#!/usr/bin/env python
"""Build a frame-aligned smoke-test dataset for GR00T N1.7 fine-tuning.

The demo dataset under experiment_data/ has two hard blockers for the GR00T
LeRobot loader:
  A) parquet has ~44,796 rows but the videos have ~1556/1557 frames, while the
     loader indexes video frames by integer parquet-row position and asserts
     len(video) == len(df) (gr00t/data/dataset/lerobot_episode_loader.py).
  B) the top-level processed_data/ uses a non-canonical "{episode_file_stem}"
     path pattern that the loader cannot format -> KeyError.

batch_000000/ already uses the canonical episode_{index:06d} naming (blocker B
absent) but still has blocker A. This script copies batch_000000/ to a new
processed_data_aligned/ and uniformly downsamples each parquet to exactly
N = min(head_frames, wrist_frames) rows so the loader's assertion passes.
Originals are never modified.

Run:  uv run python scripts/align_smoke_dataset.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "experiment_data" / "processed_data" / "batch_000000"
DST = REPO / "experiment_data" / "processed_data_aligned"
FPS = 30.0


def count_frames(mp4: Path) -> int:
    """Exact decoded frame count via ffprobe -count_frames."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(mp4),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def main() -> None:
    assert SRC.exists(), f"source not found: {SRC}"
    if DST.exists():
        print(f"[clean] removing existing {DST}")
        shutil.rmtree(DST)
    print(f"[copy ] {SRC} -> {DST}")
    shutil.copytree(SRC, DST)

    info = json.loads((DST / "meta" / "info.json").read_text())
    chunks_size = info["chunks_size"]
    data_path_tmpl = info["data_path"]
    video_path_tmpl = info["video_path"]

    episodes = [json.loads(l) for l in (DST / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]

    total_frames = 0
    global_off = 0
    for ep in episodes:
        eidx = ep["episode_index"]
        chunk = eidx // chunks_size
        pq = DST / data_path_tmpl.format(episode_chunk=chunk, episode_index=eidx)
        head = DST / video_path_tmpl.format(
            episode_chunk=chunk, video_key="observation.images.head", episode_index=eidx
        )
        wrist = DST / video_path_tmpl.format(
            episode_chunk=chunk, video_key="observation.images.wrist", episode_index=eidx
        )

        n_head, n_wrist = count_frames(head), count_frames(wrist)
        n = min(n_head, n_wrist)

        df = pd.read_parquet(pq)
        rows = len(df)
        idx = np.linspace(0, rows - 1, n).round().astype(int)
        df = df.iloc[idx].reset_index(drop=True)

        # reset cadence columns to a clean 30 fps; loader is positional and never
        # reads these for indexing, but downstream tooling expects them coherent.
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["timestamp"] = (np.arange(n) / FPS).astype(np.float32)
        df["episode_index"] = np.int64(eidx)
        if "index" in df.columns:
            df["index"] = np.arange(global_off, global_off + n, dtype=np.int64)

        df.to_parquet(pq, index=False)
        ep["length"] = int(n)
        total_frames += n
        global_off += n
        print(f"[ep {eidx}] head={n_head} wrist={n_wrist} -> N={n}; parquet {rows} -> {len(df)} rows")

    # rewrite episodes.jsonl with updated lengths
    (DST / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes)
    )

    # info.json hygiene (not load-critical) + honest codec label (torchcodec ignores it)
    info["total_frames"] = total_frames
    info["total_episodes"] = len(episodes)
    for k in ("observation.images.head", "observation.images.wrist"):
        feat = info["features"].get(k, {})
        if isinstance(feat.get("info"), dict) and "video.codec" in feat["info"]:
            feat["info"]["video.codec"] = "mpeg4"
    (DST / "meta" / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # drop stale stats so gr00t/data/stats.py regenerates them against new row counts
    for f in ("stats.json", "relative_stats.json"):
        p = DST / "meta" / f
        if p.exists():
            p.unlink()
            print(f"[stats] removed stale {p.name} (regenerate with gr00t/data/stats.py)")

    print(f"\n[done ] {len(episodes)} episodes, total_frames={total_frames}")
    print(f"        dataset: {DST}")


if __name__ == "__main__":
    main()
