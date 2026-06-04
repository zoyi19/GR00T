#!/usr/bin/env python
"""把 processed_data 整理成训练就绪数据集:逐集下采样到视频帧数 + 规范命名 + 修 meta。
   只写新目录,源目录不动。用法: uv run python scripts/prepare_dataset.py
   详见 experiment_data/数据集准备与全量微调流程.md"""
import json, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path("experiment_data/processed_data")          # 源(原始 LeRobot 数据集)
DST = Path("experiment_data/processed_data_train")    # 输出(训练就绪)
FPS = 30.0
VIDEO_KEYS = ["observation.images.head", "observation.images.wrist"]
DATA_TMPL  = "data/chunk-{chunk:03d}/episode_{idx:06d}.parquet"
VIDEO_TMPL = "videos/chunk-{chunk:03d}/{key}/episode_{idx:06d}.mp4"


def nframes(mp4: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def main():
    info = json.loads((SRC / "meta/info.json").read_text())
    chunks_size = info["chunks_size"]
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC / "meta", DST / "meta")        # 先整体拷 meta,后面改几项

    lengths, total = {}, 0
    for pq in sorted((SRC / "data").rglob("*.parquet")):
        stem, chunk_dir = pq.stem, pq.parent.name       # 例:chunk-000
        df = pd.read_parquet(pq)
        eidx = int(df["episode_index"].iloc[0])          # 从数据里取真实 episode_index
        chunk = eidx // chunks_size
        srcv = {k: SRC / "videos" / chunk_dir / k / f"{stem}.mp4" for k in VIDEO_KEYS}
        N = min(nframes(v) for v in srcv.values())       # 取两路视频帧数的最小值

        idx = np.linspace(0, len(df) - 1, N).round().astype(int)   # 均匀下采样到 N 行
        df = df.iloc[idx].reset_index(drop=True)
        df["frame_index"] = np.arange(N, dtype="int64")
        df["timestamp"] = (np.arange(N) / FPS).astype("float32")
        df["episode_index"] = np.int64(eidx)

        out_pq = DST / DATA_TMPL.format(chunk=chunk, idx=eidx)
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_pq, index=False)
        for k, v in srcv.items():                         # 视频按规范名拷贝(不转码)
            out_v = DST / VIDEO_TMPL.format(chunk=chunk, key=k, idx=eidx)
            out_v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(v, out_v)

        lengths[eidx] = N
        total += N
        print(f"ep {eidx}: rows {len(df)} == frames {N}")

    eps = [json.loads(l) for l in (DST / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    for e in eps:
        e["length"] = lengths[e["episode_index"]]
    (DST / "meta/episodes.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps))

    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["total_episodes"] = len(eps)
    info["total_frames"] = total
    info["total_videos"] = len(eps) * len(VIDEO_KEYS)
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))

    for f in ("stats.json", "relative_stats.json"):       # 删旧统计,稍后重算
        p = DST / "meta" / f
        if p.exists():
            p.unlink()
    print(f"done: {len(eps)} episodes, {total} frames -> {DST}")


if __name__ == "__main__":
    main()
