"""Small wrapper around GR00T PolicyClient for 54-DoF action chunks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import schema


class PolicyServerError(RuntimeError):
    """Raised when the GR00T policy server is unavailable or returns bad data."""


class GrootPolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 15000) -> None:
        try:
            from gr00t.policy.server_client import PolicyClient
        except Exception as exc:  # noqa: BLE001
            raise PolicyServerError(
                "failed to import gr00t.policy.server_client; run from the GR00T repo env"
            ) from exc

        self._client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=False)
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.last_latency_ms: float | None = None
        self._modality_configs: dict[str, Any] | None = None

    def ping(self) -> bool:
        return bool(self._client.ping())

    def reset(self) -> None:
        self._client.reset(options=None)

    def get_modality_config(self) -> dict[str, Any]:
        if self._modality_configs is None:
            try:
                self._modality_configs = self._client.get_modality_config()
            except Exception as exc:  # noqa: BLE001
                raise PolicyServerError(f"failed to query modality config: {exc}") from exc
        return self._modality_configs

    def predict_action_chunk(
        self,
        observation: dict[str, Any],
        *,
        min_horizon: int = 1,
    ) -> np.ndarray:
        start = time.perf_counter()
        try:
            action, _info = self._client.get_action(observation)
        except Exception as exc:  # noqa: BLE001
            raise PolicyServerError(f"policy server get_action failed: {exc}") from exc
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0

        action_chunk = self._flatten_action_dict(action)
        if action_chunk.ndim != 2:
            raise PolicyServerError(f"action chunk must be 2-D [H,54], got {action_chunk.shape}")
        if action_chunk.shape[1] != schema.ACTION_DIM:
            raise PolicyServerError(
                f"action chunk width must be {schema.ACTION_DIM}, got {action_chunk.shape[1]}"
            )
        if action_chunk.shape[0] < min_horizon:
            raise PolicyServerError(
                f"action chunk horizon {action_chunk.shape[0]} < requested {min_horizon}"
            )
        if not np.all(np.isfinite(action_chunk)):
            raise PolicyServerError("action chunk contains NaN or Inf")
        return action_chunk.astype(np.float32, copy=False)

    def _flatten_action_dict(self, action: dict[str, Any]) -> np.ndarray:
        if not isinstance(action, dict):
            raise PolicyServerError(f"policy action must be a dict, got {type(action)!r}")
        configs = self.get_modality_config()
        keys = list(configs["action"].modality_keys)
        missing = [key for key in keys if key not in action]
        if missing:
            raise PolicyServerError(f"policy action missing keys: {missing}")

        chunks = []
        horizon = None
        for key in keys:
            arr = np.asarray(action[key], dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            elif arr.ndim == 2:
                pass
            elif arr.ndim == 1:
                arr = arr[None, :]
            else:
                raise PolicyServerError(f"action[{key}] has unsupported shape {arr.shape}")
            if horizon is None:
                horizon = arr.shape[0]
            elif arr.shape[0] != horizon:
                raise PolicyServerError(
                    f"action key {key} horizon {arr.shape[0]} does not match {horizon}"
                )
            chunks.append(np.atleast_2d(arr))
        return np.concatenate(chunks, axis=1)

