<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# The Last Guardian performance fork

This branch (`tlg-readback-perf`) makes The Last Guardian (CUSA03627) faster in shadPS4 while
keeping Precise readbacks and linear image readbacks fully accurate. Those readbacks are what keep
Trico's feathers, climbing, the spear and outdoor lighting correct.

Every change is opt-in and off by default: with no extra settings the emulator behaves like
upstream.

## Results

Ryzen 7 5700X3D, RTX 4090, 60 FPS patch, Precise readbacks and linear image readbacks on:

| | Before | Now |
| :--- | :--- | :--- |
| Outdoor area near the gates (4K patch) | ~18 FPS | ~29 FPS average |
| General play, 1080p / 1440p | - | ~25-45 FPS depending on the area |
| Buffer readbacks that stall the game | ~350/s | ~6/s |

The limit now is the GPU command processor thread (one CPU core translating ~100k draws and
dispatches per second), not readbacks.

## Setup

1. Game settings (`user/custom_configs/CUSA03627.json`, or the launcher's per-game settings):
   `readbacks_mode: 2` (Precise) and `readback_linear_images_enabled: true`.
2. Create `user/custom_configs/CUSA03627.readback.json` (launchers don't touch this file):

```json
{
  "GPU": {
    "readback_async_fences_enabled": true,
    "readback_batching_enabled": true,
    "readback_fence_wait_shortcut": 3,
    "readback_stream_reuse_enabled": true,
    "readback_stats_enabled": false
  }
}
```

Set `readback_stats_enabled` to `true` to write the profiler output (`log/readback_stats.csv`
and `log/readback_top.txt`). It costs some FPS.

A lower resolution patch (1080p or 1440p) helps too: the game waits on GPU results inside every
frame, so a shorter GPU frame shortens those waits.

## Documents

- [readback-performance.md](readback-performance.md): what was changed and measured, step by step,
  and every setting.
- [gpu-side-fence-waits-plan.md](gpu-side-fence-waits-plan.md): the fence and wait design.

## Pending

- Test the spear sequence at the start of the game with `readback_stream_reuse_enabled`.
