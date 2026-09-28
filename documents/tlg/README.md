<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# The Last Guardian performance fork

> This is a fork made for fun, to see how much I could optimize this one game with no coding
> knowledge. All the code and measurements were done with Claude (Anthropic's AI) doing the
> programming. It is not an official shadPS4 project.

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
| Heaviest stretch found so far (1440p, ~66k draws/s) | - | ~40 FPS with the profiler on |
| Buffer readbacks that stall the game | ~350/s | ~6/s |

The limit now is the GPU command processor thread (one CPU core translating ~100k draws and
dispatches per second), not readbacks.

## What was done

The game reads GPU results back to the CPU every frame (Trico's feathers, animation, physics).
On a PS4 that is free: CPU and GPU share memory. On a PC each read means copying data from the
graphics card and waiting for it, which is what made the game slow. The work, in order:

1. **Measure first.** A built-in profiler (`readback_stats_enabled`) that logs, every second,
   readbacks, page faults, waits, and where the GPU command thread spends its time, down to
   single shaders.
2. **Batching.** Many readbacks waiting on the same GPU work share one wait instead of one each.
3. **Asynchronous fences.** The game's "GPU work done" signals are delivered by a background
   thread once the data is back, so the GPU command thread no longer stops to wait for them.
4. **Waits on the GPU.** When a GPU command waits for an earlier GPU result, the wait is done by
   the GPU itself instead of blocking the CPU thread (`readback_fence_wait_shortcut: 3`).
5. **Targeted waits.** When the game touches memory that is still being copied back, it waits
   for that one copy only, not for the whole GPU to go idle.
6. **Bug fixes found on the way:** a hang when loading into gameplay (pages stuck in a fault
   loop), a crash with 3D textures, and a race where a late fence could overwrite newer data.
7. **Cheaper draw recording.** Faster lookups for the small constant buffers copied before each
   draw, fetch shaders parsed once instead of at every draw, fewer command buffer submissions.
8. **Stream copy reuse.** About 70% of those small constant copies repeated data already copied
   in the same batch; they now reuse the earlier copy (`readback_stream_reuse_enabled`).
9. **Prefetch tuning.** Memory the game read back recently is copied back in advance at each
   fence. A range now stops being prefetched ~1.4 s after the game last needed it instead of
   ~14 s (`readback_prefetch_lifetime: 60`), a third fewer downloads for the same smoothness.
10. **Render target reuse.** Draws whose render target registers match the previous draw reuse
    its target images instead of searching the texture cache again (`rt_lookup_reuse_enabled`).
11. **Fewer allocations.** Vertex shader data copied at every draw no longer allocates memory.

Tried and dropped (measured, not worth it): real shared CPU/GPU memory (much slower on PC),
keeping frequently written pages unprotected (moved cost onto the bottleneck thread), and
splitting the command thread across CPU cores (too little movable work), and prefetching only
memory the game read (not wrote) (`readback_prefetch_mode: 2`, less work but twice the waiting).

Lessons from other emulators (Dolphin, Xenia, Ryujinx) are in
[readback-performance.md](readback-performance.md#lessons-from-other-emulators).

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
    "readback_prefetch_lifetime": 60,
    "rt_lookup_reuse_enabled": true,
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

- Once, the game crashed at an area transition (Havok thread, privileged instruction) with
  `rt_lookup_reuse_enabled` on; the retry through the same spot was clean. If it happens again,
  set it to `false` and report whether the crash stays.
- The command thread is still the limit, but its cost is now spread over many small steps. The
  remaining bigger lever is structural (spreading that thread's work over more CPU cores).
