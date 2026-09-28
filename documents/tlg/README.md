<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# The Last Guardian performance fork

> This is a fork made for fun, to see how much I could optimize this one game with no coding
> knowledge. All the code and measurements were done with Claude (Anthropic's AI) doing the
> programming. It is not an official shadPS4 project.

This fork makes The Last Guardian (CUSA03627) faster in shadPS4 while keeping Precise readbacks
and linear image readbacks fully accurate. Those readbacks are what keep Trico's feathers,
climbing, the spear and outdoor lighting correct.

Branches:

- `tlg-encoder-thread` (default): everything below, including a second CPU thread for the GPU
  command processor (items 12-15). Recommended.
- `tlg-readback-perf`: the readback work only (items 1-11), with a single command processor thread.

Every change is opt-in and off by default: with no extra settings the emulator behaves like
upstream.

## Results

With the 60 FPS patch and without the 4K texture patch, 60 FPS is reachable in many areas,
depending on the PC. The heaviest scenes (thousands of draws per frame) still run lower.

The limit now is the GPU command processor thread (one CPU core translating ~100k draws and
dispatches per second), not readbacks: buffer readbacks that stall the game went from ~350 to ~6
per second.

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
12. **Encoder thread** (`cp_encoder_mode: 2`). The command thread records its Vulkan commands
    into lists, and a second thread turns them into GPU command buffers and submits them. The
    command thread spends ~25% less time per draw.
13. **Full fences for compute** (`cp_full_fence_kinds: 4`). With the encoder thread, GPU work
    reaches the GPU a little later than it is recorded. Without full fences, Trico's feathers
    flickered when pulling the spear out at the start of the game. Compute fences now wait for
    all work recorded before them; that fixes it at a fraction of the cost of doing it for every
    fence kind.
14. **Shader lookup reuse** (`stage_lookup_reuse: 1`). At each draw, a shader first checks the
    variant it picked last time (right ~97% of the time) before searching all its variants.
15. **Game threads wait for their own readbacks** (`readback_async_guest_faults`). When a game
    thread touches memory the GPU wrote, the command thread starts the copy and goes on; only the
    game thread waits for it. The command thread's time waiting for the GPU in the heaviest scene
    went from ~7% to ~2%.

The profiler also tags every GPU wait of the command thread with its reason, the step that caused
it and the memory address, which is how items 13-15 were found.

Tried and dropped (measured, not worth it): real shared CPU/GPU memory (much slower on PC),
keeping frequently written pages unprotected (moved cost onto the bottleneck thread), moving the
small constant copies to the encoder thread (~3% of the work, high risk), prefetching only memory
the game read (not wrote) (`readback_prefetch_mode: 2`, less work but twice the waiting), and
reading shader code through the unprotected view of memory (`shader_backing_reads`, disabled: the
game also copies shader code with the GPU, and it crashed at game load).

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
    "readback_prefetch_mode": 1,
    "readback_prefetch_lifetime": 60,
    "rt_lookup_reuse_enabled": true,
    "stage_lookup_reuse": 1,
    "readback_async_guest_faults": true,
    "cp_encoder_mode": 2,
    "cp_full_fence_kinds": 4,
    "readback_stats_enabled": false
  }
}
```

On the `tlg-readback-perf` branch, leave out `stage_lookup_reuse`, `readback_async_guest_faults`,
`cp_encoder_mode` and `cp_full_fence_kinds` (they don't exist there).

Set `readback_stats_enabled` to `true` to write the profiler output (`log/readback_stats.csv`
and `log/readback_top.txt`). It costs some FPS.

A lower resolution patch (1080p or 1440p) helps too: the game waits on GPU results inside every
frame, so a shorter GPU frame shortens those waits.

## Documents

- [readback-performance.md](readback-performance.md): what was changed and measured, step by step,
  and every setting.
- [gpu-side-fence-waits-plan.md](gpu-side-fence-waits-plan.md): the fence and wait design.

## Pending

- Crashes seen once each, not reproduced: at an area transition, the Havok thread (privileged
  instruction) with `rt_lookup_reuse_enabled` on, and later the game's render thread (access
  violation). If one comes back at the same spot, set `rt_lookup_reuse_enabled` to `false` and
  report whether it stays.
- Trico's feathers: `cp_full_fence_kinds: 0` (no full fences) flickers when pulling the spear
  out. 4 was clean there, and so were 27 and 31, so the exact fence kind that matters isn't pinned
  down yet.
- The command thread is still the limit, but its cost is now spread over many small steps
  (bindings, draw recording, shader lookups). Further gains need less work per draw.
