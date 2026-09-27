<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# The Last Guardian (CUSA03627): readback performance work

These notes cover how GPU readbacks were made cheaper for The Last Guardian (TLG) while keeping
them fully accurate. The game needs Precise readbacks plus linear image readbacks: without them
Trico's feathers, climbing, the spear and outdoor lighting break (see upstream issue #4725).
Every change is opt-in per game and is off by default.

Test system: Ryzen 7 5700X3D, RTX 4090, 64 GB RAM, 60 FPS patch and native 4K patch.

## Results (outdoor area near the gates, walking with Trico)

| Metric (gameplay, per second) | Baseline | Current |
| :--- | :--- | :--- |
| FPS | ~18 | ~29 average (median 27, range 14-61) |
| Buffer readbacks that fault and stall | ~350 | ~6 |
| GPU thread time blocked in `Scheduler::Finish` | ~430 ms | ~9 ms |
| Guest threads blocked on readbacks | ~380 ms | ~11 ms |

## How to enable

The settings live in `user/custom_configs/<serial>.readback.json`. Only the emulator reads this
file: launchers rewrite `<serial>.json` and drop keys they don't know.

```json
{
  "GPU": {
    "readback_async_fences_enabled": true,
    "readback_batching_enabled": true,
    "readback_fence_wait_shortcut": 1,
    "readback_stats_enabled": true
  }
}
```

`readbacks_mode` (2 = Precise) and `readback_linear_images_enabled` stay in `<serial>.json`.

| Setting | Meaning |
| :--- | :--- |
| `readback_stats_enabled` | Writes `log/readback_stats.csv` (one line per second) and `log/readback_top.txt` (buffers, writer shaders and images ranked by stall time). |
| `readback_batching_enabled` | Completes image readbacks and recently read back buffers with one GPU wait per fence. |
| `readback_async_fences_enabled` | Fences are signaled from a background thread once their readbacks are in guest memory, instead of the GPU thread blocking. |
| `readback_fence_wait_shortcut` | With async fences: 0 = off, 1 = graphics queue `WaitRegMem` may pass for fence values still held back, 2 = graphics and compute, 3 = graphics and compute plus a GPU barrier after each such wait. |
| `readback_hot_write_pages_enabled` | Pages the CPU rewrites after nearly every upload stay writable and are uploaded at each use. Fewer write faults, but more command thread time: keep off. |
| `readback_stream_reuse_enabled` | A small read-only buffer bound again in the same command buffer reuses its earlier stream copy. |

## Steps taken

### 0. First attempt (discarded)
A plan based on reading the code alone (async image downloads, `HOST_CACHED` download memory,
a bigger download buffer, mid-frame submits) made no measurable difference and was reverted.
It was built without data. Changing settings helped more: turning off
`direct_memory_access_enabled` and `neo_mode` took the game from ~15 to ~20 FPS.

### 1. Measure first: readback statistics
`src/video_core/readback_stats.*` counts per second: presented frames, draws and dispatches,
page faults (read and write) and time spent handling them, protect calls, buffer and image
readbacks, bytes read back, time guest threads were blocked, `Finish` calls and time, and how
busy the GPU command thread is. It also records which shader, fill or copy last wrote each
range, so readbacks can be traced to their writer.

Finding: the GPU command thread was 100% busy, and about 45% of that time was spent waiting in
~42 `Finish` calls per frame (~20 buffer readbacks and ~24 linear image readbacks). The same
buffers and images were read back every frame.

### 2. A + B: batching (`readback_batching_enabled`)
- **A:** image readbacks queued at a fence are copied together and completed with one wait.
- **B:** buffer ranges that faulted recently ("hot ranges") are downloaded at the next fence
  whenever the GPU has written them again, so the CPU finds the data already in memory instead
  of faulting. Ranges expire after 600 submits and are relearned on the next fault.

Result: buffer faults went from ~350/s to ~0-10/s, but FPS stayed the same. The game signals
~25 fences per frame with about one image each, so there was nothing to merge. And each wait
blocks until the GPU finishes everything submitted so far, so the wait time was really the GPU's
own frame time: the CPU and GPU were taking turns instead of running in parallel.

### 3. C: asynchronous fences (`readback_async_fences_enabled`)
At EOP/EOS/ReleaseMem, the readback copies are recorded and submitted without waiting. The
scheduler's priority thread waits for the GPU, writes the data to guest memory, and only then
writes the fence value and raises the interrupt. A real PS4 behaves the same way: the fence
fires when the GPU is done. Accuracy rules (the RFC in #4316 and yuzu's async downloads work
the same way):
- Fences are signaled in the order they were issued (the priority thread is FIFO).
- `Scheduler::Finish` also waits for pending write backs, so every synchronous path is ordered
  after the asynchronous ones.
- A background write back never unprotects a range the GPU has written again since
  (`ApplyPendingUnmarks` runs on the GPU thread and checks `gpu_modified_ranges`).
- A guest access to a range whose download is still in flight waits for it.
- GDS based fences (`GdsStore`, `GdsMemStore`) stay synchronous.

Bugs found and fixed along the way:
- **`Protect` assertion (write-only permission):** a CPU write hit a range whose async readback
  had not landed. The page ended up marked both CPU- and GPU-modified. Fix: unprotects are
  applied immediately on the GPU thread during faults and fences.
- **Fence writes and faults:** fence writes from the background thread go through the memory
  backing, or are written directly when the memory has no GPU mapping and so cannot be
  protected. They only fall back to the GPU thread in the remaining rare case.

Result: GPU-thread waiting dropped from ~430 ms/s to ~5-50 ms/s, but FPS still didn't move.
New counters showed why: the game's own command streams wait on those fences with `WaitRegMem`,
so the wait simply moved there.

### 4. Command stream wait shortcut (`readback_fence_wait_shortcut`)
When a `WaitRegMem` waits for a value that a held-back fence will write, it may pass right away.
That is how it behaved before step 3, and the GPU executes commands in order. Only the
CPU-visible fence and its readback data stay deferred.
- Allowing it on compute queues broke the game's `JsComputeContext::StallUntilSignaled`
  (timeout, then crash), so the default is graphics queues only (1).
- A limit of one prefetch per range per frame was tried and reverted: rewrites later in the
  frame fell back to synchronous faults and cost ~150-270 ms/s.

### 5. Robustness
- **Launcher-proof settings:** the new keys moved to `<serial>.readback.json` after the
  launcher stripped them from `<serial>.json` several times.
- **Hang diagnostics:** waits longer than 2 s log a `Hang check:` warning describing what they
  wait on. This covers command stream `WaitRegMem` (with value, reference and pending fence
  writes), priority operations waiting on a GPU tick (and whether it was ever submitted), the GPU
  thread waiting on priority operations, and flip label waits. One intermittent hang was seen
  once and not reproduced since.

### 6. Texture cache fixes
- **Crash:** a 3D texture reused at the same address with more depth (36x36x1197
  R16G16B16A16) hit `ResolveOverlap: Unreachable code`. The old image is now replaced when it
  hasn't been used for more than 32 submits. The assertion remains for images in use.
- **Image age:** `ResolveOverlap` counted image age in scheduler ticks. Async fences flush many
  times per frame, which made recently used images look old. Age is now counted with `gc_tick`,
  which advances once per guest submit.

### 7. Shared memory experiment (parked, branch `tlg-shared-memory-experiment`)
Buffers read back in 8+ different frames (up to 16 MB) were re-created on top of guest memory
through `VK_EXT_external_memory_host`, so the GPU writes guest memory directly.
- The import only works once all tracker protection is removed from the pages. Before that, the
  NVIDIA driver returned `ErrorOutOfDeviceMemory`. `HostMappedForeignMemory` returns
  `ErrorInitializationFailed`.
- Result: 69 buffers were shared, but the title screen dropped from 60 to 13 FPS with only 3 tiny
  (16-80 KB) shared buffers, and gameplay dropped to 6-7 FPS. Every fence after a GPU write to
  shared memory must wait for the GPU, and the command processor thread then spends ~2 s/s
  (summed across queues) blocked in `WaitRegMem` on those fences. Large shared buffers (7-12 MB)
  also moved heavy GPU traffic onto PCIe.
- Conclusion: shared memory only pays off once fence waits no longer block the command processor
  thread (see `documents/tlg-gpu-side-fence-waits-plan.md`).

### 8. Fence waits on the GPU timeline (`readback_fence_wait_shortcut: 3`)
See `documents/tlg-gpu-side-fence-waits-plan.md`. Command stream waits pass as soon as the fence
they wait for is recorded, and a GPU barrier (level 3) orders the following work after it, so the
command processor thread no longer waits for the GPU at these waits. Only the latest recorded
value per label counts. The stats now split waits by class and queue, and a watchdog logs where
the command processor is stuck if it stops recording work for 3 s. Spin locks held for more than
2 s are reported too.
- `WriteData` to a label whose fence write is still held back is queued behind it (fixed the
  `StallUntilSignaled` crash that level 2 always had).
- Result: stable in gameplay, compute fence waits gone (~1.5 s/s to 0), but FPS about the same
  (~29.6 average, median 29). The command processor is still ~97% busy with other work.

### 9. Command processor profile
`readback_stats.csv` now splits command processor time per second (`cp_*_ms` columns, exclusive
time), measures GPU execution time with timestamp queries (`gpu_busy_ms`) and the CPU time of
presenting (`present_cpu_ms`). Gameplay near the gates at ~30 FPS:

| Command processor work | ms per second |
| :--- | :--- |
| Recording draws (~68k/s, excluding uploads and pipeline lookup) | ~430 |
| Recording dispatches (~26k/s) | ~145 |
| Fence handling | ~110 |
| Pipeline lookup | ~75 |
| Waiting for the GPU | ~40-130 |
| Uploads | ~60 |
| Page protection | ~30-45 |
| Idle or spinning on waits | ~5 |

The GPU is only ~31% busy and presentation costs ~0.75 ms per frame. The bottleneck is the
CPU cost of recording draws (~6 us each), not readbacks anymore.

### 10. Cheaper recording
A finer profile (`rec_*`, `buf_*`, `tex_*` columns) showed two hot spots besides the spread out
per-draw work: ~330k small read-only buffer copies per second (~150 ms/s) and ~4,800 async fences
per second that each submitted a command buffer (~125 ms/s).
- Async fences only submit when they recorded copies or a priority operation waits on the command
  buffer being recorded: submits ~4,800/s -> ~740/s, fence handling ~129 -> ~62 ms/s.
- Stream buffer copies skip the memory manager lock and mapping lookup when the range lies in an
  area already looked up and no mapping changed since (~5% cheaper, the rest is memory traffic).
- 23.6 -> 25.6 FPS in the same scene with profiling on (+8%).

### 11. Load hang fix and targeted readback waits
- **Load hang:** an async readback's unprotect was dropped for its whole range when the GPU wrote
  part of it again, leaving pages protected with nothing to download. The CPU then faulted on
  them forever (~355k empty readbacks per second, the intermittent hang when loading into
  gameplay). Landed readbacks now unprotect exactly the pages the GPU didn't write again, and the
  fault handler clears a stale mark on the faulting page (`Readback: cleared a stale GPU mark`).
- **Targeted waits:** a fault on a page whose readback is still in flight waits for that readback
  only, instead of `Finish` (submit everything and wait for the GPU to go idle).
- Keeping pages protected while any newer readback was in flight was tried and reverted: hot
  ranges are prefetched at nearly every fence, so the CPU kept waiting (~80 ms/s of GPU waits).
  The game only sees a fence after its readback landed, so this strictness isn't needed.
- Result (profiled): GPU waits ~80 -> ~22 ms/s, game threads blocked on readbacks ~83 -> ~27
  ms/s. Unprofiled play: 25-45 FPS depending on the area.
- Also: fetch shaders are parsed once per code address instead of at every draw, and pending
  operations only query the GPU progress when there is one to release (no measurable gain).
  Vulkan call timing (option B study: ~10-13% of the command processor) is off by default.

### 12. Phase 0 measurements, stream copy reuse
- Phase 0 of the multicore plan: the work an encoder thread could take (Vulkan calls ~10-13%,
  stream memcpy ~3%) is below the plan's ~20% gate, so the multicore rework is on hold.
- Hot write pages (`readback_hot_write_pages_enabled`): write faults ~35k -> ~6k/s, but uploads
  on the command thread +~55 ms/s. The game threads have headroom, the command thread doesn't:
  off by default.
- Per draw: render target registers repeat in ~98% of draws, but shader user data and context
  registers change before nearly every draw (~2% and ~5% identical), so skipping binding or
  pipeline work per draw has little to gain.
- Stream copies: ~69% repeat a range already copied in the same command buffer with identical
  data, and 0.0% of those repeats had changed data. `readback_stream_reuse_enabled` reuses the
  earlier copy; the table is dropped on a new command buffer, a stream buffer wrap, or a command
  processor write to guest memory (WriteData, occlusion results, CPU-side DMA). GPU-written ranges
  never take the stream path. Result: ~70% of stream copies skipped, stream copy time 134 -> 57
  ms/s, ~8% more draws per second on the command thread, better 1% lows.
- **Pending:** test the spear sequence (start of the game) with stream reuse on.
- Resolution: dropping the 4K patch raised FPS although the GPU was only ~37% busy. The game waits
  on GPU results (readbacks, fences, `WaitRegMem`) inside every frame, so GPU frame time is partly
  on the critical path: command thread GPU waits ~16 -> ~2 ms/s, idle ~45 -> ~12 ms/s.

## Known limits and next steps
- The GPU command thread is still ~99% busy. Compute queues still wait on held-back fences
  (~1.5 s/s of wall time across queues). About 40k write faults/s (~130 ms/s) and ~90 ms/s of
  page protection calls remain.
- **Next:** real shared memory for the hot ranges the CPU reads. Back those guest ranges with
  memory both the CPU and GPU can access, so GPU writes land in guest memory directly: no copy,
  no fault, and the fence alone orders the access.
- The long-term fix is moving GPU-internal waits (`WaitRegMem` on fences) onto the GPU timeline,
  so the command processor thread never blocks on them. That is an architectural change for
  upstream.
