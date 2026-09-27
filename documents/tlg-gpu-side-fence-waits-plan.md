<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Plan: stop the command processor from waiting on the GPU at fences

Target: The Last Guardian (CUSA03627) with Precise + linear image readbacks. It builds on
branch `tlg-readback-perf` (`78d4f381`, `6185087a`) and the notes in
`documents/tlg-readback-performance.md`.

## Status

Phases 0-3 are implemented and tested (level 3 is stable in gameplay). Results in gameplay:
compute waits on held back fences went from ~1.5 s/s to 0, ~2,700 graphics and ~500 compute
waits per second pass without blocking, but FPS only moved from ~29 (median 27) to ~29.6
(median 29): the command processor is still ~97% busy (recording ~2,400 draws and ~900
dispatches per frame, faults and protects, flip label waits). Two ordering bugs found in testing:
- **WriteData behind held back fences:** compute rings write a label (`WriteData`, 8 bytes)
  right after waiting for their own fence on it. The write happened immediately and the held
  back fence value landed later and overwrote it (`StallUntilSignaled` timeout on
  `0x7c6056f0`). Such writes are now queued behind the held back fence.
- **Labels written by someone else:** if guest memory holds a value that is neither the value
  from before the held back fences nor a recorded one, waits on it poll memory again.
- **Phase 0:** waits are classified per queue in `readback_stats.csv` (`gfx_*`, `cmp_*`,
  `sem_*` columns) and per address in `readback_top.txt` ("Command stream waits"). MemSemaphore
  and Rewind spins are timed and have hang checks. A stall watchdog logs the last packet and step
  of the command processor when it stays busy for 3 s without recording work.
- **Phase 1:** `recorded_fences` in `Liverpool` is the label timeline. It keeps only the latest
  value per label, so an older held back value can no longer satisfy a wait meant for a newer
  one (hypothesis 1 of 5.4). Levels 1 and 2 now use it.
- **Phase 2:** `Rasterizer::CommandStreamBarrier` records a full memory barrier when a wait passes
  on a fence whose GPU work hasn't finished. Barriers merge when no work was recorded between
  them.
- **Phase 3:** `readback_fence_wait_shortcut: 3` = graphics and compute queues plus the barrier.
  The first compute wait passed per label is logged (`Fence timeline:`) to match against a
  `StallUntilSignaled` label if the crash comes back.

## 1. Goal and success criteria

The emulator's command processor (CP) thread should never block waiting for the host GPU to
finish work just because the guest's command stream waits on a fence. Readbacks stay fully
accurate: the guest CPU still only sees a fence once its data is in guest memory.

| Metric (gameplay near the gates, per second) | Today | Target |
| :--- | :--- | :--- |
| FPS | ~29 average (median 27) | 35+ average, fewer dips to ~15 |
| CP thread blocked in `WaitRegMem` (wall time, all queues) | ~1.5 s | < 0.1 s |
| Compute queue waits on held-back fences | ~430 | ~0 |
| `Finish` time on the CP thread | ~9 ms | stays < 20 ms |
| Hangs / timeouts in 10 min of play | 0 (recent runs) | 0 |

Secondary goal: once this works, re-test the parked shared memory branch
(`tlg-shared-memory-experiment`). It failed only because accurate fences serialize the CP thread.

## 2. How it works today

- **One CP thread** (`AmdGpu::Liverpool::Process`, `shadPS4:GpuCommandProcessor`) runs every
  guest ring as a coroutine: graphics DCB (`ProcessGraphics`), constant engine CCB
  (`ProcessCeUpdate`) and up to 56 compute rings (`ProcessCompute`). It switches between them
  round robin (`curr_qid = (curr_qid + 1) % num_mapped_queues`) whenever one yields.
- **One Vulkan queue** (`Instance::GetGraphicsQueue`) executes all recorded work in the order
  the CP thread recorded it (`Scheduler::Flush` / `Finish`).
- **Guest-side synchronization packets** are handled on the CP thread:
  - `EventWriteEop`, `EventWriteEos`, `ReleaseMem` write a fence value (label) and may raise
    an interrupt.
  - `WaitRegMem` spins (`YIELD_GFX` / `YIELD_ASC`) until
    `(memory or register & mask) <func> ref` holds.
  - `MemSemaphore`, `Rewind`, `WaitOnCeCounter` are similar waits.
- **Before async fences**, a fence value was written the moment the CP thread processed the
  packet, before the GPU executed anything. `WaitRegMem` on it passed immediately. That is not
  accurate, which is why Precise readbacks need page faults plus `Finish` to be correct.
- **With async fences** (`readback_async_fences_enabled`), fence values are written by the
  scheduler's priority thread after the GPU finished and the readbacks landed. The guest CPU is
  now correct without faults. But the guest's command streams also wait on those fences, so the
  CP thread spins in `WaitRegMem` until the GPU catches up: CPU and GPU take turns again.
- **Current workaround:** `readback_fence_wait_shortcut` lets a graphics `WaitRegMem` pass when a
  pending (held back) fence write would satisfy it (`Liverpool::IsSatisfiedByPendingFence`).
  Enabling it for compute queues crashed the game: `JsComputeContext::StallUntilSignaled`
  timed out with `label=7c6056f0, *label=1`. So compute waits still block (~430 waits/s,
  ~1.5 s/s wall time).

## 3. Key insight

A `WaitRegMem` in a guest command stream expresses a GPU-to-GPU dependency: "don't start the
following work until the producer of this value has finished". The CP thread only needs to know
that the producer has been **recorded** ahead in the single Vulkan queue, not that it has
**executed**. Execution order can then be enforced on the GPU itself with a Vulkan barrier at
that point in the command buffer.

The CP thread only has to wait when the producer has not been recorded yet:
- it is later in another ring the CP thread hasn't reached yet (cross-queue dependency), or
- it is the guest CPU (it writes the label itself), or
- it is presentation (video out labels).

None of these involve waiting on the host GPU.

## 4. Classification of waits

| Class | Producer of the awaited value | Needed | Today (async fences) |
| :--- | :--- | :--- | :--- |
| A | Fence packet (EOP/EOS/ReleaseMem) already processed by the CP thread | GPU barrier only | Waits for the GPU (gfx: shortcut; compute: blocks) |
| B | Fence packet in another ring, not processed yet | Yield until processed (CP scheduling) | Waits for the GPU |
| C | GPU shader or DMA write to ordinary memory (not a fence) | Value unknown on the CPU; readback | Reads guest memory, may fault into a readback |
| D | Guest CPU write | Poll guest memory | Same (correct) |
| E | Video out label | Presentation | Same (special path) |
| F | `MemSemaphore` / `Rewind` | Like D, or like A if GPU-signaled | Polls memory |

Phase 0 measures how much time each class costs in TLG. Most of the gain is expected from
A (and B) on compute queues.

## 5. Design

### 5.1 Label timeline (replaces the pending-fence list)
Keep, per label address, what the command stream has **recorded** so far:

```
struct LabelState {
    u64 recorded_value;   // last value written by a processed fence packet
    u32 num_bytes;        // 4 or 8
    u64 record_seq;       // global sequence number of that fence packet
    u64 gpu_tick;         // scheduler tick of the work before the fence
    u64 landed_value;     // last value the priority thread wrote to guest memory
    bool landed;          // whether recorded_value has reached guest memory
};
```

- **Updated in processing order** by `EventWriteEop`, `EventWriteEos` and `ReleaseMem`
  (Data32/Data64 only; timestamps and GDS stay on the current path). This happens whether the
  signal is async or not.
- **`WaitRegMem` (memory space) passes when** `TestValue(recorded_value)` holds or
  `Test(guest memory)` holds.
  - This fixes a weakness of the current shortcut: it checks *any* pending write to the address
    and ignores order, so an older pending value could satisfy a wait that means a newer one.
    That is a candidate cause of the compute crash (see 5.4).
- **Guest CPU overrides:** when the priority thread writes a value, it records `landed_value`.
  If guest memory later differs from `landed_value` while nothing new was recorded, the guest CPU
  wrote the label. The timeline entry is then dropped and guest memory is authoritative (class D).
- **Retiring entries:** drop an entry once it has landed and a later check finds memory equal to
  `landed_value`, to keep the table small (expect tens of live labels).
- **Threads:** the CP thread owns the table. The priority thread only reports "landed(address,
  seq)" through a small lock-protected queue, drained on the CP thread (the same pattern as
  `BufferCache::ApplyPendingUnmarks`).

Files: `video_core/amdgpu/liverpool.{h,cpp}` (new `LabelTimeline`, replaces
`pending_fence_writes`, `IsSatisfiedByPendingFence`).

### 5.2 GPU-side ordering at satisfied waits
When a wait passes through the timeline (not through guest memory) and the producer's
`gpu_tick` has not completed, record a full barrier at that point in the current command buffer:
`ALL_COMMANDS/MEMORY_WRITE -> ALL_COMMANDS/MEMORY_READ|MEMORY_WRITE`, after `EndRendering`.

- One barrier per satisfied wait. Consecutive waits with no recorded work in between can share
  one.
- If the producer is in an earlier, already submitted command buffer, queue submission order plus
  the barrier still order the work (single queue). No semaphore is needed.
- Existing per-resource barriers (`Buffer::GetBarrier`, image transitions) stay. The full barrier
  covers what they can't see, for example data passed through memory the tracker doesn't know
  about.

Files: `vk_rasterizer.{h,cpp}` (new `Rasterizer::CommandStreamBarrier()`, next to `CpSync`),
called from the `WaitRegMem` handlers.

### 5.3 Cross-queue waits (class B)
If the awaited value is neither in the timeline nor in memory, keep yielding. This is already
correct: the round robin will process the producer ring and update the timeline, and the next
test passes. No GPU wait is involved. The hang check (`ReportLongWait`) stays to catch real
deadlocks.

### 5.4 Compute queues and the `StallUntilSignaled` crash
Before enabling 5.1/5.2 for compute rings, reproduce the crash with
`readback_fence_wait_shortcut: 2` and full logging. Hypotheses to confirm or reject:
1. **Stale match:** an older pending value at the same address satisfied a wait meant for a
   newer write (fixed by 5.1's ordered, latest-value timeline).
2. **Guest CPU reset race:** the game resets the label on the CPU while an older write is still
   held back, and the held-back write later overwrites the reset.
   - Mitigation: when an entry lands, skip the guest memory write if the CPU override rule
     detected a newer CPU value, and log it.
3. **Data, not ordering:** compute work after the wait reads data that only reaches the GPU
   through a CPU upload that happens later.
   - Check which buffers the next dispatches read, using the writer attribution already in
     `readback_stats`.

Exit condition: shortcut level 2 (or the timeline on compute) runs 10 minutes including loading
transitions without the timeout.

### 5.5 What stays the same
- CPU-visible fences stay asynchronous and ordered (priority thread FIFO). `Scheduler::Finish`
  still drains pending write backs.
- Batching and hot range prefetch (steps A/B in the notes) stay.
- Class C waits keep reading guest memory, which may fault into a readback. Phase 0 decides if
  anything more is needed. Vulkan has no in-queue "wait for a memory value", so the only other
  option would be a GPU compute spin, which is risky.

## 6. Phases

### Phase 0: measure and classify (small, low risk)
- In both `WaitRegMem` handlers (and `MemSemaphore`, `Rewind`), record per wait: queue (gfx /
  compute ring id), address, function, reference, blocked time, and class A-F.
- Writer attribution per label address: EOP / EOS / ReleaseMem / WriteData / DmaData /
  shader-written range / unknown (assume the CPU).
- Add a "Waits" section to `readback_top.txt` (top addresses by blocked time with class and
  queue) and per-class columns to `readback_stats.csv`.
- Captures: title screen, gates area, spear, climbing Trico, one loading transition. Run each
  with shortcut 1, plus one with shortcut 2 to catch the compute crash with the new logging.
- **Exit:** a table of blocked time per class and queue, and a first explanation of the compute
  crash.

### Phase 1: label timeline
- Implement 5.1 and switch the graphics shortcut to use it (it should behave the same as today).
- Add a `readback_fence_wait_shortcut` level 3 = "timeline, all queues" so the old levels stay
  available for A/B tests.
- **Exit:** graphics behavior and FPS unchanged versus level 1; no hang checks in a
  10-minute run.

### Phase 2: GPU barriers at satisfied waits
- Implement 5.2.
- Validate with `vkvalidation_enabled` + `vkvalidation_sync_enabled` in a short capture (slow, a
  correctness check only): no new hazards around the inserted barriers.
- **Exit:** clean sync validation in the gates scene, no visual regressions.

### Phase 3: compute queues
- Enable the timeline for compute rings (level 3), guided by the Phase 0 findings on the crash.
- **Exit:** the success criteria in section 1. Compute waits on held-back fences ~0, CP blocked
  time < 0.1 s/s, FPS measured in the same scenes.

### Phase 4: shared memory, second try
- Rebase `tlg-shared-memory-experiment` onto Phase 3. Only share buffers up to ~1 MB and read
  back often (avoid PCIe traffic for large GPU-heavy buffers).
- **Exit:** readback copies, prefetch MB/s and protect calls down, FPS equal or better. Otherwise
  keep it parked.

**Result (tested on top of phase 3, buffers up to 1 MB):** still much slower. The title screen
dropped from 60 to 14 FPS as soon as the first 80 KB buffer was shared, and gameplay ran at
~10 FPS. The command processor was ~5-20% busy, so it was no longer the bottleneck: the GPU
itself got slow. The buffers the CPU reads back are small but written heavily by shaders
(likely histograms or counters with atomics), and with shared memory every one of those
accesses crosses PCIe to system memory. Shared memory stays parked; the right target is data
the GPU writes once and the CPU reads, which is not what TLG reads back.

### Phase 5: cleanup
- Remove superseded paths (old pending-fence list, shortcut levels that lost), update both
  documents, commit as "TLG readback perf 3".
- Optional: turn the findings into an upstream RFC (related: #4316, #3404).

## 7. Risks and mitigations

| Risk | Mitigation |
| :--- | :--- |
| A game depends on the old early-fence behavior in a way the timeline doesn't model | Everything stays opt-in per game; levels 0-3 allow quick A/B; hang checks log the stuck wait |
| Guest CPU writes to a label misdetected as stale or fresh | Keep landed values; log every CPU override detection during testing; fall back to polling memory |
| Deadlock between the CP thread, the priority thread and guest threads | Never block the CP thread on priority work except in `Finish`; keep 2 s hang checks; test loading transitions |
| Barrier cost | One full barrier per satisfied wait, merged when adjacent; measure with the stats |
| Design assumes one Vulkan queue | True today. If upstream adds async compute queues, class A needs timeline semaphores between queues instead of barriers |
| Class C (shader-written labels) dominates | Phase 0 finds out early; this plan would then be the wrong lever and should stop there |

## 8. Validation checklist (each phase)
- Scenes: title screen, gates area (FPS reference), removing the spear, climbing Trico, outdoor
  lighting, one area load, 10 minutes of free play.
- Logs: no `Hang check`, no `Critical`, `readback_stats.csv` compared with the previous phase.
- Regression: one other game with default settings (features off) must behave exactly as before.

## 9. Effort estimate
| Phase | Size | Test runs needed |
| :--- | :--- | :--- |
| 0 | Small | 2-3 captures |
| 1 | Medium | 2 runs |
| 2 | Small-medium | 1 validation run + 1 play run |
| 3 | Medium, depends on the crash cause | 2-4 runs |
| 4 | Small (code exists) | 1-2 runs |
| 5 | Small | - |

## 10. Open questions
- Which labels do TLG's compute rings wait on, and who writes them? (Phase 0)
- Does TLG rely on `MemSemaphore` or `Rewind` in hot paths? (Phase 0)
- Is the remaining CP busy time after this plan the command recording itself (~2,800 draws and
  ~950 dispatches per frame)? If so, the next lever is CPU-side draw recording cost, not
  synchronization.
