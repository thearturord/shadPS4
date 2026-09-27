// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include "common/types.h"

// Diagnostic counters for GPU readbacks and related stalls. Enabled per game with the
// `readback_stats_enabled` GPU setting. Writes readback_stats.csv (one line per second) and
// readback_top.txt (buffers/images/shaders ranked by stall time) to the log directory.
namespace VideoCore::ReadbackStats {

/// Pseudo stage ids used for writers that are not shaders.
constexpr u32 WriterFill = 0xFD;
constexpr u32 WriterCopy = 0xFE;

namespace Detail {
extern std::atomic<bool> enabled;
extern std::atomic<u32> cp_packet;
extern std::atomic<const char*> cp_phase;
extern thread_local bool is_cp_thread;
extern thread_local u32 cp_current;
extern thread_local u64 cp_since;
extern thread_local u64 cp_packets;
extern thread_local u32 cp_yield_kind;
extern std::atomic<u64> cp_time_ns[];
} // namespace Detail


[[nodiscard]] inline bool IsEnabled() noexcept {
    return Detail::enabled.load(std::memory_order_relaxed);
}

[[nodiscard]] inline u64 NowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Called by the command processor thread once, so phases are only recorded for it.
inline void MarkCpThread() noexcept {
    Detail::is_cp_thread = true;
    Detail::cp_since = NowNs();
}

/// Records the packet the command processor is working on (queue 0 = graphics, n = compute
/// ring n - 1), reported by the stall watchdog if the command processor stops making progress.
inline void CpPacket(u32 queue, u32 opcode) noexcept {
    if (IsEnabled()) {
        Detail::cp_packet.store((queue << 16) | (opcode & 0xFFFF), std::memory_order_relaxed);
        ++Detail::cp_packets;
    }
}

/// Command processor time categories. Time is exclusive: a step nested in another one (an
/// upload inside a draw, say) only counts for the inner step.
enum class CpTime : u32 {
    Other,          ///< Packet parsing, register writes and anything not listed below.
    Draw,           ///< Recording draws, minus the nested steps below.
    Dispatch,       ///< Recording dispatches, minus the nested steps below.
    PipelineLookup, ///< Pipeline key refresh, lookup and compilation.
    Upload,         ///< Buffer and image uploads.
    Fence,          ///< EOP/EOS/ReleaseMem: readback copies, submits.
    SubmitEnd,      ///< End of a guest submission: readbacks and submit.
    Fault,          ///< Page fault handlers running on the command processor thread.
    Protect,        ///< Page protection changes.
    GpuWait,        ///< Waiting for the GPU (Finish, stream buffers, write backs).
    GuestCommand,   ///< Work sent by guest threads (SendCommand).
    FlipSleep,      ///< Sleeping on a video out (flip) label.
    Idle,           ///< Nothing submitted.
    // Steps of recording a draw or dispatch (nested in Draw/Dispatch):
    Prepare,        ///< Pending operations, draw filtering and render state setup.
    ComputeChecks,  ///< Checks for compute image copies and metadata/image clears.
    BindBuffers,    ///< Buffer bindings of all stages (buffer cache lookups).
    BindTextures,   ///< Texture and sampler bindings of all stages (texture cache lookups).
    RenderTargets,  ///< Color and depth target lookups (BeginRendering).
    VertexIndex,    ///< Vertex and index buffer bindings.
    Descriptors,    ///< Descriptor writes, barriers and push constants.
    DynamicState,   ///< Dynamic state (viewports, depth, blending...).
    BeginPass,      ///< Starting or continuing the render pass.
    // Steps of buffer and texture bindings (nested in BindBuffers/BindTextures):
    BufStreamCopy,  ///< Copying small read-only buffers into the stream buffer.
    BufLookup,      ///< GPU-modified checks and buffer cache lookups.
    BufGpuMark,     ///< Marking written ranges as GPU modified (buffers and textures).
    TexFindImage,   ///< Texture cache image lookups.
    TexViewLayout,  ///< Image view lookups and layout transitions.
    TexSampler,     ///< Sampler lookups.
    StatsOverhead,  ///< Readback stats bookkeeping (writer attribution).
    Count,
};

/// Measures command processor time for a category (see CpTime). Does nothing on other threads.
class CpTimer {
public:
    explicit CpTimer(CpTime category) noexcept {
        if (IsEnabled() && Detail::is_cp_thread) {
            const u64 now = NowNs();
            Detail::cp_time_ns[Detail::cp_current].fetch_add(now - Detail::cp_since,
                                                             std::memory_order_relaxed);
            prev = Detail::cp_current;
            Detail::cp_current = static_cast<u32>(category);
            Detail::cp_since = now;
            active = true;
        }
    }
    ~CpTimer() {
        if (active) {
            const u64 now = NowNs();
            Detail::cp_time_ns[Detail::cp_current].fetch_add(now - Detail::cp_since,
                                                             std::memory_order_relaxed);
            Detail::cp_current = prev;
            Detail::cp_since = now;
        }
    }
    CpTimer(const CpTimer&) = delete;
    CpTimer& operator=(const CpTimer&) = delete;

private:
    u32 prev{};
    bool active{};
};

/// Called right before a command stream wait yields (`flip` = waiting on a video out label).
inline void CpWaitYield(bool flip) noexcept {
    if (Detail::is_cp_thread) {
        Detail::cp_yield_kind = flip ? 2 : 1;
    }
}

/// Records time the command processor spent resuming a queue that only re-checked its wait.
void OnCpSpin(bool flip, u64 ns);

/// Records a Vulkan queue submission of the rasterizer.
void OnVkSubmit();

/// Records an asynchronous fence and whether it had to submit the command buffer.
void OnAsyncFence(bool submitted);

/// Records a small read-only buffer copied into the stream buffer at bind time.
void OnStreamCopy(u64 bytes);

/// Records GPU execution time of a command buffer of the rasterizer (timestamp queries).
void OnGpuBusy(u64 ns);

/// Records CPU time of one Presenter::Present call.
void OnPresentCpu(u64 ns);

/// Marks a potentially blocking step on the command processor thread for the stall watchdog.
/// Does nothing on other threads.
class CpPhase {
public:
    explicit CpPhase(const char* name) noexcept {
        if (IsEnabled() && Detail::is_cp_thread) {
            prev = Detail::cp_phase.exchange(name, std::memory_order_relaxed);
            active = true;
        }
    }
    ~CpPhase() {
        if (active) {
            Detail::cp_phase.store(prev, std::memory_order_relaxed);
        }
    }
    CpPhase(const CpPhase&) = delete;
    CpPhase& operator=(const CpPhase&) = delete;

private:
    const char* prev{};
    bool active{};
};

/// How a command stream wait (WaitRegMem) was resolved.
enum class WaitClass : u32 {
    FencePass,    ///< Passed right away: a fence already recorded writes the awaited value.
    Producer,     ///< Blocked until another ring recorded the fence (cross queue dependency).
    FenceBlocked, ///< Blocked until a held back fence reached guest memory.
    Memory,       ///< Blocked on a value no recorded fence writes (guest CPU or shader write).
    Count,
};

/// Reads the setting and starts the reporter thread when enabled.
void Start();

/// Stops the reporter thread and writes the final report.
void Stop();

void OnPresent();
void OnSubmit();
void OnDraw();
void OnDispatch();
void OnFault(bool is_write, u64 handler_ns);
void OnProtect(u64 ns);
void OnFinish(u64 wait_ns);
void OnGpuThreadIdle(u64 idle_ns);

/// Records which shader (or fill/copy) last wrote a guest memory range on the GPU.
void RecordWriter(VAddr base, u64 size, u64 shader_hash, u32 stage);

/// Records a buffer readback caused by a CPU access to GPU modified memory.
void OnBufferReadback(VAddr fault_addr, u64 bytes, u64 finish_ns, u64 blocked_ns,
                      bool write_triggered);

/// Records a batched readback that completed buffers and images with a single GPU wait.
void OnBatch(u32 num_buffers, u32 num_images, u64 bytes, u64 wait_ns);

/// Records a command stream wait on memory (WaitRegMem) that did not pass right away.
void OnCommandWait(bool is_compute, VAddr address, WaitClass cls, u64 wait_ns);

/// Records a MemSemaphore or Rewind wait in a command stream.
void OnSemaphoreWait(u64 wait_ns);

/// Records a GPU barrier inserted for a wait that passed on a recorded fence.
void OnStreamBarrier();

/// Records time spent waiting for the GPU to free space in a stream buffer.
void OnStreamWait(u64 wait_ns);

/// Records a linear image readback.
void OnImageReadback(VAddr addr, u64 bytes, u32 width, u32 height, u32 num_bits, u64 finish_ns);

} // namespace VideoCore::ReadbackStats
