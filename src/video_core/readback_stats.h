// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include "common/types.h"

// Diagnostic counters for GPU readbacks and related stalls. Enabled per game with the
// `readback_stats_enabled` GPU setting. Writes readback_stats.csv (one line per second) and
// readback_top.txt (buffers/images/shaders ranked by stall time) to the log directory.
namespace VideoCore {

/// Where a guest fence comes from (bit values, cp_full_fence_kinds).
enum FenceKind : u32 {
    FenceGfxEop = 1,         ///< EventWriteEop on the graphics queue.
    FenceGfxEos = 2,         ///< EventWriteEos on the graphics queue.
    FenceComputeRelease = 4, ///< ReleaseMem on a compute queue.
    FenceHeldWrite = 8,      ///< Label write queued behind a fence that has not landed.
    FenceSubmitEnd = 16,     ///< End of a guest submission (no guest fence attached).
};
inline constexpr u32 NumFenceKinds = 5;

} // namespace VideoCore

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
extern thread_local u32 wait_reason;
extern thread_local u64 wait_addr;
extern thread_local u64 wait_size;
extern thread_local u64 call_hash;
extern thread_local u32 wait_origin;
extern thread_local u32 fault_origin;
} // namespace Detail

/// Origin of a wait that no command processor step caused (a guest thread's fault).
inline constexpr u32 GuestWaitOrigin = ~0U;

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

/// Why a thread waits for the GPU (profiler: CSV columns and the GPU waits report section).
enum class WaitReason : u32 {
    Other,              ///< Finish or wait without a recorded reason.
    ReadbackCpRead,     ///< Buffer readback: the command processor read GPU-written memory.
    ReadbackCpWrite,    ///< ... wrote to GPU-written memory.
    ReadbackGuestRead,  ///< Buffer readback for a read fault of a guest thread.
    ReadbackGuestWrite, ///< ... for a write fault of a guest thread.
    InflightDownload,   ///< Waiting for an asynchronous readback that was already submitted.
    InflightCpRead,     ///< ... for a read of the command processor.
    InflightCpWrite,    ///< ... for a write of the command processor.
    InflightGuestRead,  ///< ... for a read fault of a guest thread.
    InflightGuestWrite, ///< ... for a write fault of a guest thread.
    ImageReadback,      ///< Synchronous linear image readback.
    ReadbackBatch,      ///< Readbacks of a synchronous fence or submission end (one GPU wait).
    SyncFence,          ///< Fence handled synchronously (earlier async fences drain).
    GdsStore,           ///< EOS GDS store: GDS is read after the GPU finished.
    StreamBuffer,       ///< Stream or staging buffer space still in use by the GPU.
    FaultBuffer,        ///< Fault buffer processing.
    Count,
};
inline constexpr u32 NumWaitReasons = static_cast<u32>(WaitReason::Count);

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
    Prepare,       ///< Pending operations, draw filtering and render state setup.
    ComputeChecks, ///< Checks for compute image copies and metadata/image clears.
    BindBuffers,   ///< Buffer bindings of all stages (buffer cache lookups).
    BindTextures,  ///< Texture and sampler bindings of all stages (texture cache lookups).
    RenderTargets, ///< Color and depth target lookups (BeginRendering).
    VertexIndex,   ///< Vertex and index buffer bindings.
    Descriptors,   ///< Descriptor writes, barriers and push constants.
    DynamicState,  ///< Dynamic state (viewports, depth, blending...).
    BeginPass,     ///< Starting or continuing the render pass.
    // Steps of buffer and texture bindings (nested in BindBuffers/BindTextures):
    BufStreamCopy, ///< Copying small read-only buffers into the stream buffer.
    BufLookup,     ///< GPU-modified checks and buffer cache lookups.
    BufGpuMark,    ///< Marking written ranges as GPU modified (buffers and textures).
    TexFindImage,  ///< Texture cache image lookups.
    TexViewLayout, ///< Image view lookups and layout transitions.
    TexSampler,    ///< Sampler lookups.
    StatsOverhead, ///< Readback stats bookkeeping (writer attribution).
    // Time inside the Vulkan driver (nested anywhere above):
    VulkanCmd,    ///< vkCmd* calls (recording commands, barriers, descriptors, state).
    VulkanSubmit, ///< vkQueueSubmit.
    VulkanOther,  ///< Command buffer begin/end/allocation, object creation, queries.
    StreamMemcpy, ///< The memory copy inside stream copies (nested in BufStreamCopy).
    // Steps of readback handling at fences (nested in Fence/SubmitEnd):
    FenceImageCopies, ///< Recording linear image readback copies.
    FencePrefetch,    ///< Recording readback copies of buffer ranges the CPU reads back.
    FenceSubmit,      ///< Submitting the command buffer so the readbacks start on the GPU.
    // Steps of the pipeline lookup (nested in PipelineLookup):
    PipelineStages,  ///< Shader program lookups and specialization of each stage.
    PipelineMap,     ///< Hashing the pipeline key and looking it up.
    PipelineCompile, ///< Creating pipelines not seen before.
    // Steps of render target handling:
    RtPrepare, ///< Color and depth target image lookups (PrepareRenderState, nested in Prepare).
    RtViews,   ///< Render target view lookups (nested in RenderTargets).
    // Steps of a shader stage lookup (GetProgram, nested in PipelineStages):
    StageRuntimeInfo, ///< Building the stage's runtime info from registers.
    StageFlatBuf,     ///< Re-reading the shader's user data pointers (RefreshFlatBuf).
    StageSpec,        ///< Building the specialization (reading buffer/image/sampler sharps).
    StageFind,        ///< Finding the matching variant and adding its bindings.
    Count,
};

using CpTimeArray = std::array<u64, static_cast<size_t>(CpTime::Count)>;

/// Events counted on the command processor thread, attributed to draws and dispatches.
struct CpEventCounters {
    u64 upload_bytes{};      ///< Buffer data uploaded from guest memory.
    u64 image_uploads{};     ///< Images refreshed from guest memory.
    u64 protect_calls{};     ///< Page protection changes.
    u64 faults{};            ///< Page faults handled on the command processor thread.
    u64 buffers_created{};   ///< Buffer cache buffers created (or merged).
    u64 gpu_marked_bytes{};  ///< Bytes marked as GPU modified (written buffers).
    u64 stream_copy_bytes{}; ///< Small read-only buffers copied into the stream buffer.
    u64 readback_bytes{};    ///< Synchronous buffer readbacks.

    CpEventCounters& operator+=(const CpEventCounters& o) noexcept {
        upload_bytes += o.upload_bytes;
        image_uploads += o.image_uploads;
        protect_calls += o.protect_calls;
        faults += o.faults;
        buffers_created += o.buffers_created;
        gpu_marked_bytes += o.gpu_marked_bytes;
        stream_copy_bytes += o.stream_copy_bytes;
        readback_bytes += o.readback_bytes;
        return *this;
    }
    CpEventCounters operator-(const CpEventCounters& o) const noexcept {
        return {
            .upload_bytes = upload_bytes - o.upload_bytes,
            .image_uploads = image_uploads - o.image_uploads,
            .protect_calls = protect_calls - o.protect_calls,
            .faults = faults - o.faults,
            .buffers_created = buffers_created - o.buffers_created,
            .gpu_marked_bytes = gpu_marked_bytes - o.gpu_marked_bytes,
            .stream_copy_bytes = stream_copy_bytes - o.stream_copy_bytes,
            .readback_bytes = readback_bytes - o.readback_bytes,
        };
    }
};

namespace Detail {
/// Never reset, so a draw or dispatch can take the difference before and after it.
extern thread_local CpTimeArray cp_local_ns;
extern thread_local CpEventCounters cp_events;
} // namespace Detail

/// Event counters of the command processor thread, or nullptr on other threads.
[[nodiscard]] inline CpEventCounters* CpEvents() noexcept {
    return IsEnabled() && Detail::is_cp_thread ? &Detail::cp_events : nullptr;
}

/// Command processor time per category so far, including the step in progress.
[[nodiscard]] inline CpTimeArray CpTimeSnapshot() noexcept {
    CpTimeArray totals = Detail::cp_local_ns;
    totals[Detail::cp_current] += NowNs() - Detail::cp_since;
    return totals;
}

/// Measures command processor time for a category (see CpTime). Does nothing on other threads.
class CpTimer {
public:
    explicit CpTimer(CpTime category) noexcept {
        if (IsEnabled() && Detail::is_cp_thread) {
            const u64 now = NowNs();
            Detail::cp_time_ns[Detail::cp_current].fetch_add(now - Detail::cp_since,
                                                             std::memory_order_relaxed);
            Detail::cp_local_ns[Detail::cp_current] += now - Detail::cp_since;
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
            Detail::cp_local_ns[Detail::cp_current] += now - Detail::cp_since;
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

/// Tags the GPU waits in its scope with a reason and the guest memory range involved. The
/// innermost scope wins.
class WaitScope {
public:
    /// `origin` is the command processor step (CpTime) that caused the wait, by default the
    /// one running now; GuestWaitOrigin for work done for a guest thread.
    explicit WaitScope(WaitReason reason, u64 addr = 0, u64 size = 0,
                       u32 origin = Detail::cp_current) noexcept {
        if (IsEnabled()) {
            active = true;
            prev_reason = Detail::wait_reason;
            prev_addr = Detail::wait_addr;
            prev_size = Detail::wait_size;
            prev_origin = Detail::wait_origin;
            Detail::wait_reason = static_cast<u32>(reason);
            Detail::wait_addr = addr;
            Detail::wait_size = size;
            Detail::wait_origin = origin;
        }
    }
    ~WaitScope() {
        if (active) {
            Detail::wait_reason = prev_reason;
            Detail::wait_addr = prev_addr;
            Detail::wait_size = prev_size;
            Detail::wait_origin = prev_origin;
        }
    }
    WaitScope(const WaitScope&) = delete;
    WaitScope& operator=(const WaitScope&) = delete;

private:
    u32 prev_reason{};
    u64 prev_addr{};
    u64 prev_size{};
    u32 prev_origin{};
    bool active{};
};

/// Called by the page fault handler before it starts timing: remembers the command processor
/// step that faulted, for the attribution of the readback wait that follows.
inline void NoteFaultOrigin() noexcept {
    if (Detail::is_cp_thread) {
        Detail::fault_origin = Detail::cp_current;
    }
}

/// Sets the shader of the draw or dispatch being recorded, for the attribution of GPU waits
/// (0 = none).
inline void SetCallShader(u64 hash) noexcept {
    Detail::call_hash = hash;
}

/// Records time the calling thread spent waiting for the GPU, under the current WaitScope.
void OnGpuWait(u64 ns);

/// Records time the command processor spent resuming a queue that only re-checked its wait.
void OnCpSpin(bool flip, u64 ns);

/// Kind of command a shader was recorded for (per-shader profile).
enum class ShaderCallKind : u32 {
    Draw,
    DrawIndirect,
    Dispatch,
    DispatchIndirect,
    DispatchHle,     ///< Handled by the emulator without running the shader (ExecuteShaderHLE).
    DispatchSkipped, ///< Skipped as an image copy or clear the texture cache handles.
};

/// One draw or dispatch, for the per-shader profile in readback_top.txt.
struct ShaderCall {
    ShaderCallKind kind{};
    u64 hash{};    ///< Compute shader, or vertex shader for draws.
    u64 ps_hash{}; ///< Pixel shader for draws (0 if none).
    u64 work{};    ///< Workgroups (dispatch) or vertices x instances (draw); 0 if indirect.
    u32 threads_per_group{};
    u16 buffers{};
    u16 written_buffers{};
    u16 images{};
    u16 storage_images{};
    u16 color_targets{};
    bool depth_target{};
    u64 ns{}; ///< Command processor time of the call, including nested work (uploads...).
    CpTimeArray category_ns{}; ///< Where that time went.
    CpEventCounters events{};  ///< What happened during the call.
};

/// Records a draw or dispatch for the per-shader profile.
void OnShaderCall(const ShaderCall& call);

/// Records a replay of recorded commands into a command buffer (cp_encoder_mode).
void OnCommandReplay(u64 commands, u64 ns);

/// Records a Vulkan queue submission of the rasterizer.
void OnVkSubmit();

/// Records a submission of the rasterizer's scheduler (command list shape: draws and dispatches
/// per command buffer).
void OnRasterizerSubmit();

/// Records a render pass started by the rasterizer's scheduler.
void OnRenderPassBegin();

/// Records buffer data uploaded from guest memory.
void OnUpload(u64 bytes);

/// Records pages that became hot (readback_hot_write_pages_enabled) and hot page decays.
void OnHotPages(u32 count);
void OnHotPageDecay();

/// Records a buffer created by the buffer cache: the range asked for, the range created, and the
/// buffers merged into it (their contents are copied on the GPU).
void OnBufferCreated(VAddr wanted_begin, VAddr wanted_end, VAddr begin, VAddr end, u32 num_merged,
                     u64 merged_bytes, bool stream_leap);

/// Hazards a deferred constant copy (multi-core plan) would have to wait for.
enum class Hazard : u32 {
    CpGuestWrite,   ///< The command processor wrote guest memory.
    CpWriteOverlap, ///< ... overlapping constant data copied in the same command buffer.
    GpuMarkOverlap, ///< A range marked GPU-written overlaps constant data of the same buffer.
    Count,
};
void OnHazard(Hazard hazard);

/// How much per-draw work repeats the previous call (to size skip-unchanged optimizations).
enum class Reuse : u32 {
    DrawChecked,             ///< Draws compared with the previous draw.
    DrawPipelineSame,        ///< ... same pipeline.
    DrawCtxRegsSame,         ///< ... identical context registers (all 1024).
    DrawRtRegsSame,          ///< ... identical render target registers.
    DrawUserDataSame,        ///< ... identical shader programs and user data of all stages.
    DrawAllSame,             ///< ... same pipeline, context registers and user data.
    DispatchChecked,         ///< Dispatches compared with the previous dispatch.
    DispatchPipelineSame,    ///< ... same pipeline.
    DispatchUserDataSame,    ///< ... same pipeline and identical user data.
    StreamRepeatTickSame,    ///< Stream copy of a range already copied in this command buffer,
                             ///< with identical data.
    StreamRepeatTickChanged, ///< ... with different data.
    StreamRepeatOldSame, ///< Stream copy of a range copied in an older command buffer, same data.
    StreamRepeatOldChanged, ///< ... with different data.
    StreamRepeatSameBytes,  ///< Bytes of stream copies with identical data (reported in MB).
    StreamReused,           ///< Stream copies skipped by readback_stream_reuse_enabled.
    StreamReuseResets,      ///< Reuse tables dropped by a command processor write to guest memory.
    RtReused,               ///< Render target lookups skipped by rt_lookup_reuse_enabled.
    StageLookups,           ///< Shader stage lookups (GetProgram calls for known programs).
    StageVariantCompares,   ///< Variants compared to find the matching one.
    StageSameAsLast,        ///< Lookups that picked the same variant as the program's last one.
    StageFastPath,          ///< Lookups answered by the last variant check (stage_lookup_reuse).
    StageFastPathMismatch,  ///< Verify mode: the full lookup picked another variant.
    BindStageChecked,       ///< Draw/dispatch stages whose resources were compared with the last
                            ///< binding of the same stage slot.
    BindStageSameProgram,   ///< ... same shader as last time.
    BindStageAllSame,       ///< ... same shader and identical buffer, image and sampler sharps.
    BindBufferChecked,      ///< Buffer sharps compared with the same slot's last sharp.
    BindBufferSame,         ///< ... identical.
    BindImageChecked,       ///< Image sharps compared with the same slot's last sharp.
    BindImageSame,          ///< ... identical.
    BindSamplerChecked,     ///< Sampler sharps compared with the same slot's last sharp.
    BindSamplerSame,        ///< ... identical.
    Count,
};
void OnReuse(Reuse reuse, u64 amount = 1);

/// Records an asynchronous fence of the given kind and whether it had to submit the command
/// buffer.
void OnAsyncFence(FenceKind kind, bool submitted);

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

/// Number of presents since the start (frame index for per-frame statistics).
u64 FrameIndex();

/// Records a prefetch download of a hot buffer range (a range the CPU faulted on before).
/// `repeat` is set when the same range was already downloaded earlier in the same frame.
void OnPrefetch(VAddr range_start, u64 range_size, u64 bytes, bool repeat);

/// Records a CPU fault that made (or kept) a buffer range hot for prefetching.
void OnHotRangeFault(VAddr range_start, u64 range_size);

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
