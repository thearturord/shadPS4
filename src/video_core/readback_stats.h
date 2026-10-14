// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <string_view>
#include "common/types.h"

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

// The TLG fork has a large measurement layer (per-phase command processor timers, readback
// statistics, reuse counters) behind this interface. This port keeps only the call sites: every
// function here is empty and inlined away, so they cost nothing. NowNs and the thread flag are real
// because the code also uses them for hang checks and thread decisions.
namespace VideoCore::ReadbackStats {

enum class CpTime : u32 {
    Idle,
    Fence,
    FenceImageCopies,
    FencePrefetch,
    FenceSubmit,
    GuestCommand,
    SubmitEnd,
    FlipSleep,
    Draw,
    Dispatch,
    Prepare,
    Descriptors,
    PipelineLookup,
    PipelineStages,
    PipelineMap,
    PipelineCompile,
    StageSpec,
    StageFind,
    StageRuntimeInfo,
    StageFlatBuf,
    RenderTargets,
    RtViews,
    RtPrepare,
    BeginPass,
    DynamicState,
    VertexIndex,
    BufLookup,
    BufGpuMark,
    BindBuffers,
    BindTextures,
    TexSampler,
    TexFindImage,
    TexViewLayout,
    ComputeChecks,
    StatsOverhead,
    GpuWait,
    Upload,
    Protect,
    Fault,
    StreamMemcpy,
    BufStreamCopy,
    Count,
};

using CpTimeArray = std::array<u64, static_cast<size_t>(CpTime::Count)>;

enum class Mutation : u32 {
    PassBegin,
    BufferBarrier,
    ImageBarrier,
    StreamCopy,
    BufferGpuMark,
    BufferUpload,
    BufferCreated,
    ImageUpload,
    ImageCreated,
    TargetMarked,
    PipelineCompiled,
};

enum class CallClass : u32 {
    Draw,
    Dispatch,
};

enum class Reuse : u32 {
    RtReused,
    StageFastPath,
    StageFastPathMismatch,
    StageLookups,
    StageSameAsLast,
    StageVariantCompares,
    StreamReused,
    StreamReuseResets,
};

enum class WaitReason : u32 {
    SyncFence,
    GdsStore,
    ReadbackBatch,
    ImageReadback,
    FaultBuffer,
    StreamBuffer,
};

enum class WaitClass : u32 {
    Other,
    Producer,
    FencePass,
    FenceBlocked,
    Memory,
};

enum class ShaderCallKind : u32 {
    Draw,
    DrawIndirect,
    Dispatch,
    DispatchIndirect,
    DispatchSkipped,
    DispatchHle,
};

struct CpTimer {
    explicit CpTimer(CpTime) {}
};

struct CpPhase {
    explicit CpPhase(const char*) {}
};

struct WaitScope {
    template <typename... Args>
    explicit WaitScope(Args&&...) {}
};

struct MutationScope {
    void Finish(CallClass) {}
};

struct ShaderCall {
    u64 hash{};
};

namespace Detail {
/// True on the command processor threads (front, raster).
inline thread_local bool is_cp_thread = false;
} // namespace Detail

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline void MarkCpThread() {
    Detail::is_cp_thread = true;
}

inline bool IsEnabled() {
    return false;
}

inline void CpPacket(u32, u32) {}
inline void OnSemaphoreWait(u64) {}
inline void OnStreamBarrier() {}
inline void CpWaitYield(bool) {}
inline void OnCommandWait(bool, u64, WaitClass, u64) {}
inline void NoteMutation(Mutation) {}
inline void OnReuse(Reuse, u64 = 1) {}
inline void RegisterSampledThread(const char*) {}

} // namespace VideoCore::ReadbackStats
