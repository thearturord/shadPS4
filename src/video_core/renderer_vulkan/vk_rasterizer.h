// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/amdgpu/draw_state.h"
#include <mutex>
#include <vector>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "video_core/native_draw_tag.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/readback_stats.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;
struct FixedState;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    /// With the command processor split (cp_split_mode) the pipeline lookup is done by the command
    /// processor thread (PlanDraw / PlanDispatch) and the raster thread gets its result.
    void Draw(bool is_indexed, u32 index_offset = 0, const DrawPlan* plan = nullptr);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset,
                      const DrawPlan* plan = nullptr);

    void DispatchDirect(const DispatchPlan* plan = nullptr);
    void DispatchIndirect(VAddr address, u32 offset, u32 size,
                          const DispatchPlan* plan = nullptr);

    /// Command processor thread: looks the pipeline up for `regs`. Returns null for draws that
    /// FilterDraw handles without a pipeline (the raster thread repeats that decision).
    /// `native` is the fixed-function state the native path built from the engine's objects (null: built from `regs`).
    [[nodiscard]] std::unique_ptr<DrawPlan> PlanDraw(const AmdGpu::DrawRegs& regs,
                                                     DrawIndirectParams params,
                                                     const FixedState* native = nullptr,
                                                     u32 native_stale = 0, bool verify = true);
    [[nodiscard]] std::unique_ptr<DispatchPlan> PlanDispatch(const AmdGpu::ComputeProgram& cs);

    /// Raster thread: hands a finished plan back for reuse. Allocating a plan for every draw on
    /// one thread and freeing it on another made them fight over the heap lock.
    void RecyclePlan(std::unique_ptr<DrawPlan>&& plan);
    void RecyclePlan(std::unique_ptr<DispatchPlan>&& plan);

    /// cp_flush_commands: after a draw or dispatch, hands the commands recorded so far to the
    /// encoder thread once there are enough of them.
    void MaybeFlushRecorded();

    /// A guest memory write done by the command processor itself.
    void NoteCpGuestWrite(VAddr addr, u64 size) {
        buffer_cache.NoteCpGuestWrite(addr, size);
    }

    /// Orders all later GPU work after everything recorded so far, for a command stream wait
    /// that passed because its producer (a fence recorded at `producer_tick`) was already
    /// recorded, not because it finished. Returns whether a barrier was recorded.
    bool CommandStreamBarrier(u64 producer_tick);

    [[nodiscard]] u64 CurrentTick() const noexcept;

    PipelineCache::PlanStats TakePlanStats() noexcept {
        return pipeline_cache.TakePlanStats();
    }

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    void SetDispatchCaller(u64 caller) {
        dispatch_caller = caller;
    }
    /// A memory fill (a pattern of up to four dwords repeated over `bytes`): when an image of the fill's size lives at the
    /// address the fill is an image clear. SetPendingFill covers the next dispatch.
    bool TryClearImage(VAddr dst, u32 bytes, const u32* pattern, u32 dwords);
    void SetPendingFill(VAddr dst, u32 bytes, const u32* pattern, u32 dwords) {
        pending_fill = {dst, bytes, dwords, {pattern[0], dwords > 1 ? pattern[1] : 0, dwords > 2 ? pattern[2] : 0, dwords > 3 ? pattern[3] : 0}};
    }
    /// The memory copy of two images that live on the GPU with the same layout is an image copy.
    bool TryCopyImages(VAddr dst, VAddr src, u32 num_bytes);
    /// Clears the depth / stencil target the registers describe (native depth clear record).
    void ClearDepthStencilTarget();
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);

    /// Native draw path: the tag the game wrote for the draw about to be executed (raster thread only).
    void SetNativeTag(const VideoCore::NativeTag& tag) {
        native_tag = tag;
    }
    /// Anything other than a tagged draw ran: the reuse state is stale.
    void NativeInvalidate() {
        native_last.valid = false;
    }
    struct NativeStats {
        u64 tagged{};
        u64 untagged{};
        u64 dynamic_skipped{};
        u64 bind_skipped{};
    };
    NativeStats TakeNativeStats() {
        return std::exchange(native_stats, {});
    }
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();

    void ProcessDownloadImages();

    /// Handles readbacks at an end-of-pipe fence and runs `signal` once their data is in guest
    /// memory. With async fences the GPU thread doesn't wait and `signal` runs on another thread;
    /// `must_sync` forces the blocking path (for fences whose signal needs the GPU thread).
    void OnFence(Common::UniqueFunction<void>&& signal, VideoCore::FenceKind kind,
                 bool must_sync = false);

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline, const FixedState& fixed);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();
    /// Clears the bound depth / stencil target to the clear values of the registers (native replacement of the SDK's clear quad).
    void ClearDepthTarget();

    const FixedState& FixedFor(const DrawPlan* plan, FixedState& storage) const;
    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed,
                            const FixedState& fixed) const;
    void UpdateViewportScissorState(const FixedState& fixed) const;
    void UpdateDepthStencilState(const FixedState& fixed) const;
    void UpdatePrimitiveState(bool is_indexed, const FixedState& fixed) const;
    void UpdateRasterizationState(const FixedState& fixed) const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline, const FixedState& fixed) const;

    bool FilterDraw(const DrawPlan* plan = nullptr);
    const GraphicsPipeline* ResolveDrawPipeline(const DrawPlan* plan,
                                                DrawIndirectParams params = {});
    const ComputePipeline* ResolveDispatchPipeline(const DispatchPlan* plan);

    enum class DrawClass : u8 {
        Normal,
        EliminateFastClear,
        FmaskDecompress,
        Resolve,
        PrimitiveNone,
        DepthStencilCopy,
        SkipShadow, ///< Live switch skip_shadows: a depth-only draw into a shadow map sized target.
        DepthTransition, ///< The SDK's full-screen triangle that decompresses / resummarizes depth: no effect without compression.
        DepthClear, ///< The SDK's full-screen triangle with the depth / stencil clear bits set: a clear of the target.
    };
    struct DrawClassification {
        DrawClass kind{};
        bool depth_copy{};
        bool stencil_copy{};
    };
    /// What FilterDraw does with a draw, decided from the registers only. `executed` is the fixed-function state of a draw
    /// the native path executed from its record: such a draw is a normal draw by construction, only the shadow skip
    /// switch looks at it.
    static DrawClassification ClassifyDraw(const AmdGpu::DrawRegs& regs,
                                           const FixedState* executed = nullptr);

    void ProcessReadbacksSync();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline, const FixedState* fixed = nullptr);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0, bool is_indirect = false);

    void ResetBindings(bool is_compute);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    VideoCore::NativeTag native_tag{};
    struct NativeLast {
        bool valid{};
        u64 k2{};
        const GraphicsPipeline* pipeline{};
        u64 tick{};
    } native_last{};
    NativeStats native_stats{};

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;
    const bool readback_batching;
    const bool async_fences;
    u64 recorded_work{};
    u64 last_barrier_work{~0ULL};

    /// rt_lookup_reuse_enabled: the register inputs of the last target lookup of each color slot
    /// (and the depth target, last entry) and the texture cache generation it was made at.
    struct CachedTarget {
        bool valid{};
        u64 generation{};
        u32 size{};
        std::array<u8, 256> inputs{};
    };
    std::array<CachedTarget, AmdGpu::NUM_COLOR_BUFFERS + 1> rt_cache{};
    /// Sampled images (BindTextures): the result of FindImage for a T# and the way the shader uses it, valid while
    /// no image was registered or removed (native_draw_mode >= 2).
    struct CachedTextureLookup {
        bool valid{};
        u64 generation{};
        VideoCore::ImageId image_id{};
        std::array<u64, 6> key{};
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<CachedTextureLookup, 512> texture_lookup_cache{};
    const bool rt_reuse_enabled;
    const u32 encoder_debug;
    /// Encoder thread: fences without readbacks wait for all work recorded before them.
    const bool full_eop_fences;
    /// Hand recorded commands to the encoder thread every this many commands (0 = off).
    const u32 flush_commands;
    std::mutex plan_pool_mutex;
    std::vector<std::unique_ptr<DrawPlan>> draw_plan_pool;
    struct PendingFill {
        VAddr dst{};
        u32 bytes{};
        u32 dwords{};
        std::array<u32, 4> pattern{};
    };
    PendingFill pending_fill{};
    u64 dispatch_caller{}; ///< diagnostics: the game call that made the dispatch being bound
    u8 clip_plane_mask{}; ///< user clip planes of the draw being bound
    std::vector<std::unique_ptr<DispatchPlan>> dispatch_plan_pool;
    /// True if the inputs match the cached lookup; otherwise stores them for the next draw.
    template <typename... Ts>
    bool TargetInputsMatch(CachedTarget& cache, const Ts&... parts);

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    ImageBinding db_desc;

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        vk::AccessFlags2 src_access;
    };
    std::vector<BoundBuffer> bound_buffers;
    std::vector<VideoCore::ImageId> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    bool attachment_feedback_loop{};
    bool needs_barrier{};
};

} // namespace Vulkan
