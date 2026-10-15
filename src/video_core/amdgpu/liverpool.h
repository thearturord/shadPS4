// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/native_draw_tag.h"

#include <source_location>
#include <condition_variable>
#include <coroutine>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/draw_state.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/readback_stats.h"

namespace Vulkan {
class Rasterizer;
struct DrawPlan;
struct DispatchPlan;
} // namespace Vulkan

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

struct PM4CmdWaitRegMem;

struct Liverpool {
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    u32 index_buffer_num_indices{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    /// What the rasterizer reads while it executes draws and dispatches. Without the command
    /// processor split (cp_split_mode) these are the command processor's own registers; with it
    /// they are the raster thread's mirror, kept in sync through the operation queue.
    [[nodiscard]] DrawRegs& RasterRegs() noexcept {
        return *raster_state;
    }
    [[nodiscard]] std::array<CbDbExtent, NUM_COLOR_BUFFERS>& RasterCbExtents() noexcept {
        return *raster_cb_extent;
    }
    [[nodiscard]] CbDbExtent& RasterDbExtent() noexcept {
        return *raster_db_extent;
    }
    [[nodiscard]] ComputeProgram& RasterCsRegs() noexcept {
        return raster_cs ? *raster_cs : GetCsRegs();
    }
    /// True while the raster thread runs the rasterizer (cp_split_mode): the pipeline lookup is
    /// then done by the command processor thread.
    [[nodiscard]] bool SplitActive() const noexcept {
        return split_active;
    }

    void SubmitDone() noexcept {
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this] { return num_submits == 0 && RasterIdle(); });
    }

    bool IsGpuIdle() const {
        return num_submits == 0 && RasterIdle();
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_);

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if (split_active && std::this_thread::get_id() == front_id) {
            // The command processor thread read guest memory that faulted (GPU data): the
            // callback must see all earlier work executed, as it did before the split.
            DrainRaster();
        }
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func] {
                    func();
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            WakeRaster();
            sem.acquire();
        } else {
            std::scoped_lock lk{submit_mutex};
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
        WakeRaster();
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{64};

    std::thread::id GetGpuCommandProcessorThread() {
        return gpu_id;
    }

#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId() {
        return gpu_tid;
    }
#endif

private:
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessCeUpdate(std::span<const u32> ccb);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);

    void ProcessCommands();
    void Process(std::stop_token stoken);

    // ---- Command processor split (documents/tlg/cp-split-design.md) ----
    struct RasterSplit;
    struct RasterCall;
    struct RasterOp;
    /// Tag written by the game in front of the next draw (front thread only).
    VideoCore::NativeTag native_tag_pending{};
    u64 native_exec_calls{};
    u32 native_tag_skips{};
    void StartRasterSplit();
    void StopRasterSplit();
    void RasterMain(std::stop_token stoken);
    void ExecRasterCall(const RasterCall& call, const Vulkan::DrawPlan* draw_plan,
                        const Vulkan::DispatchPlan* dispatch_plan);
    /// Queues `func` (a closure) for the raster thread, or runs it now without the split.
    /// `writes_memory`: the closure writes guest memory that later pipeline lookups may read
    /// (descriptor tables, shader code), so the command processor thread must not plan a draw
    /// before it has run.
    void RasterRun(Common::UniqueFunction<void>&& func, bool wait = false,
                   bool writes_memory = false,
                   std::source_location site = std::source_location::current());
    /// Queues a draw or dispatch. Without the split it runs right away.
    void RasterDispatch(const RasterCall& call);
    /// Draws the pending native draw tag from its record instead of from command packets.
    void ExecuteNativeDraw(const void* header = nullptr, u32 asc_vqid = ~0u, u32 index_count = ~0u);
    bool native_follow_pending{}; ///< an executed draw waits for its draw packet (index count)
    void RasterPush(RasterOp&& op, bool wait);
    /// Blocks until every queued closure that writes guest memory has run.
    void DrainRasterWrites();
    /// A command processor write to guest memory, done in order with the draws.
    void CpWriteData(void* address, const u32* data, u32 data_size, u32 queue);
    /// Blocks until the raster thread has executed everything queued so far.
    void DrainRaster();
    [[nodiscard]] bool RasterIdle() const noexcept;
    /// Register writes made by the command processor: copied to the raster mirror.
    void MirrorRegs(u32 dword_offset, u32 count);
    void MirrorRegsRaw(u32 dword_offset, u32 count);
    void MirrorObject(const void* object, size_t size);
    void MirrorSetDefaults();

    /// Registers of executed native draws stay in the register file (and the raster mirror) between draws; only the
    /// registers that differ from the previous draw are written. `truth` keeps what the command packets wrote for
    /// every register that currently holds a record's value, so the packet path (and a fast clear eliminate) gets
    /// its own state back before it runs (front thread only).
    struct NativeOverlay {
        std::vector<u64> bits; ///< register indices that hold a record's value
        std::vector<u32> list;
    };
    NativeOverlay overlay;
    bool native_exec_scope{}; ///< the draw being dispatched is an executed record
    void OverlaySet(u32 index, u32 value);
    void PushDelta(u32 index, u32 value);
    void OverlayRestore();
    void OverlayForget(u32 dword_offset, u32 count);
    void OverlayForgetAll();
    /// Wakes the raster thread (it runs SendCommand callbacks) when the split is active.
    void WakeRaster();

    /// Label timeline: per fence label, the latest value a fence packet the command processor
    /// already recorded will write, while fence writes are held back (async readback fences
    /// write them once the GPU finished and the readbacks landed). A command stream wait only
    /// needs its producer to be recorded ahead of it in the single Vulkan queue, not executed,
    /// so it may pass on that value. When everything recorded has landed, guest memory is
    /// authoritative again, and so it is as soon as someone else writes the label meanwhile.
    struct LabelState {
        VAddr address;
        u32 num_bytes;
        u32 queue; ///< Queue of the latest fence: 0 = graphics, n = compute ring n - 1.
        u64 latest_value;
        u64 latest_seq; ///< Order in which the fence packets were processed.
        u64 gpu_tick;   ///< Scheduler tick holding the GPU work before the latest fence.
        u32 pending;    ///< Recorded fences whose write hasn't landed yet.
        bool overridden;
        /// Guest memory before the first held back fence, then every value recorded since.
        std::vector<u64> known_values;
    };
    static constexpr size_t MaxKnownLabelValues = 64;
    struct FenceMatch {
        bool satisfied{};
        VAddr label_address{}; ///< The matched label and the fence recorded last for it.
        u64 seq{};
        u64 gpu_tick{};
        u32 queue{};
        u32 value{};
        u32 initial{};
    };
    u64 RecordFence(VAddr address, u64 value, u32 num_bytes, u32 queue);
    /// Raster thread: the scheduler tick holding the work before fence `seq` of a label.
    void SetFenceTick(VAddr address, u64 seq);
    [[nodiscard]] u64 GetFenceTick(VAddr address, u64 seq);
    void RetireFence(VAddr address, u64 seq);
    FenceMatch MatchRecordedFence(const PM4CmdWaitRegMem& wait_reg_mem);

    /// readback_fence_wait_shortcut: 0 = off, 1 = graphics queue waits, 2 = graphics and compute
    /// queue waits, 3 = like 2 plus a GPU barrier that orders the work after the wait on the
    /// GPU (the wait passes on a recorded fence instead of a finished one).
    [[nodiscard]] bool FenceShortcutEnabled(bool is_compute) const noexcept {
        return fence_wait_shortcut != 0 && (!is_compute || fence_wait_shortcut >= 2);
    }

    /// Called when a wait passed on a recorded fence: orders the following GPU work after it.
    void OnWaitPassedOnFence(const PM4CmdWaitRegMem& wait_reg_mem, const FenceMatch& match,
                             u32 queue);

    /// Logs a command stream wait that has been blocked for more than 2 seconds.
    void ReportLongWait(const PM4CmdWaitRegMem& wait_reg_mem, const char* queue, u64 wait_start,
                        bool& reported);
    /// Logs a MemSemaphore/Rewind wait that has been blocked for more than 2 seconds.
    void ReportLongSpin(const char* what, const void* address, u64 wait_start, bool& reported);

    /// A command processor memory write (WriteData) normally lands when the packet is processed.
    /// If a fence write to the same label is still held back, the GPU would run it after that
    /// fence, so it is queued behind it instead. Returns false when it can be written now.
    bool DeferWriteBehindFences(void* address, const u32* data, u32 num_bytes, u32 queue);

    /// Runs `signal` once the readbacks before this fence are in guest memory, recording the
    /// fence value (if known) in the label timeline meanwhile.
    /// Fence latency per FenceKind (index = bit position): the time from parsing the fence packet
    /// to the raster thread starting it ("queue"), from there to the moment its signal is about to
    /// run, after the GPU and the readbacks ("gpu"), and the sum. Published by the debug server.
    struct FenceLatencyStat {
        std::atomic<u64> count{};
        std::atomic<u64> queue_ns{};
        std::atomic<u64> gpu_ns{};
        std::atomic<u64> total_ns{};
        std::atomic<u64> max_total_ns{};
    };
    std::array<FenceLatencyStat, VideoCore::NumFenceKinds> fence_latency;
    void NoteFenceLatency(VideoCore::FenceKind kind, u64 queue_ns, u64 gpu_ns);

    void SignalFenceAfterReadbacks(Common::UniqueFunction<void>&& signal, bool must_sync,
                                   VAddr address, u64 value, u32 num_bytes, u32 queue,
                                   VideoCore::FenceKind kind);

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    u32 num_mapped_queues{1u}; // GFX is always available

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    const bool guest_markers_enabled;
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
    std::thread::id front_id;
#ifdef __linux__
    u32 gpu_tid;
#endif
    s32 curr_qid{-1};

    /// The draw state of the command processor (kept equal to `regs`, planning reads it), what the rasterizer reads
    /// without the split (rebuilt from `regs` before each operation) and the raster thread's own with it.
    DrawRegs front_state{};
    DrawRegs direct_state{};
    DrawRegs* raster_state{&direct_state};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS>* raster_cb_extent{&last_cb_extent};
    CbDbExtent* raster_db_extent{&last_db_extent};
    ComputeProgram* raster_cs{};
    bool split_active{};
    std::unique_ptr<RasterSplit> split;
    const u32 fence_wait_shortcut;
    std::mutex recorded_fence_mutex;
    std::vector<LabelState> label_states;
    u64 next_fence_seq{};
    std::unordered_set<VAddr> logged_compute_passes;
    std::unordered_set<VAddr> logged_overrides;
    std::unordered_set<VAddr> logged_deferred_writes;
};

} // namespace AmdGpu
