// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>

#include "common/arch.h"
#ifdef ARCH_X86_64
#include <immintrin.h>
#endif
#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/native_draw.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/readback_policy.h"
#include "video_core/readback_stats.h"
#include "video_core/readback_trace.h"
#include "core/debug_server.h"
#include "video_core/thread_usage.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

static void WriteFenceMemory(void* address, u64 data, u32 num_bytes) {
    auto* memory = Core::Memory::Instance();
    if (!memory->TryWriteBacking(address, &data, num_bytes)) {
        memcpy(address, &data, num_bytes);
    }
}

static void NoFenceWrite(void*, u64, u32) {}

/// Writes a fence value from outside the GPU thread. Memory without a GPU mapping is never page
/// protected by the tracker, so it can be written directly (fence labels often live in plain
/// game memory). Returns false when the write must be done on the GPU thread instead.
static bool TryWriteOffThread(Vulkan::Rasterizer* rasterizer, void* address, const void* data,
                              u32 num_bytes) {
    if (Core::Memory::Instance()->TryWriteBacking(address, data, num_bytes)) {
        return true;
    }
    if (rasterizer && !rasterizer->IsMapped(reinterpret_cast<VAddr>(address), num_bytes)) {
        std::memcpy(address, data, num_bytes);
        return true;
    }
    return false;
}

static bool TryWriteFenceOffThread(Vulkan::Rasterizer* rasterizer, void* address, u64 data,
                                   u32 num_bytes) {
    return TryWriteOffThread(rasterizer, address, &data, num_bytes);
}

/// Runs a fence signal that may execute outside the GPU thread (async readback fences). Fence
/// writes there go through the memory backing so they can never fault into the page tracker,
/// which may be waiting on that thread. If the backing can't be written, the whole signal is
/// redone on the GPU thread instead.
template <typename Write, typename Signal>
static void RunFenceSignal(AmdGpu::Liverpool& liverpool, std::thread::id gpu_id, Write&& try_writes,
                           Signal&& full_signal) {
    if (std::this_thread::get_id() == gpu_id) {
        full_signal();
        return;
    }
    if (try_writes()) {
        return;
    }
    liverpool.SendCommand(std::forward<Signal>(full_signal));
}

// ---- Command processor split (documents/tlg/cp-split-design.md) ----

/// A draw or dispatch for the rasterizer, with everything it needs besides the registers.
struct Liverpool::RasterCall {
    enum class Kind : u8 { Draw, DrawIndirect, DispatchDirect, DispatchIndirect };
    Kind kind{};
    bool asc{};       ///< Marker names come from a compute ring (vqid argument).
    bool no_marker{}; ///< Run without a ScopeMarker.
    u32 vqid{};
    const void* cmd_address{}; ///< Only printed in markers, never dereferenced.
    const char* marker{};      ///< Format string literal.
    bool indexed{};
    u32 index_offset{};
    VAddr args_addr{};
    u32 offset{};
    u32 stride{};
    u32 max_count{};
    VAddr count_addr{};
    u16 vertex_sgpr_offset{};
    u16 instance_sgpr_offset{};
    u32 dispatch_size{};
    VideoCore::NativeTag tag{}; ///< From the game's native draw hook, draws only.
};

struct Liverpool::RasterOp {
    bool has_call{};
    bool writes_memory{};
    u32 write_line{}; ///< Source line of the RasterRun call that queued a writes_memory closure.
    RasterCall call{};
    Common::UniqueFunction<void> closure{};
    std::vector<u32> deltas{}; ///< Register writes to apply first: {offset, count, data...}.
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> cb_extent{};
    CbDbExtent db_extent{};
    ComputeProgram cs{};
    std::unique_ptr<Vulkan::DrawPlan> draw_plan{};
    std::unique_ptr<Vulkan::DispatchPlan> dispatch_plan{};
};

struct Liverpool::RasterSplit {
    static constexpr size_t MaxQueued = 256;
    static constexpr u32 SetDefaultsMarker = 0xFFFFFFFFu;

    std::jthread thread;
    std::jthread housekeeping; ///< Slow periodic work that must not stall the raster thread.
    std::mutex mutex;
    std::condition_variable_any work_cv; ///< Raster thread: work arrived.
    std::condition_variable_any done_cv; ///< Front: space in the queue or operations done.
    std::vector<RasterOp> queue; // swapped with the raster thread's batch: no per-op allocation
    std::vector<std::vector<u32>> free_deltas;
    std::atomic<u64> pushed{};
    std::atomic<u64> done{};
    std::atomic<u32> pending_writes{}; ///< Queued closures with writes_memory.
    /// What the front thread is doing (for the raster thread's idle statistics): 0 working,
    /// 1 waiting for the game to submit, 2 waiting for queued memory writes, 3 queue full.
    std::atomic<u8> front_state{0};
    std::array<std::atomic<u64>, 4> raster_idle_ns_by_front_state{};
    /// Idle time by the kind of the operation that ended it: draw, indirect draw, dispatch,
    /// indirect dispatch, closure (fence, write, flip, ...).
    std::array<std::atomic<u64>, 5> raster_idle_ns_by_next_op{};
    std::array<std::atomic<u64>, 5> raster_idle_count_by_next_op{};
    /// Statistics per RasterRun call site that queues memory-writing closures: how many, and how
    /// long the command processor waited for them before a pipeline lookup.
    struct WriteSite {
        std::atomic<u32> line{};
        std::atomic<u64> pushes{};
        std::atomic<u64> wait_ns{};
    };
    std::array<WriteSite, 24> write_sites;
    u32 last_write_line{}; ///< Command processor thread: line of the newest queued write closure.
    WriteSite* SiteFor(u32 line) {
        for (auto& site : write_sites) {
            u32 expected = 0;
            if (site.line.load(std::memory_order_relaxed) == line ||
                (site.line.compare_exchange_strong(expected, line) || expected == line)) {
                return &site;
            }
        }
        return &write_sites.back();
    }
    // Statistics, logged every few seconds.
    u64 stat_ops{};
    u64 stat_calls{};
    u64 stat_write_drains{};
    std::atomic<u64> stat_front_blocked{}; ///< Pushes that found the queue full.
    std::atomic<u64> stat_raster_idle_ns{};
    u64 stat_draws{};
    u64 stat_draws_no_reg_change{};
    // Raster thread wall time by kind of work (ns), logged every few seconds.
    u64 ns_draws{};
    u64 ns_depth_only_draws{}; ///< part of ns_draws spent in draws without a color target (z-prepass, shadows)
    u64 n_draws{};
    u64 n_depth_only_draws{};
    u64 ns_dispatches{};
    u64 ns_write_closures{};
    u64 ns_other_closures{};
    u64 ns_callbacks{};
    u64 stat_last_log_ns{};
    std::vector<u32> pending; ///< Front only: register writes not attached to an operation yet.
    bool verify{};
    u32 verify_reports{};

    std::unique_ptr<Regs> regs;
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> cb_extent{};
    CbDbExtent db_extent{};
    ComputeProgram cs{};
};

void Liverpool::BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
    if (!rasterizer_ && split_active) {
        // The rasterizer goes away: finish what is queued and stop the raster thread first.
        DrainRaster();
        split_active = false;
        StopRasterSplit();
        raster_regs = &regs;
        raster_cb_extent = &last_cb_extent;
        raster_db_extent = &last_db_extent;
        raster_cs = nullptr;
        split.reset();
    }
    rasterizer = rasterizer_;
    if (rasterizer && EmulatorSettings.GetCpSplitMode() != 0 && !split) {
        StartRasterSplit();
    }
}

void Liverpool::StartRasterSplit() {
    split = std::make_unique<RasterSplit>();
    auto& sp = *split;
    sp.verify = EmulatorSettings.GetCpSplitMode() >= 2;
    sp.regs = std::make_unique<Regs>();
    std::memcpy(sp.regs.get(), &regs, sizeof(Regs));
    sp.cb_extent = last_cb_extent;
    sp.db_extent = last_db_extent;
    sp.cs = mapped_queues[GfxQueueId].cs_state;
    raster_regs = sp.regs.get();
    raster_cb_extent = &sp.cb_extent;
    raster_db_extent = &sp.db_extent;
    raster_cs = &sp.cs;
    split_active = true;
    sp.thread = std::jthread{std::bind_front(&Liverpool::RasterMain, this)};
    sp.housekeeping = std::jthread{[](std::stop_token stoken) {
        Common::SetCurrentThreadName("shadPS4:Housekeeping");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        std::mutex mutex;
        std::condition_variable_any cv;
        std::unique_lock lk{mutex};
        while (!stoken.stop_requested()) {
            if (cv.wait_for(lk, stoken, std::chrono::seconds(5), [] { return false; })) {
                break;
            }
            const auto start = std::chrono::steady_clock::now();
            if (const auto usage = VideoCore::ThreadUsageReport(); !usage.empty()) {
                LOG_INFO(Render, "Thread CPU use (100% = one core): {}", usage);
            }
            const auto middle = std::chrono::steady_clock::now();
            VideoCore::ReapplyCoreIsolation(); // threads created since (guest threads)
            const auto end = std::chrono::steady_clock::now();
            Core::DebugServer::RecordPeriodic(
                "housekeeping_thread_cpu_report",
                std::chrono::duration<double, std::milli>(middle - start).count());
            Core::DebugServer::RecordPeriodic(
                "housekeeping_core_isolation",
                std::chrono::duration<double, std::milli>(end - middle).count());
        }
    }};
    LOG_INFO(Render, "Command processor split: raster thread started{}",
             sp.verify ? " (verify mode, synchronous)" : "");
}

void Liverpool::StopRasterSplit() {
    if (!split) {
        return;
    }
    split->thread.request_stop();
    split->work_cv.notify_all();
    split->thread.join();
}

bool Liverpool::RasterIdle() const noexcept {
    return !split_active || split->pushed.load(std::memory_order_acquire) ==
                                split->done.load(std::memory_order_acquire);
}

void Liverpool::WakeRaster() {
    if (split_active) {
        std::scoped_lock lk{split->mutex};
        split->work_cv.notify_all();
    }
}

void Liverpool::MirrorRegs(u32 dword_offset, u32 count) {
    if (!overlay.list.empty()) {
        OverlayForget(dword_offset, count); // the command packets wrote the truth for these registers
    }
    MirrorRegsRaw(dword_offset, count);
}

void Liverpool::OverlaySet(u32 index, u32 value) {
    u32& current = regs.reg_array[index];
    if (current == value) {
        return;
    }
    if (!overlay.truth) {
        overlay.truth = std::make_unique<u32[]>(Regs::NumRegs);
        overlay.bits.assign((Regs::NumRegs + 63) / 64, 0);
    }
    u64& word = overlay.bits[index >> 6];
    if ((word >> (index & 63) & 1) == 0) {
        overlay.truth[index] = current;
        word |= 1ull << (index & 63);
        overlay.list.push_back(index);
        if (overlay.list.size() > 32768) {
            // duplicates pile up only when packets keep rewriting overlaid registers: start over from the truth
            current = value;
            MirrorRegsRaw(index, 1);
            OverlayRestore();
            return;
        }
    }
    current = value;
    MirrorRegsRaw(index, 1);
}

void Liverpool::OverlayRestore() {
    for (const u32 index : overlay.list) {
        u64& word = overlay.bits[index >> 6];
        if ((word >> (index & 63) & 1) != 0) {
            word &= ~(1ull << (index & 63));
            regs.reg_array[index] = overlay.truth[index];
            MirrorRegsRaw(index, 1);
        }
    }
    overlay.list.clear();
}

void Liverpool::OverlayForget(u32 dword_offset, u32 count) {
    for (u32 index = dword_offset; index < dword_offset + count; ++index) {
        overlay.bits[index >> 6] &= ~(1ull << (index & 63));
    }
}

void Liverpool::OverlayForgetAll() {
    for (const u32 index : overlay.list) {
        overlay.bits[index >> 6] &= ~(1ull << (index & 63));
    }
    overlay.list.clear();
}

void Liverpool::MirrorRegsRaw(u32 dword_offset, u32 count) {
    if (!split_active || count == 0) {
        return;
    }
    ASSERT(dword_offset + count <= Regs::NumRegs);
    auto& pending = split->pending;
    const size_t old_size = pending.size();
    pending.resize(old_size + 2 + count);
    u32* const out = pending.data() + old_size;
    out[0] = dword_offset;
    out[1] = count;
    std::memcpy(out + 2, regs.reg_array.data() + dword_offset, count * sizeof(u32));
}

void Liverpool::MirrorObject(const void* object, size_t size) {
    if (!split_active) {
        return;
    }
    const auto base = reinterpret_cast<uintptr_t>(regs.reg_array.data());
    const auto begin = reinterpret_cast<uintptr_t>(object);
    ASSERT(begin >= base && begin + size <= base + sizeof(Regs));
    const u32 first = static_cast<u32>((begin - base) / sizeof(u32));
    const u32 last = static_cast<u32>((begin + size - base + sizeof(u32) - 1) / sizeof(u32));
    MirrorRegs(first, last - first);
}

void Liverpool::MirrorSetDefaults() {
    OverlayForgetAll(); // the registers were reset to their defaults
    if (split_active) {
        split->pending.push_back(RasterSplit::SetDefaultsMarker);
        split->pending.push_back(0);
    }
}

void Liverpool::DrainRaster() {
    if (!split_active) {
        return;
    }
    auto& sp = *split;
    const u64 target = sp.pushed.load(std::memory_order_acquire);
    if (sp.done.load(std::memory_order_acquire) >= target) {
        return;
    }
    std::unique_lock lk{sp.mutex};
    sp.done_cv.wait(lk, [&] { return sp.done.load(std::memory_order_acquire) >= target; });
}

void Liverpool::RasterPush(RasterOp&& op, bool wait) {
    auto& sp = *split;
    op.deltas = std::move(sp.pending);
    sp.pending.clear();
    if (op.writes_memory) {
        sp.pending_writes.fetch_add(1, std::memory_order_acq_rel);
        sp.last_write_line = op.write_line;
        sp.SiteFor(op.write_line)->pushes.fetch_add(1, std::memory_order_relaxed);
    }
    ++sp.stat_ops;
    u64 target;
    {
        std::unique_lock lk{sp.mutex};
        if (sp.queue.size() >= RasterSplit::MaxQueued) {
            sp.stat_front_blocked.fetch_add(1, std::memory_order_relaxed);
        }
        if (sp.queue.size() >= RasterSplit::MaxQueued) {
            sp.front_state.store(3, std::memory_order_relaxed);
        }
        sp.done_cv.wait(lk, [&] { return sp.queue.size() < RasterSplit::MaxQueued; });
        sp.front_state.store(0, std::memory_order_relaxed);
        sp.queue.push_back(std::move(op));
        target = sp.pushed.fetch_add(1, std::memory_order_acq_rel) + 1;
        sp.work_cv.notify_all();
        if (!sp.free_deltas.empty()) {
            sp.pending = std::move(sp.free_deltas.back());
            sp.free_deltas.pop_back();
        }
        if (wait || sp.verify) {
            sp.done_cv.wait(lk, [&] { return sp.done.load(std::memory_order_acquire) >= target; });
        }
    }
}

void Liverpool::RasterRun(Common::UniqueFunction<void>&& func, bool wait, bool writes_memory,
                          std::source_location site) {
    if (!split_active) {
        func();
        return;
    }
    RasterOp op;
    op.closure = std::move(func);
    op.writes_memory = writes_memory;
    op.write_line = site.line();
    RasterPush(std::move(op), wait);
}

void Liverpool::DrainRasterWrites() {
    auto& sp = *split;
    if (sp.pending_writes.load(std::memory_order_acquire) == 0) {
        return;
    }
    ++sp.stat_write_drains;
    sp.front_state.store(2, std::memory_order_relaxed);
    const u64 wait_start = VideoCore::ReadbackStats::NowNs();
    {
        std::unique_lock lk{sp.mutex};
        sp.done_cv.wait(lk,
                        [&] { return sp.pending_writes.load(std::memory_order_acquire) == 0; });
    }
    sp.front_state.store(0, std::memory_order_relaxed);
    sp.SiteFor(sp.last_write_line)
        ->wait_ns.fetch_add(VideoCore::ReadbackStats::NowNs() - wait_start,
                            std::memory_order_relaxed);
}

void Liverpool::RasterDispatch(const RasterCall& call_in) {
    if (!native_exec_scope && !overlay.list.empty() &&
        (call_in.kind == RasterCall::Kind::Draw || call_in.kind == RasterCall::Kind::DrawIndirect)) {
        OverlayRestore(); // a draw of the packet path: the registers its packets built on
    }
    RasterCall call = call_in;
    if (native_tag_pending.valid) {
        // The tag belongs to the draw or dispatch the game called. The game's own tracking may issue internal draws
        // (full-screen quads) in between: a call that does not fit the tag (kind, count) leaves it pending.
        const u32 tag_kind = (native_tag_pending.meta >> 8) & 0xff;
        bool match = false;
        switch (call.kind) {
        case RasterCall::Kind::Draw:
            // internal full-screen quads (resource transitions) run the emulator's embedded vertex shader, whose
            // address has bits above the game's address space in SPI_SHADER_PGM_HI_VS
            match = tag_kind <= 1 && (tag_kind == 0) == call.indexed &&
                    regs.reg_array[Regs::ShRegWordOffset + 0x49] == 0 &&
                    (Core::NativeDraw::RecordDrawCount(native_tag_pending.seq) == regs.num_indices ||
                     Core::NativeDraw::RecordDrawCount(native_tag_pending.seq) == Core::NativeDraw::UnknownDrawCount);
            break;
        case RasterCall::Kind::DrawIndirect:
            match = tag_kind == 2;
            break;
        default:
            match = tag_kind == 3;
            break;
        }
        if (match) {
            call.tag = native_tag_pending;
            native_tag_pending.valid = false;
            native_tag_skips = 0;
        } else if (++native_tag_skips > 6) {
            native_tag_pending.valid = false;
            native_tag_skips = 0;
        }
    }
    if (!split_active) {
        ExecRasterCall(call, nullptr, nullptr);
        return;
    }
    auto& sp = *split;
    // The lookup below reads descriptor tables and shader code from guest memory: closures that
    // write guest memory (constant engine dumps, WriteData, DMA) must have run first.
    DrainRasterWrites();
    ++sp.stat_calls;
    if (call.kind == RasterCall::Kind::Draw || call.kind == RasterCall::Kind::DrawIndirect) {
        ++sp.stat_draws;
        sp.stat_draws_no_reg_change += sp.pending.empty();
    }
    RasterOp op;
    op.has_call = true;
    op.call = call;
    op.cb_extent = last_cb_extent;
    op.db_extent = last_db_extent;
    if (call.kind == RasterCall::Kind::DispatchDirect ||
        call.kind == RasterCall::Kind::DispatchIndirect) {
        op.cs = GetCsRegs();
        // The pipeline lookup runs here, on the command processor thread (stage 2 of the split).
        op.dispatch_plan = rasterizer->PlanDispatch(op.cs);
    } else {
        const Vulkan::DrawIndirectParams params =
            call.kind == RasterCall::Kind::DrawIndirect
                ? Vulkan::DrawIndirectParams{.vertex_sgpr_offset = call.vertex_sgpr_offset,
                                             .instance_sgpr_offset = call.instance_sgpr_offset,
                                             .is_indirect = true}
                : Vulkan::DrawIndirectParams{};
        op.draw_plan = rasterizer->PlanDraw(regs, params);
    }
    RasterPush(std::move(op), false);
    if (sp.verify && sp.verify_reports < 20 &&
        std::memcmp(sp.regs.get(), &regs, sizeof(Regs)) != 0) {
        // The operation is done: everything the command processor wrote so far is in the
        // mirror, so the two register files must be identical.
        const auto* a = reinterpret_cast<const u32*>(sp.regs.get());
        const auto* b = reinterpret_cast<const u32*>(&regs);
        u32 first = 0;
        u32 count = 0;
        for (u32 i = 0; i < Regs::NumRegs; ++i) {
            if (a[i] != b[i]) {
                if (count++ == 0) {
                    first = i;
                }
            }
        }
        ++sp.verify_reports;
        LOG_ERROR(Render,
                  "CP split verify: raster register mirror differs from the command processor "
                  "registers in {} dwords, first at dword {:#x} (mirror {:#x}, command processor "
                  "{:#x}); a register write is not mirrored",
                  count, first, a[first], b[first]);
    }
}

void Liverpool::ExecuteNativeDraw(const void* header, u32 asc_vqid, u32 index_count) {
    native_exec_calls++;
    Core::NativeDraw::ExecDraw draw;
    if (!rasterizer || !Core::NativeDraw::TakeExecDraw(native_tag_pending.seq, draw)) {
        native_tag_pending.valid = false;
        LOG_ERROR(Render, "native draw: the record {} is gone, a draw is lost", native_tag_pending.seq);
        return;
    }
    if (draw.kind == 6) {
        // buffer fill: destination, value and the number of dwords travel in the record
        native_tag_pending.valid = false;
        if (rasterizer) {
            RasterRun(
                [this, destination = static_cast<VAddr>(draw.args_addr), value = draw.start_index, bytes = draw.a1 * 4u] {
                    rasterizer->FillBuffer(destination, bytes, value, false);
                },
                false, true);
        }
        return;
    }
    if (draw.kind == 3) {
        // dispatch: the record holds dwords of the compute program (user data, dimensions); applied for this dispatch only
        ComputeProgram& program = GetCsRegs();
        const ComputeProgram saved_program = program;
        auto* dwords = reinterpret_cast<u32*>(&program);
        for (u32 i = 0; i < draw.count; ++i) {
            const u32 dword = draw.regs[i].index - Core::NativeDraw::ComputeRegisterBase;
            if (dword < sizeof(ComputeProgram) / sizeof(u32)) {
                dwords[dword] = draw.regs[i].value;
            }
        }
        RasterCall dispatch{};
        dispatch.kind = RasterCall::Kind::DispatchDirect;
        if (asc_vqid != ~0u) {
            dispatch.asc = true;
            dispatch.vqid = asc_vqid;
            dispatch.cmd_address = header;
            dispatch.marker = "asc[{}]:{}:NativeDispatch";
        } else {
            dispatch.marker = "gfx:{}:NativeDispatch";
        }
        if ((program.dispatch_initiator & 1) != 0) {
            RasterDispatch(dispatch);
        }
        program = saved_program;
        return;
    }
    // a fast clear eliminate takes the extent hint of its own target for this operation only
    const auto saved_cb_extent = last_cb_extent;
    if (draw.cb_extents_only) {
        native_tag_pending.valid = false;
    }
    if (draw.has_extents) {
        // the extent hints the game's target bind sends in a trailing NOP packet
        for (u32 slot = 0; slot < NUM_COLOR_BUFFERS && slot < draw.cb_extent.size(); ++slot) {
            if ((draw.cb_extent_mask >> slot) & 1) {
                last_cb_extent[slot].raw = draw.cb_extent[slot];
            }
        }
        if (!draw.cb_extents_only) {
            last_db_extent.raw = draw.db_extent;
        }
    }
    // The registers of this draw are applied for the draw only: the game's own command packets, which it still writes
    // for draws that take the packet path, assume the register file holds what they wrote.
    // The registers stay as the draw leaves them; the next draw writes only what differs, and a draw of the packet
    // path first gets the registers its own packets built (RasterDispatch).
    for (u32 i = 0; i < draw.count; ++i) {
        OverlaySet(draw.regs[i].index, draw.regs[i].value);
    }
    OverlaySet(static_cast<u32>(&regs.draw_initiator - regs.reg_array.data()), 0);
    if (index_count != ~0u) {
        OverlaySet(static_cast<u32>(&regs.num_indices - regs.reg_array.data()), index_count);
    }
    RasterCall call{};
    if (draw.kind == 2) {
        call.kind = RasterCall::Kind::DrawIndirect;
        call.marker = "gfx:{}:NativeDrawIndexIndirect";
        call.indexed = true;
        call.args_addr = draw.args_addr;
        call.offset = draw.offset;
        call.stride = sizeof(DrawIndexedIndirectArgs);
        call.max_count = 1;
        call.vertex_sgpr_offset = static_cast<u16>(draw.vertex_sgpr);
        call.instance_sgpr_offset = static_cast<u16>(draw.instance_sgpr);
    } else {
        call.kind = RasterCall::Kind::Draw;
        call.marker = draw.cb_extents_only ? "gfx:{}:NativeFastClearEliminate" : "gfx:{}:NativeDraw";
        call.indexed = draw.kind == 0;
        call.index_offset = draw.start_index;
    }
    native_exec_scope = true;
    RasterDispatch(call);
    native_exec_scope = false;
    if (draw.cb_extents_only) {
        // the fast clear eliminate registers (color control in eliminate mode) must not reach the next draw
        last_cb_extent = saved_cb_extent;
        OverlayRestore();
    }
}

void Liverpool::CpWriteData(void* address, const u32* data, u32 data_size, u32 queue) {
    const auto note = [this](void* address, u32 data_size) {
        if (rasterizer) {
            rasterizer->NoteCpGuestWrite(reinterpret_cast<VAddr>(address), data_size);
        }
    };
    if (!split_active) {
        note(address, data_size);
        if (!DeferWriteBehindFences(address, data, data_size, queue)) {
            std::memcpy(address, data, data_size);
        }
        return;
    }
    RasterRun([note, address, data_size] { note(address, data_size); }, false, true);
    // Whether a fence write to the label is held back is decided here, in stream order.
    if (!DeferWriteBehindFences(address, data, data_size, queue)) {
        RasterRun(
            [address, words = std::vector<u32>(data, data + data_size / sizeof(u32)), data_size] {
                std::memcpy(address, words.data(), data_size);
            },
            false, true);
    }
}

void Liverpool::ExecRasterCall(const RasterCall& c, const Vulkan::DrawPlan* draw_plan,
                               const Vulkan::DispatchPlan* dispatch_plan) {
    const auto run = [&](auto&& body) {
        if (c.no_marker) {
            body();
            return;
        }
        const void* cmd_address = c.cmd_address;
        u32 vqid = c.vqid;
        if (c.asc) {
            rasterizer->ScopeMarker(c.marker, fmt::make_format_args(vqid, cmd_address), body);
        } else {
            rasterizer->ScopeMarker(c.marker, fmt::make_format_args(cmd_address), body);
        }
    };
    switch (c.kind) {
    case RasterCall::Kind::Draw:
        rasterizer->SetNativeTag(c.tag);
        run([&] { rasterizer->Draw(c.indexed, c.index_offset, draw_plan); });
        break;
    case RasterCall::Kind::DrawIndirect:
        rasterizer->NativeInvalidate();
        if (c.tag.valid && c.tag.seq != 0 && (c.tag.meta & VideoCore::NativeTagExecFlag) == 0) {
            Core::NativeDraw::VerifyRecord(c.tag.seq, RasterRegs().reg_array.data());
        }
        run([&] {
            rasterizer->DrawIndirect(c.indexed, c.args_addr, c.offset, c.stride, c.max_count,
                                     c.count_addr, c.vertex_sgpr_offset, c.instance_sgpr_offset,
                                     draw_plan);
        });
        break;
    case RasterCall::Kind::DispatchDirect:
        rasterizer->NativeInvalidate();
        if (c.tag.valid && c.tag.seq != 0 && (c.tag.meta & VideoCore::NativeTagExecFlag) == 0) {
            static u32 compute_debug = 0;
            if (compute_debug++ < 6) {
                const auto& program = RasterCsRegs();
                LOG_INFO(Render, "native compute verify: marker {} initiator {:#x} dims {} {} {} address {:#x} split {}",
                         c.marker, program.dispatch_initiator, program.dim_x, program.dim_y, program.dim_z,
                         static_cast<u64>(program.address), split_active);
            }
            Core::NativeDraw::VerifyComputeRecord(c.tag.seq, reinterpret_cast<const u32*>(&RasterCsRegs()));
        }
        run([&] { rasterizer->DispatchDirect(dispatch_plan); });
        break;
    case RasterCall::Kind::DispatchIndirect:
        rasterizer->NativeInvalidate();
        run([&] {
            rasterizer->DispatchIndirect(c.args_addr, c.offset, c.dispatch_size, dispatch_plan);
        });
        break;
    }
    rasterizer->MaybeFlushRecorded();
}

void Liverpool::RasterMain(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuRaster");
    if (const u32 priority = EmulatorSettings.GetCpThreadPriority(); priority >= 2) {
        Common::SetCurrentThreadPriority(static_cast<Common::ThreadPriority>(priority));
    }
    gpu_id = std::this_thread::get_id();
    s32 applied_core = -1;
    VideoCore::ReadbackStats::MarkCpThread();
    auto& sp = *split;
    std::vector<RasterOp> batch;
    std::vector<std::vector<u32>> recycled;
    while (true) {
        // cp_raster_core can be changed live (debug server /set?name=raster_core&value=N).
        if (const s32 wanted = VideoCore::ReadbackPolicy::Get().raster_core.load();
            wanted != applied_core) {
            if (applied_core >= 0) {
                VideoCore::ClearCoreIsolation();
            }
            if (wanted >= 0) {
                VideoCore::PinThreadAndIsolateCore(static_cast<u32>(wanted));
            }
            applied_core = wanted;
        }
        {
            const u64 idle_start = VideoCore::ReadbackStats::NowNs();
            const u8 front_state_at_idle = sp.front_state.load(std::memory_order_relaxed);
            std::unique_lock lk{sp.mutex};
            Common::CondvarWait(sp.work_cv, lk, stoken,
                                [&] { return !sp.queue.empty() || num_commands != 0; });
            {
                const u64 idle = VideoCore::ReadbackStats::NowNs() - idle_start;
                sp.stat_raster_idle_ns.fetch_add(idle, std::memory_order_relaxed);
                sp.raster_idle_ns_by_front_state[front_state_at_idle & 3].fetch_add(
                    idle, std::memory_order_relaxed);
            }
            if (stoken.stop_requested() && sp.queue.empty() && num_commands == 0) {
                return;
            }
            batch.swap(sp.queue);
            if (!batch.empty()) {
                const auto& first = batch.front();
                const size_t kind = first.has_call ? static_cast<size_t>(first.call.kind) : 4;
                sp.raster_idle_ns_by_next_op[kind].fetch_add(
                    VideoCore::ReadbackStats::NowNs() - idle_start, std::memory_order_relaxed);
                sp.raster_idle_count_by_next_op[kind].fetch_add(1, std::memory_order_relaxed);
            }
            sp.done_cv.notify_all(); // The queue has room again.
        }
        for (auto& op : batch) {
            // Callbacks sent by other threads (fault readbacks, fences) run between operations,
            // like they ran between packets before.
            if (num_commands != 0) {
                const u64 callback_start = VideoCore::ReadbackStats::NowNs();
                ProcessCommands();
                sp.ns_callbacks += VideoCore::ReadbackStats::NowNs() - callback_start;
            }
            const u64 op_start = VideoCore::ReadbackStats::NowNs();
            const u32* p = op.deltas.data();
            const u32* end = p + op.deltas.size();
            while (p < end) {
                const u32 offset = p[0];
                const u32 count = p[1];
                p += 2;
                if (offset == RasterSplit::SetDefaultsMarker) {
                    sp.regs->SetDefaults();
                    continue;
                }
                std::memcpy(&sp.regs->reg_array[offset], p, count * sizeof(u32));
                p += count;
            }
            if (op.has_call) {
                sp.cb_extent = op.cb_extent;
                sp.db_extent = op.db_extent;
                if (op.call.kind == RasterCall::Kind::DispatchDirect ||
                    op.call.kind == RasterCall::Kind::DispatchIndirect) {
                    sp.cs = op.cs;
                }
                ExecRasterCall(op.call, op.draw_plan.get(), op.dispatch_plan.get());
                const u64 op_ns = VideoCore::ReadbackStats::NowNs() - op_start;
                if (op.call.kind == RasterCall::Kind::Draw ||
                    op.call.kind == RasterCall::Kind::DrawIndirect) {
                    sp.ns_draws += op_ns;
                    ++sp.n_draws;
                    if (sp.regs->color_target_mask.raw == 0) {
                        sp.ns_depth_only_draws += op_ns;
                        ++sp.n_depth_only_draws;
                    }
                } else {
                    sp.ns_dispatches += op_ns;
                }
            } else if (op.closure) {
                rasterizer->NativeInvalidate();
                op.closure();
                (op.writes_memory ? sp.ns_write_closures : sp.ns_other_closures) +=
                    VideoCore::ReadbackStats::NowNs() - op_start;
                if (op.writes_memory) {
                    sp.pending_writes.fetch_sub(1, std::memory_order_acq_rel);
                    std::scoped_lock lk{sp.mutex};
                    sp.done_cv.notify_all();
                }
            }
            op.closure = {};
            rasterizer->RecyclePlan(std::move(op.draw_plan));
            rasterizer->RecyclePlan(std::move(op.dispatch_plan));
            op.deltas.clear();
            recycled.push_back(std::move(op.deltas));
        }
        if (num_commands != 0) {
            ProcessCommands();
        }
        const u64 count = batch.size();
        batch.clear();
        if (const u64 now = VideoCore::ReadbackStats::NowNs();
            now - sp.stat_last_log_ns > 5'000'000'000ULL) {
            const u64 block_start = now;
            const auto plan_stats = rasterizer->TakePlanStats();
            if (sp.stat_last_log_ns != 0) {
                Core::DebugServer::Publish(
                    "cp_split",
                    {{"operations", sp.stat_ops},
                     {"draws_and_dispatches", sp.stat_calls},
                     {"write_drains", sp.stat_write_drains},
                     {"draws", sp.stat_draws},
                     {"draws_no_register_change", sp.stat_draws_no_reg_change},
                     {"plan_draws_all_stages_same", plan_stats.draws_all_stages_same},
                     {"plan_draws", plan_stats.draws},
                     {"note", "counts per ~5 s interval"}});
                LOG_INFO(Render,
                         "CP split: {} operations, {} draws/dispatches, {} waits for guest memory "
                         "writes before a pipeline lookup, front blocked on a full queue {} times, "
                         "raster thread idle {} ms, "
                         "draws {} of which {} had no register change since the previous op, "
                         "stage inputs same as the previous draw: draws {}/{}, stages {}/{} "
                         "(since the last line)",
                         sp.stat_ops, sp.stat_calls, sp.stat_write_drains,
                         sp.stat_front_blocked.exchange(0),
                         sp.stat_raster_idle_ns.exchange(0) / 1'000'000, sp.stat_draws,
                         sp.stat_draws_no_reg_change, plan_stats.draws_all_stages_same,
                         plan_stats.draws, plan_stats.stages_same, plan_stats.stages);
            }
            sp.stat_front_blocked = 0;
            sp.stat_raster_idle_ns = 0;
            sp.stat_ops = sp.stat_calls = sp.stat_write_drains = 0;
            sp.stat_draws = sp.stat_draws_no_reg_change = 0;
            LOG_INFO(Render,
                     "Raster thread wall time (ms per interval): draws {}, dispatches {}, memory "
                     "write closures {}, other closures (fences, submit end, flips) {}, "
                     "SendCommand callbacks {}; draws {} of which depth-only {} ms, {} draws of which "
                     "depth-only {}",
                     sp.ns_draws / 1'000'000, sp.ns_dispatches / 1'000'000,
                     sp.ns_write_closures / 1'000'000, sp.ns_other_closures / 1'000'000,
                     sp.ns_callbacks / 1'000'000, sp.ns_draws / 1'000'000,
                     sp.ns_depth_only_draws / 1'000'000, sp.n_draws, sp.n_depth_only_draws);
            {
                static constexpr const char* names[] = {"front_working", "front_waiting_for_game_submit",
                                                        "front_draining_writes", "queue_full"};
                nlohmann::json idle = nlohmann::json::object();
                for (size_t i = 0; i < 4; ++i) {
                    idle[names[i]] = sp.raster_idle_ns_by_front_state[i].exchange(0) / 1'000'000;
                }
                Core::DebugServer::Publish("raster_idle_ms_by_what_the_front_was_doing",
                                           std::move(idle));
                static constexpr const char* op_names[] = {"draw", "draw_indirect", "dispatch",
                                                           "dispatch_indirect", "closure"};
                nlohmann::json by_op = nlohmann::json::object();
                for (size_t i = 0; i < 5; ++i) {
                    by_op[op_names[i]] = {
                        {"idle_ms", sp.raster_idle_ns_by_next_op[i].exchange(0) / 1'000'000},
                        {"idle_episodes", sp.raster_idle_count_by_next_op[i].exchange(0)}};
                }
                Core::DebugServer::Publish("raster_idle_by_kind_of_op_that_ended_it",
                                           std::move(by_op));
            }
            {
                static constexpr const char* kind_names[] = {"gfx_eop", "gfx_eos", "compute_release",
                                                             "held_write", "submit_end"};
                nlohmann::json fences = nlohmann::json::object();
                for (size_t i = 0; i < fence_latency.size(); ++i) {
                    auto& stat = fence_latency[i];
                    const u64 count = stat.count.exchange(0);
                    const u64 queue_ns = stat.queue_ns.exchange(0);
                    const u64 gpu_ns = stat.gpu_ns.exchange(0);
                    const u64 total_ns = stat.total_ns.exchange(0);
                    const u64 max_ns = stat.max_total_ns.exchange(0);
                    if (count == 0) {
                        continue;
                    }
                    fences[kind_names[i]] = {
                        {"fences", count},
                        {"avg_ms_parsed_to_raster_start", queue_ns / 1e6 / count},
                        {"avg_ms_raster_start_to_signal", gpu_ns / 1e6 / count},
                        {"avg_ms_total", total_ns / 1e6 / count},
                        {"max_ms_total", max_ns / 1e6}};
                }
                Core::DebugServer::Publish("fence_latency", std::move(fences));
            }
            {
                nlohmann::json sites = nlohmann::json::object();
                for (auto& site : sp.write_sites) {
                    const u32 line = site.line.load(std::memory_order_relaxed);
                    if (line == 0) {
                        continue;
                    }
                    sites["liverpool.cpp:" + std::to_string(line)] = {
                        {"writes_queued", site.pushes.exchange(0)},
                        {"front_wait_ms", site.wait_ns.exchange(0) / 1'000'000}};
                }
                Core::DebugServer::Publish("write_drains_by_call_site", std::move(sites));
            }
            {
                const auto native = rasterizer->TakeNativeStats();
                Core::DebugServer::Publish(
                    "native_draw_raster",
                    {{"tagged_draws", native.tagged},
                     {"untagged_draws", native.untagged},
                     {"dynamic_state_skipped", native.dynamic_skipped},
                     {"bind_pipeline_skipped", native.bind_skipped},
                     {"note", "counts per ~5 s interval"}});
                if (native.tagged != 0) {
                    LOG_INFO(Render,
                             "Native draw path: {} tagged draws, {} untagged, {} dynamic state "
                             "updates skipped, {} pipeline binds skipped",
                             native.tagged, native.untagged, native.dynamic_skipped,
                             native.bind_skipped);
                }
            }
            Core::DebugServer::Publish(
                "raster_thread_ms",
                {{"draws", sp.ns_draws / 1'000'000},
                 {"depth_only_draws_ms", sp.ns_depth_only_draws / 1'000'000},
                 {"draw_count", sp.n_draws},
                 {"depth_only_draw_count", sp.n_depth_only_draws},
                 {"dispatches", sp.ns_dispatches / 1'000'000},
                 {"memory_write_closures", sp.ns_write_closures / 1'000'000},
                 {"other_closures", sp.ns_other_closures / 1'000'000},
                 {"send_command_callbacks", sp.ns_callbacks / 1'000'000},
                 {"note", "wall time per ~5 s interval"}});
            sp.ns_draws = sp.ns_dispatches = sp.ns_write_closures = sp.ns_other_closures = 0;
            sp.ns_depth_only_draws = sp.n_draws = sp.n_depth_only_draws = 0;
            sp.ns_callbacks = 0;
            // The thread CPU report and the core isolation re-apply run on the housekeeping thread:
            // each took about 70 ms here, which showed as a dip to 50 fps every 5 seconds.
            sp.stat_last_log_ns = now;
            Core::DebugServer::RecordPeriodic(
                "raster_thread_5s_stats_block",
                static_cast<double>(VideoCore::ReadbackStats::NowNs() - block_start) / 1e6);
        }
        {
            std::scoped_lock lk{sp.mutex};
            for (auto& v : recycled) {
                if (sp.free_deltas.size() < 64) {
                    sp.free_deltas.push_back(std::move(v));
                }
            }
            sp.done.fetch_add(count, std::memory_order_acq_rel);
            sp.done_cv.notify_all();
        }
        recycled.clear();
        if (RasterIdle()) {
            std::scoped_lock lk{submit_mutex};
            submit_cv.notify_all();
        }
    }
}


Liverpool::Liverpool()
    : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()},
      fence_wait_shortcut{EmulatorSettings.GetReadbackFenceWaitShortcut()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
    StopRasterSplit();
}

static u64 ReadLabel(VAddr address, u32 num_bytes) {
    u64 value{};
    std::memcpy(&value, reinterpret_cast<const void*>(address), num_bytes);
    return value;
}

u64 Liverpool::RecordFence(VAddr address, u64 value, u32 num_bytes, u32 queue) {
    // Read outside the lock, see MatchRecordedFence.
    const u64 memory = ReadLabel(address, num_bytes);
    std::scoped_lock lk{recorded_fence_mutex};
    const u64 seq = ++next_fence_seq;
    // A label reused with another size: forget the old state, waits then poll memory.
    std::erase_if(label_states, [&](const LabelState& label) {
        return label.address < address + num_bytes && address < label.address + label.num_bytes &&
               (label.address != address || label.num_bytes != num_bytes);
    });
    auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it == label_states.end() || it->overridden) {
        // Start tracking from what guest memory holds now.
        if (it == label_states.end()) {
            it = label_states.emplace(label_states.end());
            it->address = address;
            it->num_bytes = num_bytes;
        }
        it->overridden = false;
        it->known_values.clear();
        it->known_values.push_back(memory);
    }
    // Only the latest value can satisfy a later wait: the GPU runs the fences in the order they
    // were recorded, so older values are overwritten before any later wait could observe them.
    it->latest_value = value;
    it->latest_seq = seq;
    // Set by SetFenceTick once the raster thread has recorded the work before this fence.
    it->gpu_tick = 0;
    it->queue = queue;
    ++it->pending;
    if (it->known_values.size() < MaxKnownLabelValues) {
        it->known_values.push_back(value);
    }
    return seq;
}

void Liverpool::SetFenceTick(VAddr address, u64 seq) {
    const u64 tick = rasterizer ? rasterizer->CurrentTick() : 0;
    std::scoped_lock lk{recorded_fence_mutex};
    const auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it != label_states.end() && it->latest_seq == seq) {
        it->gpu_tick = tick;
    }
}

u64 Liverpool::GetFenceTick(VAddr address, u64 seq) {
    std::scoped_lock lk{recorded_fence_mutex};
    const auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it == label_states.end() || it->latest_seq < seq) {
        return 0; // Retired: the work finished, nothing to order against.
    }
    return it->gpu_tick;
}

void Liverpool::RetireFence(VAddr address, u64 seq) {
    std::scoped_lock lk{recorded_fence_mutex};
    const auto it = std::ranges::find(label_states, address, &LabelState::address);
    if (it == label_states.end() || it->latest_seq < seq) {
        return;
    }
    if (--it->pending == 0) {
        // Everything recorded has landed: guest memory is authoritative again.
        label_states.erase(it);
    }
}

Liverpool::FenceMatch Liverpool::MatchRecordedFence(const PM4CmdWaitRegMem& wait_reg_mem) {
    if (wait_reg_mem.mem_space.Value() != PM4CmdWaitRegMem::MemSpace::Memory) {
        return {};
    }
    const VAddr wait_address = wait_reg_mem.Address<VAddr>();
    // Read outside the lock: the read may fault into a readback that waits on the thread
    // retiring fences.
    const u32 memory = *wait_reg_mem.Address();
    std::scoped_lock lk{recorded_fence_mutex};
    for (auto& label : label_states) {
        if (wait_address < label.address ||
            wait_address + sizeof(u32) > label.address + label.num_bytes) {
            continue;
        }
        if (label.overridden) {
            return {};
        }
        const u32 shift = static_cast<u32>(wait_address - label.address) * 8;
        const auto slice = [shift](u64 value) { return static_cast<u32>(value >> shift); };
        // Guest memory may only hold the value from before the first held back fence, or one
        // of the recorded values. Anything else was written by someone else (typically the
        // guest CPU resetting or releasing the label), and that write is newer than the
        // recorded values: from now on waits on this label poll guest memory.
        if (std::ranges::none_of(label.known_values,
                                 [&](u64 known) { return slice(known) == memory; })) {
            label.overridden = true;
            if (logged_overrides.insert(label.address).second) {
                LOG_INFO(Render,
                         "Fence timeline: label {:#x} changed to {:#x} outside the recorded "
                         "fences (held back value {:#x}, value before them {:#x}); waits on it "
                         "poll memory until the next fence",
                         label.address, memory, slice(label.latest_value),
                         slice(label.known_values.front()));
            }
            return {};
        }
        const u32 value = slice(label.latest_value);
        return {
            .satisfied = wait_reg_mem.TestValue(value),
            .label_address = label.address,
            .seq = label.latest_seq,
            .gpu_tick = label.gpu_tick,
            .queue = label.queue,
            .value = value,
            .initial = slice(label.known_values.front()),
        };
    }
    return {};
}

void Liverpool::OnWaitPassedOnFence(const PM4CmdWaitRegMem& wait_reg_mem, const FenceMatch& match,
                                    u32 queue) {
    if (fence_wait_shortcut >= 3 && rasterizer) {
        // Recorded in order with the draws queued before and after this wait.
        // The fence's tick is set by an earlier operation of the raster queue, so it is known
        // when this one runs.
        RasterRun([this, address = match.label_address, seq = match.seq] {
            if (rasterizer->CommandStreamBarrier(GetFenceTick(address, seq))) {
                VideoCore::ReadbackStats::OnStreamBarrier();
            }
        });
    }
    if (queue != GfxQueueId) {
        const VAddr address = wait_reg_mem.Address<VAddr>();
        // Capped: every new label address logged made the console sink about 3% of the front thread.
        if (logged_compute_passes.size() < 64 && logged_compute_passes.insert(address).second) {
            LOG_INFO(Render,
                     "Fence timeline: compute ring {} passed a wait on {:#x} (func={} ref={:#x} "
                     "mask={:#x}) with value {:#x} recorded by queue {}; memory holds {:#x}, "
                     "held {:#x} before the fence",
                     queue - 1, address, static_cast<u32>(wait_reg_mem.function.Value()),
                     wait_reg_mem.ref, wait_reg_mem.mask, match.value, match.queue,
                     *wait_reg_mem.Address(), match.initial);
        }
    }
}

void Liverpool::ReportLongWait(const PM4CmdWaitRegMem& wait_reg_mem, const char* queue,
                               u64 wait_start, bool& reported) {
    if (reported || VideoCore::ReadbackStats::NowNs() - wait_start < 2'000'000'000ULL) {
        return;
    }
    reported = true;
    const bool is_memory = wait_reg_mem.mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
    const u32 value = is_memory ? *wait_reg_mem.Address() : regs.reg_array[wait_reg_mem.Reg()];
    std::string pending;
    {
        std::scoped_lock lk{recorded_fence_mutex};
        for (const auto& label : label_states) {
            pending +=
                fmt::format(" [{:#x}={:#x} q{} pending={}{}]", label.address, label.latest_value,
                            label.queue, label.pending, label.overridden ? " overridden" : "");
        }
    }
    LOG_WARNING(Render,
                "Hang check: {} queue waiting >2s on {} {:#x}: value={:#x} func={} ref={:#x} "
                "mask={:#x}; pending fence writes:{}",
                queue, is_memory ? "memory" : "register",
                is_memory ? wait_reg_mem.Address<VAddr>() : VAddr{wait_reg_mem.Reg()}, value,
                static_cast<u32>(wait_reg_mem.function.Value()), wait_reg_mem.ref,
                wait_reg_mem.mask, pending.empty() ? " none" : pending);
}

void Liverpool::ReportLongSpin(const char* what, const void* address, u64 wait_start,
                               bool& reported) {
    VideoCore::ReadbackStats::CpWaitYield(false);
    if (reported || VideoCore::ReadbackStats::NowNs() - wait_start < 2'000'000'000ULL) {
        return;
    }
    reported = true;
    LOG_WARNING(Render, "Hang check: {} waiting >2s on {}", what, fmt::ptr(address));
}

void Liverpool::NoteFenceLatency(VideoCore::FenceKind kind, u64 queue_ns, u64 gpu_ns) {
    auto& stat = fence_latency[std::countr_zero(static_cast<u32>(kind)) % fence_latency.size()];
    const u64 total = queue_ns + gpu_ns;
    stat.count.fetch_add(1, std::memory_order_relaxed);
    stat.queue_ns.fetch_add(queue_ns, std::memory_order_relaxed);
    stat.gpu_ns.fetch_add(gpu_ns, std::memory_order_relaxed);
    stat.total_ns.fetch_add(total, std::memory_order_relaxed);
    u64 max = stat.max_total_ns.load(std::memory_order_relaxed);
    while (total > max && !stat.max_total_ns.compare_exchange_weak(max, total)) {
    }
}

void Liverpool::SignalFenceAfterReadbacks(Common::UniqueFunction<void>&& signal, bool must_sync,
                                          VAddr address, u64 value, u32 num_bytes, u32 queue,
                                          VideoCore::FenceKind kind) {
    VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::Fence};
    if (!rasterizer) {
        signal();
        return;
    }
    const u64 parsed_ns = VideoCore::ReadbackStats::NowNs();
    // Wraps the signal to time the fence: parsed -> started on the raster thread -> signal runs.
    const auto timed = [this, kind, parsed_ns](Common::UniqueFunction<void>&& inner) {
        const u64 started_ns = VideoCore::ReadbackStats::NowNs();
        return Common::UniqueFunction<void>{
            [this, kind, parsed_ns, started_ns, inner = std::move(inner)]() mutable {
                const u64 now = VideoCore::ReadbackStats::NowNs();
                NoteFenceLatency(kind, started_ns - parsed_ns, now - started_ns);
                inner();
            }};
    };
    if (address == 0 || num_bytes == 0) {
        RasterRun(
            [this, signal = std::move(signal), kind, must_sync, timed]() mutable {
                rasterizer->OnFence(timed(std::move(signal)), kind, must_sync);
            },
            must_sync);
        return;
    }
    // The label timeline is updated here, when the packet is parsed, so command stream waits
    // see the fence without waiting for the raster thread. The raster thread records the work
    // before the fence and notes its tick (SetFenceTick).
    const u64 seq = RecordFence(address, value, num_bytes, queue);
    RasterRun(
        [this, address, seq, signal = std::move(signal), kind, must_sync, timed]() mutable {
            SetFenceTick(address, seq);
            rasterizer->OnFence(
                timed([this, address, seq, signal = std::move(signal)]() mutable {
                    signal();
                    RetireFence(address, seq);
                }),
                kind, must_sync);
        },
        must_sync);
}

bool Liverpool::DeferWriteBehindFences(void* address, const u32* data, u32 num_bytes, u32 queue) {
    const VAddr begin = reinterpret_cast<VAddr>(address);
    {
        std::scoped_lock lk{recorded_fence_mutex};
        const bool held_back = std::ranges::any_of(label_states, [&](const LabelState& label) {
            return label.pending != 0 && label.address < begin + num_bytes &&
                   begin < label.address + label.num_bytes;
        });
        if (!held_back) {
            return false;
        }
    }
    if (logged_deferred_writes.size() < 64 && logged_deferred_writes.insert(begin).second) {
        LOG_INFO(Render,
                 "Fence timeline: queue {} writes {} bytes to label {:#x} while a fence write to "
                 "it is held back; the write is queued behind it",
                 queue, num_bytes, begin);
    }
    std::vector<u32> words(data, data + num_bytes / sizeof(u32));
    u64 value{};
    std::memcpy(&value, words.data(), std::min<u32>(num_bytes, sizeof(u64)));
    auto signal = [this, address, words = std::move(words)] {
        const u32 size = static_cast<u32>(words.size() * sizeof(u32));
        RunFenceSignal(
            *this, gpu_id,
            [&] { return TryWriteOffThread(rasterizer, address, words.data(), size); },
            [address, words] {
                const u32 size = static_cast<u32>(words.size() * sizeof(u32));
                if (!Core::Memory::Instance()->TryWriteBacking(address, words.data(), size)) {
                    std::memcpy(address, words.data(), size);
                }
            });
    };
    const bool is_label = num_bytes == sizeof(u32) || num_bytes == sizeof(u64);
    SignalFenceAfterReadbacks(std::move(signal), false, is_label ? begin : 0, value,
                              is_label ? num_bytes : 0, queue, VideoCore::FenceHeldWrite);
    return true;
}

void Liverpool::ProcessCommands() {
    if (split_active && std::this_thread::get_id() != gpu_id) {
        return; // The raster thread runs the callbacks.
    }
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        VideoCore::ReadbackStats::CpPhase phase{"guest command (SendCommand)"};
        VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::GuestCommand};
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    if (const u32 priority = EmulatorSettings.GetCpThreadPriority(); priority >= 2) {
        Common::SetCurrentThreadPriority(static_cast<Common::ThreadPriority>(priority));
    }
    gpu_id = std::this_thread::get_id();
    front_id = gpu_id;
    if (const s32 core = EmulatorSettings.GetCpFrontCore(); core >= 0) {
        VideoCore::PinThreadAndIsolateCore(static_cast<u32>(core));
    }
    VideoCore::ReadbackStats::MarkCpThread();
    VideoCore::ReadbackStats::RegisterSampledThread("cp");
#ifdef __linux__
    gpu_tid = gettid();
#endif

    while (!stoken.stop_requested()) {
        VideoCore::ReadbackTrace::NoteMark("front_wait");
        {
            VideoCore::ReadbackStats::CpTimer idle_timer{VideoCore::ReadbackStats::CpTime::Idle};
            std::unique_lock lk{submit_mutex};
            if (split_active) {
                split->front_state.store(1, std::memory_order_relaxed);
            }
            Common::CondvarWait(submit_cv, lk, stoken, [this] {
                return (!split_active && num_commands) || num_submits || submit_done;
            });
            if (split_active) {
                split->front_state.store(0, std::memory_order_relaxed);
            }
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();
        VideoCore::ReadbackTrace::NoteMark("submit_begin");

        curr_qid = -1;

        while (num_submits || (!split_active && num_commands)) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            task.resume();

            if (task.done()) {
                task.destroy();

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();

                --num_submits;
                std::scoped_lock lock2{submit_mutex};
                submit_cv.notify_all();
            }
        }

        if (submit_done) {
            VideoCore::EndCapture();
            if (rasterizer) {
                RasterRun([this] {
                    VideoCore::ReadbackStats::CpPhase phase{"end of submit (readbacks, flush)"};
                    VideoCore::ReadbackStats::CpTimer timer{
                        VideoCore::ReadbackStats::CpTime::SubmitEnd};
                    rasterizer->OnSubmit();
                    rasterizer->Flush();
                });
            }
            submit_done = false;
        }

        RasterRun([] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle); });
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            if (split_active) {
                // A command processor write to guest memory: done in order with the draws.
                void* dst = dump_const->Address<void*>();
                const u8* src = cblock.constants_heap.data() + dump_const->Offset();
                RasterRun(
                    [dst, bytes = std::vector<u8>(src, src + dump_const->Size())] {
                        std::memcpy(dst, bytes.data(), bytes.size());
                    },
                    false, true);
            } else {
                memcpy(dump_const->Address<void*>(),
                       cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            }
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            task.handle.destroy();
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

namespace {
/// With native execution on: which command packets the game's own code still writes, grouped by the native tag that
/// came last in the stream (per 5 s, logged): 0 before any tag, 1 after the tag of an executed draw, 2 draw on the
/// packet path, 3 executed dispatch, 4 dispatch on the packet path, 5 fast clear eliminate.
void CountGameCommand(const PM4Header* header, PM4ItOpcode opcode, u32 count) {
    constexpr u32 Groups = 6;
    static std::array<std::array<u64, 256>, Groups> opcodes{};
    static std::array<u64, Regs::NumRegs> registers{};
    static std::array<u64, Groups> packets{};
    static u64 last_log_ns = 0;
    static u32 group = 0;
    if (opcode == PM4ItOpcode::Nop && count >= VideoCore::NativeTagPayloadDwords &&
        reinterpret_cast<const u32*>(header)[1] == VideoCore::NativeTagMagic) {
        const u32 meta = reinterpret_cast<const u32*>(header)[2];
        const bool exec = (meta & VideoCore::NativeTagExecFlag) != 0;
        const u32 kind = (meta >> 8) & 0xff;
        group = kind == 4 ? 5 : kind == 3 ? (exec ? 3 : 4) : (exec ? 1 : 2);
    }
    ++opcodes[group][static_cast<u32>(opcode) & 255];
    ++packets[group];
    Core::NativeDraw::Live().packets[group].fetch_add(1, std::memory_order_relaxed);
    if (opcode == PM4ItOpcode::SetContextReg || opcode == PM4ItOpcode::SetShReg ||
        opcode == PM4ItOpcode::SetUconfigReg || opcode == PM4ItOpcode::SetConfigReg) {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        const u32 base = opcode == PM4ItOpcode::SetContextReg ? Regs::ContextRegWordOffset
                         : opcode == PM4ItOpcode::SetShReg    ? Regs::ShRegWordOffset
                                                              : 0;
        for (u32 i = 0; i + 1 < count; ++i) {
            if (const u32 index = base + set_data->reg_offset + i; index < Regs::NumRegs) {
                ++registers[index];
            }
        }
    }
    const u64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    if (now - last_log_ns <= 5'000'000'000ull) {
        return;
    }
    if (last_log_ns != 0) {
        static constexpr const char* Names[Groups] = {"before any tag", "after executed draw", "after packet-path draw",
                                                      "after executed dispatch", "after packet-path dispatch",
                                                      "after fast clear eliminate"};
        for (u32 g = 0; g < Groups; ++g) {
            std::vector<std::pair<u64, u32>> top;
            for (u32 i = 0; i < 256; ++i) {
                if (opcodes[g][i] != 0) {
                    top.emplace_back(opcodes[g][i], i);
                }
            }
            std::sort(top.rbegin(), top.rend());
            std::string text;
            for (size_t i = 0; i < top.size() && i < 12; ++i) {
                text += fmt::format(" {:#x}:{}", top[i].second, top[i].first);
            }
            LOG_INFO(Render, "game packets {}: {} (opcode:count{})", Names[g], packets[g], text);
        }
        std::vector<std::pair<u64, u32>> top;
        for (u32 i = 0; i < registers.size(); ++i) {
            if (registers[i] != 0) {
                top.emplace_back(registers[i], i);
            }
        }
        std::sort(top.rbegin(), top.rend());
        std::string text;
        for (size_t i = 0; i < top.size() && i < 60; ++i) {
            text += fmt::format(" {:#x}:{}", top[i].second, top[i].first);
        }
        LOG_INFO(Render, "game register writes per 5 s (register index:count, top 60 of {}):{}", top.size(), text);
    }
    opcodes = {};
    registers = {};
    packets = {};
    last_log_ns = now;
}
} // namespace

namespace {
/// Packets the game writes to its async compute queues (per 5 s, logged).
void CountComputeQueueCommand(PM4ItOpcode opcode) {
    static std::array<u64, 256> opcodes{};
    static u64 total = 0;
    static u64 last_log_ns = 0;
    ++opcodes[static_cast<u32>(opcode) & 255];
    ++total;
    Core::NativeDraw::Live().asc_packets.fetch_add(1, std::memory_order_relaxed);
    const u64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    if (now - last_log_ns <= 5'000'000'000ull) {
        return;
    }
    if (last_log_ns != 0) {
        std::vector<std::pair<u64, u32>> top;
        for (u32 i = 0; i < 256; ++i) {
            if (opcodes[i] != 0) {
                top.emplace_back(opcodes[i], i);
            }
        }
        std::sort(top.rbegin(), top.rend());
        std::string text;
        for (size_t i = 0; i < top.size() && i < 12; ++i) {
            text += fmt::format(" {:#x}:{}", top[i].second, top[i].first);
        }
        LOG_INFO(Render, "async compute queue packets per 5 s: {} (opcode:count{})", total, text);
    }
    opcodes = {};
    total = 0;
    last_log_ns = now;
}
} // namespace

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            VideoCore::ReadbackStats::CpPacket(GfxQueueId, static_cast<u32>(opcode));
            if (VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) != 0) {
                CountGameCommand(header, opcode, count);
            }
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::NativeDrawTag: {
                    if (nop->header.count.Value() + 1 >= VideoCore::NativeTagPayloadDwords) {
                        native_tag_pending.valid = true;
                        native_tag_pending.meta = nop->data_block[1];
                        std::memcpy(&native_tag_pending.k1, &nop->data_block[2], sizeof(u64));
                        std::memcpy(&native_tag_pending.k2, &nop->data_block[4], sizeof(u64));
                        std::memcpy(&native_tag_pending.k3, &nop->data_block[6], sizeof(u64));
                        std::memcpy(&native_tag_pending.seq, &nop->data_block[8], sizeof(u64));
                        native_follow_pending = (native_tag_pending.meta & VideoCore::NativeTagFollowFlag) != 0;
                        if (native_follow_pending) {
                            // executed when the draw packet comes
                        } else if ((native_tag_pending.meta & VideoCore::NativeTagExecFlag) != 0) {
                            ExecuteNativeDraw();
                        }
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    RasterRun(
                        [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip); });
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        RasterRun([this, label = std::string(label)] {
                            rasterizer->ScopeMarkerBegin(label, true);
                        });
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        RasterRun([this, label = std::string(label), color] {
                            rasterizer->ScopedMarkerInsertColor(label, color, true);
                        });
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        RasterRun([this] { rasterizer->ScopeMarkerEnd(true); });
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                MirrorSetDefaults();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                MirrorRegs(reg_addr, count - 1);
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                MirrorRegs(reg_addr, count - 1);

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                    MirrorRegs(Regs::ShRegWordOffset + set_data->reg_offset,
                               set_size / sizeof(u32));
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                MirrorRegs(Regs::UconfigRegWordOffset + set_data->reg_offset, count - 1);
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                MirrorObject(&regs.index_buffer_type, sizeof(regs.index_buffer_type));
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                MirrorObject(&regs.max_index_size, sizeof(regs.max_index_size));
                MirrorObject(&regs.index_base_address, sizeof(regs.index_base_address));
                MirrorObject(&regs.num_indices, sizeof(regs.num_indices));
                MirrorObject(&regs.draw_initiator, sizeof(regs.draw_initiator));
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::Draw;
                call.marker = "gfx:{}:DrawIndex2";
                call.cmd_address = header;
                call.indexed = true;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                MirrorObject(&regs.max_index_size, sizeof(regs.max_index_size));
                MirrorObject(&regs.num_indices, sizeof(regs.num_indices));
                MirrorObject(&regs.draw_initiator, sizeof(regs.draw_initiator));
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::Draw;
                call.marker = "gfx:{}:DrawIndexOffset2";
                call.cmd_address = header;
                call.indexed = true;
                call.index_offset = draw_index_off->index_offset;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                MirrorObject(&regs.num_indices, sizeof(regs.num_indices));
                MirrorObject(&regs.draw_initiator, sizeof(regs.draw_initiator));
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                if (native_follow_pending) {
                    // the game skipped PrepareDraw for this draw: its registers come from the record
                    native_follow_pending = false;
                    ExecuteNativeDraw(nullptr, ~0u, draw_index->index_count);
                    native_tag_pending.valid = false;
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::Draw;
                call.marker = "gfx:{}:DrawIndexAuto";
                call.cmd_address = header;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::DrawIndirect;
                call.marker = "gfx:{}:DrawIndirect";
                call.cmd_address = header;
                call.args_addr = indirect_args_addr;
                call.offset = offset;
                call.stride = stride;
                call.max_count = 1;
                call.vertex_sgpr_offset = draw_indirect->base_vtx_loc;
                call.instance_sgpr_offset = draw_indirect->start_inst_loc;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DrawIndirectCountMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectCountMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::DrawIndirect;
                call.marker = "gfx:{}:DrawIndirectCountMulti";
                call.cmd_address = header;
                call.args_addr = indirect_args_addr;
                call.offset = offset;
                call.stride = draw_indirect->stride;
                call.max_count = draw_indirect->count;
                call.count_addr = draw_indirect->count_addr;
                call.vertex_sgpr_offset = draw_indirect->base_vtx_loc;
                call.instance_sgpr_offset = draw_indirect->start_inst_loc;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::DrawIndirect;
                call.marker = "gfx:{}:DrawIndirectMulti";
                call.cmd_address = header;
                call.args_addr = indirect_args_addr;
                call.offset = offset;
                call.stride = draw_indirect->stride;
                call.max_count = draw_indirect->count;
                call.vertex_sgpr_offset = draw_indirect->base_vtx_loc;
                call.instance_sgpr_offset = draw_indirect->start_inst_loc;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                regs.max_index_size = index_buffer_num_indices;
                MirrorObject(&regs.max_index_size, sizeof(regs.max_index_size));
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::DrawIndirect;
                call.marker = "gfx:{}:DrawIndexIndirect";
                call.cmd_address = header;
                call.indexed = true;
                call.args_addr = indirect_args_addr;
                call.offset = offset;
                call.stride = stride;
                call.max_count = 1;
                call.vertex_sgpr_offset = draw_index_indirect->base_vtx_loc;
                call.instance_sgpr_offset = draw_index_indirect->start_inst_loc;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                regs.max_index_size = index_buffer_num_indices;
                MirrorObject(&regs.max_index_size, sizeof(regs.max_index_size));
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::DrawIndirect;
                call.marker = "gfx:{}:DrawIndexIndirectMulti";
                call.cmd_address = header;
                call.indexed = true;
                call.args_addr = indirect_args_addr;
                call.offset = offset;
                call.stride = draw_index_indirect->stride;
                call.max_count = draw_index_indirect->count;
                call.vertex_sgpr_offset = draw_index_indirect->base_vtx_loc;
                call.instance_sgpr_offset = draw_index_indirect->start_inst_loc;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                regs.max_index_size = index_buffer_num_indices;
                MirrorObject(&regs.max_index_size, sizeof(regs.max_index_size));
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::DrawIndirect;
                call.marker = "gfx:{}:DrawIndexIndirectCountMulti";
                call.cmd_address = header;
                call.indexed = true;
                call.args_addr = indirect_args_addr;
                call.offset = offset;
                call.stride = draw_index_indirect->stride;
                call.max_count = draw_index_indirect->count;
                call.count_addr = draw_index_indirect->count_indirect_enable.Value()
                                      ? draw_index_indirect->count_addr
                                      : 0;
                call.vertex_sgpr_offset = draw_index_indirect->base_vtx_loc;
                call.instance_sgpr_offset = draw_index_indirect->start_inst_loc;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::DispatchDirect;
                call.marker = "gfx:{}:DispatchDirect";
                call.cmd_address = header;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                RasterCall call{};
                call.kind = RasterCall::Kind::DispatchIndirect;
                call.marker = "gfx:{}:DispatchIndirect";
                call.cmd_address = header;
                call.args_addr = indirect_args_addr;
                call.offset = offset;
                call.dispatch_size = size;
                RasterDispatch(call);
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                MirrorObject(&regs.num_instances, sizeof(regs.num_instances));
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                MirrorObject(&regs.index_base_address, sizeof(regs.index_base_address));
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                index_buffer_num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                    MirrorObject(&regs.cp_strmout_cntl, sizeof(regs.cp_strmout_cntl));
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        const u64 counter_value = pixel_counter | OcclusionCounterValidMask;
                        const s32 pairs = num_counter_pairs;
                        RasterRun([this, results, counter_value, pairs] {
                            if (rasterizer) {
                                rasterizer->NoteCpGuestWrite(reinterpret_cast<VAddr>(results),
                                                             pairs * 16);
                            }
                            u64* out = results;
                            for (s32 i = 0; i < pairs; ++i, out += 2) {
                                *out = counter_value;
                            }
                        });
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                // The packet is copied: the signal may run after the command buffer is reused.
                auto signal = [this, eos = *event_eos] {
                    RunFenceSignal(
                        *this, gpu_id,
                        [&] {
                            bool backed = true;
                            eos.SignalFence([&](void* address, u64 data, u32 num_bytes) {
                                backed &=
                                    TryWriteFenceOffThread(rasterizer, address, data, num_bytes);
                            });
                            return backed;
                        },
                        [eos] { eos.SignalFence(WriteFenceMemory); });
                };
                // GdsStore reads GDS on this thread right after, keep it synchronous.
                const bool is_gds_store =
                    event_eos->command == PM4CmdEventWriteEos::Command::GdsStore;
                SignalFenceAfterReadbacks(
                    std::move(signal), is_gds_store, is_gds_store ? 0 : event_eos->Address<VAddr>(),
                    event_eos->DataDWord(), sizeof(u32), GfxQueueId, VideoCore::FenceGfxEos);
                if (is_gds_store) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        auto* const destination = event_eos->Address();
                        const auto gds_index = event_eos->gds_index;
                        RasterRun(
                            [this, destination, gds_index] {
                                VideoCore::ReadbackStats::CpPhase phase{"EOS GDS store (Finish)"};
                                VideoCore::ReadbackStats::WaitScope wait_scope{
                                    VideoCore::ReadbackStats::WaitReason::GdsStore};
                                rasterizer->Finish();
                                *destination = rasterizer->ReadDataFromGds(gds_index);
                            },
                            true);
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                // The packet is copied: the signal may run after the command buffer is reused.
                static constexpr auto irq = [] {
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                };
                auto signal = [this, eop = *event_eop] {
                    RunFenceSignal(
                        *this, gpu_id,
                        [&] {
                            bool backed = true;
                            eop.SignalFence(
                                [&](void* address, u64 data, u32 num_bytes) {
                                    backed &= TryWriteFenceOffThread(rasterizer, address, data,
                                                                     num_bytes);
                                },
                                [] {});
                            if (backed) {
                                // Writes landed, now raise the interrupt (if any) after them.
                                eop.SignalFence(NoFenceWrite, irq);
                            }
                            return backed;
                        },
                        [eop] { eop.SignalFence(WriteFenceMemory, irq); });
                };
                const auto eop_data_sel = event_eop->data_sel.Value();
                const u32 eop_bytes = eop_data_sel == DataSelect::Data32Low ? sizeof(u32)
                                      : eop_data_sel == DataSelect::Data64  ? sizeof(u64)
                                                                            : 0;
                SignalFenceAfterReadbacks(
                    std::move(signal), false, reinterpret_cast<VAddr>(event_eop->Address<u8>()),
                    event_eop->DataQWord(), eop_bytes, GfxQueueId, VideoCore::FenceGfxEop);
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                ASSERT(dma_data->command.das == 0);
                RasterRun([this, dma = *dma_data] {
                    const auto* dma_data = &dma;
                    if (dma_data->src_sel == DmaDataSrc::Data &&
                        dma_data->dst_sel == DmaDataDst::Gds) {
                        rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                               dma_data->data, true);
                    } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                                dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                               dma_data->dst_sel == DmaDataDst::Gds) {
                        rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                               dma_data->NumBytes(), true, false);
                    } else if (dma_data->src_sel == DmaDataSrc::Data &&
                               (dma_data->dst_sel == DmaDataDst::Memory ||
                                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                               dma_data->data, false);
                    } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                               (dma_data->dst_sel == DmaDataDst::Memory ||
                                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                               dma_data->src_addr_lo, dma_data->NumBytes(), false,
                                               true);
                    } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                                dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                               (dma_data->dst_sel == DmaDataDst::Memory ||
                                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                               dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                               false, false);
                    } else {
                        UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                        u32(dma_data->src_sel), u32(dma_data->dst_sel));
                    }
                }, false, true);
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (write_data->wr_one_addr.Value()) {
                    UNREACHABLE();
                }
                CpWriteData(address, write_data->data, data_size, GfxQueueId);
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    const u64 wait_start = VideoCore::ReadbackStats::NowNs();
                    bool reported = false;
                    bool blocked = false;
                    while (!mem_semaphore->Signaled()) {
                        blocked = true;
                        ReportLongSpin("gfx MemSemaphore", mem_semaphore, wait_start, reported);
                        YIELD_GFX();
                    }
                    if (blocked) {
                        VideoCore::ReadbackStats::OnSemaphoreWait(
                            VideoCore::ReadbackStats::NowNs() - wait_start);
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                const u64 wait_start = VideoCore::ReadbackStats::NowNs();
                bool reported = false;
                bool blocked = false;
                while (!rewind->Valid()) {
                    blocked = true;
                    ReportLongSpin("gfx Rewind", rewind, wait_start, reported);
                    YIELD_GFX();
                }
                if (blocked) {
                    VideoCore::ReadbackStats::OnSemaphoreWait(VideoCore::ReadbackStats::NowNs() -
                                                              wait_start);
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    VideoCore::ReadbackStats::CpTimer timer{
                        VideoCore::ReadbackStats::CpTime::FlipSleep};
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                if (!wait_reg_mem->Test(regs.reg_array)) {
                    using VideoCore::ReadbackStats::WaitClass;
                    const u64 wait_start = VideoCore::ReadbackStats::NowNs();
                    bool reported = false;
                    bool blocked = false;
                    bool held_back = false;
                    WaitClass wait_class{};
                    while (true) {
                        const FenceMatch match = MatchRecordedFence(*wait_reg_mem);
                        if (match.satisfied && FenceShortcutEnabled(false)) {
                            OnWaitPassedOnFence(*wait_reg_mem, match, GfxQueueId);
                            wait_class = blocked ? WaitClass::Producer : WaitClass::FencePass;
                            break;
                        }
                        held_back |= match.satisfied;
                        if (wait_reg_mem->Test(regs.reg_array)) {
                            wait_class = held_back ? WaitClass::FenceBlocked : WaitClass::Memory;
                            break;
                        }
                        blocked = true;
                        ReportLongWait(*wait_reg_mem, "gfx", wait_start, reported);
                        VideoCore::ReadbackStats::CpWaitYield(vo_port->IsVoLabel(wait_addr));
                        YIELD_GFX();
                    }
                    if (VideoCore::ReadbackStats::IsEnabled()) {
                        VideoCore::ReadbackStats::OnCommandWait(
                            false, wait_reg_mem->Address<VAddr>(), wait_class,
                            blocked ? VideoCore::ReadbackStats::NowNs() - wait_start : 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                task.handle.destroy();
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                DrainRaster(); // The condition may be written by earlier work.
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        VideoCore::ReadbackStats::CpPacket(vqid + 1, static_cast<u32>(opcode));
        CountComputeQueueCommand(opcode);

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            if (nop->header.count.Value() != 0 && nop->data_block[0] == PM4CmdNop::PayloadType::NativeDrawTag &&
                nop->header.count.Value() + 1 >= VideoCore::NativeTagPayloadDwords) {
                native_tag_pending.valid = true;
                native_tag_pending.meta = nop->data_block[1];
                std::memcpy(&native_tag_pending.k1, &nop->data_block[2], sizeof(u64));
                std::memcpy(&native_tag_pending.k2, &nop->data_block[4], sizeof(u64));
                std::memcpy(&native_tag_pending.k3, &nop->data_block[6], sizeof(u64));
                std::memcpy(&native_tag_pending.seq, &nop->data_block[8], sizeof(u64));
                if ((native_tag_pending.meta & VideoCore::NativeTagExecFlag) != 0) {
                    ExecuteNativeDraw(header, vqid);
                }
            }
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            task.handle.destroy();
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            ASSERT(dma_data->command.das == 0);
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                RasterRun([this, dma = *dma_data] {
                    rasterizer->FillBuffer(dma.dst_addr_lo, dma.NumBytes(), dma.data, true);
                }, false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                RasterRun([this, dma = *dma_data] {
                    rasterizer->CopyBuffer(dma.dst_addr_lo, dma.SrcAddress<VAddr>(),
                                           dma.NumBytes(), true, false);
                }, false, true);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                RasterRun([this, dma = *dma_data] {
                    rasterizer->FillBuffer(dma.DstAddress<VAddr>(), dma.NumBytes(), dma.data,
                                           false);
                }, false, true);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                RasterRun([this, dma = *dma_data] {
                    rasterizer->CopyBuffer(dma.DstAddress<VAddr>(), dma.src_addr_lo,
                                           dma.NumBytes(), false, true);
                }, false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    RasterRun([this, dst_addr, src_addr, num_bytes] {
                        rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                    }, false, true);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
                MirrorRegs(Regs::ShRegWordOffset + set_data->reg_offset, set_size / sizeof(u32));
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                RasterCall call{};
                call.kind = RasterCall::Kind::DispatchIndirect;
                call.no_marker = true;
                call.args_addr = it->indirect_addr;
                call.dispatch_size = size;
                RasterDispatch(call);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            RasterCall call{};
            call.kind = RasterCall::Kind::DispatchDirect;
            call.asc = true;
            call.vqid = vqid;
            call.marker = "asc[{}]:{}:DispatchDirect";
            call.cmd_address = header;
            RasterDispatch(call);
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            RasterCall call{};
            call.kind = RasterCall::Kind::DispatchIndirect;
            call.asc = true;
            call.vqid = vqid;
            call.marker = "asc[{}]:{}:DispatchIndirect";
            call.cmd_address = header;
            call.args_addr = ib_address;
            call.dispatch_size = size;
            RasterDispatch(call);
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (write_data->wr_one_addr.Value()) {
                UNREACHABLE();
            }
            CpWriteData(write_data->Address<void*>(), write_data->data, data_size, vqid + 1);
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                const u64 wait_start = VideoCore::ReadbackStats::NowNs();
                bool reported = false;
                bool blocked = false;
                while (!mem_semaphore->Signaled()) {
                    blocked = true;
                    ReportLongSpin("compute MemSemaphore", mem_semaphore, wait_start, reported);
                    YIELD_ASC(vqid);
                }
                if (blocked) {
                    VideoCore::ReadbackStats::OnSemaphoreWait(VideoCore::ReadbackStats::NowNs() -
                                                              wait_start);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            if (!wait_reg_mem->Test(regs.reg_array)) {
                using VideoCore::ReadbackStats::WaitClass;
                const u64 wait_start = VideoCore::ReadbackStats::NowNs();
                bool reported = false;
                bool blocked = false;
                bool held_back = false;
                WaitClass wait_class{};
                while (true) {
                    const FenceMatch match = MatchRecordedFence(*wait_reg_mem);
                    if (match.satisfied && FenceShortcutEnabled(true)) {
                        OnWaitPassedOnFence(*wait_reg_mem, match, vqid + 1);
                        wait_class = blocked ? WaitClass::Producer : WaitClass::FencePass;
                        break;
                    }
                    held_back |= match.satisfied;
                    if (wait_reg_mem->Test(regs.reg_array)) {
                        wait_class = held_back ? WaitClass::FenceBlocked : WaitClass::Memory;
                        break;
                    }
                    blocked = true;
                    ReportLongWait(*wait_reg_mem, "compute", wait_start, reported);
                    VideoCore::ReadbackStats::CpWaitYield(false);
                    YIELD_ASC(vqid);
                }
                if (VideoCore::ReadbackStats::IsEnabled()) {
                    VideoCore::ReadbackStats::OnCommandWait(
                        true, wait_reg_mem->Address<VAddr>(), wait_class,
                        blocked ? VideoCore::ReadbackStats::NowNs() - wait_start : 0);
                }
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            // The packet is copied: the signal may run after the command buffer is reused.
            auto signal = [this, release = *release_mem, pipe_id = queue.pipe_id] {
                const auto irq = [pipe_id] {
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                };
                const auto gds_to_mem = [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                };
                RunFenceSignal(
                    *this, gpu_id,
                    [&] {
                        bool backed = true;
                        release.SignalFence(
                            [&](void* address, u64 data, u32 num_bytes) {
                                backed &=
                                    TryWriteFenceOffThread(rasterizer, address, data, num_bytes);
                            },
                            [] {}, [](VAddr, u16, u16) {});
                        if (backed) {
                            release.SignalFence(NoFenceWrite, irq, [](VAddr, u16, u16) {});
                        }
                        return backed;
                    },
                    [release, irq, gds_to_mem] {
                        release.SignalFence(WriteFenceMemory, irq, gds_to_mem);
                    });
            };
            // Copying from GDS records GPU work, which must happen on the raster thread.
            const auto release_data_sel = release_mem->data_sel.Value();
            const u32 release_bytes = release_data_sel == DataSelect::Data32Low ? sizeof(u32)
                                      : release_data_sel == DataSelect::Data64  ? sizeof(u64)
                                                                                : 0;
            SignalFenceAfterReadbacks(std::move(signal),
                                      release_data_sel == DataSelect::GdsMemStore,
                                      release_mem->Address<VAddr>(), release_mem->DataQWord(),
                                      release_bytes, vqid + 1, VideoCore::FenceComputeRelease);
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    submit_cv.notify_one();
}

} // namespace AmdGpu
