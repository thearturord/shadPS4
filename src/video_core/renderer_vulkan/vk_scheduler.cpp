// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>
#include "common/assert.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/readback_stats.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
    if (timing_pool) {
        instance.GetDevice().destroyQueryPool(timing_pool);
    }
}

void Scheduler::EnableGpuTiming() {
    const vk::QueryPoolCreateInfo pool_ci = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = NumTimingSlots * 2,
    };
    const auto [result, pool] = instance.GetDevice().createQueryPool(pool_ci);
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "GPU timing disabled: createQueryPool failed ({})",
                    vk::to_string(result));
        return;
    }
    timing_pool = pool;
    timestamp_period = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    LOG_INFO(Render_Vulkan, "GPU timing enabled (timestamp period {} ns)", timestamp_period);
}

void Scheduler::ReadGpuTiming(u32 slot) {
    std::array<u64, 2> stamps{};
    const auto result = instance.GetDevice().getQueryPoolResults(
        timing_pool, slot * 2, 2, sizeof(stamps), stamps.data(), sizeof(u64),
        vk::QueryResultFlagBits::e64);
    if (result != vk::Result::eSuccess || stamps[1] < stamps[0]) {
        return;
    }
    // Command buffers may overlap on the GPU: only count time not counted already.
    const u64 start = std::max(stamps[0], last_timing_end);
    if (stamps[1] > start) {
        VideoCore::ReadbackStats::OnGpuBusy(
            static_cast<u64>(static_cast<double>(stamps[1] - start) * timestamp_period));
    }
    last_timing_end = std::max(last_timing_end, stamps[1]);
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;
    if (timing_pool) {
        VideoCore::ReadbackStats::OnRenderPassBegin();
    }

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    CommandBuffer().beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    CommandBuffer().endRendering();
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 start = VideoCore::ReadbackStats::NowNs();
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
    // Deferred write backs (async readbacks) for work that is now complete must land in guest
    // memory before anything that relies on Finish() reads it.
    WaitPriorityOperations();
    if (VideoCore::ReadbackStats::IsEnabled()) {
        VideoCore::ReadbackStats::OnFinish(VideoCore::ReadbackStats::NowNs() - start);
    }
}

void Scheduler::Wait(u64 tick) {
    VideoCore::ReadbackStats::CpPhase phase{"waiting for a GPU tick (Scheduler::Wait)"};
    VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::GpuWait};
    if (tick >= master_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    master_semaphore.Wait(tick);
}

void Scheduler::WaitPriorityOperations() {
    if (num_priority_ops.load(std::memory_order_acquire) == 0) {
        return;
    }
    // Normally this is a short wait for write backs of already completed GPU work. Poll so a
    // wait that never ends gets reported (hang diagnostics).
    VideoCore::ReadbackStats::CpPhase phase{"waiting for priority operations (write backs)"};
    VideoCore::ReadbackStats::CpTimer timer{VideoCore::ReadbackStats::CpTime::GpuWait};
    const auto start = std::chrono::steady_clock::now();
    bool reported = false;
    u32 pending;
    while ((pending = num_priority_ops.load(std::memory_order_acquire)) != 0) {
        if (!reported && std::chrono::steady_clock::now() - start > std::chrono::seconds{2}) {
            reported = true;
            LOG_WARNING(Render_Vulkan,
                        "Hang check: waiting for {} priority operations to finish (GPU reached "
                        "tick {}, CPU recording {})",
                        pending, master_semaphore.KnownGpuTick(), master_semaphore.CurrentTick());
        }
        std::this_thread::yield();
    }
}

void Scheduler::PopPendingOperations() {
    std::unique_lock lk(pending_ops_mutex);
    // Runs at every draw and dispatch: only ask the driver for the GPU progress when there is an
    // operation it could release.
    if (pending_ops.empty()) {
        return;
    }
    if (!master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        master_semaphore.Refresh();
    }
    while (!pending_ops.empty() && master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    {
        std::scoped_lock lk{command_pool_mutex};
        current_cmdbuf = command_pool.Commit();
    }
    Check(current_cmdbuf.begin(begin_info));

    current_timing_slot = -1;
    if (timing_pool) {
        current_timing_slot = static_cast<s32>(next_timing_slot++ % NumTimingSlots);
        current_cmdbuf.resetQueryPool(timing_pool, current_timing_slot * 2, 2);
        current_cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timing_pool,
                                      current_timing_slot * 2);
    }

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SetEncoderMode(u32 mode) {
    if (mode == 0) {
        SyncedCommandBuffer();
    }
    record_commands = mode >= 1;
    if (mode >= 2 && !use_encoder) {
        use_encoder = true;
        // The command buffer allocated at construction is still open: it becomes the first
        // direct tick, submitted by the ordering thread. Later ticks go to the encoder.
        direct_active = true;
        encoder_thread = std::jthread(std::bind_front(&Scheduler::EncoderThread, this));
        LOG_INFO(Render_Vulkan, "Command processor: Vulkan encoding on an encoder thread");
    }
}

vk::CommandBuffer Scheduler::SyncedCommandBuffer() {
    if (use_encoder && !direct_active) {
        // Everything handed over must be submitted before commands recorded here, which are
        // submitted by this thread at the end of the tick.
        WaitEncoderIdle();
        AllocateWorkerCommandBuffers();
        direct_active = true;
    }
    if (record_commands && !current_list->Empty()) {
        current_list->Replay(current_cmdbuf);
    }
    return current_cmdbuf;
}

void Scheduler::WaitEncoderIdle() {
    std::unique_lock lk{encoder_mutex};
    encoder_cv.wait(lk, [this] { return jobs_done == jobs_handed; });
}

void Scheduler::HandOffToEncoder(SubmitInfo& info, u64 signal_value) {
    info.AddSignal(master_semaphore.Handle(), signal_value);
    EncoderJob job{
        .info = info,
        .signal_value = signal_value,
    };
    if (timing_pool) {
        const u32 slot = next_timing_slot++ % NumTimingSlots;
        job.timing_slot = static_cast<s32>(slot);
        std::unique_lock lk(pending_ops_mutex);
        pending_ops.emplace([this, slot] { ReadGpuTiming(slot); }, signal_value);
    }
    {
        std::unique_lock lk{encoder_mutex};
        // Back-pressure: the ordering thread may run at most a few ticks ahead of the encoder.
        encoder_cv.wait(lk, [this] { return encoder_jobs.size() < MaxQueuedLists; });
        job.list = std::move(current_list_owner);
        if (free_lists.empty()) {
            current_list_owner = std::make_unique<CommandList>();
        } else {
            current_list_owner = std::move(free_lists.back());
            free_lists.pop_back();
        }
        current_list = current_list_owner.get();
        encoder_jobs.push_back(std::move(job));
        ++jobs_handed;
    }
    encoder_cv.notify_all();
}

void Scheduler::SubmitCommandBuffer(vk::CommandBuffer cmdbuf, SubmitInfo& info) {
    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    VideoCore::ReadbackStats::OnVkSubmit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
}

void Scheduler::EncoderThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuEncoder");
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    while (true) {
        EncoderJob job;
        {
            std::unique_lock lk{encoder_mutex};
            encoder_cv.wait(lk, stoken, [this] { return !encoder_jobs.empty(); });
            if (encoder_jobs.empty()) {
                return; // Stop requested and nothing left to submit.
            }
            job = std::move(encoder_jobs.front());
            encoder_jobs.pop_front();
        }
        encoder_cv.notify_all();

        const u64 start = VideoCore::ReadbackStats::NowNs();
        vk::CommandBuffer cmdbuf;
        {
            std::scoped_lock lk{command_pool_mutex};
            cmdbuf = command_pool.Commit();
        }
        Check(cmdbuf.begin(begin_info));
        if (job.timing_slot >= 0) {
            const u32 slot = static_cast<u32>(job.timing_slot);
            cmdbuf.resetQueryPool(timing_pool, slot * 2, 2);
            cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timing_pool, slot * 2);
        }
        const u64 commands = job.list->Size();
        job.list->Replay(cmdbuf);
        if (job.timing_slot >= 0) {
            const u32 slot = static_cast<u32>(job.timing_slot);
            cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, timing_pool,
                                  slot * 2 + 1);
        }
        Check(cmdbuf.end());
        {
            std::scoped_lock lk{submit_mutex};
            SubmitCommandBuffer(cmdbuf, job.info);
        }
        VideoCore::ReadbackStats::OnCommandReplay(commands,
                                                  VideoCore::ReadbackStats::NowNs() - start);
        {
            std::unique_lock lk{encoder_mutex};
            free_lists.push_back(std::move(job.list));
            ++jobs_done;
        }
        encoder_cv.notify_all();
    }
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    // Recorded commands go into the command buffer before anything written directly at the end.
    EndRendering();

    if (use_encoder && !direct_active) {
        const u64 signal_value = master_semaphore.NextTick();
        {
            std::scoped_lock lk{submit_mutex};
            ImGui::Core::TextureManager::Submit();
        }
        if (timing_pool) {
            VideoCore::ReadbackStats::OnRasterizerSubmit();
        }
        HandOffToEncoder(info, signal_value);
        // The next tick is replayed into a new command buffer: set all dynamic state again.
        dynamic_state.Invalidate();
        master_semaphore.Refresh();
        PopPendingOperations();
        return;
    }

    if (record_commands && !current_list->Empty()) {
        const u64 replay_start = VideoCore::ReadbackStats::NowNs();
        const u64 commands = current_list->Size();
        current_list->Replay(current_cmdbuf);
        VideoCore::ReadbackStats::OnCommandReplay(commands,
                                                  VideoCore::ReadbackStats::NowNs() - replay_start);
    }

    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = master_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (current_timing_slot >= 0) {
        const u32 slot = static_cast<u32>(current_timing_slot);
        current_cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, timing_pool,
                                      slot * 2 + 1);
        std::unique_lock lk(pending_ops_mutex);
        pending_ops.emplace([this, slot] { ReadGpuTiming(slot); }, signal_value);
    }
    Check(current_cmdbuf.end());

    info.AddSignal(master_semaphore.Handle(), signal_value);
    ImGui::Core::TextureManager::Submit();
    if (timing_pool) {
        VideoCore::ReadbackStats::OnRasterizerSubmit();
    }
    SubmitCommandBuffer(current_cmdbuf, info);

    master_semaphore.Refresh();
    if (use_encoder) {
        // A direct tick (after SyncedCommandBuffer) is done: the next ones go to the encoder.
        direct_active = false;
        current_cmdbuf = vk::CommandBuffer{};
        current_timing_slot = -1;
        dynamic_state.Invalidate();
    } else {
        AllocateWorkerCommandBuffers();
    }

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        // Wait in slices so a wait that never completes gets reported (hang diagnostics).
        constexpr u64 ReportAfterNs = 2'000'000'000ULL;
        bool reported = false;
        while (!master_semaphore.WaitFor(op.gpu_tick, ReportAfterNs) && !stoken.stop_requested()) {
            if (!reported) {
                reported = true;
                LOG_WARNING(
                    Render_Vulkan,
                    "Hang check: priority operation still waiting for GPU tick {} (GPU "
                    "reached {}, CPU recording {}){}",
                    op.gpu_tick, master_semaphore.KnownGpuTick(), master_semaphore.CurrentTick(),
                    op.gpu_tick >= master_semaphore.CurrentTick() ? " - tick was never submitted"
                                                                  : "");
            }
        }
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
        num_priority_ops.fetch_sub(1, std::memory_order_acq_rel);
        num_priority_ops.notify_all();
    }
}

void DynamicState::Commit(const Instance& instance, const CommandRecorder& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
    }
}

} // namespace Vulkan
