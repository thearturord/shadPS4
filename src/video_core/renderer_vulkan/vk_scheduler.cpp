// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>
#include "common/assert.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/thread_usage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    BeginSession();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

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

    const auto recorder = CommandBuffer();
    recorder.beginRendering(rendering_info);
    recorder.record([this](vk::CommandBuffer) { replay_rendering = true; });
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    CommandBuffer().record([this](vk::CommandBuffer cmdbuf) {
        cmdbuf.endRendering();
        replay_rendering = false;
    });
}

vk::CommandBuffer Scheduler::UploadCommandBuffer() {
    auto& upload_cmdbuf = sessions.back().upload;
    if (upload_cmdbuf) {
        return upload_cmdbuf;
    }
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    upload_cmdbuf = command_pool.Commit();
    Check(upload_cmdbuf.begin(begin_info));
    return upload_cmdbuf;
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
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
    // Deferred write backs (async readbacks) for work that is now complete must land in guest
    // memory before anything that relies on Finish() reads it.
    WaitPriorityOperations();
}

void Scheduler::Wait(u64 tick) {
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    work_semaphore.Wait(tick);
}

void Scheduler::WaitPriorityOperations() {
    if (num_priority_ops.load(std::memory_order_acquire) == 0) {
        return;
    }
    // Normally this is a short wait for write backs of already completed GPU work. Poll so a
    // wait that never ends gets reported (hang diagnostics).
    const auto start = std::chrono::steady_clock::now();
    bool reported = false;
    u32 pending;
    while ((pending = num_priority_ops.load(std::memory_order_acquire)) != 0) {
        if (!reported && std::chrono::steady_clock::now() - start > std::chrono::seconds{2}) {
            reported = true;
            LOG_WARNING(Render_Vulkan,
                        "Hang check: waiting for {} priority operations to finish (GPU reached "
                        "tick {}, CPU recording {})",
                        pending, work_semaphore.KnownGpuTick(), work_semaphore.CurrentTick());
        }
        std::this_thread::yield();
    }
}

void Scheduler::PopPendingOperations() {
    std::unique_lock lk(pending_ops_mutex);
    work_semaphore.Refresh();
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::BeginSession() {
    EndSession();

    auto& session = sessions.emplace_back();
    if (use_encoder && !direct_active) {
        // Ticks are recorded into lists and replayed by the encoder thread: no command buffer here.
        dynamic_state.Invalidate();
        return;
    }

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    {
        std::scoped_lock lk{command_pool_mutex};
        session.primary = command_pool.Commit();
    }
    Check(session.primary.begin(begin_info));

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

void Scheduler::EndSession() {
    if (sessions.empty()) {
        return;
    }

    if (on_session) {
        on_session();
    }

    const auto& session = sessions.back();
    if (session.upload) {
        Check(session.upload.end());
    }

    EndRendering();
    if (session.primary) {
        Check(session.primary.end());
    }
}

void Scheduler::SetEncoderMode(u32 mode) {
    if (mode == 0) {
        SyncedCommandBuffer();
    }
    record_commands = mode >= 1;
    if (mode >= 2 && !use_encoder) {
        use_encoder = true;
        // The command buffer of the session open at construction is still open: it becomes the
        // first direct tick, submitted by the ordering thread. Later ticks go to the encoder.
        direct_active = true;
        encoder_sync = mode == 3;
        encoder_thread = std::jthread(std::bind_front(&Scheduler::EncoderThread, this));
        LOG_INFO(Render_Vulkan, "Command processor: Vulkan encoding on an encoder thread{}",
                 encoder_sync ? " (synchronous hand-off, diagnostic)" : "");
    }
}

vk::CommandBuffer Scheduler::SyncedCommandBuffer() {
    if (use_encoder && !direct_active) {
        // Everything handed over must be submitted before commands recorded here, which are
        // submitted by this thread at the end of the tick.
        WaitEncoderIdle();
        direct_active = true;
        const vk::CommandBufferBeginInfo begin_info = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };
        auto& session = sessions.back();
        {
            std::scoped_lock lk{command_pool_mutex};
            session.primary = command_pool.Commit();
        }
        Check(session.primary.begin(begin_info));
        dynamic_state.Invalidate();
    }
    const vk::CommandBuffer cmdbuf = sessions.back().primary;
    if (record_commands && !current_list->Empty()) {
        current_list->Replay(cmdbuf);
    }
    return cmdbuf;
}

void Scheduler::WaitEncoderIdle() {
    std::unique_lock lk{encoder_mutex};
    encoder_cv.wait(lk, [this] { return jobs_done == jobs_handed; });
}

void Scheduler::HandOffToEncoder(SubmitInfo& info, u64 signal_value) {
    info.AddSignal(work_semaphore.Handle(), signal_value);
    EncoderJob job{
        .info = info,
        .signal_value = signal_value,
    };
    {
        std::unique_lock lk{encoder_mutex};
        // Back-pressure: the ordering thread may run at most a few ticks ahead of the encoder.
        static const size_t max_queued =
            std::max<size_t>(1, EmulatorSettings.GetCpEncoderQueueDepth());
        encoder_cv.wait(lk, [this] { return encoder_jobs.size() < max_queued; });
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
    if (encoder_sync) {
        // Diagnostic mode 3: the ordering thread never runs ahead of the submission.
        WaitEncoderIdle();
    }
}

void Scheduler::EncoderThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuEncoder");
    if (const s32 core = EmulatorSettings.GetCpEncoderCore(); core >= 0) {
        VideoCore::PinThreadAndIsolateCore(static_cast<u32>(core));
    }
    if (const u32 priority = EmulatorSettings.GetCpThreadPriority(); priority >= 2) {
        Common::SetCurrentThreadPriority(static_cast<Common::ThreadPriority>(priority));
    }
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

        vk::CommandBuffer cmdbuf;
        {
            std::scoped_lock lk{command_pool_mutex};
            cmdbuf = command_pool.Commit();
        }
        Check(cmdbuf.begin(begin_info));
        job.list->Replay(cmdbuf);
        Check(cmdbuf.end());
        {
            // ImGui texture uploads submit to the queue too: done here, the ordering thread
            // never waits for the submit lock while this thread is in the driver.
            std::scoped_lock lk{submit_mutex};
            SubmitCommandBuffers(std::span{&cmdbuf, 1}, job.info);
        }
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
        const u64 signal_value = work_semaphore.NextTick();
        {
            std::scoped_lock lk{submit_mutex};
            if (on_submit) {
                on_submit(info);
            }
        }
        if (on_session) {
            on_session();
        }
        HandOffToEncoder(info, signal_value);
        // The next tick is replayed into a new command buffer: set all dynamic state again.
        dynamic_state.Invalidate();
        work_semaphore.Refresh();
        PopPendingOperations();
        return;
    }

    if (record_commands && !current_list->Empty()) {
        current_list->Replay(sessions.back().primary);
    }

    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = work_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }

    EndSession();

    std::vector<vk::CommandBuffer> cmd_buffers;
    cmd_buffers.reserve(sessions.size() * 2);

    for (const auto& session : sessions) {
        if (session.upload) {
            cmd_buffers.push_back(session.upload);
        }
        cmd_buffers.push_back(session.primary);
    }
    sessions.clear();

    const vk::Semaphore timeline = work_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    SubmitCommandBuffers(cmd_buffers, info);

    work_semaphore.Refresh();
    if (use_encoder) {
        // A direct tick (after SyncedCommandBuffer) is done: the next ones go to the encoder.
        direct_active = false;
    }
    BeginSession();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::SubmitCommandBuffers(std::span<const vk::CommandBuffer> cmd_buffers,
                                     SubmitInfo& info) {
    std::array<vk::PipelineStageFlags, std::tuple_size_v<decltype(info.wait_semas)>>
        wait_stage_masks;
    wait_stage_masks.fill(vk::PipelineStageFlagBits::eAllCommands);

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
        .commandBufferCount = static_cast<u32>(cmd_buffers.size()),
        .pCommandBuffers = cmd_buffers.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
}

namespace {
thread_local bool is_priority_thread = false;
} // Anonymous namespace

bool Scheduler::OnPriorityThread() noexcept {
    return is_priority_thread;
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");
    is_priority_thread = true;

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
        while (!work_semaphore.WaitFor(op.gpu_tick, ReportAfterNs) && !stoken.stop_requested()) {
            if (!reported) {
                reported = true;
                LOG_WARNING(
                    Render_Vulkan,
                    "Hang check: priority operation still waiting for GPU tick {} (GPU "
                    "reached {}, CPU recording {}){}",
                    op.gpu_tick, work_semaphore.KnownGpuTick(), work_semaphore.CurrentTick(),
                    op.gpu_tick >= work_semaphore.CurrentTick() ? " - tick was never submitted"
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
