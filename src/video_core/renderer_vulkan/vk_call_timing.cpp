// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <type_traits>

#include "video_core/readback_stats.h"
#include "video_core/renderer_vulkan/vk_call_timing.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

namespace {

using Dispatcher = std::remove_reference_t<decltype(VULKAN_HPP_DEFAULT_DISPATCHER)>;
using VideoCore::ReadbackStats::CpTime;

template <auto Member, CpTime Category, typename Pfn>
struct Hook;

/// Replaces one function pointer of the dispatcher with a wrapper that times the real call as
/// command processor time of the given category (only on the command processor thread).
template <auto Member, CpTime Category, typename R, typename... Args>
struct Hook<Member, Category, R(VKAPI_PTR*)(Args...)> {
    static inline R(VKAPI_PTR* original)(Args...) = nullptr;

    static R VKAPI_PTR Call(Args... args) {
        VideoCore::ReadbackStats::CpTimer timer{Category};
        return original(args...);
    }

    static void Install(Dispatcher& dispatcher) {
        original = dispatcher.*Member;
        if (original) {
            dispatcher.*Member = &Call;
        }
    }
};

template <auto Member, CpTime Category>
void Install(Dispatcher& dispatcher) {
    using Pfn = std::remove_cvref_t<decltype(dispatcher.*Member)>;
    Hook<Member, Category, Pfn>::Install(dispatcher);
}

} // Anonymous namespace

#define HOOK(name, category) Install<&Dispatcher::name, CpTime::category>(dispatcher)

void InstallVulkanCallTiming() {
    auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;

    // Command recording.
    HOOK(vkCmdBindPipeline, VulkanCmd);
    HOOK(vkCmdBindDescriptorSets, VulkanCmd);
    HOOK(vkCmdPushDescriptorSetKHR, VulkanCmd);
    HOOK(vkCmdPushConstants, VulkanCmd);
    HOOK(vkCmdBindIndexBuffer, VulkanCmd);
    HOOK(vkCmdBindVertexBuffers, VulkanCmd);
    HOOK(vkCmdBindVertexBuffers2, VulkanCmd);
    HOOK(vkCmdBindVertexBuffers2EXT, VulkanCmd);
    HOOK(vkCmdDraw, VulkanCmd);
    HOOK(vkCmdDrawIndexed, VulkanCmd);
    HOOK(vkCmdDrawIndirect, VulkanCmd);
    HOOK(vkCmdDrawIndexedIndirect, VulkanCmd);
    HOOK(vkCmdDrawIndirectCount, VulkanCmd);
    HOOK(vkCmdDrawIndexedIndirectCount, VulkanCmd);
    HOOK(vkCmdDispatch, VulkanCmd);
    HOOK(vkCmdDispatchIndirect, VulkanCmd);
    HOOK(vkCmdPipelineBarrier, VulkanCmd);
    HOOK(vkCmdPipelineBarrier2, VulkanCmd);
    HOOK(vkCmdPipelineBarrier2KHR, VulkanCmd);
    HOOK(vkCmdBeginRendering, VulkanCmd);
    HOOK(vkCmdBeginRenderingKHR, VulkanCmd);
    HOOK(vkCmdEndRendering, VulkanCmd);
    HOOK(vkCmdEndRenderingKHR, VulkanCmd);
    HOOK(vkCmdCopyBuffer, VulkanCmd);
    HOOK(vkCmdCopyImage, VulkanCmd);
    HOOK(vkCmdCopyBufferToImage, VulkanCmd);
    HOOK(vkCmdCopyImageToBuffer, VulkanCmd);
    HOOK(vkCmdFillBuffer, VulkanCmd);
    HOOK(vkCmdUpdateBuffer, VulkanCmd);
    HOOK(vkCmdClearColorImage, VulkanCmd);
    HOOK(vkCmdClearDepthStencilImage, VulkanCmd);
    HOOK(vkCmdResolveImage, VulkanCmd);
    HOOK(vkCmdBlitImage, VulkanCmd);
    HOOK(vkCmdResetQueryPool, VulkanCmd);
    HOOK(vkCmdWriteTimestamp, VulkanCmd);
    HOOK(vkCmdBeginDebugUtilsLabelEXT, VulkanCmd);
    HOOK(vkCmdEndDebugUtilsLabelEXT, VulkanCmd);
    HOOK(vkCmdInsertDebugUtilsLabelEXT, VulkanCmd);
    // Dynamic state.
    HOOK(vkCmdSetViewport, VulkanCmd);
    HOOK(vkCmdSetScissor, VulkanCmd);
    HOOK(vkCmdSetViewportWithCount, VulkanCmd);
    HOOK(vkCmdSetViewportWithCountEXT, VulkanCmd);
    HOOK(vkCmdSetScissorWithCount, VulkanCmd);
    HOOK(vkCmdSetScissorWithCountEXT, VulkanCmd);
    HOOK(vkCmdSetLineWidth, VulkanCmd);
    HOOK(vkCmdSetDepthBias, VulkanCmd);
    HOOK(vkCmdSetBlendConstants, VulkanCmd);
    HOOK(vkCmdSetDepthBounds, VulkanCmd);
    HOOK(vkCmdSetStencilCompareMask, VulkanCmd);
    HOOK(vkCmdSetStencilWriteMask, VulkanCmd);
    HOOK(vkCmdSetStencilReference, VulkanCmd);
    HOOK(vkCmdSetStencilOp, VulkanCmd);
    HOOK(vkCmdSetStencilOpEXT, VulkanCmd);
    HOOK(vkCmdSetStencilTestEnable, VulkanCmd);
    HOOK(vkCmdSetStencilTestEnableEXT, VulkanCmd);
    HOOK(vkCmdSetCullMode, VulkanCmd);
    HOOK(vkCmdSetCullModeEXT, VulkanCmd);
    HOOK(vkCmdSetFrontFace, VulkanCmd);
    HOOK(vkCmdSetFrontFaceEXT, VulkanCmd);
    HOOK(vkCmdSetPrimitiveTopology, VulkanCmd);
    HOOK(vkCmdSetPrimitiveTopologyEXT, VulkanCmd);
    HOOK(vkCmdSetDepthTestEnable, VulkanCmd);
    HOOK(vkCmdSetDepthTestEnableEXT, VulkanCmd);
    HOOK(vkCmdSetDepthWriteEnable, VulkanCmd);
    HOOK(vkCmdSetDepthWriteEnableEXT, VulkanCmd);
    HOOK(vkCmdSetDepthCompareOp, VulkanCmd);
    HOOK(vkCmdSetDepthCompareOpEXT, VulkanCmd);
    HOOK(vkCmdSetDepthBoundsTestEnable, VulkanCmd);
    HOOK(vkCmdSetDepthBoundsTestEnableEXT, VulkanCmd);
    HOOK(vkCmdSetDepthBiasEnable, VulkanCmd);
    HOOK(vkCmdSetDepthBiasEnableEXT, VulkanCmd);
    HOOK(vkCmdSetRasterizerDiscardEnable, VulkanCmd);
    HOOK(vkCmdSetRasterizerDiscardEnableEXT, VulkanCmd);
    HOOK(vkCmdSetPrimitiveRestartEnable, VulkanCmd);
    HOOK(vkCmdSetPrimitiveRestartEnableEXT, VulkanCmd);
    HOOK(vkCmdSetColorWriteMaskEXT, VulkanCmd);
    HOOK(vkCmdSetColorBlendEnableEXT, VulkanCmd);
    HOOK(vkCmdSetColorBlendEquationEXT, VulkanCmd);
    HOOK(vkCmdSetLogicOpEXT, VulkanCmd);
    HOOK(vkCmdSetVertexInputEXT, VulkanCmd);
    HOOK(vkCmdSetAttachmentFeedbackLoopEnableEXT, VulkanCmd);

    // Submission.
    HOOK(vkQueueSubmit, VulkanSubmit);
    HOOK(vkQueueSubmit2, VulkanSubmit);
    HOOK(vkQueueSubmit2KHR, VulkanSubmit);

    // Other driver work done while recording (not waits: those stay GPU wait time).
    HOOK(vkBeginCommandBuffer, VulkanOther);
    HOOK(vkEndCommandBuffer, VulkanOther);
    HOOK(vkAllocateCommandBuffers, VulkanOther);
    HOOK(vkResetCommandPool, VulkanOther);
    HOOK(vkResetCommandBuffer, VulkanOther);
    HOOK(vkGetSemaphoreCounterValue, VulkanOther);
    HOOK(vkGetSemaphoreCounterValueKHR, VulkanOther);
    HOOK(vkGetQueryPoolResults, VulkanOther);
    HOOK(vkUpdateDescriptorSets, VulkanOther);
    HOOK(vkFlushMappedMemoryRanges, VulkanOther);
    HOOK(vkInvalidateMappedMemoryRanges, VulkanOther);
    HOOK(vkCreateImageView, VulkanOther);
    HOOK(vkCreateBuffer, VulkanOther);
    HOOK(vkCreateImage, VulkanOther);
    HOOK(vkCreateSampler, VulkanOther);
}

#undef HOOK

} // namespace Vulkan
