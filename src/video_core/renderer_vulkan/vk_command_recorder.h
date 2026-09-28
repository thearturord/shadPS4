// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

#ifdef MemoryBarrier
#pragma push_macro("MemoryBarrier")
#undef MemoryBarrier
#define SHADPS4_RESTORE_MEMORY_BARRIER
#endif

namespace Vulkan {

/// Vulkan commands recorded as value-captured closures, replayed later into a command buffer.
/// Everything a command needs (arrays, structures with pointers, strings) is copied into the
/// list's arena, so the caller's data may change or go away right after recording.
class CommandList {
public:
    CommandList() = default;
    CommandList(const CommandList&) = delete;
    CommandList& operator=(const CommandList&) = delete;

    [[nodiscard]] bool Empty() const noexcept {
        return head == nullptr;
    }

    /// Number of commands recorded since the last replay.
    [[nodiscard]] u64 Size() const noexcept {
        return num_commands;
    }

    /// Records a command. `func` is called with the command buffer at replay time.
    template <typename F>
    void Record(F&& func) {
        using Func = std::decay_t<F>;
        static_assert(std::is_trivially_destructible_v<Func>,
                      "recorded commands must only capture values and arena pointers");
        auto* command = new (Allocate(sizeof(CommandImpl<Func>), alignof(CommandImpl<Func>)))
            CommandImpl<Func>{std::forward<F>(func)};
        command->execute = [](Command* self, vk::CommandBuffer cmdbuf) {
            static_cast<CommandImpl<Func>*>(self)->func(cmdbuf);
        };
        Link(command);
    }

    /// Copies an array into the arena. The copy lives until the next replay.
    template <typename T>
    [[nodiscard]] T* Copy(const T* data, size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (count == 0) {
            return nullptr;
        }
        auto* dst = static_cast<T*>(Allocate(sizeof(T) * count, alignof(T)));
        std::memcpy(dst, data, sizeof(T) * count);
        return dst;
    }

    /// Copies a null-terminated string into the arena.
    [[nodiscard]] const char* CopyString(const char* str) {
        if (!str) {
            return nullptr;
        }
        return Copy(str, std::strlen(str) + 1);
    }

    /// Replays every recorded command into `cmdbuf` in order and empties the list.
    void Replay(vk::CommandBuffer cmdbuf) {
        for (Command* command = head; command; command = command->next) {
            command->execute(command, cmdbuf);
        }
        Reset();
    }

private:
    struct Command {
        void (*execute)(Command*, vk::CommandBuffer){};
        Command* next{};
    };

    template <typename Func>
    struct CommandImpl : Command {
        explicit CommandImpl(Func&& func_) : func{std::move(func_)} {}
        explicit CommandImpl(const Func& func_) : func{func_} {}
        Func func;
    };

    static constexpr size_t ChunkSize = 256 * 1024;

    void Link(Command* command) {
        if (tail) {
            tail->next = command;
        } else {
            head = command;
        }
        tail = command;
        ++num_commands;
    }

    void* Allocate(size_t size, size_t align) {
        if (size + align > ChunkSize / 4) {
            // Rare large data (e.g. an image upload with many regions): a block of its own,
            // freed at the next replay.
            auto& block = large_blocks.emplace_back(std::make_unique<u8[]>(size + align));
            const auto base = reinterpret_cast<uintptr_t>(block.get());
            return reinterpret_cast<void*>((base + align - 1) & ~(uintptr_t{align} - 1));
        }
        while (true) {
            if (chunk_index < chunks.size()) {
                const size_t aligned = (offset + align - 1) & ~(align - 1);
                if (aligned + size <= ChunkSize) {
                    offset = aligned + size;
                    return chunks[chunk_index].get() + aligned;
                }
                ++chunk_index;
                offset = 0;
                continue;
            }
            chunks.push_back(std::make_unique<u8[]>(ChunkSize));
        }
    }

    void Reset() {
        head = nullptr;
        tail = nullptr;
        num_commands = 0;
        chunk_index = 0;
        offset = 0;
        large_blocks.clear();
    }

    std::vector<std::unique_ptr<u8[]>> chunks;
    std::vector<std::unique_ptr<u8[]>> large_blocks;
    size_t chunk_index{};
    size_t offset{};
    Command* head{};
    Command* tail{};
    u64 num_commands{};
};

/// What the scheduler hands out instead of a raw command buffer. In direct mode every call goes
/// straight to the command buffer (the old behavior). In recording mode calls are appended to a
/// CommandList with all their data copied, and replayed at submit time.
class CommandRecorder {
public:
    CommandRecorder() = default;
    CommandRecorder(vk::CommandBuffer cmdbuf_) : cmdbuf{cmdbuf_} {}
    CommandRecorder(vk::CommandBuffer cmdbuf_, CommandList* list_) : cmdbuf{cmdbuf_}, list{list_} {}

    explicit operator bool() const noexcept {
        return list != nullptr || static_cast<bool>(cmdbuf);
    }

    [[nodiscard]] bool IsRecording() const noexcept {
        return list != nullptr;
    }

    // ---------------------------------------------------------------- binding and state

    void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
        Do([=](vk::CommandBuffer c) { c.bindPipeline(bind_point, pipeline); });
    }

    void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
        Do([=](vk::CommandBuffer c) { c.bindIndexBuffer(buffer, offset, type); });
    }

    void bindVertexBuffers(u32 first, u32 count, const vk::Buffer* buffers,
                           const vk::DeviceSize* offsets) const {
        if (!list) {
            cmdbuf.bindVertexBuffers(first, count, buffers, offsets);
            return;
        }
        const auto* b = list->Copy(buffers, count);
        const auto* o = list->Copy(offsets, count);
        list->Record([=](vk::CommandBuffer c) { c.bindVertexBuffers(first, count, b, o); });
    }

    void bindVertexBuffers2(u32 first, u32 count, const vk::Buffer* buffers,
                            const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
                            const vk::DeviceSize* strides) const {
        if (!list) {
            cmdbuf.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
            return;
        }
        const auto* b = list->Copy(buffers, count);
        const auto* o = list->Copy(offsets, count);
        const auto* s = sizes ? list->Copy(sizes, count) : nullptr;
        const auto* st = strides ? list->Copy(strides, count) : nullptr;
        list->Record([=](vk::CommandBuffer c) { c.bindVertexBuffers2(first, count, b, o, s, st); });
    }

    void setVertexInputEXT(
        vk::ArrayProxy<const vk::VertexInputBindingDescription2EXT> bindings,
        vk::ArrayProxy<const vk::VertexInputAttributeDescription2EXT> attributes) const {
        if (!list) {
            cmdbuf.setVertexInputEXT(bindings, attributes);
            return;
        }
        const u32 nb = bindings.size();
        const u32 na = attributes.size();
        const auto* b = list->Copy(bindings.data(), nb);
        const auto* a = list->Copy(attributes.data(), na);
        list->Record([=](vk::CommandBuffer c) { c.setVertexInputEXT(nb, b, na, a); });
    }

    void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, u32 offset, u32 size,
                       const void* values) const {
        if (!list) {
            cmdbuf.pushConstants(layout, stages, offset, size, values);
            return;
        }
        const auto* v = list->Copy(static_cast<const u8*>(values), size);
        list->Record(
            [=](vk::CommandBuffer c) { c.pushConstants(layout, stages, offset, size, v); });
    }

    template <typename T>
    void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, u32 offset,
                       vk::ArrayProxy<const T> values) const {
        pushConstants(layout, stages, offset, static_cast<u32>(values.size() * sizeof(T)),
                      values.data());
    }

    void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout, u32 set,
                              vk::ArrayProxy<const vk::WriteDescriptorSet> writes) const {
        if (!list) {
            cmdbuf.pushDescriptorSetKHR(bind_point, layout, set, writes);
            return;
        }
        const u32 count = writes.size();
        auto* w = list->Copy(writes.data(), count);
        for (u32 i = 0; i < count; ++i) {
            ASSERT_MSG(!w[i].pNext, "Recorded descriptor writes can't have a pNext chain");
            w[i].pImageInfo =
                list->Copy(w[i].pImageInfo, w[i].pImageInfo ? w[i].descriptorCount : 0);
            w[i].pBufferInfo =
                list->Copy(w[i].pBufferInfo, w[i].pBufferInfo ? w[i].descriptorCount : 0);
            w[i].pTexelBufferView =
                list->Copy(w[i].pTexelBufferView, w[i].pTexelBufferView ? w[i].descriptorCount : 0);
        }
        list->Record([=](vk::CommandBuffer c) {
            c.pushDescriptorSetKHR(bind_point, layout, set, count, w);
        });
    }

    void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                            u32 first_set, vk::ArrayProxy<const vk::DescriptorSet> sets,
                            vk::ArrayProxy<const u32> dynamic_offsets) const {
        if (!list) {
            cmdbuf.bindDescriptorSets(bind_point, layout, first_set, sets, dynamic_offsets);
            return;
        }
        const u32 ns = sets.size();
        const u32 no = dynamic_offsets.size();
        const auto* s = list->Copy(sets.data(), ns);
        const auto* o = list->Copy(dynamic_offsets.data(), no);
        list->Record([=](vk::CommandBuffer c) {
            c.bindDescriptorSets(bind_point, layout, first_set, ns, s, no, o);
        });
    }

    // ---------------------------------------------------------------- dynamic state

    void setViewport(u32 first, vk::ArrayProxy<const vk::Viewport> viewports) const {
        if (!list) {
            cmdbuf.setViewport(first, viewports);
            return;
        }
        const u32 n = viewports.size();
        const auto* v = list->Copy(viewports.data(), n);
        list->Record([=](vk::CommandBuffer c) { c.setViewport(first, n, v); });
    }

    void setScissor(u32 first, vk::ArrayProxy<const vk::Rect2D> scissors) const {
        if (!list) {
            cmdbuf.setScissor(first, scissors);
            return;
        }
        const u32 n = scissors.size();
        const auto* s = list->Copy(scissors.data(), n);
        list->Record([=](vk::CommandBuffer c) { c.setScissor(first, n, s); });
    }

    void setViewportWithCount(vk::ArrayProxy<const vk::Viewport> viewports) const {
        if (!list) {
            cmdbuf.setViewportWithCount(viewports);
            return;
        }
        const u32 n = viewports.size();
        const auto* v = list->Copy(viewports.data(), n);
        list->Record([=](vk::CommandBuffer c) { c.setViewportWithCount(n, v); });
    }

    void setScissorWithCount(vk::ArrayProxy<const vk::Rect2D> scissors) const {
        if (!list) {
            cmdbuf.setScissorWithCount(scissors);
            return;
        }
        const u32 n = scissors.size();
        const auto* s = list->Copy(scissors.data(), n);
        list->Record([=](vk::CommandBuffer c) { c.setScissorWithCount(n, s); });
    }

    void setDepthTestEnable(bool enable) const {
        Do([=](vk::CommandBuffer c) { c.setDepthTestEnable(enable); });
    }
    void setDepthWriteEnable(bool enable) const {
        Do([=](vk::CommandBuffer c) { c.setDepthWriteEnable(enable); });
    }
    void setDepthCompareOp(vk::CompareOp op) const {
        Do([=](vk::CommandBuffer c) { c.setDepthCompareOp(op); });
    }
    void setDepthBoundsTestEnable(bool enable) const {
        Do([=](vk::CommandBuffer c) { c.setDepthBoundsTestEnable(enable); });
    }
    void setDepthBounds(float min, float max) const {
        Do([=](vk::CommandBuffer c) { c.setDepthBounds(min, max); });
    }
    void setDepthBiasEnable(bool enable) const {
        Do([=](vk::CommandBuffer c) { c.setDepthBiasEnable(enable); });
    }
    void setDepthBias(float constant, float clamp, float slope) const {
        Do([=](vk::CommandBuffer c) { c.setDepthBias(constant, clamp, slope); });
    }
    void setStencilTestEnable(bool enable) const {
        Do([=](vk::CommandBuffer c) { c.setStencilTestEnable(enable); });
    }
    void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
                      vk::StencilOp depth_fail, vk::CompareOp compare) const {
        Do([=](vk::CommandBuffer c) { c.setStencilOp(faces, fail, pass, depth_fail, compare); });
    }
    void setStencilReference(vk::StencilFaceFlags faces, u32 reference) const {
        Do([=](vk::CommandBuffer c) { c.setStencilReference(faces, reference); });
    }
    void setStencilWriteMask(vk::StencilFaceFlags faces, u32 mask) const {
        Do([=](vk::CommandBuffer c) { c.setStencilWriteMask(faces, mask); });
    }
    void setStencilCompareMask(vk::StencilFaceFlags faces, u32 mask) const {
        Do([=](vk::CommandBuffer c) { c.setStencilCompareMask(faces, mask); });
    }
    void setPrimitiveRestartEnable(bool enable) const {
        Do([=](vk::CommandBuffer c) { c.setPrimitiveRestartEnable(enable); });
    }
    void setRasterizerDiscardEnable(bool enable) const {
        Do([=](vk::CommandBuffer c) { c.setRasterizerDiscardEnable(enable); });
    }
    void setCullMode(vk::CullModeFlags mode) const {
        Do([=](vk::CommandBuffer c) { c.setCullMode(mode); });
    }
    void setFrontFace(vk::FrontFace face) const {
        Do([=](vk::CommandBuffer c) { c.setFrontFace(face); });
    }
    void setLineWidth(float width) const {
        Do([=](vk::CommandBuffer c) { c.setLineWidth(width); });
    }
    void setBlendConstants(const float constants[4]) const {
        const std::array<float, 4> values{constants[0], constants[1], constants[2], constants[3]};
        Do([=](vk::CommandBuffer c) { c.setBlendConstants(values.data()); });
    }
    void setColorWriteMaskEXT(u32 first,
                              vk::ArrayProxy<const vk::ColorComponentFlags> masks) const {
        if (!list) {
            cmdbuf.setColorWriteMaskEXT(first, masks);
            return;
        }
        const u32 n = masks.size();
        const auto* m = list->Copy(masks.data(), n);
        list->Record([=](vk::CommandBuffer c) { c.setColorWriteMaskEXT(first, n, m); });
    }
    void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
        Do([=](vk::CommandBuffer c) { c.setAttachmentFeedbackLoopEnableEXT(aspects); });
    }

    // ---------------------------------------------------------------- rendering and work

    void beginRendering(const vk::RenderingInfo& info) const {
        if (!list) {
            cmdbuf.beginRendering(info);
            return;
        }
        ASSERT_MSG(!info.pNext, "Recorded rendering info can't have a pNext chain");
        auto* copy = list->Copy(&info, 1);
        copy->pColorAttachments = list->Copy(info.pColorAttachments, info.colorAttachmentCount);
        copy->pDepthAttachment =
            info.pDepthAttachment ? list->Copy(info.pDepthAttachment, 1) : nullptr;
        copy->pStencilAttachment =
            info.pStencilAttachment ? list->Copy(info.pStencilAttachment, 1) : nullptr;
        list->Record([=](vk::CommandBuffer c) { c.beginRendering(*copy); });
    }

    void endRendering() const {
        Do([](vk::CommandBuffer c) { c.endRendering(); });
    }

    void draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance) const {
        Do([=](vk::CommandBuffer c) { c.draw(vertices, instances, first_vertex, first_instance); });
    }

    void drawIndexed(u32 indices, u32 instances, u32 first_index, s32 vertex_offset,
                     u32 first_instance) const {
        Do([=](vk::CommandBuffer c) {
            c.drawIndexed(indices, instances, first_index, vertex_offset, first_instance);
        });
    }

    void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, u32 count, u32 stride) const {
        Do([=](vk::CommandBuffer c) { c.drawIndirect(buffer, offset, count, stride); });
    }

    void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, u32 count,
                             u32 stride) const {
        Do([=](vk::CommandBuffer c) { c.drawIndexedIndirect(buffer, offset, count, stride); });
    }

    void drawIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
                           vk::DeviceSize count_offset, u32 max_count, u32 stride) const {
        Do([=](vk::CommandBuffer c) {
            c.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_count, stride);
        });
    }

    void drawIndexedIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
                                  vk::DeviceSize count_offset, u32 max_count, u32 stride) const {
        Do([=](vk::CommandBuffer c) {
            c.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset, max_count,
                                       stride);
        });
    }

    void dispatch(u32 x, u32 y, u32 z) const {
        Do([=](vk::CommandBuffer c) { c.dispatch(x, y, z); });
    }

    void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
        Do([=](vk::CommandBuffer c) { c.dispatchIndirect(buffer, offset); });
    }

    // ---------------------------------------------------------------- barriers

    void pipelineBarrier2(const vk::DependencyInfo& info) const {
        if (!list) {
            cmdbuf.pipelineBarrier2(info);
            return;
        }
        ASSERT_MSG(!info.pNext, "Recorded dependency info can't have a pNext chain");
        auto* copy = list->Copy(&info, 1);
        copy->pMemoryBarriers = list->Copy(info.pMemoryBarriers, info.memoryBarrierCount);
        copy->pBufferMemoryBarriers =
            list->Copy(info.pBufferMemoryBarriers, info.bufferMemoryBarrierCount);
        copy->pImageMemoryBarriers =
            list->Copy(info.pImageMemoryBarriers, info.imageMemoryBarrierCount);
        list->Record([=](vk::CommandBuffer c) { c.pipelineBarrier2(*copy); });
    }

    void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
                         vk::DependencyFlags flags,
                         vk::ArrayProxy<const vk::MemoryBarrier> memory_barriers,
                         vk::ArrayProxy<const vk::BufferMemoryBarrier> buffer_barriers,
                         vk::ArrayProxy<const vk::ImageMemoryBarrier> image_barriers) const {
        if (!list) {
            cmdbuf.pipelineBarrier(src, dst, flags, memory_barriers, buffer_barriers,
                                   image_barriers);
            return;
        }
        const u32 nm = memory_barriers.size();
        const u32 nb = buffer_barriers.size();
        const u32 ni = image_barriers.size();
        const auto* m = list->Copy(memory_barriers.data(), nm);
        const auto* b = list->Copy(buffer_barriers.data(), nb);
        const auto* i = list->Copy(image_barriers.data(), ni);
        list->Record(
            [=](vk::CommandBuffer c) { c.pipelineBarrier(src, dst, flags, nm, m, nb, b, ni, i); });
    }

    // ---------------------------------------------------------------- transfers

    void copyBuffer(vk::Buffer src, vk::Buffer dst,
                    vk::ArrayProxy<const vk::BufferCopy> regions) const {
        if (!list) {
            cmdbuf.copyBuffer(src, dst, regions);
            return;
        }
        const u32 n = regions.size();
        const auto* r = list->Copy(regions.data(), n);
        list->Record([=](vk::CommandBuffer c) { c.copyBuffer(src, dst, n, r); });
    }

    void copyImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
                   vk::ImageLayout dst_layout, vk::ArrayProxy<const vk::ImageCopy> regions) const {
        if (!list) {
            cmdbuf.copyImage(src, src_layout, dst, dst_layout, regions);
            return;
        }
        const u32 n = regions.size();
        const auto* r = list->Copy(regions.data(), n);
        list->Record(
            [=](vk::CommandBuffer c) { c.copyImage(src, src_layout, dst, dst_layout, n, r); });
    }

    void copyBufferToImage(vk::Buffer src, vk::Image dst, vk::ImageLayout dst_layout,
                           vk::ArrayProxy<const vk::BufferImageCopy> regions) const {
        if (!list) {
            cmdbuf.copyBufferToImage(src, dst, dst_layout, regions);
            return;
        }
        const u32 n = regions.size();
        const auto* r = list->Copy(regions.data(), n);
        list->Record([=](vk::CommandBuffer c) { c.copyBufferToImage(src, dst, dst_layout, n, r); });
    }

    void copyImageToBuffer(vk::Image src, vk::ImageLayout src_layout, vk::Buffer dst,
                           vk::ArrayProxy<const vk::BufferImageCopy> regions) const {
        if (!list) {
            cmdbuf.copyImageToBuffer(src, src_layout, dst, regions);
            return;
        }
        const u32 n = regions.size();
        const auto* r = list->Copy(regions.data(), n);
        list->Record([=](vk::CommandBuffer c) { c.copyImageToBuffer(src, src_layout, dst, n, r); });
    }

    void resolveImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
                      vk::ImageLayout dst_layout,
                      vk::ArrayProxy<const vk::ImageResolve> regions) const {
        if (!list) {
            cmdbuf.resolveImage(src, src_layout, dst, dst_layout, regions);
            return;
        }
        const u32 n = regions.size();
        const auto* r = list->Copy(regions.data(), n);
        list->Record(
            [=](vk::CommandBuffer c) { c.resolveImage(src, src_layout, dst, dst_layout, n, r); });
    }

    void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue& color,
                         vk::ArrayProxy<const vk::ImageSubresourceRange> ranges) const {
        if (!list) {
            cmdbuf.clearColorImage(image, layout, color, ranges);
            return;
        }
        const u32 n = ranges.size();
        const auto* r = list->Copy(ranges.data(), n);
        const vk::ClearColorValue value = color;
        list->Record([=](vk::CommandBuffer c) { c.clearColorImage(image, layout, &value, n, r); });
    }

    void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, u32 data) const {
        Do([=](vk::CommandBuffer c) { c.fillBuffer(buffer, offset, size, data); });
    }

    // ---------------------------------------------------------------- queries and markers

    void resetQueryPool(vk::QueryPool pool, u32 first, u32 count) const {
        Do([=](vk::CommandBuffer c) { c.resetQueryPool(pool, first, count); });
    }

    void writeTimestamp(vk::PipelineStageFlagBits stage, vk::QueryPool pool, u32 query) const {
        Do([=](vk::CommandBuffer c) { c.writeTimestamp(stage, pool, query); });
    }

    void beginDebugUtilsLabelEXT(const vk::DebugUtilsLabelEXT& label) const {
        if (!list) {
            cmdbuf.beginDebugUtilsLabelEXT(label);
            return;
        }
        auto copy = label;
        copy.pLabelName = list->CopyString(label.pLabelName);
        list->Record([=](vk::CommandBuffer c) { c.beginDebugUtilsLabelEXT(copy); });
    }

    void insertDebugUtilsLabelEXT(const vk::DebugUtilsLabelEXT& label) const {
        if (!list) {
            cmdbuf.insertDebugUtilsLabelEXT(label);
            return;
        }
        auto copy = label;
        copy.pLabelName = list->CopyString(label.pLabelName);
        list->Record([=](vk::CommandBuffer c) { c.insertDebugUtilsLabelEXT(copy); });
    }

    void endDebugUtilsLabelEXT() const {
        Do([](vk::CommandBuffer c) { c.endDebugUtilsLabelEXT(); });
    }

private:
    template <typename F>
    void Do(F&& func) const {
        if (list) {
            list->Record(std::forward<F>(func));
        } else {
            func(cmdbuf);
        }
    }

    vk::CommandBuffer cmdbuf{};
    CommandList* list{};
};

} // namespace Vulkan

#ifdef SHADPS4_RESTORE_MEMORY_BARRIER
#pragma pop_macro("MemoryBarrier")
#undef SHADPS4_RESTORE_MEMORY_BARRIER
#endif
