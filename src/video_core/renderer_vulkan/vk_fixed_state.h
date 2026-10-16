// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>
#include "video_core/amdgpu/draw_state.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

/// Blend state of one colour target, in Vulkan terms (no padding: compared and hashed bytewise as part of a pipeline key).
struct BlendTarget {
    vk::BlendFactor src_color{};
    vk::BlendFactor dst_color{};
    vk::BlendFactor src_alpha{};
    vk::BlendFactor dst_alpha{};
    vk::BlendOp color_op{};
    vk::BlendOp alpha_op{};
    u32 enable{};

    bool operator==(const BlendTarget&) const = default;
};

/// The fixed-function state of a draw as Vulkan wants it: viewports, scissors, depth, stencil, culling, blend constants.
/// Nothing in it is an AMD register field. It is built either from the register state of a draw
/// (`MakeFixedState`, the packet path) or by the native path straight from the game's state objects; the rasterizer reads
/// only this. Values that depend on the device (depth range limits, primitive restart support) are applied where the
/// state is committed.
struct FixedState {
    u32 viewport_count{};
    bool clip_disabled{}; ///< the viewport covers the whole render space (the shader works in screen coordinates)
    std::array<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports{};
    std::array<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors{};

    bool depth_target{}; ///< a depth target is bound
    bool depth_test{};
    bool depth_write{};
    bool depth_clear{}; ///< the draw clears the depth target (the render pass state; no depth write then)
    vk::CompareOp depth_compare{};
    bool depth_bounds_test{};
    float depth_bounds_min{};
    float depth_bounds_max{};
    bool depth_bias{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};

    bool stencil_target{}; ///< a stencil target is bound
    bool stencil_test{};
    bool stencil_clear{}; ///< the draw clears the stencil target (the write masks are zero then)
    StencilOps stencil_front_ops{};
    StencilOps stencil_back_ops{};
    std::array<u32, 2> stencil_reference{};    ///< front, back
    std::array<u32, 2> stencil_write_mask{};   ///< front, back
    std::array<u32, 2> stencil_compare_mask{}; ///< front, back

    bool primitive_restart{};
    u32 primitive_restart_index{};
    bool list_topology{};       ///< point, line or triangle list: restart needs the list restart feature
    bool patch_list_topology{}; ///< patch, quad or rect list (tessellation): restart needs the patch list feature
    bool rasterizer_discard{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};
    float line_width{};
    std::array<float, 4> blend_constants{};

    // what a pipeline is built with
    bool color_enabled{};                                       ///< false: the draw writes no colour target
    std::array<u8, AmdGpu::NUM_COLOR_BUFFERS> target_write_mask{}; ///< R, G, B, A bits per colour target
    std::array<BlendTarget, AmdGpu::NUM_COLOR_BUFFERS> blend{};
    u8 alpha_not_exported{}; ///< bit per colour target: the pixel shader does not write alpha
    bool logic_op_enabled{};
    vk::LogicOp logic_op{};
    vk::PolygonMode polygon_mode{};
    bool depth_clamp{};
    bool depth_clip{};
    bool negative_one_to_one{}; ///< clip space -w..w (otherwise 0..w)
    bool provoking_first{};

    /// True when the first target's blend state reads the second pixel shader output.
    bool DualSourceBlending() const {
        const auto& b = blend[0];
        const auto second = [](vk::BlendFactor f) {
            return f == vk::BlendFactor::eSrc1Color || f == vk::BlendFactor::eOneMinusSrc1Color ||
                   f == vk::BlendFactor::eSrc1Alpha || f == vk::BlendFactor::eOneMinusSrc1Alpha;
        };
        return b.enable && (second(b.src_color) || second(b.dst_color) || second(b.src_alpha) ||
                            second(b.dst_alpha));
    }
};

/// The packet path: the fixed-function state of the draw described by `regs`.
FixedState MakeFixedState(const AmdGpu::DrawRegs& regs);

/// Hardware state the engine only rewrites when its own cache says it changed: a draw that does not write the group runs
/// with what the hardware holds, which is not necessarily what the engine's objects say (the register file models that).
enum FixedStateStale : u32 {
    StaleSurface = 1, ///< PA_SU_SC_MODE_CNTL: culling, front face, polygon mode
    StaleClip = 2,    ///< PA_CL_CLIP_CNTL: depth clip, clip space, rasterizer discard
    StaleClear = 4,   ///< DB_RENDER_CONTROL: the depth / stencil clear flags of the render pass
};

/// Replaces the fields of the given groups with what the register state `regs` holds.
void TakeFromRegisters(FixedState& out, const AmdGpu::DrawRegs& regs, u32 groups);

/// The state of a draw whose registers hold their defaults (the base the native path starts from).
const FixedState& DefaultFixedState();

/// Sets the topology dependent fields: list / patch classification and culling (only primitives that have faces are
/// culled; `cull_mode` is what the rasterizer state asks for).
void SetFixedStateTopology(FixedState& out, AmdGpu::PrimitiveType type, vk::CullModeFlags cull_mode);

/// Names of the fields in which two states differ, separated by spaces (empty when they are equal).
std::string DiffFixedState(const FixedState& a, const FixedState& b);

} // namespace Vulkan
