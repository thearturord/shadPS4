// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "video_core/amdgpu/draw_state.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

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
    StencilOps stencil_front_ops{};
    StencilOps stencil_back_ops{};
    std::array<u32, 2> stencil_reference{};    ///< front, back
    std::array<u32, 2> stencil_write_mask{};   ///< front, back (already zero while the stencil is being cleared)
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
};

/// The packet path: the fixed-function state of the draw described by `regs`.
FixedState MakeFixedState(const AmdGpu::DrawRegs& regs);

} // namespace Vulkan
