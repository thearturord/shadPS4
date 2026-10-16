// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string_view>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"

namespace Vulkan {

namespace {

void FillViewports(FixedState& out, const AmdGpu::DrawRegs& regs) {
    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x), enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y), enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x, enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y, enable_offset ? regs.window_offset.window_y_offset : 0);

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    out.clip_disabled = regs.IsClipDisabled();
    out.viewport_count = 0;
    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // When the clip space is ranged [-1...1], the zoffset is centered.
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
        const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
        const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
        const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

        viewport.x = xoffset - xscale;
        viewport.y = yoffset - yscale;
        viewport.width = xscale * 2.0f;
        viewport.height = yscale * 2.0f;

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        out.viewports[out.viewport_count] = viewport;
        out.scissors[out.viewport_count] = vk::Rect2D{
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        };
        ++out.viewport_count;
    }

    if (out.viewport_count == 0) {
        // Vulkan requires providing at least one viewport.
        out.clip_disabled = false;
        out.viewport_count = 1;
        out.viewports[0] = vk::Viewport{
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        out.scissors[0] = vk::Rect2D{
            .offset = {0, 0},
            .extent = {1, 1},
        };
    }
}

void FillDepthStencil(FixedState& out, const AmdGpu::DrawRegs& regs) {
    out.depth_target = regs.depth_buffer.DepthValid();
    out.depth_test = regs.depth_control.depth_enable;
    out.depth_write = regs.depth_control.depth_write_enable;
    out.depth_clear = regs.depth_render_control.depth_clear_enable;
    out.depth_compare = LiverpoolToVK::CompareOp(regs.depth_control.depth_func);
    out.depth_bounds_test = regs.depth_control.depth_bounds_enable;
    out.depth_bounds_min = regs.depth_bounds_min;
    out.depth_bounds_max = regs.depth_bounds_max;

    out.depth_bias = regs.polygon_control.NeedsBias();
    const bool front_offset = regs.polygon_control.enable_polygon_offset_front;
    out.depth_bias_constant =
        front_offset ? regs.poly_offset.front_offset : regs.poly_offset.back_offset;
    out.depth_bias_clamp = regs.poly_offset.depth_bias;
    out.depth_bias_slope =
        (front_offset ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f;

    out.stencil_target = regs.depth_buffer.StencilValid();
    out.stencil_test = regs.depth_control.stencil_enable;
    out.stencil_front_ops = StencilOps{
        .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
        .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
        .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
        .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
    };
    out.stencil_back_ops = regs.depth_control.backface_enable
                               ? StencilOps{
                                     .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
                                     .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
                                     .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
                                     .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
                                 }
                               : out.stencil_front_ops;

    out.stencil_clear = regs.depth_render_control.stencil_clear_enable;
    const auto front = regs.stencil_ref_front;
    const auto back =
        regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
    // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
    // include ReplaceOp takes its Vulkan reference from op_val.
    const auto& sc = regs.stencil_control;
    const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                AmdGpu::StencilFunc zfail) {
        return fail == AmdGpu::StencilFunc::ReplaceOp || zpass == AmdGpu::StencilFunc::ReplaceOp ||
               zfail == AmdGpu::StencilFunc::ReplaceOp;
    };
    const bool front_op =
        uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
    const bool back_op =
        regs.depth_control.backface_enable
            ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
            : front_op;
    const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
        return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
               ref.stencil_test_val != ref.stencil_op_val;
    };
    if (out.stencil_test && out.stencil_target &&
        ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
         (back_op && regs.depth_control.backface_enable &&
          ref_conflict(regs.depth_control.stencil_bf_func, back)))) {
        LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                   "op_val; the stencil test will use op_val");
    }
    out.stencil_reference = {front_op ? front.stencil_op_val : front.stencil_test_val,
                             back_op ? back.stencil_op_val : back.stencil_test_val};
    out.stencil_write_mask = {front.stencil_write_mask, back.stencil_write_mask};
    out.stencil_compare_mask = {front.stencil_mask, back.stencil_mask};
}

void FillPrimitive(FixedState& out, const AmdGpu::DrawRegs& regs) {
    SetFixedStateTopology(out, regs.primitive_type,
                          LiverpoolToVK::CullMode(regs.polygon_control.CullingMode()));
    out.primitive_restart = (regs.enable_primitive_restart & 1) != 0;
    out.primitive_restart_index = regs.primitive_restart_index;
    out.front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);
    out.rasterizer_discard = regs.clipper_control.dx_rasterization_kill;
    out.line_width = regs.line_control.Width();
    out.blend_constants = regs.blend_constants;
}

BlendTarget MakeBlendTarget(const AmdGpu::BlendControl& control) {
    BlendTarget out;
    out.src_color = LiverpoolToVK::BlendFactor(control.color_src_factor);
    out.dst_color = LiverpoolToVK::BlendFactor(control.color_dst_factor);
    out.color_op = LiverpoolToVK::BlendOp(control.color_func);
    out.src_alpha = control.separate_alpha_blend
                        ? LiverpoolToVK::BlendFactor(control.alpha_src_factor)
                        : out.src_color;
    out.dst_alpha = control.separate_alpha_blend
                        ? LiverpoolToVK::BlendFactor(control.alpha_dst_factor)
                        : out.dst_color;
    out.alpha_op =
        control.separate_alpha_blend ? LiverpoolToVK::BlendOp(control.alpha_func) : out.color_op;
    out.enable = control.enable;
    return out;
}

void FillPipelineState(FixedState& out, const AmdGpu::DrawRegs& regs) {
    out.color_enabled = regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Disable;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        out.target_write_mask[cb] = static_cast<u8>(regs.color_target_mask.GetMask(cb));
        out.blend[cb] = MakeBlendTarget(regs.blend_control[cb]);
        if ((regs.color_shader_mask.GetMask(cb) & AmdGpu::ColorBufferMask::ComponentA) == 0) {
            out.alpha_not_exported |= 1u << cb;
        }
    }
    out.logic_op_enabled = regs.color_control.rop3 != AmdGpu::ColorControl::LogicOp::Copy;
    out.logic_op = LiverpoolToVK::LogicOp(regs.color_control.rop3);
    out.polygon_mode = LiverpoolToVK::PolygonMode(regs.polygon_control.PolyMode());
    out.depth_clamp = !regs.depth_render_override.disable_viewport_clamp;
    out.depth_clip = regs.clipper_control.ZclipEnable();
    out.negative_one_to_one = regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
    out.provoking_first = regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::First;
}

} // namespace

void SetFixedStateTopology(FixedState& out, AmdGpu::PrimitiveType type, vk::CullModeFlags cull_mode) {
    const auto topology = type != AmdGpu::PrimitiveType::None
                              ? LiverpoolToVK::PrimitiveType(type)
                              : vk::PrimitiveTopology::ePatchList;
    out.list_topology = topology == vk::PrimitiveTopology::ePointList ||
                        topology == vk::PrimitiveTopology::eLineList ||
                        topology == vk::PrimitiveTopology::eTriangleList ||
                        topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
                        topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    // Quad and rect lists are emulated using tessellation.
    out.patch_list_topology = type == AmdGpu::PrimitiveType::PatchPrimitive ||
                              type == AmdGpu::PrimitiveType::QuadList ||
                              type == AmdGpu::PrimitiveType::RectList;
    out.cull_mode = LiverpoolToVK::IsPrimitiveCulled(type) ? cull_mode : vk::CullModeFlagBits::eNone;
}

void TakeFromRegisters(FixedState& out, const AmdGpu::DrawRegs& regs, u32 groups) {
    if ((groups & StaleSurface) != 0) {
        SetFixedStateTopology(out, regs.primitive_type,
                              LiverpoolToVK::CullMode(regs.polygon_control.CullingMode()));
        out.front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);
        out.polygon_mode = LiverpoolToVK::PolygonMode(regs.polygon_control.PolyMode());
        out.provoking_first =
            regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::First;
    }
    if ((groups & StaleClear) != 0) {
        out.depth_clear = regs.depth_render_control.depth_clear_enable;
        out.stencil_clear = regs.depth_render_control.stencil_clear_enable;
    }
    if ((groups & StaleClip) != 0) {
        out.depth_clip = regs.clipper_control.ZclipEnable();
        out.negative_one_to_one = regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        out.rasterizer_discard = regs.clipper_control.dx_rasterization_kill;
        out.clip_disabled = regs.IsClipDisabled();
    }
}

const FixedState& DefaultFixedState() {
    static const FixedState state = [] {
        AmdGpu::DrawRegs regs{};
        regs.SetDefaults();
        regs.primitive_type = AmdGpu::PrimitiveType::TriangleList;
        return MakeFixedState(regs);
    }();
    return state;
}

std::string DiffFixedState(const FixedState& a, const FixedState& b) {
    std::string out;
    const auto note = [&](bool differs, std::string_view name) {
        if (differs) {
            out += name;
            out += ' ';
        }
    };
    const auto same_rects = [&] {
        for (u32 i = 0; i < a.viewport_count && i < b.viewport_count; ++i) {
            if (a.scissors[i] != b.scissors[i]) {
                return false;
            }
        }
        return true;
    };
    const auto same_viewports = [&] {
        for (u32 i = 0; i < a.viewport_count && i < b.viewport_count; ++i) {
            if (a.viewports[i] != b.viewports[i]) {
                return false;
            }
        }
        return true;
    };
    note(a.viewport_count != b.viewport_count, "viewport_count");
    note(!same_viewports(), "viewports");
    note(!same_rects(), "scissors");
    note(a.clip_disabled != b.clip_disabled, "clip_disabled");
    note(a.depth_target != b.depth_target, "depth_target");
    note(a.depth_test != b.depth_test, "depth_test");
    note(a.depth_write != b.depth_write, "depth_write");
    note(a.depth_clear != b.depth_clear, "depth_clear");
    note(a.depth_compare != b.depth_compare, "depth_compare");
    note(a.depth_bounds_test != b.depth_bounds_test, "depth_bounds_test");
    note(a.depth_bounds_test && (a.depth_bounds_min != b.depth_bounds_min ||
                                 a.depth_bounds_max != b.depth_bounds_max),
         "depth_bounds");
    note(a.depth_bias != b.depth_bias, "depth_bias");
    note(a.depth_bias && (a.depth_bias_constant != b.depth_bias_constant ||
                          a.depth_bias_clamp != b.depth_bias_clamp ||
                          a.depth_bias_slope != b.depth_bias_slope),
         "depth_bias_values");
    note(a.stencil_target != b.stencil_target, "stencil_target");
    note(a.stencil_test != b.stencil_test, "stencil_test");
    note(a.stencil_clear != b.stencil_clear, "stencil_clear");
    note(a.stencil_front_ops != b.stencil_front_ops, "stencil_front_ops");
    note(a.stencil_back_ops != b.stencil_back_ops, "stencil_back_ops");
    note(a.stencil_reference != b.stencil_reference, "stencil_reference");
    note(a.stencil_write_mask != b.stencil_write_mask, "stencil_write_mask");
    note(a.stencil_compare_mask != b.stencil_compare_mask, "stencil_compare_mask");
    note(a.primitive_restart != b.primitive_restart, "primitive_restart");
    note(a.primitive_restart_index != b.primitive_restart_index, "primitive_restart_index");
    note(a.list_topology != b.list_topology, "list_topology");
    note(a.patch_list_topology != b.patch_list_topology, "patch_list_topology");
    note(a.rasterizer_discard != b.rasterizer_discard, "rasterizer_discard");
    note(a.cull_mode != b.cull_mode,
         "cull_mode(" + vk::to_string(a.cull_mode) + " vs " + vk::to_string(b.cull_mode) + ")");
    note(a.front_face != b.front_face,
         "front_face(" + vk::to_string(a.front_face) + " vs " + vk::to_string(b.front_face) + ")");
    note(a.list_topology != b.list_topology || a.patch_list_topology != b.patch_list_topology,
         "topology_class");
    note(a.line_width != b.line_width, "line_width");
    note(a.blend_constants != b.blend_constants, "blend_constants");
    note(a.color_enabled != b.color_enabled, "color_enabled");
    note(a.target_write_mask != b.target_write_mask, "target_write_mask");
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
        note(!(a.blend[i] == b.blend[i]), "blend" + std::to_string(i));
    }
    note(a.alpha_not_exported != b.alpha_not_exported, "alpha_not_exported");
    note(a.logic_op_enabled != b.logic_op_enabled, "logic_op_enabled");
    note(a.logic_op_enabled && a.logic_op != b.logic_op, "logic_op");
    note(a.polygon_mode != b.polygon_mode, "polygon_mode");
    note(a.depth_clamp != b.depth_clamp, "depth_clamp");
    note(a.depth_clip != b.depth_clip, "depth_clip");
    note(a.negative_one_to_one != b.negative_one_to_one, "negative_one_to_one");
    note(a.provoking_first != b.provoking_first, "provoking_first");
    return out;
}

FixedState MakeFixedState(const AmdGpu::DrawRegs& regs) {
    FixedState out;
    FillViewports(out, regs);
    FillDepthStencil(out, regs);
    FillPrimitive(out, regs);
    FillPipelineState(out, regs);
    return out;
}

} // namespace Vulkan
