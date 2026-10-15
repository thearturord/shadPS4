// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Generated from regs.h by documents/tlg (gen_draw_state.py).

#include <cstddef>
#include <memory>
#include <vector>
#include "video_core/amdgpu/draw_state.h"

namespace AmdGpu {

namespace {

struct Range {
    u32 reg;   ///< first register index
    u32 words; ///< dwords
    u32 state; ///< first dword in DrawRegs
};

#define REG_INDEX(member) (offsetof(Regs, member) / sizeof(u32))
#define STATE_INDEX(member) (offsetof(DrawRegs, member) / sizeof(u32))
#define WORDS(member) (sizeof(DrawRegs::member) / sizeof(u32))

// members that are bit fields of one dword: register index from the layout of regs.h
static_assert(REG_INDEX(ps_input_addr) + 2 == 0xA1B6);       // num_interp
static_assert(REG_INDEX(vgt_gsvs_ring_itemsize) + 34 == 0xA2CE); // vgt_gs_max_vert_out
constexpr u32 NumInterpReg = 0xA1B6;
constexpr u32 GsMaxVertOutReg = 0xA2CE;

const std::vector<Range>& Ranges() {
    static const std::vector<Range> ranges = {
        {REG_INDEX(ps_program), WORDS(ps_program), STATE_INDEX(ps_program)},
        {REG_INDEX(vs_program), WORDS(vs_program), STATE_INDEX(vs_program)},
        {REG_INDEX(gs_program), WORDS(gs_program), STATE_INDEX(gs_program)},
        {REG_INDEX(es_program), WORDS(es_program), STATE_INDEX(es_program)},
        {REG_INDEX(hs_program), WORDS(hs_program), STATE_INDEX(hs_program)},
        {REG_INDEX(ls_program), WORDS(ls_program), STATE_INDEX(ls_program)},
        {REG_INDEX(depth_render_control), WORDS(depth_render_control), STATE_INDEX(depth_render_control)},
        {REG_INDEX(depth_view), WORDS(depth_view), STATE_INDEX(depth_view)},
        {REG_INDEX(depth_render_override), WORDS(depth_render_override), STATE_INDEX(depth_render_override)},
        {REG_INDEX(depth_htile_data_base), WORDS(depth_htile_data_base), STATE_INDEX(depth_htile_data_base)},
        {REG_INDEX(depth_bounds_min), WORDS(depth_bounds_min), STATE_INDEX(depth_bounds_min)},
        {REG_INDEX(depth_bounds_max), WORDS(depth_bounds_max), STATE_INDEX(depth_bounds_max)},
        {REG_INDEX(stencil_clear), WORDS(stencil_clear), STATE_INDEX(stencil_clear)},
        {REG_INDEX(depth_clear), WORDS(depth_clear), STATE_INDEX(depth_clear)},
        {REG_INDEX(screen_scissor), WORDS(screen_scissor), STATE_INDEX(screen_scissor)},
        {REG_INDEX(depth_buffer), WORDS(depth_buffer), STATE_INDEX(depth_buffer)},
        {REG_INDEX(ta_bc_base), WORDS(ta_bc_base), STATE_INDEX(ta_bc_base)},
        {REG_INDEX(window_offset), WORDS(window_offset), STATE_INDEX(window_offset)},
        {REG_INDEX(window_scissor), WORDS(window_scissor), STATE_INDEX(window_scissor)},
        {REG_INDEX(color_target_mask), WORDS(color_target_mask), STATE_INDEX(color_target_mask)},
        {REG_INDEX(color_shader_mask), WORDS(color_shader_mask), STATE_INDEX(color_shader_mask)},
        {REG_INDEX(generic_scissor), WORDS(generic_scissor), STATE_INDEX(generic_scissor)},
        {REG_INDEX(viewport_scissors), WORDS(viewport_scissors), STATE_INDEX(viewport_scissors)},
        {REG_INDEX(viewport_depths), WORDS(viewport_depths), STATE_INDEX(viewport_depths)},
        {REG_INDEX(index_offset), WORDS(index_offset), STATE_INDEX(index_offset)},
        {REG_INDEX(primitive_restart_index), WORDS(primitive_restart_index), STATE_INDEX(primitive_restart_index)},
        {REG_INDEX(blend_constants), WORDS(blend_constants), STATE_INDEX(blend_constants)},
        {REG_INDEX(stencil_control), WORDS(stencil_control), STATE_INDEX(stencil_control)},
        {REG_INDEX(stencil_ref_front), WORDS(stencil_ref_front), STATE_INDEX(stencil_ref_front)},
        {REG_INDEX(stencil_ref_back), WORDS(stencil_ref_back), STATE_INDEX(stencil_ref_back)},
        {REG_INDEX(viewports), WORDS(viewports), STATE_INDEX(viewports)},
        {REG_INDEX(clip_user_data), WORDS(clip_user_data), STATE_INDEX(clip_user_data)},
        {REG_INDEX(ps_inputs), WORDS(ps_inputs), STATE_INDEX(ps_inputs)},
        {REG_INDEX(vs_output_config), WORDS(vs_output_config), STATE_INDEX(vs_output_config)},
        {REG_INDEX(ps_input_ena), WORDS(ps_input_ena), STATE_INDEX(ps_input_ena)},
        {REG_INDEX(ps_input_addr), WORDS(ps_input_addr), STATE_INDEX(ps_input_addr)},
        {NumInterpReg, 1, STATE_INDEX(num_interp_word)},
        {REG_INDEX(barycentric_control), WORDS(barycentric_control), STATE_INDEX(barycentric_control)},
        {REG_INDEX(shader_pos_format), WORDS(shader_pos_format), STATE_INDEX(shader_pos_format)},
        {REG_INDEX(z_export_format), WORDS(z_export_format), STATE_INDEX(z_export_format)},
        {REG_INDEX(color_export_format), WORDS(color_export_format), STATE_INDEX(color_export_format)},
        {REG_INDEX(blend_control), WORDS(blend_control), STATE_INDEX(blend_control)},
        {REG_INDEX(index_base_address), WORDS(index_base_address), STATE_INDEX(index_base_address)},
        {REG_INDEX(draw_initiator), WORDS(draw_initiator), STATE_INDEX(draw_initiator)},
        {REG_INDEX(depth_control), WORDS(depth_control), STATE_INDEX(depth_control)},
        {REG_INDEX(color_control), WORDS(color_control), STATE_INDEX(color_control)},
        {REG_INDEX(depth_shader_control), WORDS(depth_shader_control), STATE_INDEX(depth_shader_control)},
        {REG_INDEX(clipper_control), WORDS(clipper_control), STATE_INDEX(clipper_control)},
        {REG_INDEX(polygon_control), WORDS(polygon_control), STATE_INDEX(polygon_control)},
        {REG_INDEX(viewport_control), WORDS(viewport_control), STATE_INDEX(viewport_control)},
        {REG_INDEX(vs_output_control), WORDS(vs_output_control), STATE_INDEX(vs_output_control)},
        {REG_INDEX(line_control), WORDS(line_control), STATE_INDEX(line_control)},
        {REG_INDEX(hs_clamp), WORDS(hs_clamp), STATE_INDEX(hs_clamp)},
        {REG_INDEX(vgt_gs_mode), WORDS(vgt_gs_mode), STATE_INDEX(vgt_gs_mode)},
        {REG_INDEX(vgt_gs_onchip_control), WORDS(vgt_gs_onchip_control), STATE_INDEX(vgt_gs_onchip_control)},
        {REG_INDEX(mode_control), WORDS(mode_control), STATE_INDEX(mode_control)},
        {REG_INDEX(vgt_gsvs_ring_offset_1), WORDS(vgt_gsvs_ring_offset_1), STATE_INDEX(vgt_gsvs_ring_offset_1)},
        {REG_INDEX(vgt_gsvs_ring_offset_2), WORDS(vgt_gsvs_ring_offset_2), STATE_INDEX(vgt_gsvs_ring_offset_2)},
        {REG_INDEX(vgt_gsvs_ring_offset_3), WORDS(vgt_gsvs_ring_offset_3), STATE_INDEX(vgt_gsvs_ring_offset_3)},
        {REG_INDEX(vgt_gs_out_prim_type), WORDS(vgt_gs_out_prim_type), STATE_INDEX(vgt_gs_out_prim_type)},
        {REG_INDEX(index_size), WORDS(index_size), STATE_INDEX(index_size)},
        {REG_INDEX(max_index_size), WORDS(max_index_size), STATE_INDEX(max_index_size)},
        {REG_INDEX(index_buffer_type), WORDS(index_buffer_type), STATE_INDEX(index_buffer_type)},
        {REG_INDEX(enable_primitive_id), WORDS(enable_primitive_id), STATE_INDEX(enable_primitive_id)},
        {REG_INDEX(enable_primitive_restart), WORDS(enable_primitive_restart), STATE_INDEX(enable_primitive_restart)},
        {REG_INDEX(vgt_instance_step_rate_0), WORDS(vgt_instance_step_rate_0), STATE_INDEX(vgt_instance_step_rate_0)},
        {REG_INDEX(vgt_instance_step_rate_1), WORDS(vgt_instance_step_rate_1), STATE_INDEX(vgt_instance_step_rate_1)},
        {REG_INDEX(vgt_esgs_ring_itemsize), WORDS(vgt_esgs_ring_itemsize), STATE_INDEX(vgt_esgs_ring_itemsize)},
        {REG_INDEX(vgt_gsvs_ring_itemsize), WORDS(vgt_gsvs_ring_itemsize), STATE_INDEX(vgt_gsvs_ring_itemsize)},
        {GsMaxVertOutReg, 1, STATE_INDEX(vgt_gs_max_vert_out_word)},
        {REG_INDEX(stage_enable), WORDS(stage_enable), STATE_INDEX(stage_enable)},
        {REG_INDEX(ls_hs_config), WORDS(ls_hs_config), STATE_INDEX(ls_hs_config)},
        {REG_INDEX(vgt_gs_vert_itemsize), WORDS(vgt_gs_vert_itemsize), STATE_INDEX(vgt_gs_vert_itemsize)},
        {REG_INDEX(tess_config), WORDS(tess_config), STATE_INDEX(tess_config)},
        {REG_INDEX(poly_offset), WORDS(poly_offset), STATE_INDEX(poly_offset)},
        {REG_INDEX(vgt_gs_instance_cnt), WORDS(vgt_gs_instance_cnt), STATE_INDEX(vgt_gs_instance_cnt)},
        {REG_INDEX(vgt_strmout_config), WORDS(vgt_strmout_config), STATE_INDEX(vgt_strmout_config)},
        {REG_INDEX(vgt_strmout_buffer_config), WORDS(vgt_strmout_buffer_config), STATE_INDEX(vgt_strmout_buffer_config)},
        {REG_INDEX(aa_config), WORDS(aa_config), STATE_INDEX(aa_config)},
        {REG_INDEX(color_buffers), WORDS(color_buffers), STATE_INDEX(color_buffers)},
        {REG_INDEX(cp_strmout_cntl), WORDS(cp_strmout_cntl), STATE_INDEX(cp_strmout_cntl)},
        {REG_INDEX(vgt_esgs_ring_size), WORDS(vgt_esgs_ring_size), STATE_INDEX(vgt_esgs_ring_size)},
        {REG_INDEX(vgt_gsvs_ring_size), WORDS(vgt_gsvs_ring_size), STATE_INDEX(vgt_gsvs_ring_size)},
        {REG_INDEX(primitive_type), WORDS(primitive_type), STATE_INDEX(primitive_type)},
        {REG_INDEX(num_indices), WORDS(num_indices), STATE_INDEX(num_indices)},
        {REG_INDEX(num_instances), WORDS(num_instances), STATE_INDEX(num_instances)},
        {REG_INDEX(vgt_tf_memory_base), WORDS(vgt_tf_memory_base), STATE_INDEX(vgt_tf_memory_base)},
    };
    return ranges;
}

} // namespace

const std::array<u32, Regs::NumRegs>& DrawRegs::Map() {
    static const auto map = [] {
        std::array<u32, Regs::NumRegs> table{};
        for (const Range& r : Ranges()) {
            for (u32 i = 0; i < r.words; ++i) {
                table[r.reg + i] = r.state + i + 1;
            }
        }
        return table;
    }();
    return map;
}

void DrawRegs::Gather(const Regs& regs) {
    for (const Range& r : Ranges()) {
        std::memcpy(reinterpret_cast<u32*>(this) + r.state, regs.reg_array.data() + r.reg, r.words * sizeof(u32));
    }
}

void DrawRegs::Dump(u32* out) const {
    for (const Range& r : Ranges()) {
        std::memcpy(out + r.reg, reinterpret_cast<const u32*>(this) + r.state, r.words * sizeof(u32));
    }
}

void DrawRegs::SetDefaults() {
    static const auto defaults = [] {
        auto regs = std::make_unique<Regs>();
        regs->SetDefaults();
        DrawRegs state{};
        state.Gather(*regs);
        return state;
    }();
    *this = defaults;
}

} // namespace AmdGpu
