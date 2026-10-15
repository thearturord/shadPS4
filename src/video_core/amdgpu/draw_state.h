// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Generated from regs.h by documents/tlg (gen_draw_state.py): the state of a draw as the backend reads it.

#pragma once

#include <array>
#include <cstring>
#include "common/types.h"
#include "video_core/amdgpu/regs.h"

namespace AmdGpu {

/// What the rasterizer and the pipeline cache read of a draw: the same members as `Regs` (same names, same types),
/// without the register file around them (no padding, no compute program). A record or a command packet changes a
/// member through `Set(register index, value)`; `Gather` builds the state from a register file.
struct DrawRegs {
    ShaderProgram ps_program;
    ShaderProgram vs_program;
    ShaderProgram gs_program;
    ShaderProgram es_program;
    ShaderProgram hs_program;
    ShaderProgram ls_program;
    DepthRenderControl depth_render_control;
    DepthView depth_view;
    DepthRenderOverride depth_render_override;
    Address depth_htile_data_base;
    float depth_bounds_min;
    float depth_bounds_max;
    u32 stencil_clear;
    float depth_clear;
    Scissor screen_scissor;
    DepthBuffer depth_buffer;
    BorderColorBuffer ta_bc_base;
    WindowOffset window_offset;
    ViewportScissor window_scissor;
    ColorBufferMask color_target_mask;
    ColorBufferMask color_shader_mask;
    ViewportScissor generic_scissor;
    std::array<ViewportScissor, NUM_VIEWPORTS> viewport_scissors;
    std::array<ViewportDepth, NUM_VIEWPORTS> viewport_depths;
    u32 index_offset;
    u32 primitive_restart_index;
    BlendConstants blend_constants;
    StencilControl stencil_control;
    StencilRefMask stencil_ref_front;
    StencilRefMask stencil_ref_back;
    std::array<ViewportBounds, NUM_VIEWPORTS> viewports;
    std::array<ClipUserData, NUM_CLIP_PLANES> clip_user_data;
    std::array<PsInputControl, 32> ps_inputs;
    VsOutputConfig vs_output_config;
    PsInput ps_input_ena;
    PsInput ps_input_addr;
    union {
        u32 num_interp_word;
        struct {
            u32 num_interp : 6;
        };
    };
    BarycentricControl barycentric_control;
    ShaderPosFormat shader_pos_format;
    ShaderExportFormat z_export_format;
    ColorExportFormat color_export_format;
    std::array<BlendControl, NUM_COLOR_BUFFERS> blend_control;
    IndexBufferBase index_base_address;
    u32 draw_initiator;
    DepthControl depth_control;
    ColorControl color_control;
    DepthShaderControl depth_shader_control;
    ClipperControl clipper_control;
    PolygonControl polygon_control;
    ViewportControl viewport_control;
    VsOutputControl vs_output_control;
    LineControl line_control;
    TessFactorClamp hs_clamp;
    GsMode vgt_gs_mode;
    GsOnchip vgt_gs_onchip_control;
    ModeControl mode_control;
    RingOffset vgt_gsvs_ring_offset_1;
    RingOffset vgt_gsvs_ring_offset_2;
    RingOffset vgt_gsvs_ring_offset_3;
    GsOutPrimitiveType vgt_gs_out_prim_type;
    u32 index_size;
    u32 max_index_size;
    IndexBufferType index_buffer_type;
    u32 enable_primitive_id;
    u32 enable_primitive_restart;
    u32 vgt_instance_step_rate_0;
    u32 vgt_instance_step_rate_1;
    u32 vgt_esgs_ring_itemsize;
    u32 vgt_gsvs_ring_itemsize;
    union {
        u32 vgt_gs_max_vert_out_word;
        struct {
            u32 vgt_gs_max_vert_out : 11;
        };
    };
    ShaderStageEnable stage_enable;
    LsHsConfig ls_hs_config;
    u32 vgt_gs_vert_itemsize[4];
    TessellationConfig tess_config;
    PolygonOffset poly_offset;
    GsInstances vgt_gs_instance_cnt;
    StreamOutConfig vgt_strmout_config;
    StreamOutBufferConfig vgt_strmout_buffer_config;
    AaConfig aa_config;
    ColorBuffer color_buffers[NUM_COLOR_BUFFERS];
    StreamOutControl cp_strmout_cntl;
    u32 vgt_esgs_ring_size;
    u32 vgt_gsvs_ring_size;
    PrimitiveType primitive_type;
    u32 num_indices;
    VgtNumInstances num_instances;
    TessFactorMemoryBase vgt_tf_memory_base;

    /// Writes one register dword (ignored for registers no member covers).
    void Set(u32 register_index, u32 value) {
        const u32 word = Map()[register_index];
        if (word != 0) {
            reinterpret_cast<u32*>(this)[word - 1] = value;
        }
    }
    /// The dword of one register (nullptr for registers no member covers).
    u32* Word(u32 register_index) {
        const u32 word = Map()[register_index];
        return word != 0 ? &reinterpret_cast<u32*>(this)[word - 1] : nullptr;
    }
    /// Reads one register dword back (0 for registers no member covers); `known` tells which.
    u32 Get(u32 register_index, bool* known = nullptr) const {
        const u32 word = Map()[register_index];
        if (known) {
            *known = word != 0;
        }
        return word != 0 ? reinterpret_cast<const u32*>(this)[word - 1] : 0;
    }
    /// Copies every member from a register file.
    void Gather(const Regs& regs);
    /// Writes every member to its register of a register file image (`out` holds Regs::NumRegs dwords).
    void Dump(u32* out) const;
    /// True when a member covers the register.
    static bool IsMapped(u32 register_index) {
        return Map()[register_index] != 0;
    }
    /// The state of a register file after `Regs::SetDefaults`.
    void SetDefaults();

    const ShaderProgram* ProgramForStage(u32 index) const {
        switch (index) {
        case 0:
            return &ps_program;
        case 1:
            return &vs_program;
        case 2:
            return &gs_program;
        case 3:
            return &es_program;
        case 4:
            return &hs_program;
        case 5:
            return &ls_program;
        }
        return nullptr;
    }

    bool IsClipDisabled() const {
        return clipper_control.clip_disable || primitive_type == PrimitiveType::RectList;
    }

    /// Register index -> 1 + dword index in this struct (0: not part of the draw state).
    static const std::array<u32, Regs::NumRegs>& Map();
};

} // namespace AmdGpu
