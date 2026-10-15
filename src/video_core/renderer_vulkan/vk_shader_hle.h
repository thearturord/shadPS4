// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace AmdGpu {
struct ComputeProgram;
union Regs;
struct DrawRegs;
} // namespace AmdGpu

namespace Shader {
struct Info;
}

namespace Vulkan {

class Rasterizer;

/// Attempts to execute a shader using HLE if possible.
bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::DrawRegs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer);

} // namespace Vulkan
