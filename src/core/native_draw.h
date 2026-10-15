// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>
#include "common/types.h"

// Host functions called from the guest at chosen game functions (hook mode "call", see guest_hooks.h).
//
// The game's render layer (JsDeviceContext) is a D3D11-style object model; every draw goes through
// PrepareDraw. These handlers read the context's object pointers at the draw entry points. The first
// handlers only measure (how much of the draw state changes from one draw to the next); the native path
// builds on the same reading code. Layout of the context and the objects: documents/tlg/spike/js_ctx_state.h.
namespace Core::NativeDraw {

/// Address of the host function registered under `name` (a PS4_SYSV_ABI function taking
/// (a0..a5, return address) and returning non-zero when it handled the call and the guest function must
/// be skipped), or 0.
u64 FindHandler(const std::string& name);

/// Runtime address of the eboot image (callers are classified by their Ghidra address).
void SetImageBase(VAddr base);

/// Raster thread: compares the register state recorded for tagged draw `seq` with the raster thread's register file
/// (`reg_array`, index = register dword offset) and counts the differences (shadow mode).
void VerifyRecord(u64 seq, const u32* reg_array);

/// Index base of the compute program registers in a record: dword `d` of the compute program is index `ComputeRegisterBase + d`.
constexpr u32 ComputeRegisterBase = 0xE000;

/// Raster thread: the same comparison for a dispatch; `cs_dwords` is the compute program the dispatch runs with.
void VerifyComputeRecord(u64 seq, const u32* cs_dwords);

struct ExecRegister {
    u32 index; ///< index in the register file (dword offset)
    u32 value;
};

/// A draw the game's render layer did not write as command packets: the registers its objects describe and the
/// draw arguments.
struct ExecDraw {
    static constexpr u32 MaxRegisters = 320;
    std::array<ExecRegister, MaxRegisters> regs;
    u32 count{};
    u32 kind{};        ///< 0 = indexed, 1 = auto
    u32 start_index{}; ///< indexed draws
    u64 args_addr{};   ///< indirect draws (kind 2)
    u32 offset{};
    u32 vertex_sgpr{};
    u32 instance_sgpr{};
};

/// Command processor: the index / vertex count the record's draw was called with (0xffffffff when the record is gone).
u32 RecordDrawCount(u64 seq);

/// Command processor: the record of a draw the hook executed natively. False when the record was overwritten.
bool TakeExecDraw(u64 seq, ExecDraw& out);

} // namespace Core::NativeDraw
