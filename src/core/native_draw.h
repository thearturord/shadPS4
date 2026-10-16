// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include <array>
#include <string>
#include "common/types.h"
#include "video_core/amdgpu/draw_state.h"

// Host functions called from the guest at chosen game functions (hook mode "call", see guest_hooks.h).
//
// The game's render layer (JsDeviceContext) is a D3D11-style object model; every draw goes through
// PrepareDraw. These handlers read the context's object pointers at the draw entry points. The first
// handlers only measure (how much of the draw state changes from one draw to the next); the native path
// builds on the same reading code. Layout of the context and the objects: documents/tlg/spike/js_ctx_state.h.
namespace Vulkan {
struct FixedState;
}

namespace Core::NativeDraw {

/// Address of the host function registered under `name` (a PS4_SYSV_ABI function taking
/// (a0..a5, return address) and returning non-zero when it handled the call and the guest function must
/// be skipped), or 0.
u64 FindHandler(const std::string& name);

/// Runtime address of the eboot image (callers are classified by their Ghidra address).
/// Draw count of a record whose draw packet is not known yet (matches any count).
constexpr u32 UnknownDrawCount = 0xfffffffe;

/// Cumulative counters of the native path (read by the debug server's /native endpoint; the log lines use their own).
struct LiveCounters {
    std::atomic<u64> gfx_tags{}, gfx_exec{};
    std::atomic<u64> compute_tags{}, compute_exec{};
    std::atomic<u64> asc_exec{}, asc_packet{}, asc_no_record{}, fill_exec{}, mode_switch_exec{}, asc_barrier_exec{}, clear_exec{}, asc_begin_stripped{}, counter_readback{}, async_copy_exec{}, device_begin_stripped{};
    std::atomic<u64> track_native{}, track_fce{}, track_game_second{}, track_game_depth{};
    /// fixed-function state of native draws: used from the engine's objects, compared with the registers, and the compares that differed
    std::atomic<u64> fixed_native{}, fixed_checked{}, fixed_diff{};
    /// command packets by the group of the last native tag: 0 none, 1 executed draw, 2 packet-path draw, 3 executed
    /// dispatch, 4 packet-path dispatch, 5 fast clear eliminate
    std::array<std::atomic<u64>, 6> packets{};
    std::atomic<u64> asc_packets{};
};
LiveCounters& Live();

/// The engine's JsDevice (0 until known).
u64 EngineDevice();

void SetImageBase(VAddr base);
/// Load address of the game's main module (0 until known).
VAddr ImageBase();
/// True on a thread that is running a guarded record build: a fault there is handled by the builder, not fatal.
bool IsGuarded();

/// Raster thread: compares the register state recorded for tagged draw `seq` with the raster thread's register file
/// (`reg_array`, index = register dword offset) and counts the differences (shadow mode).
void VerifyRecord(u64 seq, const u32* reg_array);
/// The same for the raster thread's draw state: only the registers that belong to the state are compared.
void VerifyRecord(u64 seq, const AmdGpu::DrawRegs& state);

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
    u64 extra_addr{};  ///< copy records (kind 10): the source address
    u32 offset{};
    u32 vertex_sgpr{};
    u32 instance_sgpr{};
    std::array<u32, 8> cb_extent{}; ///< render target extent hints (the game's trailing NOP of a target bind)
    u32 cb_extent_mask{};           ///< bit per slot whose hint is set by this draw
    u32 db_extent{};
    bool has_extents{};
    u32 a1{}; ///< kind 6 (buffer fill): number of dwords; start_index holds the value, args_addr the destination
    bool cb_extents_only{}; ///< fast clear eliminate: only the render target extents change
    u64 copy_destination{}; ///< dispatches of the device's buffer copy shader: the memory copy it performs (null: not one)
    u64 copy_source{};
    u32 copy_bytes{};
    const Vulkan::FixedState* fixed{}; ///< draws: the fixed-function state built from the engine's objects (null: from the registers)
    u32 fixed_stale{};                 ///< FixedStateStale groups of `fixed` the hardware keeps from earlier draws
};

/// Command processor: the index / vertex count the record's draw was called with (0xffffffff when the record is gone).
u32 RecordDrawCount(u64 seq);

/// Command processor: the record of a draw the hook executed natively. False when the record was overwritten.
bool TakeExecDraw(u64 seq, ExecDraw& out);

/// The register effect of a constant command packet sequence of the emulated Gnm driver (the hardware default state the game
/// sends at the start of every command buffer): the writes in order, parsed once. A tag record names the template; the
/// command processor applies the writes in place of the packets.
struct StateWrite {
    enum Kind : u8 { Context, Sh, Uconfig, NumInstances, ClearState, IndexType };
    u8 kind;
    u32 reg;   ///< register offset within its range
    u32 value;
};
struct StateTemplate {
    std::vector<StateWrite> writes;
};

/// The template of `prefix` followed by `sequence` (both packet dwords); the same pointers always give the same template.
/// Null when the packets contain anything but register writes, clear state and no-ops.
const StateTemplate* HardwareStateTemplate(const u32* prefix, u32 prefix_dwords, const u32* sequence, u32 sequence_dwords);

/// The template of a small indirect buffer that holds nothing but register writes, clear state and padding (the game records its
/// default state once and submits it every frame). Cached by the content; null for anything else.
const StateTemplate* IndirectStateTemplate(const u32* packets, u32 dwords);

} // namespace Core::NativeDraw
