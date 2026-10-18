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
    /// synchronization packets taken out of the stream: scratch barrier units (label, end of shader event, wait on the label,
    /// cache acquire), stand-alone cache acquires and no-op events
    /// draws of a shader without a fetch object yet (`native_sdk_strip` bit 2147483648): the game's draw function ran, the record was built again afterwards
    /// the texture map helper's upload dispatch (`map_native`): calls seen, records patched into the tag, refused, executed
    /// compute program audit (`cs_audit`): record dispatches seen, those whose program (queue state plus record) differs from the record alone on a zeroed program, and the dwords that differ
    std::atomic<u64> cs_audit_dispatches{}, cs_audit_differing{};
    std::array<std::atomic<u64>, 128> cs_audit_dword{};
    std::atomic<u64> map_tried{}, map_patched{}, map_refused{}, map_exec{};
    std::atomic<u64> fetch_rebuild_tried{}, fetch_rebuilt{}, fetch_rebuild_refused{}, fetch_rebuild_exec{};
    std::atomic<u64> sync_units_stripped{}, sync_acquires_stripped{}, sync_events_stripped{}, barrier_calls_skipped{};
    /// direct submission, step 1 (shadow index, `native_sdk_strip` bit 1048576): tags registered at their address, command
    /// buffer spans walked, spans without any packet besides entries ("pure") and with some ("mixed"), tags the walk found that
    /// the index has / has not, indexed tags the walk did not find
    std::atomic<u64> shadow_anchors{}, shadow_spans{}, shadow_pure{}, shadow_mixed{}, shadow_matched{}, shadow_extra{}, shadow_missing{}, shadow_nested{}, shadow_stale{};
    /// sync records (`native_sdk_strip` bit 2097152): packets turned into tags, by kind (1 graphics end-of-pipe, 2 release memory, 3 wait, 4 write data)
    /// and by place (the hook of FUN_02480d60, the sweep when the next tag is written, the submit of a command buffer); packets the
    /// hook left as written; tags the command processor executed, by kind
    std::array<std::atomic<u64>, 5> sync_converted{}, sync_exec{};
    std::atomic<u64> sync_converted_hook{}, sync_converted_sweep{}, sync_converted_submit{}, fence_packets_kept{};
    /// direct submission, step 3a (bit 4194304): graphics buffers whose entry list was built, pure / mixed ones, entries listed, entries the walk
    /// found in the list, entries that differ or are missing, buffers whose non-entry packets differ between list and walk, buffers whose list is incomplete (not compared)
    std::atomic<u64> direct_built{}, direct_pure{}, direct_mixed{}, direct_entries{}, direct_matched{}, direct_entry_mismatch{}, direct_purity_mismatch{}, direct_incomplete{}, direct_executed{}, direct_executed_asc{};
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
    u64 fill_destination{}; ///< dispatches of the device's fill shader: the pattern fill it performs (0: not one)
    u32 fill_bytes{};
    u32 fill_dwords{}; ///< dwords of the pattern (at most 4)
    std::array<u32, 4> fill_pattern{};
    u64 copy_destination{}; ///< dispatches of the device's buffer copy shader: the memory copy it performs (null: not one)
    u64 copy_source{};
    u32 copy_bytes{};
    const Vulkan::FixedState* fixed{}; ///< draws: the fixed-function state built from the engine's objects (null: from the registers)
    u32 fixed_stale{};                 ///< FixedStateStale groups of `fixed` the hardware keeps from earlier draws
    u32 dst_usage{~0u};                ///< copy and counter read-back records: Usage of the destination resource (3 staging)
};

/// Command processor: the index / vertex count the record's draw was called with (0xffffffff when the record is gone).
u32 RecordDrawCount(u64 seq);

/// Direct submission, step 1 (documents/tlg/native-direct-submission.md). The record builders register the address of every tag
/// they write; the packet walk of a command buffer span compares what it finds with the registered tags of the span and counts
/// the packets that are not entries (inert padding aside), which tells whether the span could be submitted without the walk.
bool SubmitShadowEnabled();
void NoteAnchor(u64 address, u64 seq, u32 kind);
/// One line that describes the tag at `header` and the record it names (kind, mode, calling function as a Ghidra address, operands). Used by the entry dump.
std::string DescribeEntry(const void* header);
/// Submit time: the synchronization packets of the command buffer (and of the indirect buffers it calls) become tags.
void PrepareSubmittedBuffer(const u32* dcb, u32 dwords);
/// Command processor: when the no-op `tag` stands for a synchronization packet, writes that packet into `packet` and returns its
/// opcode (0x47, 0x49, 0x3c, 0x37); 0 for any other no-op.
u32 RestoreSyncTag(const u32* tag, std::array<u32, 8>& packet, bool count_execution = true);
/// Per opcode: packets in walked spans that are not entries, flips and indirect buffers included (/native).
u64 ShadowNonEntryCount(u32 opcode);

class SubmitShadow {
public:
    SubmitShadow() = default;
    SubmitShadow(const SubmitShadow&) = delete;
    SubmitShadow& operator=(const SubmitShadow&) = delete;
    ~SubmitShadow() {
        End();
    }
    /// `queue`: 0 graphics, 1 + n async compute queue n; the span is the dwords of the command buffer.
    void Begin(u32 queue, const u32* begin, u32 dwords);
    /// The walk found a tag at `header`.
    void Tag(const void* header);
    /// The walk found another packet: padding / no-ops the walk ignores are `inert`.
    void Packet(u32 opcode, bool inert);
    /// The walk follows an indirect buffer: its tags belong to that span, even when it lies inside this one.
    void Indirect(u64 address, u32 dwords);
    /// The walk skipped the payload of a no-op: an old tag left in it (the buffers are reused) is not an entry of this span.
    void Skipped(u64 address, u32 dwords);
    void End();
    bool Active() const {
        return active;
    }

private:
    bool active{};
    u32 queue{};
    u64 begin{};
    u64 end{};
    std::vector<u64> expected; ///< sorted tag addresses of the span
    std::vector<u8> seen;
    std::array<u32, 256> other{}; ///< packets by opcode that are not entries
    u32 extra{};
    std::vector<std::pair<u64, u64>> nested;  ///< address ranges of the indirect buffers the walk followed
    std::vector<std::pair<u64, u64>> skipped; ///< payloads of the no-ops the walk stepped over
};
/// Tags the walk found without an index entry, by tag kind (/native).
u64 ShadowExtraByKind(u32 kind);

/// The Ghidra address of the call that made record `seq` (0 when the record is gone).
u64 RecordCaller(u64 seq);

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
