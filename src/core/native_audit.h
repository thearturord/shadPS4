// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

// Tracker audit (documents/tlg/native-isolation.md). The buffer cache decides which bytes of guest memory the GPU sees: it uploads the
// pages the CPU wrote since the last upload, which it learns from page protection faults. If it misses a write, the native renderer
// draws with stale data and the picture is wrong although the native path did everything right. This audit measures how often that
// happens: it keeps a hash of a sample of the pages at the moment they are uploaded, and when the buffer cache later says a range is clean
// it hashes the current guest memory of the sampled pages again. A page that differs was written by the CPU (or by the emulator itself)
// without the tracker noticing. Instrumentation only: nothing here changes what is uploaded.
namespace Core::NativeAudit {

/// Bit 67108864 of `native_sdk_strip`.
bool Enabled();

/// The buffer cache uploaded [address, address + size) (page aligned) from `data`.
void NoteUpload(VAddr address, const u8* data, u64 size);

/// The GPU writes [address, address + size): the guest memory will differ from the upload legitimately, forget the hashes.
void NoteGpuWrite(VAddr address, u64 size);

/// The buffer cache calls [address, address + size) clean (no CPU write pending, not GPU modified): compares the sampled pages with the
/// hashes taken at their upload. `gpu_modified(page)` answers whether the tracker marks a page GPU modified.
void CheckClean(VAddr address, u64 size);

/// Bit 134217728 of `native_sdk_strip`: the stale state audit. A native draw takes the groups the engine did not write for it (surface,
/// clip, clear flags) from the register mirror, which holds what earlier draws wrote. The audit counts the draws where that value is
/// not the one the engine's own objects describe.
bool StaleEnabled();
void NoteStale(u32 groups, bool differs, const std::string& diff, u64 caller, u64 present);
/// The Ghidra address of the call that made the record of the draw being planned on this thread (0 when not known).
void SetPlanCaller(u64 caller);
u64 PlanCaller();

nlohmann::json Report();

} // namespace Core::NativeAudit
