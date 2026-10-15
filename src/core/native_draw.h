// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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

} // namespace Core::NativeDraw
