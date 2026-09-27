// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

/// Wraps the Vulkan functions used while recording so the time spent in the driver on the
/// command processor thread shows up in the readback stats. Call after the dispatcher has been
/// initialized for the device.
void InstallVulkanCallTiming();

} // namespace Vulkan
