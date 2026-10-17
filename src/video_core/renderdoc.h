// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include "common/types.h"

namespace VideoCore {

/// Loads renderdoc dynamic library module.
void LoadRenderDoc();

/// Begins a capture if a renderdoc instance is attached.
void StartCapture();

/// Ends current renderdoc capture.
void EndCapture();

/// Triggers capturing process.
void TriggerCapture();

/// Sets output directory for captures
void SetOutputDir(const std::filesystem::path& path, const std::string& prefix);

/// Returns true when RenderDoc API was loaded and is usable.
bool IsRenderDocLoaded();

enum class ScreenshotRequest : u32 {
    None = 0,
    GameOnly = 1,
    WithOverlays = 2,
};

struct ScreenshotRequests {
    u32 game_only_count{};
    u32 with_overlays_count{};
};

/// Queues an in-emulator screenshot request to be consumed by the presenter.
void RequestScreenshot(ScreenshotRequest request);

/// Atomically consumes and returns pending "game only" screenshot request counter.
u32 ConsumeGameOnlyScreenshotRequests();

/// Atomically consumes and returns pending "with overlays" screenshot request counter.
u32 ConsumeWithOverlaysScreenshotRequests();

/// Atomically consumes and returns pending screenshot request counters.
ScreenshotRequests ConsumeScreenshotRequests();

/// Debug: capture the guest output of the next `frames` presented game frames (every flip once), cropped to the rectangle given in
/// fractions of the image, into log/frame_sequence (debug server /capture).
struct FrameSequenceStep {
    u32 index{};
    float x0{}, y0{}, x1{1.0f}, y1{1.0f};
};
void RequestFrameSequence(u32 frames, float x0, float y0, float x1, float y1);
/// True when the game frame being prepared is part of the sequence; fills in what to capture.
bool TakeFrameSequenceStep(FrameSequenceStep& step);
/// Frames of the sequence not captured yet.
u32 FrameSequenceRemaining();

} // namespace VideoCore
