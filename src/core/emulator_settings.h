// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "common/types.h"

#define EmulatorSettings (*EmulatorSettingsImpl::GetInstance())

enum HideCursorState : int {
    Never,
    Idle,
    Always,
};

enum UsbBackendType : int {
    Real,
    SkylandersPortal,
    InfinityBase,
    DimensionsToypad,
};

enum GpuReadbacksMode : int {
    Disabled,
    Relaxed,
    Precise,
};

enum class ConfigMode {
    Default,
    Global,
    Clean,
};

enum AudioBackend : int {
    SDL,
    OpenAL,
    // Add more backends as needed
};

enum OpenALHrtfMode : int {
    HrtfAuto, // Let OpenAL Soft decide (on for headphone-like stereo outputs)
    HrtfOn,   // Force HRTF binaural rendering
    HrtfOff,  // Never use HRTF
};

enum OpenALOutputMode : int {
    OutputAuto,       // Let OpenAL Soft negotiate with the device
    OutputStereo,     // Force stereo output
    OutputQuad,       // Force quadraphonic output
    OutputSurround51, // Force 5.1 surround output
    OutputSurround71, // Force 7.1 surround output
};

template <typename T>
struct Setting {
    T default_value{};
    T value{};
    std::optional<T> game_specific_value{};

    Setting() = default;
    // Single-argument ctor: initialises both default_value and value so
    // that CleanMode can always recover the intended factory default.
    /*implicit*/ Setting(T init) : default_value(std::move(init)), value(default_value) {}

    /// Return the active value under the given mode.
    T get(ConfigMode mode = ConfigMode::Default) const {
        switch (mode) {
        case ConfigMode::Default:
            return game_specific_value.value_or(value);
        case ConfigMode::Global:
            return value;
        case ConfigMode::Clean:
            return default_value;
        }
        return value;
    }

    /// Write v to the base layer.
    /// Set proper value as base or game_specific
    void set(const T& v, bool game_specific = false) {
        if (game_specific) {
            game_specific_value = v;
        } else {
            value = v;
        }
    }

    /// Discard the game-specific override; subsequent get(Default) will
    /// fall back to the base value.
    void reset_game_specific() {
        game_specific_value = std::nullopt;
    }
};

template <typename T>
void to_json(nlohmann::json& j, const Setting<T>& s) {
    j = s.value;
}

template <typename T>
void from_json(const nlohmann::json& j, Setting<T>& s) {
    s.value = j.get<T>();
}

struct OverrideItem {
    const char* key;
    std::function<void(void* group_ptr, const nlohmann::json& entry,
                       std::vector<std::string>& changed)>
        apply;
    /// Return the value that should be written to the per-game config file.
    /// Falls back to base value if no game-specific override is set.
    std::function<nlohmann::json(const void* group_ptr)> get_for_save;

    /// Clear game_specific_value for this field.
    std::function<void(void* group_ptr)> reset_game_specific;
};

template <typename Struct, typename T>
inline OverrideItem make_override(const char* key, Setting<T> Struct::* member) {
    return OverrideItem{
        key,
        [member, key](void* base, const nlohmann::json& entry, std::vector<std::string>& changed) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            Setting<T>& dst = obj->*member;
            try {
                T newValue = entry.get<T>();
                if (dst.value != newValue) {
                    std::ostringstream oss;
                    oss << key << " ( " << dst.value << " -> " << newValue << " )";
                    changed.push_back(oss.str());
                }
                dst.game_specific_value = newValue;
            } catch (const std::exception& e) {
                LOG_ERROR(Config, "[make_override] error parsing {}: {}", key, e.what());
                LOG_ERROR(Config, "[make_override] Entry was: {}", entry.dump());
                LOG_ERROR(Config, "[make_override] Type name: {}", entry.type_name());
            }
        },

        // --- get_for_save -------------------------------------------
        // Returns game_specific_value when present, otherwise base value.
        // This means a freshly-opened game-specific dialog still shows
        // useful (current-global) values rather than empty entries.
        [member](const void* base) -> nlohmann::json {
            const Struct* obj = reinterpret_cast<const Struct*>(base);
            const Setting<T>& src = obj->*member;
            return nlohmann::json(src.game_specific_value.value_or(src.value));
        },

        // --- reset_game_specific ------------------------------------
        [member](void* base) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            (obj->*member).reset_game_specific();
        }};
}

// -------------------------------
// Support types
// -------------------------------
struct GameInstallDir {
    std::filesystem::path path;
    bool enabled;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GameInstallDir, path, enabled)

// -------------------------------
// General settings
// -------------------------------
struct GeneralSettings {
    Setting<std::vector<GameInstallDir>> install_dirs;
    Setting<std::filesystem::path> addon_install_dir;
    Setting<std::filesystem::path> home_dir;
    Setting<std::filesystem::path> sys_modules_dir;
    Setting<std::filesystem::path> font_dir;

    Setting<int> volume_slider{100};
    Setting<bool> neo_mode{false};
    Setting<bool> dev_kit_mode{false};
    Setting<int> extra_dmem_in_mbytes{0};
    Setting<int> extra_fmem_in_mbytes{0};
    Setting<bool> trophy_popup_disabled{false};
    Setting<double> trophy_notification_duration{6.0};
    Setting<std::string> trophy_notification_side{"right"};
    Setting<bool> show_splash{false};
    Setting<bool> discord_rpc_enabled{false};
    Setting<bool> show_fps_counter{false};
    Setting<int> console_language{1};
    Setting<int> big_picture_scale{1000};
    Setting<int> big_picture_folder_depth{2};
    Setting<bool> redzone_patches{false};
    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<GeneralSettings>("volume_slider", &GeneralSettings::volume_slider),
            make_override<GeneralSettings>("neo_mode", &GeneralSettings::neo_mode),
            make_override<GeneralSettings>("dev_kit_mode", &GeneralSettings::dev_kit_mode),
            make_override<GeneralSettings>("extra_dmem_in_mbytes",
                                           &GeneralSettings::extra_dmem_in_mbytes),
            make_override<GeneralSettings>("extra_fmem_in_mbytes",
                                           &GeneralSettings::extra_fmem_in_mbytes),
            make_override<GeneralSettings>("trophy_popup_disabled",
                                           &GeneralSettings::trophy_popup_disabled),
            make_override<GeneralSettings>("trophy_notification_duration",
                                           &GeneralSettings::trophy_notification_duration),
            make_override<GeneralSettings>("show_splash", &GeneralSettings::show_splash),
            make_override<GeneralSettings>("trophy_notification_side",
                                           &GeneralSettings::trophy_notification_side),
            make_override<GeneralSettings>("console_language", &GeneralSettings::console_language),
            make_override<GeneralSettings>("redzone_patches", &GeneralSettings::redzone_patches)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneralSettings, install_dirs, addon_install_dir, home_dir,
                                   sys_modules_dir, font_dir, volume_slider, neo_mode, dev_kit_mode,
                                   extra_dmem_in_mbytes, extra_fmem_in_mbytes,
                                   trophy_popup_disabled, trophy_notification_duration, show_splash,
                                   trophy_notification_side, discord_rpc_enabled, show_fps_counter,
                                   console_language, big_picture_scale, big_picture_folder_depth,
                                   redzone_patches)

// -------------------------------
// Network settings
// -------------------------------
struct NetworkSettings {
    Setting<bool> connected_to_network{false};
    Setting<bool> shad_net_enabled{false};
    Setting<std::string> shadnet_server{"srv.shadps4.net:31313"};
    Setting<std::string> shadnet_webapi_server{"http://srv.shadps4.net:31315"};
    Setting<bool> enable_upnp{true};
    Setting<int> p2p_port{3658}; // 0 is any free port, 3658 is the default PS4 port. With UPnP
                                 // enabled it is forwarded on the router.
    Setting<bool> disable_https{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<NetworkSettings>("connected_to_network",
                                           &NetworkSettings::connected_to_network),
            make_override<NetworkSettings>("shad_net_enabled", &NetworkSettings::shad_net_enabled),
            make_override<NetworkSettings>("shadnet_server", &NetworkSettings::shadnet_server),
            make_override<NetworkSettings>("shadnet_webapi_server",
                                           &NetworkSettings::shadnet_webapi_server),
            make_override<NetworkSettings>("enable_upnp", &NetworkSettings::enable_upnp),
            make_override<NetworkSettings>("p2p_port", &NetworkSettings::p2p_port),
            make_override<NetworkSettings>("disable_https", &NetworkSettings::disable_https)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkSettings, connected_to_network, shad_net_enabled,
                                   shadnet_server, shadnet_webapi_server, enable_upnp, p2p_port,
                                   disable_https)

// -------------------------------
// Log settings
// -------------------------------
struct LogSettings {
    Setting<bool> append{false}; // specific
    Setting<bool> enable{true};  // specific
    Setting<std::string> filter{""};
    Setting<std::string> flush_level{""};
    Setting<u32> max_skip_duration{5'000};
    Setting<bool> separate{false}; // specific
    Setting<unsigned long long> size_limit{100_MB};
    Setting<bool> skip_duplicate{true};
    Setting<bool> sync{true};
#ifdef _WIN32
    Setting<std::string> type{"wincolor"};
#endif

    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<LogSettings>("append", &LogSettings::append),
            make_override<LogSettings>("enable", &LogSettings::enable),
            make_override<LogSettings>("filter", &LogSettings::filter),
            make_override<LogSettings>("flush_level", &LogSettings::flush_level),
            make_override<LogSettings>("max_skip_duration", &LogSettings::max_skip_duration),
            make_override<LogSettings>("separate", &LogSettings::separate),
            make_override<LogSettings>("size_limit", &LogSettings::size_limit),
            make_override<LogSettings>("skip_duplicate", &LogSettings::skip_duplicate),
            make_override<LogSettings>("sync", &LogSettings::sync),
#ifdef _WIN32
            make_override<LogSettings>("type", &LogSettings::type),
#endif
        };
    }
};
#ifdef _WIN32
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, flush_level,
                                   max_skip_duration, separate, size_limit, skip_duplicate, sync,
                                   type)
#else
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, flush_level,
                                   max_skip_duration, separate, size_limit, skip_duplicate, sync)
#endif

// -------------------------------
// Debug settings
// -------------------------------
struct DebugSettings {
    Setting<bool> debug_dump{false};         // specific
    Setting<bool> shader_collect{false};     // specific
    Setting<std::string> config_version{""}; // specific

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<DebugSettings>("debug_dump", &DebugSettings::debug_dump),
            make_override<DebugSettings>("shader_collect", &DebugSettings::shader_collect)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DebugSettings, debug_dump, shader_collect, config_version)

// -------------------------------
// Input settings
// -------------------------------

struct InputSettings {
    Setting<int> cursor_state{HideCursorState::Idle};      // specific
    Setting<int> cursor_hide_timeout{5};                   // specific
    Setting<int> usb_device_backend{UsbBackendType::Real}; // specific
    Setting<bool> use_special_pad{false};
    Setting<int> special_pad_class{1};
    Setting<bool> motion_controls_enabled{true}; // specific
    Setting<bool> use_unified_input_config{true};
    Setting<std::string> default_controller_id{""};
    Setting<bool> background_controller_input{false}; // specific
    Setting<bool> ime_accessibility_enabled{false};   // specific
    Setting<bool> ime_url_mail_short_panel{false};    // specific
    Setting<bool> is_circle_enter{false};             // specific
    Setting<s32> camera_id{-1};
    Setting<bool> use_mice_as_mice{false};
    Setting<bool> use_keyboard_as_keyboard{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<InputSettings>("cursor_state", &InputSettings::cursor_state),
            make_override<InputSettings>("cursor_hide_timeout",
                                         &InputSettings::cursor_hide_timeout),
            make_override<InputSettings>("usb_device_backend", &InputSettings::usb_device_backend),
            make_override<InputSettings>("motion_controls_enabled",
                                         &InputSettings::motion_controls_enabled),
            make_override<InputSettings>("background_controller_input",
                                         &InputSettings::background_controller_input),
            make_override<InputSettings>("ime_accessibility_enabled",
                                         &InputSettings::ime_accessibility_enabled),
            make_override<InputSettings>("ime_url_mail_short_panel",
                                         &InputSettings::ime_url_mail_short_panel),
            make_override<InputSettings>("is_circle_enter", &InputSettings::is_circle_enter),
            make_override<InputSettings>("camera_id", &InputSettings::camera_id),
            make_override<InputSettings>("use_mice_as_mice", &InputSettings::use_mice_as_mice),
            make_override<InputSettings>("use_keyboard_as_keyboard",
                                         &InputSettings::use_keyboard_as_keyboard)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InputSettings, cursor_state, cursor_hide_timeout,
                                   usb_device_backend, use_special_pad, special_pad_class,
                                   motion_controls_enabled, use_unified_input_config,
                                   default_controller_id, background_controller_input,
                                   ime_accessibility_enabled, ime_url_mail_short_panel, camera_id,
                                   is_circle_enter, use_mice_as_mice, use_keyboard_as_keyboard)
// -------------------------------
// Audio settings
// -------------------------------
struct AudioSettings {
    Setting<u32> audio_backend{AudioBackend::SDL};
    Setting<std::string> sdl_mic_device{"Default Device"};
    Setting<std::string> sdl_main_output_device{"Default Device"};
    Setting<std::string> sdl_padSpk_output_device{"Default Device"};
    Setting<std::string> openal_mic_device{"Default Device"};
    Setting<std::string> openal_main_output_device{"Default Device"};
    Setting<std::string> openal_padSpk_output_device{"Default Device"};
    Setting<u32> openal_hrtf{OpenALHrtfMode::HrtfAuto};
    Setting<u32> openal_output_mode{OpenALOutputMode::OutputAuto};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<AudioSettings>("audio_backend", &AudioSettings::audio_backend),
            make_override<AudioSettings>("sdl_mic_device", &AudioSettings::sdl_mic_device),
            make_override<AudioSettings>("sdl_main_output_device",
                                         &AudioSettings::sdl_main_output_device),
            make_override<AudioSettings>("sdl_padSpk_output_device",
                                         &AudioSettings::sdl_padSpk_output_device),
            make_override<AudioSettings>("openal_mic_device", &AudioSettings::openal_mic_device),
            make_override<AudioSettings>("openal_main_output_device",
                                         &AudioSettings::openal_main_output_device),
            make_override<AudioSettings>("openal_padSpk_output_device",
                                         &AudioSettings::openal_padSpk_output_device),
            make_override<AudioSettings>("openal_hrtf", &AudioSettings::openal_hrtf),
            make_override<AudioSettings>("openal_output_mode", &AudioSettings::openal_output_mode)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AudioSettings, audio_backend, sdl_mic_device,
                                   sdl_main_output_device, sdl_padSpk_output_device,
                                   openal_mic_device, openal_main_output_device,
                                   openal_padSpk_output_device, openal_hrtf, openal_output_mode)

// -------------------------------
// GPU settings
// -------------------------------
struct GPUSettings {
    Setting<u32> window_width{1280};
    Setting<u32> window_height{720};
    Setting<u32> internal_screen_width{1280};
    Setting<u32> internal_screen_height{720};
    Setting<bool> null_gpu{false};
    Setting<bool> copy_gpu_buffers{false};
    Setting<u32> readbacks_mode{GpuReadbacksMode::Disabled};
    Setting<bool> readback_linear_images_enabled{false};
    Setting<bool> direct_memory_access_enabled{false};
    // ---- TLG readback / command processor tuning (documents/tlg/settings-reference.md) ----
    // Completes image readbacks and recently read back buffers with one GPU wait per fence.
    Setting<bool> readback_batching_enabled{false};
    // Signals end-of-pipe fences after their readbacks complete on another thread instead of
    // blocking the GPU command thread.
    Setting<bool> readback_async_fences_enabled{false};
    // With async fences, lets command stream waits pass for fence values still held back:
    // 0 = off, 1 = graphics queue only, 2 = graphics and compute queues, 3 = like 2 plus a GPU
    // barrier after each wait that passed on a fence recorded but not finished yet.
    Setting<u32> readback_fence_wait_shortcut{1};
    // Small read-only buffers copied again in the same command buffer reuse the earlier copy.
    Setting<bool> readback_stream_reuse_enabled{false};
    // Buffer ranges the CPU faulted on are downloaded in advance at fences where the GPU wrote them
    // again. 0 = never, 1 = ranges that faulted on a read or a write, 2 = only on a read.
    Setting<u32> readback_prefetch_mode{1};
    // Fences a prefetched range stays hot after its last CPU fault before it is dropped.
    Setting<u32> readback_prefetch_lifetime{600};
    // A draw whose render target registers match the previous draw reuses its target images.
    Setting<bool> rt_lookup_reuse_enabled{false};
    // Shader stage lookups first check the variant picked last time: 0 = off, 1 = on, 2 = verify.
    Setting<u32> stage_lookup_reuse{0};
    // Guest threads that fault on GPU written memory wait for their own readback; the command
    // processor records and submits the download and goes on.
    Setting<bool> readback_async_guest_faults{false};
    // Linear images that are read back are copied into private staging buffers and written to guest
    // memory by the priority thread once the GPU finished them (no fence waits for them).
    Setting<bool> readback_linear_images_async{false};
    // DMA copies into guest memory (the game's own GPU readbacks) are also copied into a private
    // staging buffer that the priority thread writes to guest memory once the GPU finished it.
    Setting<bool> readback_dma_async{false};
    // File with the 512 KB windows of guest memory the game reads while GPU data is in flight
    // (one "start end" pair per line, hex). Windows listed at start are prefetched for good, and new
    // ones learned from read faults are added to the file. Empty = off.
    Setting<std::string> readback_hot_file{""};
    // GPU command processor recording: 0 = record Vulkan commands directly, 1 = record them into a
    // list replayed at submit (same thread), 2 = replay and submit on an encoder thread,
    // 3 = like 2 but waiting for each submission (diagnostic).
    Setting<u32> cp_encoder_mode{0};
    // Encoder thread diagnostics (bit mask): 1 = wait for the encoder at every fence, 2 = at the end
    // of every guest submission, 4 = fences without readbacks wait for all earlier work.
    Setting<u32> cp_encoder_debug{0};
    // Priority of the command processor and encoder threads: 0 or 1 = default, 2 = above normal,
    // 3 = highest.
    Setting<u32> cp_thread_priority{0};
    // Encoder thread: how many recorded command lists the command processor may hand over before it
    // waits for the encoder (encoder mode 2).
    Setting<u32> cp_encoder_queue_depth{8};
    // Fence kinds that wait for all earlier work (bit mask, see VideoCore::FenceKind): 1 = graphics
    // EOP, 2 = graphics EOS, 4 = compute ReleaseMem, 8 = label writes held behind a fence, 16 = end
    // of a guest submission. 0 makes Trico's feathers flicker in The Last Guardian.
    Setting<u32> cp_full_fence_kinds{31};
    // Command processor split: 0 = one thread does everything, 1 = a front thread parses packets and a
    // raster thread executes draws, dispatches, fences and copies in order, 2 = like 1 but the front
    // thread waits after every operation and compares the register mirror (diagnostic).
    Setting<u32> cp_split_mode{0};
    // With the encoder thread: hand the recorded Vulkan commands to it every N commands instead of
    // only at fences and flips. 0 = off.
    Setting<u32> cp_flush_commands{0};
    // Windows: pin the raster / front / encoder thread to this logical processor and keep other
    // threads of the process off it and off its hyper-threading sibling. -1 = off.
    Setting<s32> cp_raster_core{-1};
    Setting<s32> cp_front_core{-1};
    Setting<s32> cp_encoder_core{-1};
    // Read-only development server on 127.0.0.1 (core/debug_server.h): live counters and policy
    // switches as JSON. 0 = off.
    Setting<u32> debug_server_port{0};
    // Windows: patch every thread-control-block access (mov reg, fs:[0] and similar) of a module when it
    // is loaded, before the guest runs, instead of the first time a thread executes it.
    Setting<bool> cpu_prepatch_tcb{false};
    // Native draw path (documents/tlg/Legacy/native-render-plan.md): 0 off, 1 the game writes a tag packet in front of
    // every draw of the enabled classes (no effect on rendering), 2 the rasterizer also uses the tag to skip
    // redundant per-draw work.
    Setting<u32> native_draw_mode{0};
    // Pass classes the native draw path covers: 1 z-prepass, 2 G-buffer, 4 forward, 8 everything else.
    Setting<u32> native_draw_classes{3};
    // Shadow mode of the native front end: the game's draw hook builds the register state of every tagged draw from the
    // engine's objects, the rasterizer compares it with the registers the command packets produced and logs the
    // differences. Changes nothing in the rendering.
    Setting<bool> native_synth_verify{false};
    // Native draw execution: tagged draws whose state the native front end could build are not written as command
    // packets; the command processor draws them from the record (every 64th draw still takes the packet path and is
    // compared with the record). Needs native_draw_mode and native_synth_verify.
    Setting<bool> native_draw_exec{false};
    // With native execution: every Nth draw / dispatch of a call site still takes the game's packet path and is compared
    // with its record (the safety net that notices a state the builder does not know). 0 = never.
    Setting<u32> native_audit_every{64};
    Setting<u32> native_map{0};
    Setting<u32> native_packet_path{1};
    // With native execution: the packets of SDK helper calls the command processor has no use for are replaced by one NOP
    // (bit 0: depth transition quads of the render target tracking, bit 1: resource tracking, bit 2: copies).
    Setting<u32> native_sdk_strip{0};
    Setting<bool> dump_shaders{false};
    Setting<bool> patch_shaders{false};
    Setting<u32> vblank_frequency{60};
    Setting<bool> full_screen{false};
    Setting<std::string> full_screen_mode{"Windowed"};
    Setting<std::string> present_mode{"Mailbox"};
    Setting<bool> hdr_allowed{false};
    Setting<bool> fsr_enabled{false};
    Setting<bool> rcas_enabled{true};
    Setting<int> rcas_attenuation{250};
    Setting<bool> userfaultfd{false};
    Setting<bool> inline_fetch_shader{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<GPUSettings>("null_gpu", &GPUSettings::null_gpu),
            make_override<GPUSettings>("copy_gpu_buffers", &GPUSettings::copy_gpu_buffers),
            make_override<GPUSettings>("full_screen", &GPUSettings::full_screen),
            make_override<GPUSettings>("full_screen_mode", &GPUSettings::full_screen_mode),
            make_override<GPUSettings>("present_mode", &GPUSettings::present_mode),
            make_override<GPUSettings>("window_height", &GPUSettings::window_height),
            make_override<GPUSettings>("window_width", &GPUSettings::window_width),
            make_override<GPUSettings>("hdr_allowed", &GPUSettings::hdr_allowed),
            make_override<GPUSettings>("fsr_enabled", &GPUSettings::fsr_enabled),
            make_override<GPUSettings>("rcas_enabled", &GPUSettings::rcas_enabled),
            make_override<GPUSettings>("rcas_attenuation", &GPUSettings::rcas_attenuation),
            make_override<GPUSettings>("dump_shaders", &GPUSettings::dump_shaders),
            make_override<GPUSettings>("patch_shaders", &GPUSettings::patch_shaders),
            make_override<GPUSettings>("readbacks_mode", &GPUSettings::readbacks_mode),
            make_override<GPUSettings>("readback_linear_images_enabled",
                                       &GPUSettings::readback_linear_images_enabled),
            make_override<GPUSettings>("direct_memory_access_enabled",
                                       &GPUSettings::direct_memory_access_enabled),
            make_override<GPUSettings>("readback_batching_enabled", &GPUSettings::readback_batching_enabled),
            make_override<GPUSettings>("readback_async_fences_enabled", &GPUSettings::readback_async_fences_enabled),
            make_override<GPUSettings>("readback_fence_wait_shortcut", &GPUSettings::readback_fence_wait_shortcut),
            make_override<GPUSettings>("readback_stream_reuse_enabled", &GPUSettings::readback_stream_reuse_enabled),
            make_override<GPUSettings>("readback_prefetch_mode", &GPUSettings::readback_prefetch_mode),
            make_override<GPUSettings>("readback_prefetch_lifetime", &GPUSettings::readback_prefetch_lifetime),
            make_override<GPUSettings>("rt_lookup_reuse_enabled", &GPUSettings::rt_lookup_reuse_enabled),
            make_override<GPUSettings>("stage_lookup_reuse", &GPUSettings::stage_lookup_reuse),
            make_override<GPUSettings>("readback_async_guest_faults", &GPUSettings::readback_async_guest_faults),
            make_override<GPUSettings>("readback_linear_images_async", &GPUSettings::readback_linear_images_async),
            make_override<GPUSettings>("readback_dma_async", &GPUSettings::readback_dma_async),
            make_override<GPUSettings>("readback_hot_file", &GPUSettings::readback_hot_file),
            make_override<GPUSettings>("cp_encoder_mode", &GPUSettings::cp_encoder_mode),
            make_override<GPUSettings>("cp_encoder_debug", &GPUSettings::cp_encoder_debug),
            make_override<GPUSettings>("cp_thread_priority", &GPUSettings::cp_thread_priority),
            make_override<GPUSettings>("cp_encoder_queue_depth", &GPUSettings::cp_encoder_queue_depth),
            make_override<GPUSettings>("cp_full_fence_kinds", &GPUSettings::cp_full_fence_kinds),
            make_override<GPUSettings>("cp_split_mode", &GPUSettings::cp_split_mode),
            make_override<GPUSettings>("cp_flush_commands", &GPUSettings::cp_flush_commands),
            make_override<GPUSettings>("cp_raster_core", &GPUSettings::cp_raster_core),
            make_override<GPUSettings>("cp_front_core", &GPUSettings::cp_front_core),
            make_override<GPUSettings>("cp_encoder_core", &GPUSettings::cp_encoder_core),
            make_override<GPUSettings>("debug_server_port", &GPUSettings::debug_server_port),
            make_override<GPUSettings>("cpu_prepatch_tcb", &GPUSettings::cpu_prepatch_tcb),
            make_override<GPUSettings>("native_draw_mode", &GPUSettings::native_draw_mode),
            make_override<GPUSettings>("native_draw_classes", &GPUSettings::native_draw_classes),
            make_override<GPUSettings>("native_synth_verify", &GPUSettings::native_synth_verify),
            make_override<GPUSettings>("native_draw_exec", &GPUSettings::native_draw_exec),
            make_override<GPUSettings>("native_audit_every", &GPUSettings::native_audit_every),
            make_override<GPUSettings>("native_map", &GPUSettings::native_map),
            make_override<GPUSettings>("native_packet_path", &GPUSettings::native_packet_path),
            make_override<GPUSettings>("native_sdk_strip", &GPUSettings::native_sdk_strip),
            make_override<GPUSettings>("vblank_frequency", &GPUSettings::vblank_frequency),
            make_override<GPUSettings>("userfaultfd", &GPUSettings::userfaultfd),
            make_override<GPUSettings>("inline_fetch_shader", &GPUSettings::inline_fetch_shader),
        };
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GPUSettings, window_width, window_height, internal_screen_width,
                                   internal_screen_height, null_gpu, copy_gpu_buffers,
                                   readbacks_mode, readback_linear_images_enabled,
                                   direct_memory_access_enabled, readback_batching_enabled, readback_async_fences_enabled, readback_fence_wait_shortcut, readback_stream_reuse_enabled, readback_prefetch_mode, readback_prefetch_lifetime, rt_lookup_reuse_enabled, stage_lookup_reuse, readback_async_guest_faults, readback_linear_images_async, readback_dma_async, readback_hot_file, cp_encoder_mode, cp_encoder_debug, cp_thread_priority, cp_encoder_queue_depth, cp_full_fence_kinds, cp_split_mode, cp_flush_commands, cp_raster_core, cp_front_core, cp_encoder_core, debug_server_port, cpu_prepatch_tcb, native_draw_mode, native_draw_classes, native_synth_verify, native_draw_exec, native_audit_every, native_map, native_packet_path,
                                   dump_shaders, patch_shaders,
                                   vblank_frequency, full_screen, full_screen_mode, present_mode,
                                   hdr_allowed, fsr_enabled, rcas_enabled, rcas_attenuation,
                                   userfaultfd, inline_fetch_shader)

// -------------------------------
// Vulkan settings
// -------------------------------
struct VulkanSettings {
    Setting<s32> gpu_id{-1};
    Setting<bool> renderdoc_enabled{false};
    Setting<bool> vkvalidation_enabled{false};
    Setting<bool> vkvalidation_core_enabled{true};
    Setting<bool> vkvalidation_sync_enabled{false};
    Setting<bool> vkvalidation_gpu_enabled{false};
    Setting<bool> vkcrash_diagnostic_enabled{false};
    Setting<bool> vkhost_markers{false};
    Setting<bool> vkguest_markers{false};
    Setting<bool> pipeline_cache_enabled{false};
    Setting<bool> pipeline_cache_archived{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<VulkanSettings>("gpu_id", &VulkanSettings::gpu_id),
            make_override<VulkanSettings>("renderdoc_enabled", &VulkanSettings::renderdoc_enabled),
            make_override<VulkanSettings>("vkvalidation_enabled",
                                          &VulkanSettings::vkvalidation_enabled),
            make_override<VulkanSettings>("vkvalidation_core_enabled",
                                          &VulkanSettings::vkvalidation_core_enabled),
            make_override<VulkanSettings>("vkvalidation_sync_enabled",
                                          &VulkanSettings::vkvalidation_sync_enabled),
            make_override<VulkanSettings>("vkvalidation_gpu_enabled",
                                          &VulkanSettings::vkvalidation_gpu_enabled),
            make_override<VulkanSettings>("vkcrash_diagnostic_enabled",
                                          &VulkanSettings::vkcrash_diagnostic_enabled),
            make_override<VulkanSettings>("vkhost_markers", &VulkanSettings::vkhost_markers),
            make_override<VulkanSettings>("vkguest_markers", &VulkanSettings::vkguest_markers),
            make_override<VulkanSettings>("pipeline_cache_enabled",
                                          &VulkanSettings::pipeline_cache_enabled),
            make_override<VulkanSettings>("pipeline_cache_archived",
                                          &VulkanSettings::pipeline_cache_archived),
        };
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VulkanSettings, gpu_id, renderdoc_enabled, vkvalidation_enabled,
                                   vkvalidation_core_enabled, vkvalidation_sync_enabled,
                                   vkvalidation_gpu_enabled, vkcrash_diagnostic_enabled,
                                   vkhost_markers, vkguest_markers, pipeline_cache_enabled,
                                   pipeline_cache_archived)

// -------------------------------
// Main manager
// -------------------------------
class EmulatorSettingsImpl {
public:
    EmulatorSettingsImpl();
    ~EmulatorSettingsImpl();

    static std::shared_ptr<EmulatorSettingsImpl> GetInstance();
    static void SetInstance(std::shared_ptr<EmulatorSettingsImpl> instance);

    bool Save(const std::string& serial = "");
    bool Load(const std::string& serial = "");
    void SetDefaultValues();
    bool TransferSettings();

    // Config mode
    ConfigMode GetConfigMode() const {
        return m_configMode;
    }
    void SetConfigMode(ConfigMode mode) {
        m_configMode = mode;
    }

    //
    // Game-specific override management
    /// Clears all per-game overrides.  Call this when a game exits so
    /// the emulator reverts to global settings.
    void ClearGameSpecificOverrides();

    // general accessors
    bool AddGameInstallDir(const std::filesystem::path& dir, bool enabled = true);
    std::vector<std::filesystem::path> GetGameInstallDirs() const;
    void SetAllGameInstallDirs(const std::vector<GameInstallDir>& dirs);
    void RemoveGameInstallDir(const std::filesystem::path& dir);
    void SetGameInstallDirEnabled(const std::filesystem::path& dir, bool enabled);
    void SetGameInstallDirs(const std::vector<std::filesystem::path>& dirs_config);
    const std::vector<bool> GetGameInstallDirsEnabled();
    const std::vector<GameInstallDir>& GetAllGameInstallDirs() const;

    std::filesystem::path GetHomeDir();
    void SetHomeDir(const std::filesystem::path& dir);
    std::filesystem::path GetSysModulesDir();
    void SetSysModulesDir(const std::filesystem::path& dir);
    std::filesystem::path GetFontsDir();
    void SetFontsDir(const std::filesystem::path& dir);
    std::filesystem::path GetAddonInstallDir();
    void SetAddonInstallDir(const std::filesystem::path& dir);

private:
    GeneralSettings m_general{};
    NetworkSettings m_network{};
    LogSettings m_log{};
    DebugSettings m_debug{};
    InputSettings m_input{};
    AudioSettings m_audio{};
    GPUSettings m_gpu{};
    VulkanSettings m_vulkan{};
    ConfigMode m_configMode{ConfigMode::Default};

    // Runtime-only override: when true, IsShadNetEnabled() reports false for the
    // rest of this run regardless of the persisted setting
    std::atomic<bool> m_shadnet_session_disabled{false};

    static std::shared_ptr<EmulatorSettingsImpl> s_instance;
    static std::mutex s_mutex;

    /// Apply overrideable fields from groupJson into group.game_specific_value.
    template <typename Group>
    void ApplyGroupOverrides(Group& group, const nlohmann::json& groupJson,
                             std::vector<std::string>& changed) {
        for (auto& item : group.GetOverrideableFields()) {
            if (!groupJson.contains(item.key))
                continue;
            item.apply(&group, groupJson.at(item.key), changed);
        }
    }

    // Write all overrideable fields from group into out (for game-specific save).
    template <typename Group>
    static void SaveGroupGameSpecific(const Group& group, nlohmann::json& out) {
        for (auto& item : group.GetOverrideableFields())
            out[item.key] = item.get_for_save(&group);
    }

    // Discard every game-specific override in group.
    template <typename Group>
    static void ClearGroupOverrides(Group& group) {
        for (auto& item : group.GetOverrideableFields())
            item.reset_game_specific(&group);
    }

    static void PrintChangedSummary(const std::vector<std::string>& changed);
    void MigrateNetworkKeys(const nlohmann::json& general, const nlohmann::json& network);

public:
    // Used by the launcher to back up / restore settings (e.g. per-game settings dialog)
    EmulatorSettingsImpl& operator=(const EmulatorSettingsImpl& other) {
        if (this != &other) {
            m_shadnet_session_disabled.store(other.m_shadnet_session_disabled.load());
            m_general = other.m_general;
            m_network = other.m_network;
            m_log = other.m_log;
            m_debug = other.m_debug;
            m_input = other.m_input;
            m_audio = other.m_audio;
            m_gpu = other.m_gpu;
            m_vulkan = other.m_vulkan;
            m_configMode = other.m_configMode;
        }
        return *this;
    }

    // Add these getters to access overrideable fields
    std::vector<OverrideItem> GetGeneralOverrideableFields() const {
        return m_general.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetNetworkOverrideableFields() const {
        return m_network.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetDebugOverrideableFields() const {
        return m_debug.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetInputOverrideableFields() const {
        return m_input.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetAudioOverrideableFields() const {
        return m_audio.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetGPUOverrideableFields() const {
        return m_gpu.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetVulkanOverrideableFields() const {
        return m_vulkan.GetOverrideableFields();
    }
    std::vector<std::string> GetAllOverrideableKeys() const;

#define SETTING_FORWARD(group, Name, field)                                                        \
    auto Get##Name() const {                                                                       \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(const decltype((group).field.value)& v, bool specific = false) {                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL(group, Name, field)                                                   \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(bool v, bool specific = false) {                                                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL_READONLY(group, Name, field)                                          \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }

    // General settings
    SETTING_FORWARD(m_general, VolumeSlider, volume_slider)
    SETTING_FORWARD_BOOL(m_general, Neo, neo_mode)
    SETTING_FORWARD_BOOL(m_general, DevKit, dev_kit_mode)
    SETTING_FORWARD(m_general, ExtraDmemInMBytes, extra_dmem_in_mbytes)
    SETTING_FORWARD(m_general, ExtraFmemInMBytes, extra_fmem_in_mbytes)
    bool IsShadNetEnabled() const {
        return m_network.shad_net_enabled.get(m_configMode) &&
               !m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    void SetShadNetEnabled(bool v, bool specific = false) {
        m_network.shad_net_enabled.set(v, specific);
    }
    bool IsShadNetEnabledSetting() const {
        return m_network.shad_net_enabled.get(m_configMode);
    }
    void SetShadNetSessionDisabled(bool v) {
        m_shadnet_session_disabled.store(v, std::memory_order_relaxed);
    }
    bool IsShadNetSessionDisabled() const {
        return m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    SETTING_FORWARD_BOOL(m_general, TrophyPopupDisabled, trophy_popup_disabled)
    SETTING_FORWARD(m_general, TrophyNotificationDuration, trophy_notification_duration)
    SETTING_FORWARD(m_general, TrophyNotificationSide, trophy_notification_side)
    SETTING_FORWARD_BOOL(m_general, ShowSplash, show_splash)
    SETTING_FORWARD_BOOL(m_general, DiscordRPCEnabled, discord_rpc_enabled)
    SETTING_FORWARD_BOOL(m_general, ShowFpsCounter, show_fps_counter)
    SETTING_FORWARD(m_general, ConsoleLanguage, console_language)
    SETTING_FORWARD(m_general, BigPictureScale, big_picture_scale)
    SETTING_FORWARD(m_general, BigPictureFolderDepth, big_picture_folder_depth)
    SETTING_FORWARD_BOOL(m_general, RedZonePatchingEnabled, redzone_patches)

    // Network settings
    SETTING_FORWARD_BOOL(m_network, ConnectedToNetwork, connected_to_network)
    SETTING_FORWARD(m_network, ShadNetServer, shadnet_server)
    SETTING_FORWARD(m_network, ShadNetWebApiServer, shadnet_webapi_server)
    SETTING_FORWARD_BOOL(m_network, UPnPEnabled, enable_upnp)
    SETTING_FORWARD(m_network, P2PPort, p2p_port)
    SETTING_FORWARD_BOOL(m_network, ForcedHttpsDisabled, disable_https)

    // Log settings
    SETTING_FORWARD_BOOL(m_log, LogAppend, append)
    SETTING_FORWARD_BOOL(m_log, LogEnable, enable)
    SETTING_FORWARD(m_log, LogFilter, filter)
    SETTING_FORWARD(m_log, LogFlushLevel, flush_level)
    SETTING_FORWARD(m_log, LogMaxSkipDuration, max_skip_duration)
    SETTING_FORWARD_BOOL(m_log, LogSeparate, separate)
    SETTING_FORWARD(m_log, LogSizeLimit, size_limit)
    SETTING_FORWARD_BOOL(m_log, LogSkipDuplicate, skip_duplicate)
    SETTING_FORWARD_BOOL(m_log, LogSync, sync)
#ifdef _WIN32
    SETTING_FORWARD(m_log, LogType, type)
#endif

    // Audio settings
    SETTING_FORWARD(m_audio, AudioBackend, audio_backend)
    SETTING_FORWARD(m_audio, SDLMicDevice, sdl_mic_device)
    SETTING_FORWARD(m_audio, SDLMainOutputDevice, sdl_main_output_device)
    SETTING_FORWARD(m_audio, SDLPadSpkOutputDevice, sdl_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALMicDevice, openal_mic_device)
    SETTING_FORWARD(m_audio, OpenALMainOutputDevice, openal_main_output_device)
    SETTING_FORWARD(m_audio, OpenALPadSpkOutputDevice, openal_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALHrtf, openal_hrtf)
    SETTING_FORWARD(m_audio, OpenALOutputMode, openal_output_mode)

    // Debug settings
    SETTING_FORWARD_BOOL(m_debug, DebugDump, debug_dump)
    SETTING_FORWARD_BOOL(m_debug, ShaderCollect, shader_collect)
    SETTING_FORWARD(m_debug, ConfigVersion, config_version)

    // GPU Settings
    SETTING_FORWARD_BOOL(m_gpu, NullGPU, null_gpu)
    SETTING_FORWARD_BOOL(m_gpu, DumpShaders, dump_shaders)
    SETTING_FORWARD_BOOL(m_gpu, CopyGpuBuffers, copy_gpu_buffers)
    SETTING_FORWARD_BOOL(m_gpu, FullScreen, full_screen)
    SETTING_FORWARD(m_gpu, FullScreenMode, full_screen_mode)
    SETTING_FORWARD(m_gpu, PresentMode, present_mode)
    SETTING_FORWARD(m_gpu, WindowHeight, window_height)
    SETTING_FORWARD(m_gpu, WindowWidth, window_width)
    SETTING_FORWARD(m_gpu, InternalScreenHeight, internal_screen_height)
    SETTING_FORWARD(m_gpu, InternalScreenWidth, internal_screen_width)
    SETTING_FORWARD_BOOL(m_gpu, HdrAllowed, hdr_allowed)
    SETTING_FORWARD_BOOL(m_gpu, FsrEnabled, fsr_enabled)
    SETTING_FORWARD_BOOL(m_gpu, RcasEnabled, rcas_enabled)
    SETTING_FORWARD(m_gpu, RcasAttenuation, rcas_attenuation)
    SETTING_FORWARD(m_gpu, ReadbacksMode, readbacks_mode)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackLinearImagesEnabled, readback_linear_images_enabled)
    SETTING_FORWARD_BOOL(m_gpu, DirectMemoryAccessEnabled, direct_memory_access_enabled)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackBatchingEnabled, readback_batching_enabled)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackAsyncFencesEnabled, readback_async_fences_enabled)
    SETTING_FORWARD(m_gpu, ReadbackFenceWaitShortcut, readback_fence_wait_shortcut)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackStreamReuseEnabled, readback_stream_reuse_enabled)
    SETTING_FORWARD(m_gpu, ReadbackPrefetchMode, readback_prefetch_mode)
    SETTING_FORWARD(m_gpu, ReadbackPrefetchLifetime, readback_prefetch_lifetime)
    SETTING_FORWARD_BOOL(m_gpu, RtLookupReuseEnabled, rt_lookup_reuse_enabled)
    SETTING_FORWARD(m_gpu, StageLookupReuse, stage_lookup_reuse)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackAsyncGuestFaultsEnabled, readback_async_guest_faults)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackLinearImagesAsync, readback_linear_images_async)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackDmaAsync, readback_dma_async)
    SETTING_FORWARD(m_gpu, ReadbackHotFile, readback_hot_file)
    SETTING_FORWARD(m_gpu, CpEncoderMode, cp_encoder_mode)
    SETTING_FORWARD(m_gpu, CpEncoderDebug, cp_encoder_debug)
    SETTING_FORWARD(m_gpu, CpThreadPriority, cp_thread_priority)
    SETTING_FORWARD(m_gpu, NativeDrawMode, native_draw_mode)
    SETTING_FORWARD(m_gpu, NativeDrawClasses, native_draw_classes)
    SETTING_FORWARD_BOOL(m_gpu, NativeSynthVerify, native_synth_verify)
    SETTING_FORWARD_BOOL(m_gpu, NativeDrawExec, native_draw_exec)
    SETTING_FORWARD(m_gpu, NativeAuditEvery, native_audit_every)
    SETTING_FORWARD(m_gpu, NativeMap, native_map)
    SETTING_FORWARD(m_gpu, NativePacketPath, native_packet_path)
    SETTING_FORWARD(m_gpu, NativeSdkStrip, native_sdk_strip)
    SETTING_FORWARD(m_gpu, CpEncoderQueueDepth, cp_encoder_queue_depth)
    SETTING_FORWARD(m_gpu, CpFullFenceKinds, cp_full_fence_kinds)
    SETTING_FORWARD(m_gpu, CpSplitMode, cp_split_mode)
    SETTING_FORWARD(m_gpu, CpFlushCommands, cp_flush_commands)
    SETTING_FORWARD(m_gpu, CpRasterCore, cp_raster_core)
    SETTING_FORWARD(m_gpu, CpFrontCore, cp_front_core)
    SETTING_FORWARD(m_gpu, CpEncoderCore, cp_encoder_core)
    SETTING_FORWARD(m_gpu, DebugServerPort, debug_server_port)
    SETTING_FORWARD_BOOL(m_gpu, CpuPrepatchTcb, cpu_prepatch_tcb)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, PatchShaders, patch_shaders)
    SETTING_FORWARD_BOOL(m_gpu, UserfaultfdTracking, userfaultfd)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, InlineFetchShader, inline_fetch_shader)

    u32 GetVblankFrequency() {
        if (m_gpu.vblank_frequency.value < 30) {
            return 30;
        }
        return m_gpu.vblank_frequency.get();
    }
    void SetVblankFrequency(const u32& v, bool is_specific = false) {
        u32 val = v < 30 ? 30 : v;
        if (is_specific) {
            m_gpu.vblank_frequency.game_specific_value = val;
        } else {
            m_gpu.vblank_frequency.value = val;
        }
    }

    // Input Settings
    SETTING_FORWARD(m_input, CursorState, cursor_state)
    SETTING_FORWARD(m_input, CursorHideTimeout, cursor_hide_timeout)
    SETTING_FORWARD(m_input, UsbDeviceBackend, usb_device_backend)
    SETTING_FORWARD_BOOL(m_input, MotionControlsEnabled, motion_controls_enabled)
    SETTING_FORWARD_BOOL(m_input, BackgroundControllerInput, background_controller_input)
    SETTING_FORWARD_BOOL(m_input, ImeAccessibilityEnabled, ime_accessibility_enabled)
    SETTING_FORWARD_BOOL(m_input, ImeUrlMailShortPanel, ime_url_mail_short_panel)
    SETTING_FORWARD(m_input, DefaultControllerId, default_controller_id)
    SETTING_FORWARD_BOOL(m_input, UsingSpecialPad, use_special_pad)
    SETTING_FORWARD(m_input, SpecialPadClass, special_pad_class)
    SETTING_FORWARD_BOOL(m_input, UseUnifiedInputConfig, use_unified_input_config)
    SETTING_FORWARD(m_input, CameraId, camera_id)
    SETTING_FORWARD_BOOL(m_input, CircleEnter, is_circle_enter)
    SETTING_FORWARD_BOOL(m_input, MiceUsedAsMice, use_mice_as_mice)
    SETTING_FORWARD_BOOL(m_input, KeyboardUsedAsKeyboard, use_keyboard_as_keyboard)

    // Vulkan settings
    SETTING_FORWARD(m_vulkan, GpuId, gpu_id)
    SETTING_FORWARD_BOOL(m_vulkan, RenderdocEnabled, renderdoc_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationEnabled, vkvalidation_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationCoreEnabled, vkvalidation_core_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationSyncEnabled, vkvalidation_sync_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationGpuEnabled, vkvalidation_gpu_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkCrashDiagnosticEnabled, vkcrash_diagnostic_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkHostMarkersEnabled, vkhost_markers)
    SETTING_FORWARD_BOOL(m_vulkan, VkGuestMarkersEnabled, vkguest_markers)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheEnabled, pipeline_cache_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheArchived, pipeline_cache_archived)

#undef SETTING_FORWARD
#undef SETTING_FORWARD_BOOL
#undef SETTING_FORWARD_BOOL_READONLY
};
