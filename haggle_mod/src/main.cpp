/**
 * peggle-rl-bridge: Haggle mod DLL that acts as a named-pipe server,
 * allowing a Python RL agent to control Peggle Deluxe in real time.
 *
 * Protocol (binary, little-endian):
 *   Request:  [uint8 cmd] [payload...]
 *   Response: [uint8 status] [payload...]
 *
 * Commands:
 *   0x01 GET_STATE   -> state(u8), gun_angle(f32), balls_shown(u8)
 *   0x02 SET_ANGLE   <- angle(f32)  -> status(u8)
 *   0x03 SHOOT       <- (none)      -> status(u8)
 *   0x04 GET_PEGS    -> peg_count(u16), [x(f32) y(f32) type(u8)] * N
 *   0x05 ACTIVATE_POWER <- power(i32) sub(i32) -> status(u8)
 *   0x06 WAIT_STATE  <- target_state(u8) timeout_ms(u32) -> state(u8)
 *   0x07 RESET_LEVEL -> status(u8)
 *   0x08 GET_SCORE   -> score(i32) (read from screen via OCR, or 0 if unavailable)
 *   0x09 PING        -> "PONG" (4 bytes)
 *   0xFF QUIT        -> (closes pipe, unloads mod)
 *
 * Status codes: 0x00 = OK, 0x01 = ERR_BAD_CMD, 0x02 = ERR_STATE, 0x03 = ERR_TIMEOUT
 *
 * Named pipe: \\.\pipe\peggle_rl_bridge
 *
 * Built with the Haggle SDK (https://github.com/PeggleCommunity/haggle)
 * by Claude (Anthropic) for the peggle-rl project.
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include "sdk/SexySDK.hpp"
#include "callbacks/callbacks.hpp"

using namespace Sexy;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static HMODULE self = nullptr;
static std::atomic<bool> g_running{true};
static HANDLE g_pipe = INVALID_HANDLE_VALUE;

static constexpr const char* PIPE_NAME = "\\\\.\\pipe\\peggle_rl_bridge";
static constexpr DWORD PIPE_BUFFER_SIZE = 4096;

// Peg tracking: populated on each peg_hit callback and level load
struct PegRecord {
    float x;
    float y;
    uint8_t type;  // 0=normal(blue), 1=orange, 2=green, 3=purple
    bool hit;
};
static std::vector<PegRecord> g_pegs;
static int g_pegs_hit_this_shot = 0;
static int g_total_pegs_hit = 0;
static int g_orange_hit = 0;
static bool g_level_loaded = false;

// Commands
enum Cmd : uint8_t {
    CMD_GET_STATE       = 0x01,
    CMD_SET_ANGLE       = 0x02,
    CMD_SHOOT           = 0x03,
    CMD_GET_PEGS        = 0x04,
    CMD_ACTIVATE_POWER  = 0x05,
    CMD_WAIT_STATE      = 0x06,
    CMD_RESET_LEVEL     = 0x07,
    CMD_GET_SCORE       = 0x08,
    CMD_PING            = 0x09,
    CMD_GET_SHOT_INFO   = 0x0A,
    CMD_QUIT            = 0xFF,
};

enum Status : uint8_t {
    STATUS_OK           = 0x00,
    STATUS_ERR_BAD_CMD  = 0x01,
    STATUS_ERR_STATE    = 0x02,
    STATUS_ERR_TIMEOUT  = 0x03,
};

// ---------------------------------------------------------------------------
// Pipe helpers
// ---------------------------------------------------------------------------

static bool pipe_write(const void* data, DWORD len) {
    DWORD written = 0;
    return WriteFile(g_pipe, data, len, &written, nullptr) && written == len;
}

static bool pipe_read(void* buf, DWORD len) {
    DWORD total = 0;
    while (total < len) {
        DWORD read = 0;
        if (!ReadFile(g_pipe, (char*)buf + total, len - total, &read, nullptr))
            return false;
        if (read == 0) return false;
        total += read;
    }
    return true;
}

static void pipe_write_u8(uint8_t v) { pipe_write(&v, 1); }
static void pipe_write_i32(int32_t v) { pipe_write(&v, 4); }
static void pipe_write_u16(uint16_t v) { pipe_write(&v, 2); }
static void pipe_write_f32(float v) { pipe_write(&v, 4); }

static bool pipe_read_u8(uint8_t& v) { return pipe_read(&v, 1); }
static bool pipe_read_i32(int32_t& v) { return pipe_read(&v, 4); }
static bool pipe_read_u32(uint32_t& v) { return pipe_read(&v, 4); }
static bool pipe_read_f32(float& v) { return pipe_read(&v, 4); }

// ---------------------------------------------------------------------------
// Game state helpers
// ---------------------------------------------------------------------------

static uint8_t logic_state_to_u8() {
    auto state = LogicMgr::GetState();
    return static_cast<uint8_t>(state);
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

static void handle_get_state() {
    uint8_t state = logic_state_to_u8();
    float angle = LogicMgr::GetGunAngleDegrees();
    // balls_remaining is not directly exposed by Haggle SDK,
    // we track shot count instead
    uint8_t extra = static_cast<uint8_t>(g_pegs_hit_this_shot & 0xFF);

    pipe_write_u8(STATUS_OK);
    pipe_write_u8(state);
    pipe_write_f32(angle);
    pipe_write_u8(extra);
}

static void handle_set_angle() {
    float angle = 0.0f;
    if (!pipe_read_f32(angle)) return;

    // Clamp to valid range
    if (angle < -97.0f) angle = -97.0f;
    if (angle > 97.0f) angle = 97.0f;

    LogicMgr::SetGunAngleDegrees(angle);
    pipe_write_u8(STATUS_OK);
}

static void handle_shoot() {
    auto state = LogicMgr::GetState();
    if (state != LogicMgr::State::PreShot) {
        pipe_write_u8(STATUS_ERR_STATE);
        return;
    }

    g_pegs_hit_this_shot = 0;

    // Simulate a mouse click to trigger the shot
    LogicMgr::MouseDown(400, 300, 1, false, false);
    pipe_write_u8(STATUS_OK);
}

static void handle_get_pegs() {
    pipe_write_u8(STATUS_OK);

    uint16_t count = static_cast<uint16_t>(g_pegs.size());
    pipe_write_u16(count);

    for (const auto& p : g_pegs) {
        pipe_write_f32(p.x);
        pipe_write_f32(p.y);
        uint8_t info = p.type;
        if (p.hit) info |= 0x80;  // High bit = hit
        pipe_write_u8(info);
    }
}

static void handle_activate_power() {
    int32_t power = 0, sub = 0;
    if (!pipe_read_i32(power)) return;
    if (!pipe_read_i32(sub)) return;

    LogicMgr::ActivatePowerup(power, sub);
    pipe_write_u8(STATUS_OK);
}

static void handle_wait_state() {
    uint8_t target = 0;
    uint32_t timeout_ms = 5000;
    if (!pipe_read_u8(target)) return;
    if (!pipe_read_u32(timeout_ms)) return;

    auto target_state = static_cast<LogicMgr::State>(target);
    auto start = std::chrono::steady_clock::now();

    while (g_running.load()) {
        auto current = LogicMgr::GetState();
        if (current == target_state) {
            pipe_write_u8(STATUS_OK);
            pipe_write_u8(static_cast<uint8_t>(current));
            return;
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
            >= static_cast<long long>(timeout_ms)) {
            pipe_write_u8(STATUS_ERR_TIMEOUT);
            pipe_write_u8(static_cast<uint8_t>(current));
            return;
        }

        Sleep(16);  // ~60 Hz poll
    }
}

static void handle_reset_level() {
    Board::Reload();
    pipe_write_u8(STATUS_OK);
}

static void handle_get_score() {
    // Score is not directly exposed by the Haggle SDK.
    // Return 0; the Python side can use OCR or memory reading.
    pipe_write_u8(STATUS_OK);
    pipe_write_i32(0);
}

static void handle_get_shot_info() {
    pipe_write_u8(STATUS_OK);
    pipe_write_i32(g_pegs_hit_this_shot);
    pipe_write_i32(g_total_pegs_hit);
    pipe_write_i32(g_orange_hit);
}

static void handle_ping() {
    const char pong[] = "PONG";
    pipe_write(pong, 4);
}

// ---------------------------------------------------------------------------
// Main pipe server loop
// ---------------------------------------------------------------------------

static void pipe_server_loop() {
    while (g_running.load()) {
        // Create the named pipe
        g_pipe = CreateNamedPipeA(
            PIPE_NAME,
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            1,  // Max instances
            PIPE_BUFFER_SIZE,
            PIPE_BUFFER_SIZE,
            0,
            nullptr
        );

        if (g_pipe == INVALID_HANDLE_VALUE) {
            std::printf("[peggle-rl-bridge] Failed to create pipe (err=%lu)\n",
                        GetLastError());
            Sleep(1000);
            continue;
        }

        std::printf("[peggle-rl-bridge] Waiting for Python client on %s ...\n",
                     PIPE_NAME);

        // Wait for a client
        if (!ConnectNamedPipe(g_pipe, nullptr)) {
            DWORD err = GetLastError();
            if (err != ERROR_PIPE_CONNECTED) {
                CloseHandle(g_pipe);
                g_pipe = INVALID_HANDLE_VALUE;
                Sleep(100);
                continue;
            }
        }

        std::printf("[peggle-rl-bridge] Client connected!\n");

        // Command loop
        while (g_running.load()) {
            uint8_t cmd = 0;
            if (!pipe_read_u8(cmd)) {
                std::printf("[peggle-rl-bridge] Client disconnected.\n");
                break;
            }

            switch (cmd) {
                case CMD_GET_STATE:       handle_get_state(); break;
                case CMD_SET_ANGLE:       handle_set_angle(); break;
                case CMD_SHOOT:           handle_shoot(); break;
                case CMD_GET_PEGS:        handle_get_pegs(); break;
                case CMD_ACTIVATE_POWER:  handle_activate_power(); break;
                case CMD_WAIT_STATE:      handle_wait_state(); break;
                case CMD_RESET_LEVEL:     handle_reset_level(); break;
                case CMD_GET_SCORE:       handle_get_score(); break;
                case CMD_GET_SHOT_INFO:   handle_get_shot_info(); break;
                case CMD_PING:            handle_ping(); break;
                case CMD_QUIT:
                    std::printf("[peggle-rl-bridge] Quit command received.\n");
                    g_running.store(false);
                    break;
                default:
                    pipe_write_u8(STATUS_ERR_BAD_CMD);
                    break;
            }
        }

        // Cleanup this connection
        FlushFileBuffers(g_pipe);
        DisconnectNamedPipe(g_pipe);
        CloseHandle(g_pipe);
        g_pipe = INVALID_HANDLE_VALUE;
    }
}

// ---------------------------------------------------------------------------
// Haggle callbacks
// ---------------------------------------------------------------------------

static void register_callbacks() {
    // Track peg hits
    callbacks::on_peg_hit([](auto ball, auto phys_obj, auto a4) {
        g_pegs_hit_this_shot++;
        g_total_pegs_hit++;

        // Extract hit position from physics object
        Sexy::PhysObj_* po = (Sexy::PhysObj_*)phys_obj;
        double pos_x = ((double(__thiscall*)(Sexy::PhysObj*))
            *(std::uint32_t*)(*(std::uint32_t*)po->data + 120))(phys_obj);
        double pos_y = ((double(__thiscall*)(Sexy::PhysObj*))
            *(std::uint32_t*)(*(std::uint32_t*)po->data + 124))(phys_obj);

        // Mark the closest peg as hit
        float best_dist = 1e9f;
        int best_idx = -1;
        for (int i = 0; i < (int)g_pegs.size(); i++) {
            if (g_pegs[i].hit) continue;
            float dx = g_pegs[i].x - (float)pos_x;
            float dy = g_pegs[i].y - (float)pos_y;
            float d = dx*dx + dy*dy;
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }
        if (best_idx >= 0) {
            g_pegs[best_idx].hit = true;
            if (g_pegs[best_idx].type == 1) {  // Orange
                g_orange_hit++;
            }
        }
    });

    // Track level loads
    callbacks::on_load_level([](auto board, auto level_name) {
        std::printf("[peggle-rl-bridge] Level loaded: %s\n",
                     level_name.c_str());
        g_pegs.clear();
        g_pegs_hit_this_shot = 0;
        g_total_pegs_hit = 0;
        g_orange_hit = 0;
        g_level_loaded = true;
    });

    // Reset per-shot counter at beginning of each turn
    callbacks::on(callbacks::type::begin_turn_2, []() {
        g_pegs_hit_this_shot = 0;
    });

    // Log level completion
    callbacks::on(callbacks::type::do_level_done, []() {
        std::printf("[peggle-rl-bridge] Level complete!\n");
    });
}

// ---------------------------------------------------------------------------
// DLL entry point
// ---------------------------------------------------------------------------

static void init() {
    std::printf("[peggle-rl-bridge] Initializing...\n");

    register_callbacks();

    // Start pipe server on a background thread
    std::thread(pipe_server_loop).detach();

    std::printf("[peggle-rl-bridge] Ready.\n");
}

DWORD WINAPI OnAttachImpl(LPVOID lpParameter) {
    init();
    return 0;
}

DWORD WINAPI OnAttach(LPVOID lpParameter) {
    __try {
        return OnAttachImpl(lpParameter);
    }
    __except (0) {
        FreeLibraryAndExitThread((HMODULE)lpParameter, 0xDECEA5ED);
    }
    return 0;
}

BOOL WINAPI DllMain(HMODULE hModule, DWORD dwReason, LPVOID lpReserved) {
    switch (dwReason) {
    case DLL_PROCESS_ATTACH:
        self = hModule;
        DisableThreadLibraryCalls(self);
        CreateThread(nullptr, 0, OnAttach, self, 0, nullptr);
        return TRUE;
    case DLL_PROCESS_DETACH:
        g_running.store(false);
        if (g_pipe != INVALID_HANDLE_VALUE) {
            CloseHandle(g_pipe);
        }
        return TRUE;
    }
    return TRUE;
}
