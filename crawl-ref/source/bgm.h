/**
 * @file
 * @brief Procedural, adaptive, pure-synthesis Roguelike Dungeon Synth BGM engine.
**/

#pragma once

#include <cstdint>

#if defined(USE_SOUND) && defined(USE_SDL)
# define BGM_BACKEND "Procedural Dungeon Synth BGM (SDL2)"
#else
# define BGM_BACKEND "Procedural Dungeon Synth BGM (headless)"
#endif

/// Initialise the BGM synth and open the audio output device (safe to call repeatedly).
void bgm_init();

/// Fade/stop and close the audio output device before SDL shutdown.
void bgm_shutdown();

/// Return true if the audio output device is open and active.
bool bgm_is_active();

/// Enable or mute BGM output (ramps smoothly to avoid clicks).
void bgm_set_enabled(bool enabled);

/// Set master BGM volume in [0, 100] percent.
void bgm_set_volume(int percent);

/// Push raw adaptive synthesis parameters (lock-free, thread-safe).
void bgm_update_params(uint32_t seed, float depth, float tension, float mystery);

/// Derive adaptive BGM parameters from the current DCSS game state and push them.
void bgm_update();

/// Render interleaved stereo float frames directly (useful for audio callback and tests).
void bgm_render_stereo(float *out_interleaved, int num_frames);
