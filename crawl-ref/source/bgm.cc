/**
 * @file
 * @brief Procedural, adaptive, pure-synthesis Roguelike Dungeon Synth BGM engine.
 *
 * Pure mathematical C++ synthesis (zero external audio files or codecs):
 *  - Modal theory sequencer (Dorian / Phrygian / Harmonic Minor, 56-78 BPM,
 *    6/8 / 3/4 / slow 4/4, 4-bar Call-and-Response Markov motif, harmonic ostinato)
 *  - 5-instrument ensemble: Drone Pad, Karplus-Strong Pluck Lute/Harp,
 *    2-Op FM Spectral Glass/Bell Lead, Formant-Filtered Dark Cello,
 *    Sub-Timpani & High-Q Cavern Droplets
 *  - Dotted-eighth stereo ping-pong delay, Schroeder/Freeverb cavern reverb,
 *    and RMS compressor + soft tanh limiter.
**/

#include "AppHdr.h"

#include "bgm.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>

#if defined(USE_SOUND) && defined(USE_SDL)
# ifdef __ANDROID__
#  include <SDL.h>
# elif defined(TARGET_COMPILER_VC)
#  include <SDL.h>
# else
#  include <SDL2/SDL.h>
# endif
#endif

#include "branch.h"
#include "options.h"
#include "player.h"
#include "religion.h"
#include "state.h"

namespace
{

constexpr float SAMPLE_RATE = 44100.0f;
constexpr float INV_SAMPLE_RATE = 1.0f / SAMPLE_RATE;
constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float PI_VAL = 3.14159265358979323846f;

static inline float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float lerpf(float a, float b, float t)
{
    return a + (b - a) * t;
}

static inline float flush_denormal(float x)
{
    if (!std::isfinite(x) || std::fabs(x) < 1e-20f)
        return 0.0f;
    return x;
}

static inline float midi_to_hz(float midi)
{
    return 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f);
}

/// PolyBLEP anti-aliasing correction for saw/pulse discontinuities.
static inline float poly_blep(float phase, float phase_inc)
{
    if (phase < phase_inc)
    {
        float t = phase / std::max(phase_inc, 1e-7f);
        return t + t - t * t - 1.0f;
    }
    if (phase > 1.0f - phase_inc)
    {
        float t = (phase - 1.0f) / std::max(phase_inc, 1e-7f);
        return t * t + t + t + 1.0f;
    }
    return 0.0f;
}

/// Deterministic Mulberry32 PRNG so each floor seed yields a signature motif.
struct Mulberry32
{
    uint32_t state;

    explicit Mulberry32(uint32_t seed = 0x517cc1b7u)
        : state(seed ? seed : 0x9e3779b9u)
    {
    }

    uint32_t next_u32()
    {
        uint32_t z = (state += 0x6d2b79f5u);
        z = (z ^ (z >> 15)) * (z | 1u);
        z ^= z + (z ^ (z >> 7)) * (z | 61u);
        return z ^ (z >> 14);
    }

    float next_float()
    {
        return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f);
    }

    int range(int lo, int hi)
    {
        if (hi <= lo)
            return lo;
        const uint32_t span = static_cast<uint32_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u32() % span);
    }
};

/// Paul Kellet's pink noise generator (accurate +/-0.05 dB slope, no heap).
struct PinkNoise
{
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
    float b3 = 0.0f, b4 = 0.0f, b5 = 0.0f, b6 = 0.0f;

    float process(Mulberry32 &rng)
    {
        const float white = rng.next_float() * 2.0f - 1.0f;
        b0 = 0.99886f * b0 + white * 0.0555179f;
        b1 = 0.99332f * b1 + white * 0.0750759f;
        b2 = 0.96900f * b2 + white * 0.1538520f;
        b3 = 0.86650f * b3 + white * 0.3104856f;
        b4 = 0.55000f * b4 + white * 0.5329522f;
        b5 = -0.7616f * b5 - white * 0.0168980f;
        const float pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362f;
        b6 = white * 0.115926f;
        return pink * 0.12f;
    }
};

/// Simple 1-pole high-pass filter to keep sub-160 Hz clean on non-bass buses.
struct OnePoleHPF
{
    float prev_x = 0.0f;
    float prev_y = 0.0f;
    float alpha = 0.977f;

    void set_cutoff(float hz)
    {
        const float rc = 1.0f / (TWO_PI * std::max(hz, 20.0f));
        alpha = rc / (rc + INV_SAMPLE_RATE);
    }

    float process(float x)
    {
        const float y = alpha * (prev_y + x - prev_x);
        prev_x = x;
        prev_y = flush_denormal(y);
        return prev_y;
    }
};

/// State-variable filter providing clean 12 dB/oct LP and BP without coefficient bombs.
struct StateVarFilter
{
    float ic1eq = 0.0f;
    float ic2eq = 0.0f;

    void reset()
    {
        ic1eq = 0.0f;
        ic2eq = 0.0f;
    }

    void process(float x, float cutoff_hz, float q, float &out_lp, float &out_bp)
    {
        const float clamped_hz = clampf(cutoff_hz, 30.0f, 16000.0f);
        const float g = std::tan(PI_VAL * clamped_hz * INV_SAMPLE_RATE);
        const float k = 1.0f / std::max(q, 0.5f);
        const float a1 = 1.0f / (1.0f + g * (g + k));
        const float a2 = g * a1;
        const float a3 = g * a2;
        const float v3 = x - ic2eq;
        const float v1 = a1 * ic1eq + a2 * v3;
        const float v2 = ic2eq + a2 * ic1eq + a3 * v3;
        ic1eq = flush_denormal(2.0f * v1 - ic1eq);
        ic2eq = flush_denormal(2.0f * v2 - ic2eq);
        out_bp = v1;
        out_lp = v2;
    }
};

// =========================================================================
// Musical Theory & Call-and-Response Sequencer
// =========================================================================

enum ScaleMode
{
    SCALE_DORIAN = 0,
    SCALE_PHRYGIAN = 1,
    SCALE_HARMONIC_MINOR = 2,
};

constexpr int NUM_BARS = 4;
constexpr int MAX_STEPS_PER_BAR = 16;

const std::array<std::array<int, 7>, 3> SCALE_INTERVALS = {{
    {{ 0, 2, 3, 5, 7, 9, 10 }}, // Dorian
    {{ 0, 1, 3, 5, 7, 8, 10 }}, // Phrygian
    {{ 0, 2, 3, 5, 7, 8, 11 }}, // Harmonic Minor
}};

static int degree_to_midi(int root_midi, ScaleMode mode, int degree)
{
    int oct = degree / 7;
    int deg = degree % 7;
    if (deg < 0)
    {
        deg += 7;
        --oct;
    }
    const auto &ints = SCALE_INTERVALS[static_cast<size_t>(mode)];
    return root_midi + oct * 12 + ints[static_cast<size_t>(deg)];
}

struct ChordSlot
{
    int root_deg;
    int tones[4]; // MIDI note offsets relative to scale root + octave (in C3/D3 range)
};

static ChordSlot build_chord(ScaleMode mode, int root_midi, int prog_idx, int chord_idx)
{
    ChordSlot slot{};
    if (prog_idx == 0)
    {
        // i -> VI -> iv -> V7(b9)
        static const int roots[4] = { 0, 5, 3, 4 };
        const int r = roots[chord_idx & 3];
        slot.root_deg = r;
        if ((chord_idx & 3) == 3)
        {
            // Explicit V7(b9) gothic cadence relative to root
            const int fifth_midi = root_midi + 7;
            slot.tones[0] = fifth_midi;
            slot.tones[1] = fifth_midi + 4;  // major 3rd of V
            slot.tones[2] = fifth_midi + 10; // b7 of V
            slot.tones[3] = fifth_midi + 13; // b9 of V
        }
        else
        {
            slot.tones[0] = degree_to_midi(root_midi, mode, r);
            slot.tones[1] = degree_to_midi(root_midi, mode, r + 2);
            slot.tones[2] = degree_to_midi(root_midi, mode, r + 4);
            slot.tones[3] = degree_to_midi(root_midi, mode, r + 7);
        }
    }
    else
    {
        // i -> bII -> i -> vii°
        static const int roots[4] = { 0, 1, 0, 6 };
        const int r = roots[chord_idx & 3];
        slot.root_deg = r;
        if ((chord_idx & 3) == 1)
        {
            // Authentic bII (Neapolitan/Phrygian II) major triad
            const int b2 = root_midi + 1;
            slot.tones[0] = b2;
            slot.tones[1] = b2 + 4;
            slot.tones[2] = b2 + 7;
            slot.tones[3] = b2 + 12;
        }
        else if ((chord_idx & 3) == 3)
        {
            // Leading-tone diminished 7th vii°7
            const int lead = root_midi + (mode == SCALE_HARMONIC_MINOR ? 11 : 10);
            slot.tones[0] = lead;
            slot.tones[1] = lead + 3;
            slot.tones[2] = lead + 6;
            slot.tones[3] = lead + 9;
        }
        else
        {
            slot.tones[0] = degree_to_midi(root_midi, mode, 0);
            slot.tones[1] = degree_to_midi(root_midi, mode, 2);
            slot.tones[2] = degree_to_midi(root_midi, mode, 4);
            slot.tones[3] = degree_to_midi(root_midi, mode, 7);
        }
    }
    return slot;
}

struct MotifNote
{
    bool active = false;
    int degree = 0;
    float vel = 0.8f;
    float dur_beats = 1.0f;
};

// =========================================================================
// 5-Instrument Voices
// =========================================================================

/// 1. Subterranean Drone Pad (dual detuned PolyBLEP saws + sub-sine + LFO -> LPF)
struct DroneVoice
{
    bool active = false;
    bool releasing = false;
    float freq = 65.41f;
    float phase1 = 0.0f;
    float phase2 = 0.33f;
    float phase_sub = 0.67f;
    float lfo_phase = 0.0f;
    float lfo_hz = 0.11f;
    float env = 0.0f;
    float sustain_gain = 0.45f;
    StateVarFilter lpf;

    void start(float hz, float lfo_rate, float level)
    {
        active = true;
        releasing = false;
        freq = hz;
        lfo_hz = lfo_rate;
        sustain_gain = level;
        env = 0.0f;
        lpf.reset();
    }

    void trigger_release()
    {
        if (active)
            releasing = true;
    }

    float process(float tension, float &out_r)
    {
        if (!active)
        {
            out_r = 0.0f;
            return 0.0f;
        }

        if (!releasing)
        {
            env += INV_SAMPLE_RATE / 1.5f; // 1.5s slow atmospheric attack
            if (env >= 1.0f)
                env = 1.0f;
        }
        else
        {
            env -= INV_SAMPLE_RATE / 2.5f; // 2.5s release
            if (env <= 0.0001f)
            {
                active = false;
                out_r = 0.0f;
                return 0.0f;
            }
        }

        // Detuned -7 cents & +7 cents
        const float inc1 = freq * 0.995965f * INV_SAMPLE_RATE;
        const float inc2 = freq * 1.004051f * INV_SAMPLE_RATE;
        const float inc_sub = (freq * 0.5f) * INV_SAMPLE_RATE;

        phase1 += inc1;
        if (phase1 >= 1.0f) phase1 -= 1.0f;
        phase2 += inc2;
        if (phase2 >= 1.0f) phase2 -= 1.0f;
        phase_sub += inc_sub;
        if (phase_sub >= 1.0f) phase_sub -= 1.0f;

        const float saw1 = (2.0f * phase1 - 1.0f) - poly_blep(phase1, inc1);
        const float saw2 = (2.0f * phase2 - 1.0f) - poly_blep(phase2, inc2);
        const float sub = std::sin(TWO_PI * phase_sub);

        lfo_phase += lfo_hz * INV_SAMPLE_RATE;
        if (lfo_phase >= 1.0f) lfo_phase -= 1.0f;
        const float lfo = 0.5f + 0.5f * std::sin(TWO_PI * lfo_phase);

        // Sweep LPF cutoff in 180 Hz - 620 Hz modulated by LFO and tension
        const float base_cut = lerpf(180.0f, 380.0f, tension);
        const float span_cut = lerpf(140.0f, 240.0f, tension);
        const float cutoff = clampf(base_cut + lfo * span_cut, 180.0f, 620.0f);

        float lp1 = 0.0f, bp_unused = 0.0f;
        lpf.process(0.45f * (saw1 + saw2), cutoff, 1.15f, lp1, bp_unused);

        const float mono = (lp1 * 0.65f + sub * 0.35f) * env * sustain_gain;
        // Subtle stereo spread from detuned pair
        const float spread = (saw1 - saw2) * 0.06f * env * sustain_gain;
        out_r = mono - spread;
        return mono + spread;
    }
};

/// 2. Ancient Pluck Harp / Lute (Karplus-Strong physical string model)
struct PluckVoice
{
    static constexpr int MAX_DELAY = 512;

    bool active = false;
    float buffer[MAX_DELAY]{};
    int write_idx = 0;
    float period = 100.0f;
    float damping = 0.495f;
    float prev_sample = 0.0f;
    float attack_ramp = 0.0f;
    float gain = 0.3f;
    float pan = 0.0f;
    float energy = 0.0f;
    OnePoleHPF hpf;

    void trigger(float hz, float vel, float p_pan, Mulberry32 &rng)
    {
        const float clamped_hz = clampf(hz, 95.0f, 1600.0f);
        period = clampf(SAMPLE_RATE / clamped_hz, 8.0f, static_cast<float>(MAX_DELAY - 3));
        const int plen = static_cast<int>(period) + 2;

        // Low-pass smoothed noise burst exciter
        float lp = 0.0f;
        for (int i = 0; i < MAX_DELAY; ++i)
        {
            if (i < plen)
            {
                const float w = rng.next_float() * 2.0f - 1.0f;
                lp = lp * 0.45f + w * 0.55f;
                buffer[i] = lp;
            }
            else
            {
                buffer[i] = 0.0f;
            }
        }
        write_idx = plen % MAX_DELAY;
        prev_sample = 0.0f;
        attack_ramp = 0.0f;
        // Brighter/longer strings for higher notes, natural 450-900ms decay
        damping = clampf(0.4935f + (clamped_hz / 1800.0f) * 0.0045f, 0.492f, 0.498f);
        gain = vel * 0.34f;
        pan = clampf(p_pan, -0.7f, 0.7f);
        energy = 1.0f;
        hpf.set_cutoff(165.0f);
        hpf.prev_x = 0.0f;
        hpf.prev_y = 0.0f;
        active = true;
    }

    void process(float &out_l, float &out_r)
    {
        if (!active)
        {
            out_l = out_r = 0.0f;
            return;
        }

        // Fractional read position for accurate tuning
        float read_pos = static_cast<float>(write_idx) - period;
        while (read_pos < 0.0f)
            read_pos += static_cast<float>(MAX_DELAY);

        const int i0 = static_cast<int>(read_pos) % MAX_DELAY;
        const int i1 = (i0 + 1) % MAX_DELAY;
        const float frac = read_pos - std::floor(read_pos);
        const float raw = lerpf(buffer[i0], buffer[i1], frac);

        // Karplus-Strong damping filter: y[n] = damping * (x[n] + x[n-1])
        const float damped = flush_denormal(damping * (raw + prev_sample));
        prev_sample = raw;

        buffer[write_idx] = damped;
        write_idx = (write_idx + 1) % MAX_DELAY;

        if (attack_ramp < 1.0f)
            attack_ramp = std::min(1.0f, attack_ramp + INV_SAMPLE_RATE / 0.003f); // 3ms

        energy = energy * 0.99985f + std::fabs(damped) * 0.0002f;
        if (energy < 1e-4f && attack_ramp >= 1.0f)
        {
            active = false;
            out_l = out_r = 0.0f;
            return;
        }

        const float sig = hpf.process(damped) * attack_ramp * gain;
        out_l = sig * (0.707f - 0.35f * pan);
        out_r = sig * (0.707f + 0.35f * pan);
    }
};

/// 3. Spectral Glass / FM Bell Lead (2-Op FM + decaying mod index + delayed vibrato)
struct FMBellVoice
{
    bool active = false;
    float carrier_hz = 440.0f;
    float harmonicity = 2.0f;
    float phase_c = 0.0f;
    float phase_m = 0.0f;
    float vib_phase = 0.0f;
    float age = 0.0f;
    float max_dur = 2.2f;
    float gain = 0.25f;
    float pan = 0.0f;
    OnePoleHPF hpf;

    void trigger(float hz, float vel, float ratio, float dur_sec, float p_pan)
    {
        carrier_hz = clampf(hz, 220.0f, 1900.0f);
        harmonicity = ratio;
        phase_c = 0.0f;
        phase_m = 0.0f;
        vib_phase = 0.0f;
        age = 0.0f;
        max_dur = clampf(dur_sec, 0.5f, 3.8f);
        gain = vel * 0.26f;
        pan = clampf(p_pan, -0.6f, 0.6f);
        hpf.set_cutoff(165.0f);
        hpf.prev_x = 0.0f;
        hpf.prev_y = 0.0f;
        active = true;
    }

    void process(float &out_l, float &out_r)
    {
        if (!active)
        {
            out_l = out_r = 0.0f;
            return;
        }

        age += INV_SAMPLE_RATE;
        if (age >= max_dur)
        {
            active = false;
            out_l = out_r = 0.0f;
            return;
        }

        // Delayed 5.2 Hz pitch vibrato (+/- 4 cents) fading in after 250ms
        vib_phase += 5.2f * INV_SAMPLE_RATE;
        if (vib_phase >= 1.0f) vib_phase -= 1.0f;
        const float vib_fade = clampf((age - 0.25f) / 0.30f, 0.0f, 1.0f);
        const float vib_cents = 4.0f * vib_fade * std::sin(TWO_PI * vib_phase);
        const float fc = carrier_hz * std::pow(2.0f, vib_cents / 1200.0f);
        const float fm = fc * harmonicity;

        // Modulation index I(t): 2.2 decaying exponentially toward 0.15 over 350ms
        const float mod_index = 0.15f + 2.05f * std::exp(-age / 0.13f);

        phase_m += fm * INV_SAMPLE_RATE;
        if (phase_m >= 1.0f) phase_m -= 1.0f;
        const float mod_sig = mod_index * std::sin(TWO_PI * phase_m);

        phase_c += fc * INV_SAMPLE_RATE;
        if (phase_c >= 1.0f) phase_c -= 1.0f;
        const float raw = std::sin(TWO_PI * phase_c + mod_sig);

        // Smooth attack (4ms) + crystalline decay + soft fadeout tail
        const float att = clampf(age / 0.004f, 0.0f, 1.0f);
        const float dec = std::exp(-age / (max_dur * 0.42f));
        const float tail = clampf((max_dur - age) / 0.08f, 0.0f, 1.0f);
        const float sig = hpf.process(raw) * att * dec * tail * gain;

        out_l = sig * (0.707f - 0.3f * pan);
        out_r = sig * (0.707f + 0.3f * pan);
    }
};

/// 4. Bowed Dark Cello / Vihuela Swell (Saw + sub-triangle -> dual formant BPFs)
struct CelloVoice
{
    bool active = false;
    float freq = 146.83f;
    float phase = 0.0f;
    float phase_sub = 0.0f;
    float vib_phase = 0.0f;
    float age = 0.0f;
    float hold_dur = 1.55f; // 350ms attack + 1200ms sustain
    float rel_dur = 0.80f;  // 800ms release
    float gain = 0.25f;
    StateVarFilter formant1;
    StateVarFilter formant2;
    OnePoleHPF hpf;

    void trigger(float hz, float vel)
    {
        freq = clampf(hz, 95.0f, 480.0f);
        phase = 0.0f;
        phase_sub = 0.0f;
        vib_phase = 0.0f;
        age = 0.0f;
        gain = vel * 0.24f;
        formant1.reset();
        formant2.reset();
        hpf.set_cutoff(165.0f);
        hpf.prev_x = 0.0f;
        hpf.prev_y = 0.0f;
        active = true;
    }

    float process(float &out_r)
    {
        if (!active)
        {
            out_r = 0.0f;
            return 0.0f;
        }

        age += INV_SAMPLE_RATE;
        const float total = hold_dur + rel_dur;
        if (age >= total)
        {
            active = false;
            out_r = 0.0f;
            return 0.0f;
        }

        // Bow ADSR envelope (350ms attack, 1.2s sustain, 800ms release)
        float env = 1.0f;
        if (age < 0.35f)
            env = age / 0.35f;
        else if (age > hold_dur)
            env = 1.0f - (age - hold_dur) / rel_dur;

        vib_phase += 4.6f * INV_SAMPLE_RATE;
        if (vib_phase >= 1.0f) vib_phase -= 1.0f;
        const float vib_fade = clampf((age - 0.20f) / 0.35f, 0.0f, 1.0f);
        const float f = freq * std::pow(2.0f, (6.0f * vib_fade * std::sin(TWO_PI * vib_phase)) / 1200.0f);

        const float inc = f * INV_SAMPLE_RATE;
        const float inc_sub = (f * 0.5f) * INV_SAMPLE_RATE;

        phase += inc;
        if (phase >= 1.0f) phase -= 1.0f;
        phase_sub += inc_sub;
        if (phase_sub >= 1.0f) phase_sub -= 1.0f;

        const float saw = (2.0f * phase - 1.0f) - poly_blep(phase, inc);
        const float tri = 4.0f * std::fabs(phase_sub - 0.5f) - 1.0f;
        const float osc = 0.72f * saw + 0.38f * tri;

        float lp_unused = 0.0f, bp420 = 0.0f, bp1100 = 0.0f;
        formant1.process(osc, 420.0f, 3.0f, lp_unused, bp420);
        formant2.process(osc, 1100.0f, 3.8f, lp_unused, bp1100);

        const float body = hpf.process(osc * 0.22f + bp420 * 0.85f + bp1100 * 0.50f);
        const float sig = body * env * gain;
        out_r = sig * 0.75f;
        return sig * 0.65f;
    }
};

/// 5a. Ritual Sub-Timpani (115 Hz -> 48 Hz pitch drop + 160 Hz LP noise thud)
struct TimpaniVoice
{
    bool active = false;
    float age = 0.0f;
    float phase = 0.0f;
    float noise_lp = 0.0f;
    float gain = 0.38f;

    void trigger(float vel)
    {
        active = true;
        age = 0.0f;
        phase = 0.0f;
        noise_lp = 0.0f;
        gain = vel * 0.42f;
    }

    float process(Mulberry32 &rng)
    {
        if (!active)
            return 0.0f;

        age += INV_SAMPLE_RATE;
        if (age >= 1.35f)
        {
            active = false;
            return 0.0f;
        }

        // Pitch drops 115 Hz -> 48 Hz over ~180ms
        const float hz = 48.0f + 67.0f * std::exp(-age / 0.055f);
        phase += hz * INV_SAMPLE_RATE;
        if (phase >= 1.0f) phase -= 1.0f;

        const float att = clampf(age / 0.005f, 0.0f, 1.0f);
        const float sine_body = std::sin(TWO_PI * phase) * std::exp(-age / 0.34f);

        // Low-passed noise thud (160 Hz cutoff, 60ms decay)
        const float white = rng.next_float() * 2.0f - 1.0f;
        const float lp_alpha = 0.022f; // ~160 Hz at 44.1kHz
        noise_lp = flush_denormal(noise_lp + lp_alpha * (white - noise_lp));
        const float thud = noise_lp * std::exp(-age / 0.022f) * 1.4f;

        return (sine_body + thud) * att * gain;
    }
};

/// 5b. Cavern Droplet / Metallic Air (High-Q bandpass Q=18 pink noise, 900-2400 Hz)
struct DropletVoice
{
    bool active = false;
    float age = 0.0f;
    float center_hz = 1400.0f;
    float sweep_hz = 400.0f;
    float gain = 0.15f;
    float pan = 0.0f;
    StateVarFilter bpf;
    PinkNoise pink;
    OnePoleHPF hpf;

    void trigger(float hz, float sweep, float vel, float p_pan)
    {
        active = true;
        age = 0.0f;
        center_hz = clampf(hz, 900.0f, 2400.0f);
        sweep_hz = sweep;
        gain = vel * 0.42f;
        pan = clampf(p_pan, -0.8f, 0.8f);
        bpf.reset();
        hpf.set_cutoff(180.0f);
        hpf.prev_x = 0.0f;
        hpf.prev_y = 0.0f;
    }

    void process(Mulberry32 &rng, float &out_l, float &out_r)
    {
        if (!active)
        {
            out_l = out_r = 0.0f;
            return;
        }

        age += INV_SAMPLE_RATE;
        if (age >= 0.45f)
        {
            active = false;
            out_l = out_r = 0.0f;
            return;
        }

        const float pn = pink.process(rng);
        const float hz = clampf(center_hz + sweep_hz * (1.0f - std::exp(-age / 0.04f)),
                                750.0f, 2800.0f);
        float lp_unused = 0.0f, bp = 0.0f;
        bpf.process(pn, hz, 18.0f, lp_unused, bp);

        const float env = clampf(age / 0.004f, 0.0f, 1.0f) * std::exp(-age / 0.085f);
        const float sig = hpf.process(bp) * env * gain;
        out_l = sig * (0.707f - 0.45f * pan);
        out_r = sig * (0.707f + 0.45f * pan);
    }
};

// =========================================================================
// Spatial Effects (Ping-Pong Delay + Schroeder Reverb + Master Limiter)
// =========================================================================

struct PingPongDelay
{
    static constexpr int MAX_DELAY = 88200; // 2.0 seconds at 44100 Hz

    float buf_l[MAX_DELAY]{};
    float buf_r[MAX_DELAY]{};
    int write_idx = 0;
    float lp_l = 0.0f;
    float lp_r = 0.0f;
    float smooth_delay_samples = 28000.0f;

    void process(float in_l, float in_r, float bpm, float &out_l, float &out_r)
    {
        // Dotted-eighth delay: (60 / bpm) * 0.75 seconds
        const float target_samples = clampf(((60.0f / clampf(bpm, 56.0f, 78.0f)) * 0.75f) * SAMPLE_RATE,
                                            4000.0f, static_cast<float>(MAX_DELAY - 2));
        smooth_delay_samples += 0.0005f * (target_samples - smooth_delay_samples);

        const int delay_int = static_cast<int>(smooth_delay_samples);
        const int r_idx = (write_idx - delay_int + MAX_DELAY) % MAX_DELAY;

        const float d_l = buf_l[r_idx];
        const float d_r = buf_r[r_idx];

        // 1-pole LP at 1800 Hz in feedback path
        const float lp_alpha = 0.225f;
        lp_l = flush_denormal(lp_l + lp_alpha * (d_l - lp_l));
        lp_r = flush_denormal(lp_r + lp_alpha * (d_r - lp_r));

        // Cross-coupled ping-pong with 0.32 feedback
        buf_l[write_idx] = flush_denormal(in_l + lp_r * 0.32f);
        buf_r[write_idx] = flush_denormal(in_r + lp_l * 0.32f);

        write_idx = (write_idx + 1) % MAX_DELAY;
        out_l = d_l;
        out_r = d_r;
    }
};

struct SchroederReverb
{
    // 4 parallel feedback combs: 29.7ms, 37.1ms, 41.1ms, 43.7ms (+ stereo R offset)
    // 2 series all-pass diffusers: 5.0ms, 1.7ms
    static constexpr int COMB_MAX = 2048;
    static constexpr int AP_MAX = 512;

    struct Comb
    {
        float buf[COMB_MAX]{};
        int idx = 0;
        int len = 1024;
        float lp = 0.0f;

        float process(float x, float fb, float damp)
        {
            const float out = buf[idx];
            lp = flush_denormal(out * (1.0f - damp) + lp * damp);
            buf[idx] = flush_denormal(x + lp * fb);
            if (++idx >= len)
                idx = 0;
            return out;
        }
    };

    struct AllPass
    {
        float buf[AP_MAX]{};
        int idx = 0;
        int len = 220;

        float process(float x)
        {
            const float bufout = buf[idx];
            const float y = -x + bufout;
            buf[idx] = flush_denormal(x + bufout * 0.5f);
            if (++idx >= len)
                idx = 0;
            return y;
        }
    };

    Comb comb_l[4];
    Comb comb_r[4];
    AllPass ap_l[2];
    AllPass ap_r[2];
    OnePoleHPF return_hpf_l;
    OnePoleHPF return_hpf_r;

    SchroederReverb()
    {
        static const int comb_lengths[4] = {
            1310, // 29.7 ms
            1636, // 37.1 ms
            1813, // 41.1 ms
            1927  // 43.7 ms
        };
        static const int ap_lengths[2] = {
            221, // 5.0 ms
            75   // 1.7 ms
        };

        for (int i = 0; i < 4; ++i)
        {
            comb_l[i].len = comb_lengths[i];
            comb_r[i].len = std::min(COMB_MAX - 1, comb_lengths[i] + 23);
        }
        for (int i = 0; i < 2; ++i)
        {
            ap_l[i].len = ap_lengths[i];
            ap_r[i].len = std::min(AP_MAX - 1, ap_lengths[i] + 11);
        }
        return_hpf_l.set_cutoff(165.0f);
        return_hpf_r.set_cutoff(165.0f);
    }

    void process(float in_l, float in_r, float depth, float &out_l, float &out_r)
    {
        // Deeper cavern increases tail feedback subtly (0.81 -> 0.87)
        const float fb = lerpf(0.81f, 0.87f, depth);
        const float damp = 0.28f; // ~2200 Hz damping inside comb loop

        float sum_l = 0.0f;
        float sum_r = 0.0f;
        for (int i = 0; i < 4; ++i)
        {
            sum_l += comb_l[i].process(in_l, fb, damp);
            sum_r += comb_r[i].process(in_r, fb, damp);
        }

        sum_l *= 0.25f;
        sum_r *= 0.25f;

        for (int i = 0; i < 2; ++i)
        {
            sum_l = ap_l[i].process(sum_l);
            sum_r = ap_r[i].process(sum_r);
        }

        out_l = return_hpf_l.process(sum_l);
        out_r = return_hpf_r.process(sum_r);
    }
};

struct MasterCompressor
{
    float env = 0.0f;

    void process(float &l, float &r)
    {
        const float peak = std::max(std::fabs(l), std::fabs(r));
        const float coeff = (peak > env)
            ? (1.0f - std::exp(-1.0f / (0.010f * SAMPLE_RATE)))   // 10ms attack
            : (1.0f - std::exp(-1.0f / (0.120f * SAMPLE_RATE)));  // 120ms release
        env = flush_denormal(env + coeff * (peak - env));

        // Threshold -14 dB (~0.20 linear), Ratio 3:1
        constexpr float THRESH = 0.20f;
        float gain = 1.0f;
        if (env > THRESH)
        {
            const float over_db = 20.0f * std::log10(env / THRESH);
            const float reduced_db = over_db * (1.0f - 1.0f / 3.0f);
            gain = std::pow(10.0f, -reduced_db / 20.0f);
        }

        // Gentle makeup + soft tanh limiter
        constexpr float MAKEUP = 1.15f;
        l = std::tanh(l * gain * MAKEUP);
        r = std::tanh(r * gain * MAKEUP);
    }
};

// =========================================================================
// Unified Adaptive Engine
// =========================================================================

struct PendingEvent
{
    enum Type { NONE, HARP, BELL, CELLO, TIMPANI, DROPLET };

    bool active = false;
    Type type = NONE;
    int countdown = 0;
    float hz = 440.0f;
    float vel = 0.8f;
    float dur = 1.0f;
    float pan = 0.0f;
};

struct BgmEngine
{
    Mulberry32 rng;
    uint32_t current_seed = 0x44435353u; // "DCSS"
    float smooth_depth = 0.15f;
    float smooth_tension = 0.10f;
    float smooth_mystery = 0.35f;
    float smooth_vol = 0.70f;
    float smooth_mute = 1.0f;

    // Musical sequencer state
    int bar_in_phrase = 0;
    int step_in_bar = 0;
    int steps_per_bar = 12; // 12 for 6/8 & 3/4, 16 for 4/4
    int meter_type = 0;     // 0=6/8, 1=3/4, 2=4/4
    int bars_per_chord = 1;
    int prog_idx = 0;
    int phrase_count = 0;
    float step_sample_accum = 0.0f;
    bool initialized_phrase = false;
    int arp_cursor = 0;
    MotifNote motif[NUM_BARS][MAX_STEPS_PER_BAR]{};

    // Instrument pools
    DroneVoice drones[2];
    int active_drone = 0;
    PluckVoice plucks[8];
    int next_pluck = 0;
    FMBellVoice bells[6];
    int next_bell = 0;
    CelloVoice cellos[3];
    int next_cello = 0;
    TimpaniVoice timpanis[2];
    int next_timpani = 0;
    DropletVoice droplets[3];
    int next_droplet = 0;
    PendingEvent queue[32];

    PingPongDelay delay;
    SchroederReverb reverb;
    MasterCompressor comp;

    ScaleMode current_scale() const
    {
        if (smooth_depth <= 0.35f)
            return SCALE_DORIAN;
        if (smooth_depth <= 0.70f)
            return SCALE_PHRYGIAN;
        return SCALE_HARMONIC_MINOR;
    }

    int base_root_midi() const
    {
        // Root in C or D; octave lowers with depth
        const int pitch_class = ((current_seed & 1u) == 0u) ? 0 : 2; // C or D
        const int base_oct = (smooth_depth > 0.70f) ? 36 : 48; // C2/D2 vs C3/D3
        return base_oct + pitch_class;
    }

    void schedule_event(PendingEvent::Type t, float hz, float vel,
                        float dur, float pan)
    {
        // +/- 12ms timing jitter centered at +12ms baseline latency, +/- 15% velocity
        const int delay_smp = rng.range(0, static_cast<int>(0.024f * SAMPLE_RATE));
        const float hum_vel = clampf(vel * (0.85f + 0.30f * rng.next_float()), 0.05f, 1.0f);

        for (PendingEvent &ev : queue)
        {
            if (!ev.active)
            {
                ev.active = true;
                ev.type = t;
                ev.countdown = delay_smp;
                ev.hz = hz;
                ev.vel = hum_vel;
                ev.dur = dur;
                ev.pan = pan;
                return;
            }
        }
    }

    void fire_event(const PendingEvent &ev)
    {
        switch (ev.type)
        {
        case PendingEvent::HARP:
            plucks[next_pluck].trigger(ev.hz, ev.vel, ev.pan, rng);
            next_pluck = (next_pluck + 1) % 8;
            break;
        case PendingEvent::BELL:
        {
            const float ratio = (smooth_mystery < 0.5f)
                ? lerpf(2.0f, 2.414f, smooth_mystery * 2.0f)
                : lerpf(2.414f, 3.0f, (smooth_mystery - 0.5f) * 2.0f);
            bells[next_bell].trigger(ev.hz, ev.vel, ratio, ev.dur, ev.pan);
            next_bell = (next_bell + 1) % 6;
            break;
        }
        case PendingEvent::CELLO:
            cellos[next_cello].trigger(ev.hz, ev.vel);
            next_cello = (next_cello + 1) % 3;
            break;
        case PendingEvent::TIMPANI:
            timpanis[next_timpani].trigger(ev.vel);
            next_timpani = (next_timpani + 1) % 2;
            break;
        case PendingEvent::DROPLET:
        {
            const float sweep = (rng.next_float() - 0.3f) * 500.0f;
            droplets[next_droplet].trigger(ev.hz, sweep, ev.vel, ev.pan);
            next_droplet = (next_droplet + 1) % 3;
            break;
        }
        case PendingEvent::NONE:
        default:
            break;
        }
    }

    int markov_next_degree(Mulberry32 &prng, int cur) const
    {
        const float r = prng.next_float();
        int next = cur;
        if (r < 0.55f)
            next += (prng.next_u32() & 1u) ? 1 : -1;
        else if (r < 0.75f)
            next += (prng.next_u32() & 1u) ? 2 : -2;
        else if (r < 0.90f)
            next = cur;
        else
            next += (prng.next_u32() & 1u) ? 4 : ((cur > 3) ? -7 : 7);

        if (next < -2) next += 7;
        if (next > 9) next -= 7;
        return next;
    }

    void generate_phrase(uint32_t seed)
    {
        Mulberry32 prng(seed ^ (static_cast<uint32_t>(phrase_count) * 0x9e3779b9u));

        const int m_sel = prng.range(0, 2);
        meter_type = m_sel;
        steps_per_bar = (meter_type == 2) ? 16 : 12; // 6/8 & 3/4 = 12, slow 4/4 = 16
        bars_per_chord = (prng.next_float() < 0.35f) ? 2 : 1;

        const ScaleMode mode = current_scale();
        prog_idx = (mode == SCALE_PHRYGIAN && prng.next_float() < 0.65f) ? 1 : 0;

        for (int b = 0; b < NUM_BARS; ++b)
            for (int s = 0; s < MAX_STEPS_PER_BAR; ++s)
                motif[b][s] = MotifNote{};

        // Build Bar 1 seed call motif on melodic steps (even steps or dotted beats)
        int cur_deg = prng.range(0, 2) * 2; // Start on 1st, 3rd, or 5th
        int last_active_bar1 = 0;
        for (int s = 0; s < steps_per_bar; s += 2)
        {
            // Keep rhythm spacious so notes breathe in cavern reverb
            const bool play = (s == 0) || (prng.next_float() < (s % 4 == 0 ? 0.68f : 0.38f));
            if (play)
            {
                motif[0][s].active = true;
                motif[0][s].degree = cur_deg;
                motif[0][s].vel = (s == 0) ? 0.85f : (0.55f + 0.30f * prng.next_float());
                motif[0][s].dur_beats = (s % 4 == 0) ? 1.8f : 1.1f;
                last_active_bar1 = s;
                cur_deg = markov_next_degree(prng, cur_deg);
            }
        }

        // Bar 3 mirrors Bar 1 seed call with slight variation on the final 2 notes
        for (int s = 0; s < steps_per_bar; ++s)
            motif[2][s] = motif[0][s];
        int altered = 0;
        for (int s = steps_per_bar - 1; s >= 0 && altered < 2; --s)
        {
            if (motif[2][s].active)
            {
                motif[2][s].degree = markov_next_degree(prng, motif[2][s].degree);
                ++altered;
            }
        }

        // Bar 2: unanswered question variation ending on 2nd (1) or 5th (4) degree
        int deg_b2 = cur_deg;
        int last_b2 = -1;
        for (int s = 0; s < steps_per_bar; s += 2)
        {
            if (s == 0 || prng.next_float() < (s % 4 == 0 ? 0.62f : 0.32f))
            {
                motif[1][s].active = true;
                motif[1][s].degree = deg_b2;
                motif[1][s].vel = 0.58f + 0.28f * prng.next_float();
                motif[1][s].dur_beats = 1.4f;
                last_b2 = s;
                deg_b2 = markov_next_degree(prng, deg_b2);
            }
        }
        if (last_b2 >= 0)
            motif[1][last_b2].degree = (prng.next_u32() & 1u) ? 1 : 4;
        else
        {
            motif[1][last_active_bar1].active = true;
            motif[1][last_active_bar1].degree = 4;
            motif[1][last_active_bar1].vel = 0.72f;
            motif[1][last_active_bar1].dur_beats = 1.8f;
        }

        // Bar 4: resolution ending back on 1st (0) or 3rd (2) scale degree
        int deg_b4 = deg_b2;
        int last_b4 = -1;
        for (int s = 0; s < steps_per_bar; s += 2)
        {
            if (s == 0 || prng.next_float() < (s % 4 == 0 ? 0.58f : 0.25f))
            {
                motif[3][s].active = true;
                motif[3][s].degree = deg_b4;
                motif[3][s].vel = (s == 0) ? 0.80f : (0.52f + 0.25f * prng.next_float());
                motif[3][s].dur_beats = 1.6f;
                last_b4 = s;
                deg_b4 = markov_next_degree(prng, deg_b4);
            }
        }
        if (last_b4 >= 0)
        {
            motif[3][last_b4].degree = (prng.next_float() < 0.70f) ? 0 : 2;
            motif[3][last_b4].dur_beats = 2.6f;
        }
        else
        {
            motif[3][0].active = true;
            motif[3][0].degree = 0;
            motif[3][0].vel = 0.82f;
            motif[3][0].dur_beats = 2.8f;
        }
    }

    void trigger_bar_instruments(int chord_idx, int root_midi, ScaleMode mode, float bpm)
    {
        const ChordSlot chord = build_chord(mode, root_midi, prog_idx, chord_idx);

        // Update drone chord on Bar 0 or every chord transition
        if ((bar_in_phrase % bars_per_chord) == 0 || !drones[active_drone].active)
        {
            drones[active_drone].trigger_release();
            active_drone = 1 - active_drone;
            int drone_midi = chord.tones[0];
            while (drone_midi < 34) drone_midi += 12; // >= ~58 Hz
            while (drone_midi > 55) drone_midi -= 12; // <= ~196 Hz
            const float lfo_hz = lerpf(0.08f, 0.18f, smooth_mystery);
            drones[active_drone].start(midi_to_hz(static_cast<float>(drone_midi)),
                                       lfo_hz,
                                       lerpf(0.32f, 0.44f, smooth_tension));
        }

        // Ritual Sub-Timpani on Bar 1 downbeat (plus occasional Bar 3 punctuation under high tension)
        if (bar_in_phrase == 0)
        {
            const float timp_vel = clampf(0.35f + smooth_tension * 0.60f, 0.30f, 0.95f);
            schedule_event(PendingEvent::TIMPANI, 48.0f, timp_vel, 1.0f, 0.0f);
        }
        else if (bar_in_phrase == 2 && smooth_tension > 0.62f)
        {
            schedule_event(PendingEvent::TIMPANI, 48.0f, 0.45f * smooth_tension, 1.0f, 0.0f);
        }

        // Bowed Dark Cello Swell on Bar 2 / Bar 4, or any bar when tension rises
        const bool cello_bar = (bar_in_phrase == 1 || bar_in_phrase == 3)
                               || (smooth_tension > 0.45f && (bar_in_phrase & 1) == 0);
        if (cello_bar)
        {
            int cello_midi = (bar_in_phrase == 3) ? chord.tones[0] : chord.tones[1];
            while (cello_midi < 45) cello_midi += 12; // >= 110 Hz (A2)
            while (cello_midi > 67) cello_midi -= 12; // <= 440 Hz
            const float c_vel = clampf(0.45f + smooth_tension * 0.50f, 0.40f, 0.95f);
            schedule_event(PendingEvent::CELLO, midi_to_hz(static_cast<float>(cello_midi)),
                           c_vel, 2.3f, 0.0f);
        }

        // Sparse Cavern Droplet / Scrape (~15% base probability per bar + mystery)
        const float drop_chance = 0.15f + 0.20f * smooth_mystery;
        if (rng.next_float() < drop_chance)
        {
            const float drop_hz = lerpf(900.0f, 2400.0f, rng.next_float());
            const float pan = rng.next_float() * 1.4f - 0.7f;
            schedule_event(PendingEvent::DROPLET, drop_hz,
                           0.55f + 0.40f * rng.next_float(),
                           60.0f / bpm, pan);
        }
    }

    void trigger_step(float bpm)
    {
        const ScaleMode mode = current_scale();
        const int root_midi = base_root_midi();
        const int chord_idx = (bar_in_phrase / bars_per_chord) % 4;

        if (step_in_bar == 0)
            trigger_bar_instruments(chord_idx, root_midi, mode, bpm);

        // 1. Pluck Harp / Lute ostinato: 1/8 notes on even steps; 1/16 unlocked with tension
        const bool eighth_step = (step_in_bar % 2) == 0;
        const bool do_harp = eighth_step
                             || (smooth_tension > 0.42f && rng.next_float() < (smooth_tension - 0.30f));
        if (do_harp && (rng.next_float() < lerpf(0.68f, 0.95f, smooth_tension)))
        {
            const ChordSlot chord = build_chord(mode, root_midi + 12, prog_idx, chord_idx);
            const int pat_idx = arp_cursor % 4;
            ++arp_cursor;
            int harp_midi = chord.tones[pat_idx];
            while (harp_midi < 55) harp_midi += 12; // >= 196 Hz
            while (harp_midi > 79) harp_midi -= 12; // <= 784 Hz
            const float accent = (step_in_bar == 0) ? 0.92f
                               : ((meter_type == 0 && step_in_bar == 6) ? 0.78f : 0.62f);
            const float harp_pan = ((arp_cursor & 1) ? 0.28f : -0.28f);
            schedule_event(PendingEvent::HARP,
                           midi_to_hz(static_cast<float>(harp_midi)),
                           accent, 0.7f, harp_pan);
        }

        // 2. Spectral Glass / FM Bell Lead carrying 4-bar Call-and-Response motif
        if (step_in_bar < MAX_STEPS_PER_BAR)
        {
            const MotifNote &mn = motif[bar_in_phrase][step_in_bar];
            if (mn.active)
            {
                int mel_midi = degree_to_midi(root_midi + 24, mode, mn.degree);
                while (mel_midi < 67) mel_midi += 12; // >= 392 Hz (G4)
                while (mel_midi > 91) mel_midi -= 12; // <= ~1568 Hz
                const float beat_sec = 60.0f / bpm;
                const float mel_pan = (bar_in_phrase & 1) ? -0.20f : 0.20f;
                schedule_event(PendingEvent::BELL,
                               midi_to_hz(static_cast<float>(mel_midi)),
                               mn.vel, mn.dur_beats * beat_sec, mel_pan);
            }
        }

        // Advance sequencer counters
        ++step_in_bar;
        if (step_in_bar >= steps_per_bar)
        {
            step_in_bar = 0;
            ++bar_in_phrase;
            if (bar_in_phrase >= NUM_BARS)
            {
                bar_in_phrase = 0;
                ++phrase_count;
                generate_phrase(current_seed);
            }
        }
    }

    void render_sample(uint32_t target_seed, float t_depth, float t_tension,
                       float t_mystery, float t_vol, bool t_enabled,
                       float &out_l, float &out_r)
    {
        // Smooth adaptive parameters & volume/mute slew to eliminate clicks
        constexpr float PARM_SLEW = 1.0f / (0.250f * SAMPLE_RATE);
        constexpr float VOL_SLEW = 1.0f / (0.050f * SAMPLE_RATE);

        smooth_depth += PARM_SLEW * (t_depth - smooth_depth);
        smooth_tension += PARM_SLEW * (t_tension - smooth_tension);
        smooth_mystery += PARM_SLEW * (t_mystery - smooth_mystery);
        smooth_vol += VOL_SLEW * (t_vol - smooth_vol);
        smooth_mute += VOL_SLEW * ((t_enabled ? 1.0f : 0.0f) - smooth_mute);

        if (target_seed != current_seed)
        {
            current_seed = target_seed;
            rng = Mulberry32(current_seed);
            bar_in_phrase = 0;
            step_in_bar = 0;
            step_sample_accum = 0.0f;
            initialized_phrase = false;
        }

        if (!initialized_phrase)
        {
            initialized_phrase = true;
            generate_phrase(current_seed);
        }

        const float seed_bpm_bias = static_cast<float>((current_seed >> 4) & 0xFFu) / 255.0f;
        const float bpm = clampf(56.0f + 10.0f * seed_bpm_bias + 12.0f * smooth_tension,
                                 56.0f, 78.0f);
        // 16th-note step duration in samples
        const float step_samples = ((60.0f / bpm) * 0.25f) * SAMPLE_RATE;

        if (step_sample_accum <= 0.0f)
        {
            trigger_step(bpm);
            step_sample_accum += step_samples;
        }
        step_sample_accum -= 1.0f;

        // Dispatch humanized sample-accurate pending events
        for (PendingEvent &ev : queue)
        {
            if (!ev.active)
                continue;
            if (ev.countdown <= 0)
            {
                fire_event(ev);
                ev.active = false;
            }
            else
            {
                --ev.countdown;
            }
        }

        // Sum 5 instruments + delay/reverb sends
        float dry_l = 0.0f;
        float dry_r = 0.0f;
        float delay_in_l = 0.0f;
        float delay_in_r = 0.0f;
        float rev_in_l = 0.0f;
        float rev_in_r = 0.0f;

        for (DroneVoice &dv : drones)
        {
            float dr = 0.0f;
            const float dl = dv.process(smooth_tension, dr);
            dry_l += dl;
            dry_r += dr;
            rev_in_l += dl * 0.25f;
            rev_in_r += dr * 0.25f;
        }

        for (PluckVoice &pv : plucks)
        {
            float pl = 0.0f, pr = 0.0f;
            pv.process(pl, pr);
            dry_l += pl;
            dry_r += pr;
            delay_in_l += pl * 0.35f;
            delay_in_r += pr * 0.35f;
            rev_in_l += pl * 0.35f;
            rev_in_r += pr * 0.35f;
        }

        for (FMBellVoice &bv : bells)
        {
            float bl = 0.0f, br = 0.0f;
            bv.process(bl, br);
            dry_l += bl;
            dry_r += br;
            delay_in_l += bl * 0.45f;
            delay_in_r += br * 0.45f;
            rev_in_l += bl * 0.50f;
            rev_in_r += br * 0.50f;
        }

        for (CelloVoice &cv : cellos)
        {
            float cr = 0.0f;
            const float cl = cv.process(cr);
            dry_l += cl;
            dry_r += cr;
            rev_in_l += cl * 0.40f;
            rev_in_r += cr * 0.40f;
        }

        for (TimpaniVoice &tv : timpanis)
        {
            const float ts = tv.process(rng);
            dry_l += ts * 0.707f;
            dry_r += ts * 0.707f;
        }

        for (DropletVoice &wv : droplets)
        {
            float wl = 0.0f, wr = 0.0f;
            wv.process(rng, wl, wr);
            dry_l += wl * 0.25f;
            dry_r += wr * 0.25f;
            rev_in_l += wl * 0.85f;
            rev_in_r += wr * 0.85f;
        }

        float del_l = 0.0f, del_r = 0.0f;
        delay.process(delay_in_l, delay_in_r, bpm, del_l, del_r);

        float rev_l = 0.0f, rev_r = 0.0f;
        reverb.process(rev_in_l + del_l * 0.30f, rev_in_r + del_r * 0.30f,
                       smooth_depth, rev_l, rev_r);

        const float wet_mix = lerpf(0.28f, 0.60f, smooth_depth);
        float mix_l = dry_l + del_l * 0.50f + rev_l * wet_mix;
        float mix_r = dry_r + del_r * 0.50f + rev_r * wet_mix;

        comp.process(mix_l, mix_r);

        const float master = smooth_vol * smooth_mute * 0.72f;
        out_l = flush_denormal(clampf(mix_l * master, -0.98f, 0.98f));
        out_r = flush_denormal(clampf(mix_r * master, -0.98f, 0.98f));
    }
};

// =========================================================================
// Shared Atomic Control State & SDL2 Audio Backend
// =========================================================================

std::atomic<uint32_t> g_seed{0x44435353u};
std::atomic<float> g_depth{0.15f};
std::atomic<float> g_tension{0.10f};
std::atomic<float> g_mystery{0.35f};
std::atomic<bool> g_enabled{true};
std::atomic<float> g_volume{0.70f};

BgmEngine *g_engine = nullptr;

#if defined(USE_SOUND) && defined(USE_SDL)
SDL_AudioDeviceID g_audio_dev = 0;

static void SDLCALL _bgm_sdl_callback(void * /*userdata*/, Uint8 *stream, int len)
{
    const int num_floats = len / static_cast<int>(sizeof(float));
    float *out = reinterpret_cast<float *>(stream);
    if (!g_engine || num_floats < 2)
    {
        std::memset(stream, 0, static_cast<size_t>(len));
        return;
    }
    bgm_render_stereo(out, num_floats / 2);
}
#endif

static float _branch_base_depth(branch_type br)
{
    switch (br)
    {
    case BRANCH_DUNGEON:
        return lerpf(0.05f, 0.42f, clampf((you.depth - 1) / 14.0f, 0.0f, 1.0f));
    case BRANCH_TEMPLE:
        return 0.22f;
    case BRANCH_ORC:
    case BRANCH_LAIR:
        return 0.30f + clampf((you.depth - 1) * 0.03f, 0.0f, 0.14f);
    case BRANCH_SWAMP:
    case BRANCH_SHOALS:
    case BRANCH_SNAKE:
    case BRANCH_SPIDER:
        return 0.48f + clampf((you.depth - 1) * 0.04f, 0.0f, 0.15f);
    case BRANCH_ELF:
    case BRANCH_SLIME:
    case BRANCH_VAULTS:
        return 0.58f + clampf((you.depth - 1) * 0.04f, 0.0f, 0.16f);
    case BRANCH_CRYPT:
    case BRANCH_DEPTHS:
        return 0.72f + clampf((you.depth - 1) * 0.04f, 0.0f, 0.16f);
    case BRANCH_TOMB:
    case BRANCH_VESTIBULE:
    case BRANCH_DIS:
    case BRANCH_GEHENNA:
    case BRANCH_COCYTUS:
    case BRANCH_TARTARUS:
    case BRANCH_ZOT:
    case BRANCH_ABYSS:
    case BRANCH_PANDEMONIUM:
    case BRANCH_ZIGGURAT:
        return 0.86f + clampf(you.depth * 0.02f, 0.0f, 0.14f);
    default:
        // Portal vaults & mini-branches
        return 0.52f;
    }
}

static float _branch_base_mystery(branch_type br)
{
    switch (br)
    {
    case BRANCH_TEMPLE:
    case BRANCH_ELF:
    case BRANCH_WIZLAB:
        return 0.72f;
    case BRANCH_VAULTS:
    case BRANCH_CRYPT:
    case BRANCH_SLIME:
    case BRANCH_TOMB:
        return 0.65f;
    case BRANCH_ABYSS:
    case BRANCH_PANDEMONIUM:
    case BRANCH_ZOT:
    case BRANCH_DESOLATION:
        return 0.88f;
    case BRANCH_BAZAAR:
    case BRANCH_OSSUARY:
    case BRANCH_ICE_CAVE:
    case BRANCH_VOLCANO:
        return 0.60f;
    default:
        return 0.35f;
    }
}

} // anonymous namespace

void bgm_init()
{
    if (!g_engine)
        g_engine = new BgmEngine();

    g_enabled.store(Options.bgm_on, std::memory_order_relaxed);
    bgm_set_volume(Options.bgm_volume);
    bgm_update();

#if defined(USE_SOUND) && defined(USE_SDL)
    if (g_audio_dev != 0)
        return;

    if (!(SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO))
    {
        if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0)
            return;
    }

    SDL_AudioSpec want{};
    SDL_AudioSpec have{};
    want.freq = static_cast<int>(SAMPLE_RATE);
    want.format = AUDIO_F32SYS;
    want.channels = 2;
    want.samples = 1024;
    want.callback = _bgm_sdl_callback;
    want.userdata = nullptr;

    g_audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
    if (g_audio_dev != 0)
        SDL_PauseAudioDevice(g_audio_dev, 0);
#endif
}

void bgm_shutdown()
{
#if defined(USE_SOUND) && defined(USE_SDL)
    if (g_audio_dev != 0)
    {
        SDL_PauseAudioDevice(g_audio_dev, 1);
        SDL_CloseAudioDevice(g_audio_dev);
        g_audio_dev = 0;
    }
#endif
    delete g_engine;
    g_engine = nullptr;
}

bool bgm_is_active()
{
#if defined(USE_SOUND) && defined(USE_SDL)
    return g_audio_dev != 0;
#else
    return g_engine != nullptr;
#endif
}

void bgm_set_enabled(bool enabled)
{
    g_enabled.store(enabled, std::memory_order_relaxed);
}

void bgm_set_volume(int percent)
{
    const float vol = clampf(static_cast<float>(percent) / 100.0f, 0.0f, 1.0f);
    g_volume.store(vol, std::memory_order_relaxed);
}

void bgm_update_params(uint32_t seed, float depth, float tension, float mystery)
{
    g_seed.store(seed ? seed : 0x44435353u, std::memory_order_relaxed);
    g_depth.store(clampf(depth, 0.0f, 1.0f), std::memory_order_relaxed);
    g_tension.store(clampf(tension, 0.0f, 1.0f), std::memory_order_relaxed);
    g_mystery.store(clampf(mystery, 0.0f, 1.0f), std::memory_order_relaxed);
}

void bgm_update()
{
    g_enabled.store(Options.bgm_on, std::memory_order_relaxed);
    bgm_set_volume(Options.bgm_volume);

    if (!crawl_state.game_started || you.hp_max <= 0)
    {
        bgm_update_params(0x44435353u, 0.15f, 0.08f, 0.35f);
        return;
    }

    const uint64_t raw_seed = you.game_seed ? you.game_seed : 0x243f6a8885a308d3ULL;
    const uint64_t mixed = raw_seed
        ^ (static_cast<uint64_t>(you.where_are_you) * 0x9e3779b97f4a7c15ULL)
        ^ (static_cast<uint64_t>(you.depth) * 0xbf58476d1ce4e5b9ULL);
    const uint32_t floor_seed = static_cast<uint32_t>(mixed ^ (mixed >> 32));

    const float depth = _branch_base_depth(you.where_are_you);

    // Derive tension from Crawl's tactical tension metric, HP deficit, and Orb run
    const int crawl_tension = get_tension(GOD_NO_GOD);
    const float monster_t = clampf(static_cast<float>(crawl_tension) / 70.0f, 0.0f, 0.65f);
    const float hp_frac = clampf(static_cast<float>(you.hp) / static_cast<float>(you.hp_max),
                                 0.0f, 1.0f);
    const float hp_t = clampf((0.80f - hp_frac) * 0.70f, 0.0f, 0.45f);
    float orb_t = player_on_orb_run() ? 0.55f : 0.0f;
    if (you.berserk())
        orb_t = std::max(orb_t, 0.35f);

    const float tension = clampf(0.08f + monster_t + hp_t + orb_t, 0.0f, 1.0f);

    float mystery = _branch_base_mystery(you.where_are_you);
    if (you.invisible())
        mystery = std::min(1.0f, mystery + 0.10f);

    bgm_update_params(floor_seed, depth, tension, mystery);
}

void bgm_render_stereo(float *out_interleaved, int num_frames)
{
    if (!out_interleaved || num_frames <= 0)
        return;

    if (!g_engine)
        g_engine = new BgmEngine();

    const uint32_t seed = g_seed.load(std::memory_order_relaxed);
    const float depth = g_depth.load(std::memory_order_relaxed);
    const float tension = g_tension.load(std::memory_order_relaxed);
    const float mystery = g_mystery.load(std::memory_order_relaxed);
    const float vol = g_volume.load(std::memory_order_relaxed);
    const bool enabled = g_enabled.load(std::memory_order_relaxed);

    for (int i = 0; i < num_frames; ++i)
    {
        float l = 0.0f, r = 0.0f;
        g_engine->render_sample(seed, depth, tension, mystery, vol, enabled, l, r);
        out_interleaved[i * 2] = l;
        out_interleaved[i * 2 + 1] = r;
    }
}
