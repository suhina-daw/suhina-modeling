# A device-agnostic modeling toolkit

Status: proposal, not built. Nothing has moved yet.

The reverb research (`reverb/`) built a model, fitted it against one captured reference IR and
built tools around the fit: paths behind environment variables, a level-matched listening
package, comparison pages on a common time axis, onset kurtosis and an attribute-based objective
with held-out sources. This document separates what in that work fits any device from what each
device needs for itself, proposes a shared package beside `reverb/` and lists the tests it needs.

## 1. What is generic

These parts know nothing about reverbs. They move into a shared package and every device folder
uses them.

**Capture folder convention.** Each device reads its captures from `SUHINA_<DEVICE>_CAPTURES` and
writes to `SUHINA_<DEVICE>_WORK` (default `target/<device>`). A native probe, when the device has
one, comes from `SUHINA_<DEVICE>_PROBE`. `SUHINA_REPO` is shared. The reverb names
(`SUHINA_REVERB_*`) already follow this pattern and keep working. Required variables have no
default and fail on first use with the variable's name, as `reverb/paths.py` does now. Nothing is
created on import. A captures folder holds:

```text
<captures>/
  capture.json          sample rate, pre-roll before the stimulus (s), input calibration (dBFS
                        of the stimulus at the device input), output gain, the settings captured
  stimuli/<name>.wav    the dry, self-authored stimuli exactly as played
  <setting>/<name>.wav  the capture of each stimulus at each setting
```

`capture.json` replaces what the reverb code hard-codes: the 1 s pre-roll (`a[FS:]`) and the
`/ .25` gain. Captures stay uncommitted (`*.wav`, `*.npy` in `.gitignore`). Stimuli are generated
by the toolkit (`stimuli.py`), so a stimulus can be rebuilt from its name and seed and its hash
checked against the capture's manifest.

**Listening package.** `make_listening.py` becomes a library function plus a thin command. A
version is either an IR (`label=ir.npy`, convolved with each dry source as now) or a folder of
rendered WAVs (`label=renders/`, one file per source with the source's name). Nonlinear and
time-varying devices can only use the second form. The rest stays as it is: A is the captured
reference, then B, C, D ... in argument order, so the files sort in the order Juha stacks them in
his DAW; one common max-300 ms-RMS target per source with a 4x-oversampled -1 dBFS peak ceiling;
no limiter or time-varying gain; refuses to overwrite; a verification manifest with SHA-256 of
every source and output file; `00 - START HERE.txt`. The source list (`SPECS`, now hard-coded)
comes from the device folder.

One rule changes for nonlinear devices. The reverb energy-matches each candidate to A before the
common target. That is right for a linear device, where level is only a gain. For a nonlinear
device the input level is part of the sound: the renders use the same calibrated input level as
the capture, and the package matches levels only at the output, for listening. The measurements
never normalise the level away.

**Comparison figures.** `compare_ir.py` and `compare_ir2.py` become functions that take a
reference and any number of candidates (labelled, in the same A, B, C order) on one time axis.
Two page kinds:

- *IR pages* (linear devices): impulse dB, ETC, step response, Schroeder decay, smoothed
  magnitude, spectrograms, L/R correlation and coherence, octave EDC, C50/C80. These exist now.
- *Signal pages* (any device, from renders): the waveform and RMS envelope of each source, its
  spectrum and spectrogram, the difference against A, plus the device-specific measurement plots
  in section 2.

Before any figure or measurement compares two signals, `align()` estimates the delay between them
by cross-correlation and removes it, so latency (lookahead, oversampling filters) is not counted
as a difference. The delay found is reported.

**Measurements that are not reverb-specific.** Band T60 and EDC, ETC, clarity, L/R coherence,
the kurtosis of 10 ms frames (discrete versus diffuse onsets), RMS envelope error, band power
balance, modulation error (`fit_perceptual.modulation_error`). They move into
`measures/ir.py` and `measures/signal.py`.

**Attribute-based objective with held-out sources.** The shape of `perceptual_objective.py` is
generic: a `Reference` that precomputes its features once, a set of named terms each compared
with the reference's own value, a fixed scale per term so a typical value is near 1, and weights.
The terms themselves are per device. The shared part provides:

- `Objective(terms, scales, weights)` with `objective(x) -> (score, terms)`;
- `Source(name, family, holdout)`, and an evaluation that weights each family equally (as
  round 13 weights the originals and the pitch bank), never lets a held-out source into the
  score and reports it separately;
- the eligibility guard from `run_five_rounds.py`: a candidate is eligible only if no family's
  mean regresses by more than a set margin and no single case by more than another, against a
  baseline.

## 2. What each device needs

A device is **linear and time-invariant** when its output is fully described by an impulse
response. Then the reverb approach carries over almost as is: measure the IR (impulse or
log sweep), render candidates by convolution, fit in Python. A device that is **nonlinear** or
**time-varying** has no single IR: candidates must be rendered natively through a probe, and the
device needs its own measurements.

| Device | Kind | Renders | Measurements it needs |
|---|---|---|---|
| `reverb` (chorus off, freeze off) | linear | convolution with the model's IR | what exists |
| `dj-eq` (kills included), filters | linear | IR | magnitude and phase per band, crossover sums |
| `nam-cab-loader` | linear | IR | magnitude, the effect of the tone cuts and delay |
| `utility` (gain, width, bass mono) | linear, stereo | 2x2 IR matrix | M/S balance per band |
| `delay`, static times | linear | IR (long) | tap times and levels, band-pass in the loop |
| `delay` with LFO, Repitch/Fade/Jump | time-varying | probe | delay-time tracking, the transition on a time change |
| `reverb` with chorus | time-varying | probe | modulation tracking of the tail |
| `auto-pan` | time-varying | probe | gain and pan tracking of a steady sine: rate, depth, shape |
| `overdrive`, `vinyl-distortion` | nonlinear | probe | static curve, harmonics against level and frequency, aliasing |
| `vinyl-distortion` crackle | random | probe, fixed seed | rate and level spectrum of the clicks, statistics only |
| `compressor`, `limiter` | nonlinear, level-dependent | probe | static gain curve, attack and release from level steps, overshoot |

Instruments (`fm-synth`, `sampler`) and MIDI effects (`pitch`) take notes, not audio. They are
left out until the probe can play notes.

The device-specific measurements, all from self-authored stimuli:

- **Distortion against level.** Stepped sines (for example -40 to +6 dBFS in 2 dB steps at a few
  frequencies). Per step: output level (the static curve), the level of each harmonic, THD,
  and the inharmonic residue (aliasing of an oversampled shaper). A log sweep separates the
  harmonic IRs in time and gives the same per frequency.
- **Attack and release.** Level steps of a steady tone (for example -30 to -6 dBFS and back). The
  gain envelope (output envelope over input envelope) gives the time constants as the time to
  63 % of the step, the overshoot, and for the limiter the peak against the ceiling.
- **Modulation tracking.** A steady sine (for gain and pan) or a broadband signal (for delay
  time) through the device. Windowed cross-correlation or the instantaneous phase gives the delay
  or gain over time; its rate, depth and shape are compared. The LFO phase of a capture and of a
  render are not related, so these comparisons use phase-invariant values (rate, depth, the shape
  over one cycle), never sample by sample.

## 3. Native renders: the probe

The reverb has its own probe (`crates/dsp/examples/fdn_probe.rs`). Writing one per device does
not scale. Every built-in device implements the same `Device` trait, so one probe in the main
repository could render any of them: a device id, an input WAV, parameter values and a tail,
and a WAV out (for example `suhina-cli device run <device-id> <in.wav> <out.wav> --set
<param>=<value> --tail <seconds>`, beside the existing `plugin run`). That is work in
suhina-daw/suhina-daw, its own issue; the toolkit only needs the contract. `probe.py` here wraps
it: it writes the input, runs the executable from `SUHINA_<DEVICE>_PROBE`, reads the output and
checks the sample rate, the channel count and the length.

## 4. Proposed layout

```text
suhina_modeling/        the shared package; no device knowledge
  paths.py              device_paths('<device>') -> captures, work, probe; Missing; nothing on import
  audio.py              read and write WAV (integer to float, sample rate and channel checks)
  captures.py           capture.json, finding stimuli and captures, pre-roll and gain
  stimuli.py            impulse, log sweep and its inverse, stepped sines, level steps, noise
                        bursts, tone bursts; deterministic from a name and a seed
  levels.py             max-RMS, 4x true peak, energy match, the common package target
  align.py              delay between two signals by cross-correlation
  listening.py          the package builder (IR or rendered versions)
  figures.py            IR pages and signal pages on a common time axis
  objective.py          Objective, Source, family-weighted evaluation, eligibility guard
  probe.py              runs a native probe and checks its output
  measures/
    ir.py               EDC, band T60, ETC, clarity, coherence, kurtosis
    signal.py           envelope error, band balance, modulation error
    nonlinear.py        static curve, harmonics, THD, aliasing residue
    dynamics.py         gain envelope, attack and release, overshoot
    modulation.py       delay and gain tracking, rate, depth, shape
  cli.py                `python -m suhina_modeling listen ...`, `... compare ...`
reverb/                 unchanged at first; then imports the shared parts instead of its copies
<device>/               each new device: README (its variables), sources.json, terms, data/
tests/                  pytest, synthetic signals only
.github/workflows/tests.yml
```

The reverb folder moves to the shared code one script at a time, after the shared package and
its tests are in. Each step keeps the reverb's own outputs the same: a listening package built
both ways from the same inputs has the same file names and the same output hashes.

## 5. Tests for the shared part

Every test builds its signals in code. No captures, no network, no probe executable.

| Module | Test |
|---|---|
| `paths` | an unset required variable fails with its name; defaults; importing creates nothing |
| `audio` | int16, int32 and float files read to the same float values; a wrong rate or mono file is refused |
| `stimuli` | the same name and seed give the same samples; a sweep convolved with its inverse is an impulse |
| `levels` | a sine at fs/4 with a 45° phase: sample peak 0.707, true peak 1.0; max-RMS of a constant |
| `align` | a known delay (whole and fractional samples) is found |
| `listening` | from synthetic sources and IRs: files sort A, B, C; every group at the same max-RMS within 0.01 dB; every peak at or under -1 dBFS; a second run refuses to overwrite; the manifest hashes match the files; an IR version and a rendered-WAV version of the same convolution give the same files |
| `measures.ir` | exponentially decaying noise with a known T60 is measured within 5 %; Gaussian noise has kurtosis near 3, sparse impulses far above; a known energy split gives its C50 |
| `measures.nonlinear` | a pure gain has no harmonics; a small cubic gives the third harmonic its formula predicts; `tanh` gives a static curve that rises then saturates |
| `measures.dynamics` | a synthetic one-pole compressor with known attack and release is measured within 10 % |
| `measures.modulation` | a sinusoidally modulated delay with known rate and depth is recovered; a random LFO phase does not change the result |
| `objective` | the reference against itself scores 0; changing only a held-out source changes the reported held-out value and not the score; the eligibility guard rejects a family regression over the margin |
| `probe` | a fake probe (a small Python script) is run; wrong rate, channels or length are reported |
| `figures` | each page renders to a PNG (Agg) from synthetic signals |

A licence-header check (`.py` files start with the two header lines) runs with the tests. Other
product names are kept out by review: a check would have to list the names it looks for.

**Workflow.** `.github/workflows/tests.yml` on push and pull request: `ubuntu-latest`, Python
3.12, `pip install -r requirements.txt -r requirements-dev.txt` (pytest), the header check, then
`pytest`. `OPENBLAS_NUM_THREADS=1`. No secrets, no caches beyond pip's.

## 6. Open questions

- The order of the first devices after the move. Linear ones (`dj-eq`, `nam-cab-loader`) need no
  probe and test the shared code first; the compressor or overdrive needs the probe issue in the
  main repository.
- Whether `00 - START HERE.txt` stays in Finnish (it is read only by the listener) or the code
  writes it in English and `--note` carries any Finnish.
