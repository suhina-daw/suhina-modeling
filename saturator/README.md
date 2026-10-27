# Saturator reference measurements

Measures eight curve modes at the captured settings using the production Rust device.
This is a focused research workflow, not the proposed general modeling toolkit.
Captured audio, projects, source paths, inventories and listening packages stay local.

## Inputs and provenance

Set `SUHINA_SATURATOR_CAPTURES` to the folder containing the processed WAVs and `dry/`.
Set `SUHINA_SATURATOR_WORK` to an output folder (default `target/saturator`).
Set `SUHINA_SATURATOR_PROBE` to the release `saturator_probe` executable built from the
device crate in the application repository. Use a separate executable for each DSP revision.

The inventory reads a self-authored `.als` project supplied with `--project`. It records
actual Type events, clip positions, gains and source offsets, not rounded bar boundaries.
The current capture convention is stereo 48 kHz, 120 BPM, eight modes in ascending order,
unwarped, untransposed, non-looped clips without fades and a unity track mixer. Unsupported
transforms are refused. A source is matched to `dry/` by SHA-256. If that copy is missing,
the exact project source is used and that fallback is recorded. Similar names never prove
equivalent content. Source offsets of unwarped clips are seconds; arrangement times are beats.

The project proves saved settings, not which settings were used for an earlier export.
Check reconstructed input against the captures before fitting. The main fader value is
recorded but is not applied to individual track exports. Verify export calibration separately
on a quiet, approximately linear passage before interpreting gain differences as DSP errors.

## Commands

From the repository root, using Python with `requirements.txt` installed:

```text
python -m saturator inventory --project <capture-project.als>
python -m saturator render --label baseline --commit <full-DSP-commit>
python -m saturator compare --label baseline
python -m saturator render --label candidate --commit <full-DSP-commit>
python -m saturator compare --label candidate
python -m saturator listen --name listening-01 baseline candidate
python -m saturator.candidate_report baseline candidate
python -m saturator.waveshaper_captures
python -m saturator.curve_sweep
python -m unittest saturator.test_measurements saturator.test_probe -v
```

Inventories, render directories and listening packages refuse to overwrite existing results.
Each render stores the input WAV, schedule, executable hash, inventory hash and output hash.
The commit argument identifies the DSP baseline; the executable hash also distinguishes an
uncommitted research build. Preserve its patch alongside the results when evaluating a candidate.
The probe integration tests run when `SUHINA_SATURATOR_PROBE` is set; otherwise they are
explicitly skipped. The A/B/C diagnostic expects independently rendered `waveshaper-A`,
`waveshaper-B`, `waveshaper-C` folders containing numbered track exports. It records their
hashes and measures `C - (2*A + B)` without alignment or gain fitting, testing whether the
isolated contributions add. It does not infer the Curve control's full mapping.
The Curve sweep diagnostic additionally reads `waveshaper-D` and `waveshaper-E` exports:
Curve 25 % and 50 %, Linear 0 %, other settings unchanged from B (Curve 5 %, Linear 0 %).
It needs tracks 2 (triangle steps) and 3 (sine steps), at least the first 16 seconds each.
Its base-rate research prediction is separate from the native production-device comparison.

## Current research status (2026-10-28)

The measured baseline is application PR #900 at
`a7ffab52a4170623e6941ad593196315ff62e042`. The native probe is an example in that
application worktree (`cargo build --release -p suhina-device-saturator --example saturator_probe`);
it must be available in the selected application revision before running `render`.
These Python tools do not contain or install the production DSP corrections.

The 17-track comparison supports two corrections: a quadratic shoulder for Medium Curve,
and independently additive Linear/Curve contributions with DC rejection for Waveshaper.
Both candidates reduced waveform and spectral error in every measured steady-state clip
of their respective modes. Separate four-source listening packages (sine, bass, vocal,
snare) received user listening acceptance at the captured settings.

The Waveshaper model predicts the isolated Curve captures at 5 %, 25 % and 50 % without
refitting to the latter two. This does not validate Curve above 50 %, other Drive levels,
Color, non-default Damp/Depth/Period/Shaper Drive, or a different quality setting.
Application merge readiness and the remaining parameter research are separate decisions.
The production changes were still local at this research snapshot; this document does not
claim that PR #900 contains them or that all application-wide checks passed.

## Interpretation

`comparison.json` reports every mode and clip separately, with holdout membership, fixed
delay, gain error, sample error, spectral error and detected quiet-period output. The Markdown
table is a navigation summary; it does not replace the per-clip results. The first 100 ms after
a mode change is reported separately. Active windows come only from the dry input, at an
absolute energy floor of -100 dBFS over 10 ms. All candidates use the same windows.

One common delay per track is estimated from its first-mode clip. Fractional compensation uses
a windowed-sinc filter. This is an alignment aid, not phase equalization; strongly periodic
sources can make the estimate ambiguous. Raw output is retained. No gain normalization is used
in measurements. Spectral error uses contiguous windows, never samples concatenated across gaps.
Harmonic measurements require a stationary sine verified from the dry input. Harmonic content
already present in saw/square/musical inputs is not reported as device THD. Aliasing requires
an appropriate separate known-frequency probe; arbitrary residual energy is not labeled aliasing.
When the fundamental is suppressed below -100 dB relative to the fitted harmonics (or an
absolute amplitude of 1e-12), THD is null with an explicit reason; amplitudes remain available.

The sine and synthetic stepped waveforms are fitting material. Musical sources are holdouts.
Inspect per-mode and per-source regressions before accepting a change; an improved average alone
does not pass a candidate. This capture fixes one input-drive setting and default Waveshaper
amounts, so it cannot validate other settings or identify every model coefficient.

Listening packages contain original-level and constant-gain-matched versions of the same
interval. The matched versions share a conservative 4x-estimated -1 dBFS ceiling; no limiter
or time-varying gain is used. A Finnish instruction file and hashes accompany the WAVs.
