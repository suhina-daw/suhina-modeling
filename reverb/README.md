# Reverb research

Fits and measures the linear model of Suhina's diffused FDN reverb wrapper against one captured
reference impulse response and source captures. The findings are recorded in the main
repository's spec,
[`docs/spec/reverb-prototype.md`](https://github.com/suhina-daw/suhina-daw/blob/main/docs/spec/reverb-prototype.md)
› *Discrete early reflections and fitted input weights*. The model itself (`model.py`) and its
native parity checks are kept in the main repository, `scripts/reverb/model/`, because they change
together with `crates/dsp/src/fdn_diffuse.rs`; these scripts import it from a checkout.

## Setup

```sh
pip install -r requirements.txt
export SUHINA_REPO=/path/to/suhina-daw                 # required: the model comes from here
export SUHINA_REVERB_CAPTURES=/path/to/captures/48k   # required; never committed
export SUHINA_REVERB_WORK=target/reverb               # optional, the default
export OPENBLAS_NUM_THREADS=1                         # the fits run several processes
```

[`paths.py`](paths.py) describes every variable. The captures folder holds the reference
(`* - 6000ms.wav`) and `samples/` with the dry stimuli and their captures. `data/` holds the small
committed inputs: round 18's selected settings (`selected-settings.json`), the E output weights
(`e-output-weights.json`) and the fitted parameters and measurements of the candidates that were
listened to (`round5-*`, `round7-*`, `round8-*`).

Scripts that read earlier results (listening packages, comparisons) read them from
`SUHINA_REVERB_WORK`; run the round that writes them first.

## What is here

| Script | Does |
|---|---|
| `perceptual_objective.py` | the fit objective: band decay relief, onset/body energy, coloration, recurrence, stereo, octave build-up (window `set_buildup(w, window)`), broadband bursts (`set_burst`) |
| `metrics.py`, `ir_diagnostics.py` | shared IR metrics and the reference loader |
| `run_five_rounds.py` | source stimuli, `measure` and `evaluate` used by the evaluations (from round 13) |
| `round2.py` | the four evaluations every candidate gets (fresh40, sources, objective, cases) |
| `round3.py` ... `round8.py` | the fitting rounds; each docstring says what changed and how to run it |
| `early_density.py` | 0-50 ms dips and 10 ms-frame kurtosis (`kurt`) |
| `etc_early.py`, `compare_ir.py`, `compare_ir2.py` | comparison figures on a common time axis: ETC, impulse, step, Schroeder and octave decay, magnitude, spectrogram, L/R correlation, clarity |
| `sine_balance.py`, `sine_static_check.py`, `lr_spectrum.py`, `release_step.py` | single-sine channel balance and release |
| `rew_compare.py` | ISO 3382 values from a running REW (API on port 4735) |
| `make_listening.py` | a level-matched listening package from any IRs: A the reference, then B, C, D ... in folder order (usage in its docstring) |
| `exp*.py`, `fit_*.py`, `native_e.py` | the first day's experiments and the native render of candidate E |

Round 8, the latest:

```sh
python -I reverb/round8.py 4      # seed 4: the best so far
```
