# neural-fm: results summary

Masked-reconstruction spike transformer on NLB MC_Maze (co-smoothing). All scores are **co-bps** (NLB-official
definition) on the **public validation split** (573 trials), scored once per run. Hyperparameters, learning rates and early
stopping were chosen on an internal dev fold (15% of the 1,721 training trials). Selection was not perfectly blind: the first two
step-4 sweeps and the discarded first pretraining run also printed validation scores (for step 4 the dev fold picked the same winner).

Generated from the saved files in `results/` on 2026-10-30; status section updated afterwards. Status: steps 1-7, README and lint/test CI
complete; serving API and container image not started.

## 1. Final evaluation (step 5)

Frozen config: 4 layers, 4 heads, d_model 128, dropout 0.5, weight decay 0.1, mask ratio 0.25, span 5; 852,534 parameters.

| Model | co-bps |
|---|---|
| Mean-rate null | -0.001 |
| Smoothing + Poisson GLM, 40 ms / alpha 0.1 | 0.188 [0.181, 0.193] |
| Smoothing + Poisson GLM, 50 ms / alpha 0.01 (NLB reference settings) | 0.212 [0.203, 0.219] |
| **Transformer, 5 seeds** | **0.335 ± 0.004** (SD; seeds 0.331, 0.337, 0.340, 0.331, 0.335) |
| Transformer, 5-seed ensemble (extra) | 0.352 [0.340, 0.361] |

* Paired gain over the reference GLM: +0.119 to +0.128 per seed, every bootstrap interval above 0; +0.140 for the ensemble.
* Per-neuron (step-5 notebook, 5-seed ensemble vs the reference GLM): 44 of 45 held-out neurons improved. A re-check from the saved
  checkpoints gave 44, 44, 44, 43, 44 of 45 for the individual seeds.
* Published NLB'21 Table 1 for context (NDT 0.329, AutoLFADS 0.346): **not a like-for-like comparison**. The split used by the
  paper's table is unconfirmed, and mine is the public validation split. Do not claim to beat it; quote the 5-seed mean.

## 2. Ablations (step 6; 3 seeds, one change at a time)

Seed SD is mostly 0.002-0.006 (0.010 for d_model 256), so differences below roughly 0.005 are noise. Mask ratio is the nominal
span-start rate; the realised masked fraction is lower (about 0.095 / 0.22 / 0.40 for ratio 0.10 / 0.25 / 0.50 at span 5).

| Change | Val co-bps | Change vs baseline | Verdict |
|---|---|---|---|
| Baseline | 0.336 | | |
| No masking | 0.249 | -0.087 | essential |
| No position embedding | 0.136 | -0.200 | essential (below the GLM) |
| Mask ratio 0.10 / 0.50 | 0.321 / 0.340 | -0.015 / +0.004 | 0.10 worse; 0.50 within noise |
| Mask span 1 / 10 | 0.329 / 0.335 | -0.007 / -0.001 | not reliable |
| 1 / 2 heads | 0.335 / 0.336 | -0.002 / -0.000 | no difference from 4 |
| 8 heads | 0.323 | -0.013 | worse in all 3 seeds |
| 1 / 2 / 6 layers | 0.262 / 0.311 / 0.345 | -0.074 / -0.025 / +0.009 | depth helps; 6 layers small gain |
| d_model 64 / 256 | 0.291 / 0.327 | -0.045 / -0.009 | 64 too small; 256 slightly worse (seed SD 0.010, so only weak evidence) |

Caveats: 3 seeds only; baseline dropout and weight decay were tuned at 4 layers / d_model 128; the frozen config was not changed
on the basis of these validation numbers (dev-fold ranking agrees).

## 3. Pretrain then adapt (step 7; 5 seeds)

Pretraining uses held-in spikes only (no held-out labels). Adaptation uses a fraction of held-out labels. Every arm has the same
labelled subsets, an 8,000-step budget, and dev-chosen learning rate (GLM: dev-chosen alpha).

| Labelled trials | GLM | Scratch | Pretrained + fine-tune | Pretrained, frozen + probe |
|---|---|---|---|---|
| 10% (146) | 0.139 | 0.023 | **0.176** | 0.182 (seeds 0.11-0.25) |
| 25% (366) | 0.171 | 0.204 | **0.261** | 0.252 |
| 50% (732) | 0.186 | 0.283 | **0.304** | 0.274 |
| 100% (1,463) | 0.209 | 0.332 | **0.337** | 0.282 |

Paired gain of fine-tune over scratch: +0.153 (10%), +0.057 (25%), +0.021 (50%), +0.005 (100%); positive in 5/5, 5/5, 4/5, 4/5 seeds.
Over the GLM on the same labels: +0.036, +0.090, +0.118, +0.128, positive in every seed.

* Pretraining helps label efficiency when labels are scarce; there is **no evidence of a gain at full labels**.
* The scratch transformer at 10% labels (0.023) is worse than the GLM (0.139), so the fair headline gain there is +0.04 over the GLM.
* The frozen probe is competitive at 10-25% but varies a lot across seeds and falls behind as labels grow.

## 4. Claims supported and not supported

Supported: the transformer clearly beats the NLB-reference GLM on this split; masking, position information and depth matter;
pretraining improves label efficiency for held-out neurons.

Not supported (yet): beating published NLB numbers; a "foundation model" claim (one animal, one session, pretraining on the same
trials as the labels, no cross-session or cross-dataset test); any claim from the untested train/test mismatch (held-out outputs are
trained from masked inputs, tested on unmasked inputs).

## 5. Process notes

* A first pretrain run (single learning rate) was discarded because the scratch arm was unstable at full data (0.249 ± 0.085 vs 0.336
  in step 5). Redone with per-fraction learning rates; old runs kept in `results/pretrain_v1_scratch_undertuned/`.
* The probe was changed to train unmasked after a masked-train/unmasked-test failure, diagnosed on the dev fold.
* Earlier wrong claims were corrected: dataset size, baseline parameters, a "matches published 0.19" statement.

## 6. State of the repository

* Code: `src/neuralfm/` (model, data, metrics, baselines, run_baselines, train, ablate, pretrain, glm_fraction); tests: 4 pass.
* Notebooks: step 3-4, 5, 6, 7. Figures: architecture and fig1-fig6. Results: about 264 MB in `results/` on disk (checkpoints and the data cache are untracked).
* `results/baselines.json` and `run_baselines.py` are current (null, 40 ms / 0.1 and the 50 ms / 0.01 reference GLM).
* The `val_co_bps` key means different things in different files: in `train.py` run JSONs (`results/step5/final_s*.json`, `results/*.json`)
  it uses the stricter training-mean null; in `ablate.py`, `pretrain.py`, `glm_fraction.py` and `step5_results.json` it is the NLB-official score
  (`val_co_bps_official` in the ablation run files). All numbers quoted above are the official definition.
* MLflow: logged for train.py runs (steps 4-6; `results/step5` shows 15 runs for 5 final seeds, probably notebook re-runs); `pretrain.py` does not log to MLflow.
* Not started: FastAPI service and tests, Dockerfile, second dataset, cloud deployment.
