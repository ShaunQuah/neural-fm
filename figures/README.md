# Figure legends

Figure numbers follow the file names and the notebooks (`fig1` to `fig6`). The legends for the architecture diagram and
Figures 1, 5 and 6 are in the [main README](../README.md), next to the figures. The legends below cover the rest.
Abbreviations (co-bps, GLM, NLB, SD and others) are defined in the README's
[Abbreviations](../README.md#abbreviations) table. In all figures, "val" means the official validation split
(573 trials) and "GLM (ref.)" is the generalised linear model with the NLB reference settings (50 ms smoothing, alpha 0.01).

## Figure 2: per-neuron scores (`fig2_per_neuron.png`)

Each point is one of the 45 held-out neurons. The x axis is its co-smoothing bits per spike (co-bps) under the reference
GLM, and the y axis is its co-bps under the 5-seed transformer ensemble (the average of the 5 seeds' predicted rates),
both on the validation split. The dashed line marks equal scores, so points above it are neurons the transformer
predicts better. Green points are neurons that improved (44 of 45); the one red point is a neuron that got worse. Point
size is proportional to the neuron's number of validation spikes, so small points are noisy estimates. Per-neuron
scores run higher than the overall co-bps because the overall score is weighted by spike count and the busiest neurons
(large points) mostly score lower.

## Figure 3: trial-averaged predictions (`fig3_trial_averages.png`)

Firing rate over the 700 ms trial, averaged across all validation trials, for six held-out neurons chosen to span the range
of spike counts. Each panel title gives the neuron index and its total validation spikes. Black: the actual
spikes per 5 ms bin, averaged over trials. Grey: the reference GLM. Green: the 5-seed transformer ensemble. The
models never saw these neurons' validation spikes; they predicted them from the 137 held-in neurons.

## Figure 4: single-trial predictions (`fig4_single_trials.png`)

Predicted firing rate (spikes per 5 ms bin) for held-out neuron 28, the neuron with the most validation spikes, on
three validation trials (rows). Lines are the reference GLM (grey) and the 5-seed transformer ensemble (green).
Ticks along the bottom are the neuron's actual spikes on that trial. Predictions are smooth rates, not spike trains:
spiking is noisy and only the part shared with the held-in neurons can be predicted.

## Step 3: baselines to beat (`fig_step3_baselines.png`)

Validation co-bps of the three reference points. Bars are scores and whiskers are 95% bootstrap confidence intervals
over validation trials; the mean-rate null has no interval because it is re-estimated on every resample. "GLM 40 ms /
alpha 0.1" is the first setting I tried (smoothing width 40 ms, regularisation strength 0.1) and "GLM 50 ms / alpha
0.01" is the NLB reference setting, which is the baseline used in the rest of the project. "NLB-official" refers to the
official definition of the metric (null model = mean rate of the evaluated spikes).

## Step 4: hyperparameter sweep (`fig_step4_dev_sweep.png`)

Co-bps on the internal dev fold (15% of the training trials held out for model selection; the official validation split
is not used) for eight training configurations, each with a single seed (seed 0). The orange bar is the configuration
that was selected. Labels: do = dropout, wd = weight decay, mask = mask ratio (share of time bins hidden during
training), lr = learning rate (5e-4 = 0.0005), long400 = a 400-epoch training budget, base = the starting
configuration. With one seed per bar, small gaps are within noise.
