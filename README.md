# MLP Projection Experiment

This prototype simulates adversarial projection updates for a bias-free tanh
MLP. The default mode is the original row-space projection

```text
theta <- theta + Proj_{range(J^T)}(theta_star - theta)
```

where `J` is the batch Jacobian of the flattened network outputs with respect
to the flattened parameter vector.

It can also run input-Hessian-span and combined jet-span modes. In Hessian mode,
sampled points `x_i` define mixed-derivative blocks

```text
H_i = d_x d_theta F_theta(x_i)
```

and the update projects `theta_star - theta` onto the column span of
`[H_1 ... H_N]`. The implementation also constructs the paired points
`x_i' = x_i + delta v_i` and records how closely the finite-difference batch
gradient matches the linearized update.

The combined jet-span mode projects onto the span of both ordinary gradients
and mixed-derivative columns:

```text
[g(x_1) ... g(x_N) H_1 ... H_N]
```

The implementation is intentionally explicit:

- no autodiff dependency
- analytic Jacobian by backpropagation
- analytic mixed derivative `d_x d_theta F_theta(x)` for scalar-output tanh MLPs
- SVD projection onto `range(J^T)`
- SVD projection onto the mixed-derivative column span in Hessian-span mode
- SVD projection onto the combined gradient/mixed-derivative jet span
- rank-matched random-subspace baseline using the exact beta distribution
- rank-matched random orthogonal projection trajectory for distance comparison

## Run

The bundled Codex Python has NumPy available:

```bash
/Users/starpoint/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 mlp_projection_experiment.py
```

Example with a larger network:

```bash
/Users/starpoint/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 mlp_projection_experiment.py \
  --widths 25,25,25,1 \
  --batch-size 64 \
  --steps 100 \
  --output-dir runs/widths_25_25_25_1
```

Example Hessian-span paired-point run:

```bash
/Users/starpoint/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 mlp_projection_experiment.py \
  --widths 2,5,1 \
  --batch-size 8 \
  --steps 10 \
  --projection-mode input-hessian-span \
  --target-delta-norm 0.01 \
  --pair-delta 1e-12 \
  --output-dir runs/hessian_span_taylor_smoke
```

Example combined jet-span run:

```bash
/Users/starpoint/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 mlp_projection_experiment.py \
  --widths 10,10,10,1 \
  --batch-size 16 \
  --steps 50 \
  --projection-mode combined-jet-span \
  --pair-delta 1e-9 \
  --output-dir runs/combined_jet_span_smoke
```

The Hessian-span and combined jet-span modes currently require scalar output,
i.e. the final width must be `1`. The paired finite-difference diagnostic is sensitive to the size
of the recovered `v_i`; if coefficients are large, `pair_delta` may need to be
much smaller than the point displacement alone would suggest.

## Outputs

Each run writes:

- `history.csv`: per-step projection and random-subspace distances, rank, projection ratio, singular-value diagnostics
- `random_baseline.csv`: rank-matched random-subspace projection ratios
- `summary.json`: configuration and aggregate metrics
- `distance.svg`: projection and random-subspace `||theta - theta_star||` over time
- `rank.svg`: sampled subspace rank over time
- `projection_ratio.svg`: actual projection ratio over time
- `projection_ratio_histogram.svg`: actual ratios compared with rank-matched random subspaces

Derivative-span runs also write:

- `finite_difference_error.svg`: paired finite-difference realization error over time

Derivative-span `history.csv` rows include additional diagnostics such as
`finite_difference_relative_error`, `max_v_norm`, `mean_v_norm`,
`max_pair_displacement`, and `mean_pair_displacement`.

## Three-Method HTML Interface

Open the minimal comparison interface directly:

```text
hessian_span_distance_report.html
```

It compares gradient span, Hessian span, and 2-Jet span from the same initial
and target parameters. Each method has architecture widths and sampled points
controls. The play button runs until paused.

For the fastest projection solves, serve the folder locally or host it as static
files so the browser can use the Worker-backed LAPACK bundle:

```bash
python3 -m http.server 8765
```

Then open `http://127.0.0.1:8765/hessian_span_distance_report.html`. Opening the
HTML file directly still works, but it falls back to loading the LAPACK bundle on
the main browser thread because many browsers restrict local-file Workers.

## Playback Report

Generate a standalone HTML report from a saved run:

```bash
/Users/starpoint/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 generate_playback_report.py \
  --run-dir runs/widths_25_25_25_1_sphere_smoke \
  --comparison-run-dir runs/default_smoke \
  --output gradient_descent_playback_report.html
```

The report embeds the saved `history.csv`, `random_baseline.csv`, and
`summary.json` data directly into the HTML. The browser only plays back the
recorded run by default.

The same HTML also has a live-run panel. Enter architecture widths such as
`25,25,25,1`, choose the number `N` of sampled points, steps, seed, sampler, and
`# random subspaces / step`, then press **Run**. The browser runs a fresh
experiment with analytic Jacobians and modified Gram-Schmidt row-space
projection, updating the report plots as the run progresses. It also runs a
rank-matched random orthogonal projection trajectory from the same initial
parameters and plots its distance beside the Jacobian row-space trajectory. The
default input sampler is `sphere`, i.e. uniform samples from the unit sphere in
`R^{n_0}`. Browser-side live runs are still row-space-only; saved derivative-span
runs can be loaded and played back.
