# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14.0",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # DriftWorld on Push-T: evidence first

    **Result.** The one-pass latency claim reproduced, but the reported drifting-field
    quality advantage did not appear in this bounded reconstruction. Eight independent
    seeds per objective gave:

    | Model | Held-out MSE ↓ | 64-frame MSE ↓ | Latency ↓ |
    |---|---:|---:|---:|
    | MSE one-pass | **0.000792** | **0.00557** | 1.058 ms/frame |
    | Drift one-pass | 0.045031 | 0.10285 | **1.052 ms/frame** |
    | Diffusion, 20 steps | 0.14677 | 0.20815 | 24.840 ms/frame |

    This notebook contains the already-produced evidence, so opening it in Molab does
    not rerun expensive training or depend on repository-relative files.
    """)
    return


@app.cell
def _():
    results = {
        "MSE": {
            "heldout_mse": (0.0007920305, 0.0000120761),
            "rollout_mse": (0.0055697905, 0.0007576110),
            "latency": (1.0578830, 0.0439160),
            "action_gain": 0.00000546856,
        },
        "Drift": {
            "heldout_mse": (0.0450313585, 0.0002189236),
            "rollout_mse": (0.1028471915, 0.0258041564),
            "latency": (1.0520106, 0.0342399),
            "action_gain": 0.000000717584,
        },
        "Diffusion (20-step)": {
            "heldout_mse": (0.1467726175, 0.02136567),
            "rollout_mse": (0.2081539854, 0.0255715),
            "latency": (24.8399542, 0.58423),
            "action_gain": 0.0000113966,
        },
    }
    return (results,)


@app.cell
def _(mo):
    metric = mo.ui.dropdown(
        options={
            "Held-out one-pass MSE": "heldout_mse",
            "64-frame autoregressive MSE": "rollout_mse",
            "Inference latency (ms/frame)": "latency",
        },
        value="Held-out one-pass MSE",
        label="Explore a measured result",
    )
    metric
    return (metric,)


@app.cell
def _(metric, np, plt, results):
    names = list(results)
    means = [results[name][metric.value][0] for name in names]
    stds = [results[name][metric.value][1] for name in names]
    colors = ["#2878B5", "#D9534F", "#6C757D"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors)
    ax.set_xticks(x, names)
    ax.set_yscale("log")
    ax.set_title(metric.selected_key)
    ax.set_ylabel("Measured value (log scale; lower is better)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        label = f"{value:.4g}" if value < 1 else f"{value:.2f}"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 7), textcoords="offset points", ha="center")
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What was tested

    Push-T is a planar manipulation dataset: an agent must move a T-shaped block by
    choosing 2-D actions. A world model sees four 96×96 RGB history frames and four
    future actions, then predicts the next four frames. Rolling the model forward
    sixteen times produces the 64-frame test.

    The public archive contained 25,650 frames in 206 episodes. The last 50 episodes
    were fixed as test data, leaving 18,043 training and 6,165 test windows. The MSE,
    drifting, and diffusion runs used the same reconstructed 4.80M-parameter FiLM
    spatiotemporal U-Net and eight independently initialized models.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How the drifting field differs from MSE

    The direct control learns the future pixels:

    \[
    \mathcal L_{\text{MSE}} = \lVert G_\theta(z, h, a) - x^+ \rVert_2^2.
    \]

    DriftWorld instead constructs a local vector field. Generated candidates are
    attracted to the demonstrated future and repelled from generated peers plus a
    static-frame negative. The network follows that field through a stopped-gradient
    fixed-point target:

    \[
    \mathcal L_{\text{drift}} =
    \lVert G_\theta(z,h,a) - \operatorname{sg}[G_\theta(z,h,a)+v] \rVert_2^2.
    \]

    This reconstruction used eight negative samples and temperatures 0.02, 0.05,
    and 0.2, matching the paper's appendix. The exact field normalization and update
    magnitude remain uncertain because the public author repository contains no
    implementation.
    """)
    return


@app.cell
def _(mo, results):
    drift_quality_ratio = results["Drift"]["heldout_mse"][0] / results["MSE"]["heldout_mse"][0]
    rollout_ratio = results["Drift"]["rollout_mse"][0] / results["MSE"]["rollout_mse"][0]
    speedup = results["Diffusion (20-step)"]["latency"][0] / results["Drift"]["latency"][0]
    mo.md(
        f"""
        ## Reading the result

        - The reconstructed Drift model's held-out MSE was **{drift_quality_ratio:.1f}×**
          the MSE control's, and its rollout MSE was **{rollout_ratio:.1f}×** higher.
          This run therefore did not show the paper's reported quality direction.
        - Drift was **{speedup:.1f}× faster** than the 20-step sampler. This is the cleanest
          aligned claim: one network evaluation is substantially cheaper than twenty.
        - Supplying true rather than shuffled actions improved Drift MSE by only
          **{results['Drift']['action_gain']:.2e}**. The sign was consistent across all
          eight seeds, but the effect is too small to establish useful control.

        The diffusion model was trained for only 6,000 updates and is included as a timing
        reference, not as a quality benchmark.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Evidence boundary and compute

    The paper reports 8.73M parameters and does not publish the exact Push-T source,
    split, field constants, or training duration. This reconstruction has 4.80M
    parameters and 6,000 updates per seed. It omits LPIPS and policy ranking.

    Formal runs used the configured **Kubernetes** backend, **NVIDIA RTX PRO 6000
    Blackwell** GPUs, and a peak of **16 concurrent GPUs**. The detailed public report
    records the per-run wall times, sensitivity checks, experiment branches, and final
    campaign elapsed time.

    **Bottom line:** the latency advantage reproduced; the Drift-over-MSE visual-quality
    effect did not appear under this bounded public-data reconstruction.
    """)
    return


if __name__ == "__main__":
    app.run()
