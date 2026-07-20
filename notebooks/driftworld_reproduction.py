import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # DriftWorld on Push-T: inspect the evidence

    This tutorial is self-contained: the formal Kubernetes results are embedded below, so opening it in Molab does **not** rerun training. We ask whether a one-step drifting-field objective improves a matched MSE video predictor while preserving the latency advantage of one-pass generation.

    **Bottom line:** speed partially reproduced; quality did not align in this bounded reconstruction. Drift was 23.6× faster than a matched 20-step sampler, but its 64-frame predictions were substantially worse than direct MSE.
    """)
    return


@app.cell
def _():
    evidence = {
        "MSE": {"heldout_mse": 0.00079203, "rollout_mse": 0.00556979, "rollout_ssim": 0.88244869, "latency_ms": 1.05788302, "action_gain": 5.4686e-6},
        "Drift": {"heldout_mse": 0.04503136, "rollout_mse": 0.10284719, "rollout_ssim": 0.54635274, "latency_ms": 1.05201057, "action_gain": 0.7176e-6},
        "Diffusion-20": {"heldout_mse": 0.14677262, "rollout_mse": 0.20815399, "rollout_ssim": 0.16883701, "latency_ms": 24.83995419, "action_gain": 11.3966e-6},
    }
    return (evidence,)


@app.cell
def _(evidence, mo):
    summary_rows = [
        {
            "method": method,
            "held-out MSE ↓": values["heldout_mse"],
            "64-frame MSE ↓": values["rollout_mse"],
            "64-frame SSIM ↑": values["rollout_ssim"],
            "ms/frame ↓": values["latency_ms"],
        }
        for method, values in evidence.items()
    ]
    mo.vstack([
        mo.md("## The measured comparison"),
        mo.ui.table(summary_rows, selection=None),
        mo.md("Values are means over eight independent GPU seeds. The 20-step model is an undertrained latency reference, not a competitive quality baseline."),
    ])
    return


@app.cell
def _(mo):
    metric_picker = mo.ui.dropdown(
        options={
            "64-frame SSIM (higher is better)": "rollout_ssim",
            "64-frame MSE (lower is better)": "rollout_mse",
            "Latency in ms/frame (lower is better)": "latency_ms",
        },
        value="64-frame SSIM (higher is better)",
        label="Explore a measured metric",
    )
    metric_picker
    return (metric_picker,)


@app.cell
def _(evidence, metric_picker, mo):
    selected_key = metric_picker.value
    selected_values = {name: vals[selected_key] for name, vals in evidence.items()}
    max_value = max(selected_values.values())
    bar_rows = "".join(
        f"<div style='margin:10px 0'><b>{name}</b>: {value:.6g}<div style='height:22px;background:#5b8ff9;width:{max(2, 100*value/max_value):.1f}%'></div></div>"
        for name, value in selected_values.items()
    )
    mo.Html(f"<div style='max-width:700px'>{bar_rows}</div>")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What was implemented

    The generator consumes four RGB history frames, Gaussian noise for four future frames, and four 2-D actions. Spatial convolutions are followed by temporal convolutions; action embeddings provide frame-wise FiLM scale and shift.

    For each generated sample $x$, drifting constructs

    $$V(x)=V^+_p(x)-V^-_q(x),$$

    where the positive mean shift points toward the single ground-truth future and the negative mean shift points toward generated peers plus a no-action frame. Fields are normalized and summed for temperatures 0.02, 0.05, and 0.2. Training regresses to the detached target $x+V(x)$. Direct MSE changes only this objective; diffusion changes it to cosine noise prediction and uses 20 DDIM steps at evaluation.

    The exact paper training code was unavailable. Material substitutions were the public 206-episode replay instead of the paper's 500 mixed trajectories, 6,000 updates, and a 4.80M rather than 8.73M parameter reconstruction.
    """)
    return


@app.cell
def _(evidence, mo):
    drift_speedup = evidence["Diffusion-20"]["latency_ms"] / evidence["Drift"]["latency_ms"]
    mse_ratio = evidence["Drift"]["rollout_mse"] / evidence["MSE"]["rollout_mse"]
    mo.md(
        f"""
        ## Read the result carefully

        - **Latency aligned:** the iterative sampler was **{drift_speedup:.1f}× slower** than Drift, while Drift and MSE had matched one-pass latency.
        - **Quality diverged:** Drift's 64-frame MSE was **{mse_ratio:.1f}× higher** than the MSE control in this setup.
        - **Conditioning was weak:** shuffled actions increased Drift MSE by only {evidence['Drift']['action_gain']:.2e} on average. The sign was positive in all eight seeds, but the magnitude and poor rollout quality do not establish useful simulated rollouts.

        This is therefore a **partial reproduction**, not evidence that the paper's general claim is false. The tested dataset, architecture details, and training duration differ materially from the unreleased author setup.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Compute provenance

    All formal runs used Kubernetes on NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. Each method trained eight independent seeds on eight GPUs; peak concurrency was 16 GPUs. The complete campaign lasted 0.541 wall hours. Formal train/eval spans were 170.0 s (MSE), 662.4 s (Drift), and 207.7 s (diffusion). The exact command on every experiment was `bash run.sh`.
    """)
    return


if __name__ == "__main__":
    app.run()
