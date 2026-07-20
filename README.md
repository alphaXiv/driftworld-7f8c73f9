# Reproduction: one-step drifting on Push-T

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/driftworld-7f8c73f9/blob/main/notebooks/driftworld_reproduction.py)

We tested the central claim of [*DriftWorld: Fast World Modeling through Drifting* (arXiv:2607.15065)](https://arxiv.org/abs/2607.15065): a pixel-space drifting-field objective should improve Push-T prediction over a same-architecture one-pass MSE model while retaining one-pass speed relative to diffusion. We trained eight independent seeds per objective on the public 206-episode Diffusion Policy Push-T replay, using a reconstructed 4.80M-parameter action-FiLM video U-Net. The formal MSE/Drift pair ran for 60,000 updates; the diffusion timing reference ran for 6,000.

**Assessment: partially reproduced.** The latency result aligned: Drift and MSE took 1.061 and 1.059 ms/frame, versus 24.840 ms/frame for a matched 20-step sampler (23.4× slower than Drift). The quality result reversed under this reconstruction: at matched 60k updates, Drift reached 64-frame MSE 0.09947 / SSIM 0.51595 versus the MSE control's 0.00416 / 0.91927. The paper reports Drift 0.0007 / 0.9925 versus MSE 0.0035 / 0.9704. The main substitutions are the smaller public replay (206 episodes versus the paper's 500 mixed trajectories), a bounded 60k schedule because the author schedule is unpublished, a 4.80M reconstruction versus the paper's 8.73M U-Net, and a 20-step undertrained diffusion latency reference.

Diagnostic ablations localized the mismatch to the reconstructed repulsion. At matched 60k updates, attraction-only reached 0.000780 held-out MSE and 0.004518 rollout MSE versus MSE's 0.000439 / 0.004165; the dramatic full-field gap disappeared, though MSE retained the better means. At 6k, increasing reconstructed repulsion weights 0, 0.001, 0.01, 0.1, 0.5, and 1.0 worsened held-out MSE monotonically from 0.00140 to 0.04503. This narrows the uncertainty but cannot validate the authors' unavailable source implementation.

Read the [detailed claim-by-claim report](reports/pusht/report.md), inspect the [aggregate evidence](reports/pusht/evidence.json) and [primary per-seed data](reports/pusht/results.json), or open the [self-contained tutorial notebook](notebooks/driftworld_reproduction.py). The exact public Molab URL is <https://molab.marimo.io/github/alphaXiv/driftworld-7f8c73f9/blob/main/notebooks/driftworld_reproduction.py>.

Compute: Kubernetes, NVIDIA RTX PRO 6000 Blackwell Server Edition, peak 16 concurrent GPUs. The complete compute campaign ran from 2026-07-20 09:35:13Z to 13:58:34Z (**4.389 elapsed wall hours**); successful headline train/eval spans were 1,481.2 s (MSE 60k), 6,402.7 s (Drift 60k), and 207.7 s (diffusion timing reference).

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Polished publication surface | Not run as an experiment (publication surface) | Report, notebook, figures, and implementation | — |
| [MSE, 60k updates](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/mse-60k-updates) | Same U-Net, direct pixel MSE; eight seeds | `bash run.sh` | Control: 64-frame MSE 0.00416, SSIM 0.91927; 1.059 ms/frame | Kubernetes; 8× RTX PRO 6000 Blackwell; 1,481.2 s train/eval |
| [Drift, 60k updates](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/drift-60k-updates) | Normalized pixel drifting, 8 negatives, three temperatures; eight seeds | `bash run.sh` | Quality gain not seen: 64-frame MSE 0.09947, SSIM 0.51595; 1.061 ms/frame | Kubernetes; 8× RTX PRO 6000 Blackwell; 6,402.7 s train/eval |
| [Attraction-only, 60k updates](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/attraction-only-60k-updates) | Remove both reconstructed repulsive terms at the matched schedule; eight seeds | `bash run.sh` | Held-out MSE 0.000780; rollout MSE 0.004518, close to but behind matched MSE means | Kubernetes; 8× RTX PRO 6000 Blackwell; 6,402.4 s train/eval |
| Repulsion interpolation: [0.001](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/drift-repulsion-weight-0-001), [0.01](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/drift-repulsion-weight-0-01), [0.1](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/drift-repulsion-weight-0-1), [0.5](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/drift-repulsion-weight-0-5) | Vary only repulsion coefficient at 6k; eight seeds each | `bash run.sh` | Held-out MSE rises monotonically: 0.00160, 0.00300, 0.01240, 0.03376 | Kubernetes; 8× RTX PRO 6000 Blackwell per run; ~662 s train/eval each |
| [20-step diffusion independent seeds](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/20-step-diffusion-independent-seeds) | Matched cosine diffusion and 20-step DDIM; eight seeds | `bash run.sh` | Latency reference: 24.840 ms/frame; quality undertrained | Kubernetes; 8× RTX PRO 6000 Blackwell; 207.7 s train/eval |

---

# DriftWorld: Fast World Modeling through Drifting

### [Paper](https://arxiv.org/abs/2607.15065) | [Project Page](https://susie-lu.github.io/driftworld/)

<img src="assets/Teaser.png" width="700px"/>

This codebase will contain the official implementation for the paper:

> **DriftWorld: Fast World Modeling through Drifting**  
> Susie Lu, Haonan Chen, Weirui Ye, Yilun Du 
