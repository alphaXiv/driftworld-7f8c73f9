# Reproduction: one-step drifting on Push-T

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/driftworld-7f8c73f9/blob/main/notebooks/driftworld_reproduction.py)

We tested the central claim of [*DriftWorld: Fast World Modeling through Drifting* (arXiv:2607.15065)](https://arxiv.org/abs/2607.15065): a pixel-space drifting-field objective should improve Push-T prediction over a same-architecture one-pass MSE model while retaining one-pass speed relative to diffusion. We trained eight independent seeds per objective on the public 206-episode Diffusion Policy Push-T replay, using a reconstructed 4.80M-parameter action-FiLM video U-Net for 6,000 updates.

**Assessment: partially reproduced.** The latency result aligned: Drift and MSE took 1.052 and 1.058 ms/frame, versus 24.840 ms/frame for a matched 20-step sampler (23.6× slower than Drift). The quality result reversed under this downscaled reconstruction: on 64-frame rollouts, Drift reached MSE 0.10285 / SSIM 0.54635 versus the MSE control's 0.00557 / 0.88245. The paper reports Drift 0.0007 / 0.9925 versus MSE 0.0035 / 0.9704. The main substitutions are the smaller public replay (206 episodes versus the paper's 500 mixed trajectories), 6,000 updates because the author schedule is unpublished, a 4.80M reconstruction versus the paper's 8.73M U-Net, and a 20-step undertrained diffusion latency reference.

Read the [detailed claim-by-claim report](reports/pusht/report.md), inspect the [embedded result data](reports/pusht/results.json), or open the [self-contained tutorial notebook](notebooks/driftworld_reproduction.py). The exact public Molab URL is <https://molab.marimo.io/github/alphaXiv/driftworld-7f8c73f9/blob/main/notebooks/driftworld_reproduction.py>.

Compute: Kubernetes, NVIDIA RTX PRO 6000 Blackwell Server Edition, peak 16 concurrent GPUs. The complete compute campaign ran from 2026-07-20 09:35:13Z to 10:07:39Z (0.541 elapsed wall hours); successful per-model train/eval spans were 170.0 s (MSE), 662.4 s (Drift), and 207.7 s (diffusion).

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Polished publication surface | Not run as an experiment (publication surface) | Report, notebook, figures, and implementation | — |
| [MSE independent seeds](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/mse-independent-seeds) | Same U-Net, direct pixel MSE; eight seeds | `bash run.sh` | Control: 64-frame MSE 0.00557, SSIM 0.88245; 1.058 ms/frame | Kubernetes; 8× RTX PRO 6000 Blackwell; 170.0 s train/eval |
| [Drift independent seeds](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/drift-independent-seeds) | Normalized pixel drifting, 8 negatives, three temperatures; eight seeds | `bash run.sh` | Quality gain not seen in this setup: MSE 0.10285, SSIM 0.54635; 1.052 ms/frame | Kubernetes; 8× RTX PRO 6000 Blackwell; 662.4 s train/eval |
| [20-step diffusion independent seeds](https://github.com/alphaXiv/driftworld-7f8c73f9/tree/orx/20-step-diffusion-independent-seeds) | Matched cosine diffusion and 20-step DDIM; eight seeds | `bash run.sh` | Latency reference: 24.840 ms/frame; quality undertrained | Kubernetes; 8× RTX PRO 6000 Blackwell; 207.7 s train/eval |

---

# DriftWorld: Fast World Modeling through Drifting

### [Paper](https://arxiv.org/abs/2607.15065) | [Project Page](https://susie-lu.github.io/driftworld/)

<img src="assets/Teaser.png" width="700px"/>

This codebase will contain the official implementation for the paper:

> **DriftWorld: Fast World Modeling through Drifting**  
> Susie Lu, Haonan Chen, Weirui Ye, Yilun Du 
