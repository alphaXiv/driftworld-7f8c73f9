"""Render the two report figures from reports/pusht/results.json."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "reports/pusht/results.json").read_text())
OUT = ROOT / "reports/pusht/images"
OUT.mkdir(parents=True, exist_ok=True)


def mean_sd(values):
    return statistics.mean(values), statistics.pstdev(values)


def quality_svg():
    mse = DATA["methods"]["MSE"]
    drift = DATA["methods"]["Drift"]
    methods = [("MSE", mse, "#4C78A8"), ("Drift", drift, "#E45756")]
    parts = ["<svg xmlns='http://www.w3.org/2000/svg' width='920' height='430' viewBox='0 0 920 430'>",
             "<rect width='920' height='430' fill='white'/>",
             "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.label{font-size:16px}.small{font-size:13px;fill:#4b5563}</style>",
             "<text x='36' y='38' class='title'>Observed 64-frame quality (8 seeds, mean ± SD)</text>"]
    panels = [(40, "SSIM ↑", "rollout64Ssim", 1.0), (480, "MSE ↓", "rollout64Mse", 0.16)]
    for x0, title, key, ymax in panels:
        parts.append(f"<text x='{x0}' y='78' class='label'>{title}</text>")
        parts.append(f"<line x1='{x0}' y1='360' x2='{x0+360}' y2='360' stroke='#9ca3af'/>")
        for i, (name, values, color) in enumerate(methods):
            mean, sd = mean_sd(values[key])
            height = 250 * mean / ymax
            x = x0 + 70 + i * 150
            y = 360 - height
            err = 250 * sd / ymax
            parts += [f"<rect x='{x}' y='{y:.1f}' width='90' height='{height:.1f}' rx='5' fill='{color}'/>",
                      f"<line x1='{x+45}' y1='{y-err:.1f}' x2='{x+45}' y2='{y+err:.1f}' stroke='#111827' stroke-width='2'/>",
                      f"<line x1='{x+35}' y1='{y-err:.1f}' x2='{x+55}' y2='{y-err:.1f}' stroke='#111827' stroke-width='2'/>",
                      f"<text x='{x+45}' y='386' text-anchor='middle' class='label'>{name}</text>",
                      f"<text x='{x+45}' y='{max(102,y-err-10):.1f}' text-anchor='middle' class='small'>{mean:.5f}</text>"]
        parts.append(f"<text x='{x0+180}' y='414' text-anchor='middle' class='small'>Whiskers: population SD across seeds</text>")
    parts.append("</svg>")
    (OUT / "quality.svg").write_text("".join(parts))


def latency_svg():
    names = [("MSE", "MSE", "#4C78A8"), ("Drift", "Drift", "#E45756"), ("Diffusion-20", "Diffusion20", "#72B7B2")]
    parts = ["<svg xmlns='http://www.w3.org/2000/svg' width='920' height='420' viewBox='0 0 920 420'>",
             "<rect width='920' height='420' fill='white'/>",
             "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.label{font-size:16px}.small{font-size:13px;fill:#4b5563}</style>",
             "<text x='36' y='38' class='title'>Observed inference latency (8 seeds, mean ± SD)</text>",
             "<text x='36' y='70' class='small'>NVIDIA RTX PRO 6000 Blackwell; batch one; four-frame chunks</text>",
             "<line x1='90' y1='350' x2='860' y2='350' stroke='#9ca3af'/>"]
    ymax = 27.0
    for i, (label, key, color) in enumerate(names):
        values = DATA["methods"][key]["latencyMsPerFrame"]
        mean, sd = mean_sd(values)
        height = 235 * mean / ymax
        x = 160 + i * 240
        y = 350 - height
        err = 235 * sd / ymax
        parts += [f"<rect x='{x}' y='{y:.1f}' width='120' height='{height:.1f}' rx='5' fill='{color}'/>",
                  f"<line x1='{x+60}' y1='{y-err:.1f}' x2='{x+60}' y2='{y+err:.1f}' stroke='#111827' stroke-width='2'/>",
                  f"<line x1='{x+48}' y1='{y-err:.1f}' x2='{x+72}' y2='{y-err:.1f}' stroke='#111827' stroke-width='2'/>",
                  f"<text x='{x+60}' y='378' text-anchor='middle' class='label'>{label}</text>",
                  f"<text x='{x+60}' y='{max(100,y-err-12):.1f}' text-anchor='middle' class='label'>{mean:.3f} ms</text>"]
    parts += ["<text x='54' y='220' transform='rotate(-90 54 220)' text-anchor='middle' class='label'>milliseconds per generated frame ↓</text>",
              "<text x='460' y='408' text-anchor='middle' class='small'>Drift is 23.6× faster than the 20-step sampler; one-pass MSE and Drift are matched.</text>", "</svg>"]
    (OUT / "latency.svg").write_text("".join(parts))


quality_svg()
latency_svg()
