"""Matched Push-T reproduction for DriftWorld (arXiv:2607.15065).

The only experiment-level switch is config.json: objective is one of mse, drift,
or diffusion. Architecture, data split, optimizer, update count, evaluation, and
Kubernetes shape otherwise remain fixed. The terminal JSON block is the durable
evidence channel consumed by OpenResearch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import zarr


def distributed_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # NCCL communicator creation stalls on the target cluster. Gloo is used only
    # for control/metric aggregation; each rank trains a complete independent
    # model on its own GPU, giving eight measured seeds per objective.
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    return rank, local_rank, dist.get_world_size()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset(zip_path: str, rank: int):
    cache = Path("/tmp/driftworld_pusht")
    ready = cache / "READY"
    if rank == 0 and not ready.exists():
        cache.mkdir(parents=True, exist_ok=True)
        store = zarr.ZipStore(zip_path, mode="r")
        root = zarr.group(store=store)
        image_key = "data/img" if "img" in root["data"] else "data/image"
        images = np.asarray(root[image_key][:], dtype=np.uint8)
        actions = np.asarray(root["data/action"][:], dtype=np.float32)
        ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        np.save(cache / "images.npy", images)
        np.save(cache / "actions.npy", actions)
        np.save(cache / "episode_ends.npy", ends)
        store.close()
        ready.touch()
        print(f"DATASET frames={len(images)} episodes={len(ends)} image_shape={images.shape[1:]}")
    dist.barrier()
    images = np.load(cache / "images.npy", mmap_mode="r")
    actions = np.load(cache / "actions.npy", mmap_mode="r")
    ends = np.load(cache / "episode_ends.npy")
    return images, actions, ends


def valid_starts(ends: np.ndarray, history: int, future: int, test_eps: int):
    starts = np.r_[0, ends[:-1]]
    train, test, rollout = [], [], []
    split = len(ends) - test_eps
    for ep, (s, e) in enumerate(zip(starts, ends)):
        valid = np.arange(s + history - 1, e - future, dtype=np.int64)
        if ep < split:
            train.extend(valid.tolist())
        else:
            test.extend(valid.tolist())
            if e - s >= history + 64:
                rollout.append(int(s + history - 1))
    return np.asarray(train), np.asarray(test), np.asarray(rollout)


def get_batch(images, actions, indices, history, horizon, device):
    # t is the final history frame; action[t+i] leads to image[t+i+1].
    obs = np.stack([images[np.arange(t - history + 1, t + 1)] for t in indices])
    fut = np.stack([images[np.arange(t + 1, t + horizon + 1)] for t in indices])
    act = np.stack([actions[np.arange(t, t + horizon)] for t in indices])
    obs = torch.from_numpy(obs.copy()).to(device, non_blocking=True).float().div_(127.5).sub_(1)
    fut = torch.from_numpy(fut.copy()).to(device, non_blocking=True).float().div_(127.5).sub_(1)
    act = torch.from_numpy(act.copy()).to(device, non_blocking=True).float()
    obs = obs.permute(0, 1, 4, 2, 3).contiguous()
    fut = fut.permute(0, 1, 4, 2, 3).contiguous()
    return obs, act, fut


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, temporal: bool = True):
        super().__init__()
        self.norm1 = nn.GroupNorm(12, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, channels * 2))
        self.norm2 = nn.GroupNorm(12, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.temporal = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), groups=channels) if temporal else None

    def forward(self, x, cond):
        # x [B,T,C,H,W], cond [B,T,D]
        b, t, c, h, w = x.shape
        residual = x
        y = self.norm1(x.reshape(b * t, c, h, w))
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        y = y.reshape(b, t, c, h, w) * (1 + scale[..., None, None]) + shift[..., None, None]
        y = self.conv1(F.silu(y).reshape(b * t, c, h, w))
        y = self.conv2(F.silu(self.norm2(y))).reshape(b, t, c, h, w)
        if self.temporal is not None:
            y = self.temporal(y.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return residual + y


class BottleneckAttention(nn.Module):
    """Spatial attention at the paper's 8x downsampling resolution."""
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(12, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.heads = heads

    def forward(self, x):
        b, t, c, h, w = x.shape
        z = self.norm(x.reshape(b * t, c, h, w))
        q, k, v = self.qkv(z).chunk(3, dim=1)
        d = c // self.heads
        def shape(a):
            return a.reshape(b * t, self.heads, d, h * w).transpose(-1, -2)
        z = F.scaled_dot_product_attention(shape(q), shape(k), shape(v))
        z = z.transpose(-1, -2).reshape(b * t, c, h, w)
        return x + self.out(z).reshape(b, t, c, h, w)


class VideoUNet(nn.Module):
    def __init__(self, history=4, base=96, res_blocks=2):
        super().__init__()
        self.history = history
        self.base = base
        cond_dim = base * 4
        self.action_mlp = nn.Sequential(nn.Linear(3, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.in_conv = nn.Conv2d(3 + history * 3, base, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for level in range(4):
            self.down_blocks.append(nn.ModuleList([FiLMResBlock(base, cond_dim) for _ in range(res_blocks)]))
            if level < 3:
                self.downsample.append(nn.Conv2d(base, base, 3, stride=2, padding=1))
        self.attn = BottleneckAttention(base)
        self.mid = nn.ModuleList([FiLMResBlock(base, cond_dim) for _ in range(2)])
        self.upsample = nn.ModuleList([nn.ConvTranspose2d(base, base, 4, stride=2, padding=1) for _ in range(3)])
        self.merge = nn.ModuleList([nn.Conv2d(base * 2, base, 1) for _ in range(3)])
        self.up_blocks = nn.ModuleList([
            nn.ModuleList([FiLMResBlock(base, cond_dim) for _ in range(res_blocks)]) for _ in range(3)
        ])
        self.out = nn.Sequential(nn.GroupNorm(12, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))

    def forward(self, noise, obs, action, timestep=None):
        b, t, _, h, w = noise.shape
        if timestep is None:
            timestep = torch.zeros(b, device=noise.device, dtype=noise.dtype)
        if timestep.ndim == 0:
            timestep = timestep.expand(b)
        time_feature = timestep[:, None, None].expand(b, t, 1)
        cond = self.action_mlp(torch.cat([action, time_feature], dim=-1))
        history = obs.reshape(b, -1, h, w)[:, None].expand(-1, t, -1, -1, -1)
        x = torch.cat([noise, history], dim=2)
        x = self.in_conv(x.reshape(b * t, -1, h, w)).reshape(b, t, self.base, h, w)
        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                x = block(x, cond)
            skips.append(x)
            if level < 3:
                bt, tt, cc, hh, ww = x.shape
                x = self.downsample[level](x.reshape(bt * tt, cc, hh, ww)).reshape(bt, tt, cc, hh // 2, ww // 2)
        x = self.attn(x)
        for block in self.mid:
            x = block(x, cond)
        for i in range(3):
            b0, t0, c0, h0, w0 = x.shape
            x = self.upsample[i](x.reshape(b0 * t0, c0, h0, w0)).reshape(b0, t0, c0, h0 * 2, w0 * 2)
            skip = skips[-2 - i]
            x = self.merge[i](torch.cat([x, skip], dim=2).reshape(b0 * t0, c0 * 2, h0 * 2, w0 * 2)).reshape(b0, t0, c0, h0 * 2, w0 * 2)
            for block in self.up_blocks[i]:
                x = block(x, cond)
        return self.out(x.reshape(b * t, self.base, h, w)).reshape(b, t, 3, h, w)


def normalized_drift(pred, target, still, temperatures, repulsion_weight=1.0):
    """Pixel-space conditional attraction-repulsion field from the paper appendix.

    Each spatial location is an independent field whose feature is the flattened
    T*C vector. Generated peers plus the repeated current frame are negatives.
    """
    b, n, t, c, h, w = pred.shape
    x = pred.permute(0, 4, 5, 1, 2, 3).reshape(b, h, w, n, t * c)
    pos = target.permute(0, 3, 4, 1, 2).reshape(b, h, w, 1, t * c)
    still_chunk = still[:, None].expand(-1, t, -1, -1, -1)
    still_chunk = still_chunk.permute(0, 3, 4, 1, 2).reshape(b, h, w, 1, t * c)
    negatives = torch.cat([x.detach(), still_chunk], dim=3)
    cloud = torch.cat([pos, negatives], dim=3)
    # Mean pairwise scale, detached, with a floor for nearly-static pixels.
    pair = torch.cdist(cloud.float(), cloud.float())
    scale = pair.sum(dim=(-1, -2), keepdim=True) / max(1, cloud.shape[3] * (cloud.shape[3] - 1))
    scale = scale.clamp_min(1e-3).to(x.dtype)
    xn, pn, nn = x / scale, pos / scale, negatives / scale
    total = torch.zeros_like(x)
    root_c = math.sqrt(t * c)
    self_mask = torch.eye(n, device=x.device, dtype=torch.bool)[None, None, None]
    for tau in temperatures:
        dpos = torch.linalg.vector_norm(xn - pn, dim=-1, keepdim=True)
        kpos = torch.exp(-dpos / (tau * root_c))
        vpos = kpos * (pos - x) / kpos.clamp_min(1e-8)
        delta = nn[:, :, :, None, :, :] - x[:, :, :, :, None, :]
        dneg = torch.linalg.vector_norm(xn[:, :, :, :, None, :] - nn[:, :, :, None, :, :], dim=-1)
        kneg = torch.exp(-dneg / (tau * root_c))
        kneg[..., :n] = kneg[..., :n].masked_fill(self_mask, 0)
        vneg = (kneg[..., None] * delta).sum(dim=-2) / kneg.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        field = vpos - repulsion_weight * vneg
        field = field / torch.linalg.vector_norm(field.float(), dim=-1, keepdim=True).clamp_min(1e-6).to(field.dtype)
        total = total + field
    return total.reshape(b, h, w, n, t, c).permute(0, 3, 4, 5, 1, 2).contiguous()


def cosine_alpha_bar(step, total=1000):
    s = 0.008
    f = math.cos(((step / total + s) / (1 + s)) * math.pi / 2) ** 2
    return f / (math.cos((s / (1 + s)) * math.pi / 2) ** 2)


def diffusion_schedule(device, steps=1000):
    abar = torch.tensor([cosine_alpha_bar(i, steps) for i in range(steps + 1)], device=device)
    return abar.clamp(1e-5, 1.0)


@torch.no_grad()
def generate(model, obs, action, objective, diffusion_steps=20, noise=None):
    b, t = action.shape[:2]
    if noise is None:
        noise = torch.randn(b, t, 3, obs.shape[-2], obs.shape[-1], device=obs.device)
    if objective != "diffusion":
        return model(noise, obs, action, torch.zeros(b, device=obs.device)).clamp(-1, 1)
    abar = diffusion_schedule(obs.device)
    x = noise
    schedule = torch.linspace(999, 0, diffusion_steps + 1, device=obs.device).long()
    for i in range(diffusion_steps):
        ti, tj = schedule[i], schedule[i + 1]
        eps = model(x, obs, action, (ti.float() / 999).expand(b))
        a_i, a_j = abar[ti], abar[tj]
        x0 = ((x - (1 - a_i).sqrt() * eps) / a_i.sqrt()).clamp(-1, 1)
        x = a_j.sqrt() * x0 + (1 - a_j).sqrt() * eps
    return x.clamp(-1, 1)


def image_metrics(pred, target):
    pred = pred.float().clamp(-1, 1).add(1).div(2)
    target = target.float().clamp(-1, 1).add(1).div(2)
    mse = F.mse_loss(pred, target).item()
    psnr = -10 * math.log10(max(mse, 1e-12))
    # Standard local SSIM, averaged over frames/channels/images.
    k = 11
    x = pred.reshape(-1, 3, pred.shape[-2], pred.shape[-1])
    y = target.reshape_as(x)
    ux, uy = F.avg_pool2d(x, k, 1, k // 2), F.avg_pool2d(y, k, 1, k // 2)
    vx = F.avg_pool2d(x * x, k, 1, k // 2) - ux * ux
    vy = F.avg_pool2d(y * y, k, 1, k // 2) - uy * uy
    vxy = F.avg_pool2d(x * y, k, 1, k // 2) - ux * uy
    ssim = (((2 * ux * uy + 0.01**2) * (2 * vxy + 0.03**2)) /
            ((ux * ux + uy * uy + 0.01**2) * (vx + vy + 0.03**2))).mean().item()
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


@torch.no_grad()
def evaluate(model, objective, cfg, images, actions, test_idx, rollout_idx, device):
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 99)
    chosen = test_idx[torch.randperm(len(test_idx), generator=gen)[:cfg["eval_windows"]].numpy()]
    preds, targets, shuffled_preds = [], [], []
    for chunk in np.array_split(chosen, max(1, len(chosen) // 8)):
        obs, act, fut = get_batch(images, actions, chunk, cfg["history"], cfg["horizon"], device)
        torch.manual_seed(cfg["seed"] + int(chunk[0]))
        noise = torch.randn_like(fut)
        pred = generate(model, obs, act, objective, cfg["diffusion_steps"], noise)
        shuffled = generate(model, obs, act.roll(1, 0), objective, cfg["diffusion_steps"], noise)
        preds.append(pred.cpu())
        shuffled_preds.append(shuffled.cpu())
        targets.append(fut.cpu())
    pred, target, shuffled = torch.cat(preds), torch.cat(targets), torch.cat(shuffled_preds)
    one = image_metrics(pred, target)
    shuffled_mse = image_metrics(shuffled, target)["mse"]
    action_effect = F.l1_loss(pred, shuffled).item()
    one.update({
        "shuffled_action_mse": shuffled_mse,
        "true_vs_shuffled_mse_gain": shuffled_mse - one["mse"],
        "action_effect_l1": action_effect,
    })

    rpred, rtgt = [], []
    for s in rollout_idx[:cfg["rollout_sequences"]]:
        obs_np = images[np.arange(s - cfg["history"] + 1, s + 1)]
        obs = torch.from_numpy(obs_np.copy()).to(device).float().div_(127.5).sub_(1).permute(0, 3, 1, 2)[None]
        seq_pred, seq_tgt = [], []
        for off in range(0, cfg["rollout_frames"], cfg["horizon"]):
            idx = np.arange(s + off, s + off + cfg["horizon"])
            act = torch.from_numpy(actions[idx].copy()).to(device).float()[None]
            gt = torch.from_numpy(images[idx + 1].copy()).to(device).float().div_(127.5).sub_(1).permute(0, 3, 1, 2)[None]
            torch.manual_seed(cfg["seed"] + int(s) + off)
            out = generate(model, obs, act, objective, cfg["diffusion_steps"])
            seq_pred.append(out.cpu())
            seq_tgt.append(gt.cpu())
            obs = torch.cat([obs, out], dim=1)[:, -cfg["history"]:]
        rpred.append(torch.cat(seq_pred, dim=1))
        rtgt.append(torch.cat(seq_tgt, dim=1))
    rollout = image_metrics(torch.cat(rpred), torch.cat(rtgt))

    # Batch-one latency for one T-frame chunk; report per generated frame.
    s = int(test_idx[0])
    obs, act, fut = get_batch(images, actions, np.asarray([s]), cfg["history"], cfg["horizon"], device)
    for _ in range(10):
        generate(model, obs, act, objective, cfg["diffusion_steps"])
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    repeats = 100 if objective != "diffusion" else 20
    starter.record()
    for _ in range(repeats):
        generate(model, obs, act, objective, cfg["diffusion_steps"])
    ender.record(); torch.cuda.synchronize()
    ms_per_chunk = starter.elapsed_time(ender) / repeats
    return {
        "heldout_one_pass": one,
        "autoregressive_64_frame": rollout,
        "latency_ms_per_chunk": ms_per_chunk,
        "latency_ms_per_frame": ms_per_chunk / cfg["horizon"],
        "fps": 1000 * cfg["horizon"] / ms_per_chunk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path("config.json").read_text())
    rank, local_rank, world = distributed_setup()
    device = torch.device("cuda", local_rank)
    seed_everything(cfg["seed"] + rank)
    start_wall = time.time()
    if rank == 0:
        print("CONFIG_JSON " + json.dumps(cfg, sort_keys=True))
        print(f"COMPUTE backend=kubernetes gpu_model={torch.cuda.get_device_name(0)} gpu_count={world}")
        print(f"PYTORCH version={torch.__version__} cuda={torch.version.cuda}")

    images, actions, ends = prepare_dataset(args.dataset, rank)
    train_idx, test_idx, rollout_idx = valid_starts(ends, cfg["history"], cfg["horizon"], cfg["test_episodes"])
    train_cutoff = int(ends[len(ends) - cfg["test_episodes"] - 1])
    train_min = np.asarray(actions[:train_cutoff], dtype=np.float32).min(0)
    train_max = np.asarray(actions[:train_cutoff], dtype=np.float32).max(0)
    if rank == 0:
        print(f"SPLIT train_windows={len(train_idx)} test_windows={len(test_idx)} test_episodes={cfg['test_episodes']}")
        print(f"ACTION_RANGE min={train_min.tolist()} max={train_max.tolist()}")
    # Normalize in a compact RAM array; images stay memory-mapped.
    norm_actions = (np.asarray(actions, dtype=np.float32) - train_min) / np.maximum(train_max - train_min, 1e-6) * 2 - 1

    model = VideoUNet(cfg["history"], cfg["base_channels"], cfg["res_blocks"]).to(device)
    if rank == 0:
        print(f"MODEL parameters={sum(p.numel() for p in model.parameters())}")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    diffusion_abar = diffusion_schedule(device)
    rng = np.random.default_rng(cfg["seed"] + rank * 1009)
    scaler = None

    for step in range(1, cfg["max_steps"] + 1):
        ids = rng.choice(train_idx, cfg["batch_per_gpu"], replace=False)
        obs, act, target = get_batch(images, norm_actions, ids, cfg["history"], cfg["horizon"], device)
        warm = min(1.0, step / cfg["warmup_steps"])
        decay = 0.5 * (1 + math.cos(math.pi * max(0, step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])))
        lr = cfg["learning_rate"] * warm * decay
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if cfg["objective"] == "mse":
                pred = model(torch.randn_like(target), obs, act)
                loss = F.mse_loss(pred, target)
            elif cfg["objective"] == "drift":
                n = cfg["negative_samples"]
                b, t, c, h, w = target.shape
                obs_n = obs[:, None].expand(-1, n, -1, -1, -1, -1).reshape(b * n, cfg["history"], c, h, w)
                act_n = act[:, None].expand(-1, n, -1, -1).reshape(b * n, t, 2)
                pred = model(torch.randn(b * n, t, c, h, w, device=device), obs_n, act_n).reshape(b, n, t, c, h, w)
                field = normalized_drift(
                    pred,
                    target,
                    obs[:, -1],
                    cfg["temperatures"],
                    cfg.get("repulsion_weight", 1.0),
                )
                loss = F.mse_loss(pred, (pred + field).detach())
            elif cfg["objective"] == "diffusion":
                b = target.shape[0]
                ti = torch.randint(1, 1000, (b,), device=device)
                noise = torch.randn_like(target)
                a = diffusion_abar[ti].sqrt()[:, None, None, None, None]
                noised = a * target + (1 - diffusion_abar[ti]).sqrt()[:, None, None, None, None] * noise
                pred = model(noised, obs, act, ti.float() / 999)
                loss = F.mse_loss(pred, noise)
            else:
                raise ValueError(cfg["objective"])
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
        opt.step()
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, 1 - cfg["ema_decay"])
        if step == 1 or step % 100 == 0:
            value = loss.detach().float().cpu()
            dist.all_reduce(value)
            if rank == 0:
                elapsed = time.time() - start_wall
                print(f"TRAIN step={step} loss={value.item()/world:.8f} lr={lr:.3e} grad_norm={float(grad):.4f} elapsed_s={elapsed:.1f}")

    dist.barrier()
    eval_start = time.time()
    metrics = evaluate(ema, cfg["objective"], cfg, images, norm_actions, test_idx, rollout_idx, device)
    checkpoint = {
        "model": ema.state_dict(), "config": cfg,
        "action_min": train_min, "action_max": train_max,
    }
    checkpoint_path = f"/tmp/driftworld_checkpoint_seed{rank}.pt"
    torch.save(checkpoint, checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        checkpoint_sha = hashlib.sha256(f.read()).hexdigest()
    total = time.time() - start_wall
    seed_result = {
        "seed_rank": rank,
        "seed": cfg["seed"] + rank,
        "metrics": metrics,
        "wall_seconds": total,
        "checkpoint_sha256": checkpoint_sha,
    }
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(seed_result, gathered, dst=0)
    if rank == 0:
        def aggregate_metric(path):
            values = []
            for item in gathered:
                value = item["metrics"]
                for key in path:
                    value = value[key]
                values.append(float(value))
            return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}

        aggregate = {
            "heldout_one_pass": {
                key: aggregate_metric(["heldout_one_pass", key])
                for key in metrics["heldout_one_pass"]
            },
            "autoregressive_64_frame": {
                key: aggregate_metric(["autoregressive_64_frame", key])
                for key in metrics["autoregressive_64_frame"]
            },
            "latency_ms_per_chunk": aggregate_metric(["latency_ms_per_chunk"]),
            "latency_ms_per_frame": aggregate_metric(["latency_ms_per_frame"]),
            "fps": aggregate_metric(["fps"]),
        }
        result = {
            "schema": "driftworld-reproduction-v1",
            "objective": cfg["objective"],
            "paper_id": "2607.15065",
            "dataset": "public Diffusion Policy Push-T demonstrations",
            "split": {"train_windows": len(train_idx), "test_windows": len(test_idx), "test_episodes": cfg["test_episodes"]},
            "metrics": aggregate,
            "per_seed": gathered,
            "compute": {
                "backend": "kubernetes",
                "gpu_model": torch.cuda.get_device_name(0),
                "gpu_count": world,
                "train_eval_wall_seconds": max(item["wall_seconds"] for item in gathered),
                "eval_seconds": time.time() - eval_start,
            },
            "parallelism": "8 independent GPU seeds coordinated with Gloo",
        }
        print("FINAL_RESULT_JSON_BEGIN")
        print(json.dumps(result, indent=2, sort_keys=True))
        print("FINAL_RESULT_JSON_END")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
