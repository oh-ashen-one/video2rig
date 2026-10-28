"""The device PyTorch work runs on, and the mixed precision to run it in: one place instead of `torch.device('cuda')`
in every file, so the reader trains and reads on an NVIDIA card (CUDA), an Apple Silicon Mac (MPS) or the processor.

    from ..device import pick, autocast
    dev = pick(cpu=a.cpu)                  # cuda > mps > cpu; V2R_DEVICE=cuda|mps|cpu overrides; cpu=True forces cpu
    with autocast(dev):                    # bf16 on cuda (as upstream); on mps bf16, else fp16, else fp32 (probed once)
        out = model(x)

V2R_AMP=bf16|fp16|fp32 forces the precision on any device (fp32 = autocast off). On CUDA nothing changes against
upstream: the same device and torch.autocast('cuda', dtype=torch.bfloat16).
"""
import contextlib
import os

import torch

_AMP = {}
_NAMES = {'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16, 'fp16': torch.float16, 'float16': torch.float16,
          'half': torch.float16, 'fp32': torch.float32, 'float32': torch.float32, 'off': torch.float32}


def _available(kind):
    if kind == 'cuda':
        return torch.cuda.is_available()
    if kind == 'mps':
        return getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
    return kind == 'cpu'


def pick(cpu=False):
    """torch.device: 'cpu' when cpu is set, else V2R_DEVICE when given (an unavailable one is an error), else the
    first of cuda, mps, cpu that this PyTorch has."""
    if cpu:
        return torch.device('cpu')
    want = os.environ.get('V2R_DEVICE', '').strip().lower()
    if want:
        kind = want.split(':')[0]
        if kind not in ('cuda', 'mps', 'cpu'):
            raise SystemExit(f'V2R_DEVICE={want}: use cuda, mps or cpu')
        if not _available(kind):
            raise SystemExit(f'V2R_DEVICE={want}: this PyTorch has no {kind} device')
        return torch.device(want)
    for kind in ('cuda', 'mps'):
        if _available(kind):
            return torch.device(kind)
    return torch.device('cpu')


def _probe(dev, dtype):
    """A forward and backward pass through the ops the reader uses (convolution, batch and layer norm, attention,
    GELU, bilinear resize, grid_sample, gather) under autocast(dev, dtype); False if any of them fails."""
    import torch.nn.functional as F
    try:
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(3, 8, 3, 1, 1).to(dev)
        bn = torch.nn.BatchNorm2d(8).to(dev)
        lin = torch.nn.Linear(8, 8).to(dev)
        x = torch.randn(2, 3, 28, 28, device=dev)
        with torch.autocast(dev.type, dtype=dtype):
            f = F.relu(bn(conv(x)))
            t = f.flatten(2).transpose(1, 2)                                  # (B, N, C) tokens
            t = F.layer_norm(t, (8,))
            q = lin(t).view(2, -1, 2, 4).transpose(1, 2)
            t = F.scaled_dot_product_attention(q, q, q).transpose(1, 2).reshape(2, -1, 8)
            t = F.gelu(lin(t))
            f = t.transpose(1, 2).reshape(2, 8, 28, 28)
            f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
            g = torch.zeros(2, 4, 1, 2, device=dev, dtype=f.dtype)
            s = F.grid_sample(f, g, mode='bilinear', padding_mode='border', align_corners=False)
            flat = f.reshape(2, 8, -1)
            m = flat.gather(-1, flat.argmax(-1, keepdim=True))
            loss = f.float().square().mean() + s.float().mean() + m.float().mean()
        loss.backward()
        ok = bool(torch.isfinite(loss).item()) and conv.weight.grad is not None
        if dev.type == 'mps':
            torch.mps.synchronize()
        return ok
    except Exception:                                                          # noqa: BLE001 (any op gap counts)
        return False


def amp_dtype(dev):
    """The autocast dtype for this device: V2R_AMP when set, bf16 on cuda (upstream), on mps and cpu the first of
    bf16, fp16 that passes the op probe, else fp32 (autocast off). Probed once per device type."""
    want = os.environ.get('V2R_AMP', '').strip().lower()
    if want:
        if want not in _NAMES:
            raise SystemExit(f'V2R_AMP={want}: use bf16, fp16 or fp32')
        return _NAMES[want]
    if dev.type == 'cuda':
        return torch.bfloat16
    if dev.type not in _AMP:
        order = (torch.bfloat16, torch.float16) if dev.type == 'mps' else (torch.bfloat16,)
        _AMP[dev.type] = next((d for d in order if _probe(dev, d)), torch.float32)
    return _AMP[dev.type]


def autocast(dev, dtype=None):
    """torch.autocast for this device in amp_dtype(dev) (or `dtype`); a no-op context when that is fp32."""
    dtype = amp_dtype(dev) if dtype is None else dtype
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(dev.type, dtype=dtype)


def describe(dev):
    """'mps, bf16' and the like, for the logs."""
    return f'{dev}, {str(amp_dtype(dev)).replace("torch.", "").replace("bfloat16", "bf16").replace("float16", "fp16").replace("float32", "fp32")}'


def synchronize(dev):
    """Wait for the device's queued work (timing)."""
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    elif dev.type == 'mps':
        torch.mps.synchronize()


def area_resize(x, **kw):
    """F.interpolate(x, mode='area', ...): on mps through the processor, as MPS's adaptive average pool refuses sizes
    that do not divide evenly (pytorch/pytorch#96056); elsewhere unchanged."""
    import torch.nn.functional as F
    if x.device.type == 'mps':
        return F.interpolate(x.cpu(), mode='area', **kw).to(x.device)
    return F.interpolate(x, mode='area', **kw)
