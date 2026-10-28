# video2rig on macOS (Apple Silicon, MPS)

This branch (`mac-mps`) runs the reader road on an Apple Silicon Mac: the joint reader trains and reads on the Mac's
GPU through PyTorch's MPS backend. Measured on a Mac Studio, Apple M3 Ultra, 256 GB, macOS, Blender 5.2 LTS,
Python 3.12, PyTorch 2.14.1, transformers 5.21. On an NVIDIA machine nothing changes: the same device and bf16
autocast as upstream.

## Install

```sh
cd video2rig
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt torch torchvision   # the default index: the macOS arm64 build with MPS
brew install ffmpeg                                                      # ffmpeg on PATH (or V2R_FFMPEG)
export V2R_BLENDER=/Applications/Blender.app/Contents/MacOS/Blender      # prepare also finds Blender.app by itself
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # True
```

Run every Blender script with `--background --factory-startup`: without `--factory-startup` Blender loads your own
add-ons (an MCP add-on registered itself in our first export). Nothing here opens a Blender window.

## The device helper (`video2rig/device.py`)

- `pick(cpu=False)`: `cuda` > `mps` > `cpu`. `V2R_DEVICE=cuda|mps|cpu` forces one (an unavailable one is an error);
  `run --cpu` (and `read_clip(cpu=True)`) still forces the processor.
- `autocast(dev)`: `torch.autocast` in `amp_dtype(dev)`. CUDA: bf16, as upstream. MPS: the first of bf16, fp16 that
  passes a forward and backward probe through the reader's ops (convolution, batch and layer norm, attention, GELU,
  bilinear resize, grid_sample, gather), else fp32. CPU: bf16 if the probe passes, else fp32. `V2R_AMP=bf16|fp16|fp32`
  forces it (fp32 turns autocast off).
- `area_resize(x, ...)`: `F.interpolate(mode='area')`, done on the processor for MPS tensors (see the gaps below).

Used by `reader/train.py`, `reader/read.py`, `reader/dense.py`, `reader/road.py` and `tracker_worker.py` (the
tracker keeps fp32 on MPS; its fp16 autocast stays CUDA-only). `reader/crops.py` and `reader/prepare.py` take the
device from their callers. `raster.py` still uses CUDA only: it rasterizes in float64, which MPS does not have, so on
a Mac it runs its numpy path (same pixels).

## Measured on the M3 Ultra (the dragon boss, 51 bones, 71k vertices, front | side start frame)

| Stage | Time |
| --- | --- |
| `export_rig.py` | 11 s |
| `render_start.py` (Cycles CPU, 64 samples, two panels) | 7 s |
| poses (3000) | 9 s |
| Blender renders, 3000 poses x 2 views (EEVEE, headless, `--copies 16`) | 146 s (0.049 s per pose) |
| hidden-joint check (6 processes, beside the render) | 25 s |
| training, MPS bf16, batch 24 | 1.04-1.08 s per step alone (300 steps: 5.4 min); 1.18-1.20 s per step with two other Blender renders on the same GPU: 9000 steps about 2.6-3.0 h |
| reading (inference) on MPS | about 0.2 s per panel picture once, little more with TTA 8 (4 frames x 2 panels: 1.6 s, 2.0 s with TTA 8) |

Per training step (a synthetic batch of 24, timed apart): forward + backward 0.87 s in bf16, 0.90 s in fp32,
1.12 s in fp16; channels-last or not makes no measurable difference; the picture damage (`augment`) 0.11 s. So the
MPS step is about 3-4x an RTX 4070's (upstream: 9000 steps in 35-46 minutes). bf16 is kept as the MPS default: it is
the fastest and it trained without trouble.

Held-out error after the 300-step smoke test (a sanity check, not a usable reader): front 13.4 px, side 13.6 px median
(p90 106 / 93). Upstream's readers in use reach 2-4 px at 9000 steps.

## What works, what does not

Works (run on this machine): install, `export_rig.py`, `render_start.py`, `video2rig.reader.prepare` end to end
(poses, headless EEVEE renders, hidden joints, training on MPS, held-out check, `reader.pt` saved), reading frames
with the reader on MPS (`read_frames`, with and without TTA).

Not tested here: `run` / `check` / `review` on a real clip (no generated clip yet: this port spent nothing on video
generation), the reader fit (`reader/fit.py`, processor only, should not care about the device), `dense.py` (patched
the same way, not run), the older road's BootsTAPIR tracker on MPS (patched to pick MPS, fp32; not run, its weights
were not downloaded).

Does not work: the auto-rig's UniRig skeleton (`video2rig.autorig.unirig`) needs CUDA and spconv; out of scope.
`autorig.skeleton` (the centre-line fallback) does not need it but was not tried.

## MPS op gaps hit

- `F.interpolate(..., mode='area')` with a scale that does not divide the size (the training damage's "lost
  resolution" step, `scale_factor` 0.5-0.8) raises `Adaptive pool MPS: input sizes must be divisible by output
  sizes` (pytorch/pytorch#96056). Fixed by `device.area_resize` (that one resize on the processor; one picture at a
  time, cheap).
- float64 is not available on MPS: `raster.py` stays on its numpy path.
- Nothing else failed in training or reading under bf16 autocast with PyTorch 2.14.1.

## Our dragon (Veyr boss), as set up for the smoke test

The rig comes from Tripo's auto-rig, and its names do not mean what the tool's name rules expect:

- `0_Left_Limb_*` / `1_Left_Limb_*` are the two FRONT legs (the -Y and +Y side), `0_Right_Limb_*` / `1_Right_Limb_*`
  the two HIND legs. The tool's side rule reads `_L` in `_Limb`, so every limb counts as "left" and no twin pairs are
  found: the left / right check (`fix_twins`) does nothing for this rig. Fixed by the `twins` field (docs/CONFIG.md):
  creature.json names the chain roots of the front legs, hind legs and wings, and `python -m video2rig.twins <project>`
  prints the 15 pairs it walks from them (`0_Left_Limb_3` has no partner and is logged). Without `twins` the tool
  pairs by names, as upstream.
- The front legs have different chains: +Y `bone_6 > bone_7 > 1_Left_Limb_0 > {bone_9, 1_Left_Limb_1}`, -Y
  `bone_11 > 0_Left_Limb_0 > 0_Left_Limb_1 > {bone_14, 0_Left_Limb_2 > 0_Left_Limb_3}`. Nothing in the reader needs
  them symmetric (every bone's head is its own keypoint); the hinges are named per side. `twins` pairs them by
  position: shoulder, upper arm, forearm, then the branches by where they lie mirrored (`bone_9`/`bone_14`,
  `1_Left_Limb_1`/`0_Left_Limb_2`); `0_Left_Limb_3` stays unpaired on the right side.
- Only the neck is a limb chain by the skeleton's shape (the legs and wings branch at their ends, and the spine bones
  nest), so the tool found no standing legs: no foot lock, no steps. Fixed by the `limbs` field (docs/CONFIG.md): the
  four legs are named down to their paw bones (`bone_9`, `bone_14`, `1_Right_Limb_4`, `0_Right_Limb_4`, the bones that
  carry most of each paw's skin) and stand; the wings (root > wrist > tip) are free limbs. `python -m video2rig.limbs
  <project>` prints them. The front paw bones point down into the floor (Tripo), so the fit's flat-foot term skips
  them; the foot lock still pins them. The right hind toe's tip lies 0.027 above the floor, just over the flat-foot
  test's 5% of the height, so that foot gets the lock but not the flat-foot term either.
- `Head_0..3`, `bone_36`, `bone_37` carry no skin weight (a copy of a wing chain): `fit_bones: auto` leaves them out
  of the fit; their keypoints are never "seen" (no skin of their own), so the reader learns them as "not here".
- No foot or toe names: `facing` must be given.

`creature.json` used:

```json
{
 "name": "dragon",
 "root_bone": "tripo::Spine_0",
 "facing": "+X",
 "fit_bones": "auto",
 "hinges": {
  "tripo::1_Left_Limb_0": "front",
  "tripo::0_Left_Limb_1": "front",
  "tripo::1_Right_Limb_2": "back",
  "tripo::0_Right_Limb_2": "back"
 },
 "twins": [["bone_11", "bone_6"], ["tripo::0_Right_Limb_0", "tripo::1_Right_Limb_0"],
           ["VeyrWingPosZRoot", "VeyrWingNegZRoot"]],
 "limbs": {
  "front_l": {"chain": ["bone_6", "bone_7", "tripo::1_Left_Limb_0", "bone_9"], "foot": "bone_9", "standing": true},
  "front_r": {"chain": ["bone_11", "tripo::0_Left_Limb_0", "tripo::0_Left_Limb_1", "bone_14"], "foot": "bone_14", "standing": true},
  "hind_l": {"chain": ["tripo::1_Right_Limb_0", "tripo::1_Right_Limb_1", "tripo::1_Right_Limb_2", "tripo::1_Right_Limb_3", "tripo::1_Right_Limb_4"], "foot": "tripo::1_Right_Limb_4", "standing": true},
  "hind_r": {"chain": ["tripo::0_Right_Limb_0", "tripo::0_Right_Limb_1", "tripo::0_Right_Limb_2", "tripo::0_Right_Limb_3", "tripo::0_Right_Limb_4"], "foot": "tripo::0_Right_Limb_4", "standing": true},
  "wing_l": {"chain": ["VeyrWingNegZRoot", "VeyrWingNegZWrist", "VeyrWingNegZTip"], "standing": false},
  "wing_r": {"chain": ["VeyrWingPosZRoot", "VeyrWingPosZWrist", "VeyrWingPosZTip"], "standing": false}
 }
}
```

`limbs`, unlike `twins`, changes the keypoints: the new limbs' end bones whose skin spreads get side points (paws,
wing tips): 78 keypoints instead of 64. The reader trained before it (64 keypoints) still reads and fits: reading and
the fit take the keypoint list stored in `reader.pt`, never the creature's. But the next reader trained gets 78, and
`--keep` on the old renders needs the hidden-joint check made again first
(`python -m video2rig.reader.visible <project>`; train.py now stops with that message instead of misreading
`visible.npz`). The joint ranges did not change (0 of 129 channels).

(`twins` was added after the reader's training started; it does not change the reader: the keypoints are the same
64, checked against `reader-300.pt`, and the training loss never asks for a twin. It changes the reading's fit, and
the random poses a reader trained from now on would get: twin paws and wing tips brought together in some.)

Start frame: the default front | side framing cut the dragon's head and tail out of the side panel (it is twice as
long as it is tall). Rendered with
`--facing +X --widths 840,1080 --margin-wide 1.25`: the whole body in both panels, room above for rearing.

## Commands that worked

```sh
P=~/dragon-boss-fight/Local/video2rig/dragon
B=/Applications/Blender.app/Contents/MacOS/Blender
cd $P
$B model.blend --background --factory-startup --python ~/video2rig/blender/export_rig.py -- --out rig
$B model.blend --background --factory-startup --python ~/video2rig/blender/render_start.py -- --out start \
   --facing +X --widths 840,1080 --margin-wide 1.25
cd ~/video2rig
V2R_BLENDER=$B .venv/bin/python -m video2rig.reader.prepare $P --blend $P/model.blend --steps 300
# the full reader, training only on the pictures already made, in the background:
nohup .venv/bin/python -m video2rig.reader.train $P --steps 9000 --out reader/reader.pt > $P/reader/train-full.log 2>&1 &
```
