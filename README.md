<div align="center">

<img src="docs/assets/banner.svg" alt="horos — annotate, train, evaluate, deploy" width="100%">

<br>

[![CI][ci-shield]][ci-url]
[![License][license-shield]][license-url]

**horos** (ὅρος — *boundary, definition*) is the path that takes a detection
or instance-segmentation model into production: one tool that carries a
dataset from raw images through an active-learning annotation loop, training,
and evaluation to a deployable artifact — with a web UI, a Python API, and a
CLI that share one capability set.

[Quickstart](#quickstart) ·
[Web UI](#web-ui) ·
[Active learning loop](#loop--active-learning) ·
[Models](#models) ·
[Platforms](#platform-support) ·
[Installation](#installation) ·
[Roadmap](#roadmap)

</div>

## Quickstart

Install ([details & Jetson notes below](#installation)):

```bash
pip install horos   # lightweight core: datasets, annotation, web UI — no torch
horos install       # ML stack (torch / rfdetr / albumentations / transformers), matched to your machine
horos doctor        # verifies the environment; --fix installs what's missing
```

`pip install horos` deliberately ships without the ML stack: the right torch
build depends on your platform (Windows needs a CUDA index, Jetson needs the
JetPack wheel, GPU-less Linux wants the 2 GB-smaller CPU build) and pip cannot
make that call. `horos install` detects your GPU and installs the right
builds; ML commands check the environment on startup and tell you exactly
what to run if something is missing or mis-built.

Run the whole pipeline from the terminal:

```bash
mkdir my-project && cd my-project
horos init my-project        # an empty directory becomes the project itself
horos import path/to/data    # COCO / YOLO / VOC / Darknet / VIA / LabelMe, dir or zip
horos loop select --count 20 # pick the next batch to label (diverse, or model-scored)
horos loop train             # train the open round on everything labeled so far
horos loop close             # review, then open the next round
horos train                  # or a full training run with derived hyperparameters
horos models                 # the project's trained models (completed runs)
horos infer photo.jpg        # newest completed run, unless you pass --run
horos export-model --format onnx   # ONNX / TensorRT / TFLite with a model card
horos serve                  # POST /predict from an export bundle or checkpoint
horos ui                     # web UI: dataset, annotate, loop, train, evaluate, lab
horos catalog                # architectures horos can train, with their licenses
```

Project commands find the project by walking up from the current directory, so
`--project` is optional once you are inside one; `--run` defaults to the newest
completed run. Both still accept an explicit value from anywhere.

Or from Python — every UI action has a scriptable twin:

```python
import horos.api as api

project = api.open_project("my-project")

# the active-learning loop: pick → label (here, or elsewhere) → train → review
rnd = api.select_round(project, count=20)       # every pick carries a score and a reason
rnd = api.train_round(project, rnd.number)      # trains on all labels, holds out test/valid
print(api.loop_advice(project).title)           # "keep going" / "flattening" / "goal reached"
api.close_round(project, rnd.number)

# or a plain training run
record = api.start_training(project, api.TrainRunConfig(model="rfdetr-small"))
# ... poll api.training_status(project, record.run_id) ...
report = api.get_eval_report(project, record.run_id, "test")
```

<div align="center">
  <img src="docs/assets/pipeline.svg" alt="annotate → train → evaluate → deploy" width="100%">
</div>

## Web UI

`horos ui <project>` serves six pages on localhost: Dataset, Annotate, Train,
Evaluate, Experiments and Lab.

### Dataset

Import by dropping a zip (COCO / YOLO / VOC / Darknet / VIA / LabelMe — format
is auto-detected), get a validation report with actionable errors, per-class
statistics, and train / valid / test sets that only labeled photos belong to: a
photo joins a set the first time it is labeled, by a stable hash in the shares
you choose (70 / 10 / 20 by default), and never changes set — so the test set
is never trained on. Unlabeled photos are in no set.

<img src="docs/assets/screens/dataset.png" alt="Dataset page" width="100%">

### Annotate

The keyboard-first annotator on its own page: browse every photo (filter by
queue state, split and class), draw boxes and polygons by clicking with
SAM 2.1, manage classes, run auto-label, review pseudo-labels. Accepting a
shape opens the class menu pre-filled from the pseudo label or model
prediction under it; one tick keeps a class for the next objects.

### Loop — active learning

The loop is a four-step page you can follow without reading any help text:
**Select → Label → Train → Review**, then the next round. Its Label step
opens the annotate page on the round's photos and tracks the progress.

- **Select.** Choose how many photos the round should have (a fixed count, or a
  percentage of the unlabeled pool). With no labels yet, the batch is spread
  over the data by DINOv2 embeddings (k-center greedy). Once labels exist, the
  loop scores the pool with your latest model — or OWLv2 zero-shot from the
  class names before there is one — using
  [Portable Active Learning](https://arxiv.org/abs/2605.10349) (PAL): per-class
  true-positive probabilities, class-weighted image entropy, rare-class
  budgets and a similarity penalty. Every pick records its score and reason.
  Pick the model family (boxes or segmentation), pseudo-labeling on/off and
  box vs. polygon shapes right here; the choices persist per project.
- **Label.** The annotate page opens on the round's photos, pre-filled
  with pending pseudo-labels to correct instead of a blank image. Boxes and
  polygons are drawn by clicking with SAM 2.1; the edit tool adds a vertex on
  an edge click and merges vertices when one is dragged onto another. Photos
  unfit for training can be skipped together with their look-alikes (embedding
  similarity, adjustable threshold) — skipped photos leave the pool and the
  round refills at the end of the queue. Several annotators can open the same
  round and each get their own share of its photos.
- **Train.** One button. Each round continues from the previous run's weights
  (optimizer fresh, class head resized, so new classes are fine) and trains
  about half the epochs; switch to Fresh to start from the published weights.
  Newly labeled photos are bucketed by a stable hash
  into a held-out test set (20 %), a validation set (10 %) and training data;
  a photo never changes split, and the test set is never trained on. Live
  loss curves while it runs.
- **Review.** A learning curve of train vs. test mAP against photos labeled,
  the round's metric and its delta to the previous round, and a plain verdict:
  keep going, gains are flattening, check the labels, or goal reached.

Machine-generated geometry — pseudo-labels, autolabel, SAM polygons — is
always stored as pending with a score, never as human work.

<img src="docs/assets/screens/annotate.png" alt="Annotate page" width="100%">

### Train

One click to start: hyperparameters are derived from your dataset's statistics
**with the reasoning shown**, and every value can be overridden. Live loss/mAP
curves, a run queue with in-place editing, resume with full optimizer state,
OOM auto-backoff, a selectable best-checkpoint criterion, and a post-run
verdict with concrete suggestions.

<img src="docs/assets/screens/train.png" alt="Training page" width="100%">

### Evaluate

Drop photos, GIFs, or videos onto a trained model and browse per-frame
predictions in a gallery viewer (confidence slider, frame-by-frame
navigation). COCO metrics with per-class AP and PR curves, persisted per run.

<img src="docs/assets/screens/evaluate.png" alt="Evaluate page" width="100%">

### Experiments and Lab

**Experiments** lists every run with its scores, compares hyperparameters and
metrics side by side, flags runs whose dataset fingerprint differs (their
metrics are not comparable), and keeps notes and tags. **Lab** is where a
trained model meets new data: drop photos, GIFs or videos, see boxes or
polygons overlaid, export to ONNX / TensorRT / TFLite with a `model_card.json`,
and start `horos serve` for an HTTP `POST /predict` endpoint.

## Models

All registered weights are Apache-2.0. Nothing is bundled — weights download
on first use and cache locally.

**Detection (trainable)**

| Model | Params | Input | Notes |
|---|---|---|---|
| RF-DETR Nano | 30.5 M | 384 px | fastest — Jetson-friendly real-time |
| RF-DETR Small | 32.1 M | 512 px | fast — good default for Jetson |
| RF-DETR Medium | 33.7 M | 576 px | balanced accuracy/latency |
| RF-DETR Large | 129 M | 704 px | highest accuracy — desktop GPU recommended |

**Instance segmentation (trainable)**

| Model | Params | Input | Notes |
|---|---|---|---|
| RF-DETR-Seg Nano | 33.6 M | 312 px | fastest masks — Jetson-friendly |
| RF-DETR-Seg Small | 33.7 M | 384 px | fast masks — good default for Jetson |
| RF-DETR-Seg Medium | 35.7 M | 432 px | balanced mask quality/latency |
| RF-DETR-Seg Large | 36.2 M | 504 px | high mask quality — desktop GPU recommended |
| RF-DETR-Seg XLarge | 38.1 M | 624 px | highest mask quality — desktop GPU only |
| RF-DETR-Seg 2XLarge | 38.6 M | 768 px | best masks, slowest — desktop GPU only |

The loop picks a model for you — RF-DETR-Seg Nano when most labels are
polygons, RF-DETR Nano otherwise — and records why; any trainable key can be
chosen instead.

**Annotation assistants (not for deployment)**

| Model | Params | Role |
|---|---|---|
| OWLv2 Base / Large | 155 M / 437 M | open-vocabulary zero-shot pseudo-labels from class names or text prompts |
| DINOv2 Small | 22.1 M | one embedding per image — cold-start batch selection, look-alike skipping, PAL similarity |
| SAM 2.1 Hiera-Tiny / Small | 38.9 M / 46 M | click-to-mask drawing on the canvas; refines pseudo-label boxes into polygons |
| SAM ViT-B | 94 M | batch conversion of existing box annotations into polygons |

RF-DETR XL/2XL are deliberately unregistered: their weights are not Apache-2.0
(PML 1.0). Loading them requires an explicit `acknowledge_non_apache=True`.

## Platform support

| Capability | Ubuntu (CUDA) | Windows | macOS | Jetson |
|---|:-:|:-:|:-:|:-:|
| Dataset management & annotation | ✅ | ✅ | ✅ | ✅ |
| Pseudo-labeling & selection (OWLv2, DINOv2, SAM 2.1) | ✅ | ✅ | ✅ (MPS/CPU, slower) | ✅ |
| Training | ✅ | ✅ | small-dataset validation only | discouraged, not blocked |
| Inference & evaluation | ✅ | ✅ | ✅ | ✅ |
| TensorRT export | ✅ | ✅ | ❌ refused explicitly | ✅ |
| TFLite export (`horos install --tflite`) | ✅ | ✅ | ✅ | ✅ (CPU conversion) |
| `horos serve` — ONNX / TFLite | ✅ | ✅ | ✅ | ✅ |
| `horos serve` — TensorRT engine | ✅ | ✅ | ❌ refused explicitly | ✅ (the engine built there) |

Unsupported combinations raise a clear error at the API layer and show up as
disabled buttons with an explanation in the UI — never a silent CPU fallback.
Device priority: CUDA → MPS → CPU, recorded in each run's metadata.

## Installation

Two steps, on every platform:

```bash
pip install horos   # the core — datasets, annotation, web UI (no ML deps)
horos install       # the ML stack, matched to this machine
```

`horos install` detects your OS, NVIDIA driver and CUDA version and runs the
right pip commands (`--dry-run` shows them first, `--cpu` forces the CPU
build). `horos doctor` re-checks everything and plans the same fixes — it also
catches the classic trap of a CPU-only torch sitting on a GPU machine.

<details>
<summary>What <code>horos install</code> decides for you</summary>

| Platform | torch source |
|---|---|
| Linux + NVIDIA GPU | PyPI (Linux wheels bundle CUDA) |
| Linux without GPU | PyTorch CPU index (saves ~2 GB) |
| macOS | PyPI universal build (MPS) |
| Windows + NVIDIA GPU | PyTorch index matching your driver's CUDA (cu118 … cu132) — the PyPI Windows wheel is CPU-only |
| Windows without GPU | PyPI (CPU) |
| AMD GPU (Linux or Windows) | AMD's ROCm index, for the detected gfx architecture; PyPI has no AMD build |
| Jetson | **never pip-installed** — see below |

For Linux/x86_64 CI and containers where the default PyPI torch is already
right, `pip install horos[ml]` installs the same stack in one shot.

**AMD GPUs (ROCm).** No flag needed: an AMD GPU is handled like an NVIDIA
one. `horos install` finds the card, works out its gfx architecture and
installs AMD's ROCm wheels, because PyPI has no AMD torch at all. `--cpu`
opts out.

The architecture comes from `clinfo` (the AMD display driver installs it, so
this works before ROCm exists), from AMD's own `rocm-bootstrap` when present,
or from `rocminfo`. If none of them can tell, horos installs the CPU wheel
and says why rather than guessing, because the wrong architecture installs
kernels the GPU cannot run; set `HOROS_ROCM_ARCH=gfx1201` to name it
yourself. `horos doctor` reports a CPU-only torch on an AMD machine and
`doctor --fix` repairs it.

The wheels carry the ROCm runtime (~1.4 GB), so only a current driver is
needed, no HIP SDK. torch exposes a ROCm GPU through `torch.cuda`, so horos
selects it as device `cuda` and records the real GPU name in the run
metadata. TensorRT export stays NVIDIA-only.

</details>

The repo also ships bootstrap scripts that create `./.venv`, install the core,
and run `horos install` for you:

```bash
./install.sh        # Ubuntu / macOS / Jetson
install.bat         # Windows
```

On a fresh Windows machine `python` on PATH is only the Microsoft Store
placeholder, not an interpreter. `install.bat` detects this and asks whether to
install Python 3.12 for you (per-user, via winget or the python.org installer)
and whether to add it to your user PATH, then continues with the horos install.
Set `HOROS_AUTO_INSTALL_PYTHON=1` to answer yes to both without prompting (CI).

Recreating `.venv` reinstalls the ML stack from scratch, and `horos
install` detects the GPU again, so GPU support comes back on its own.
Both scripts forward their arguments to `horos install` if you need to
steer it (`install.bat --cpu`).

**Use a dedicated environment.** horos pins `rfdetr` exactly (upstream has had
silent annotation-corruption bugs; reproducibility wins) and requires
`transformers >= 5.1` — installing into a shared ML environment will upgrade
`transformers`, `supervision`, `huggingface-hub` and friends, which can break
other projects living in that environment.

### Jetson (read this — it matters)

On Jetson, torch **must** come from NVIDIA's JetPack-matched wheel — the PyPI
torch has no CUDA support there. `pip install horos` is safe (the core has no
torch dependency), and `horos install` never pip-installs torch on Jetson: it
prints the JetPack steps, installs `rfdetr` with `--no-deps` so pip can never
swap torch out, and adds the training stack once the JetPack torch is in
place. horos also warns at backend load time when it detects a Jetson
platform where `torch.cuda.is_available()` is False.

Use a venv created with `--system-site-packages` so the JetPack torch stays
visible (`./install.sh` does this automatically on Jetson):

```bash
pip install horos
# torch/torchvision: install the NVIDIA wheel matching your JetPack version —
# https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
horos install       # rfdetr (--no-deps), training stack, albumentations, transformers
```

## Roadmap

- [x] Project & dataset core — formats, validation, stats, splits
- [x] Manual annotation — bbox + polygon, SAM click-to-draw, multi-annotator
- [x] Auto-labeling — OWLv2 open-vocabulary, SAM boxes-to-polygons
- [x] Active learning loop — DINOv2 cold start, PAL acquisition, pseudo-labels, skip look-alikes, growing held-out test set, learning curve
- [x] Instance segmentation — RF-DETR-Seg training, polygon pseudo-labels, mask output in exports
- [x] Training — derived hyperparameters, queue, resume, live monitoring
- [x] Evaluation — media gallery, COCO metrics, per-class analysis
- [x] Error analysis — confusion matrix, worst-case mining, colour-coded overlays
- [x] Experiment management — run comparison, dataset fingerprints, notes & tags
- [x] Export & deploy — ONNX / TensorRT / TFLite, model cards, parity checks, `horos serve`

## Development

```bash
bash scripts/setup_local.sh --dev          # install.sh/.bat + [dev] extras + horos doctor
bash scripts/setup_local.sh --light --dev  # torch-free core only (annotation/dataset work)
bash scripts/local_test.sh --lint          # invariants first, then pytest and ruff
```

The setup script runs the same `install.sh` / `install.bat` users run, then
`horos doctor` as the installation check — a missing or mis-built dependency
fails the script instead of surfacing later as a training-time ImportError.
Doing it by hand is equivalent:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[dev]      # the core is torch-free by design
horos install              # ML stack — needed for the backend/training tests
horos doctor               # must print "Environment OK."
pytest tests/test_invariants.py && pytest
```

`tests/test_invariants.py` runs first for a reason: it statically enforces the
architecture — model dependencies live only in `horos/backends/`, `import horos`
never drags in torch, and the UI talks to the core exclusively through the web
API. Models are adapters; the workflow is the product.

## License

Distributed under the [Apache License 2.0](LICENSE). Model weights are
downloaded at runtime and cached locally — horos never bundles or
redistributes them, and each model's license is recorded in the registry,
shown in the UI, and stamped into every training run.

## Acknowledgments

[RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow ·
[OWLv2](https://arxiv.org/abs/2306.09683) by Google Research ·
[DINOv2](https://github.com/facebookresearch/dinov2) and
[Segment Anything 2](https://ai.meta.com/sam2/) by Meta AI ·
[Portable Active Learning for Object Detection](https://arxiv.org/abs/2605.10349)
by Sharma, Bersamin & Subramanian

[ci-shield]: https://img.shields.io/github/actions/workflow/status/SJ-Chuang/horos/ci.yml?branch=main&style=for-the-badge&label=CI
[ci-url]: https://github.com/SJ-Chuang/horos/actions/workflows/ci.yml
[license-shield]: https://img.shields.io/github/license/SJ-Chuang/horos.svg?style=for-the-badge
[license-url]: https://github.com/SJ-Chuang/horos/blob/main/LICENSE
