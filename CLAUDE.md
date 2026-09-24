# CLAUDE.md — horos

> Working document for Claude Code on this project. Before implementing anything, read §2 "Rules That Must Never Be Violated" and §7 "Development Process" in full.

---

## 1. What This Project Is

**horos** (ὅρος, ancient Greek for "boundary stone, limit, definition") is an end-to-end toolchain for detection tasks: annotate → train → evaluate → deploy.

The problem it solves is not "which model is more accurate". It is **the path that gets a detection model into production**.

### Core design premises

1. **Models expire; workflows do not.** The first supported backend is RF-DETR, but anything `rfdetr`-specific must be isolated behind an adapter layer. When the architecture is replaced two years from now, users' code must not have to change.
2. **Licensing is a first-class concern, not a footnote in the docs.** Every model, weight file and dataset carries queryable license metadata.
3. **The target deployment environment is NVIDIA Jetson.** When "runs on a desktop" conflicts with "runs on Jetson", Jetson wins.
4. **Naming is not tied to detection.** The project will expand beyond detection; no new module may contain `det` in its name.

### Support matrix (v1)

| Item | Scope |
|---|---|
| Models | RF-DETR Nano / Small / Medium / Large (detection) and RF-DETR-Seg Nano … 2XLarge (instance segmentation), all Apache 2.0 |
| Auto-labeling | OWLv2 open-vocabulary zero-shot (Apache 2.0) |
| Data formats | COCO JSON, YOLO, LabelMe (read + write); Pascal VOC, Darknet, VIA (import only); plain photos with no labels (import only) |
| Export | ONNX, TensorRT, TFLite |
| Interfaces | Python API, Web API (Flask), WebUI (Flask) |

### Explicit non-goals

- No bundling or redistribution of model weights (licensing risk) — always download at runtime and cache
- No semantic segmentation. horos handles detection and instance segmentation
- No wrapping of cloud training services; no uploading user data to any external service
- No search-based hyperparameter optimization in v1 (see E5)

---

## 2. Rules That Must Never Be Violated

These are the load-bearing rules. A change that violates any one of them is rejected outright, no matter what else it does well.

### R1 — Model dependencies may only appear under backends/

`horos/core`, `horos/api`, `horos/web` and `horos/ui` must **not** contain `import rfdetr` or `from rfdetr import`, and must not import `torch` or `transformers` directly.

Those dependencies may only appear under `horos/backends/<backend_name>/`. Everything above calls through the abstract interface defined in `horos/backends/base.py`.

The same rule applies to OWLv2 (`transformers`) and to any model backend added later.

**The purpose of this rule is replaceability.** When RF-DETR is superseded by a better architecture two years from now, the change should be confined to one directory and users' code should not need to change.

`tests/test_invariants.py` enforces this by statically scanning the source.

### R1b — Backends are always lazily loaded

After `import horos`, `sys.modules` must **not** contain `torch`, `rfdetr` or `transformers`.

Backend modules are imported only on first actual use. None of the following may trigger a backend load:

- `import horos`
- Creating or opening a project, reading/writing datasets, format conversion
- Listing available models and their metadata (the registry holds static data only and does not import backends)
- Starting the web service or opening the annotation page

Therefore **type annotations always go inside `TYPE_CHECKING` blocks**; backend types must never be imported at module level.

This rule has a dedicated runtime test (see E4-T10). Without it, lazy loading will silently break in some commit that added a type annotation, and nobody will notice.

Practical effect: for someone who only uses the annotation features, `import horos` should complete in under a second and must not require a working GPU.

### R2 — The three layers may not be bypassed

```
WebUI  ──→  Web API  ──→  Python API  ──→  core
(templates/JS) (Flask routes) (public functions)
```

- The WebUI must not import horos core modules; it only calls the Web API
- Web API routes contain no business logic — only parameter validation and delegation
- Business logic always lives in the Python API layer

**The purpose of this rule is testability.** Follow it and verifying a feature only requires testing the Python API, with UI tests degenerating into "does the button hit the right endpoint". Break it and the UI needs a second complete set of test logic.

### R3 — License metadata is a first-class citizen

Every model definition must carry a `license` field, and it must surface in three places: the WebUI model picker, the training run metadata, and `model_card.json` next to the export artifact.

Hardcoding the assumption "it's all Apache anyway" is not allowed.

### R4 — Long-running work always reports structured progress events

Training, batch inference and auto-labeling must not return results only at the end. They must report progress continuously through a unified event interface.

- Events are schema'd `pydantic` objects, not free-form strings
- The Python API delivers them via callback or generator; the Web API turns them into SSE or a polling endpoint
- Event kinds must cover at least: started, progress update, metrics update, warning, completed, failed

Backend implementations do not get to choose their own reporting format — they always use the event types from `horos/backends/base.py`. This is what lets all three interface layers share one progress display.

### R5 — Pin versions

The `rfdetr` version must be pinned exactly. The package has had several **silent annotation-corruption bugs** (under specific augmentation settings), and a floating version makes training results unreproducible. Upgrading is a standalone task that requires a full regression run, not something done in passing.

### R6 — ROS 2-compatible naming

Module and package names must follow ROS 2 conventions: all lowercase, underscore-separated, not starting with a digit. Topos may reference horos directly in the future.

### R7 — Platform neutrality

Four platforms are supported: Ubuntu, macOS, Windows, Jetson. The following are always forbidden:

- **Hardcoded path separators.** Always use `pathlib.Path`; never string-concatenate `/` or `\\`
- **Assuming POSIX file locks.** `fcntl` does not exist on Windows. Cross-platform locking must use atomic file creation (`O_EXCL`) or a database-level mechanism
- **Assuming fork.** Windows and macOS default to `spawn`, which re-imports the main module. Any code that starts a subprocess (including DataLoader `num_workers`) must be spawn-safe, and entry points need an `if __name__ == "__main__"` guard
- **Assuming CUDA exists.** Device selection always goes through the `backends/device.py` abstraction; never write `.cuda()` or `device="cuda"` directly
- **Assuming symlinks work.** Creating a symlink on Windows requires administrator rights; dataset splitting must not depend on symlinks — use copies or an index file
- **Long paths.** Windows defaults to a 260-character path limit; be careful when generating nested output directories

CI must run at least two runners: Ubuntu and Windows. macOS and Jetson are verified manually.

### R8 — Every commit and push must leave the remote pullable

The author works across several machines and updates content and versions on the others. A commit or push that breaks the next `git fetch` / `git pull` — or that silently discards another machine's work — is a defect regardless of the code it carries.

Concretely:

- **Never rewrite published history.** No `git push --force` (or `--force-with-lease`), and no amending, rebasing or squashing of commits that have already been pushed. Another clone is already built on them.
- **Always push a real, tracking branch.** Never push from a detached HEAD, and make sure the local branch tracks its remote counterpart (`git push -u` on the first push) so that a bare `git pull` works afterwards.
- **Integrate the remote before pushing.** Fetch first; if the remote has moved on, merge it (or rebase only your *unpushed* commits) and re-run the tests before pushing.
- **Leave a clean working tree.** No half-staged or uncommitted leftovers at the end of a task — they turn the next machine's `git pull` into a conflict.
- **Never commit anything that makes a clone expensive or broken**: weights, datasets, `~/.horos/` cache contents, run outputs, virtualenvs. `.gitignore` must actually cover these (see §10).
- **Verify, don't assume.** After pushing, `git fetch && git status` must report the branch as up to date with its upstream and the working tree as clean. Report that check as part of the task.

---

## 3. Directory Structure

```
horos/
├── core/                  # data models, project structure, configuration
│   ├── project.py         # Project: the single authority on the on-disk layout
│   ├── dataset.py         # Dataset / Split / Annotation data models
│   ├── formats/           # COCO and YOLO read/write and conversion
│   └── registry.py        # model registry (including license metadata)
├── backends/              # the only home for model dependencies (R1)
│   ├── base.py            # abstract interface + event types (R4)
│   ├── env.py             # torch / CUDA environment checks
│   ├── rfdetr/            # the only place allowed to import rfdetr
│   └── owlv2/             # the only place allowed to import transformers
├── api/                   # Python API — business logic lives in this layer
│   ├── annotate.py
│   ├── autolabel.py
│   ├── train.py
│   ├── evaluate.py
│   ├── experiment.py
│   └── export.py
├── web/                   # Web API (Flask)
│   ├── app.py
│   └── routes/            # thin routes
├── ui/                    # WebUI (Flask templates + frontend)
│   ├── templates/
│   └── static/
└── cli.py

tests/                     # all test scripts live here
├── test_invariants.py     # static R1/R2 checks, runs first in CI
├── unit/
├── api/                   # Python API tests
├── web/                   # Web API tests
├── contract/              # three-layer capability parity
├── fixtures/              # small test datasets
└── ui_scenarios/          # interface scenario checklists (markdown)
```

`~/.horos/weights/` holds the runtime-downloaded weight cache.

---

## 4. Technical Constraints

- Python >= 3.10
- Web framework: Flask (do not introduce FastAPI or Django)
- Configuration objects always use `pydantic`, never bare dicts
- Tests use `pytest`; formatting/linting uses `ruff`
- **The `hidden` attribute hides.** `static/controls.css` resets `[hidden]` with `!important` because every control class sets `display`, and an author rule beats the browser's own `[hidden]` rule — without the reset a hidden `.btn` renders as an empty pill
- Library code must not call `print()` — always use `logging`
- TensorRT engines are not portable: the export flow must be run on the target device, and both the API and the UI must say so explicitly

### Dependency strategy: a single environment

Both `rfdetr` and `transformers` are **primary dependencies**, installed alongside `pip install horos`. No virtual environment isolation, no subprocess isolation.

The cost is install size (~2–3 GB including torch). That is a deliberate trade for a simpler workflow.

### Platform support matrix

| Feature | Ubuntu (x86 + CUDA) | Windows | macOS | Jetson |
|---|---|---|---|---|
| Dataset management, format conversion | Full | Full | Full | Full |
| Manual annotation | Full | Full | Full | Full |
| Auto-labeling (OWLv2) | Full | Full | Works; slower on MPS/CPU | Full |
| Training | Full | Full | Only for validating the workflow on small datasets | Not recommended, but not blocked |
| ONNX / TFLite export | Full | Full | Full | Full |
| TensorRT export | Full | Full | **Unsupported** | Full |
| Inference serving | Full | Full | Full | Full |

Unsupported combinations must raise an explicit error at the API layer and disable the button with an explanation at the UI layer. They must **never fail silently or quietly fall back to CPU**.

Device priority: CUDA → MPS (Apple Silicon) → CPU. The device actually selected must be recorded in the run metadata.

### The Jetson torch trap (required reading)

**On Jetson, torch must be the dedicated wheel NVIDIA releases for the matching JetPack version. The torch on PyPI has no CUDA support.**

Running `pip install horos` directly on Jetson may install a CPU-only torch that overwrites the system's CUDA build. This failure is **silent** — the program still runs, just an order of magnitude slower at inference, and it is extremely hard to trace.

Therefore:

1. The Jetson install path must use `--no-deps`, or a pre-prepared environment
2. `horos/backends/env.py` must check at import time and emit an explicit warning when the platform is detected as Jetson but `torch.cuda.is_available()` is False
3. The install documentation must give Jetson its own section, not mixed into the general instructions

---

## 5. Phase Plan

The user's priority order is: annotate → train → evaluate → deploy. Annotation has prerequisites, so the actual schedule is:

| Phase | Content | Rationale |
|---|---|---|
| **P0** | E1 + E4 + E9 skeleton | Prerequisites for annotation. Without data models there is nowhere to store annotations; without the model layer there is no auto-labeling |
| **P1** | E2 manual annotation + E3 auto-labeling | The user's top priority |
| **P2** | E5 training | |
| **P3** | E6 evaluation and testing | |
| **P4** | E7 experiment management + E8 export and deployment | |
| **P5** | E10 active learning loop | Replaces the standalone annotate / autolabel / review pages with one select → label → train → review loop; the canvas engine from E2 is kept |

---

## 6. Epics

Each Epic contains User stories (S) and Tasks (T). A Task's definition of done must include the corresponding test file.

---

### E1 — Project and Dataset Core

**Goal:** provide the single source of truth for data. Every other Epic builds on this layer.

#### User stories

- **E1-S1** (Python API) A researcher creates a horos project from an existing COCO directory in three lines of code
- **E1-S2** (Python API) An engineer converts a CVAT-exported YOLO dataset to COCO without losing any annotation
- **E1-S3** (WebUI) A non-engineer uploads a zip and sees a parse summary: how many images, how many classes, how many instances per class
- **E1-S4** (Python API) A user imports a problematic dataset (boxes out of bounds, non-contiguous class ids, missing images) and the system states exactly what is wrong instead of failing silently
- **E1-S5** (Web API) The frontend requests dataset statistics for use in downstream hyperparameter derivation

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E1-T1 | `Project` object and on-disk layout spec | Create, load, validate; `tests/unit/test_project.py` |
| E1-T2 | `Dataset` / `Split` / `Annotation` data models | Support bbox and polygon; `tests/unit/test_dataset_model.py` |
| E1-T3 | COCO JSON read/write | Support the `_annotations.coco.json` convention; `tests/api/test_format_coco.py` |
| E1-T4 | YOLO format read/write | Including `data.yaml`; `tests/api/test_format_yolo.py` |
| E1-T5 | Round-trip format conversion | COCO→YOLO→COCO leaves annotations identical; `tests/api/test_format_roundtrip.py` |
| E1-T6 | Dataset validator | Five common error classes, each with an explicit message; `tests/api/test_dataset_validate.py` |
| E1-T7 | Statistics computation | Class distribution, relative object area distribution, image size distribution; `tests/api/test_dataset_stats.py` |
| E1-T8 | Split management — only labeled photos belong to train / valid / test: a photo joins a set when it first carries a confirmed annotation, by a stable hash in the project's ratios (default 70/10/20), and never changes set; unlabeled photos are in no set and reach no training snapshot; "Assign splits" fills in the unassigned, "Reshuffle" re-draws labeled photos with a warning | `tests/api/test_split.py` |
| E1-T9 | Web API endpoints | `tests/web/test_dataset_routes.py` |
| E1-T10 | WebUI upload and summary page | Interface scenario (see §8) |
| E1-T11 | Photos without labels are an import (format `images`, detected last so no real layout is ever mistaken for one): they join the project unlabeled and in no set, from a directory, a zip or photos dropped on the Dataset page. Labels arriving later for photos the project already has attach to them (same name and content, or a bare label file whose recorded size agrees) instead of vanishing with the duplicate; identical labels are a duplicate, differing ones ask (`on_annotations`: ask / replace / merge / skip) before anything is written | `tests/api/test_format_images.py`, `tests/api/test_import_labels.py`, `tests/web/test_dataset_routes.py`, interface scenario (`tests/ui_scenarios/upload-conflicts-and-formats.md` §D–E) |

#### How it is accepted

API tests, primarily. The core condition is **E1-T5: lossless round-tripping** — once that passes, the data layer can be trusted.

---

### E2 — Manual Annotation

**Goal:** an annotation interface fast enough that nobody wants to go back to CVAT.

#### User stories

- **E2-S1** (WebUI) An annotator works entirely from the keyboard: switch class, draw box, next image — without touching a mouse menu
- **E2-S2** (WebUI) An annotator closes the browser and comes back to find progress fully preserved at the point they left
- **E2-S3** (WebUI) An annotator edits an existing box: drag corners, delete, change class
- **E2-S4** (WebUI) An annotator draws polygons for instance segmentation annotation
- **E2-S5** (Python API) An engineer inserts or corrects annotations programmatically in bulk
- **E2-S6** (WebUI) Several people annotate different images of the same project simultaneously without overwriting each other

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E2-T1 | Annotation canvas (zoom, pan, draw) | Interface scenario |
| E2-T2 | bbox creation and editing | `tests/api/test_annotate_bbox.py` + interface scenario |
| E2-T3 | polygon creation and editing | `tests/api/test_annotate_polygon.py` + interface scenario |
| E2-T4 | Class management (add, rename, recolor, merge, delete as a job); renamed and merged names stay as `Category.aliases`, and every place model output meets project classes (autolabel, loop scoring and pseudo-labels, Lab, single-image inference) maps names through them, so a model trained before a rename still lands on the right class | `tests/api/test_labels.py` |
| E2-T5 | Keyboard shortcuts | Interface scenario (including a shortcut reference table) |
| E2-T6 | Progress persistence and resume | `tests/api/test_annotate_progress.py` |
| E2-T7 | Image queue and navigation | `tests/api/test_image_queue.py` |
| E2-T8 | Optimistic locking for concurrent writes | Cross-platform implementation (no `fcntl`); two sessions writing the same image, the second gets a conflict; `tests/api/test_annotate_concurrency.py` |
| E2-T9 | Web API endpoints | `tests/web/test_annotate_routes.py` |
| E2-T10 | The editor sidebar keeps the shapes list usable: the Tools heading folds the active tool's options away (remembered per browser, and it names what it hid), the tools card takes at most half the sidebar and scrolls inside it, and the shapes list keeps a floor of 190px | Interface scenario (`tests/ui_scenarios/E2-annotator.md`) |

#### How it is accepted

Interface scenarios primarily, with API tests covering persistence and concurrency. **E2-T8 cannot be skipped** — multi-person annotation is the norm, and retrofitting this mechanism later means rewriting the data layer.

---

### E3 — Auto-labeling

**Goal:** let annotators start from corrections rather than from a blank canvas.

The backend uses **OWLv2** (`google/owlv2-*`, Apache 2.0), implemented in `horos/backends/owlv2/`.

#### User stories

- **E3-S1** (WebUI) A user enters text prompts (`forklift`, `pallet`, `person`) and the system produces pre-annotations for an entire batch of unlabeled images
- **E3-S2** (WebUI) A user adjusts the confidence threshold and sees the retained box count change live
- **E3-S3** (WebUI) The system sorts images by confidence so the user reviews the least certain first
- **E3-S4** (Python API) An engineer calls auto-labeling programmatically and post-processes the results themselves
- **E3-S5** (WebUI) A user accepts, corrects or bulk-rejects pre-annotations; accepted ones enter the official annotation set
- **E3-S6** (WebUI) First use requires a weight download, and the user sees download progress instead of a frozen screen

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E3-T1 | OWLv2 backend implementation | Follows the `backends/base.py` interface; `tests/api/test_backend_owlv2.py` |
| E3-T2 | Text-prompt-to-class mapping | One class may map to several prompt terms; `tests/api/test_autolabel_prompt.py` |
| E3-T3 | Batch inference and progress streaming | Line-delimited JSON events; `tests/api/test_autolabel_stream.py` |
| E3-T4 | Confidence filtering and NMS post-processing | `tests/api/test_autolabel_postprocess.py` |
| E3-T5 | Pre-annotations written as "pending review" | Distinguishable from human annotations in the data model; `tests/api/test_autolabel_review.py` |
| E3-T6 | Uncertainty ranking | `tests/api/test_autolabel_ranking.py` |
| E3-T7 | Weight download and caching | Resumable, with progress reporting; `tests/api/test_weights_cache.py` |
| E3-T8 | Review UI (accept/correct/reject) | Interface scenario |
| E3-T9 | Web API endpoints | `tests/web/test_autolabel_routes.py` |

#### How it is accepted

API tests cover pipeline correctness (fixture images with known prompts, asserting box counts and classes fall in a sensible range). The UI goes through interface scenarios.

---

### E4 — Model Integration Layer

**Goal:** implement R1 and R4. All model dependencies converge into `horos/backends/`, and the layers above have no idea which model is underneath. This comes first in P0.

#### User stories

- **E4-S1** (Python API) A user lists available models and sees size, expected latency and **license**
- **E4-S2** (WebUI) A user sees the Apache 2.0 marking directly in the model picker
- **E4-S3** (Python API) A maintainer adds a third model backend by implementing only the `base.py` interface, without touching core
- **E4-S4** (Python API) A user running on Jetson gets an explicit warning at startup if torch has no CUDA support, rather than discovering it when inference turns out ten times slower
- **E4-S5** (Python API) A user attempting to load RF-DETR XL / 2XL is blocked, with an error message explaining the license difference
- **E4-S6** (Python API) Exceptions raised by a backend are translated into horos's unified error types; the layers above never need to know rfdetr's exceptions
- **E4-S7** (Python API) A user who only annotates gets `import horos` in under a second, with no torch loaded along the way
- **E4-S8** (Python API) A user calling TensorRT export on macOS gets an explicit "unsupported on this platform" error, not a strange low-level exception
- **E4-S9** (WebUI) A user on macOS sees the TensorRT export button disabled, with an explanation on hover
- **E4-S10** (Python API) A user queries which features the current platform supports and gets a structured capability list

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E4-T1 | Model registry and metadata schema | Includes license, input resolution, parameter count; `tests/unit/test_registry.py` |
| E4-T2 | `backends/base.py` abstract interface | Three method groups: train, infer, export; `tests/api/test_backend_interface.py` |
| E4-T3 | Event types and progress reporting interface (R4) | pydantic schema; `tests/unit/test_events.py` |
| E4-T4 | RF-DETR backend implementation | Training + inference; `tests/api/test_backend_rfdetr.py` |
| E4-T5 | Exception translation layer | Backend exceptions become horos error types; `tests/api/test_backend_errors.py` |
| E4-T6 | **torch / CUDA environment check** | Warn on Jetson without CUDA; `tests/api/test_env_check.py` |
| E4-T7 | **Static R1 check** | `core`/`api`/`web`/`ui` must not import rfdetr, torch or transformers; `tests/test_invariants.py` |
| E4-T8 | **Static R2 check** | `horos/ui/` must not import core modules; `tests/test_invariants.py` |
| E4-T9 | Blocking non-Apache models | Loading XL / 2XL raises an error explaining the license difference; requires `acknowledge_non_apache=True` to proceed; `tests/api/test_license_guard.py` |
| E4-T10 | **Lazy-loading invariant (R1b)** | Runtime assertion that `sys.modules` has no torch/rfdetr/transformers after `import horos`; `tests/test_invariants.py` |
| E4-T11 | Lazy-loading mechanism | Backends imported on first use, type annotations guarded by `TYPE_CHECKING`; `tests/api/test_lazy_backend.py` |
| E4-T12 | `backends/device.py` device abstraction | CUDA → MPS → CPU priority, overridable; `tests/api/test_device.py` |
| E4-T13 | Platform capability query | Returns a feature-availability list for the current platform; `tests/api/test_platform_capabilities.py` |
| E4-T14 | Error handling for unsupported combinations | macOS + TensorRT raises an explicit error, no CPU fallback; `tests/api/test_unsupported_combos.py` |
| E4-T15 | RF-DETR-Seg instance segmentation models: registry entries (all sizes Apache 2.0, verified against the open package's weight list), class mapping, masks → polygons on inference, mask output in the export spec, resolution snapping to the model's patch step | `tests/unit/test_registry.py`, `tests/api/test_backend_rfdetr.py`, `tests/api/test_hparam_derive.py` |

#### How it is accepted

`tests/test_invariants.py` is the core acceptance gate and must run first in CI.

E4-T7 statically scans import statements (in a single environment rfdetr is always installable, so a runtime check cannot detect the violation); E4-T10 is the opposite — it must be a runtime check, because only a real import reveals whether torch got dragged in. Neither substitutes for the other.

E4-T13's capability list feeds both the Web API and the WebUI; the UI's disabled-button state is driven directly by it, never by hardcoded platform checks in the frontend.

---

### E5 — Training

**Goal:** a user who knows nothing about hyperparameters still gets reasonable results, while a user who does can take over completely.

#### Hyperparameter adaptation strategy

**Rule-based first; no search-based tuning in v1.** Starting values are derived from the E1-T7 statistics:

| Statistic | Influences |
|---|---|
| Total image count | epochs, augmentation strength |
| Instances per class | warmup length, class weights |
| Mean relative object area | training resolution (raise it for small objects) |
| Available VRAM | batch size, with automatic step-down retry on OOM |
| Class imbalance | sampling strategy |

Every derived value must be explicitly overridable, and **the reason for the derivation must be recorded in the run metadata** — the user has to be able to see "why did the system pick this resolution".

Search-based HPO (Optuna and friends) is left as a pluggable extension. Rationale: rule-based is usually sufficient on small datasets, and search-based has poor returns in a compute-constrained iteration setting like Jetson.

#### User stories

- **E5-S1** (WebUI) A non-engineer presses one "start training" button and it runs without filling in any hyperparameter
- **E5-S2** (WebUI) A user sees the system-derived hyperparameters together with **the reason for each**, and can override them individually
- **E5-S3** (WebUI) A user watches loss curves and validation metrics live
- **E5-S4** (WebUI) A user stops training partway through and the best weights are already preserved
- **E5-S5** (Python API) An engineer bypasses adaptation entirely and specifies every hyperparameter
- **E5-S6** (Python API) A user resumes training from an existing checkpoint, or warm-starts a new run from an earlier run's weights with a changed class set (`init_from`)
- **E5-S7** (WebUI) When training fails with OOM, the system automatically lowers the batch size, retries, and tells the user

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E5-T1 | Hyperparameter deriver | Every derived value carries a reason string; `tests/api/test_hparam_derive.py` |
| E5-T2 | Override mechanism | A partial override does not disturb the other derived values; `tests/api/test_hparam_override.py` |
| E5-T3 | Training run lifecycle | Create, run, stop, clean up; `tests/api/test_train_lifecycle.py` |
| E5-T4 | Metric streaming and persistence | `tests/api/test_train_metrics.py` |
| E5-T5 | Checkpoint management and resume | `tests/api/test_train_resume.py` |
| E5-T6 | Automatic OOM step-down | Simulated OOM halves the batch size and retries; `tests/api/test_train_oom.py` |
| E5-T6b | **spawn-safe training entry point** | Starting training from a Flask process with `num_workers>0` does not re-initialize the app; must run in Windows CI; `tests/api/test_train_spawn.py` |
| E5-T6c | Cross-platform memory detection | CUDA reads VRAM, MPS reads unified memory, CPU uses a conservative default; `tests/api/test_memory_probe.py` |
| E5-T7 | End-to-end small-dataset training | Fixture dataset completes 2 epochs and produces weights; `tests/api/test_train_e2e.py` |
| E5-T8 | Training monitoring UI | Interface scenario |
| E5-T9 | Web API endpoints | `tests/web/test_train_routes.py` |
| E5-T10 | End-to-end small-dataset **segmentation** training | Polygon fixture completes 2 epochs with RF-DETR-Seg Nano and inference returns polygons; `tests/api/test_train_seg_e2e.py` |

#### How it is accepted

**E5-T7 is a hard acceptance condition**: a 20-image fixture dataset completes 2 epochs and produces a loadable weight file. It is fast, covers the full path, and is suitable for CI.

---

### E6 — Evaluation and Testing

**Goal:** let the user know whether the model is actually usable, not just look at a single mAP number.

#### User stories

- **E6-S1** (WebUI) A user uploads a few new photos and immediately sees detection results overlaid
- **E6-S2** (WebUI) A user uploads a video and sees per-frame detection results
- **E6-S3** (WebUI) A user adjusts the confidence threshold and sees results change live
- **E6-S4** (Python API) An engineer obtains full metrics on the held-out test set (mAP, per-class AP, PR curves)
- **E6-S5** (WebUI) A user sees error analysis: which classes are missed most often, which are false-positived most often, which pairs are most confused
- **E6-S6** (WebUI) A user sees the N worst predictions and can judge directly whether it is a model problem or an annotation problem
- **E6-S7** (Python API) An engineer runs a whole directory of images in batch and exports the results

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E6-T1 | Single-image and batch inference API | `tests/api/test_inference.py` |
| E6-T2 | Per-frame video inference | `tests/api/test_inference_video.py` |
| E6-T3 | COCO metric computation | Aligned with the reference implementation; `tests/api/test_metrics.py` |
| E6-T4 | Confusion matrix and per-class analysis | `tests/api/test_error_analysis.py` |
| E6-T5 | Worst-case mining | `tests/api/test_worst_cases.py` |
| E6-T6 | Result visualization (overlay generation) | `tests/api/test_visualize.py` |
| E6-T7 | Upload-and-test UI | Interface scenario |
| E6-T8 | Error analysis UI; Per class and the confusion matrix are tabs, one full-width view at a time (side by side left the matrix scrolling and the table cut off), with the choice remembered per browser | Interface scenario |
| E6-T9 | Web API endpoints | `tests/web/test_eval_routes.py` |
| E6-T10 | Suggested operating confidence threshold: one matching pass over the split's saved detections yields the precision / recall / F-beta sweep, and the recommendation is the middle of the plateau that scores within 1 % of the peak (not the bare peak, which moves with the data); per class as well, marked when the class has under 10 boxes; `beta` picks the trade (1 balanced, 2 fewer misses, 0.5 fewer false alarms); tuning on test is called out | `tests/api/test_threshold_advice.py`, `tests/web/test_eval_routes.py`, interface scenario (`tests/ui_scenarios/E6-T8.md` §B2) |
| E6-T11 | From an error straight to the labels: the worst-image thumbnails and the overlay viewer link to `/annotate#<image_id>` in a new tab (a run's split snapshot keeps the project's own image ids, so the id needs no mapping); the annotate page's deep link widens its queue to `file_name` with the filters cleared when the photo is outside the current view, which every labeled photo is under the default "To do" queue — this also repairs the Dataset page's validation-issue and cluster-sample links | Interface scenario (`tests/ui_scenarios/E6-T8.md` §F, `tests/ui_scenarios/E2-annotator.md` A.5) |
| E6-T12 | The jump from an error to its labels is direct and the annotate page loads faster: `Project` caches the parsed `images.json` keyed on the file's (mtime_ns, size) and mutators take a private copy, so one request parses it once instead of dozens of times; the annotate boot fetches `/project`, `/loop` and `/dataset/stats` together and stops blocking on `/progress`; a `#<image_id>` link picks the queue that holds the photo up front, opens the editor from `AnnotationSetView.image` while the queue loads behind it, and never paints a grid nobody sees | `tests/unit/test_project.py`, interface scenario (`tests/ui_scenarios/E2-annotator.md` A.5-6) |
| E6-T13 | Evaluation scores the project's labels as they are now by default (`labels="current"`; `"snapshot"` reproduces an older number). Held-out sets were never trained on, so a corrected box counts on the next evaluation instead of on the next training; photos of the run's own train snapshot are held back in case a reshuffle moved one in, and the report says what the set turned out to be. The ground truth an evaluation used is persisted as `<split>.gt.json`, so error analysis, worst cases, overlays and the threshold sweep re-match the boxes the metrics came from | `tests/api/test_eval_labels.py`, `tests/web/test_eval_routes.py`, interface scenario (`tests/ui_scenarios/E6-T8.md` §G) |
| E6-T14 | Export the evaluation as one 16:9 sheet: the confusion matrix beside the per-class performance table (PNG or PDF), rendered from the same error analysis the page draws at the threshold and IoU asked for. The matrix is one sequential hue shaded by each cell's share of its row with the count printed in every cell, so colour is a scan aid and never the only way to read a number; the diagonal is outlined, not recoloured | `tests/api/test_export_eval_chart.py`, `tests/web/test_export_routes.py`, interface scenario (`tests/ui_scenarios/E6-T8.md` §H) |

#### User stories (added)

- **E6-S8** (WebUI) A user sees which confidence threshold to operate at, why, and how much it beats the 0.50 default by — and applies it with one button
- **E6-S9** (Python API) An engineer asks for the threshold that favours recall over precision and gets it with the sweep it came from
- **E6-S10** (WebUI) A user who decides a worst image is an annotation problem opens that photo in the annotator from where they are, without searching for it by name
- **E6-S11** (WebUI) A user fixes a wrong label in the test set, re-runs the evaluation, and sees the corrected number without retraining
- **E6-S12** (WebUI) A user exports the evaluation as one image and drops it straight into a report or a slide

#### How it is accepted

E6-T3 validates metric values against a fixture with known answers. The rest goes through API tests plus interface scenarios.

E6-T10's sweep is derived from a single matching pass rather than one pass per threshold; its test asserts that every grid point agrees with `analyze_detections` at the same threshold, which is what makes the shortcut safe.

---

### E7 — Experiment Management

**Goal:** answer "which training run was best, and why".

#### User stories

- **E7-S1** (WebUI) A user compares hyperparameters and metrics across several runs side by side
- **E7-S2** (WebUI) A user picks a run straight from the comparison table and enters the export flow
- **E7-S3** (Python API) An engineer queries all runs programmatically, sorted by a metric
- **E7-S4** (WebUI) A user sees which dataset version each run used — after the dataset changes, older runs' metrics are no longer directly comparable and the system must flag that
- **E7-S5** (WebUI) A user adds notes and tags to a run

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E7-T1 | Run metadata schema and storage | `tests/api/test_run_store.py` |
| E7-T2 | Dataset version fingerprint | Content hash, detects dataset changes; `tests/api/test_dataset_fingerprint.py` |
| E7-T3 | Run query and sorting API | `tests/api/test_run_query.py` |
| E7-T4 | Non-comparability warning logic | Flag when dataset fingerprints differ; `tests/api/test_run_comparability.py` |
| E7-T5 | Notes and tags | `tests/api/test_run_tags.py` |
| E7-T6 | Comparison UI | Interface scenario |
| E7-T7 | Web API endpoints | `tests/web/test_experiment_routes.py` |

#### How it is accepted

API tests. **E7-T2, the dataset fingerprint, is the key to this Epic** — without it, comparisons between runs produce misleading conclusions.

---

### E8 — Export and Deployment

**Goal:** get the trained model onto Jetson.

#### User stories

- **E8-S1** (WebUI) A user selects a run, exports to ONNX and downloads it
- **E8-S2** (Python API) An engineer exports a TensorRT engine on Jetson
- **E8-S3** (WebUI) A user exporting sees an explicit notice: a TensorRT engine is bound to the current GPU architecture and TensorRT version and cannot be moved to another machine
- **E8-S4** (Python API) A user exports TFLite
- **E8-S5** (Python API) The export artifact is accompanied by `model_card.json` containing the model license, training dataset fingerprint, metrics, and input/output specification
- **E8-S8** (Python API) The export artifact carries the confidence threshold to run it at, so the deployer does not have to guess or go back to the evaluate page
- **E8-S6** (Python API) A user runs inference directly with the export artifact and verifies the results match the original weights
- **E8-S7** (WebUI) A user starts a local inference service and tests it by posting images over HTTP

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E8-T1 | ONNX export | `tests/api/test_export_onnx.py` |
| E8-T2 | TensorRT export | Availability decided by the E4-T13 capability list; macOS explicitly refused; `tests/api/test_export_tensorrt.py` |
| E8-T3 | TFLite export | `tests/api/test_export_tflite.py` |
| E8-T4 | `model_card.json` generation: license (R3), classes, I/O spec, dataset fingerprint, metrics, parity, and the confidence to run the artifact at — taken from the evaluation's F-score sweep (E6-T10) on test, else valid, with the per-class figures and what it was derived from; a run with no evaluation ships the reason, never an invented number | Includes the license field; `tests/api/test_export_model.py`, `tests/api/test_model_card_threshold.py` |
| E8-T5 | Post-export parity verification | Output difference from the original weights within tolerance on the same input; `tests/api/test_export_parity.py` |
| E8-T6 | Portability warnings | `tests/api/test_export_warnings.py` |
| E8-T7 | Local inference service | `tests/web/test_serve.py` |
| E8-T8 | Export UI | Interface scenario |

#### How it is accepted

**E8-T5 is the core acceptance gate**: an export feature whose results differ before and after export is worthless.

---

### E9 — Three-Layer Interface Parity

**Goal:** ensure that whatever the Python API can do, the Web API and WebUI can do too.

#### User stories

- **E9-S1** (Maintainer) After adding a Python API feature, the contract test immediately points out that the Web API has no corresponding endpoint yet
- **E9-S2** (User) Any operation performed in the WebUI can be reproduced as a script with the Python API
- **E9-S3** (User) The full workflow can be run from the CLI without opening a browser
- **E9-S4** (Maintainer) The Web API has a machine-readable specification document

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E9-T1 | Capability manifest | The Python API's public capabilities are enumerable; `tests/contract/test_capabilities.py` |
| E9-T2 | Contract test framework | Compares capability coverage across the three layers; `tests/contract/test_layer_parity.py` |
| E9-T3 | Flask app skeleton and error handling | Unified error format; `tests/web/test_error_format.py` |
| E9-T4 | OpenAPI specification generation | `tests/web/test_openapi.py` |
| E9-T5 | CLI | `tests/api/test_cli.py` |

#### How it is accepted

Contract tests. Deliberate exceptions are allowed (some features are intentionally not exposed over the Web), but every exception must be explicitly registered in the capability manifest — never left to an omission.

---

### E10 — Active Learning Loop

**Goal:** replace the standalone annotate, autolabel and review pages with one loop that a user can follow without reading any text: select a batch → label it → train → review → next round. Works for any labeled fraction, from zero to complete.

Decisions confirmed on 2026-09-13:

- The **canvas engine** (E2-T1..T8: zoom, draw, polygon, shortcuts, optimistic lock) is kept and embedded; the page shell, autolabel page and review page are rebuilt as the loop page
- Cold-start similarity uses **DINOv2 small** (Apache 2.0, code and weights) as a new `horos/backends/dinov2/` backend; selection is k-center greedy in embedding space
- With a trained model, selection follows **Portable Active Learning (PAL)** — Sharma, Bersamin & Subramanian, CVPR 2026, arXiv 2605.10349 — as the default acquisition metric: per-class logistic classifiers on (raw-candidate support, confidence) give a true-positive probability whose entropy is the instance uncertainty (LIUS); class-weighted image entropy, rare-class diversity and a rank-conditioned similarity penalty over embeddings (GUIDE) refine the ranking with α = 0.9, β = 0.04, γ = 0.02; annotation budget is split by class rarity. Backends therefore report raw `candidates` and, where available, `class_probs` on every prediction. Every picked image carries a score and a reason string; details the paper leaves open are decided and documented in `horos/core/pal.py`
- With no trained model but named classes, **OWLv2 zero-shot** is the round-0 pre-annotator and uncertainty source; the user is never asked to pick a model
- **Only labeled photos belong to a set; the sets grow with the labels** (2026-09-13, two passes on the user's request): a photo arrives in no split and is bucketed by a stable hash into test (20 %), valid (10 %) or train the first time it carries a confirmed annotation; the ratios and seed live in the project manifest (Dataset page), a photo never changes set, test is never trained on and is the learning curve's honest line, valid is what the trainer selects its checkpoint on; unlabeled photos are the loop's pool and are never negatives in a training snapshot
- Batch size per round defaults to a **fixed number**; a percentage of the unlabeled pool is selectable
- **Multiple annotators**: a round's images are assigned per annotator on top of the E2-T8 claims
- Machine-generated geometry (autolabel, SAM boxes-to-polygons, round pre-annotation) is **always** `source="auto", status="pending"` with a score — never stored as human work
- Confirmed 2026-09-13 (second pass): annotators can **skip** a photo as unfit for training and take visually similar photos with it (embedding similarity, user-set threshold); UI text is kept short and plain, Apple-like in tone. The loop was folded into the annotate page on 2026-09-13 and **moved back to its own `/loop` page on 2026-09-14** — the user wants a plain annotate page too

#### User stories

- **E10-S1** (WebUI) A user with an unlabeled folder presses one button, gets a diverse first batch, labels it, presses train, and sees a model
- **E10-S2** (WebUI) A user chooses how many images the next round should contain and sees the remaining unlabeled count update live
- **E10-S3** (WebUI) A user with a partly labeled dataset starts the loop and the existing labels are round 0 — nothing is relabeled
- **E10-S4** (WebUI) In every round after the first model, the user corrects pre-annotations instead of drawing from scratch
- **E10-S5** (WebUI) A user sees, per round, the validation metric and how many labels it took, and decides whether another round is worth it
- **E10-S6** (Python API) An engineer runs the loop from a script: select, label externally, train, repeat
- **E10-S7** (WebUI) Two annotators open the same round and each gets their own share of its images
- **E10-S8** (Python API) A user asks why an image was picked and gets the strategy, score and reason recorded for it

#### Tasks

| ID | Content | Definition of done |
|---|---|---|
| E10-T1 | Round data model and storage (`rounds/<n>/round.json`, state machine selecting → labeling → training → reviewing → closed) | `tests/unit/test_round_model.py` |
| E10-T2 | `ImageEmbedder` interface in `backends/base.py` + DINOv2 backend + registry entry with both licenses | `tests/api/test_backend_dinov2.py` |
| E10-T3 | Project embedding store: per model, incremental, invalidated when an image file changes, progress events (R4) | `tests/api/test_embedding_store.py` |
| E10-T4 | Diversity selection (k-center greedy) with a reason per pick | `tests/unit/test_selection_diversity.py` |
| E10-T5 | PAL acquisition (LIUS + GUIDE, class budgets) with a reason per pick; backends report raw candidates; class balance on top (horos choice, 2026-09-14): budgets weighted by inverse label frequency, and a rare class the model cannot propose yet gets look-alikes of its few photos (≤ ¼ of the round); `LoopSettings.balance` switches it off | `tests/unit/test_selection_uncertainty.py`, `tests/api/test_loop_select.py` |
| E10-T6 | Round selection API: count or percent; strategy auto-chosen from model availability; pool = every unlabeled, unskipped photo (labeled photos are the ones in a set); the scorer runs batched (`infer_many`, boxes only) over a seeded sample of at most `scan_factor` × the round size photos (default 100×, 0 = all) | `tests/api/test_loop_select.py`, `tests/api/test_infer_many.py` |
| E10-T7 | Round pre-annotation: own model when a completed run exists, else OWLv2 from class names; per-class NMS (IoU 0.5) so one object gets one pseudo-label; written pending with score | `tests/api/test_loop_preannotate.py` |
| E10-T8 | Round training: readiness threshold, quick derived config; the round trains on the project's train set and holds out its test (20 %) and valid (10 %) sets, which E1-T8 assigns per photo by stable hash as labels arrive; with the loop's training set to "continue" (default) the run warm-starts from the newest completed run of the same model (`TrainRunConfig.init_from`: weights kept, optimizer fresh, class head resized so classes may change, half the epochs); when only the per-class minimum blocks, `train_round(ignore_short_classes=True)` trains without those classes (`TrainReadiness.short_classes` / `ready_without_short`, UI "Train without N classes", CLI `--ignore-short-classes`) | `tests/api/test_loop_train.py` |
| E10-T9 | Round history: per-round metrics, labels spent, delta to the previous round | `tests/api/test_loop_history.py` |
| E10-T10 | Per-round assignment of images to annotators | `tests/api/test_loop_assign.py` |
| E10-T11 | Machine geometry always pending with score (boxes-to-polygons, autolabel, pre-annotation) | `tests/api/test_generated_pending.py` |
| E10-T12 | Web API endpoints | `tests/web/test_loop_routes.py` |
| E10-T13 | Canvas embeddable without its page shell: `/annotate?embed=1&round=<n>[&annotator=]` hides the site header, swaps the project queue for the round's picks and reports progress to the parent via `postMessage` (the engine itself is not split into a separate file) | Interface scenario (`tests/ui_scenarios/E10-T14.md`) |
| E10-T14 | Loop page: four-step stepper, one primary action at a time, count slider, embedded canvas, live curves, round history | Interface scenario |
| E10-T15 | CLI `horos loop` (status / select / train / close) | `tests/api/test_cli.py` |
| E10-T16 | Skip unfit photos, and similar ones with them: `ImageRecord.excluded`, skipped images leave the pool, statistics, snapshots and training; similar photos found by embedding cosine similarity with a threshold the user adjusts; undo via restore | `tests/api/test_loop_skip.py` |
| E10-T18 | The loop picks the model itself: RF-DETR-Seg Nano when most confirmed labels are polygons, RF-DETR Nano otherwise; reason recorded on the round | `tests/api/test_loop_train.py` |
| E10-T19 | Loop settings chosen once and kept in `loop.json`: training model (auto / any trainable key), suggestions on/off, suggestion shapes auto / box / polygon (SAM refines boxes); Select step controls, API GET/PUT, CLI flags | `tests/api/test_loop_settings.py` |
| E10-T20 | Photo groups on the Dataset page: spherical k-means (`core/clustering.py`) over the DINOv2 embeddings, `k` chosen or automatic; each group shows its closest samples, label and skip counts; "Skip group" / "Restore" go through `images.skip` / `images.restore` so a whole group of unfit photos leaves the pool in one action | `tests/unit/test_clustering.py`, `tests/api/test_image_clusters.py`, interface scenario (`tests/ui_scenarios/E1-T10.md`) |
| E10-T17 | Reversed 2026-09-14 on the user's request: the loop is its own page again (`/loop`, template `loop.html`, four-step shell) and `/annotate` is the plain annotator (`canvas.html`); the Label step no longer embeds a canvas (dropped 2026-09-14 on the user's request) — its "Annotate" button opens `/annotate?round=<n>` and the page polls progress; both have a nav entry; wording stays short and plain with an Apple-like calm look | Interface scenario (`tests/ui_scenarios/E10-T14.md`) |

#### How it is accepted

**E10-T6 is the core acceptance gate**: with fake backends, a project with zero labels yields a diverse batch, a project with a completed run yields an uncertainty-ranked batch, and every pick carries a reason. E10-T14 is accepted through its interface scenario: the whole loop is completed once without reading any help text.

---

## 7. Development Process

### Confirm before starting each Epic

**Before implementing any Epic, present the design options for that Epic to the user and wait for a reply.** The options must be concrete enough to affect code structure — do not ask vague questions like "what do you think".

Decisions already made; no need to ask again:

- **Single environment**: rfdetr and transformers are both primary dependencies installed with horos, with no isolation of any kind
- The licensing boundary is **model size** (XL/2XL are PML 1.0), not install behavior. Install prompts talk about cost, not licensing
- The Jetson install path uses `--no-deps` and checks CUDA availability at import time
- **Lazy backend loading**: `import horos` must not pull in torch; people who only annotate should not pay that cost
- **Four platforms supported**: Ubuntu, macOS, Windows, Jetson. Unsupported combinations error explicitly and never fall back silently
- Auto-labeling uses **OWLv2 open-vocabulary zero-shot** (Apache 2.0)
- First-version priority: annotate → train → evaluate → deploy
- Hyperparameter adaptation is **rule-based**; search-based is left as a later extension
- **Evaluation scores the project's current labels** (decided 2026-09-17): held-out sets are never trained on, so a label correction must count on the next evaluation, not the next training. The run's frozen snapshot stays available as an explicit option
- **Active learning loop (E10)**: canvas engine kept, DINOv2 small for cold-start similarity, PAL (arXiv 2605.10349) as the default uncertainty metric once labels exist, OWLv2 as the round-0 pre-annotator, fixed per-round count by default, multi-annotator assignment required, labeled-only splits assigned per photo by stable hash (70/10/20) that grow with the labels

### Definition of done for a task

A task card is done when three things hold simultaneously:

1. The feature works
2. The corresponding test is written and passing
3. If UI is involved, the interface scenario has been reported in the §8 format

All three are required. "Implement first, add tests later" is not accepted.

### Test location

**All test scripts live under `tests/`**, organized into the subdirectories from §3. Do not put test files next to the source.

### Commit

The commit message format is in §10. The format is fixed; variants are not accepted. Pushing must satisfy R8.

---

## 8. Interface Scenario Report Format

When a completed task involves the WebUI, **tell the user in the following format — do not just say "done"**:

```
[Done] E2-T5 keyboard shortcuts

[How to run]
  horos ui ./demo_project
  Open http://localhost:5000 in a browser

[Test steps]
  1. Go to the "Annotate" page and open any image
  2. Press number keys 1–9 to switch class; confirm the highlight in the
     left-hand class list follows
  3. Hold W and drag the mouse to draw a box; the box should remain on release
  4. Press D for the next image, A for the previous one
  5. Press Ctrl+Z to undo the last action

[Expected result]
  The whole annotate-and-advance loop can be completed without clicking any menu

[Known limitations]
  Polygon shortcuts are not implemented yet (to be added once E2-T3 lands)
```

All four blocks are required. Even when "Known limitations" is empty, write "None" — do not omit it.

Write the report in the language the user is using; only the block labels are fixed.

---

## 9. License Discipline

horos itself is released under **Apache 2.0**. The user has commercial requirements, so licensing is a hard constraint.

**Any new dependency must have its license verified as compatible before being added.** Forbidden list:

| Forbidden | Reason |
|---|---|
| `ultralytics` (YOLOv8/11/12/26) | AGPL-3.0, viral |
| `mmyolo` / `mmdetection` / YOLO-World | GPL-3.0 |
| `rfdetr[plus]` (RF-DETR XL / 2XL) | PML 1.0, not Apache |
| SegFormer official pretrained weights | NVIDIA Source Code License, research use only |

**Code license is not the same as weight license.** Before introducing any pretrained weights, verify the code license and the weight license separately and record both in the model registry.

**RF-DETR size boundary:** Nano / Small / Medium / Large are Apache 2.0 along with the code. The registry lists only those four. If a user attempts to load XL or 2XL, an explicit error must be raised explaining the license difference and requiring `acknowledge_non_apache=True` — never allow it silently.

---

## 10. Commit Conventions

This project will be public on GitHub. The commit message format is fixed as follows, and **variants are not accepted**:

```
[Commit Type] Title for this commit

[Description]
1.

[Verification]
1.
```

### Commit Type

Use only these nine, with the first letter capitalized inside the brackets:

| Type | Use |
|---|---|
| `Feat` | New feature |
| `Fix` | Bug fix |
| `Refactor` | Refactoring, behavior unchanged |
| `Test` | Tests only |
| `Docs` | Documentation only |
| `Chore` | Build, CI, dependency versions, miscellany |
| `Perf` | Performance improvement |
| `Style` | Formatting, naming, no logic change |
| `Revert` | Revert |

### Title

- English, imperative present tense (`Add`, `Fix`, `Remove` — not `Added`, `Fixes`)
- No trailing period, 50 characters or fewer
- **If it corresponds to a task card, put the id in parentheses at the end**: `[Feat] Add COCO format reader (E1-T3)`

Task card ids are this project's traceability mechanism. When someone later asks "what exactly did E1-T3 change", `git log --grep="E1-T3"` has to find it.

### Description

Enumerate **what changed, and why**. "What changed" alone is not enough — three months later, the reasoning is the valuable part.

One commit corresponds to one task card as a rule. If a card needs to be split across several commits, each commit's Description must state where it sits within the card.

### Verification

Enumerate **how someone else confirms this commit is correct**. This section must not be left empty, and must not contain content-free statements like "tested".

Two styles, depending on the kind of change:

**Changes with tests** — give commands that can be copy-pasted and run:

```
[Verification]
1. pytest tests/api/test_format_coco.py -v
2. pytest tests/api/test_format_roundtrip.py -v
```

**Interface changes** — give a summary of the steps in the §8 format, and point at the full scenario file:

```
[Verification]
1. horos ui ./demo_project, open http://localhost:5000
2. On the annotate page, press number keys 1-9 to switch class and confirm
   the left-hand highlight follows
3. Full steps in tests/ui_scenarios/E2-T5.md
```

### Example

```
[Feat] Add lazy backend loading (E4-T11)

[Description]
1. Backend modules are now imported on first use; import horos no longer
   pulls in torch
2. Type annotations moved into TYPE_CHECKING blocks to avoid loading backend
   types at module level
3. For annotation-only usage, import time drops from 8.2s to 0.4s

[Verification]
1. pytest tests/api/test_lazy_backend.py -v
2. pytest tests/test_invariants.py::test_no_torch_after_import -v
3. python -c "import horos, sys; assert 'torch' not in sys.modules"
```

### Rules that cannot be skipped

- **Never credit AI collaborators in a commit message** — no `Co-Authored-By`, no generated-by tool attribution
- All four blocks are required. A commit without a Description or a Verification is rejected
- Never commit weight files, datasets, or `~/.horos/` cache contents. `.gitignore` must cover these
- Pushing must satisfy R8: never rewrite published history, and after pushing confirm that `git fetch && git status` reports the branch up to date with its upstream and the tree clean
