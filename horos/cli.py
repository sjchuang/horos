"""horos CLI (E9-T5): the full workflow without a browser (E9-S3).

This is an interface layer like horos.web — it may print (it IS the output
device) but all logic lives in horos.api (R2).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

import horos
import horos.api as api
from horos.core.streams import use_utf8_streams
from horos.errors import HorosError, ProjectError

MANIFEST_NAME = "horos.json"


def find_project_root(start: Path | str | None = None) -> Path | None:
    """The nearest horos project at or above `start` (default: the cwd).

    Lets every project command be run from inside the project — the same way
    git works — instead of repeating --project on each invocation."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    return None


def _project_arg(args, attribute: str = "project"):
    """Open the project named by the flag, else the one containing the cwd."""
    explicit = getattr(args, attribute, None)
    if explicit:
        return api.open_project(explicit)
    root = find_project_root()
    if root is None:
        raise ProjectError(
            f"No horos project here: run 'horos {args.command}' from inside a "
            f"project directory (one containing {MANIFEST_NAME}), or pass "
            f"--project <dir>. 'horos init' creates one."
        )
    return api.open_project(root)


def _resolve_run(project, run_id: str | None, *, need_checkpoint: bool = True) -> str:
    """`--run` defaults to the newest usable run of the project.

    With `need_checkpoint` (infer, evaluate, export-model) that means the
    newest completed run that actually has weights; report accepts any run."""
    if run_id:
        return run_id
    runs = api.list_runs(project)
    if not runs:
        raise ProjectError(
            f"Project '{project.manifest.name}' has no training runs yet — "
            f"run 'horos train' first."
        )
    usable = [
        r for r in runs
        if not need_checkpoint or (r.state == "completed" and r.checkpoint)
    ]
    if not usable:
        states = ", ".join(sorted({r.state for r in runs}))
        raise ProjectError(
            f"No completed training run with a checkpoint in project "
            f"'{project.manifest.name}' (runs are: {states}). Pass --run <id> "
            f"to pick one explicitly."
        )
    chosen = usable[0]  # list_runs is newest first
    print(f"using run {chosen.run_id} ({chosen.model})", file=sys.stderr)  # noqa: T201
    return chosen.run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="horos",
        description="horos: annotate, train, evaluate, deploy perception models.",
        epilog=(
            "Project commands find the project by walking up from the current "
            "directory, so --project is optional once you are inside one:\n"
            "  mkdir beds && cd beds\n"
            "  horos init beds              # empty directory -> the project IS this directory\n"
            "  horos import ~/data.zip      # no --project needed\n"
            "  horos train --epochs 25\n"
            "  horos models                 # trained models of this project\n"
            "  horos infer photo.jpg        # --run defaults to the newest completed run\n"
            "  horos ui                     # opens this project\n"
            "  horos catalog                # architectures horos can train, with licenses"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=horos.__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "init",
        help="Create a horos project (in the current directory when it is empty)",
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Project name, or a path. A bare name in an empty directory turns "
        "THAT directory into the project; otherwise a subdirectory of this name "
        "is created. Omit it to use the current directory.",
    )
    p.add_argument("--name", help="Project name (default: the directory's name)")

    p = sub.add_parser(
        "import",
        help="Import a COCO / YOLO / VOC / Darknet / VIA / LabelMe dataset "
        "(directory or .zip) into a project, or plain photos with no labels",
    )
    p.add_argument("source")
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--format",
        choices=["coco", "yolo", "voc", "darknet", "via", "labelme", "images"],
        help="Skip detection and read the source as this format ('images': photos only)",
    )
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="Reference images in place instead of copying them into the project",
    )
    p.add_argument(
        "--on-conflict",
        choices=["ask", "overwrite", "skip", "rename"],
        default="ask",
        help="What to do when a file name already exists with different content "
        "(default: ask — fail with the conflict list, importing nothing)",
    )
    p.add_argument(
        "--on-annotations",
        choices=["ask", "replace", "merge", "skip"],
        default="ask",
        help="What to do when the import brings labels for a photo that already "
        "has some (default: ask — fail with the list, importing nothing)",
    )
    p.add_argument(
        "--class-names",
        help="Comma-separated class names for Darknet datasets without "
        "_darknet.labels, or VIA datasets without class attributes",
    )

    p = sub.add_parser(
        "export", help="Export the project's dataset as COCO, YOLO, or LabelMe"
    )
    p.add_argument("out_dir")
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument("--format", choices=["coco", "yolo", "labelme"], default="coco")

    p = sub.add_parser("convert", help="Convert a dataset between formats")
    p.add_argument("source")
    p.add_argument("out_dir")
    p.add_argument("--to", required=True, choices=["coco", "yolo", "labelme"], dest="to_format")
    p.add_argument("--from", choices=["coco", "yolo", "voc", "darknet", "via", "labelme"],
                   dest="from_format")

    p = sub.add_parser(
        "validate", help="Validate the project dataset; exit 1 when it has errors"
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Repair auto-fixable boxes: clamp small annotation-tool overshoots back "
        "into the image and refit a drifted bbox to its polygons, then re-validate",
    )

    p = sub.add_parser(
        "stats", help="Show dataset statistics (classes, splits, object sizes)"
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )

    p = sub.add_parser(
        "clear", help="Delete every image and annotation of the project (runs are kept)"
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument("--yes", action="store_true", help="Required: confirm the deletion")
    p.add_argument(
        "--drop-classes", action="store_true", help="Also delete the class list"
    )
    p = sub.add_parser(
        "split",
        help="Set the train/valid/test ratios and give labeled photos without a split "
             "their set (only labeled photos belong to a set)",
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument("--train", type=float, default=None,
                   help="share of labeled photos (default 0.7)")
    p.add_argument("--valid", type=float, default=None,
                   help="share of labeled photos (default 0.1)")
    p.add_argument("--test", type=float, default=None,
                   help="share of labeled photos (default 0.2)")
    p.add_argument("--seed", type=int, default=None, help="hash seed for the assignment")
    p.add_argument(
        "--reshuffle", action="store_true",
        help="re-draw EVERY labeled photo at random — photos past models trained on may "
             "land in test, so their learning curve is no longer clean",
    )

    p = sub.add_parser(
        "autolabel", help="Zero-shot pre-labels from text prompts (runs in foreground)"
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--prompt",
        action="append",
        required=True,
        dest="prompts",
        metavar="CLASS=P1[,P2...]",
        help="Class and its prompt(s), repeatable: --prompt forklift=forklift,lift truck",
    )
    p.add_argument("--model", default="owlv2-base")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument(
        "--output",
        choices=["bbox", "polygon"],
        default="bbox",
        help="polygon refines each kept box into a mask outline with SAM "
        "(sam-base, downloaded on first use)",
    )
    p.add_argument("--split", choices=["train", "valid", "test"])
    p.add_argument(
        "--include-annotated",
        action="store_true",
        help="Also pre-label images that already have confirmed annotations",
    )

    p = sub.add_parser(
        "boxes-to-polygons",
        help="Rewrite box annotations as SAM polygons — each box is the prompt (foreground)",
    )
    p.add_argument("--project", help="Project directory (default: the enclosing project)")
    p.add_argument(
        "--class", action="append", dest="classes", metavar="NAME",
        help="Only this class (repeatable); default: every class",
    )
    p.add_argument("--model", default="sam2.1-tiny", help="Segmenter (default sam2.1-tiny)")
    p.add_argument("--split", choices=["train", "valid", "test"])
    p.add_argument(
        "--skip-pending", action="store_true",
        help="Leave pending pre-labels as boxes; only confirmed boxes are rewritten",
    )

    p = sub.add_parser(
        "train", help="Train a model (runs in a worker subprocess, streams events)"
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument("--model", default="rfdetr-nano")
    p.add_argument("--epochs", type=int, help="Omit to derive from dataset stats")
    p.add_argument("--batch-size", type=int, help="Omit to derive from memory probe")
    p.add_argument("--resolution", type=int)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int)
    p.add_argument("--resume-from", help="Checkpoint path to continue training from")
    p.add_argument(
        "--classes",
        help="Comma-separated category names to train on (default: all); "
        "objects of unselected classes become background",
    )
    p.add_argument(
        "--include-background",
        action="store_true",
        help="With --classes: keep images that contain none of the selected "
        "classes as background negatives (default: drop them)",
    )

    p = sub.add_parser(
        "report",
        help="Render a run's training report (16:9 PNG dashboard, PDF, or Excel), or "
        "with --evaluation the evaluation sheet: confusion matrix + per-class table",
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--run",
        dest="run_id",
        help="Training run id (default: the newest completed run of this project)",
    )
    p.add_argument("--format", choices=["png", "pdf", "xlsx"], default="png")
    p.add_argument("--out", help="Output file (default: <run>/exports/training_report.<format>)")
    p.add_argument(
        "--evaluation", action="store_true",
        help="Render the evaluation sheet instead: the confusion matrix beside the "
        "per-class performance table (needs a prior 'evaluate'; png or pdf)",
    )
    p.add_argument("--split", choices=["train", "valid", "test"], default="test",
                   help="Split for --evaluation (default test)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Operating confidence for --evaluation (default 0.5)")
    p.add_argument("--iou", type=float, default=0.5,
                   help="Matching IoU for --evaluation (default 0.5)")

    p = sub.add_parser(
        "export-model",
        help="Export a completed run's model with its model card (streams events)",
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--run",
        dest="run_id",
        help="Training run id (default: the newest completed run of this project)",
    )
    p.add_argument(
        "--format", choices=["pytorch", "onnx", "tensorrt", "tflite"], default="onnx"
    )
    p.add_argument("--dynamic-batch", action="store_true", help="ONNX: dynamic batch axis")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")

    p = sub.add_parser(
        "infer", help="Detect objects in image(s) with a trained run's model"
    )
    p.add_argument("images", nargs="+")
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--run",
        dest="run_id",
        help="Training run id (default: the newest completed run of this project)",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--overlay-dir", metavar="DIR",
        help="Also write each image with its predictions drawn on it into DIR",
    )

    p = sub.add_parser(
        "evaluate", help="COCO metrics for a run on its held-out split"
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--run",
        dest="run_id",
        help="Training run id (default: the newest completed run of this project)",
    )
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument(
        "--labels", choices=["current", "snapshot"], default="current",
        help="Ground truth to score against: the project's labels as they are "
        "now (default), or the export this run trained with — use 'snapshot' "
        "to reproduce an older number",
    )

    p = sub.add_parser(
        "analyze",
        help="Error analysis of an evaluated run: suggested confidence threshold, "
        "confusion matrix, per-class misses and false positives, worst images "
        "(needs a prior 'evaluate')",
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--run",
        dest="run_id",
        help="Training run id (default: the newest completed run of this project)",
    )
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Operating confidence threshold (default 0.5)",
    )
    p.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU needed for a prediction to match a ground-truth box (default 0.5)",
    )
    p.add_argument(
        "--beta", type=float, default=1.0,
        help="F-score weight for the suggested threshold: 1 balances precision and "
        "recall, 2 weights recall, 0.5 weights precision (default 1)",
    )
    p.add_argument(
        "--worst", type=int, default=20, metavar="N",
        help="How many worst images to list (default 20; 0 for none)",
    )
    p.add_argument(
        "--overlays", metavar="DIR",
        help="Also write a colour-coded overlay PNG of each listed worst image into DIR",
    )

    p = sub.add_parser(
        "models",
        help="List this project's trained models (completed runs, newest first)",
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Include runs that are still queued, running, stopped, or failed",
    )
    p = sub.add_parser(
        "serve",
        help="Serve a trained model over HTTP (POST /predict) from an export bundle or "
        "checkpoint — the command that runs on the deployment machine (E8)",
    )
    p.add_argument(
        "source", nargs="?",
        help="Export bundle directory or zip, model_card.json, a bare .onnx / .trt / .tflite "
        "artifact, or a checkpoint; omit to serve --run from the project",
    )
    p.add_argument("--run", metavar="RUN_ID",
                   help="Serve this run of the project (default: the newest completed run)")
    p.add_argument(
        "--format", default="onnx", choices=("onnx", "tensorrt", "tflite", "pytorch", "checkpoint"),
        help="With --run: which export bundle to serve (onnx / tensorrt engine / tflite / "
        "pytorch weights), or the raw checkpoint (default onnx)",
    )
    p.add_argument("--project", help="Project directory (default: the enclosing project)")
    p.add_argument("--model", help="Model key when serving a bare checkpoint file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Default confidence threshold (per-request 'threshold' overrides)")
    p.add_argument("--device", help="cuda | cpu (default: auto — an engine needs cuda, "
                   "TFLite runs on cpu; recorded in /health)")

    p = sub.add_parser(
        "runs",
        help="List training runs with their scores, sorted by any metric (E7)",
    )
    p.add_argument("--project", help="Project directory (default: the enclosing project)")
    p.add_argument(
        "--sort", default="created_at", metavar="KEY",
        help="Sort key: a record field (created_at, model, epochs_completed, ...), a "
        "score key (map50, loss, ...) or eval.<split>.<metric>; an unknown key lists "
        "the valid ones",
    )
    p.add_argument("--asc", action="store_true", help="Ascending order (default: descending)")
    p.add_argument(
        "--state", action="append", default=[], help="Keep only runs in this state (repeatable)"
    )
    p.add_argument(
        "--tag", action="append", default=[], help="Keep only runs carrying this tag (repeatable)"
    )
    p.add_argument(
        "--reference", default="project", metavar="RUN_ID|project|none",
        help="Judge each run's comparability against this run, the project's current "
        "data (default), or skip it",
    )
    p = sub.add_parser(
        "compare", help="Compare runs side by side: hyperparameters, metrics, dataset (E7)"
    )
    p.add_argument("run_ids", nargs="+", metavar="RUN_ID", help="Runs to compare (2-8)")
    p.add_argument("--project", help="Project directory (default: the enclosing project)")
    p.add_argument(
        "--all", action="store_true", help="Print every row, not only the ones that differ"
    )
    p = sub.add_parser(
        "tag", help="Set a training run's notes or edit its tags (E7)"
    )
    p.add_argument("run_id", help="The run to annotate (see 'horos models --all')")
    p.add_argument("--project", help="Project directory (default: the enclosing project)")
    p.add_argument("--notes", help="Replace the run's free-text notes")
    p.add_argument(
        "--add", action="append", default=[], metavar="TAG", help="Add a tag (repeatable)"
    )
    p.add_argument(
        "--remove", action="append", default=[], metavar="TAG",
        help="Remove a tag (repeatable)",
    )
    p.add_argument(
        "--set", metavar="TAG[,TAG...]", help="Replace the whole tag list (comma-separated)"
    )
    sub.add_parser(
        "catalog", help="List the model architectures horos can train or run, with licenses"
    )
    sub.add_parser("capabilities", help="Show what this platform supports")

    p = sub.add_parser(
        "install",
        help="Install the ML stack (torch, rfdetr, albumentations, transformers, "
        "plus the ONNX export and report libraries) matched to this machine",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force the CPU-only torch build even if a GPU is present",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned pip commands without running them",
    )
    p.add_argument(
        "--tflite",
        action="store_true",
        help="Also install the TFLite conversion toolchain (onnx2tf + tensorflow, ~600 MB, "
        "Apache 2.0 / MIT; needed for TFLite export)",
    )
    p.add_argument(
        "--tensorrt",
        action="store_true",
        help="Also install NVIDIA's TensorRT wheels for this GPU (NVIDIA license; "
        "needed for TensorRT engine export)",
    )

    p = sub.add_parser(
        "doctor", help="Check dependencies for this platform; --fix installs what's missing"
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Run the planned pip installs (torch on Jetson is never automated)",
    )

    p = sub.add_parser(
        "loop",
        help="Active-learning loop: show where it stands, or select the next round",
    )
    p.add_argument(
        "action", nargs="?", choices=("status", "select", "train", "close"), default="status",
        help="'status' (default) prints pool, labels and rounds; 'select' opens the "
             "next round and picks its images; 'train' trains the open round on every "
             "labeled image; 'close' closes the open round",
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument("--count", type=int, help="Images for the round (default 20)")
    p.add_argument("--percent", type=float, help="Round size as a percentage of the pool")
    p.add_argument(
        "--strategy", choices=("auto", "pal", "diversity", "random"), default="auto",
        help="auto (default): PAL once labels exist, diversity before",
    )
    p.add_argument("--device", help="Device override (cuda, mps, cpu)")
    p.add_argument(
        "--no-suggestions", action="store_true",
        help="loop select: do not pre-label the picked photos this time",
    )
    p.add_argument(
        "--shapes", choices=("auto", "box", "polygon"),
        help="loop select: suggestion geometry for this round (default: the loop settings)",
    )
    p.add_argument(
        "--no-balance", action="store_true",
        help="loop select: do not tilt this round towards under-labeled classes",
    )
    p.add_argument(
        "--scan", type=int, default=None,
        help="loop select: score this many times the round size (random sample of the pool; "
             "0 = all; default: the loop settings, 100)",
    )
    p.add_argument(
        "--model", help="loop train: model key (default: RF-DETR Nano, or RF-DETR-Seg Nano "
                        "when the labels are mostly polygons)"
    )
    p.add_argument("--epochs", type=int, help="loop train: epochs (default: derived)")
    p.add_argument(
        "--fresh", action="store_true",
        help="loop train: start from the published weights instead of continuing from the "
             "previous run of the same model (default: the loop settings, continue)",
    )
    p.add_argument(
        "--ignore-short-classes", action="store_true",
        help="loop train: leave out the classes with too few labels instead of waiting "
             "for them",
    )

    p = sub.add_parser("ui", help="Start the Web API + WebUI server")
    p.add_argument(
        "project_path",
        nargs="?",
        default=None,
        metavar="project",
        help="Project directory (default: the project containing the current directory)",
    )
    # kept for compatibility with older docs/scripts: horos ui --project <dir>
    p.add_argument("--project", dest="project_flag", help=argparse.SUPPRESS)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)

    return parser


def _emit(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))  # noqa: T201


#: commands that cannot run without the ML stack `horos install` provides
_ML_GATED_COMMANDS = frozenset(
    {"autolabel", "boxes-to-polygons", "train", "infer", "evaluate", "export-model"}
)


def _ml_preflight(command: str) -> int | None:
    """Fail fast (with the fix) when an ML command lacks its dependencies.

    `pip install horos` ships without torch/rfdetr/transformers on purpose;
    this is the moment the gap becomes the user's problem, so this is where
    the answer must be. `ui` only warns — dataset management and annotation
    work without the ML stack.
    """
    from horos.api.install import check_ml_ready

    readiness = check_ml_ready()
    for message in readiness.warnings:
        print(f"warning: {message}", file=sys.stderr)  # noqa: T201
    if not readiness.missing:
        return None
    names = ", ".join(readiness.missing)
    if command == "ui":
        print(  # noqa: T201
            f"warning: ML dependencies are not installed ({names}) — "
            "autolabel, training and inference will be unavailable. "
            "Run 'horos install' to add them.",
            file=sys.stderr,
        )
        return None
    print(  # noqa: T201
        f"error [ml-not-installed]: 'horos {command}' needs the ML stack, "
        f"but these packages are missing: {names}.\n"
        "Run 'horos install' — it detects your platform and GPU and installs "
        "the matching builds ('horos install --cpu' forces CPU-only).",
        file=sys.stderr,
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    # `horos dataset stats > out.json` and every other redirected command
    # would otherwise encode with the locale code page, and class names or a
    # progress table are not always in it (R7)
    use_utf8_streams()
    args = build_parser().parse_args(argv)
    gated = args.command in _ML_GATED_COMMANDS or args.command == "ui"
    if args.command == "loop" and args.action in ("select", "train"):
        gated = True  # embedding / scoring / training models; status and close work without
    if gated:
        exit_code = _ml_preflight(args.command)
        if exit_code is not None:
            return exit_code
    try:
        if args.command == "init":
            cwd = Path.cwd()
            target, name = args.target, args.name
            looks_like_path = target is not None and (
                "/" in target or "\\" in target or Path(target).is_absolute()
                or target in (".", "..")
            )
            from horos.core.project import occupied_by

            if target is None:
                root = cwd  # no argument: this directory becomes the project
            elif looks_like_path:
                root = Path(target)
            elif not occupied_by(cwd):
                # a bare name in an empty directory: no pointless nesting
                root, name = cwd, name or target
            else:
                root, name = cwd / target, name or target
            existing = find_project_root(root) if root.is_dir() else None
            if existing is not None and existing != root.resolve():
                print(  # noqa: T201
                    f"note: this is inside the existing project at {existing}",
                    file=sys.stderr,
                )
            project = api.create_project(root, name=name or Path(root).resolve().name)
            print(  # noqa: T201
                f"created project '{project.manifest.name}' in {project.root.resolve()}",
                file=sys.stderr,
            )
            _emit({"root": str(project.root), "name": project.manifest.name})
        elif args.command == "import":
            last_phase = [""]

            def report(event) -> None:
                # one stderr line per phase change plus the final tick of each
                if event.type != "progress":
                    return
                done = event.total is not None and event.current == event.total
                if event.phase != last_phase[0] or done:
                    last_phase[0] = event.phase
                    count = f" {event.current}/{event.total}" if event.total else ""
                    note = f" ({event.message})" if event.message else ""
                    print(f"{event.phase}{count}{note}", file=sys.stderr)  # noqa: T201

            project = _project_arg(args)
            names = (
                [n.strip() for n in args.class_names.split(",")] if args.class_names else None
            )
            if zipfile.is_zipfile(args.source):
                summary = api.import_zip(
                    project,
                    args.source,
                    on_conflict=args.on_conflict,
                    on_annotations=args.on_annotations,
                    class_names=names,
                    progress=report,
                )
            else:
                summary = api.import_dataset(
                    project,
                    args.source,
                    format=args.format,
                    copy_images=not args.no_copy,
                    on_conflict=args.on_conflict,
                    on_annotations=args.on_annotations,
                    class_names=names,
                    progress=report,
                )
            _emit(summary.model_dump())
        elif args.command == "export":
            written = api.export_dataset(
                _project_arg(args), args.out_dir, format=args.format
            )
            _emit({"path": str(written)})
        elif args.command == "convert":
            written = api.convert_dataset(
                args.source, args.out_dir,
                to_format=args.to_format, from_format=args.from_format,
            )
            _emit({"path": str(written)})
        elif args.command == "validate":
            project = _project_arg(args)
            if args.fix:
                result = api.fix_validation_issues(project)
                _emit(result.model_dump() | {"ok": result.report.ok})
                return 0 if result.report.ok else 1
            report = api.validate_project(project)
            _emit(report.model_dump() | {"ok": report.ok})
            return 0 if report.ok else 1
        elif args.command == "stats":
            _emit(api.dataset_stats(_project_arg(args)).model_dump())
        elif args.command == "clear":
            project = _project_arg(args)
            if not args.yes:
                raise ProjectError(
                    f"This deletes every image and annotation of project "
                    f"{project.manifest.name!r} (training runs are kept). Re-run with --yes."
                )
            _emit(api.clear_dataset(
                project, confirm=project.manifest.name, keep_categories=not args.drop_classes,
            ).model_dump())
        elif args.command == "split":
            counts = api.resplit(
                _project_arg(args),
                train=args.train, valid=args.valid, test=args.test, seed=args.seed,
                reshuffle=args.reshuffle,
            )
            _emit(counts)
        elif args.command == "autolabel":
            from horos.api.autolabel import autolabel_events
            from horos.backends.base import dump_event

            prompts: dict[str, list[str]] = {}
            for entry in args.prompts:
                cls, _, plist = entry.partition("=")
                prompts[cls.strip()] = (
                    [p.strip() for p in plist.split(",")] if plist else [cls.strip()]
                )
            failed = False
            for event in autolabel_events(
                _project_arg(args),
                api.PromptSpec(prompts=prompts),
                model=args.model,
                threshold=args.threshold,
                nms_iou=args.nms_iou,
                output=args.output,
                split=args.split,
                only_unannotated=not args.include_annotated,
            ):
                sys.stdout.write(dump_event(event) + "\n")  # JSONL stream (E3-T3)
                sys.stdout.flush()
                failed = failed or event.type == "failed"
            if failed:
                return 2
        elif args.command == "boxes-to-polygons":
            from horos.api.segment import boxes_to_polygons_events
            from horos.backends.base import dump_event

            failed = False
            for event in boxes_to_polygons_events(
                _project_arg(args),
                categories=args.classes or None,
                split=args.split,
                include_pending=not args.skip_pending,
                model=args.model,
            ):
                sys.stdout.write(dump_event(event) + "\n")  # JSONL stream, like autolabel
                sys.stdout.flush()
                failed = failed or event.type == "failed"
            if failed:
                return 2
        elif args.command == "train":
            import time as time_mod

            from horos.api.train import TrainRunConfig

            project = _project_arg(args)
            record = api.start_training(
                project,
                TrainRunConfig(
                    model=args.model,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    resolution=args.resolution,
                    device=args.device,
                    seed=args.seed,
                    resume_from=args.resume_from,
                    categories=(
                        [c.strip() for c in args.classes.split(",")]
                        if args.classes
                        else None
                    ),
                    include_background=args.include_background,
                ),
            )
            print(f"run {record.run_id} started (pid {record.pid})", file=sys.stderr)  # noqa: T201
            seen = 0
            try:
                while True:
                    status = api.training_status(project, record.run_id, after=seen)
                    for event in status.events:
                        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
                        sys.stdout.flush()
                    seen = status.num_events
                    if status.run.state not in ("pending", "running"):
                        _emit(status.run.model_dump())
                        # the conclusion, checked even when numbers look perfect
                        _emit(api.run_verdict(project, record.run_id).model_dump())
                        return 0 if status.run.state == "completed" else 2
                    time_mod.sleep(1.0)
            except KeyboardInterrupt:
                api.stop_training(project, record.run_id)
                print(f"stopping run {record.run_id} ...", file=sys.stderr)  # noqa: T201
                return 130
        elif args.command == "report":
            project = _project_arg(args)
            # a report is readable for any run, finished or not
            run_id = _resolve_run(project, args.run_id, need_checkpoint=False)
            if args.evaluation:
                path = api.export_evaluation_chart(
                    project, run_id, args.split, threshold=args.threshold,
                    iou=args.iou, format=args.format, out_path=args.out,
                )
            else:
                path = api.export_training_report(
                    project, run_id, format=args.format, out_path=args.out,
                )
            _emit({"path": str(path), "format": args.format})
        elif args.command == "export-model":
            from horos.api.export import model_export_events

            project = _project_arg(args)
            failed = False
            for event in model_export_events(
                project, _resolve_run(project, args.run_id), format=args.format,
                options={"dynamic_batch": args.dynamic_batch, "opset": args.opset},
            ):
                sys.stdout.write(event.model_dump_json() + "\n")
                sys.stdout.flush()
                failed = failed or event.type == "failed"
            if failed:
                return 2
        elif args.command == "infer":
            project = _project_arg(args)
            run_id = _resolve_run(project, args.run_id)
            for image in args.images:
                prediction = api.infer_image(
                    project, run_id, image, threshold=args.threshold
                )
                if args.overlay_dir:
                    api.render_prediction_overlay(
                        image, prediction,
                        out=Path(args.overlay_dir) / f"{Path(image).stem}.overlay.png",
                    )
                sys.stdout.write(prediction.model_dump_json() + "\n")
                sys.stdout.flush()
        elif args.command == "evaluate":
            from horos.api.evaluate import evaluation_events
            from horos.backends.base import dump_event

            project = _project_arg(args)
            failed = False
            for event in evaluation_events(
                project, _resolve_run(project, args.run_id),
                split=args.split, labels=args.labels,
            ):
                sys.stdout.write(dump_event(event) + "\n")  # JSONL stream (R4)
                sys.stdout.flush()
                failed = failed or event.type == "failed"
            if failed:
                return 2
        elif args.command == "analyze":
            project = _project_arg(args)
            run_id = _resolve_run(project, args.run_id)
            advice = api.suggest_threshold(
                project, run_id, args.split, iou=args.iou, beta=args.beta
            )
            payload = {
                "analysis": api.analyze_errors(
                    project, run_id, args.split, threshold=args.threshold, iou=args.iou
                ).model_dump(mode="json"),
                # where to operate, next to how the chosen threshold behaves
                "threshold": advice.model_dump(mode="json"),
            }
            if args.worst > 0:
                worst = api.worst_cases(
                    project, run_id, args.split,
                    threshold=args.threshold, iou=args.iou, top_k=args.worst,
                )
                payload["worst"] = worst.model_dump(mode="json")
                if args.overlays:
                    out_dir = Path(args.overlays)
                    written = []
                    for image in worst.images:
                        target = out_dir / f"{Path(image.file_name).stem}.overlay.png"
                        api.render_error_overlay(
                            project, run_id, args.split, image.image_id,
                            threshold=args.threshold, iou=args.iou, out=target,
                        )
                        written.append(str(target))
                    payload["overlays"] = written
            sys.stdout.write(json.dumps(payload) + "\n")
        elif args.command == "models":
            from horos.api.report import _series_from_events, run_scores
            from horos.api.train import _read_events, _run_dir

            project = _project_arg(args)
            runs = api.list_runs(project)
            if not args.all:
                runs = [r for r in runs if r.state == "completed" and r.checkpoint]
            default_id = next(
                (r.run_id for r in runs if r.state == "completed" and r.checkpoint), None
            )
            rows = []
            for run in runs:
                events, _ = _read_events(_run_dir(project, run.run_id))
                best_epoch, scores = run_scores(_series_from_events(events))
                rows.append({
                    "run_id": run.run_id,
                    "model": run.model,
                    "state": run.state,
                    "created_at": run.created_at,
                    "epochs_completed": run.epochs_completed,
                    "classes": run.dataset_classes,
                    "dataset_images": run.dataset_images,
                    "best_epoch": None if best_epoch is None else best_epoch + 1,
                    "scores": scores,
                    "checkpoint": run.checkpoint,
                    # the run `horos infer` / `export-model` use when --run is omitted
                    "default": run.run_id == default_id,
                })
            if not rows:
                print(  # noqa: T201
                    "no completed training runs yet — 'horos train' creates one"
                    + ("" if args.all else "; --all also lists unfinished runs"),
                    file=sys.stderr,
                )
            _emit(rows)
        elif args.command == "serve":
            from horos.web.serve_app import create_serve_app

            if args.source:
                source = api.resolve_source(path=args.source, model=args.model)
            else:
                project = _project_arg(args)
                run_id = args.run
                if not run_id:
                    completed = [
                        r for r in api.list_runs(project)
                        if r.state == "completed" and r.checkpoint
                    ]
                    if not completed:
                        raise ProjectError(
                            "No completed training run to serve — pass a bundle path, or "
                            "train first."
                        )
                    run_id = completed[0].run_id
                source = api.resolve_source(project, run_id=run_id, format=args.format)
            server = api.create_inference_server(
                source, device=args.device, threshold=args.threshold
            )
            print(  # noqa: T201 — the CLI is the output device
                f"serving {source.kind} {Path(source.path).name} "
                f"({source.model or 'unknown model'}, {len(source.classes)} classes, "
                f"device {server.device or 'auto'}) on http://{args.host}:{args.port} — "
                f"POST /predict, GET /health, GET /model_card",
                file=sys.stderr,
            )
            create_serve_app(server).run(host=args.host, port=args.port, threaded=True)
        elif args.command == "runs":
            result = api.query_runs(
                _project_arg(args),
                sort_by=args.sort,
                descending=not args.asc,
                states=args.state or None,
                tags=args.tag or None,
                reference=None if args.reference == "none" else args.reference,
            )
            _emit([
                {
                    "run_id": s.run.run_id,
                    "model": s.run.model,
                    "state": s.run.state,
                    "created_at": s.run.created_at,
                    "epochs_completed": s.run.epochs_completed,
                    "best_epoch": s.best_epoch,
                    "scores": s.scores,
                    "evals": s.evals,
                    "tags": s.tags,
                    "notes": s.notes,
                    "fingerprint": s.fingerprint.digest if s.fingerprint else None,
                    "comparable": None if s.comparability is None else s.comparability.comparable,
                    "comparability": None if s.comparability is None else s.comparability.reason,
                }
                for s in result.runs
            ])
        elif args.command == "compare":
            comparison = api.compare_runs(_project_arg(args), args.run_ids)

            def _table(rows):
                return [
                    row.model_dump() for row in rows if args.all or row.differs
                ]

            _emit({
                "runs": [
                    {
                        "run_id": s.run.run_id,
                        "model": s.run.model,
                        "state": s.run.state,
                        "tags": s.tags,
                        "comparable": None if s.comparability is None
                        else s.comparability.comparable,
                        "comparability": None if s.comparability is None
                        else s.comparability.reason,
                    }
                    for s in comparison.runs
                ],
                "hparams": _table(comparison.hparams),
                "metrics": _table(comparison.metrics),
                "dataset": _table(comparison.dataset),
            })
        elif args.command == "tag":
            summary = api.update_run_notes(
                _project_arg(args),
                args.run_id,
                notes=args.notes,
                tags=None if args.set is None else args.set.split(","),
                add_tags=args.add or None,
                remove_tags=args.remove or None,
            )
            _emit({
                "run_id": summary.run.run_id,
                "notes": summary.notes,
                "tags": summary.tags,
            })
        elif args.command == "catalog":
            _emit([m.model_dump() for m in api.list_models()])
        elif args.command == "capabilities":
            _emit(api.platform_capabilities().model_dump())
        elif args.command == "install":
            import subprocess

            from horos.api.install import plan_install

            plan = plan_install(cpu=args.cpu, tensorrt=args.tensorrt, tflite=args.tflite)
            plat = plan.platform
            print(f"platform : {plat.os_family}/{plat.arch}"  # noqa: T201
                  f"{' (Jetson)' if plat.is_jetson else ''}  python {plat.python_version}")
            print(f"cuda     : driver supports {plan.cuda_version}"  # noqa: T201
                  if plan.cuda_version else "cuda     : no NVIDIA GPU detected")
            for note in plan.notes:
                print(f"note     : {note}")  # noqa: T201
            if plan.empty:
                print("ML stack already installed — nothing to do.")  # noqa: T201
                return 0
            for command in plan.pip_commands:
                print(f"plan     : pip install {' '.join(command)}")  # noqa: T201
            for action in plan.manual_actions:
                print(f"manual   : {action}")  # noqa: T201
            if args.dry_run:
                return 0
            for command in plan.pip_commands:
                print(f"==> pip install {' '.join(command)}")  # noqa: T201
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *command], check=True
                )
            if plan.manual_actions:
                print("Manual steps remain (see above) — not automated on purpose.")  # noqa: T201
                return 1
            print("ML stack installed. Run 'horos doctor' to verify.")  # noqa: T201
        elif args.command == "doctor":
            import subprocess

            report = api.doctor_report()
            plat = report.platform
            print(f"platform : {plat.os_family}/{plat.arch}"  # noqa: T201
                  f"{' (Jetson)' if plat.is_jetson else ''}  python {plat.python_version}")
            for dep in report.dependencies:
                # BAD = installed but wrong (e.g. a CPU torch on a GPU machine)
                mark = "ok " if dep.ok else ("BAD" if dep.installed else "MISSING")
                extra = f"  ({dep.note})" if dep.note else ""
                print(f"  [{mark}] {dep.name:<12} {dep.installed or '-':<10} "  # noqa: T201
                      f"requires {dep.required}{extra}")
            if report.torch_cuda_available is not None:
                print(f"device   : cuda={report.torch_cuda_available} "  # noqa: T201
                      f"mps={report.torch_mps_available}")
            for action in report.manual_actions:
                print(f"manual   : {action}")  # noqa: T201
            if report.ok:
                print("Environment OK.")  # noqa: T201
                return 0
            if not args.fix:
                for command in report.fix_commands:
                    print(f"fix      : pip install {' '.join(command)}")  # noqa: T201
                print("Run 'horos install' (or 'horos doctor --fix') to install the above.")  # noqa: T201
                return 1
            for command in report.fix_commands:
                print(f"==> pip install {' '.join(command)}")  # noqa: T201
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *command], check=True
                )
            if report.manual_actions:
                print("Manual steps remain (see above) — not automated on purpose.")  # noqa: T201
                return 1
            print("Fixes applied. Re-run 'horos doctor' to verify.")  # noqa: T201
        elif args.command == "loop":
            project = _project_arg(args)
            if args.action == "select":
                from horos.api.loop import select_round_events
                from horos.backends.base import dump_event

                final = None
                for event in select_round_events(
                    project, count=args.count, percent=args.percent,
                    strategy=args.strategy, device=args.device,
                    preannotate=False if args.no_suggestions else None, shapes=args.shapes,
                    scan_factor=args.scan, balance=False if args.no_balance else None,
                ):
                    print(dump_event(event), file=sys.stderr)  # noqa: T201
                    final = event
                if final is None or final.type != "completed" or final.result.get("cancelled"):
                    return 1
                record = api.get_round(project, int(final.result["round"]))
                _emit(record.model_dump(mode="json"))
            elif args.action in ("train", "close"):
                status = api.loop_status(project)
                if status.current is None:
                    raise ProjectError("No open round — run 'horos loop select' first")
                if args.action == "train":
                    record = api.train_round(
                        project, status.current.number, model=args.model, epochs=args.epochs,
                        device=args.device, warm_start=False if args.fresh else None,
                        ignore_short_classes=args.ignore_short_classes,
                    )
                    print(  # noqa: T201
                        f"training run {record.train_run_id} started for round {record.number}; "
                        f"follow it with 'horos loop' or the Loop page",
                        file=sys.stderr,
                    )
                else:
                    record = api.close_round(project, status.current.number)
                _emit(record.model_dump(mode="json"))
            else:
                _emit(api.loop_status(project).model_dump(mode="json"))
        elif args.command == "ui":
            from horos.web.app import create_app

            project_path = args.project_path or args.project_flag or find_project_root()
            if not project_path:
                print(  # noqa: T201
                    "usage: horos ui [project] — run it inside a project directory, "
                    "or name one. 'horos init' creates a project.",
                    file=sys.stderr,
                )
                return 2
            app = create_app(project_path)
            app.run(host=args.host, port=args.port)
    except HorosError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
