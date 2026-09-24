"""Unified horos error types.

Layers above `horos/backends/` must never need to catch a backend library's
exception classes (E4-S6). Backends translate everything they raise into this
hierarchy via `horos.backends.base.translate_backend_errors`.
"""

from __future__ import annotations


class HorosError(Exception):
    """Base class for every error horos raises deliberately."""

    #: Stable machine-readable code, used by the Web API error format (E9-T3).
    code = "horos_error"

    #: Optional structured payload for the Web API error format (E9-T3),
    #: e.g. conflict file lists — set per-instance by subclasses that need it.
    details: dict | None = None


class ProjectError(HorosError):
    """Project structure on disk is missing, corrupt, or already exists."""

    code = "project_error"


class DatasetFormatError(HorosError):
    """A dataset file could not be parsed as the claimed format."""

    code = "dataset_format_error"


class DatasetValidationError(HorosError):
    """A dataset failed validation and the caller asked for strictness."""

    code = "dataset_validation_error"


class AnnotationConflictError(HorosError):
    """Optimistic-lock conflict: someone else wrote this image's annotations first."""

    code = "annotation_conflict"


class UnknownModelError(HorosError):
    """Requested model key is not in the registry."""

    code = "unknown_model"


class LicenseError(HorosError):
    """Model requires explicit acknowledgement of a non-Apache license (E4-T9)."""

    code = "license_error"


class UnsupportedPlatformError(HorosError):
    """The requested feature is not available on the current platform (E4-T14).

    Never fall back silently — raise this instead.
    """

    code = "unsupported_platform"


class BackendError(HorosError):
    """An error raised inside a model backend, translated to horos (E4-T5).

    The original exception is preserved as ``__cause__``.
    """

    code = "backend_error"

    def __init__(self, message: str, *, backend: str | None = None):
        super().__init__(message)
        self.backend = backend


class BackendOutOfMemoryError(BackendError):
    """Device ran out of memory; training may retry with a smaller batch (E5-T6)."""

    code = "backend_out_of_memory"


class WeightsError(HorosError):
    """Model weights could not be downloaded or the cache is corrupt (E3-T7)."""

    code = "weights_error"


class ImportConflictError(HorosError):
    """Uploaded dataset contains file names that already exist with different
    content, and the caller asked to be consulted (on_conflict="ask")."""

    code = "import_conflict"

    def __init__(self, message: str, *, conflicts: list[str]):
        super().__init__(message)
        self.conflicts = conflicts
        self.details = {"conflicts": conflicts}


class LabelConflictError(HorosError):
    """An import brings labels for photos the project already has labels on,
    and the caller asked to be consulted (on_annotations="ask"). Uploading a
    label file for photos uploaded earlier is the normal way to hit this, so
    the answer belongs to the user, not to a default. Distinct from
    AnnotationConflictError, the optimistic-lock conflict of the annotator."""

    code = "label_conflict"

    def __init__(self, message: str, *, conflicts: list[str]):
        super().__init__(message)
        self.conflicts = conflicts
        self.details = {"conflicts": conflicts}


class CategoryInUseError(ProjectError):
    """Deleting a class that annotations still reference was asked without
    force. The UI turns this into a confirm ("delete the class and its N
    labels?"); any other failure is a real error."""

    code = "category_in_use"

    def __init__(self, message: str, *, annotations: int, images: int):
        super().__init__(message)
        self.details = {"annotations": annotations, "images": images}


class ClassNamesRequiredError(HorosError):
    """A Darknet import has no _darknet.labels and the caller requires explicit
    class names (the WebUI upload path — it shows an editable list instead)."""

    code = "class_names_required"

    def __init__(self, message: str, *, default_names: list[str]):
        super().__init__(message)
        self.default_names = default_names
        self.details = {
            "num_classes": len(default_names),
            "default_names": default_names,
        }
