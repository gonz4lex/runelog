__version__ = "0.1.0"

from .runelog import RuneLog
from .exceptions import (
    RunelogException,
    ExperimentNotFound,
    RunNotFound,
    ModelNotFound,
    ModelVersionNotFound,
    ArtifactNotFound,
    NoActiveRun,
    InvalidExperimentId,
    InvalidRunId,
    RegistryError,
    DuplicateModelRegistration,
    InvalidModelVersion,
    CorruptedMetadata,
    InsufficientPermissions,
    StorageError,
    NoVersionsFound,
)


def get_tracker(path: str = ".") -> RuneLog:
    """
    Initializes and returns the main tracker instance.

    This is the primary entry point for interacting with the Runelog library.
    The returned object is used to create experiments, start runs, and log
    all relevant machine learning data.

    Args:
        path (str, optional): The root directory for storing experiments.
            Defaults to the current directory.

    Returns:
        RuneLog: An instance of the main tracker class, ready to be used.

    Example:
        >>> from runelog import get_tracker
        >>>
        >>> tracker = get_tracker()
        >>> with tracker.start_run(experiment_name="my-test"):
        ...     tracker.log_parameter("learning_rate", 0.01)
        ...     tracker.log_metric("accuracy", 0.95)


    """
    return RuneLog(path=path)


__all__ = ["get_tracker", "RuneLog", "exceptions", "__version__"]
