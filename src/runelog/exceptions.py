"""
Custom exception classes for the Runelog ML experiment tracking system.

These exceptions provide more specific error handling and better debugging
information compared to generic Python exceptions.
"""


class RunelogException(Exception):
    """Base exception class for all runelog-specific errors."""

    pass


class ExperimentNotFound(RunelogException):
    """Raised when an experiment with the specified ID or name is not found."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        super().__init__(f"Experiment with ID '{experiment_id}' not found.")


class RunNotFound(RunelogException):
    """Raised when a run with the specified ID is not found."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"Run with ID '{run_id}' not found.")


class ModelNotFound(RunelogException):
    """Raised when a model with the specified name is not found in the registry."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' not found in registry.")


class ModelVersionNotFound(RunelogException):
    """Raised when a specific version of a model is not found."""

    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        super().__init__(f"Model '{model_name}' version '{version}' not found.")


class ArtifactNotFound(RunelogException):
    """Raised when an artifact file is not found."""

    def __init__(self, artifact_path: str, run_id: str = None):
        self.artifact_path = artifact_path
        self.run_id = run_id
        run_context = f" in run '{run_id}'" if run_id else ""
        super().__init__(f"Artifact '{artifact_path}' not found{run_context}.")


class NoActiveRun(RunelogException):
    """Raised when trying to log parameters, metrics, or artifacts without an active run."""

    def __init__(self):
        super().__init__("No active run. Use start_run() context manager.")


class InvalidExperimentId(RunelogException):
    """Raised when an experiment ID has an invalid format."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        super().__init__(f"Invalid experiment ID format: '{experiment_id}'.")


class InvalidRunId(RunelogException):
    """Raised when a run ID has an invalid format."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"Invalid run ID format: '{run_id}'.")


class RegistryError(RunelogException):
    """Raised for general model registry operation errors."""

    pass


class DuplicateModelRegistration(RegistryError):
    """Raised when trying to register a model that already exists with the same version."""

    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        super().__init__(f"Model '{model_name}' version '{version}' already exists.")


class InvalidModelVersion(RegistryError):
    """Raised when a model version has an invalid format."""

    def __init__(self, version: str):
        self.version = version
        super().__init__(f"Invalid model version format: '{version}'.")


class CorruptedMetadata(RunelogException):
    """Raised when metadata files are corrupted or unreadable."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        super().__init__(f"Corrupted or unreadable metadata file: {file_path}")


class InsufficientPermissions(RunelogException):
    """Raised when file system permissions prevent required operations."""

    def __init__(self, path: str, operation: str):
        self.path = path
        self.operation = operation
        super().__init__(f"Insufficient permissions to {operation} at: {path}")


class StorageError(RunelogException):
    """Raised when file system operations fail."""

    def __init__(self, operation: str, path: str, original_error: Exception = None):
        self.operation = operation
        self.path = path
        self.original_error = original_error
        error_details = f": {original_error}" if original_error else ""
        super().__init__(f"Storage error during {operation} at {path}{error_details}")


class NoVersionsFound(ModelNotFound):
    """Raised when no versions are found for a registered model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"No versions found for model '{model_name}'.")
