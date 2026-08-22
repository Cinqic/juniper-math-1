"""Shared exception types for clear, explicit failure reporting."""


class JuniperConfigError(ValueError):
    """Raised when configuration is missing, malformed, or fails validation."""


class JuniperManifestError(ValueError):
    """Raised when a manifest is missing, malformed, or fails validation."""
