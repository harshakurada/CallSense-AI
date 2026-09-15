"""Pipeline-specific exceptions, one per stage. Raised (not printed) by
src/* modules so the API layer can translate them into proper HTTP errors."""


class CallSenseError(Exception):
    """Base class for all CallSense AI pipeline errors."""


class AudioProcessingError(CallSenseError):
    pass


class ASRError(CallSenseError):
    pass


class DiarizationError(CallSenseError):
    pass


class NLPModelError(CallSenseError):
    pass


class ConfigurationError(CallSenseError):
    pass
