"""Domain exceptions for the formal geometry world."""


class GeometryWorldError(Exception):
    """Base class for geometry-world failures."""


class InvalidGeometryInput(GeometryWorldError, ValueError):
    """Raised when a schema or operation payload is invalid."""


class DuplicateObjectError(GeometryWorldError):
    """Raised when an object id is registered twice."""


class UnknownObjectError(GeometryWorldError):
    """Raised when a construction references a missing object."""


class DegenerateGeometryError(GeometryWorldError):
    """Raised when a construction would create degenerate geometry."""


class NumericalInstabilityError(GeometryWorldError):
    """Raised when a value cannot be represented as stable real geometry."""


class VerificationError(GeometryWorldError):
    """Raised when a requested admission or verification cannot be established."""


class UnsupportedConstructionError(GeometryWorldError):
    """Raised when an operation is intentionally reserved for a later phase."""
