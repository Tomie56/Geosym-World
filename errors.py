"""Domain exceptions for the formal geometry world."""


class GeometryWorldError(Exception):
    """Base class for geometry-world failures."""


class DuplicateObjectError(GeometryWorldError):
    """Raised when an object id is registered twice."""


class UnknownObjectError(GeometryWorldError):
    """Raised when a construction references a missing object."""


class DegenerateGeometryError(GeometryWorldError):
    """Raised when a construction would create degenerate geometry."""


class UnsupportedConstructionError(GeometryWorldError):
    """Raised when an operation is intentionally reserved for a later phase."""
