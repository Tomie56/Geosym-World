# GeoSym Formal Geometry Schema

This document is the working schema for `geosym_world`. It describes the executable construction trace, formal state objects, derived fact families, and the intended boundaries of the current prototype.

## Construction Trace

Every construction operation is a JSON object with:

```json
{"op": "operation_name", "id": "optional_output_id", "...": "operation parameters"}
```

The trace is replayed by `ConstructionBuilder` into a `FormalGeometryState`. The verifier performs lightweight static reference checks, then executes the trace.

## Core 2D/Generic Operations

| op | output | purpose |
|---|---|---|
| `add_point` | point | Add a 2D or 3D point from coordinates. |
| `add_segment` | segment | Add a segment from two points. |
| `add_line` | line | Add a line from two points or point + direction. |
| `add_ray` | ray | Add a ray from start + through point or start + direction. |
| `add_circle` | circle | Add a circle from center + radius. |
| `add_arc` | arc | Add a circle arc and an `arc` construction relation. |
| `add_polygon` | polygon | Add a polygon from ordered vertices. |
| `add_region` | region | Add a generic region. |
| `mark_polygon_region` | region | Mark a polygon boundary as a region. |
| `mark_sector_region` | region | Mark a circular sector region. |
| `set_target` | target | Select the formal target fact for export. |

## Derived 2D Construction Operations

| op | output | purpose |
|---|---|---|
| `midpoint` | point | Construct midpoint of two points. |
| `intersection_line_line` | point | Construct 2D line-line intersection. |
| `intersection_line_circle` | point(s) | Construct 2D line-circle intersections. |
| `intersection_circle_circle` | point(s) | Construct 2D circle-circle intersections. |
| `tangent_points_from_point` | point(s) | Construct tangent points from an external point to a circle. |
| `add_chord` | segment | Add a segment and a `chord` relation. |
| `parallel_through` | line | Construct line through point parallel to a line-like object. |
| `perpendicular_through` | line | Construct 2D perpendicular line through a point. |
| `line_through_angle` | line | Construct a line by rotating a base direction. |
| `rotate_ray` | ray | Rotate an existing ray. |
| `angle_bisector` | line | Construct internal or external angle bisector. |
| `foot_of_perpendicular` | point | Project a 2D point to a 2D line. |
| `point_on_ray_distance` | point | Place a point along a ray or direction at a given distance. |
| `point_on_segment_ratio` | point | Place a point at a segment ratio. |

## 3D Interface Operations

| op | output | purpose |
|---|---|---|
| `add_point3d` | point | Add a 3D point. |
| `add_plane` | plane | Add a plane from three points or point + normal. |
| `add_polyhedron` | polyhedron | Add a polyhedron from vertices and faces. |
| `project_point_to_plane` | point | Orthogonally project a 3D point to a plane. |
| `project_line_to_plane` | line | Orthogonally project a 3D line-like object to a plane. |
| `dihedral_angle` | relation | Declare a dihedral angle from planes or polyhedron face indices. |
| `section_polyhedron_with_plane` | relation, optional polygon | Declare or auto-generate a plane section of a polyhedron. |

`section_polyhedron_with_plane` has two modes:

- Explicit boundary mode: pass `boundary`.
- Automatic mode: omit `boundary`; the builder intersects polyhedron edges with the plane, creates section points, sorts them on the section plane, and materializes `{section_id}_polygon` when at least three points are found.

## Formal State Objects

| object | key fields |
|---|---|
| point | `id`, `coords`, `dim`, `metadata` |
| line | `id`, `point`, `direction`, `source_points`, `metadata` |
| segment | `id`, `start`, `end`, `metadata` |
| ray | `id`, `start`, `direction`, `through`, `metadata` |
| circle | `id`, `center`, `radius`, `metadata` |
| arc | `id`, `circle`, `start`, `end`, `central_angle`, `orientation`, `metadata` |
| polygon | `id`, `vertices`, `metadata` |
| region | `id`, `boundary`, `kind`, `metadata` |
| plane | `id`, `point`, `normal`, `source_points`, `metadata` |
| polyhedron | `id`, `vertices`, `faces`, `metadata` |
| target | `type`, `object`, `metadata` |

## Fact Schema

Each fact has:

| field | meaning |
|---|---|
| `id` | Stable local fact id within one solve pass. |
| `type` | Fact type, such as `length`, `area`, `space_angle`. |
| `object` | Object or relation identifier. |
| `value` | Exact symbolic value when available. |
| `numeric_value` | Float approximation when parseable. |
| `source` | `constructed` or `derived`. |
| `rule` | Rule or formula that produced the fact. |
| `dependencies` | Input object/fact references. |
| `metadata` | Structured auxiliary details. |

## Core Metric Facts

| family | fact types |
|---|---|
| Object existence | `exists` |
| Point/segment metrics | `distance`, `length` |
| Circle metrics | `radius`, `circumference`, `area` |
| Arc/sector metrics | `arc_length`, `sector_area` |
| Polygon/region metrics | `perimeter`, `area` |
| Polyhedron metrics | `surface_area`, `volume` |
| Target lookup | `target`, `target_unresolved` |

## Topology and Relation Facts

| family | fact types |
|---|---|
| Line-like topology | `angle`, `parallel`, `perpendicular`, `line_intersection` |
| Point incidence | `point_on_line`, `point_on_segment`, `point_on_circle`, `collinear` |
| Plane topology | `point_on_plane`, `line_parallel_plane`, `line_on_plane`, `line_perpendicular_plane`, `line_plane_normal_dot`, `plane_parallel`, `plane_perpendicular` |
| 3D line-plane measures | `line_plane_angle`, `line_plane_distance` |
| Unified 3D angles | `space_angle` with metadata `kind` |

## Theorem-Lite Fact Families

These facts are coordinate-backed theorem checks. They are not a full synthetic proof system.

| family | fact types |
|---|---|
| Right triangle | `right_triangle`, `pythagorean_relation` |
| Isosceles triangle | `isosceles_triangle`, `equal_base_angles`, `equal_base_angle_residual` |
| Similar triangle | `similar_triangle`, `similarity_ratio_residual`, `similarity_scale_squared`, `equal_corresponding_angles`, `similar_angle_residual`, `aa_similarity`, `aa_similarity_residual`, `parallel_cut_similarity`, `parallel_cut_ratio_residual` |
| Angle bisector | `angle_bisector`, `equal_angle`, `angle_bisector_residual` |
| Circle chord | `chord_of_circle`, `central_angle`, `chord_length_formula_check`, `equal_chord_equal_central_angle` |
| Inscribed angle | `inscribed_angle`, `inscribed_angle_central_residual`, `arc_subtends_inscribed_angle`, `inscribed_angle_arc_residual`, `equal_inscribed_angles` |
| Tangent-chord | `tangent_to_circle`, `tangent_chord_angle`, `alternate_segment_angle`, `tangent_chord_angle_residual` |

## 3D Fact Families

| family | fact types |
|---|---|
| Projection | `point_projection_plane`, `projection_distance`, `projection_plane_residual`, `line_projection_plane`, `line_projection_plane_residual` |
| Dihedral angle | `dihedral_angle`, `dihedral_shared_edge` |
| Solid section | `section_interface`, `section_polygon`, `section_area`, `section_perimeter`, `section_degeneracy_measure`, `section_boundary_size`, `section_boundary_plane_residual` |
| Space angle | `space_angle` with `kind=line_line`, `line_plane`, or `plane_plane` |

## Current Boundaries

- The system is exact-coordinate-backed, using SymPy expressions.
- The theorem-lite layer derives verifiable facts but does not attempt a full proof graph or natural-language proof.
- Automatic polyhedron-plane section currently intersects edges and sorts section points on the plane; it does not handle every degenerate coplanar-edge case.
- 3D operations are intentionally interface-first, with small pilot coverage rather than large data generation.
- Pilot export is the main integration smoke path; `tools/smoke_check.py` provides an in-memory assertion layer for selected invariants.
