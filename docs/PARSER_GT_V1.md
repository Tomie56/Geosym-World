# Parser Ground-Truth Guideline v1

Contract version: `parser-gt-v1`.

This guideline defines trusted human ParseGraph references for clean 2D geometry diagrams. It is intentionally narrower than open-world geometry understanding. A record is trusted only after primary annotation, independent review, and adjudication pass the contracts in `geosym_world.benchmark.parser_annotation`.

## 1. Annotation Boundary

Annotators may inspect only the public `ParserInput` image and public question text. They must not inspect synthetic scene metadata, formal state, construction trace, derivation, answer, source sidecars, model output, or another annotator's graph before completing their independent graph.

Trusted v1 annotation is unassisted human work:

- `model_assistance` is `false` at every stage.
- Human-observed candidates use `origin=image`, `text`, or `image_text`, `confidence=1.0`, and `score_kind=human_annotation`.
- Candidates deterministically compiled from accepted human candidates use `origin=compiled`, `confidence=1.0`, and `score_kind=deterministic_rule`.
- `ParseGraph.metadata.reference_kind` is `human_annotation`.

Confidence is not used to hide uncertainty in trusted GT. Genuine ambiguity is represented with explicit candidates and an unresolved conflict during draft/review. An adjudicated evaluator reference cannot retain unresolved conflicts.

## 2. Evidence Before Semantics

Every human-observed candidate must reference evidence from the same source.

| Evidence kind | Required payload | Use |
| --- | --- | --- |
| `pixel_point` | `point` | Point center, intersection, or marker anchor |
| `image_region` | `bbox` or localized geometry | Text, marker, curve, shaded component |
| `text_span` | `span` and `raw_content` | Public question wording |
| `source_document` | source-level location | Page or problem context when licensed and public |

Use `localized` when evidence only identifies a region. Use `bound` only when the evidence is explicitly assigned to one semantic object or relation. Do not mark a whole image as bound evidence for a local label.

## 3. Objects

Annotate visible mathematical objects, not detector fragments.

### Points

- Create a point for every visible named point marker.
- Create an anonymous stable point for every crossing, tangent contact, endpoint, or construction point needed by topology, even when no label is printed.
- Do not assign a visible name to an anonymous point.
- Two labels at the same geometric location do not create two points unless the source explicitly distinguishes them.
- A line crossing hidden by a label or marker is annotated only when continuity on both sides or the question provides sufficient evidence. Otherwise record an ambiguity.

Real raster annotation stores observed `pixel` coordinates. It does not invent `exact_coords`. Exact coordinates are allowed only for a verified native structured source and remain realization geometry, not automatically public conditions.

### Linear Objects

- `segment`: finite support with two endpoints.
- `ray`: one endpoint and a forward direction visibly or textually established.
- `line`: support intended to continue in both directions.
- A dashed object is not automatically an extension. Use `visible_extension` only when it extends a specific admitted support.
- Merge collinear raster fragments that represent one mathematical support. Keep distinct overlapping supports when the source semantics distinguishes them.

Use object references:

- segment: `start`, `end`;
- ray: `start`, `through`;
- line from two known points: `point_0`, `point_1`;
- exact line awaiting a direction construction: one `point` reference.

### Circles And Arcs

- A full circumference is a `circle`; a bounded curved part is an `arc`.
- A circle references one `center` only when the center is visible or stated.
- An arc references `circle`, `start`, and `end`; orientation is `cw` or `ccw` only when it can be resolved.
- Do not infer a center from visual symmetry alone as a source assertion.

### Polygons, Text, And Markers

- A polygon references ordered `vertex_0`, `vertex_1`, and subsequent vertices.
- Text remains a `text` object even after binding.
- Right-angle boxes, parallel arrows, equal-length ticks, angle marks, and arrowheads remain `marker` evidence and bind to explicit relations.
- Decorative arrows, dimension arrows, and ray arrows must not be conflated with parallel markers.

Each separately drawn glyph is one marker object with bound `image_region` evidence and the tight pixel bbox of its visible stroke. Canonical `metadata.marker_kind` values are `right_angle`, `parallel`, `equal_length_tick`, and `angle_arc`. Marker references identify the local support whenever it is observable: right-angle boxes reference their two supports or grounded vertex/line, parallel arrows reference the support carrying that glyph, equal-length ticks reference the bounded segment interval containing the glyph, and angle arcs reference their ordered arm/vertex points. A tick on a merged line may reference a subsegment only when already observed topology points delimit that interval; annotators must not invent a midpoint from the tick location. Repeated glyphs that jointly express one relation remain separate marker objects; the relation may depend on all of them. A marker object records visual evidence and does not by itself admit the associated semantic relation.

## 4. Labels And Binding

Point names are represented by a text object plus a `binds` relation with `metadata.binding_kind=point_name`. A visible name may also be projected onto the point for evaluator convenience, but the text object and binding remain independently scorable.

- Bind a label only when one object is clearly intended.
- If two objects are plausible, retain both binding candidates in one unresolved conflict.
- The same visible point name cannot bind to multiple points in one accepted graph.
- Multiple point-name labels cannot bind to one point unless the source visibly repeats the same name.
- Numeric text is not a point name.

## 5. Topology And Relations

Relations describe source semantics. They are not a place to store every coordinate fact that happens to hold in the drawing.

Required v1 role signatures for executable relations are:

| Predicate | Arguments | Additional fields |
| --- | --- | --- |
| `intersection` | two leading `support`; then one `point` or contiguous `point_0...` | optional `selection`; `contact=crossing` or `tangent` |
| `ordered_points` | one `support`, ordered `point_0...` | finite-support order only |
| `midpoint` | `point`, `of_0`, `of_1` | explicit midpoint evidence or text |
| `foot_of_perpendicular` | `point`, `from`, `line` | perpendicular-foot evidence |
| `point_on_segment_ratio` | `point`, `segment` | typed `metadata.ratio`, strictly between 0 and 1 |
| `point_on_ray_distance` | `point`, `ray` | typed positive `metadata.distance` |
| `parallel` / `perpendicular` | `object_0`, `object_1` | visible marker or explicit text evidence |
| `angle_bisector` | `line`, `arm_0`, `vertex`, `arm_1` | `metadata.mode=internal` or `external` |
| `angle_from_line` | `line`, `point`, `base_line` | typed numeric `metadata.angle` in radians |
| `point_on_line` | `point`, `line` | use `support` only for format adapters |
| `point_on_circle` | `point`, `circle` | source-supported incidence |
| `point_on_arc` | `point`, `arc` | source-supported bounded incidence |

Parallelism, perpendicularity, equal length, equal angle, similarity, congruence, tangency, cyclicity, and bisectors require a marker or explicit public text. Visual appearance alone may produce a Parser hypothesis, but not trusted source GT.

All crossings that affect topology must be represented. A dashed auxiliary line that crosses another support must keep its intersection point even when the point is unnamed or visually suppressed.

## 6. Quantities

Quantities bind a typed safe `MathExpression` to explicit object references.

- Preserve the visible or textual raw value in `raw_text`.
- Normalize integers, rational values, roots, powers, and `pi` into the safe expression AST.
- Never evaluate arbitrary source strings.
- Record a unit only when printed or stated.
- A nearby number is not a quantity until its object binding is resolved.
- Radius and diameter bind to the circle object; length binds to a segment; area and perimeter bind to a region or polygon.
- If one label can bind to multiple objects, retain competing quantity candidates in a conflict.

Pixel measurements are not source quantities. They may be used by the numeric-witness pipeline and must remain realization facts.

## 7. Regions And Shading

Annotate region semantics separately from fill appearance.

- A simple region lists an oriented boundary of existing point, segment, arc, circle, or polygon objects.
- A hole is an explicit referenced region, not merely a white patch.
- A disconnected shaded answer has multiple components.
- Use typed `union`, `intersection`, and `difference` expressions over explicit region references.
- Touching a boundary, crossing a boundary, and strict containment are distinct cases.
- Do not close an occluded boundary unless visible continuation or public text supports it.
- Shading color and opacity are rendering attributes; the selected mathematical set is the GT.
- Parser-side `fill_status=observed` records only pixel appearance. An `interior_gap` may be a true hole, text occlusion, marker, or rendering artifact and must not become a GT hole without boundary grounding.

## 8. Targets

The target comes from the public question, not from a private answer.

- `mode=value`: asks for a quantity.
- `mode=proof`: asks to establish a supported relation.
- `mode=constraint`: compares a target quantity with an explicit typed expression and operator.
- Resolve the target object only when names and topology identify it uniquely.
- If wording permits multiple target objects, retain an unresolved target conflict in draft/review.

Do not add givens solely because they make the target solvable. An unsolved but faithful graph is preferable to a solvable fabricated graph.

## 9. Construction And Exactness

The annotation describes observed objects and stated semantics. It does not claim to recover the source author's original construction order.

The reconstruction compiler may choose a replayable exact construction when the graph supports one. Current deterministic inverse constructions include intersections, midpoint, perpendicular foot, segment ratio, ray distance, parallel/perpendicular through a point, relative-angle line, angle bisector, and a circle radius derived from a consistent radius/diameter quantity.

Annotators must not select an exact circle-intersection branch from pixel proximity alone. If the public source distinguishes the branch, record `selection`; otherwise retain branch ambiguity.

## 10. Independent Review

The reviewer creates `independent_graph` before seeing the primary graph. After both graphs are frozen:

1. Compare canonical graphs.
2. Use `agree` only when checksums are identical.
3. Otherwise use `disagree` and enumerate every material difference.
4. Assign one category: missing candidate, extra candidate, type, geometry, binding, topology, quantity, region, target, evidence, or other.
5. Use a stable candidate path such as `objects/point_a/geometry/pixel`.
6. Do not resolve differences inside free-form notes.

## 11. Adjudication

Every disagreement receives exactly one typed resolution and rationale. The final graph is rebuilt as one canonical graph; it is not a patch log. `accepted` requires a final graph and no unresolved conflicts. `unresolved` cannot materialize evaluator truth.

The adjudicator checks:

- public input and source manifest checksums match all stages;
- source ID and guideline version match;
- actor independence policy holds;
- every candidate has valid evidence, provenance fields, and reference roles;
- all visible topology-changing intersections are represented;
- label, quantity, region, and target bindings are unique or explicitly resolved;
- no exact value was inferred only from pixel scale;
- no private answer, scene, trace, or model output entered the graph.

## 12. Versioning

`parser-gt-v1` is the only guideline accepted by the current annotation contracts. Any semantic change to inclusion rules, role signatures, evidence policy, or trusted-assistance policy requires a new guideline version, explicit migration, updated tests, and a benchmark manifest change. Existing references retain their original version and are never silently reinterpreted.
