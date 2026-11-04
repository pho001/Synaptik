# Failures and [ownership](../../glossary.md#term-ownership-meanings) summary

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Find the stage that can know the answer

When application or layer code calls a Tensor method, that method can reject a wrong rank or
type because descriptors cross the call boundary. It cannot reject an out-of-range stored index
without reading values, which expression construction deliberately does not do.
Later Compiler checks can reason about the captured descriptions; concrete binding or execution
must discharge remaining obligations. A successful constructor is therefore not proof of a runnable
backend route. Use [indexing examples](gather.md) for descriptor-versus-value checks and the
[production support matrix](../../backend-guide/production-support.md) for executable domains.

**How to read this summary.** Local [validation](../../glossary.md#term-validation) checks [descriptors](../../glossary.md#term-descriptor) and request [parameters](../../glossary.md#term-parameter) when
an expression is constructed; it cannot check input values it never reads.
A **[deferred obligation](../../glossary.md#term-deferred-obligation)** is a condition left for the stage that knows concrete sizes or values,
not permission to ignore that condition. **Checked [overflow](../../glossary.md#term-overflow)** throws when an integer size
calculation cannot be represented, unlike deliberate fixed-width wrapping in numeric operations.
**Ownership** asks who must keep memory alive and who may [release](../../glossary.md#term-release) it; a [borrowed](../../glossary.md#ownership-borrowing-and-lease) association
does not transfer that responsibility. For a shorter first explanation before the detailed
failure inventory, use [shapes](shapes.md)<span data-glossary-navigation="true"> ([shapes glossary](../../glossary.md#shape))</span>, [layouts](layouts.md)<span data-glossary-navigation="true"> ([layouts glossary](../../glossary.md#layout))</span>, and
[host-storage lifetimes](host-storage.md)<span data-glossary-navigation="true"> ([host glossary](../../glossary.md#term-host))</span>.

<a id="failures-and-ownership-summary"></a>
## Failures and [ownership](../../glossary.md#term-ownership-meanings) summary

- Null inputs are rejected where documented by each current method's [Javadoc](../../glossary.md#term-javadoc).
- Negative sizes, offsets, [strides](../../glossary.md#term-stride), and identifiers are rejected.
- Checked size, stride, and span arithmetic throws `ArithmeticException` on [overflow](../../glossary.md#term-overflow).
- `Tensor.contiguous()` accepts every data type, preserves exact Shape, type, and gradient
  eligibility, and performs checked canonical layout construction only for fully static Shapes.
  Overflow consumes no Tensor identity; identifier exhaustion occurs after local immutable
  expression metadata construction. Valid results remain unlabeled and storage-free.<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient); [canonical glossary](../../glossary.md#term-canonical-form); [layout glossary](../../glossary.md#layout); [metadata glossary](../../glossary.md#term-metadata))</span>
- `Tensor.reshape(long...)` accepts non-negative extents plus at most one inferable `-1`, treats an
  empty request as scalar Shape, copies request semantics without retaining the caller array, and
  rejects unavailable, ambiguous, non-divisible, overflowing, or known-mismatched requests before
  identity allocation.<span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar))</span>
- `Tensor.reshape(Shape)` retains the exact non-null normalized target reference. Both overloads
  defer equality when either count is dynamic and resolve same-offset canonical view geometry only
  for resolved contiguous input plus a fully static target; every other layout remains unresolved.
  Valid results retain exact type and gradient eligibility, use exact RESHAPE/target-shape
  semantics and <span data-glossary-navigation="true"> ([view glossary](../../glossary.md#term-view-meanings); [contiguous glossary](../../glossary.md#term-contiguous))</span>`[input]` [provenance](../../glossary.md#provenance), and remain fresh, unlabeled, and storage-free.
- `Tensor.expand(long...)` accepts only literal non-negative extents, treats an empty request as
  scalar Shape, and neither retains nor mutates the caller array. `Tensor.expand(Shape)` retains
  the exact non-null target reference. Both require target rank at least input rank. Equal pairs,
  a static source singleton, and leading target axes are immediately valid; a fully static
  contradiction fails, while a pair containing an unresolved dimension retains the later
  source-one-or-source-equal requirement without binding or a Model-created constraint.<span data-glossary-navigation="true"> ([singleton glossary](../../glossary.md#term-singleton); [dimension glossary](../../glossary.md#dimension); [binding glossary](../../glossary.md#term-binding); [Model glossary](../../glossary.md#term-model))</span>
- A static expand target and resolved input layout produce a fresh same-offset view layout that
  preserves unchanged aligned strides and uses zero strides for leading and expanded-singleton
  axes. Binding-dependent, dynamic, or otherwise unresolved geometry remains unresolved. Every
  result retains exact type and gradient eligibility, records exact EXPAND/target-shape semantics
  and `[input]` provenance, and remains fresh, unlabeled, and storage-free.
- `Tensor.permute(int...)` requires exactly one raw axis per input rank, copies the request after
  count validation, normalizes each negative entry once, and rejects an out-of-range or duplicate
  normalized axis before identity allocation. Empty axes are valid exactly for scalar input.<span data-glossary-navigation="true"> ([validation glossary](../../glossary.md#term-validation))</span>
- Every resolved permute input layout produces one new same-offset view layout with exact reordered
  strides; unresolved input remains unresolved. `Tensor.transpose()` requires rank two and uses
  normalized `[1, 0]`. Both methods retain exact type and eligibility, record PERMUTE attributes
  and `[input]` provenance, and remain fresh, unlabeled, storage-free logical metadata without an
  alias, copy, materialization, or execution guarantee.<span data-glossary-navigation="true"> ([alias glossary](../../glossary.md#aliasing-and-arena); [materialization glossary](../../glossary.md#materialization))</span>
- `Tensor.expandDims(int)` normalizes an insertion position against rank plus one, inserts exactly
  one new static singleton while preserving exact unaffected Dimension references, and records
  the normalized output position. For resolved input geometry it inserts one checked following-
  axis stride, or one at the end, in a new same-offset view descriptor; unresolved input remains
  unresolved.
- `Tensor.squeeze(int)` uses existing Shape-axis normalization, requires the selected Dimension to
  be statically known as one, and removes exactly that Dimension and stride while preserving exact
  unaffected references and offset. Dynamic, zero, and non-one dimensions are rejected without a
  symbolic constraint. Both rank-edit operations preserve exact type/eligibility, record matching<span data-glossary-navigation="true"> ([normalization glossary](../../glossary.md#term-normalization))</span>
  `AxisTransformAttrs` and `[input]` provenance, and remain fresh, unlabeled, storage-free logical
  metadata without an alias, copy, canonicalization, gradient, materialization, or execution
  guarantee.<span data-glossary-navigation="true"> ([canonicalization glossary](../../glossary.md#occurrence-identity-and-canonical-form))</span>
- `Tensor.slice(long[], long[], int[], long[])` requires four non-null equal-length arrays and
  clones them before element inspection. It normalizes each axis and negative bound once, clamps
  bounds according to step direction, rejects repeated normalized axes and zero steps, and
  preserves rank plus exact unselected Dimension references. Empty arrays and zero-extent results
  are valid; normalized attributes store start plus selected length, not an end sentinel.<span data-glossary-navigation="true"> ([element glossary](../../glossary.md#term-element))</span>
- A non-empty all-positive slice of resolved dense, offset, strided, or broadcast geometry derives
  a new checked start-adjusted offset and step-multiplied original strides in a view-marked
  descriptor. Unresolved input, empty results, and every request containing a negative step remain
  unresolved. Every success preserves exact type and eligibility, records normalized `SliceAttrs`
  and `[input]` provenance at output index zero, and is fresh, unlabeled, and storage-free.
  `Tensor.sliceAxis` has step-one and explicit-step forms; `Tensor.flip` creates one negative-step
  SLICE occurrence for its explicit axes, with empty axes meaning identity. None promises a
  physical alias, copy, canonicalization, gradient, materialization, or execution.
- `Tensor.sliceByLength(long[], long[], int[], long[])` requires four non-null equal-length arrays,
  clones them in parameter order, normalizes distinct axes once, and accepts only non-negative
  starts/lengths plus signed non-zero steps. Every non-empty sequence proves a representable
  non-negative final coordinate; static selected extents prove the upper bound, while an
  unresolved selected extent defers only that comparison. Zero length performs no bound proof and
  stores canonical start zero. Selected result Dimensions are new static lengths and unaffected
  references remain exact. Layout follows the same resolved-input/non-empty/all-positive rule as
  raw slicing. Every success is a fresh <span data-glossary-navigation="true"> ([parameter glossary](../../glossary.md#term-parameter); [coordinate glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>[`SLICE`](../../glossary.md#slice)/`SliceAttrs`/`[input]` occurrence at output index
  zero without a Model-owned deferred constraint, attached storage, or execution promise.
- `Tensor.sliceUpdate(update, starts, axes, steps)` requires non-null equal-length parallel arrays,
  exact base/update type and rank, distinct normalized axes, signed non-zero steps, static selected
  update Dimensions, and an exact complete update Shape. It normalizes one negative start only
  against a static base extent, rejects statically out-of-bounds first/final coordinates, and
  defers only an unresolved base axis's upper bound. Empty arrays are full-Shape replacement and a
  zero selected update extent is canonical empty placement. Every success retains exact base
  Shape/type, combines eligibility, leaves layout unresolved, and records `SLICE_UPDATE`,
  `SliceAttrs`, and ordered `[base, update]` provenance without mutation, addition, or value work.
- `Tensor.sliceUpdate(update, prefixShape)` requires exact base/update type, equal base/update
  rank, and equal prefix/base rank. Every fully static axis proves checked
  `prefix + update <= base`; an [axis](../../glossary.md#term-axis) with any unresolved [extent](../../glossary.md#term-extent) defers its whole fit. The
  `CropToShapeAttrs` target is the exact update Shape and its prefix is the exact caller Shape,
  while the result retains exact base Shape/type, eligibility OR, unresolved layout, and ordered
  `[base, update]` provenance. The operation performs functional replacement without mutation,
  addition, partial fit arithmetic, or a Model-owned constraint object.
- `Tensor.cropToShape(targetShape, prefixShape)` requires equal input/target/prefix ranks and
  validates every fully static `prefix + target <= input` bound with checked arithmetic. Any axis
  involving an unresolved extent retains that inequality for later binding or execution. Every
  success retains the exact target Shape, input type/eligibility, unresolved layout, and one
  `SLICE`/`CropToShapeAttrs` [occurrence](../../glossary.md#term-occurrence) with `[input]` provenance. It does not clamp, bind, infer,
  mutate, promise a view/copy, or read values.
- `Tensor.select(int, long)` normalizes one existing axis. A static selected extent normalizes one
  negative index and bounds-checks every resulting coordinate; a dynamic selected extent retains
  a non-negative index with its upper bound deferred and rejects a negative index. The result
  removes exactly that Dimension, preserves every unaffected exact reference, and maps rank one
  to the canonical scalar Shape.
- Resolved input geometry with a non-empty select result produces one new view descriptor after
  checked offset advancement and selected-stride removal. Unresolved input and empty results stay
  unresolved. Every success preserves exact type and eligibility, records normalized
  `SelectAttrs` and `[input]` provenance, and remains fresh, unlabeled, and storage-free without a
  physical alias, gradient, compiler, materialization, backend, or execution guarantee.<span data-glossary-navigation="true"> ([compiler glossary](../../glossary.md#term-compiler); [backend glossary](../../glossary.md#backend))</span>
- `Tensor.pad(long[], long[], double)` requires two non-null rank-sized arrays, clones them before
  element inspection, rejects negative widths, and performs canonical checked static or symbolic
  addition in before-then-after order. Static zero extents and empty scalar arrays are valid; zero
  widths preserve the exact input Dimension reference. The raw `double` constant is retained
  without conversion.<span data-glossary-navigation="true"> ([constant glossary](../../glossary.md#term-constant))</span>
- `Tensor.tile(long...)` requires one non-null rank-sized array, clones it before element
  inspection, rejects non-positive repeats, and performs canonical checked static or symbolic
  multiplication. Static zero extents and empty scalar arrays are valid; repeat one preserves the
  exact input Dimension reference. Both operations accept every data type, preserve exact type and
  gradient eligibility, always leave result layout unresolved, record normalized attributes and
  `[input]` provenance, and remain fresh, unlabeled, and storage-free without binding or
  evaluating formulas, performing materialization, or executing work.
- `Tensor.unfold` requires rank at least one, a static selected extent, positive size and step,
  and a fitting window. It normalizes the raw source axis against input rank, derives a checked
  window count, appends size as the final dimension, and accepts every current data type.
- `Tensor.foldAxis` requires input rank at least two, normalizes the raw axis against target rank,
  accepts floating or signed-integral input, and requires positive static final window size plus
  a static matching window count. It supports explicit zero output only with zero windows, removes
  the final Dimension, restores `outputSize`, and sums overlaps semantically.
- Both `Tensor.unfold2d` overloads require floating rank-four NCHW input but accept static or
  unresolved channel/height/width Dimensions. Direct attributes mean conceptual positive-zero
  padding; the second overload requires an exact matching typed scalar and retains its reference
  and raw bits for all out-of-domain positions. <span data-glossary-navigation="true"> ([NCHW glossary](../../glossary.md#term-nchw); [channel glossary](../../glossary.md#term-channel); [padding glossary](../../glossary.md#padding); [raw bits glossary](../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>`Tensor.fold2d` requires floating rank-three
  canonical columns structurally equal to formulas derived from an explicit rank-four NCHW target,
  including an equal batch Dimension. Unrelated unresolved symbols fail locally. Every window-
  transform result preserves type and eligibility, leaves layout unresolved, records exact<span data-glossary-navigation="true"> ([batch Dimension glossary](../../glossary.md#batch-dimension))</span>
  `[input]` provenance, and remains fresh, unlabeled, and storage-free.
- Current value objects are [immutable](../../glossary.md#term-immutable) and defensively copy caller-owned arrays where applicable.
- Operation-kind implementations must return a non-null, non-blank name and a stable immutable
  non-empty signature list. Operation-attribute implementations must preserve immutable value
  semantics; the marker interface cannot enforce immutability at runtime.<span data-glossary-navigation="true"> ([signature glossary](../../glossary.md#term-signature); [marker interface glossary](../../glossary.md#term-marker-interface); [runtime glossary](../../glossary.md#term-runtime))</span>
- [`Operation`](../../glossary.md#operation) rejects a null kind or attributes value and any incompatible exact attributes class,
  retains valid references unchanged, and exposes the selected family-owned signature without
  storing it as another record component.
- `IndexAxisAttrs` rejects a negative [normalized axis](../../glossary.md#normalized-axis) with `IllegalArgumentException` and exact
  message `axis must be non-negative: <axis>`. It retains every non-negative `int`, including
  `Integer.MAX_VALUE`, without proving that the axis exists for an eventual data input.
- `ScatterElementsAttrs` checks its normalized axis first with the same exact negative-axis
  failure, then requires a non-null `ScatterReduction` with `NullPointerException("reduction")`.
  It retains every non-negative axis and exact reduction unchanged without validating an input
  rank, Shape, data type, index bound, or duplicate target. `NONE` duplicates remain invalid but
  require later value-aware detection.
- `CompositionAxisAttrs` rejects a negative normalized axis with `IllegalArgumentException` and
  exact message `axis must be non-negative: <axis>`. It retains every non-negative `int`, including
  `Integer.MAX_VALUE`, without proving an input or insertion bound.
- `PadAttrs` null-checks `before` and `after` in component order, rejects unequal sizes, then
  validates paired entries in ascending index order: before null, after null, before negative,
  after negative. Only complete valid input is snapshotted in component order. Empty lists,
  `Long.MAX_VALUE`, [signed zero](../../glossary.md#term-signed-zero), [NaN](../../glossary.md#term-nan), and [infinities](../../glossary.md#nan-infinity-and-signed-zero) are structurally valid; no [rank](../../glossary.md#term-rank), [Shape](../../glossary.md#shape),
  [`DataType`](../../glossary.md#data-type--datatype), or result behavior is inferred.
- `TileAttrs` null-checks `repeats`, then rejects the first null, zero, or negative entry in
  ascending order before taking one immutable snapshot. Empty lists and <span data-glossary-navigation="true"> ([snapshot glossary](../../glossary.md#term-snapshot))</span>`Long.MAX_VALUE` are
  structurally valid; no input rank or result-Shape multiplication is performed.
- `TargetShapeAttrs` rejects a null [target](../../glossary.md#term-target) with `NullPointerException("targetShape")`, retains the
  exact non-null immutable Shape reference, and performs no input-dependent compatibility check.
- `PermutationAttrs` validates a non-null list and indexed elements in encounter order, rejects a
  negative, out-of-range, or first duplicate axis with the documented indexed message, and only
  then stores one immutable snapshot. An empty axes list is the rank-zero identity. The value
  validates a complete permutation of its own list size but no eventual input rank.<span data-glossary-navigation="true"> ([rank-zero glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix); [permutation glossary](../../glossary.md#permutation))</span>
- `AxisTransformAttrs` rejects a negative normalized position with `IllegalArgumentException` and
  exact message `axis must be non-negative: <axis>`. It retains every non-negative value without
  proving an insertion bound, removal bound, or singleton input dimension.
- `CastAttrs` rejects a null target [data type](../../glossary.md#data-type--datatype) with `NullPointerException("targetDataType")`,
  retains every valid `DataType` reference unchanged, and performs no source compatibility,
  conversion, or backend-capability validation.
- `AxisReductionAttrs` rejects a negative normalized axis with `IllegalArgumentException` and
  retains the exact non-negative axis and dimension choice. `ArgExtremaAttrs` performs that axis
  check before rejecting a null `ArgExtremaTiePolicy` with `NullPointerException("tiePolicy")`. Ordinary
  full reductions use `NoOperationAttrs.INSTANCE`; no negative all-axis sentinel exists. These
  semantic values do not by themselves validate rank, construct Tensor results, or define
  executable or backend behavior. Public `sum`, [`mean`](../../glossary.md#term-mean), `prod`, reduction `min`, reduction `max`,
  boolean `all`, [boolean](../../glossary.md#data-type--datatype) `any`, and `argMin`/`argMax` perform the local rank and result construction
  described above. Ordinary integral sum/product use exact-width modular semantics and bounded
  empty identities; integral min/max use signed order and bounded extrema identities. Arg extrema
  accept only floating or integral input, require an explicit policy at their shared helper
  boundary, fix results to non-differentiable INT64, use the documented shared floating/integral
  ordering, and reject a statically empty selected axis.<span data-glossary-navigation="true"> ([extrema glossary](../../glossary.md#saturation-pole-and-extremum); [differentiable glossary](../../glossary.md#derivative-and-chain-rule))</span>
- `MaskedReductionAttrs` rejects a negative normalized reduction axis with
  `IllegalArgumentException` and exact message `axis must be non-negative: <axis>`. It retains
  every non-negative axis without storing a Shape, mask, mapping, or broadcast plan. The value
  records masked SUM/MEAN semantics only and performs no Shape validation, Tensor construction,
  provenance construction, value selection, counting, or execution. Public masked <span data-glossary-navigation="true"> ([mask glossary](../../glossary.md#term-boolean-mask))</span>`sum` and
  `mean` separately require floating input and an exact BOOL mask, normalize one axis, require an
  ordinary right-aligned broadcast whose result equals the input Shape, remove the axis, and
  construct exact `[input, mask]` provenance. Null, type, axis, and broadcast failures occur
  before result identity allocation. Construction performs no storage alignment, numerical work,
  gradient rule, compiler behavior, backend behavior, or execution.
- `CumulativeScanAttrs` rejects a negative normalized axis with `IllegalArgumentException` and the
  exact message `axis must be non-negative: <axis>`. It retains every non-negative axis and both
  mode flags unchanged. The value itself neither validates an input rank nor constructs or
  executes a Tensor result. Public `Tensor.cumSum` and `Tensor.cumProd` separately accept floating
  or integral input, reject BOOL before axis validation, normalize one positive or negative axis,
  retain the exact Shape/type/eligibility metadata in an unresolved descriptor, and record exact
  one-input provenance. Each valid call is fresh and performs no accumulation, gradient work,
  compiler capture, backend behavior, or execution.<span data-glossary-navigation="true"> ([capture glossary](../../glossary.md#term-capture-meanings))</span>
- `SoftmaxAttrs` rejects a negative normalized axis with `IllegalArgumentException` and the exact
  message `axis must be non-negative: <axis>`. It retains every non-negative axis unchanged but
  does not itself validate rank, construct a Tensor or provenance, evaluate normalization, define
  a gradient or numerical algorithm, report backend support, or execute work. Public
  `Tensor.softmax` and `Tensor.logSoftmax` separately require floating input, normalize one caller
  axis, retain exact Shape/type/eligibility metadata in an unresolved descriptor, and create fresh
  exact-kind one-input provenance without numerical, gradient, compiler, backend, or execution
  behavior.
- `Tensor.cast` requires a non-null target and accepts all 49 current source/target pairs. Each
  successful call returns a fresh unlabeled storage-free expression, including for a same-type
  request, with the exact input Shape reference, unresolved layout, typed target attributes, and
  exact one-input provenance. Gradient eligibility survives only an already-eligible
  floating-to-floating cast. Construction neither reads values/storage nor defines conversion,
  canonicalization, gradient-rule, compiler, or backend behavior.
- `TensorDescriptor` rejects null components, resolved layouts incompatible with their paired
  shape, and gradient requests for non-differentiable data types. Layout reconstruction can report
  checked arithmetic overflow.
- [`Tensor`](../../glossary.md#tensor) retains exact stable ID and descriptor references, a normalized label, and immutable
  optional provenance, and uses object identity for equality and hashing. A present provenance
  retains the exact <span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-reference-identity))</span>`TensorProvenance` reference for the tensor's lifetime without synchronization
  and must select that same exact descriptor reference from its producer position; structural
  descriptor equality alone is insufficient.
  It borrows optional host storage, validates data type, resolved referenced span, and
  attachment-time liveness in that order, and changes only the storage reference through
  synchronized snapshot, replacement, and clearing methods. Failed replacement preserves the
  previous reference, later storage death remains observable, and neither transition changes
  provenance or metadata-only diagnostics.<span data-glossary-navigation="true"> ([lifetime glossary](../../glossary.md#term-lifetime); [synchronization glossary](../../glossary.md#term-synchronization); [host storage glossary](../../glossary.md#host-storage); [referenced span glossary](../../glossary.md#referenced-element-span); [liveness glossary](../../glossary.md#term-liveness))</span>
- [`TensorProducer`](../../glossary.md#producer--tensorproducer) rejects null components and indexed elements, empty output descriptors, and
  input/output counts outside the selected operation signature. It snapshots ordered inputs and
  output descriptors with exact element references, derives output count from the descriptor
  list, and uses ordinary object identity rather than structural equality or an allocated producer
  ID. It retains one canonical exact Tensor wrapper per output slot in a final immutable snapshot;
  indexed lookup validates both bounds and returns the retained wrapper without allocation,
  traversal, or reconstruction.<span data-glossary-navigation="true"> ([operation signature glossary](../../glossary.md#operation-signature); [structural equality glossary](../../glossary.md#term-structural-equality); [lookup glossary](../../glossary.md#term-lookup))</span>
- `TensorProvenance` rejects a null [producer](../../glossary.md#term-producer) and every output index outside
  `[0, producer.outputCount())`. It retains the exact producer and index, then derives the exact
  operation, immutable ordered inputs, and selected descriptor. Record equality uses producer
  identity and the index; neither producer nor provenance establishes graph membership or
  executable support.<span data-glossary-navigation="true"> ([graph glossary](../../glossary.md#graph))</span>
- Static `Tensor.concat` and `Tensor.stack` reject a null input array, empty input sequence, or the
  first null copied element before descriptor inspection. Axis validation precedes encounter-order
  type/Shape checks. Both preserve ordered exact inputs, require exact data types, OR gradient
  eligibility, and return fresh unresolved unlabeled storage-free results. CONCAT additionally
  requires equal rank and non-axis Dimensions and encounter-order folds every selected extent
  through canonical checked symbolic addition. STACK requires identical Shapes and inserts a
  static input-count Dimension. Neither binds or evaluates symbolic formulas, promotes,
  broadcasts, reads values/storage, or materializes.
- `Tensor.unstack` normalizes one existing axis, rejects a dynamic selected extent or a static
  extent above `Integer.MAX_VALUE`, then returns one fresh scalar SELECT per coordinate in an
  immutable ordered List. Each output preserves input metadata, uses SELECT's conditional layout,
  and has an independent one-output producer over <span data-glossary-navigation="true"> ([scalar SELECT glossary](../../glossary.md#scalar-select))</span>`[input]` with provenance index zero. A zero
  extent consumes no ID. Mid-request ID exhaustion may consume earlier IDs but returns no partial
  List.
- `Tensor.add`, `sub`, `mul`, `minimum`, and `maximum` require same-category floating or signed-
  integral operands; `div` and tensor-valued `pow` remain floating-only. Every form requires a
  non-null right operand and locally provable right-aligned broadcasting. Floating extrema
  propagate NaN, select negative or positive zero for minimum or maximum, and otherwise use
  ordinary numeric order; integral extrema use signed order. Each successful call returns a fresh
  unlabeled storage-free Tensor with unresolved layout, gradient eligibility equal to input OR,
  and exact matching parameterless operation plus ordered <span data-glossary-navigation="true"> ([broadcasting glossary](../../glossary.md#broadcasting); [positive zero glossary](../../glossary.md#nan-infinity-and-signed-zero))</span>`[left, right]` provenance. Null, type,
  and shape failures precede ID allocation; identity exhaustion follows local descriptor and
  provenance construction. These methods do not calculate values, simplify expressions, capture
  graphs, or define gradient or backend behavior.
- `Tensor.greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, and `notEqualTo`
  require a non-null right operand, accept same-category floating or signed-integral input pairs,
  and require locally provable right-aligned broadcasting. Each successful call returns a fresh
  unlabeled storage-free [`BOOL`](../../glossary.md#data-type--datatype) Tensor with unresolved layout, false gradient eligibility, the
  exact parameterless comparison operation, and ordered `[left, right]` provenance. Floating
  ordered relations are false for NaN and treat signed zeros as equal; equality is exact numeric
  equality and inequality is its complement. Integral comparison is exact after signed promotion.
  Null, type, and shape failures precede ID allocation; identity exhaustion follows local
  descriptor and provenance construction. These methods do not inspect values, create gradient
  rules, capture graphs, or provide backend execution.
- `Tensor.logicalAnd` and `logicalOr` require a non-null right operand and exact `BOOL` data type
  on both ordered inputs, then derive one locally provable right-aligned broadcast shape.
  `Tensor.logicalNot` requires an exact `BOOL` receiver and retains its exact input `Shape`
  reference without broadcasting. Every successful call returns a fresh unlabeled storage-free
  `BOOL` Tensor with unresolved layout, false gradient eligibility, the exact parameterless
  logical operation, and ordered binary or one-input provenance. Null, kind, type, and shape
  failures precede ID allocation; identity exhaustion follows fixed descriptor and provenance
  construction. These methods do not inspect truth bytes, short-circuit, simplify or reorder
  expressions, insert casts, create gradient rules, capture graphs, or provide backend execution.<span data-glossary-navigation="true"> ([binary glossary](../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Static `Tensor.where` requires non-null condition, true-branch, and false-branch inputs in that
  order, exact BOOL condition type, floating branches, and two locally provable broadcasts in
  branch-first then condition-second order. Each successful call returns a fresh unlabeled,
  storage-free Tensor with promoted branch type, final three-way shape, unresolved layout,
  branch-only gradient eligibility OR, exact `WHERE` operation, and ordered
  `[condition, ifTrue, ifFalse]` provenance. Validation failures precede ID allocation; identity
  exhaustion follows local descriptor and provenance construction. The method does not inspect
  values, choose or evaluate a branch, define gradient routing, capture a graph, or provide ONNX
  or backend execution.<span data-glossary-navigation="true"> ([ONNX glossary](../../glossary.md#term-onnx))</span>
- `Tensor.abs`, `neg`, `reciprocal`, [`log`](../../glossary.md#term-log), `log1p`, [`exp`](../../glossary.md#term-expx), `expm1`, [`erf`](../../glossary.md#term-erf), `sqrt`, `rsqrt`,
  `floor`, `ceil`, `sign`, `relu`, [`sigmoid`](../../glossary.md#term-sigmoid), and [`tanh`](../../glossary.md#term-tanh) accept only floating receiver data types and retain
  the exact data type, shape reference, and gradient-eligibility flag. Each successful call returns
  a fresh unlabeled storage-free Tensor with unresolved layout and exact matching parameterless
  operation plus one-input provenance. Type failures precede ID allocation; identity exhaustion
  follows local descriptor and provenance construction. The methods do not inspect mathematical
  domains, simplify chains, define numerical accuracy, capture graphs, or provide gradient or
  backend behavior. `rsqrt`, `log1p`, and `expm1` are first-class transforms; together with `exp`
  and `tanh`, they select no algorithm, bitwise result, fixed accuracy bound, or [backend route](../../glossary.md#backend-route).
- `Tensor.isFinite`, `isNaN`, and `isInf` accept only floating receiver data types and construct
  fixed BOOL results with the exact input Shape, unresolved layout, false gradient eligibility,
  and exact matching one-input provenance. They record future value classifications without
  inspecting storage, eagerly producing booleans, defining gradients, capturing a graph, or
  executing work.
- The [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) overloads of `Tensor.add`, `sub`, `mul`, `minimum`, `maximum`, `clampMin`, and
  `clampMax` accept matching floating or signed-integral receivers; `div`, `pow`, and first-class
  `clamp` remain floating-only. Every scalar or bound requires exact receiver/value data-type
  equality. Their `double` overloads adapt only to exact FLOAT64. All forms preserve the input
  type, shape reference, and gradient eligibility. Each
  successful call returns a fresh unlabeled storage-free Tensor with unresolved layout and exact
  matching one-input provenance. <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../glossary.md#term-float64))</span>`clamp` validates the input type before rejecting only a strictly
  inverted represented range and remains one `CLAMP` operation with ordered
  `MIN(MAX(input, minValue), maxValue)` meaning. Scalar extrema use the fixed NaN-propagating and
  signed-zero-selecting policy; exact scalar bits remain metadata rather than bitwise comparison.
  Type and range failures precede ID allocation; the methods do not convert parameters, simplify
  expressions, inspect values, capture graphs, or provide gradient or backend behavior.<span data-glossary-navigation="true"> ([signed-zero glossary](../../glossary.md#nan-infinity-and-signed-zero))</span>
- `TensorFactory` rejects null argument containers before ID allocation, delegates label and
  storage semantics to <span data-glossary-navigation="true"> ([argument glossary](../../glossary.md#term-argument))</span>`Tensor` after allocation, and assigns IDs unique among its allocations in
  one JVM. Delegated semantic failures consume IDs; exhaustion after <span data-glossary-navigation="true"> ([JVM glossary](../../glossary.md#term-java-virtual-machine-jvm))</span>`Long.MAX_VALUE` is permanent.
  Its allocation overloads require resolved layout, reject referenced span above
  `Integer.MAX_VALUE`, create exact-span type-matched primitive-array heap storage, and delegate
  only after allocation and wrapping. Preallocation and JVM allocation failures consume no ID;
  blank-label failure and identifier exhaustion occur after heap allocation. Automatic scope keeps
  factory-created arrays reachable without an arena, close operation, or external owner.<span data-glossary-navigation="true"> ([heap glossary](../../glossary.md#term-heap); [scope glossary](../../glossary.md#term-scope); [arena glossary](../../glossary.md#term-arena); [owner glossary](../../glossary.md#term-owner))</span>
- Its six flat-array overloads require exact [carrier](../../glossary.md#term-storage-carrier)/data-type matching, resolved
  `DENSE_CONTIGUOUS` layout, and source length equal to logical element count. Numeric carriers and
  raw BFLOAT16 bits are copied unchanged; BOOL bytes are normalized to zero or one. The source is
  not retained. Import validation failures consume no ID, while blank-label failure and exhaustion
  occur after destination allocation and before population.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../glossary.md#term-bfloat16))</span>
- Its nested-array method requires runtime rank at least two, one of the same six ultimate
  primitive carriers, a completely rectangular non-null structure, and no empty non-final axis.
  It infers an exact fully static dense descriptor, flattens into a fresh matching carrier, and
  delegates once to flat import. Numeric and raw BFLOAT16 values remain unchanged; BOOL
  normalization remains in flat import. Structural and descriptor failures consume no ID; blank
  label and exhaustion retain the flat-import destination-allocation and ID side effects. Source
  ownership remains with the caller, and concurrent mutation has no deep-snapshot guarantee.
- Its scalar methods infer exact data type from primitive signatures and create rank-0 dense
  values; BFLOAT16 alone performs an explicit binary32-to-BFLOAT16 conversion. Zeros and ones
  require fully static shapes, synthesize dense descriptors, and support scalar and empty shapes.
  Like methods copy only shape and data type from the template. Static/count/layout/gradient
  validation happens before source or destination allocation. Blank-label failure and exhaustion
  happen after destination allocation; scalar and one creation have also allocated their source
  carrier, while zero creation has no source carrier. Every successful result has independent
  descriptor, layout, storage, backing array, Tensor object, and factory ID.<span data-glossary-navigation="true"> ([binary32 glossary](../../glossary.md#floating-point-precision-and-rounding))</span>
- Its type-safe full-value methods require a fully static shape and infer the exact data type
  from a primitive value; named low factories round binary32 once to the requested format.
  Each method fills one exact carrier and delegates once to raw import, preserving stored
  values and canonical BOOL bytes. `identityMatrix` supports every Model data type and non-negative
  rectangular dimensions, writes typed one only on the main diagonal of one default-zero carrier,
  and delegates once; `eye` is a pure canonical delegation. Early shape, dimension, count, layout,
  and gradient failures consume no ID. Blank labels, exhaustion, and unexpected copy failures have
  the same late source/destination/ID effects described by the factory contract.
- Its `int` and `long` range overloads create [eager](../../glossary.md#term-eager-evaluation) non-empty, inclusive-start, exclusive-end
  [`INT32`](../../glossary.md#data-type--datatype) and [`INT64`](../../glossary.md#data-type--datatype) tensors with gradients disabled. Positive or negative non-zero steps must
  advance toward the end. Exact count arithmetic and a no-post-final-addition fill loop preserve
  primitive-boundary behavior, while counts above `Integer.MAX_VALUE` fail before allocation.
  One exact carrier is delegated to flat import and is not retained.
- Prefix shaping is not a public product capability. Package-private test-source fixtures may
  prepare strict or cyclic carriers and delegate to public flat import, but production source,
  generated production Javadoc, and the TensorFactory surface expose no prefix method.<span data-glossary-navigation="true"> ([prepare glossary](../../glossary.md#prepare))</span>
- `TensorRandoms.randomNormal` requires a caller-owned `RandomGenerator`, a fully static
  Java-array-sized shape, a floating data type, finite mean, and finite numerically non-negative
  standard deviation. It consumes one <span data-glossary-navigation="true"> ([finite glossary](../../glossary.md#term-finite); [standard deviation glossary](../../glossary.md#term-standard-deviation))</span>`nextGaussian()` call per row-major element, transforms in
  binary64 with ordinary multiplication then addition, converts to the exact FLOAT64, FLOAT32,
  BFLOAT16, or FLOAT16 carrier, and delegates once to flat import. The source is never retained, substituted,
  synchronized, seeded, reset, split, or closed. Prevalidation and source-carrier allocation
  failures consume no calls or ID; a source exception preserves prior source advancement; blank
  label and exhaustion occur after all calls and destination allocation under the delegated
  flat-import side effects.<span data-glossary-navigation="true"> ([row-major glossary](../../glossary.md#term-row-major); [binary64 glossary](../../glossary.md#floating-point-precision-and-rounding); [FLOAT32 glossary](../../glossary.md#term-float32); [FLOAT16 glossary](../../glossary.md#term-float16))</span>
- `TensorRandoms.randomUniform` requires a caller-owned `RandomGenerator`, a fully static
  Java-array-sized shape, a floating data type, finite binary64 bounds, and a lower bound strictly
  less than the upper bound. It consumes one bounded `nextDouble(lower, upper)` call per row-major
  element, stores the returned binary64 value directly or narrows it to FLOAT32/BFLOAT16/FLOAT16,
  and delegates once to flat import. A conforming source's binary64 result is half-open; narrowing may
  equal the corresponding narrowed upper bound. The source is never retained, substituted,
  synchronized, seeded, reset, split, or closed. Prevalidation and source-carrier allocation
  failures consume no calls or ID; a source exception preserves prior advancement; blank label and
  exhaustion occur after all calls and destination allocation under the delegated flat-import
  side effects.<span data-glossary-navigation="true"> ([half-open glossary](../../glossary.md#term-half-open-interval); [narrowing glossary](../../glossary.md#term-narrowing))</span>
- The two `TensorRandoms.randomInt` overloads infer `INT32` from primitive `int` bounds and
  `INT64` from primitive `long` bounds. They require a fully static Java-array-sized shape and a
  strict half-open interval, always disable gradients, and consume one matching bounded<span data-glossary-navigation="true"> ([half-open interval glossary](../../glossary.md#term-half-open-interval))</span>
  `nextInt(origin, bound)` or `nextLong(origin, bound)` call per row-major element. Each direct
  result is stored without modulo, unbounded sampling, floating arithmetic, or conversion in one
  exact carrier and delegated once to flat import. The same-carrier bound cannot express an
  exclusive value above the carrier maximum, so no full-domain form is provided. Prevalidation
  and source-carrier allocation failures consume no calls or ID; a source exception preserves
  prior calls; blank-label failure and exhaustion occur after all calls and destination
  allocation without rollback.<span data-glossary-navigation="true"> ([sampling glossary](../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [rollback glossary](../../glossary.md#term-rollback))</span>
- `TensorRandoms.randomBernoulli` requires a caller-owned `RandomGenerator`, a fully static
  Java-array-sized shape, and a finite probability in `[0, 1]`. It always creates a BOOL result
  with gradients disabled. Every row-major element consumes one unbounded `nextDouble()` call,
  including at probability zero and one, and stores canonical byte one exactly when the draw is
  strictly less than the probability. A custom source result is not post-validated. One complete
  `byte[]` is delegated once to BOOL flat import. Prevalidation and source-carrier allocation
  failures consume no calls or ID; a source exception preserves prior calls; blank-label failure
  and exhaustion occur after all calls and destination allocation without rollback.
- `MemorySegmentStorage` rejects null inputs, negative capacity, checked byte-size overflow, an
  inexact segment byte size, and an initially dead scope in that order. It borrows the exact segment
  and owns no allocation or lifetime.
- `GraphValue` rejects null [identity](../../glossary.md#term-identity-meanings) or [descriptor](../../glossary.md#term-descriptor) components and stores no producer relationship.
- `CompiledNode` snapshots ordered lists, permits repeated inputs, requires non-empty
  within-node-unique outputs, and validates final input/output counts against the selected operation
  signature. It does not perform graph-wide or operand-aware descriptor validation.<span data-glossary-navigation="true"> ([node glossary](../../glossary.md#node))</span>
- [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) snapshots all collections and validates graph-wide structural closure,
  topological order, declared boundaries, producer rules, and exact node-phase coverage. It
  accepts zero-input nodes, repeated node inputs, unused inputs, and zero-node pass-through graphs.<span data-glossary-navigation="true"> ([topological order glossary](../../glossary.md#term-topological-order))</span>
- `ForwardPublicationBinding` rejects null identities and remains a standalone requested-forward
  association without owning-graph validation, policy, storage, backend, or runtime behavior.
- None of the current types owns device storage, runtime residency, or backend selection. The
  current <span data-glossary-navigation="true"> ([device glossary](../../glossary.md#term-device); [residency glossary](../../glossary.md#residency))</span>`Tensor` also owns no storage lifetime, graph-local identity, compiler graph capture, or
  execution state; its provenance is model origin metadata only.<span data-glossary-navigation="true"> ([graph capture glossary](../../glossary.md#capture-graph-ports-and-reconstruction))</span>

See generated Javadoc for the exact member-level exception and [nullability](../../glossary.md#term-nullability) contract.

---


