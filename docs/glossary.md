# Synaptik glossary

This glossary is the central dictionary for Synaptik terminology. It explains how terms are used in this repository; it is not an independent architecture contract. If a definition conflicts with [`ARCHITECTURE.md`](../ARCHITECTURE.md), the architecture contract wins.

Some entries describe contracts planned by the architecture but not yet implemented. Planning status belongs in the [implementation roadmap](planning/roadmap.md), not in this glossary.

## Implementation-status convention

The current public planning foundation consists of `OperationCapabilityQuery` and
`BackendCapabilityProvider`: one immutable operation-occurrence question and one explicitly
supplied boolean provider collaboration. Planning also has one package-private per-query hard-
eligibility result and evaluator that combine backend-level support, supplied availability, and
an optional exact hard requirement into provider-ordered `BackendId` values. A second
package-private step uses those identities directly as the candidate set and selects one exact
`BackendId` owner through optional preferred-class match and provider order. The public immutable
`PlannedPartition` recipe and a package-private maximal consecutive same-owner generator are also
implemented. The public immutable `LogicalMemoryRequirement` and `LogicalMemoryPlan` recipes and
a package-private derivation from the closed graph plus ordered complete partitions are current as
well. Reusable or public capability matrices, public planning orchestration or owner selection,
numeric or cost scoring, owner-map assembly, compiler consumers, physical memory, provider
implementations, device-level capability, and device selection remain planned.

The compiler currently provides package-private structural graph capture, binding-free
captured-graph verification inference, mandatory dense canonicalization, a bounded first-order
pre-capture autograd transformation, and one optional bounded transformation pipeline over either
the forward-only or combined graph. Verification independently derives every current production
operation occurrence's output descriptors, rejects semantic contradictions, and retains only
undecidable descriptor-visible Shape obligations as typed internal deferred graph constraints.
Forward and generated gradient expressions use one public Tensor algebra and one numerical,
inference, validation, and exact-optimization contract. The optional pipeline performs one guarded
exact arithmetic scan, one bounded logical-splat folding scan, and one sidecar-aware
`DCE -> phase-local CSE -> DCE` sequence, revalidating only changed candidates. This is not a
public compiler, concrete binding service, compile artifact, backend decision, or execution path.

The runtime currently provides the public immutable
[`BufferSlot`](#buffer-slot--bufferslot) and
[`WorkspaceSlot`](#workspace-slot--workspaceslot) identities plus
[`PreparedMemoryPlan`](#prepared-memory-plan--preparedmemoryplan) final slot geometry. It also
provides the nominal `BufferRepresentation` and `WorkspaceRepresentation` lifecycle roles,
`BufferRepresentationBinding`, `RunResourceOwnership`, and one array-backed
[`RunState`](#run-state--runstate) for per-run reference retention and cleanup. Prepare-owned
requirement-to-slot assignment and association are now current. Runtime also has a reusable
[`PreparedRepresentationPlan`](#prepared-representation-plan--preparedrepresentationplan),
package-private all-or-cleaned cold setup, structural residency, and explicit
[`buffer validity`](#buffer-validity) per resident copy. Concrete physical representations remain
backend-owned, and public output-value access remains planned. The immutable
creation-plus-execution-plus-transfer-plus-publication
[`PreparedSchedule`](#prepared-schedule--preparedschedule) contract is current.
Current [`PreparedExecutable`](#prepared-executable--preparedexecutable) recipes perform checked
cold binding and create current [`BoundInvocation`](#bound-invocation--boundinvocation) objects;
current [`PreparedBufferTransfer`](#prepared-buffer-transfer--preparedbuffertransfer) recipes
similarly create [`BoundBufferTransfer`](#bound-buffer-transfer--boundbuffertransfer) actions;
current [`PreparedPublication`](#prepared-publication--preparedpublication) recipes create
[`BoundPublication`](#bound-publication--boundpublication) occurrences, and current
[`RunResult`](#run-result--runresult) leases complete states. The CPU backend supplies concrete
borrowed and run-owned native buffer representations plus bounded fully static pointwise and
affine-copy families. Its forty-eight-opcode pointwise vocabulary includes same-typed binary/scalar extrema, floating
Tensor/Tensor DIV and POW, Tensor/scalar DIV and POW, first-class floating range CLAMP,
canonical-BOOL AND/OR/NOT, all nineteen FLOAT32/FLOAT64 unary kinds, and three separate floating
classifications, while its normalized access plans cover resolved
right-broadcastable scalar/rank/singleton/multi-axis inputs, zero extents, offsets, positive and
broadcast-zero strides, injective output layouts, and derived heap/segment carrier patterns.
Connected pointwise chains contain one through eight occurrences, keep single-use intermediates
virtual, and produce one final store. The affine family composes one-through-eight resolved-layout
CONTIGUOUS, RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, SELECT, or positive-step SLICE
occurrences into one represented-bit source/result boundary copy. It accepts all six data types;
BFLOAT16 remains opaque two-byte movement rather than numerical execution. Preferred-species
vector execution covers selected FLOAT32/FLOAT64
value rows, INT32/INT64 arithmetic and extrema, canonical BOOL logic, and virtual floating
predicate masks through logical masks into WHERE. Other operation topologies, types, strategies,
and routes fail closed. The earlier per-node ADD/worker pipeline is Superseded historical evidence rather than
the current implementation baseline.

Prepare currently provides the public immutable analysis-side contracts in
`io.github.pho001.synaptik.prepare.analysis`: `BackendAnalysisInputs`,
`BackendPreparationPlan`, `PrepareContext`, `PreparationResourceRequirement` with its `Buffer`
and `Workspace` variants, `BackendPartitionAnalysis`, and `BackendPartitionPreparer`. They
project and validate one fully static planned partition, carry exact typed logical-splat constants
for projected graph inputs, retain backend inputs and the selected plan opaquely, and declare
exact shared buffer/workspace needs. Slot assignment, finalization, prepared-executable
construction, the minimal `PreparedPartition` association, complete graph preparation, explicit
schedule assembly, and schedule validation are current Prepare contracts. Executable,
buffer-transfer, and publication scheduling plus shared runner execution are current Runtime
contracts. Other physical resources, production operation routes, Engine composition, and public
output-value access remain planned. The Runtime executable contract itself is current; a current
Prepare finalizer constructs a backend subclass against assigned slots.

The exact arithmetic scan contains seven semantic rules: duplicate-input binary `MIN` and `MAX`;
scalar `MUL` by exact typed positive one for all five numeric types; scalar `DIV` and `POW` by exact
typed positive one for the three floating types; and scalar `ADD` and `SUB` by exact typed zero for
the two integral types. A bypass requires a one-output internal `FORWARD` occurrence, complete
input/output descriptor equality, false gradient eligibility, and a result that is not a graph
output. Scalar rules read immutable `ScalarValueAttrs` metadata, not a Tensor constant or storage.
The following current fold scan consumes only explicitly supplied logical splats and selects BOOL
logic, signed-integral binary arithmetic/comparison, and same-typed signed-integral scalar
arithmetic. Broader algebra, floating/BFLOAT16 evaluation, autograd, downstream lifecycle work,
and backend execution are not current effects of either scan.

The currently implemented terms are the model foundations: data type, static, named dynamic, and
symbolic-expression dimension, shape, broadcasting, layout, element stride, referenced element
span, view, `TensorDescriptor`, typed
`TensorId`, `NodeId`, and `ValueId` values, `OperationKind`, `OperationAttrs`, `NoOperationAttrs`,
`OperationSignature`, the `Operation` descriptor, the `GraphValue` and `CompiledNode` graph-element records,
`GraphPhase`, `CompiledGraphModel`, `ForwardPublicationBinding`, and the raw host-storage contracts
`HostTensorStorage` and `MemorySegmentStorage`, plus public `Tensor` state and the descriptor-based
`TensorFactory` construction boundary with JVM-scoped factory-assigned identity and exact-span
JVM-managed heap allocation for resolved layouts, plus copied flat typed import for resolved
dense-contiguous layouts and copied rectangular nested primitive-array import with exact carrier,
static-shape, and dense-layout inference, plus exact typed rank-0 scalars and independent dense
zero, one, zero-like, and one-like constants, plus exact typed full-value tensors and dense
rectangular identity matrices with `eye` as a pure alias, plus eager typed integer ranges.
Public stateless `TensorRandoms` separately owns explicit-source normal and bounded
continuous-uniform population for the three floating types, bounded integral population for exact
`INT32` and `INT64` output, and BOOL Bernoulli population from a finite scalar probability.
Opaque `GraphRngState` separately owns explicit storage-free graph RNG state construction from two
raw unsigned 64-bit key/counter words; it does not replace eager `TensorRandoms`.
Training-only inverted dropout is also implemented as explicit-state model construction. One
producer consumes the input and private state Tensor, describes public output, auxiliary BOOL
keep-mask, and next-state slots, and performs no sampling or execution.
Strict and cyclic prefix preparation is test-fixture mechanics rather than a production Tensor
capability. Immutable `TensorProvenance` origin metadata is also implemented. That origin model
now uses identity-bearing immutable `TensorProducer` occurrences and indexed
`TensorProvenance` results. Concrete operation kind support now includes the parameterless
`BinaryArithmeticKind` vocabulary for `ADD`, `SUB`,
`MUL`, `DIV`, `MIN`, `MAX`, and `POW`, plus matching public Tensor expression construction with
same-category floating promotion and selected signed-integral ADD/SUB/MUL/MIN/MAX domains, local
broadcasting, descriptor derivation, and ordered provenance. Integral ADD/SUB/MUL are fixed-width
two's-complement modular requests; integral extrema use signed order. DIV and POW remain
floating-only.
The parameterless `UnaryElementwiseKind` vocabulary is also implemented for nineteen unary
arithmetic, transcendental, and activation meanings, plus matching
public floating unary Tensor expression construction with exact type/shape retention and one-input
provenance. The family includes exact GELU, a separately named fixed tanh-approximation GELU, and
canonical SiLU without a `swish` alias. The separate parameterless `FloatingClassificationKind`
vocabulary is implemented for
finite, NaN, and infinity classification, together with floating-only public construction that
returns fixed non-differentiable BOOL metadata. Public pairwise extrema are named `minimum` and
`maximum`; aggregate reductions retain
`min` and `max`. The parameterized `ScalarElementwiseKind` vocabulary is implemented for scalar
`ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW`, and `CLAMP`, together with exact typed
`ScalarValue`, `ScalarValueAttrs`, and `ClampRangeAttrs`, plus matching public floating and selected
signed-integral Tensor expression construction with exact receiver/value type equality, exact
type/shape retention, and one-input provenance. Integral DIV, POW, and range CLAMP remain
unsupported. One-bound clamp methods are conveniences over scalar MAX and MIN and therefore
accept exact integral bounds.
The parameterless `BinaryComparisonKind` vocabulary is implemented for ordered `GREATER_THAN`,
`GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and `NOT_EQUAL` meanings, plus matching
public same-category floating or signed-integral comparison Tensor expression construction with
local broadcasting, fixed non-differentiable BOOL descriptors, signed integral ordering, and
ordered provenance. The parameterless
`BooleanLogicalKind` vocabulary is implemented for elementwise `AND`, `OR`, and `NOT` truth
meanings, plus matching public BOOL-only logical Tensor expression construction. Binary AND and OR
use local broadcasting and ordered provenance; unary NOT retains the exact input shape; and every
logical result has fixed non-differentiable BOOL descriptor facts.
The parameterless `WhereSelectionKind` vocabulary is implemented with the sole `WHERE` identity
and ordered condition, true-branch, and false-branch roles, plus matching static public
`Tensor.where` expression construction. The method requires an exact BOOL condition, promotes two
floating branches, composes branch-first and condition-second local broadcasts, derives an
unresolved result with branch-only gradient eligibility, and records exact three-input provenance.
Value selection, gradient routing and rules, compiler capture, ONNX/backend execution, and
scalar-index `select` remain separate concerns.
The parameterized `CastKind` vocabulary is implemented with the sole `CAST` identity, together
with `CastAttrs` carrying one exact non-null target `DataType`. All six current data types are
representable targets, and public `Tensor.cast` now creates a fresh explicit storage-free
expression for all 36 source/target pairs. It retains the exact input Shape, leaves layout
unresolved, preserves a true gradient request only for floating-to-floating casts, and records
typed target attributes plus exact one-input provenance. Numerical conversion behavior, gradient
rules, compiler capture and canonicalization, and backend execution remain separately owned.
Current package-private first-order compiler autograd reverses a selected floating-to-floating
cast with one ordinary cast to the source floating type; a same-type cast passes the cotangent
through. This adds no model conversion promise or gradient-specific rounding, NaN-payload, or
backend-instruction contract.
The `AggregateReductionKind` vocabulary is implemented for `SUM`, `MEAN`, `PROD`, `MIN`, `MAX`,
`ALL`, `ANY`, `ARG_MAX`, `ARG_MIN`, `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`,
and `L2_NORM`, together with normalized single-axis `AxisReductionAttrs`, ordered
`MultiAxisReductionAttrs`, correction-bearing `StatisticalReductionAttrs`, explicit full-form
`NoOperationAttrs.INSTANCE`, shared `ArgExtremaAttrs`, `ArgExtremaTiePolicy`, and masked SUM/MEAN
`MaskedReductionAttrs`. The masked attributes preserve only one normalized axis. Public floating
`Tensor.sum`, `mean`, `prod`, reduction `min`, and reduction `max` now construct full,
single-axis, and ordered multi-axis expressions with locally derived shapes and one-input provenance.
Except for floating-only mean, the ordinary families also accept exact INT32/INT64. Integral sum
and product use fixed-width modular meaning with reassociation permitted; integral min/max use
signed order. Empty identities are zero, one, the bounded type maximum, and the bounded type
minimum, respectively. Floating and BOOL ordinary empty/special-value policies are also fixed.
Public floating log-sum-exp, corrected variance/standard deviation, and L1/L2 norm expressions
preserve exact type/eligibility and record first-class semantics without decomposition.
The masked `sum(axis, mask)` and `mean(axis, mask)` overloads require ordinary right-aligned
broadcasting to produce exactly the input Shape, remove the selected axis, and record exact
`[input, mask]` provenance. False mask positions exclude their values, including NaN and infinity;
all-false sum is zero and all-false mean is NaN without a payload guarantee.
Aggregate `MIN`/`MAX` remain typed separately from equally named binary elementwise kinds. Boolean
`Tensor.all` and `Tensor.any` now provide the corresponding exact-BOOL expressions with false
gradient eligibility and one-input provenance; aggregate `ALL`/`ANY` remain typed separately from
elementwise `AND`/`OR`. Axis-only `Tensor.argMin` and `Tensor.argMax` accept floating or integral
input and produce fixed non-differentiable INT64 expressions with explicit first- or last-index
policy. Their shared ordering prefers NaN, orders signed zero and infinities deterministically,
uses signed integral order, and rejects a statically empty selected axis. Numerical or truth
evaluation, lowering, backend support, and execution remain planned. Current package-private
compiler autograd supports the floating SUM/MEAN/PROD/MIN/MAX, masked SUM/MEAN, SUM_TO_SHAPE,
LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, and L2_NORM roles described under
[pre-capture Tensor-expression autograd](#pre-capture-tensor-expression-autograd); BOOL and
arg-extrema results remain non-differentiable.
The `CumulativeScanKind` vocabulary is implemented with `CUM_SUM` and `CUM_PROD` semantic
identities, together with `CumulativeScanAttrs` carrying one normalized axis and exact exclusive/
reverse mode flags. Public `Tensor.cumSum` and `Tensor.cumProd` construct all four traversal/
inclusion modes for floating and integral inputs, retaining exact Shape/type/eligibility metadata
in an unresolved descriptor and recording one-input provenance. Value accumulation and execution
remain planned. Current package-private compiler autograd supports floating CUM_SUM and CUM_PROD;
integral scans remain non-differentiable.
The `SoftmaxKind` vocabulary is implemented with distinct `SOFTMAX` probability and
`LOG_SOFTMAX` log-probability meanings, together with `SoftmaxAttrs` carrying one normalized axis.
These semantic values preserve logical positions and describe complete normalization slices
without storing a Tensor or Shape. Public `Tensor.softmax` and `Tensor.logSoftmax` now construct
fresh floating expressions with Shape-aware axis normalization, exact Shape/type/eligibility
retention, unresolved layout, and one-input provenance. Tensor construction performs no numerical
evaluation, gradient construction, compiler work, backend work, or execution. Current
package-private compiler autograd supports both floating kinds through their exact forward
outputs; numerical evaluation, lowering, backend support, and execution remain planned.
The `LossKind` vocabulary is implemented with mean-squared error plus dense-target and index-target
categorical cross entropy directly from logits. The shared loss-only `LossReduction` values are
`NONE`, `SUM`, and `MEAN`; exact attributes retain each meaning's reduction and, for categorical
losses, normalized class axis and optional exact typed ignore metadata. Public Tensor loss methods
construct fresh two-input expressions with ordered provenance and perform no value evaluation,
Model-owned gradient construction, backend work, runtime execution, or training coordination.
Current package-private compiler autograd supports both roles of MSE and dense categorical loss,
plus the logits role of index categorical loss when class depth is positive and static.
The `BatchNormKind` vocabulary is implemented with distinct stateless `BATCH_NORM_INFERENCE` and
pure `BATCH_NORM_TRAINING` meanings. Both have fixed ordered inputs
`[input, scale, bias, runningMean, runningVariance]`. `BatchNormInferenceAttrs` retains one
normalized layout-neutral channel axis and exact typed positive epsilon;
`BatchNormTrainingAttrs` also retains exact typed new-batch-weight momentum. Public
`Tensor.batchNormInference` requires rank-at-least-two floating input and exact rank-one `[C]`
affine/statistic vectors, preserves the exact input Shape, and records one fresh output at
provenance index zero. It reads or updates no statistic, creates no saved output or hidden state,
and performs no compiler, backend, runtime, or execution work. Public `Tensor.batchNormTraining`
records five shared outputs and exposes normalized output plus explicit next running statistics;
saved batch mean and inverse standard deviation remain producer-described. Current package-private
compiler autograd retrieves those exact wrappers and implements the public-output role matrix. The
Tensor construction path retains no state across calls and performs no compiler, backend,
runtime, or execution work.
The `ScaledDotProductAttentionKind` vocabulary is implemented with one first-class attention
meaning and an exact one-through-two-output occurrence signature, together with
`ScaledDotProductAttentionAttrs` carrying optional exact scale and causal eligibility. The four
public `Tensor.scaledDotProductAttention` forms preserve exact one-output construction with no
hidden weights. The four `scaledDotProductAttentionWithWeights` forms instead return
`ScaledDotProductAttentionResult(output, weights)` at shared producer slots zero and one. Both
families accept floating query/key/value inputs and an optional exact BOOL mask, derive exact
broadcast-batch, weights, and output metadata, and record ordered three- or four-input provenance.
They expose no dropout state. Numerical evaluation, compiler capture and deferred-constraint
proof, gradients, saved-value lifetime, lowering, backend support, and execution remain planned.
The parameterless `ContiguousKind` vocabulary is implemented with the sole `CONTIGUOUS` identity.
It preserves logical values, Shape, DataType, and row-major element order while requesting
canonical dense row-major, zero-logical-offset result geometry. It is a semantic request rather
than the resolved `LayoutKind.DENSE_CONTIGUOUS` classification. Public `Tensor.contiguous()` now
constructs a fresh storage-free expression: static Shapes receive newly resolved canonical
geometry and dynamic Shapes remain unresolved. Input-layout inspection, alias or copy choice,
materialization, compiler behavior, backend behavior, and execution remain owned by later layers.
The `ShapeTransformKind` vocabulary is implemented with distinct `RESHAPE` and `EXPAND`
identities, together with `TargetShapeAttrs` carrying one exact normalized target `Shape`.
`RESHAPE` preserves ordered logical elements under new coordinates, while `EXPAND` logically
repeats compatible singleton or leading axes. Public `Tensor.reshape(long...)` and
`Tensor.reshape(Shape)` now construct fresh storage-free expressions with local count validation,
conditional same-offset view geometry, and one-input provenance. Public `Tensor.expand(long...)`
and `Tensor.expand(Shape)` now add directional right-aligned compatibility plus conditional
same-offset, zero-stride view geometry. An aligned pair containing an unresolved dimension is
retained as a binding-dependent Model expression; the later condition is that the source extent
is one or equals the target extent, and Model creates no deferred constraint. Current compiler
inference adopts that widened case with an occurrence-owned proof and matching preflight and
gradient handling. Concrete binding, materialization, backend behavior, and execution remain
planned.
The `AxisTransformKind` vocabulary is implemented with distinct `PERMUTE`, `EXPAND_DIMS`, and
`SQUEEZE` meanings. `PermutationAttrs` stores an immutable complete normalized output-to-input
axis mapping, and `AxisTransformAttrs` stores one normalized non-negative insertion or removal
position. Public `Tensor.permute` and rank-two `Tensor.transpose()` now construct fresh expressions
with input-rank validation, Shape/layout derivation, and one-input provenance. Public
`Tensor.expandDims` and `Tensor.squeeze` now construct fresh rank-editing expressions with their
distinct normalization domains, static singleton proof for removal, conditional same-offset view
geometry, and one-input provenance. Gradients, compiler behavior, materialization, backend
behavior, and execution remain planned.
The `SliceKind` vocabulary is implemented with `SLICE` extraction and `SLICE_UPDATE` functional
replacement. `SliceAttrs` carries immutable parallel lists of normalized starts, selected lengths,
distinct axes, and signed non-zero steps for both meanings. `CropToShapeAttrs` carries exact target
and prefix Shapes for target-relative `SLICE` extraction or `SLICE_UPDATE` placement whose extents
may remain symbolic. Public `Tensor.slice`, both `sliceAxis` overloads, and `flip` add directional
raw-bound normalization against selected static dimensions, same-rank Shape derivation,
positive-only resolved view geometry, and exact one-input provenance. Public `sliceByLength`
accepts normalized non-negative starts and finite lengths, including when only the selected
unresolved extent's upper bound must be deferred. Public `sliceUpdate` has signed-array and exact
target-relative forms, both retaining exact base Shape with ordered `[base, update]` provenance;
public `cropToShape` retains the exact target/prefix Shape references. Placement and crop leave
layout unresolved. Model stores no deferred constraint object. Current compiler inference,
preflight, and first-order autograd have adopted length-defined extraction and target-relative
placement, including occurrence-owned unresolved fit constraints; materialization, backend
behavior, ONNX mapping, and execution remain planned in later owning layers.
The `PadKind` and `TileKind` vocabularies are implemented with the sole `PAD` and `TILE` meanings,
respectively. `PadAttrs` stores immutable ordered non-negative before/after widths plus one exact
typed scalar constant; `TileAttrs` stores immutable ordered positive complete-pattern repeat
counts. Empty lists are scalar identity parameters, and structurally valid extreme widths and
typed scalar bits are retained without rank, Shape, input compatibility, layout, or result
interpretation. Public
`Tensor.pad` and `Tensor.tile` construction now adds rank validation, checked Shape derivation,
unresolved result layout, exact one-input provenance, and fresh result identity. Gradients,
materialization, compiler/backend/ONNX behavior, and execution remain planned for later owning
tasks and layers.
The `TensorCompositionKind` vocabulary is implemented with distinct ordered `CONCAT` and
inserted-axis `STACK` meanings. `CompositionAxisAttrs` stores one normalized non-negative axis
shared by them. Public unstack is ordered repeated scalar select: every result records `SELECT`,
has an independent one-output producer, and uses provenance output index zero.
The `SelectKind` vocabulary is implemented with the sole scalar-index `SELECT` meaning, together
with `SelectAttrs` carrying one normalized non-negative source axis and one normalized
non-negative scalar coordinate. Selection fixes that coordinate and removes its axis from the
conceptual result. The semantic values contain no input Shape, so they perform no rank or bounds
validation. Public `Tensor.select` now normalizes a raw axis and scalar coordinate, validates
static selected extents, accepts a non-negative index on a dynamic selected extent with its upper
bound deferred, removes the selected axis, derives conditional logical-view geometry, and records
fresh one-input provenance. Value access, physical aliasing, gradients, compiler behavior,
materialization, backend behavior, and execution remain planned or separately owned.
The `AxisGatherKind` vocabulary is implemented with distinct `GATHER` and `GATHER_ELEMENTS`
tensor-index meanings, together with `IndexAxisAttrs` carrying one normalized non-negative data
axis. Every kind has ordered logical inputs `[data, indices]`. Public `gather` and `gatherElements`
construction validates exact `INT32`/`INT64` index type, normalizes the data axis, applies the
family-specific structural Shape rule, preserves data metadata with unresolved layout, and records
fresh ordered two-input provenance. Index-value access and bounds checks, gradients, compiler
behavior, backend behavior, and execution remain planned or separately owned.
The `OneHotKind` vocabulary is implemented with the sole `ONE_HOT` index-encoding meaning and
`OneHotAttrs` carrying one positive static depth. Public `Tensor.oneHot` construction accepts
INT32/INT64 index metadata, appends one fresh static trailing Dimension after the exact input
Dimension references, and records a non-differentiable BOOL result with fresh one-input
provenance. It reads no index values. Negative or out-of-range values are invalid for eventual
execution rather than wrapping, clamping, selecting a default, or producing an all-false row;
compiler analysis, bounds enforcement, lowering, and execution remain planned.
The `GatherNdKind` vocabulary is implemented with the sole tuple-index `GATHER_ND` meaning,
together with `GatherNdAttrs` carrying one normalized non-negative count of shared leading batch
Dimensions. Gather-ND has ordered logical inputs `[data, indices]`; the final indices Dimension
supplies tuple depth rather than duplicating that occurrence-specific fact in attributes. Public
`Tensor.gatherNd` construction now validates exact index type, both ranks, structural shared batch
prefix, static positive tuple depth, and result Shape; it records fresh ordered provenance while
preserving data metadata with unresolved layout. Index-value bounds, gradients, compiler behavior,
backend behavior, and execution remain planned or separately owned.
The `AxisScatterKind` vocabulary is implemented with functional `SCATTER_ELEMENTS`, then
Gather-compatible fixed-add `SCATTER_ADD`. Both use ordered logical inputs
`[data, indices, updates]` and conceptually produce a new result with the exact data Shape without
mutating the base data. Scatter Elements uses `ScatterElementsAttrs` and the reusable
`ScatterReduction` choices `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`; `NONE` duplicate targets are
invalid, with value-aware detection deferred. Scatter Add instead reuses `IndexAxisAttrs`,
requires the exact ordinary-Gather result Shape for updates, and intrinsically accumulates every
update including duplicates. Public Tensor construction validates each contract's exact type,
axis, and Shape rules, combines data/update gradient eligibility, leaves layout unresolved, and
records ordered provenance. Index bounds, value writes/reductions, gradients, compiler behavior,
backend behavior, and execution remain planned or separately owned.
The `ScatterNdKind` vocabulary is implemented with the sole functional tuple-index `SCATTER_ND`
meaning, together with `ScatterNdAttrs` carrying a normalized non-negative shared-batch count and
an explicit `ScatterReduction`. Its ordered logical inputs are `[data, indices, updates]`; tuple
depth remains the final indices Dimension, updates follow the Gather-ND result-Shape formula, and
the conceptual result is a new value with the exact data Shape. Public Scatter-ND Tensor
construction is current: it validates input types, reduction eligibility, ranks, the shared batch
prefix, tuple depth, and updates Shape before producing an unresolved-layout result with exact
ordered provenance. Values, bounds, duplicate detection, writes or reductions, gradients,
compiler behavior, backend behavior, and execution remain planned or separately owned.
The `WindowTransformKind` vocabulary is implemented with distinct general-axis `UNFOLD_AXIS` and
`FOLD_AXIS` meanings plus NCHW `UNFOLD2D` and `FOLD2D` meanings. `UnfoldAxisAttrs` stores one
normalized axis, positive window size, and positive step. `FoldAxisAttrs` stores one normalized
target axis, explicit non-negative output extent, and positive step; the eventual input's final
dimension supplies window size. `Window2dAttrs` stores positive kernel, stride, and dilation,
non-negative symmetric padding, and a floor/ceil rounding choice. `Unfold2dAttrs` adds one exact
typed out-of-domain scalar while retaining that geometry by reference. `Fold2dAttrs` retains one
exact output Shape and one exact window-geometry reference. These values define materialized
windows, scatter-add folding, NCHW im2col, and overlap-summing col2im without Tensor construction.
Public `unfold`, `foldAxis`, both `unfold2d` forms, and `fold2d` expressions add checked static or
canonical symbolic Shape arithmetic, unresolved result layout, preserved data type and gradient
eligibility, and one-input provenance. Gradients,
materialization, compiler capture, lowering, backend/ONNX behavior, and execution remain planned.
Other concrete kind families and expression families, their family attributes, random Operations,
typed access and export, native/runtime/backend allocation,
gradient and publication behavior, compiler entry points and artifacts, planning, prepare,
runtime, concrete backends, concrete trace payload families, trace correlation beyond event
identity and the current node/value/Tensor domains, trace serialization/emission, and training
remain architecture or planning contracts. The implemented trace foundation consists of
`TraceEventId`, `TracePhase`, `TraceLevel`, the open `TracePayload` marker, the generic
`TraceEvent` envelope, and the trace-local `TraceNodeId`, `TraceValueId`, and `TraceTensorId`
correlation values. The implemented backend-contract foundation consists of `BackendId`,
`BackendDeviceId`, the coarse `DeviceClass` categories `CPU` and `ACCELERATOR`, and
`BackendAvailabilitySnapshot` for one backend's caller-supplied point-in-time device-to-class
availability association. It also contains the sealed, method-free `BackendRequirement` family
with exact-backend, exact-device, and device-class hard targets. Planning currently evaluates one
query against explicitly supplied providers, snapshots, and intent through a package-private
result/factory. Discovery and refresh, registration, concrete provider implementations, public
planning orchestration, concrete backend integration, and trace-local backend/device correlations
remain planned. The implemented
config foundation contains `BackendIntent`, `CompileMode`, `GraphOptimizationConfig`, and
`PartitionScoringConfig`. They record hard-requirement optionality, graph scope,
optional-optimization permission, and one optional soft coarse class preference, respectively,
without planning evaluation or lifecycle behavior. Compile aggregation, profiles, and all other
config contracts remain planned. The planning foundation also contains the public
`PlannedPartition` recipe plus internal complete-map grouping; this adds no public orchestration,
logical boundary, memory, prepare, or execution behavior. A definition explains intended meaning;
it is not by itself evidence that a Java type exists.

## Terms

### Aggregate reduction

A computation that combines values from a selected domain into fewer logical positions. The
implemented `AggregateReductionKind` vocabulary includes numeric `SUM`, `MEAN`, `PROD`, `MIN`, and
`MAX`, boolean `ALL` and `ANY`, floating `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`,
`L1_NORM`, and `L2_NORM`, plus index-producing `ARG_MIN` and `ARG_MAX`. An ordinary full
reduction selects every input axis and uses `NoOperationAttrs.INSTANCE`; an ordinary single-axis
reduction uses `AxisReductionAttrs`; ordered multi-axis forms use `MultiAxisReductionAttrs`, or
`StatisticalReductionAttrs` when correction is intrinsic. Representing the full form through
parameter absence avoids a negative all-axis sentinel. False `keepDimensions` removes selected
axes, while true retains them with extent one.

The public sum, product, minimum, and maximum families accept floating or exact INT32/INT64 input;
mean remains floating-only. Integral sum and product use fixed-width modular arithmetic in the
exact result type and permit reassociation. Integral minimum and maximum use signed order. Their
empty-domain identities are zero, one, the bounded type maximum, and the bounded type minimum,
respectively, for full and selected-axis domains. Floating SUM/MEAN/PROD/MIN/MAX and BOOL ALL/ANY
have explicit NaN, infinity, signed-zero, and empty-domain policies. Dynamic selected extents
use the same identities when later bound to zero. Aggregate minimum/maximum use one input and are
distinct from pairwise `minimum`/`maximum`, which use two ordered inputs and
`BinaryArithmeticKind`.

A masked, axis-removing `SUM` or `MEAN` instead uses `MaskedReductionAttrs`, which stores one
normalized axis. Public `sum(axis, mask)` and `mean(axis, mask)` require ordinary right-aligned
broadcasting of the BOOL mask to produce exactly the input Shape and record provenance order
`[input, mask]`. False mask positions exclude their inputs, including NaN and infinity. No selected
values produces zero for masked sum and NaN for masked mean. Masked forms remain floating-only.

Public `argMin` and `argMax` accept floating or integral input, normalize one selected axis, and
produce exact INT64 with false gradient eligibility. They use shared `ArgExtremaAttrs` because the
explicit first/last tie policy is intrinsic to both semantics; neither has a full form. A
statically empty selected axis is invalid, while an unselected zero axis and an unbound selected
extent are structurally accepted. Expression construction records all these meanings but performs
no value aggregation or selection, gradient work, compiler capture, lowering, or execution. See
[Numeric aggregate expressions](api/tensor-api.md#numeric-aggregate-expressions), [Aggregate
reduction semantic kinds and attributes](api/tensor-api.md#aggregate-reduction-semantic-kinds-and-attributes),
[Boolean aggregate expressions](api/tensor-api.md#boolean-aggregate-expressions), [Arg-extrema
expressions](api/tensor-api.md#arg-extrema-expressions), [Multi-axis and statistical reduction
expressions](api/tensor-api.md#multi-axis-and-statistical-reduction-expressions), and [Masked sum and mean
expressions](api/tensor-api.md#masked-sum-and-mean-expressions).

### Sum-to-Shape

An implemented binding-aware [`aggregate reduction`](#aggregate-reduction) that adds a numeric
Tensor into an exact target `Shape`. `Tensor.sumToShape(Shape)` right-aligns that target with the
input: unmatched leading input axes reduce and disappear, aligned target-one axes reduce and
remain with extent one, and equal aligned axes preserve their coordinates. Other fully concrete
pairs are invalid. A pair involving an unresolved Dimension is accepted as a
target-one-or-input-equal obligation for later binding validation.

The operation uses the existing `AggregateReductionKind.SUM` with
`SumToShapeAttrs(targetShape)`; it is not a new operation kind. The fresh result retains exact
input type and gradient eligibility, the exact target Shape, unresolved layout, and ordered
one-input/output-index-zero provenance. It does not bind dimensions, resolve axes, read values,
capture a graph, build a gradient, lower, or execute.

Sum-to-Shape differs from reshape, which changes coordinate interpretation while preserving
element count; from ordinary fixed-axis SUM, whose axes are known when the expression is built;
and from broadcasting, which repeats values toward a larger compatible Shape. A future compiler
may use this public semantic variant when constructing adjoints, but compiler-generated backward
graphs and target binding proof are not current capabilities. See [Binding-aware sum-to-Shape
expressions](api/tensor-api.md#binding-aware-sum-to-shape-expressions).

### Multi-axis reduction

An implemented aggregate occurrence over a caller-ordered set of distinct normalized axes.
`MultiAxisReductionAttrs` snapshots the ordered axes plus `keepDimensions`;
`StatisticalReductionAttrs` adds correction. Axis order is semantic metadata but does not prescribe
physical traversal or a sequential fold. Shape derivation uses membership: selected axes are
removed or replaced with extent one, while unselected Dimension references remain exact. An empty
axis list selects a point domain and is distinct from full reduction over every axis.

### Statistical correction

The implemented non-negative `long` value subtracted from selected-domain count `N` in variance and
standard deviation. Correction zero requests population statistics; correction one is the usual
sample estimator only when `N > 1`. A statically known `N <= correction` is rejected during Tensor
construction, while dynamic proof belongs to a later compiler or binding layer. Correction is not
an accumulator type, rounding mode, denominator clamp, or invalid-domain sentinel.

### Log-sum-exp

The implemented floating reduction meaning `log(sum(exp(x_i)))` over selected axes. It is one
first-class `LOG_SUM_EXP` occurrence rather than stored exp, sum, and log operations. The contract
fixes empty, NaN, infinity, singleton, and signed-zero meaning while leaving stable algorithm,
gradient, compiler lowering, backend route, and execution separately owned.

### Arg-extrema tie policy

The implemented `ArgExtremaTiePolicy` choice for selecting a logical axis index when several
values share a minimum or maximum. `FIRST_INDEX` requests the smallest logical index, and
`LAST_INDEX` requests the largest. A logical index is a position along the selected axis rather
than a physical storage offset. `ArgExtremaAttrs` requires an explicit non-null policy; the
semantic value supplies no default. Public `Tensor.argMin` and `Tensor.argMax` convenience
overloads supply `FIRST_INDEX` explicitly, while complete overloads retain a caller policy.
Integral candidates use signed order. Floating candidates prefer NaN to non-NaN, treat multiple
NaNs as ties, order negative zero below positive zero, and order infinities normally. These
semantics define the requested index without choosing an algorithm or executing a reduction.

### Stable full ordering

An implemented axis-wise request represented by `OrderingKind.SORT` for values or
`OrderingKind.ARGSORT` for logical indices. **Stable** means values in the same ordering class
retain increasing original logical-axis index, independently of physical layout or backend
traversal. **NaN-last** means all floating not-a-number (NaN) values form one final class after
every non-NaN in both ascending and descending requests; NaNs remain stable within that class.
Negative zero precedes positive zero ascending and follows it descending, while infinities use
ordinary numerical order. `SortAttrs` stores only the normalized axis and descending flag because
stability and NaN placement are fixed family semantics. Public `Tensor.sort` preserves input type,
Shape reference, and gradient eligibility; `Tensor.argsort` preserves the exact Shape reference
but produces non-differentiable INT64 indices. Both construct metadata and independent
single-output provenance without comparing values or providing compiler, backend, runtime, or
execution behavior. See [Stable sort and argsort
expressions](api/tensor-api.md#stable-sort-and-argsort-expressions).

### Top-K selected set

The implemented model request represented by `TopKKind.TOP_K` and
`TopKAttrs(axis, k, largest, sorted)`. The **selected set** is the first `k` entries of the same
complete stable numerical order used by full ordering, requested largest or smallest. NaNs remain
after all non-NaNs in either direction, signed zero and infinities use the fixed numerical policy,
and equal candidates prefer increasing original logical-axis index. When `sorted` is false, the
same selected value/index pairs use deterministic increasing original logical-index order; the
flag never permits a backend-dependent permutation.

Public `Tensor.topK` returns `TopKResult(values, indices)` from one shared producer at output slots
zero and one. Construction replaces the selected Shape dimension with static `k`, checks a known
static extent, and defers a dynamic or expression-bound capacity obligation to later compiler or
binding validation. This is model meaning and provenance, not value evaluation, an algorithm,
gradient construction, compiler support, backend lowering, runtime behavior, or execution. See
[Top-K values and indices](api/tensor-api.md#top-k-values-and-indices).

### Gather

An implemented backend-independent tensor-index meaning represented by `AxisGatherKind.GATHER`
and `IndexAxisAttrs(axis)`. It consumes ordered logical inputs `[data, indices]` and replaces
the selected data axis with the complete indices Shape:

```text
result = data[:axis] + indices[:] + data[axis + 1:]
```

Data `[2, 3, 4]`, axis `1`, and indices `[5, 6]` therefore produce
`[2, 5, 6, 4]`; scalar indices produce `[2, 4]`. Public `Tensor.gather` requires INT32 or
INT64 indices, normalizes the caller axis, preserves data type and gradient eligibility, leaves
layout unresolved, and records exact `[data, indices]` provenance without reading index values
or checking bounds. See [Gather and Gather Elements semantic kinds and
attributes](api/tensor-api.md#gather-and-gather-elements-semantic-kinds-and-attributes) and
[axis-gather expressions](api/tensor-api.md#axis-gather-expressions).

### Embedding

An implemented public convenience for using a rank-two floating weight table as an ordinary
axis-zero [Gather](#gather). For weights `[vocabulary, embeddingSize]` and INT32 or INT64 indices,
`weights.embedding(indices)` produces metadata shaped
`indices.shape + [embeddingSize]`. It retains the exact indices Dimensions followed by the exact
weight axis-one Dimension, with result type and gradient eligibility inherited only from weights.

Embedding is not a separate operation kind: each call creates one `GATHER` occurrence with
`IndexAxisAttrs(0)`, ordered `[weights, indices]` provenance, one output at index zero, and one
fresh Tensor identity. Construction reads no index values. Negative and out-of-range values are
invalid for later ordinary Gather execution, and there is no wrapping, padding row, sparse
gradient, maximum-norm, or frequency-scaling option. Gradient construction, compiler analysis,
bounds enforcement, lowering, backend behavior, and execution remain separately owned. See the
[embedding convenience](api/tensor-api.md#embedding-convenience).

### One-hot encoding

An implemented backend-independent index-encoding meaning represented by
`OneHotKind.ONE_HOT` and `OneHotAttrs(depth)`. It consumes one INT32 or INT64 indices Tensor and
adds a positive static trailing class axis. For each input coordinate `p` with eventual value `i`,
its dense BOOL meaning is:

```text
result[p..., j] = (i == j), for 0 <= j < depth
```

Public `indices.oneHot(depth)` retains every input Dimension reference, appends one fresh
`StaticDimension(depth)`, and creates a storage-free BOOL result with `requiresGrad=false`, one
producer, output index zero, and one fresh Tensor identity. Scalar indices produce Shape
`[depth]`; zero-element input and `Long.MAX_VALUE` depth are structurally valid without result
materialization.

One-hot differs from [Gather](#gather): Gather uses index values to select existing data, whereas
one-hot turns each index into a new trailing indicator row. Construction reads no values. Valid
eventual execution requires `0 <= i < depth`; negative and out-of-range values do not wrap,
clamp, select a default, or produce an all-false row. Compiler analysis, bounds enforcement,
gradient construction, lowering, backend support, and execution remain separately owned. See
[one-hot encoding expressions in the Tensor API](api/tensor-api.md#one-hot-encoding-expressions).

### Gather Elements

An implemented aligned tensor-index meaning represented by
`AxisGatherKind.GATHER_ELEMENTS` and `IndexAxisAttrs(axis)`. Data and indices must have equal
rank and structurally equal Dimensions away from the selected axis; the selected extents may
differ. The result retains the exact indices Shape. Data `[2, 3, 4]`, axis `1`, and indices
`[2, 7, 4]` therefore produce `[2, 7, 4]`.

Public `Tensor.gatherElements` requires INT32 or INT64 indices and performs the rank and
increasing-axis alignment checks before constructing fresh unresolved-layout exact
`[data, indices]` provenance. It reads no index value and defines no bound, gradient, compiler,
backend, or execution behavior.

### Scatter Add

An implemented Gather-compatible functional fixed-add meaning represented by
`AxisScatterKind.SCATTER_ADD` and `IndexAxisAttrs(axis)`. Its ordered inputs are
`[data, indices, updates]`. For normalized axis `a`, updates must have the exact Shape produced by
ordinary [Gather](#gather):

```text
updates.shape = data.shape[:a] + indices.shape + data.shape[a + 1:]
```

The fresh functional result retains the exact data Shape and type. An updates coordinate
`[before..., i..., after...]` addresses result coordinate
`[before..., indices[i...], after...]`. Each target starts with its base data value and adds all
addressed updates; duplicate indices therefore accumulate. Signed-integral addition is fixed-
width modular, while floating addition may be reassociated and has no bitwise-order guarantee.

Public `Tensor.scatterAdd(indices, updates, axis)` accepts exact INT32/INT64 indices, exact matching
numeric data/update types, normalizes the axis, and validates the formula without broadcasting or
binding. Scalar indices insert no Dimensions; zero, dynamic, and expression Dimensions are
preserved structurally. Every success has data/update gradient-eligibility OR, unresolved layout,
no label or storage, output index zero, a fresh Tensor identity, and exact ordered provenance.
Construction reads no values, checks no bounds, performs no addition, mutates no input, and defines
no gradient, compiler, backend, runtime, or execution behavior. This differs from
[Scatter Elements](#scatter-elements), whose indices and updates are aligned and same-rank with
data, and [Scatter-ND](#scatter-nd), whose indices contain multi-axis coordinate tuples. See
[Gather-compatible Scatter Add expressions](api/tensor-api.md#gather-compatible-scatter-add-expressions).

### Functional-scatter target group

For one concrete result coordinate `c` in configurable Scatter Elements or Scatter-ND reduction,
the logical multiset containing exactly one base value `data[c]` and exactly one scalar from every
updates position addressed to `c`. Equal values from distinct positions remain distinct members,
so duplicate indices or tuples contribute more than once. In Scatter-ND, every tuple position
contributes its complete update suffix slice scalar by scalar. If no update addresses `c`, the
result is the exact unchanged representation of `data[c]`; no identity arithmetic or
canonicalization occurs.

For `MUL`, `MIN`, and `MAX`, the target-group result is independent of encounter order, layout,
strides, atomic scheduling, reduction tree, and backend traversal. When at least one update
addresses the coordinate, floating `MUL` is one abstract unchanged-format product: NaN propagates;
zero with infinity produces NaN; otherwise zero or infinity has the parity sign of all negative
factors; and a finite non-zero exact product rounds once with round-to-nearest, ties-to-even,
including signed overflow, underflow, subnormal, and zero results. BFLOAT16 begins from its exact
represented 16-bit values and rounds the final product to BFLOAT16. Floating extrema for such a
non-empty update group propagate NaN, use ordinary numeric order including infinities, and select
negative zero for `MIN` or positive zero for `MAX` when both zero signs occur. No NaN
payload, sign, signaling, source, fixed factor sequence, or bitwise result is promised. Integral
`MUL` is exact-width two's-complement modular multiplication; integral extrema use signed order.
Data and updates keep one exact type without promotion, widening, saturation, or conversion.
`NONE`, configurable `ADD`, and fixed [Scatter Add](#scatter-add) retain their existing meanings,
and the forward contract chooses no derivative or subgradient policy. See
[Scatter Elements expressions](api/tensor-api.md#scatter-elements-expressions).

### Scatter Elements

An implemented functional same-rank scatter meaning represented by
`AxisScatterKind.SCATTER_ELEMENTS`, `ScatterElementsAttrs(axis, reduction)`, and
`ScatterReduction`. Its ordered inputs are `[data, indices, updates]`. Indices and updates have
equal rank and Shape and match data away from the selected axis; the functional result has exact
data Shape and does not mutate data.

`NONE` means replacement and requires unique targets. `ADD`, `MUL`, `MAX`, and `MIN`
combine the [functional-scatter target group](#functional-scatter-target-group). Public
construction validates types, reduction
eligibility, axis, rank, and Shape, preserves exact data Shape/type with unresolved layout,
combines data/update gradient eligibility, and records ordered provenance. It reads no values,
checks no bounds or duplicate targets, and performs no write or reduction. See
[Scatter Elements expressions](api/tensor-api.md#scatter-elements-expressions) and
[axis-scatter semantic kinds, reduction, and
attributes](api/tensor-api.md#axis-scatter-semantic-kinds-reduction-and-attributes).

### Gather-ND

An implemented backend-independent tuple-index meaning represented by
`GatherNdKind.GATHER_ND` and `GatherNdAttrs(batchDimensions)`. Its ordered logical inputs are
`[data, indices]`. The normalized non-negative batch count `B` identifies shared leading data and
indices Dimensions `[0, B)`. The final indices Dimension supplies tuple depth `K`; each tuple
indexes data axes `[B, B + K)`, while data axes after `B + K` form the untouched suffix.

For data rank `R` and indices rank `Q`, the conceptual result Shape is:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

The indices prefix is equivalently `indices.shape[:-1]`. Data `[2, 3, 4]` with indices `[5, 2]`,
`B=0`, and `K=2` therefore means result `[5, 4]`. The same data with indices `[2, 5, 1]`, `B=1`,
and `K=1` means `[2, 5, 4]`. Data `[2, 3]` with indices `[2]`, `B=0`, and `K=2` means the canonical
scalar Shape `[]` because neither formula term contributes a Dimension.

`GatherNdAttrs` rejects a negative batch count but stores no input rank, Shape, tuple depth, or
index type. Tuple depth is not an attribute because it belongs to the final indices Dimension of
each operation occurrence. The family signature enforces `GatherNdAttrs` and declares the ordered
two-input, one-output occurrence. The current zero-batch `Tensor.gatherNd(indices)`
convenience uses `new GatherNdAttrs(0)`, not another kind or default value.

The current public Tensor boundary accepts only `INT32` or `INT64` indices, requires both ranks to
fit the batch count, compares shared batch Dimensions structurally, and requires static positive
tuple depth no greater than the remaining data rank. It constructs the formula above once,
retaining exact prefix/suffix Dimension references and returning canonical scalar Shape when both
parts are empty. Each result preserves the data type and gradient eligibility, leaves layout
unresolved, has no label or storage, and records fresh exact `[data, indices]` provenance. It never
reads an index value or checks its bounds. Gather-ND differs from [scalar select](#scalar-select),
whose coordinate is an intrinsic attribute; [Gather](#gather), whose index values
address one selected data axis; and scatter-ND, which writes or combines updates. Gradients,
compiler behavior, materialization, backend behavior, and execution remain separately owned. See
[Gather-ND semantic kind and attributes](api/tensor-api.md#gather-nd-semantic-kind-and-attributes)
and [Gather-ND expressions](api/tensor-api.md#gather-nd-expressions).

### Scatter-ND

An implemented backend-independent functional tuple-index scatter meaning represented by
`ScatterNdKind.SCATTER_ND` and `ScatterNdAttrs(batchDimensions, reduction)`. It has ordered logical
inputs `[data, indices, updates]`. `data` supplies the base and exact result Shape, `indices`
supplies coordinate tuples, and `updates` supplies the values applied at addressed targets. The
result is a new value and does not mutate `data` in place.

For data rank `R`, indices rank `Q`, normalized shared-batch count `B`, and tuple depth `K` from
the final indices Dimension, each tuple indexes data axes `[B, B + K)`. Data axes `[B + K, R)`
form the untouched suffix represented by one update slice. The exact updates formula is
`updates.shape == indices.shape[0:Q-1] + data.shape[B+K:R]`, equivalently
`indices.shape[:-1] + data.shape[batchDimensions + K:]`, and the result Shape is exactly the data
Shape. Data `[2, 3, 4]` with indices `[5, 2]`, `B=0`, and `K=2` therefore uses updates `[5, 4]`;
the same data with indices `[2, 5, 1]`, `B=1`, and `K=1` uses updates `[2, 5, 4]`. Data `[2, 3]`
with indices `[2]`, `B=0`, and `K=2` uses canonical scalar updates `[]`. Their respective result
Shapes are `[2, 3, 4]`, `[2, 3, 4]`, and `[2, 3]`.

The shared `ScatterReduction` values mean replacement, addition, multiplication, maximum, or
minimum over each [functional-scatter target group](#functional-scatter-target-group). `NONE`
replacement requires unique target tuples; duplicate tuples are invalid, but
detection requires index values and does not occur during current expression construction. The
attributes reject a negative batch count before rejecting a null reduction. They store no inputs,
ranks, Shapes, tuple depth, types, descriptor, or provenance. The three current public overloads
default to `NONE` and zero batch Dimensions, accept an explicit reduction with zero batch
Dimensions, or accept both values explicitly. They require exact `INT32`/`INT64` indices and exact
matching data/update types; `NONE` accepts every current data type, while arithmetic reductions
accept floating or integral values and reject `BOOL`. They validate both ranks, the structurally
equal shared batch prefix, static positive tuple depth, and the exact updates Shape. Every valid
request is fresh and produces an unlabeled, storage-free, unresolved-layout result with exact data
Shape/type, data/update gradient-eligibility OR, and ordered `[data, indices, updates]` provenance.
Construction does not read values, check bounds or duplicates, mutate data, write, or reduce.
Scatter-ND differs from [Gather-ND](#gather-nd), which
reads selected data, and [Scatter Elements](#scatter-elements), whose indices address one selected axis.
See [Scatter-ND semantic kind and attributes](api/tensor-api.md#scatter-nd-semantic-kind-and-attributes)
and [Scatter-ND expressions](api/tensor-api.md#scatter-nd-expressions).

### Axis transform

An implemented backend-independent semantic change to axis coordinates. The
`AxisTransformKind` vocabulary contains exactly `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`.
`PERMUTE` uses [`PermutationAttrs`](#permutation); `EXPAND_DIMS` and `SQUEEZE` use one
`AxisTransformAttrs` value whose non-negative axis is interpreted as an output insertion position
or an input removal position, respectively.

These semantic values do not retain a Tensor, Shape, layout, or input rank and cannot by themselves
prove an insertion bound, removal bound, or singleton dimension. Public `Tensor.permute` now owns
raw negative-axis normalization, complete input-rank validation, result Shape/layout derivation,
and provenance. Rank-two `transpose()` is implemented over `PERMUTE` mapping `[1, 0]`, not a
separate semantic kind. Public `Tensor.expandDims` normalizes against rank plus one and inserts a
static singleton; public `Tensor.squeeze` normalizes an existing axis and removes it only when the
dimension is statically known as one. Resolved geometry inserts or removes one stride in a new
same-offset logical view descriptor, while unresolved geometry remains unresolved. Current
package-private compiler autograd reverses supported floating rank edits through the opposite
public Tensor operation. Public gradient requests/publication, materialization, backend behavior,
and execution remain planned. See
[Expand-dimensions and squeeze expressions](api/tensor-api.md#expand-dimensions-and-squeeze-expressions)
and [Axis-transform semantic kinds and
attributes](api/tensor-api.md#axis-transform-semantic-kinds-and-attributes).

### Architecture contract

The normative rules that define Synaptik's module responsibilities, dependency direction, lifecycle boundaries, and core invariants. The repository's architecture contract is [`ARCHITECTURE.md`](../ARCHITECTURE.md). Guides, plans, ADRs, and this glossary may explain those rules but do not override them.

### Autograd

Automatic differentiation: a compiler transformation that derives gradient computations from the
forward computation when the compile mode requires them. In Synaptik, the compiler owns global
reverse traversal, gradient-rule selection, seeds, targets, ordered contribution accumulation,
one phase-aware combined capture, validation, and combined-graph optimization. A concrete backend
does not perform global autograd; it prepares and executes only its assigned regions.

The accepted [pre-capture Tensor-expression autograd](#pre-capture-tensor-expression-autograd)
design uses ordinary public Tensor operations over the original forward expression. This strategy
does not add public Tensor gradient/backward lifecycle state. Model task 0025 is Complete and
supplies canonical producer-output wrappers; Compiler task 0004 is Complete with its first
bounded package-private implementation; Compiler task 0004A is Complete with its first
policy-free exact-composition rule extension; and Compiler task 0004B is Complete with mixed-
floating cotangent normalization, ordinary DIV/MEAN formulas, and the selected direct-zero local
conventions. Compiler task 0005A is Complete with the exact elementwise/activation policy, and
Compiler task 0005B is Complete with binding-aware expansion plus reduction, scan, softmax,
statistics, norm, and normalization formulas. Compiler task 0005C is Complete with layout,
window, indexing/scatter, ordering, and explicit-state dropout formulas. Compiler task 0005D is
Complete with the representable attention, convolution, pooling, and loss formulas. See
[Training graph](architecture/training-graph.md).

### Autotuning / model autotuning

One explicit optional workflow for a model and target. Its first phase extracts actual tunable
workloads, reuses compatible canonical signatures from an explicit workload tuning cache, and
measures only misses. Its second phase measures a bounded set of complete valid graph and plan
candidates end to end and selects an explicit model plan or prepared artifact without repeating
local route-parameter search.

The caller supplies representative input or shape profiles, objective, budget, constraints, and
explicit cache locations. Tuning is optional for correctness and never runs in the runtime hot
path. Running the same workflow over a representative model corpus may pre-seed the same workload
cache; this is not a separate platform-calibration subsystem. The workflow and artifact contracts
remain planned.

A **workload tuning cache** is the explicit file-backed reusable artifact keyed by canonical
workload and target compatibility. A **model plan cache** or **prepared-plan record** is the
model-specific artifact for the selected complete plan. Compatible hits are reused; misses may be
tuned and atomically persisted. Both roles require explicit schema and backend candidate-schema
versions, fingerprints, objective and constraints, and a compact measurement summary. They are
not hidden global state, Java object serialization, or assumed executable payloads.

### Auxiliary mask

A non-public BOOL result of a multi-output operation that records which logical positions were
selected during eventual forward execution. Current dropout describes its keep mask at producer
output slot one while returning only output slots zero and two. A later compiler may capture and
retain the mask for backward construction; no current compiler or gradient rule does so.

### Backend

An execution target identified at planning time, such as CPU, Metal, or CUDA. A backend is
responsible for reporting capabilities and preparing the partitions assigned to it. Generic
architecture discussions use “backend” for this role; a [concrete backend](#concrete-backend) is
the module that implements it. The current [`BackendId`](#backend-identity--backendid) value names
this ownership domain but does not implement the backend role.

### Backend availability snapshot / `BackendAvailabilitySnapshot`

The implemented immutable point-in-time fact for one backend's currently reported available
devices. It stores an explicit [`BackendId`](#backend-identity--backendid) and an immutable
structural copy of a map from same-backend
[`BackendDeviceId`](#backend-device-identity--backenddeviceid) values to
[`DeviceClass`](#device-class--deviceclass) values. The exact backend, device-key, and class-value
references are retained, but later structural changes to a mutable source map cannot affect the
snapshot and the copied map's iteration order is unspecified. An empty map means that the
supplying context reports no currently available device for the named backend.

The snapshot is supplied data, not a provider or service. It performs no discovery, registration,
refresh, liveness monitoring, capability evaluation, ownership selection, planning, preparation,
execution, or trace translation and carries no timestamp, status, or absence reason.

### Backend capability

A backend's declarative answer about computation it can semantically own. The current planning
contract asks about one immutable [`OperationCapabilityQuery`](#operation-capability-query--operationcapabilityquery)
through one explicitly supplied
[`BackendCapabilityProvider`](#backend-capability-provider--backendcapabilityprovider). The answer
is backend-level and boolean; it is separate from registration, availability, device selection,
scoring, route selection, preparation, and execution. Current internal hard eligibility combines
that answer with separate availability and intent facts without changing its level. A capability is not a
kernel, a live executable service, a rejection diagnostic, or a promise that one fixed
implementation route will always be selected. See [Partition scoring](architecture/partition-scoring.md).

### Backend capability provider / `BackendCapabilityProvider`

The implemented planning-owned interface through which one named backend answers
[backend-capability](#backend-capability) questions. `backendId()` returns one stable non-null
[`BackendId`](#backend-identity--backendid). `supports(query)` deterministically reports semantic
ownership support for an immutable query and unchanged immutable provider configuration, rejects
null with `NullPointerException("query")`, and returns no rejection reason.

The provider is an explicitly supplied compile-time collaboration, not a registry, discovery
mechanism, `ServiceLoader` lookup, service locator, availability source, requirement evaluator,
scoring policy, route selector, preparer, or execution service. The CPU backend implements it
through the architecture-approved inward dependency on planning and supports only exact
`BinaryArithmeticKind.ADD` with `NoOperationAttrs`, two inputs, one output, equal fully static
shapes, one of `FLOAT64`, `FLOAT32`, `INT32`, or `INT64`, and either unresolved or
`DENSE_CONTIGUOUS` non-view zero-offset layout. The package-private hard-eligibility step is the
first internal planning consumer. Compile-time plans retain `BackendId`, not a provider object.

### Backend hard eligibility

The implemented internal planning step that determines which backend identities may still own one
[`OperationCapabilityQuery`](#operation-capability-query--operationcapabilityquery) before
scoring. It validates a complete one-to-one association between explicitly supplied
[`BackendCapabilityProvider`](#backend-capability-provider--backendcapabilityprovider) values and
[`BackendAvailabilitySnapshot`](#backend-availability-snapshot--backendavailabilitysnapshot)
values by equal [`BackendId`](#backend-identity--backendid). Only after all duplicates and missing
counterparts have been rejected does it evaluate providers in provider encounter order.

An empty snapshot or exact [`BackendRequirement`](#backend-requirement--backendrequirement)
mismatch skips a provider before its capability call. Every remaining provider receives the exact
query once; a true answer retains that provider's exact backend-identity reference. The result is
an immutable provider-ordered list and may be empty. A later public planning consumer must treat
emptiness as terminal before scoring and must not relax the hard requirement.

Exact-device and device-class requirements use the matching snapshot only to prove current
matching availability. Capability remains backend-level, and hard eligibility neither selects nor
retains a device. The result and evaluator are package-private, so this current behavior creates no
public matrix, public planner workflow, score, ownership decision, diagnostic taxonomy, compiler
integration, preparation, runtime, or execution contract.

### Backend device identity / `BackendDeviceId`

The implemented immutable identity for one opaque device token inside a
[`BackendId`](#backend-identity--backendid) namespace. It retains the exact caller-supplied backend
identity and nonblank backend-defined string references. Equal tokens under different backends are
unequal device identities; equal component values have ordinary composite record equality. The
record performs no normalization and proves no discovery, presence, availability, capability,
resource access, or device-handle ownership. It is a producer-domain identity, not a trace-local
device correlation; that trace translation remains planned.

### Backend identity / `BackendId`

The implemented immutable identity for a backend ownership domain. It retains the exact
caller-supplied nonblank string reference without trimming, case folding, Unicode normalization,
syntax validation, interning, or alias resolution, so examples such as `"cpu"`, `"metal"`, and
`"cuda"` do not form a closed vocabulary. Ordinary record equality compares the stored string
content. The value can name a compile-time owner but does not register, discover, locate, or prove
the availability or capability of a [concrete backend](#concrete-backend). It is not a
trace-local backend correlation; that trace translation remains planned.

### Backend ownership

The compile-time decision that assigns work to a
[`BackendId`](#backend-identity--backendid), such as an identity whose value is `"cpu"`, `"metal"`,
or `"cuda"`. The current package-private baseline selects one owner for one operation occurrence
from the complete provider-ordered [hard-eligible](#backend-hard-eligibility) identity list. With
no preferred [`DeviceClass`](#device-class--deviceclass), the first eligible identity wins. With a
preference, the first eligible identity whose equal-ID availability snapshot reports that class
wins; when none matches, the first eligible identity still wins. Provider order resolves ties, an
empty matching snapshot is a preference nonmatch, and the exact identity reference from hard
eligibility is returned. Empty hard eligibility fails internally before snapshot elements are
read.

This internal baseline is not a public planner API, numeric or cost-scoring model, device choice,
or same-owner partition. The broader planned ownership workflow may assign nodes or segments and
use additional backend-neutral costs. Ownership always answers “where should this work run?” It
does not answer “which kernel should run it?” The owning concrete backend makes that implementation
choice during [prepare](#prepare).

### Backend intent / `BackendIntent`

The implemented immutable config value that records whether later backend planning has one
[`BackendRequirement`](#backend-requirement--backendrequirement) as a hard eligibility target.
`BackendIntent.unconstrained()` contains an empty optional, while
`BackendIntent.requiring(requirement)` contains the exact supplied requirement reference. Direct
construction retains the exact non-null `Optional<BackendRequirement>` reference.

An unconstrained intent means only that no hard target filters later candidates. It does not mean
that a default backend exists, discovery occurs, fallback will succeed, or any backend is
available or capable. Backend intent does not evaluate the target, rank candidates, express
preference, contain planning-cost or model-autotuning data, select ownership, locate services, or
perform compile, prepare, run, publication, or execution behavior. The current separate
[`PartitionScoringConfig`](#partition-scoring-configuration--partitionscoringconfig) may retain a
soft coarse class preference, while planning-cost and model-autotuning inputs remain planned.
Current internal planning owns the per-query hard-eligibility intersection and its cost-free
preferred-class/provider-order owner selection. Later public planning orchestration and cost
scoring remain planned, including the public no-match failure contract.

### Backend requirement / `BackendRequirement`

The implemented sealed, method-free family for one hard backend eligibility target.
`BackendIdRequirement` targets later ownership by a `BackendId` equal to its retained component.
`BackendDeviceIdRequirement` targets a `BackendDeviceId` equal to its retained component and
therefore also that device identity's owning backend. `DeviceClassRequirement` accepts any later
eligible device whose `DeviceClass` equals its retained component; it identifies neither a
particular backend nor a particular device.

A backend requirement is requested data, not a capability claim, availability fact, preference,
fallback, score, or evaluator. The family has no absence sentinel and does not combine or match
targets. Current [`BackendIntent`](#backend-intent--backendintent) owns whether a requirement is
present. Current internal planning evaluates it with availability and capability facts, represents
no match with an empty immutable backend-identity list, and fails before baseline selection can
choose an owner. The later public no-match exception type and message are not yet defined.

### Backend-owned lowering

The rule that each concrete backend translates its assigned planned partitions into its own executable representation during prepare. Lowering, fusion, specialization, and kernel selection stay together because they depend on backend-specific execution and storage models. There is no shared lowering module. See [Runtime, Prepare, and Backend Boundary](architecture/runtime-prepare-backend-boundary.md).

### Backend route

A concrete implementation choice inside one backend, such as a CPU scalar loop, Vector API
routine, generated CPU computation kernel, OpenBLAS call, MPSGraph executable, custom Metal
kernel, or CUDA kernel. A route is selected during backend prepare after planning has chosen the
backend owner. Routes and provider libraries are not separate backend identities. CPU route
selection therefore retains one `BackendId("cpu")` and has no fixed vendor-priority list.

### CPU portable route

The bytecode-first Java Class-File API plus Vector API production baseline inside the single CPU
backend. It is the always-available semantic fallback for every operation occurrence it truthfully
supports, although a scalar reference remains available separately for conformance and fail-closed
checking. Metadata-only work may require no generated class. Task 0005A implemented the portable
scalar proving slice and `internal.route.portable`; completed CPU 0005C added preferred-species
FLOAT64 vector compute and explicit CPU-private parallel orchestration. Completed CPU 0005E now
uses that route for a bounded five-type pointwise matrix and connected one-to-eight-occurrence
chains. CPU 0005G completes the selected exact algebraic/logical increment through scalar or
parallel-scalar compute. CPU 0005H closes all nineteen same-typed FLOAT32/FLOAT64 unary kinds and
keeps the three floating classifications separate. CPU 0005I gives preferred-species FLOAT32 and
FLOAT64 parity to its original twenty-one eligible opcodes. Completed CPU 0005J adds the exact
floating extrema/clamp/ReLU/sign/cast, signed-integral arithmetic/extrema/cast, canonical-BOOL
logic/cast, and virtual floating-mask-to-WHERE vector rows; every other admitted row or ineligible
access pattern retains deterministic scalar or parallel-scalar fallback.

Completed CPU 0006 adds one static resolved-layout affine-copy family through the same portable
route. It composes bounded straight-line view chains during cold analysis, keeps eligible internal
views virtual, and materializes only the final boundary through scalar or parallel-scalar
represented-bit copying. Its seventh carrier form, `SHORT_ARRAY`, is limited to raw BFLOAT16
movement and does not add BFLOAT16 arithmetic, conversion, numerical meaning, or vector support.

Completed CPU 0006A adds exactly one fully static resolved-layout PAD, TILE, CONCAT, or STACK
occurrence through compact movement IR and cold geometry. It preserves one through sixteen
semantic composition occurrences while declaring repeated graph inputs once, requires one
distinct injective output, copies represented bits for all six Model data types, and uses scalar
or parallel-scalar execution. Vector preference falls back to scalar.

Cross-type cast, BFLOAT16/FLOAT16 numerical execution, relaxed math, native/vendor realization,
window/index/scatter/order/random families, functional updates, and dynamic layouts are not part
of this route increment. General partition-DAG decomposition and bounded vertical/horizontal
fusion remain Draft CPU 0008A work.

OpenBLAS is not another meaning of portable route. It is a narrow cross-platform native fallback
for eligible BLAS-compatible linear algebra. It is neither a universal fallback nor preferred over
generated code by library identity.

### CPU native peer route (planned)

An exact-capability native realization compared with the portable route by CPU Prepare using
whole-plan cost. Families include OpenBLAS for narrow BLAS-compatible work; Accelerate
BLAS/vDSP/vForce on Apple CPU; distinct oneMKL BLAS/VML and oneDNN routes on Intel; and distinct
AOCL-BLAS/AOCL-LibM and optional ZenDNN routes on AMD. Apple Silicon may use Accelerate, while
other ARM targets retain portable code generation unless a later task supplies an explicitly
verified provider. ARM itself does not name one native library.

Planning still selects one CPU backend owner. Common partition lowering, fusion legality and
profitability, canonical CPU IR, access plans, materialization accounting, numerical/determinism
checks, and representation planning remain route-independent. A provider adapter owns ABI and
provider-specific compatibility mechanics only; it does not reinterpret graphs, duplicate
broadcast/fusion planning, or own shared resource lifetimes.

### Generated CPU computation kernel

A CPU-backend-internal executable route whose Java Virtual Machine (JVM) bytecode is produced for
one selected lowering and specialization. CPU analysis owns the selection and exact resource
requirements; CPU finalization generates and defines the kernel or reuses a compatible CPU-owned
generated artifact after shared slot assignment. Runtime invokes only the resulting prepared
executable. The term describes the generated artifact and its lifecycle, not the bytecode-
generation library or builder API used by the current implementation.

The current package-private implementation uses the Java 26 Class-File API to emit and verify one
static typed entry method for the specialization's exact ordered primitive-array or
`MemorySegment` pattern, defines the result as a hidden nestmate class, and retains its exact
method handle. FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL heap boundaries use `double[]`,
`float[]`, opaque `short[]`, `int[]`, `long[]`, and canonical `byte[]` respectively when their
family admits that type. The
Class-File API and Java Vector API are CPU-internal implementation choices, not architecture
invariants. The production kernel executes one admitted bounded pointwise chain with one generated
scalar or exactly eligible typed preferred-species vector body. Floating bodies may retain
comparison/classification BOOL values as unit-private virtual masks through logical combination
and floating WHERE; materialized masks remain scalar. It
emits generation-time-selected primitive access state machines, direct array or native-order
`MemorySegment` vector access, unmasked complete vectors, and scalar remainders. Parallel
orchestration remains outside generated code. The family-level
`CpuVectorInstructionEmitter` owns the closed operation-to-vector-bytecode switch, while
`CpuVectorMath` owns only pure typed multi-instruction formulas, including the selected ERF and
GELU approximations. Class/loop/carrier/store/tail generation remains in the class generator.
When analysis selects one FLOAT64 contiguous input copy, the generated body consumes the adjusted
dense segment at that original boundary position; FLOAT32 vector parity does not broaden this
materialization policy. The static entry retains the lowering-derived boundary count.
An affine-copy class instead has exactly two typed carriers plus one cold-composed address table
and primitive `start`/`end` bounds. It transfers one payload per pair with scalar instructions;
parallel orchestration remains outside the class, and no Shape, layout, mapping, or vector
decision occurs in the generated loop.

### CPU kernel specialization

The implemented backend-private immutable description of every fact allowed to change one
generated CPU class. The current form includes schema 12, the canonical lowering fingerprint with
typed opcode sequence, exact scalar-immediate bits, exact ordered clamp-bound bits, and selected
scalar-power realizations, exact/default numerical mode, generated
scalar/vector compute form, exact preferred FLOAT32, FLOAT64, INT32, INT64, or BOOL species bit
size for vector compute,
and the complete ordered boundary data-type/carrier pattern. Schema 12 also distinguishes the
pointwise, affine, or movement IR family; affine mapping topology and write domain; movement
family, rank, occurrence map, and exact padding bits; and the seventh opaque BFLOAT16
`SHORT_ARRAY` carrier. The homogeneous boundary types
distinguish FLOAT32 from FLOAT64; no second lane-type field is retained.
The direct-versus-materialized input position and adjusted canonical consumer access form are also
structural when present.
Canonical IR separately supplies value kind, data type, ordered semantics/stores, iteration rank,
axis roles, contiguous-suffix form, and access regime. Their derived compatibility bytes and
structural identity are order-sensitive.

Concrete compatible extents, element count, layout offsets, effective stride values, starting
coordinates and addresses, carrier objects, slots, physical addresses, selected range count,
minimum chunk size, source/workspace objects, graph `ValueId`, declaration/slot identities,
materialization costs and expected runs, worker identity, run identity, cache state, and mutable
observations are excluded. Parallel-scalar reuses the scalar artifact and parallel-vector reuses
the vector artifact. The current hard budget permits four complete candidates, one realized
artifact, zero fixed-shape variants, and zero unrolled variants. A CPU kernel specialization is not
a capability claim, prepared executable, worker lifecycle, or persistent-cache record.

### CPU generated-kernel artifact

The implemented backend-private immutable identity object that retains one verified generated
class's bytes, specialization, full-privilege hidden-class lookup, hidden class, exact static
`MethodHandle`, and method type. Primitive arrays, exact `MemorySegment` values, and mixed ordered
signatures are direct entry carriers; the artifact does not copy, retain ownership of, or close
invocation memory.

Keeping the artifact reachable keeps its hidden-class state reachable, without promising when an
unreferenced class unloads. Direct generator calls produce equal class bytes but distinct hidden
classes and artifact identities. The current durable generated-kernel artifact store may instead
reuse compatible class bytes and weakly intern one loaded artifact while it remains live. The
artifact is not by itself a prepared route. Current schema-12 artifacts execute admitted bounded
forty-eight-opcode pointwise chains, one static affine represented-bit copy, or one static
PAD/TILE/CONCAT/STACK movement across the implemented carrier patterns. Schema 11 and older
artifacts are incompatible misses with no migration reader.

### CPU generated-kernel artifact store

The implemented package-private optional trusted-root mechanism for reusing deterministic
generated class bytes. With no root, realization is entirely in memory. With a root, one
current-schema envelope per key can supply bytes only when schema, structural key, complete
compatibility metadata—including carrier/materialization structure—lengths, checksum, class shape,
and entry descriptor verify. The envelope is bounded to 2 MiB, with metadata bounded to 64 KiB and
class bytes bounded to 1 MiB. A structural key alone never establishes compatibility.

Missing, corrupt, incompatible, oversized, malformed, or unusable entries are safe misses and
return to verified generation. Publication uses a forced temporary file and atomic move; failure
never changes correctness.
Compatible age does not invalidate an entry, and the mechanism has no automatic eviction, expiry,
quota, directory sweep, or background maintenance. Loaded artifacts are interned weakly by exact
structural key and remain usable through their strong callers.

Checksums and structural validation detect accidental damage but do not authenticate executable
content. The caller must isolate the root from untrusted writers. The store is neither a workload
tuning cache nor a model-plan cache: it reuses already-selected executable class bytes during cold
CPU finalization and performs no route selection, measurement, or Runtime hot-path work. A hit
still defines a fresh hidden class and preserves no Java Virtual Machine just-in-time machine code
or profile. The future workload tuning cache instead records compatible route/configuration
decisions. No public composition owner supplies a default root: the default remains off after the
CPU 0005D `KEEP_DISABLED` evidence verdict. The current CPU-private finalizer calls the store only
after shared assignment, and the resulting prepared executable strongly retains the loaded
artifact.

### CPU portable preparation plan

The implemented backend-private immutable result of route-neutral CPU analysis before shared slot
assignment. `CpuPartitionPreparationPlan` retains one fused execution unit, its canonical kernel
IR and portable specialization, exact/default compute and orchestration strategy, the derived
ordered boundary values and data-type/carrier pattern, normalized access bindings, exact per-boundary
declarations, compatible extents/count, selected range count, positive minimum range size, exact
vector species when applicable, optional selected materialization, original and adjusted generated
carrier patterns, optional exact workspace declaration, specialization budget, optional compact
movement geometry, and an optional cold lowering manifest.

CPU analysis validates that Planning selected CPU ownership, admits one connected straight-line
chain of one through eight supported pointwise occurrences or one exact static movement
occurrence, derives every family-specific Model
shape, normalizes resolved layouts, proves output write injectivity, derives external boundaries
and virtual single-use results, and compares direct plus at most three eligible one-input copy
candidates. It declares each boundary's referenced byte span and, when selected, appends workspace
ID `0` with exact dense byte geometry. Shared Prepare sees the plan opaquely and later validates
declarations against assigned `PreparedMemoryPlan` geometry.

The plan is not a generated artifact, assigned slot, executable, physical resource, per-run
binding, capability claim, tuning result, registry, or public route API. It fails closed before
artifact work for unsupported rows, disconnected or fan-out topology, multiple final outputs, or
more than eight occurrences.

### CPU portable prepared executable

The implemented backend-private immutable Runtime recipe constructed by CPU finalization after
shared slot assignment. `CpuPreparedExecutable` strongly retains the partition's one generated
artifact and exact static `MethodHandle`, the lowering-derived selected buffers, full normalized geometry, ordered
original/generated carrier patterns, optional materialization/workspace selection, optional
compact movement geometry, one half-open
logical range, selected range bounds, and an optional borrowed CPU worker group. It owns no
physical representation or worker close lifecycle and may bind
concurrently to distinct open `RunState` instances.

Cold binding consumes `CpuBorrowedBuffer` and `CpuNativeBuffer` through Runtime's
`BufferRepresentation` boundary. It validates selected carrier form, liveness/accessibility,
writability, exact size/alignment, canonical BOOL input bytes, and actual input/output accessed
spans once, then constructs a
signature-specific `BoundInvocation` with direct typed fields and primitive address state. No executable accepts
`HostTensorStorage` directly. The Runtime hot call performs no storage discovery, slot lookup,
argument classification, artifact access, reflection, handle adaptation, route selection, or
allocation. A selected copy reads the original source into the contiguous workspace once on the
invoking thread before any consumer chunk; copy failure prevents consumer execution. A bound
partition invocation performs the Runtime state guard. Movement binding packs the current
range-start coordinate, carrier bases, strides, and compact family facts without a per-element
table. It invokes one range inline or
synchronously joins deterministic disjoint chunks through the borrowed group; generated code
performs no scheduling. The current production families establish only the admitted bounded
pointwise, affine, and movement matrices; they make no Tensor-result, gradient, compiler,
conformance, integration, or performance claim.

### CPU execution unit

A backend-private computation-oriented group inside one CPU-owned planned
partition. CPU analysis forms units and applies legal fusion before declaring physical resources.
One unit has one canonical loop-oriented kernel intermediate representation and one selected
generated-Class-File, metadata-only, or peer native realization. A scalar reference realization
is reserved for conformance and fail-closed fallback, not Runtime IR interpretation. A partition
may contain multiple units, but Runtime receives one partition-level prepared executable rather
than one executable per graph node or unit.

The term is CPU-private lowering vocabulary. It is not a Planning `PlannedPartition`,
Runtime `PreparedUnit`, public backend SPI, operation family, registry entry, or shared Prepare
type. Shared Prepare sees only the resulting opaque backend plan and exact resource declarations.
Task 0005A creates only the portable route package; later native packages appear only with their
concrete Draft tasks.

### CPU kernel intermediate representation

The immutable CPU-private portable representation of one CPU execution unit. The sealed family
contains the established pointwise form, static affine-copy form, and static movement form. The
pointwise route-
independent canonical form records typed boundary and virtual values, one
forty-eight-opcode family-oriented pointwise vocabulary, exact typed scalar-immediate and ordered
two-bound clamp bits, selected scalar-power realizations, ordered computation semantics,
normalized access-plan form,
a universal primitive `start`/`end` loop
model, fusion form, output stores, and the numerical semantic version. It contains no compatible
concrete extents or element count by default, selected route/configuration, thread count, vector
species, cache root, generator/artifact compatibility setting, `Operation`, `CompiledNode`, graph
identity, Runtime slot, physical segment, method handle, or run state.

Generated and scalar-reference realizations consume the same canonical representation. Canonical
topology and semantics participate in generated-artifact identity; specialization separately adds
structural access regime, fusion form, data type, route/strategy/configuration, and compatibility
facts. Instance extents, offsets, carriers, slots, and segments participate in neither. A later
fixed-shape or unrolled variant may add extents only as an explicit evidence-selected specialization
under a class-count budget.

The affine form records the copied data type, exact structural read/write access forms, ordered
view-mapping kinds and ranks, and the logical-element or distinct-address write domain. Cold
lowering separately retains the exact composed source/result addresses. It contains no operation,
graph identity, concrete carrier, slot, address, worker, or Runtime state. Encoding it for the
current generator produces two boundary values, no computation instructions, one loop, and one
ordered store.

The movement form records PAD, TILE, CONCAT, or STACK identity, represented type, unique
structural input accesses, ordered occurrence-to-boundary mapping, one injective output access,
and exact padding bits. Concrete movement extents, offsets, strides, axes, repeats, and segment
prefixes remain compact instance geometry outside structural identity.

This is backend-internal state, not another Model, Compiler, Planning, Prepare, or Runtime IR.

### CPU access plan

The implemented normalized per-value address description shared by the current bounded CPU
pointwise, affine, and movement units. CPU 0005B right-aligns pointwise rank and separates the
structural regime/axis pattern used by generated-class identity from cold concrete extents, base
element offset, effective `long` strides, exact carrier objects, slots, and addresses. Different
inputs and the output of one fused unit may have different plans. The plan consumes current Model
`ShapeBroadcast` and `LayoutDescriptor` contracts rather than duplicating shape or layout
semantics.

The ordered heap-array/`MemorySegment` carrier access pattern is a separate code-shaping
structural specialization fact. Each generated class has exactly one direct static entry for its
actual ordered pattern, and that pattern participates in compatibility/class identity. Exact
carrier objects and byte offsets remain cold bindings. Equal patterns can reuse class bytes and
loaded identity across compatible extents; a different pattern may use a distinct specialization.
CPU analysis receives the backend-owned prepared pattern, finalization realizes the matching
artifact, and Runtime binding only validates matching concrete carriers. This avoids Runtime
generation and avoids placing every possible carrier combination in each class as boundary count
grows.

The current `CpuPartitionAnalysisInputs.DEFAULT` disables lowering-manifest retention and leaves
the explicit carrier list empty. Analysis interprets that empty list as one exact
`MEMORY_SEGMENT` access form per lowering-derived boundary. Explicit analysis inputs may instead
immutably select a non-null ordered typed heap/segment pattern. CPU analysis validates the snapshot
against boundary count, data type, and order; it carries no physical carrier object or general
Config policy.

The fully static scope includes scalars, rank expansion, singleton and multi-axis
broadcast, zero extents, offset-contiguous and positive-strided layouts, zero-stride views, heap
arrays, `MemorySegment`, and mixed carriers. Its ordered scalar regimes are dense linear,
scalar/all-zero stride, last-axis bias, block/outer with a contiguous inner loop, and a complete
general positive-strided odometer fallback. The five regimes emit distinct state machines: one
linear increment, a constant address, a final-axis counter/reset, a contiguous-inner plus outer
carry/reset, or a full-axis odometer. Generated hot code uses primitive address/carry
arithmetic without per-element cursor allocation, semantic dispatch, division, or modulo.
Completed CPU 0005J vectorizes direct contiguous runs and scalar broadcasts for exactly eligible
FLOAT32, FLOAT64, INT32, INT64, or canonical-BOOL IR and admits narrowly virtual floating masks
when every non-scalar boundary has a complete preferred-species run. It uses the scalar body for
tails and selects scalar fallback for a general odometer, a too-short run, or any ineligible instruction.
It implements no gather or masked tail. Completed CPU 0005D can replace one eligible FLOAT64
input with canonical dense workspace access; FLOAT32 vector parity does not broaden that policy,
and complete access semantics do not promise universal vectorization.

Cold bindings retain output extents, base offset, effective strides, start coordinates/address,
and the exact half-open accessed span for their selected logical range. Output normalization uses
a complete bounded static injectivity decision: interleaved positive strides are accepted when
distinct coordinates remain distinct, while actual repeated writes are rejected. Declaration
geometry remains each layout's referenced span, not the iteration element count.

Static affine lowering also derives a compact structural read/write plan over its cold-composed
address pairs. Exact source and final-result declaration geometry remains the respective resolved
layout's referenced span, including offsets, holes, positive strides, and zero strides. A
non-injective result uses a deduplicated distinct-address domain only when every repeated logical
coordinate selects the same represented source value.

Current Prepare has no exact dynamic binding and rejects non-static projected shapes. CPU
therefore fails closed for dynamic or symbolic dimensions. Supporting them requires a future
explicit exact-binding contract rather than a CPU assumption. Broadcast gradients remain
Compiler/Model `SUM_TO_SHAPE` semantics plus later CPU reduction coverage.

### CPU contiguous materialization plan

The implemented CPU-private immutable decision to copy at most one eligible FLOAT64 read boundary
of the current pointwise unit into run-owned contiguous workspace. `CpuMaterializationPlan` retains the
original source boundary identity and normalized binding, a canonical dense consumer binding,
checked element/byte geometry, analysis-local workspace ID `0`, derived lowered-unit use count,
expected runs, dimensionless direct/copy/contiguous estimates, positive net benefit, basis-point
benefit, and a diagnostic selection reason.

Analysis enumerates direct plus at most the first three eligible reads in derived boundary order;
direct wins ties. It rejects
scalar/all-zero and already-dense sources, unused values, excess additional memory, non-beneficial
costs, and ineligible consumer forms. A selected plan appends one workspace declaration after all
derived graph-value buffers. Shared Prepare assigns that workspace opaquely, and CPU finalization
verifies it before artifact access. The copy completes once per bound invocation before consumer
work. This plan does not add a graph value, mutate `LogicalMemoryPlan`, measure during prepare,
select a future tuning-cache entry, or authorize more than one copy.

### CPU portable execution strategy

One of exactly `scalar`, `vector`, `parallel-scalar`, or `parallel-vector`. Scalar/vector is the
compute axis and single-thread/parallel is the orchestration axis. CPU analysis selects among all
four before shared assignment. It uses the exact Java 26 preferred FLOAT64, FLOAT32, INT32, INT64,
or ByteVector species for an eligible lane type. All use unmasked complete vectors plus scalar
tails and scalar fallback for general odometers or short runs. Generated kernels accept primitive `start` and
`end`; worker orchestration remains outside generated code.

The floating value set is the CPU 0005I twenty-one-opcode set plus binary/scalar extrema, ordered
clamp, ReLU, sign, and same-type cast. The INT32/INT64 set is binary/scalar ADD/SUB/MUL/MIN/MAX
plus same-type cast. The canonical BOOL set is AND/OR/NOT plus same-type cast. Matching virtual
floating comparison/classification masks may be combined logically and consumed by floating
WHERE; a scalar/all-zero external BOOL condition is the only external mask form that vectorizes.
Scalar power is eligible only for positive-one, identity, square, and reciprocal plans; direct
power remains scalar. Materialized masks, non-scalar external WHERE conditions, mixed vector
types, and any chain containing another ineligible instruction select scalar compute without
losing semantic support or splitting the chain. The Java 26 Vector API may be scalarized by the
JVM and makes no hardware-intrinsic or performance promise.

Floating vector extrema preserve NaN propagation and directional signed zero; clamp is lower
maximum followed by upper minimum; ReLU is maximum with positive zero; and sign preserves signed
zero and NaN while mapping other values to exact signed one. Integral arithmetic remains
fixed-width modular and integral extrema use signed order. External BOOL bytes are validated as
canonical zero or one, and vector NOT masks its bitwise complement back to those values rather
than materializing `0xFF`.

Configured and available parallelism are positive immutable CPU-private snapshots. Analysis also
uses a positive minimum elements per worker, while cold binding forms deterministic contiguous,
non-empty, ascending chunks for the requested range. Zero work submits nothing and one chunk runs
inline. Two or more chunks borrow an explicit caller-owned [CPU worker
group](#cpu-worker-group--cpuworkergroup). These facts are prepared/cold state rather than
generated-class identity, Config policy, Runtime selection, tuning evidence, or a performance
claim.

### CPU worker group / `CpuWorkerGroup`

The implemented CPU-private caller-owned lifetime for a fixed positive number of named daemon
platform workers. Composition or tests create and close the group. A CPU finalizer verifies and
borrows it for a parallel plan; the prepared executable and bound invocation never close it. The
group is not a public executor facade, common pool, virtual-thread scheduler, Runtime service,
global singleton, or shared `PreparedExecution` lifecycle.

Multi-chunk submissions complete synchronously. Detected failure or close cancels unclaimed work,
started work quiesces, and failure selection is deterministic by ascending range index. Joining
interruption restores the submitter's interrupt status and reports a CPU-private coordination
failure. Close is thread-safe and idempotent, rejects new work, joins started work, and returns
only after every owned worker terminates. Owned workers may neither submit nested multi-chunk work
nor close their group. No write rollback or performance guarantee is implied.

### CPU generated-class persistence policy

An optional cold-path policy for compatible deterministic class bytes. With no configured trusted
artifact root, CPU emits, verifies, and defines in memory. With a root, a verified compatible hit
may avoid emission, but it is still defined in the current JVM and cannot preserve JIT machine
code or profiling. Missing, corrupt, incompatible, oversized, or inaccessible storage safely
returns to in-memory emission.

Persistence is neither correctness-critical nor Runtime state. CPU 0005D's fixed opt-in evidence
suite recorded `KEEP_DISABLED`, so the default remains persistence-free; an explicit trusted root
enables only bounded verified lookup/store behavior. Class-byte reuse is distinct from JVM JIT
machine code/profile and from the future workload tuning cache of route/configuration decisions.
Deterministic structural identity and prepared-execution strong artifact ownership remain
mandatory, and there is no reader for earlier schemas.

### CPU affine boundary copy

The implemented CPU-private materialization of one bounded static resolved-layout affine view
chain. Analysis composes one through eight admitted one-input/one-output occurrences into exact
source/result element-address pairs. Eligible internal results remain graph and logical-memory
values without CPU declarations or Runtime slots. The final result always has a distinct declared
buffer because current shared preparation has no cross-value alias assignment.

The copy preserves represented bits for all six current Model data types. FLOAT64/FLOAT32 NaN
payloads and signed zeros, raw BFLOAT16 16-bit payloads, signed integral patterns, and canonical
BOOL bytes are transferred without conversion or arithmetic. An injective result has one pair per
logical coordinate; a zero-stride/non-injective result has one pair per distinct address after
lowering proves repeated coordinates agree. Scalar results have one pair and zero-element results
have none.

This term does not mean that Model view construction attaches shared storage. It does not include
dynamic layouts, BFLOAT16 numerical behavior, Vector API affine access, static non-affine movement,
index-tensor work, scatter/fold, ordering, random generation, general partition-DAG fusion, or a
performance claim.

### CPU static data movement

The implemented CPU-private represented-bit realization of exactly one fully static,
resolved-layout PAD, TILE, CONCAT, or STACK occurrence. It retains one through sixteen semantic
input occurrences in order while declaring each distinct graph input once in first-occurrence
order. The final output has one distinct declared representation and an injective layout.

Canonical movement IR contains family, rank, represented type, unique structural accesses,
composition occurrence mapping, one output store, and exact PAD bits. Compact cold geometry owns
concrete extents, layout offsets and strides, normalized axis, padding widths, repeats, segment
prefixes, and arbitrary-range initial state. Generated scalar loops advance that state directly;
they retain no per-output-element table and perform no per-element division or modulo. Parallel-
scalar orchestration is permitted over disjoint output ranges, while vector preference falls back
to scalar.

This term does not include Model expression construction, affine view folding, window extraction,
index tensors, functional updates, scatter/fold, dynamic Shape binding, native routing, general
partition-DAG fusion, or a performance claim.

### CPU virtual intermediate

A graph value whose producer and every consumer are inside one fused CPU execution unit and which
has no publication or other unit/partition boundary obligation. It remains present in the
compiled graph and `LogicalMemoryPlan`, but CPU analysis emits no buffer declaration for it, so
shared Prepare assigns no physical slot. A pointwise virtual remains a typed IR value; an affine
view intermediate is folded into the composed mapping and has no computation instruction, store,
or affine boundary of its own.

Virtuality is decided only by CPU analysis after legal fusion. Shared Prepare does not infer it or
require one buffer per projected value. A fan-out, publication, cross-partition consumer, or
execution-unit split makes the value a materialized boundary instead.

### Benchmark report / benchmarking

**Benchmarking** is observational execution of fixed reproducible operation, operation-family,
model, or end-to-end workloads to compare commits, models, or environments. A **benchmark report**
is the rich immutable evidence from one recorded run: workload and environment identity, supplied
configuration, lifecycle boundary, samples or distributions, and summary statistics. Benchmarking
never selects or mutates production settings and never substitutes for correctness tests. The
report contract and harness remain planned in `tools/benchmarks`.

### Backward graph

The graph nodes produced by capturing gradient Tensor expressions derived from an original
forward Tensor expression. In backward-capable compile modes, the compiler captures forward
outputs and gradient roots together once, then validates and optimizes the immutable combined
graph before planning. See [Training graph](architecture/training-graph.md).

### Batch dimension

A leading Shape dimension that identifies independent groups of the same logical operation rather
than one operation's elementwise or matrix coordinates. For matrix multiplication, every axis
before the final two matrix axes is a batch dimension; rank-one operands have no batch prefix.
The two prefixes broadcast right-aligned before row and column result axes are appended. A batch
dimension is logical Shape metadata, not a runtime batch scheduler, storage partition, or backend
execution unit. See [Matrix-multiplication expressions](api/tensor-api.md#matrix-multiplication-expressions).

### Scaled dot-product attention

An implemented backend-independent operation meaning that compares query rows with key rows,
softmax-normalizes the resulting eligible [attention scores](#attention-score), and uses those
weights to combine value rows. For query `[..., L, E]`, key `[..., S, E]`, and value
`[..., S, Ev]`, its score Shape is `[..., L, S]` and output Shape is `[..., L, Ev]` after
right-aligned three-way batch broadcasting. An absent scale means `1 / sqrt(E)` after positive
embedding binding; a present scale is an exact finite positive floating `ScalarValue`.

Current public Tensor construction records ordered inputs `[query, key, value]` or
`[query, key, value, mask]`. The original method family records output slot zero only and creates
no hidden weights. The explicit `WithWeights` family returns output slot zero and normalized
[attention weights](#attention-weight) slot one from one exact shared producer. Neither form
applies dropout, evaluates numbers, creates Model-owned gradients, proves deferred constraints,
chooses a backend, or executes. Current package-private compiler autograd supports query, key, and
value from slot zero and query and key from canonical weights slot one only for the exact
two-output occurrence. The one-output occurrence fails closed, and a BOOL mask remains
non-differentiable.

### Attention score

One scaled dot product between a query row and a key row before softmax normalization. In scaled
dot-product attention, scores have Shape `[..., L, S]`, and each query row normalizes over the
final key-sequence axis `S`. Masked-out positions are excluded before their score or value special
values participate.

### Attention weight

One normalized, post-mask coefficient produced from an eligible
[attention score](#attention-score) along the final key-sequence axis. In scaled dot-product
attention, weights have Shape `[..., L, S]` and the promoted query/key/value data type. Excluded
positions, no-eligible rows, and all-eligible-negative-infinity rows have positive-zero weights;
eligible NaN produces NaN weights; eligible positive-infinity ties divide unit weight equally;
otherwise eligible finite weights follow stable softmax and total one ideally.

`ScaledDotProductAttentionResult.weights()` exposes these exact same-occurrence coefficients at
producer slot one. Its eligibility metadata is query OR key, unlike output slot zero's query OR
key OR value. This output is model metadata and semantic meaning, not evidence that weights were
evaluated, captured by a compiler, retained for gradients, materialized, or executed.

### Causal mask

Eligibility that prevents a query position from attending to a later key position. Synaptik's
current attention semantics are top-left aligned: zero-based key position `j` is eligible for
query position `i` exactly when `j <= i`, including rectangular query/key sequence lengths. A
separate explicit BOOL mask combines with causal eligibility by logical AND.

### Categorical cross entropy

A loss meaning that compares a logits slice with either a dense target distribution or one index
target. Synaptik represents both directly from logits with a stable negative-log-softmax meaning;
it does not first create a probability Tensor. The current model meaning includes an explicit
class axis and loss reduction. Current package-private compiler autograd supports logits and dense
target roles; index targets remain non-differentiable, while their logits role requires a positive
static class depth. Lowering, bounds enforcement, and numerical execution remain owned by later
lifecycle layers.

### Class axis

The logits axis whose positions represent alternative classes in a categorical loss. Removing it
leaves the [sample domain](#sample-domain). Callers may supply a positive or negative axis; current
Tensor construction normalizes it once to a non-negative attributes value.

### All-masked row

One attention query row with no eligible key positions after explicit and causal masking. The
scaled-dot-product-attention contract assigns positive-zero weights and positive-zero output
components to such a row rather than applying ordinary softmax to an all-negative-infinity
sentinel row. An empty key sequence creates this case for every existing query row.

### Broadcasting

A shape rule that combines compatible inputs by aligning axes from the right and expanding static
singleton dimensions as needed. In the implemented local shape model, equal dimensions remain
equal and size `1` expands to the opposing dimension. Named dimensions are equal by canonical
symbol, exact expression dimensions are equal structurally, and constrained unknowns are equal
only when the same unknown is reused. Other symbolic pairings remain unprovable locally.
Broadcasting describes logical repetition; a resolved layout may represent that repetition with
a zero element stride. Directional [`Tensor.expand`](#reshape-and-expand-semantics) is a separate
one-input operation. It may retain an exact target when an aligned unresolved pair has the later
source-one-or-source-equal requirement; this does not broaden the conservative local
multi-input-broadcasting rule above.

### Bounded replay

A reproducibility promise limited to the same conforming prepared implementation path and
configuration. Equal graph RNG key/counter state and equal consuming-operation inputs request
equal results within that boundary. It is not a cross-backend or cross-version bitstream promise,
because no portable pseudorandom-number-generator algorithm is currently selected.

### Bindable input

A structural graph input that a later caller or run is allowed to supply. In the current internal
compiler boundary, bindability is derived from the ordered `CompiledGraphModel.inputs()` list:
inputs absent from the immutable compile-time constant sidecar are bindable, while inputs with
logical-splat facts are fixed [compile-time constant sources](#compile-time-constant-source).

The classification is logical and immutable for that candidate graph. It does not read Tensor
storage, bind a value now, allocate a buffer, define a public input API, or implement runtime
binding. Sidecar-aware dead-code elimination preserves every bindable input even when unused.

### Cast expression

An implemented public `Tensor.cast(targetDataType)` request that records elementwise conversion
metadata without converting a value. Every current source/target data-type pair is representable.
The result is a fresh unlabeled, storage-free [`Tensor`](#tensor), including when source and target
types are equal. It retains the input descriptor's exact [`Shape`](#shape), leaves layout
unresolved, and records `CastKind.CAST`, `CastAttrs(targetDataType)`, and the exact input as its
sole [provenance](#provenance) reference.

Gradient eligibility survives only when it was already requested and both source and target are
floating. That descriptor fact is not a gradient rule or backend differentiability promise.
Numerical conversion, redundant-cast and cast-chain canonicalization, autograd expansion, backend
support, and execution belong to later owning layers. A cast expression is therefore not converted
storage, a compiler graph node, or proof that a requested conversion can execute.

### Canonical workload signature

The compatibility key for one local tunable workload. It includes operation semantics and
attributes, input and output data types, shapes, layouts, relevant policies, and target
compatibility. Identical signatures can reuse one workload tuning result across occurrences and
models. Operation family only selects the candidate generator and is not itself a universal cache
key. The contract remains planned.

### Candidate generator

A typed, version-controlled, tested backend or lifecycle-owner component that derives and prunes
complete valid tuning candidates from target capabilities, workload facts, and a tuning budget.
Concrete backend generators are colocated with their routes and own private configuration
vocabulary. Shared tuning and prepare orchestration sees candidates opaquely; it does not use a
generic parameter map, string dispatch, reflective annotations, or a central knob registry. The
contract remains planned.

### Compile

The lifecycle stage that optionally builds gradient Tensor expressions, captures one forward-only
or combined graph, validates and transforms immutable graph semantics, assigns backend ownership,
and creates immutable compile artifacts. Compile creates a logical recipe; it does not allocate
physical buffers, choose concrete kernels, or create backend executables. See
[Lifecycle](architecture/lifecycle.md).

The current package-private compiler implements the lifecycle through a validated forward-only or
combined one/two-stage functional-derivative graph followed by publication roles, per-node owner
selection, maximal partitions, logical memory, constants, diagnostics, derivative-order metadata,
and immutable `CompileArtifacts`. Public compilation, publication delivery, preparation, and
execution remain planned.

### Compile artifacts

The current public immutable eight-component `CompileArtifacts` record returned by the
package-private complete compiler entry. It contains the exact compile mode and final
`CompiledGraphModel`, an immutable membership snapshot of maximal backend-owned partitions, the
derived `LogicalMemoryPlan`, and exact `PublicationPlan`, `CompileConstantPlan`, and
`CompileDiagnostics`, and `DerivativeGraphMetadata` references.

The constructor cross-validates graph identity, mode and phase roles, maximal graph-order
partitioning, logical memory, complete graph-input source roles, constant type and gradient
eligibility, and diagnostic node membership. Compile artifacts are a recipe for later prepare
work, not executable state. They contain no provider, availability snapshot, selected device,
physical buffer, concrete route or kernel, transfer, schedule, executable, runtime residency, or
mutable run state. The artifact type is public, but no public compile entry currently produces it.

### Compile constant plan

The current compiler-owned output-only classification of every final graph input as either a
[bindable input](#bindable-input) or one exact
[compile-time constant source](#compile-time-constant-source). `CompileConstantPlan` snapshots
both ordered role lists while retaining exact immutable element references.
`ConstantSource(ValueId, ScalarValue)` carries the exact input identity and exact typed scalar
repeated at every logical coordinate. It is not a Tensor, Shape, dense payload, storage object,
materialization instruction, backend value, or physical allocation.

### Compile diagnostics

The current compiler-owned output-only successful-compile diagnostic bundle.
`CompileDiagnostics` exposes ordered immutable
`DeferredConstraintDiagnostic(NodeId, subject, predicate)` projections from the final deferred
graph constraints and privately retains the exact internal constraints for later compiler-owned
binding validation. Subject and predicate text are deterministic and nonblank, but they are not a
public predicate language, binding API, trace payload, warning taxonomy, rejection result, or
serialization. A failed compilation returns no partial artifact or successful diagnostic bundle.

### Compile-time constant source

A current package-private compiler classification for a fixed logical graph source. Structurally,
the source remains an ordinary `CompiledGraphModel` input, satisfying the existing graph-closure
rules. An immutable sidecar associates its `ValueId` with one [logical splat](#logical-splat), and
the source is therefore excluded from the derived [bindable inputs](#bindable-input).

Initial facts enter only through explicit exact Tensor-identity bindings for reachable
provenance-free, non-gradient leaves. The compiler does not infer them from a Tensor factory,
descriptor, label, layout, provenance absence, or mutable host storage. Folding may create a
synthetic constant source deterministically. The current complete compiler entry transports the
facts into `CompileConstantPlan`; future prepare/backend work owns physical materialization. The current
classification is not a public binding API, dense payload, physical buffer, runtime value, or
executable constant.

### Compile mode / `CompileMode`

The implemented standalone config enum that records requested compile-time graph scope. Its exact
values, in declaration order, are `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and `TRAINING_STEP`.
Forward-only requests forward graph construction and requested forward publications. Forward-and-
backward requests compiler autograd expansion plus combined forward/backward compile-time graph
work. Package-private `GraphCompiler` currently consumes these two modes and `TRAINING_STEP` for
both graph-stage and complete artifact compilation; the latter performs the same internal
first-order combined graph construction without adding an optimizer, optimizer-update graph,
training session, publication delivery, schedule, or execution behavior.

A compile mode is a request, not completed public compiler behavior. No current public aggregate
or compiler entry point consumes it, and constructing a value does not capture or transform a
graph, construct gradients, bind publications, prepare an executable, or run a computation. It
differs from
[`GraphPhase`](#graph-phase), which classifies a compiled node's forward or backward origin.

### Contiguous request

The implemented parameterless `ContiguousKind.CONTIGUOUS` operation meaning. It describes one
logical input and requests logically equivalent canonical dense row-major result geometry with
logical storage offset zero. Logical values, Shape, DataType, and row-major element order are
preserved. The kind composes with `Operation` and `NoOperationAttrs.INSTANCE`; its signature
enforces that pairing and declares one input and one output.

A contiguous request is not the implemented `LayoutKind.DENSE_CONTIGUOUS` classification: the
request states desired computation semantics, while the layout kind classifies already-resolved
strides and zero offset for a static Shape. It is also not a materialization decision. Public
`Tensor.contiguous()` constructs a fresh expression that preserves exact Shape, DataType, and
gradient eligibility. Fully static results receive newly resolved canonical layout geometry;
dynamic results remain unresolved. Construction records exact one-input provenance without
inspecting input geometry or storage and without choosing an alias or copy, allocating storage,
or defining compiler, planning, prepare, backend, runtime, gradient, or execution behavior. See
[Contiguous expressions](api/tensor-api.md#contiguous-expressions), [Contiguous semantic
kind](api/tensor-api.md#contiguous-semantic-kind), [Layout](#layout), and
[Materialization](#materialization).

### Cotangent

The reverse-mode sensitivity that flows backward from a selected output toward an earlier Tensor.
Current package-private autograd accepts exact-Shape explicit cotangents or supplies an exact typed
scalar unit for an eligible scalar output, combines multiple output and route contributions with
ordinary `Tensor.add`, normalizes each selected contribution to its input contract, and returns
the accumulated cotangent for each requested target. A cotangent here is a temporary compiler
Tensor expression, not mutable Tensor gradient state, a runtime tape value, or an optimizer slot.

### Cotangent normalization

The current first-order compiler step that makes one generated contribution match its selected
input's exact Shape and floating data type before deterministic accumulation. For the selected
0004B binary, WHERE-branch, and MATMUL roles, normalization first calls ordinary
`sumToShape` when broadcast reversal is required and then calls ordinary `cast` exactly when the
result type differs from the input type. A same-Shape or same-type row adds no redundant
operation.

Cotangent normalization is not a second algebra or a special conversion policy. Its SUM and CAST
occurrences enter the same combined capture, model inference, validation, numerical-semantics, and
exact optimization pipeline as forward expressions.

### Compiled graph / `CompiledGraphModel`

The implemented immutable compile-time graph representation. `CompiledGraphModel` stores ordered
`GraphValue` values, topologically ordered `CompiledNode` nodes, ordered input and output
`ValueId` boundaries, and an exact `NodeId`-to-[`GraphPhase`](#graph-phase) mapping. Construction
validates structural closure, producer rules, topology, and phase coverage and then stores
immutable collection snapshots without derived indexes. It performs no graph capture, compiler
transformation, planning, preparation, or execution. The planned public engine facade named
`CompiledGraph` will expose compiled artifacts and lifecycle orchestration; it is not the graph
model. See the [Compile API](api/compile-api.md#current-model-contracts).

### Concrete backend

A module that implements a backend, such as `backends/cpu`, `backends/metal`, or `backends/cuda`.
It owns backend-specific capability reporting, prepare-time lowering, fusion, specialization,
kernel selection, executable units, storage, workspaces, and native integration. Concrete
backends do not own public tensor semantics or global graph compilation. The CPU module is the
first concrete implementation and currently exposes only its exact fully static FLOAT64
`ADD -> exact GELU -> MUL` partition route over resolved right-broadcastable access plans;
other backend identities still contain no concrete backend behavior. See [Module
boundaries](architecture/module-boundaries.md).

### Cumulative scan

An ordered one-input operation that cumulatively combines values along one axis while preserving
one output position for every input position. The implemented `CumulativeScanKind` contains
`CUM_SUM` for addition and `CUM_PROD` for multiplication.
`CumulativeScanAttrs(axis, exclusive, reverse)` carries an already normalized non-negative axis
plus the two independent mode choices.

For logical input `[1, 2, 3]`, inclusive forward produces `[1, 3, 6]` from `1`, `1 + 2`, and
`1 + 2 + 3`. Exclusive forward produces `[0, 1, 3]` from the empty prefix, `1`, and `1 + 2`.
Inclusive reverse produces `[6, 5, 3]` from `1 + 2 + 3`, `2 + 3`, and `3`. Exclusive reverse
produces `[5, 3, 0]` from `2 + 3`, `3`, and the empty reverse prefix. Reverse changes traversal
direction, not output order.

Public `Tensor.cumSum(axis)` selects inclusive forward mode, and the complete overload preserves
explicit exclusive and reverse flags. Both validate numeric input, normalize one caller axis,
retain exact Shape/type/gradient-eligibility metadata with unresolved layout, and record exact
one-input provenance in a fresh unlabeled, storage-free Tensor. They calculate none of the example
values.

Public `Tensor.cumProd` has the same two forms and metadata behavior. For `[2, 3, 4]`, its
inclusive-forward, exclusive-forward, inclusive-reverse, and exclusive-reverse meanings are
`[2, 6, 24]`, `[1, 2, 6]`, `[24, 12, 4]`, and `[12, 4, 1]`. Positive one is the exclusive
boundary identity. A zero-length axis has no result position and therefore emits no identity.
Integral product uses exact-width two's-complement modular multiplication. Floating product
propagates NaN, makes zero times infinity NaN, and follows multiplication parity for zero and
infinity signs. Current package-private first-order autograd supports floating `CUM_SUM` and
`CUM_PROD`. `CUM_SUM` preserves exclusivity and reverses direction. `CUM_PROD` uses safe products,
cumulative zero counts, and an opposite-direction cumulative sum rather than dividing by the
original input. This is compiler expression construction, not a forward algorithm, intermediate
rounding rule, NaN-payload promise, bitwise backend result, runtime behavior, backend support, or
execution. See
[Cumulative-scan semantic kinds and attributes](api/tensor-api.md#cumulative-scan-semantic-kinds-and-attributes).

### Dropout / inverted dropout

A training-only stochastic operation that drops a requested fraction of logical input positions.
Current `Tensor.dropout(probability, state)` represents inverted dropout: a kept value is ideally
scaled by `1 / (1 - probability)`, while a dropped value becomes positive zero. It consumes
explicit graph RNG state and returns the next state; construction itself performs no draw,
scaling, masking, or execution. There is no model-level training flag or inference rewrite.

### Contraction dimension

The shared input extent over which pairwise products are summed by a contraction operation. In
current `MATMUL`, it is the final left axis and either the final right axis for a vector or the
penultimate right axis otherwise. Equal static extents are required; equality involving an
unresolved extent is deferred because the contraction extent is absent from the output Shape.
The term describes mathematical and Shape meaning, not an accumulation loop or kernel choice.
See [Matrix-multiplication expressions](api/tensor-api.md#matrix-multiplication-expressions).

### Convolution / Conv2d

Current `Tensor.conv2d` records grouped two-dimensional cross-correlation in NCHW order: batch,
channel, height, width. Cross-correlation uses the weight kernel in stored order; it does not
reverse the kernel as mathematical convolution does. Input Shape is `[N, C_in, H, W]`, weight is
`[C_out, C_in/groups, K_h, K_w]`, optional bias is `[C_out]`, and result is
`[N, C_out, H_out, W_out]`. A group selects one contiguous input-channel subset and its associated
output channels. Dilation is the positive spacing between stored kernel positions; the effective
kernel extent is `dilation * (kernel - 1) + 1`. Symmetric conceptual positive-zero padding is
applied on both sides of each spatial axis. Current model construction derives metadata and
provenance only. Package-private compiler capture and grouped input, weight, and optional bias
first-order formulas through public unfold/matrix/fold expressions are current; concrete binding,
decomposition, backend algorithms, and execution remain planned. See
[Tensor API](api/tensor-api.md#grouped-nchw-conv2d-expressions).

### Data type / `DataType`

The logical kind of scalar stored in each element of a tensor, such as `FLOAT32`, `INT64`, or `BOOL`. `DataType` records model-level facts including category and width, which lets model and compiler code interpret values consistently. It does not claim that every backend supports the type, prescribe a physical allocation alignment, or select a conversion route. See [Data types](api/tensor-api.md#data-types).

### Numeric promotion

The implemented backend-independent rule that selects a common operation data type for two
operands from the same numeric category. Floating pairs use
`BFLOAT16 < FLOAT32 < FLOAT64`; signed-integral pairs use `INT32 < INT64`. When an integral pair
promotes to INT64, the INT32 operand participates by conceptual sign extension. Same-width pairs
retain their type. BOOL and mixed floating/integral pairs are rejected, so a caller records an
explicit [`cast`](#cast-expression) when cross-category conversion is intended.

Promotion selects expression metadata only. It does not insert a cast producer, convert or read
values, capture a graph, choose a backend, or execute. See
[numeric promotion](api/tensor-api.md#numeric-promotion).

### Numerical permission (planned)

The future small backend-neutral Prepare configuration that tells candidate selection whether the
caller permits behavior outside an operation's ordinary exact/default conformance contract and
states the required determinism. Its default grants no relaxed or fast-math behavior. It names no
backend, provider, route, kernel, approximation algorithm, compiler pass, or operation-specific
rewrite.

Permission is an input, not a fact inferred from hardware, provider availability, workload size,
a performance objective, tuning, or a benchmark. Those mechanisms may compare only candidates
already eligible under the caller's permission and the semantic operation contract. Forward and
compiler-generated gradient operations share this policy; there is no gradient-specific numerical
permission.

### Scalar-power realization

The implemented CPU-prepare choice that retains semantic scalar `POW` in the compiled graph while
selecting one exact/default realization from the exponent's exact FLOAT32 or FLOAT64 bits. Common
route-independent CPU analysis owns exponent classification and the selected plan; emitters only
consume it.

Positive and negative zero select exact positive one, positive one selects identity, positive two
selects one typed multiply, and negative one selects one typed division of positive one by the
base. Every other finite value, infinity, and NaN uses direct power. In particular, reciprocal
power remains semantic `SCALAR_POW(-1)` rather than Tensor/Tensor or Tensor/scalar DIV, even though
the low-level realization performs division. Multiply chains and exponentiation by squaring are
not current plans because intermediate rounding, overflow, underflow, and subnormal transitions
lack a universal exact/default proof.

The current Compiler already bypasses typed scalar `POW(+1)` under task 0003A's exact guards. A
future graph-level `POW(0)` identity needs a typed shape-correct logical one-splat and a complete
exceptional-value, constant-sidecar, output/publication, phase/autograd, and descriptor proof.
`POW(0.5)` is not silently `SQRT`, because domain, signed-zero, exceptional-value, and rounding
behavior can differ. Current same-typed FLOAT32/FLOAT64 Tensor/Tensor power is instead always a
direct `StrictMath.pow` realization and is never inferred from Tensor storage or factory history.
The scalar semantic opcode, exact exponent bits, selected realization, and numerical mode
participate in generated-artifact compatibility and the cold lowering manifest under generator
schema 10, with no Runtime hot-path lookup.

### Typed scalar value / `ScalarValue`

An implemented immutable backend-independent operation parameter containing one exact
[`DataType`](#data-type--datatype) and that type's primitive bits. FLOAT64 and FLOAT32 retain raw
signed-zero and NaN payload bits, raw BFLOAT16 retains all 16 supplied bits, signed INT32 and
INT64 retain exact two's-complement values, and BOOL is canonical false or true. Equality and
hashing include both type and bits.

A typed scalar value is not a rank-zero [Tensor](#tensor): it has no Shape, identity, layout,
storage, provenance, or executable state. Its inspectors require the exact stored type and do not
convert. Scalar/clamp and padding attributes retain these values, while receiver-aware public
Tensor expression construction enforces exact value/input type equality. Existing primitive
TensorFactory scalar/full methods remain separate eager Tensor-construction APIs. See [exact typed
scalar values](api/tensor-api.md#exact-typed-scalar-values).

### Data-transfer object / DTO

A value whose purpose is to carry structured data across a boundary without owning the behavior
that produced it. Synaptik's implemented trace-event foundation uses typed DTO contracts so later
diagnostic producers and consumers can exchange explicit fields without importing compiler,
runtime, or backend business objects. Concrete lifecycle payload DTOs remain planned.

### Device class / `DeviceClass`

The implemented coarse declarative category of a device. `CPU` means a device that executes
through a general-purpose central processing unit; `ACCELERATOR` means a non-CPU compute device
intended for offloaded computation. The category does not distinguish graphics processing units,
neural processing units, vendors, memory topologies, or backend-internal routes. It is separate
from [`BackendId`](#backend-identity--backendid), which names an ownership domain, and
[`BackendDeviceId`](#backend-device-identity--backenddeviceid), which names one device inside that
domain. A current
[`BackendAvailabilitySnapshot`](#backend-availability-snapshot--backendavailabilitysnapshot)
associates those two facts without storing the class in the identity. The declaration order
conveys no preference, score, priority, capability, or fallback policy. Only an explicitly present
[`PartitionScoringConfig`](#partition-scoring-configuration--partitionscoringconfig) value records
a soft class preference; it does not change the category itself.

### Dense target

A dense target supplies one floating weight for every class in a categorical-loss slice. For
current dense categorical cross entropy, its Shape matches logits exactly and each class-axis
slice carries a caller obligation to be finite, non-negative, and normalized to one. It differs
from an [index target](#index-target), which identifies one class with an exact integer. The
three-argument categorical-loss method dispatches between these meanings by target data type.

### Index target

An INT32 or INT64 Tensor containing one exact class index for each coordinate in a categorical
loss's [sample domain](#sample-domain). Its Shape equals the logits Shape with the class axis
removed. An index target is not promoted, cast, broadcast, wrapped, clamped, or converted to one
hot. Every non-ignored value must eventually be in the half-open class range; model construction
does not read values or prove bounds.

### Ignore index

One exact INT32 or INT64 `ScalarValue` whose type matches an index target. A matching target is
ignored before bounds checking or logits evaluation, contributes positive zero, and is excluded
from the categorical-loss `MEAN` denominator. The value is operation attributes metadata rather
than a Tensor producer input; the public API uses one required scalar overload rather than a
public `Optional`.

### Dimension

The size description for one axis of a [shape](#shape). A `StaticDimension` has a known
non-negative `long` size; zero is valid and represents an empty extent. A `DynamicDimension`
names one extent with a canonical non-blank symbol. An `ExpressionDimension` retains either an
exact [symbolic extent expression](#symbolic-extent-expression) or an identity-based constrained
unknown.

Every non-static form is dynamic. Only the named form has a `dynamicSymbol()`: an expression or
unknown has no caller-defined name. Dynamic dimensions are explicit values, not negative-number
sentinels, and local code does not assume that different names or distinct unknowns are equal. See
[Shapes and dimensions](api/tensor-api.md#shapes-and-dimensions).

### Deferred graph constraint

A **deferred graph constraint** is one typed, occurrence-owned Shape obligation that current
package-private compiler verification cannot prove or disprove from immutable graph descriptors.
It records the owning `NodeId`, a concise semantic subject, and a closed immutable predicate such
as dimension equality, a minimum or divisibility requirement, Shape element-count equality, or a
fit-within relation. Equal-looking obligations from different operation occurrences remain
distinct because later diagnostics and transformations need their graph context.

Verification uses three outcomes. A proven predicate disappears, a disproven predicate rejects
the graph in deterministic node/rule/axis order, and an undecidable predicate is retained in the
internal `ValidatedGraph`. Proof may use static values, structural model equality, and conservative
bounds already carried by symbolic expressions. It never assigns a value to a symbol, unifies
different symbols, or treats unavailable or overflowing interval arithmetic as contradiction.

This term is narrower than a **concrete binding constraint**. The current internal value provides
no binding map, substitution, public evaluation API, serialization format, or executable promise.
It also excludes value-dependent obligations such as index contents and bounds, duplicate scatter
targets, categorical target contents, numerical results, and storage validity, which cannot be
decided from descriptor-only graph values. See [current package-private verification
inference](api/compile-api.md#current-package-private-verification-inference).

Accepting a binding-dependent `Tensor.expand` expression in Model does not itself create a
deferred graph constraint. Model retains only the exact target Shape and the EXPAND occurrence.
Current compiler inference translates each non-equal, non-static-source-singleton unresolved
aligned pair to
`AnyOf(DimensionEqual(source, 1), DimensionEqual(source, target))` in aligned-axis order.
Preflight admits the inverse `SUM_TO_SHAPE` only through the same predicate relationship. A
deferred result remains an occurrence-owned compile obligation; it is not a concrete binding,
stride choice, materialization decision, or execution permission.

### Element stride

The number of storage elements advanced when one logical index advances by one position along an axis. Strides are measured in elements, not bytes. A stride of zero can represent repeated data, while canonical row-major strides describe contiguous geometry.

### Floating classification

An elementwise semantic request that reports which broad value class a floating input represents.
The implemented `FloatingClassificationKind` family distinguishes finite values, NaN, and either
infinity. Public `Tensor.isFinite`, `isNaN`, and `isInf` accept only `BFLOAT16`, `FLOAT32`, or
`FLOAT64` and construct fixed non-differentiable BOOL result metadata with the exact input Shape.

A floating classification is a graph-visible value-producing operation, not a numeric transform,
trace diagnostic, validation warning, eager Java boolean, or storage inspection. Model
construction records the request and provenance without calculating classification values.

### Gaussian error linear unit / GELU

A smooth floating activation with exact mathematical target `x * Phi(x)`, where `Phi` is the
standard normal cumulative distribution function. Equivalently, exact GELU is
`0.5 * x * (1 + erf(x / sqrt(2)))`. Synaptik represents it as the parameterless first-class
`UnaryElementwiseKind.GELU` semantic identity.

`UnaryElementwiseKind.GELU_TANH_APPROXIMATION` is a separate fixed function:
`0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`. It is not a configurable mode of
exact GELU. Both functions use the continuous extension negative infinity to negative zero,
preserve signed zero, map positive infinity to positive infinity, and propagate NaN. Current
`Tensor.gelu()` and `geluTanhApproximation()` construct model metadata and provenance only; they do
not execute either function or define backend algorithms. The current package-private compiler
does define first-order formulas: finite and NaN inputs use the corresponding analytic
derivative, positive infinity returns the incoming cotangent, and negative infinity returns exact
positive zero.

### Functional gradient request

The current public immutable `FunctionalGradientRequest` for exactly one or two reverse-mode
stages. Stage one selects exact requested forward outputs; stage two may select generated
first-stage gradients by target-list index. Every stage aligns optional explicit
[cotangent](#cotangent) seeds with its outputs and names an ordered, non-empty,
exact-identity-unique target list from the complete original forward inventory. An absent seed is
an exact typed positive one only for an eligible scalar output. `DisconnectedPolicy.ERROR`
rejects an unreachable target; `ZERO` returns an ordinary exact typed zero expression.
`FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` require this request, while `FORWARD_ONLY`
rejects it. The value exposes no public compile call or Tensor gradient lifecycle.

### Forward graph

The graph that computes outputs from user inputs in the original direction of the tensor expression. It exists before any gradient computation is added. A forward-only compile uses this computation; a backward-capable compile may expand it with a [backward graph](#backward-graph).

### Forward graph optimization

The historical name for the current package-private compiler's optional semantics-preserving
transformation pipeline. It now operates on the complete graph selected by the compile mode:
forward-only or combined forward/backward. The bounded order is exact arithmetic rewriting,
logical-splat constant folding, whole-graph dead-code elimination (DCE), phase- and
derivative-order-local exact
common-subexpression elimination (CSE), and whole-graph DCE cleanup, once each. A changed
candidate is revalidated before the following transform.

DCE walks from the complete ordered graph-output boundary and does not keep a node merely because
its phase is `BACKWARD`; every live occurrence retains all output slots. The sidecar-aware form
additionally prunes only unused fixed constant sources and retains every bindable input. Exact CSE
merges whole occurrences only when their operation value, ordered remapped inputs, phase, and
complete ordered output descriptors match. It never merges across phases, and a graph-output
producer cannot merge or represent another occurrence.

This optional work is distinct from mandatory [graph canonicalization](#graph-canonicalization).
The selected fold matrix is limited to BOOL logic, signed-integral binary arithmetic/comparison,
and same-typed signed-integral scalar arithmetic over explicit splats. It does not evaluate
floating/BFLOAT16 facts, casts, unselected operations, or dense payloads; construct autograd;
derive publication or planning; allocate physical values; or expose a public compiler API. See
[Compile API](api/compile-api.md#current-package-private-canonicalization-and-whole-graph-optimization).

### Pre-capture Tensor-expression autograd

The current bounded package-private compiler design for constructing gradient formulas with the
existing public Tensor
operation vocabulary before graph capture. Named compiler gradient-rule components traverse the
original forward expression, use exact Tensor identity for temporary contribution bookkeeping,
and combine contributions with ordinary `Tensor.add`.

Seeds and derivative constants are storage-free leaves registered explicitly as logical splats.
One capture receives forward outputs, gradient roots and target roles, original forward-producer
identities, and constant facts. It assigns graph-local IDs once and records each node's
`FORWARD` or `BACKWARD` phase. The immutable combined graph then receives inference, validation,
exact optimization, and final validation.

Compiler task 0004 implements the general package-private entry owner `GraphCompiler` and its
mode-neutral internal result `GraphCompilation`; it adds no request aggregate. `FORWARD_ONLY`
produces no BACKWARD nodes and empty gradient results, while backward-capable modes may produce
the combined forward/backward graph described above. `GraphCompilation` is graph-stage state and
is distinct from the later-lifecycle `CompileArtifacts` aggregate.

The current request contains one exact requested scalar floating objective, an ordered non-empty
identity-unique target list in its selected differentiable ancestry, and one implicit exact typed
unit seed. Targets may differ from the objective's floating type. Its closed matrix covers all
seven promoted floating binary arithmetic kinds, all eight exact-type floating scalar kinds,
promoted branch-only WHERE, floating-to-floating CAST, all nineteen floating unary kinds,
ordinary full/axis/multi-axis SUM/MEAN/PROD/MIN/MAX, masked SUM/MEAN, binding-aware SUM_TO_SHAPE,
LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, L2_NORM, CUM_SUM, CUM_PROD, SOFTMAX,
LOG_SOFTMAX, Layer/RMS/batch normalization, every floating MATMUL vector/matrix rank pairing, and
the guarded floating
CONTIGUOUS/RESHAPE/EXPAND/EXPAND_DIMS/SQUEEZE/PERMUTE/SLICE/SLICE_UPDATE/SELECT/PAD/TILE/
CONCAT/STACK families. It also covers floating window transforms, Gather/scatter data roles,
floating SORT and TOP_K values, explicit-state dropout values, exact two-output attention,
grouped convolution, maximum and average pooling, MSE, dense categorical loss, and positive-
static-depth index categorical logits.

The current source-backed closure checkpoint matches the compiled production Model inventory:
37 operation-kind enum families, 107 constants, and 128 exact signature fingerprints, including
both `SLICE_UPDATE` attributes variants. Each legal output/input role is conditionally
differentiable, intentionally non-differentiable, or fail-closed. A conditionally differentiable
role becomes usable only after occurrence-local preflight proves its exact Shape, data type,
cardinality, canonical-auxiliary, normalization, and formula-construction prerequisites. The
successful decision also selects one existing compiler formula family, keeping preflight and
formula dispatch aligned.

Mixed-floating binary, WHERE-branch, and MATMUL contributions use ordinary
[cotangent normalization](#cotangent-normalization). Ordinary MEAN derives its denominator by
reducing an input-shaped logical-one expression across the selected axes. Masked MEAN counts true
positions with ordinary WHERE and SUM, divides, then uses a final ordinary WHERE whose selected
all-false-slice convention returns exact zero. FLOOR/CEIL/SIGN return a direct exact-zero
first-order cotangent without `g * 0` or a floating comparison. These conventions do not create a
gradient-only arithmetic, cast, exceptional-value, validation, rewrite, fold, or optimizer
contract.

Compiler 0005A selects the elementwise boundary policy. Tensor extrema split exact numeric ties
equally; scalar extrema give the Tensor receiver one half; ordered first-class CLAMP composes the
same rule and can therefore produce one quarter when both stages tie. Piecewise extrema, CLAMP,
ABS, and RELU return exact positive zero at unordered NaN positions. Exact GELU, fixed
tanh-approximation GELU, and SiLU return the incoming cotangent at positive infinity and exact
positive zero at negative infinity. Other analytic formulas use ordinary Tensor operations
without gradient-only domain masks. Comparisons, Boolean logic/classification, condition,
attribute, bound, and non-floating cast roles remain non-differentiable.

SLICE_UPDATE accepts its exact retained static or symbolic target-relative placement constraints,
and SUM_TO_SHAPE accepts aligned exact Dimension equality, a static target singleton, or the exact
binding-dependent inverse predicate. Reduction products are division-free, reduction extrema
share exact ties, and softmax formulas consume exact forward outputs. Batch training selects roles
by public output slot and consumes canonical saved-statistic slots three and four only as formula
auxiliaries.

Compiler 0005D reuses exact same-occurrence attention weights and the public maximum-pool output;
it introduces no hidden outputs or tape. Grouped convolution preserves group isolation through
unfold/matrix/fold expressions. Average pooling divides by a logical typed fixed kernel count, and
maximum pooling reconstructs the first eligible logical winner. MSE uses exact typed `2` and `-2`
scalar-operation coefficients. Dense categorical loss supports logits and target; index
categorical loss supports logits only and fails closed for dynamic or zero class depth.
Everything else fails closed on a selected route until its shared formula or derivative policy is
selected.

The technique adds no placeholder Tensor for a captured `ValueId`, second gradient algebra,
public gradient, runtime tape, model derivative rule, or Tensor gradient/backward lifecycle.
Model task 0025 and Compiler tasks 0004, 0004A, 0004B, 0005, and 0005A–0005D are Complete. Public
requests, higher derivatives, optimizer updates, preparation, and execution remain planned.

### Neural-network module, parameter, buffer, and forward context

These are planned `extensions/nn` contracts; no Gradle `nn` project or production Java API exists
yet. A **module** is a stateful neural-network composition unit that can declare child modules,
trainable **parameters**, and persistent **buffers**. A parameter is module-owned state that an
optimizer may update. A buffer is persistent module-owned state that an optimizer does not update,
such as a running statistic. A **forward context** carries the module's selected train/eval mode
through a module tree so layers can choose their forward behavior consistently.

`extensions/nn` composes generic [`Tensor`](#tensor) and operation semantics from
`modules/model`. [`extensions/training`](#training-graph) consumes nn-declared parameters for
optimizer algorithms and training orchestration, but it does not own modules, buffers, or
train/eval behavior. None of these planned contracts grants autograd, backend storage, kernel
selection, runtime execution, or a concrete backend dependency. See [Module
boundaries](architecture/module-boundaries.md#extensions).

### Graph

A directed dataflow model of a computation. [Nodes](#node) represent computation occurrences, and [values](#graph-value) carry logical data from graph inputs or producing nodes to consuming nodes and graph outputs. A graph describes computation and data dependencies; it is not a runtime schedule or a collection of physical buffers.

### Graph value

Logical data flowing through a graph. The implemented immutable `GraphValue` record stores exactly
one graph-local [`ValueId`](#valueid) and one [`TensorDescriptor`](#tensor-descriptor). A value may
be a graph input, an intermediate result, or a graph output. It can exist without a producing node,
one node can produce multiple values, and one value can have multiple consumers. The record stores
no producer, consumer, or role flag. The owning implemented
[`CompiledGraphModel`](#compiled-graph--compiledgraphmodel) derives producers during construction
and validates graph-wide existence, uniqueness, and topology without storing an index. A graph
value is compile-time model state, not the implemented public mutable [`Tensor`](#tensor), a physical
memory slot, storage, or runtime residency. See [Graph values and compiled
nodes](api/tensor-api.md#graph-values-and-compiled-nodes).

### Graph phase

The implemented `GraphPhase` enum classifies compiled nodes as exactly `FORWARD` or `BACKWARD`
compile-time work. It helps later compiler, publication, planning, and diagnostic work distinguish
the two regions. It is not a compile mode, runtime schedule, prepared-execution boundary, ordinal
serialization, or optimizer-update phase.

### Gradient publication binding

The current public immutable `GradientPublicationBinding` association from derivative order and
stage-local target index to the stable target `TensorId` and final gradient `ValueId`, exposed as
`valueId()`. Bindings are ordered first by derivative order and then target index and remain
distinct when multiple targets share one captured value. The combined graph output boundary
instead lists each distinct gradient value once. A binding is not mutable Tensor state, storage
allocation, optimizer state, or runtime delivery.

### Derivative graph metadata

The current public immutable `DerivativeGraphMetadata` sidecar with one encounter-ordered entry
per final graph node: zero for an original forward producer, one for a producer first owned by the
first reverse-mode stage, and two for one first owned by the second stage. It retains the exact
graph reference and augments unchanged [`GraphPhase`](#graph-phase); it contains no Tensor,
request, storage, backend, runtime, or execution state.

### Graph optimization configuration / `GraphOptimizationConfig`

The implemented standalone config record whose sole boolean component,
`optionalOptimizationsEnabled`, permits or suppresses optional semantics-preserving compiler
optimization. `disabled()` creates a fresh false value that requests skipping optional work;
`standard()` creates a fresh true value that permits the compiler's standard optional pipeline.
Direct construction retains either primitive value.

False cannot disable graph capture, ordering, inference, validation, mandatory canonical
representation, mode-required autograd, publication binding, backend-neutral planning,
preparation, or execution. True does not freeze a pass list, pass order, internal graph shape, or
implementation strategy and permits no approximate mathematics, changed numerical semantics,
backend-specific fusion, preparation, or execution behavior. No current `CompileConfig` or public
compiler entry point consumes this value. The current package-private `GraphCompiler` consumes it
only to enable or skip the optional pipeline over the complete mode-selected graph; mandatory
[graph canonicalization](#graph-canonicalization) and validation always run.

### Graph canonicalization

The current mandatory package-private compiler normalization that rebuilds an accepted immutable
graph with dense graph-local identifiers. Graph inputs are allocated first in boundary order,
followed by node outputs in stored topological and output-slot order; node identifiers follow
stored node order. Canonicalization preserves operations, descriptors, phases, repeated inputs,
every output slot, and ordered graph boundaries, then the compiler revalidates the rebuilt graph.

Canonicalization is not optional [forward graph optimization](#forward-graph-optimization), an
operation-specific rewrite, numerical execution, concrete dimension binding, publication, or a
public compiler API.

### Graph RNG state / `GraphRngState`

The implemented opaque public value for one explicit random-number-generator (RNG) state
occurrence in a Tensor expression graph. `GraphRngState.initial(key, counter)` accepts every Java
`long` bit pattern for both words. Key is caller-selected stream/domain identity; counter is the
next abstract logical sample position. Both are interpreted as unsigned 64-bit raw bits, so signed
decimal rendering and ordering have no semantic meaning.

Each initializer creates a fresh zero-input, one-output `GraphRngKind.INITIAL_STATE` producer with
exact `GraphRngStateAttrs(long key, long counter)`. Its private Tensor is `INT64 Shape[2]`, with
lane zero carrying key bits and lane one counter bits, unresolved layout, false gradient
eligibility, no label or storage, and provenance output index zero. The lanes are an opaque state
representation, not public numerical Tensor inputs. The wrapper exposes no public Tensor, words,
storage mutation, generator, split, copy, or arbitrary-wrap API.

State objects use identity equality. Equal words in distinct initializers request
replay-equivalent abstract positions but create different state, Tensor, producer, and identifier
occurrences. Current dropout retains the key and means counter advancement modulo `2^64` by its
bound logical element count. Reusing one state on separate branches deliberately reuses its
interval; sequential callers thread the returned next state. The wrapper is shallowly immutable
and may be shared, but does not synchronize execution.

The words, state descriptor, and ordered state edges are portable model semantics. No random
algorithm, key schedule, counter-to-bits function, floating conversion, or bitstream is selected,
so sampled replay is bounded to the same conforming prepared implementation and configuration.
A future graph serializer must preserve both words losslessly, but no byte encoding, parser,
schema version, or stable enum token exists. Current construction does not sample, materialize an
advanced state value, allocate storage, execute, capture a graph, define gradients, or add runtime
or backend state. See [Explicit graph RNG state](api/tensor-api.md#explicit-graph-rng-state).

### State threading

Passing one state-consuming expression's returned next state into the following expression.
Threading expresses sequential, non-overlapping abstract random intervals without hidden mutation.
Reusing the same input state for multiple consumers instead expresses branching and intentional
interval reuse; sharing an immutable state wrapper does not serialize execution.

### Host storage

Implemented model-level raw storage for physical tensor-element capacity visible in host memory.
The sealed `HostTensorStorage` abstraction currently permits exactly the final
`MemorySegmentStorage` identity wrapper. That wrapper binds a non-null logical data type and a
non-negative physical element capacity to one exact-size, initially live JDK `MemorySegment`.
Capacity is measured in complete physical elements and is independent of logical element count,
layout offset, and referenced span. Byte size is the checked product of capacity and data-type byte
width; zero capacity requires a zero-byte segment.

The wrapper borrows and returns the exact segment. It does not allocate memory, own or close an
arena, extend a scope's lifetime, or implement `AutoCloseable`. For a caller-supplied arena-backed
segment, the caller remains responsible for scope lifetime and JDK thread-access rules. A
factory-created primitive-array segment instead has an automatic scope that keeps its heap base
reachable and is accessible from any thread. Read-only state is descriptive, and liveness is a
point-in-time observation: after caller-controlled closure, the wrapper reports not alive but still
returns the exact dead segment so JDK access rules enforce failure. It defines no typed element
access, alignment, byte order, conversion, synchronization, or mutation-version policy.

Host storage is distinct from logical [layout](#layout), public [`Tensor`](#tensor) state,
device/backend storage, prepared memory, workspaces, and runtime [residency](#residency). The
implemented `Tensor` may borrow the exact storage object and validates matching data type,
resolved referenced span when geometry is available, and attachment-time liveness. The
implemented `TensorFactory` may pass an existing caller-supplied borrowed object through that same
construction path or create a matching primitive-array heap segment whose capacity is exactly the
resolved referenced span. The factory introduces no arena, close operation, external owner, or
deterministic lifetime. Neither path changes the wrapper's sizing, borrowed/non-closing ownership,
or raw-access contract. See [Host-visible storage](api/tensor-api.md#host-visible-storage).

### Kernel

A concrete backend implementation route for executing prepared work, such as a scalar CPU
routine, an OpenBLAS call, a Metal kernel, or a CUDA kernel. Planning never selects executable
kernels. The owning backend selects one during prepare. In convolution documentation, **weight
kernel** instead means the logical `K_h` by `K_w` coefficient window stored in the weight Tensor;
that semantic kernel is not an executable implementation route.

### General matrix multiplication / GEMM

The low-level operation `C[m,n] = alpha * (A[m,k] x B[k,n]) + beta * C[m,n]`. The current
OpenBLAS provider implements only FLOAT32 and FLOAT64 dense row-major, non-transposed GEMM over
caller-owned native memory beginning at byte offset zero. It validates and forwards one already-
normalized call; it does not own Tensor rank adaptation, batching, broadcasting, transposition,
layout conversion, packing, storage allocation, CPU route selection, or fallback.

GEMM is therefore a native invocation primitive, not Synaptik's backend-independent
[`MATMUL`](#matrix-multiplication--matmul) semantic contract. Future CPU lowering may use one or
more GEMM calls to implement MATMUL, linear projection, attention, or convolution, but this
provider surface alone does not implement those operations.

### OpenBLAS library handle / `OpenBlasLibrary`

The implemented caller-owned lifetime for one explicitly selected OpenBLAS native library lookup
and its complete required symbol bindings. A caller supplies either one nonblank operating-system
library name or one absolute path. The provider does not discover a platform filename, search a
path, read configuration, or choose fallback.

A successful open binds `cblas_sgemm`, `cblas_dgemm`, `openblas_set_num_threads`, and
`openblas_get_num_threads` under the ordinary 32-bit-`blasint` C interface. Each open returns a
fresh Java owner whose shared Foreign Function and Memory arena remains alive until idempotent
close. Closing ends only that owner's lookup lifetime; it does not promise physical unloading or
isolate process-global OpenBLAS state from another owner.

The current handle exposes exact FLOAT32/FLOAT64 dense row-major, non-transposed
[`GEMM`](#general-matrix-multiplication--gemm) invocation plus direct positive thread-count query
and control, but no native address or JDK native handle. The thread count is conservatively
treated as mutable library/process state that owners of one loaded binary may observe in common;
this is not a coordination guarantee across independently loaded copies, loader namespaces, or
arbitrary native consumers. Query/set sequences are not atomic, concurrent setters have no
provider-defined winner, thread changes are not serialized with GEMM, and close does not restore
the count. Callers own coordination and any restoration through a still-open owner.

Caller matrix segments remain borrowed and the handle promises no OpenBLAS numerical policy. CPU
prepare later owns OpenBLAS route selection and normalization, while CPU or composition policy
owns configuration and fallback. See the [CPU backend guide](backend-guide/cpu-backend.md#current-low-level-openblas-foundation).

### Layout

The logical mapping from a tensor's multidimensional indices to positions in storage. A layout can describe contiguous, offset-contiguous, strided, or broadcast views using facts such as element strides and storage offset. Layout describes geometry and aliasing; it does not own storage or decide whether a copy must be materialized. See the [Tensor API](api/tensor-api.md#resolved-layouts).

### Logical splat

One exact typed immutable `ScalarValue` that applies to every logical coordinate of one graph
value. The current compiler sidecar uses logical splats only for fixed structural graph inputs.
Equality preserves the scalar data type and exact represented bits, so floating signed zeros and
distinct NaN payloads remain distinct, raw BFLOAT16 bits remain exact, and integral/BOOL values
retain their current canonical representations.

A logical splat has constant payload size even for a dynamic, scalar, or empty Shape. It contains
no Shape, Tensor, host storage, array, memory segment, dense element payload, physical buffer,
backend value, or runtime state. The compiler can fold only the explicitly selected BOOL and
signed-integral operation matrix over these facts. Floating/BFLOAT16 splats are representable but
are not currently evaluated. Current pre-capture autograd caches generated floating scalar bases
by exact data type and represented bits, including signed zeros and NaN payloads, and registers
each created base as one logical splat without reading or allocating storage. The cache supplies
typed zero/unit leaves, extrema or clamp bounds needed as Tensor comparison operands, and
logical-one expansion for ordinary and masked MEAN counts. Creation and binding order follow
deterministic first use. Only generated splat bases reachable from returned gradient expressions
remain in combined-capture ingress. Physical repetition and storage allocation remain future
prepare/backend materialization work. Current Prepare requires a compiler constant source to have
at least one backend-created initialized representation; Runtime carries only the initial-validity
fact, not the scalar.

### Logical memory plan

The implemented public `LogicalMemoryPlan` is an immutable ordered snapshot of distinct
per-value `LogicalMemoryRequirement` records. A current generated requirement retains one graph
value's exact [`ValueId`](#valueid) and [`TensorDescriptor`](#tensor-descriptor), its optional
producing [`PlannedPartition`](#partition), every distinct consuming partition in partition order,
and whether the graph declares it as an output. Generated plan order is
`CompiledGraphModel.values()` order.

Current package-private derivation first validates that the supplied partition recipes cover
every graph node exactly once in graph order and contain maximal adjacent owner runs. It then
derives producers from node outputs, consumers from node inputs, and the output flag from
`CompiledGraphModel.outputs()`. These primitive facts express graph inputs, partition inputs and
outputs, same-owner and cross-owner boundaries, graph-output preservation, and
partition-internal values without a separate role enum. A standalone public plan may be empty;
direct construction validates DTO state but cannot prove graph-relative facts without an owning
graph.

A logical memory plan is not a physical memory plan. A requirement does not itself create a
Runtime buffer slot; shared Prepare creates slots only for exact backend buffer declarations made
after backend-private lowering, fusion, and materialization decisions. It contains no
`ForwardPublicationBinding`, `GradientPublicationBinding`, public Tensor identity, element or
byte count, lifetime, transfer, copy, device, address, buffer, slot, arena, allocation, route,
kernel, schedule, or runtime residency. Retaining `TensorDescriptor` keeps static, dynamic, and
expression Shapes representable. Prepare and concrete backends later turn logical obligations
into physical representations and schedules.

### Lifecycle

The ordered stages through which Synaptik state moves. The core lifecycle is compile, prepare, and
run: compile creates immutable artifacts, prepare creates immutable reusable executable recipes,
and each active run creates exactly one invocation-specific mutable `RunState`. “Lifecycle” also
includes ownership and validity rules for objects and resources within those stages. See
[Lifecycle](architecture/lifecycle.md).

### Lowering

Translation from a planned, backend-neutral graph region into a backend-specific executable form. Lowering may include backend-specific decomposition, fusion, specialization, and representation building. In Synaptik it happens during prepare inside the owning concrete backend, not in planning or the runtime hot path. See [Backend-owned lowering](#backend-owned-lowering).

### Loss

A backend-independent computation meaning that compares model predictions with target values and
produces one or more error values. A **prediction** is the model-produced operand being assessed;
a **target** is the caller-supplied reference operand it is compared with. The current implemented
loss family contains [mean-squared error](#mean-squared-error--mse) and dense-target categorical
plus index-target categorical cross entropy directly from logits. A loss operation is model
metadata, not a gradient rule, optimizer, training session, compiler pass, backend kernel, or
executed value.

### Loss reduction

The explicit policy that maps a loss's logical error domain to its result Shape. The current
`LossReduction` vocabulary contains `NONE`, which retains every error coordinate; `SUM`, which
combines the complete domain into one scalar; and `MEAN`, which divides that sum by a
family-defined denominator. The vocabulary stores no default, axis, denominator, weight, mask,
ignore value, Tensor, or executable behavior. It is distinct from aggregate Tensor reduction
because a loss first combines ordered prediction and target roles.

### Logit

An unnormalized score for one class. Dense and index categorical cross entropy consume logits
directly so their negative-log-softmax meanings remain stable semantic operations rather than
first materializing probabilities.

### Materialization

Producing an equivalent concrete stored representation when a logical value cannot be consumed in
its current representation. For example, a backend route may require the value in an already-
created contiguous destination rather than its current strided source. Planning expresses logical
materialization requirements; prepare selects exact work; and the current Runtime transfer
contract can realize that value by explicitly copying between two already-created representations
of the same prepared buffer. Materialization is that transfer in this context, not a second
Runtime operation, allocation, transfer-route search, hidden coherence policy, or property decided
by `LayoutDescriptor` alone.

The current CPU proving slice also implements one backend-private workspace materialization. It
copies one selected input into a CPU-owned run workspace before the generated consumer, without
creating another graph value or Runtime transfer recipe. See [CPU contiguous materialization
plan](#cpu-contiguous-materialization-plan).

The CPU affine family performs a different boundary materialization: after folding eligible
same-unit view intermediates, it copies the original source's represented bits into the final
result's distinct declared buffer according to both resolved layouts. It allocates no optional
workspace and creates no shared alias. See [CPU affine boundary copy](#cpu-affine-boundary-copy).

### Linear projection

A convenience that maps an input's final **in-features** axis to an **out-features** axis using a
conventional rank-two weight with Shape `[outFeatures, inFeatures]`. The implemented
`Tensor.linear(weight)` is exactly PERMUTE `[1, 0]` followed by MATMUL;
`Tensor.linear(weight, bias)` adds an exact rank-one `[outFeatures]` bias through ordinary ADD.
There is no LINEAR operation kind. The visible primitive producer chain remains available for
future compiler inspection.

The optional bias Dimension must be structurally equal to the weight out-features Dimension; this
is stricter than general broadcasting. No-bias returns the MATMUL product directly. In the biased
form, ADD produces a structurally equal final Shape with the exact product Dimension references,
although its outer Shape object may be distinct. A linear projection here is storage-free model
metadata, not a stateful linear layer: it owns no parameters, initialization, serialization,
dedicated gradient rule, compiler pass, fusion, backend kernel, or execution. Its visible
floating PERMUTE/MATMUL/ADD primitive chain is differentiable by current package-private compiler
autograd when each primitive satisfies its closed rule guards. See
[Linear-projection convenience](api/tensor-api.md#linear-projection-convenience).

### Masked reduction

An aggregate `SUM` or `MEAN` that excludes input positions whose aligned boolean mask position is
false. The implemented `MaskedReductionAttrs` semantic value stores one already normalized,
non-negative reduction axis and no Shape, mask, or broadcast plan. The selected public baseline
removes the reduction axis. False positions exclude their corresponding inputs before aggregation,
including NaN and infinity. A masked sum with no selected values is zero. A masked mean divides by
the number of true selected positions for each output and is NaN when that count is zero; its NaN
payload or bit pattern is unspecified.

Public masked Tensor expressions are also current. They validate floating input and an exact BOOL
mask, require ordinary right-aligned broadcasting to produce exactly the input Shape, remove the
normalized reduction axis, preserve input type and gradient eligibility, and record exact
`[input, mask]` provenance. Callers make other axis intent visible with rank-edit or Shape
expressions before the reduction. A static zero-sized reduction axis produces zero sum slices and
NaN mean slices; runtime zero-sized or all-false dynamic slices follow the same semantics. Storage
alignment, value selection, counting, aggregation, division, backend behavior, and numerical
execution remain planned. Current package-private compiler autograd supports masked floating SUM
and MEAN. Both restore the reduced axis and expand the cotangent. SUM routes it with the original
mask and an exact typed zero. MEAN constructs the true count with ordinary WHERE and SUM, divides
the restored cotangent, then applies a final ordinary WHERE. Its explicit all-false-slice
first-order convention returns zero at every input coordinate without prescribing branch
evaluation or suppressing the intermediate quotient. The BOOL mask receives no cotangent.

### Mean-squared error / MSE

The implemented loss meaning represented by `LossKind.MEAN_SQUARED_ERROR` and
`MeanSquaredErrorAttrs(reduction)`. For prediction `p_i` and exact-shape target `t_i`, each logical
error is `(p_i - t_i)^2`. `NONE` preserves the exact prediction Shape, while `SUM` and `MEAN`
produce a scalar. The MSE mean denominator is the complete logical element count after unresolved
Dimensions are bound: a scalar count is one, and a zero-extent domain has empty `NONE`, positive-
zero `SUM`, and NaN `MEAN`.

Current public construction accepts BFLOAT16, FLOAT32, and FLOAT64, promotes the two operands,
rejects unequal static Dimensions, defers equality involving unresolved Dimensions, and never
broadcasts the target. It records exact ordered `[prediction, target]` provenance and reads no
values. Current package-private compiler autograd supports both roles: prediction uses
`2 * (prediction - target)` and target uses `-2 * (prediction - target)`, with reduced cotangents
restored through logical Tensor counts before exact descriptor normalization. See
[Mean-squared-error loss expressions](api/tensor-api.md#mean-squared-error-loss-expressions).

### Matrix multiplication / `MATMUL`

The implemented backend-independent operation that contracts the final left axis with the right
vector axis or the penultimate right matrix axis, broadcasts leading batch dimensions
right-aligned, and retains the remaining row and column axes. Rank-one operands are temporarily
promoted to matrices and their inserted result axes are removed, so vector-vector `MATMUL`
produces a scalar. `MatmulKind.MATMUL` is parameterless with exactly two logical inputs and one
output; public `Tensor.matmul` constructs the current storage-free expression metadata. The
operation does not itself capture a graph, multiply stored values, select a backend algorithm, or
execute. Current package-private compiler autograd constructs cotangents for every floating
vector/matrix rank pairing, unbroadcasts batch axes with ordinary `sumToShape`, and then uses one
ordinary cast when a promoted contribution differs from the selected operand type. Integral
MATMUL remains rejected. See
[Matrix-multiplication expressions](api/tensor-api.md#matrix-multiplication-expressions).

### Buffer slot / `BufferSlot`

The implemented public runtime record that carries one non-negative `long` identity for a buffer
position within one owning prepared-memory-plan context. Zero through `Long.MAX_VALUE` are valid,
with no reserved sentinel. The record stores no plan reference, so its ordinary equality and
hashing compare only the exact number; equal values used by different plans do not establish one
cross-plan identity.

A `BufferSlot` is deeply immutable and may be retained by the current
[`PreparedMemoryPlan`](#prepared-memory-plan--preparedmemoryplan). It is not a logical
[`ValueId`](#valueid), physical buffer, storage handle, address, allocation, device, resource,
residency fact, or per-run binding. Constructing one performs no allocation or resource operation.
Prepare assignment is current; physical binding and scheduled execution remain planned. Current
`RunState` uses the entry-list position rather than this numeric value for direct representation access. See the
[Runtime API](api/runtime-api.md#current-slot-identities).

### Buffer validity

The implemented per-run boolean fact stating whether one structurally resident physical buffer
representation currently contains its logical slot value. Each dense buffer/representation
position in an open [`RunState`](#run-state--runstate) has one independent bit. Borrowed caller
inputs start valid; newly created run-owned buffers start invalid. Zero, one, or multiple copies
of one logical buffer may be valid, and repeating a value is permitted. Workspaces are
backend-local scratch and have no buffer validity.

Validity query and mutation are constant-time array operations. Mutation records exactly the
supplied fact and performs no storage inspection, copy, transfer, kernel call, ownership change,
implicit invalidation, or coherence. Later transfer and execution work must update bits explicitly
after successful physical actions. Validity is per-run mutable state, not prepared recipe state,
model Tensor state, or a physical representation capability.

### Workspace slot / `WorkspaceSlot`

The implemented public Runtime record that carries one non-negative `long` identity for a
workspace position within one owning prepared-memory-plan context. Zero through
`Long.MAX_VALUE` are valid, with no reserved sentinel. It is nominally distinct from
[`BufferSlot`](#buffer-slot--bufferslot) and from a Prepare analysis-local workspace requirement
ID even when their numbers match.

The record stores no plan reference, so ordinary equality and hashing compare only another
`WorkspaceSlot` with the same number. It is deeply immutable and may be retained by a current
`PreparedMemoryPlan`, but it is not physical scratch storage, a handle, address, allocation,
device, resource, residency fact, lifetime, or per-run binding. Construction performs no
allocation or resource operation.

### Prepared memory plan / `PreparedMemoryPlan`

The implemented public Runtime record that describes final ordered reusable slot geometry for one
prepared plan. Its immutable `buffers` and `workspaces` list snapshots contain
`BufferEntry(BufferSlot, byteSize, byteAlignment)` and
`WorkspaceEntry(WorkspaceSlot, byteSize, byteAlignment)` values. Size is an exact non-negative
byte count, including zero. Alignment is an exact positive power of two from `1` through
`1L << 62`.

Supplied entry order is preserved, exact immutable entry and slot references are retained, and
the caller's list containers are not retained. Buffer slots are unique within buffer entries and
workspace slots within workspace entries; equal numeric buffer/workspace values are valid because
the domains are nominally separate. Either list may be empty. Ordinary record equality and
hashing use the retained immutable values; record text is diagnostic only.

The plan contains geometry, not provenance or resources. It retains no Prepare analysis
requirement, graph `ValueId`, analysis-local workspace ID, source-to-slot association, physical
storage, address, allocation, ownership, aliasing or lifetime proof, device, residency, or per-run
binding. It does not sort, derive, assign, allocate, bind, access, or release anything. Current
Prepare assignment constructs it while retaining source associations. Current `RunState`
consumes the plan's entry order for per-run representation access without changing the geometry;
later Runtime/backend contracts own physical allocation, access mechanics, and execution.

### Prepared representation plan / `PreparedRepresentationPlan`

The implemented immutable reusable Runtime description of how every dense buffer representation
and workspace for one exact `PreparedMemoryPlan` originates. Each non-empty buffer-position list
contains `CallerInput` occurrences, `CreatedBuffer(BufferCreator)` values,
`InitializedBuffer(BufferCreator)` values, or a permitted combination; the workspace
list contains one `WorkspaceCreator` per prepared workspace. The record snapshots both buffer-list
levels and the workspace list while retaining exact preparation and callback references. It
invokes no callback and owns no physical object.

Caller-input occurrences consume a later flat caller list in buffer/representation encounter
order. `CreatedBuffer` results start invalid, while `InitializedBuffer` results start valid; both
are run-owned and otherwise share creator, identity, ordering, rollback, and cleanup rules.
Concrete backends implement immutable thread-safe creators, and every successful call must return
a fresh non-null physical result for that position and run. Current package-private cold
setup validates all callers before invoking callbacks, creates buffers then workspaces, and rolls
back successfully created results in reverse creation order on failure. A current first-only
`PreparedSchedule.RepresentationCreationStep` makes the plan reachable through the unchanged
two-component `PreparedExecution`. Explicit transfer/materialization, validity, and publication
contracts, public runner orchestration, and executable-output validity transitions are current.

### Memory slot

An architecture-level prepared-plan position through which a schedule can refer to reusable
storage without embedding a raw address. Current
[`BufferSlot`](#buffer-slot--bufferslot) and
[`WorkspaceSlot`](#workspace-slot--workspaceslot) values provide nominally separate plan-local
identities, and current [`PreparedMemoryPlan`](#prepared-memory-plan--preparedmemoryplan) records
their final byte geometry. Current `RunState` accesses representations by dense plan-entry
position rather than by interpreting either slot's numeric component. Current Prepare retains
logical source association separately through `PreparedBufferAssignment`; physical allocation and
access, aliasing, and lifetime analysis remain later work. A memory-slot
identity is not itself a physical execution resource and must not be confused with a logical
[`ValueId`](#valueid) or an analysis requirement ID.

### Physical runtime representation

A concrete backend-owned implementation of one buffer or workspace in host, device, or native
storage. Current Runtime defines the distinct nominal `BufferRepresentation` and
`WorkspaceRepresentation` lifecycle roles, each exposing only unchecked cleanup through
`close()`. CPU now provides package-private borrowed and run-owned native implementations;
operation routes, transfers, and other backend representations remain planned. Runtime owns the
logical per-run association, ownership, structural residency, explicit buffer validity, and
cleanup orchestration. The backend representation owns physical allocation, release, transfer,
and access mechanics. A
buffer slot may have multiple representations only when prepared work explicitly requires them.
A workspace representation is backend-local scratch and is not a transferable logical graph
value.

Current `BufferRepresentationBinding` retains one exact buffer representation plus
`RunResourceOwnership.BORROWED` or `RUN_OWNED`. Borrowed cleanup remains with the caller;
run-owned cleanup transfers only after `RunState` construction succeeds. Every workspace supplied
to a successfully constructed state is run-owned. A repeated exact representation identity is
rejected across every buffer and workspace position in one state.

Physical representations use heterogeneous Java types. The current cold binding phase performs
explicit checked compatibility once and creates backend-owned typed
[`BoundInvocation`](#bound-invocation--boundinvocation) objects with direct references before the
hot path. No raw `Object`, unchecked generic API, registry, reflection, or public
concrete-backend switch is part of the shared contract.

### CPU buffer argument

The implemented backend-private, non-owning direct access form created once when a CPU
`PreparedExecutable` cold-binds a selected physical buffer representation. When the exact
primitive heap carrier is observable and matches the logical data type, an array form retains the
carrier, its carrier-relative byte offset, exact byte size, and read-only state. Otherwise
`CpuBufferArgument.Segment` retains the exact selected `MemorySegment` or slice with byte offset
zero relative to that retained region.

`Segment` describes exact-segment access, not native provenance. Genuine native segments use it,
but a heap-backed segment also uses it when the JDK does not expose its primitive carrier. In JDK
26, `heapBase()` is empty for a read-only heap segment, so cold binding preserves that exact
read-only segment without copying. Every form preserves the underlying ownership, lifetime,
current-thread access, byte geometry, and writability restrictions; the argument owns and closes
no memory. Mixed array and exact-segment arguments are valid in one ordered invocation.

### `NoOperationAttrs`

The implemented canonical immutable attribute value for an operation kind that has no semantic parameters. It is a single-value enum whose only value is `NoOperationAttrs.INSTANCE`. The singleton makes “no parameters” explicit and non-null rather than representing absence with `null`, an empty map, or a newly allocated placeholder. See [`OperationAttrs`](#operationattrs).

### Normalized axis

A non-negative axis index in the range established by a tensor's rank. The implemented
`Shape.normalizeAxis` method accepts a caller-facing positive or negative axis and returns this
form. Reduction attributes store only an already normalized non-negative `int`; they do not retain
a Shape or prove that the index exists for a particular input. A negative stored axis is invalid,
and no negative value is reused as an all-axis sentinel. Current `sum`, `mean`, `prod`, reduction
`min`, reduction `max`, boolean `all`, and boolean `any` axis methods normalize against the input
Shape before constructing semantic attributes. The current axis-only `argMin` and `argMax`
methods use the same boundary before constructing `ArgExtremaAttrs`.
`MaskedReductionAttrs` follows the same normalized-axis boundary, and current public masked
expression construction normalizes that axis and validates ordinary broadcast compatibility
before creating the attributes.
`CumulativeScanAttrs` also stores only an already normalized non-negative axis; current public
Shape-aware `Tensor.cumSum` and `Tensor.cumProd` construction normalizes the caller axis before
creating it.
`SoftmaxAttrs` likewise stores only an already normalized non-negative axis; current public
Shape-aware `Tensor.softmax` and `Tensor.logSoftmax` construction normalizes the caller axis
before creating it.
`AxisTransformAttrs` stores one normalized non-negative position without a Shape or rank. The
public boundary interprets it according to the operation kind: current `expandDims` uses an output
insertion position normalized against rank plus one, while current `squeeze` uses an input removal
position normalized against the existing Shape and validates the selected static singleton.
`SliceAttrs` stores an ordered list of distinct normalized non-negative input axes. It has no Shape
or rank, so it cannot prove that an axis exists for a future input. The current public raw slice,
length-defined slice, flip, and array-based slice-update boundaries normalize caller-facing
negative axes before constructing these attributes; duplicates fail after normalization in caller
order. The length-defined path differs only in coordinate syntax: starts are already non-negative
and lengths are explicit rather than derived from raw half-open bounds.
`CompositionAxisAttrs` stores the normalized existing CONCAT input axis or inserted STACK result
axis, with the paired kind supplying the interpretation. It contains no rank context.
`IndexAxisAttrs` stores the normalized data axis for Gather, Gather Elements, and
Gather-compatible Scatter Add.
`ScatterElementsAttrs` stores the corresponding axis plus an explicit
reduction for scatter-elements. `ScatterNdAttrs` instead stores a shared batch count and reduction;
tuple depth remains in the final indices Dimension. The current gather and scatter expression
boundaries validate their public caller parameters before constructing these attributes.
`UnfoldAxisAttrs` stores a normalized source axis, while `FoldAxisAttrs` stores a normalized
restored target axis. Neither contains rank context. Current public `unfold` normalizes raw syntax
against the input rank, while public `foldAxis` normalizes against target rank, which is one less
than input rank.

### Node

One occurrence of computation in a graph. The implemented immutable `CompiledNode` record stores a
graph-local [`NodeId`](#nodeid), one [`Operation`](#operation), and ordered immutable input and
output `ValueId` snapshots. Empty and repeated inputs are valid; outputs must be non-empty and
unique within that node. Reusing the same operation kind in two places creates two node
occurrences. A node is not the operation semantics alone and is not the data flowing between
computations; that data is represented by [graph values](#graph-value). The record validates its
local list invariants and checks final input/output counts against the selected
[`OperationSignature`](#operation-signature). The owning implemented `CompiledGraphModel` validates
referenced-value existence, producer uniqueness, topology, and graph boundaries, while
operand-aware descriptor compatibility remains separate. A compiled node is compile-time model state and
must not enter runtime hot paths. See [Graph values and compiled
nodes](api/tensor-api.md#graph-values-and-compiled-nodes).

### `NodeId`

A validated non-negative identifier for a node occurrence within one owning graph. It identifies where operation semantics occur, not the operation kind itself. Its numeric value may be reused in another graph, so it has meaning only with its graph context. See [Identifiers](api/tensor-api.md#typed-identifiers).

### Operation capability query / `OperationCapabilityQuery`

The implemented immutable planning question for one structurally valid operation occurrence. It
retains the exact backend-independent [`Operation`](#operation) reference and immutable ordered
membership snapshots of the exact input and output
[`TensorDescriptor`](#tensor-descriptor) references. Construction rejects null top-level or
element references, snapshots inputs before outputs, and then validates only the final input and
output counts through the operation's [`OperationSignature`](#operation-signature).

The query has no graph, node, value, phase, backend, device, availability, requirement, score,
route, kernel, prepare, runtime, or execution state. It does not validate descriptor data-type,
Shape, layout, gradient, operand-role, or graph compatibility. A
[`BackendCapabilityProvider`](#backend-capability-provider--backendcapabilityprovider) answers the
query for its own stable backend identity.

### Operation

The implemented immutable value that keeps two parts of a computation description together: an
[`OperationKind`](#operationkind), which says which computation is meant, and
[`OperationAttrs`](#operationattrs), which carries its typed parameters. Both parts must be
non-null and are retained unchanged. Construction resolves the kind's family-owned
[`OperationSignature`](#operation-signature) and rejects an attributes value whose exact concrete
class is not accepted. The signature is derived, not stored as a third record component. Record
equality and hashing therefore still use only kind and attributes, while text is diagnostic rather
than serialization. An operation does not report backend support, perform compiler work, choose a
kernel, execute computation, own runtime state, or identify a graph occurrence; an implemented
[`CompiledNode`](#node) represents that occurrence.

### `OperationAttrs`

The implemented zero-method marker contract for immutable, typed parameters that refine which
computation an [`Operation`](#operation) describes. Implemented scalar-family values are
`ScalarValueAttrs`, which holds one exact [`ScalarValue`](#typed-scalar-value--scalarvalue), and
`ClampRangeAttrs`, which holds exact same-type numeric inclusive lower and upper bounds.
Implemented cast-family `CastAttrs` holds one exact
non-null target `DataType` without duplicating a source type. Implemented reduction-family
`AxisReductionAttrs` holds one normalized axis and retained-dimension choice, while `ArgExtremaAttrs`
adds an explicit tie policy and `MaskedReductionAttrs` holds exactly one normalized reduction
axis. Its concrete attributes class distinguishes the first-class masked occurrence. Implemented
scan-family `CumulativeScanAttrs` holds one
normalized axis plus exact exclusive and reverse flags. Implemented normalization-family
`SoftmaxAttrs` holds one normalized axis shared by softmax and log-softmax;
implemented loss-family `MeanSquaredErrorAttrs` holds one exact non-null `LossReduction`;
`BatchNormInferenceAttrs` holds one normalized layout-neutral channel axis and exact typed epsilon
for stateless five-input inference. `BatchNormTrainingAttrs` adds exact typed new-batch-weight
momentum for pure five-input/five-output training and running-statistic transition. Implemented
layout-operation `TargetShapeAttrs` holds one exact normalized semantic result Shape shared by
reshape and expand. Layout-operation `PermutationAttrs` holds one complete normalized
output-to-input axis mapping, while `AxisTransformAttrs` holds one normalized non-negative
insertion or removal position. `SliceAttrs` holds immutable ordered parallel lists of normalized
inclusive starts, selected lengths, distinct axes, and signed non-zero steps. Each slice entry is
an exact finite coordinate sequence rather than a stored raw or normalized exclusive end.
`CropToShapeAttrs` holds exact target and per-axis prefix Shapes for target-relative `SLICE`
extraction or `SLICE_UPDATE` placement; the Shapes may contain unresolved Dimensions and are
retained without binding.
`PadAttrs` holds immutable
ordered before/after widths and one exact typed scalar constant, while `TileAttrs` holds immutable
ordered positive complete-pattern repeat counts. `CompositionAxisAttrs` holds one normalized axis
shared by concat and stack. `SelectAttrs` holds one normalized source axis and one normalized
scalar coordinate for axis-removing scalar select and repeated-select unstack. `IndexAxisAttrs`
holds one normalized data axis shared by `GATHER`, `GATHER_ELEMENTS`, and `SCATTER_ADD`.
`OneHotAttrs` holds one positive static `long` depth for the trailing class axis of `ONE_HOT` and
contains no input value, Shape, axis, on/off value, output type, or execution policy.
`GatherNdAttrs` holds the normalized count of shared leading batch Dimensions
for `GATHER_ND`; tuple depth remains the final indices Dimension rather than attribute state.
`ScatterElementsAttrs` holds one normalized data axis plus an explicit non-null
`ScatterReduction` for `SCATTER_ELEMENTS`. `ScatterNdAttrs` holds one normalized non-negative
shared-batch count plus an explicit non-null `ScatterReduction` for `SCATTER_ND`; tuple depth
remains the final indices Dimension. `UnfoldAxisAttrs`
carries normalized general-axis window size and step; `FoldAxisAttrs` carries a normalized target
axis, explicit restored extent, and step. `Window2dAttrs` carries symmetric NCHW
kernel/stride/padding/dilation geometry and its rounding flag. `Unfold2dAttrs` carries that exact
geometry plus one exact typed padding scalar, while `Fold2dAttrs` carries one exact output Shape
plus the shared geometry. Other
families may define records for another operation-specific value. Implementations use typed
fields, defensively isolate mutable inputs, and provide structural equality and hashing; they do not use a primary
string-keyed map or contain backend, compiler-service, mutable tensor, storage, or runtime state.
Kinds without parameters use [`NoOperationAttrs.INSTANCE`](#nooperationattrs). The marker
identifies the role of a value but does not enforce immutability at runtime.

### `OperationKind`

The implemented open typed discriminator that supplies the “which computation” part of an
[`Operation`](#operation). `name()` provides stable diagnostic text. `signatures()` returns the
family-owned immutable structural variants, and `signatureFor(attrs)` resolves one by exact
attributes runtime class. Equality belongs to the typed kind value, so equal name text from
unrelated kind types does not create equivalence or a global string registry. A kind signature
describes attributes and occurrence counts, not backend support, cost, fusion, storage, execution
behavior, or a kernel route.

Public exposure is separate from semantic representability. For example, `SliceKind.SLICE`
supports public slice, flip, and target-relative crop expressions, while
`SliceKind.SLICE_UPDATE` supports public coordinate-sequence and target-relative functional
replacement. Current
`WindowTransformKind.FOLD_AXIS` also has public `Tensor.foldAxis` construction, but future
gradient use remains compiler-owned. Every kind still supplies its exact family-owned structural
signatures.

### Operation signature

The implemented immutable `OperationSignature` record describes one accepted structural variant
of an operation kind. It stores the exact concrete `OperationAttrs` class and inclusive minimum
and maximum logical input and output counts. These counts are **occurrence cardinality**: the
number of ordered values consumed or produced by one node, not Tensor rank, Shape, or element
count.

A **fixed** cardinality has equal minimum and maximum values, such as exactly two inputs. A
**bounded** cardinality allows a finite inclusive range. A **variadic** input accepts a variable
count; the current representation uses `Integer.MAX_VALUE` as the real inclusive upper bound of a
Java list size, not as a sentinel. Signatures match attributes by exact class, fail closed for
missing or malformed family declarations, and contain no operand inference, graph-wide rule,
backend capability, or executable behavior.

The first production family is `BinaryArithmeticKind`, an enum containing exactly `ADD`, `SUB`,
`MUL`, `DIV`, `MIN`, `MAX`, and `POW`. These values identify ordered tensor-to-tensor elementwise
arithmetic meanings. They have no intrinsic parameters and therefore compose with
`NoOperationAttrs.INSTANCE`. The enum stores no operands or broadcast metadata and does not build
Tensor expressions by itself, infer shapes or data types, identify graph occurrences, execute
computation, or report backend support. The implemented public Tensor methods consume these values
while separately owning local expression construction. The enum's inherited names are diagnostics
rather than serialization or dispatch keys.

The second production family is `UnaryElementwiseKind`, an enum containing exactly `ABS`, `NEG`,
`RECIPROCAL`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`,
`RELU`, `SIGMOID`, and `TANH`. These values identify one-input elementwise mathematical or
activation meanings and compose with `NoOperationAttrs.INSTANCE`. Their shared signature declares
one input and one output. `RSQRT`, `LOG1P`, and `EXPM1` remain first-class rather than decomposed
transforms. No kind promises an algorithm, bitwise result, fixed accuracy bound, or backend route.
The enum defines no descriptor inference, provenance, gradient, execution, or backend support.
The implemented public unary Tensor methods consume these values while separately owning local
expression construction.

The separate `FloatingClassificationKind` family contains exactly `IS_FINITE`, `IS_NAN`, and
`IS_INF`. It uses the same parameterless one-input, one-output structural signature but remains a
distinct family because public construction fixes the result to non-differentiable BOOL instead
of preserving the floating input type. The enum stores no result metadata and does not inspect or
classify values.

The next production family is `ScalarElementwiseKind`, an enum containing exactly `ADD`, `SUB`,
`MUL`, `DIV`, `MIN`, `MAX`, `POW`, and `CLAMP`. These values identify parameterized one-input
elementwise meanings. The first seven pair with `ScalarValueAttrs`; `CLAMP` pairs with
`ClampRangeAttrs`. The scalar values are attributes rather than additional Tensor inputs, and
family signatures enforce each exact attributes class and one-input, one-output cardinality. The
attributes retain exact typed scalar values by reference. Clamp-range construction requires the
same non-BOOL data type and rejects only a strict inversion in that represented primitive type,
so equal bounds, both signed-zero orderings, ordered infinities, and floating NaN endpoints are
valid. The enum and attributes perform no Tensor expression construction,
descriptor inference, numerical execution behavior, gradients, or backend support by themselves.
The implemented public scalar Tensor methods consume these values while separately owning
floating or selected signed-integral validation, descriptor derivation, exact attribute
composition, and one-input provenance. Signed-integral support is limited to ADD, SUB, MUL, MIN,
and MAX; DIV, POW, and first-class range CLAMP remain floating-only.

Floating scalar and binary MIN/MAX propagate NaN, order infinities normally, and select negative
zero for minimum or positive zero for maximum when comparing opposite signed zeros, independent
of operand order. Unequal non-NaN values use ordinary numeric order; equal nonzero candidates
produce that numeric value without identifying one source operand. Scalar extrema compare the
input's represented numeric value with the exact same-typed `ScalarValue` represented value; raw
scalar bits remain metadata equality rather than bitwise elementwise equality. FLOAT64 and
FLOAT32 use their represented IEEE-754 values, and BFLOAT16 uses the value represented by its
exact 16-bit storage. Range CLAMP means one first-class request with exactly ordered
`MIN(MAX(input, minValue), maxValue)` value semantics and no stored intermediate producers. Any
NaN input or bound produces NaN; equal non-NaN same-representation bounds produce that bound;
bounds `[-0, +0]` select the corresponding directional zero; and bounds `[+0, -0]` produce
negative zero for every non-NaN input. Public `clampMin` creates one scalar MAX producer and
`clampMax` one scalar MIN producer; no separate one-bound kind exists. These semantics promise no
NaN payload, algorithm, gradient rule, backend route, or execution. Integral ADD, SUB, and MUL use
fixed-width two's-complement modular meaning; integral MIN and MAX use ordinary signed order.

The fourth production family is `BinaryComparisonKind`, an enum containing exactly
`GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and `NOT_EQUAL`. These
values identify ordered, parameterless tensor-to-tensor comparison meanings and compose with
`NoOperationAttrs.INSTANCE`. The enum stores no operands, broadcast geometry, BOOL result facts,
or execution metadata. Its floating edge policy is ordinary represented numeric comparison after
existing promotion. Every ordered relation is false if either operand is NaN. Negative and
positive zero compare equal, so neither strict relation holds and both inclusive relations hold.
`EQUAL` is exact numeric rather than raw-bit or tolerance equality: NaN is unequal to every value,
including itself, and opposite signed zeros are equal. `NOT_EQUAL` is its logical complement.
Finite values and infinities otherwise use ordinary numeric order. FLOAT64, FLOAT32, and BFLOAT16
use the represented-value boundaries described above. The implemented public comparison Tensor
methods consume these values while separately owning same-category floating or signed-integral input
validation, local broadcasting, fixed BOOL result derivation, and ordered provenance. Integral
relations use ordinary signed order after promotion; EQUAL and NOT_EQUAL compare exact promoted
signed values. Compiler capture, derivative policy, execution, and backend support remain planned.
Its inherited names are diagnostic rather than serialization or dispatch keys, and an equally
named kind from another family remains a different typed value.

The fifth production family is `BooleanLogicalKind`, an enum containing exactly `AND`, `OR`, and
`NOT`. These values identify parameterless elementwise boolean conjunction, disjunction, and
negation and compose with `NoOperationAttrs.INSTANCE`. `AND` and `OR` have two logical input roles;
`NOT` has one. Family signatures declare those exact input counts and one output. The enum itself
defines no BOOL descriptor eligibility, binary broadcasting, unary shape
preservation, provenance, storage representation, numeric truthiness, gradient, execution, or
backend support. The implemented public logical Tensor methods separately own exact BOOL input
validation, binary broadcast or unary shape rules, fixed BOOL results, and provenance. Its
inherited names are diagnostic rather than serialization or dispatch keys, and an equally named
kind from another family remains a different typed value.

The sixth production family is `WhereSelectionKind`, an enum containing exactly `WHERE`. This
parameterless value identifies elementwise conditional choice with three ordered logical roles:
condition, true branch, and false branch. A true condition chooses the corresponding true-branch
value; otherwise it chooses the false-branch value. The kind composes with
`NoOperationAttrs.INSTANCE`. It is distinct from scalar-index `select`, gather, take, and scatter.
Its signature declares the three ordered inputs and one output.
The enum itself defines no Tensor construction, condition or branch eligibility, promotion,
three-way broadcasting, result descriptor, provenance, evaluation order, gradient, compiler,
ONNX, execution, or backend-support behavior. Static `Tensor.where` separately owns local
BOOL/floating validation, branch promotion, ordered pairwise broadcasting, descriptor derivation,
and exact three-input provenance. Its inherited name is diagnostic rather than a serialization or
dispatch key, and an equally named kind from another family remains a different typed value.

The seventh production family is `CastKind`, an enum containing exactly `CAST`. This value
identifies parameterized elementwise conversion of one logical input and pairs with `CastAttrs`,
whose sole component is the exact non-null target `DataType`. Every current data type is a valid
target. The source type remains a fact of the later input descriptor rather than duplicated
attribute state. The family signature enforces `CastAttrs` and declares one input and one output. The kind and
attributes alone define no source compatibility, same-type handling, result descriptor, numerical
conversion rules, gradients, provenance, compiler capture, execution, or backend support. The
implemented public `Tensor.cast` method separately owns fresh expression construction, exact shape
retention, unresolved layout, floating-only eligibility retention, and one-input provenance. Their
text forms are diagnostic rather than serialization or dispatch contracts.

The eighth production family is `AggregateReductionKind`, an enum containing exactly `SUM`,
`MEAN`, `PROD`, `MIN`, `MAX`, `ALL`, `ANY`, `ARG_MAX`, `ARG_MIN`, `LOG_SUM_EXP`, `VARIANCE`,
`STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM`. The first seven ordinary kinds pair
with `NoOperationAttrs.INSTANCE` for a full reduction over every input axis or with
`AxisReductionAttrs` for one already normalized axis or `MultiAxisReductionAttrs` for ordered
distinct axes. `ARG_MIN` and `ARG_MAX` pair with shared
`ArgExtremaAttrs`, which adds an explicit `FIRST_INDEX` or `LAST_INDEX` tie policy. Family signatures enforce these exact
variants and distinguish one-input ordinary forms from two-input masked forms. Masked
axis-removing `SUM` and `MEAN` pair with `MaskedReductionAttrs`, whose sole component is the
normalized reduction axis. Their semantic contract excludes false-position values, including NaN
and infinity, before aggregation; a zero selected-count produces zero for sum and NaN for mean
without a payload guarantee. The family stores no Tensor input, result descriptor, negative all-axis
sentinel, gradient rule, executable behavior, or backend support. Public `sum`, `prod`, reduction
`min`, and reduction `max` separately own floating-or-integral eligibility, while `mean` owns
floating-only eligibility and public `all` and `any` own exact BOOL eligibility and fixed false
gradient eligibility. All seven ordinary families own axis normalization, result-shape derivation,
and one-input provenance. Advanced floating reductions use `MultiAxisReductionAttrs`, while
variance and standard deviation use correction-bearing `StatisticalReductionAttrs`. Public
`argMin` and `argMax` separately own floating-or-integral
eligibility, fixed INT64 false-gradient results, explicit tie policy, axis-only shape derivation,
static selected-axis non-emptiness, shared candidate ordering, and one-input
provenance. Public masked `sum` and `mean` separately own floating/BOOL validation, ordinary
right-aligned broadcast-to-input validation, axis-removing result derivation, exact input type and
gradient eligibility, and `[input, mask]` provenance.

The ninth production family is `CumulativeScanKind`, an enum containing exactly `CUM_SUM` and
`CUM_PROD`. It identifies one-input cumulative addition or multiplication along the already
normalized axis in `CumulativeScanAttrs`. The attributes also select inclusive or exclusive output
and forward or reverse traversal. The family preserves logical positions; reverse traversal does
not reverse output order. Exclusive traversal emits additive zero for sum or multiplicative
positive one for product at its first visited position. Each kind's signature enforces
`CumulativeScanAttrs` and declares one input and one output. The kind and attributes contain no
Tensor, Shape, result descriptor, provenance, data-type policy, gradient rule, algorithm,
executable behavior, or backend support. Public `Tensor.cumSum` and `Tensor.cumProd` separately own
numeric validation, Shape-aware axis normalization, descriptor construction, fresh identity, and
one-input provenance without accumulating values.

The tenth production family is `SoftmaxKind`, an enum containing exactly `SOFTMAX` and
`LOG_SOFTMAX`. Both kinds pair with `SoftmaxAttrs`, whose sole component is an already normalized
non-negative axis. A normalization slice fixes every other logical coordinate and varies only
along that axis. `SOFTMAX` identifies positive normalized probabilities whose ideal slice total is
one; `LOG_SOFTMAX` identifies their natural logarithms, so exponentiating each ideal log-softmax
value recovers the corresponding softmax probability. The family preserves logical positions and
is distinct from both axis-contracting aggregate reduction and ordered-prefix scan semantics.
The shared signature enforces `SoftmaxAttrs` and declares one input and one output. The kind and attributes store no Tensor,
Shape, data-type policy, result descriptor, provenance, numerical algorithm, gradient, compiler
decomposition, storage, executable behavior, or backend support. Public `Tensor.softmax` and
`Tensor.logSoftmax` separately own floating validation, Shape-aware axis normalization, exact
Shape/type/eligibility retention in an unresolved descriptor, fresh identity, and one-input
provenance without calculating values.

The eleventh production family is `ContiguousKind`, whose sole `CONTIGUOUS` value is a
parameterless request for canonical dense row-major, zero-offset result geometry. The twelfth is
`ShapeTransformKind`, whose `RESHAPE` and `EXPAND` values pair with `TargetShapeAttrs`. The
thirteenth is `AxisTransformKind`, whose `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` values pair with
`PermutationAttrs` or `AxisTransformAttrs` as described under [axis transform](#axis-transform).
The fourteenth is `SliceKind`: `SLICE` pairs with `SliceAttrs` or `CropToShapeAttrs`, while
`SLICE_UPDATE` pairs with `SliceAttrs` or `CropToShapeAttrs`, as described under [slice](#slice), [Slice
Update](#slice-update), and [Target-relative crop](#target-relative-crop). The fifteenth and
sixteenth are `PadKind` and `TileKind`, whose sole `PAD` and
`TILE` values pair with `PadAttrs` and `TileAttrs` as described under [padding](#padding) and
[tiling](#tiling). The seventeenth is `TensorCompositionKind`, whose `CONCAT` and `STACK` values
pair with `CompositionAxisAttrs`, as described under [tensor composition](#tensor-composition).
The eighteenth is `SelectKind`, whose
sole `SELECT` value pairs with `SelectAttrs` as described under [scalar select](#scalar-select).
The nineteenth is `AxisGatherKind`, whose `GATHER` and `GATHER_ELEMENTS` values pair with
`IndexAxisAttrs` as described under [Gather](#gather) and [Gather Elements](#gather-elements).
The twentieth is `OneHotKind`, whose sole `ONE_HOT` value pairs with `OneHotAttrs` as described
under [one-hot encoding](#one-hot-encoding). The twenty-first is `GatherNdKind`, whose sole
`GATHER_ND` value pairs with `GatherNdAttrs` as described under [Gather-ND](#gather-nd). The
twenty-second is `AxisScatterKind`, whose `SCATTER_ELEMENTS` value pairs with
`ScatterElementsAttrs` and one `ScatterReduction`, while `SCATTER_ADD` pairs with
`IndexAxisAttrs`, as described under [Scatter Elements](#scatter-elements) and
[Scatter Add](#scatter-add).
The twenty-third is `ScatterNdKind`, whose sole `SCATTER_ND` value pairs with `ScatterNdAttrs`
and the shared reduction vocabulary, as described under [Scatter-ND](#scatter-nd). The
twenty-fourth is `WindowTransformKind`: `UNFOLD_AXIS` pairs with
`UnfoldAxisAttrs`, `FOLD_AXIS` with
`FoldAxisAttrs`, `UNFOLD2D` with `Window2dAttrs` or `Unfold2dAttrs`, and `FOLD2D` with
`Fold2dAttrs`, as described
under [window transform](#window-transform). The twenty-fifth is `LossKind`, whose sole
`MEAN_SQUARED_ERROR` value pairs with `MeanSquaredErrorAttrs` and the separate `LossReduction`
vocabulary, as described under [mean-squared error](#mean-squared-error--mse). These semantic
families do not by themselves
construct Tensors or define compiler, backend, or execution behavior.

### Padding

An implemented backend-independent constant-padding meaning represented by `PadKind.PAD` and
`PadAttrs`. For each normalized axis position `i`, `before[i]` and `after[i]` give the
non-negative numbers of constant-filled logical positions requested before and after the complete
input extent. The two equal-size lists are immutable ordered snapshots. Two empty lists are the
rank-zero scalar identity parameters.

For conceptual input `[10, 20]`, before `[1]`, after `[2]`, and constant `-1`, the requested
logical result is `[-1, 10, 20, -1, -1]`. The exact typed `ScalarValue` constant is retained by
reference, including raw floating signed-zero and NaN payload bits or exact integral/BOOL values.
The attributes do not know an input Tensor or Shape, so they do not match rank, add result extents,
check overflow, prove constant/input type equality, derive layout or provenance, materialize
values, define gradients, or execute padding.
`Long.MAX_VALUE` widths are therefore structurally valid. Public `Tensor.pad` separately requires
one width per input axis, clones the arrays, and applies canonical checked symbolic addition in
before-then-after order. Static extents fold to static results, dynamic extents retain exact
symbolic-extent expressions, and zero widths preserve the exact input Dimension reference. The
result preserves exact type and gradient eligibility and is a fresh unresolved storage-free
expression with normalized attributes and provenance `[input]`. It requires exact constant/input
type equality but does not bind or evaluate a symbolic extent, convert the constant, or
materialize values. Current package-private compiler autograd crops a supported floating PAD
cotangent by the exact before widths. See [Pad and tile
expressions](api/tensor-api.md#pad-and-tile-expressions) and [Pad and tile semantic kinds and
normalized attributes](api/tensor-api.md#pad-and-tile-semantic-kinds-and-normalized-attributes).

### Permutation

An implemented complete reordering of logical axes represented by `PermutationAttrs`. Its
`axes` list uses output-to-input order: `axes[i]` is the normalized input axis occupying output
position `i`. For input-axis labels `[rows, columns, channels]`, mapping `[1, 0, 2]` produces
output-axis labels `[columns, rows, channels]`.

The list size defines the permutation rank. Every value must lie in `[0, rank)` and occur exactly
once; an empty list is the rank-zero scalar identity. Construction validates elements in index
order before taking one immutable snapshot. The attributes do not know an eventual input Tensor
rank or construct a Shape, layout, storage view, provenance, gradient, or executable result by
themselves. Public `Tensor.permute` validates that rank, normalizes each negative raw axis once,
reorders exact Dimension references and resolved strides, preserves resolved element offset, and
records normalized attributes with provenance `[input]`. Unresolved layout remains unresolved. A
rank-two transpose convenience uses mapping `[1, 0]`. The resulting view descriptor is logical
metadata, not attached storage or a zero-copy guarantee. See [Permute and transpose
expressions](api/tensor-api.md#permute-and-transpose-expressions) and [Axis-transform semantic kinds
and attributes](api/tensor-api.md#axis-transform-semantic-kinds-and-attributes).

### Partition

An implemented immutable compile-time recipe for one non-empty consecutive region of an owning
compiled graph. The public `PlannedPartition(owner, nodeIds)` record stores one non-null
[`BackendId`](#backend-identity--backendid) and one immutable, non-empty, ordered snapshot of
unique non-null [`NodeId`](#nodeid) values. It retains the exact owner and node-ID references and
uses ordinary record value equality. It stores no graph values, partition ID, boundary edges,
transfer or memory facts, device, route, kernel, executable, or runtime state.

Current package-private generation consumes a `CompiledGraphModel` plus a complete
`Map<NodeId, BackendId>`. It validates all keys, graph coverage, and owner values before creating
output, then scans `graph.nodes()` in stored validated topological order. Consecutive nodes join
while their owners are equal by `BackendId.equals`; an owner transition starts a new partition.
The recipe retains the graph's exact node-ID references and the first node's exact owner reference.
The outer result and every membership list are immutable.

Adjacency here means consecutive stored node positions, not a graph edge. Equal-owner independent
nodes join when consecutive, while equal-owner graph-connected nodes remain separate when another
owner lies between them. Graph phase, fan-out, merges, repeated inputs, graph input/output values,
and multiple outputs from one producer do not independently split a run. A zero-node pass-through
graph has no partition. The generator is not public, and no current orchestration assembles its
complete owner map. Current package-private logical-memory derivation consumes the graph plus an
ordered complete partition list and produces public immutable per-value requirements, but public
orchestration, compiler artifacts, physical memory, prepare, and execution remain separately owned
or planned.

### Partition scoring configuration / `PartitionScoringConfig`

The implemented immutable config record that retains one non-null
`Optional<DeviceClass>` named `preferredDeviceClass`. An empty optional means only that no
explicit coarse device-class preference was supplied; it promises no default, fallback, equal
candidate scores, discovery, or successful ownership selection. A present optional records a soft
preference for the exact retained CPU or accelerator enum value after hard eligibility has
established the candidate set. It neither filters another eligible candidate, weakens a hard
[`BackendRequirement`](#backend-requirement--backendrequirement), nor guarantees that a candidate
of the preferred class wins.

Direct construction retains the exact optional reference and exact contained enum reference and
rejects a null optional with `NullPointerException("preferredDeviceClass")`. `neutral()` returns a
fresh empty value but does not select the later aggregate's default. `preferring(deviceClass)`
returns a fresh present value and rejects null with `NullPointerException("deviceClass")`.

This record is ranking input, not a scoring policy implementation. It does not evaluate
eligibility or candidates, calculate or compare scores, contain profile measurements, choose an
owner or device, select a backend route or kernel, compile or prepare a graph, inspect runtime
state, or execute work. Planning interpretation, planning-cost profiles, `CompileConfig`, compiler
consumption, owner-map assembly, and execution remain planned. The current partition and logical-
memory recipes and their internal generators do not make this config value perform planning.

### Planning cost model / planning cost profile

A **planning cost model** is backend-neutral compile-time estimation used to compare eligible
`BackendId` owners. A **planning cost profile** is a possible compact immutable input to that
model. Planning may interpret graph, operation, data-type, shape/size, transfer, materialization,
and boundary estimates, but it never interprets backend route, vector, thread, tile, chunk,
kernel, workload-cache, or other model-autotuning values.

This concept is separate from workload tuning caches and model-plan results. No stable cost
consumer or shared backend-neutral cost classification exists yet, so the planning cost-profile
contract remains Draft rather than defining a global backend average prematurely.

### Planning

The backend-neutral compile-time work that decides backend ownership, forms same-owner partitions,
and derives logical memory or materialization requirements from graph facts, configuration, and
declarative backend capabilities. Current public recipes cover capability questions, partitions,
and logical per-value memory relationships, while their end-to-end orchestration remains internal
or planned. Planning answers where a node or segment should run and which logical values its
regions consume or produce. It does not choose concrete kernels, perform backend-specific
lowering, allocate physical buffers, or inspect runtime residency. See [Partition
scoring](architecture/partition-scoring.md).

### Prepare

The lifecycle stage that turns immutable compile artifacts into reusable runtime-ready state.
Shared Prepare projects partition-scoped facts, coordinates backend analysis, assigns stable
buffer/workspace slots from exact declarations, invokes backend finalization against those slots,
and validates the result. Each concrete backend owns lowering, route selection, the opaque
analysis plan, and executable construction. Prepare creates analysis results, prepared partitions,
memory plans, schedules, and `PreparedExecution`. See [Runtime, Prepare, and Backend
Boundary](architecture/runtime-prepare-backend-boundary.md#the-staged-prepare-handoff).

### Prepare context: `PrepareContext`

The implemented public immutable projection that shared Prepare supplies for analysis of one
`PlannedPartition`. It retains the exact partition and backend-input references plus immutable
membership snapshots of supplied nodes, projected graph values, matching logical-memory
requirements, and exact typed logical-splat constants. List order and constant-map encounter
order preserve the supplying containers' deterministic order.

The projected nodes must match `PlannedPartition.nodeIds()` exactly and in order. Projected
values are unique by `ValueId`; every node input and output must resolve to one projected value;
and every projected value has exactly one descriptor-matching `LogicalMemoryRequirement`. The
last rule does not imply the converse that every otherwise valid projected value must occur in a
node input or output.

Every projected `TensorDescriptor` Shape is fully static. A constant key must be a projected
graph input: the value is referenced as a partition-node input and its logical-memory requirement
has no producing partition. The constant's exact `ScalarValue` data type must equal the projected
descriptor's data type. The map carries logical splats only; it contains no dense payload or
physical materialization.

`PrepareContext` has no `CompileArtifacts`, Compiler diagnostic, publication plan, dimension
binding, route choice, slot, allocation, executable, schedule, or per-run state. Its generic
backend input is one [opaque backend analysis role](#opaque-backend-analysis-roles), retained
without shared interpretation.

### Opaque backend analysis roles

The two implemented method-free marker interfaces that keep a concrete backend's analysis
boundary typed while hiding private fields from shared Prepare.
`BackendAnalysisInputs` marks one immutable backend-owned input object carrying resolved target
capabilities, applicable configuration, and any already-compatible cached decision.
`BackendPreparationPlan` marks the immutable selected lowering, route, and private configuration
returned by that backend for later finalization.

Shared Prepare retains exact objects implementing these roles but does not inspect, downcast,
copy, or interpret them. They are not generic string/object parameter maps. Neither role is an
executable, physical resource, assigned slot, mutable cache, measurement result, or per-run state.
The interfaces cannot enforce immutability themselves, so concrete implementations must expose
immutable state.

### Preparation resource requirement

The implemented sealed Prepare-owned declaration family for exact shared resource needs discovered
during backend analysis. `PreparationResourceRequirement.Buffer` associates one non-null projected
graph `ValueId` with an exact non-negative byte size and positive power-of-two byte alignment.
`PreparationResourceRequirement.Workspace` carries the same size and alignment facts under one
non-negative requirement ID that is local to a single backend partition analysis.

Zero byte size is valid. Buffer IDs must be unique among buffer declarations in one analysis
result, and workspace IDs must be unique among workspace declarations in that result. The two
identity domains remain separate. A buffer `ValueId` is a compile/prepare association, while a
workspace requirement ID is backend scratch identity; neither is a Runtime slot, address,
allocation, storage object, handle, lifetime interval, executable, or per-run binding.

### Backend partition analysis: `BackendPartitionAnalysis`

The implemented immutable result of the first backend preparation stage. It retains one exact
planned-partition reference, one exact backend-owned
`BackendPreparationPlan` reference, and an immutable ordered snapshot of exact shared buffer and
workspace declarations. Duplicate buffer `ValueId` values and duplicate analysis-local workspace
requirement IDs are rejected independently in encounter order.

Shared Prepare currently interprets the declarations in its package-private complete-set handoff
to assign Runtime slot types and construct final geometry, but it does not inspect private plan
state. The analysis itself is not a prepared executable,
slot assignment, physical allocation, resource handle, schedule, or per-run binding. The
constructing record cannot by itself prove which context produced it; the implemented
`BackendPartitionPreparer` collaboration requires the returned analysis to retain the exact
`PrepareContext.partition()` reference.

### Backend partition preparer / `BackendPartitionPreparer`

The implemented Prepare-owned generic collaboration through which one concrete backend analyzes a
validated `PrepareContext`. Its input type extends `BackendAnalysisInputs`, and its result plan
type extends `BackendPreparationPlan`, preserving their backend-specific relationship without
casts, raw `Object`, or a shared parameter map.

`analyze` must be deterministic from the complete context and return an immutable
`BackendPartitionAnalysis` with the exact context partition, opaque selected plan, and every exact
shared requirement. It performs no tuning measurement or search, cache mutation, physical
allocation, executable construction, slot assignment, scheduling, or Runtime execution. No
public concrete-backend implementation is composed yet. The current CPU module has an internal
implementation for one connected one-through-eight-occurrence fully static pointwise partition;
its exact supported matrix is described under [CPU portable route](#cpu-portable-route).

### Preparation resource assignment

The implemented sealed Prepare-owned association between one exact
[`PreparationResourceRequirement`](#preparation-resource-requirement), one exact Runtime slot,
and the dense list index of that slot's entry in a `PreparedMemoryPlan`. The `Buffer` variant uses
a `BufferSlot`; the `Workspace` variant uses a `WorkspaceSlot`. Assignment order matches the
owning analysis declaration order.

Assignments are immutable preparation results. They allocate, bind, own, access, or release no
physical resource and contain no per-run state. The current package-private complete-set handoff
shares one buffer slot across declarations of the same `ValueId`, using maximum declared size and
alignment, and gives each workspace declaration its own slot. This is conservative assignment,
not aliasing, lifetime analysis, reuse, residency, or materialization.

### Backend partition finalization

The implemented second backend preparation stage after shared slot assignment.
`BackendPartitionFinalization<P>` retains one exact typed `BackendPartitionAnalysis<P>`, the exact
shared `PreparedMemoryPlan`, and an immutable list of
[`PreparationResourceAssignment`](#preparation-resource-assignment) values in declaration order.
The owning backend implements `BackendPartitionFinalizer<P>` and returns one non-null immutable
`PreparedExecutable` retaining that exact plan.

Finalization does not authorize route reselection, undeclared resource requirements, physical
allocation, native or closeable prepared-resource acquisition, run-state construction, binding,
execution, scheduling, transfer, publication, measurement, or workload tuning-cache mutation.
An owning backend may perform an already-selected compatible executable-artifact load or
publication where its established cold-finalization contract permits it. The batch handoff is
package-private behind current public graph preparation. The current CPU module has one
package-private portable finalizer that resolves assigned resources and loads or generates an
already-selected artifact; it is not publicly composed and does not select or widen the current
exact fused FLOAT64 operation topology.

### Prepared partition / `PreparedPartition`

The implemented immutable Prepare-owned association between one exact `PlannedPartition` and one
exact finalized `PreparedExecutable`. Backend ownership remains available from the partition and
prepared-memory identity remains available from the executable, so the record duplicates neither
value. It owns no physical or per-run resource and is not a schedule step, `PreparedUnit`,
`PreparedSchedule`, or `PreparedExecution`.

### Prepared buffer assignment / `PreparedBufferAssignment`

The implemented immutable Prepare-owned translation from one compile graph `ValueId` to the exact
assigned Runtime `BufferSlot` and its non-negative dense index in
`PreparedMemoryPlan.buffers()`. Complete-set assignment produces one value in first-declaration
order for every distinct declared buffer value. It is schedule-construction context, not a
physical binding, representation, residency fact, or mutable runtime state.

### Graph preparation / `GraphPreparation`

The implemented stateless public Prepare operation that consumes one exact `CompileArtifacts`,
one typed positional `PartitionPreparation` per planned partition, and one explicit
`PreparedScheduleAssembler`. It constructs every partition `PrepareContext` before backend work,
coordinates ordered analysis and finalization, supplies one complete immutable
`PreparedScheduleContext` to the assembler, validates the returned recipe, and returns the exact
`PreparedExecution`.

Validation covers exact plan identity, caller-input order, initialized compile constants,
executable coverage and order, representation coordinates, and forward-then-gradient publication
occurrences. The operation invokes no creator, allocates no physical resource, executes no work,
discovers no backend, and retains no collaborator. Concrete backend-facing contexts contain no
Compiler aggregate; future Engine composition explicitly supplies the collaborators.

### Prepared execution / `PreparedExecution`

The implemented immutable Runtime record that forms the complete current reusable prepared root.
It retains exactly one `PreparedMemoryPlan` and one `PreparedSchedule`, requires both to be
non-null, and requires the schedule to report the exact same plan reference. Structurally equal
but separately constructed plans do not satisfy that association.

The record owns no closeable, persistent, or per-run resource and transfers no ownership. It has
no run or close method and does not create a `RunState`, consume a schedule, bind or execute an
invocation, allocate a representation, or publish a result. It may be read concurrently because
its current components are immutable; every later active logical run still requires its own
isolated mutable `RunState`. Future persistent prepared resources require an explicit ownership
and failure-lifecycle contract rather than an empty lifecycle on this record. See the
[Runtime API](api/runtime-api.md#current-prepared-execution).

### Prepared executable / `PreparedExecutable`

The implemented Runtime-owned abstract recipe boundary for one backend-prepared computation
region. It retains one exact `PreparedMemoryPlan` reference and immutable private snapshots of
ordered dense `BufferSelection` and `WorkspaceSelection` values plus one aligned `BufferAccess`
per buffer selection. A buffer selection addresses a
prepared buffer position and then one ordered representation position in a matching `RunState`;
a workspace selection addresses a prepared workspace position. These are list indices, not slot
numeric components. Empty and repeated selections are valid. `READ_ONLY` requires the old copy,
`WRITE_ONLY` declares only a successful write, and `READ_WRITE` does both. Runtime validates reads
before invalidating every copy of an output buffer and validates exact declared writes only after
backend success.

`bind` is the cold path. It requires an open state associated by exact plan reference identity,
resolves buffers before workspaces in selection order, and delegates explicit checked
compatibility to concrete-backend hooks. After all checks pass, the backend constructs a
[`BoundInvocation`](#bound-invocation--boundinvocation) with direct concrete typed fields. Binding
may allocate temporary JVM arrays and the invocation, but it acquires no auxiliary closeable
resource, changes no ownership, and performs no cleanup on failure.

The recipe and concrete subclass must be immutable and thread-safe so one instance can bind
concurrently to distinct run states. Backend analysis selects the implementation, while the current
backend-finalization contract constructs it only after shared slot assignment. The executable
does not own selected representations and supplies no allocation, transfer, residency, schedule,
publication, result, or cleanup lifecycle.

### Prepared buffer transfer / `PreparedBufferTransfer`

The implemented immutable Runtime recipe for one explicit transfer between two distinct,
already-created representation positions of the same prepared buffer. It retains one exact
`PreparedMemoryPlan`, one dense buffer index, and dense source and destination representation
indices. Its cold `bind` operation requires a matching open `RunState`, validates both positions,
checks source then destination compatibility through backend hooks, and returns a matching
[`BoundBufferTransfer`](#bound-buffer-transfer--boundbuffertransfer) whose concrete subclass has
direct typed references.

The recipe is reusable and thread-safe when its backend subclass is immutable. It owns no
representation and performs no allocation, transfer, validity change, route search, fallback,
schedule traversal, or publication. Transfer to an equivalent already-created destination is
materialization; there is no second materialization recipe or schedule kind.

### Prepared publication / `PreparedPublication`

The implemented immutable Runtime recipe that names one already-created buffer representation as
one ordered result position. It retains one exact `PreparedMemoryPlan` and three dense zero-based
Runtime coordinates: buffer position, representation position within that buffer for a run, and
result position. These positions are not `BufferSlot.value()`, `TensorId`, `ValueId`, graph
membership, or compiler publication roles. Prepare will later translate compiler logical roles
into these coordinates.

Cold `bind` requires one open `RunState` associated with the same exact plan reference, validates
the per-run representation position, and resolves that exact physical representation once. The
returned [`BoundPublication`](#bound-publication--boundpublication) retains it directly.
Construction and binding perform no validity change, publication, physical access, transfer,
conversion, allocation, backend call, or ownership transition. The recipe is reusable and may
bind concurrently to distinct states.

### Prepared schedule / `PreparedSchedule`

The implemented Runtime-owned immutable recipe that orders already-prepared work against one
exact `PreparedMemoryPlan`. The schedule retains the plan by reference identity and an immutable
ordered snapshot of `Step` occurrences. Every step reports that same exact plan. Empty schedules
and repeated exact or equal occurrences are valid; each list position is one occurrence rather
than a new ownership relationship.

The nested `Step` interface is sealed and currently permits
`RepresentationCreationStep(PreparedRepresentationPlan)` and
`ExecutionStep(PreparedExecutable)`, `BufferTransferStep(PreparedBufferTransfer)`, and
`PublicationStep(PreparedPublication)`. The creation variant retains the sole optional cold-setup
prefix and may occur only at index zero. Publication variants form a final suffix whose result
indices are dense `0..N-1` in encounter order. Distinct positions may intentionally name the same
exact representation. The other variants retain their exact recipes. All derive their plan from
the retained recipe. The family has no ID, kind enum, generic payload,
visitor, binding method, or execution method. No `PreparedUnit` is present: occurrence position
supplies order, while each variant supplies the stable recipe and exact-plan invariant.

Construction scans supplied steps in encounter order, compares plan references with identity,
and snapshots through `List.copyOf` only after every step passes. It owns only that list snapshot
and closes nothing. Empty, executable-only, transfer-only, and zero-publication schedules remain
valid; executable or transfer occurrences may repeat. `publicationCount()` reports the suffix
length. A schedule may be traversed concurrently while distinct runs are prepared, but this does
not make per-run bound actions thread-safe. Current schedule construction does not invoke a
creator, bind, execute, or publish work, allocate a representation, or transfer ownership.
Transfer and publication steps only make their current recipes reachable.

### Bound invocation / `BoundInvocation`

The implemented Runtime-owned abstract lifecycle guard for one backend invocation bound to one
exact open [`RunState`](#run-state--runstate). A concrete backend subclass stores the compatible
buffer and workspace representations directly in concrete typed fields established during cold
binding. The invocation strongly retains the state but owns or closes neither the state nor any
representation or immutable prepared resource.

`execute()` first rejects a closed state, then delegates once to the backend's `executeBound()`.
The hot call performs no resource or slot lookup, compatibility test or cast, graph inspection,
backend discovery, route/configuration selection, allocation, transfer, residency decision,
publication, tuning, tracing, retry, fallback, or cleanup. Sequential calls while open are
permitted, and backend `RuntimeException` or `Error` values propagate unchanged. An invocation is
not thread-safe and must not be executed concurrently or raced with state closure.

### Bound buffer transfer / `BoundBufferTransfer`

The implemented Runtime-owned abstract validity guard for one backend transfer bound to one exact
open [`RunState`](#run-state--runstate). Its backend subclass stores direct concrete source and
destination representation fields established by `PreparedBufferTransfer.bind`. The shared base
retains only that state and the dense coordinates needed for validity orchestration; it owns and
closes no state or representation.

`execute()` makes an already-valid destination a no-op without requiring a valid source. For an
invalid destination, it requires the source valid, delegates once to `executeTransfer()`, and
marks only the destination valid after successful return. A backend `RuntimeException` or `Error`
propagates unchanged and leaves every Runtime validity bit unchanged. Source and unrelated-copy
validity are never changed. The bound action is not thread-safe and must not race another
validity transition or state closure. It performs no representation lookup, compatibility cast,
allocation, backend or route discovery, retry, invalidation, publication, or cleanup.

### Bound publication / `BoundPublication`

The implemented per-run one-shot publication occurrence created only by
[`PreparedPublication.bind`](#prepared-publication--preparedpublication). It retains one exact
`RunState`, prepared recipe, and directly resolved physical buffer representation. Distinct
ordered result positions may use separate bound occurrences that retain the same exact
representation.

`publish()` first requires the state open, rejects an already-completed occurrence, and requires
the selected copy valid at that exact moment. Success changes only the occurrence's local flag.
Failure changes no flag, validity, ownership, or resource. Publication performs no representation
lookup, backend callback, transfer, conversion, allocation, copy, fallback, or cleanup.
`isPublished()` remains available after state closure. The bound occurrence is not thread-safe and
must not race validity changes, execution, transfer, result construction, or closure.

### Producer / `TensorProducer`

Implemented immutable pre-capture identity for one public Tensor-expression occurrence. A
`TensorProducer` retains one exact backend-independent [`Operation`](#operation), an immutable
ordered snapshot of exact input-Tensor references, and an immutable non-empty ordered snapshot of
exact output [`TensorDescriptor`](#tensor-descriptor) references. Repeated input or descriptor
references are preserved when the operation signature permits the final counts. Output count is
derived from the descriptor-list size rather than stored independently.

Separate producers use ordinary object identity even when operation, inputs, and descriptors are
structurally equal. This is occurrence identity in the public expression model, not an allocated
producer ID, `NodeId`, `ValueId`, graph membership, graph capture, or a common-subexpression key.
The producer retains one canonical exact Tensor wrapper for every output slot, and
`output(outputIndex)` returns the retained wrapper without allocation or reconstruction. Each
wrapper's indexed provenance points back to the same producer, producing one immutable, safely
published `Tensor -> TensorProvenance -> TensorProducer -> outputs -> Tensor` cycle. Retaining one
result may retain siblings; the cycle owns no external resource and is ordinarily collectable
when unreachable. The producer still gains no label, storage, value, gradient, compiler, backend,
runtime, or execution state. See [Expression producers and indexed
provenance](api/tensor-api.md#expression-producers-and-indexed-provenance).

### Provenance

Implemented immutable association between one public [`Tensor`](#tensor) and one zero-based output
of one exact [`TensorProducer`](#producer--tensorproducer). The two-component
`TensorProvenance(producer, outputIndex)` record validates the index against the producer's output
count and derives its exact operation, immutable ordered inputs, and selected exact descriptor
from that producer. Single-output expressions use output index zero. A derived Tensor must retain
the exact descriptor object selected by provenance; structural descriptor equality is
insufficient.

Record equality and hashing combine ordinary producer identity with the output index. Two
positions of one producer differ, and the same index on separately invoked structurally equal
producers also differs. Provenance is not intermediate representation (IR), graph membership,
graph capture, or executable behavior. It contains no graph-local `NodeId` or `ValueId`, does not
validate cycles or graph structure, and does not change when Tensor host storage is replaced,
cleared, or becomes dead. A later compiler owns traversal and conversion into immutable graph
records. See [Public Tensor state](api/tensor-api.md#public-tensor-state).

The implemented binary Tensor methods create provenance whose operation uses the exact matching
`BinaryArithmeticKind` and `NoOperationAttrs.INSTANCE`, and whose two input positions preserve the
receiver as left and argument as right. Binary comparison methods use the same exact ordered input
contract with `BinaryComparisonKind`, including for symmetric equality and inequality, while
producing a storage-free non-differentiable BOOL result. Boolean logical AND and OR likewise
retain exact ordered receiver/argument inputs with `BooleanLogicalKind`; NOT retains exactly the
receiver as its one input. The implemented floating unary methods use the exact matching
`UnaryElementwiseKind` and canonical no-attributes value, also with exactly the receiver as their
one input. Floating-classification methods instead use the exact matching
`FloatingClassificationKind` with the same one-input parameterless provenance while constructing
fixed BOOL metadata. Static `Tensor.where` uses `WhereSelectionKind.WHERE` and retains exact ordered inputs
`[condition, ifTrue, ifFalse]`, including repeated branch references. These construction paths do
not change provenance's general role or make it graph membership. `Tensor.cast` uses
`CastKind.CAST`, retains its exact target in `CastAttrs`, and records exactly the receiver as its
one immediate input, including in same-type and chained requests. `Tensor.cumSum` and
`Tensor.cumProd` use the matching `CumulativeScanKind.CUM_SUM` or `CUM_PROD`, retain the normalized
axis and exact mode flags in `CumulativeScanAttrs`, and likewise record exactly the receiver as
their sole input.
`Tensor.softmax` and `Tensor.logSoftmax` use the exact corresponding `SoftmaxKind`, retain the
normalized axis in `SoftmaxAttrs`, and record exactly the receiver as their sole input.
`Tensor.meanSquaredError` uses `LossKind.MEAN_SQUARED_ERROR`, retains its explicit reduction in
`MeanSquaredErrorAttrs`, and records ordered inputs `[prediction, target]` under one fresh
single-output producer.
`Tensor.contiguous()` uses `ContiguousKind.CONTIGUOUS` and `NoOperationAttrs.INSTANCE`, and records
exactly the receiver even when it already has canonical dense layout. That provenance does not
imply that a copy or materialization occurred.
`Tensor.permute` and `Tensor.transpose()` use `AxisTransformKind.PERMUTE`, retain their complete
normalized output-to-input mapping in `PermutationAttrs`, and record exactly the receiver as their
sole input. Resolved view layout in the result does not imply that storage was attached, aliased,
copied, or materialized.
`Tensor.expandDims` and `Tensor.squeeze` use `AxisTransformKind.EXPAND_DIMS` and `SQUEEZE`,
respectively, retain one normalized position in `AxisTransformAttrs`, and record exactly the
receiver as their sole input. Conditional resolved view geometry still does not imply attached or
aliased storage, a copy-free route, or execution.
`Tensor.slice`, `Tensor.sliceByLength`, both `Tensor.sliceAxis` overloads, and `Tensor.flip` use
`SliceKind.SLICE`, retain normalized finite start/length/signed-step sequences on distinct axes in
`SliceAttrs`, and record exactly the receiver as their sole input. Each success has one
identity-distinct producer, one output descriptor, and provenance output index zero, including
identity, dynamic selected-extent length-defined extraction, and multi-axis flip.
Conditional resolved positive-step view geometry and unresolved negative-step geometry imply
neither attached or physically aliased storage, copying, materialization, nor execution.
Both `Tensor.sliceUpdate` overloads use `SliceKind.SLICE_UPDATE` with ordered inputs
`[base, update]` and retain the exact base Shape in one unresolved descriptor. The array form uses
`SliceAttrs`; the target-relative form uses `CropToShapeAttrs`, whose target is the exact update
Shape and whose prefix is the exact caller Shape. `Tensor.cropToShape` uses `SliceKind.SLICE` with
`CropToShapeAttrs` and exact input `[input]`; there the target is the exact result Shape. These
forms create one fresh producer at output index zero without mutating inputs or attaching storage.
Static `Tensor.concat` and `Tensor.stack` retain immutable ordered snapshots of their exact input
Tensor references in provenance with matching `TensorCompositionKind` and normalized
`CompositionAxisAttrs`. Each `Tensor.unstack` output instead carries its own `SELECT` operation,
`SelectAttrs` with an increasing coordinate, and exactly the source Tensor as input. Each is an
independent one-output producer with provenance index zero. Those individual
origins distinguish outputs without establishing one shared producer occurrence, grouped result
identity, or graph output slot.

### Forward publication binding

The implemented immutable `ForwardPublicationBinding` record associates one
[`TensorId`](#tensorid) with one graph-local [`ValueId`](#valueid). It is standalone Model data
used for requested forward results by the current compiler-owned `PublicationPlan`, not a
component of `CompiledGraphModel`. A binding cannot by itself prove that its value belongs to a
particular graph, and it carries no public `Tensor`, gradient role, publication policy or target,
storage, backend, or runtime state.

### Publication plan

The current compiler-owned output-only graph context for ordered publication bindings.
`PublicationPlan` retains the exact final `CompiledGraphModel` and immutable membership snapshots
exposed by `forwardBindings()` and `gradientBindings()`. `ForwardPublicationBinding` values pair
requested Tensor IDs with the forward graph-output prefix. `GradientPublicationBinding` values
pair derivative order and stage-local target position with target Tensor ID and final gradient
`valueId()`. A target ID may occur once in each derivative order, and gradient values may repeat
or equal a forward value.

The plan validates every binding against the exact graph and requires the complete graph-output
boundary to be the forward prefix followed by each previously unseen gradient value in first role
order. It proves graph membership and boundary order only. It contains no publication delivery
policy, alias/copy decision, Tensor gradient state, storage, backend behavior, preparation, or
runtime publication.

### Residency

Runtime knowledge that an explicitly prepared physical representation exists and is bound to one
run. The current foundation uses structural residency: every buffer and workspace representation
bound during complete state construction remains resident until `RunState` closure, with no lazy
creation, eviction, replacement, or removal. [`Buffer validity`](#buffer-validity) is a separate
fact about whether a resident buffer copy contains the logical slot value; workspace scratch has
no such fact. Residency belongs to one `RunState`; it is not shared prepared state and does not
belong to the model `Tensor`, compile-time planning, or `LayoutDescriptor`. It implies no
automatic coherence or hidden write-back.

### Run / runtime

**Run** is one invocation of a `PreparedExecution`. It creates exactly one `RunState` for the
complete heterogeneous logical run, borrows caller inputs, follows explicit prepared resource and
schedule work, executes cold-bound prepared units, manages residency/materialization/transfer,
publishes results, and cleans resources still owned by the run. **Runtime** is the module and
machinery that performs this work. Run does not optimize graphs, discover backends, lower
partitions, or choose kernels. See [Lifecycle](architecture/lifecycle.md#run-lifecycle).

### Prepared execution runner / `PreparedExecutionRunner`

The implemented public stateless Runtime orchestrator for one synchronous invocation of an
immutable `PreparedExecution`. One call validates dense borrowed caller inputs, creates one
isolated `RunState`, cold-binds every executable, transfer, and publication occurrence before the
first action, and traverses direct bound references in schedule order. Successful completion
returns the whole-state `RunResult` lease. Any post-creation failure closes the state once,
preserves the original unchecked failure, and suppresses a distinct cleanup failure.

Executable traversal validates declared reads before invalidating all resident copies of every
declared output buffer. It validates only exact declared written copies after backend success;
failure leaves all output copies invalid. The runner performs no graph inspection, backend
discovery, route selection, physical allocation, retry, Trace emission, or output conversion.
Separate calls may run concurrently because each has isolated mutable state; a single call is
synchronous and must not race its state or bound actions.

### Runtime profiling

Passive observation of actual prepared execution. `modules/runtime` owns the execution facts and
later translates selected observations into typed `modules/trace` data-transfer objects. Runtime
profiling does not compare tuning candidates, select or mutate settings, or feed hidden state back
into the current run. Concrete runtime-profile payloads and emitters remain planned.

### Run result / `RunResult`

The implemented closeable Runtime lease over one complete open [`RunState`](#run-state--runstate)
after all dense ordered bound publication occurrences have completed. Construction requires every
occurrence to belong to that exact state, match its list position by result index, and report
successful publication. It privately snapshots the direct representation references in result
order, preserving intentional aliases, but exposes only result count and lifecycle state. An
empty result is valid.

Successful construction semantically transfers responsibility for closing the whole state to the
result without changing any individual `RunResourceOwnership`. Constructor failure or partial
publication transfers nothing and closes nothing. `close()` delegates to the state's idempotent,
ownership-sensitive reverse cleanup; borrowed representations remain caller-owned for the entire
lease. The result exposes no representation, storage, Tensor, value, or `RunState` accessor and is
not thread-safe. Public output access remains planned; `PreparedExecutionRunner` currently
assembles this lease after successful traversal.

### Run state / `RunState`

The implemented public final lifecycle owner for physical representation bindings in one complete
logical run. It spans all backend partitions in that run; it is not one state per backend or
partition. Construction retains the exact immutable `PreparedMemoryPlan`, snapshots supplied list
structure into private arrays in plan encounter order, and retains every exact binding and
representation reference. Dense access indices are entry-list positions, not
`BufferSlot.value()` or `WorkspaceSlot.value()`.

Each buffer position contains one or more ordered bindings, and each workspace position contains
one run-owned representation. Binding presence is structural residency until closure. One
independent validity bit per buffer copy starts true for `BORROWED` and false for `RUN_OWNED`;
query and mutation are explicit constant-time operations with no physical side effect. Workspaces
have no logical validity. Successful construction transfers cleanup responsibility for
`RUN_OWNED` buffers and all workspaces; failed construction transfers nothing. Closing marks the
state closed first, skips borrowed buffers, attempts owned resources once in deterministic reverse
order, and preserves the first `RuntimeException` or `Error`. Later failure objects distinct from
that primary object are suppressed in encounter order. If another resource throws the same exact
primary object, that occurrence is not self-suppressed because Java forbids self-suppression; it
does not replace the primary failure or interrupt the remaining cleanup. Representation access is
closed after that transition, but the exact plan and counts remain inspectable. Repeated close is
inert.

One state is not thread-safe. Concurrent runs have distinct states and distinct run-owned
representations and validity arrays, though they may share the immutable plan. A current
`PreparedExecutable` may cold-bind selected representations while the state is open, and the
resulting current `BoundInvocation` retains this exact state and rejects execution after closure.
A current `PreparedPublication` may likewise cold-bind one selected representation into a
`BoundPublication`; after publication completes, a `RunResult` leases cleanup of the complete
state. Bound actions own neither the state nor its resources. Package-private cold creation now
constructs states from one `PreparedRepresentationPlan`, validates all caller inputs before
callbacks, and rolls back created results on failure. The state itself does not implement physical
access, transfer/materialization, schedule execution, publication traversal, output access, or a
runner.

### Sample domain

For a class-axis loss, the sample domain is the coordinate space remaining after the class axis is
removed. Its logical element count is the denominator for current dense categorical-cross-entropy
`MEAN`. For index categorical cross entropy, the exact target Shape represents this domain and
`MEAN` divides by its non-ignored target count. Neither denominator is the class count or logits
element count; the dense denominator is also not positive-target count or target-weight sum.

### Shape

An immutable ordered collection of [dimensions](#dimension) describing the logical size of a
tensor along each axis. The number of dimensions is the shape's rank; a rank-0 shape represents a
scalar. A shape describes extents only: it does not define strides, storage, layout, backend
support, or runtime allocation. Its total element count is known only when every dimension is
static. A prefix Shape used by [target-relative crop](#target-relative-crop) applies that same
extent model to the number of logical positions preceding a region on each axis; it is not an
index Tensor or a storage offset. See [Shapes and
dimensions](api/tensor-api.md#shapes-and-dimensions) and [symbolic extent
expressions](#symbolic-extent-expression).

### Symbolic extent expression

An implemented immutable model value that retains a small exact formula or a constrained unknown
for one Shape axis. `DimensionExpressions` is the only public construction boundary. It provides
checked addition, signed constant offset, multiplication by a non-negative constant,
Dimension-to-Dimension multiplication, floor or ceiling division by a positive constant, and
identity-based unknown construction.

Exact formulas use structural equality. Canonical linear combinations have positive dimension
coefficients and one signed constant offset: nested sums flatten, static terms fold, repeated
terms combine, and addition order is not semantic. A **canonical symbolic product** has one
positive constant coefficient and a non-empty immutable map from complete Dimension factors to
positive exponents. It folds static factors, flattens nested products, combines repeated factors,
and makes factor order non-semantic without distributing across sums. Zero and one use their
ordinary identities, and checked coefficient or exponent overflow fails rather than wrapping.
Floor and ceiling division retain explicit dividend and divisor nodes. A constrained unknown has
an inclusive non-negative minimum and an optional inclusive maximum Dimension. It deliberately
uses object identity, so only reuse of the same unknown proves equality.

Expression construction performs local checked arithmetic and canonicalization only. It does not
bind named dimensions, evaluate concrete sizes, solve graph-wide constraints, construct Tensor
operations, or add compiler, prepare, runtime, storage, or backend state. See [Symbolic extent
expressions](api/tensor-api.md#symbolic-extent-expressions).

### Scalar select

An implemented backend-independent scalar-index meaning represented by `SelectKind.SELECT` and
`SelectAttrs(axis, index)`. The normalized zero-based `axis` identifies one existing source axis,
and the normalized zero-based `index` fixes one coordinate on it. That axis is removed from the
conceptual result. For source Shape `[2, 3, 4]`, axis `1` and index `2` therefore mean conceptual
result Shape `[2, 4]`.

The attributes reject negative axis and index values, checking the axis first, but store no input
Shape, rank, or selected-axis extent. They consequently cannot prove that an axis exists or that
an index is in bounds. The family signature enforces `SELECT` with `SelectAttrs` and declares one
input and one output.

Public `Tensor.select(axis, index)` supplies that input-aware expression boundary. It normalizes a
positive or negative axis against input rank. A static selected extent normalizes one negative
index and bounds-checks the resulting coordinate; a dynamic selected extent accepts a non-negative
coordinate unchanged with its upper bound deferred and rejects a negative coordinate. The fresh
result removes exactly that axis, retains every unaffected exact Dimension reference, preserves
data type and gradient eligibility, and records exact `[input]` provenance. Rank-one selection
produces the scalar Shape. Resolved input geometry with a non-empty result removes the selected
stride and advances the element offset in new logical view metadata; unresolved input and empty
results remain unresolved. No host storage is attached, and view metadata is not a physical alias
promise.

Scalar select differs from conditional `WHERE`, which chooses between branch values at
corresponding positions, and general `SLICE`, which selects half-open intervals without removing
an axis. Public unstack composes ordered scalar-select occurrences. Tensor-index gather instead
uses one or more tensors to supply indices.
Current package-private compiler autograd scatters a supported floating SELECT cotangent into an
input-shaped exact typed zero at the restored axis coordinate. Compiler canonicalization,
materialization, backend lowering, and execution remain planned. See [Scalar select semantic kind and
attributes](api/tensor-api.md#scalar-select-semantic-kind-and-attributes).

### Sigmoid linear unit / SiLU

A smooth floating activation with mathematical target `x * sigmoid(x)`, equivalently
`x / (1 + exp(-x))`. Synaptik represents it as the parameterless first-class
`UnaryElementwiseKind.SILU` semantic identity and exposes the canonical `Tensor.silu()` spelling;
the public API has no `swish` alias. Its continuous extension maps negative infinity to negative
zero, preserves signed zero, maps positive infinity to positive infinity, and propagates NaN.
Current Tensor construction records type-, Shape-, eligibility-, and provenance metadata without
executing the function or defining compiler decomposition, gradients, or backend algorithms.

### Slice

An implemented backend-independent logical selection represented by `SliceKind.SLICE` with
`SliceAttrs`. The attributes use four equal-size parallel lists. Entry `i` selects the half-open
coordinate sequence on normalized input axis `axes[i]`: coordinate `k` is
`starts[i] + k * steps[i]` for `0 <= k < lengths[i]`. Steps are signed and non-zero. Unlisted axes
retain their complete logical coordinate range.

Starts, lengths, and steps use `long`, while axes use `int`. The lists are immutable snapshots;
their values and entry order define record equality. Starts, lengths, and axes are non-negative,
axes are distinct, empty entries use canonical start zero, and a non-empty entry must have a
representable non-negative final coordinate. Four empty lists describe normalized identity
semantics. The value stores no input Shape, raw bounds, or direction-dependent exclusive-end
sentinel, so rank, selected-dimension staticity, and directional normalization belong to the
public Tensor expression boundary.

For selected extent `D`, the original public request is directional and half-open. A negative raw bound
adds `D` once. Positive-step bounds clamp to `[0, D]`; a negative-step start clamps to
`[0, D - 1]` and its end to `[-1, D - 1]`. Raw `-1` therefore means relative coordinate
`D - 1`, not the boundary before zero. On conceptual values `[0, 1, 2, 3, 4]`,
`sliceAxis(0, 1, 5, 2)` selects `[1, 3]`, while `sliceAxis(0, 4, -6, -1)` and `flip(0)` both select
`[4, 3, 2, 1, 0]`. The former entries normalize to starts `[1]`, lengths `[2]`, steps `[2]`; the
latter normalize to starts `[4]`, lengths `[5]`, steps `[-1]`.

The separate `sliceByLength` request consumes already normalized non-negative starts plus explicit
finite non-negative lengths. It proves non-negative first and final coordinates and every static
upper bound. If a selected Dimension is unresolved, only the upper-bound comparison is deferred;
the selected result Dimension is still the requested static length and Model retains no constraint
object. A zero length selects no coordinate, checks no input bound, and stores canonical start
zero.

Raw-bound slicing requires every selected Dimension static; length-defined slicing does not.
Unselected Dimensions retain exact references. Empty arrays and empty flip axes are fresh explicit
identity occurrences. Resolved non-empty all-positive requests derive checked start-adjusted,
step-multiplied logical view geometry. Unresolved input, empty results, and every negative-step
request remain layout-unresolved because the current descriptor forbids negative strides. This
boundary promises neither a physical alias nor a copy.

Every public slice or flip result preserves exact data type and gradient eligibility, remains
fresh, unlabeled, and storage-free, and records one `SliceKind.SLICE` producer with exact input
`[input]`, one output descriptor, and provenance output index zero. The three-argument
`sliceAxis` uses step one; the four-argument form uses an explicit signed step; `flip` places all
requested axes into one operation. There is no `SLICE_AXIS` or `FLIP` kind. These methods do not
read values, capture or canonicalize a graph, materialize storage, lower a backend or ONNX
operation, or execute selection. Current package-private compiler autograd scatters a supported
floating SLICE cotangent into an input-shaped exact typed zero, including length-defined regions
whose selected source extent remains unresolved after occurrence-owned constraint retention. See
[Slice expressions](api/tensor-api.md#slice-expressions)
and [Slice semantic kind and normalized
attributes](api/tensor-api.md#slice-semantic-kind-and-normalized-attributes).

### Slice Update

An implemented backend-independent functional replacement represented by
`SliceKind.SLICE_UPDATE` with ordered inputs `[base, update]`. With `SliceAttrs`, entry `i` maps
update coordinate `k` to base coordinate `normalizedStart[i] + k * steps[i]` on one distinct
normalized axis. The selected update Dimension must be static and supplies `lengths[i]`;
unselected update Dimensions must exactly equal the base Dimensions. The result retains the exact
base Shape and values outside the mapped Cartesian region.

A negative start adds one static base extent once. Static first/final coordinates must fit; an
unresolved base axis accepts only a non-negative start and defers its upper-bound proof. Empty
arrays mean explicit full-Shape replacement, while a selected zero update extent maps no
coordinates. The mapping is functional replacement, not mutation, addition, reduction, or an
operation-specific backward kind.

With `CropToShapeAttrs`, the exact update Shape is the target region and the exact caller prefix
Shape identifies the non-negative logical extent preceding it on each axis. Every fully static
axis proves `prefix + update <= base` with checked arithmetic; any axis containing an unresolved
base, prefix, or update extent defers its whole fit. The result still retains the exact base Shape.

Both public `Tensor.sliceUpdate` forms accept all current data types with exact base/update type
equality, combine their gradient eligibility, leave result layout unresolved, and create one fresh,
unlabeled, storage-free result at provenance index zero. They read no values and define no
lowering, backend, or execution behavior. Current package-private compiler autograd masks the
supported floating base cotangent and extracts the supported floating update cotangent for both
`SliceAttrs` and target-relative `CropToShapeAttrs` after exact constraint-aware preflight. See
[Slice update and
target-relative crop expressions](api/tensor-api.md#slice-update-and-target-relative-crop-expressions).

### Target-relative crop

An implemented target-relative region represented by `CropToShapeAttrs(targetShape, prefixShape)`.
For `SliceKind.SLICE`, the target Shape is the exact extraction result Shape. For
`SliceKind.SLICE_UPDATE`, it is the exact update-region Shape and the result retains the base
Shape. The prefix Shape gives the non-negative number of logical positions preceding the region
on each axis, so axis `i` identifies `[prefix[i], prefix[i] + target[i])`. Both exact immutable
Shape references are retained and may contain [symbolic extent
expressions](#symbolic-extent-expression).

Public `Tensor.cropToShape` requires input, target, and prefix ranks to match. A fully static
prefix-plus-target bound is checked locally with exact arithmetic; an inequality involving any
unresolved Dimension remains a later binding/execution obligation. The operation does not infer,
bind, clamp, shift, truncate, pad, or reverse a region. Its fresh result retains input type and
gradient eligibility, leaves layout unresolved, and records exact `[input]` provenance without a
view or copy promise. The Shape-based `Tensor.sliceUpdate` uses the same exact region language for
placement and ordered `[base, update]` provenance. Model stores no deferred constraint object;
current compiler inference, preflight, and first-order autograd retain and adopt the placement
form. See
[Slice update and target-relative crop
expressions](api/tensor-api.md#slice-update-and-target-relative-crop-expressions).

### Softmax / log-softmax

Two implemented, shape-preserving normalization meanings represented by `SoftmaxKind.SOFTMAX`
and `SoftmaxKind.LOG_SOFTMAX`. Both pair with `SoftmaxAttrs`, which carries one already normalized
non-negative axis. A normalization slice contains positions that vary along that axis while every
other logical coordinate remains fixed.

For slice values `x_i`, ideal softmax produces positive normalized probabilities
`exp(x_i) / sum_j(exp(x_j))` whose slice total is one. Ideal log-softmax produces the natural
logarithm of each corresponding softmax probability. For `[1, 2, 3]`, those meanings are
approximately `[0.09003057, 0.24472847, 0.66524096]` and
`[-2.40760596, -1.40760596, -0.40760596]`; exponentiating the latter values recovers the former.
The example states ideal mathematics, not a finite-precision algorithm.

The semantic values do not retain a Tensor or Shape, normalize caller-facing negative axes,
construct descriptors or provenance, define data-type eligibility or gradients, select compiler
decomposition, report backend support, or execute normalization. Public `Tensor.softmax` and
`Tensor.logSoftmax` now add floating input validation, Shape-aware caller-axis normalization,
exact Shape/type/eligibility retention with unresolved layout, and fresh one-input provenance.
They still calculate none of the example values. Current package-private compiler autograd uses
the exact forward output `y`: softmax constructs `y * (g - sum(g * y, axis, true))`, while
log-softmax constructs `g - exp(y) * sum(g, axis, true)`. This compiler behavior defines no
forward numerical algorithm, backend route, or execution result. See [Softmax
expressions](api/tensor-api.md#softmax-expressions) and [Softmax
semantic kinds and attributes](api/tensor-api.md#softmax-semantic-kinds-and-attributes).

### Layer normalization

Implemented shape-preserving normalization that standardizes every non-empty trailing slice by
subtracting its mean and dividing by `sqrt(populationVariance + epsilon)`. A
[`normalized Shape`](#normalized-shape) identifies the trailing axes. The no-affine form records
one input; the affine form then applies explicit scale and bias and records ordered inputs
`[input, scale, bias]`. Both forms produce exactly one result and no public saved statistic.

`LayerNormKind.LAYER_NORM` has distinct `LayerNormAttrs` and `AffineLayerNormAttrs` signatures so
the exact valid input counts are one and three, never two. Current `Tensor.layerNorm` constructs
metadata and provenance but does not calculate statistics, capture a graph, select an algorithm,
or execute. Current package-private compiler autograd supports the input in both forms and the
scale/bias roles in the affine form. It rebuilds the population statistics through ordinary
Tensor expressions over the exact trailing axes. See [Layer-normalization
expressions](api/tensor-api.md#layer-normalization-expressions).

### RMS normalization

Implemented shape-preserving root-mean-square normalization that divides every value in a
non-empty trailing slice by `sqrt(uncenteredMeanSquare + epsilon)`. It does not center values,
calculate variance, apply a correction, or accept bias. The no-scale form records `[input]`; the
scaled form records `[input, scale]`, with scale Shape exactly equal to the
[`normalized Shape`](#normalized-shape). `RmsNormKind.RMS_NORM` and one
`RmsNormAttrs(normalizedShape, epsilon)` input-range signature represent both safe cardinalities
and exactly one output.

Current `Tensor.rmsNorm` constructs metadata and provenance but does not calculate the root mean
square, capture a graph, create saved statistics, select an algorithm, or execute. Current
package-private compiler autograd supports the input and optional scale roles through ordinary
population-mean Tensor expressions.
See [RMS-normalization expressions](api/tensor-api.md#rms-normalization-expressions).

### Batch normalization / batch-normalization inference and training

Batch normalization uses statistics associated with a channel to center and scale values. The
implemented `BatchNormKind.BATCH_NORM_INFERENCE` meaning is one stateless five-input occurrence
with ordered roles `[input, scale, bias, runningMean, runningVariance]`, one output, and
`BatchNormInferenceAttrs(channelAxis, epsilon)`. The distinct implemented
`BATCH_NORM_TRAINING` meaning uses the same Tensor roles, five ordered outputs, and
`BatchNormTrainingAttrs(channelAxis, momentum, epsilon)`.

The **channel axis** is one normalized non-negative logical input-axis position selected explicitly
by the caller. It is layout-neutral: it does not imply NCHW, NHWC, a physical stride, or a batch
scheduler. Its extent `C` must match each exact rank-one `[C]` scale, bias, running-mean, and
running-variance vector, with unresolved equality deferred when the exact result Shape is still
known.

For channel `c`, inference requests
`((input - runningMean[c]) / sqrt(runningVariance[c] + epsilon)) * scale[c] + bias[c]`. A
**running mean** or **running variance** is an explicit estimated per-channel statistic supplied
to this occurrence. The variance is used directly: the operation performs no correction
conversion, recomputation, clamp, momentum update, or mutation. Construction also creates no
training/evaluation mode, saved statistic, auxiliary output, or hidden state.

Current `Tensor.batchNormInference` constructs metadata and provenance only. Current
package-private compiler autograd constructs contributions for all five floating Tensor inputs,
aligning each channel vector to the input Shape. Legal decomposition, backend preparation, and
runtime execution remain planned in their owning layers.

**Batch-normalization training** reduces every non-channel axis to calculate batch mean and
variance. Forward normalization uses biased population variance, while the explicit next running
variance uses the corresponding unbiased estimate divided by `N - 1`. The exact domain obligation
is `C == 0 || N >= 2`. A **running-statistic transition** is the pure output value
`(1 - momentum) * old + momentum * batch`; it neither assigns that value to a buffer nor retains it
for another call. Here **momentum** is the weight of the new batch, not the old statistic.

`Tensor.batchNormTraining` returns normalized output, next running mean, and next running variance.
Its shared producer additionally describes saved batch mean and saved inverse standard deviation.
Current package-private compiler autograd routes normalized output to input/scale/bias, next
running mean to input/running mean, and next running variance to input/running variance. It
retrieves the exact same-occurrence saved wrappers for formulas; saved slots are not independent
cotangent roots. Training-session and checkpoint ownership, physical saved-value lifetime,
publication, lowering, and execution remain planned in their owning layers.
See [batch-normalization inference](api/tensor-api.md#batch-normalization-inference-expressions) and
[batch-normalization training](api/tensor-api.md#batch-normalization-training-and-statistic-transition-expressions).

### Normalized Shape

The exact positive-rank `Shape` that identifies the trailing axes normalized by layer or RMS
normalization. If input rank is `R` and normalized rank is `K`, normalized axis `j` corresponds to
input axis `R - K + j`. Known static extents must match; unresolved equality may be deferred when
the result Shape remains exactly the input Shape. This is different from a [normalized
axis](#normalized-axis), which is one non-negative integer axis position.

### Population variance

Variance with correction zero: the sum of squared deviations from the mean is divided by the
population count `N`, not `N - 1`. Current layer normalization uses population variance and adds
epsilon inside the denominator square root. The term describes mathematical semantics, not a
reduction traversal or backend algorithm.

### Uncentered mean square

The sum of squared values divided by the population count `N`, without first subtracting a mean:
`sum_i(x_i * x_i) / N`. Current RMS normalization uses this quantity instead of variance and has
no statistical-correction option. The term defines mathematical semantics, not accumulation
order or a backend algorithm.

### Root mean square

The positive square root of an uncentered mean square. Current RMS normalization adds epsilon to
the mean square before taking that root, then divides each input value by the result. This meaning
is distinct from standard deviation because the input is not centered and variance is not
calculated.

### Epsilon

A small positive value added to a denominator-related quantity to state a numerical semantic
boundary. In current layer and RMS normalization it is an exact typed `ScalarValue`, must be
finite and strictly positive, and must match the result data type. Layer normalization adds it to
population variance inside the square root; RMS normalization adds it to uncentered mean square
inside the square root. Batch-normalization inference likewise requires exact-result-typed finite
positive epsilon and adds it to the supplied running variance inside the square root. Training
adds it only to biased batch variance inside the saved inverse-standard-deviation square root; it
does not alter either variance estimate or running transition. Epsilon is operation metadata, not
a Tensor input or a default hidden constant.

### Affine transform

An elementwise scale followed by bias: `standardized * scale + bias`. Current affine layer
normalization requires both explicit operands, each with Shape exactly equal to the normalized
Shape. Both batch-normalization forms also require both explicit operands, but each is an exact
rank-one per-channel vector. Neither contract infers or initializes parameters.

### Accumulator type

The floating format in which a semantic reduction such as a sum, mean, or variance is accumulated
before conversion to the result format. It can differ from result type: current BFLOAT16 layer
normalization accumulates mean and variance in FLOAT32, and BFLOAT16 RMS normalization accumulates
squares and sums in FLOAT32. An accumulator type constrains numerical meaning without selecting
traversal order, a kernel, or an executable algorithm.

A **computation format** is the floating format used for non-reduction formula arithmetic. Current
batch-normalization inference uses FLOAT32 formula arithmetic for BFLOAT16 or FLOAT32 results and
FLOAT64 for FLOAT64 results. It has no reduction accumulator. Training uses FLOAT32 reduction and
formula arithmetic for BFLOAT16/FLOAT32 results and FLOAT64 for FLOAT64 results. Like an
accumulator type, a computation format constrains meaning without selecting an algorithm, kernel,
or traversal.

### Saved statistic

An intermediate statistic retained for a later transformation, commonly a mean or inverse
standard deviation retained for backward construction. Current public layer and RMS normalization
and batch-normalization inference create no saved-statistic output. Batch-normalization training
instead describes saved batch mean and inverse standard deviation at producer positions three and
four, without exposing their Tensor wrappers in `BatchNormTrainingResult`. They are current model
metadata. Current package-private compiler autograd retrieves those canonical wrappers for
batch-training output-slot-zero and output-slot-two formulas without recomputation. They remain
logical forward values rather than a runtime tape, physical saved buffer, public publication, or
independent cotangent root; materialization and runtime lifetime remain future concerns.

### Referenced element span

The minimum count of storage elements needed to include every index referenced by a resolved layout. For non-empty shapes it is the greatest referenced element index plus one; for shapes with a zero-sized dimension it is zero. The span includes a storage offset and is not necessarily equal to the logical element count.

### Reshape and expand semantics

Two implemented one-input target-shape meanings represented by `ShapeTransformKind.RESHAPE` and
`ShapeTransformKind.EXPAND`. Both pair with `TargetShapeAttrs`, which retains one exact non-null
[`Shape`](#shape) as normalized semantic result state. Scalar, zero-extent, static, mixed dynamic,
and fully dynamic Shapes are structurally valid attributes. A raw public reshape request and its
numeric `-1` inference syntax are not stored; static Shape dimensions are non-negative and dynamic
dimensions use explicit symbols.

`RESHAPE` preserves the ordered logical element sequence while interpreting it through the target
coordinates. `EXPAND` logically repeats compatible singleton dimensions or adds repeated leading
dimensions. The shared attributes do not inspect an input, compare element counts, validate
expansion compatibility, bind symbols, create a proof or constraint, derive a
[`Layout`](#layout), or decide whether a result is a [view](#view) or materialized copy.

These semantic values are current. Public `Tensor.reshape(long...)` normalizes a defensively owned
raw request containing non-negative extents and at most one inferable `-1`; an empty request means
scalar Shape. Inference requires a known input count and non-zero product of all other requested
extents. It may infer zero from a zero input count, but a requested zero product is ambiguous.
Public `Tensor.reshape(Shape)` instead retains the exact normalized target reference. Both methods
reject unequal known counts and defer equality when either Shape is dynamic.

When the input has resolved contiguous geometry and the target is fully static, reshape publishes
a new view-marked layout with canonical target strides and the same element offset. Unresolved,
strided, broadcast, or dynamic geometry remains unresolved; no implicit contiguous operation or
materialization is inserted. Every valid result is fresh, unlabeled, storage-free, retains input
type and gradient eligibility, and records exact RESHAPE/target-shape semantics with provenance
`[input]`.

Public `Tensor.expand(long...)` treats raw extents as literal non-negative dimensions, with an
empty request denoting scalar Shape and no `-1` inference. `Tensor.expand(Shape)` retains the exact
target reference. Both align axes from the right, permit new leading target axes, and accept an
aligned pair immediately when its dimensions are structurally equal or the input dimension is a
static singleton. A fully static unequal pair is rejected if the source is not one; zero is an
ordinary static extent, so zero-to-one is invalid. A pair containing an unresolved dimension is
accepted as a binding-dependent Model expression with the later condition
`source == 1 || source == target`. Model does not bind, unify, or evaluate dimensions and creates
no deferred constraint. For a static target and resolved input layout, expand preserves the input
offset and unchanged aligned strides while assigning zero strides to leading and
expanded-singleton axes. A dynamic target, unresolved input geometry, or binding-dependent pair
stays layout-unresolved. Results are fresh, unlabeled, storage-free, retain exact input type and
gradient eligibility, and record exact EXPAND/target-shape semantics with provenance `[input]`.

The current compiler accepts a binding-dependent EXPAND occurrence by emitting the exact
source-one-or-source-equal predicate for each unresolved aligned pair. Preflight admits the
inverse SUM_TO_SHAPE only when it uses the same obligation. This does not bind a Dimension or
resolve layout. Compiler canonicalization, materialization, backend support, and execution remain
outside Model construction. Generic
[`Operation`](#operation) composition retains the exact kind and attributes references but does
not enforce the family pairing or one-input context. See [Reshape
expressions](api/tensor-api.md#reshape-expressions) and [Expand
expressions](api/tensor-api.md#expand-expressions).

### Tensor composition

The implemented backend-independent vocabulary for joining tensors. `TensorCompositionKind`
contains exactly `CONCAT` and `STACK`. CONCAT joins ordered non-empty inputs along an existing
axis and preserves rank; STACK joins same-shaped inputs along a newly inserted axis and increases
rank. Both pair with `CompositionAxisAttrs(axis)`.

Public `Tensor.unstack` is not a third composition kind. It validates one static `int`-sized
axis count up front, then returns an immutable ordered list of independent scalar
`SELECT(axis, coordinate)` expressions. Each has its own one-output producer and provenance index
zero and uses SELECT's conditional layout behavior. A zero count returns an immutable empty list
without creating a producer, Tensor, or identifier. This differs from future genuine multi-output
operations, which share one producer across distinct output indices.

Current package-private compiler autograd crops a supported floating CONCAT cotangent by each
ordered symbolic input prefix and selects the corresponding inserted-axis coordinate for each
supported floating STACK input. Repeated positions remain repeated contributions.

### Tensor

The implemented public model object for stable tensor metadata and optional host-visible state.
The current final `Tensor` retains one exact immutable [`TensorId`](#tensorid), one exact
immutable [`TensorDescriptor`](#tensor-descriptor), one normalized immutable optional label, and
immutable optional [`TensorProvenance`](#provenance). Its only current mutation is a synchronized
optional borrowed [`HostTensorStorage`](#host-storage) association. Replacement validates matching
data type, resolved referenced span when layout is available, and point-in-time liveness before
changing the exact reference. Read-only storage is accepted; later caller-controlled scope death
remains observable; and the tensor retains the wrapper reference without allocating backing
memory itself, copying or accessing contents, owning a closeable resource, or closing storage.
Provenance remains the same exact value across every storage transition and is accessed without
synchronization because it is final. When present, its selected output descriptor must be the
same exact reference as the Tensor descriptor; an equal but separately constructed descriptor is
rejected.

Tensor owns no gradient field, backward method, or gradient-lifecycle state. Each producer now
retains the canonical exact Tensor wrapper for every output slot, so hidden same-occurrence
auxiliaries can be retrieved without reconstructing wrappers. Retaining one derived Tensor may
therefore retain its producer's sibling outputs; the immutable cycle owns no external resource and
remains ordinarily collectable when unreachable.

Construction remains package-private, and the implemented [`TensorFactory`](#tensor-factory) is
the supported public construction boundary. The object uses ordinary identity equality and
hashing, while its diagnostic text contains stable ID, descriptor, and label facts without
provenance expansion, storage, or runtime state. Copied flat typed import is implemented through
the factory for resolved
dense-contiguous layouts. Copied rectangular nested primitive-array import is also implemented;
the factory infers its exact type, fully static shape, and dense-contiguous layout before returning
a Tensor. Exact typed scalar, zero, one, zero-like, and one-like creation is implemented with new
dense storage and explicit label and gradient intent. Type-safe full-value creation is implemented
for every current primitive meaning, and rectangular identity creation is implemented for all six
data types with typed main-diagonal ones and off-diagonal zeros. Eager non-empty `INT32` and
`INT64` range creation is also implemented as copied canonical dense leaf data. Public
[`TensorRandoms`](#tensor-random-initialization) owns caller-source normal,
continuous-uniform, bounded-integral, and Bernoulli eager initialization. Strict and cyclic
prefix preparation exists only in test source and is not a product capability. Random Operations,
typed access and export, and deterministic native-resource ownership remain planned. The current `add`, `sub`,
`mul`, `div`, `minimum`, `maximum`, and tensor-valued `pow` methods create fresh storage-free binary
arithmetic expression tensors from floating operands. They promote data type, broadcast shape,
leave layout unresolved, propagate gradient eligibility as input OR, and retain exact matching
operation semantics plus ordered provenance. The current `greaterThan`, `greaterOrEqual`,
`lessThan`, `lessOrEqual`, `equalTo`, and `notEqualTo` methods also accept ordered floating pairs
and broadcast shapes, but create fixed BOOL descriptors with false gradient eligibility while
retaining exact comparison semantics and ordered provenance. The current `logicalAnd` and
`logicalOr` methods accept only exact BOOL inputs, broadcast their shapes, retain ordered
provenance, and create fresh fixed BOOL results. `logicalNot` also requires exact BOOL, but retains
the exact input Shape and one-input provenance without broadcasting. All logical results have
unresolved layout, false gradient eligibility, no label, and no storage, and construction does not
read truth bytes, short-circuit, simplify, or execute them. Static `Tensor.where` requires an exact
BOOL condition and two floating branches. It promotes the branch types, broadcasts the branches
before combining the condition shape, creates a fresh unresolved storage-free result with
branch-only gradient eligibility, and records exact ordered
`[condition, ifTrue, ifFalse]` provenance. It does not inspect values, choose or evaluate a branch,
define gradient routing, capture a graph, or execute selection. The current `cast` method accepts
all current source/target pairs and creates a fresh explicit result even for a same-type request. It
retains the exact input Shape, leaves layout unresolved, retains gradient eligibility only across
an already-eligible floating-to-floating cast, and records typed target attributes plus exact
one-input provenance. It does not inspect or convert values/storage, define numerical or gradient
rules, canonicalize casts, capture a graph, or execute conversion. The current nineteen
zero-argument unary methods also create fresh floating expression tensors. They retain the exact
input data type and Shape,
leave layout unresolved, preserve gradient eligibility, and record the matching parameterless kind
plus exactly one input reference without domain checks or canonicalization. The three floating-
classification methods instead return fixed non-differentiable BOOL metadata with the exact input
Shape and one-input provenance. They do not inspect or classify stored values. The current scalar
`add`, `sub`, `mul`, `div`, `minimum`, `maximum`, `pow`, `clamp`, `clampMin`, and `clampMax`
methods likewise create fresh floating
one-input expressions. They retain the exact type and Shape, preserve gradient eligibility, and
store exact matching typed scalar attributes without conversion or canonicalization; retained
`double` overloads mean exact FLOAT64, range clamp remains one first-class `CLAMP` operation, and
the one-bound conveniences create scalar MAX/MIN producers.
Other expression families, gradient rules and objects, trainable
role, publication behavior, compiler integration, device buffers, numerical execution, and
runtime residency remain planned. The current `sum`, `prod`, reduction `min`, and reduction `max`
methods create floating or signed-integral full, single-axis, or multi-axis aggregate expressions; `mean`
remains floating-only. Current `all` and `any` create exact-BOOL, non-differentiable forms. Full
forms produce canonical rank-zero shape; axis forms normalize, then remove or retain selected
axes with extent one. Results preserve the family-specific type and eligibility, leave layout
unresolved, and record one-input provenance without aggregating, comparing, or evaluating values.
Integral sum/product use exact-width modular arithmetic with reassociation permitted, integral
min/max use signed order, and their empty identities are zero, one, the bounded type maximum, and
the bounded type minimum, respectively. Floating/BOOL ordinary empty and special-value policy is
fixed. Ordered multi-axis ordinary reductions and floating log-sum-exp, corrected
variance/standard deviation, and L1/L2 norm construction are current; algorithms, gradients,
compiler, backend, and executable behavior remain separately owned. Current
`sumToShape` adds a distinct numeric SUM form with an exact right-aligned target Shape, retaining
unresolved target-one-or-input-equal obligations without selecting axes or binding dimensions.
Current
`argMin` and `argMax` methods accept one axis of floating or integral input and create fixed INT64,
non-differentiable results with shared `ArgExtremaAttrs`. Convenience forms request the first equal
extremum; complete forms retain an explicit first- or last-index policy. The shared semantic order
prefers NaN, orders negative zero below positive zero, orders infinities normally, and uses signed
integral order. Construction rejects a statically empty selected axis but performs no comparison
or actual index selection.
A masked `sum(axis, mask)` or `mean(axis, mask)` requires an exact BOOL mask and ordinary
right-aligned broadcasting that produces exactly the input Shape, removes the selected axis, and
records exact `[input, mask]` provenance. Callers explicitly reshape or expand masks to express
other axis intent. It preserves input type and gradient eligibility but does not align storage,
select or aggregate values, count true positions, divide, or define a gradient rule.
A `cumSum` or `cumProd` request accepts floating or integral input, preserves its exact Shape, data
type, and gradient eligibility in an unresolved descriptor, and records one normalized axis plus
exact exclusive/reverse flags. Each valid call is fresh and storage-free; construction performs no
addition or multiplication and defines no gradient, compiler, backend, or execution behavior.
A `softmax` or `logSoftmax` request accepts floating input, preserves its exact Shape, data type,
and gradient eligibility in an unresolved descriptor, and records one normalized axis plus the
exact probability or log-probability kind. Each valid call is fresh, unlabeled, and storage-free;
construction performs no normalization and defines no numerical algorithm, gradient, compiler,
backend, or execution behavior.
A `contiguous` request accepts every data type and preserves exact Shape, type, and gradient
eligibility. A fully static result receives new canonical dense row-major, zero-offset geometry;
a dynamic result remains unresolved. Every request is fresh, unlabeled, and storage-free, records
exact one-input provenance, and neither inspects input layout or values nor performs a copy.
A `reshape` request also accepts every data type and preserves exact type and gradient eligibility,
but changes to a normalized target Shape while preserving ordered logical elements. Raw requests
support scalar empty varargs, zero extents, and one locally inferable `-1`; exact Shape requests
retain their target reference and defer dynamic count equality. A resolved contiguous input and
static target produce same-offset canonical view metadata, while other geometry remains
unresolved. Each result is fresh, unlabeled, storage-free, and records one-input RESHAPE
provenance without moving values or choosing materialization.
A `Tensor.expand` request also accepts every data type and preserves exact type and gradient
eligibility. It directionally right-aligns the input with a literal or exact target Shape, accepts
equal aligned dimensions, input-side static singletons, and new leading axes. A fully static
unequal pair whose source is not one is rejected; a pair containing an unresolved dimension is
retained with the later source-one-or-source-equal requirement and no Model-created constraint.
Static target plus resolved input geometry produces same-offset view metadata with preserved
aligned and zero leading/expanded strides; binding-dependent or otherwise unresolved geometry
remains unresolved. Each result is fresh, unlabeled, storage-free, and records one-input EXPAND
provenance without repeating values, attaching an alias, or choosing materialization.
A `Tensor.permute` request accepts every data type and preserves exact gradient eligibility. It
requires a complete output-to-input mapping, normalizes each negative axis once, and reorders exact
Dimension references. Resolved input geometry produces a new same-offset view descriptor with
reordered exact strides; unresolved input stays unresolved. `Tensor.transpose()` requires rank two
and supplies `[1, 0]`. Every result is fresh, unlabeled, storage-free, and records one-input PERMUTE
provenance without attaching an alias, canonicalizing, choosing materialization, or executing.
A `Tensor.expandDims` request accepts every data type, normalizes one insertion position against
rank plus one, inserts one static singleton, and preserves exact unaffected Dimension references.
A `Tensor.squeeze` request likewise accepts every data type but normalizes an existing axis and
requires its Dimension to be statically known as one before removing it. Resolved layouts produce
new same-offset view descriptors by inserting a checked deterministic stride or removing one
stride; unresolved layout remains unresolved. Every result preserves exact gradient eligibility,
is fresh, unlabeled, and storage-free, and records matching one-input axis-transform provenance
without binding a dynamic symbol, attaching an alias, canonicalizing inverse edits, choosing
materialization, or executing.
A `Tensor.pad` request accepts every data type, requires non-negative before/after widths for
every input axis, and derives canonical checked static or symbolic extents by applying the before
and after widths in order. A `Tensor.tile` request likewise accepts every data type, requires one
positive complete-pattern repeat per axis, and derives a canonical checked static or symbolic
product. Neutral requests preserve exact Dimension references. Both clone caller arrays, preserve
exact gradient eligibility, always produce fresh unresolved unlabeled storage-free results, and
record their normalized attributes plus provenance `[input]` without binding or evaluating
symbolic extents, materializing values, or defining execution.
Static `Tensor.concat` and `Tensor.stack` accept ordered non-empty inputs of one exact data type.
CONCAT validates equal rank and non-axis Dimensions, derives the selected extent through canonical
checked symbolic addition, and preserves rank. STACK requires identical Shapes and inserts one
input-count Dimension. Both propagate gradient eligibility by input OR. Instance `Tensor.unstack`
removes one static `int`-sized axis and returns an immutable ordered List of individually indexed
outputs. All created composition results are fresh, unresolved, unlabeled, and storage-free;
each unstack output has an independent one-output producer and provenance index zero, so unstack
does not claim shared producer grouping; its zero-count result creates no Tensor or ID.
A `Tensor` is not an
intermediate-representation node or [graph value](#graph-value). See [Public Tensor
state](api/tensor-api.md#public-tensor-state).

### Tensor factory

The implemented public static `TensorFactory` creates a fresh [`Tensor`](#tensor) from a completed
[`TensorDescriptor`](#tensor-descriptor), with optional diagnostic label text and optional
existing borrowed [host storage](#host-storage). It is the public construction boundary while the
Tensor constructor remains package-private. For a descriptor with a resolved layout, the factory
can also allocate one matching JVM primitive array whose length is exactly the referenced element
span, wrap its heap segment, and attach that writable storage. `FLOAT64`, `FLOAT32`, `BFLOAT16`,
`INT32`, `INT64`, and `BOOL` use `double[]`, `float[]`, raw `short[]`, `int[]`, `long[]`, and raw
`byte[]`, respectively. The raw array starts at the JVM default zero representation.

The factory can import those same six flat carriers into a new resolved dense-contiguous tensor.
Carrier data type and logical element count must match exactly. Numeric carriers and raw BFLOAT16
bits are copied unchanged; BOOL treats zero as false and canonicalizes every non-zero byte to one.
The source is not retained or mutated, and later source mutation cannot change the tensor. Scalar
and empty dense imports follow their logical counts. Offset, strided, broadcast, and unresolved
layouts are rejected because they require a separate scatter or view-population policy.

The factory also accepts one rank-two-or-greater rectangular Java primitive-array graph through an
`Object` parameter, which is necessary because arbitrary array rank has no finite Java overload
family. Runtime class metadata must prove an ultimate `double`, `float`, `short`, `int`, `long`, or
`byte` carrier. The factory validates every reachable subarray for non-null rectangular structure,
rejects an empty non-final axis whose trailing extents are unobservable, accepts an empty final
primitive axis, and infers an exact fully static dense-contiguous descriptor. It flattens leaves in
row-major order into a fresh matching carrier and delegates to flat import. Numeric and raw
BFLOAT16 values remain unchanged; BOOL normalization remains centralized in flat import. No source
level is retained or mutated, later source mutation cannot change the tensor, and concurrent
mutation during import has no atomic deep-snapshot guarantee.

The heap segment's automatic scope keeps the primitive array reachable and permits access from
any thread. No arena, close operation, external owner, native fallback, or deterministic release
is introduced. Descriptor-based creation and flat import do not construct descriptors or resolve
absent layouts; nested import constructs only the exact static dense descriptor inferred from the
validated source. Constant creation constructs only canonical dense descriptors for exact
primitive rank-0 scalars or fully static requested/template shapes. Scalars infer data type from
their declared primitive inputs; `scalarBFloat16(float)` alone converts with BFLOAT16
round-to-nearest, ties-to-even semantics. Zeros use default-zero allocation, while scalars and
ones use exact typed flat import. Like methods read only template shape and data type and preserve
neither layout nor mutable or diagnostic state. Full-value methods infer exact type from primitive
values, with only `fullBFloat16` converting, and fill one exact source before one flat import.
`identityMatrix` creates square or rectangular dense matrices for every current data type with
typed one on the main diagonal and typed zero elsewhere; `eye` delegates unchanged to that
canonical method. Separate calls create fresh metadata, storage, Tensor identity, and backing
arrays. Every public factory path creates a
provenance-free leaf. Package-private derived-construction seams create one validated producer and
attach index-zero provenance to one result or indexed provenance to an immutable ordered result
list through the existing ID allocator. Producer validation precedes allocation; a multi-output ID
exhaustion can consume earlier IDs but returns no partial list. These seams add no storage, graph
capture, traversal, family-specific inference, or execution. The factory retains no tensor, graph,
runtime, backend, registry, or service state.

The factory's two eager range overloads map `int` bounds and step to `INT32`, and `long` bounds and
step to `INT64`. Each result is non-empty, rank one, non-differentiable, inclusive at the start,
exclusive at the end, and stored in new canonical dense storage. Positive and negative non-zero
steps are accepted only when they advance toward the end. Exact overflow-safe sizing rejects a
count above `Integer.MAX_VALUE` before allocation and does not evaluate an unused addition after
the final emitted value.

Prefix shaping is not part of the production factory. Package-private test-source fixtures may
prepare strict or cyclic exact-carrier arrays and delegate them to public flat import, but this
mechanic is absent from production source, generated production Javadoc, and the public product
inventory.

### Tensor random initialization

The public final stateless `TensorRandoms` namespace creates eager provenance-free leaf tensors
from one explicit caller-owned `RandomGenerator`. It retains no source or result and gives the
source no lifecycle. The caller creates, configures, seeds, owns, advances, and coordinates access
to the exact generator. Synaptik supplies no default, global, thread-local, engine-owned,
runtime-owned, or service-located source and does not select, seed, synchronize, reset, split,
serialize, or close the supplied object.

Normal random creation accepts one transient caller-owned `RandomGenerator`, fully static
Java-array-sized shape, explicit `FLOAT64`, `FLOAT32`, or `BFLOAT16` output, finite mean, finite
numerically non-negative standard deviation, label, and gradient intent. It consumes exactly one
`nextGaussian()` call per logical row-major element, transforms with ordinary binary64
multiplication then addition, converts to one exact carrier, and delegates once to flat import.
`TensorRandoms` never selects, seeds, retains, substitutes, synchronizes, resets, splits, or closes
the source. Reproducibility is consequently bounded to equivalent generator implementation and
state, identical arguments, and no interfering use. Random Operations, typed access or export,
general numeric conversion, native/runtime/backend allocation, and deterministic resource
ownership remain planned.

Continuous-uniform random creation accepts the same transient caller-owned source, static
Java-array-sized shapes, and three floating output types. Its finite binary64 lower bound must be
strictly less than its finite upper bound. Each row-major element consumes exactly one
`nextDouble(lower, upper)` call. A conforming source result is in the binary64 half-open interval;
FLOAT64 stores it directly, FLOAT32 narrows once, and BFLOAT16 narrows to binary32 before
`BFloat16Bits.fromFloat`. Narrowing may produce a stored value equal to the corresponding narrowed
upper bound. `TensorRandoms` does not clamp or resample, post-validate custom source results, or
retain the generator. The same caller ownership, no-synchronization, and bounded reproducibility
policy applies.

Bounded integral random creation has two `randomInt` overloads. Primitive `int` bounds infer
`INT32`, primitive `long` bounds infer `INT64`, and both results disable gradients. Each row-major
element consumes exactly one matching bounded `nextInt(origin, bound)` or
`nextLong(origin, bound)` call and is stored directly in one exact carrier before one flat import.
Bounds define a strict half-open interval and are validated even for empty output. No modulo,
unbounded draw, floating conversion, data-type parameter, gradient parameter, or default source is
added. Because the exclusive bound uses the result carrier, the API cannot express a mathematical
exclusive bound above `Integer.MAX_VALUE` or `Long.MAX_VALUE`; no full-domain convenience is
provided. The same caller ownership, no-synchronization, bounded reproducibility, and late
failure/no-rollback rules apply.

Bernoulli random creation has one `randomBernoulli` method. It requires a fully static
Java-array-sized shape and a finite binary64 probability in the closed interval `[0, 1]`, always
produces canonical BOOL storage, and always disables gradients. Each row-major element consumes
exactly one unbounded `nextDouble()` call, including when probability is zero or one, and stores
byte one exactly when the draw is strictly less than the probability. Equal or custom
non-conforming draws are not post-validated or coerced. Positive and negative zero are both
accepted as probability zero. `TensorRandoms` builds one complete byte carrier and delegates once
to BOOL flat import; it exposes no data-type, gradient, numeric-truthiness, default-source, or
probability-tensor option. The same caller ownership, no-synchronization, bounded reproducibility,
and late failure/no-rollback rules apply.

Every successful method builds one exact primitive source carrier and delegates once to
`TensorFactory.fromFlatArray`, which allocates destination storage and the fresh factory ID.
Pre-sampling validation consumes no source call or ID. A source exception leaves completed calls
consumed but creates no destination or ID. Blank-label rejection and ID exhaustion happen after
all calls and destination allocation, without rollback. Reproducible values require an equivalent
generator implementation and state, identical arguments, and no interfering use; there is no
cross-algorithm, provider, Java-version, seed-expansion, serialization, concurrent-use, or global
sequence promise. Eager random initialization is not graph RNG state, a random Operation, or a
dropout contract.

### Tensor factory allocation and failure effects

For every attempted construction that reaches identifier allocation, the factory issues one
non-negative [`TensorId`](#tensorid) unique among its allocations in the current Java virtual
machine (JVM), including concurrent calls. Numeric values are opaque: semantic construction
failures create permanent gaps, and numeric order need not match method-completion order. Null
argument containers fail before allocation, while delegated label or storage validation fails
afterward and consumes the candidate. `Long.MAX_VALUE` can be claimed once; every later allocation
fails permanently instead of wrapping or reusing an ID. The guarantee does not cover manually
constructed IDs, another JVM, process restarts, persisted artifacts, or distributed identity.

Heap allocation additionally requires resolved layout and a referenced span no greater than
`Integer.MAX_VALUE`. Null, unresolved-layout, over-limit, JVM array, segment, and storage-wrapper
failures occur before identifier allocation and consume no ID. Heap allocation and wrapping occur
before delegated creation, so a blank label or exhausted identifier space is observed only after
the heap work; a blank-label failure consumes its allocated ID.

Flat import performs carrier, dense-layout, and logical-count validation before destination or ID
allocation. A blank label and identifier exhaustion are observed after destination allocation and
before copying. Unexpected population failures occur after ID allocation and are not rolled back.

Nested import performs complete structural and checked-count validation before allocating its
intermediate flat carrier. Descriptor gradient eligibility is checked after flattening but before
destination or ID allocation. A blank label and identifier exhaustion then have the same
destination-allocation and ID side effects as delegated flat import. The intermediate carrier is
never exposed or retained.

Constant creation validates static shape, checked logical count, the Java array limit, dense
layout geometry, and gradient eligibility before destination or ID allocation. Scalar and one
source carriers are allocated after descriptor validation; zeros have no source carrier. Blank
labels fail after destination and ID allocation and consume that ID. Exhaustion is also observed
after destination allocation. Every successful constant has a new Tensor, descriptor, layout,
storage wrapper, backing array, and factory ID; like-shaped results retain no template object or
template state beyond the immutable shape and data-type values used to build the result.

Range label and argument validation runs before result-carrier, destination, or ID allocation.
Each successful path builds one complete exact carrier and delegates once to flat import. A blank
label is rejected after carrier, destination, and ID allocation but before copying, and consumes
that ID. Exhaustion is observed after both arrays exist. These failures do not roll back
identifiers.

Normal-random null, shape, count, type, distribution, layout, and descriptor validation completes
before source-carrier allocation, sampling, destination allocation, or ID allocation. Source
allocation failure consumes neither calls nor an ID. A generator exception preserves completed
source calls but creates no destination or ID. After sampling, delegated flat import allocates the
destination and then the ID; blank-label failure consumes all calls and one ID, while exhaustion
consumes all calls without rollback.

Continuous-uniform null, shape, count, type, bound, layout, and descriptor validation likewise
completes before source-carrier allocation, sampling, destination allocation, or ID allocation.
Source allocation failure consumes neither calls nor an ID. A generator exception preserves prior
bounded calls but creates no destination or ID. After sampling, delegated flat import allocates the
destination and then the ID; blank-label failure consumes all calls and one ID, while exhaustion
consumes all calls without rollback.

Bounded-integral null, shape, count, and bound validation likewise completes before source-carrier
allocation, sampling, destination allocation, or ID allocation. Source allocation failure
consumes neither calls nor an ID. A generator exception preserves prior bounded calls but creates
no destination or ID. After sampling, delegated flat import allocates the destination and then the
ID; blank-label failure consumes all calls and one ID, while exhaustion consumes all calls without
rollback.

Bernoulli null, shape, count, and probability validation likewise completes before source-carrier
allocation, sampling, destination allocation, or ID allocation. Source allocation failure
consumes neither calls nor an ID. A generator exception preserves prior unbounded calls but
creates no destination or ID. After sampling, delegated BOOL flat import allocates the destination
and then the ID; blank-label failure consumes all calls and one ID, while exhaustion consumes all
calls without rollback.

### Tensor descriptor

The implemented immutable combination of one non-null data type, one non-null shape, an explicit
resolved-or-unresolved layout state, and a `requiresGrad` flag. A present
`Optional<LayoutDescriptor>` contains resolved numeric geometry; `Optional.empty()` means geometry
is unresolved. Dynamic shapes must be unresolved, while fully static shapes may also remain
unresolved without implying a default layout. A present layout is reconstructed against the paired
shape and accepted only when its complete public geometry remains equal. This proves geometric
compatibility, not the identity of the shape that originally created the layout. `requiresGrad`
may be true only for a differentiable data type and records model eligibility, not the existence of
a gradient rule or backend support. The optional is compared by value rather than container
identity; when present, it contains the exact immutable layout object supplied at construction. A
descriptor describes logical facts and does not own a public mutable `Tensor`, graph identity, host
storage, device buffers, runtime residency, materialization policy, or backend execution state. See
[Tensor descriptors](api/tensor-api.md#tensor-descriptors).

### `TensorId`

A validated non-negative identifier retained by implemented public mutable `Tensor` state. It
belongs to the tensor identity domain and is distinct from graph-local node and value identities.
The value type itself does not allocate or guarantee uniqueness. The implemented
[`TensorFactory`](#tensor-factory) provides the narrower guarantee that IDs it allocates are unique
among its allocations in one JVM; callers may still construct equal numeric values manually. Two
tensor objects remain unequal even when their IDs compare equal. An implemented
`ForwardPublicationBinding` can associate an ID with a graph value without storing graph-local
IDs on the tensor. See [Identifiers](api/tensor-api.md#typed-identifiers).

### Tiling

An implemented backend-independent complete-pattern repetition meaning represented by
`TileKind.TILE` and `TileAttrs`. Entry `repeats[i]` is the strictly positive number of times the
complete input pattern is requested along normalized axis position `i`; this differs from
repeating each scalar into one adjacent run. The ordered list is an immutable snapshot, and an
empty list is the rank-zero scalar identity parameter.

For conceptual input `[[1, 2], [3, 4]]` and repeats `[2, 3]`, each complete row pattern occurs
three times along axis one and the complete two-row pattern occurs twice along axis zero, producing
the conceptual pattern `[[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4], [1, 2, 1, 2, 1, 2],
[3, 4, 3, 4, 3, 4]]`. The attributes have no input Tensor or Shape, so they do not match rank,
multiply extents, check overflow, derive layout or provenance, materialize values, define
gradients, or execute tiling. `Long.MAX_VALUE` is structurally valid. Public `Tensor.tile`
separately requires one repeat per input axis, clones the array, and performs canonical checked
static or symbolic multiplication. Repeat one preserves the exact input Dimension reference;
other positive repeats retain an exact symbolic-extent expression. The result preserves exact
type and gradient eligibility and is a fresh unresolved storage-free expression with normalized
attributes and provenance `[input]`. It does not bind or evaluate the formula, repeat values, or
materialize storage. Current package-private compiler autograd reshapes a supported floating TILE
cotangent to interleaved repeat/input axes, sums the repeat axes, and restores the input Shape. See
[Pad and tile
expressions](api/tensor-api.md#pad-and-tile-expressions) and [Pad and tile semantic kinds and
normalized attributes](api/tensor-api.md#pad-and-tile-semantic-kinds-and-normalized-attributes).

### Trace

Structured diagnostic information about compile, prepare, run, and backend activity. A trace
helps people and tools understand what happened without becoming business logic or execution
state. The implemented module currently defines the common event envelope and trace-local event,
node, logical-value, and public-Tensor identities; it does not emit, store, filter, or serialize
events. The trace module is a dependency leaf, so producers translate their identities rather than
make trace import producer-layer domain objects. See [Tracing](architecture/tracing.md).

### Trace event envelope

The implemented generic `TraceEvent<T extends TracePayload>` record that carries one non-null
producer-assigned [`TraceEventId`](#trace-event-identity--traceeventid), one non-null lifecycle
[`TracePhase`](#trace-phase--tracephase), one non-null [`TraceLevel`](#trace-level--tracelevel), a
producer-supplied monotonic-clock reading in nanoseconds, and one non-null typed payload. The
record retains all components unchanged and has ordinary component-based record value semantics.
Its timestamp accepts every `long` value, is not wall-clock or epoch time, and is meaningful for
differences only within the producer's documented clock domain. The envelope allocates no ID,
reads no clock, and provides no serialization, filtering, storage, sink, or emission behavior.

### Trace event identity / `TraceEventId`

The implemented non-negative `long` identity for one diagnostic event within a
producer-defined trace stream. The producer assigns the value and defines its uniqueness domain;
zero is valid, no sentinel is reserved, and the trace module provides no allocator or global
uniqueness guarantee.

### Trace level / `TraceLevel`

The implemented diagnostic classification with the exact detail-to-severity order `TRACE`,
`DEBUG`, `INFO`, `WARN`, and `ERROR`. A level classifies an event but defines no filtering
threshold, sink behavior, logging integration, failure response, or process-exit policy.

### Trace payload / `TracePayload`

The implemented open method-free marker for a typed diagnostic DTO carried by a
[`TraceEvent`](#trace-event-envelope). Implementations are required to be immutable and to
describe producer facts in trace-owned terms, but the open marker cannot enforce those properties
at runtime. Concrete compile, prepare, run, and backend payload records remain planned.

### Trace-local correlation identifier

An immutable trace-owned value used to relate diagnostic facts without storing or importing a
producer-domain identity. The producer defines the trace stream or correlation domain, assigns
the value, and owns allocation, uniqueness, lifetime, and any mapping from its own identity. A
trace-local numeric value is not required to equal the producer ID's numeric value and has no
process-wide or cross-stream guarantee. The implemented model-correlation domains are
[`TraceNodeId`](#trace-node-correlation-identity--tracenodeid),
[`TraceValueId`](#trace-value-correlation-identity--tracevalueid), and
[`TraceTensorId`](#trace-tensor-correlation-identity--tracetensorid).

### Trace node correlation identity / `TraceNodeId`

The implemented non-negative one-`long` trace-local identity for one computation occurrence. It
does not identify operation semantics, an output value, a producer object, or a runtime unit.
Zero is valid, no sentinel is reserved, and ordinary record equality applies only to another
`TraceNodeId` with the same value.

### Trace value correlation identity / `TraceValueId`

The implemented non-negative one-`long` trace-local identity for logical graph data. It is
nominally distinct from node and public-Tensor correlations and does not identify storage, a
buffer, or a runtime slot. Zero is valid, no sentinel is reserved, and ordinary record equality
applies only within the `TraceValueId` domain.

### Trace Tensor correlation identity / `TraceTensorId`

The implemented non-negative one-`long` trace-local identity for public Tensor state. It is
nominally distinct from graph node and logical-value correlations and does not identify a storage
address, device allocation, or runtime residency. Zero is valid, no sentinel is reserved, and
ordinary record equality applies only within the `TraceTensorId` domain.

### Trace phase / `TracePhase`

The implemented lifecycle classification with exactly `COMPILE`, `PREPARE`, and `RUN`.
`COMPILE` covers capture through logical planning, `PREPARE` covers backend preparation and
executable-state construction, and `RUN` covers invocation and execution activity. Backend is a
producer role and planned payload family, not a fourth phase; a backend fact uses the phase in
which it occurs.

### Training graph

The compile-time computation used for a training-capable mode. It contains the forward computation and backward gradient computation, and a later architecture version may also represent optimizer updates as graph operations. It remains compile-time graph state; backends only prepare and execute their assigned regions. See [Training graph](architecture/training-graph.md).

### Typed trace DTO

A typed data-transfer object used to carry one defined category of diagnostic facts. The current
foundation implements the event envelope and an open payload marker; concrete compile, prepare,
run, and backend payload DTOs remain planned. Typed fields preserve meaning and machine-readable
types. They are preferred over a primary `Map<String,String>` model, while a typed backend
attribute escape hatch also remains planned. See [Tracing](architecture/tracing.md#current-event-foundation).

### `ValueId`

A validated non-negative identifier for an input, intermediate, or output logical value within one owning graph. It does not identify a computation occurrence or a physical buffer. Its numeric value may be reused in another graph. See [Identifiers](api/tensor-api.md#typed-identifiers).

### View

A tensor or layout interpretation that aliases storage also used by another logical tensor representation. A view may change shape, strides, or offset without copying elements. In the implemented `LayoutDescriptor`, view is explicit metadata independent of geometric kind, except that broadcast repetition through a zero stride must be marked as a view.

### Window transform

An implemented backend-independent sliding-window meaning represented by
`WindowTransformKind`. The exact kind/attributes pairings are `UNFOLD_AXIS` with
`UnfoldAxisAttrs`, `FOLD_AXIS` with `FoldAxisAttrs`, `UNFOLD2D` with direct `Window2dAttrs` or
explicit-padding `Unfold2dAttrs`, and `FOLD2D` with `Fold2dAttrs`. Family signatures enforce all
five pairings and declare one input and one output.

General-axis unfold materializes windows along one normalized source axis without padding,
dilation, or image assumptions. For static selected extent `D`, positive window `size`, and
positive `step`, current public expression Shape construction uses
`floor((D - size) / step) + 1` window positions after proving that `size <= D`. It replaces the
selected extent with that count and appends `size` as the final result axis. Conceptual Shape
`[2, 5, 3]` with axis `1`, size `3`, and step `1` therefore becomes `[2, 3, 3, 3]`. The semantic
contract describes materialized windows rather than promising a storage view.

General-axis fold is the scatter-add adjoint. It interprets the eventual input's final dimension
as window size, removes it, restores one normalized target axis to the explicit `outputSize`, and
adds overlapping contributions. Conceptual windows of Shape `[3, 3]` with values
`[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, target axis `0`, output size `5`, and step `1` produce Shape
`[5]` with values `[1, 6, 15, 14, 9]`. Uncovered valid positions remain zero. Output size cannot
always be inferred because trailing source positions might have appeared in no window. No public
Tensor execution occurs here. Public `Tensor.foldAxis` normalizes its raw axis against target
rank, validates numeric input and static window geometry, and records the `FOLD_AXIS` expression;
future gradient construction remains compiler-owned.

The two-dimensional forms use NCHW axis order: batch, channel, height, width. Im2col places
sampled image windows into canonical rank-three columns; col2im scatters column entries back into
an explicit NCHW output and sums entries targeting the same coordinate without overlap averaging.
For conceptual NCHW Shape `[1, 1, 3, 3]`, a 2-by-2 kernel, unit stride and dilation, zero symmetric
padding, and floor mode produce im2col Shape `[1, 4, 4]`. Folding compatible columns into explicit
Shape `[1, 1, 3, 3]` gives four contributions to the center and one to each corner. Dynamic
channel and spatial Dimensions retain the same canonical rank-three form by using exact symbolic
products for `C * kernelHeight * kernelWidth` and `outputHeight * outputWidth`.

Stride is the positive distance between consecutive window starts. Dilation is the positive
spacing between kernel samples. Symmetric padding is the same non-negative width on both sides of
one spatial dimension. Direct `UNFOLD2D + Window2dAttrs` samples outside the source as conceptual
positive zero. `UNFOLD2D + Unfold2dAttrs` instead uses its exact typed scalar for every symmetric
padding sample and terminal ceil-grid sample beyond the padded extent, preserving raw floating
bits. Effective kernel is the span after dilation. For each spatial dimension, current public
expression construction uses checked static or canonical symbolic arithmetic:

```text
effectiveKernel = dilation * (kernel - 1) + 1
numerator       = input + 2 * padding - effectiveKernel
output          = floor(numerator / stride) + 1       in floor mode
output          = ceil(numerator / stride) + 1        in ceil mode
```

The immutable attributes validate only intrinsic component signs and nullness. They retain valid
values and exact immutable scalar/Shape/window references without calculating geometry or proving
rank, axis bounds, fit, column compatibility, or arithmetic representability. Current public
`unfold`, `foldAxis`, both `unfold2d` forms, and `fold2d` construction supplies input-aware local
validation, exact Shape arithmetic, preserved data type and gradient eligibility, unresolved
layout, and exact one-input provenance. `fold2d` requires complete structural equality with the
canonical column formulas and rejects unrelated unresolved symbols rather than registering an
equality constraint.
Gradient rules, compiler behavior, materialization,
lowering, backend/ONNX behavior, and execution remain planned. See
[Window-transform semantic kinds and
attributes](api/tensor-api.md#window-transform-semantic-kinds-and-attributes).

### NCHW pooling window geometry

NCHW names the rank-four logical axis order batch, channel, height, width. **Pooling** moves a
spatial **window** over each batch/channel plane and combines sampled values into one result per
window position. A window is the logical coordinate set; it does not imply a storage view or a
physical traversal order.

For each spatial axis, **padding** adds the same non-negative coordinate width on both sides,
**dilation** is the positive spacing between kernel samples, and **effective kernel** is the span
covered by those samples: `dilation * (kernel - 1) + 1`. **Ceil mode** rounds the output-position
quotient upward; current maximum and average pooling use the literal symmetric padded grid and do
not remove a terminal window that begins entirely in trailing padding.

The meaning of a padding coordinate depends on the operation. Direct `UNFOLD2D` reads it as
conceptual positive zero, while its explicit-padding variant reads the exact supplied typed scalar.
Conv2d includes conceptual positive zero in multiplication, and `MAX_POOL2D` excludes padding
from maximum selection. An all-padding maximum-pooling window therefore returns negative infinity.
`AVERAGE_POOL2D` instead uses **count-padding**: every logical kernel position contributes one to
the positive **divisor** `kernelHeight * kernelWidth`, while an out-of-bounds position contributes
conceptual positive zero to the numerator. This differs from valid-sample averaging, which would
divide only by in-bounds sample count. An all-padding average-pooling window returns positive zero.

`MaxPool2dAttrs`, `AveragePool2dAttrs`, and window-transform `Window2dAttrs` remain distinct because
extrema selection, fixed-divisor averaging, and window extraction have different semantics.
Average pooling accumulates and divides BFLOAT16/FLOAT32 in FLOAT32 and FLOAT64 in FLOAT64, with
one final division; **accumulation** here names the intermediate arithmetic domain, not an
algorithm or traversal order. Current package-private compiler autograd routes maximum pooling
through the reconstructed first eligible logical winner without adding saved indices, and routes
average pooling through the fixed logical kernel count. See [NCHW maximum-pooling
expressions](api/tensor-api.md#nchw-maximum-pooling-expressions), [NCHW average-pooling
expressions](api/tensor-api.md#nchw-average-pooling-expressions), and [Window transform](#window-transform).

## Common distinctions

### Tensor versus graph value

| `Tensor` | Graph value |
|---|---|
| Implemented public model state with immutable identity, descriptor, label, and provenance plus one existing mutable borrowed-host-storage association | Immutable compile-time graph state |
| Retains stable ID, descriptor, label, and optional provenance plus optional mutable host storage | Represents logical data flowing between graph nodes |
| Identified by `TensorId` | Identified by graph-local `ValueId` |
| Can participate in more than one separately compiled graph | Belongs to one owning graph context |
| Must not become runtime device residency | Must not be confused with a physical buffer or slot |

The implemented standalone `ForwardPublicationBinding` connects the public Tensor and graph-value
identity domains for requested forward results. Current compiler-owned `PublicationPlan` provides
owning-graph and separate forward/gradient binding context but no publication delivery policy.
Public
descriptor-based leaf construction, immutable provenance, and floating plus selected signed-
integral binary arithmetic, comparison, and scalar Tensor expression construction are
implemented. Unary construction remains floating-only; BOOL-only logical expression construction,
explicit cast expression construction, floating numeric aggregate expression construction for
sum, mean, product, minimum, and maximum, selected signed-integral aggregate construction for sum,
product, minimum, and maximum, and BOOL aggregate expression construction for all and any.
Ordered multi-axis ordinary/log-sum-exp/statistical/norm construction and axis-only
index-producing construction for arg-min and arg-max, shape-preserving cumulative-scan
construction, and shape-preserving softmax/log-softmax construction are also implemented.
Shape-preserving contiguous request construction is
implemented with static-resolved and dynamic-unresolved result layout rules. Ordered-element-
preserving reshape construction is implemented with raw inference, exact-Shape retention, and
conditional contiguous-input/static-target view geometry. Directional right-aligned expand
construction is implemented with exact target retention and conditional zero-stride view geometry.
Complete axis-permutation and rank-two transpose construction is implemented with exact
Dimension/stride reordering and conditional same-offset view geometry. Singleton-axis insertion
and selected static-singleton removal construction are implemented with exact unaffected
Dimension retention and conditional same-offset stride insertion/removal.
Scalar select, axis gather, Gather-ND, axis-scatter, and Scatter-ND expression construction are
implemented with their family-specific local type, axis, and Shape validation plus exact provenance. These
metadata-only expressions do not read index/update values, apply scatter writes or reductions, or
define index bounds, duplicate-target detection, gradients, compiler behavior, or execution.
Ordered concat and stack construction plus immutable-list unstack construction are implemented
with unresolved layouts and exact ordered or individually indexed provenance. The unstack list is
not evidence of one grouped graph producer.
Gradient rules and objects, compiler
graph capture and canonicalization, dynamic constraint solving, truth or numerical execution,
materialization, and publication behavior are not part of the current Tensor contract.

### Node versus value

| Node | Value |
|---|---|
| Implemented as `CompiledNode` | Implemented as `GraphValue` |
| A computation occurrence | Logical data consumed or produced by computation |
| Identified by `NodeId` | Identified by `ValueId` |
| Stores an `Operation` and ordered value-ID positions | Stores a `TensorDescriptor`, not an operation |
| May produce multiple values | May exist without a producer and may have multiple consumers |

Neither local record is the graph container. The implemented `CompiledGraphModel` owns producer
derivation and validates whole-graph uniqueness, topology, boundaries, and exact phase coverage
without storing derived indexes.

### Operation kind versus attributes versus operation

| Concept | Meaning | Current status |
|---|---|---|
| `OperationKind` | Which backend-independent computation is meant | Interface plus binary arithmetic, binary comparison, boolean logical, conditional selection, unary elementwise, floating-classification, scalar elementwise, cast, aggregate reduction, cumulative scan, softmax, layer normalization, RMS normalization, contiguous-request, reshape/expand, axis-transform, slice, pad, tile, tensor-composition, scalar-select, axis-gather, gather-ND, axis-scatter, scatter-ND, and window-transform families implemented; other families planned |
| `OperationAttrs` | Immutable typed parameters that refine that meaning | Marker plus scalar-value, clamp-range, cast-target, ordinary single-/multi-axis and statistical-reduction, shared arg-extrema, masked-reduction, cumulative-scan, softmax, layer-normalization, RMS-normalization, target-shape, permutation, single-axis-transform, normalized slice and crop-to-Shape, pad, tile, composition-axis, scalar-select, gather-axis, gather-ND, scatter-elements, scatter-ND, and window-transform values implemented; other family-specific values planned |
| `NoOperationAttrs.INSTANCE` | Explicit parameter value for a kind with no parameters | Implemented canonical singleton |
| `OperationSignature` | Exact accepted attributes class plus inclusive occurrence input/output bounds | Implemented family-owned structural contract |
| `Operation` | Immutable validated pairing of one kind with one caller-supplied `OperationAttrs` value | Implemented descriptor with derived signature |

A kind distinguishes computations, while attributes carry parameters within a computation family.
`Operation` stores both as one value and validates exact family compatibility through the derived
signature. None of these
values identifies where computation occurs in a graph; an implemented [node](#node) represents
that occurrence. Binary arithmetic, binary comparison, boolean logical, conditional selection,
unary elementwise, floating-classification, scalar elementwise, cast, and aggregate reduction
kinds are implemented. Arithmetic, unary, floating-classification, scalar, and comparison public
Tensor construction paths are also implemented,
together with boolean logical, conditional-selection, cast, and
sum/mean/product/minimum/maximum/all/any/arg-min/arg-max aggregate Tensor construction, including
ordered multi-axis forms, floating log-sum-exp, corrected variance/standard deviation, L1/L2
norms, and masked sum/mean. Cumulative-sum, softmax/log-softmax, layer-normalization, and RMS-
normalization semantics
and public Tensor construction are also implemented. Contiguous-request semantics and public contiguous Tensor construction are
also
implemented. Reshape and expand semantics plus target-shape attributes are implemented; public
reshape and expand Tensor construction is current. Axis-transform semantics, complete permutation
attributes, and single-axis insertion/removal attributes are implemented; public permute and
transpose, expand-dimensions, and squeeze Tensor construction is current. Slice extraction and
functional-update semantics, normalized parallel and exact crop-Shape attributes, plus public
general/single-axis slice, slice-update, and target-relative crop Tensor construction are current.
Pad and tile semantics, normalized immutable attributes, and public Tensor construction
are current. Tensor-composition semantics, normalized axis/index attributes, and public concat,
stack, and immutable-list unstack Tensor construction are current. Scalar-select semantics and
normalized axis/index attributes plus public scalar-select Tensor construction are current.
Axis-gather, Gather-ND, axis-scatter, and Scatter-ND semantics, attributes, and public Tensor
construction are also current. Other
concrete families and their family-specific
attributes, compiler capture and canonicalization, materialization policy, and execution remain
planned.
The compiled graph container is implemented model state.

### Compile versus prepare versus run

| Stage | Main question | Produces | Must not do |
|---|---|---|---|
| Compile | What does the graph mean, and which backend owns each region? | Immutable compile artifacts | Allocate physical buffers or choose kernels |
| Prepare | How will each owned region execute? | Reusable prepared executables, memory plan, and schedule | Perform global graph optimization or per-run work |
| Run | Execute this prepared schedule with these inputs. | Per-run state and published results | Discover backends, lower partitions, or select kernels |

### Logical `ValueId` versus physical memory slot

A `ValueId` names logical data in one compiled graph. Current `BufferSlot` and `WorkspaceSlot`
instead name nominally separate plan-local positions without naming physical storage. Current
`PreparedMemoryPlan` associates those slot identities only with final byte size and alignment.
Current Prepare assignment maps declared source requirements to slots conservatively; later
Runtime/backend state binds those positions to per-run physical representations. Any future reuse
requires a proved lifetime/interference model. The slot and plan records themselves still contain
no source mapping or physical binding, so graph and analysis identities remain distinct from both
slot identities and physical resources.
