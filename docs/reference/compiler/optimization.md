# [Canonicalization](../../glossary.md#occurrence-identity-and-canonical-form) and whole-[graph optimization](../../glossary.md#forward-graph-optimization)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

**Canonicalization** gives an equivalent [graph](../../glossary.md#graph) a consistent internal numbering and representation.
**Optimization** then removes or combines work only where the documented exact rules allow it.
[Dead-code elimination](../../glossary.md#term-dead-code-elimination-dce) removes work not needed by outputs; [common-subexpression elimination](../../glossary.md#term-common-subexpression-elimination-cse) shares
equivalent occurrences where permitted. Neither step chooses a [backend](../../glossary.md#backend) [kernel](../../glossary.md#kernel) or reads [Tensor](../../glossary.md#tensor) values.

<a id="current-package-private-canonicalization-and-whole-graph-optimization"></a>
### Current package-private [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form) and whole-[graph optimization](../../glossary.md#forward-graph-optimization)

Package-private `ForwardGraphOptimization.optimize(ValidatedGraph, GraphOptimizationConfig)`
consumes only a successful verification result and the standalone optimization permission. It
always rebuilds the [graph](../../glossary.md#graph) through `GraphCanonicalization` before validating the rebuilt candidate.
Graph inputs receive dense [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) values first in boundary order. [Node](../../glossary.md#node) outputs then receive
dense values in stored [topological](../../glossary.md#term-topological-order) and output-slot order, while [`NodeId`](../../glossary.md#nodeid) values follow stored node
order. Operations, [descriptors](../../glossary.md#term-descriptor), phases, repeated input positions, every output slot, and ordered
graph boundaries are preserved.

When optional optimization is disabled, that canonicalization and [validation](../../glossary.md#term-validation) still occur. When it
is enabled, the internal pipeline runs exactly once in this order:

```text
guarded exact arithmetic rewriting -> constant folding
                                   -> sidecar-aware whole-graph dead-code elimination
                                   -> exact phase-local common-subexpression elimination
                                   -> sidecar-aware whole-graph dead-code elimination cleanup
```

The first scan recognizes exactly seven internal one-output [identities](../../glossary.md#term-identity-meanings) in either [graph phase](../../glossary.md#graph-phase):
duplicate-input
[binary](../../glossary.md#term-binary-descriptor-and-metadata) `MIN` and `MAX`; [scalar](../../glossary.md#term-scalar) `MUL` by exact typed positive one for [`BFLOAT16`](../../glossary.md#term-bfloat16), [`FLOAT32`](../../glossary.md#term-float32),
[`FLOAT64`](../../glossary.md#term-float64), [`INT32`](../../glossary.md#data-type--datatype), and [`INT64`](../../glossary.md#data-type--datatype); scalar `DIV` and `POW` by exact typed positive one for the three
floating types; and scalar `ADD` and `SUB` by exact typed zero for `INT32` and `INT64`. Every bypass
requires complete equality between the input and output `TensorDescriptor`, including [Shape](../../glossary.md#shape),
[layout](../../glossary.md#layout), [data type](../../glossary.md#data-type--datatype), and `requiresGrad`; the equal descriptor must have `requiresGrad == false`.
The result must not be a graph output. The [occurrence](../../glossary.md#term-occurrence) phase is preserved. Duplicate [extrema](../../glossary.md#saturation-pole-and-extremum)
compare already-remapped input IDs, while
scalar rules require the exact `ScalarValueAttrs` [carrier](../../glossary.md#term-storage-carrier), matching scalar/input/output type, and
matching typed value. A prior bypass can therefore expose a later eligible occurrence during the
same [scan](../../glossary.md#model-module-buffer-and-scan-distinctions) without iteration.

`ScalarValueAttrs` is [immutable](../../glossary.md#term-immutable) [operation](../../glossary.md#operation) [metadata](../../glossary.md#term-metadata) already stored in the graph. Reading its exact
typed [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) for an [identity](../../glossary.md#term-identity-meanings) proof is not [Tensor](../../glossary.md#tensor) [constant](../../glossary.md#term-constant) discovery or [host](../../glossary.md#term-host)-storage
inspection. The scan deliberately does not add floating ADD/SUB-zero, multiplication by
zero, [cancellation](../../glossary.md#term-cancellation), other powers, bounds/clamp identities, broader algebra, or relaxed numerical
rules. It retains graph-output, [gradient](../../glossary.md#term-gradient)-eligible, and multi-output occurrences.

The following constant-folding scan reads only [logical splat](../../glossary.md#logical-splat) facts already present in the
[compiler](../../glossary.md#term-compiler) sidecar. It folds exactly these current rows:

| Family | Current selected kinds and domains |
|---|---|
| [Boolean](../../glossary.md#data-type--datatype) logical | `NOT`, `AND`, and `OR` over exact [`BOOL`](../../glossary.md#data-type--datatype) splats |
| Binary arithmetic | `ADD`, `SUB`, `MUL`, `MIN`, and `MAX` over `INT32`/`INT64` splats, including mixed-width signed promotion |
| Binary comparison | All six comparisons over `INT32`/`INT64` splats, producing exact `BOOL` splats |
| Scalar [elementwise](../../glossary.md#term-elementwise) | `ADD`, `SUB`, `MUL`, `MIN`, and `MAX` over one integral splat and an exact same-typed integral `ScalarValueAttrs` value |

Integral addition, subtraction, and multiplication use the current fixed-width modular semantics;
extrema and comparisons use signed order. If either binary operand is `INT64`, an `INT32` operand
is sign-extended and the result domain is `INT64`. The scalar input remains the left operand, so
subtraction preserves operand order. A folded splat can feed a later selected occurrence during
the same topological scan, but the compiler does not seek a fixed point or [run](../../glossary.md#term-run) a second folding
scan.

Each selected fold replaces one internal, one-output, non-gradient occurrence in either phase with
one
synthetic constant graph input. Original inputs remain first; synthetic sources follow in original
fold-occurrence order; retained outputs and nodes are then rebuilt densely and deterministically.
Graph-output [producers](../../glossary.md#term-producer), all multi-output producers, and gradient-eligible values remain
occurrences. Equal splats are not interned.

Floating and BFLOAT16 splats retain exact type and bits in the sidecar, including signed zeros and
[NaN](../../glossary.md#term-nan) payloads, but no floating or BFLOAT16 operation is evaluated. Casts, unselected scalar kinds,
[unary](../../glossary.md#term-unary) numerical functions, reductions, scans, [normalization](../../glossary.md#term-normalization), [loss](../../glossary.md#loss), [linear](../../glossary.md#term-linear-meanings) algebra, indexing,
layout/[view](../../glossary.md#term-view-meanings) operations, random/state operations, and generic partial [evaluation](../../glossary.md#term-evaluation) are also outside
the current fold [matrix](../../glossary.md#term-matrix-meanings). The compiler reads no [host storage](../../glossary.md#host-storage), creates no dense constant [payload](../../glossary.md#artifact-envelope-and-atomic-publication),
and allocates no physical value.

[Dead-code elimination](../../glossary.md#term-dead-code-elimination-dce) ([DCE](../../glossary.md#compiler-transformations-and-dominance)) derives [liveness](../../glossary.md#term-liveness) only from the complete ordered graph-output boundary,
so an unused `BACKWARD` node is removable just like unused `FORWARD` work. Every live occurrence
retains all output slots. The sidecar-aware overload then removes only fixed constant inputs that
are neither graph outputs nor inputs to a retained node; it always preserves bindable inputs,
even when unused.

Exact [common-subexpression elimination](../../glossary.md#term-common-subexpression-elimination-cse) ([CSE](../../glossary.md#compiler-transformations-and-dominance)) compares graph phase, complete immutable operation
value, ordered remapped inputs, and every ordered output descriptor. It merges a whole
multi-output occurrence slot by slot within `FORWARD` or within `BACKWARD`, never across phases.
A [producer](../../glossary.md#term-producer) containing a graph output may neither merge nor serve as a representative, so
requested boundaries remain distinct. Fixed recurrent-scan occurrences are also ineligible
because their [Model](../../glossary.md#term-model) producer identity defines a distinct occurrence even when all structural
facts compare equal. Each changed candidate is immediately revalidated through the verification
pass; an unchanged helper result is not redundantly revalidated.

These are internal graph transformations, not a public compiler surface. They do not rewrite
casts, broader arithmetic, algebra, or views; construct [autograd](../../glossary.md#autograd); derive [publication](../../glossary.md#term-publication), [planning](../../glossary.md#term-planning), or
diagnostics; or affect [trace](../../glossary.md#trace), preparation, [runtime](../../glossary.md#term-runtime), [backend](../../glossary.md#backend), or [engine](../../glossary.md#term-engine) behavior. The current
complete compiler entry transports the exact immutable facts into `CompileConstantPlan`, exposing
only derived bindable inputs and exact logical-splat sources. Planning sees the [topology](../../glossary.md#term-topology) and
semantic operations left by those transformations. [Prepare](../../glossary.md#prepare) and backend work own physical splat
[materialization](../../glossary.md#materialization), storage allocation, [lowering](../../glossary.md#lowering), and execution.

After this final optimization has fixed the surviving topology, Compiler closes one narrower
descriptor case before deriving forward or gradient [bindings](../../glossary.md#term-binding), publication, [backend ownership](../../glossary.md#backend-ownership),
partitions, or logical memory. A final [graph value](../../glossary.md#graph-value) receives the [canonical](../../glossary.md#term-canonical-form) [contiguous](../../glossary.md#term-contiguous) logical
layout for its fully static Shape only when it is all of the following: a graph input, an explicit
[compile](../../glossary.md#compile)-time splat, a graph output, absent from every final node input, and still layout-
unresolved. This is a source-only published compile-time constant. Dynamic Shapes, consumed or
unpublished constants, caller-bindable inputs, produced values, non-constants, and values with an
already resolved layout are unchanged.

The closure preserves the value's data type, Shape, gradient eligibility, `ValueId`, splat, and
publication role. It also preserves graph and boundary order, exact node references, phases,
deferred constraints, bindable Tensor identities, and [derivative](../../glossary.md#term-derivative)-order metadata. Scalars receive
the Model's [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) span-one contiguous layout; static Shapes with a zero [extent](../../glossary.md#term-extent) receive the
same canonical layout factory's span-zero result; all other eligible static Shapes receive its
checked [row-major](../../glossary.md#term-row-major) geometry. [Overflow](../../glossary.md#term-overflow) while deriving canonical [strides](../../glossary.md#term-stride) or [referenced span](../../glossary.md#referenced-element-span) rejects
[compilation](../../glossary.md#compiler-engine-and-execution-stage-names) with the affected `ValueId` and the arithmetic failure as the cause.

This closure is logical descriptor completion only. It creates no node, [partition](../../glossary.md#partition), byte or
[alignment](../../glossary.md#term-alignment) declaration, slot, representation, initialized value, backend choice, caller binding,
or executable publication. Planning therefore still reports a source-only published value as
graph-output-required with no producer or consumer partition. The value is not yet preparable or
executable from this Compiler fact alone: current Prepare and [CPU](../../glossary.md#term-cpu) behavior separately own resource
[handoff](../../glossary.md#term-handoff) and physical materialization.

The Compiler then performs one final [convolution](../../glossary.md#term-channels-first-convolution-layers) logical-layout closure. It visits final nodes in
stored order and assigns `LayoutDescriptor.contiguous(shape)` if and only if a final [`CONV2D`](../../glossary.md#convolution--conv2d) or
[`CONV3D`](../../glossary.md#conv3d--ncdhw) occurrence has its validated single output, the output Shape is fully static, and its
layout remains unresolved. The replacement preserves the exact data type, Shape,
`requiresGrad` value, `ValueId`, operation occurrence, and graph topology. Fully or partially
dynamic outputs and already resolved outputs remain unchanged; the Compiler does not bind a
[dimension](../../glossary.md#dimension) or inspect Tensor storage to make a Shape static.

One additional rule carries this newly selected fact through the public [Conv1d composition](../../glossary.md#conv1d-composition). An
unresolved `SQUEEZE(axis=2)` output receives a view layout only when its exact input is a Conv2d
output newly closed by the same invocation, that input has a fully static [rank](../../glossary.md#term-rank)-four Shape with
extent one at [axis](../../glossary.md#term-axis) two, and the validated output is the corresponding rank-three Shape. The
Compiler removes stride index two, preserves the input [storage offset](../../glossary.md#term-storage-offset), and marks the result as a
view. It does not identify a synthetic Conv1d operation or propagate layout through another
squeeze, expand-dimensions, [permutation](../../glossary.md#permutation), reshape, contiguous, [slice](../../glossary.md#slice), publication boundary, or
unrelated operation.

Both replacements use the Model's checked layout factories. Arithmetic overflow rejects
compilation with operation-occurrence and value context while retaining the arithmetic failure as
the cause. If neither rule is eligible, the exact validated graph state is returned. Otherwise
only the changed graph-value descriptors are rebuilt; node and operation references, IDs,
boundaries, phases, constants, caller-bindable Tensor IDs, deferred constraints, derivative-order
values, publication identities, optimization results, and ordering are preserved.

This convolution closure runs after final optimization, validation, and published compile-time
constant descriptor closure, and before forward or gradient binding derivation, publication,
Planning capability queries, partitioning, or logical-memory planning. Those consumers therefore
observe the final descriptors. The closure asserts no [backend capability](../../glossary.md#backend-capability) and chooses no physical
representation, allocation, storage, materialization, preparation, [kernel](../../glossary.md#kernel), or execution behavior.
It is not a general layout-[inference](../../glossary.md#term-inference) or propagation policy.

The Compiler then applies one operation-aware static-result logical-layout closure. A fully static,
layout-unresolved result receives `LayoutDescriptor.contiguous(shape)` only when its exact
operation family proves a newly materialized logical result, or when the operation is the explicit
`CONTIGUOUS` request. The allowlist covers the current elementwise, indexing, functional update,
materializing movement, linear-algebra, loss, normalization, ordering, [pooling](../../glossary.md#term-pooling), random, recurrent,
reduction, scan, and attention result families. Unknown future kinds remain unresolved. Exact
unary NEG results are one ordinary allowlisted case, replacing the former NEG-only pass.

Affine/view transforms are excluded from that family-wide classifier. Their supported views keep
the exact input-derived offsets/strides, never a canonical replacement. The earlier specialized
convolution closure continues to derive the direct Conv1d squeeze view. The second topological
phase below also distinguishes a selected materializing RESHAPE from a proved view.

After materialized results close, a topological view pass retries these transforms using the latest
input descriptors. Scalar `SELECT` reuses indexing-family inference; [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) `SLICE` and
[target](../../glossary.md#term-target)-relative crops reuse layout-family inference. Only exact inferred layouts whose data type
and Shape match the captured output are accepted. For example, selecting axis 0 at index 2 from a
newly closed `[3,2,2]` product gives a `[2,2]` view with strides `[2,1]` and [element](../../glossary.md#term-element) offset `8`.
A gate slice selecting columns 2 and 3 of a `[2,6]` projection keeps strides `[6,1]` and offset `2`,
not canonical `[2,1]` geometry. Later views in the same pass consume those exact descriptors.
Negative-step and empty extraction remain unresolved during raw inference. Final ordinary layout
completion selects canonical non-view storage for qualified fully static signed/empty
`SliceAttrs` results with known resolved nonnegative input geometry. This is materialization,
not a signed-stride or empty [alias](../../glossary.md#aliasing-and-arena). Unknown input geometry and other unproved layouts remain
unresolved; backend [admission](../../glossary.md#term-admission) and the actual physical copy remain independent obligations.

After exact view inference, a validated fully static RESHAPE with a present noncontiguous source
layout and no inferred supported view may select canonical non-view output geometry. For example,
`[3,2]/[1,3] -> [6]` becomes a compact logical value, not an invented source alias. Scalar/empty
Shapes retain ordinary checked stride arithmetic. Missing source layouts and dynamic Shapes stay
unresolved; existing resolved/offset views remain exact. This final choice changes no raw Model,
source/cold inference, derivative formula, node/identity/sidecar or wire record. CPU/[Metal](../../glossary.md#term-metal) admission
and actual physical copy remain independent backend obligations.

Explicit compile-time splat graph inputs receive canonical logical geometry because they have no
caller binding whose descriptor must remain exact. The pass preserves caller-bindable graph
inputs, dynamic or partially dynamic Shapes, and every already resolved layout.

Rank-zero and zero-extent Shapes use the Model's existing checked canonical layout rules. Checked
stride or referenced-span overflow rejects compilation with the affected node-output or splat
input identity and retains the arithmetic failure as the cause. Resolving a descriptor does not
admit its operation: capability providers still validate exact kind, attributes, [arity](../../glossary.md#term-arity), data types,
layout restrictions, gradient metadata, and backend availability, while backend lowering
independently revalidates complete partition topology and executable geometry.

Static-result closure runs after optimization, validation, published-constant closure, and the
specialized convolution/Conv1d-view closure in both forward-only and backward-capable compilation.
It completes before forward and gradient bindings, publication, constant/bindable source
projection, capability queries, [owner](../../glossary.md#term-owner) selection, partitioning, and logical-memory planning. The
pass changes only eligible `GraphValue` descriptors and graph-owning immutable sidecars. It
preserves topology, IDs, exact node and operation references, boundary order, phases, constants,
bindable Tensor identities, constraints, derivative-order metadata, and optimization results, and
returns the exact validated graph object when no value is eligible.

This is backend-neutral logical descriptor completion, not physical layout selection. It creates
no storage, representation, allocation, slot, route, kernel, executable, or backend capability.
Concrete backends remain responsible for lowering the completed logical geometry to physical
representations and [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) when an otherwise completed occurrence is outside their supported
matrix.


---

[Previous topic](verification-scan.md) · [Next topic](autograd.md)
