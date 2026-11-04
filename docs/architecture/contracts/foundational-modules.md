# Foundational [module](../../glossary.md#term-module) contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.<span data-glossary-navigation="true"> ([normative glossary](../../glossary.md#term-normative); [scope glossary](../../glossary.md#term-scope); [dependency glossary](../../glossary.md#term-dependency))</span>

## [Scope](../../glossary.md#term-scope)

This contract owns the detailed responsibilities and prohibitions for [Trace](../../glossary.md#trace), [Backend](../../glossary.md#backend) Contract,
[Model](../../glossary.md#term-model), Config, and [Planning](../../glossary.md#term-planning). Model's scope includes the sole [normative](../../glossary.md#term-normative) definition of profile-free
[operation](../../glossary.md#operation)-family numerical semantics and allowed results. Planning scope here is the [module](../../glossary.md#term-module)
boundary and [ownership](../../glossary.md#term-ownership-meanings) question; the detailed [partition](../../glossary.md#partition)-scoring algorithm boundary belongs to the
[compiler](../../glossary.md#term-compiler)/[autograd](../../glossary.md#autograd) contract. This file does not own recurrent-[scan](../../glossary.md#term-cumulative-scan) semantics, compiler behavior,
execution lifecycles, [concrete backend](../../glossary.md#concrete-backend) implementation, or extensions.

## [Module](../../glossary.md#term-module) responsibilities

### `modules/trace`

`modules/trace` owns neutral typed diagnostic envelopes, correlation IDs and shared [payload](../../glossary.md#artifact-envelope-and-atomic-publication) DTOs.
The open [`TracePayload`](../../glossary.md#trace-payload--tracepayload) marker also permits [immutable](../../glossary.md#term-immutable) [producer](../../glossary.md#term-producer)-specific DTOs in their owning
[module](../../glossary.md#term-module). Those DTOs contain [detached](../../glossary.md#detached-model-results) diagnostic values only, not [graph](../../glossary.md#graph), [Tensor](../../glossary.md#tensor), storage, native
handle, exception or producer-domain object references. Producer [ownership](../../glossary.md#term-ownership-meanings) does not introduce
any [dependency](../../glossary.md#term-dependency) from [Trace](../../glossary.md#trace) back to [Engine](../../glossary.md#term-engine) or a [backend](../../glossary.md#backend).

Allowed:

- trace event envelopes
- typed [compile](../../glossary.md#compile) payloads
- typed [prepare](../../glossary.md#prepare) payloads
- typed [run](../../glossary.md#term-run) payloads
- typed backend payloads
- trace-local IDs
- typed trace attributes

Forbidden:

- importing [model](../../glossary.md#term-model)
- importing [planning](../../glossary.md#term-planning)
- importing [compiler](../../glossary.md#term-compiler)
- importing [runtime](../../glossary.md#term-runtime)
- importing prepare
- importing engine
- importing concrete backends
- graph traversal
- backend execution
- business logic
- runtime state

Trace must use typed DTOs.

`Map<String,String>` must not be used as the primary trace model.

Backend-specific details may use typed `TraceAttributes` as an escape hatch.

### `modules/backend-contract`

`modules/backend-contract` owns minimal [backend](../../glossary.md#backend) [identities](../../glossary.md#term-identity-meanings) and declarative requirements.

Allowed:

- [`BackendId`](../../glossary.md#backend-identity--backendid)
- [`BackendDeviceId`](../../glossary.md#backend-device-identity--backenddeviceid)
- [`BackendAvailabilitySnapshot`](../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot)
- [`BackendRequirement`](../../glossary.md#backend-requirement--backendrequirement)
- [`DeviceClass`](../../glossary.md#device-class--deviceclass)

Forbidden:

- [kernel](../../glossary.md#kernel) [registry](../../glossary.md#term-registry)
- [operation](../../glossary.md#operation) support logic
- backend [prepare](../../glossary.md#prepare) services
- executable units
- [runtime](../../glossary.md#term-runtime) storage
- physical [buffers](../../glossary.md#term-buffer)
- [cost model](../../glossary.md#compiler-transformations-and-dominance) implementation

[Compile](../../glossary.md#compile)-time plans must hold [backend identity](../../glossary.md#backend-identity--backendid), not live backend services.

Use `BackendId`, not [concrete backend](../../glossary.md#concrete-backend) objects, in compile-time [ownership](../../glossary.md#term-ownership-meanings) and partitioning data.

### `modules/model`

`modules/model` owns the public [tensor](../../glossary.md#tensor) [model](../../glossary.md#term-model), [operation](../../glossary.md#operation) semantics, [shape](../../glossary.md#shape)/[data type](../../glossary.md#data-type--datatype)/[layout](../../glossary.md#layout) model, [host storage](../../glossary.md#host-storage) abstraction, and [immutable](../../glossary.md#term-immutable) [graph](../../glossary.md#graph) model.

Allowed:

- `Tensor`
- [`TensorId`](../../glossary.md#tensorid)
- `TensorFactory`
- [`DataType`](../../glossary.md#data-type--datatype)
- `Shape`
- `LayoutDescriptor`
- `HostTensorStorage`
- `MemorySegmentStorage`
- `Operation`
- [`OperationAttrs`](../../glossary.md#operationattrs)
- [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel)
- `CompiledNode`
- `GraphValue`
- [`NodeId`](../../glossary.md#nodeid)
- [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot)
- `GraphPhase`
- `ForwardPublicationBinding`
- `TensorDescriptor`
- operation-family formulas and allowed-result sets

Forbidden:

- [backend](../../glossary.md#backend) support
- `supportedBackends()`
- [device](../../glossary.md#term-device) [residency](../../glossary.md#residency)
- [runtime](../../glossary.md#term-runtime) workspaces
- physical device [buffers](../../glossary.md#term-buffer)
- [kernel](../../glossary.md#kernel) selection
- backend-specific storage
- [prepared execution](../../glossary.md#prepared-execution--preparedexecution)
- runtime state

`Tensor` may retain its existing [mutable](../../glossary.md#term-mutable) [borrowed](../../glossary.md#ownership-borrowing-and-lease) [host](../../glossary.md#term-host)-storage association, but it must not own
[gradient](../../glossary.md#term-gradient) state or runtime device residency. Its [identity](../../glossary.md#term-identity-meanings), [descriptor](../../glossary.md#term-descriptor), and expression [provenance](../../glossary.md#provenance)
remain immutable.

Every derived expression [producer](../../glossary.md#term-producer) owns one operation [occurrence](../../glossary.md#term-occurrence), its ordered input tensors, its
ordered output descriptors, and the [canonical](../../glossary.md#term-canonical-form) exact `Tensor` wrapper for every output slot.
`TensorFactory` constructs those wrappers atomically with their indexed provenance. A producer
must return the retained wrapper for a slot rather than reconstructing an equal wrapper. This
pre-[capture](../../glossary.md#term-capture-meanings) model relationship is not graph [IR](../../glossary.md#term-intermediate-representation-ir), graph membership, or graph-local identity.

The accepted [general Scan](../../glossary.md#general-differentiable-scan) continuation adds a context-qualified overload on the existing Model
[owner](../../glossary.md#term-owner): `ScanBody.describeOrdinary(Operation, List<TensorDescriptor>, int, Optional<DynamicDimension>)`.
It reuses ordinary [inference](../../glossary.md#term-inference) and the same factored source/deferred symbolic proof with the exact
declared [symbol](../../glossary.md#term-symbol). The three-[argument](../../glossary.md#term-argument) ordinary [API](../../glossary.md#term-application-programming-interface-api) keeps its deferred semantics. This is descriptor
proof, not [Compiler](../../glossary.md#term-compiler) inference, ROW provenance, backend [admission](../../glossary.md#term-admission) or a new public type; the
[Scan owner](recurrent-scan.md#general-ordinary-unit-scan-cutover) defines its semantic obligations.

The reviewed logical-layout prerequisite adds two [metadata](../../glossary.md#term-metadata) operations on the same existing owner:
`ScanBody.closeOrdinaryLayouts(Operation, List<TensorDescriptor>, List<TensorDescriptor>)` and
`ScanBody.hasCanonicalContiguousResults(Operation)`. The former completes eligible unresolved
static computed results through the moved ordinary closure and existing family layout helpers;
the latter preserves the exact generic [materialization](../../glossary.md#materialization) fact also consumed by Compiler symbolic
[binding](../../glossary.md#term-binding). Specialized [convolution](../../glossary.md#term-channels-first-convolution-layers) completion does not widen that classifier. Neither operation
changes ordinary construction, performs independent type/Shape inference or grants [PURE](../../glossary.md#term-pure)/[Scan](../../glossary.md#term-cumulative-scan)/
backend admission. Preserve resolved layouts, exact affine [views](../../glossary.md#term-view-meanings) and unresolved unproved cases.
No new production type, Tensor/ID allocation or cross-[module](../../glossary.md#term-module) [dependency edge](../../glossary.md#term-dependency-edge) is selected.

The user-authorized backend coverage completion extends that existing completion decision for
validated fully static signed-step or empty [SLICE](../../glossary.md#slice) extraction: when an exact nonnegative-[stride](../../glossary.md#term-stride)
view is unproved, select canonical non-view materialization. Raw family/source/wire inference
remains unchanged, and no negative physical stride, sampled symbolic [extent](../../glossary.md#term-extent), new operation owner
or [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) [normalization](../../glossary.md#term-normalization) is introduced. Flat Compiler and containing-body physical closure
consume the same decision; ordinary and Scan execution still qualify independently.

The reviewed source-[publication](../../glossary.md#term-publication) prerequisite renames the existing [Prepare](../../glossary.md#prepare) contribution owner
to [`ProducerlessPublishedValueResource`](../../glossary.md#producerless-published-value-resource--producerlesspublishedvalueresource), retaining five components: stable value, logical
requirement, backend owner, existing Runtime `ByteSize size`, and [byte alignment](../../glossary.md#term-byte-alignment). The migrated
default assembler hook takes value, requirement and `Optional<ScalarValue>`: present only for
an authenticated [constant](../../glossary.md#term-constant), empty only for an authenticated caller. Prepare owns complete source/
publication roles and ordering; [CPU](../../glossary.md#term-cpu) owns physical geometry and fixed-unit compatibility.
No Model mathematics, Runtime type/field, dependency edge or parallel contribution owner is added.

The accepted bounded scan exception adds source-body containment on [TensorProducer](../../glossary.md#producer--tensorproducer), distinct from
the single compiled-body attachment on CompiledNode. The complete inventory, canonical [residual
port](../../glossary.md#scan-residual-port-and-history-demand) construction, formal/role binding and scoping semantics belong solely to the
[scan semantic owner](recurrent-scan.md#general-differentiable-scan). Model0035 implements and
validates these Java source/IR attachments. Planning0008 implements query transport, and
Compiler0011 implements scoped capture/inference, first-order [AD](../../glossary.md#autograd) and final query/demand emission.
Prepare0010 implements scoped projection and static binding/demand [validation](../../glossary.md#term-validation). CPU0030 implements
the bounded [FLOAT32](../../glossary.md#term-float32) same-shape [pointwise](../../glossary.md#term-pointwise) forward/actual-reverse execution slice. CPU0031 adds
qualified FLOAT32 broadcast, SUM/sumToShape and bounded shape/layout closure, with actual reverse
bodies independently qualified. CPU0032 adds static [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul) and
independently qualified actual reverse bodies. CPU0033 implements homogeneous [FLOAT64](../../glossary.md#term-float64) across
the complete bounded declared algebra and actual first-order reverse closure; source admission
still grants no generic [backend capability](../../glossary.md#backend-capability). Preserve
empty-attachment constructor compatibility. OperationAttrs remains Tensor/graph-free [scalar](../../glossary.md#term-scalar)
semantic metadata; no body [registry](../../glossary.md#term-registry) or source closure belongs in attrs.

Device storage belongs to runtime/backend layers, not model.

Model0038 implements the length-aware header, overload, bounded row-role source subset and outer/local
count distinction defined only by the [scan owner](recurrent-scan.md#header-bindings-and-source-admission).
Model construction/structural attachments do not grant Compiler, cold, AD or execution capability.
Compiler0013 independently implements scoped length inference and bounded first-order reverse
construction/query emission; cold, Prepare, CPU and encoding remain closed.
The old all-valid PURE/type admission remains unchanged; no new production type or [dependency](../../glossary.md#term-dependency) is selected.

`ScanBody.describeOrdinary` exposes descriptor-only checking through the same existing ordinary
family guards used by Tensor construction. It returns all canonical output descriptors without
creating Tensors, producers, storage or identifiers. This Model check is not independent Compiler
inference and grants no row-coordinate, AD, execution or [persistence](../../glossary.md#term-persistence) capability.

The separately accepted [Scan coordinate seam](recurrent-scan.md#bounded-row-role-analysis)
recomputes per-canonical-output row/dependency/terminal semantics from independently checked
descriptors. It is not descriptor inference or an admission [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions). Model0048 is implemented
and source-qualified; source validation and raw Compiler use the same semantic owner while
retaining independent ordinary descriptor guards and downstream support boundaries.

### [Profile-free numerical semantics](../../glossary.md#profile-free-numerical-semantics)

<a id="numerical-profiles"></a>

The legacy anchor above preserves links from historical profile-era records. [Model](../../glossary.md#term-model)
alone owns each [operation](../../glossary.md#operation) family's formula, contributor domain, guards, [precision](../../glossary.md#term-precision)/[rounding](../../glossary.md#term-rounding) sites,
special-value behavior, and allowed results. There is no numerical selector, strict/[accelerator](../../glossary.md#term-accelerator)
result-set union, public accuracy envelope, production [tolerance](../../glossary.md#term-tolerance) [registry](../../glossary.md#term-registry), or [runtime](../../glossary.md#term-runtime) numerical
[certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions). [Backend](../../glossary.md#backend) support remains an [occurrence](../../glossary.md#term-occurrence)-specific provider answer, not a consequence of
removing a selector.

#### Family formulas and [unary](../../glossary.md#term-unary) sites

Every current unary kind retains its formula for each accepted floating type, including both
[OFP8](../../glossary.md#term-ofp8) types below. ABS and NEG perform exact represented sign clearing/inversion. SIGN, FLOOR, CEIL,
and RELU retain their exact represented-value family rules. Those six sites never use arithmetic
[DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz). RECIPROCAL is one division at the declared working [dtype](../../glossary.md#data-type--datatype), not an implicit reciprocal
estimate. [LOG](../../glossary.md#term-log), LOG1P, [EXP](../../glossary.md#term-expx), EXPM1, [ERF](../../glossary.md#term-erf), SQRT, and [TANH](../../glossary.md#term-tanh) retain their mathematical function, domain,
range, class, and sign rules. LOG1P refers to the real `ln(1+x)` and EXPM1 to real `exp(x)-1`;
their reference additions/subtractions are not separately rounded primitive sites. ERF retains
`erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t*t) dt` as its mathematical reference.
RSQRT and the [activation](../../glossary.md#term-activation) composites retain the named-site [dependency](../../glossary.md#term-dependency) formulas below; none is
one blanket approximate output. A named real [constant](../../glossary.md#term-constant) is rounded before first use as the family
specifies. [Finite](../../glossary.md#term-finite) accuracy is qualified with test-only [backend](../../glossary.md#backend)/route/family/dtype/size tolerances
and an explicit [oracle](../../glossary.md#term-test-oracle), [metric](../../glossary.md#term-metric), threshold, and [cancellation](../../glossary.md#term-cancellation) boundary, not [Model](../../glossary.md#term-model)-owned
ordered-distance or forward-error envelopes.

The following top-level domain and special-value results override the ordinary-finite constructions.
Every listed [NaN](../../glossary.md#term-nan) result must be NaN, but [payload](../../glossary.md#artifact-envelope-and-atomic-publication), quieting, and sign are unspecified.

| Kinds | [Signed zero](../../glossary.md#term-signed-zero), [infinity](../../glossary.md#term-infinity), domain, and NaN rules |
|---|---|
| `ABS` | Either zero becomes `+0`; either infinity becomes `+infinity`; NaN remains NaN. |
| `NEG` | The sign of zero and infinity is inverted; NaN remains NaN. |
| `RECIPROCAL` | Each signed zero becomes same-signed infinity; each infinity becomes same-signed zero; NaN remains NaN. |
| `LOG` | Either zero becomes `-infinity`; negative finite inputs and `-infinity` become NaN; `+infinity` remains; NaN remains NaN. |
| `LOG1P` | Signed zero is preserved; `-1` becomes `-infinity`; values below `-1`, including `-infinity`, become NaN; `+infinity` remains; NaN remains NaN. |
| `EXP` | Either zero becomes `+1`; `-infinity` becomes `+0`; `+infinity` remains; NaN remains NaN. |
| `EXPM1` | Signed zero is preserved; `-infinity` becomes `-1`; `+infinity` remains; NaN remains NaN. |
| `ERF` | Signed zero is preserved; each infinity becomes the same-signed unit value; NaN remains NaN. |
| `SQRT` | Signed zero is preserved; negative finite inputs and `-infinity` become NaN; `+infinity` remains; NaN remains NaN. |
| `RSQRT` | Each signed zero becomes same-signed infinity; negative finite inputs and `-infinity` become NaN; `+infinity` becomes `+0`; NaN remains NaN. |
| `FLOOR`, `CEIL` | Signed zero and signed infinity are preserved; NaN remains NaN. |
| `SIGN` | Signed zero is preserved; [infinities](../../glossary.md#nan-infinity-and-signed-zero) become same-signed unit values; NaN remains NaN. |
| `RELU` | Negative finite inputs, `-infinity`, and either zero become `+0`; `+infinity` remains; NaN remains NaN. |
| [`SIGMOID`](../../glossary.md#term-sigmoid) | Either zero becomes `0.5`; `-infinity` becomes `+0`; `+infinity` becomes `+1`; NaN remains NaN. |
| `TANH` | Signed zero is preserved; infinities become same-signed unit values; NaN remains NaN. |
| [`GELU`](../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, `SILU` | `-infinity` becomes `-0` by continuous extension; signed zero is preserved; `+infinity` remains; NaN remains NaN. |

#### Arithmetic-site [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz)

[Denormals-are-zero](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) (DAZ) may read a [subnormal](../../glossary.md#term-subnormal) named floating arithmetic primitive input as
same-[signed zero](../../glossary.md#term-signed-zero) without changing its stored bits. [Flush-to-zero](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) (FTZ) may map a [finite](../../glossary.md#term-finite) subnormal
named primitive result to zero subject to the existing family-specific [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) rule; it
grants no universal choice of zero sign.

| Primitive-site [dtype](../../glossary.md#data-type--datatype) | Input DAZ | Result FTZ |
|---|---|---|
| [FLOAT32](../../glossary.md#term-float32) | Permitted | Permitted |
| [BFLOAT16](../../glossary.md#term-bfloat16) | Permitted, including multiplication | Permitted, including multiplication |
| [FLOAT16](../../glossary.md#term-float16) | Permitted, including multiplication | Permitted, including multiplication |
| FLOAT8_E4M3 / FLOAT8_E5M2 | Permitted at [F32](../../glossary.md#floating-point-precision-and-rounding) working sites | Permitted at F32 working sites |
| [FLOAT64](../../glossary.md#term-float64) | Forbidden | Forbidden |

This permission does not attach to an [operation](../../glossary.md#operation) output merely because it is floating. Neither
applies at stored comparison, [extrema](../../glossary.md#saturation-pole-and-extremum)/winner selection, raw movement, cast, selection, index,
[mask](../../glossary.md#term-boolean-mask), state, or exact [unary](../../glossary.md#term-unary) ABS/NEG/SIGN/FLOOR/CEIL/RELU sites. Composite formulas and
[Compiler](../../glossary.md#term-compiler)-generated [gradients](../../glossary.md#term-gradient) inherit only their named arithmetic sites' permissions. [NaN](../../glossary.md#term-nan) class,
[infinity](../../glossary.md#term-infinity), [overflow](../../glossary.md#term-overflow), domain, and exact [guard](../../glossary.md#term-guard) rules are not finite-[tolerance](../../glossary.md#term-tolerance) decisions.

#### Exact and discrete floor

Kind and attributes, input/output [arity](../../glossary.md#term-arity) and [descriptors](../../glossary.md#term-descriptor), [Shape](../../glossary.md#shape)/[layout](../../glossary.md#layout)/[axis](../../glossary.md#term-axis) mapping, contributor
membership, [masks](../../glossary.md#term-boolean-mask), indices, [graph](../../glossary.md#graph)-[RNG state](../../glossary.md#graph-rng-state--graphrngstate) transitions, traversal, casts, conversions, ordering,
stability, ties, empty-domain behavior, [identities](../../glossary.md#term-identity-meanings), divisors, and [publication](../../glossary.md#term-publication) remain exact. A
selected stored value is one original candidate representation. Predicates gain no [epsilon](../../glossary.md#epsilon), and
approximation cannot choose an index, mask, winner, ordering, or state transition. Floating
comparisons and [extrema](../../glossary.md#saturation-pole-and-extremum) observe stored numeric values, including represented [subnormals](../../glossary.md#term-subnormal); they
retain family [NaN](../../glossary.md#term-nan), opposite-[signed-zero](../../glossary.md#nan-infinity-and-signed-zero), tie, winner, and index rules. A selected result remains
an original candidate. For example, positive minimum subnormal compares greater than +0 and
remains the MAX winner; MIN selects +0. WHERE makes an exact [Boolean](../../glossary.md#data-type--datatype) branch choice, then applies
the declared cast if promotion requires one; same-type selection preserves the chosen bits.
Movement, copying,
[CONTIGUOUS](../../glossary.md#term-contiguous), classification, Boolean logic, casts, indexing, ordering, arg-extrema, RNG
state/masks, and max-[pooling](../../glossary.md#term-pooling) selection gain no arithmetic relaxation from adjacent work.

#### Floating arithmetic primitives

Each named ADD, SUB, MUL, and DIV site uses the family's declared working [dtype](../../glossary.md#data-type--datatype) and [rounding](../../glossary.md#term-rounding).
A corresponding multiply/add may fuse only without a [Model](../../glossary.md#term-model)-observable intermediate; no
cross-[node](../../glossary.md#node) [contraction](../../glossary.md#contraction-dimension) is authorized. Elementary POW, [LOG](../../glossary.md#term-log), LOG1P, [EXP](../../glossary.md#term-expx), EXPM1, [ERF](../../glossary.md#term-erf), SQRT, and
[TANH](../../glossary.md#term-tanh) sites retain their named mathematical function and domain/special-value rules. [Finite](../../glossary.md#term-finite)
ordinary-result accuracy is route-qualified in tests, not bounded by a public ordered-distance
ceiling. An ordinary finite result cannot become [NaN](../../glossary.md#term-nan) or [infinity](../../glossary.md#term-infinity) except through genuine
[overflow](../../glossary.md#term-overflow) or a domain path after a permitted [DAZ](../../glossary.md#term-daz) choice. Required infinity and zero signs remain
intact except at approved arithmetic DAZ/[FTZ](../../glossary.md#term-ftz) sites or under the family's existing final
exact-zero [publication](../../glossary.md#term-publication) rule. No finite [tolerance](../../glossary.md#term-tolerance) substitutes for class or sign checks.

#### [Aggregate floor](../../glossary.md#term-aggregate-floor)

After exact mapping, masking, [padding](../../glossary.md#padding), bounds, and selection determine one logical cell, every
declared contributor participates exactly once. Floating [evaluation](../../glossary.md#term-evaluation) may choose a rounded [binary](../../glossary.md#term-binary-descriptor-and-metadata)
tree, use the permitted arithmetic-site [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) rules, and fuse only a corresponding multiply/add
without an observable intermediate. [FLOAT32](../../glossary.md#term-float32) sites round to FLOAT32; [homogeneous](../../glossary.md#term-homogeneous) low sites work
and accumulate in FLOAT32 with one final low [narrowing](../../glossary.md#term-narrowing). Exact empty/point [identities](../../glossary.md#term-identity-meanings) and mandatory
divisors remain. No evaluation may drop, duplicate, invent, pretruncate, or replace a term.

This floor applies recursively to ordinary and masked reductions, [sum-to-Shape](../../glossary.md#sum-to-shape), scans,
overlap/scatter reductions, statistics and norms, [loss](../../glossary.md#loss) reductions, [matrix](../../glossary.md#term-matrix-meanings)/[convolution](../../glossary.md#term-channels-first-convolution-layers)/attention/
recurrent contractions, and average [pooling](../../glossary.md#term-pooling). [Extrema](../../glossary.md#saturation-pole-and-extremum) and winner selection use the exact/discrete
floor. The established final exact-zero [publication](../../glossary.md#term-publication) freedoms for qualifying SUM, [MEAN](../../glossary.md#term-mean),
SUM-to-Shape, and nonempty [MATMUL](../../glossary.md#matrix-multiplication--matmul) results remain part of their [Model](../../glossary.md#term-model) formulas; they do not grant
intermediate or neighboring-[operation](../../glossary.md#operation) sign freedom.

For floating scatter MUL, the base and every addressed update contribute exactly once, but a
rounded multiplication tree may differ from a once-rounded exact abstract product: an
[overflow](../../glossary.md#term-overflow)-first `maxFinite * 2 * +0` may produce [NaN](../../glossary.md#term-nan). A NaN factor or a zero/[infinity](../../glossary.md#term-infinity) pair still
requires NaN class. Integral fixed-width modular MUL, exact [target](../../glossary.md#term-target) mapping, and raw unaddressed
cells remain unchanged.

#### Composite inheritance

[Normalization](../../glossary.md#term-normalization), [activation](../../glossary.md#term-activation), [loss](../../glossary.md#loss), attention, [pooling](../../glossary.md#term-pooling), [convolution](../../glossary.md#term-channels-first-convolution-layers), random/[dropout](../../glossary.md#dropout--inverted-dropout), recurrent,
visible convenience composition, and [Compiler](../../glossary.md#term-compiler)-generated [gradient](../../glossary.md#term-gradient) formulas receive no separate
[error envelope](../../glossary.md#term-error-envelope). Their exact guards [run](../../glossary.md#term-run) first, and their authoritative [Model](../../glossary.md#term-model) formulas recurse
through the exact/discrete, primitive, and aggregate sites. Saved outputs, statistics, [masks](../../glossary.md#term-boolean-mask), and
indices are exact stored values. [Opaque](../../glossary.md#term-opaque) routes require the same [occurrence](../../glossary.md#term-occurrence)-scoped exact, special,
and [finite](../../glossary.md#term-finite) test [qualification](../../glossary.md#term-qualification) as explicit routes; no [operation](../../glossary.md#operation) name or [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) grants support.

[General scan](../../glossary.md#general-differentiable-scan) composition inherits these same family arithmetic and exact saved-value rules; it
introduces no numerical selector, alternative [derivative](../../glossary.md#term-derivative) numerics or blanket [dtype](../../glossary.md#data-type--datatype) capability.
The [scan owner](recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([scan glossary](../../glossary.md#term-cumulative-scan); [owner glossary](../../glossary.md#term-owner))</span> defines its
bounded algebra and qualification sequence, not a replacement family formula.

The recursive sites for every current first-class composite formula are closed as follows. An
exact typed attribute or stored input is an exact [leaf](../../glossary.md#term-leaf). A formula-named real [constant](../../glossary.md#term-constant) is rounded
once to nearest-even [`FLOAT32`](../../glossary.md#term-float32) before its first use and is not an approximation site.

| Formula family | Complete recursive sites |
|---|---|
| [Unary](../../glossary.md#term-unary) composites | `RSQRT` is `SQRT` then typed `+1 / root`. [`SIGMOID`](../../glossary.md#term-sigmoid) first applies its exact sign [guard](../../glossary.md#term-guard); the nonnegative branch has negation, [`EXP`](../../glossary.md#term-expx), typed-one addition, and division, while the negative branch has `EXP`, typed-one addition, and division. Exact [GELU](../../glossary.md#gaussian-error-linear-unit--gelu) has typed `0.5`, `1`, and `2`, `SQRT(2)`, `x / sqrt(2)`, [`ERF`](../../glossary.md#term-erf), addition, and two multiplications. [Tanh](../../glossary.md#term-tanh) GELU has typed `0.5`, `1`, `2`, `pi`, and `0.044715`, `x*x`, `x^2*x`, `2/pi`, `SQRT`, the inner multiply/adds, `TANH`, and the outer add/multiplies. [SiLU](../../glossary.md#sigmoid-linear-unit--silu) uses the exact sign guard and the corresponding sigmoid branch plus the final multiplication; equivalently its nonnegative branch is `x/(1+EXP(-x))` and its negative branch is `x*EXP(x)/(1+EXP(x))`. None of `RSQRT`, `SIGMOID`, GELU, tanh GELU, or SiLU is one elementary site. |
| Aggregate/statistical composites | [`MEAN`](../../glossary.md#term-mean) is an all-terms-once sum then mandatory count division. [`VARIANCE`](../../glossary.md#term-variance) is mean, one subtraction and `x*x` per contributor, aggregate sum, then `N-correction` division; [standard deviation](../../glossary.md#term-standard-deviation) adds `SQRT`. [L1 norm](../../glossary.md#norms-elementary-functions-and-probability-distributions) applies `ABS` then sum. [L2 norm](../../glossary.md#norms-elementary-functions-and-probability-distributions) applies `x*x`, sum, then `SQRT`. `LOG_SUM_EXP` applies `EXP` once per contributor, aggregate sum, then [`LOG`](../../glossary.md#term-log); a stable replacement must preserve guards, special values, and family formula under route qualification. Cumulative and overlap/scatter arithmetic use their declared per-contributor operation and aggregate tree. |
| [Softmax](../../glossary.md#softmax--log-softmax) and normalization | On an all-finite nonempty [slice](../../glossary.md#slice), softmax and [log-softmax](../../glossary.md#softmax--log-softmax) use the maximum-shifted sites defined below; unshifted `EXP` is not a required intermediate. Layer [norm](../../glossary.md#term-norm) uses sum/count mean, per-value subtraction and `x*x`, sum/count variance, [epsilon](../../glossary.md#epsilon) addition, `SQRT`, division, and optional scale multiply/[bias](../../glossary.md#term-bias) add. RMS norm uses `x*x`, sum/count, epsilon addition, `SQRT`, division, and optional scale multiply. BatchNorm [inference](../../glossary.md#term-inference) uses subtraction, variance/epsilon addition, `SQRT`, division, scale multiply, and bias add. Training additionally uses sum/count mean, centered `x*x` sums with the exact `N` and `N-1` divisors, epsilon addition, `SQRT`, typed-one division, and the fixed multiply/add running-[statistic](../../glossary.md#term-statistic) transitions. |
| Losses | [MSE](../../glossary.md#mean-squared-error--mse) uses one subtraction and `delta*delta` per position, then its exact reduction and optional divisor. Categorical cross-entropy uses exact max selection, score-minus-max subtraction, `EXP`, class sum, `LOG`, max addition, [logit](../../glossary.md#logit) subtraction, exact zero-[target](../../glossary.md#term-target)/ignore guards, target multiplication where declared, loss sum, negation, and optional reduction divisor. |
| [Linear](../../glossary.md#term-linear-meanings), convolution, pooling, and attention | [MATMUL](../../glossary.md#matrix-multiplication--matmul) and [Conv2d](../../glossary.md#convolution--conv2d)/[Conv3d](../../glossary.md#conv3d--ncdhw) use one multiply per exact mapped pair and an all-terms-once [contraction](../../glossary.md#contraction-dimension); Conv2d/Conv3d then apply their optional bias addition, while visible linear/[Conv1d composition](../../glossary.md#conv1d-composition) adds only its documented bias or [rank](../../glossary.md#term-rank) edits. Average Pool2d/[Pool3d](../../glossary.md#pool3d--ncdhw-pooling) sums every declared [kernel](../../glossary.md#kernel) position then divides by the exact fixed divisor; max pooling is exact winner selection. Attention scores use query/key multiply-contractions and scale multiplication; an absent scale uses exact positive [embedding](../../glossary.md#embedding) [extent](../../glossary.md#term-extent) conversion, `SQRT`, and typed-one division. Exact mask/causal guards precede attention's own literal eligible-score `EXP`/sum/division sites, and output rows use value-[weight](../../glossary.md#term-weight) multiply-contractions. The first-class stable finite-softmax rule does not replace those guarded attention sites. |
| Random and recurrent | A kept dropout value uses typed-one-minus-probability, typed-one division, then input multiplication; the draw, mask, dropped [positive zero](../../glossary.md#nan-infinity-and-signed-zero), and state transition are exact. [RNN](../../glossary.md#term-recurrent-neural-network-rnn) uses two multiply-contractions, optional bias addition, addition, and `TANH`. [GRU](../../glossary.md#term-gated-recurrent-unit-gru) uses the documented packed contractions/bias adds, gate additions and sigmoid branches, reset multiplication, candidate `TANH`, `h-n`, update multiplication, and final addition. [LSTM](../../glossary.md#term-long-short-term-memory-lstm) uses its packed contractions/bias adds, gate additions and sigmoid/tanh branches, cell multiplications/addition, final `TANH`, and hidden multiplication. Traversal, valid-length guards, skipped work, outputs, and state [publication](../../glossary.md#term-publication) remain exact. |

For each all-finite nonempty softmax slice, select the greatest represented input value `m` by
exact stored-value comparison, with no [DAZ](../../glossary.md#term-daz) or arithmetic approximation at that selection. In the
family's working dtype, subtract `m` from each represented `x_i`, and include each
`EXP(x_i-m)` contributor exactly once in the sum using the [aggregate floor](../../glossary.md#term-aggregate-floor). An implementation
may recompute a shifted exponential when producing its output, without adding a sum contributor. The
shift, exponential result, and aggregate accumulation are [FLOAT64](../../glossary.md#term-float64) for FLOAT64 inputs and FLOAT32
for FLOAT32, [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16), FLOAT8_E4M3, and FLOAT8_E5M2 inputs. The resulting sum is positive and finite when all
working shifts are finite. SOFTMAX divides each shifted exponential by that sum; LOG_SOFTMAX
subtracts `LOG(sum)` from each working shift. Each output is then published in its input dtype,
with a single final [ties-to-even](../../glossary.md#term-ties-to-even) [narrowing](../../glossary.md#term-narrowing) for a low output. LOG_SOFTMAX does not take the
logarithm of a rounded SOFTMAX output: an underflowed probability can therefore have a finite
log-probability. Finite represented inputs can still produce a negative-[infinity](../../glossary.md#term-infinity) working shift
when subtraction genuinely overflows the working dtype; its exponential is positive zero and its
log-softmax result is [negative infinity](../../glossary.md#nan-infinity-and-signed-zero). No [backend](../../glossary.md#backend) may instead evaluate unshifted `EXP` and use
its incidental [overflow](../../glossary.md#term-overflow) to turn an otherwise finite-slice probability into [NaN](../../glossary.md#term-nan).

The shifted-exponential sum may use compensated accumulation, including Kahan's rounded
sum/[residual](../../glossary.md#scan-residual-port-and-history-demand) update, as an explicit softmax-family exception to the ordinary rounded-[binary](../../glossary.md#term-binary-descriptor-and-metadata)-tree
[evaluation](../../glossary.md#term-evaluation) [shape](../../glossary.md#shape). The compensation is [rounding](../../glossary.md#term-rounding)-error state, not an additional slice contributor;
each shifted exponential still enters the aggregate exactly once. Compensation arithmetic uses
the same working dtype and named arithmetic-site permissions. It may not move an exact guard,
alter contributor membership, or cross a public or saved-value boundary.

Special-value and empty-slice rules are separate from that all-finite construction. An empty slice
publishes no positions. Any NaN in a nonempty slice makes every result NaN. With one or more
positive [infinities](../../glossary.md#nan-infinity-and-signed-zero) and no NaN, each positive-infinity position is NaN; every other position is
positive zero for SOFTMAX and negative infinity for LOG_SOFTMAX. A slice with negative infinity
and no [positive infinity](../../glossary.md#nan-infinity-and-signed-zero) or NaN retains the pre-existing literal formula: working `EXP(x_i)` for
every position, an [all-contributors-once](../../glossary.md#rounding-sites-fma-and-aggregate-floor) rounded sum, then division for SOFTMAX or `LOG(sum)`
and input subtraction for LOG_SOFTMAX. Thus finite members are not automatically stabilized in
a mixed-infinity slice. For example, `[-infinity, 1000]` at FLOAT32 has SOFTMAX `[+0, NaN]` and
LOG_SOFTMAX `[-infinity, -infinity]` because the finite member's unshifted EXP overflows; this
preserves the prior result classes. An all-negative-infinity nonempty slice produces NaN at
every position. [NaN payload](../../glossary.md#nan-infinity-and-signed-zero) and sign are unspecified. These are Model results, not a promise that
every backend admits every represented value. [CPU](../../glossary.md#term-cpu)'s narrower [runtime](../../glossary.md#term-runtime) input domain belongs to
[its finite-softmax admission owner](backend-execution.md#cpu0077-finite-softmax-admission-parity)<span data-glossary-navigation="true"> ([admission glossary](../../glossary.md#term-admission))</span>;
it does not redefine this formula or let an incidental unshifted overflow change the result.

Guarded attention's no-eligible, all-negative-infinity, and positive-infinity-tie cases retain
their family-specific results rather than inheriting ordinary softmax's result by approximation.
Categorical cross-entropy already has its own guarded max-shifted formula; this decision does not
change its target/ignore guards, [accumulator](../../glossary.md#accumulator-type) sites, or allowed results. LOG_SUM_EXP and other
composites retain their own formulas and cannot inherit this shift by analogy.

Visible convenience compositions recurse through the actual public [Tensor](../../glossary.md#tensor) operations they create.
Compiler-generated gradients likewise recurse through every captured arithmetic, elementary,
aggregate, comparison, `WHERE`, cast, [layout](../../glossary.md#layout), index, and saved-value site. Another internal
algorithm must retain exact guards, contributor domain, observable boundaries, and special-value
rules and pass route-specific finite qualification. A passing sample or [tolerance](../../glossary.md#term-tolerance) cannot license
an algebraic rewrite, omitted contributor, or new provider capability.

#### [Model](../../glossary.md#term-model) low-[precision](../../glossary.md#term-precision) contract

The Model-level [FLOAT16](../../glossary.md#term-float16), [BFLOAT16](../../glossary.md#term-bfloat16) and [OFP8](../../glossary.md#term-ofp8) types own value, promotion, cast and factory
semantics independently of native availability. [CPU](../../glossary.md#term-cpu) low-precision parity is authorized by the
user; [Metal](../../glossary.md#term-metal) retains its independently qualified BFLOAT16/FLOAT16 domain and rejects [FP8](../../glossary.md#term-fp8).
This subsection defines allowed results, not an [inference](../../glossary.md#term-inference) of support from another [dtype](../../glossary.md#data-type--datatype) or a
permission to replace [Compiler](../../glossary.md#term-compiler)-owned [derivatives](../../glossary.md#term-derivative) with vendor [AD](../../glossary.md#autograd).

FLOAT16, BFLOAT16, FLOAT8_E4M3 and FLOAT8_E5M2 are distinct storage types. An [operation](../../glossary.md#operation)
with unlike low floating types promotes to [FLOAT32](../../glossary.md#term-float32); any FLOAT32 operand promotes low types
to FLOAT32, and a [FLOAT64](../../glossary.md#term-float64) operand promotes to FLOAT64. [Homogeneous](../../glossary.md#term-homogeneous) low occurrences retain
their declared low output type. CPU follows this promotion; other [backends](../../glossary.md#backend) qualify independently.

Kind, attributes, [arity](../../glossary.md#term-arity), [descriptors](../../glossary.md#term-descriptor), [Shape](../../glossary.md#shape)/[layout](../../glossary.md#layout) mapping, raw storage and movement, contributor
domain, guards, predicates, indices, [masks](../../glossary.md#term-boolean-mask), state transitions, selection, ordering, casts, public
outputs, and saved values remain exact. Casts retain their declared conversion. Each declared low
arithmetic output performs one final [round-to-nearest, ties-to-even](../../glossary.md#floating-point-precision-and-rounding) [narrowing](../../glossary.md#term-narrowing); no implementation
may pre-narrow an input, contributor, [accumulator](../../glossary.md#accumulator-type), public value, or saved value.

Every admitted family must have a qualified implementation whose low arithmetic uses `FLOAT32`
working and accumulator values and the arithmetic-site [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) table above. Its Model contract
retains family-specific arithmetic exact-zero sign choices. Arithmetic [NaN](../../glossary.md#term-nan) must retain NaN class and
the Model formula's domain behavior, but its [payload](../../glossary.md#artifact-envelope-and-atomic-publication) and sign are not accuracy requirements. These
freedoms never affect a predicate, index, mask, selection, state transition, raw copy, or cast.

Within one operation, arithmetic may reassociate and use corresponding `FLOAT32` [FMA](../../glossary.md#term-fused-multiply-add-fma). [Fusion](../../glossary.md#term-fusion) may
cross only a semantically unobservable intermediate with exactly one consumer. It may never cross
a public output, saved value, [fan-out](../../glossary.md#term-fan-out), predicate, index, mask, selection, or state boundary.

[Finite](../../glossary.md#term-finite) accuracy [qualification](../../glossary.md#term-qualification) belongs only in tests with explicit family/dtype/backend/route/size
oracles, [metrics](../../glossary.md#machine-learning-and-hyperparameters), thresholds, and [cancellation](../../glossary.md#term-cancellation) boundaries. NaN/[infinity](../../glossary.md#term-infinity), [signed zero](../../glossary.md#term-signed-zero), [subnormal](../../glossary.md#term-subnormal)/
[underflow](../../glossary.md#term-underflow), domain and exact cases have separate [conformance](../../glossary.md#term-conformance) checks. Model owns no public forward-
[error envelope](../../glossary.md#term-error-envelope); tests do not become production policy.

[Activation](../../glossary.md#term-activation) is [occurrence](../../glossary.md#term-occurrence)- and backend-specific. Project actual current [ACCELERATOR](../../glossary.md#term-accelerator) true and false
CPU/Metal provider answers to profile-free occurrence keys; independently query low types and
retain negative predicates. No answer is inferred from semantic reach or another dtype. Provider
capability and any architecture [target](../../glossary.md#term-target) mapping remain separate from route,
[runtime](../../glossary.md#term-runtime)/[device](../../glossary.md#term-device), and generated-backward facts.

Metal realizes every admitted BFLOAT16/FLOAT16 occurrence through its fixed custom program.
Environment records, program certificates, vendor-route probes, and [trace](../../glossary.md#trace) [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) verdicts
are not active contract state. The current route decision is governed by the backend-execution
contract and
[ADR 0025](../../design/decisions/0025-custom-only-metal-low-precision.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>. The [canonical](../../glossary.md#term-canonical-form) ledgers
and append-only allocations originated in
[ADR 0023](../../design/decisions/0023-low-precision-accelerator-parity.md).

No family permits unguarded reciprocal substitution,
algebraic [identities](../../glossary.md#term-identity-meanings) absent from the Model formula, cross-[node](../../glossary.md#node) [contraction](../../glossary.md#contraction-dimension), term [loss](../../glossary.md#loss), hidden
state changes, or [tolerance](../../glossary.md#term-tolerance)-based predicate/selection changes. In particular,
`maxFinite / maxFinite -> 0`, `+infinity / maxFinite -> NaN`, `RELU(NaN) -> +0`, and
`TANH(NaN) -> +1` remain invalid. Low-precision transformed algorithms and observable-single-use
fusion remain subject to the same exact/special boundaries and route qualification.

Model defines the single allowed-result contract. Config, [Planning](../../glossary.md#term-planning), Compiler, [Prepare](../../glossary.md#prepare), [Engine](../../glossary.md#term-engine),
Runtime, Trace, and backends do not transport a replacement numerical selector. Planning queries
actual backend occurrence capability without interpreting numerical semantics. Backend preparation
must [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) if its route is unqualified. Changed route, generated-[artifact](../../glossary.md#term-artifact), native, and tuning
identities are append-only and must reject stale serialized or cached input; the implementation
cutover assigns exact new versions.

#### [OFP8](../../glossary.md#term-ofp8) representation and [operation](../../glossary.md#operation) boundaries

`FLOAT8_E4M3` and `FLOAT8_E5M2` are [differentiable](../../glossary.md#derivative-and-chain-rule) one-byte OFP8 storage types appended
after [FLOAT16](../../glossary.md#term-float16). [E4M3](../../glossary.md#term-e4m3) has sign/[exponent](../../glossary.md#term-exponent)/fraction widths 1/4/3, [bias](../../glossary.md#term-bias) 7, [signed zero](../../glossary.md#term-signed-zero), gradual
[subnormals](../../glossary.md#term-subnormal), maximum [finite](../../glossary.md#term-finite) magnitude 448, and [NaN](../../glossary.md#term-nan) encodings 0x7f/0xff; it has no [infinity](../../glossary.md#term-infinity).
[E5M2](../../glossary.md#term-e5m2) has widths 1/5/2, bias 15, signed zero, gradual subnormals, maximum finite magnitude
57344, infinity encodings 0x7c/0xfc, and NaNs with exponent 31 and nonzero fraction.

Numerical conversion is non-[saturating](../../glossary.md#saturation-pole-and-extremum) [round-to-nearest](../../glossary.md#floating-point-precision-and-rounding) [ties-to-even](../../glossary.md#term-ties-to-even). E5M2 [overflow](../../glossary.md#term-overflow) produces
signed infinity; E4M3 out-of-range [rounding](../../glossary.md#term-rounding) and infinity input produce [canonical NaN](../../glossary.md#term-canonical-nan) 0x7f.
Narrowed NaNs canonicalize to 0x7f for E4M3 and 0x7e for E5M2. Direct [FLOAT64](../../glossary.md#term-float64) and integer
casts round to the destination without an intervening lossy [FLOAT32](../../glossary.md#term-float32) conversion. Exact same-type
casts, raw factories, movement, [scalar](../../glossary.md#term-scalar) [metadata](../../glossary.md#term-metadata) and [persistence](../../glossary.md#term-persistence) preserve every stored bit.
[Widening](../../glossary.md#term-widening) represents the original value and preserves represented NaN sign/[payload](../../glossary.md#artifact-envelope-and-atomic-publication) where the
destination permits it; no numeric conversion changes a raw import.

Each ordinary operation uses its existing formula with FLOAT32 working values and accumulators,
then one final [FP8](../../glossary.md#term-fp8) [narrowing](../../glossary.md#term-narrowing) at each public/saved/ordinary operation-result boundary. There is
no implicit scale, clipping, dynamic range calibration, stochastic rounding or [loss](../../glossary.md#loss) scaler.
An E4M3 result whose mathematical/working class is infinity consequently narrows to NaN,
including empty floating [extrema](../../glossary.md#saturation-pole-and-extremum) and function poles. Nonempty extrema still select an exact
original represented winner; an unrepresentable infinity [identity](../../glossary.md#term-identity-meanings) must not poison that selection.
[Masks](../../glossary.md#term-boolean-mask), indices, comparison, [RNG state](../../glossary.md#graph-rng-state--graphrngstate) and raw paths do not acquire arithmetic latitude.

Existing [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd), [Scan](../../glossary.md#term-cumulative-scan) and fixed recurrent [owners](../../glossary.md#term-owner) are reused with their original rounding
boundaries. FP8 coefficients for [precision](../../glossary.md#term-precision)-sensitive [normalization](../../glossary.md#term-normalization)/attention metadata may use
FLOAT32, just as FLOAT16 metadata does. Training uses FLOAT32 masters for all low [parameter](../../glossary.md#term-parameter)
types and rejects nonfinite represented candidates transactionally. Its [objective](../../glossary.md#term-objective) remains an
explicit scalar FLOAT32/FLOAT64 value; conversion to that objective is visible in the [graph](../../glossary.md#graph).
Generic source/value persistence appends new [dtype](../../glossary.md#data-type--datatype) IDs without changing any old dtype ID or
old accepted bytes; new FP8 wire domains are separately versioned. FP8 does not authorize a
new [backend](../../glossary.md#backend), graph compiler, execution [owner](../../glossary.md#term-owner) or [Metal](../../glossary.md#term-metal) native [wire value](../../glossary.md#term-wire-value).


### `modules/config`

`modules/config` owns declarative configuration only.

Allowed:

- `CompileConfig`
- [`CompileMode`](../../glossary.md#compile-mode--compilemode)
- [`BackendIntent`](../../glossary.md#backend-intent--backendintent)
- [`GraphOptimizationConfig`](../../glossary.md#graph-optimization-configuration--graphoptimizationconfig)
- [`PartitionScoringConfig`](../../glossary.md#partition-scoring-configuration--partitionscoringconfig)
- `PartitionScoringPolicy`
- `PrepareConfig`
- `CpuPrepareConfig`
- `AcceleratorPrepareConfig`
- `RunOptions`
- `PublicationPolicy`
- [immutable](../../glossary.md#term-immutable) declarative [planning](../../glossary.md#term-planning)-cost inputs, after their planning consumer is stable
- immutable declarative [model](../../glossary.md#term-model)-[autotuning](../../glossary.md#autotuning--model-autotuning) inputs, after their owning contracts are stable

Forbidden:

- live services
- [concrete backend](../../glossary.md#concrete-backend) classes
- [kernel](../../glossary.md#kernel) class references
- [runtime](../../glossary.md#term-runtime) state
- executable units
- [backend](../../glossary.md#backend)-specific implementation logic
- benchmark runners, model-autotuning search, [cache](../../glossary.md#term-cache) mutation, or measurement algorithms
- live platform discovery or [mutable](../../glossary.md#term-mutable) measurement evidence

Backend-specific interpretation of config belongs to backend [prepare](../../glossary.md#prepare).

### `modules/planning`

`modules/planning` owns [backend](../../glossary.md#backend)-neutral [compile](../../glossary.md#compile)-time [planning](../../glossary.md#term-planning).

Allowed:

- [backend intent](../../glossary.md#backend-intent--backendintent) propagation
- capability query contracts
- capability [matrix](../../glossary.md#term-review-matrix) construction
- backend-neutral [partition](../../glossary.md#partition) scoring
- [node](../../glossary.md#node) [ownership](../../glossary.md#term-ownership-meanings) decisions
- segment ownership decisions
- maximal same-[owner](../../glossary.md#term-owner) partitioning
- logical [materialization](../../glossary.md#materialization) requirements
- logical memory requirements
- backend-neutral planning cost estimates and profiles

Forbidden:

- [fusion](../../glossary.md#term-fusion) implementation
- [specialization](../../glossary.md#term-specialization)
- concrete [kernel](../../glossary.md#kernel) selection
- [OpenBLAS](../../glossary.md#term-openblas) route selection
- [Vector API](../../glossary.md#term-vector-api) route selection
- [MPSGraph](../../glossary.md#term-mpsgraph) route selection
- [CUDA](../../glossary.md#term-cuda) kernel selection
- physical memory allocation
- [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency)
- prepared schedules
- prepared executables
- backend-specific [DAG](../../glossary.md#term-directed-acyclic-graph-dag) construction
- backend-specific [lowering](../../glossary.md#lowering)
- runtime execution units
- concrete kernel/runtime scoring
- [backend route](../../glossary.md#backend-route) names or route-selection [parameters](../../glossary.md#term-parameter)
- [vector species](../../glossary.md#term-vector-species), [lane](../../glossary.md#term-lane) counts, unroll factors, thread counts, chunk sizes, or tile sizes

Planning answers:

```text
Where should this node or segment run?
```

Planning must not answer:

```text
Which concrete kernel, executable, BLAS route, MPSGraph route, or CUDA implementation should run it?
```

For a [general scan](../../glossary.md#general-differentiable-scan), Planning selects one owner for the complete body, never mixed-owner body
partitions. Logical memory planning carries ordinary-use [residual](../../glossary.md#scan-residual-port-and-history-demand) demand, not source [Tensors](../../glossary.md#tensor) or
an [opaque](../../glossary.md#term-opaque) saved workspace. Complete [scan](../../glossary.md#term-cumulative-scan) semantics remain with the
[scan owner](recurrent-scan.md#canonical-residual-ports-and-demand).

Compiler0013 emits length-mode queries with the entire header, ordered row roles and body/demand
without treating the extra lengths actual as a body formal. Existing transport and [Compiler](../../glossary.md#term-compiler)
[inference](../../glossary.md#term-inference)/[AD](../../glossary.md#autograd) alone grant no execution capability; providers reject until their complete binder/body is qualified under
the [scan owner's gates](recurrent-scan.md#runtime-preflight-and-implementation-gates).

## Profile-free capability queries

<a id="numerical-profile-transport"></a>

Profile transport is removed. [`OperationCapabilityQuery`](../../glossary.md#operation-capability-query--operationcapabilityquery)
retains complete [operation](../../glossary.md#operation), attribute, ordered [descriptor](../../glossary.md#term-descriptor), and [gradient](../../glossary.md#term-gradient) facts, but no numerical
profile. Its deterministic [boolean](../../glossary.md#data-type--datatype) provider answer cannot be inferred from a family formula,
another [dtype](../../glossary.md#data-type--datatype), [device](../../glossary.md#term-device) [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters), or the removed selector.

The accepted general-[scan](../../glossary.md#term-cumulative-scan) query extends OperationCapabilityQuery with
`Optional<CompiledScanBody>` while retaining the three-[argument](../../glossary.md#term-argument) empty-body construction. [Compiler](../../glossary.md#term-compiler)
supplies the validated complete scoped body, role/port descriptors and demanded outer-port indices;
queries remain [Tensor](../../glossary.md#tensor)-free. Propagate and revalidate demand after rewrites through [Prepare](../../glossary.md#prepare).
The deterministic provider boolean must qualify the header, every body [occurrence](../../glossary.md#term-occurrence)/descriptor,
[residual](../../glossary.md#scan-residual-port-and-history-demand) and [binding](../../glossary.md#term-normative) requirements, and generated [topology](../../glossary.md#term-topology) together with the loop binder. Supported
child queries are necessary but insufficient. Missing/malformed bodies or unsupported binders
reject; [Planning](../../glossary.md#term-planning) forwards semantic facts without interpreting [backend](../../glossary.md#backend) routes or implementing [AD](../../glossary.md#autograd).
Planning0008 implements the optional-body transport and provider contract tests. Compiler0011
implements body-bearing query emission, scoped [capture](../../glossary.md#term-capture-meanings)/[inference](../../glossary.md#term-inference) and first-order AD/demand propagation.
Prepare0010 implements scoped projection and static [binding](../../glossary.md#term-binding)/demand [validation](../../glossary.md#term-validation). CPU0030 adds a
bounded [FLOAT32](../../glossary.md#term-float32) same-[shape](../../glossary.md#shape) [pointwise](../../glossary.md#term-pointwise) provider predicate; both the forward body and its actual
Compiler-generated reverse must independently qualify. CPU0031 extends that predicate to
[homogeneous](../../glossary.md#term-homogeneous) FLOAT32 broadcast, SUM/sumToShape and bounded shape/[layout](../../glossary.md#layout) closure. CPU0032 adds static
homogeneous FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul) under the same independent whole-body/binder proof for actual
forward and reverse occurrences. CPU0033 extends the complete bounded declared algebra and actual
first-order reverse closure to homogeneous [FLOAT64](../../glossary.md#term-float64) under that same independent [qualification](../../glossary.md#term-qualification);
[Model](../../glossary.md#term-model) source [admission](../../glossary.md#term-admission) and other dtype/layout capability remain separate. See the
[sole scan protocol](recurrent-scan.md#general-differentiable-scan).

### [General Scan](../../glossary.md#general-differentiable-scan) activity transport

Implemented coordinated Class C transport, accepted with Compiler0017/CPU0081 after independent
authority/interface and exact-source implementation reviews and the affected integration gate.
[Planning](../../glossary.md#term-planning) owns only the [immutable](../../glossary.md#term-immutable) [carrier](../../glossary.md#term-storage-carrier)/query transport, not [derivative](../../glossary.md#term-derivative) formulas/proof,
physical storage or [runtime](../../glossary.md#term-runtime) control. The
[Scan owner](recurrent-scan.md#general-ordinary-unit-scan-cutover) owns its semantic meaning.
Exactly one new public Planning type in existing `planning.capability` is selected:

```java
public record ScanActivityPlan(
        CompiledGraphModel graph, List<Carry> carries, List<Integer> seedFormals,
        List<List<Integer>> scopes, Map<NodeId, Integer> nodeScopes,
        Map<NodeId, Merge> merges, List<Accumulator> accumulators) {
    public record Carry(int numericCarryPosition, boolean initialPresence, int nextScope) {}
    public record Merge(int leftScope, int rightScope) {}
    public record Accumulator(int numericSharedCarryPosition, int contributionScope) {}
}
```

[`graph`](../../glossary.md#graph) is the exact independently scoped compiled body. Root order is carries then seedFormals;
[scope](../../glossary.md#term-scope) lists are defensive immutable sorted unique root-index sets. [Carry](../../glossary.md#term-carry)/[seed](../../glossary.md#term-seed)/[accumulator](../../glossary.md#accumulator-type)
positions are declaration positions, not deduplicated actual [identities](../../glossary.md#term-identity-meanings). Every executable local
[node](../../glossary.md#node) has a scope; merge keys identify actual local ordinary ADDs. Intrinsic constructor checks
cover complete membership, root/scope ranges, unique positions, deterministic immutable maps and
valid local merge [shapes](../../glossary.md#shape). Containing-[occurrence](../../glossary.md#term-occurrence) consumers independently check header/formal/
[binding](../../glossary.md#term-binding) applicability, scopes/[dataflow](../../glossary.md#term-dataflow) and ordinary predicates. No [Tensor](../../glossary.md#tensor), [producer](../../glossary.md#term-producer), numerical
[cotangent](../../glossary.md#cotangent), derivative formula, address, physical slot or [mutable](../../glossary.md#term-mutable) runtime object is retained.
The carrier is strictly local: it contains no external presence/length references and does not
transport activity through ordinary Tensor publications. Root initialization follows the
[Scan](../../glossary.md#term-cumulative-scan) [owner](../../glossary.md#term-owner)'s materialized-boundary semantics; the selected record [signatures](../../glossary.md#term-signature) remain unchanged.

[OperationCapabilityQuery](../../glossary.md#operation-capability-query--operationcapabilityquery) becomes `(operation, inputs, outputs, Optional<CompiledScanBody> scanBody,
Optional<ScanActivityPlan> scanActivityPlan)`. A plan applies only to a body-bearing Scan and
names that exact body graph. Migrate every four-[argument](../../glossary.md#term-argument) body-bearing caller; the meaningful
three-argument ordinary empty-attachment constructor remains. Missing ordinary support or
malformed applicable control still fails closed independently. No empty optional [serialization](../../glossary.md#term-serialization)
field or [identity](../../glossary.md#term-identity-meanings) suffix is added to old recipes, and no new numerical profile is introduced.

