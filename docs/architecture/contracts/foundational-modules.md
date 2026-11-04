# Foundational module contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns the detailed responsibilities and prohibitions for Trace, Backend Contract,
Model, Config, and Planning. Model's scope includes the sole normative definition of profile-free
operation-family numerical semantics and allowed results. Planning scope here is the module
boundary and ownership question; the detailed partition-scoring algorithm boundary belongs to the
compiler/autograd contract. This file does not own recurrent-scan semantics, compiler behavior,
execution lifecycles, concrete backend implementation, or extensions.

## Module responsibilities

### `modules/trace`

`modules/trace` owns neutral typed diagnostic envelopes, correlation IDs and shared payload DTOs.
The open `TracePayload` marker also permits immutable producer-specific DTOs in their owning
module. Those DTOs contain detached diagnostic values only, not graph, Tensor, storage, native
handle, exception or producer-domain object references. Producer ownership does not introduce
any dependency from Trace back to Engine or a backend.

Allowed:

- trace event envelopes
- typed compile payloads
- typed prepare payloads
- typed run payloads
- typed backend payloads
- trace-local IDs
- typed trace attributes

Forbidden:

- importing model
- importing planning
- importing compiler
- importing runtime
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

`modules/backend-contract` owns minimal backend identities and declarative requirements.

Allowed:

- `BackendId`
- `BackendDeviceId`
- `BackendAvailabilitySnapshot`
- `BackendRequirement`
- `DeviceClass`

Forbidden:

- kernel registry
- operation support logic
- backend prepare services
- executable units
- runtime storage
- physical buffers
- cost model implementation

Compile-time plans must hold backend identity, not live backend services.

Use `BackendId`, not concrete backend objects, in compile-time ownership and partitioning data.

### `modules/model`

`modules/model` owns the public tensor model, operation semantics, shape/data type/layout model, host storage abstraction, and immutable graph model.

Allowed:

- `Tensor`
- `TensorId`
- `TensorFactory`
- `DataType`
- `Shape`
- `LayoutDescriptor`
- `HostTensorStorage`
- `MemorySegmentStorage`
- `Operation`
- `OperationAttrs`
- `CompiledGraphModel`
- `CompiledNode`
- `GraphValue`
- `NodeId`
- `ValueId`
- `GraphPhase`
- `ForwardPublicationBinding`
- `TensorDescriptor`
- operation-family formulas and allowed-result sets

Forbidden:

- backend support
- `supportedBackends()`
- device residency
- runtime workspaces
- physical device buffers
- kernel selection
- backend-specific storage
- prepared execution
- runtime state

`Tensor` may retain its existing mutable borrowed host-storage association, but it must not own
gradient state or runtime device residency. Its identity, descriptor, and expression provenance
remain immutable.

Every derived expression producer owns one operation occurrence, its ordered input tensors, its
ordered output descriptors, and the canonical exact `Tensor` wrapper for every output slot.
`TensorFactory` constructs those wrappers atomically with their indexed provenance. A producer
must return the retained wrapper for a slot rather than reconstructing an equal wrapper. This
pre-capture model relationship is not graph IR, graph membership, or graph-local identity.

The accepted general Scan continuation adds a context-qualified overload on the existing Model
owner: `ScanBody.describeOrdinary(Operation, List<TensorDescriptor>, int, Optional<DynamicDimension>)`.
It reuses ordinary inference and the same factored source/deferred symbolic proof with the exact
declared symbol. The three-argument ordinary API keeps its deferred semantics. This is descriptor
proof, not Compiler inference, ROW provenance, backend admission or a new public type; the
[Scan owner](recurrent-scan.md#general-ordinary-unit-scan-cutover) defines its semantic obligations.

The reviewed logical-layout prerequisite adds two metadata operations on the same existing owner:
`ScanBody.closeOrdinaryLayouts(Operation, List<TensorDescriptor>, List<TensorDescriptor>)` and
`ScanBody.hasCanonicalContiguousResults(Operation)`. The former completes eligible unresolved
static computed results through the moved ordinary closure and existing family layout helpers;
the latter preserves the exact generic materialization fact also consumed by Compiler symbolic
binding. Specialized convolution completion does not widen that classifier. Neither operation
changes ordinary construction, performs independent type/Shape inference or grants PURE/Scan/
backend admission. Preserve resolved layouts, exact affine views and unresolved unproved cases.
No new production type, Tensor/ID allocation or cross-module dependency edge is selected.

The user-authorized backend coverage completion extends that existing completion decision for
validated fully static signed-step or empty SLICE extraction: when an exact nonnegative-stride
view is unproved, select canonical non-view materialization. Raw family/source/wire inference
remains unchanged, and no negative physical stride, sampled symbolic extent, new operation owner
or coordinate normalization is introduced. Flat Compiler and containing-body physical closure
consume the same decision; ordinary and Scan execution still qualify independently.

The reviewed source-publication prerequisite renames the existing Prepare contribution owner
to `ProducerlessPublishedValueResource`, retaining five components: stable value, logical
requirement, backend owner, existing Runtime `ByteSize size`, and byte alignment. The migrated
default assembler hook takes value, requirement and `Optional<ScalarValue>`: present only for
an authenticated constant, empty only for an authenticated caller. Prepare owns complete source/
publication roles and ordering; CPU owns physical geometry and fixed-unit compatibility.
No Model mathematics, Runtime type/field, dependency edge or parallel contribution owner is added.

The accepted bounded scan exception adds source-body containment on TensorProducer, distinct from
the single compiled-body attachment on CompiledNode. The complete inventory, canonical residual
port construction, formal/role binding and scoping semantics belong solely to the
[scan semantic owner](recurrent-scan.md#general-differentiable-scan). Model0035 implements and
validates these Java source/IR attachments. Planning0008 implements query transport, and
Compiler0011 implements scoped capture/inference, first-order AD and final query/demand emission.
Prepare0010 implements scoped projection and static binding/demand validation. CPU0030 implements
the bounded FLOAT32 same-shape pointwise forward/actual-reverse execution slice. CPU0031 adds
qualified FLOAT32 broadcast, SUM/sumToShape and bounded shape/layout closure, with actual reverse
bodies independently qualified. CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL and
independently qualified actual reverse bodies. CPU0033 implements homogeneous FLOAT64 across
the complete bounded declared algebra and actual first-order reverse closure; source admission
still grants no generic backend capability. Preserve
empty-attachment constructor compatibility. OperationAttrs remains Tensor/graph-free scalar
semantic metadata; no body registry or source closure belongs in attrs.

Device storage belongs to runtime/backend layers, not model.

Model0038 implements the length-aware header, overload, bounded row-role source subset and outer/local
count distinction defined only by the [scan owner](recurrent-scan.md#header-bindings-and-source-admission).
Model construction/structural attachments do not grant Compiler, cold, AD or execution capability.
Compiler0013 independently implements scoped length inference and bounded first-order reverse
construction/query emission; cold, Prepare, CPU and encoding remain closed.
The old all-valid PURE/type admission remains unchanged; no new production type or dependency is selected.

`ScanBody.describeOrdinary` exposes descriptor-only checking through the same existing ordinary
family guards used by Tensor construction. It returns all canonical output descriptors without
creating Tensors, producers, storage or identifiers. This Model check is not independent Compiler
inference and grants no row-coordinate, AD, execution or persistence capability.

The separately accepted [Scan coordinate seam](recurrent-scan.md#bounded-row-role-analysis)
recomputes per-canonical-output row/dependency/terminal semantics from independently checked
descriptors. It is not descriptor inference or an admission certificate. Model0048 is implemented
and source-qualified; source validation and raw Compiler use the same semantic owner while
retaining independent ordinary descriptor guards and downstream support boundaries.

### Profile-free numerical semantics

<a id="numerical-profiles"></a>

The legacy anchor above preserves links from historical profile-era records. Model
alone owns each operation family's formula, contributor domain, guards, precision/rounding sites,
special-value behavior, and allowed results. There is no numerical selector, strict/accelerator
result-set union, public accuracy envelope, production tolerance registry, or runtime numerical
certificate. Backend support remains an occurrence-specific provider answer, not a consequence of
removing a selector.

#### Family formulas and unary sites

Every current unary kind retains its formula for each accepted floating type, including both
OFP8 types below. ABS and NEG perform exact represented sign clearing/inversion. SIGN, FLOOR, CEIL,
and RELU retain their exact represented-value family rules. Those six sites never use arithmetic
DAZ/FTZ. RECIPROCAL is one division at the declared working dtype, not an implicit reciprocal
estimate. LOG, LOG1P, EXP, EXPM1, ERF, SQRT, and TANH retain their mathematical function, domain,
range, class, and sign rules. LOG1P refers to the real `ln(1+x)` and EXPM1 to real `exp(x)-1`;
their reference additions/subtractions are not separately rounded primitive sites. ERF retains
`erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t*t) dt` as its mathematical reference.
RSQRT and the activation composites retain the named-site dependency formulas below; none is
one blanket approximate output. A named real constant is rounded before first use as the family
specifies. Finite accuracy is qualified with test-only backend/route/family/dtype/size tolerances
and an explicit oracle, metric, threshold, and cancellation boundary, not Model-owned
ordered-distance or forward-error envelopes.

The following top-level domain and special-value results override the ordinary-finite constructions.
Every listed NaN result must be NaN, but payload, quieting, and sign are unspecified.

| Kinds | Signed zero, infinity, domain, and NaN rules |
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
| `SIGN` | Signed zero is preserved; infinities become same-signed unit values; NaN remains NaN. |
| `RELU` | Negative finite inputs, `-infinity`, and either zero become `+0`; `+infinity` remains; NaN remains NaN. |
| `SIGMOID` | Either zero becomes `0.5`; `-infinity` becomes `+0`; `+infinity` becomes `+1`; NaN remains NaN. |
| `TANH` | Signed zero is preserved; infinities become same-signed unit values; NaN remains NaN. |
| `GELU`, `GELU_TANH_APPROXIMATION`, `SILU` | `-infinity` becomes `-0` by continuous extension; signed zero is preserved; `+infinity` remains; NaN remains NaN. |

#### Arithmetic-site DAZ/FTZ

Denormals-are-zero (DAZ) may read a subnormal named floating arithmetic primitive input as
same-signed zero without changing its stored bits. Flush-to-zero (FTZ) may map a finite subnormal
named primitive result to zero subject to the existing family-specific signed-zero rule; it
grants no universal choice of zero sign.

| Primitive-site dtype | Input DAZ | Result FTZ |
|---|---|---|
| FLOAT32 | Permitted | Permitted |
| BFLOAT16 | Permitted, including multiplication | Permitted, including multiplication |
| FLOAT16 | Permitted, including multiplication | Permitted, including multiplication |
| FLOAT8_E4M3 / FLOAT8_E5M2 | Permitted at F32 working sites | Permitted at F32 working sites |
| FLOAT64 | Forbidden | Forbidden |

This permission does not attach to an operation output merely because it is floating. Neither
applies at stored comparison, extrema/winner selection, raw movement, cast, selection, index,
mask, state, or exact unary ABS/NEG/SIGN/FLOOR/CEIL/RELU sites. Composite formulas and
Compiler-generated gradients inherit only their named arithmetic sites' permissions. NaN class,
infinity, overflow, domain, and exact guard rules are not finite-tolerance decisions.

#### Exact and discrete floor

Kind and attributes, input/output arity and descriptors, Shape/layout/axis mapping, contributor
membership, masks, indices, graph-RNG state transitions, traversal, casts, conversions, ordering,
stability, ties, empty-domain behavior, identities, divisors, and publication remain exact. A
selected stored value is one original candidate representation. Predicates gain no epsilon, and
approximation cannot choose an index, mask, winner, ordering, or state transition. Floating
comparisons and extrema observe stored numeric values, including represented subnormals; they
retain family NaN, opposite-signed-zero, tie, winner, and index rules. A selected result remains
an original candidate. For example, positive minimum subnormal compares greater than +0 and
remains the MAX winner; MIN selects +0. WHERE makes an exact Boolean branch choice, then applies
the declared cast if promotion requires one; same-type selection preserves the chosen bits.
Movement, copying,
CONTIGUOUS, classification, Boolean logic, casts, indexing, ordering, arg-extrema, RNG
state/masks, and max-pooling selection gain no arithmetic relaxation from adjacent work.

#### Floating arithmetic primitives

Each named ADD, SUB, MUL, and DIV site uses the family's declared working dtype and rounding.
A corresponding multiply/add may fuse only without a Model-observable intermediate; no
cross-node contraction is authorized. Elementary POW, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, and
TANH sites retain their named mathematical function and domain/special-value rules. Finite
ordinary-result accuracy is route-qualified in tests, not bounded by a public ordered-distance
ceiling. An ordinary finite result cannot become NaN or infinity except through genuine
overflow or a domain path after a permitted DAZ choice. Required infinity and zero signs remain
intact except at approved arithmetic DAZ/FTZ sites or under the family's existing final
exact-zero publication rule. No finite tolerance substitutes for class or sign checks.

#### Aggregate floor

After exact mapping, masking, padding, bounds, and selection determine one logical cell, every
declared contributor participates exactly once. Floating evaluation may choose a rounded binary
tree, use the permitted arithmetic-site DAZ/FTZ rules, and fuse only a corresponding multiply/add
without an observable intermediate. FLOAT32 sites round to FLOAT32; homogeneous low sites work
and accumulate in FLOAT32 with one final low narrowing. Exact empty/point identities and mandatory
divisors remain. No evaluation may drop, duplicate, invent, pretruncate, or replace a term.

This floor applies recursively to ordinary and masked reductions, sum-to-Shape, scans,
overlap/scatter reductions, statistics and norms, loss reductions, matrix/convolution/attention/
recurrent contractions, and average pooling. Extrema and winner selection use the exact/discrete
floor. The established final exact-zero publication freedoms for qualifying SUM, MEAN,
SUM-to-Shape, and nonempty MATMUL results remain part of their Model formulas; they do not grant
intermediate or neighboring-operation sign freedom.

For floating scatter MUL, the base and every addressed update contribute exactly once, but a
rounded multiplication tree may differ from a once-rounded exact abstract product: an
overflow-first `maxFinite * 2 * +0` may produce NaN. A NaN factor or a zero/infinity pair still
requires NaN class. Integral fixed-width modular MUL, exact target mapping, and raw unaddressed
cells remain unchanged.

#### Composite inheritance

Normalization, activation, loss, attention, pooling, convolution, random/dropout, recurrent,
visible convenience composition, and Compiler-generated gradient formulas receive no separate
error envelope. Their exact guards run first, and their authoritative Model formulas recurse
through the exact/discrete, primitive, and aggregate sites. Saved outputs, statistics, masks, and
indices are exact stored values. Opaque routes require the same occurrence-scoped exact, special,
and finite test qualification as explicit routes; no operation name or sample grants support.

General scan composition inherits these same family arithmetic and exact saved-value rules; it
introduces no numerical selector, alternative derivative numerics or blanket dtype capability.
The [scan owner](recurrent-scan.md#traversal-empty-domains-and-declared-qualification) defines its
bounded algebra and qualification sequence, not a replacement family formula.

The recursive sites for every current first-class composite formula are closed as follows. An
exact typed attribute or stored input is an exact leaf. A formula-named real constant is rounded
once to nearest-even `FLOAT32` before its first use and is not an approximation site.

| Formula family | Complete recursive sites |
|---|---|
| Unary composites | `RSQRT` is `SQRT` then typed `+1 / root`. `SIGMOID` first applies its exact sign guard; the nonnegative branch has negation, `EXP`, typed-one addition, and division, while the negative branch has `EXP`, typed-one addition, and division. Exact GELU has typed `0.5`, `1`, and `2`, `SQRT(2)`, `x / sqrt(2)`, `ERF`, addition, and two multiplications. Tanh GELU has typed `0.5`, `1`, `2`, `pi`, and `0.044715`, `x*x`, `x^2*x`, `2/pi`, `SQRT`, the inner multiply/adds, `TANH`, and the outer add/multiplies. SiLU uses the exact sign guard and the corresponding sigmoid branch plus the final multiplication; equivalently its nonnegative branch is `x/(1+EXP(-x))` and its negative branch is `x*EXP(x)/(1+EXP(x))`. None of `RSQRT`, `SIGMOID`, GELU, tanh GELU, or SiLU is one elementary site. |
| Aggregate/statistical composites | `MEAN` is an all-terms-once sum then mandatory count division. `VARIANCE` is mean, one subtraction and `x*x` per contributor, aggregate sum, then `N-correction` division; standard deviation adds `SQRT`. L1 norm applies `ABS` then sum. L2 norm applies `x*x`, sum, then `SQRT`. `LOG_SUM_EXP` applies `EXP` once per contributor, aggregate sum, then `LOG`; a stable replacement must preserve guards, special values, and family formula under route qualification. Cumulative and overlap/scatter arithmetic use their declared per-contributor operation and aggregate tree. |
| Softmax and normalization | On an all-finite nonempty slice, softmax and log-softmax use the maximum-shifted sites defined below; unshifted `EXP` is not a required intermediate. Layer norm uses sum/count mean, per-value subtraction and `x*x`, sum/count variance, epsilon addition, `SQRT`, division, and optional scale multiply/bias add. RMS norm uses `x*x`, sum/count, epsilon addition, `SQRT`, division, and optional scale multiply. BatchNorm inference uses subtraction, variance/epsilon addition, `SQRT`, division, scale multiply, and bias add. Training additionally uses sum/count mean, centered `x*x` sums with the exact `N` and `N-1` divisors, epsilon addition, `SQRT`, typed-one division, and the fixed multiply/add running-statistic transitions. |
| Losses | MSE uses one subtraction and `delta*delta` per position, then its exact reduction and optional divisor. Categorical cross-entropy uses exact max selection, score-minus-max subtraction, `EXP`, class sum, `LOG`, max addition, logit subtraction, exact zero-target/ignore guards, target multiplication where declared, loss sum, negation, and optional reduction divisor. |
| Linear, convolution, pooling, and attention | MATMUL and Conv2d/Conv3d use one multiply per exact mapped pair and an all-terms-once contraction; Conv2d/Conv3d then apply their optional bias addition, while visible linear/Conv1d composition adds only its documented bias or rank edits. Average Pool2d/Pool3d sums every declared kernel position then divides by the exact fixed divisor; max pooling is exact winner selection. Attention scores use query/key multiply-contractions and scale multiplication; an absent scale uses exact positive embedding extent conversion, `SQRT`, and typed-one division. Exact mask/causal guards precede attention's own literal eligible-score `EXP`/sum/division sites, and output rows use value-weight multiply-contractions. The first-class stable finite-softmax rule does not replace those guarded attention sites. |
| Random and recurrent | A kept dropout value uses typed-one-minus-probability, typed-one division, then input multiplication; the draw, mask, dropped positive zero, and state transition are exact. RNN uses two multiply-contractions, optional bias addition, addition, and `TANH`. GRU uses the documented packed contractions/bias adds, gate additions and sigmoid branches, reset multiplication, candidate `TANH`, `h-n`, update multiplication, and final addition. LSTM uses its packed contractions/bias adds, gate additions and sigmoid/tanh branches, cell multiplications/addition, final `TANH`, and hidden multiplication. Traversal, valid-length guards, skipped work, outputs, and state publication remain exact. |

For each all-finite nonempty softmax slice, select the greatest represented input value `m` by
exact stored-value comparison, with no DAZ or arithmetic approximation at that selection. In the
family's working dtype, subtract `m` from each represented `x_i`, and include each
`EXP(x_i-m)` contributor exactly once in the sum using the aggregate floor. An implementation
may recompute a shifted exponential when producing its output, without adding a sum contributor. The
shift, exponential result, and aggregate accumulation are FLOAT64 for FLOAT64 inputs and FLOAT32
for FLOAT32, BFLOAT16, FLOAT16, FLOAT8_E4M3, and FLOAT8_E5M2 inputs. The resulting sum is positive and finite when all
working shifts are finite. SOFTMAX divides each shifted exponential by that sum; LOG_SOFTMAX
subtracts `LOG(sum)` from each working shift. Each output is then published in its input dtype,
with a single final ties-to-even narrowing for a low output. LOG_SOFTMAX does not take the
logarithm of a rounded SOFTMAX output: an underflowed probability can therefore have a finite
log-probability. Finite represented inputs can still produce a negative-infinity working shift
when subtraction genuinely overflows the working dtype; its exponential is positive zero and its
log-softmax result is negative infinity. No backend may instead evaluate unshifted `EXP` and use
its incidental overflow to turn an otherwise finite-slice probability into NaN.

The shifted-exponential sum may use compensated accumulation, including Kahan's rounded
sum/residual update, as an explicit softmax-family exception to the ordinary rounded-binary-tree
evaluation shape. The compensation is rounding-error state, not an additional slice contributor;
each shifted exponential still enters the aggregate exactly once. Compensation arithmetic uses
the same working dtype and named arithmetic-site permissions. It may not move an exact guard,
alter contributor membership, or cross a public or saved-value boundary.

Special-value and empty-slice rules are separate from that all-finite construction. An empty slice
publishes no positions. Any NaN in a nonempty slice makes every result NaN. With one or more
positive infinities and no NaN, each positive-infinity position is NaN; every other position is
positive zero for SOFTMAX and negative infinity for LOG_SOFTMAX. A slice with negative infinity
and no positive infinity or NaN retains the pre-existing literal formula: working `EXP(x_i)` for
every position, an all-contributors-once rounded sum, then division for SOFTMAX or `LOG(sum)`
and input subtraction for LOG_SOFTMAX. Thus finite members are not automatically stabilized in
a mixed-infinity slice. For example, `[-infinity, 1000]` at FLOAT32 has SOFTMAX `[+0, NaN]` and
LOG_SOFTMAX `[-infinity, -infinity]` because the finite member's unshifted EXP overflows; this
preserves the prior result classes. An all-negative-infinity nonempty slice produces NaN at
every position. NaN payload and sign are unspecified. These are Model results, not a promise that
every backend admits every represented value. CPU's narrower runtime input domain belongs to
[its finite-softmax admission owner](backend-execution.md#cpu0077-finite-softmax-admission-parity);
it does not redefine this formula or let an incidental unshifted overflow change the result.

Guarded attention's no-eligible, all-negative-infinity, and positive-infinity-tie cases retain
their family-specific results rather than inheriting ordinary softmax's result by approximation.
Categorical cross-entropy already has its own guarded max-shifted formula; this decision does not
change its target/ignore guards, accumulator sites, or allowed results. LOG_SUM_EXP and other
composites retain their own formulas and cannot inherit this shift by analogy.

Visible convenience compositions recurse through the actual public Tensor operations they create.
Compiler-generated gradients likewise recurse through every captured arithmetic, elementary,
aggregate, comparison, `WHERE`, cast, layout, index, and saved-value site. Another internal
algorithm must retain exact guards, contributor domain, observable boundaries, and special-value
rules and pass route-specific finite qualification. A passing sample or tolerance cannot license
an algebraic rewrite, omitted contributor, or new provider capability.

#### Model low-precision contract

The Model-level FLOAT16, BFLOAT16 and OFP8 types own value, promotion, cast and factory
semantics independently of native availability. CPU low-precision parity is authorized by the
user; Metal retains its independently qualified BFLOAT16/FLOAT16 domain and rejects FP8.
This subsection defines allowed results, not an inference of support from another dtype or a
permission to replace Compiler-owned derivatives with vendor AD.

FLOAT16, BFLOAT16, FLOAT8_E4M3 and FLOAT8_E5M2 are distinct storage types. An operation
with unlike low floating types promotes to FLOAT32; any FLOAT32 operand promotes low types
to FLOAT32, and a FLOAT64 operand promotes to FLOAT64. Homogeneous low occurrences retain
their declared low output type. CPU follows this promotion; other backends qualify independently.

Kind, attributes, arity, descriptors, Shape/layout mapping, raw storage and movement, contributor
domain, guards, predicates, indices, masks, state transitions, selection, ordering, casts, public
outputs, and saved values remain exact. Casts retain their declared conversion. Each declared low
arithmetic output performs one final round-to-nearest, ties-to-even narrowing; no implementation
may pre-narrow an input, contributor, accumulator, public value, or saved value.

Every admitted family must have a qualified implementation whose low arithmetic uses `FLOAT32`
working and accumulator values and the arithmetic-site DAZ/FTZ table above. Its Model contract
retains family-specific arithmetic exact-zero sign choices. Arithmetic NaN must retain NaN class and
the Model formula's domain behavior, but its payload and sign are not accuracy requirements. These
freedoms never affect a predicate, index, mask, selection, state transition, raw copy, or cast.

Within one operation, arithmetic may reassociate and use corresponding `FLOAT32` FMA. Fusion may
cross only a semantically unobservable intermediate with exactly one consumer. It may never cross
a public output, saved value, fan-out, predicate, index, mask, selection, or state boundary.

Finite accuracy qualification belongs only in tests with explicit family/dtype/backend/route/size
oracles, metrics, thresholds, and cancellation boundaries. NaN/infinity, signed zero, subnormal/
underflow, domain and exact cases have separate conformance checks. Model owns no public forward-
error envelope; tests do not become production policy.

Activation is occurrence- and backend-specific. Project actual current ACCELERATOR true and false
CPU/Metal provider answers to profile-free occurrence keys; independently query low types and
retain negative predicates. No answer is inferred from semantic reach or another dtype. Provider
capability and any architecture target mapping remain separate from route,
runtime/device, and generated-backward facts.

Metal realizes every admitted BFLOAT16/FLOAT16 occurrence through its fixed custom program.
Environment records, program certificates, vendor-route probes, and trace certificate verdicts
are not active contract state. The current route decision is governed by the backend-execution
contract and
[ADR 0025](../../design/decisions/0025-custom-only-metal-low-precision.md). The canonical ledgers
and append-only allocations originated in
[ADR 0023](../../design/decisions/0023-low-precision-accelerator-parity.md).

No family permits unguarded reciprocal substitution,
algebraic identities absent from the Model formula, cross-node contraction, term loss, hidden
state changes, or tolerance-based predicate/selection changes. In particular,
`maxFinite / maxFinite -> 0`, `+infinity / maxFinite -> NaN`, `RELU(NaN) -> +0`, and
`TANH(NaN) -> +1` remain invalid. Low-precision transformed algorithms and observable-single-use
fusion remain subject to the same exact/special boundaries and route qualification.

Model defines the single allowed-result contract. Config, Planning, Compiler, Prepare, Engine,
Runtime, Trace, and backends do not transport a replacement numerical selector. Planning queries
actual backend occurrence capability without interpreting numerical semantics. Backend preparation
must fail closed if its route is unqualified. Changed route, generated-artifact, native, and tuning
identities are append-only and must reject stale serialized or cached input; the implementation
cutover assigns exact new versions.

#### OFP8 representation and operation boundaries

`FLOAT8_E4M3` and `FLOAT8_E5M2` are differentiable one-byte OFP8 storage types appended
after FLOAT16. E4M3 has sign/exponent/fraction widths 1/4/3, bias 7, signed zero, gradual
subnormals, maximum finite magnitude 448, and NaN encodings 0x7f/0xff; it has no infinity.
E5M2 has widths 1/5/2, bias 15, signed zero, gradual subnormals, maximum finite magnitude
57344, infinity encodings 0x7c/0xfc, and NaNs with exponent 31 and nonzero fraction.

Numerical conversion is non-saturating round-to-nearest ties-to-even. E5M2 overflow produces
signed infinity; E4M3 out-of-range rounding and infinity input produce canonical NaN 0x7f.
Narrowed NaNs canonicalize to 0x7f for E4M3 and 0x7e for E5M2. Direct FLOAT64 and integer
casts round to the destination without an intervening lossy FLOAT32 conversion. Exact same-type
casts, raw factories, movement, scalar metadata and persistence preserve every stored bit.
Widening represents the original value and preserves represented NaN sign/payload where the
destination permits it; no numeric conversion changes a raw import.

Each ordinary operation uses its existing formula with FLOAT32 working values and accumulators,
then one final FP8 narrowing at each public/saved/ordinary operation-result boundary. There is
no implicit scale, clipping, dynamic range calibration, stochastic rounding or loss scaler.
An E4M3 result whose mathematical/working class is infinity consequently narrows to NaN,
including empty floating extrema and function poles. Nonempty extrema still select an exact
original represented winner; an unrepresentable infinity identity must not poison that selection.
Masks, indices, comparison, RNG state and raw paths do not acquire arithmetic latitude.

Existing Compiler AD, Scan and fixed recurrent owners are reused with their original rounding
boundaries. FP8 coefficients for precision-sensitive normalization/attention metadata may use
FLOAT32, just as FLOAT16 metadata does. Training uses FLOAT32 masters for all low parameter
types and rejects nonfinite represented candidates transactionally. Its objective remains an
explicit scalar FLOAT32/FLOAT64 value; conversion to that objective is visible in the graph.
Generic source/value persistence appends new dtype IDs without changing any old dtype ID or
old accepted bytes; new FP8 wire domains are separately versioned. FP8 does not authorize a
new backend, graph compiler, execution owner or Metal native wire value.


### `modules/config`

`modules/config` owns declarative configuration only.

Allowed:

- `CompileConfig`
- `CompileMode`
- `BackendIntent`
- `GraphOptimizationConfig`
- `PartitionScoringConfig`
- `PartitionScoringPolicy`
- `PrepareConfig`
- `CpuPrepareConfig`
- `AcceleratorPrepareConfig`
- `RunOptions`
- `PublicationPolicy`
- immutable declarative planning-cost inputs, after their planning consumer is stable
- immutable declarative model-autotuning inputs, after their owning contracts are stable

Forbidden:

- live services
- concrete backend classes
- kernel class references
- runtime state
- executable units
- backend-specific implementation logic
- benchmark runners, model-autotuning search, cache mutation, or measurement algorithms
- live platform discovery or mutable measurement evidence

Backend-specific interpretation of config belongs to backend prepare.

### `modules/planning`

`modules/planning` owns backend-neutral compile-time planning.

Allowed:

- backend intent propagation
- capability query contracts
- capability matrix construction
- backend-neutral partition scoring
- node ownership decisions
- segment ownership decisions
- maximal same-owner partitioning
- logical materialization requirements
- logical memory requirements
- backend-neutral planning cost estimates and profiles

Forbidden:

- fusion implementation
- specialization
- concrete kernel selection
- OpenBLAS route selection
- Vector API route selection
- MPSGraph route selection
- CUDA kernel selection
- physical memory allocation
- runtime residency
- prepared schedules
- prepared executables
- backend-specific DAG construction
- backend-specific lowering
- runtime execution units
- concrete kernel/runtime scoring
- backend route names or route-selection parameters
- vector species, lane counts, unroll factors, thread counts, chunk sizes, or tile sizes

Planning answers:

```text
Where should this node or segment run?
```

Planning must not answer:

```text
Which concrete kernel, executable, BLAS route, MPSGraph route, or CUDA implementation should run it?
```

For a general scan, Planning selects one owner for the complete body, never mixed-owner body
partitions. Logical memory planning carries ordinary-use residual demand, not source Tensors or
an opaque saved workspace. Complete scan semantics remain with the
[scan owner](recurrent-scan.md#canonical-residual-ports-and-demand).

Compiler0013 emits length-mode queries with the entire header, ordered row roles and body/demand
without treating the extra lengths actual as a body formal. Existing transport and Compiler
inference/AD alone grant no execution capability; providers reject until their complete binder/body is qualified under
the [scan owner's gates](recurrent-scan.md#runtime-preflight-and-implementation-gates).

## Profile-free capability queries

<a id="numerical-profile-transport"></a>

Profile transport is removed. `OperationCapabilityQuery`
retains complete operation, attribute, ordered descriptor, and gradient facts, but no numerical
profile. Its deterministic boolean provider answer cannot be inferred from a family formula,
another dtype, device sample, or the removed selector.

The accepted general-scan query extends OperationCapabilityQuery with
`Optional<CompiledScanBody>` while retaining the three-argument empty-body construction. Compiler
supplies the validated complete scoped body, role/port descriptors and demanded outer-port indices;
queries remain Tensor-free. Propagate and revalidate demand after rewrites through Prepare.
The deterministic provider boolean must qualify the header, every body occurrence/descriptor,
residual and binding requirements, and generated topology together with the loop binder. Supported
child queries are necessary but insufficient. Missing/malformed bodies or unsupported binders
reject; Planning forwards semantic facts without interpreting backend routes or implementing AD.
Planning0008 implements the optional-body transport and provider contract tests. Compiler0011
implements body-bearing query emission, scoped capture/inference and first-order AD/demand propagation.
Prepare0010 implements scoped projection and static binding/demand validation. CPU0030 adds a
bounded FLOAT32 same-shape pointwise provider predicate; both the forward body and its actual
Compiler-generated reverse must independently qualify. CPU0031 extends that predicate to
homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded shape/layout closure. CPU0032 adds static
homogeneous FLOAT32 rank-two MATMUL under the same independent whole-body/binder proof for actual
forward and reverse occurrences. CPU0033 extends the complete bounded declared algebra and actual
first-order reverse closure to homogeneous FLOAT64 under that same independent qualification;
Model source admission and other dtype/layout capability remain separate. See the
[sole scan protocol](recurrent-scan.md#general-differentiable-scan).

### General Scan activity transport

Implemented coordinated Class C transport, accepted with Compiler0017/CPU0081 after independent
authority/interface and exact-source implementation reviews and the affected integration gate.
Planning owns only the immutable carrier/query transport, not derivative formulas/proof,
physical storage or runtime control. The
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

`graph` is the exact independently scoped compiled body. Root order is carries then seedFormals;
scope lists are defensive immutable sorted unique root-index sets. Carry/seed/accumulator
positions are declaration positions, not deduplicated actual identities. Every executable local
node has a scope; merge keys identify actual local ordinary ADDs. Intrinsic constructor checks
cover complete membership, root/scope ranges, unique positions, deterministic immutable maps and
valid local merge shapes. Containing-occurrence consumers independently check header/formal/
binding applicability, scopes/dataflow and ordinary predicates. No Tensor, producer, numerical
cotangent, derivative formula, address, physical slot or mutable runtime object is retained.
The carrier is strictly local: it contains no external presence/length references and does not
transport activity through ordinary Tensor publications. Root initialization follows the
Scan owner's materialized-boundary semantics; the selected record signatures remain unchanged.

OperationCapabilityQuery becomes `(operation, inputs, outputs, Optional<CompiledScanBody> scanBody,
Optional<ScanActivityPlan> scanActivityPlan)`. A plan applies only to a body-bearing Scan and
names that exact body graph. Migrate every four-argument body-bearing caller; the meaningful
three-argument ordinary empty-attachment constructor remains. Missing ordinary support or
malformed applicable control still fails closed independently. No empty optional serialization
field or identity suffix is added to old recipes, and no new numerical profile is introduced.

