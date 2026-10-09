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

`modules/trace` owns typed diagnostic DTOs only.

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

Device storage belongs to runtime/backend layers, not model.

### Profile-free numerical semantics

<a id="numerical-profiles"></a>

The legacy anchor above preserves links from historical profile-era records. Model
alone owns each operation family's formula, contributor domain, guards, precision/rounding sites,
special-value behavior, and allowed results. There is no numerical selector, strict/accelerator
result-set union, public accuracy envelope, production tolerance registry, or runtime numerical
certificate. Backend support remains an occurrence-specific provider answer, not a consequence of
removing a selector.

#### Family formulas and unary sites

Every current unary kind retains its formula for each accepted BFLOAT16, FLOAT16, FLOAT32, and
FLOAT64 type. ABS and NEG perform exact represented sign clearing/inversion. SIGN, FLOOR, CEIL,
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

The recursive sites for every current first-class composite formula are closed as follows. An
exact typed attribute or stored input is an exact leaf. A formula-named real constant is rounded
once to nearest-even `FLOAT32` before its first use and is not an approximation site.

| Formula family | Complete recursive sites |
|---|---|
| Unary composites | `RSQRT` is `SQRT` then typed `+1 / root`. `SIGMOID` first applies its exact sign guard; the nonnegative branch has negation, `EXP`, typed-one addition, and division, while the negative branch has `EXP`, typed-one addition, and division. Exact GELU has typed `0.5`, `1`, and `2`, `SQRT(2)`, `x / sqrt(2)`, `ERF`, addition, and two multiplications. Tanh GELU has typed `0.5`, `1`, `2`, `pi`, and `0.044715`, `x*x`, `x^2*x`, `2/pi`, `SQRT`, the inner multiply/adds, `TANH`, and the outer add/multiplies. SiLU uses the exact sign guard and the corresponding sigmoid branch plus the final multiplication; equivalently its nonnegative branch is `x/(1+EXP(-x))` and its negative branch is `x*EXP(x)/(1+EXP(x))`. None of `RSQRT`, `SIGMOID`, GELU, tanh GELU, or SiLU is one elementary site. |
| Aggregate/statistical composites | `MEAN` is an all-terms-once sum then mandatory count division. `VARIANCE` is mean, one subtraction and `x*x` per contributor, aggregate sum, then `N-correction` division; standard deviation adds `SQRT`. L1 norm applies `ABS` then sum. L2 norm applies `x*x`, sum, then `SQRT`. `LOG_SUM_EXP` applies `EXP` once per contributor, aggregate sum, then `LOG`; a stable replacement must preserve guards, special values, and family formula under route qualification. Cumulative and overlap/scatter arithmetic use their declared per-contributor operation and aggregate tree. |
| Softmax and normalization | Literal softmax applies `EXP` per slice value, aggregate sum, then division per output; literal log-softmax applies the same `EXP`/sum, then `LOG` and output subtraction. Layer norm uses sum/count mean, per-value subtraction and `x*x`, sum/count variance, epsilon addition, `SQRT`, division, and optional scale multiply/bias add. RMS norm uses `x*x`, sum/count, epsilon addition, `SQRT`, division, and optional scale multiply. BatchNorm inference uses subtraction, variance/epsilon addition, `SQRT`, division, scale multiply, and bias add. Training additionally uses sum/count mean, centered `x*x` sums with the exact `N` and `N-1` divisors, epsilon addition, `SQRT`, typed-one division, and the fixed multiply/add running-statistic transitions. |
| Losses | MSE uses one subtraction and `delta*delta` per position, then its exact reduction and optional divisor. Categorical cross-entropy uses exact max selection, score-minus-max subtraction, `EXP`, class sum, `LOG`, max addition, logit subtraction, exact zero-target/ignore guards, target multiplication where declared, loss sum, negation, and optional reduction divisor. |
| Linear, convolution, pooling, and attention | MATMUL and Conv2d/Conv3d use one multiply per exact mapped pair and an all-terms-once contraction; Conv2d/Conv3d then apply their optional bias addition, while visible linear/Conv1d composition adds only its documented bias or rank edits. Average Pool2d/Pool3d sums every declared kernel position then divides by the exact fixed divisor; max pooling is exact winner selection. Attention scores use query/key multiply-contractions and scale multiplication; an absent scale uses exact positive embedding extent conversion, `SQRT`, and typed-one division. Exact mask/causal guards precede the literal softmax sites above, and output rows use value-weight multiply-contractions. |
| Random and recurrent | A kept dropout value uses typed-one-minus-probability, typed-one division, then input multiplication; the draw, mask, dropped positive zero, and state transition are exact. RNN uses two multiply-contractions, optional bias addition, addition, and `TANH`. GRU uses the documented packed contractions/bias adds, gate additions and sigmoid branches, reset multiplication, candidate `TANH`, `h-n`, update multiplication, and final addition. LSTM uses its packed contractions/bias adds, gate additions and sigmoid/tanh branches, cell multiplications/addition, final `TANH`, and hidden multiplication. Traversal, valid-length guards, skipped work, outputs, and state publication remain exact. |

Guarded attention's no-eligible, all-negative-infinity, and positive-infinity-tie cases retain
their family-specific results rather than inheriting literal softmax's result by approximation.

Visible convenience compositions recurse through the actual public Tensor operations they create.
Compiler-generated gradients likewise recurse through every captured arithmetic, elementary,
aggregate, comparison, `WHERE`, cast, layout, index, and saved-value site. Another internal
algorithm must retain exact guards, contributor domain, observable boundaries, and special-value
rules and pass route-specific finite qualification. A passing sample or tolerance cannot license
an algebraic rewrite, omitted contributor, or new provider capability.

#### Model low-precision contract

The Model-level `FLOAT16` type and its value, promotion, cast, and factory semantics are active.
CPU and Metal backend wire/schema/identity/provider/route/runtime support is active for the
documented exact and arithmetic occurrence domains. This subsection owns their allowed results; it
does not imply an unlisted occurrence, a direct mixed BFLOAT16/FLOAT16 operation, or generated
backward ownership beyond the separately admitted primitive closure.

`FLOAT16` and `BFLOAT16` are distinct, incomparable storage types. A mixed
`FLOAT16`/`BFLOAT16` floating operation promotes to `FLOAT32`; either low type combined with
`FLOAT32` promotes to `FLOAT32`, and a combination containing `FLOAT64` promotes to `FLOAT64`. A
homogeneous low-precision occurrence retains its declared low output type.

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

## Profile-free capability queries

<a id="numerical-profile-transport"></a>

Profile transport is removed. `OperationCapabilityQuery`
retains complete operation, attribute, ordered descriptor, and gradient facts, but no numerical
profile. Its deterministic boolean provider answer cannot be inferred from a family formula,
another dtype, device sample, or the removed selector.
