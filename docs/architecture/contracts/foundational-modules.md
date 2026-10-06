# Foundational module contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns the detailed responsibilities and prohibitions for Trace, Backend Contract,
Model, Config, and Planning. Model's scope includes the sole normative definition of graph
numerical profiles and profile-indexed allowed-result sets. Planning scope here is the module
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
- profile-indexed operation allowed-result sets

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

### Numerical profiles

Model is the sole semantic owner of graph numerical profiles and the allowed-result set for every
operation under each profile. The profile names in this contract establish semantic identities;
they do not by themselves add a Java enum, configuration API, backend capability, runtime policy,
native schema, cache codec, or trace field.

`STRICT_IEEE` means the exact current per-operation Model contract. It preserves every existing
family-specific rounding, approximation, NaN-payload, signed-zero, accumulation, reassociation,
and fused-multiply-add promise or freedom. The name does not promise universal bitwise identity,
correct rounding, Java `strictfp`, a fixed instruction sequence, or cross-backend equality beyond
each operation's existing contract.

For unary semantics, the strict baseline is complete for every current kind and every accepted
`BFLOAT16`, `FLOAT16`, `FLOAT32`, and `FLOAT64` input. Let `F` denote one of those formats, let
`RN_F(z)` round the exact real `z` once to `F` using round-to-nearest, ties-to-even with gradual
underflow, and let `q_F(f,x) = RN_F(f(real(x)))`. Overflow produces the correctly signed infinity.
No denormals-are-zero (DAZ), flush-to-zero (FTZ), reduced precision, reciprocal estimate,
unlisted reassociation, or contraction applies.

Finite accuracy uses an ordered-representation distance where stated. For an unsigned raw
`BFLOAT16` or `FLOAT16` word `b`, `key16(b)` is unsigned `~b` when its sign bit is set and unsigned
`b ^ 0x8000` otherwise. `key32` and `key64` apply the same transform with their respective sign
bits and word widths. `distance_F(a,b)` is the mathematical absolute difference between the two
unsigned keys. Thus identical references have distance zero and negative zero and positive zero
are adjacent. Every bound below is inclusive.

For a distance-bounded primitive, a zero or infinity reference requires that exact signed result.
A finite nonzero reference requires a finite same-sign result within the stated distance; a
subnormal reference may not become zero. The result also remains inside the mathematical range:
`EXP` is nonnegative, `SQRT` is nonnegative, and `TANH` is in `[-1,1]`. These class, sign, range,
overflow, underflow, and domain requirements are independent of distance.

The Model-owned strict primitive result sets are:

| Kinds | Exact mathematical reference and strict result rule |
|---|---|
| `ABS`, `NEG` | Exact represented sign clearing or sign inversion respectively. |
| `RECIPROCAL` | Exactly `RN_F(1 / real(x))`; this is one `F` division site, not an estimate. |
| `FLOOR`, `CEIL` | The exact represented greatest integer not above, or least integer not below, the input respectively. |
| `SIGN` | Exact same-format `-1`, signed zero, or `+1` according to the represented input sign and zero class. |
| `RELU` | Exact family-extrema `max(x,+0)`. |
| `LOG` | `q_F(ln,x)`, with `distance_F <= 2`. |
| `LOG1P` | `q_F(x -> ln(1+x),x)`, with `distance_F <= 2`; the real addition is part of the mathematical reference and is not rounded first. |
| `EXP` | `q_F(exp,x)`, with `distance_F <= 2`. |
| `EXPM1` | `q_F(x -> exp(x)-1,x)`, with `distance_F <= 2`; the real subtraction is not rounded separately. |
| `SQRT` | `q_F(sqrt,x)`, with `distance_F <= 1`. |
| `TANH` | `q_F(tanh,x)`, with `distance_F <= 5`. |

`ERF` has exact real reference
`erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t*t) dt`. For a finite nonzero input
its strict result
`y` is finite, nonzero, same-signed, in `[-1,1]`, and satisfies
`abs(real(y)-erf(real(x))) <= max(A_F, R_F*abs(erf(real(x))))`, with inclusive constants:

| Format `F` | `A_F` | `R_F` |
|---|---:|---:|
| `BFLOAT16` | `2^-7` | `2^-7` |
| `FLOAT16` | `2^-10` | `2^-10` |
| `FLOAT32` | `2e-5` | `2e-5` |
| `FLOAT64` | `2e-7` | `2e-7` |

All remaining first-class unary functions are Model-owned recursive result sets, not aliases and
not backend-algorithm identities. A strict realization format `E` is either the native `F`, or
one wider format (`BFLOAT16 -> FLOAT32`, `FLOAT16 -> FLOAT32`, or `FLOAT32 -> FLOAT64`);
`FLOAT64` has only its native realization. The represented input injects exactly into `E`. Each
named real constant is rounded once by `RN_E` before first use; every named negation, addition, multiplication, and division is
one `RN_E` site; every elementary call chooses a result from the strict primitive set above for
`E`. Evaluation follows the listed data dependencies with no reassociation or fused operation,
then rounds the final value once by `RN_F`. The strict result is the union over those Model-owned
native and one-wider realizations:

| Kind | Recursive sites, in dependency order |
|---|---|
| `RSQRT` | `s in SQRT_E(x)`; `y = RN_E(1 / s)`. |
| `SIGMOID`, `x >= 0` | `n = RN_E(-x)`; `e in EXP_E(n)`; `d = RN_E(1+e)`; `y = RN_E(1/d)`. |
| `SIGMOID`, `x < 0` | `e in EXP_E(x)`; `d = RN_E(1+e)`; `y = RN_E(e/d)`. |
| `GELU` | `m = RN_E(0.5*x)`; `s in SQRT_E(2)`; `u = RN_E(x/s)`; `e in ERF_E(u)`; `v = RN_E(1+e)`; `y = RN_E(m*v)`. |
| `GELU_TANH_APPROXIMATION` | `x2 = RN_E(x*x)`; `x3 = RN_E(x2*x)`; `c = RN_E(0.044715*x3)`; `u = RN_E(x+c)`; `r = RN_E(2/pi)`; `s in SQRT_E(r)`; `v = RN_E(s*u)`; `h in TANH_E(v)`; `j = RN_E(1+h)`; `m = RN_E(0.5*x)`; `y = RN_E(m*j)`. |
| `SILU`, `x >= 0` | `n = RN_E(-x)`; `e in EXP_E(n)`; `d = RN_E(1+e)`; `y = RN_E(x/d)`. |
| `SILU`, `x < 0` | `e in EXP_E(x)`; `p = RN_E(x*e)`; `d = RN_E(1+e)`; `y = RN_E(p/d)`. |

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

The bounds are Model decisions supported, not defined, by retained conformance evidence. The
one-ULP logarithmic/exponential evidence converts to at most two ordered steps at a binade
boundary; the 2.5-ULP `TANH` evidence converts to at most five; retained square-root qualification
fits one. The retained direct-reference `ERF` gates supply `2e-5` for `FLOAT32` and `2e-7` for
`FLOAT64`. For `BFLOAT16`, `2^-7` dominates one ties-to-even BFLOAT16 narrowing plus the retained
FLOAT32 `ERF` error, while the distance bounds admit a conforming wider-format evaluation after one
final narrowing. Current CPU routes are members because native FLOAT64, native FLOAT32, and
one-wider FLOAT32 evaluation all use the same mathematical references and site graph; no CPU,
library, or coefficient table is semantic authority.

`ACCELERATOR` is an opt-in superset of the complete `STRICT_IEEE` allowed-result set. A backend may
always produce a strict result. `FLOAT32` uses the three recursive floors below. Homogeneous
BFLOAT16/FLOAT16 occurrences admitted under ACCELERATOR use the separate low-precision contract
below; exact-valid low occurrences may also remain profile-common. For one operation occurrence
and descriptor domain, strict capability and behavior must remain subsets of accelerator
capability and behavior.

#### Exact and discrete floor

Kind and attributes, input/output arity and descriptors, Shape/layout/axis mapping, contributor
membership, masks, indices, graph-RNG state transitions, traversal, casts, conversions, ordering,
stability, ties, empty-domain behavior, identities, divisors, and publication remain exact. A
selected stored value is one original candidate representation. Predicates gain no epsilon, and
approximation cannot choose an index, mask, winner, ordering, or state transition. A floating
comparison may compare DAZ-normalized operands, and an extrema operation may treat DAZ-normalized
values as tied; its existing NaN and signed-zero rules still apply and a selected result remains an
original candidate. Movement, copying, `CONTIGUOUS`, `WHERE`, classification, Boolean logic,
casts, indexing, ordering, arg-extrema, RNG state/masks, and max-pooling selection otherwise gain
no arithmetic relaxation from adjacent work.

#### Primitive `FLOAT32` floor

At each arithmetic primitive site named by the operation's Model formula, a subnormal operand may
be read as same-signed zero (denormals-are-zero, DAZ), and a finite subnormal site result may become
either signed zero (flush-to-zero, FTZ). These choices do not rewrite a stored input or another
observable value. Basic `ADD`, `SUB`, `MUL`, and `DIV` still perform one `FLOAT32`
round-to-nearest-even operation. A multiply and corresponding add with no Model-observable
intermediate may use one `FLOAT32` fused multiply-add; contraction never crosses graph nodes.

Only an irreducible elementary-function site — `POW`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`,
`SQRT`, or `TANH` — receives an elementary allowance. Let `r` be the correctly rounded exact
binary32 result after DAZ and before FTZ, and let `a` be the site's result. For an ordinary finite
non-subnormal `r`, the ordered-representation distance defined above must satisfy
`distance(a,r) <= 5`; the ceiling is inclusive. A subnormal `r` is either exact or FTZ.

An ordered-representation distance ceiling of five is conservative, not “five ULP” and not a
claim of an observed or globally minimal error. Java's scalar and Vector `TANH` operations carry
the equivalent Java method's at-most-2.5-ULP exact-result contract; FLOAT32 lanes use the specified
widen-to-binary64, evaluate, narrow-to-binary32 adaptation. At a power-of-two binade boundary the
spacing on the smaller side is half the Java ULP at the boundary, so 2.5 Java ULP can span an
ordered distance of five adjacent-representation steps; away from that boundary the distance is
at most four. The Java one-ULP logarithmic/exponential contract converts to ordered distance at
most two, and
retained Metal `EXP`/`SIGMOID` observations are also within two, so they do not raise the uniform
ceiling. No claim that five is minimal is made.

Domain and special classes remain formula-derived. NaN cannot become ordinary. An ordinary finite
result cannot become NaN or infinity except through genuine overflow or a domain path after DAZ.
Required infinity and zero signs remain intact except for explicit DAZ/FTZ, the established
extrema tie rule, or an operation's existing final exact-zero publication freedom. Distance never
substitutes for those class and sign checks.

#### Aggregate floor

After exact mapping, masking, padding, bounds, and selection determine one logical cell, every
declared contributor participates exactly once. `FLOAT32` evaluation may choose any binary tree,
reassociate, round each arithmetic step to `FLOAT32`, use the primitive DAZ/FTZ rules, and fuse
only a corresponding multiply/add. Exact empty/point identities and mandatory divisors remain.
No evaluation may drop, duplicate, invent, pretruncate, or replace a term.

This floor applies recursively to ordinary and masked reductions, sum-to-Shape, scans,
overlap/scatter reductions, statistics and norms, loss reductions, matrix/convolution/attention/
recurrent contractions, and average pooling. Extrema and winner selection use the exact/discrete
floor. The established final exact-zero publication freedoms for qualifying SUM, MEAN,
SUM-to-Shape, and nonempty MATMUL results remain part of their Model formulas; they do not grant
intermediate or neighboring-operation sign freedom.

#### Composite inheritance

Normalization, activation, loss, attention, pooling, convolution, random/dropout, recurrent,
visible convenience composition, and Compiler-generated gradient formulas receive no separate
error envelope. Their exact guards run first, and their authoritative Model formulas recurse
through the exact/discrete, primitive, and aggregate floors. Saved outputs, statistics, masks, and
indices are exact stored values. An opaque vendor selector is eligible only when its complete
output set is proved to be a subset of this recursive set.

The recursive sites for every current first-class composite formula are closed as follows. An
exact typed attribute or stored input is an exact leaf. A formula-named real constant is rounded
once to nearest-even `FLOAT32` before its first use and is not an approximation site.

| Formula family | Complete recursive sites |
|---|---|
| Unary composites | `RSQRT` is `SQRT` then typed `+1 / root`. `SIGMOID` first applies its exact sign guard; the nonnegative branch has negation, `EXP`, typed-one addition, and division, while the negative branch has `EXP`, typed-one addition, and division. Exact GELU has typed `0.5`, `1`, and `2`, `SQRT(2)`, `x / sqrt(2)`, `ERF`, addition, and two multiplications. Tanh GELU has typed `0.5`, `1`, `2`, `pi`, and `0.044715`, `x*x`, `x^2*x`, `2/pi`, `SQRT`, the inner multiply/adds, `TANH`, and the outer add/multiplies. SiLU uses the exact sign guard and the corresponding sigmoid branch plus the final multiplication; equivalently its nonnegative branch is `x/(1+EXP(-x))` and its negative branch is `x*EXP(x)/(1+EXP(x))`. None of `RSQRT`, `SIGMOID`, GELU, tanh GELU, or SiLU is one elementary site. |
| Aggregate/statistical composites | `MEAN` is an all-terms-once sum then mandatory count division. `VARIANCE` is mean, one subtraction and `x*x` per contributor, aggregate sum, then `N-correction` division; standard deviation adds `SQRT`. L1 norm applies `ABS` then sum. L2 norm applies `x*x`, sum, then `SQRT`. `LOG_SUM_EXP` applies `EXP` once per contributor, aggregate sum, then `LOG`; a stable replacement is valid only by complete-subset proof. Cumulative and overlap/scatter arithmetic use their declared per-contributor operation and aggregate tree. |
| Softmax and normalization | Literal softmax applies `EXP` per slice value, aggregate sum, then division per output; literal log-softmax applies the same `EXP`/sum, then `LOG` and output subtraction. Layer norm uses sum/count mean, per-value subtraction and `x*x`, sum/count variance, epsilon addition, `SQRT`, division, and optional scale multiply/bias add. RMS norm uses `x*x`, sum/count, epsilon addition, `SQRT`, division, and optional scale multiply. BatchNorm inference uses subtraction, variance/epsilon addition, `SQRT`, division, scale multiply, and bias add. Training additionally uses sum/count mean, centered `x*x` sums with the exact `N` and `N-1` divisors, epsilon addition, `SQRT`, typed-one division, and the fixed multiply/add running-statistic transitions. |
| Losses | MSE uses one subtraction and `delta*delta` per position, then its exact reduction and optional divisor. Categorical cross-entropy uses exact max selection, score-minus-max subtraction, `EXP`, class sum, `LOG`, max addition, logit subtraction, exact zero-target/ignore guards, target multiplication where declared, loss sum, negation, and optional reduction divisor. |
| Linear, convolution, pooling, and attention | MATMUL and Conv2d/Conv3d use one multiply per exact mapped pair and an all-terms-once contraction; Conv2d/Conv3d then apply their optional bias addition, while visible linear/Conv1d composition adds only its documented bias or rank edits. Average Pool2d/Pool3d sums every declared kernel position then divides by the exact fixed divisor; max pooling is exact winner selection. Attention scores use query/key multiply-contractions and scale multiplication; an absent scale uses exact positive embedding extent conversion, `SQRT`, and typed-one division. Exact mask/causal guards precede the literal softmax sites above, and output rows use value-weight multiply-contractions. |
| Random and recurrent | A kept dropout value uses typed-one-minus-probability, typed-one division, then input multiplication; the draw, mask, dropped positive zero, and state transition are exact. RNN uses two multiply-contractions, optional bias addition, addition, and `TANH`. GRU uses the documented packed contractions/bias adds, gate additions and sigmoid branches, reset multiplication, candidate `TANH`, `h-n`, update multiplication, and final addition. LSTM uses its packed contractions/bias adds, gate additions and sigmoid/tanh branches, cell multiplications/addition, final `TANH`, and hidden multiplication. Traversal, valid-length guards, skipped work, outputs, and state publication remain exact. |

Visible convenience compositions recurse through the actual public Tensor operations they create.
Compiler-generated gradients likewise recurse through every captured arithmetic, elementary,
aggregate, comparison, `WHERE`, cast, layout, index, and saved-value site. For the current
`ACCELERATOR FLOAT32` recursive floors, a first-class or opaque selector may use another internal
algorithm only after its complete output set, for the complete descriptor domain, is proved to be
a subset of the results generated by the current ledger and three floors; a passing sample,
operation name, or final-output tolerance is insufficient. The active low-precision contract below
instead uses its public cancellation- and size-aware family envelope.

#### Model low-precision contract and active ACCELERATOR extension

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

Every admitted family has a complete custom implementation whose low arithmetic uses `FLOAT32`
working and accumulator values. Its Model contract declares the permitted low-input DAZ,
low-result FTZ, and arithmetic exact-zero sign choices. Arithmetic NaN must retain NaN class and
the Model formula's domain behavior, but its payload and sign are not accuracy requirements. These
freedoms never affect a predicate, index, mask, selection, state transition, raw copy, or cast.

Within one operation, arithmetic may reassociate and use corresponding `FLOAT32` FMA. Fusion may
cross only a semantically unobservable intermediate with exactly one consumer. It may never cross
a public output, saved value, fan-out, predicate, index, mask, selection, or state boundary.

Model owns a public accuracy envelope for each admitted low-precision family. It fixes the exact
domain and special-class rules, reference result, accumulator/final narrowing, and a forward-error
bound whose scale depends explicitly on family size and cancellation sensitivity. It is not a
generic absolute/relative `allclose` tolerance. Every implementation must satisfy that envelope;
an operation name, vendor claim, passing examples, or `allclose` sample alone is insufficient.

Activation is occurrence- and backend-specific. A row is capable only when it maps to a supported
frozen `ACCELERATOR FLOAT32` occurrence for that backend, while every frozen unsupported row
remains excluded. Provider capability and target mapping remain separate from route,
runtime/device, and generated-backward facts.

Metal realizes every admitted BFLOAT16/FLOAT16 occurrence through its fixed custom program.
Environment records, program certificates, vendor-route probes, and trace certificate verdicts
are not active contract state. The current route decision is governed by the backend-execution
contract and
[ADR 0025](../../design/decisions/0025-custom-only-metal-low-precision.md). The canonical ledgers
and append-only allocations originated in
[ADR 0023](../../design/decisions/0023-low-precision-accelerator-parity.md).

For the current `FLOAT32` recursive floors, neither profile permits reciprocal substitution,
algebraic identities absent from the Model formula, cross-node contraction, term loss, hidden
state changes, or tolerance-based predicate/selection changes. In particular,
`maxFinite / maxFinite -> 0`, `+infinity / maxFinite -> NaN`, `RELU(NaN) -> +0`, and
`TANH(NaN) -> +1` remain invalid. The low-precision extension does not weaken its exact discrete
boundaries, but its bounded transformed algorithms and observable-single-use fusion are governed
by its public per-family envelope instead of the literal `FLOAT32` tree rule.

The selected profile is graph-wide and cold. Model defines its result sets. A later Config change
may own only the immutable declarative selector, without semantic or route logic. Planning may
later ask profile-qualified capability questions but must not reinterpret a result set. Later
concrete-backend preparation may realize any result allowed by the selected profile and must fail
closed when it cannot; shared Prepare transports the selection without owning numerical meaning.
The selection must participate in backend route, specialization, generated-artifact, and tuning-
cache identity before relaxed capability is advertised. Runtime and Trace remain profile-free
unless a later coordinated architecture update explicitly changes their contracts.

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

## Numerical-profile transport

Planning has a public API dependency on Config so each `OperationCapabilityQuery` carries the exact
non-null graph-wide `NumericalProfile`. Planning treats that value as query identity only; it does
not define allowed results, select a default, or infer a profile from backend or device facts.
