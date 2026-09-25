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

For unary semantics, the strict baseline is now explicit for all nineteen current kinds. It
preserves represented normal and subnormal inputs and results: no denormals-are-zero (DAZ),
flush-to-zero (FTZ), reduced precision, reciprocal estimate, or unlisted reassociation applies.
NaN must remain NaN, although payload, quieting, and sign are not promised. The per-kind domain,
infinity, and signed-zero rules in `UnaryElementwiseKind` remain mandatory.

Finite same-format accuracy uses an ordered-representation distance only where stated. For a raw
binary32 word `b`, define `key32(b)` as unsigned `~b` when its sign bit is set and unsigned
`b ^ 0x80000000` otherwise; binary64 uses the analogous sign-bit transform. Distance is the
unsigned absolute difference between keys. Thus the reference has distance zero and negative zero
and positive zero are adjacent. Every ceiling is inclusive, while zero sign, NaN, infinity,
overflow, underflow, and domain are checked separately.

The current strict unary result set is:

| Kinds | Strict ordinary-finite result rule |
|---|---|
| `ABS`, `NEG`, `RECIPROCAL`, `FLOOR`, `CEIL`, `SIGN`, `RELU` | Exact selected represented operation: sign edit, typed `+1 / x`, integral rounding, sign classification, or `max(x,+0)` respectively. |
| `LOG`, `LOG1P`, `EXP`, `EXPM1` | The selected scalar or lane-wise Java operation. Its one-unit-in-the-last-place contract is at most ordered distance two from the correctly rounded exact same-format reference at a binade boundary. FLOAT32 lane operations use the specified exact widening to binary64 and one final narrowing. |
| `SQRT` | Correctly rounded same-format principal square root. |
| `TANH` | The selected scalar or lane-wise Java operation. Its 2.5-unit-in-the-last-place contract is at most ordered distance five from the correctly rounded exact same-format reference at a binade boundary. The retained five-step scalar differential gate is corroboration, not an additional error term. |
| `RSQRT` | The exact union of the retained first-class scalar and lane realizations of typed `+1 / sqrt(x)`: FLOAT32 scalar widens once, performs binary64 square root and division, then narrows once; the FLOAT32 lane form performs typed square root then typed division. |
| `ERF` | The exact union of the selected scalar Cephes-derived piecewise realization and selected typed lane realization, including their fixed coefficients, branches, operation order, and special corrections. |
| `SIGMOID` | The selected stable branch: `1/(1+exp(-x))` for nonnegative input and `exp(x)/(1+exp(x))` for negative input, in the result format's retained scalar realization. |
| `GELU` | The selected exact-GELU realization `0.5*x*(1+erf(x/sqrt(2)))`, using its selected strict `ERF` realization and fixed operation order. |
| `GELU_TANH_APPROXIMATION` | The selected fixed formula `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))` and fixed operation order. |
| `SILU` | The selected stable branch: `x/(1+exp(-x))` for nonnegative input and `x*exp(x)/(1+exp(x))` for negative input. |

The retained CPU relative-error checks for `ERF`, `SIGMOID`, both GELU kinds, and `SILU` qualify
those fixed realizations; they are not public strict result envelopes. In particular, they cannot
turn an FTZ result or a wrong zero, infinity, NaN, or domain class into a strict result. BFLOAT16
has the same mathematical targets and mandatory special classes, but no current backend unary
capability; its future finite realization requires separate qualification rather than inference
from storage width.

`ACCELERATOR` is an opt-in superset of the complete `STRICT_IEEE` allowed-result set. A backend may
always produce a strict result. For `FLOAT32`, it may additionally evaluate any current operation
through the three recursive floors below. Every other data type retains strict behavior. For one
operation occurrence and descriptor domain, strict capability and behavior must remain subsets of
accelerator capability and behavior.

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
aggregate, comparison, `WHERE`, cast, layout, index, and saved-value site. A first-class or opaque
selector may use another internal algorithm only after its complete output set, for the complete
descriptor domain, is proved to be a subset of the results generated by this ledger and the three
floors; a passing sample, operation name, or final-output tolerance is insufficient.

Neither profile permits reduced precision, reciprocal substitution for division, algebraic
identities absent from the Model formula, cross-node contraction, term loss, hidden state changes,
or tolerance-based predicate/selection changes. In particular, `maxFinite / maxFinite -> 0`,
`+infinity / maxFinite -> NaN`, `RELU(NaN) -> +0`, and `TANH(NaN) -> +1` remain invalid.
`FLOAT64`, `BFLOAT16`, future `FLOAT16`, integral values, and `BOOL` acquire no relaxed arithmetic
meaning.

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
