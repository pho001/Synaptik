# Tensor qualification context

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="purpose-and-mental-model"></a>
## Purpose and mental model

This reference documents the public model contracts that are implemented today. The mutable
`Tensor` now connects stable logical metadata to an optional borrowed host-storage association,
and `TensorFactory` provides the public construction boundary for completed descriptors, copied
primitive-array import, independent dense constants including full-value and rectangular identity
tensors, and integer ranges. `TensorRandoms` is the sole public owner of eager normal,
continuous-uniform, bounded-integral, and Bernoulli initialization from an explicit caller-owned
source. `GraphRngState` separately represents an explicit, opaque random-number-generator (RNG)
state occurrence for later graph execution; it neither exposes a numerical Tensor nor performs a
random draw. `Tensor.dropout(probability, state)` constructs one training-only inverted-dropout
occurrence and returns its public output together with the explicitly threaded next state. The
current concrete expression surface contains seven tensor-to-tensor binary arithmetic
methods, of which ADD, SUB, MUL, MIN, and MAX also accept signed integral operands; six floating
or signed-integral tensor-to-tensor comparison methods; three BOOL-only logical
methods, nineteen floating unary elementwise methods, three floating-classification methods, seven
exact-typed plus seven exact-FLOAT64 scalar arithmetic methods, and six range/one-bound clamp
methods, plus one static conditional-selection method and one explicit cast method. Fifteen
numeric aggregate methods add full, one-axis, and retained-axis `sum`, `mean`, `prod`, reduction
`min`, and reduction `max` expression construction: mean remains floating-only, while the other
four families also accept signed-integral input. Six BOOL methods provide full/single-axis
`all`/`any`; ten additional numeric and four BOOL methods add ordered distinct multi-axis forms.
Twelve floating methods add first-class log-sum-exp, corrected
variance/standard deviation, and L1/L2 norms. Six axis-only `argMin`/`argMax` methods add fixed
INT64 index results. Two one-axis `cumSum` methods and two matching `cumProd` methods add
shape-preserving numeric scan expressions, and one-axis `softmax` and `logSoftmax` add shape-
preserving floating probability-normalization expressions. Two `layerNorm` methods add population-variance
normalization over an exact trailing Shape, either without affine operands or with explicit scale
and bias. Two `rmsNorm` methods separately add uncentered root-mean-square normalization over an
exact trailing Shape, either without scale or with one explicit exact-Shape scale. One
`batchNormInference` method adds stateless per-channel inference from mandatory explicit scale,
bias, running-mean, and running-variance vectors along a layout-neutral channel axis. One
`batchNormTraining` method uses the same five Tensor roles plus exact typed momentum and epsilon
to describe current-batch normalization and an explicit running-statistic transition. It returns
the normalized output and next running statistics while its shared producer also describes two
compiler-facing saved outputs. The
parameterless `contiguous` method adds a shape-preserving request
for canonical dense row-major result geometry. Two `reshape` overloads add ordered-element-
preserving coordinate changes from either raw `long...` dimensions or an exact normalized
`Shape`. Two `expand` overloads add directional right-aligned singleton and leading-axis
repetition with locally derived zero-stride view geometry when possible. `permute(int...)` adds
arbitrary complete axis reordering, and `transpose()` adds its rank-two `[1, 0]` convenience.
Two `linear` overloads compose that transpose with MATMUL and optional exact rank-one ADD bias.
Two `conv1d` overloads compose input/weight singleton-height expansion, ordinary grouped Conv2d,
and height squeeze, with optional output-channel bias and no new operation kind.
Two `conv3d` overloads construct one first-class grouped cross-correlation occurrence in NCDHW
axis order, with optional output-channel bias and explicit rank-three-spatial geometry.
One `maxPool3d` and one `averagePool3d` receiver construct first-class NCDHW pooling metadata with
rank-specific depth, height, and width geometry. They do not evaluate or materialize a result.
`expandDims(int)` inserts one singleton axis, while `squeeze(int)` removes one selected statically
known singleton axis. `slice(long[], long[], int[], long[])` adds general signed-step directional
half-open selection. `sliceAxis(int, long, long)` supplies its one-axis step-one convenience,
`sliceAxis(int, long, long, long)` supplies an explicit signed step, and `flip(int...)` reverses
explicit axes through one slice occurrence.
`select(int, long)` fixes one scalar coordinate and removes its source axis, with locally derived
logical view geometry when the input layout is resolved and the result is non-empty.
`gather(Tensor, int)` replaces one data axis with the complete indices Shape.
`gatherElements(Tensor, int)` retains the exact same-rank indices Shape after checking every
non-selected Dimension. These are the complete public axis-index gather spellings.
`oneHot(long)` treats an INT32 or INT64 receiver as logical indices and appends one positive
static class axis. It produces storage-free BOOL metadata for one first-class `ONE_HOT`
occurrence without reading any index value.
`gatherNd(Tensor)` and `gatherNd(Tensor, int)` use coordinate tuples to index several data axes
after zero or more structurally equal shared batch Dimensions. They require a static positive
tuple depth, derive the result from the indices prefix plus the untouched data suffix, and never
inspect index values.
The two `scatterElements` overloads construct functional same-rank axis-scatter expressions from
ordered `[data, indices, updates]` inputs. They preserve the exact data Shape and type, combine
data/update gradient eligibility, and never inspect values or mutate the data input.
`scatterAdd(Tensor, Tensor, int)` constructs the distinct Gather-compatible fixed-add scatter:
the complete indices Shape replaces one data axis in the required updates Shape, while the fresh
functional result retains the exact data Shape and accumulates duplicate targets.
The three `scatterNd` overloads construct functional tuple-index scatter expressions with
replacement or an explicit reduction and with zero or an explicit shared-batch count. They apply
the Gather-ND result-Shape formula to `updates`, preserve exact data metadata in a fresh unresolved
result, and never inspect values or mutate an input.
`pad(long[], long[], ScalarValue)` adds exactly typed constant-filled positions around every axis;
its retained `double` overload is an exact-FLOAT64 convenience. `tile(long...)` repeats each
complete input pattern along every axis. Static `concat` joins an
ordered non-empty input sequence along an existing axis, static `stack` inserts a count axis for
same-shaped inputs, and instance `unstack` returns an immutable ordered list of independent scalar
select expressions. `unfold` adds general-axis window materialization and `foldAxis` restores its
overlap-summing target transformation. The two `unfold2d` forms add NCHW im2col with conceptual
positive-zero or exact typed padding, while `fold2d` adds overlap-summing col2im with exact static
or symbolic compatibility.
`HostTensorStorage` exposes current raw host-storage facts, but `Tensor` has no typed element
getters or setters. The public Engine now owns the separate compile, prepare, and run
lifecycle for Tensor expressions. Public compilation can request forward and supported
first-order gradient publication occurrences. Engine.run binds exact input identities in any order;
Engine.runValues binds fresh fully static exact-descriptor values in compiled order without
mutating Tensor prototypes. Both use the same prepared owner and an open
`RunResult` can explicitly materialize one occurrence as a detached host value.
`Engine.standard()` uses one fixed CPU-only composition; preparation and execution remain bounded
by exact CPU capability and fail closed for
unsupported operation, data-type, Shape, or layout combinations. This current lifecycle does not
move compiler, prepared-execution, runtime-residency, or backend state into Tensor. Tensor-owned
device residency, generic or mixed-backend composition, and universal execution coverage remain
absent.
The current Model contract defines one numerical meaning per family and dtype, with no graph-wide
selector. Kinds, attributes, mapping, contributors, masks, indices, state, traversal, casts,
ordering, guards, identities, divisors, stored comparisons and extrema, raw movement and
selection, and publication remain exact. Denormals-are-zero (DAZ) at named arithmetic inputs and
flush-to-zero (FTZ) at named arithmetic results are permitted for FLOAT32 and all four low types,
including low multiplication, but forbidden for FLOAT64. They do not apply to stored comparison,
extrema, cast, raw movement, selection, or the exact unary ABS/NEG/SIGN/FLOOR/CEIL/RELU sites.
Every floating scatter MUL includes the base and each addressed update exactly once but may use a
rounded tree; integral modular multiplication and raw unaddressed cells remain exact.
Qualified low arithmetic uses FLOAT32 working values and accumulators followed by one final
ties-to-even narrowing per declared low output, without hidden intermediate low narrowing.
Finite accuracy thresholds qualify backend routes only in tests; they do not define a public
envelope or authorize compiler rewrites. Composite and generated-gradient formulas inherit only
their named primitive-site permissions. Tensor construction performs no numerical evaluation.

Metal's exact domain contains the exact unary,
affine, canonicalization, indexing, BOOL-domain, Task-0059 movement, Task-0060
replacement/fold/aggregate, Task-0063 ordering/top-K/numeric arg-extrema, no-gradient promoted
INT32/INT64 MATMUL, Task-0064 maximum pooling, and Task-0065 raw INITIAL_STATE rows. Metal
also admits the documented FLOAT32 arithmetic, extrema, scalar, reduction, and scan rows;
every positive-static FLOAT32 MATMUL vector, matrix, batched, and broadcast geometry; no-gradient
BFLOAT16/FLOAT32 mixed MATMUL with FLOAT32 output; Task-0064 convolution/average pooling; Task-0065
FLOAT32 dropout; and three Task-0069 source-owned custom programs. Task 0069 admits no-gradient,
canonical rank-one FLOAT32 `L1_NORM` with one positive extent; no-gradient, axis-zero, canonical
rank-one FLOAT32 `SCATTER_ADD` with positive data and update extents, a materialized canonical
rank-one INT32 or INT64 index feed, and identical base/result and index/update extents; and
no-gradient FLOAT32 singleton `VARIANCE` over input `[1]`, axis `[0]`, correction zero, with
canonical scalar or retained `[1]` output. Compiler-generated rank-one Gather data cotangents close
through an explicit canonical zero `EXPAND -> CONTIGUOUS -> SCATTER_ADD`; gradient-bearing
Scatter-Add and its own cotangents remain unsupported. The resulting Metal capability ledger is
exactly 88 admitted kinds and 27 remaining false, with 103 structurally executable kinds and 12
remaining nonexecutable. The later bounded FLOAT32 and homogeneous BFLOAT16/FLOAT16 `EXP` routes
and sign-guarded composed FLOAT32 `SIGMOID` route do not make every dtype or occurrence of those
kinds available. Homogeneous BFLOAT16 and FLOAT16 `SIGMOID` use independently qualified custom
steps. These counts describe kinds with
admitted occurrences, not whole-kind admission; every unlisted occurrence fails closed before
route selection.

Metal uses ABI 7 with thirteen exports and one bounded schema-20 route-bearing program image.
Operation wires `1..115`, attribute wires `0..41`, and type wires `1..7` cover current structural
vocabulary. The custom-program image authenticates compact materialized slots, deterministic
pointwise units, and qualified MATMUL/Conv2d anchor epilogues. Metal-local workload,
exact-policy, candidate, compatibility, route-policy, and session codec identities are version 40;
every other identity fails closed rather than falling back. Model remains the sole semantic owner
of numerical meaning.

The authoritative module boundary remains [`ARCHITECTURE.md`](../../../ARCHITECTURE.md).


---

[Next topic](compiler-qualification.md)
