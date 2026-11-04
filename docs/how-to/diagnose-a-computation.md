# Diagnose a failed computation

## Outcome

Locate the failed boundary before changing a [backend](../glossary.md#backend), suppressing an exception or enabling traces.
The [learning project](../examples/README.md) includes an `invalid-shape` mode: it deliberately
supplies the wrong static [signature](../glossary.md#term-signature) and checks rejection rather than silently [padding](../glossary.md#padding) the data.

A **signature** is the complete expected type and geometry of an input, not just its value count.
For the learning input, `[2,1]` means two [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) rows with one [feature](../glossary.md#term-feature) each; `[1,2]` changes those
roles. A **boundary** below is a stage where one kind of responsibility is checked before the
next stage can proceed. **[Admission](../glossary.md#term-admission)** means accepting work after those checks.

## Read the failure in [lifecycle](../glossary.md#lifecycle) order

| Boundary | Question | Useful evidence |
|---|---|---|
| [Tensor](../glossary.md#tensor)/[model](../glossary.md#term-model) construction | Do [dtype](../glossary.md#data-type--datatype), [axes](../glossary.md#term-axis), [contraction](../glossary.md#contraction-dimension) and declared attributes agree? | Concrete [descriptors](../glossary.md#term-descriptor) and the focused [Tensor reference](../reference/tensor/index.md) |
| [Compile](../glossary.md#compile)/[AD](../glossary.md#autograd) | Is the [graph](../glossary.md#graph) valid and is this requested [derivative](../glossary.md#term-derivative) supported? | Requested [objective](../glossary.md#term-objective)/[targets](../glossary.md#term-target), compile failure [scope](../glossary.md#term-scope) and [Compiler reference](../reference/compiler/index.md)<span data-glossary-navigation="true"> ([Compiler glossary](../glossary.md#term-compiler))</span> |
| Capability/[prepare](../glossary.md#prepare) | Can the exact [owner](../glossary.md#term-owner) realize the [occurrence](../glossary.md#term-occurrence) and assign resources? | Support [matrix](../glossary.md#term-review-matrix) and optional cold preparation events |
| Input [binding](../glossary.md#term-binding)/[run](../glossary.md#term-run) | Do fresh values match the prepared [signature](../glossary.md#term-signature) and remain live? | Actual descriptors, storage spans and caller [ownership](../glossary.md#term-ownership-meanings) |
| [Publication](../glossary.md#term-publication)/[materialization](../glossary.md#materialization) | Is the result [lease](../glossary.md#term-lease) open and is the byte ceiling sufficient? | Publication [metadata](../glossary.md#term-metadata) and [lifetime explanation](../explanation/ownership.md) |
| Training update | Are [gradients](../glossary.md#term-gradient)/updates [finite](../glossary.md#term-finite) and live [parameters](../glossary.md#term-parameter) unchanged? | Training failure and unchanged-state guarantees, not a retried partial update |
| [Persistence](../glossary.md#term-persistence) | Is source/value/wire [admission](../glossary.md#term-admission) satisfied independently of execution? | [Checkpoint](../glossary.md#term-checkpoint) limits, source scope and the original I/O/[validation](../glossary.md#term-validation) cause |

**AD** means [automatic differentiation](../glossary.md#autograd): constructing derivative calculations from expressions.
A **contraction** pairs an axis from each operand and sums products along it, as in [matrix
multiplication](../glossary.md#matrix-multiplication--matmul). A **storage span** is the physical byte range referenced by a [layout](../glossary.md#layout), including
offsets and gaps. A **lease** keeps resources available for a bounded [lifetime](../glossary.md#term-lifetime); materialization
copies a publication's values out of that lifetime. **Finite** means neither [infinity](../glossary.md#term-infinity) nor [NaN](../glossary.md#term-nan)
(“[not a number](../glossary.md#nan-infinity-and-signed-zero),” an invalid or undefined numerical result). These distinctions help separate
bad geometry, unavailable resources and invalid numerical values.

A wrong descriptor should be corrected at the [producer](../glossary.md#term-producer) or call boundary. Changing `requiresGrad`,
adding a cast or changing [loss reduction](../glossary.md#loss-reduction) alters the program; it is not a neutral way to hide a
rejection. Do not close an [Arena](../glossary.md#term-arena) early just because the [synchronous](../glossary.md#term-synchronous) [kernel](../glossary.md#kernel) finished while a
low-level result still borrows its inputs.

## Observe selectively

Use [diagnostic observers](../developer-guide/debugging-trace.md) when the exception alone cannot
answer a concrete question. [CPU](../glossary.md#term-cpu) emits cold selection/finalization facts; [Engine](../glossary.md#term-engine) reports configured
tuning fallback/rejection; [Metal](../glossary.md#term-metal) can report the first failing [Scan](../glossary.md#term-cumulative-scan) [coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix) and native cause.
The default no-[observer](../glossary.md#term-observer) path does not create these diagnostic events or clocks.

Correlation comes from typed IDs, not adjacent list positions during concurrent preparation.
Bounded lists have [truncation](../glossary.md#term-truncation-toward-zero) flags. An absent event is not proof that nothing happened; a failure
can precede the observation boundary. A diagnostic [callback](../glossary.md#term-callback) is not a place to recursively close
the active Engine or mutate live [model](../glossary.md#term-model) state.

## Check numeric expectations separately

For the learning project, the returned first training [objective](../glossary.md#term-objective) is the pre-update squared-error
sum. Recompute after the update to evaluate new [predictions](../glossary.md#term-prediction). Exact teaching numbers do not imply
bit-identical arbitrary [FLOAT32](../glossary.md#term-float32) reductions across all supported [backends](../glossary.md#backend). For low types, logical
storage and FLOAT32 [working arithmetic](../glossary.md#accumulator-type) are different facts; use the [Model](../glossary.md#term-model) semantic contract and
current qualified [operation](../glossary.md#operation) domain, not a guessed [accumulator](../glossary.md#accumulator-type) policy.

FLOAT32 rounds intermediate results to representable values, so changing the order in which
many numbers are summed can change final bits. An **accumulator** holds a running sum; its
working [precision](../glossary.md#term-precision) need not be the storage precision of each input. This does not change the
learning example's exact expected `40`, `-40` and `2`.

Useful variation: change only the learning input [shape](../glossary.md#shape) while retaining its two numeric values.
Observe the documented [descriptor](../glossary.md#term-descriptor) rejection, then restore `[2,1]` and rerun. This isolates geometry
from arithmetic and avoids diagnosing a shape problem as a native [kernel](../glossary.md#kernel) failure.
