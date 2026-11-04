# Diagnose a failed computation

## Outcome

Locate the failed boundary before changing a backend, suppressing an exception or enabling traces.
The [learning project](../examples/README.md) includes an `invalid-shape` mode: it deliberately
supplies the wrong static signature and checks rejection rather than silently padding the data.

A **signature** is the complete expected type and geometry of an input, not just its value count.
For the learning input, `[2,1]` means two sample rows with one feature each; `[1,2]` changes those
roles. A **boundary** below is a stage where one kind of responsibility is checked before the
next stage can proceed. **Admission** means accepting work after those checks.

## Read the failure in lifecycle order

| Boundary | Question | Useful evidence |
|---|---|---|
| Tensor/model construction | Do dtype, axes, contraction and declared attributes agree? | Concrete descriptors and the focused [Tensor reference](../reference/tensor/index.md) |
| Compile/AD | Is the graph valid and is this requested derivative supported? | Requested objective/targets, compile failure scope and [Compiler reference](../reference/compiler/index.md) |
| Capability/prepare | Can the exact owner realize the occurrence and assign resources? | Support matrix and optional cold preparation events |
| Input binding/run | Do fresh values match the prepared signature and remain live? | Actual descriptors, storage spans and caller ownership |
| Publication/materialization | Is the result lease open and is the byte ceiling sufficient? | Publication metadata and [lifetime explanation](../explanation/ownership.md) |
| Training update | Are gradients/updates finite and live parameters unchanged? | Training failure and unchanged-state guarantees, not a retried partial update |
| Persistence | Is source/value/wire admission satisfied independently of execution? | Checkpoint limits, source scope and the original I/O/validation cause |

**AD** means automatic differentiation: constructing derivative calculations from expressions.
A **contraction** pairs an axis from each operand and sums products along it, as in matrix
multiplication. A **storage span** is the physical byte range referenced by a layout, including
offsets and gaps. A **lease** keeps resources available for a bounded lifetime; materialization
copies a publication's values out of that lifetime. **Finite** means neither infinity nor NaN
(“not a number,” an invalid or undefined numerical result). These distinctions help separate
bad geometry, unavailable resources and invalid numerical values.

A wrong descriptor should be corrected at the producer or call boundary. Changing `requiresGrad`,
adding a cast or changing loss reduction alters the program; it is not a neutral way to hide a
rejection. Do not close an Arena early just because the synchronous kernel finished while a
low-level result still borrows its inputs.

## Observe selectively

Use [diagnostic observers](../developer-guide/debugging-trace.md) when the exception alone cannot
answer a concrete question. CPU emits cold selection/finalization facts; Engine reports configured
tuning fallback/rejection; Metal can report the first failing Scan coordinate and native cause.
The default no-observer path does not create these diagnostic events or clocks.

Correlation comes from typed IDs, not adjacent list positions during concurrent preparation.
Bounded lists have truncation flags. An absent event is not proof that nothing happened; a failure
can precede the observation boundary. A diagnostic callback is not a place to recursively close
the active Engine or mutate live model state.

## Check numeric expectations separately

For the learning project, the returned first training objective is the pre-update squared-error
sum. Recompute after the update to evaluate new predictions. Exact teaching numbers do not imply
bit-identical arbitrary FLOAT32 reductions across all supported backends. For low types, logical
storage and FLOAT32 working arithmetic are different facts; use the Model semantic contract and
current qualified operation domain, not a guessed accumulator policy.

FLOAT32 rounds intermediate results to representable values, so changing the order in which
many numbers are summed can change final bits. An **accumulator** holds a running sum; its
working precision need not be the storage precision of each input. This does not change the
learning example's exact expected `40`, `-40` and `2`.

Useful variation: change only the learning input shape while retaining its two numeric values.
Observe the documented descriptor rejection, then restore `[2,1]` and rerun. This isolates geometry
from arithmetic and avoids diagnosing a shape problem as a native kernel failure.
