# ADR 0041 Runtime length scan Model inference

## Status

Accepted decision — 2026-10-20. The user approved the exact API and inference-only Model cutover.

## Context

NN0022 supplies runtime-length recurrent expressions through existing RNN, GRU and LSTM sequences.
CPU0028 executes their canonical FLOAT32 forward occurrences through public Engine preparations.
Graph-model v3 already copies, saves and restores flat scans and ordered inputs. The remaining
workflow gap is constructing and computing an ordinary owned Model without manual list indexing.
The prior owned-Model contract explicitly withheld scan compute and training support.

NN already owns private model state and high-level exclusion. Engine owns compile, prepare and
run; Training owns optimizer sessions; Compiler rejects scan backpropagation through time (BPTT).
Existing configure setup reaches that rejection only after binding freeze, callback invocation
and native parameter admission. An inference-only Model must reject earlier to preserve its state
and inference cache.

## Decision drivers

Reuse existing owners and ordinary Tensor provenance, preserve static recurrence and training,
keep two declared inputs truthful in the generic type, and reject unsupported training without
setup side effects. Avoid Data prerequisites, hidden recurrent carry and persistence changes.

## Options considered

Keeping direct Engine/list orchestration alone preserves the status quo but leaves the owned
Model workflow unavailable. Replacing static builder recurrence with scans would change existing
semantics and training. A new runner or model hierarchy would duplicate lifecycle ownership.
Explicit scan recipes on the existing builder provide the bounded inference capability without
those migrations.

## Decision

Add exactly these methods to the existing public types:

```java
// ModelBuilder
ModelBuilder rnnScan(long hiddenSize);
ModelBuilder gruScan(long hiddenSize);
ModelBuilder lstmScan(long hiddenSize);
ModelBuilder finalHidden();
Model<List<Tensor>, List<Tensor>> buildWithLengths();
// Model
HostTensorValue compute(Tensor firstInput, Tensor secondInput);
```

Scan recipes use existing biased FORWARD Sequence.forwardScan, positive hidden widths, existing
seeds, independent parameters and zero initial hidden/cell states. The first scan establishes
positive static time/batch/features (TBF) geometry and one shared non-gradient INT64 `[B]` length
input. Later scans preserve T/B. The terminal declares `[values, validLengths]` and one result;
existing build rejects scan recipes, and buildWithLengths rejects chains without scans before
acquisition effects. finalHidden selects only the immediately preceding scan's existing port;
explicit state, finalCell, reverse and bias-free choices remain advanced graph/list workflows.

Ordinary layers keep their existing dense semantics and shape guards. They do not inherit a
global mask; bias and sigmoid may change padded zeros. Static recurrence and lastTimeStep retain
their existing meaning. For lengths `[4,2,0]`, a scan's finalHidden returns states after four,
two and zero transitions; selecting dense time index 3 is not equivalent. A classifier head after
finalHidden may produce nonzero scores for the zero-length row.

The compute overload applies to any owned two-input/one-output Model in declared positional order,
including loaded graphs, without role inference. It uses one admission, checks arity before
arguments, and delegates internally to the existing computeOpen with Long.MAX_VALUE. Optional
cold forward is not required before compute. Builder cold composition retains instantiated owned
steps through existing adapters, Sequential ownership and ModelForward, not recipes or callbacks.

Cache scan presence from ordinary acquired provenance, including heads and final-state-only roots.
Both configure/train overload families reject with UnsupportedOperationException after existing
lifecycle/live-source admission but before binding freeze, objective callbacks, Engine creation,
inference-cache closure or native copy. Preserve existing freeze status, stores and bytes; never
clear gradient eligibility or depend on late Compiler rejection.

## Rationale

Explicit recipe names preserve the distinction between static unrolling and runtime length values.
The list-boundary Model represents both inputs honestly, while the convenience removes indexing
without inventing scan-specific execution. Existing source and execution owners already separate
cold composition from cached inference. Early training rejection prevents an unsupported request
from disrupting usable inference state.

## Consequences

Construction, save/loadGraph and compute can share the existing private working state and Engine
path. Static builders and supported static training remain unchanged. No new production type,
including nested types, dependency edge, graph representation or execution owner is authorized.
If existing adapters cannot express the composition, implementation must stop and replan.

The builder intentionally offers neither explicit carry nor finalCell output; callers needing
those use advanced boundaries. Positive builder extents remain narrower than the scan operation's
zero T/B allowance. Execution qualification stays canonical FLOAT32 CPU, independently of broader
valid floating expressions. No backend numerics, BPTT, Data, native ABI or persistence format changes.
Existing v3 preserves sibling ports and declared input positions, not user-input role metadata.

NN0022B must validate mixed stacks and padding, generic two-input arity/lifecycle precedence,
save-before-compute and load round trips, early rejection without setup effects, and unchanged
static training. Independent code/documentation review and the coordinated contract-boundary
test/Javadoc checkpoint remain implementation gates, not completed by this policy decision.

## Related documentation

- [Root lifecycle](../../../ARCHITECTURE.md#core-lifecycle)
- [Owning NN contract](../../architecture/contracts/extensions-training.md#owned-tensor-models)
- [Fixed scan semantics](../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs)
- [Graph-model v3](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)
- [Owned composition explanation](../../architecture/module-boundaries.md#owned-tensor-model-composition)
- [ADR 0039 graph persistence](0039-portable-graph-model-persistence.md)
- [ADR 0040 fluent builder](0040-fluent-model-builder.md)
- [NN0022B implementation and validation](../../planning/extensions/nn/tasks/0022b-runtime-length-scan-model-workflow.md)
