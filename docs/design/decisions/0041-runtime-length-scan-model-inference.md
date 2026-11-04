# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0041 [Runtime](../../glossary.md#term-runtime) length [scan](../../glossary.md#term-cumulative-scan) [Model](../../glossary.md#term-model) [inference](../../glossary.md#term-inference)


## Editorial reading context

The application binds values and valid-length Tensors in declared positional order. A final hidden state is the state after each row's last valid transition, not necessarily the last dense time cell. The inference-only guard below belonged to the fixed-scan workflow; general trainable RNN composition is a separate, later decision.

Use the [later general-scan RNN direction](0058-trainable-runtime-length-rnn-composition.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted decision — 2026-10-20. The user approved the exact [API](../../glossary.md#term-application-programming-interface-api) and [inference](../../glossary.md#term-inference)-only [Model](../../glossary.md#term-model) cutover.

## Context

NN0022 supplies [runtime](../../glossary.md#term-runtime)-length recurrent expressions through existing [RNN](../../glossary.md#term-recurrent-neural-network-rnn), [GRU](../../glossary.md#term-gated-recurrent-unit-gru) and [LSTM](../../glossary.md#term-long-short-term-memory-lstm) sequences.
CPU0028 executes their [canonical](../../glossary.md#term-canonical-form) [FLOAT32](../../glossary.md#term-float32) forward occurrences through public [Engine](../../glossary.md#term-engine) preparations.
[Graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) v3 already copies, saves and restores flat scans and ordered inputs. The remaining
workflow gap is constructing and computing an ordinary [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) without manual list indexing.
The prior owned-Model contract explicitly withheld [scan](../../glossary.md#term-cumulative-scan) compute and training support.

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) already owns private model state and high-level exclusion. Engine owns [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare) and
[run](../../glossary.md#term-run); Training owns [optimizer](../../glossary.md#term-optimizer) sessions; [Compiler](../../glossary.md#term-compiler) rejects scan [backpropagation through time](../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../glossary.md#backpropagation-through-time-and-recurrent-state)).
Existing configure setup reaches that rejection only after [binding](../../glossary.md#term-binding) freeze, [callback](../../glossary.md#term-callback) invocation
and native [parameter](../../glossary.md#term-parameter) [admission](../../glossary.md#term-admission). An [inference](../../glossary.md#term-inference)-only Model must reject earlier to preserve its state
and inference [cache](../../glossary.md#term-cache).

## Decision drivers

Reuse existing [owners](../../glossary.md#term-owner) and ordinary [Tensor](../../glossary.md#tensor) [provenance](../../glossary.md#provenance), preserve static recurrence and training,
keep two declared inputs truthful in the generic type, and reject unsupported training without
setup side effects. Avoid Data prerequisites, hidden recurrent [carry](../../glossary.md#term-carry) and [persistence](../../glossary.md#term-persistence) changes.

## Options considered

Keeping direct [Engine](../../glossary.md#term-engine)/list orchestration alone preserves the status quo but leaves the [owned
Model](../../glossary.md#model-module-buffer-and-scan-distinctions) workflow unavailable. Replacing static builder recurrence with scans would change existing
semantics and training. A new runner or [model](../../glossary.md#term-model) hierarchy would duplicate [lifecycle](../../glossary.md#lifecycle) [ownership](../../glossary.md#term-ownership-meanings).
Explicit [scan](../../glossary.md#term-cumulative-scan) recipes on the existing builder provide the bounded [inference](../../glossary.md#term-inference) capability without
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

[Scan](../../glossary.md#term-cumulative-scan) recipes use existing biased FORWARD Sequence.forwardScan, positive hidden widths, existing
[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), independent [parameters](../../glossary.md#term-parameter) and zero initial hidden/cell states. The first scan establishes
positive static time/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[features](../../glossary.md#term-feature) ([TBF](../../glossary.md#term-tbf)) geometry and one shared non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype) `[B]` length
input. Later scans preserve T/B. The terminal declares `[values, validLengths]` and one result;
existing build rejects scan recipes, and buildWithLengths rejects chains without scans before
acquisition effects. finalHidden selects only the immediately preceding scan's existing port;
explicit state, finalCell, reverse and [bias](../../glossary.md#term-bias)-free choices remain advanced [graph](../../glossary.md#graph)/list workflows.

Ordinary layers keep their existing dense semantics and [shape](../../glossary.md#shape) guards. They do not inherit a
global [mask](../../glossary.md#term-boolean-mask); bias and [sigmoid](../../glossary.md#term-sigmoid) may change padded zeros. Static recurrence and lastTimeStep retain
their existing meaning. For lengths `[4,2,0]`, a scan's finalHidden returns states after four,
two and zero transitions; selecting dense time index 3 is not equivalent. A classifier head after
finalHidden may produce nonzero scores for the zero-length row.

The compute overload applies to any owned two-input/one-output [Model](../../glossary.md#term-model) in declared positional order,
including loaded graphs, without role [inference](../../glossary.md#term-inference). It uses one [admission](../../glossary.md#term-admission), checks [arity](../../glossary.md#term-arity) before
[arguments](../../glossary.md#term-argument), and delegates internally to the existing computeOpen with Long.MAX_VALUE. Optional
cold forward is not required before compute. Builder cold composition retains instantiated owned
steps through existing adapters, Sequential [ownership](../../glossary.md#term-ownership-meanings) and ModelForward, not recipes or callbacks.

[Cache](../../glossary.md#term-cache) scan presence from ordinary acquired [provenance](../../glossary.md#provenance), including heads and final-state-only roots.
Both configure/train overload families reject with UnsupportedOperationException after existing
[lifecycle](../../glossary.md#lifecycle)/live-source admission but before [binding](../../glossary.md#term-binding) freeze, [objective](../../glossary.md#term-objective) callbacks, [Engine](../../glossary.md#term-engine) creation,
inference-cache closure or native copy. Preserve existing freeze status, stores and bytes; never
clear gradient eligibility or depend on late [Compiler](../../glossary.md#term-compiler) rejection.

## Rationale

Explicit recipe names preserve the distinction between static [unrolling](../../glossary.md#term-unrolling) and [runtime](../../glossary.md#term-runtime) length values.
The list-boundary [Model](../../glossary.md#term-model) represents both inputs honestly, while the convenience removes indexing
without inventing [scan](../../glossary.md#term-cumulative-scan)-specific execution. Existing source and execution [owners](../../glossary.md#term-owner) already separate
cold composition from cached [inference](../../glossary.md#term-inference). Early training rejection prevents an unsupported request
from disrupting usable inference state.

## Consequences

Construction, save/loadGraph and compute can share the existing private working state and [Engine](../../glossary.md#term-engine)
path. Static builders and supported static training remain unchanged. No new production type,
including nested types, [dependency edge](../../glossary.md#term-dependency-edge), [graph](../../glossary.md#graph) representation or execution [owner](../../glossary.md#term-owner) is authorized.
If existing adapters cannot express the composition, implementation must stop and replan.

The builder intentionally offers neither explicit [carry](../../glossary.md#term-carry) nor finalCell output; callers needing
those use advanced boundaries. Positive builder extents remain narrower than the [scan](../../glossary.md#term-cumulative-scan) [operation](../../glossary.md#operation)'s
zero T/B allowance. Execution [qualification](../../glossary.md#term-qualification) stays [canonical](../../glossary.md#term-canonical-form) [FLOAT32](../../glossary.md#term-float32) [CPU](../../glossary.md#term-cpu), independently of broader
valid floating expressions. No [backend](../../glossary.md#backend) numerics, [BPTT](../../glossary.md#term-backpropagation-through-time-bptt), Data, native [ABI](../../glossary.md#term-application-binary-interface-abi) or [persistence](../../glossary.md#term-persistence) format changes.
Existing v3 preserves sibling ports and declared input positions, not user-input role [metadata](../../glossary.md#term-metadata).

NN0022B must validate mixed stacks and [padding](../../glossary.md#padding), generic two-input [arity](../../glossary.md#term-arity)/[lifecycle](../../glossary.md#lifecycle) precedence,
save-before-compute and load round trips, early rejection without setup effects, and unchanged
static training. Independent code/documentation review and the coordinated contract-boundary
test/[Javadoc](../../glossary.md#term-javadoc) [checkpoint](../../glossary.md#term-checkpoint) remain implementation gates, not completed by this policy decision.

## Related documentation

- [Root lifecycle](../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Owning NN contract](../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Fixed scan semantics](../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs)<span data-glossary-navigation="true"> ([scan glossary](../../glossary.md#term-cumulative-scan))</span>
- [Graph-model v3](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)<span data-glossary-navigation="true"> ([Graph glossary](../../glossary.md#graph); [model glossary](../../glossary.md#term-model))</span>
- [Owned composition explanation](../../architecture/module-boundaries.md#owned-tensor-model-composition)
- [ADR 0039 graph persistence](0039-portable-graph-model-persistence.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [persistence glossary](../../glossary.md#term-persistence))</span>
- [ADR 0040 fluent builder](0040-fluent-model-builder.md)
- [NN0022B implementation and validation](../../planning/extensions/nn/tasks/0022b-runtime-length-scan-model-workflow.md)<span data-glossary-navigation="true"> ([validation glossary](../../glossary.md#term-validation))</span>
