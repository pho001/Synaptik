# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0012: Fixed recurrent [scan](../../glossary.md#term-fixed-recurrent-scan) without [graph](../../glossary.md#graph) regions


## Editorial reading context

The application supplies sequence values, valid-length counts and initial states as ordinary Tensor inputs. A fixed recurrent scan describes one named transition family; the backend executes its prepared loop. “Fixed” distinguishes that family from a user-described general body, not from changing valid-length values. The persistence amendment and initial backward rejection are separate decisions.

Use the [fixed versus general scan boundaries](../../architecture/general-differentiable-scan.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-08-20

Amended — 2026-10-20: permit cold semantic [persistence](../../glossary.md#term-persistence) under the owning [Checkpoint](../../glossary.md#term-checkpoint) contract.
No executable [scan](../../glossary.md#term-fixed-recurrent-scan) or [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) is accepted by this amendment.

## Context

Synaptik's current recurrent [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) containers accept snapshotted Java `long[]` valid lengths and
construct a static [Tensor](../../glossary.md#tensor) expression for each selected length pattern. They return compact
per-step output lists. A future [API](../../glossary.md#term-application-programming-interface-api) needs valid-length values to remain ordinary [runtime](../../glossary.md#term-runtime) Tensor
inputs so one compiled and prepared computation can be reused for different [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters) without
changing [graph](../../glossary.md#graph) [topology](../../glossary.md#term-topology).

The current architecture has flat [immutable](../../glossary.md#term-immutable) [Model](../../glossary.md#term-model) expressions and compiled graphs. [Compiler](../../glossary.md#term-compiler) owns
[capture](../../glossary.md#term-capture-meanings), [inference](../../glossary.md#term-inference-meanings), [validation](../../glossary.md#term-validation), and [autograd](../../glossary.md#autograd); [Planning](../../glossary.md#term-planning) chooses [backend ownership](../../glossary.md#backend-ownership); staged [backend](../../glossary.md#backend)
preparation owns [lowering](../../glossary.md#lowering) and route choice; Runtime invokes only cold-bound prepared actions.
There is no body capture, nested graph, region [identity](../../glossary.md#term-identity-meanings), free-variable capture, loop
[intermediate representation](../../glossary.md#term-intermediate-representation-ir), or Runtime graph interpreter.

The decision must therefore fix both the recurrent semantics and their [lifecycle](../../glossary.md#lifecycle) placement before
any public API is published. Adding only an [operation](../../glossary.md#operation) kind would leave Compiler's [closed inventory](../../glossary.md#term-closed-inventory)
and backend execution undefined. Adding a general body or region abstraction would invent a much
larger architecture whose [ownership](../../glossary.md#term-ownership-meanings) and identity rules have not been designed.

## Decision drivers

- preserve the current flat [`TensorProducer`](../../glossary.md#producer--tensorproducer) and [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) meanings;
- make [runtime](../../glossary.md#term-runtime) valid lengths compatible with fully static [Shapes](../../glossary.md#shape) and reusable preparation;
- retain [Compiler](../../glossary.md#term-compiler) [ownership](../../glossary.md#term-ownership-meanings) of [capture](../../glossary.md#term-capture-meanings), [validation](../../glossary.md#term-validation), and [fail-closed](../../glossary.md#term-fail-closed) [autograd](../../glossary.md#autograd);
- keep [backend](../../glossary.md#backend)-specific loop [lowering](../../glossary.md#lowering) and execution out of shared [Prepare](../../glossary.md#prepare) and Runtime;
- validate invalid lengths without partially written published results;
- keep [recurrent state](../../glossary.md#backpropagation-through-time-and-recurrent-state), [parameters](../../glossary.md#term-parameter), and side effects explicit;
- make [graph](../../glossary.md#graph) and prepared-transition size independent of `time`; and
- preserve the current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) static APIs until migration is decided from executable evidence.

## Options considered

### General user-defined body or [graph](../../glossary.md#graph) region

[Model](../../glossary.md#term-model) could publish a [callback](../../glossary.md#term-callback), lambda, [Tensor](../../glossary.md#tensor) body, nested graph, callable block, or region and
let callers define the recurrent transition. This would require new body-input/output [identity](../../glossary.md#term-identity-meanings),
free-variable [capture](../../glossary.md#term-capture-meanings), cross-graph [ownership](../../glossary.md#term-ownership-meanings), nested [inference](../../glossary.md#term-inference-meanings), optimization, [autograd](../../glossary.md#autograd), [planning](../../glossary.md#term-planning),
preparation, [serialization](../../glossary.md#term-serialization), and [Runtime](../../glossary.md#term-runtime) contracts. None exists today. Publishing a partial region
surface would make later consumers invent incompatible semantics.

### [Host](../../glossary.md#term-host)-side unroll selected by [runtime](../../glossary.md#term-runtime) lengths

An adapter could read a Java `long[]` and construct a different [Tensor](../../glossary.md#tensor) expression for each length
pattern. This is the current static-container model, not runtime-value reuse. [Graph](../../glossary.md#graph) construction
and [compiled graph](../../glossary.md#compiled-graph--compiledgraphmodel) size can grow with `time`, and changing lengths changes [topology](../../glossary.md#term-topology).

### Dense masking around ordinary cell expressions

The [graph](../../glossary.md#graph) could unroll a cell at every time step and use [masks](../../glossary.md#term-boolean-mask) to select states and zero outputs.
That preserves dense values but still grows the graph with `time` and executes recurrent [dot
products](../../glossary.md#linear-algebra-building-blocks) and gates for invalid [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix). It therefore fails the selected graph-size and
genuine skipped-arithmetic requirements.

### Fixed first-class recurrent [operation](../../glossary.md#operation) with [backend](../../glossary.md#backend)-internal loop

[Model](../../glossary.md#term-model) defines a closed recurrent family with fixed cell equations and ordinary ordered inputs and
outputs. [Compiler](../../glossary.md#term-compiler) [captures](../../glossary.md#term-capture-meanings) one [node](../../glossary.md#node). Backend analysis lowers it once to a bounded loop plan and
finalization constructs one reusable executable. [Runtime](../../glossary.md#term-runtime) invokes the prepared bound action
without interpreting recurrence.

## Decision

Synaptik adopts the fixed first-class recurrent [operation](../../glossary.md#operation) with a [backend](../../glossary.md#backend)-internal loop.

One public call will create exactly one [identity](../../glossary.md#term-identity-meanings)-distinct multi-output [`TensorProducer`](../../glossary.md#producer--tensorproducer), and
[Compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) will emit exactly one ordinary flat `CompiledNode`. The [occurrence](../../glossary.md#term-occurrence) has no body,
region, nested [graph](../../glossary.md#graph), [callback](../../glossary.md#term-callback), captured [Tensor](../../glossary.md#tensor) beyond its ordered inputs, graph-local child
identity, free variable, or cross-graph [ownership](../../glossary.md#term-ownership-meanings) rule. A general loop, conditional, or region
system remains forbidden until another explicit architecture update.

The distinction is:

```text
declarative Model operation
  = immutable fixed transition semantics and ordinary Tensor inputs/outputs

runtime control flow
  = concrete-backend prepared loop implementing that operation
```

### Exact family and current surface

The exact variants are `RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../../glossary.md#term-long-short-term-memory-lstm). Each has one [immutable](../../glossary.md#term-immutable)
`FORWARD` or `REVERSE` direction. The fixed equations and [parameter](../../glossary.md#term-parameter) packing match the current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
cells: [tanh](../../glossary.md#term-tanh) [RNN](../../glossary.md#term-recurrent-neural-network-rnn) with separate input/hidden [weights](../../glossary.md#term-weight); reset-after [GRU](../../glossary.md#term-gated-recurrent-unit-gru) with reset, update, candidate
gate order and `candidate + update * (hidden - candidate)`; and LSTM with input, forget,
candidate, output gate order and explicit hidden/cell states. Each family has one optional packed
input-side [bias](../../glossary.md#term-bias).

[Model](../../glossary.md#term-model) task 0025E originally published exactly one direction enum, `RecurrentScanResult` with
`outputs` and `finalHidden`, `LstmRecurrentScanResult` with `outputs`, `finalHidden`, and
`finalCell`, and six provisional [Tensor](../../glossary.md#tensor) receiver methods. Task 0025F immediately corrects only
that [API](../../glossary.md#term-application-programming-interface-api) placement: the result carriers and [operation](../../glossary.md#operation) semantics remain unchanged, while public
final field-free `model.tensor.RecurrentScan` exposes exactly six static biased or bias-free
`rnn`, `gru`, and `lstm` methods with explicit [time-major](../../glossary.md#term-time-major) `input` first. This amendment preserves
the historical 0025E sequence rather than presenting the corrected placement as its original
decision. The complete exact current [signatures](../../glossary.md#term-signature) are recorded in the
[architecture contract](../../architecture/contracts/recurrent-scan.md#fixed-family-and-current-model-surface)<span data-glossary-navigation="true"> ([architecture contract glossary](../../glossary.md#architecture-contract))</span>.

The namespace is advanced low-level Model expression construction. It is not a layer, [module](../../glossary.md#term-module),
execution service, [registry](../../glossary.md#term-registry), or [general scan](../../glossary.md#general-differentiable-scan)-body abstraction, and `Tensor` has no recurrent
receiver [alias](../../glossary.md#aliasing-and-arena). Operation input order is:

```text
RNN/GRU without bias:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight]
RNN/GRU with bias:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]
LSTM without bias:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight]
LSTM with bias:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias]
```

Output order is:

```text
RNN/GRU: [outputs, finalHidden]
LSTM:    [outputs, finalHidden, finalCell]
```

Every result component is the [canonical](../../glossary.md#term-canonical-form) wrapper for its slot from the one shared [producer](../../glossary.md#term-producer).

### Static [descriptors](../../glossary.md#term-descriptor) and [runtime](../../glossary.md#term-runtime) valid lengths

The input [Shape](../../glossary.md#shape) is fully static `[time, batch, inputSize]`. Valid lengths are a fully static
[rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) `INT64[batch]` [Tensor](../../glossary.md#tensor) with `requiresGrad == false`. Initial hidden and [LSTM](../../glossary.md#term-long-short-term-memory-lstm) cell are
`[batch, hiddenSize]`. Input [weight](../../glossary.md#term-weight) is `[gateCount * hiddenSize, inputSize]`, hidden weight is
`[gateCount * hiddenSize, hiddenSize]`, and optional [bias](../../glossary.md#term-bias) is `[gateCount * hiddenSize]`, where
`gateCount` is one, three, or four for [RNN](../../glossary.md#term-recurrent-neural-network-rnn), [GRU](../../glossary.md#term-gated-recurrent-unit-gru), or LSTM. `inputSize` and `hiddenSize` are positive;
`time` and [`batch`](../../glossary.md#term-batch-machine-learning-and-hyperparameters) may be zero. All value, state, weight, and bias roles share one exact floating
[data type](../../glossary.md#data-type--datatype).

The dense output is `[time, batch, hiddenSize]`; final states are `[batch, hiddenSize]`. [Model](../../glossary.md#term-model)
construction leaves [layout](../../glossary.md#layout) unresolved. [Gradient](../../glossary.md#term-gradient) eligibility is the OR of [differentiable](../../glossary.md#derivative-and-chain-rule) floating
roles, never the valid-length role.

Only valid-length values vary across compatible executions. Dynamic or [binding](../../glossary.md#term-binding)-dependent time,
batch, [feature](../../glossary.md#term-feature), [parameter](../../glossary.md#term-parameter), or output Shapes remain outside this program. A valid-length Tensor
does not authorize a hidden dynamic-Shape [lifecycle](../../glossary.md#lifecycle).

### Traversal, [padding](../../glossary.md#padding), and failure

For each original [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) row `b`, the [backend](../../glossary.md#backend) validates `L[b]` in `[0, time]`. `FORWARD` consumes
[coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) `0 .. L[b]-1`; `REVERSE` consumes `L[b]-1 .. 0`. Reverse traversal never passes
through the padded suffix. Each valid coordinate stores the hidden state produced after consuming
that coordinate. Every padded coordinate stores exact [positive zero](../../glossary.md#nan-infinity-and-signed-zero) in the common [data type](../../glossary.md#data-type--datatype).

A zero-length row returns its exact initial hidden state semantically, plus its exact initial cell
for [LSTM](../../glossary.md#term-long-short-term-memory-lstm), and has zero output at every time. A zero-time input requires all lengths to be zero,
returns an empty dense output, and preserves the initial states semantically. Lengths are never
inferred from padding, values, [NaN](../../glossary.md#term-nan), [token](../../glossary.md#term-token) identifiers, labels, or storage.

Before any output representation is mutated, the executable backend validates the complete length
[vector](../../glossary.md#term-vector), all bounds, and representation-specific access preconditions. Failure cannot expose
partially written published results. [Engine](../../glossary.md#term-engine) later owns public exception translation.

### Purity, state, and [serialization](../../glossary.md#term-serialization)

The [operation](../../glossary.md#operation) is pure. All carried states are explicit inputs and final outputs. It owns no
hidden [module](../../glossary.md#term-module), [compiler](../../glossary.md#term-compiler), prepared, or [runtime](../../glossary.md#term-runtime) state; [RNG](../../glossary.md#graph-rng-state--graphrngstate); mode; counter; [parameter](../../glossary.md#term-parameter) or [buffer](../../glossary.md#term-buffer);
mutation; [callback](../../glossary.md#term-callback); I/O; or external resource. Each [run](../../glossary.md#term-run) uses its own [`RunState`](../../glossary.md#run-state--runstate).

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) retains parameter and persistent-buffer wrappers and state paths and supplies current parameter
[Tensor](../../glossary.md#tensor) [bindings](../../glossary.md#term-binding) as ordinary [scan](../../glossary.md#term-fixed-recurrent-scan) inputs. NN state dictionaries contain those parameter and buffer
bindings. The 2026-10-20 amendment permits cold semantic [graphs](../../glossary.md#graph) and named [host](../../glossary.md#term-host) values under the
owning [generic source](../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)
and [graph-model v3 contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model))</span>.
The fixed operation and its ordered ports are data, not an executable or serialized result-carrier
object. Prepared executables, native/[backend](../../glossary.md#backend) [artifacts](../../glossary.md#term-artifact) and Runtime state remain rebuilt, never
persisted. [Prototype](../../glossary.md#term-prototype) valid-length storage is not saved or read by cold [preflight](../../glossary.md#term-preflight); length bounds
belong to execution. This does not add a body, callback, unroll, [backend route](../../glossary.md#backend-route) or [gradient](../../glossary.md#term-gradient) rule.

Keeping the original serialization exclusion would prevent cold save/load without adding any
execution safety. Serializing prepared loops instead would cross backend and resource-[lifetime](../../glossary.md#term-lifetime)
boundaries. The selected amendment preserves the flat semantic [occurrence](../../glossary.md#term-occurrence) and [canonical](../../glossary.md#term-canonical-form) sibling
wrappers through existing Compiler reconstruction and [Checkpoint](../../glossary.md#term-checkpoint) [persistence](../../glossary.md#term-persistence). Its cost is a
closed multi-output wire and [producer](../../glossary.md#term-producer)-aware [validation](../../glossary.md#term-validation), implemented jointly with
[CP0010](../../planning/extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md)
and [CP0011](../../planning/extensions/checkpoint/tasks/0011-flat-recurrent-scan-persistence.md).
Cold roundtrip, [descriptor](../../glossary.md#term-descriptor) and [identity](../../glossary.md#term-identity-meanings) tests are required; runnable scan/[BPTT](../../glossary.md#term-backpropagation-through-time-bptt) tests are not a
prerequisite or a claim of this amendment. Existing static NN containers stay unchanged.

### Layer [ownership](../../glossary.md#term-ownership-meanings)

- Model owns fixed semantics, kind/attributes, descriptor-visible validation, result metadata,
  canonical provenance, and the focused static expression namespace.<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [descriptor glossary](../../glossary.md#term-descriptor); [validation glossary](../../glossary.md#term-validation); [metadata glossary](../../glossary.md#term-metadata); [canonical glossary](../../glossary.md#term-canonical-form); [provenance glossary](../../glossary.md#provenance))</span>
- Compiler owns one-node capture, inference, final validation, and forward inventory adoption. It
  rejects every backward-capable request reaching the family before constructing any gradient
  Tensor until BPTT is separately designed.<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler); [node glossary](../../glossary.md#node); [capture glossary](../../glossary.md#term-capture-meanings); [inference glossary](../../glossary.md#term-inference-meanings); [gradient glossary](../../glossary.md#term-gradient); [Tensor glossary](../../glossary.md#tensor); [BPTT glossary](../../glossary.md#term-backpropagation-through-time-bptt))</span>
- [Planning](../../glossary.md#term-planning) uses the existing ordinary capability query and chooses only [backend ownership](../../glossary.md#backend-ownership).
- Shared Prepare uses the existing static projection, staged analysis, exact declaration,
  assignment, and finalization contracts; it gains no loop/body contract.<span data-glossary-navigation="true"> ([Prepare glossary](../../glossary.md#prepare))</span>
- Runtime uses the existing representations, schedule, isolated state, and cold-bound invocation;
  it never inspects lengths or recurrence.<span data-glossary-navigation="true"> ([bound invocation glossary](../../glossary.md#bound-invocation--boundinvocation))</span>
- Engine owns typed logical caller-input binding and typed publication mapping and never rebuilds
  or specializes the graph from length values.<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine); [binding glossary](../../glossary.md#term-binding); [publication glossary](../../glossary.md#term-publication))</span>
- A concrete backend owns exact capability truth, one-time lowering, loop planning, complete
  length validation, physical work, and reusable executable construction.<span data-glossary-navigation="true"> ([concrete backend glossary](../../glossary.md#concrete-backend); [lowering glossary](../../glossary.md#lowering))</span>
- NN retains module composition and parameter/state bindings. A later NN/Data task owns
  runtime-valid-length convenience and schema integration without inferring lengths from values.<span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../glossary.md#term-module); [schema glossary](../../glossary.md#term-schema))</span>
- Training retains optimizer and orchestration ownership. Compiler, not Training, owns the later
  BPTT decision and gradient construction.<span data-glossary-navigation="true"> ([optimizer glossary](../../glossary.md#term-optimizer))</span>

Planning does not interpret recurrence, direction, lengths, active rows, loop [parameters](../../glossary.md#term-parameter), or
routes. [Runtime](../../glossary.md#term-runtime) does not select loop count, compact rows, or inspect an [operation](../../glossary.md#operation) or [graph](../../glossary.md#graph).

### Performance contract

One [occurrence](../../glossary.md#term-occurrence) remains one compiled [node](../../glossary.md#node) independent of `time` and lengths; [graph](../../glossary.md#graph) construction
and graph size for that occurrence are [`O(1)`](../../glossary.md#memory-management-atomic-counters-and-complexity) in `time`. [Backend](../../glossary.md#backend) analysis prepares the transition
once and never constructs or retains one node, graph, executable, or body per step. The [Runtime](../../glossary.md#term-runtime)
[hot path](../../glossary.md#term-hot-path) performs no reflection, string dispatch, scalar boxing, graph inspection, [operation](../../glossary.md#operation)
dispatch, backend [lookup](../../glossary.md#term-lookup), route selection, or per-step object-graph growth.

Invalid [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) execute no recurrent [dot products](../../glossary.md#linear-algebra-building-blocks), gates, or state update. Branch-based
row/time traversal satisfies the first capability. Physical active-row compaction, sorting,
packed [buffers](../../glossary.md#term-buffer), vectorized packed [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters), workspace reuse, and claims that [validation](../../glossary.md#term-validation) or zeroing
perform no work remain deferred backend choices requiring evidence.

## Rationale

The fixed [operation](../../glossary.md#operation) is the smallest architecture that supports [runtime](../../glossary.md#term-runtime) lengths without weakening
current [ownership](../../glossary.md#term-ownership-meanings). [Model](../../glossary.md#term-model) expresses one [immutable](../../glossary.md#term-immutable) meaning, [Compiler](../../glossary.md#term-compiler) and [Planning](../../glossary.md#term-planning) retain ordinary
flat behavior, a [concrete backend](../../glossary.md#concrete-backend) owns the only component that needs a loop, and Runtime stays a
[graph](../../glossary.md#graph)-free schedule executor. It also permits genuine skipped invalid-[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) arithmetic,
which dense masked [unrolling](../../glossary.md#term-unrolling) cannot provide.

The closed family deliberately trades generality for an executable contract. A future general
region system can still be designed from concrete needs, but it must not inherit accidental body,
[capture](../../glossary.md#term-capture-meanings), or [lifecycle](../../glossary.md#lifecycle) rules from this operation.

## Consequences

### Positive

- [Runtime](../../glossary.md#term-runtime) length values can vary without [graph](../../glossary.md#graph) reconstruction or preparation.
- Graph and prepared-transition size do not grow with `time`.
- The existing flat graph, staged [Prepare](../../glossary.md#prepare), and cold-[binding](../../glossary.md#term-binding) boundaries remain intact.
- [Recurrent state](../../glossary.md#backpropagation-through-time-and-recurrent-state) and side effects remain explicit and compatible with isolated runs.
- [Backend capability](../../glossary.md#backend-capability) and skipped-work claims can remain exact and evidence-based.

### Negative and risks

- The first family is intentionally narrow and cannot express an arbitrary recurrent cell.
- Each [backend](../../glossary.md#backend) must implement and validate each claimed variant/type/[Shape](../../glossary.md#shape)/direction combination.
- Dense outputs require padded zero storage even though padded recurrent arithmetic is skipped.
- BPTT is unavailable until a later Compiler decision selects saved-state versus recomputation
  and derivative semantics.<span data-glossary-navigation="true"> ([BPTT glossary](../../glossary.md#term-backpropagation-through-time-bptt); [Compiler glossary](../../glossary.md#term-compiler); [derivative glossary](../../glossary.md#term-derivative))</span>
- Fully dynamic Shapes and physical active-row compaction remain unsupported.

### Compatibility, migration, testing, and follow-up

Current `RnnSequence`, `GruSequence`, `LstmSequence`, and their bidirectional containers keep
their Java `long[]`, static-unroll, compact-output-list, [provenance](../../glossary.md#provenance), and final-state contracts.
They are not redirected to the new [operation](../../glossary.md#operation). [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0022 later decides overload, migration,
retention, or deprecation and must account for the new dense zero-padded original-time-aligned
output versus current compact per-step lists. Any bidirectional migration preserves independent
[parameters](../../glossary.md#term-parameter), valid-prefix-only reverse traversal, forward-first [feature](../../glossary.md#term-feature) concatenation, original-
time [alignment](../../glossary.md#term-alignment), and typed final states.

[Model](../../glossary.md#term-model) 0025E owns the semantic family and its original provisional receiver surface; Model 0025F
corrects that surface to `RecurrentScan.rnn/gru/lstm` without changing the semantics. [Compiler](../../glossary.md#term-compiler)
0006A owns forward adoption and the initial [fail-closed](../../glossary.md#term-fail-closed) [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) boundary. A later concrete-[backend](../../glossary.md#backend)
task owns truthful execution.
[Engine](../../glossary.md#term-engine) 0001–0002 own [lifecycle](../../glossary.md#lifecycle) composition and typed caller/output [binding](../../glossary.md#term-binding). NN 0021B coordinates
the executable [owner](../../glossary.md#term-owner) sequence; NN 0022 owns Data-facing [runtime](../../glossary.md#term-runtime)-length [API](../../glossary.md#term-application-programming-interface-api) and compatibility.
Dynamic [Shapes](../../glossary.md#shape), arbitrary [masks](../../glossary.md#term-boolean-mask) with holes, active-row compaction, BPTT, and general regions stay
separate.

The placement correction changes no [module](../../glossary.md#term-module) [dependency](../../glossary.md#term-dependency). Existing architecture tests must pass but
require no source update. This [ADR](../../glossary.md#term-architecture-decision-record-adr) documents the public Model construction boundary; it introduces
no Compiler, backend, Runtime, or execution behavior.

## Related documentation

- [Architecture contract](../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../glossary.md#architecture-contract))</span>
- [Lifecycle](../../architecture/lifecycle.md#planned-fixed-recurrent-scan-through-the-lifecycle)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Module boundaries](../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>
- [Runtime, Prepare, and Backend boundary](../../architecture/runtime-prepare-backend-boundary.md#planned-fixed-recurrent-scan-handoff)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Backend glossary](../../glossary.md#backend))</span>
- [Training graph](../../architecture/training-graph.md#fixed-recurrent-scan-and-the-initial-bptt-boundary)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [ADR 0002: Backend-owned lowering](0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [Backend-owned lowering glossary](../../glossary.md#backend-owned-lowering))</span>
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler); [pre-capture Tensor-expression autograd glossary](../../glossary.md#pre-capture-tensor-expression-autograd))</span>
- [ADR 0010: Staged backend preparation](0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership](0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../../glossary.md#term-run); [ownership glossary](../../glossary.md#term-ownership))</span>
- [NN task 0021A](../../planning/extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)<span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Model task 0025F](../../planning/modules/model/tasks/0025f-recurrent-scan-expression-namespace-correction.md)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model))</span>
