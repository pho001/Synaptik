# Fixed and general [differentiable](../../glossary.md#derivative-and-chain-rule)-scan contract

<a id="fixed-recurrent-scan-contract"></a>

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.<span data-glossary-navigation="true"> ([normative glossary](../../glossary.md#term-normative); [scope glossary](../../glossary.md#term-scope); [dependency glossary](../../glossary.md#term-dependency))</span>

## [Scope](../../glossary.md#term-scope)

This contract is the sole complete semantic [owner](../../glossary.md#term-owner) for fixed recurrent scans and the bounded [general
differentiable scan](../../glossary.md#general-differentiable-scan): [Model](../../glossary.md#term-model) surface, formals/results/residuals, lexical scopes, static-[Shape](../../glossary.md#shape)/[runtime](../../glossary.md#term-runtime)-
value boundary, traversal, empties, purity, failure, performance and migration invariants. [Compiler](../../glossary.md#term-compiler),
[Planning](../../glossary.md#term-planning), [Prepare](../../glossary.md#prepare)/Runtime, [backend](../../glossary.md#backend) and extension contracts own only their respective boundaries;
they link here rather than independently redefining scan semantics. General [graph](../../glossary.md#graph) regions and
arbitrary control flow remain forbidden. [General scan](../../glossary.md#general-differentiable-scan) has implemented Model source/[IR](../../glossary.md#term-intermediate-representation-ir), Planning
query transport, Compiler scoped [capture](../../glossary.md#term-capture-meanings)/first-order [AD](../../glossary.md#autograd)/demand and Prepare scoped projection/static
[binding](../../glossary.md#term-binding)/demand [validation](../../glossary.md#term-validation). CPU0030 implements bounded [FLOAT32](../../glossary.md#term-float32) same-shape [pointwise](../../glossary.md#term-pointwise) execution and
its independently qualified actual Compiler-generated reverse sibling. CPU0031 adds qualified
[homogeneous](../../glossary.md#term-homogeneous) FLOAT32 broadcast, SUM/sumToShape and bounded shape/[layout](../../glossary.md#layout) closure, including actual
reverse bodies. CPU0032 adds static homogeneous FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul) inside the emitted scan,
including independently qualified actual reverse bodies. CPU0033 implements homogeneous [FLOAT64](../../glossary.md#term-float64)
across the complete bounded declared algebra and actual first-order reverse closure. The bounded
owned [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) workflow is implemented under [its owner](extensions-training.md#owned-tensor-models);
[padding](../../glossary.md#padding) and low-type execution remain excluded. Implemented bounded
[body persistence](extensions-training.md#canonical-graph-model-version-4)<span data-glossary-navigation="true"> ([persistence glossary](../../glossary.md#term-persistence))</span>
changes no Scan semantics here. The
existing fixed family and executable routes remain unchanged.
The [accepted symbolic-batch extension](#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span> has implemented Model-only
static/bounded-symbolic source under Model0042, independently reviewed and MAIN accepted 2026-10-25.
Model0041's coordinated authority is Complete. Compiler0014's [inference](../../glossary.md#term-inference-meanings)/actual first-order AD/
all-port binding is independently reviewed and MAIN accepted 2026-10-25 after compatibility validation.
Prepare0014 implements bounded FLOAT32 symbolic projection/execution; its task tracks final
[qualification](../../glossary.md#term-qualification). Cold/copy/wire still requires static Scan shapes, and [Engine](../../glossary.md#term-engine) [memoization](../../glossary.md#term-memoization) remains future.

## Fixed recurrent scan without [graph](../../glossary.md#graph) regions

The first recurrent [operation](../../glossary.md#operation) whose valid sequence lengths can change between runs is one fixed,
first-class [Model](../../glossary.md#term-model) operation family. It is an ordinary flat, [identity](../../glossary.md#term-identity-meanings)-distinct, multi-output
[`TensorProducer`](../../glossary.md#producer--tensorproducer) [occurrence](../../glossary.md#term-occurrence) before [capture](../../glossary.md#term-capture-meanings) and exactly one ordinary flat `CompiledNode` after
capture. This fixed-family decision does not authorize a user-defined [callback](../../glossary.md#term-callback), lambda, [Tensor](../../glossary.md#tensor) body,
nested [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel), callable graph, block, region, loop [intermediate representation](../../glossary.md#term-intermediate-representation-ir), captured
subgraph, or other body value.

```text
declarative Model operation
  = immutable fixed transition meaning plus ordered Tensor inputs and outputs

backend-internal runtime control flow
  = one prepared bounded loop implementing that meaning
```

The operation captures no Tensor reference beyond its ordered ordinary inputs and owns no body
input or output, free variable, nested graph-local identity, region identity, cross-graph
[ownership](../../glossary.md#term-ownership-meanings) rule, or loop condition. `Tensor`, `TensorProducer`, `Operation`, `CompiledNode`, and
`CompiledGraphModel` retain their flat meanings for fixed scans. The bounded exception in
[General differentiable scan](#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../glossary.md#general-differentiable-scan))</span> does not turn fixed scans into bodies;
a general control-flow or graph-region system remains forbidden.

<a id="fixed-family-and-planned-model-surface"></a>

### Fixed family and current [Model](../../glossary.md#term-model) surface

The exact semantic variants are:

```text
RNN_TANH
GRU_RESET_AFTER
LSTM
```

Each [occurrence](../../glossary.md#term-occurrence) has exactly one [immutable](../../glossary.md#term-immutable) `FORWARD` or `REVERSE` direction attribute. The first
family has no bidirectional, stacked, arbitrary-cell, configurable-gate, configurable-[activation](../../glossary.md#term-activation),
peephole, projection, recurrent-[dropout](../../glossary.md#dropout--inverted-dropout), [residual](../../glossary.md#scan-residual-port-and-history-demand), stateful, sparse, or quantized variant.

The fixed transitions match the current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) cells. `RNN_TANH` uses one [tanh](../../glossary.md#term-tanh) transition, separate
input and hidden [weights](../../glossary.md#term-weight), and one optional shared input-side [bias](../../glossary.md#term-bias). `GRU_RESET_AFTER` uses reset,
update, and candidate gate order, applies reset after the recurrent candidate projection, computes
`candidate + update * (hidden - candidate)`, and has one optional packed input-side bias. [`LSTM`](../../glossary.md#term-long-short-term-memory-lstm)
uses input, forget, candidate, and output gate order, explicit hidden and cell states, the current
fixed [sigmoid](../../glossary.md#term-sigmoid)/tanh equations, and one optional packed input-side bias.

Model publishes exactly one direction enum, a two-output recurrent result, a three-output LSTM
result, and one public final field-free `model.tensor.RecurrentScan` static namespace in the
existing [Tensor](../../glossary.md#tensor) package. The namespace has exactly the following six biased or bias-free
constructors, with the [time-major](../../glossary.md#term-time-major) input explicit and first. This is advanced low-level Java
expression-construction [API](../../glossary.md#term-application-programming-interface-api), not a layer, [module](../../glossary.md#term-module), execution service, [registry](../../glossary.md#term-registry), [general scan](../../glossary.md#general-differentiable-scan) body,
or runnable [Compiler](../../glossary.md#term-compiler), [Engine](../../glossary.md#term-engine), [Runtime](../../glossary.md#term-runtime), or [backend](../../glossary.md#backend) API:

```java
RecurrentScanResult RecurrentScan.rnn(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

RecurrentScanResult RecurrentScan.rnn(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

RecurrentScanResult RecurrentScan.gru(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

RecurrentScanResult RecurrentScan.gru(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

LstmRecurrentScanResult RecurrentScan.lstm(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

LstmRecurrentScanResult RecurrentScan.lstm(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)
```

`Tensor` has no recurrent receiver method. `RecurrentScanResult` exposes exactly `outputs` and
`finalHidden`; `LstmRecurrentScanResult` exposes exactly `outputs`, `finalHidden`, and
`finalCell`. These typed Tensor-output carriers remain in `model.tensor`, while
`RecurrentScanKind` and `RecurrentDirection` remain [operation](../../glossary.md#operation) semantics. Every component is the
[canonical](../../glossary.md#term-canonical-form) wrapper from one exact shared [producer](../../glossary.md#term-producer) in the order shown below.

Ordinary neural-network callers continue to use `RnnSequence`, `GruSequence`, and
`LstmSequence`, whose current Java `long[]` lengths select static [unrolling](../../glossary.md#term-unrolling) and compact outputs.
A later NN runtime-length API may delegate to this namespace only after Compiler and backend
adoption. `Tensor.linear` remains unchanged because a [linear projection](../../glossary.md#linear-projection) is a natural one-input,
one-output last-[axis](../../glossary.md#term-axis) [contraction](../../glossary.md#contraction-dimension); the NN [`Linear`](../../glossary.md#term-linear-meanings) module owns [parameter](../../glossary.md#term-parameter) state, while
`Tensor.linear` is its stateless Model operation composition.

Operation input order is fixed:

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

Operation output order is fixed:

```text
RNN/GRU: [outputs, finalHidden]
LSTM:    [outputs, finalHidden, finalCell]
```

### Static [Shape](../../glossary.md#shape) and [runtime](../../glossary.md#term-runtime)-value boundary

The first executable capability has fully static Shapes while valid-length values remain ordinary
runtime inputs:

- input is [rank](../../glossary.md#term-rank) three `[time, batch, inputSize]` in [time-major](../../glossary.md#term-time-major) order;
- valid lengths are a [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) [`INT64`](../../glossary.md#data-type--datatype) [Tensor](../../glossary.md#tensor) `[batch]` with `requiresGrad == false`;
- initial hidden and the [LSTM](../../glossary.md#term-long-short-term-memory-lstm) initial cell are `[batch, hiddenSize]`;
- input [weight](../../glossary.md#term-weight) is `[gateCount * hiddenSize, inputSize]`;
- hidden weight is `[gateCount * hiddenSize, hiddenSize]`;
- optional [bias](../../glossary.md#term-bias) is `[gateCount * hiddenSize]`;
- `gateCount` is one for [RNN](../../glossary.md#term-recurrent-neural-network-rnn), three for [GRU](../../glossary.md#term-gated-recurrent-unit-gru), and four for LSTM;
- `inputSize` and `hiddenSize` are positive, while `time` and [`batch`](../../glossary.md#term-batch-machine-learning-and-hyperparameters) may be zero; and
- input, states, weights, and optional bias have one exact common floating [data type](../../glossary.md#data-type--datatype).

The dense output Shape is `[time, batch, hiddenSize]`, and each final-state Shape is
`[batch, hiddenSize]`. [Layout](../../glossary.md#layout) remains unresolved at [Model](../../glossary.md#term-model) construction. Output [gradient](../../glossary.md#term-gradient)
eligibility is the OR of the [differentiable](../../glossary.md#derivative-and-chain-rule) floating input, state, and [parameter](../../glossary.md#term-parameter) roles; valid
lengths never contribute.

Dynamic or [binding](../../glossary.md#term-binding)-dependent `time`, `batch`, `inputSize`, `hiddenSize`, parameter Shapes, or
output Shapes are outside this first program. [Prepared execution](../../glossary.md#prepared-execution--preparedexecution) must be reusable across
different valid-length values for the same compatible static [descriptors](../../glossary.md#term-descriptor), but not across
arbitrary Shapes. Runtime valid lengths must not be used to disguise a dynamic-Shape [lifecycle](../../glossary.md#lifecycle).

### Valid lengths, traversal, and outputs

For each original [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) row `b`, execution validates the [runtime](../../glossary.md#term-runtime) value `L[b]` in the inclusive
range `[0, time]`. Lengths are never inferred from input contents, [padding](../../glossary.md#padding) values, zero, [NaN](../../glossary.md#term-nan),
[token](../../glossary.md#term-token) identifiers, labels, or storage.

`FORWARD` applies the fixed transition at original [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) `0 .. L[b]-1`. `REVERSE` applies
the same transition at original coordinates `L[b]-1 .. 0`; it reverses only the valid prefix and
never traverses the padded suffix. In either direction:

- each valid coordinate stores the next hidden state produced after consuming that coordinate;
- each padded coordinate `t >= L[b]` stores the exact [positive zero](../../glossary.md#nan-infinity-and-signed-zero) of the common [data type](../../glossary.md#data-type--datatype);
- final hidden, and final cell for [LSTM](../../glossary.md#term-long-short-term-memory-lstm), is the state after the row's last executed transition;
- a zero-length row has positive-zero output at every time and returns its exact initial state
  values semantically; and
- when `time` is zero, every length must be zero, the dense output is empty, and the initial states
  are the semantic final states.

Padding is never a recurrent input. Before mutating any output representation, an executable
[backend](../../glossary.md#backend) must validate the complete length [vector](../../glossary.md#term-vector), every bound, and any representation-specific
access precondition. Invalid lengths fail the [run](../../glossary.md#term-run) without partially written published results.
[Engine](../../glossary.md#term-engine) owns any future public recurrent-execution exception translation; this architecture does
not select the public exception type.

### Purity and [lifecycle](../../glossary.md#lifecycle) [ownership](../../glossary.md#term-ownership-meanings)

The [operation](../../glossary.md#operation) is functionally pure. All carried state is explicit in its ordinary inputs and
final outputs. It owns no hidden [module](../../glossary.md#term-module), [compiler](../../glossary.md#term-compiler), prepared, or [runtime](../../glossary.md#term-runtime) state; [RNG](../../glossary.md#graph-rng-state--graphrngstate); mode; counter;
[`Buffer`](../../glossary.md#term-buffer); [`Parameter`](../../glossary.md#term-parameter); mutation; [callback](../../glossary.md#term-callback); I/O; or external resource. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) continues to own
parameter wrappers and state paths, while the operation consumes their current [Tensor](../../glossary.md#tensor) [bindings](../../glossary.md#term-binding) as
ordinary inputs. Different executions use isolated [`RunState`](../../glossary.md#run-state--runstate) instances under the existing
Runtime contract.

NN state dictionaries contain only parameter and persistent-buffer Tensor bindings. Cold semantic
scan [graphs](../../glossary.md#graph) and named [host](../../glossary.md#term-host) values may be copied, saved, loaded and rebound under [Checkpoint](../../glossary.md#term-checkpoint)'s
[generic source contract](extensions-training.md#generic-graph-source-and-values) and its owning
[graph-model v3 contract](extensions-training.md#canonical-graph-model-version-3)<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model))</span>. This persists
fixed operation meaning, [descriptors](../../glossary.md#term-descriptor) and ordered ports, not the result-[carrier](../../glossary.md#term-storage-carrier) Java objects or an
executable graph. Prepared executables, native/[backend](../../glossary.md#backend) [artifacts](../../glossary.md#term-artifact) and runtime state are rebuilt,
never serialized. Cold [preflight](../../glossary.md#term-preflight) does not read valid-length values; runtime bounds remain an
execution responsibility. [Persistence](../../glossary.md#term-persistence) grants neither backend execution nor [derivative](../../glossary.md#term-derivative) capability;
those require the actual forward and first-order closures described below.

[Owner](../../glossary.md#term-owner) boundaries are exact:

- Model owns operation kind and attributes, fixed semantics, descriptor-visible validation,
  result metadata, canonical multi-output provenance, and the current focused<span data-glossary-navigation="true"> ([validation glossary](../../glossary.md#term-validation); [metadata glossary](../../glossary.md#term-metadata); [canonical glossary](../../glossary.md#term-canonical-form); [provenance glossary](../../glossary.md#provenance))</span>
  `model.tensor.RecurrentScan` expression surface.
- Compiler captures one ordinary flat fixed node, owns inference and final validation, and adopts
  the operation in its exact closed inventory. First-order BPTT uses the shared length-aware
  Scan transition recipe and ordinary Tensor derivative rules described below.<span data-glossary-navigation="true"> ([captures glossary](../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([inference glossary](../../glossary.md#term-inference-meanings); [closed inventory glossary](../../glossary.md#term-closed-inventory); [BPTT glossary](../../glossary.md#term-backpropagation-through-time-bptt))</span>
- The recipe owns internal saved sites through the existing Scan residual protocol; it adds no
  public fixed-family saved-gate output, independent backward operation, backward kernel, or
  second recurrent derivative algebra. Higher-order/createGraph requests remain rejected before
  derivative allocation.<span data-glossary-navigation="true"> ([residual glossary](../../glossary.md#scan-residual-port-and-history-demand); [kernel glossary](../../glossary.md#kernel))</span>
- Planning performs its existing ordinary operation capability query and selects backend
  ownership. It does not interpret recurrence, direction, valid lengths, active rows, loop
  parameters, or routes.<span data-glossary-navigation="true"> ([operation capability query glossary](../../glossary.md#operation-capability-query--operationcapabilityquery); [backend ownership glossary](../../glossary.md#backend-ownership))</span>
- Shared Prepare uses its existing static-Shape projection, staged backend analysis, exact shared-
  resource declaration, slot assignment, and finalization lifecycle. This fixed route adds no
  loop-body or control-flow contract.
- Runtime uses its existing caller-input representation, immutable prepared recipe, cold binding,
  schedule, isolated run state, and bound invocation contracts. It does not inspect valid
  lengths, select a loop count, compact rows, or interpret recurrence.<span data-glossary-navigation="true"> ([immutable glossary](../../glossary.md#term-immutable); [run state glossary](../../glossary.md#run-state--runstate); [bound invocation glossary](../../glossary.md#bound-invocation--boundinvocation))</span>
- Engine owns typed logical caller-input to ordered Runtime-representation binding and typed
  publication mapping. A valid-length value never causes Engine to specialize or rebuild a graph.<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine); [publication glossary](../../glossary.md#term-publication))</span>
- A concrete backend advertises only exact implemented variant, type, Shape, and direction
  combinations. It lowers the occurrence once, declares all shared resources before slot
  assignment, and constructs the reusable executable during finalization.<span data-glossary-navigation="true"> ([occurrence glossary](../../glossary.md#term-occurrence))</span>
- NN retains module composition and current parameter/state bindings; a later NN/Data task owns
  runtime-valid-length convenience and schema integration. Data does not own recurrence and must
  not infer lengths from values.<span data-glossary-navigation="true"> ([schema glossary](../../glossary.md#term-schema))</span>
- Training retains optimizer and training-orchestration ownership. It does not implement BPTT,
  recurrent execution, saved-state policy, or a backend route.<span data-glossary-navigation="true"> ([optimizer glossary](../../glossary.md#term-optimizer); [backend route glossary](../../glossary.md#backend-route))</span>

The current [Planning](../../glossary.md#term-planning), [Prepare](../../glossary.md#prepare), and Runtime shared contracts are sufficient for this static-[Shape](../../glossary.md#shape)
ordinary-[node](../../glossary.md#node) design. A later implementation must stop and provide evidence if a [concrete backend](../../glossary.md#concrete-backend)
reveals an actual shared-contract gap; it must not add speculative region or loop-body types.

### Fixed-family first-order differentiation and numerical sites

`RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../../glossary.md#term-long-short-term-memory-lstm) admit first-order differentiation of every public
output: dense outputs, final hidden, and LSTM final cell. Each floating input, initial-state,
[weight](../../glossary.md#term-weight), and optional-[bias](../../glossary.md#term-bias) position is a [differentiable](../../glossary.md#derivative-and-chain-rule) role; valid lengths remain a discrete
non-[gradient](../../glossary.md#term-gradient) input. Selected output [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) may be supplied independently or jointly. Missing
seeds are typed zeros, and positional ties and cross-[occurrence](../../glossary.md#term-occurrence) contributions add exactly once
each through ordinary [Tensor](../../glossary.md#tensor) algebra. A recurrent occurrence on an unrelated forward branch
does not by itself reject a backward-capable or training request.

[Compiler](../../glossary.md#term-compiler) constructs a bounded shared transition recipe using ordinary operations inside one
length-aware Scan, then applies the existing first-order Scan/ordinary [AD](../../glossary.md#autograd) protocol. It carries
the exact original lengths actual into the generated opposite-traversal sibling. No [host](../../glossary.md#term-host) length
read, static time unroll, separately maintained gate [derivative](../../glossary.md#term-derivative), or [backend](../../glossary.md#backend)-specific backward
[kernel](../../glossary.md#kernel) is permitted. Public fixed-result wrappers retain their original [identities](../../glossary.md#term-identity-meanings) and port
order; internal recipe histories are not additional public fixed outputs. Recipe and derivative
[graph](../../glossary.md#graph) size remain independent of `time`.
Incoming cotangents generated by ordinary consumers bind through the internal recipe [preflight](../../glossary.md#term-preflight),
as in a [Scan body](../../glossary.md#backpropagation-through-time-and-recurrent-state); they are not reclassified as user-provided explicit seeds. Public explicit
[seed](../../glossary.md#term-seed) eligibility restrictions remain unchanged.

The transition recipe must preserve the fixed [operation](../../glossary.md#operation)'s numerical sites, not merely equivalent
real-number equations. For [BFLOAT16](../../glossary.md#term-bfloat16) and [FLOAT16](../../glossary.md#term-float16) forward execution, projections and gate arithmetic
use [FLOAT32](../../glossary.md#term-float32) working sites; recurrent hidden and cell states narrow to their declared [carrier](../../glossary.md#term-storage-carrier) at
each step. LSTM's hidden update consumes the already narrowed cell state. Retained values and
ordinary derivative operands must denote these exact sites. FLOAT32/[FLOAT64](../../glossary.md#term-float64) retain their declared
ordinary sites. Nonempty recurrent contractions begin with their first actual product, retaining
[singleton](../../glossary.md#term-singleton) [negative zero](../../glossary.md#nan-infinity-and-signed-zero) through the transition and public state. No positive-zero [accumulator](../../glossary.md#accumulator-type)
seed or post-[contraction](../../glossary.md#contraction-dimension) zero addition may replace that site; [MATMUL](../../glossary.md#matrix-multiplication--matmul)'s final-zero freedom does
not extend to recurrent intermediate values. Compiler [admission](../../glossary.md#term-admission) of floating derivative graphs
does not grant backend arithmetic capability: Java [CPU](../../glossary.md#term-cpu) BFLOAT16/FLOAT16 arithmetic admission
remains locked, while each backend must qualify the complete actual generated closure independently.

Length-mode activity rules apply to the whole recipe and reverse body. Padded input and seed
[coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) cause no gate, contraction, or dependent arithmetic; inactive state cotangents pass
through unchanged, and shared [parameter](../../glossary.md#term-parameter) gradients accumulate only active row contributions under
the existing ordered reduction rules. A zero-length row passes final-state seeds to the
corresponding initial states, with zero input and parameter contributions. At `time == 0`, a
selected final-state seed is the exact initial-state gradient and may be published directly
through the existing producerless-value contract; no fictitious reverse Scan, copy, or dummy
state is allowed. Empty [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) and time domains retain their actual [descriptor](../../glossary.md#term-descriptor) geometry.

This admission is first-order only. Higher-order/createGraph requests, including a second stage
rooted at a first-stage recurrent gradient, reject before any derivative allocation. Existing
general-Scan [residual](../../glossary.md#scan-residual-port-and-history-demand)-root, unsupported-body and generated-reverse guards remain unchanged.


### Performance and migration invariants

One [Model](../../glossary.md#term-model) scan [occurrence](../../glossary.md#term-occurrence) remains one [compiler](../../glossary.md#term-compiler) [graph](../../glossary.md#graph) [node](../../glossary.md#node) regardless of `time` or valid-length
values, so its graph construction and graph size are [`O(1)`](../../glossary.md#memory-management-atomic-counters-and-complexity) in `time`. [Backend](../../glossary.md#backend) analysis prepares
the transition implementation once for that occurrence and must not construct, [compile](../../glossary.md#compile), or retain
one graph, node, or executable body per time step.

The [runtime](../../glossary.md#term-runtime) [hot path](../../glossary.md#term-hot-path) performs no reflection, string [dispatch](../../glossary.md#term-dispatch), [scalar](../../glossary.md#term-scalar)-[element](../../glossary.md#term-element) boxing, graph
inspection, [operation](../../glossary.md#operation) dispatch, backend [lookup](../../glossary.md#term-lookup), route selection, or per-step object-graph growth.
Invalid [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) execute no recurrent [dot products](../../glossary.md#linear-algebra-building-blocks), gates, or state update. A bounded branch-
based row/time traversal is sufficient for the first capability. Physical active-row compaction,
sorting, packed [buffers](../../glossary.md#term-buffer), vectorized packed [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters), workspace reuse, and claims that length checks
or zero initialization perform no work remain deferred [backend route](../../glossary.md#backend-route) decisions requiring truthful
resource and performance evidence.

Current `RnnSequence`, `GruSequence`, `LstmSequence`, and the three bidirectional containers keep
their snapshotted Java `long[]`, compact-output-list, static-unroll, exact-[provenance](../../glossary.md#provenance), and final-
state contracts unchanged. Model and Compiler implementation must not redirect those APIs to the
new operation. A later [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) decision must address whether to add runtime-length overloads, migrate,
retain compatibility, or deprecate. It must acknowledge that the new scan returns dense zero-
padded original-time-aligned outputs, while current static containers return compact per-step
output lists. A [host](../../glossary.md#term-host) `long[]` adapter that builds a different graph for each length pattern is not
an implementation of this runtime-valid-length decision.

Any later bidirectional migration must preserve independent [parameter](../../glossary.md#term-parameter) [ownership](../../glossary.md#term-ownership-meanings), valid-prefix-
only reverse traversal, forward-first final-[axis](../../glossary.md#term-axis) concatenation, original-time [alignment](../../glossary.md#term-alignment), and
type-specific final hidden and cell states.

## [General differentiable scan](../../glossary.md#general-differentiable-scan)

The accepted exception is a pure, fixed-count scan over ordinary [Tensor](../../glossary.md#tensor) algebra. It is not a
general region system or while loop. [Automatic differentiation](../../glossary.md#autograd) ([AD](../../glossary.md#autograd)) constructs an ordinary sibling
scan in the opposite traversal before one combined [capture](../../glossary.md#term-capture-meanings). This section specifies the accepted
protocol. Model0035 implements and validates its source/[IR](../../glossary.md#term-intermediate-representation-ir) foundation; Planning0008 implements
optional compiled-body query transport and provider contract tests. Compiler0011 implements scoped
capture/[inference](../../glossary.md#term-inference-meanings), bounded [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) first-order AD and final body-bearing query/demand
emission. Prepare0010 implements scoped projection and static [binding](../../glossary.md#term-binding)/demand [validation](../../glossary.md#term-validation). CPU0030
implements the bounded FLOAT32 same-[shape](../../glossary.md#shape) [pointwise](../../glossary.md#term-pointwise) forward/actual-reverse [slice](../../glossary.md#slice). CPU0031 adds
[homogeneous](../../glossary.md#term-homogeneous) FLOAT32 broadcast, SUM/sumToShape and bounded shape/[layout](../../glossary.md#layout) closure, including qualified
actual reverse bodies. CPU0032 adds static homogeneous FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul) and independently
qualified actual [Compiler](../../glossary.md#term-compiler) reverse bodies. CPU0033 implements homogeneous FLOAT64 across that
complete bounded declared algebra and actual first-order reverse closure. The [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) [CPU](../../glossary.md#term-cpu) domain
is implemented for static T and [canonical](../../glossary.md#term-canonical-form) boundaries plus proved zero-offset all-zero-[stride](../../glossary.md#term-stride)
[scalar](../../glossary.md#term-scalar) reads. The narrow dense actual-read-[view](../../glossary.md#term-view-meanings) exception below was approved on 2026-10-24;
CPU0037 implements it with [qualification](../../glossary.md#term-qualification) and main's final capability acceptance on 2026-10-24.
Wider layouts, low types and the later migration gates remain excluded.
Exact provider/body/binder qualification still
applies independently to each [occurrence](../../glossary.md#term-occurrence). Ordinary users compose Tensor
algebra; they never choose saved-tensor recipes. The
[focused explanation](../general-differentiable-scan.md)
and [ADR 0045](../../design/decisions/0045-general-differentiable-scan.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span> give examples and rationale.

### Source body, attachments and scoped [identities](../../glossary.md#term-identity-meanings)

The implemented static [admission](../../glossary.md#term-admission) below is retained. The sole bounded source relaxation is
[Accepted symbolic-batch scan](#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>; its semantics are owned here,
not by the numeric resource boundary. Model0042 implements this source-only relaxation, independently
reviewed and MAIN accepted 2026-10-25; Compiler0014's symbolic [inference](../../glossary.md#term-inference-meanings)/[AD](../../glossary.md#autograd)/[binding](../../glossary.md#term-binding) is also
independently reviewed and MAIN accepted after compatibility [validation](../../glossary.md#term-validation). Prepare0014 implements
bounded [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) symbolic forward and actual first-order reverse through the existing
[Prepare](../../glossary.md#prepare)/[CPU](../../glossary.md#term-cpu)/[Engine](../../glossary.md#term-engine) [owners](../../glossary.md#term-owner); final [qualification](../../glossary.md#term-qualification) is tracked in its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md).
Cold support remains static-only. Ordinary symbolic-B execution retains its separate qualification.
The user-approved length-mode source combination is specified under
[Header, bindings and source admission](#header-bindings-and-source-admission). Model0044 implements
that source-only combination under the authority accepted on 2026-10-26. Independent source/docs
review and MAIN's root test/[Javadoc](../../glossary.md#term-javadoc) [checkpoint](../../glossary.md#term-checkpoint) passed; MAIN accepted Model0044 as Complete on 2026-10-26.

The complete new [Model](../../glossary.md#term-model) inventory, under `io.github.pho001.synaptik.`, is:

- `model.tensor.ScanBody`: [immutable](../../glossary.md#term-immutable) ordered ordinary [Tensor](../../glossary.md#tensor) formals `(carry, step, invariants)`,
  roots `(nextCarry, emissions)`, and derived ordered retained values. This is source expression
  containment, not a second DAG, captured graph, callback or runnable closure.<span data-glossary-navigation="true"> ([DAG glossary](../../glossary.md#term-directed-acyclic-graph-dag); [callback glossary](../../glossary.md#term-callback))</span>
- `model.tensor.Scan.apply(initialCarry, sequences, invariants, body, reverse)` and
  `model.tensor.ScanResult(finalCarry, outputs, residuals)`: explicit lists of canonical Tensors.
  Construct formals once with TensorFactory; retain no callback.
- `model.operation.scan.ScanKind` and `ScanAttrs`: fixed-count meaning and scalar header only,
  containing inferred static time count T, carry/sequence/invariant/emission/residual counts and
  reverse flag on the implemented all-valid path. The implemented length-mode source additions are
  defined below. Attributes contain no Tensor, graph, registry key,
  executable or saved runtime values.<span data-glossary-navigation="true"> ([registry glossary](../../glossary.md#term-registry); [runtime glossary](../../glossary.md#term-runtime))</span>
- `model.graph.CompiledScanBody`: one existing [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel), ordered role-result ValueId
  bindings and ordered demanded outer-port indices. This is the one captured body representation.

Extend existing [TensorProducer](../../glossary.md#producer--tensorproducer) with `Optional<ScanBody>`, validated before [canonical](../../glossary.md#term-canonical-form) output
allocation, and CompiledNode with `Optional<CompiledScanBody>`. Preserve old constructor forms
with empty attachments for ordinary and fixed operations. No additional type/layer is selected.

On the static path, formals are [identity](../../glossary.md#term-identity-meanings)-distinct, [provenance](../../glossary.md#provenance)/storage-free, static, canonical and disjoint from actual
outer bindings. Removing [axis](../../glossary.md#term-axis) zero from a sequence [descriptor](../../glossary.md#term-descriptor) yields its step formal with canonical
step geometry. [Carry](../../glossary.md#term-carry)/[invariant](../../glossary.md#term-invariant) formals match actual [shape](../../glossary.md#shape), [dtype](../../glossary.md#data-type--datatype) and [gradient](../../glossary.md#term-gradient) eligibility. Actual
inputs may repeat exact identities across roles; never deduplicate positions. Body [leaves](../../glossary.md#term-leaf) must be
exactly declared formals by identity, except declared unused formals are allowed. Hidden [scalar](../../glossary.md#term-scalar)
Tensor [constants](../../glossary.md#term-constant) must become explicit invariants; scalar [operation](../../glossary.md#operation) attributes remain ordinary algebra.
Reject free leaves, [host](../../glossary.md#term-host)-backed constants, [RNG](../../glossary.md#graph-rng-state--graphrngstate), mutation and nested/recursive bodies.

Model source construction admits existing known [PURE](../../glossary.md#term-pure) Tensor operation semantics and their
applicable existing Model dtypes, independently of the initial CPU execution algebra and [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding)
domain below. Validate the whole contained body, including canonical siblings, against existing
static or admitted named-[symbol](../../glossary.md#term-symbol), type, shape and role rules. Thus a valid [FLOAT16](../../glossary.md#term-float16) [TANH](../../glossary.md#term-tanh) body or FLOAT32 RELU body may
construct without belonging to that first CPU domain. This grants no custom/unknown semantics,
new operation capability, AD rule or [backend](../../glossary.md#backend) support. Effects and the other source exclusions
above still reject at construction; unsupported AD and execution occurrences still reject in
their owning [Compiler](../../glossary.md#term-compiler)/capability/backend boundaries.

Cold copying has a separate bounded constructor inventory in
[Compiler's cold lifecycle](compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>.
Model-valid PURE does not mean reconstructible, executable, [differentiable](../../glossary.md#derivative-and-chain-rule) or serializable.
The cold exception preserves applicable Model dtypes rather than importing CPU qualification.

For outer result construction, let `G` be the OR of `requiresGrad` over every actual initial-carry,
sequence and [invariant input](../../glossary.md#term-invariant-input) position. Every differentiable final-carry, stacked-emission and
stacked-[residual](../../glossary.md#scan-residual-port-and-history-demand) descriptor receives `requiresGrad = G`; every nondifferentiable result dtype
receives false. Include unused actual inputs and apply the same rule for T=0. Formal flags still
match their corresponding actuals, including sequence-step formals; neither formal nor ordinary
body-value flags are rewritten to G. This conservative construction flag is not a claim of a
nonzero [dependency](../../glossary.md#term-dependency), an available [derivative](../../glossary.md#term-derivative) rule or backend execution support. In particular,
with `h0.requiresGrad=false`, `x.requiresGrad=true`, `nextCarry=h+xStep` and `emit=h`, all
differentiable outer final-carry/emission/residual descriptors receive true even though the carry
formal and source emission have false flags: later emissions can depend on earlier sequence steps.

The same immutable template and its [producers](../../glossary.md#term-producer) may serve multiple scan occurrences, with independent
[capture](../../glossary.md#term-capture-meanings) scopes. Reject [producer](../../glossary.md#term-producer) escape into the outer [graph](../../glossary.md#graph) and sharing across different lexical
templates, not reuse of one template. No outer [ValueId](../../glossary.md#logical-valueid-versus-physical-memory-slot) appears inside a body. Capture assigns IDs
once per [scope](../../glossary.md#term-scope); numeric IDs may coincide across scopes and are qualified only by their containing
CompiledScanBody. TensorIds remain global. Body graph inputs list every formal in declaration order,
including unused ones; graph outputs are unique, but ordered role bindings may repeat (for example,
one value serving as both next carry and emission) without duplicate graph values.

### [Canonical](../../glossary.md#term-canonical-form) [residual](../../glossary.md#scan-residual-port-and-history-demand) ports and demand

A [residual port](../../glossary.md#scan-residual-port-and-history-demand) names a body value available for later saving, not an allocation. Construction-time
canonical ports are Synaptik's [immutable](../../glossary.md#term-immutable)-[producer](../../glossary.md#term-producer) design choice, not an industry requirement.

1. ScanBody derives retained values by deterministic producer postorder and ascending canonical
   output slot, identity-unique, including unused siblings and each carry formal (previous state).
   Step/invariant formals instead use explicit outer inputs, without duplicate saved sequences.
   Every other reachable body value receives a latent residual slot; Model selects no derivative rule.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings); [carry glossary](../../glossary.md#term-carry); [invariant glossary](../../glossary.md#term-invariant); [Model glossary](../../glossary.md#term-model); [derivative glossary](../../glossary.md#term-derivative))</span>
2. Original Scan.apply construction reserves canonical residual descriptors/wrappers after final
   carry and emissions, never through producer replacement, postcapture augmentation or replay.
   These descriptors describe stacked histories but allocate none. Aliased roles share body-value
   bindings while outer canonical wrappers remain distinct.<span data-glossary-navigation="true"> ([descriptors glossary](../../glossary.md#term-descriptor); [bindings glossary](../../glossary.md#term-binding))</span>
3. Only publication/downstream consumption demands history. Internal body use needs step temporaries,
   not history. Propagate ordinary-use demand into queries and Prepare and revalidate after every
   rewrite. Live histories store at original time indices; a previous carry is saved before its step.<span data-glossary-navigation="true"> ([publication glossary](../../glossary.md#term-publication); [Prepare glossary](../../glossary.md#prepare))</span>
4. Only live residual history restricts fusion/recompute at saved sites. Dead ports allocate/store
   zero history and impose no residual fusion restriction, including the generated reverse scan's
   own residuals. Other ordinary observable boundaries still apply. This requires new liveness and
   Prepare work, not existing sibling dead-code elimination or an opaque saved-workspace handshake.<span data-glossary-navigation="true"> ([liveness glossary](../../glossary.md#term-liveness); [dead-code elimination glossary](../../glossary.md#term-dead-code-elimination-dce); [opaque glossary](../../glossary.md#term-opaque))</span>

Demanded conservative histories cost `O(T * retained elements)` storage and constrain [fusion](../../glossary.md#term-fusion).
More selective saving or recomputation requires a separate reviewed decision.

### Reverse scan and exact [primal](../../glossary.md#term-primal) [bindings](../../glossary.md#term-binding)

A [cotangent](../../glossary.md#cotangent) is [reverse-mode](../../glossary.md#primal-adjoint-and-differentiation-seed) [sensitivity](../../glossary.md#term-sensitivity); a primal is an original forward body value. [Compiler](../../glossary.md#term-compiler) owns
the following first-order protocol through existing [Tensor](../../glossary.md#tensor) [derivative](../../glossary.md#term-derivative) algebra, not captured-[IR](../../glossary.md#term-intermediate-representation-ir) [AD](../../glossary.md#autograd):

1. Inventory the body and combine cotangents for every selected final-carry and emission port once
   per scan occurrence. Missing port seeds are explicit typed zeros; final-carry and sequence seeds
   may coexist. Existing public request seed guards still apply.<span data-glossary-navigation="true"> ([carry glossary](../../glossary.md#term-carry); [occurrence glossary](../../glossary.md#term-occurrence); [seeds glossary](../../glossary.md#commit-release-distribution-log-and-seed-meanings); [seed glossary](../../glossary.md#term-seed))</span>
2. Extend selected Compiler rule entry points with an explicit compile-local primal-operand binding
   function. Every formula operand denoting a primal resolves from its exact original body Tensor
   to a reverse-body formal backed by the corresponding residual, step or invariant input. TANH's
   saved output resolves to its residual formal, not recomputed tanh. Contribution maps continue
   to address original Tensor identities. No ValueId-to-Tensor AD map or second algebra is allowed.<span data-glossary-navigation="true"> ([compile glossary](../../glossary.md#compile); [invariant input glossary](../../glossary.md#term-invariant-input); [TANH glossary](../../glossary.md#term-tanh); [ValueId glossary](../../glossary.md#logical-valueid-versus-physical-memory-slot))</span>
3. Close derivative constants, including TANH/SIGMOID oneLike scalar leaves, through explicit
   non-gradient invariant formals bound to outer Compiler-created splats via existing constant
   ingress. Expand remains ordinary body algebra. Zero seeds and accumulator initializers use
   registered outer splats/expressions; carry initializers bind through carry formals and body
   constants through invariant formals, never undeclared free leaves.<span data-glossary-navigation="true"> ([leaves glossary](../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([constants glossary](../../glossary.md#term-constant); [SIGMOID glossary](../../glossary.md#term-sigmoid); [scalar glossary](../../glossary.md#term-scalar); [gradient glossary](../../glossary.md#term-gradient); [invariant glossary](../../glossary.md#term-invariant); [accumulator glossary](../../glossary.md#accumulator-type))</span>
4. Construct an ordinary sibling Scan.apply in opposite traversal. Cotangent carry starts with
   final-carry seeds; output seeds and residual histories are explicit sequences; shared parameters
   are invariants. Initialize parameter sums to exact typed zero and add each positional contribution
   once in traversal order. Return dSequence emissions, final dInitialCarry and parameter sums.
   Sum tied actual-role and cross-occurrence terms afterward with ordinary Tensor.add, never
   identity-deduplicate contributions.<span data-glossary-navigation="true"> ([parameters glossary](../../glossary.md#term-parameter); [identity glossary](../../glossary.md#term-identity-meanings))</span>
5. One combined capture includes original forward wrappers and generated reverse scan, each body
   captured once in its independent scope. Public TensorId/publication roles remain unchanged;
   ordinary residual edges enforce forward-before-reverse liveness. Exact saved sites survive
   layout closure, reindexing, DCE/CSE and fusion. Never merge scan occurrences or cross live saved
   boundaries.<span data-glossary-navigation="true"> ([capture glossary](../../glossary.md#term-capture-meanings); [scope glossary](../../glossary.md#term-scope); [TensorId glossary](../../glossary.md#tensorid); [publication glossary](../../glossary.md#term-publication); [liveness glossary](../../glossary.md#term-liveness); [layout glossary](../../glossary.md#layout); [DCE glossary](../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../glossary.md#term-common-subexpression-elimination-cse); [fusion glossary](../../glossary.md#term-fusion))</span>

[Residual](../../glossary.md#scan-residual-port-and-history-demand) ports use the conservative outer construction flags defined above, but differentiation
rooted at a [residual port](../../glossary.md#scan-residual-port-and-history-demand), unsupported body rules, and every scan higher-order/createGraph request
reject before derivative allocation. [Preflight](../../glossary.md#term-preflight) includes unsupported occurrences on otherwise
unrelated original branches.
Fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) first-order [admission](../../glossary.md#term-admission) is defined separately by the shared recipe above;
it grants no higher-order exception to this protocol.

### Traversal, empty domains and declared [qualification](../../glossary.md#term-qualification)

The first completed [CPU](../../glossary.md#term-cpu) domain is [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) and [FLOAT64](../../glossary.md#term-float64), [canonical](../../glossary.md#term-canonical-form) boundaries, static
T >= 0, at least one [carry](../../glossary.md#term-carry) and one sequence, zero or more invariants/emissions, common sequence T,
and fixed body [shapes](../../glossary.md#shape)/[dtypes](../../glossary.md#data-type--datatype). Forward traverses `0 .. T-1`; reverse traverses `T-1 .. 0`. Emissions
and live [residual](../../glossary.md#scan-residual-port-and-history-demand) histories are stored at original indices, not traversal-order positions.

The approved CPU0037 extension admits a resolved actual read with `storageOffset=0` and [strides](../../glossary.md#term-stride)
exactly equal, [axis](../../glossary.md#term-axis) by axis, to `LayoutDescriptor.contiguous(shape).strides()`, even when
`isView=true`. This applies to every actual initial-carry, sequence and [invariant](../../glossary.md#term-invariant) read position
in the existing [pointwise](../../glossary.md#term-pointwise), shaped and [MATMUL](../../glossary.md#matrix-multiplication--matmul) families, including each independently qualified
actual [Compiler](../../glossary.md#term-compiler)-generated reverse [occurrence](../../glossary.md#term-occurrence) ([seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), saved histories and [primal](../../glossary.md#term-primal) [bindings](../../glossary.md#term-binding)).
Canonical reads and the existing zero-offset all-zero-stride [scalar](../../glossary.md#term-scalar) reads remain admitted;
the latter do not have to satisfy the dense-stride predicate. Internal formals and outer
outputs/write boundaries retain complete non-[view](../../glossary.md#term-view-meanings) canonical [descriptor](../../glossary.md#term-descriptor) requirements. Existing
proved local body views are unchanged. This is read [admission](../../glossary.md#term-admission), not removal of view [metadata](../../glossary.md#term-metadata) or
changes to LayoutDescriptor equality, [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings)/[provenance](../../glossary.md#provenance), source or storage [ownership](../../glossary.md#term-ownership-meanings).

The implemented read admission retains exact shape/type/role, bounds, referenced-span,
storage accessibility, [lifetime](../../glossary.md#term-lifetime) and overlap proofs. Read/read [aliases](../../glossary.md#aliasing-and-arena) remain legal; writable
overlap does not. Absent/unresolved [layouts](../../glossary.md#layout), nonzero offsets and noncanonical strides remain
rejected except for the retained scalar-read case, as do low/mixed unsupported types and bodies.
The added read uses the canonical twin's dense address/access/resource/[IR](../../glossary.md#term-intermediate-representation-ir) and generated-code
path, without inserted copies or [CONTIGUOUS](../../glossary.md#term-contiguous). This grants no general affine-layout support.

For T=0 no body executes: emissions/residuals are empty, final carry is the exact initial carry,
dInitialCarry equals dFinalCarry, and invariant sums are typed zero. Empty [dimensions](../../glossary.md#dimension) follow each
constituent's semantics; an empty dimension does not excuse skipping a nonempty reduction or an
unsupported [operation](../../glossary.md#operation).

Exact initial carry here means its represented values/[raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity), not reuse of the input Tensor
wrapper or its flags. Construction still reserves distinct canonical outer ports and applies G
as defined above; outer flags are not copied from body roots or initial-carry descriptors.

The initial declared CPU execution algebra is ADD/SUB/MUL/DIV (ordinary floating tensor and existing
scalar forms), NEG/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid), [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) MATMUL and rank-two `[1,0]` PERMUTE under existing guards.
Ordinary DIV [derivatives](../../glossary.md#term-derivative) are reused without new [masks](../../glossary.md#term-boolean-mask)/numerics. Qualification covers the whole generated closure:
broadcast reductions/sumToShape, expands, views/contiguous [materialization](../../glossary.md#materialization), [constants](../../glossary.md#term-constant) and required
casts. [NaN](../../glossary.md#term-nan)/Inf, [signed zero](../../glossary.md#term-signed-zero), zero divisors, [overflow](../../glossary.md#term-overflow)/[underflow](../../glossary.md#term-underflow) and empty reductions retain existing
family arithmetic and saved-value rules; unsupported occurrences reject, not blanket all-op admission.
[BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16) require later independent qualification under existing working/[rounding](../../glossary.md#term-rounding)/saved-state
semantics, not whole-[model](../../glossary.md#term-model) conversion or support inferred from emitters.

### [Qualification](../../glossary.md#term-qualification), performance and migration gates

The implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
qualifies only its selected exec-v1 terminal arithmetic and [synchronous](../../glossary.md#term-synchronous) [CPU](../../glossary.md#term-cpu) region [dispatch](../../glossary.md#term-dispatch).
It does not admit parallel/associative time traversal or change [scalar](../../glossary.md#term-scalar) CPU0073.

Construction, captured/prepared objects and generated code are `O(body)` rather than T-fold copies;
live saved bytes may grow with T, while undemanded [inference](../../glossary.md#term-inference) history has zero bytes/stores. The
[capability boundary](foundational-modules.md#profile-free-capability-queries),
[Compiler boundary](compiler-autograd.md#compiler-owned-automatic-differentiation),
[Prepare boundary](runtime-prepare-engine.md#modulesprepare) and
[CPU boundary](backend-execution.md#cpu-backend-routes) own their respective [admission](../../glossary.md#term-admission), projection,
resources and emitted-loop duties. No [runtime](../../glossary.md#term-runtime) interpreter, per-op [lookup](../../glossary.md#term-lookup), per-step [callback](../../glossary.md#term-callback) or
per-step allocation is allowed. Materialize only when needed; this is no universal no-copy or speed claim.

The user-approved 2026-10-30 [CPU0073 mechanism](backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
is implemented and MAIN accepted after independent review and named gates on 2026-10-30.
It uses exactly three CPU-internal top-level
types (`CpuRowMapReductionIr`, `CpuRowMapReductionLowering`, `CpuRowMapReductionEmitter`), nested
[`Contraction`](../../glossary.md#contraction-dimension)/`Sum`, and their specified generation seams/selected [identity](../../glossary.md#term-identity-meanings) suffixes. That owning
subsection defines packages, [signatures](../../glossary.md#term-signature), proof limits and retention. A [fused](../../glossary.md#specialization-and-fusion) region is emitted
inside the existing Scan wrapper/time loop, not a new body [graph](../../glossary.md#graph) or executor. Existing semantic
sections and admission remain unchanged. The [CPU0073 result](../../planning/backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md#result)
records the bounded execution/retention evidence, not a performance claim.
No other type, runtime edge or namespace change is permitted.

Implementation gates are serial: independently reviewed authority cutover and main acceptance;
separately [Ready](../../glossary.md#term-ready) [Model](../../glossary.md#term-model) source/[IR](../../glossary.md#term-intermediate-representation-ir); [Planning](../../glossary.md#term-planning) query; [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd)/[capture](../../glossary.md#term-capture-meanings); [Prepare](../../glossary.md#prepare) projection; a complete
bounded [pointwise](../../glossary.md#term-pointwise) [FLOAT32](../../glossary.md#term-float32) forward/automatic-reverse vertical milestone; remaining declared forms
and closure plus [FLOAT64](../../glossary.md#term-float64) domain completion; then the low-level join. No successor becomes Ready
automatically. First execution is all-valid, static and [canonical](../../glossary.md#term-canonical-form). Retain all existing four-[dtype](../../glossary.md#data-type--datatype)/
read-[layout](../../glossary.md#layout) fixed routes and static [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) routes without redirection or retirement.

CPU0037 is a separately approved same-address read-admission extension, not NN migration.
Acceptance requires focused [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) pointwise/shaped/[MATMUL](../../glossary.md#matrix-multiplication--matmul) and actual-reverse qualification,
canonical-twin access/resource/IR/identity/generated-byte equality, preserved formal/write guards,
and real Conv1d-prefix scan plus [singleton](../../glossary.md#term-singleton)-CONCAT-seeded reverse numerical [gradients](../../glossary.md#term-gradient) against
independent ordinary [unrolling](../../glossary.md#term-unrolling). Include T=0/empty domains where applicable, [heap](../../glossary.md#term-heap)/segment reads,
changed inputs across reusable runs and [alias](../../glossary.md#aliasing-and-arena)/write guards. With generator, physical [binding](../../glossary.md#term-binding) and
identities unchanged, reuse CPU0036's specialized clean-Java/Class-File/performance evidence and
its limitations; no new benchmark is required. If that premise fails, stop for a main decision
before [widening](../../glossary.md#term-widening) implementation. Neither this exception nor its completion unlocks `.rnn`,
[padding](../../glossary.md#padding), low types or fluent migration authority.

CPU0037's focused qualification covers the above read extension in both dtypes, including real
Conv1d-prefix and singleton-CONCAT workflows in both traversals against ordinary-[unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) outputs
and first-order gradients. Eighteen canonical/[view](../../glossary.md#term-view-meanings) pairs retain complete [lowering](../../glossary.md#lowering) facts,
resources, canonical and both [specialization](../../glossary.md#term-specialization) identities, generated class bytes and cached [artifact](../../glossary.md#term-artifact)
reuse. Generator, physical binding, [schema](../../glossary.md#term-schema)/family 71 and staged v4 identities are unchanged;
CPU0036's evidence and limitations therefore remain applicable, without a new speed claim.
Main accepted the final [module](../../glossary.md#term-module)/[Javadoc](../../glossary.md#term-javadoc), named [conformance](../../glossary.md#term-conformance)/integration and configured root gates
on 2026-10-24; [CPU0037](../../planning/backends/cpu/tasks/0037-general-scan-dense-read-views.md#result)
records final capability acceptance.

Future implementation acceptance includes reusable-template/escape cases, closed [constants](../../glossary.md#term-constant), tied
roles/roots and cross-[occurrence](../../glossary.md#term-occurrence) sums, two carries, both [seed](../../glossary.md#term-seed) classes, shared [parameters](../../glossary.md#term-parameter), exact saved
[rounding](../../glossary.md#term-rounding)/special values, T=0/empties, [residual](../../glossary.md#scan-residual-port-and-history-demand)-root/higher-order rejection, [overflow](../../glossary.md#term-overflow)/overlap/cleanup/
retry/[concurrency](../../glossary.md#term-concurrency), unsupported interior/generated occurrences, missing body, supported children with
unsupported loop binder, preserved projection and no runtime graph access. Performance evidence uses
the specialized optimal clean Java algorithm/hot-loop/[dataflow](../../glossary.md#term-dataflow) [oracle](../../glossary.md#term-test-oracle) and Class-File inspection,
plus short warmed matched-work generated/direct measurements; separate cold generation/class-load/[JIT](../../glossary.md#term-just-in-time-compilation-jit),
entry/[preflight](../../glossary.md#term-preflight) and [kernel](../../glossary.md#kernel)/step costs. Fix or explicitly justify diagnosed avoidable overhead. No
broad/five-[fork](../../glossary.md#term-fork-meanings) campaign, zero-penalty/JIT or CPU0029 timing-parity promise follows. Future slices own
focused/affected-module test and Javadoc; main owns new architecture/conformance/integration coverage
and one final source-matched native-enabled root test/Javadoc gate at the low-level join. The
original authority-only cutover performed no code/[dependency](../../glossary.md#term-dependency) change, executable [validation](../../glossary.md#term-validation) or
measurement; CPU0037's later implementation evidence is distinguished above.

The approved all-valid owned NN direction uses existing `Model.graph`/`defineTensor`, `compute`
and `configureTraining`/`train`. Its first tested execution is static canonical [homogeneous](../../glossary.md#term-homogeneous)
FLOAT32/FLOAT64. Compiler [cold reconstruction](../../glossary.md#term-cold-reconstruction), [Checkpoint](../../glossary.md#term-checkpoint) attachment-preserving copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) and
NN [ownership](../../glossary.md#term-ownership-meanings)/[lifecycle](../../glossary.md#lifecycle) qualification were serial prerequisites. The bounded owned workflow is now
implemented under [the NN owner](extensions-training.md#owned-tensor-models); no fixed builder
redirection follows. Fixed-family [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) is separately owned by the shared recipe above.

The separate per-row semantic decision is now selected below; implementation and padded NN
migration remain gated. [General Scan](../../glossary.md#general-differentiable-scan) grants no automatic Model.train/save/load admission.
If/while, arbitrary regions, recursion, nested scans, parallel or associative general scans,
runtime tape, mixed-[owner](../../glossary.md#term-owner) bodies and a second executor remain excluded.

### CPU0074 selected terminal grouping exception

Implemented, independently reviewed and MAIN accepted, 2026-10-30. CPU0074 is Complete;
the [task Result](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result)
records [validation](../../glossary.md#term-validation) and unqualified steady-state profitability. This is the sole Scan semantic exception for the
[CPU0074 mechanism](backend-execution.md#cpu0074-selected-row-region-execution-exception).
Only selected CPU0073-shaped exec-v1 regions in already-admitted domains may use bounded
[feature](../../glossary.md#term-feature)/output [SIMD](../../glossary.md#simd), row-partial workers or both. The complete [occurrence](../../glossary.md#term-occurrence)/binder must qualify,
independently for genuinely [fused](../../glossary.md#specialization-and-fusion) forward and actual [Compiler](../../glossary.md#term-compiler)-generated reverse. A standard
[RNN](../../glossary.md#term-recurrent-neural-network-rnn) forward lacking a computed-row-[producer](../../glossary.md#term-producer)/terminal region remains [scalar](../../glossary.md#term-scalar). Static/length/combined
[F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) and all-valid symbolic F32 bounds do not admit all-valid symbolic F64 or low/wider types.

Within a selected terminal row-[axis](../../glossary.md#term-axis) floating [contraction](../../glossary.md#contraction-dimension) or represented Scan SUM, a fixed
partitioned rounded tree may regroup the declared contributors. Every active contributor still
participates exactly once; both contraction operands and dependent math remain activity-dominated.
Within each worker visit rows in increasing original row index; merge valid partials in fixed
ascending range-index order on the caller. Empty/inactive partials have validity flags and do
not invent +0 terms. [Singleton](../../glossary.md#term-singleton)/no-axis raw movement is determined by geometry, not active count.
The existing foundational [Aggregate floor](../../glossary.md#term-aggregate-floor) is unchanged; this lifts only the narrower ordered
CPU0073/Scan override at selected terminal sites, not arbitrary neighboring operations.

Feature-K contraction order, separate typed MUL/ADD/result [rounding](../../glossary.md#term-rounding), same compute/[accumulator](../../glossary.md#accumulator-type)
[dtype](../../glossary.md#data-type--datatype) and no implicit [FMA](../../glossary.md#term-fused-multiply-add-fma)/[TF32](../../glossary.md#term-tf32)/lower [precision](../../glossary.md#term-precision) remain mandatory. Ordinary exact SUM/[MEAN](../../glossary.md#term-mean) and
standalone [MATMUL](../../glossary.md#matrix-multiplication--matmul)-K receive no new permission. Time traversal and [carry](../../glossary.md#term-carry) are sequential: finish
all row partials and merge before exactly one distinct shared-carry ADD per active time at its
own dtype/[rounding site](../../glossary.md#term-rounding-site). Never absorb that ADD into a row partial, flatten across time, or update
shared floating state atomically. Inactive time preserves shared carry raw, not old+0.
[Masks](../../glossary.md#term-boolean-mask), original indices, casts, raw movement, empty [identities](../../glossary.md#term-identity-meanings), [padding](../../glossary.md#padding) and selected-realization
saved values remain exact. Load saved primals, never replay [TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid). [Finite](../../glossary.md#term-finite) tolerances are
per-[operation](../../glossary.md#operation)/dtype [qualification](../../glossary.md#term-qualification) tests only; no numerical mode/certification or cross-[device](../../glossary.md#term-device),
thread-count or strategy bitwise claim is created.

The emitted outer time loop remains the sole time-control [owner](../../glossary.md#term-owner). Only its generation-selected,
typed StepDispatch boundary may synchronously submit prebound row ranges and allocate existing
scheduler [metadata](../../glossary.md#term-metadata) per step. This qualifies the no-per-step-[dispatch](../../glossary.md#term-dispatch)/[callback](../../glossary.md#term-callback)/allocation language
in Qualification, performance and migration gates; Qualification and mechanism gates; and
[Runtime](../../glossary.md#term-runtime) [preflight](../../glossary.md#term-preflight) and implementation gates solely for that seam. Body/[element](../../glossary.md#term-element) allocation,
per-op [lookup](../../glossary.md#term-lookup), [graph](../../glossary.md#graph) interpretation, geometry repacking, a second executor and T-fold code/[buffers](../../glossary.md#term-buffer)
remain forbidden. [Runtime reservation/lifetime rules](runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation)<span data-glossary-navigation="true"> ([lifetime glossary](../../glossary.md#term-lifetime))</span>
govern [borrowing](../../glossary.md#term-borrowing), errors, interruption/close and [quiescence](../../glossary.md#term-quiescence). Join precedes consumer execution,
carry transition and scratch reuse; no nested worker submissions. One [RunState](../../glossary.md#run-state--runstate) owns all [mutable](../../glossary.md#term-mutable)
frames/staging/partials, not prepared state. Existing length preflight after producers and before
any Scan-local write remains intact, including all B values and T=0.

Only bounded raw feature-[transpose](../../glossary.md#term-transpose) staging proved by the [CPU](../../glossary.md#term-cpu) owner is permitted: once per [run](../../glossary.md#term-run)
for a true [invariant](../../glossary.md#term-invariant) across T, otherwise at the exact active time after the relevant producer/
SHARED history [binding](../../glossary.md#term-binding), or scalar fallback. No stale reuse, row compaction/sorting or B×[parameter](../../glossary.md#term-parameter)
bank. Activity gating encloses staging; T=0/B=0/all-inactive reads no body history or invariant.
All resource declarations precede slots. CPU0073 scalar-v1 algorithms/keys/ClassFiles/resources
and nonselected cases retain their existing rules. Only new exec-v1 [identity](../../glossary.md#term-identity-meanings) appendages and exact
internal seams in the CPU owner qualify the prior no-other-type/namespace/runtime-edge clauses.
No new Runtime protocol, public [API](../../glossary.md#term-application-programming-interface-api) or production-default strategy follows.

### Accepted symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) scan

Accepted direction on 2026-10-25: extend existing all-valid `Scan.apply`/`ScanBody` [admission](../../glossary.md#term-admission) to
one exact named [batch dimension](../../glossary.md#batch-dimension) B, retaining fixed time T, [features](../../glossary.md#term-feature) and state widths. This is an
accepted contract; Model0041's authority delivery is Complete, MAIN accepted
2026-10-25 after independent APPROVE. Model0042 implements only static/bounded-symbolic [Model](../../glossary.md#term-model)
source construction and consumer rejection evidence, independently reviewed and MAIN accepted 2026-10-25.
Compiler0014 implements symbolic [inference](../../glossary.md#term-inference-meanings), actual [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) first-order [AD](../../glossary.md#autograd) and fresh
scoped all-port [binding](../../glossary.md#term-binding), independently reviewed and MAIN accepted 2026-10-25 after compatibility [validation](../../glossary.md#term-validation).
Prepare0014 implements bounded [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 forward and actual [Compiler](../../glossary.md#term-compiler)-generated reverse,
generated [scalar](../../glossary.md#term-scalar) and single-range, through [Prepare](../../glossary.md#prepare)/[CPU](../../glossary.md#term-cpu)/[Engine](../../glossary.md#term-engine); its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md) tracks final [qualification](../../glossary.md#term-qualification).
Symbolic cold/copy/wire and Engine geometry [memoization](../../glossary.md#term-memoization) remain unimplemented.
Retain all implemented static FLOAT32/FLOAT64
paths, the actual-read exceptions above and fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) restrictions. Model0044 implements
the symbolic length-mode Model source combination below. Compiler0015 implements bounded
inference/actual first-order AD/fresh scoped all-port binding under
[the implementation gates](#runtime-preflight-and-implementation-gates).
Prepare0015 implements symbolic-length Prepare/CPU/Engine execution: fixed T/features, one named B,
homogeneous FLOAT32/FLOAT64 forward and actual Compiler-generated reverse, through existing [owners](../../glossary.md#term-owner)
and numeric protocols. Owning validation and independent code/structural review passed; root validation and the
sole bounded probe completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending. This does not qualify all-valid
symbolic FLOAT64 or implement Engine memoization.
Variable T/features, higher-order, low/wider CPU [dtype](../../glossary.md#data-type--datatype), native/mixed, [vector](../../glossary.md#term-vector)/parallel,
cold/copy/wire and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training reuse require separate qualification.

#### Symbolic templates, roles and ports

Each value has fixed extents or at most one [occurrence](../../glossary.md#term-occurrence) of the same named B; anonymous unknowns,
independent [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) symbols and unproved transformations of B reject. Sequence [axis](../../glossary.md#term-axis) zero remains
the common static T. Removing it yields the exact step-formal [shape](../../glossary.md#shape). [Carry](../../glossary.md#term-carry)/[invariant](../../glossary.md#term-invariant) actuals
match their formal's symbolic shape, [dtype](../../glossary.md#data-type--datatype) and [gradient](../../glossary.md#term-gradient) flag; next-carry shape/type matches its
carry. [Model](../../glossary.md#term-model) retains known [PURE](../../glossary.md#term-pure) operations and applicable dtype guards, independently of [Compiler](../../glossary.md#term-compiler)'s
bounded [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) first-order [admission](../../glossary.md#term-admission) and initial [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 [CPU](../../glossary.md#term-cpu) [qualification](../../glossary.md#term-qualification).
The symbolic branch does not turn the CPU whitelist into Model source semantics.

Symbolic template formals/results have unresolved numeric [layouts](../../glossary.md#layout), never a [canonical](../../glossary.md#term-canonical-form) layout
fabricated at a [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) B. Static formals/results keep their exact canonical guards. After [binding](../../glossary.md#term-binding),
all final-carry, stacked-emission and stacked-[residual](../../glossary.md#scan-residual-port-and-history-demand) ports have actual canonical non-[view](../../glossary.md#term-view-meanings) layouts;
all ports, including latent/dead siblings, retain ordered [identity](../../glossary.md#term-identity-meanings), shape/type and conservative
outer gradient OR. Repeated role bindings do not merge wrappers or contributions. Existing scoped
containment, storage-free formals, explicit [constants](../../glossary.md#term-constant), lexical IDs and demand rules still apply.
Compiler independently validates every local declaration, actual/formal/result binding and saved
site; [Prepare](../../glossary.md#prepare) preserves that complete projection rather than equating local and outer numeric IDs.

Named-axis [provenance](../../glossary.md#provenance), not equal numeric sizes, distinguishes row-dependent from shared values.
`[B,F]` transposes to `[F,B]`; reducing the identified axis removes B. A fixed [parameter](../../glossary.md#term-parameter) `[F,H]`
stays shared even at B=F=H, and its demanded history is `[T,F,H]`, not `[T,B,F,H]`.
Original recurrent row carries/steps and generated shared [accumulator](../../glossary.md#accumulator-type) positions stay distinct.
All-valid symbolic mode retains default ScanAttrs counts; do not use length-mode `rowCarryCount`
to label it, add a lengths actual, store B in the header or introduce another body representation.

#### Actual first-order closure and arithmetic

Only the implemented [CPU0074 terminal exception](#cpu0074-selected-terminal-grouping-exception)
qualifies ordered producing-site SUM/grouping language here for selected exec-v1 regions.
Ordinary outer exact SUM/[MEAN](../../glossary.md#term-mean) and all other arithmetic sites remain unchanged.

The bounded closure is [tensor](../../glossary.md#tensor)/[scalar](../../glossary.md#term-scalar) ADD/SUB/MUL/DIV, NEG/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid), [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul)/[transpose](../../glossary.md#term-transpose),
SUM/sumToShape, explicit [constants](../../glossary.md#term-constant) and necessary expand/[rank](../../glossary.md#term-rank) edits/[contiguous](../../glossary.md#term-contiguous)/same-[dtype](../../glossary.md#data-type--datatype) casts
under existing family guards. [Feature](../../glossary.md#term-feature)-only edits may introduce a [singleton](../../glossary.md#term-singleton) [axis](../../glossary.md#term-axis) but must preserve
the named B [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix); unproved reshape/mixing of B rejects. Qualify required rank-three outer
[seed](../../glossary.md#term-seed)/zero/constant/[pointwise](../../glossary.md#term-pointwise)/[layout](../../glossary.md#layout) plumbing for `[T,B,*]` through existing ordinary [owners](../../glossary.md#term-owner);
this is not arbitrary-rank [admission](../../glossary.md#term-admission). Forward validity grants no actual reverse capability.

Use the existing opposite-traversal sibling construction with joint final/emission [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings),
original-time histories, closed [derivative](../../glossary.md#term-derivative) constants and exact [primal](../../glossary.md#term-primal) [bindings](../../glossary.md#term-binding). Saved TANH/SIGMOID
outputs are loaded at their represented sites, never replayed. Shared [invariant](../../glossary.md#term-invariant) [gradients](../../glossary.md#term-gradient) use
[parameter](../../glossary.md#term-parameter)-shaped additive carries, not B-by-parameter banks; positional ties and cross-[occurrence](../../glossary.md#term-occurrence)
terms add once each. [Residual](../../glossary.md#scan-residual-port-and-history-demand)-root, unsupported-rule and higher-order/createGraph [preflight](../../glossary.md#term-preflight) still
reject before derivative allocation, including unsupported unrelated source branches.

For X `[T,B,F]`, h `[B,H]`, Wx `[F,H]`, Wh `[H,H]`, [bias](../../glossary.md#term-bias) `[H]` and
`hNext=tanh(xStep@Wx+h@Wh+bias)`, bind forward MATMUL M/K/N=B/F/H,
dX=B/H/F and dWx=F/B/H; dWh likewise contracts K=B with actual transposed [strides](../../glossary.md#term-stride).
Bias unbroadcast reduces B even at B=0/1; size equality cannot erase axis [identity](../../glossary.md#term-identity-meanings). For
`L=mean(finalCarry)`, preserve `oneLike(input).sum(...)`, restored/expanded division and actual
count B*H, never a [prototype](../../glossary.md#term-prototype) denominator. Preserve constituent [rounding](../../glossary.md#term-rounding), [NaN](../../glossary.md#term-nan)/Inf, [signed-zero](../../glossary.md#nan-infinity-and-signed-zero),
empty and saved-value semantics. Scan's typed producing-site SUM and ordinary outer exact
SUM/MEAN remain distinct algorithms; the accepted bounded [FLOAT64](../../glossary.md#term-float64) outer [qualification](../../glossary.md#term-qualification) and retained
[FLOAT32](../../glossary.md#term-float32) five-[limb](../../glossary.md#exact-superaccumulator-and-limbs) [specialization](../../glossary.md#term-specialization) belong to the [CPU owner](backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu); [owner glossary](../../glossary.md#term-owner))</span>.
sumToShape's singleton/raw-copy and empty
reduction cases remain [operation](../../glossary.md#operation)-correct; no new arithmetic latitude is selected.

#### Empty domains and numeric resources

T=0 requires its own fixed-T [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)/preparation, not a changing-T invocation of a T>0 plan.
No body or hoisted body [operation](../../glossary.md#operation) executes. Final [carry](../../glossary.md#term-carry) preserves initial [represented bits](../../glossary.md#term-represented-bits) in a
distinct [canonical](../../glossary.md#term-canonical-form) output, dInitialCarry equals its final [seed](../../glossary.md#term-seed), stacked emissions/residuals/dX
are empty, and [invariant](../../glossary.md#term-invariant)-[gradient](../../glossary.md#term-gradient) sums are typed zero. Full-body structural validity still holds.

B=0 empties row work, not the whole Scan/[run](../../glossary.md#term-run). Required shared carries/emissions, initializers and
[parameter](../../glossary.md#term-parameter)-shaped empty-[contraction](../../glossary.md#contraction-dimension)/SUM outputs still execute where nonempty. K=0 [MATMUL](../../glossary.md#matrix-multiplication--matmul) and
empty SUM produce typed zero without nonexistent row reads; [mean](../../glossary.md#term-mean)(empty finalCarry) is [NaN](../../glossary.md#term-nan).
Empty dX/dh0 does not excuse nonempty parameter gradients. Test B=0 and T=0 independently and
together, including shared work beside empty rows. Neither emptiness nor dead demand hides an
unsupported body or overflowing trailing domain. Geometry reads no lengths. CPU0042's fixed-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)
length execution has separate [producer](../../glossary.md#term-producer)-ordered value [preflight](../../glossary.md#term-preflight). Symbolic-B plus lengths has
implemented [Compiler](../../glossary.md#term-compiler) [qualification](../../glossary.md#term-qualification) below; Model0044 source alone did not grant it. Prepare0015's
combined execution is implemented through the existing [owners](../../glossary.md#term-owner). It changes none of the
numeric/resource algorithms in this section.

One compilation/preparation fixes code/routes/slots, representation positions and storage [topology](../../glossary.md#term-topology).
Numeric recipes bind actual B: [TBF](../../glossary.md#term-tbf) time [stride](../../glossary.md#term-stride) B*F, row frame B*H, demanded row history
T*B*localFeatures and demanded shared history T*sharedElements. Dead histories retain zero
bytes/stores. Exact checked [alignment](../../glossary.md#term-alignment)/[padding](../../glossary.md#padding), conditional empty scratch and B=0/1 cases may be
piecewise, not necessarily affine. Never use representative fake B, B_max banks, per-B relowering,
route/slot/topology selection or executable caching. Declared zero-byte workspaces retain their
structural slots/creators. Carry backing and inclusive cut lifetimes remain fixed; reuse a cut
only when [lifetime](../../glossary.md#term-lifetime) and size dominance hold for every admitted B, otherwise keep disjoint regions.

#### [Qualification](../../glossary.md#term-qualification) and mechanism gates

The implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
adds only its exact [CPU](../../glossary.md#term-cpu)-internal [payload](../../glossary.md#artifact-envelope-and-atomic-publication)/[dispatch](../../glossary.md#term-dispatch)/math seams and selected exec-v1 [identities](../../glossary.md#term-identity-meanings);
the no-extra-seam and [scalar](../../glossary.md#term-scalar)-only gates below otherwise remain in force.

Existing [owners](../../glossary.md#term-owner) and numeric protocols are selected, not new production types/public [signatures](../../glossary.md#term-signature).
[Compiler](../../glossary.md#term-compiler) owns [metadata](../../glossary.md#term-metadata) [binding](../../glossary.md#term-binding); [Prepare](../../glossary.md#prepare) owns scoped transport/stable declarations; CPU owns
numeric recipes/emission; [Engine](../../glossary.md#term-engine) owns entry [admission](../../glossary.md#term-admission) and the separately bounded
[geometry memo](runtime-prepare-engine.md#accepted-engine-geometry-memo). [Runtime](../../glossary.md#term-runtime) gets only
authenticated numeric geometry, no body/symbols/[memo](../../glossary.md#cache-and-memoization). Every invocation retains complete fresh
Compiler binding/private obligations, current input-backed [publication](../../glossary.md#term-publication) [descriptors](../../glossary.md#term-descriptor), physical
admission and fresh [mutable](../../glossary.md#term-mutable) arrays/resources. No per-[element](../../glossary.md#term-element)/time-step memo [lookup](../../glossary.md#term-lookup) is allowed.

Implementation acceptance must qualify real symbolic forward and actual Compiler-generated reverse
independently against nonuniform ordinary [Tensor](../../glossary.md#tensor) [unrolling](../../glossary.md#term-unrolling) and specialized clean Java: both
traversals/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), fixed [parameter](../../glossary.md#term-parameter) [gradients](../../glossary.md#term-gradient), local/outer ports, exact demanded/dead/shared bytes,
64→40→1→64 reuse, separate T=0, B=0/1, [overflow](../../glossary.md#term-overflow) before borrow, [aliases](../../glossary.md#aliasing-and-arena), cleanup/retry/close and
concurrent mixed-B results. Add affected-[module](../../glossary.md#term-module), [backend](../../glossary.md#backend)-[conformance](../../glossary.md#term-conformance) and Engine integration tests,
retained static compatibility and unsupported symbolic/length-boundary rejection tests, and generated Class-File checks for algorithm/hot-loop/
[dataflow](../../glossary.md#term-dataflow), hidden allocation/boxing/reflection/dispatch. Use only bounded warmed matched-work
generated/direct evidence with cold setup, admission and [kernel](../../glossary.md#kernel) costs separated; no speed promise.

Order remains authority review/MAIN acceptance → separately [Ready](../../glossary.md#term-ready) [Model](../../glossary.md#term-model) foundation with old-consumer
rejection → Compiler [inference](../../glossary.md#term-inference-meanings)/actual [AD](../../glossary.md#autograd)/all-port binding → coordinated Prepare/CPU/Engine vertical.
No successor becomes Ready automatically. Prepare0014 uses existing owners and the symbolic
[identity](../../glossary.md#term-identity-meanings) branch documented by the CPU [owner](../../glossary.md#term-owner), with no new Runtime [handoff](../../glossary.md#term-handoff) or supported signature.
The future Engine memo's [atomic publication](../../glossary.md#term-atomic-publication)/close mechanism remains a later scoped-brief decision.
CPU0073's exact [accepted internal mechanism and identity exception](backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
alone qualifies the new-type and retained-key/ClassFile gates here; noneligible controls stay
literal unchanged and Scan projection stays 71. It admits no new symbolic [dtype](../../glossary.md#data-type--datatype) or arithmetic.
If any other new
named type or changed Runtime handoff is necessary, stop for exact package/name/purpose/signature
and user approval; zero proposed types is not an implementation proof. The implemented Compiler0015
combination qualifies inference, actual first-order sibling AD and full fresh scoped all-port binding
together, not backend execution. Its numerical evidence uses existing test-local [evaluation](../../glossary.md#term-evaluation) of the
actual generated [graph](../../glossary.md#graph) against independent nonuniform active-prefix unrolling; only that [oracle](../../glossary.md#term-test-oracle)
inspects length values. Backend/emitter/performance gates above apply to separately admitted
execution, not this Compiler-only phase. Prepare0015 implements combined execution,
including the bounded ordinary typed/control/ROW(1)
prerequisite under the CPU owner. Both complete forward and actual reverse occurrences must qualify
independently in each dtype; no handwritten reverse substitutes for Compiler siblings. Existing
header/equations, original-source ROW(0) AD restrictions, generated mixed roles, exact saved bits,
length [preflight](../../glossary.md#term-preflight) and resource algorithms are unchanged. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training and all other excluded successors
remain closed; the future Engine memo is not part of Prepare0015.

### Length-aware whole-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) scan

The user approved the independently reviewed length-aware design on 2026-10-24. This section
defines source semantics, first-order reverse construction and the length execution protocol. MAIN accepted
Model0037's coordinated authority on 2026-10-24 after Hume's final overall independent Class C
APPROVE, zero open P1/P2. Model0038 implements the bounded [Model](../../glossary.md#term-model) source header/overload, row-role
[validation](../../glossary.md#term-validation) and structural [IR](../../glossary.md#term-intermediate-representation-ir) counts. Compiler0013 implements independent scoped [inference](../../glossary.md#term-inference-meanings),
first-order reverse construction and role/demand query emission for the subset below. It preserves
exact saved primals, normalizes each incoming SHARED-[target](../../glossary.md#term-target) [cotangent](../../glossary.md#cotangent) before accumulation and
binds row-shaped zero/literal [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) through explicit sequence roles. These are [compile](../../glossary.md#compile)-time
facts, not execution evidence. Prepare0011 implements static lengths/formal separation, boundary
role checks and complete-[graph](../../glossary.md#graph) demand projection through the existing staged resource [handoff](../../glossary.md#term-handoff).
It preserves broader forward sources and actual generated reverse mixed roles without duplicating
body row-algebra proof. CPU0042 implements fixed T/B/[features](../../glossary.md#term-feature) [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) forward and actual
[Compiler](../../glossary.md#term-compiler)-generated first-order reverse through generated [scalar](../../glossary.md#term-scalar), single-range [CPU](../../glossary.md#term-cpu) execution.
Independent code/docs review and MAIN's named [conformance](../../glossary.md#term-conformance)/[Engine](../../glossary.md#term-engine) [checkpoint](../../glossary.md#term-checkpoint) and single bounded
probe passed; MAIN accepted the capability on 2026-10-26. The
[CPU0042 Result](../../planning/backends/cpu/tasks/0042-length-aware-general-scan-execution.md#result)
records evidence and [nonstationary](../../glossary.md#term-nonstationary) timing limits, not a steady-state speed or parity claim.
Static [homogeneous](../../glossary.md#term-homogeneous) length-mode [cold reconstruction](../../glossary.md#term-cold-reconstruction) is implemented and accepted under
[Compiler0018](../../planning/modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md#result)
and the [Compiler cold contract](compiler-autograd.md#bounded-general-scan-cold-reconstruction).
[Checkpoint0014](../../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result)
implements its separate static length-aware v5 wire; symbolic cold/wire remains unimplemented.
[Model0044](../../planning/modules/model/tasks/0044-symbolic-length-scan-source-foundation.md)
records the user's 2026-10-26 approval of one cohesive Model source foundation, including the bounded
source-contract wording before Java. Mencius final pre-Java APPROVE (0 P1/P2, no missing contracts/[owners](../../glossary.md#term-owner))
and MAIN authority/[Ready](../../glossary.md#term-ready) acceptance are recorded on 2026-10-26.
The static-or-one-named-B source extension below is Complete: independent source/docs review and
MAIN's configured root test/[Javadoc](../../glossary.md#term-javadoc) checkpoint passed, with MAIN acceptance on 2026-10-26.
Compiler0015 implements the static/one-B length-mode intersection for independent inference,
actual first-order sibling [AD](../../glossary.md#autograd) and full fresh scoped all-port [metadata](../../glossary.md#term-metadata) [binding](../../glossary.md#term-binding). The Compiler-only
task is Complete. Prepare0015 implements static-or-one-B projection and combined FLOAT32/FLOAT64
CPU/Engine execution under the gates below. Owning validation and independent code/structural
review passed; root validation and the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending.
Static homogeneous cold [admission](../../glossary.md#term-admission) and inherited Checkpoint/[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) cold consumer [qualification](../../glossary.md#term-qualification) are
Complete under Compiler0018. V5 and the complete owned length-aware numerical workflow remain
separate capabilities.
The existing all-valid [PURE](../../glossary.md#term-pure)/type inventory, fixed scans, overloads and compatibility [identities](../../glossary.md#term-identity-meanings)
remain unchanged. Length-mode source validity never implies a downstream capability answer.

#### Header, [bindings](../../glossary.md#term-binding) and source [admission](../../glossary.md#term-admission)

The implemented Model0038/Model0044 baseline appended `int rowCarryCount` and [immutable](../../glossary.md#term-immutable) ordered
`List<OptionalInt> sequenceRowAxes` after the original seven ScanAttrs components, using zero rows
as the all-valid sentinel. The accepted source amendment below replaces that [canonical form](../../glossary.md#term-canonical-form) with
an explicit mode while preserving the seven-[argument](../../glossary.md#term-argument) all-valid convenience and Scan.apply overload.
Constructor migration is not preservation of record reflection, equality/[hash](../../glossary.md#term-hash) algorithms or
diagnostic text; none is a stable wire/[cache](../../glossary.md#term-cache) encoding. Old-default keys must not derive from
changed record text, and qualified all-valid/fixed [identity](../../glossary.md#term-identity-meanings) byte streams remain unchanged.

The implemented additional source-only overload is:

```java
ScanResult Scan.apply(List<Tensor> initialCarry, List<Tensor> sequences,
        List<Tensor> invariants, Tensor validLengths, int rowCarryCount,
        List<OptionalInt> sequenceRowAxes, ScanBody body, boolean reverse);
```

The implemented positive-ROW consumer baseline requires `1 <= rowCarryCount <= carryCount` and exactly sequenceCount non-null
optional entries. A present entry denotes ROW([axis](../../glossary.md#term-axis) in the step formal); absent denotes SHARED.
ROW means dependence on one identified [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix), not an [extent](../../glossary.md#term-extent) numerically equal to B.
Initially only ROW(0) and [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1) are admitted. Leading carries are ROW(0); remaining
carries are terminal SHARED additive accumulators. Invariants are SHARED. Actual order is [carry](../../glossary.md#term-carry),
sequence, [invariant](../../glossary.md#term-invariant), then one non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype) `[B]` validLengths actual, with **no body formal**.
Thus outer inputCount is checked `c+s+i+1`, while local formal count remains checked `c+s+i`.
All-valid counts remain `c+s+i`. Output groups and CompiledScanBody's representation are unchanged.

The implemented positive-ROW baseline infers common static T from sequence axis zero and B from row-carry axis zero.
Retain fully static source as a first-class path; alternatively B may be one exact named [dimension](../../glossary.md#dimension),
compared by [canonical](../../glossary.md#occurrence-identity-and-canonical-form) name rather than object [identity](../../glossary.md#term-reference-identity) or the spelling `B`. Removing outer time
yields each step formal. Every declared row axis has that exact Dimension; every row carry and
emission is row-leading. All other extents, including [features](../../glossary.md#term-feature)/state widths and SHARED [shapes](../../glossary.md#shape),
remain static. A symbolic row value contains B exactly once, at its proved ROW(0) or rank-two
ROW(1) axis; SHARED values, including invariants and terminal accumulators, cannot contain B.
Reject anonymous unknowns, independent/repeated/derived B, variable T/features and unproved
coordinate transformations, including in unused declarations and shape-bearing attributes.
Symbolic formals, actuals and ports retain unresolved numeric [layouts](../../glossary.md#layout); static formals and ports
retain canonical layouts, and existing static actual-layout admission is unchanged. No [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) B
or [runtime](../../glossary.md#term-runtime) binding is performed by source construction. Check lengths [dtype](../../glossary.md#data-type--datatype)/[rank](../../glossary.md#term-rank)/flag/exact B,
admitted shapes, exact formal bindings, role counts,
containment, all canonical siblings and checked arithmetic before canonical output IDs. Length
values are never read at construction. Repeated actual positions retain positional identity and
contributions, with compatible explicit roles; do not deduplicate them. Compatible tied actuals
identify the same outer batch axis, or are both SHARED; step axis r maps to outer axis r+1.
Lengths cannot [alias](../../glossary.md#aliasing-and-arena) a body formal. The existing outer gradient OR applies to
carry/sequence/invariant positions only;
lengths is discrete, never a [differentiation target](../../glossary.md#functional-gradient-request).

The Model0038/Model0044 source baseline required [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) or [FLOAT64](../../glossary.md#term-float64) body formals,
values and matching actuals, plus separate INT64 lengths. Model0045 supersedes that source-wide
dtype restriction and same-type-only CAST restriction as specified below. The closed algebra
and ordinary family [descriptor](../../glossary.md#term-descriptor) guards remain; this is [Model](../../glossary.md#term-model) source admission, not a [CPU](../../glossary.md#term-cpu) provider
query. All-valid source still admits its existing known [PURE](../../glossary.md#term-pure) operations and applicable Model dtypes.

#### Accepted empty-[carry](../../glossary.md#term-carry) and explicit length-mode source

MAIN accepts [ADR0059](../../design/decisions/0059-empty-carry-scan-and-explicit-length-mode.md)
and the implemented, source-qualified [Model0046 cutover](../../planning/modules/model/tasks/0046-empty-carry-scan-and-explicit-length-mode.md).
It supersedes only source representation/count/mode assumptions above, not current consumer
[qualification](../../glossary.md#term-qualification). Independent [Compiler](../../glossary.md#term-compiler)/[Prepare](../../glossary.md#prepare)/[CPU](../../glossary.md#term-cpu)/cold/raw-[codec](../../glossary.md#term-codec) guards remain bounded.

ScanBody permits an actual empty carry/nextCarry pair, with nonempty step sequences and at least
one nextCarry or emission root. Existing carry [shape](../../glossary.md#shape)/[dtype](../../glossary.md#data-type--datatype) stability, formals, containment,
[aliases](../../glossary.md#aliasing-and-arena), roles, [canonical](../../glossary.md#term-canonical-form) outputs and checked arithmetic remain. CompiledScanBody permits empty
nextCarry with nonempty semantic roots; ScanKind's structural minimum becomes one input.

ScanAttrs canonical components become `(timeCount, carryCount, sequenceCount, invariantCount,
emissionCount, residualCount, reverse, lengthAware, rowCarryCount, sequenceRowAxes)`.
Retain its seven-[argument](../../glossary.md#term-argument) all-valid constructor; migrate/remove every nine-argument sentinel
caller, with no alias/shim. Require checked T>=0, c>=0, s>=1, i/e/r>=0 and c+e>=1. Explicit false
requires rowCarryCount=0 and empty [axes](../../glossary.md#term-axis). Explicit true permits rowCarryCount in [0,c] and exactly
one existing optional role per sequence. inputCount depends on lengthAware, never the row count.
Lengths stays the final non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype)[B] actual, with no body formal.

Length source derives static/named B from lengths [metadata](../../glossary.md#term-metadata) before validating every explicit ROW
[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix), without reading lengths or guessing from coincident extents/first carry. Existing
row/algebra/activity rules stay unchanged. A no-ROW reverse can later use only SHARED floating
accumulators, or no carry when only floating sequence emissions are required. This authorizes no
dummy state, discrete [cotangent](../../glossary.md#cotangent), replay, [unrolling](../../glossary.md#term-unrolling), map [API](../../glossary.md#term-application-programming-interface-api) or additional execution [owner](../../glossary.md#term-owner).

The original positive-ROW consumer [checkpoint](../../glossary.md#term-checkpoint) rejected empty-carry/zero-ROW headers before
first-carry assumptions. Subsequent ordinary [inference](../../glossary.md#term-inference-meanings)/[AD](../../glossary.md#autograd) and CPU qualification supersede that
checkpoint independently. Current [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) v4/v5 [cold reconstruction](../../glossary.md#term-cold-reconstruction) and encoding/decoding
admit zero carry and zero ROW counts under their
[existing scoped wire rules](extensions-training.md#canonical-graph-model-version-4).
Mode remains [opcode](../../glossary.md#term-opcode)-derived on wire, with no new [boolean](../../glossary.md#data-type--datatype), opcode or [identity](../../glossary.md#term-identity-meanings)-wide bump.
Preserve static geometry, role/axis/dtype guards and the qualified nonempty [publication](../../glossary.md#term-publication) inventory.


<a id="prospective-per-value-typed-length-aware-source"></a>

#### Accepted per-value typed length-aware source

SOURCE COMPLETE — MAIN accepted 2026-11-01 after independent pre-Java and frozen
source/tests/[Javadoc](../../glossary.md#term-javadoc) review, including deliberate consumer rejection evidence. The
[Model0045 Result](../../planning/modules/model/tasks/0045-per-value-typed-length-scan-source.md#result)
records retained source-matched coverage and three approved Class A test-only recoveries;
original failed commands remain failed, with no clean root PASS claim.
This subsection supersedes only the historical [Model](../../glossary.md#term-model) source [dtype](../../glossary.md#data-type--datatype)/CAST restrictions; every other
source rule and downstream [qualification](../../glossary.md#term-qualification) remains unchanged.

Scan controls traversal, [bindings](../../glossary.md#term-binding) and row activity, not one global arithmetic dtype. Each body
value retains its own ordinary Model dtype. Length-aware Model source admits [FLOAT64](../../glossary.md#term-float64), [FLOAT32](../../glossary.md#term-float32),
[BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16), [INT64](../../glossary.md#data-type--datatype), [INT32](../../glossary.md#data-type--datatype) and [BOOL](../../glossary.md#data-type--datatype) per value where the existing closed row-preserving
[operation](../../glossary.md#operation) inventory and ordinary family guards permit them. [Carry](../../glossary.md#term-carry), sequence and [invariant](../../glossary.md#term-invariant) slots
may differ in dtype, including unused slots. An [identity](../../glossary.md#term-identity-meanings) ROW(0) BOOL or integral carry is valid;
this does not admit BOOL arithmetic, new logical/comparison/selection operations, RELU or [SLICE](../../glossary.md#slice).
Trailing SHARED carries still require the existing terminal additive update; they do not acquire
identity-update permission merely because ROW carries have it.

Every corresponding actual/formal position still matches exact [shape](../../glossary.md#shape), dtype and [gradient](../../glossary.md#term-gradient)
eligibility; sequence matching removes only time. Each next-carry root retains its own declared
carry's exact shape and dtype. Scan inserts no promotion, conversion, common working dtype or
hidden cast. Ordinary body operations alone own their existing promotion, compute/[accumulator](../../glossary.md#accumulator-type),
conversion, [rounding](../../glossary.md#term-rounding) and saved-value rules. An operation's ordinary promoted result cannot change
the declared carry dtype. A trailing SHARED update remains its own `old.add(contribution)` with
contribution shape **and dtype exactly equal to that carry's**, even when ordinary ADD promotion
would otherwise preserve the root dtype (for example FLOAT64 old plus FLOAT32 contribution).
Scan validates this dtype equality explicitly rather than relying on global homogeneity.

An ordinary explicit CAST may change dtype while preserving the exact shape and proved row role.
Use the ordinary cast [descriptor](../../glossary.md#term-descriptor)/value/eligibility rules, not a Scan conversion policy or an
implicit cast at a carry binding. CAST must preserve terminal row-reduction [dependency](../../glossary.md#term-dependency) facts and
SHARED-carry dependency isolation: casting cannot launder a terminal contribution into row
feedback or allow a shared carry formal outside its own permitted ADD. Every intermediate and
[canonical](../../glossary.md#term-canonical-form) sibling remains validated; no cast-chain folding or saved-site change is authorized.

The existing static-or-one-named-B geometry, ROW(0)/[rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1)/SHARED mappings, terminal
rules, [feature](../../glossary.md#term-feature)-[overflow](../../glossary.md#term-overflow) checks, containment/effect/free-[leaf](../../glossary.md#term-leaf)/nesting guards and pre-output-ID
failure ordering remain unchanged, including T=0/B=0. Control lengths remains a distinct
non-gradient INT64[B] actual with no body formal, outside the input-gradient OR. For each outer
final/emission/[residual port](../../glossary.md#scan-residual-port-and-history-demand), preserve its own dtype and use `G && dtype.isDifferentiable()`;
G still includes unused carry/sequence/invariant actuals. Do not rewrite body/formal flags.

This amendment grants no [Compiler](../../glossary.md#term-compiler) [inference](../../glossary.md#term-inference-meanings)/[AD](../../glossary.md#autograd), [Prepare](../../glossary.md#prepare), [CPU](../../glossary.md#term-cpu), [cold reconstruction](../../glossary.md#term-cold-reconstruction), wire, [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) or
Training [admission](../../glossary.md#term-admission) for newly Model-valid occurrences. Existing independent [homogeneous](../../glossary.md#term-homogeneous) [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding)
consumer guards remain [fail-closed](../../glossary.md#term-fail-closed), including low/mixed interiors hidden behind F32 boundary
casts or unused declarations. Negative tests must reach each [owner](../../glossary.md#term-owner) directly through existing
raw [graph](../../glossary.md#graph)/attachment/query seams; an earlier Compiler rejection is not Prepare/CPU/[codec](../../glossary.md#term-codec) evidence.
No new production type, [signature](../../glossary.md#term-signature), wrapper, [registry](../../glossary.md#term-registry), execution architecture or dependency is
selected. All-valid paths, fixed [kernels](../../glossary.md#kernel), ordinary [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) `.rnn`/`.gru`/`.lstm`, static
`Sequence.forward` APIs, old [checkpoint](../../glossary.md#term-checkpoint) formats, [constants](../../glossary.md#term-constant) and numerics remain unchanged.

#### Bounded row-role analysis

Model0048 implements the accepted ordinary-[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) contract below, replacing the initial
source [operation](../../glossary.md#operation) whitelist and [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix)-only intermediate ROW(1) restriction. Its independently
reviewed source/forward-coordinate result does not widen typed [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd), [Prepare](../../glossary.md#prepare), [CPU](../../glossary.md#term-cpu), cold
or wire [qualification](../../glossary.md#term-qualification). Retained physical recipes and old constructor/wire intersections keep
their independent [admission](../../glossary.md#term-admission) until separately qualified.

The single [descriptor](../../glossary.md#term-descriptor)-only semantic seam belongs to existing `model.tensor.Scan`:

```java
public record ValueSemantics(OptionalInt rowAxis, boolean terminal, List<Integer> directInputs) {}
public static ValueSemantics declaredSemantics(
        TensorDescriptor descriptor, OptionalInt rowAxis, Dimension batch);
public static List<ValueSemantics> ordinarySemantics(Operation operation,
        List<TensorDescriptor> inputs, List<ValueSemantics> inputSemantics,
        List<TensorDescriptor> outputs, Dimension batch);
```

Ordinary family guards remain with [Model](../../glossary.md#term-model) constructors; raw Compiler independently infers and
compares every [canonical](../../glossary.md#term-canonical-form) descriptor. Only then do both recompute these semantic facts. The seam
derives neither ordinary result [shapes](../../glossary.md#shape) nor promotion and creates no [Tensor](../../glossary.md#tensor), [producer](../../glossary.md#term-producer) or [identity](../../glossary.md#term-identity-meanings).
Facts are not a persisted/source admission [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions), another [graph](../../glossary.md#graph), [registry](../../glossary.md#term-registry) or proof framework.
Migrate Model [validation](../../glossary.md#term-validation), raw scoped [inference](../../glossary.md#term-inference-meanings) and [derivative](../../glossary.md#term-derivative) role discovery together; do not
retain [aliases](../../glossary.md#aliasing-and-arena) or a second transfer implementation.

ROW([axis](../../glossary.md#term-axis)) identifies exactly one original B coordinate at any valid intermediate [rank](../../glossary.md#term-rank)/axis.
SHARED has no row axis and fully static shape. Terminal implies SHARED. Direct inputs are
[immutable](../../glossary.md#term-immutable) increasing distinct producing-[occurrence](../../glossary.md#term-occurrence) positions, preserving tied operands as
separate positions. They include conditions, indices, [masks](../../glossary.md#term-boolean-mask) and [targets](../../glossary.md#term-target), not [cotangent](../../glossary.md#cotangent) eligibility
or every descriptor consulted for shape/promotion. Declared facts have no producing-input routes.

Check complete bodies, unused declarations and all canonical siblings even at T=0/B=0. Compare B
[Dimensions](../../glossary.md#dimension) structurally; only the identified row axis may be named and all other axes are static.
Equal F=B/H=B, B=0/1, zero [element](../../glossary.md#term-element) counts and [gradient](../../glossary.md#term-gradient) flags never infer or erase row identity.
Check fixed-[feature](../../glossary.md#term-feature) products independently of T/B emptiness and retain legitimate zero-feature
domains. In ordinary right [alignment](../../glossary.md#term-alignment), a SHARED operand must omit the row coordinate or supply a
static [singleton](../../glossary.md#term-singleton) there. An independently indexed shared [lookup](../../glossary.md#term-lookup)-table axis is not an aligned B
axis merely because its size equals B.

For every output row b, each contributing ROW input access must use that same identified b.
Apply ordinary contributor mappings, not a mathematical-kind whitelist:

- Pointwise, comparison, classification, logical, WHERE and CAST preserve compatible broadcast
  coordinates. CONTIGUOUS preserves logical coordinates. Permutation/rank edits map explicit axes.<span data-glossary-navigation="true"> ([Pointwise glossary](../../glossary.md#term-pointwise); [CONTIGUOUS glossary](../../glossary.md#term-contiguous); [Permutation glossary](../../glossary.md#permutation))</span>
- Reshape preserves the identified coordinate boundary and checked fixed feature products on each
  side, not merely total count or a matching extent. Preserve an existing exact-shape row position.
  For a changed shape, each candidate must independently prove the same row mapping and both
  feature boundaries. If several axes pass that complete proof, choose the lowest axis
  deterministically; multiple proved coordinate aliases are not a failed proof. Reject when no
  axis is proved. Never use B=0/1 as a substitute for boundary/overflow checks. Under the accepted
  general ordinary-unit cutover below, this is the additional length-mode ROW proof. Ordinary
  all-valid symbolic reshape instead requires proved canonical element-count equality and the
  declared-symbol/descriptor guards; it may move B through fixed features without claiming ROW
  provenance. Historical qualified backend restrictions remain independently effective until cutover.<span data-glossary-navigation="true"> ([extent glossary](../../glossary.md#term-extent); [overflow glossary](../../glossary.md#term-overflow); [symbol glossary](../../glossary.md#term-symbol))</span>
- Feature-only slicing, selection, updates, padding, tiling, composition, indexing, ordering,
  cumulative operations, reductions, normalization, windows, convolution, pooling, losses and
  attention are legal whenever their actual ordinary mapping preserves that coordinate. Reject
  partial/reversed/arbitrary row access, row neighborhoods or normalization over B, not the family.
  Unknown/custom semantics, effects, RNG/dropout, fixed recurrence and nested scans remain excluded.<span data-glossary-navigation="true"> ([padding glossary](../../glossary.md#padding); [tiling glossary](../../glossary.md#tiling); [normalization glossary](../../glossary.md#term-normalization); [convolution glossary](../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../glossary.md#term-pooling); [RNG glossary](../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../glossary.md#dropout--inverted-dropout))</span>
- A known entirely SHARED PURE occurrence has no row-coordinate reason to reject by kind. Its
  ordinary guards and terminal/shared-carry restrictions still apply; it cannot invent a row axis.<span data-glossary-navigation="true"> ([PURE glossary](../../glossary.md#term-pure))</span>
- Grouped [Conv2d](../../glossary.md#convolution--conv2d)'s actual [VJP](../../glossary.md#term-vector-jacobian-product-vjp) may permute `[B,G,O,P]` to `[G,B,O,P]`, use batched MATMUL and reduce
  axis one. Arbitrary proved intermediate axes/ranks are necessary; numerical derivatives remain
  in the ordinary owner. Sequence-header/formal admission is separate: this amendment alone does
  not widen ScanAttrs' declared 0/1 axes or direct Prepare's retained rank-two ROW(1) boundaries.<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>

Compute each canonical sibling separately. BatchNorm training dependencies are normalized
slot0:[0,1,2], next-[mean](../../glossary.md#term-mean) slot1:[0,3], next-[variance](../../glossary.md#term-variance) slot2:[0,4], saved slots3/4:[0].
Attention output depends on Q/K/V/optional mask; [weights](../../glossary.md#term-weight) depend on Q/K/optional mask, not V.
Ordinary [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-prefix [broadcasting](../../glossary.md#broadcasting) may nevertheless give weights a row domain from V's shape.
Domain [provenance](../../glossary.md#provenance), numerical/control routes and ordinary validation must not be conflated.

Only established active contributor meanings eliminate ROW: represented SUM/sumToShape,
matched [MATMUL](../../glossary.md#matrix-multiplication--matmul) [contraction](../../glossary.md#contraction-dimension) and functional shared-target ADD-scatter. [Feature](../../glossary.md#term-feature) reduction shifts
the axis only by removed preceding features; equal aligned sumToShape extents retain it even
at B=1, unlike explicit row-axis SUM. Do not infer active MEAN/variance/[softmax](../../glossary.md#softmax--log-softmax)/order/[loss](../../glossary.md#loss)-divisor
semantics. Shared-target ADD-scatter includes its base once, applies eligible updates in original
logical order and preserves raw unaddressed cells; its complete active index domain must validate
before writes. Replacement/MUL terminal meanings do not follow.

Terminal taint follows each output's actual routes and never feeds ROW work. A terminal
ADD-scatter base is legal only at shared targets, not as terminal-derived row indices/updates.
ROW next-[carry](../../glossary.md#term-carry) and emission boundaries remain ROW(0), nonterminal, never a SHARED expand guessed
from size. Each trailing SHARED carry occurs syntactically only as the exact left/old operand of
its own distinct tensor old.add(contribution) root. Contribution shape/type match exactly and
depend on no SHARED carry. Check every syntactic use before consulting per-output routes; an
unused numerical route cannot conceal an illegal carry use. Previous-state residuals remain legal.

This source semantics changes no operation's ordinary numerical meaning outside length-aware
Scan and grants no [backend](../../glossary.md#backend), derivative or [persistence](../../glossary.md#term-persistence) capability. Cold/wire owners must preserve
their effective old operation/rank/role/attribute intersection explicitly when retiring the old
transfer helper; static [homogeneous](../../glossary.md#term-homogeneous) types alone are insufficient to preserve that boundary.

#### Activity, raw [identity](../../glossary.md#term-identity-meanings) and retained histories

The implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
qualifies the increasing-b requirement only by fixed row partitions and range-index merge at
selected exec-v1 terminals; activity dominance and raw/saved-history rules below are unchanged.

Define `active(t,b) = t < L[b]`, `active(t) = exists b: active(t,b)`. Forward visits original
t=0..T−1; reverse visits t=T−1..0, hence each row's valid prefix L[b]−1..0, never its padded tail.
At a time, [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) contributors are increasing b. The activity gate dominates padded sequence,
[seed](../../glossary.md#term-seed) and row-history loads and dependent arithmetic. Result masking, `0*x` and compute-then-where
are not skipping. Inactive row [carry](../../glossary.md#term-carry) is raw identity; inactive emissions/dSequence are exact
[positive zero](../../glossary.md#nan-infinity-and-signed-zero). Empty-row final carry is raw initial carry and dInitialCarry is raw dFinalCarry
seed. Active [NaN](../../glossary.md#term-nan)/Inf retain ordinary family meaning; [padding](../../glossary.md#padding) cannot contaminate active results.

Retain [canonical](../../glossary.md#term-canonical-form) residuals and ordinary use/[publication](../../glossary.md#term-publication) demand. Save previous carry before the
active step and produced primals at exact represented sites, without replay. ROW([axis](../../glossary.md#term-axis)) histories
remain `[T]+localShape`, storing active row cells and raw positive-zero inactive cells. SHARED
histories remain `[T]+localShape`, with **no added B axis**: save exact values only at active(t),
otherwise raw positive zero. Shared [accumulator](../../glossary.md#accumulator-type) history is its pre-time value. Read row
histories/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) only at active(t,b), shared histories only at active(t); prior zero-fill does not
permit padded reverse reads. Shared hoisting/publication also needs any-active/active-time gates:
all-zero lengths, T=0 or B=0 perform no body work. Undemanded history costs zero bytes/stores.

The implemented [CPU0076 scalar load-placement exception](backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)
does not relax these activity scopes: true invocation-[invariant](../../glossary.md#term-invariant) cells and SHARED original-time
history cells require different lifetimes. Worker-local staging also requires an active consumer
in that range. Its conservative source scopes exclude ROW histories even if a consumer [view](../../glossary.md#term-view-meanings)
selects a [constant](../../glossary.md#term-constant) row: source activity remains active(t, selectedRow), not merely active(t).
Neither zero [stride](../../glossary.md#term-stride) nor [scalar](../../glossary.md#term-scalar) [broadcasting](../../glossary.md#broadcasting) proves readiness or temporal invariance. Complete
body/[lifetime](../../glossary.md#term-lifetime) proof failures retain checked reads. This adds no Scan semantic exception or
across-time carry promotion.

For example, shared `w.permute(1,0)` `[F,H]` retains `[T,F,H]`, not `[T,B,F,H]`, even at F=B.
Output seed `[T,B,H]` is ROW(0); a transposed row [residual](../../glossary.md#scan-residual-port-and-history-demand) is ROW(1). Reverse construction derives
these roles from checked original [bindings](../../glossary.md#term-binding)/[dataflow](../../glossary.md#term-dataflow), pruning sequence axes together with
usedStep/usedSequences. Ordered roles live in reverse ScanAttrs and survive demand refresh,
query/projection and future cold/v5 roundtrip, not just temporary [AD](../../glossary.md#autograd) labels.

#### First-order length-mode reverse

Only the implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
qualifies the ordered row-contributor grouping below; [feature](../../glossary.md#term-feature) K, sequential time and one distinct
shared [carry](../../glossary.md#term-carry) ADD per active time remain mandatory for every realization.

Before [derivative](../../glossary.md#term-derivative) allocation, original length-aware sources qualify only when
`rowCarryCount == carryCount` and **every original sequence** is explicitly ROW(0), including
unused/non-[gradient](../../glossary.md#term-gradient) positions. Original SHARED carries or SHARED/ROW(1)/other nonzero-[axis](../../glossary.md#term-axis)
sequences reject for [AD](../../glossary.md#autograd). Broader forward source validity grants no AD. Generated reverse inputs
may mix explicit ROW(0), ROW(1), SHARED histories/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings); that original-source restriction does
not apply to them. No differentiation of generated shared-carry reverse scans is admitted;
existing [residual](../../glossary.md#scan-residual-port-and-history-demand)-root and higher-order/createGraph rejection is unchanged.

Reuse ordinary derivative formulas and exact [primal](../../glossary.md#term-primal) [bindings](../../glossary.md#term-binding). Carry the same lengths actual
into the opposite-traversal sibling, with row [cotangent](../../glossary.md#cotangent) carries first and [invariant](../../glossary.md#term-invariant)-shaped
SHARED gradient accumulators appended. Preserve joint final/emission seeds, false-flag internal
carry propagation, closed [constants](../../glossary.md#term-constant), tied positional contributions and cross-[occurrence](../../glossary.md#term-occurrence) additions.

At active(t), [parameter](../../glossary.md#term-parameter) SUM/sumToShape and transposed [MATMUL](../../glossary.md#matrix-multiplication--matmul) reduce only active [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) contributors
in increasing b. The gate dominates both [contraction](../../glossary.md#contraction-dimension) operand loads and each contributor's
dependent arithmetic. Start each output contribution at the family's typed zero; preserve
ordinary multiply/accumulation/result sites, then perform **one distinct shared carry ADD per
time** at its own [dtype](../../glossary.md#data-type--datatype)/[rounding site](../../glossary.md#term-rounding-site). Never fuse it into contraction or update per row.
At inactive(t), shared state is raw unchanged, not old+0. Feature contractions retain ordinary
domains. Inactive cotangent carries bypass the body, including poisoned emission seeds; an
empty row's final [seed](../../glossary.md#term-seed) copies to dInitialCarry without entering parameter sums.

Use one invariant-shaped [accumulator](../../glossary.md#accumulator-type) plus bounded same-[shape](../../glossary.md#shape) contribution/old-new scratch as
required, never a B×parameter gradient bank. Demanded shared history can cost O(T×P), row history
O(T×B×retained-row-[elements](../../glossary.md#term-element)). New length-mode reductions do not promise bit-identical results
to a differently grouped whole-batch unroll; old all-valid arithmetic/bytes remain unchanged.

#### [Runtime](../../glossary.md#term-runtime) [preflight](../../glossary.md#term-preflight) and implementation gates

The implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
qualifies [scalar](../../glossary.md#term-scalar)/[vector](../../glossary.md#term-vector)/parallel and no-per-step-[dispatch](../../glossary.md#term-dispatch)/allocation restrictions below only for
the selected exec-v1 seam, scheduler [metadata](../../glossary.md#term-metadata) and bounded [feature](../../glossary.md#term-feature)-[transpose](../../glossary.md#term-transpose) staging. All other
preflight, activity, source/[AD](../../glossary.md#autograd), resource and unsupported-domain gates remain unchanged.

For the separately qualified symbolic-B extension, geometry [admission](../../glossary.md#term-admission) reads no lengths. Validate
all actual B values after [producers](../../glossary.md#term-producer) and before this Scan's first output/[padding](../../glossary.md#padding)/history/scratch
write, even for an invalid last value or T=0 (all lengths must be zero). B=0 has an empty length
domain and no body work, but other/shared outputs retain their semantic [identities](../../glossary.md#term-identity-meanings); never skip
an entire [run](../../glossary.md#term-run) because one domain is empty. T=0 all-valid [carry](../../glossary.md#term-carry) preserves raw initial values,
dInitialCarry preserves dFinalCarry and [invariant](../../glossary.md#term-invariant) sums are typed zero. Empty geometry bypasses
no structural/validity check. Scan-local rejection rolls back no upstream writes. Existing static
length semantics below remain unchanged. Lengths imply neither [model](../../glossary.md#term-model)-wide masking nor a changed
outer [MEAN](../../glossary.md#term-mean) denominator; Training accumulation remains a SUM of supplied [gradients](../../glossary.md#term-gradient).

After length producers execute, recheck accessible typed spans/[alignment](../../glossary.md#term-alignment)/overlap and validate
**all B lengths** in `[0,T]` before this Scan's first output, padding, history or scratch write.
An invalid last length rejects before earlier rows write; T=0 still validates all lengths.
This is scan-local rejection, not upstream [rollback](../../glossary.md#term-rollback). [Borrowed](../../glossary.md#ownership-borrowing-and-lease) storage remains stable during run.
The [INT64](../../glossary.md#data-type--datatype) boundary is separate from [homogeneous](../../glossary.md#term-homogeneous) floating body boundaries, not an invariant formal.

Reuse existing staged [Prepare](../../glossary.md#prepare) and the generated [CPU](../../glossary.md#term-cpu) scan path, not the fixed executor. Declare
all resources before slots; keep changing [bindings](../../glossary.md#term-binding)/banks in [RunState](../../glossary.md#run-state--runstate) and code/recipes O(body).
Qualify forward and actual reverse independently for both [dtypes](../../glossary.md#data-type--datatype), gate hoisted/shared/row work
at the proper activity [scope](../../glossary.md#term-scope), and retain every demanded saved site. No packing, sorting,
compaction, per-step allocation/[callback](../../glossary.md#term-callback), new executor, vector, parallel, [OpenBLAS](../../glossary.md#term-openblas) or [Metal](../../glossary.md#term-metal) route.
Version changed length-mode [canonical](../../glossary.md#term-canonical-form)/[specialization](../../glossary.md#term-specialization) identities and every baked role/demand/
geometry fact independently from the wire; preserve old-default/fixed keys and generated bytes.

The later [CPU0073 accepted mechanism](backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
permits only its exact CPU types/[signatures](../../glossary.md#term-signature) and selected `|scan-row-map-multi-reduction-v1=`
appendage to the existing baked mode [payload](../../glossary.md#artifact-envelope-and-atomic-publication) in both streams, retaining Scan projection 71.
Existing canonical identities and noneligible ClassFiles/keys/resources stay unchanged; this
qualifies retention references here only for selected eligible [fusion](../../glossary.md#term-fusion), not a semantic exception.
Length/activity/raw-[identity](../../glossary.md#term-identity-meanings)/saved-history and distinct per-time carry-ADD rules are unchanged.
CPU0073 implements this mechanism; independent code/tests/[Javadoc](../../glossary.md#term-javadoc) review and MAIN's named
[conformance](../../glossary.md#term-conformance)/[Engine](../../glossary.md#term-engine) join are accepted on 2026-10-30. No new Runtime/Engine/Prepare protocol or
measurement grant follows; the linked CPU mechanism owns its bounded eligibility and evidence.

Authority review/MAIN acceptance preceded the separately [dependency](../../glossary.md#term-dependency)-verified Model0038 foundation.
Model0038 supports static source construction and structural CompiledNode counts. Model0044's
implemented source-only extension retains the same overload, header, ordered roles and counts.
Compiler0015 implements the support-gate [qualification](../../glossary.md#term-qualification) accepted on 2026-10-26: one exact
named B with discrete non-gradient INT64[B] lengths, fixed T/[features](../../glossary.md#term-feature) and homogeneous [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64):
independent scoped [inference](../../glossary.md#term-inference-meanings), the existing actual first-order opposite-traversal sibling AD and
complete fresh scoped all-port metadata binding together. Final acceptance is tracked in its
[task result](../../planning/modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result).
Existing header/row/activity rules, original-source ROW(0) AD
restrictions, generated mixed roles, numeric/resource algorithms and public APIs remain unchanged.
[Compiler](../../glossary.md#term-compiler) must prove [Dimensions](../../glossary.md#dimension) by canonical value/name, not sampled extents; independently validate
all declarations, bindings and latent ports; retain per-[occurrence](../../glossary.md#term-occurrence) obligations and complete fresh
binding. [Compile](../../glossary.md#compile)/bind reads no input/length values and allocates no history or physical resources.
Original-source preflight restrictions still reject before [derivative](../../glossary.md#term-derivative) IDs; raw [IR](../../glossary.md#term-intermediate-representation-ir)/[descriptor](../../glossary.md#term-descriptor) seams
must independently reject invalid or unsupported metadata rather than rely on Model constructors.
Prepare0015 implements
static-or-one-named-B length projection, combined homogeneous FLOAT32/FLOAT64 generated scalar,
single-range forward/actual reverse and existing Engine admission/[publication](../../glossary.md#term-publication). Its bounded ordinary
outer [F64](../../glossary.md#floating-point-precision-and-rounding)/control/ROW(1) prerequisite is implemented under the
[CPU contract](backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract), not by general
F64 [MATMUL](../../glossary.md#matrix-multiplication--matmul) or all-valid symbolic F64 admission. The implementation preserves complete
pre-analyzer declaration/binding/latent-port/demand checks,
explicit [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-required mode even for `[B,0]`, and explicit B in fresh numeric packing rather than
derivation from a frame or feature [extent](../../glossary.md#term-extent). INT64 lengths retains its separate eight-byte boundary.
No new production type, public signature, Runtime protocol, storage/[lease](../../glossary.md#term-lease)/dependency or [memo](../../glossary.md#cache-and-memoization) is added.
Combined canonical `:symbolic-length-v1` and baked `|scan-symbolic-length-v1=` in both specialization
streams bind dtype/roles/terminal/demand/control/numeric-pack/recipe facts, excluding actual B/L,
resolved bytes, slots and addresses. Old all-valid static/symbolic, static-length and ordinary [F32](../../glossary.md#floating-point-precision-and-rounding)
identity streams and generated bytes remain unchanged. Owning executable [validation](../../glossary.md#term-validation) and independent
code/structural review passed, including actual Compiler reverse, independent active-prefix ordinary
[Tensor](../../glossary.md#tensor) AD and generated/optimal-Java Class-File inspection. Root validation and the sole bounded probe
completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending; no speed or [JIT](../../glossary.md#term-just-in-time-compilation-jit)-parity claim follows.
Incidental Optional/index failures or successful old-wire encoding are not guards.
Compiler0013 adds scoped inference, bounded first-order reverse construction and complete final queries/demand,
without reading lengths or allocating history. Original-source AD restrictions above still apply
before derivative allocation; broader forward inference does not relax them. Prepare0011 now
validates static bindings, declared boundary extents, tied positions and ordered demand before
analysis, retaining exact saved-[primal](../../glossary.md#term-primal) ports and original local history [shapes](../../glossary.md#shape). Actual [layouts](../../glossary.md#layout),
including lengths, are preserved independently of [backend](../../glossary.md#backend) read predicates. Test-only analyzers
declare checked mixed-boundary resources before assignment; this proves transport/associations,
not physical allocation or execution. CPU0042 separately qualifies mixed-boundary binding,
[producer](../../glossary.md#term-producer)-ordered value preflight and generated active-prefix execution in both floating dtypes.
Cold [capture](../../glossary.md#term-capture-meanings)/reconstruction and [Checkpoint](../../glossary.md#term-checkpoint) encoding gates still explicitly reject length mode.
No accidental success, incidental index failure or seven-field v4 encoding substitutes for rejection.
Compiler0015's completion itself authorizes no runtime, cold/v5 or owned-Model successor.
Model0044 remains Complete for source construction and its consumer-rejection baseline only;
active Model frontiers are None. The user's subsequent “pokracuj” authorizes only Prepare0015's
bounded complete vertical. Its pre-Java gate was independently approved and MAIN-accepted on
2026-10-26; the bounded combined implementation is now frozen. Final task acceptance remains
MAIN-owned under the gates above.
Compiler0013 and Prepare0011 create no distant task cascade or automatic successor launch.

CPU0042 qualification covers zero/full/mixed/invalid lengths, both traversals/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), empty domains,
F=B role ambiguity, shared transpose and ROW(1) history pruning, terminal-only reduction,
pre-write failure, raw identity/+0, poisoned padding/[seed](../../glossary.md#term-seed) rejection, reuse/[concurrency](../../glossary.md#term-concurrency)/overlap,
old compatibility and independent active-prefix oracles. CPU hot paths require the specialized
optimal clean-Java algorithm/[dataflow](../../glossary.md#term-dataflow) [oracle](../../glossary.md#term-test-oracle), Class-File inspection and bounded matched-work
evidence. CPU0042 records those checks and their limits; Compiler0013 alone supplies no such
execution or performance qualification.
Cold/wire and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) admission remain with [their owner](extensions-training.md#canonical-graph-model-version-5)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>.
Lengths govern only this scan: downstream [bias](../../glossary.md#term-bias)/head/[loss](../../glossary.md#loss) remains ordinary dense algebra with
no model-wide [mask](../../glossary.md#term-boolean-mask) or changed MEAN denominator. `.rnn` migration and [GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) NN migration,
low-type CPU arithmetic and higher-order waivers remain excluded; fixed-family first-order [BPTT](../../glossary.md#term-backpropagation-through-time-bptt)
is separately specified above. NN0032's second-order CPU compile [regression](../../glossary.md#term-regression), [singleton](../../glossary.md#term-singleton)-CONCAT
copy and unproved mixed-engine compatibility remain separate gates.

### General ordinary-unit Scan cutover

Implemented coordinated Class C cutover — MAIN accepted Compiler0017/CPU0081 after the exact
[Ready](../../glossary.md#term-ready)/source freeze, independent authority/interface and implementation APPROVE0 reviews, actual
ordinary/selected-backward/typed/activity consumers and the unfiltered twelve-project test/[Javadoc](../../glossary.md#term-javadoc)
gate. This supersedes historical [homogeneous](../../glossary.md#term-homogeneous)/closed body and generated-history header restrictions
for the execution/[AD](../../glossary.md#autograd) continuation below. Cold/wire constructor intersections remain independent;
there is no [Metal](../../glossary.md#term-metal) Scan grant, Java low-arithmetic unlock or universal performance claim.

Both all-valid and length-aware bodies compose known ordinary [PURE](../../glossary.md#term-pure) semantics through existing
[owners](../../glossary.md#term-owner), including per-value floating/discrete types, casts, [canonical](../../glossary.md#term-canonical-form) saved siblings and all-SHARED
work. Ordinary [descriptor](../../glossary.md#term-descriptor), effect, lexical containment, stable [carry](../../glossary.md#term-carry), public eligibility, resource
and per-[backend](../../glossary.md#backend) numerical predicates still apply. No separate permanent Scan mathematical-kind
list is permitted. All-valid keeps ordinary whole-[Tensor](../../glossary.md#tensor) meaning: do not impose length-mode row
locality or fabricate a [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) on it. Length mode keeps the proved original-row,
terminal, shared-carry isolation and exact contribution rules above.

The existing [Model](../../glossary.md#term-model) [owner](../../glossary.md#term-owner) adds the scoped overload
`ScanBody.describeOrdinary(Operation, List<TensorDescriptor>, int, Optional<DynamicDimension>)`.
Its optional is the exact sole declared batch [symbol](../../glossary.md#term-symbol), or empty for a static template. Reuse the
same ordinary descriptor and factored source/deferred symbolic guards; validate every operand,
canonical result and [shape](../../glossary.md#shape)-bearing attribute even when counts are zero. Keep the three-[argument](../../glossary.md#term-argument)
ordinary deferred-[inference](../../glossary.md#term-inference-meanings) [API](../../glossary.md#term-application-programming-interface-api) unchanged. Source, observed pre-ID construction and independent
[Prepare](../../glossary.md#prepare) call the scoped [guard](../../glossary.md#term-guard); no Tensor/[producer](../../glossary.md#term-producer), sampled B, duplicate family validator or new
type is required. Reshape count equality uses Model's canonical [dimension](../../glossary.md#dimension) arithmetic, separately
from length-mode contributor-preserving row-boundary proof and fixed-[feature](../../glossary.md#term-feature) [overflow](../../glossary.md#term-overflow) checks.

The same existing descriptor [facade](../../glossary.md#term-facade) exposes
`ScanBody.closeOrdinaryLayouts(Operation, List<TensorDescriptor>, List<TensorDescriptor>)`
for cold completion of independently validated computed result [layouts](../../glossary.md#layout), and
`ScanBody.hasCanonicalContiguousResults(Operation)` for the existing geometry-independent generic
[materialization](../../glossary.md#materialization) fact. Move the ordinary [Compiler](../../glossary.md#term-compiler) closure decisions, rather than copying a family
inventory into [CPU](../../glossary.md#term-cpu). Preserve the generic classifier's exact set and its symbolic
`BatchBindingPlan` acceptance boundary; specialized [Conv2d](../../glossary.md#convolution--conv2d)/[Conv3d](../../glossary.md#conv3d--ncdhw) closure remains separate from
that generic fact. This is not a second Scan mathematical-kind list or [admission](../../glossary.md#term-admission) predicate.

Shared completion preserves types, Shapes, [gradient](../../glossary.md#term-gradient) flags, ordering and every resolved layout.
It reuses exact ordinary affine [views](../../glossary.md#term-view-meanings), moves the existing noncontiguous-RESHAPE materialization
decision and factors the existing positive-static crop layout recipe into its Model family owner.
Negative-step, empty, named and unproved extraction layouts remain unresolved. Ordinary Tensor
construction and three-/four-argument descriptor guards do not change. No Tensor, producer,
storage or identifier is created, and Compiler type/Shape/constraint inference stays independent.
CPU may overlay only computed local projection descriptors and matching memory requirements;
the captured body and exact activity [graph](../../glossary.md#graph) associations remain [immutable](../../glossary.md#term-immutable). Caller/formal layouts,
actual-only batch authority and independent physical [qualification](../../glossary.md#term-qualification) remain strict.

A legal [finite](../../glossary.md#term-finite)-time fold, including T=0, may publish an existing caller [seed](../../glossary.md#term-seed) directly as a
selected gradient. That value still requires a real complete-graph physical declaration and
the existing caller-owned representation/[publication](../../glossary.md#term-publication) path. Reuse the
[producerless published-value contract](runtime-prepare-engine.md#producerless-published-values)
for caller or [constant](../../glossary.md#term-constant) sources; do not retain a fictitious reverse Scan, insert COPY/[Scalar](../../glossary.md#term-scalar)/
dummy dependencies, allocate a substitute Tensor or skip the publication. [Target](../../glossary.md#term-target) [TensorId](../../glossary.md#tensorid),
[derivative](../../glossary.md#term-derivative) order, target index, [dtype](../../glossary.md#data-type--datatype), Shape, [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity) and caller [lifetime](../../glossary.md#term-lifetime) remain exact.

Support true empty carry and explicit length mode independently of floating or ROW carry count,
including generated no-ROW/empty reverse. No dummy state, discrete [cotangent](../../glossary.md#cotangent) or first-carry dtype/
batch inference. Each canonical value and positional boundary has its own ordinary dtype/layout;
carry updates retain exact own descriptors. Intermediate proved ROW([axis](../../glossary.md#term-axis)) histories require
sequence/formal support at every valid axis/[rank](../../glossary.md#term-rank), not only [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1). Complete canonical-B
and fixed-feature proofs remain; no equal-[extent](../../glossary.md#term-extent), empty-domain, sampled-B or B=1 shortcut.
Original length-AD sources still require all original carries and sequences ROW(0); broader
generated histories do not broaden those source [bindings](../../glossary.md#term-binding). Fixed-family first-order [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) uses
the shared recipe above; [residual](../../glossary.md#scan-residual-port-and-history-demand)-root and higher-order exclusions remain independent. Conv3d
uses the shared ordinary [convolution](../../glossary.md#term-channels-first-convolution-layers) adjoints,
not a Scan-specific exclusion or derivative. CPU [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) full-body forward and actual first-order
reverse composition is qualified for all-valid and length-aware scans in both traversal directions;
ordinary body geometry, row proof and complete backend-closure checks still apply. This does not
widen ordinary public named-B Conv3d admission or grant Metal Scan execution.

Validate the entire original lexical inventory before derivative allocation, including unused
and unrelated branches. Separate source validity from selected derivative constructibility.
Preserve exact selected source-root/input pairs, ordinary D/[ND](../../glossary.md#term-nd)/FC and ND-only versus disconnected
ZERO/ERROR policy. Compute finite-T carry/step/[invariant](../../glossary.md#term-invariant) routes and every legal length prefix,
including T=0 [identity](../../glossary.md#term-identity-meanings) and T=1's missing second transition. Do not replace them with Cartesian
root/target unions, unrestricted transitive closure, sampled lengths or a T-sized derivative [DAG](../../glossary.md#term-directed-acyclic-graph-dag).
Compact selected [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), cotangent channels, sequence emissions, [parameter](../../glossary.md#term-parameter) accumulators and [primal](../../glossary.md#term-primal)
histories retain explicit original declaration maps even when actual Tensor identities are tied.

[Metadata](../../glossary.md#term-metadata)-first and real expansion use the SAME ordinary derivative bodies and contribution
orchestration, including seed [normalization](../../glossary.md#term-normalization), cast-back, term normalization, ordered merging,
disconnected publication and separate temporal ADDs. Model checks and independent Compiler
inference prove every actual created [operation](../../glossary.md#operation) before derivative IDs, including canonical saved
siblings, [eager](../../glossary.md#term-eager-evaluation) unused temporaries and primal-only work. Descriptor reads and resolved-but-unused
handles do not demand history; reachable actual value use does. No fake Tensor/ID, shadow
derivative formula, source replay or numerical zero as an admission proof.

Structural cotangent roots are required carried-cotangent channels followed by selected
emission-seed formals. Scopes are finite unions of those roots, not numerical tests or a general
[Boolean](../../glossary.md#data-type--datatype) language. Each selected canonical-output [VJP](../../glossary.md#term-vector-jacobian-product-vjp) and all its generated prerequisites inherit
its incoming [scope](../../glossary.md#term-scope). Ordinary contribution-list merge executes no ADD, raw sole identity or the
existing ordered ADD for neither/one/both present terms. Temporal parameter accumulation is
different: each present completed contribution executes its distinct old.add(contribution),
including initial +0; absence preserves old raw. No has-ever-term or scatter-cell presence.

Activity is local to the exact generated [occurrence](../../glossary.md#term-occurrence). A retained external cotangent Tensor actual
initializes a present root regardless of its bits or its producer's locally absent work. Ordinary
Tensor publication exports numerical values, not structural activity; shared lengths alone do not
couple different occurrences' initial-presence bits. A missing [channel](../../glossary.md#term-channel) remains different from a
bound Tensor containing zero.

Complete selected routes proved impossible for every compatible legal prefix assignment may be
omitted uniformly; selected-pair [planning](../../glossary.md#term-planning) and actual construction must omit the same channels.
Once retained for any prefix, a bound Tensor seed is present when qualifying the consuming local
reverse over its legal prefixes. Source feasibility and this materialized reverse's FC and
target-aggregate ND-only obligations are separate proofs. Do not reuse producer-prefix absence
as consuming-root absence. Correlating prefixes also requires a proved canonical row mapping
through outer operations: identical lengths do not equate different rows, and an unproved mapping
does not itself invalidate ordinary mathematics.

Absence dominates validators, operand loads and arithmetic. Explicit zero, [cancellation](../../glossary.md#term-cancellation),
FLOOR/SIGN zeros and numerical WHERE/attention selection remain present. Inactive length rows
preserve numeric/control carry raw, and inactive emissions/sequence gradients publish [positive
zero](../../glossary.md#nan-infinity-and-signed-zero). All-valid empty Tensor occurrences remain present independently of scalar iteration count;
length B=0/all-zero/inactive-time no-body-work is separate. Terminal SUM, matched [MATMUL](../../glossary.md#matrix-multiplication--matmul) and
shared-target ADD-scatter exclude absent/inactive contributors before their operand/index loads
while retaining ordinary represented order, [rounding](../../glossary.md#term-rounding) and the separate completed per-time ADD.

The finite neutral compiled activity [carrier](../../glossary.md#term-storage-carrier) and its exact-body association belong to
[Planning](foundational-modules.md#general-scan-activity-transport). Compiler owns proof/[capture](../../glossary.md#term-capture-meanings);
Prepare/providers independently validate applicability, membership, bindings and ordinary
predicates. Scope metadata is not an admission [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) or a second numerical graph.
Rebuild/demand/optimization must preserve/remap it; no [CSE](../../glossary.md#term-common-subexpression-elimination-cse)/[fusion](../../glossary.md#term-fusion)/hoist may cross incompatible
activity, optional-merge or temporal-ADD meaning.

For each physical unit, complete all eligible active value-domain [validation](../../glossary.md#term-validation) after inputs are
ready and BEFORE any of that unit's numerical output/history/scratch writes. Earlier completed
units need no [rollback](../../glossary.md#term-rollback). Inactive/absent values are not inspected; ignored categorical targets,
noncanonical [BOOL](../../glossary.md#data-type--datatype), [gather](../../glossary.md#gather)/scatter component bounds, bounds-before-duplicates and nonempty-index/
empty-output behavior retain ordinary meanings. No new finite-input rule for normalization or
categorical [loss](../../glossary.md#loss) follows. All lengths still validate before this Scan's first write, including
T=0. Metadata admission reads no values.

Reuse one existing emitted time loop and existing ordinary physical/math owners. No interpreter,
prepared-body child executor, [runtime](../../glossary.md#term-runtime) unit/graph [lookup](../../glossary.md#term-lookup), per-step [callback](../../glossary.md#term-callback)/packing/allocation or
new scheduler. Typed physical storage, control workspace, cursor reset and actual method inventory
are owned by the [CPU realization boundary](backend-execution.md#general-scan-ordinary-unit-realization).
Absent activity introduces no empty identity fields; unchanged frozen recipes retain bytes/keys.
New compact controlled graphs have selective identity and are not mislabeled old recipes.
The independently old cold/source/v1-v5 constructor/type/header/role intersection stays closed;
compiled activity is never serialized. CPU low numerical arithmetic follows the independently
qualified parity cutover; Metal's separate first-production support remains mandatory.

