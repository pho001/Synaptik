# Fixed recurrent-scan contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns the complete fixed recurrent-scan family, public Model surface, semantics,
static-Shape/runtime-value boundary, operation-specific application of existing lifecycle
ownership, failure behavior, performance, and migration invariants. Generic compile, prepare,
run, backend, and training lifecycles remain with their respective scoped contracts. This file
does not authorize general graph regions or user-defined loop bodies.

## Fixed recurrent scan without graph regions

The first recurrent operation whose valid sequence lengths can change between runs is one fixed,
first-class Model operation family. It is an ordinary flat, identity-distinct, multi-output
`TensorProducer` occurrence before capture and exactly one ordinary flat `CompiledNode` after
capture. This decision does not authorize a user-defined callback, lambda, Tensor body, nested
`CompiledGraphModel`, callable graph, block, region, loop intermediate representation, captured
subgraph, or other body value.

```text
declarative Model operation
  = immutable fixed transition meaning plus ordered Tensor inputs and outputs

backend-internal runtime control flow
  = one prepared bounded loop implementing that meaning
```

The operation captures no Tensor reference beyond its ordered ordinary inputs and owns no body
input or output, free variable, nested graph-local identity, region identity, cross-graph
ownership rule, or loop condition. `Tensor`, `TensorProducer`, `Operation`, `CompiledNode`, and
`CompiledGraphModel` retain their current flat meanings. A general control-flow or graph-region
system remains forbidden until another explicit architecture update.

<a id="fixed-family-and-planned-model-surface"></a>

### Fixed family and current Model surface

The exact semantic variants are:

```text
RNN_TANH
GRU_RESET_AFTER
LSTM
```

Each occurrence has exactly one immutable `FORWARD` or `REVERSE` direction attribute. The first
family has no bidirectional, stacked, arbitrary-cell, configurable-gate, configurable-activation,
peephole, projection, recurrent-dropout, residual, stateful, sparse, or quantized variant.

The fixed transitions match the current NN cells. `RNN_TANH` uses one tanh transition, separate
input and hidden weights, and one optional shared input-side bias. `GRU_RESET_AFTER` uses reset,
update, and candidate gate order, applies reset after the recurrent candidate projection, computes
`candidate + update * (hidden - candidate)`, and has one optional packed input-side bias. `LSTM`
uses input, forget, candidate, and output gate order, explicit hidden and cell states, the current
fixed sigmoid/tanh equations, and one optional packed input-side bias.

Model publishes exactly one direction enum, a two-output recurrent result, a three-output LSTM
result, and one public final field-free `model.tensor.RecurrentScan` static namespace in the
existing Tensor package. The namespace has exactly the following six biased or bias-free
constructors, with the time-major input explicit and first. This is advanced low-level Java
expression-construction API, not a layer, module, execution service, registry, general scan body,
or runnable Compiler, Engine, Runtime, or backend API:

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
`RecurrentScanKind` and `RecurrentDirection` remain operation semantics. Every component is the
canonical wrapper from one exact shared producer in the order shown below.

Ordinary neural-network callers continue to use `RnnSequence`, `GruSequence`, and
`LstmSequence`, whose current Java `long[]` lengths select static unrolling and compact outputs.
A later NN runtime-length API may delegate to this namespace only after Compiler and backend
adoption. `Tensor.linear` remains unchanged because a linear projection is a natural one-input,
one-output last-axis contraction; the NN `Linear` module owns parameter state, while
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

### Static Shape and runtime-value boundary

The first executable capability has fully static Shapes while valid-length values remain ordinary
runtime inputs:

- input is rank three `[time, batch, inputSize]` in time-major order;
- valid lengths are a rank-one `INT64` Tensor `[batch]` with `requiresGrad == false`;
- initial hidden and the LSTM initial cell are `[batch, hiddenSize]`;
- input weight is `[gateCount * hiddenSize, inputSize]`;
- hidden weight is `[gateCount * hiddenSize, hiddenSize]`;
- optional bias is `[gateCount * hiddenSize]`;
- `gateCount` is one for RNN, three for GRU, and four for LSTM;
- `inputSize` and `hiddenSize` are positive, while `time` and `batch` may be zero; and
- input, states, weights, and optional bias have one exact common floating data type.

The dense output Shape is `[time, batch, hiddenSize]`, and each final-state Shape is
`[batch, hiddenSize]`. Layout remains unresolved at Model construction. Output gradient
eligibility is the OR of the differentiable floating input, state, and parameter roles; valid
lengths never contribute.

Dynamic or binding-dependent `time`, `batch`, `inputSize`, `hiddenSize`, parameter Shapes, or
output Shapes are outside this first program. Prepared execution must be reusable across
different valid-length values for the same compatible static descriptors, but not across
arbitrary Shapes. Runtime valid lengths must not be used to disguise a dynamic-Shape lifecycle.

### Valid lengths, traversal, and outputs

For each original batch row `b`, execution validates the runtime value `L[b]` in the inclusive
range `[0, time]`. Lengths are never inferred from input contents, padding values, zero, NaN,
token identifiers, labels, or storage.

`FORWARD` applies the fixed transition at original coordinates `0 .. L[b]-1`. `REVERSE` applies
the same transition at original coordinates `L[b]-1 .. 0`; it reverses only the valid prefix and
never traverses the padded suffix. In either direction:

- each valid coordinate stores the next hidden state produced after consuming that coordinate;
- each padded coordinate `t >= L[b]` stores the exact positive zero of the common data type;
- final hidden, and final cell for LSTM, is the state after the row's last executed transition;
- a zero-length row has positive-zero output at every time and returns its exact initial state
  values semantically; and
- when `time` is zero, every length must be zero, the dense output is empty, and the initial states
  are the semantic final states.

Padding is never a recurrent input. Before mutating any output representation, an executable
backend must validate the complete length vector, every bound, and any representation-specific
access precondition. Invalid lengths fail the run without partially written published results.
Engine owns any future public recurrent-execution exception translation; this architecture does
not select the public exception type.

### Purity and lifecycle ownership

The operation is functionally pure. All carried state is explicit in its ordinary inputs and
final outputs. It owns no hidden module, compiler, prepared, or runtime state; RNG; mode; counter;
`Buffer`; `Parameter`; mutation; callback; I/O; or external resource. NN continues to own
parameter wrappers and state paths, while the operation consumes their current Tensor bindings as
ordinary inputs. Different executions use isolated `RunState` instances under the existing
Runtime contract.

NN state dictionaries contain only parameter and persistent-buffer Tensor bindings. Future model
checkpoints may persist the materialized values of those entries. The scan operation, compiler
graph, prepared executable, runtime state, and backend artifacts are rebuilt and are not
serialized by this decision.

Owner boundaries are exact:

- Model owns operation kind and attributes, fixed semantics, descriptor-visible validation,
  result metadata, canonical multi-output provenance, and the current focused
  `model.tensor.RecurrentScan` expression surface.
- Compiler captures one ordinary flat node, owns inference and final validation, adopts the
  operation in its exact closed inventory, and initially rejects every backward-capable request
  that reaches the family before constructing any gradient Tensor.
- Backpropagation through time (BPTT) is a later explicit Compiler decision. No saved-gate output,
  tape, checkpointing or recomputation policy, backward operation, or derivative formula is
  selected here.
- Planning performs its existing ordinary operation capability query and selects backend
  ownership. It does not interpret recurrence, direction, valid lengths, active rows, loop
  parameters, or routes.
- Shared Prepare uses its existing static-Shape projection, staged backend analysis, exact shared-
  resource declaration, slot assignment, and finalization lifecycle. It gains no loop-body or
  control-flow contract.
- Runtime uses its existing caller-input representation, immutable prepared recipe, cold binding,
  schedule, isolated run state, and bound invocation contracts. It does not inspect valid
  lengths, select a loop count, compact rows, or interpret recurrence.
- Engine owns typed logical caller-input to ordered Runtime-representation binding and typed
  publication mapping. A valid-length value never causes Engine to specialize or rebuild a graph.
- A concrete backend advertises only exact implemented variant, type, Shape, and direction
  combinations. It lowers the occurrence once, declares all shared resources before slot
  assignment, and constructs the reusable executable during finalization.
- NN retains module composition and current parameter/state bindings; a later NN/Data task owns
  runtime-valid-length convenience and schema integration. Data does not own recurrence and must
  not infer lengths from values.
- Training retains optimizer and training-orchestration ownership. It does not implement BPTT,
  recurrent execution, saved-state policy, or a backend route.

The current Planning, Prepare, and Runtime shared contracts are sufficient for this static-Shape
ordinary-node design. A later implementation must stop and provide evidence if a concrete backend
reveals an actual shared-contract gap; it must not add speculative region or loop-body types.

### Performance and migration invariants

One Model scan occurrence remains one compiler graph node regardless of `time` or valid-length
values, so its graph construction and graph size are `O(1)` in `time`. Backend analysis prepares
the transition implementation once for that occurrence and must not construct, compile, or retain
one graph, node, or executable body per time step.

The runtime hot path performs no reflection, string dispatch, scalar-element boxing, graph
inspection, operation dispatch, backend lookup, route selection, or per-step object-graph growth.
Invalid coordinates execute no recurrent dot products, gates, or state update. A bounded branch-
based row/time traversal is sufficient for the first capability. Physical active-row compaction,
sorting, packed buffers, vectorized packed batches, workspace reuse, and claims that length checks
or zero initialization perform no work remain deferred backend route decisions requiring truthful
resource and performance evidence.

Current `RnnSequence`, `GruSequence`, `LstmSequence`, and the three bidirectional containers keep
their snapshotted Java `long[]`, compact-output-list, static-unroll, exact-provenance, and final-
state contracts unchanged. Model and Compiler implementation must not redirect those APIs to the
new operation. A later NN decision must address whether to add runtime-length overloads, migrate,
retain compatibility, or deprecate. It must acknowledge that the new scan returns dense zero-
padded original-time-aligned outputs, while current static containers return compact per-step
output lists. A host `long[]` adapter that builds a different graph for each length pattern is not
an implementation of this runtime-valid-length decision.

Any later bidirectional migration must preserve independent parameter ownership, valid-prefix-
only reverse traversal, forward-first final-axis concatenation, original-time alignment, and
type-specific final hidden and cell states.
