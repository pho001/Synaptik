# ADR 0012: Fixed recurrent scan without graph regions

## Status

Accepted — 2026-08-20

## Context

Synaptik's current recurrent NN containers accept snapshotted Java `long[]` valid lengths and
construct a static Tensor expression for each selected length pattern. They return compact
per-step output lists. A future API needs valid-length values to remain ordinary runtime Tensor
inputs so one compiled and prepared computation can be reused for different batches without
changing graph topology.

The current architecture has flat immutable Model expressions and compiled graphs. Compiler owns
capture, inference, validation, and autograd; Planning chooses backend ownership; staged backend
preparation owns lowering and route choice; Runtime invokes only cold-bound prepared actions.
There is no body capture, nested graph, region identity, free-variable capture, loop
intermediate representation, or Runtime graph interpreter.

The decision must therefore fix both the recurrent semantics and their lifecycle placement before
any public API is published. Adding only an operation kind would leave Compiler's closed inventory
and backend execution undefined. Adding a general body or region abstraction would invent a much
larger architecture whose ownership and identity rules have not been designed.

## Decision drivers

- preserve the current flat `TensorProducer` and `CompiledGraphModel` meanings;
- make runtime valid lengths compatible with fully static Shapes and reusable preparation;
- retain Compiler ownership of capture, validation, and fail-closed autograd;
- keep backend-specific loop lowering and execution out of shared Prepare and Runtime;
- validate invalid lengths without partially written published results;
- keep recurrent state, parameters, and side effects explicit;
- make graph and prepared-transition size independent of `time`; and
- preserve the current NN static APIs until migration is decided from executable evidence.

## Options considered

### General user-defined body or graph region

Model could publish a callback, lambda, Tensor body, nested graph, callable block, or region and
let callers define the recurrent transition. This would require new body-input/output identity,
free-variable capture, cross-graph ownership, nested inference, optimization, autograd, planning,
preparation, serialization, and Runtime contracts. None exists today. Publishing a partial region
surface would make later consumers invent incompatible semantics.

### Host-side unroll selected by runtime lengths

An adapter could read a Java `long[]` and construct a different Tensor expression for each length
pattern. This is the current static-container model, not runtime-value reuse. Graph construction
and compiled graph size can grow with `time`, and changing lengths changes topology.

### Dense masking around ordinary cell expressions

The graph could unroll a cell at every time step and use masks to select states and zero outputs.
That preserves dense values but still grows the graph with `time` and executes recurrent dot
products and gates for invalid coordinates. It therefore fails the selected graph-size and
genuine skipped-arithmetic requirements.

### Fixed first-class recurrent operation with backend-internal loop

Model defines a closed recurrent family with fixed cell equations and ordinary ordered inputs and
outputs. Compiler captures one node. Backend analysis lowers it once to a bounded loop plan and
finalization constructs one reusable executable. Runtime invokes the prepared bound action
without interpreting recurrence.

## Decision

Synaptik adopts the fixed first-class recurrent operation with a backend-internal loop.

One public call will create exactly one identity-distinct multi-output `TensorProducer`, and
Compiler capture will emit exactly one ordinary flat `CompiledNode`. The occurrence has no body,
region, nested graph, callback, captured Tensor beyond its ordered inputs, graph-local child
identity, free variable, or cross-graph ownership rule. A general loop, conditional, or region
system remains forbidden until another explicit architecture update.

The distinction is:

```text
declarative Model operation
  = immutable fixed transition semantics and ordinary Tensor inputs/outputs

runtime control flow
  = concrete-backend prepared loop implementing that operation
```

### Exact family and planned surface

The exact variants are `RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM`. Each has one immutable
`FORWARD` or `REVERSE` direction. The fixed equations and parameter packing match the current NN
cells: tanh RNN with separate input/hidden weights; reset-after GRU with reset, update, candidate
gate order and `candidate + update * (hidden - candidate)`; and LSTM with input, forget,
candidate, output gate order and explicit hidden/cell states. Each family has one optional packed
input-side bias.

Model task 0025E must publish exactly one direction enum, `RecurrentScanResult` with `outputs` and
`finalHidden`, `LstmRecurrentScanResult` with `outputs`, `finalHidden`, and `finalCell`, and six
Tensor receiver methods: bias-free and biased `rnnScan`, `gruScan`, and `lstmScan`. These names
and signatures are planned architecture, not current runnable Java. The complete exact planned
signatures are recorded in the [architecture contract](../../../ARCHITECTURE.md#fixed-family-and-planned-model-surface).

The receiver is the time-major input Tensor. Operation input order is:

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

Every result component is the canonical wrapper for its slot from the one shared producer.

### Static descriptors and runtime valid lengths

The input Shape is fully static `[time, batch, inputSize]`. Valid lengths are a fully static
rank-one `INT64[batch]` Tensor with `requiresGrad == false`. Initial hidden and LSTM cell are
`[batch, hiddenSize]`. Input weight is `[gateCount * hiddenSize, inputSize]`, hidden weight is
`[gateCount * hiddenSize, hiddenSize]`, and optional bias is `[gateCount * hiddenSize]`, where
`gateCount` is one, three, or four for RNN, GRU, or LSTM. `inputSize` and `hiddenSize` are positive;
`time` and `batch` may be zero. All value, state, weight, and bias roles share one exact floating
data type.

The dense output is `[time, batch, hiddenSize]`; final states are `[batch, hiddenSize]`. Model
construction leaves layout unresolved. Gradient eligibility is the OR of differentiable floating
roles, never the valid-length role.

Only valid-length values vary across compatible executions. Dynamic or binding-dependent time,
batch, feature, parameter, or output Shapes remain outside this program. A valid-length Tensor
does not authorize a hidden dynamic-Shape lifecycle.

### Traversal, padding, and failure

For each original batch row `b`, the backend validates `L[b]` in `[0, time]`. `FORWARD` consumes
coordinates `0 .. L[b]-1`; `REVERSE` consumes `L[b]-1 .. 0`. Reverse traversal never passes
through the padded suffix. Each valid coordinate stores the hidden state produced after consuming
that coordinate. Every padded coordinate stores exact positive zero in the common data type.

A zero-length row returns its exact initial hidden state semantically, plus its exact initial cell
for LSTM, and has zero output at every time. A zero-time input requires all lengths to be zero,
returns an empty dense output, and preserves the initial states semantically. Lengths are never
inferred from padding, values, NaN, token identifiers, labels, or storage.

Before any output representation is mutated, the executable backend validates the complete length
vector, all bounds, and representation-specific access preconditions. Failure cannot expose
partially written published results. Engine later owns public exception translation.

### Purity, state, and serialization

The operation is pure. All carried states are explicit inputs and final outputs. It owns no
hidden module, compiler, prepared, or runtime state; RNG; mode; counter; parameter or buffer;
mutation; callback; I/O; or external resource. Each run uses its own `RunState`.

NN retains parameter and persistent-buffer wrappers and state paths and supplies current parameter
Tensor bindings as ordinary scan inputs. NN state dictionaries contain those parameter and buffer
bindings, and future model checkpoints may persist their materialized values. Operations,
compiler graphs, prepared executables, Runtime state, and backend artifacts are rebuilt and are
not serialized by this decision.

### Layer ownership

- Model owns fixed semantics, kind/attributes, descriptor-visible validation, result metadata,
  canonical provenance, and the planned Tensor surface.
- Compiler owns one-node capture, inference, final validation, and forward inventory adoption. It
  rejects every backward-capable request reaching the family before constructing any gradient
  Tensor until BPTT is separately designed.
- Planning uses the existing ordinary capability query and chooses only backend ownership.
- Shared Prepare uses the existing static projection, staged analysis, exact declaration,
  assignment, and finalization contracts; it gains no loop/body contract.
- Runtime uses the existing representations, schedule, isolated state, and cold-bound invocation;
  it never inspects lengths or recurrence.
- Engine owns typed logical caller-input binding and typed publication mapping and never rebuilds
  or specializes the graph from length values.
- A concrete backend owns exact capability truth, one-time lowering, loop planning, complete
  length validation, physical work, and reusable executable construction.
- NN retains module composition and parameter/state bindings. A later NN/Data task owns
  runtime-valid-length convenience and schema integration without inferring lengths from values.
- Training retains optimizer and orchestration ownership. Compiler, not Training, owns the later
  BPTT decision and gradient construction.

Planning does not interpret recurrence, direction, lengths, active rows, loop parameters, or
routes. Runtime does not select loop count, compact rows, or inspect an operation or graph.

### Performance contract

One occurrence remains one compiled node independent of `time` and lengths; graph construction
and graph size for that occurrence are `O(1)` in `time`. Backend analysis prepares the transition
once and never constructs or retains one node, graph, executable, or body per step. The Runtime
hot path performs no reflection, string dispatch, scalar boxing, graph inspection, operation
dispatch, backend lookup, route selection, or per-step object-graph growth.

Invalid coordinates execute no recurrent dot products, gates, or state update. Branch-based
row/time traversal satisfies the first capability. Physical active-row compaction, sorting,
packed buffers, vectorized packed batches, workspace reuse, and claims that validation or zeroing
perform no work remain deferred backend choices requiring evidence.

## Rationale

The fixed operation is the smallest architecture that supports runtime lengths without weakening
current ownership. Model expresses one immutable meaning, Compiler and Planning retain ordinary
flat behavior, a concrete backend owns the only component that needs a loop, and Runtime stays a
graph-free schedule executor. It also permits genuine skipped invalid-coordinate arithmetic,
which dense masked unrolling cannot provide.

The closed family deliberately trades generality for an executable contract. A future general
region system can still be designed from concrete needs, but it must not inherit accidental body,
capture, or lifecycle rules from this operation.

## Consequences

### Positive

- Runtime length values can vary without graph reconstruction or preparation.
- Graph and prepared-transition size do not grow with `time`.
- The existing flat graph, staged Prepare, and cold-binding boundaries remain intact.
- Recurrent state and side effects remain explicit and compatible with isolated runs.
- Backend capability and skipped-work claims can remain exact and evidence-based.

### Negative and risks

- The first family is intentionally narrow and cannot express an arbitrary recurrent cell.
- Each backend must implement and validate each claimed variant/type/Shape/direction combination.
- Dense outputs require padded zero storage even though padded recurrent arithmetic is skipped.
- BPTT is unavailable until a later Compiler decision selects saved-state versus recomputation
  and derivative semantics.
- Fully dynamic Shapes and physical active-row compaction remain unsupported.

### Compatibility, migration, testing, and follow-up

Current `RnnSequence`, `GruSequence`, `LstmSequence`, and their bidirectional containers keep
their Java `long[]`, static-unroll, compact-output-list, provenance, and final-state contracts.
They are not redirected to the new operation. NN 0022 later decides overload, migration,
retention, or deprecation and must account for the new dense zero-padded original-time-aligned
output versus current compact per-step lists. Any bidirectional migration preserves independent
parameters, valid-prefix-only reverse traversal, forward-first feature concatenation, original-
time alignment, and typed final states.

Model 0025E owns the semantic family and Tensor surface. Compiler 0006A owns forward adoption and
the initial fail-closed BPTT boundary. A later concrete-backend task owns truthful execution.
Engine 0001–0002 own lifecycle composition and typed caller/output binding. NN 0021B coordinates
the executable owner sequence; NN 0022 owns Data-facing runtime-length API and compatibility.
Dynamic Shapes, arbitrary masks with holes, active-row compaction, BPTT, and general regions stay
separate.

The decision changes no module dependency or source inventory. Existing architecture tests must
pass but require no source update. No production API or executable behavior is introduced by this
ADR.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md#fixed-recurrent-scan-without-graph-regions)
- [Lifecycle](../../architecture/lifecycle.md#planned-fixed-recurrent-scan-through-the-lifecycle)
- [Module boundaries](../../architecture/module-boundaries.md)
- [Runtime, Prepare, and Backend boundary](../../architecture/runtime-prepare-backend-boundary.md#planned-fixed-recurrent-scan-handoff)
- [Training graph](../../architecture/training-graph.md#fixed-recurrent-scan-and-the-initial-bptt-boundary)
- [ADR 0002: Backend-owned lowering](0002-backend-owned-lowering.md)
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [ADR 0010: Staged backend preparation](0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership](0011-per-run-runtime-resource-ownership.md)
- [NN task 0021A](../../planning/extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)
