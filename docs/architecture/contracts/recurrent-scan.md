# Fixed and general differentiable-scan contract

<a id="fixed-recurrent-scan-contract"></a>

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract is the sole complete semantic owner for fixed recurrent scans and the bounded general
differentiable scan: Model surface, formals/results/residuals, lexical scopes, static-Shape/runtime-
value boundary, traversal, empties, purity, failure, performance and migration invariants. Compiler,
Planning, Prepare/Runtime, backend and extension contracts own only their respective boundaries;
they link here rather than independently redefining scan semantics. General graph regions and
arbitrary control flow remain forbidden. General scan has implemented Model source/IR, Planning
query transport, Compiler scoped capture/first-order AD/demand and Prepare scoped projection/static
binding/demand validation. CPU0030 implements bounded FLOAT32 same-shape pointwise execution and
its independently qualified actual Compiler-generated reverse sibling. CPU0031 adds qualified
homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded shape/layout closure, including actual
reverse bodies. CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL inside the emitted scan,
including independently qualified actual reverse bodies. CPU0033 implements homogeneous FLOAT64
across the complete bounded declared algebra and actual first-order reverse closure. The bounded
owned NN workflow is implemented under [its owner](extensions-training.md#owned-tensor-models);
padding and low-type execution remain excluded. Implemented bounded
[body persistence](extensions-training.md#canonical-graph-model-version-4)
changes no Scan semantics here. The
existing fixed family and executable routes remain unchanged.
The [accepted symbolic-batch extension](#accepted-symbolic-batch-scan) has implemented Model-only
static/bounded-symbolic source under Model0042, independently reviewed and MAIN accepted 2026-10-25.
Model0041's coordinated authority is Complete. Compiler0014's inference/actual first-order AD/
all-port binding is independently reviewed and MAIN accepted 2026-10-25 after compatibility validation.
Prepare0014 implements bounded FLOAT32 symbolic projection/execution; its task tracks final
qualification. Cold/copy/wire still requires static Scan shapes, and Engine memoization remains future.

## Fixed recurrent scan without graph regions

The first recurrent operation whose valid sequence lengths can change between runs is one fixed,
first-class Model operation family. It is an ordinary flat, identity-distinct, multi-output
`TensorProducer` occurrence before capture and exactly one ordinary flat `CompiledNode` after
capture. This fixed-family decision does not authorize a user-defined callback, lambda, Tensor body,
nested `CompiledGraphModel`, callable graph, block, region, loop intermediate representation, captured
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
`CompiledGraphModel` retain their flat meanings for fixed scans. The bounded exception in
[General differentiable scan](#general-differentiable-scan) does not turn fixed scans into bodies;
a general control-flow or graph-region system remains forbidden.

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

NN state dictionaries contain only parameter and persistent-buffer Tensor bindings. Cold semantic
scan graphs and named host values may be copied, saved, loaded and rebound under Checkpoint's
[generic source contract](extensions-training.md#generic-graph-source-and-values) and its owning
[graph-model v3 contract](extensions-training.md#canonical-graph-model-version-3). This persists
fixed operation meaning, descriptors and ordered ports, not the result-carrier Java objects or an
executable graph. Prepared executables, native/backend artifacts and runtime state are rebuilt,
never serialized. Cold preflight does not read valid-length values; runtime bounds remain an
execution responsibility. Persistence grants neither backend execution nor BPTT.

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
  resource declaration, slot assignment, and finalization lifecycle. This fixed route adds no
  loop-body or control-flow contract.
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

## General differentiable scan

The accepted exception is a pure, fixed-count scan over ordinary Tensor algebra. It is not a
general region system or while loop. Automatic differentiation (AD) constructs an ordinary sibling
scan in the opposite traversal before one combined capture. This section specifies the accepted
protocol. Model0035 implements and validates its source/IR foundation; Planning0008 implements
optional compiled-body query transport and provider contract tests. Compiler0011 implements scoped
capture/inference, bounded FLOAT32/FLOAT64 first-order AD and final body-bearing query/demand
emission. Prepare0010 implements scoped projection and static binding/demand validation. CPU0030
implements the bounded FLOAT32 same-shape pointwise forward/actual-reverse slice. CPU0031 adds
homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded shape/layout closure, including qualified
actual reverse bodies. CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL and independently
qualified actual Compiler reverse bodies. CPU0033 implements homogeneous FLOAT64 across that
complete bounded declared algebra and actual first-order reverse closure. The F32/F64 CPU domain
is implemented for static T and canonical boundaries plus proved zero-offset all-zero-stride
scalar reads. The narrow dense actual-read-view exception below was approved on 2026-10-24;
CPU0037 implements it with qualification and main's final capability acceptance on 2026-10-24.
Wider layouts, low types and the later migration gates remain excluded.
Exact provider/body/binder qualification still
applies independently to each occurrence. Ordinary users compose Tensor
algebra; they never choose saved-tensor recipes. The
[focused explanation](../general-differentiable-scan.md)
and [ADR 0045](../../design/decisions/0045-general-differentiable-scan.md) give examples and rationale.

### Source body, attachments and scoped identities

The implemented static admission below is retained. The sole bounded source relaxation is
[Accepted symbolic-batch scan](#accepted-symbolic-batch-scan); its semantics are owned here,
not by the numeric resource boundary. Model0042 implements this source-only relaxation, independently
reviewed and MAIN accepted 2026-10-25; Compiler0014's symbolic inference/AD/binding is also
independently reviewed and MAIN accepted after compatibility validation. Prepare0014 implements
bounded homogeneous FLOAT32 symbolic forward and actual first-order reverse through the existing
Prepare/CPU/Engine owners; final qualification is tracked in its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md).
Cold support remains static-only. Ordinary symbolic-B execution retains its separate qualification.
The user-approved length-mode source combination is specified under
[Header, bindings and source admission](#header-bindings-and-source-admission). Model0044 implements
that source-only combination under the authority accepted on 2026-10-26. Independent source/docs
review and MAIN's root test/Javadoc checkpoint passed; MAIN accepted Model0044 as Complete on 2026-10-26.

The complete new Model inventory, under `io.github.pho001.synaptik.`, is:

- `model.tensor.ScanBody`: immutable ordered ordinary Tensor formals `(carry, step, invariants)`,
  roots `(nextCarry, emissions)`, and derived ordered retained values. This is source expression
  containment, not a second DAG, captured graph, callback or runnable closure.
- `model.tensor.Scan.apply(initialCarry, sequences, invariants, body, reverse)` and
  `model.tensor.ScanResult(finalCarry, outputs, residuals)`: explicit lists of canonical Tensors.
  Construct formals once with TensorFactory; retain no callback.
- `model.operation.scan.ScanKind` and `ScanAttrs`: fixed-count meaning and scalar header only,
  containing inferred static time count T, carry/sequence/invariant/emission/residual counts and
  reverse flag on the implemented all-valid path. The implemented length-mode source additions are
  defined below. Attributes contain no Tensor, graph, registry key,
  executable or saved runtime values.
- `model.graph.CompiledScanBody`: one existing `CompiledGraphModel`, ordered role-result ValueId
  bindings and ordered demanded outer-port indices. This is the one captured body representation.

Extend existing TensorProducer with `Optional<ScanBody>`, validated before canonical output
allocation, and CompiledNode with `Optional<CompiledScanBody>`. Preserve old constructor forms
with empty attachments for ordinary and fixed operations. No additional type/layer is selected.

On the static path, formals are identity-distinct, provenance/storage-free, static, canonical and disjoint from actual
outer bindings. Removing axis zero from a sequence descriptor yields its step formal with canonical
step geometry. Carry/invariant formals match actual shape, dtype and gradient eligibility. Actual
inputs may repeat exact identities across roles; never deduplicate positions. Body leaves must be
exactly declared formals by identity, except declared unused formals are allowed. Hidden scalar
Tensor constants must become explicit invariants; scalar operation attributes remain ordinary algebra.
Reject free leaves, host-backed constants, RNG, mutation and nested/recursive bodies.

Model source construction admits existing known PURE Tensor operation semantics and their
applicable existing Model dtypes, independently of the initial CPU execution algebra and F32/F64
domain below. Validate the whole contained body, including canonical siblings, against existing
static or admitted named-symbol, type, shape and role rules. Thus a valid FLOAT16 TANH body or FLOAT32 RELU body may
construct without belonging to that first CPU domain. This grants no custom/unknown semantics,
new operation capability, AD rule or backend support. Effects and the other source exclusions
above still reject at construction; unsupported AD and execution occurrences still reject in
their owning Compiler/capability/backend boundaries.

Cold copying has a separate bounded constructor inventory in
[Compiler's cold lifecycle](compiler-autograd.md#bounded-general-scan-cold-reconstruction).
Model-valid PURE does not mean reconstructible, executable, differentiable or serializable.
The cold exception preserves applicable Model dtypes rather than importing CPU qualification.

For outer result construction, let `G` be the OR of `requiresGrad` over every actual initial-carry,
sequence and invariant input position. Every differentiable final-carry, stacked-emission and
stacked-residual descriptor receives `requiresGrad = G`; every nondifferentiable result dtype
receives false. Include unused actual inputs and apply the same rule for T=0. Formal flags still
match their corresponding actuals, including sequence-step formals; neither formal nor ordinary
body-value flags are rewritten to G. This conservative construction flag is not a claim of a
nonzero dependency, an available derivative rule or backend execution support. In particular,
with `h0.requiresGrad=false`, `x.requiresGrad=true`, `nextCarry=h+xStep` and `emit=h`, all
differentiable outer final-carry/emission/residual descriptors receive true even though the carry
formal and source emission have false flags: later emissions can depend on earlier sequence steps.

The same immutable template and its producers may serve multiple scan occurrences, with independent
capture scopes. Reject producer escape into the outer graph and sharing across different lexical
templates, not reuse of one template. No outer ValueId appears inside a body. Capture assigns IDs
once per scope; numeric IDs may coincide across scopes and are qualified only by their containing
CompiledScanBody. TensorIds remain global. Body graph inputs list every formal in declaration order,
including unused ones; graph outputs are unique, but ordered role bindings may repeat (for example,
one value serving as both next carry and emission) without duplicate graph values.

### Canonical residual ports and demand

A residual port names a body value available for later saving, not an allocation. Construction-time
canonical ports are Synaptik's immutable-producer design choice, not an industry requirement.

1. ScanBody derives retained values by deterministic producer postorder and ascending canonical
   output slot, identity-unique, including unused siblings and each carry formal (previous state).
   Step/invariant formals instead use explicit outer inputs, without duplicate saved sequences.
   Every other reachable body value receives a latent residual slot; Model selects no derivative rule.
2. Original Scan.apply construction reserves canonical residual descriptors/wrappers after final
   carry and emissions, never through producer replacement, postcapture augmentation or replay.
   These descriptors describe stacked histories but allocate none. Aliased roles share body-value
   bindings while outer canonical wrappers remain distinct.
3. Only publication/downstream consumption demands history. Internal body use needs step temporaries,
   not history. Propagate ordinary-use demand into queries and Prepare and revalidate after every
   rewrite. Live histories store at original time indices; a previous carry is saved before its step.
4. Only live residual history restricts fusion/recompute at saved sites. Dead ports allocate/store
   zero history and impose no residual fusion restriction, including the generated reverse scan's
   own residuals. Other ordinary observable boundaries still apply. This requires new liveness and
   Prepare work, not existing sibling dead-code elimination or an opaque saved-workspace handshake.

Demanded conservative histories cost `O(T * retained elements)` storage and constrain fusion.
More selective saving or recomputation requires a separate reviewed decision.

### Reverse scan and exact primal bindings

A cotangent is reverse-mode sensitivity; a primal is an original forward body value. Compiler owns
the following first-order protocol through existing Tensor derivative algebra, not captured-IR AD:

1. Inventory the body and combine cotangents for every selected final-carry and emission port once
   per scan occurrence. Missing port seeds are explicit typed zeros; final-carry and sequence seeds
   may coexist. Existing public request seed guards still apply.
2. Extend selected Compiler rule entry points with an explicit compile-local primal-operand binding
   function. Every formula operand denoting a primal resolves from its exact original body Tensor
   to a reverse-body formal backed by the corresponding residual, step or invariant input. TANH's
   saved output resolves to its residual formal, not recomputed tanh. Contribution maps continue
   to address original Tensor identities. No ValueId-to-Tensor AD map or second algebra is allowed.
3. Close derivative constants, including TANH/SIGMOID oneLike scalar leaves, through explicit
   non-gradient invariant formals bound to outer Compiler-created splats via existing constant
   ingress. Expand remains ordinary body algebra. Zero seeds and accumulator initializers use
   registered outer splats/expressions; carry initializers bind through carry formals and body
   constants through invariant formals, never undeclared free leaves.
4. Construct an ordinary sibling Scan.apply in opposite traversal. Cotangent carry starts with
   final-carry seeds; output seeds and residual histories are explicit sequences; shared parameters
   are invariants. Initialize parameter sums to exact typed zero and add each positional contribution
   once in traversal order. Return dSequence emissions, final dInitialCarry and parameter sums.
   Sum tied actual-role and cross-occurrence terms afterward with ordinary Tensor.add, never
   identity-deduplicate contributions.
5. One combined capture includes original forward wrappers and generated reverse scan, each body
   captured once in its independent scope. Public TensorId/publication roles remain unchanged;
   ordinary residual edges enforce forward-before-reverse liveness. Exact saved sites survive
   layout closure, reindexing, DCE/CSE and fusion. Never merge scan occurrences or cross live saved
   boundaries.

Residual ports use the conservative outer construction flags defined above, but differentiation
rooted at a residual port, unsupported body rules, and every scan higher-order/createGraph request
reject before derivative allocation. Preflight includes unsupported occurrences on otherwise
unrelated original branches.
This exception grants no fixed RNN/GRU/LSTM BPTT.

### Traversal, empty domains and declared qualification

The first completed CPU domain is homogeneous FLOAT32 and FLOAT64, canonical boundaries, static
T >= 0, at least one carry and one sequence, zero or more invariants/emissions, common sequence T,
and fixed body shapes/dtypes. Forward traverses `0 .. T-1`; reverse traverses `T-1 .. 0`. Emissions
and live residual histories are stored at original indices, not traversal-order positions.

The approved CPU0037 extension admits a resolved actual read with `storageOffset=0` and strides
exactly equal, axis by axis, to `LayoutDescriptor.contiguous(shape).strides()`, even when
`isView=true`. This applies to every actual initial-carry, sequence and invariant read position
in the existing pointwise, shaped and MATMUL families, including each independently qualified
actual Compiler-generated reverse occurrence (seeds, saved histories and primal bindings).
Canonical reads and the existing zero-offset all-zero-stride scalar reads remain admitted;
the latter do not have to satisfy the dense-stride predicate. Internal formals and outer
outputs/write boundaries retain complete non-view canonical descriptor requirements. Existing
proved local body views are unchanged. This is read admission, not removal of view metadata or
changes to LayoutDescriptor equality, Tensor identity/provenance, source or storage ownership.

The implemented read admission retains exact shape/type/role, bounds, referenced-span,
storage accessibility, lifetime and overlap proofs. Read/read aliases remain legal; writable
overlap does not. Absent/unresolved layouts, nonzero offsets and noncanonical strides remain
rejected except for the retained scalar-read case, as do low/mixed unsupported types and bodies.
The added read uses the canonical twin's dense address/access/resource/IR and generated-code
path, without inserted copies or CONTIGUOUS. This grants no general affine-layout support.

For T=0 no body executes: emissions/residuals are empty, final carry is the exact initial carry,
dInitialCarry equals dFinalCarry, and invariant sums are typed zero. Empty dimensions follow each
constituent's semantics; an empty dimension does not excuse skipping a nonempty reduction or an
unsupported operation.

Exact initial carry here means its represented values/raw bits, not reuse of the input Tensor
wrapper or its flags. Construction still reserves distinct canonical outer ports and applies G
as defined above; outer flags are not copied from body roots or initial-carry descriptors.

The initial declared CPU execution algebra is ADD/SUB/MUL/DIV (ordinary floating tensor and existing
scalar forms), NEG/TANH/SIGMOID, rank-two MATMUL and rank-two `[1,0]` PERMUTE under existing guards.
Ordinary DIV derivatives are reused without new masks/numerics. Qualification covers the whole generated closure:
broadcast reductions/sumToShape, expands, views/contiguous materialization, constants and required
casts. NaN/Inf, signed zero, zero divisors, overflow/underflow and empty reductions retain existing
family arithmetic and saved-value rules; unsupported occurrences reject, not blanket all-op admission.
BFLOAT16/FLOAT16 require later independent qualification under existing working/rounding/saved-state
semantics, not whole-model conversion or support inferred from emitters.

### Qualification, performance and migration gates

The implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
qualifies only its selected exec-v1 terminal arithmetic and synchronous CPU region dispatch.
It does not admit parallel/associative time traversal or change scalar CPU0073.

Construction, captured/prepared objects and generated code are `O(body)` rather than T-fold copies;
live saved bytes may grow with T, while undemanded inference history has zero bytes/stores. The
[capability boundary](foundational-modules.md#profile-free-capability-queries),
[Compiler boundary](compiler-autograd.md#compiler-owned-automatic-differentiation),
[Prepare boundary](runtime-prepare-engine.md#modulesprepare) and
[CPU boundary](backend-execution.md#cpu-backend-routes) own their respective admission, projection,
resources and emitted-loop duties. No runtime interpreter, per-op lookup, per-step callback or
per-step allocation is allowed. Materialize only when needed; this is no universal no-copy or speed claim.

The user-approved 2026-10-30 [CPU0073 mechanism](backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
is implemented and MAIN accepted after independent review and named gates on 2026-10-30.
It uses exactly three CPU-internal top-level
types (`CpuRowMapReductionIr`, `CpuRowMapReductionLowering`, `CpuRowMapReductionEmitter`), nested
`Contraction`/`Sum`, and their specified generation seams/selected identity suffixes. That owning
subsection defines packages, signatures, proof limits and retention. A fused region is emitted
inside the existing Scan wrapper/time loop, not a new body graph or executor. Existing semantic
sections and admission remain unchanged. The [CPU0073 result](../../planning/backends/cpu/tasks/0073-row-map-multi-reduction-fusion.md#result)
records the bounded execution/retention evidence, not a performance claim.
No other type, runtime edge or namespace change is permitted.

Implementation gates are serial: independently reviewed authority cutover and main acceptance;
separately Ready Model source/IR; Planning query; Compiler AD/capture; Prepare projection; a complete
bounded pointwise FLOAT32 forward/automatic-reverse vertical milestone; remaining declared forms
and closure plus FLOAT64 domain completion; then the low-level join. No successor becomes Ready
automatically. First execution is all-valid, static and canonical. Retain all existing four-dtype/
read-layout fixed routes and static NN routes without redirection or retirement.

CPU0037 is a separately approved same-address read-admission extension, not NN migration.
Acceptance requires focused F32/F64 pointwise/shaped/MATMUL and actual-reverse qualification,
canonical-twin access/resource/IR/identity/generated-byte equality, preserved formal/write guards,
and real Conv1d-prefix scan plus singleton-CONCAT-seeded reverse numerical gradients against
independent ordinary unrolling. Include T=0/empty domains where applicable, heap/segment reads,
changed inputs across reusable runs and alias/write guards. With generator, physical binding and
identities unchanged, reuse CPU0036's specialized clean-Java/Class-File/performance evidence and
its limitations; no new benchmark is required. If that premise fails, stop for a main decision
before widening implementation. Neither this exception nor its completion unlocks `.rnn`,
padding, low types or fluent migration authority.

CPU0037's focused qualification covers the above read extension in both dtypes, including real
Conv1d-prefix and singleton-CONCAT workflows in both traversals against ordinary-unrolled outputs
and first-order gradients. Eighteen canonical/view pairs retain complete lowering facts,
resources, canonical and both specialization identities, generated class bytes and cached artifact
reuse. Generator, physical binding, schema/family 71 and staged v4 identities are unchanged;
CPU0036's evidence and limitations therefore remain applicable, without a new speed claim.
Main accepted the final module/Javadoc, named conformance/integration and configured root gates
on 2026-10-24; [CPU0037](../../planning/backends/cpu/tasks/0037-general-scan-dense-read-views.md#result)
records final capability acceptance.

Future implementation acceptance includes reusable-template/escape cases, closed constants, tied
roles/roots and cross-occurrence sums, two carries, both seed classes, shared parameters, exact saved
rounding/special values, T=0/empties, residual-root/higher-order rejection, overflow/overlap/cleanup/
retry/concurrency, unsupported interior/generated occurrences, missing body, supported children with
unsupported loop binder, preserved projection and no runtime graph access. Performance evidence uses
the specialized optimal clean Java algorithm/hot-loop/dataflow oracle and Class-File inspection,
plus short warmed matched-work generated/direct measurements; separate cold generation/class-load/JIT,
entry/preflight and kernel/step costs. Fix or explicitly justify diagnosed avoidable overhead. No
broad/five-fork campaign, zero-penalty/JIT or CPU0029 timing-parity promise follows. Future slices own
focused/affected-module test and Javadoc; main owns new architecture/conformance/integration coverage
and one final source-matched native-enabled root test/Javadoc gate at the low-level join. The
original authority-only cutover performed no code/dependency change, executable validation or
measurement; CPU0037's later implementation evidence is distinguished above.

The approved all-valid owned NN direction uses existing `Model.graph`/`defineTensor`, `compute`
and `configureTraining`/`train`. Its first tested execution is static canonical homogeneous
FLOAT32/FLOAT64. Compiler cold reconstruction, Checkpoint attachment-preserving copy/rebind and
NN ownership/lifecycle qualification were serial prerequisites. The bounded owned workflow is now
implemented under [the NN owner](extensions-training.md#owned-tensor-models); no fixed builder
redirection or fixed-scan BPTT follows.

The separate per-row semantic decision is now selected below; implementation and padded NN
migration remain gated. General Scan grants no automatic Model.train/save/load admission.
If/while, arbitrary regions, recursion, nested scans, parallel or associative general scans,
runtime tape, mixed-owner bodies and a second executor remain excluded.

### CPU0074 selected terminal grouping exception

Implemented, independently reviewed and MAIN accepted, 2026-10-30. CPU0074 is Complete;
the [task Result](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result)
records validation and unqualified steady-state profitability. This is the sole Scan semantic exception for the
[CPU0074 mechanism](backend-execution.md#cpu0074-selected-row-region-execution-exception).
Only selected CPU0073-shaped exec-v1 regions in already-admitted domains may use bounded
feature/output SIMD, row-partial workers or both. The complete occurrence/binder must qualify,
independently for genuinely fused forward and actual Compiler-generated reverse. A standard
RNN forward lacking a computed-row-producer/terminal region remains scalar. Static/length/combined
F32/F64 and all-valid symbolic F32 bounds do not admit all-valid symbolic F64 or low/wider types.

Within a selected terminal row-axis floating contraction or represented Scan SUM, a fixed
partitioned rounded tree may regroup the declared contributors. Every active contributor still
participates exactly once; both contraction operands and dependent math remain activity-dominated.
Within each worker visit rows in increasing original row index; merge valid partials in fixed
ascending range-index order on the caller. Empty/inactive partials have validity flags and do
not invent +0 terms. Singleton/no-axis raw movement is determined by geometry, not active count.
The existing foundational Aggregate floor is unchanged; this lifts only the narrower ordered
CPU0073/Scan override at selected terminal sites, not arbitrary neighboring operations.

Feature-K contraction order, separate typed MUL/ADD/result rounding, same compute/accumulator
dtype and no implicit FMA/TF32/lower precision remain mandatory. Ordinary exact SUM/MEAN and
standalone MATMUL-K receive no new permission. Time traversal and carry are sequential: finish
all row partials and merge before exactly one distinct shared-carry ADD per active time at its
own dtype/rounding site. Never absorb that ADD into a row partial, flatten across time, or update
shared floating state atomically. Inactive time preserves shared carry raw, not old+0.
Masks, original indices, casts, raw movement, empty identities, padding and selected-realization
saved values remain exact. Load saved primals, never replay TANH/SIGMOID. Finite tolerances are
per-operation/dtype qualification tests only; no numerical mode/certification or cross-device,
thread-count or strategy bitwise claim is created.

The emitted outer time loop remains the sole time-control owner. Only its generation-selected,
typed StepDispatch boundary may synchronously submit prebound row ranges and allocate existing
scheduler metadata per step. This qualifies the no-per-step-dispatch/callback/allocation language
in Qualification, performance and migration gates; Qualification and mechanism gates; and
Runtime preflight and implementation gates solely for that seam. Body/element allocation,
per-op lookup, graph interpretation, geometry repacking, a second executor and T-fold code/buffers
remain forbidden. [Runtime reservation/lifetime rules](runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation)
govern borrowing, errors, interruption/close and quiescence. Join precedes consumer execution,
carry transition and scratch reuse; no nested worker submissions. One RunState owns all mutable
frames/staging/partials, not prepared state. Existing length preflight after producers and before
any Scan-local write remains intact, including all B values and T=0.

Only bounded raw feature-transpose staging proved by the CPU owner is permitted: once per run
for a true invariant across T, otherwise at the exact active time after the relevant producer/
SHARED history binding, or scalar fallback. No stale reuse, row compaction/sorting or B×parameter
bank. Activity gating encloses staging; T=0/B=0/all-inactive reads no body history or invariant.
All resource declarations precede slots. CPU0073 scalar-v1 algorithms/keys/ClassFiles/resources
and nonselected cases retain their existing rules. Only new exec-v1 identity appendages and exact
internal seams in the CPU owner qualify the prior no-other-type/namespace/runtime-edge clauses.
No new Runtime protocol, public API or production-default strategy follows.

### Accepted symbolic-batch scan

Accepted direction on 2026-10-25: extend existing all-valid `Scan.apply`/`ScanBody` admission to
one exact named batch dimension B, retaining fixed time T, features and state widths. This is an
accepted contract; Model0041's authority delivery is Complete, MAIN accepted
2026-10-25 after independent APPROVE. Model0042 implements only static/bounded-symbolic Model
source construction and consumer rejection evidence, independently reviewed and MAIN accepted 2026-10-25.
Compiler0014 implements symbolic inference, actual FLOAT32/FLOAT64 first-order AD and fresh
scoped all-port binding, independently reviewed and MAIN accepted 2026-10-25 after compatibility validation.
Prepare0014 implements bounded homogeneous FLOAT32 forward and actual Compiler-generated reverse,
generated scalar and single-range, through Prepare/CPU/Engine; its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md) tracks final qualification.
Symbolic cold/copy/wire and Engine geometry memoization remain unimplemented.
Retain all implemented static FLOAT32/FLOAT64
paths, the actual-read exceptions above and fixed RNN/GRU/LSTM restrictions. Model0044 implements
the symbolic length-mode Model source combination below. Compiler0015 implements bounded
inference/actual first-order AD/fresh scoped all-port binding under
[the implementation gates](#runtime-preflight-and-implementation-gates).
Prepare0015 implements symbolic-length Prepare/CPU/Engine execution: fixed T/features, one named B,
homogeneous FLOAT32/FLOAT64 forward and actual Compiler-generated reverse, through existing owners
and numeric protocols. Owning validation and independent code/structural review passed; root validation and the
sole bounded probe completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending. This does not qualify all-valid
symbolic FLOAT64 or implement Engine memoization.
Variable T/features, higher-order, low/wider CPU dtype, native/mixed, vector/parallel,
cold/copy/wire and NN/Training reuse require separate qualification.

#### Symbolic templates, roles and ports

Each value has fixed extents or at most one occurrence of the same named B; anonymous unknowns,
independent batch symbols and unproved transformations of B reject. Sequence axis zero remains
the common static T. Removing it yields the exact step-formal shape. Carry/invariant actuals
match their formal's symbolic shape, dtype and gradient flag; next-carry shape/type matches its
carry. Model retains known PURE operations and applicable dtype guards, independently of Compiler's
bounded FLOAT32/FLOAT64 first-order admission and initial homogeneous FLOAT32 CPU qualification.
The symbolic branch does not turn the CPU whitelist into Model source semantics.

Symbolic template formals/results have unresolved numeric layouts, never a canonical layout
fabricated at a sample B. Static formals/results keep their exact canonical guards. After binding,
all final-carry, stacked-emission and stacked-residual ports have actual canonical non-view layouts;
all ports, including latent/dead siblings, retain ordered identity, shape/type and conservative
outer gradient OR. Repeated role bindings do not merge wrappers or contributions. Existing scoped
containment, storage-free formals, explicit constants, lexical IDs and demand rules still apply.
Compiler independently validates every local declaration, actual/formal/result binding and saved
site; Prepare preserves that complete projection rather than equating local and outer numeric IDs.

Named-axis provenance, not equal numeric sizes, distinguishes row-dependent from shared values.
`[B,F]` transposes to `[F,B]`; reducing the identified axis removes B. A fixed parameter `[F,H]`
stays shared even at B=F=H, and its demanded history is `[T,F,H]`, not `[T,B,F,H]`.
Original recurrent row carries/steps and generated shared accumulator positions stay distinct.
All-valid symbolic mode retains default ScanAttrs counts; do not use length-mode `rowCarryCount`
to label it, add a lengths actual, store B in the header or introduce another body representation.

#### Actual first-order closure and arithmetic

Only the implemented [CPU0074 terminal exception](#cpu0074-selected-terminal-grouping-exception)
qualifies ordered producing-site SUM/grouping language here for selected exec-v1 regions.
Ordinary outer exact SUM/MEAN and all other arithmetic sites remain unchanged.

The bounded closure is tensor/scalar ADD/SUB/MUL/DIV, NEG/TANH/SIGMOID, rank-two MATMUL/transpose,
SUM/sumToShape, explicit constants and necessary expand/rank edits/contiguous/same-dtype casts
under existing family guards. Feature-only edits may introduce a singleton axis but must preserve
the named B coordinate; unproved reshape/mixing of B rejects. Qualify required rank-three outer
seed/zero/constant/pointwise/layout plumbing for `[T,B,*]` through existing ordinary owners;
this is not arbitrary-rank admission. Forward validity grants no actual reverse capability.

Use the existing opposite-traversal sibling construction with joint final/emission seeds,
original-time histories, closed derivative constants and exact primal bindings. Saved TANH/SIGMOID
outputs are loaded at their represented sites, never replayed. Shared invariant gradients use
parameter-shaped additive carries, not B-by-parameter banks; positional ties and cross-occurrence
terms add once each. Residual-root, unsupported-rule and higher-order/createGraph preflight still
reject before derivative allocation, including unsupported unrelated source branches.

For X `[T,B,F]`, h `[B,H]`, Wx `[F,H]`, Wh `[H,H]`, bias `[H]` and
`hNext=tanh(xStep@Wx+h@Wh+bias)`, bind forward MATMUL M/K/N=B/F/H,
dX=B/H/F and dWx=F/B/H; dWh likewise contracts K=B with actual transposed strides.
Bias unbroadcast reduces B even at B=0/1; size equality cannot erase axis identity. For
`L=mean(finalCarry)`, preserve `oneLike(input).sum(...)`, restored/expanded division and actual
count B*H, never a prototype denominator. Preserve constituent rounding, NaN/Inf, signed-zero,
empty and saved-value semantics. Scan's typed producing-site SUM and ordinary outer exact
SUM/MEAN remain distinct algorithms; the accepted bounded FLOAT64 outer qualification and retained
FLOAT32 five-limb specialization belong to the [CPU owner](backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).
sumToShape's singleton/raw-copy and empty
reduction cases remain operation-correct; no new arithmetic latitude is selected.

#### Empty domains and numeric resources

T=0 requires its own fixed-T compilation/preparation, not a changing-T invocation of a T>0 plan.
No body or hoisted body operation executes. Final carry preserves initial represented bits in a
distinct canonical output, dInitialCarry equals its final seed, stacked emissions/residuals/dX
are empty, and invariant-gradient sums are typed zero. Full-body structural validity still holds.

B=0 empties row work, not the whole Scan/run. Required shared carries/emissions, initializers and
parameter-shaped empty-contraction/SUM outputs still execute where nonempty. K=0 MATMUL and
empty SUM produce typed zero without nonexistent row reads; mean(empty finalCarry) is NaN.
Empty dX/dh0 does not excuse nonempty parameter gradients. Test B=0 and T=0 independently and
together, including shared work beside empty rows. Neither emptiness nor dead demand hides an
unsupported body or overflowing trailing domain. Geometry reads no lengths. CPU0042's fixed-batch
length execution has separate producer-ordered value preflight. Symbolic-B plus lengths has
implemented Compiler qualification below; Model0044 source alone did not grant it. Prepare0015's
combined execution is implemented through the existing owners. It changes none of the
numeric/resource algorithms in this section.

One compilation/preparation fixes code/routes/slots, representation positions and storage topology.
Numeric recipes bind actual B: TBF time stride B*F, row frame B*H, demanded row history
T*B*localFeatures and demanded shared history T*sharedElements. Dead histories retain zero
bytes/stores. Exact checked alignment/padding, conditional empty scratch and B=0/1 cases may be
piecewise, not necessarily affine. Never use representative fake B, B_max banks, per-B relowering,
route/slot/topology selection or executable caching. Declared zero-byte workspaces retain their
structural slots/creators. Carry backing and inclusive cut lifetimes remain fixed; reuse a cut
only when lifetime and size dominance hold for every admitted B, otherwise keep disjoint regions.

#### Qualification and mechanism gates

The implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
adds only its exact CPU-internal payload/dispatch/math seams and selected exec-v1 identities;
the no-extra-seam and scalar-only gates below otherwise remain in force.

Existing owners and numeric protocols are selected, not new production types/public signatures.
Compiler owns metadata binding; Prepare owns scoped transport/stable declarations; CPU owns
numeric recipes/emission; Engine owns entry admission and the separately bounded
[geometry memo](runtime-prepare-engine.md#accepted-engine-geometry-memo). Runtime gets only
authenticated numeric geometry, no body/symbols/memo. Every invocation retains complete fresh
Compiler binding/private obligations, current input-backed publication descriptors, physical
admission and fresh mutable arrays/resources. No per-element/time-step memo lookup is allowed.

Implementation acceptance must qualify real symbolic forward and actual Compiler-generated reverse
independently against nonuniform ordinary Tensor unrolling and specialized clean Java: both
traversals/seeds, fixed parameter gradients, local/outer ports, exact demanded/dead/shared bytes,
64→40→1→64 reuse, separate T=0, B=0/1, overflow before borrow, aliases, cleanup/retry/close and
concurrent mixed-B results. Add affected-module, backend-conformance and Engine integration tests,
retained static compatibility and unsupported symbolic/length-boundary rejection tests, and generated Class-File checks for algorithm/hot-loop/
dataflow, hidden allocation/boxing/reflection/dispatch. Use only bounded warmed matched-work
generated/direct evidence with cold setup, admission and kernel costs separated; no speed promise.

Order remains authority review/MAIN acceptance → separately Ready Model foundation with old-consumer
rejection → Compiler inference/actual AD/all-port binding → coordinated Prepare/CPU/Engine vertical.
No successor becomes Ready automatically. Prepare0014 uses existing owners and the symbolic
identity branch documented by the CPU owner, with no new Runtime handoff or supported signature.
The future Engine memo's atomic publication/close mechanism remains a later scoped-brief decision.
CPU0073's exact [accepted internal mechanism and identity exception](backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
alone qualifies the new-type and retained-key/ClassFile gates here; noneligible controls stay
literal unchanged and Scan projection stays 71. It admits no new symbolic dtype or arithmetic.
If any other new
named type or changed Runtime handoff is necessary, stop for exact package/name/purpose/signature
and user approval; zero proposed types is not an implementation proof. The implemented Compiler0015
combination qualifies inference, actual first-order sibling AD and full fresh scoped all-port binding
together, not backend execution. Its numerical evidence uses existing test-local evaluation of the
actual generated graph against independent nonuniform active-prefix unrolling; only that oracle
inspects length values. Backend/emitter/performance gates above apply to separately admitted
execution, not this Compiler-only phase. Prepare0015 implements combined execution,
including the bounded ordinary typed/control/ROW(1)
prerequisite under the CPU owner. Both complete forward and actual reverse occurrences must qualify
independently in each dtype; no handwritten reverse substitutes for Compiler siblings. Existing
header/equations, original-source ROW(0) AD restrictions, generated mixed roles, exact saved bits,
length preflight and resource algorithms are unchanged. NN/Training and all other excluded successors
remain closed; the future Engine memo is not part of Prepare0015.

### Length-aware whole-batch scan

The user approved the independently reviewed length-aware design on 2026-10-24. This section
defines source semantics, first-order reverse construction and the length execution protocol. MAIN accepted
Model0037's coordinated authority on 2026-10-24 after Hume's final overall independent Class C
APPROVE, zero open P1/P2. Model0038 implements the bounded Model source header/overload, row-role
validation and structural IR counts. Compiler0013 implements independent scoped inference,
first-order reverse construction and role/demand query emission for the subset below. It preserves
exact saved primals, normalizes each incoming SHARED-target cotangent before accumulation and
binds row-shaped zero/literal seeds through explicit sequence roles. These are compile-time
facts, not execution evidence. Prepare0011 implements static lengths/formal separation, boundary
role checks and complete-graph demand projection through the existing staged resource handoff.
It preserves broader forward sources and actual generated reverse mixed roles without duplicating
body row-algebra proof. CPU0042 implements fixed T/B/features FLOAT32/FLOAT64 forward and actual
Compiler-generated first-order reverse through generated scalar, single-range CPU execution.
Independent code/docs review and MAIN's named conformance/Engine checkpoint and single bounded
probe passed; MAIN accepted the capability on 2026-10-26. The
[CPU0042 Result](../../planning/backends/cpu/tasks/0042-length-aware-general-scan-execution.md#result)
records evidence and nonstationary timing limits, not a steady-state speed or parity claim.
Static homogeneous length-mode cold reconstruction is implemented and accepted under
[Compiler0018](../../planning/modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md#result)
and the [Compiler cold contract](compiler-autograd.md#bounded-general-scan-cold-reconstruction).
[Checkpoint0014](../../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result)
implements its separate static length-aware v5 wire; symbolic cold/wire remains unimplemented.
[Model0044](../../planning/modules/model/tasks/0044-symbolic-length-scan-source-foundation.md)
records the user's 2026-10-26 approval of one cohesive Model source foundation, including the bounded
source-contract wording before Java. Mencius final pre-Java APPROVE (0 P1/P2, no missing contracts/owners)
and MAIN authority/Ready acceptance are recorded on 2026-10-26.
The static-or-one-named-B source extension below is Complete: independent source/docs review and
MAIN's configured root test/Javadoc checkpoint passed, with MAIN acceptance on 2026-10-26.
Compiler0015 implements the static/one-B length-mode intersection for independent inference,
actual first-order sibling AD and full fresh scoped all-port metadata binding. The Compiler-only
task is Complete. Prepare0015 implements static-or-one-B projection and combined FLOAT32/FLOAT64
CPU/Engine execution under the gates below. Owning validation and independent code/structural
review passed; root validation and the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending.
Static homogeneous cold admission and inherited Checkpoint/NN cold consumer qualification are
Complete under Compiler0018. V5 and the complete owned length-aware numerical workflow remain
separate capabilities.
The existing all-valid PURE/type inventory, fixed scans, overloads and compatibility identities
remain unchanged. Length-mode source validity never implies a downstream capability answer.

#### Header, bindings and source admission

The implemented Model0038/Model0044 baseline appended `int rowCarryCount` and immutable ordered
`List<OptionalInt> sequenceRowAxes` after the original seven ScanAttrs components, using zero rows
as the all-valid sentinel. The accepted source amendment below replaces that canonical form with
an explicit mode while preserving the seven-argument all-valid convenience and Scan.apply overload.
Constructor migration is not preservation of record reflection, equality/hash algorithms or
diagnostic text; none is a stable wire/cache encoding. Old-default keys must not derive from
changed record text, and qualified all-valid/fixed identity byte streams remain unchanged.

The implemented additional source-only overload is:

```java
ScanResult Scan.apply(List<Tensor> initialCarry, List<Tensor> sequences,
        List<Tensor> invariants, Tensor validLengths, int rowCarryCount,
        List<OptionalInt> sequenceRowAxes, ScanBody body, boolean reverse);
```

The implemented positive-ROW consumer baseline requires `1 <= rowCarryCount <= carryCount` and exactly sequenceCount non-null
optional entries. A present entry denotes ROW(axis in the step formal); absent denotes SHARED.
ROW means dependence on one identified batch coordinate, not an extent numerically equal to B.
Initially only ROW(0) and rank-two ROW(1) are admitted. Leading carries are ROW(0); remaining
carries are terminal SHARED additive accumulators. Invariants are SHARED. Actual order is carry,
sequence, invariant, then one non-gradient INT64 `[B]` validLengths actual, with **no body formal**.
Thus outer inputCount is checked `c+s+i+1`, while local formal count remains checked `c+s+i`.
All-valid counts remain `c+s+i`. Output groups and CompiledScanBody's representation are unchanged.

The implemented positive-ROW baseline infers common static T from sequence axis zero and B from row-carry axis zero.
Retain fully static source as a first-class path; alternatively B may be one exact named dimension,
compared by canonical name rather than object identity or the spelling `B`. Removing outer time
yields each step formal. Every declared row axis has that exact Dimension; every row carry and
emission is row-leading. All other extents, including features/state widths and SHARED shapes,
remain static. A symbolic row value contains B exactly once, at its proved ROW(0) or rank-two
ROW(1) axis; SHARED values, including invariants and terminal accumulators, cannot contain B.
Reject anonymous unknowns, independent/repeated/derived B, variable T/features and unproved
coordinate transformations, including in unused declarations and shape-bearing attributes.
Symbolic formals, actuals and ports retain unresolved numeric layouts; static formals and ports
retain canonical layouts, and existing static actual-layout admission is unchanged. No sample B
or runtime binding is performed by source construction. Check lengths dtype/rank/flag/exact B,
admitted shapes, exact formal bindings, role counts,
containment, all canonical siblings and checked arithmetic before canonical output IDs. Length
values are never read at construction. Repeated actual positions retain positional identity and
contributions, with compatible explicit roles; do not deduplicate them. Compatible tied actuals
identify the same outer batch axis, or are both SHARED; step axis r maps to outer axis r+1.
Lengths cannot alias a body formal. The existing outer gradient OR applies to
carry/sequence/invariant positions only;
lengths is discrete, never a differentiation target.

The Model0038/Model0044 source baseline required homogeneous FLOAT32 or FLOAT64 body formals,
values and matching actuals, plus separate INT64 lengths. Model0045 supersedes that source-wide
dtype restriction and same-type-only CAST restriction as specified below. The closed algebra
and ordinary family descriptor guards remain; this is Model source admission, not a CPU provider
query. All-valid source still admits its existing known PURE operations and applicable Model dtypes.

#### Accepted empty-carry and explicit length-mode source

MAIN accepts [ADR0059](../../design/decisions/0059-empty-carry-scan-and-explicit-length-mode.md)
and the implemented, source-qualified [Model0046 cutover](../../planning/modules/model/tasks/0046-empty-carry-scan-and-explicit-length-mode.md).
It supersedes only source representation/count/mode assumptions above, not current consumer
qualification. Independent Compiler/Prepare/CPU/cold/raw-codec guards remain bounded.

ScanBody permits an actual empty carry/nextCarry pair, with nonempty step sequences and at least
one nextCarry or emission root. Existing carry shape/dtype stability, formals, containment,
aliases, roles, canonical outputs and checked arithmetic remain. CompiledScanBody permits empty
nextCarry with nonempty semantic roots; ScanKind's structural minimum becomes one input.

ScanAttrs canonical components become `(timeCount, carryCount, sequenceCount, invariantCount,
emissionCount, residualCount, reverse, lengthAware, rowCarryCount, sequenceRowAxes)`.
Retain its seven-argument all-valid constructor; migrate/remove every nine-argument sentinel
caller, with no alias/shim. Require checked T>=0, c>=0, s>=1, i/e/r>=0 and c+e>=1. Explicit false
requires rowCarryCount=0 and empty axes. Explicit true permits rowCarryCount in [0,c] and exactly
one existing optional role per sequence. inputCount depends on lengthAware, never the row count.
Lengths stays the final non-gradient INT64[B] actual, with no body formal.

Length source derives static/named B from lengths metadata before validating every explicit ROW
coordinate, without reading lengths or guessing from coincident extents/first carry. Existing
row/algebra/activity rules stay unchanged. A no-ROW reverse can later use only SHARED floating
accumulators, or no carry when only floating sequence emissions are required. This authorizes no
dummy state, discrete cotangent, replay, unrolling, map API or additional execution owner.

Until separately qualified, Compiler inference/AD, direct Prepare and direct CPU independently
reject `carryCount==0` or `lengthAware && rowCarryCount==0` before first-carry assumptions,
derivative IDs, analyzer calls, resources or realization. Cold reconstruction and raw encoding/
decoding independently retain opcode23's nonempty-carry and opcode33's positive-ROW domains,
including pre-sizing/pre-allocation rejection. Mode remains opcode-derived on wire; no boolean,
v6 or new opcode. Existing v1-v5 accepted bytes and qualified CPU internal IR/keys/generated
bytes remain unchanged. The explicit source flag is not a blanket identity bump.


<a id="prospective-per-value-typed-length-aware-source"></a>

#### Accepted per-value typed length-aware source

SOURCE COMPLETE — MAIN accepted 2026-11-01 after independent pre-Java and frozen
source/tests/Javadoc review, including deliberate consumer rejection evidence. The
[Model0045 Result](../../planning/modules/model/tasks/0045-per-value-typed-length-scan-source.md#result)
records retained source-matched coverage and three approved Class A test-only recoveries;
original failed commands remain failed, with no clean root PASS claim.
This subsection supersedes only the historical Model source dtype/CAST restrictions; every other
source rule and downstream qualification remains unchanged.

Scan controls traversal, bindings and row activity, not one global arithmetic dtype. Each body
value retains its own ordinary Model dtype. Length-aware Model source admits FLOAT64, FLOAT32,
BFLOAT16, FLOAT16, INT64, INT32 and BOOL per value where the existing closed row-preserving
operation inventory and ordinary family guards permit them. Carry, sequence and invariant slots
may differ in dtype, including unused slots. An identity ROW(0) BOOL or integral carry is valid;
this does not admit BOOL arithmetic, new logical/comparison/selection operations, RELU or SLICE.
Trailing SHARED carries still require the existing terminal additive update; they do not acquire
identity-update permission merely because ROW carries have it.

Every corresponding actual/formal position still matches exact shape, dtype and gradient
eligibility; sequence matching removes only time. Each next-carry root retains its own declared
carry's exact shape and dtype. Scan inserts no promotion, conversion, common working dtype or
hidden cast. Ordinary body operations alone own their existing promotion, compute/accumulator,
conversion, rounding and saved-value rules. An operation's ordinary promoted result cannot change
the declared carry dtype. A trailing SHARED update remains its own `old.add(contribution)` with
contribution shape **and dtype exactly equal to that carry's**, even when ordinary ADD promotion
would otherwise preserve the root dtype (for example FLOAT64 old plus FLOAT32 contribution).
Scan validates this dtype equality explicitly rather than relying on global homogeneity.

An ordinary explicit CAST may change dtype while preserving the exact shape and proved row role.
Use the ordinary cast descriptor/value/eligibility rules, not a Scan conversion policy or an
implicit cast at a carry binding. CAST must preserve terminal row-reduction dependency facts and
SHARED-carry dependency isolation: casting cannot launder a terminal contribution into row
feedback or allow a shared carry formal outside its own permitted ADD. Every intermediate and
canonical sibling remains validated; no cast-chain folding or saved-site change is authorized.

The existing static-or-one-named-B geometry, ROW(0)/rank-two ROW(1)/SHARED mappings, terminal
rules, feature-overflow checks, containment/effect/free-leaf/nesting guards and pre-output-ID
failure ordering remain unchanged, including T=0/B=0. Control lengths remains a distinct
non-gradient INT64[B] actual with no body formal, outside the input-gradient OR. For each outer
final/emission/residual port, preserve its own dtype and use `G && dtype.isDifferentiable()`;
G still includes unused carry/sequence/invariant actuals. Do not rewrite body/formal flags.

This amendment grants no Compiler inference/AD, Prepare, CPU, cold reconstruction, wire, NN or
Training admission for newly Model-valid occurrences. Existing independent homogeneous F32/F64
consumer guards remain fail-closed, including low/mixed interiors hidden behind F32 boundary
casts or unused declarations. Negative tests must reach each owner directly through existing
raw graph/attachment/query seams; an earlier Compiler rejection is not Prepare/CPU/codec evidence.
No new production type, signature, wrapper, registry, execution architecture or dependency is
selected. All-valid paths, fixed kernels, ordinary unrolled `.rnn`/`.gru`/`.lstm`, static
`Sequence.forward` APIs, old checkpoint formats, constants and numerics remain unchanged.

#### Bounded row-role analysis

Model0048 implements the accepted ordinary-coordinate contract below, replacing the initial
source operation whitelist and rank-two-only intermediate ROW(1) restriction. Its independently
reviewed source/forward-coordinate result does not widen typed Compiler AD, Prepare, CPU, cold
or wire qualification. Retained physical recipes and old constructor/wire intersections keep
their independent admission until separately qualified.

The single descriptor-only semantic seam belongs to existing `model.tensor.Scan`:

```java
public record ValueSemantics(OptionalInt rowAxis, boolean terminal, List<Integer> directInputs) {}
public static ValueSemantics declaredSemantics(
        TensorDescriptor descriptor, OptionalInt rowAxis, Dimension batch);
public static List<ValueSemantics> ordinarySemantics(Operation operation,
        List<TensorDescriptor> inputs, List<ValueSemantics> inputSemantics,
        List<TensorDescriptor> outputs, Dimension batch);
```

Ordinary family guards remain with Model constructors; raw Compiler independently infers and
compares every canonical descriptor. Only then do both recompute these semantic facts. The seam
derives neither ordinary result shapes nor promotion and creates no Tensor, producer or identity.
Facts are not a persisted/source admission certificate, another graph, registry or proof framework.
Migrate Model validation, raw scoped inference and derivative role discovery together; do not
retain aliases or a second transfer implementation.

ROW(axis) identifies exactly one original B coordinate at any valid intermediate rank/axis.
SHARED has no row axis and fully static shape. Terminal implies SHARED. Direct inputs are
immutable increasing distinct producing-occurrence positions, preserving tied operands as
separate positions. They include conditions, indices, masks and targets, not cotangent eligibility
or every descriptor consulted for shape/promotion. Declared facts have no producing-input routes.

Check complete bodies, unused declarations and all canonical siblings even at T=0/B=0. Compare B
Dimensions structurally; only the identified row axis may be named and all other axes are static.
Equal F=B/H=B, B=0/1, zero element counts and gradient flags never infer or erase row identity.
Check fixed-feature products independently of T/B emptiness and retain legitimate zero-feature
domains. In ordinary right alignment, a SHARED operand must omit the row coordinate or supply a
static singleton there. An independently indexed shared lookup-table axis is not an aligned B
axis merely because its size equals B.

For every output row b, each contributing ROW input access must use that same identified b.
Apply ordinary contributor mappings, not a mathematical-kind whitelist:

- Pointwise, comparison, classification, logical, WHERE and CAST preserve compatible broadcast
  coordinates. CONTIGUOUS preserves logical coordinates. Permutation/rank edits map explicit axes.
- Reshape preserves the identified coordinate boundary and checked fixed feature products on each
  side, not merely total count or a matching extent. Preserve an existing exact-shape row position.
  For a changed shape, each candidate must independently prove the same row mapping and both
  feature boundaries. If several axes pass that complete proof, choose the lowest axis
  deterministically; multiple proved coordinate aliases are not a failed proof. Reject when no
  axis is proved. Never use B=0/1 as a substitute for boundary/overflow checks. Under the accepted
  general ordinary-unit cutover below, this is the additional length-mode ROW proof. Ordinary
  all-valid symbolic reshape instead requires proved canonical element-count equality and the
  declared-symbol/descriptor guards; it may move B through fixed features without claiming ROW
  provenance. Historical qualified backend restrictions remain independently effective until cutover.
- Feature-only slicing, selection, updates, padding, tiling, composition, indexing, ordering,
  cumulative operations, reductions, normalization, windows, convolution, pooling, losses and
  attention are legal whenever their actual ordinary mapping preserves that coordinate. Reject
  partial/reversed/arbitrary row access, row neighborhoods or normalization over B, not the family.
  Unknown/custom semantics, effects, RNG/dropout, fixed recurrence and nested scans remain excluded.
- A known entirely SHARED PURE occurrence has no row-coordinate reason to reject by kind. Its
  ordinary guards and terminal/shared-carry restrictions still apply; it cannot invent a row axis.
- Grouped Conv2d's actual VJP may permute `[B,G,O,P]` to `[G,B,O,P]`, use batched MATMUL and reduce
  axis one. Arbitrary proved intermediate axes/ranks are necessary; numerical derivatives remain
  in the ordinary owner. Sequence-header/formal admission is separate: this amendment alone does
  not widen ScanAttrs' declared 0/1 axes or direct Prepare's retained rank-two ROW(1) boundaries.

Compute each canonical sibling separately. BatchNorm training dependencies are normalized
slot0:[0,1,2], next-mean slot1:[0,3], next-variance slot2:[0,4], saved slots3/4:[0].
Attention output depends on Q/K/V/optional mask; weights depend on Q/K/optional mask, not V.
Ordinary batch-prefix broadcasting may nevertheless give weights a row domain from V's shape.
Domain provenance, numerical/control routes and ordinary validation must not be conflated.

Only established active contributor meanings eliminate ROW: represented SUM/sumToShape,
matched MATMUL contraction and functional shared-target ADD-scatter. Feature reduction shifts
the axis only by removed preceding features; equal aligned sumToShape extents retain it even
at B=1, unlike explicit row-axis SUM. Do not infer active MEAN/variance/softmax/order/loss-divisor
semantics. Shared-target ADD-scatter includes its base once, applies eligible updates in original
logical order and preserves raw unaddressed cells; its complete active index domain must validate
before writes. Replacement/MUL terminal meanings do not follow.

Terminal taint follows each output's actual routes and never feeds ROW work. A terminal
ADD-scatter base is legal only at shared targets, not as terminal-derived row indices/updates.
ROW next-carry and emission boundaries remain ROW(0), nonterminal, never a SHARED expand guessed
from size. Each trailing SHARED carry occurs syntactically only as the exact left/old operand of
its own distinct tensor old.add(contribution) root. Contribution shape/type match exactly and
depend on no SHARED carry. Check every syntactic use before consulting per-output routes; an
unused numerical route cannot conceal an illegal carry use. Previous-state residuals remain legal.

This source semantics changes no operation's ordinary numerical meaning outside length-aware
Scan and grants no backend, derivative or persistence capability. Cold/wire owners must preserve
their effective old operation/rank/role/attribute intersection explicitly when retiring the old
transfer helper; static homogeneous types alone are insufficient to preserve that boundary.

#### Activity, raw identity and retained histories

The implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
qualifies the increasing-b requirement only by fixed row partitions and range-index merge at
selected exec-v1 terminals; activity dominance and raw/saved-history rules below are unchanged.

Define `active(t,b) = t < L[b]`, `active(t) = exists b: active(t,b)`. Forward visits original
t=0..T−1; reverse visits t=T−1..0, hence each row's valid prefix L[b]−1..0, never its padded tail.
At a time, batch contributors are increasing b. The activity gate dominates padded sequence,
seed and row-history loads and dependent arithmetic. Result masking, `0*x` and compute-then-where
are not skipping. Inactive row carry is raw identity; inactive emissions/dSequence are exact
positive zero. Empty-row final carry is raw initial carry and dInitialCarry is raw dFinalCarry
seed. Active NaN/Inf retain ordinary family meaning; padding cannot contaminate active results.

Retain canonical residuals and ordinary use/publication demand. Save previous carry before the
active step and produced primals at exact represented sites, without replay. ROW(axis) histories
remain `[T]+localShape`, storing active row cells and raw positive-zero inactive cells. SHARED
histories remain `[T]+localShape`, with **no added B axis**: save exact values only at active(t),
otherwise raw positive zero. Shared accumulator history is its pre-time value. Read row
histories/seeds only at active(t,b), shared histories only at active(t); prior zero-fill does not
permit padded reverse reads. Shared hoisting/publication also needs any-active/active-time gates:
all-zero lengths, T=0 or B=0 perform no body work. Undemanded history costs zero bytes/stores.

The implemented [CPU0076 scalar load-placement exception](backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)
does not relax these activity scopes: true invocation-invariant cells and SHARED original-time
history cells require different lifetimes. Worker-local staging also requires an active consumer
in that range. Its conservative source scopes exclude ROW histories even if a consumer view
selects a constant row: source activity remains active(t, selectedRow), not merely active(t).
Neither zero stride nor scalar broadcasting proves readiness or temporal invariance. Complete
body/lifetime proof failures retain checked reads. This adds no Scan semantic exception or
across-time carry promotion.

For example, shared `w.permute(1,0)` `[F,H]` retains `[T,F,H]`, not `[T,B,F,H]`, even at F=B.
Output seed `[T,B,H]` is ROW(0); a transposed row residual is ROW(1). Reverse construction derives
these roles from checked original bindings/dataflow, pruning sequence axes together with
usedStep/usedSequences. Ordered roles live in reverse ScanAttrs and survive demand refresh,
query/projection and future cold/v5 roundtrip, not just temporary AD labels.

#### First-order length-mode reverse

Only the implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
qualifies the ordered row-contributor grouping below; feature K, sequential time and one distinct
shared carry ADD per active time remain mandatory for every realization.

Before derivative allocation, original length-aware sources qualify only when
`rowCarryCount == carryCount` and **every original sequence** is explicitly ROW(0), including
unused/non-gradient positions. Original SHARED carries or SHARED/ROW(1)/other nonzero-axis
sequences reject for AD. Broader forward source validity grants no AD. Generated reverse inputs
may mix explicit ROW(0), ROW(1), SHARED histories/seeds; that original-source restriction does
not apply to them. No differentiation of generated shared-carry reverse scans is admitted;
existing residual-root and higher-order/createGraph rejection is unchanged.

Reuse ordinary derivative formulas and exact primal bindings. Carry the same lengths actual
into the opposite-traversal sibling, with row cotangent carries first and invariant-shaped
SHARED gradient accumulators appended. Preserve joint final/emission seeds, false-flag internal
carry propagation, closed constants, tied positional contributions and cross-occurrence additions.

At active(t), parameter SUM/sumToShape and transposed MATMUL reduce only active batch contributors
in increasing b. The gate dominates both contraction operand loads and each contributor's
dependent arithmetic. Start each output contribution at the family's typed zero; preserve
ordinary multiply/accumulation/result sites, then perform **one distinct shared carry ADD per
time** at its own dtype/rounding site. Never fuse it into contraction or update per row.
At inactive(t), shared state is raw unchanged, not old+0. Feature contractions retain ordinary
domains. Inactive cotangent carries bypass the body, including poisoned emission seeds; an
empty row's final seed copies to dInitialCarry without entering parameter sums.

Use one invariant-shaped accumulator plus bounded same-shape contribution/old-new scratch as
required, never a B×parameter gradient bank. Demanded shared history can cost O(T×P), row history
O(T×B×retained-row-elements). New length-mode reductions do not promise bit-identical results
to a differently grouped whole-batch unroll; old all-valid arithmetic/bytes remain unchanged.

#### Runtime preflight and implementation gates

The implemented [CPU0074 exception](#cpu0074-selected-terminal-grouping-exception)
qualifies scalar/vector/parallel and no-per-step-dispatch/allocation restrictions below only for
the selected exec-v1 seam, scheduler metadata and bounded feature-transpose staging. All other
preflight, activity, source/AD, resource and unsupported-domain gates remain unchanged.

For the separately qualified symbolic-B extension, geometry admission reads no lengths. Validate
all actual B values after producers and before this Scan's first output/padding/history/scratch
write, even for an invalid last value or T=0 (all lengths must be zero). B=0 has an empty length
domain and no body work, but other/shared outputs retain their semantic identities; never skip
an entire run because one domain is empty. T=0 all-valid carry preserves raw initial values,
dInitialCarry preserves dFinalCarry and invariant sums are typed zero. Empty geometry bypasses
no structural/validity check. Scan-local rejection rolls back no upstream writes. Existing static
length semantics below remain unchanged. Lengths imply neither model-wide masking nor a changed
outer MEAN denominator; Training accumulation remains a SUM of supplied gradients.

After length producers execute, recheck accessible typed spans/alignment/overlap and validate
**all B lengths** in `[0,T]` before this Scan's first output, padding, history or scratch write.
An invalid last length rejects before earlier rows write; T=0 still validates all lengths.
This is scan-local rejection, not upstream rollback. Borrowed storage remains stable during run.
The INT64 boundary is separate from homogeneous floating body boundaries, not an invariant formal.

Reuse existing staged Prepare and the generated CPU scan path, not the fixed executor. Declare
all resources before slots; keep changing bindings/banks in RunState and code/recipes O(body).
Qualify forward and actual reverse independently for both dtypes, gate hoisted/shared/row work
at the proper activity scope, and retain every demanded saved site. No packing, sorting,
compaction, per-step allocation/callback, new executor, vector, parallel, OpenBLAS or Metal route.
Version changed length-mode canonical/specialization identities and every baked role/demand/
geometry fact independently from the wire; preserve old-default/fixed keys and generated bytes.

The later [CPU0073 accepted mechanism](backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
permits only its exact CPU types/signatures and selected `|scan-row-map-multi-reduction-v1=`
appendage to the existing baked mode payload in both streams, retaining Scan projection 71.
Existing canonical identities and noneligible ClassFiles/keys/resources stay unchanged; this
qualifies retention references here only for selected eligible fusion, not a semantic exception.
Length/activity/raw-identity/saved-history and distinct per-time carry-ADD rules are unchanged.
CPU0073 implements this mechanism; independent code/tests/Javadoc review and MAIN's named
conformance/Engine join are accepted on 2026-10-30. No new Runtime/Engine/Prepare protocol or
measurement grant follows; the linked CPU mechanism owns its bounded eligibility and evidence.

Authority review/MAIN acceptance preceded the separately dependency-verified Model0038 foundation.
Model0038 supports static source construction and structural CompiledNode counts. Model0044's
implemented source-only extension retains the same overload, header, ordered roles and counts.
Compiler0015 implements the support-gate qualification accepted on 2026-10-26: one exact
named B with discrete non-gradient INT64[B] lengths, fixed T/features and homogeneous FLOAT32/FLOAT64:
independent scoped inference, the existing actual first-order opposite-traversal sibling AD and
complete fresh scoped all-port metadata binding together. Final acceptance is tracked in its
[task result](../../planning/modules/compiler/tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result).
Existing header/row/activity rules, original-source ROW(0) AD
restrictions, generated mixed roles, numeric/resource algorithms and public APIs remain unchanged.
Compiler must prove Dimensions by canonical value/name, not sampled extents; independently validate
all declarations, bindings and latent ports; retain per-occurrence obligations and complete fresh
binding. Compile/bind reads no input/length values and allocates no history or physical resources.
Original-source preflight restrictions still reject before derivative IDs; raw IR/descriptor seams
must independently reject invalid or unsupported metadata rather than rely on Model constructors.
Prepare0015 implements
static-or-one-named-B length projection, combined homogeneous FLOAT32/FLOAT64 generated scalar,
single-range forward/actual reverse and existing Engine admission/publication. Its bounded ordinary
outer F64/control/ROW(1) prerequisite is implemented under the
[CPU contract](backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract), not by general
F64 MATMUL or all-valid symbolic F64 admission. The implementation preserves complete
pre-analyzer declaration/binding/latent-port/demand checks,
explicit batch-required mode even for `[B,0]`, and explicit B in fresh numeric packing rather than
derivation from a frame or feature extent. INT64 lengths retains its separate eight-byte boundary.
No new production type, public signature, Runtime protocol, storage/lease/dependency or memo is added.
Combined canonical `:symbolic-length-v1` and baked `|scan-symbolic-length-v1=` in both specialization
streams bind dtype/roles/terminal/demand/control/numeric-pack/recipe facts, excluding actual B/L,
resolved bytes, slots and addresses. Old all-valid static/symbolic, static-length and ordinary F32
identity streams and generated bytes remain unchanged. Owning executable validation and independent
code/structural review passed, including actual Compiler reverse, independent active-prefix ordinary
Tensor AD and generated/optimal-Java Class-File inspection. Root validation and the sole bounded probe
completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending; no speed or JIT-parity claim follows.
Incidental Optional/index failures or successful old-wire encoding are not guards.
Compiler0013 adds scoped inference, bounded first-order reverse construction and complete final queries/demand,
without reading lengths or allocating history. Original-source AD restrictions above still apply
before derivative allocation; broader forward inference does not relax them. Prepare0011 now
validates static bindings, declared boundary extents, tied positions and ordered demand before
analysis, retaining exact saved-primal ports and original local history shapes. Actual layouts,
including lengths, are preserved independently of backend read predicates. Test-only analyzers
declare checked mixed-boundary resources before assignment; this proves transport/associations,
not physical allocation or execution. CPU0042 separately qualifies mixed-boundary binding,
producer-ordered value preflight and generated active-prefix execution in both floating dtypes.
Cold capture/reconstruction and Checkpoint encoding gates still explicitly reject length mode.
No accidental success, incidental index failure or seven-field v4 encoding substitutes for rejection.
Compiler0015's completion itself authorizes no runtime, cold/v5 or owned-Model successor.
Model0044 remains Complete for source construction and its consumer-rejection baseline only;
active Model frontiers are None. The user's subsequent “pokracuj” authorizes only Prepare0015's
bounded complete vertical. Its pre-Java gate was independently approved and MAIN-accepted on
2026-10-26; the bounded combined implementation is now frozen. Final task acceptance remains
MAIN-owned under the gates above.
Compiler0013 and Prepare0011 create no distant task cascade or automatic successor launch.

CPU0042 qualification covers zero/full/mixed/invalid lengths, both traversals/seeds, empty domains,
F=B role ambiguity, shared transpose and ROW(1) history pruning, terminal-only reduction,
pre-write failure, raw identity/+0, poisoned padding/seed rejection, reuse/concurrency/overlap,
old compatibility and independent active-prefix oracles. CPU hot paths require the specialized
optimal clean-Java algorithm/dataflow oracle, Class-File inspection and bounded matched-work
evidence. CPU0042 records those checks and their limits; Compiler0013 alone supplies no such
execution or performance qualification.
Cold/wire and NN admission remain with [their owner](extensions-training.md#canonical-graph-model-version-5).
Lengths govern only this scan: downstream bias/head/loss remains ordinary dense algebra with
no model-wide mask or changed MEAN denominator. `.rnn` migration, fixed BPTT, GRU/LSTM migration,
low types and higher-order waivers remain excluded; NN0032's second-order CPU compile regression,
singleton-CONCAT copy and unproved mixed-engine compatibility remain separate gates.

### General ordinary-unit Scan cutover

Implemented coordinated Class C cutover — MAIN accepted Compiler0017/CPU0081 after the exact
Ready/source freeze, independent authority/interface and implementation APPROVE0 reviews, actual
ordinary/selected-backward/typed/activity consumers and the unfiltered twelve-project test/Javadoc
gate. This supersedes historical homogeneous/closed body and generated-history header restrictions
for the execution/AD continuation below. Cold/wire constructor intersections remain independent;
there is no Metal Scan grant, Java low-arithmetic unlock or universal performance claim.

Both all-valid and length-aware bodies compose known ordinary PURE semantics through existing
owners, including per-value floating/discrete types, casts, canonical saved siblings and all-SHARED
work. Ordinary descriptor, effect, lexical containment, stable carry, public eligibility, resource
and per-backend numerical predicates still apply. No separate permanent Scan mathematical-kind
list is permitted. All-valid keeps ordinary whole-Tensor meaning: do not impose length-mode row
locality or fabricate a batch coordinate on it. Length mode keeps the proved original-row,
terminal, shared-carry isolation and exact contribution rules above.

The existing Model owner adds the scoped overload
`ScanBody.describeOrdinary(Operation, List<TensorDescriptor>, int, Optional<DynamicDimension>)`.
Its optional is the exact sole declared batch symbol, or empty for a static template. Reuse the
same ordinary descriptor and factored source/deferred symbolic guards; validate every operand,
canonical result and shape-bearing attribute even when counts are zero. Keep the three-argument
ordinary deferred-inference API unchanged. Source, observed pre-ID construction and independent
Prepare call the scoped guard; no Tensor/producer, sampled B, duplicate family validator or new
type is required. Reshape count equality uses Model's canonical dimension arithmetic, separately
from length-mode contributor-preserving row-boundary proof and fixed-feature overflow checks.

The same existing descriptor facade exposes
`ScanBody.closeOrdinaryLayouts(Operation, List<TensorDescriptor>, List<TensorDescriptor>)`
for cold completion of independently validated computed result layouts, and
`ScanBody.hasCanonicalContiguousResults(Operation)` for the existing geometry-independent generic
materialization fact. Move the ordinary Compiler closure decisions, rather than copying a family
inventory into CPU. Preserve the generic classifier's exact set and its symbolic
`BatchBindingPlan` acceptance boundary; specialized Conv2d/Conv3d closure remains separate from
that generic fact. This is not a second Scan mathematical-kind list or admission predicate.

Shared completion preserves types, Shapes, gradient flags, ordering and every resolved layout.
It reuses exact ordinary affine views, moves the existing noncontiguous-RESHAPE materialization
decision and factors the existing positive-static crop layout recipe into its Model family owner.
Negative-step, empty, named and unproved extraction layouts remain unresolved. Ordinary Tensor
construction and three-/four-argument descriptor guards do not change. No Tensor, producer,
storage or identifier is created, and Compiler type/Shape/constraint inference stays independent.
CPU may overlay only computed local projection descriptors and matching memory requirements;
the captured body and exact activity graph associations remain immutable. Caller/formal layouts,
actual-only batch authority and independent physical qualification remain strict.

A legal finite-time fold, including T=0, may publish an existing caller seed directly as a
selected gradient. That value still requires a real complete-graph physical declaration and
the existing caller-owned representation/publication path. Reuse the
[producerless published-value contract](runtime-prepare-engine.md#producerless-published-values)
for caller or constant sources; do not retain a fictitious reverse Scan, insert COPY/Scalar/
dummy dependencies, allocate a substitute Tensor or skip the publication. Target TensorId,
derivative order, target index, dtype, Shape, raw bits and caller lifetime remain exact.

Support true empty carry and explicit length mode independently of floating or ROW carry count,
including generated no-ROW/empty reverse. No dummy state, discrete cotangent or first-carry dtype/
batch inference. Each canonical value and positional boundary has its own ordinary dtype/layout;
carry updates retain exact own descriptors. Intermediate proved ROW(axis) histories require
sequence/formal support at every valid axis/rank, not only rank-two ROW(1). Complete canonical-B
and fixed-feature proofs remain; no equal-extent, empty-domain, sampled-B or B=1 shortcut.
Original length-AD sources still require all original carries and sequences ROW(0); broader
generated histories do not broaden those source bindings. Existing fixed/Conv3d-backward,
residual-root and higher-order exclusions remain independent.

Validate the entire original lexical inventory before derivative allocation, including unused
and unrelated branches. Separate source validity from selected derivative constructibility.
Preserve exact selected source-root/input pairs, ordinary D/ND/FC and ND-only versus disconnected
ZERO/ERROR policy. Compute finite-T carry/step/invariant routes and every legal length prefix,
including T=0 identity and T=1's missing second transition. Do not replace them with Cartesian
root/target unions, unrestricted transitive closure, sampled lengths or a T-sized derivative DAG.
Compact selected seeds, cotangent channels, sequence emissions, parameter accumulators and primal
histories retain explicit original declaration maps even when actual Tensor identities are tied.

Metadata-first and real expansion use the SAME ordinary derivative bodies and contribution
orchestration, including seed normalization, cast-back, term normalization, ordered merging,
disconnected publication and separate temporal ADDs. Model checks and independent Compiler
inference prove every actual created operation before derivative IDs, including canonical saved
siblings, eager unused temporaries and primal-only work. Descriptor reads and resolved-but-unused
handles do not demand history; reachable actual value use does. No fake Tensor/ID, shadow
derivative formula, source replay or numerical zero as an admission proof.

Structural cotangent roots are required carried-cotangent channels followed by selected
emission-seed formals. Scopes are finite unions of those roots, not numerical tests or a general
Boolean language. Each selected canonical-output VJP and all its generated prerequisites inherit
its incoming scope. Ordinary contribution-list merge executes no ADD, raw sole identity or the
existing ordered ADD for neither/one/both present terms. Temporal parameter accumulation is
different: each present completed contribution executes its distinct old.add(contribution),
including initial +0; absence preserves old raw. No has-ever-term or scatter-cell presence.

Activity is local to the exact generated occurrence. A retained external cotangent Tensor actual
initializes a present root regardless of its bits or its producer's locally absent work. Ordinary
Tensor publication exports numerical values, not structural activity; shared lengths alone do not
couple different occurrences' initial-presence bits. A missing channel remains different from a
bound Tensor containing zero.

Complete selected routes proved impossible for every compatible legal prefix assignment may be
omitted uniformly; selected-pair planning and actual construction must omit the same channels.
Once retained for any prefix, a bound Tensor seed is present when qualifying the consuming local
reverse over its legal prefixes. Source feasibility and this materialized reverse's FC and
target-aggregate ND-only obligations are separate proofs. Do not reuse producer-prefix absence
as consuming-root absence. Correlating prefixes also requires a proved canonical row mapping
through outer operations: identical lengths do not equate different rows, and an unproved mapping
does not itself invalidate ordinary mathematics.

Absence dominates validators, operand loads and arithmetic. Explicit zero, cancellation,
FLOOR/SIGN zeros and numerical WHERE/attention selection remain present. Inactive length rows
preserve numeric/control carry raw, and inactive emissions/sequence gradients publish positive
zero. All-valid empty Tensor occurrences remain present independently of scalar iteration count;
length B=0/all-zero/inactive-time no-body-work is separate. Terminal SUM, matched MATMUL and
shared-target ADD-scatter exclude absent/inactive contributors before their operand/index loads
while retaining ordinary represented order, rounding and the separate completed per-time ADD.

The finite neutral compiled activity carrier and its exact-body association belong to
[Planning](foundational-modules.md#general-scan-activity-transport). Compiler owns proof/capture;
Prepare/providers independently validate applicability, membership, bindings and ordinary
predicates. Scope metadata is not an admission certificate or a second numerical graph.
Rebuild/demand/optimization must preserve/remap it; no CSE/fusion/hoist may cross incompatible
activity, optional-merge or temporal-ADD meaning.

For each physical unit, complete all eligible active value-domain validation after inputs are
ready and BEFORE any of that unit's numerical output/history/scratch writes. Earlier completed
units need no rollback. Inactive/absent values are not inspected; ignored categorical targets,
noncanonical BOOL, gather/scatter component bounds, bounds-before-duplicates and nonempty-index/
empty-output behavior retain ordinary meanings. No new finite-input rule for normalization or
categorical loss follows. All lengths still validate before this Scan's first write, including
T=0. Metadata admission reads no values.

Reuse one existing emitted time loop and existing ordinary physical/math owners. No interpreter,
prepared-body child executor, runtime unit/graph lookup, per-step callback/packing/allocation or
new scheduler. Typed physical storage, control workspace, cursor reset and actual method inventory
are owned by the [CPU realization boundary](backend-execution.md#general-scan-ordinary-unit-realization).
Absent activity introduces no empty identity fields; unchanged frozen recipes retain bytes/keys.
New compact controlled graphs have selective identity and are not mislabeled old recipes.
The independently old cold/source/v1-v5 constructor/type/header/role intersection stays closed;
compiled activity is never serialized. CPU low numerical arithmetic remains locked and Metal's
separate first-production support remains mandatory.

