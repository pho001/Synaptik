# Task 0025E: Fixed Recurrent-[Scan](../../../../glossary.md#term-cumulative-scan) Semantic Family and [Tensor](../../../../glossary.md#tensor) Expressions

## Status

Complete

## Goal

Add the smallest [Model](../../../../glossary.md#term-model)-owned fixed recurrent-[scan](../../../../glossary.md#term-cumulative-scan) capability selected by
[ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>: one flat,
[identity](../../../../glossary.md#term-identity-meanings)-distinct, multi-output [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence) for each of `RNN_TANH`,
`GRU_RESET_AFTER`, and [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm), with one [immutable](../../../../glossary.md#term-immutable) `FORWARD` or `REVERSE` direction.

The public [Tensor](../../../../glossary.md#tensor) receiver is a fully static [time-major](../../../../glossary.md#term-time-major) sequence. Ordinary Tensor inputs [carry](../../../../glossary.md#term-carry)
[runtime](../../../../glossary.md#term-runtime) valid lengths, initial states, [weights](../../../../glossary.md#term-weight), and the optional packed input-side [bias](../../../../glossary.md#term-bias). One exact
[producer](../../../../glossary.md#term-producer) exposes a dense original-time-aligned output and explicit final states through [canonical](../../../../glossary.md#term-canonical-form)
Tensor wrappers:

```text
time-major values + runtime lengths + explicit state + explicit parameters + direction
  -> one immutable recurrent-scan producer occurrence
  -> [dense outputs, final hidden] or [dense outputs, final hidden, final cell]
```

This task constructs immutable Model [metadata](../../../../glossary.md#term-metadata) only. It does not read length values, execute a
loop, bind caller data, lower or [prepare](../../../../glossary.md#prepare) work, select a [backend](../../../../glossary.md#backend), or make the new operation
executable. [Compiler](../../../../glossary.md#term-compiler) task 0006A and later execution-layer work must adopt the closed family
explicitly.

## Motivation

The completed static [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) sequence containers construct one [Tensor](../../../../glossary.md#tensor) expression per time step and fix
sequence length in the Java call. That surface cannot represent an ordinary [runtime](../../../../glossary.md#term-runtime) length [vector](../../../../glossary.md#term-vector),
keep [graph](../../../../glossary.md#graph) size [constant](../../../../glossary.md#term-constant) in `time`, or allow a [backend](../../../../glossary.md#backend) to own a prepared bounded loop. [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012
selects a fixed flat [operation](../../../../glossary.md#operation) instead of a general region system so the [Model](../../../../glossary.md#term-model) can describe the
needed meaning without weakening existing graph, [compiler](../../../../glossary.md#term-compiler), Runtime, or [backend ownership](../../../../glossary.md#backend-ownership).

Publishing the Model contract first is also required by the repository's closed inventories.
Compiler 0006A cannot truthfully add [capture](../../../../glossary.md#term-capture-meanings) and [inference](../../../../glossary.md#term-inference) for a semantic family that Model does
not yet own, while Model must not imply executability before Compiler and a [concrete backend](../../../../glossary.md#concrete-backend) adopt
that family. This task establishes that narrow prerequisite and keeps every downstream adoption
explicit and [fail-closed](../../../../glossary.md#term-fail-closed).

## [Scope](../../../../glossary.md#term-scope)

- Add one [operation](../../../../glossary.md#operation)-family package containing exactly:
  - `RecurrentDirection`, the two-value [immutable](../../../../glossary.md#term-immutable) attributes enum;
  - `RecurrentScanKind`, the three-value kind enum with closed structural [signatures](../../../../glossary.md#term-signature).
- Add exactly two shallowly immutable public result records:
  - `RecurrentScanResult(Tensor outputs, Tensor finalHidden)`;
  - `LstmRecurrentScanResult(Tensor outputs, Tensor finalHidden, Tensor finalCell)`.
- Add one package-private, field-free `TensorRecurrentScanExpressions` construction helper.
- Add exactly six public receiver methods to [`Tensor`](../../../../glossary.md#tensor): [bias](../../../../glossary.md#term-bias)-free and biased overloads of
  `rnnScan`, `gruScan`, and `lstmScan`.
- Fix the exact operation input and output order, gate packing and transition equations, static
  Shape rules, exact data-type rules, gradient-request metadata, validation order, failure
  effects, identity, provenance, and zero-length semantics below.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Construct all outputs atomically through the existing canonical multi-output Tensor factory
  path.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- Extend production-signature coverage and every live public Tensor method-count lock from 202
  to 208.
- Add focused semantic and Tensor-construction tests for every family, direction, bias variant,
  descriptor rule, validation branch, effect boundary, and provenance invariant in this task.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Finalize affected Javadocs, architecture-status explanations, Tensor/Compile/Training API
  explanations, glossary terminology, Model capabilities, and planning records through the
  mandatory independent documentation-focused pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- arbitrary cells, callbacks, bodies, graph regions, nested graphs, free-variable capture,
  subgraphs, loop-carried graph values, general control flow, or another graph identity domain<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [capture glossary](../../../../glossary.md#term-capture-meanings); [identity domain glossary](../../../../glossary.md#term-identity-domain))</span>
- another recurrent kind, direction, activation, gate order, bias convention, peephole,
  projection, bidirectional occurrence, stacking, residual, dropout, sparse, quantized, stateful,
  or configurable variant<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [bias glossary](../../../../glossary.md#term-bias); [occurrence glossary](../../../../glossary.md#term-occurrence); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- dynamic or binding-dependent Shapes, arbitrary masks, inferred lengths, padding sentinels,
  active-row compaction, sorting, packed-sequence buffers, or skipped-work claims<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Shapes glossary](../../../../glossary.md#shape); [masks glossary](../../../../glossary.md#term-boolean-mask); [padding glossary](../../../../glossary.md#padding); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- reading, inspecting, validating, copying, or retaining runtime valid-length values during Model
  construction<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model))</span>
- storage, caller binding, graph compilation, inference adoption, graph finalization, backward
  graph construction, backpropagation through time (BPTT), canonicalization, lowering, planning,
  preparation, execution, backend capability, runtime, engine, or training behavior<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span><span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [inference glossary](../../../../glossary.md#term-inference); [backward graph glossary](../../../../glossary.md#backward-graph); [backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [lowering glossary](../../../../glossary.md#lowering); [planning glossary](../../../../glossary.md#term-planning); [backend capability glossary](../../../../glossary.md#backend-capability); [engine glossary](../../../../glossary.md#term-engine))</span>
- compiler, planning, prepare, runtime, engine, backend, NN, Data, architecture-test,
  backend-conformance, integration-test, Gradle, dependency, module-descriptor, or serialization
  source changes<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [descriptor glossary](../../../../glossary.md#term-descriptor); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- migrating or replacing the current static [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) cell and sequence APIs
- hidden recurrent state, hidden outputs, parameters, buffers, random-number-generator state,
  mode, counters, mutation, I/O, or external resources<span data-glossary-navigation="true"> ([recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- changing current scalar, elementwise, MATMUL, shape, promotion, broadcasting, inference,
  autograd, capability-query, Tensor factory, Tensor producer, or graph-capture contracts<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [elementwise glossary](../../../../glossary.md#term-elementwise); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [broadcasting glossary](../../../../glossary.md#broadcasting); [autograd glossary](../../../../glossary.md#autograd); [Tensor factory glossary](../../../../glossary.md#tensor-factory); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer))</span>
- creating or promoting Compiler 0006A, backend, Engine, NN 0021B, NN 0022, or Data task
  specifications
- editing the global roadmap, completed task specifications, ADR 0012, or unrelated source and
  documentation<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially fixed recurrent scan,
  Model ownership, immutable operation/provenance, static Shape, compile, Runtime, Engine, and
  training boundaries<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [Model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [Shape glossary](../../../../glossary.md#shape); [compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)
- [Model capabilities](../capabilities.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0019](0019-matmul-semantics-and-tensor-expression.md)
- [Task 0025](0025-canonical-tensor-producer-outputs.md)
- [NN task 0021A](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [NN master plan](../../../extensions/nn/master-plan.md), including the completed recurrent-cell
  and sequence tasks 0012 through 0020C
- [Compiler master plan](../../compiler/master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>, especially future task 0006A
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work remains inside `modules/model` plus directly affected architecture-status,
  explanatory/API, glossary, capability, and Model planning documentation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- One recurrent [scan](../../../../glossary.md#term-cumulative-scan) is one ordinary flat [`Operation`](../../../../glossary.md#operation) and one [identity](../../../../glossary.md#term-identity-meanings)-distinct
  [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), independent of `time`. It has no body, region, callback, nested graph, or
  auxiliary per-time Tensor producer.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [graph glossary](../../../../glossary.md#graph); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer))</span>
- `RecurrentDirection` is the complete [immutable](../../../../glossary.md#term-immutable) attributes value. It contains only `FORWARD` and
  `REVERSE` and implements [`OperationAttrs`](../../../../glossary.md#operationattrs); no second attributes or [descriptor](../../../../glossary.md#term-descriptor) type is added.
- `RecurrentScanKind` is the complete semantic family. Its [constants](../../../../glossary.md#term-constant) contain only `RNN_TANH`,
  `GRU_RESET_AFTER`, and [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) and implement [`OperationKind`](../../../../glossary.md#operationkind).
- The family owns logical meaning and descriptor-visible construction. It owns no execution
  algorithm, backend route, capability claim, prepared state, storage, caller value, gradient
  rule, or lifecycle state.<span data-glossary-navigation="true"> ([backend route glossary](../../../../glossary.md#backend-route); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Shapes are fully static at construction. The valid-length Tensor is runtime data, not a hidden
  Shape-binding mechanism.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding))</span>
- Every floating value/state/parameter input has the receiver's exact floating data type. The
  helper must not invoke promotion or accept an otherwise promotable mixed-type combination.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Valid lengths have exact type [`INT64`](../../../../glossary.md#data-type--datatype), [rank](../../../../glossary.md#term-rank) one, [Shape](../../../../glossary.md#shape) `[batch]`, and
  `requiresGrad == false`. [Model](../../../../glossary.md#term-model) does not read or infer their values.
- Every output uses unresolved layout and the same exact floating data type. Gradient eligibility
  is the OR of all floating input, state, weight, and supplied bias roles; valid lengths never
  contribute.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias))</span>
- Every output returned through a result record is the canonical exact wrapper for its indexed
  producer slot. A successful call exposes every producer output; none is hidden.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Current graph capture may observe the producer as one ordinary flat multi-output occurrence.
  Compiler adoption remains closed: forward compilation fails on the unsupported kind until
  Compiler 0006A, and every backward-capable request fails closed before derivative Tensor
  construction. Unrelated kinds retain their current behavior.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [occurrence glossary](../../../../glossary.md#term-occurrence); [Compiler glossary](../../../../glossary.md#term-compiler); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- No dependency direction, module ownership, graph identity, public exception layer, or
  architecture decision changes.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Stop if implementation requires a different public signature, another type, a compiler/runtime/
  backend/NN edit, dynamic Shapes, region semantics, or any architecture change.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [backend glossary](../../../../glossary.md#backend); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.datatype`

Packages added or changed:

- add `io.github.pho001.synaptik.model.operation.recurrent` for the closed recurrent-scan
  semantic vocabulary;
- change `io.github.pho001.synaptik.model.tensor` only for the public facade, two result records,
  and one package-private construction helper.

Type placement:

- `io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection` — owns the exact
  direction parameter and is itself the immutable <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [immutable glossary](../../../../glossary.md#term-immutable))</span>[`OperationAttrs`](../../../../glossary.md#operationattrs) value.
- `io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind` — owns the three fixed
  semantic identities and their exact structural signatures.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- `io.github.pho001.synaptik.model.tensor.RecurrentScanResult` — exposes the two canonical RNN/GRU
  outputs in producer-slot order.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [producer glossary](../../../../glossary.md#term-producer))</span>
- `io.github.pho001.synaptik.model.tensor.LstmRecurrentScanResult` — exposes the three canonical
  LSTM outputs in producer-slot order.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- `io.github.pho001.synaptik.model.tensor.TensorRecurrentScanExpressions` — owns family-local
  validation, descriptor inference, operation construction, and result-carrier assembly.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [inference glossary](../../../../glossary.md#term-inference); [operation glossary](../../../../glossary.md#operation); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` — remains the public receiver [facade](../../../../glossary.md#term-facade).

No generic [scan](../../../../glossary.md#term-cumulative-scan) utility, recurrent [descriptor](../../../../glossary.md#term-descriptor), recurrent attributes record, builder, facade,
manager, [compiler](../../../../glossary.md#term-compiler) adapter, or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) type is added.

## Exact public [API](../../../../glossary.md#term-application-programming-interface-api) and [occurrence](../../../../glossary.md#term-occurrence) [signatures](../../../../glossary.md#term-signature)

### Direction and kind enums

Add exactly:

```java
public enum RecurrentDirection implements OperationAttrs {
    FORWARD,
    REVERSE
}

public enum RecurrentScanKind implements OperationKind {
    RNN_TANH,
    GRU_RESET_AFTER,
    LSTM
}
```

`RecurrentScanKind.signatures()` returns stable [immutable](../../../../glossary.md#term-immutable) lists with exactly these structural
contracts:

```text
RNN_TANH:       RecurrentDirection, 5..6 inputs, exactly 2 outputs
GRU_RESET_AFTER: RecurrentDirection, 5..6 inputs, exactly 2 outputs
LSTM:           RecurrentDirection, 6..7 inputs, exactly 3 outputs
```

Use `OperationSignature.inputRange(...)`. Each kind accepts the exact
`RecurrentDirection.class` only. Do not add a permissive [signature](../../../../glossary.md#term-signature), direction-specific kind,
no-attributes form, or second attributes implementation.

### Result records

Add exactly:

```java
public record RecurrentScanResult(Tensor outputs, Tensor finalHidden)

public record LstmRecurrentScanResult(
        Tensor outputs, Tensor finalHidden, Tensor finalCell)
```

Each compact constructor checks component [nullability](../../../../glossary.md#term-nullability) in declaration order, with the component
name as the null message, and retains each exact reference. The records perform no [descriptor](../../../../glossary.md#term-descriptor),
[producer](../../../../glossary.md#term-producer), [operation](../../../../glossary.md#operation), slot, storage, [compiler](../../../../glossary.md#term-compiler), or execution [validation](../../../../glossary.md#term-validation) for independently supplied
[Tensors](../../../../glossary.md#tensor). They add no alternate constructor, factory, accessor [alias](../../../../glossary.md#aliasing-and-arena), method, interface, or state.

### [Tensor](../../../../glossary.md#tensor) receiver methods

Add exactly these six public non-static, non-synchronized, non-varargs methods:

```java
public RecurrentScanResult rnnScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

public RecurrentScanResult rnnScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

public RecurrentScanResult gruScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

public RecurrentScanResult gruScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

public LstmRecurrentScanResult lstmScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

public LstmRecurrentScanResult lstmScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)
```

The receiver is `input`. Each [facade](../../../../glossary.md#term-facade) method contains one return statement delegating directly to
the matching helper entry with the exact [argument](../../../../glossary.md#term-argument) order. It does not read fields, validate,
construct [descriptors](../../../../glossary.md#term-descriptor) or operations, or delegate through another public Tensor method.

There are exactly two public overloads for each of `rnnScan`, `gruScan`, and `lstmScan`; no [alias](../../../../glossary.md#aliasing-and-arena),
static form, optional/null [bias](../../../../glossary.md#term-bias), default direction, generic recurrent method, descriptor argument,
or convenience method is added. The public Tensor declared-method inventory changes exactly from
202 to 208.

### Ordered inputs and outputs

The helper constructs exact [immutable](../../../../glossary.md#term-immutable) input snapshots in this order:

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

Output [descriptor](../../../../glossary.md#term-descriptor) and [canonical](../../../../glossary.md#term-canonical-form) wrapper order is exactly:

```text
RNN/GRU: [outputs, finalHidden]
LSTM:    [outputs, finalHidden, finalCell]
```

## Fixed mathematical semantics

For one valid original-time [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), let row [vector](../../../../glossary.md#term-vector) `x` be the input, `h` the previous hidden
state, and, for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), `c` the previous cell state. Let `W` be `inputWeight`, `U` be
`hiddenWeight`, and let optional `b` be added only to the input projection. [Matrix](../../../../glossary.md#term-matrix-meanings) products,
[elementwise](../../../../glossary.md#term-elementwise) arithmetic, [sigmoid](../../../../glossary.md#term-sigmoid), and [tanh](../../../../glossary.md#term-tanh) have the represented-value meanings of the current
[Model](../../../../glossary.md#term-model) operations. These equations fix observable semantic association and gate packing; they do
not prescribe a [decomposition](../../../../glossary.md#term-decomposition), numerical [kernel](../../../../glossary.md#kernel), [accumulator](../../../../glossary.md#accumulator-type) [widening](../../../../glossary.md#term-widening), [fusion](../../../../glossary.md#term-fusion), or [backend](../../../../glossary.md#backend)
algorithm.

`RNN_TANH`, gate count 1:

```text
h' = tanh((x @ transpose(W) + optional b) + (h @ transpose(U)))
```

`GRU_RESET_AFTER`, gate count 3, packed reset/update/candidate order:

```text
P_x = x @ transpose(W) + optional b
P_h = h @ transpose(U)

r = sigmoid(P_x[r] + P_h[r])
z = sigmoid(P_x[z] + P_h[z])
n = tanh(P_x[n] + r * P_h[n])
h' = n + z * (h - n)
```

The reset gate is applied after the recurrent candidate projection. There is no recurrent-side
[bias](../../../../glossary.md#term-bias).

`LSTM`, gate count 4, packed input/forget/candidate/output order:

```text
P_x = x @ transpose(W) + optional b
P_h = h @ transpose(U)

i = sigmoid(P_x[i] + P_h[i])
f = sigmoid(P_x[f] + P_h[f])
g = tanh(P_x[g] + P_h[g])
o = sigmoid(P_x[o] + P_h[o])
c' = f * c + i * g
h' = o * tanh(c')
```

There is no recurrent-side bias, peephole, projection, configurable [activation](../../../../glossary.md#term-activation), or hidden
[parameter](../../../../glossary.md#term-parameter) convention. The [scan](../../../../glossary.md#term-cumulative-scan) is semantically equivalent to repeated application of these fixed
transitions but is not represented as repeated [Tensor](../../../../glossary.md#tensor) expressions.

## Static [descriptors](../../../../glossary.md#term-descriptor), valid lengths, and traversal

For receiver [Shape](../../../../glossary.md#shape) `[time, batch, inputSize]` and hidden size `hiddenSize`:

| Role | Exact requirement |
|---|---|
| `input` | fully static [rank](../../../../glossary.md#term-rank) 3 `[time, batch, inputSize]`; floating |
| `validLengths` | fully static rank 1 `INT64[batch]`; `requiresGrad == false` |
| `initialHidden` | fully static `[batch, hiddenSize]` |
| `initialCell` | [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) only; fully static `[batch, hiddenSize]` |
| `inputWeight` | fully static `[gateCount * hiddenSize, inputSize]` |
| `hiddenWeight` | fully static `[gateCount * hiddenSize, hiddenSize]` |
| [`bias`](../../../../glossary.md#term-bias) | optional; fully static `[gateCount * hiddenSize]` |
| `outputs` | `[time, batch, hiddenSize]` |
| `finalHidden` | `[batch, hiddenSize]` |
| `finalCell` | LSTM only; `[batch, hiddenSize]` |

`inputSize` and `hiddenSize` are positive. `time` and [`batch`](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) may be zero. Gate count is 1 for
[RNN](../../../../glossary.md#term-recurrent-neural-network-rnn), 3 for [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru), and 4 for LSTM. The checked product `gateCount * hiddenSize` must be representable
as a Shape [extent](../../../../glossary.md#term-extent).

Input, initial states, [weights](../../../../glossary.md#term-weight), and optional bias have one exact common floating [data type](../../../../glossary.md#data-type--datatype). No
promotion or conversion is implied. Outputs retain that exact type. Every output [layout](../../../../glossary.md#layout) is
unresolved.

Every output has one common `requiresGrad` value equal to the logical OR of the request flags on
`input`, `initialHidden`, LSTM `initialCell`, `inputWeight`, `hiddenWeight`, and supplied `bias`.
`validLengths` is excluded. The flag records only [gradient](../../../../glossary.md#term-gradient) eligibility/request [metadata](../../../../glossary.md#term-metadata); this task
does not make a backward-capable [compile](../../../../glossary.md#compile) succeed.

At future execution, for each original batch row `b`, the complete [runtime](../../../../glossary.md#term-runtime) length [vector](../../../../glossary.md#term-vector) is
validated with `0 <= L[b] <= time` before output representation mutation. [Model](../../../../glossary.md#term-model) neither reads nor
stores those [scalar](../../../../glossary.md#term-scalar) values beyond the ordinary `validLengths` [Tensor](../../../../glossary.md#tensor) input.

- `FORWARD` consumes original [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `0` through `L[b] - 1`.
- `REVERSE` consumes original coordinates `L[b] - 1` through `0`; it reverses only the valid
  prefix and never traverses the padded suffix.
- Each valid original coordinate stores the next hidden state after consuming that coordinate.
- Each padded original coordinate `t >= L[b]` stores the exact positive zero of the common data
  type.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Final hidden, and LSTM final cell, is the state after the last consumed coordinate.
- A zero-length row has positive-zero output at every coordinate and returns the exact initial
  state values semantically.
- If `time == 0`, every future runtime length must be zero, the dense output is empty, and the
  initial states are preserved semantically.

These are semantic obligations for later executable adoption, not Model-time value checks or a
claim that a [backend](../../../../glossary.md#backend) currently skips work.

## Construction [validation](../../../../glossary.md#term-validation) and effects

The package-private helper performs validation in this observable order. Each step completes
before the next begins:

1. Null-check supplied [parameters](../../../../glossary.md#term-parameter) in public declaration order. For biased overloads, check
   [`bias`](../../../../glossary.md#term-bias) before `direction`; for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), check `initialCell` before weights. The receiver cannot be
   null during an instance call.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
2. Validate `input`: exact floating [data type](../../../../glossary.md#data-type--datatype), [rank](../../../../glossary.md#term-rank) 3, fully static [Shape](../../../../glossary.md#shape); extract `time`, [`batch`](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters),
   and `inputSize`; allow zero `time` and `batch` and require positive `inputSize`.
3. Validate `validLengths`: exact [`INT64`](../../../../glossary.md#data-type--datatype), `requiresGrad == false`, rank 1, fully static Shape,
   and exact extent `batch`.
4. Validate `initialHidden`: exact input data type, rank 2, fully static Shape, exact batch extent,
   and positive `hiddenSize`.
5. For LSTM, validate `initialCell`: exact input data type, rank 2, fully static Shape, and exact
   `[batch, hiddenSize]`.
6. Compute `gateCount * hiddenSize` with checked arithmetic. [Overflow](../../../../glossary.md#term-overflow) fails here.
7. Validate `inputWeight`: exact input data type, rank 2, fully static Shape, and exact
   `[gateCount * hiddenSize, inputSize]`.
8. Validate `hiddenWeight`: exact input data type, rank 2, fully static Shape, and exact
   `[gateCount * hiddenSize, hiddenSize]`.
9. When supplied, validate `bias`: exact input data type, rank 1, fully static Shape, and exact
   `[gateCount * hiddenSize]`.
10. Derive ordered output [descriptors](../../../../glossary.md#term-descriptor), create `Operation(kind, direction)`, form the exact ordered
    input snapshot, and invoke the existing factory-atomic canonical multi-output path.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
11. Assemble the appropriate result record from the returned [canonical](../../../../glossary.md#term-canonical-form) wrappers in slot order.

Use stable, role-specific exception messages consistent with current [Tensor](../../../../glossary.md#tensor) expression helpers.
Null failures use `NullPointerException`; invalid type, [gradient](../../../../glossary.md#term-gradient) flag, rank, static [extent](../../../../glossary.md#term-extent),
[dimension](../../../../glossary.md#dimension) relation, or checked-size condition uses `IllegalArgumentException`. Do not catch,
translate, or suppress Tensor identifier exhaustion.

Every null/type/rank/static-Shape/size/overflow failure occurs before a Tensor identifier is
allocated and performs no mutation. [Runtime](../../../../glossary.md#term-runtime) length bounds cannot fail here because no value is
read. During factory output creation, output identifiers are allocated in slot order; if later
identifier allocation fails, earlier identifiers may remain consumed, no [producer](../../../../glossary.md#term-producer) or result
[carrier](../../../../glossary.md#term-storage-carrier) is published, and no partial result is returned. This preserves the existing
`TensorFactory.createDerivedOutputs` failure contract.

Descriptor construction should preserve exact [immutable](../../../../glossary.md#term-immutable) Shape references when their meaning is
already exact: final hidden may reuse `initialHidden.shape()`, final cell may reuse
`initialCell.shape()`, and the dense output should use the exact static `time`, `batch`, and
`hiddenSize` Dimension values obtained from the validated inputs. Do not mutate or normalize any
input Tensor, Shape, descriptor, producer, or caller collection.

## [Identity](../../../../glossary.md#term-identity-meanings), [provenance](../../../../glossary.md#provenance), [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), and [fail-closed](../../../../glossary.md#term-fail-closed) adoption

Every successful call creates one fresh [`Operation`](../../../../glossary.md#operation), one fresh identity-distinct [producer](../../../../glossary.md#term-producer), one
fresh result [carrier](../../../../glossary.md#term-storage-carrier), and exactly two or three fresh output [Tensors](../../../../glossary.md#tensor) and Tensor identifiers in
output order. Two calls with equal operands and direction remain distinct occurrences.

All result components from one call:

- retain the same exact producer;
- expose the same exact operation, kind, direction object, immutable ordered input snapshot, and
  ordered output-descriptor snapshot through that producer;<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- are the producer's [canonical](../../../../glossary.md#term-canonical-form) exact wrappers at indices 0, 1, and, for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), 2;
- have distinct Tensor identifiers and the correct `producerOutputIndex`; and
- create no gate, projection, state-step, [mask](../../../../glossary.md#term-boolean-mask), zero-fill, or per-time intermediate Tensor.

The existing generic [graph](../../../../glossary.md#graph)-[capture](../../../../glossary.md#term-capture-meanings) path may therefore capture one ordinary multi-output [node](../../../../glossary.md#node) and
its ordinary input edges. This task does not change graph capture or promise a successful complete
[compile](../../../../glossary.md#compile).

Until [Compiler](../../../../glossary.md#term-compiler) 0006A explicitly adopts the family:

- a forward-only complete compile that reaches current [inference](../../../../glossary.md#term-inference) [dispatch](../../../../glossary.md#term-dispatch) rejects the unknown
  `RecurrentScanKind` as unsupported before [planning](../../../../glossary.md#term-planning) or [backend capability](../../../../glossary.md#backend-capability) selection;
- a backward-capable request fails closed through the current closed autograd/signature coverage
  boundary before constructing derivative Tensors; and<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [signature glossary](../../../../glossary.md#term-signature); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- existing compiler branches and capability inventories remain unchanged, so unrelated operation
  families compile exactly as before.

No current [backend](../../../../glossary.md#backend) advertises this family, and no execution, [runtime](../../../../glossary.md#term-runtime) [validation](../../../../glossary.md#term-validation), skipped-work,
[lowering](../../../../glossary.md#lowering), or backend-support claim is made.

## Affected files

Expected production source:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentDirection.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/RecurrentScanResult.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/LstmRecurrentScanResult.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRecurrentScanExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRecurrentScanExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `ARCHITECTURE.md`
- `docs/architecture/lifecycle.md`
- `docs/architecture/module-boundaries.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/architecture/training-graph.md`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md`

No other file is expected.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- 6 production Java files;
- 16 [Model](../../../../glossary.md#term-model) test files;
- 12 architecture-status, explanatory/[API](../../../../glossary.md#term-application-programming-interface-api), glossary, capability, and Model [planning](../../../../glossary.md#term-planning) files;
- 34 files total.

Documentation changes may update current-versus-planned status and explain the exact public
[metadata](../../../../glossary.md#term-metadata) contract. They must not change the architecture decision. `current-architecture-plan.md`
and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012 remain accurate navigation/history and are review-only.

If another file, type, public member, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), [architecture test](../../../../glossary.md#term-architecture-test), [conformance](../../../../glossary.md#term-conformance) test,
[integration test](../../../../glossary.md#term-integration-test), build change, or broader documentation change is needed, stop and propose a
follow-up task.

## Acceptance criteria

- `RecurrentDirection` contains exactly `FORWARD` and `REVERSE`, implements [`OperationAttrs`](../../../../glossary.md#operationattrs), and
  is the exact attributes type for every recurrent kind.
- `RecurrentScanKind` contains exactly `RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) and exposes only
  the exact 5..6/2, 5..6/2, and 6..7/3 signatures defined above.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- The two records have exactly the specified components, check nulls in component order, and
  retain exact references without extra validation or methods.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [`Tensor`](../../../../glossary.md#tensor) adds exactly the six specified receiver methods and no other public declaration; its
  declared public method count is exactly 208.
- Every kind, bias variant, ordered input snapshot, output order, direction, gate packing,
  equation, static Shape, exact data type, unresolved layout, and gradient-request rule matches
  this specification.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [snapshot glossary](../../../../glossary.md#term-snapshot); [Shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- RNN/GRU and LSTM results expose exactly two and three canonical producer wrappers respectively;
  all identity, shared-provenance, indexed-slot, freshness, and no-intermediate invariants pass.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- `time == 0` and `batch == 0` construct valid metadata when all other static requirements hold;
  positive <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>`inputSize` and `hiddenSize` remain mandatory.
- Every construction-time failure class, validation order, no-identifier-allocation boundary, and
  factory identifier-exhaustion effect matches this specification.
- Tests prove that Model construction never reads length storage or values and accepts ordinary
  unbound runtime-input metadata satisfying the descriptor contract.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [runtime glossary](../../../../glossary.md#term-runtime); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Tests and documentation state zero-length, reverse-prefix, positive-zero padding, and final-state
  semantics without claiming current execution or skipped work.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- Existing Compiler, autograd, graph-capture, capability, Runtime, Engine, backend, and unrelated
  Model code remain byte-for-byte unchanged except for the explicitly listed Model files.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [graph glossary](../../../../glossary.md#graph); [capture glossary](../../../../glossary.md#term-capture-meanings); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- Documentation clearly distinguishes the now-current Model construction surface from still-
  planned Compiler, execution, NN coordination, and BPTT support.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- Model capabilities list the family as constructible metadata and explicitly non-executable
  until downstream closed inventories adopt it.
- No compiler/backend/NN/Data/global-roadmap task is promoted or specified by this change; Model
  0026 remains Draft without a task specification.
- Every affected public or package-private type and method has detailed Javadoc covering inputs,
  constraints, nullability, outputs, failure conditions, provenance, and non-execution boundaries.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- A separate documentation-focused agent pass has finalized affected explanatory documentation,
  Javadoc, glossary impact, and planning status in this same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation context runs focused tests while developing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanSemanticsTest \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorRecurrentScanExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMeanSquaredErrorExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSlicePlacementExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSumToShapeExpressionTest
```

Then [run](../../../../glossary.md#term-run) the [Model](../../../../glossary.md#term-model) task tier once:

```bash
./gradlew :modules:model:test
```

The separate documentation-focused context reuses the successful Java-test evidence unless it
changes executable Java behavior. It runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The implementation and documentation contexts also record these automated or justified manual
checks:

- exact enum constants, signatures, record components, six Tensor methods, and public method
  count 208 are locked by Model tests rather than ad hoc reflection or bytecode commands;<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- package/import [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) and [Javadoc](../../../../glossary.md#term-javadoc) prove the public [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape);
- documentation links, heading anchors, code fences, final newlines, and trailing whitespace are
  checked against the final changed Markdown files and recorded with the checker or manual method
  used;
- the final changed-path inventory is exactly within the 34 allowed paths, with no Compiler,
  Runtime, Engine, backend, NN, Data, architecture-test, integration-test, conformance-test,
  Gradle, dependency, ADR, current-architecture-index, or global-roadmap change; and<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- task/master status is synchronized only after all implementation, documentation, and validation
  evidence passes.

Repository-wide validation is deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) or the next recorded cross-[module](../../../../glossary.md#term-module) recurrent-[scan](../../../../glossary.md#term-cumulative-scan)
capability [checkpoint](../../../../glossary.md#term-checkpoint). This is one Model-module task; it changes no [dependency](../../../../glossary.md#term-dependency), architecture
boundary, shared build configuration, or executable cross-module contract. [Compiler](../../../../glossary.md#term-compiler) 0006A owns
its focused [capture](../../../../glossary.md#term-capture-meanings)/[inference](../../../../glossary.md#term-inference)/[preflight](../../../../glossary.md#term-preflight) tests and later end-to-end execution adoption owns
integration and [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) validation.

## Dependencies

- Accepted [ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
  and completed [NN task 0021A](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
  fix the architecture, surface, ordering, static descriptors, and layer ownership.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Completed [Model](../../../../glossary.md#term-model) 0018K supplies exact family-owned `OperationSignature` [validation](../../../../glossary.md#term-validation).
- Completed Model 0018L and 0025 supply shared multi-output provenance and canonical output
  wrappers.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Completed Model 0018N and existing Shape/data-type contracts supply exact floating type and
  static descriptor vocabulary.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Completed Model 0019 and current NN recurrent-cell tasks supply the selected MATMUL,
  sigmoid/tanh, gate-packing, bias, and transition meanings.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [sigmoid glossary](../../../../glossary.md#term-sigmoid); [tanh glossary](../../../../glossary.md#term-tanh); [bias glossary](../../../../glossary.md#term-bias))</span>
- Model 0025D is the latest completed detailed Model task. Therefore 0025E is the first unfinished
  task in Model master-plan order and is the next executable Model frontier.

## Follow-up tasks

- Compiler 0006A must explicitly adopt one-node forward capture, descriptor inference, final
  validation, closed inventory coverage, and fail-closed backward handling for this family. It
  must not be specified or promoted in this task.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [node glossary](../../../../glossary.md#node); [capture glossary](../../../../glossary.md#term-capture-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [closed inventory glossary](../../../../glossary.md#term-closed-inventory); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Later concrete-backend and execution-layer tasks must choose exact capability truth, lowering,
  prepared-loop behavior, complete runtime length validation before output mutation, Engine
  binding/publication, and evidence for any skipped-work claim.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication))</span>
- Later NN 0021B/0022 and Data coordination may expose sequence/module conveniences while
  preserving existing static APIs and passing lengths as an ordinary Tensor.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [Tensor glossary](../../../../glossary.md#tensor))</span>
- BPTT requires a separate compiler architecture decision covering saved state versus
  recomputation and exact derivative semantics.<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Dynamic Shapes, bidirectional composition, packed/compacted execution, broader recurrent
  variants, and serialization changes remain separate future decisions.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Model 0026 remains the next Draft Model task after 0025E and receives no detailed specification
  here.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

## Architecture impact

Expected impact: None.

This task implements the exact [Model](../../../../glossary.md#term-model) surface and [ownership](../../../../glossary.md#term-ownership-meanings) already authorized by
`ARCHITECTURE.md` and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012. The documentation pass changes planned/current status wording but
does not change an architecture rule. If implementation requires another [signature](../../../../glossary.md#term-signature), [operation](../../../../glossary.md#operation)
[shape](../../../../glossary.md#shape), ownership rule, execution promise, or architecture edit beyond status reconciliation, stop
and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are the clean-context implementation agent for Synaptik Model task 0025E in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/model/master-plan.md, and
docs/planning/modules/model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md
completely. Read the directly referenced ADR, focused architecture documents, prerequisite Model
and NN contracts, and actual affected source/tests before editing.

Implement task 0025E exactly as specified and run its focused and Model validation. Do not
implement out-of-scope Compiler, Runtime, Engine, backend, NN, Data, roadmap, architecture, or
general-control-flow work. Preserve unrelated and concurrent worktree changes. Stop and report if
the specification conflicts with architecture or requires more than its maximum scope.

After Java implementation and successful Model tests, hand the diff and exact test evidence to a
separate documentation-focused agent/thread with clean context. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs,
architecture-status explanations, API documentation, glossary, capabilities, and planning status,
reuse successful Java evidence unless it changes executable behavior, and run the specified
documentation validation.

At the end, update the task's validation evidence, local decisions, implementation notes,
completion summary, and final status. Mark Complete only after every acceptance criterion and the
documentation pass succeed.
```

## Local decisions

- The direction enum itself is the attributes value. This uses the intended immutable-enum form
  of <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>[`OperationAttrs`](../../../../glossary.md#operationattrs), keeps operation-to-tensor dependency direction intact, and avoids a
  redundant one-field <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [tensor glossary](../../../../glossary.md#tensor); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>`RecurrentScanAttrs` record.
- One `RecurrentScanKind` owns all three fixed meanings because they share direction vocabulary,
  time-major/static descriptor rules, and one construction helper while retaining distinct enum
  identity and kind-specific signatures.<span data-glossary-navigation="true"> ([time-major glossary](../../../../glossary.md#term-time-major); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [Bias](../../../../glossary.md#term-bias) optionality is represented only by exact input-count ranges and two explicit overloads.
  `null`, an option wrapper, a [boolean](../../../../glossary.md#data-type--datatype), or a second attributes type cannot create a hidden variant.
- The task extends current canonical multi-output factory behavior instead of adding a recurrent
  producer/result mechanism.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>
- No unresolved architecture or public-[API](../../../../glossary.md#term-application-programming-interface-api) decision remains for implementation.

## Known limitations

- The new Model metadata is intentionally non-executable until Compiler 0006A and downstream
  closed inventories adopt it.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Every backward-capable [compile](../../../../glossary.md#compile) remains unsupported; [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) is not designed by this task.
- Only fully static [Shapes](../../../../glossary.md#shape), [runtime](../../../../glossary.md#term-runtime) `INT64[batch]` prefix lengths, one direction per occurrence,
  dense original-time-aligned output, and the three fixed cell equations are represented.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Model construction cannot validate runtime length scalar bounds and does not promise physical
  skipped work.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Current [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) APIs are not migrated and no public runtime input-[binding](../../../../glossary.md#term-binding) convenience is added.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/model_0025e_planning` read the required repository instructions,
[architecture contract](../../../../glossary.md#architecture-contract) and focused [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[runtime](../../../../glossary.md#term-runtime)/training explanations, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012,
planning guide and roadmap, [Model](../../../../glossary.md#term-model)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Compiler](../../../../glossary.md#term-compiler) master plans, completed prerequisite Model and NN
tasks, current Model source/tests/Javadocs/capabilities, and the current closed Compiler [capture](../../../../glossary.md#term-capture-meanings),
[inference](../../../../glossary.md#term-inference), [autograd](../../../../glossary.md#autograd), and capability boundaries. It found no architecture or ordering conflict.

Planning-stage validation:

- A read-only Markdown checker validated this task and the Model master plan: two files and 209
  repository-local links, including heading anchors, with balanced fences, final newlines, and no
  trailing whitespace.
- `git diff --check` passed for tracked worktree changes, and
  `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md`
  returned the expected difference status 1 with no whitespace diagnostic for the new untracked
  specification.
- Exact-status checks found one [`Ready`](../../../../glossary.md#term-ready) row in the Model master plan and this task as the only
  `Ready` detailed Model task. No stale 0025E Draft wording remains in the Model master plan.
- Model 0025D remains Complete; 0025E is the first unfinished Model task; Model 0026 and Compiler
  0006A remain Draft without task specifications.
- The planning context created or changed exactly the Model master plan and this new task. It did
  not touch the pre-existing concurrent CPU/backend, Data-master, or global-roadmap worktree
  changes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- The future implementation maximum contains exactly 34 paths: 6 production Java files, 16 Model
  test files, and 12 directly affected documentation/planning files. It contains no Compiler,
  Runtime, Engine, backend, NN, Data, architecture-test, conformance-test, integration-test,
  Gradle, dependency, ADR, current-architecture-index, or global-roadmap change.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>

No Java test, [Javadoc](../../../../glossary.md#term-javadoc) generation, or build command ran in the planning context because that pass
created planning only and did not implement task 0025E.

Implementation context `/root/model_0025e_implementation` validation after executable freeze:

- `./gradlew :modules:model:test` with the sixteen task-specified focused test filters passed 153
  tests with zero failures, errors, or skips. An earlier focused development run exposed two test-
  fixture ID-accounting mistakes; the fixtures were corrected before this final focused run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- The one authoritative unfiltered `./gradlew :modules:model:test` passed 1,046 tests with zero
  failures, errors, or skips. Executable Java did not change afterward.
- Preliminary `./gradlew :modules:model:javadoc` passed after the production surface was complete.
  The documentation-focused context must run the final Javadoc gate after its independent review.
- External package-use [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) against `modules/model/build/classes/java/main` passed for all
  three result forms, both directions, and biased/unbiased public signatures. <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`javap` showed
  exactly two direction constants, three kind constants, the two- and three-component records,
  and exactly six Tensor scan overloads.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- A temporary compiler-package boundary probe compiled and ran after
  `./gradlew :modules:compiler:classes`: generic graph capture produced exactly one flat node with
  two ordered outputs; current captured inference rejected <span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [node glossary](../../../../glossary.md#node))</span>`RecurrentScanKind` as an unsupported
  operation kind; current autograd preflight rejected it as an unknown/unclassified kind before a
  derivative Tensor identifier was allocated. No Compiler source or test was changed.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [preflight glossary](../../../../glossary.md#term-preflight); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Imports and source inventories contain no downstream `RecurrentScanKind` adoption. No current
  capability provider advertises the family. A focused source scan found no callback, lambda,
  region/body value, nested graph, runtime length read, reflection/string dispatch, RNG, mask,
  dynamic Shape, storage, backend, Runtime, or Engine mechanism in the new production types.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [graph glossary](../../../../glossary.md#graph); [dispatch glossary](../../../../glossary.md#term-dispatch); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [mask glossary](../../../../glossary.md#term-boolean-mask); [Shape glossary](../../../../glossary.md#shape))</span>
- The implementation pass changes exactly 24 task paths: 6 production Java files, 16 Model test
  files, and 2 Model planning files. Global status review found the concurrent CPU/backend,
  Data-master, and global-roadmap changes preserved and outside this inventory; no source under
  Compiler, Planning, Prepare, Runtime, Engine, backend, NN, Data, architecture tests,
  conformance tests, or integration tests was changed by this context.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- New-file terminal-byte checks reported `0a` for every new task path, and `git diff --check`
  passed. The first untracked-file no-index loop used zsh's read-only `status` name and stopped
  before checking files; the corrected loop used a task-specific variable and passed all eight
  new files with the expected difference status and no whitespace diagnostics. Architecture tests
  and the repository-wide suite were not run because this task changes no dependency,
  authoritative architecture boundary, shared build configuration, or executable cross-module
  adoption; the planning guide defers those gates to the recurrent capability checkpoint or CI.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>
- Development-only [compile](../../../../glossary.md#compile)/test commands were also explicit: initial Model compile/Javadoc and
  `:modules:model:testClasses` passed. The compiler probe's first autograd extension revealed the
  existing explicit-seed precondition, and its next run exposed the exact current<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
  `unknown or unclassified operation kind` diagnostic; after aligning the probe with those
  existing contracts, its final compile/run passed. These temporary `/tmp` sources changed no
  repository path and made no implementation decision.
- Documentation context `/root/model_0025e_docs` independently read the architecture contract,
  focused architecture explanations, ADR 0012, documentation rules and applicable profiles,
  planning guide and roadmap, Model task/master/capability contracts, NN 0021A and current cells,
  all six production files, all sixteen affected tests, shared multi-output provenance and graph
  capture, current Compiler inference/autograd boundaries, and every affected documentation page.
  It found no executable, API, semantic, provenance, fail-closed, or documentation blocker and
  made no executable Java or test change.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Final `./gradlew :modules:model:javadoc` passed. Generated pages were inspected for both enums,
  both result records, and all six Tensor overloads, including ordered parameters, outputs,
  Shapes, nullability, failure documentation, and current-versus-future boundaries. No production
  Javadoc correction was required after the independent review.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Independent `javap`, reflection, and external-package compilation/runtime probes confirmed the
  exact two direction constants, three kind constants, two-/three-component records, six public
  overloads, and all biased/unbiased forms. A compiler-package probe confirmed one flat captured
  node with two canonical outputs, current inference rejection as unsupported, and current
  autograd rejection as unknown/unclassified before formula construction.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Import, inventory, capability, and forbidden-mechanism probes found no downstream recurrent
  adoption or capability advertisement and no callback, functional body, nested graph, region,
  runtime-value read, storage access, reflection, service discovery, RNG, backend, Runtime, or
  Engine mechanism in the family implementation. The final task inventory is exactly 34 paths:
  6 production files, 16 Model tests, and 12 documentation/planning files.
- A targeted Markdown validator passed all 12 affected Markdown files, resolving 351 local links
  including 46 heading anchors and checking balanced fences, final newlines, carriage returns,
  and trailing whitespace. Final status checks leave 0025E Complete while Model 0026 and Compiler
  0006A remain Draft without task specifications and no Model task is Ready.
- Architecture/ADR tests, backend conformance, integration tests, repository-wide tests, and
  other module suites were not rerun: executable Java remained frozen after the successful 153-
  test focused and 1,046-test authoritative Model runs, and the task changes no dependency,
  module boundary, build configuration, Compiler/Runtime/Engine/backend implementation, or NN,
  Training, Data, or Text behavior. ADR 0012 and `current-architecture-plan.md` remain accurate
  unchanged; global roadmap and concurrent CPU/backend/Data work were preserved outside scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Implementation notes

- Implementation context `/root/model_0025e_implementation` added the exact closed recurrent
  semantic vocabulary, two exact result records, six Tensor receiver overloads, and one
  package-private field-free construction helper. One call creates one ordinary flat canonical
  multi-output producer and never reads runtime valid-length values.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The helper validates caller-visible metadata in the specified order, preserves exact static
  Dimension and final-state Shape references where required, checks packed gate extents with
  overflow translation to <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [Dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape); [overflow glossary](../../../../glossary.md#term-overflow))</span>`IllegalArgumentException`, and delegates output identity allocation
  atomically to <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`TensorFactory.createDerivedOutputs`.
- Focused tests lock all kinds, directions, bias forms, signatures, packing-derived Shapes,
  ordered inputs and outputs, canonical shared provenance, result records, zero time and batch,
  exact types, gradient-request OR, validation order, no-ID local failures, and partial allocation
  effects at identifier exhaustion. Existing public Tensor method-count locks now expect 208.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [signatures glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [gradient glossary](../../../../glossary.md#term-gradient); [validation glossary](../../../../glossary.md#term-validation))</span>
- Compiler, Planning, Prepare, Runtime, Engine, backend, NN, Data, architecture-test,
  conformance-test, integration-test, build, dependency, ADR, current-architecture-index, and
  global-roadmap sources remain unchanged by this implementation context. Temporary clean-room
  checks confirmed one-node graph capture and the current Compiler inference/autograd fail-closed
  boundaries without adding downstream adoption.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [node glossary](../../../../glossary.md#node); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Executable Java remained frozen after the successful authoritative Model test. Documentation
  context <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>`/root/model_0025e_docs` independently finalized the ten additional authorized
  architecture/API/capability/glossary pages plus the Model task/master evidence and status. It
  confirmed that existing production Javadocs are complete and accurate without modification.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Completion summary

- Completed changes: implemented fixed RNN-tanh, reset-after GRU, and LSTM Model metadata with
  exact forward/reverse attributes, static descriptor validation, canonical result carriers, and
  the six specified Tensor receiver methods.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [tanh glossary](../../../../glossary.md#term-tanh); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [Model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: exactly 6 production Java files, 16 Model test files, and 12
  documentation/planning files; all 34 are the task-authorized paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and [validation](../../../../glossary.md#term-validation): focused task [matrix](../../../../glossary.md#term-matrix-meanings) passed 153 tests; the single authoritative
  `./gradlew :modules:model:test` passed 1,046 tests with zero failures, errors, or skips;
  final Model Javadoc, generated-page inspection, external-use compilation/runtime, <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [runtime glossary](../../../../glossary.md#term-runtime))</span>`javap`,
  reflection, compiler-boundary, import/inventory/capability/forbidden-mechanism, Markdown, exact-
  scope, status, newline, whitespace, and <span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` validation passed.
- Documentation-agent review: complete in [canonical](../../../../glossary.md#term-canonical-form) context `/root/model_0025e_docs`.
- Documentation impact: reconciled the Tensor, Compile, and Training APIs; focused architecture
  status; glossary; Model capabilities; and task/master evidence while preserving the accepted
  architecture decision and downstream ownership boundaries.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: all six affected production files were independently reviewed and generated;
  existing Javadocs remained accurate, detailed, and complete without a final source edit.
- Glossary impact: added the current fixed recurrent-scan and valid-length Tensor distinction from
  cumulative scans and static NN sequence containers.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Unresolved issues: none for task 0025E.
- Follow-up required: Compiler 0006A and later execution/BPTT/NN tasks remain separately Draft or
  intentionally unspecified as recorded above.<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>

Status: Complete
