# Task 0025E: Fixed Recurrent-Scan Semantic Family and Tensor Expressions

## Status

Complete

## Goal

Add the smallest Model-owned fixed recurrent-scan capability selected by
[ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md): one flat,
identity-distinct, multi-output operation occurrence for each of `RNN_TANH`,
`GRU_RESET_AFTER`, and `LSTM`, with one immutable `FORWARD` or `REVERSE` direction.

The public Tensor receiver is a fully static time-major sequence. Ordinary Tensor inputs carry
runtime valid lengths, initial states, weights, and the optional packed input-side bias. One exact
producer exposes a dense original-time-aligned output and explicit final states through canonical
Tensor wrappers:

```text
time-major values + runtime lengths + explicit state + explicit parameters + direction
  -> one immutable recurrent-scan producer occurrence
  -> [dense outputs, final hidden] or [dense outputs, final hidden, final cell]
```

This task constructs immutable Model metadata only. It does not read length values, execute a
loop, bind caller data, lower or prepare work, select a backend, or make the new operation
executable. Compiler task 0006A and later execution-layer work must adopt the closed family
explicitly.

## Motivation

The completed static NN sequence containers construct one Tensor expression per time step and fix
sequence length in the Java call. That surface cannot represent an ordinary runtime length vector,
keep graph size constant in `time`, or allow a backend to own a prepared bounded loop. ADR 0012
selects a fixed flat operation instead of a general region system so the Model can describe the
needed meaning without weakening existing graph, compiler, Runtime, or backend ownership.

Publishing the Model contract first is also required by the repository's closed inventories.
Compiler 0006A cannot truthfully add capture and inference for a semantic family that Model does
not yet own, while Model must not imply executability before Compiler and a concrete backend adopt
that family. This task establishes that narrow prerequisite and keeps every downstream adoption
explicit and fail-closed.

## Scope

- Add one operation-family package containing exactly:
  - `RecurrentDirection`, the two-value immutable attributes enum;
  - `RecurrentScanKind`, the three-value kind enum with closed structural signatures.
- Add exactly two shallowly immutable public result records:
  - `RecurrentScanResult(Tensor outputs, Tensor finalHidden)`;
  - `LstmRecurrentScanResult(Tensor outputs, Tensor finalHidden, Tensor finalCell)`.
- Add one package-private, field-free `TensorRecurrentScanExpressions` construction helper.
- Add exactly six public receiver methods to `Tensor`: bias-free and biased overloads of
  `rnnScan`, `gruScan`, and `lstmScan`.
- Fix the exact operation input and output order, gate packing and transition equations, static
  Shape rules, exact data-type rules, gradient-request metadata, validation order, failure
  effects, identity, provenance, and zero-length semantics below.
- Construct all outputs atomically through the existing canonical multi-output Tensor factory
  path.
- Extend production-signature coverage and every live public Tensor method-count lock from 202
  to 208.
- Add focused semantic and Tensor-construction tests for every family, direction, bias variant,
  descriptor rule, validation branch, effect boundary, and provenance invariant in this task.
- Finalize affected Javadocs, architecture-status explanations, Tensor/Compile/Training API
  explanations, glossary terminology, Model capabilities, and planning records through the
  mandatory independent documentation-focused pass.

## Out of scope

- arbitrary cells, callbacks, bodies, graph regions, nested graphs, free-variable capture,
  subgraphs, loop-carried graph values, general control flow, or another graph identity domain
- another recurrent kind, direction, activation, gate order, bias convention, peephole,
  projection, bidirectional occurrence, stacking, residual, dropout, sparse, quantized, stateful,
  or configurable variant
- dynamic or binding-dependent Shapes, arbitrary masks, inferred lengths, padding sentinels,
  active-row compaction, sorting, packed-sequence buffers, or skipped-work claims
- reading, inspecting, validating, copying, or retaining runtime valid-length values during Model
  construction
- storage, caller binding, graph compilation, inference adoption, graph finalization, backward
  graph construction, backpropagation through time (BPTT), canonicalization, lowering, planning,
  preparation, execution, backend capability, runtime, engine, or training behavior
- compiler, planning, prepare, runtime, engine, backend, NN, Data, architecture-test,
  backend-conformance, integration-test, Gradle, dependency, module-descriptor, or serialization
  source changes
- migrating or replacing the current static NN cell and sequence APIs
- hidden recurrent state, hidden outputs, parameters, buffers, random-number-generator state,
  mode, counters, mutation, I/O, or external resources
- changing current scalar, elementwise, MATMUL, shape, promotion, broadcasting, inference,
  autograd, capability-query, Tensor factory, Tensor producer, or graph-capture contracts
- creating or promoting Compiler 0006A, backend, Engine, NN 0021B, NN 0022, or Data task
  specifications
- editing the global roadmap, completed task specifications, ADR 0012, or unrelated source and
  documentation

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially fixed recurrent scan,
  Model ownership, immutable operation/provenance, static Shape, compile, Runtime, Engine, and
  training boundaries
- [ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
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
- [NN task 0021A](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)
- [NN master plan](../../../extensions/nn/master-plan.md), including the completed recurrent-cell
  and sequence tasks 0012 through 0020C
- [Compiler master plan](../../compiler/master-plan.md), especially future task 0006A
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work remains inside `modules/model` plus directly affected architecture-status,
  explanatory/API, glossary, capability, and Model planning documentation.
- One recurrent scan is one ordinary flat `Operation` and one identity-distinct
  `TensorProducer`, independent of `time`. It has no body, region, callback, nested graph, or
  auxiliary per-time Tensor producer.
- `RecurrentDirection` is the complete immutable attributes value. It contains only `FORWARD` and
  `REVERSE` and implements `OperationAttrs`; no second attributes or descriptor type is added.
- `RecurrentScanKind` is the complete semantic family. Its constants contain only `RNN_TANH`,
  `GRU_RESET_AFTER`, and `LSTM` and implement `OperationKind`.
- The family owns logical meaning and descriptor-visible construction. It owns no execution
  algorithm, backend route, capability claim, prepared state, storage, caller value, gradient
  rule, or lifecycle state.
- Shapes are fully static at construction. The valid-length Tensor is runtime data, not a hidden
  Shape-binding mechanism.
- Every floating value/state/parameter input has the receiver's exact floating data type. The
  helper must not invoke promotion or accept an otherwise promotable mixed-type combination.
- Valid lengths have exact type `INT64`, rank one, Shape `[batch]`, and
  `requiresGrad == false`. Model does not read or infer their values.
- Every output uses unresolved layout and the same exact floating data type. Gradient eligibility
  is the OR of all floating input, state, weight, and supplied bias roles; valid lengths never
  contribute.
- Every output returned through a result record is the canonical exact wrapper for its indexed
  producer slot. A successful call exposes every producer output; none is hidden.
- Current graph capture may observe the producer as one ordinary flat multi-output occurrence.
  Compiler adoption remains closed: forward compilation fails on the unsupported kind until
  Compiler 0006A, and every backward-capable request fails closed before derivative Tensor
  construction. Unrelated kinds retain their current behavior.
- No dependency direction, module ownership, graph identity, public exception layer, or
  architecture decision changes.
- Stop if implementation requires a different public signature, another type, a compiler/runtime/
  backend/NN edit, dynamic Shapes, region semantics, or any architecture change.

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
  direction parameter and is itself the immutable `OperationAttrs` value.
- `io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind` — owns the three fixed
  semantic identities and their exact structural signatures.
- `io.github.pho001.synaptik.model.tensor.RecurrentScanResult` — exposes the two canonical RNN/GRU
  outputs in producer-slot order.
- `io.github.pho001.synaptik.model.tensor.LstmRecurrentScanResult` — exposes the three canonical
  LSTM outputs in producer-slot order.
- `io.github.pho001.synaptik.model.tensor.TensorRecurrentScanExpressions` — owns family-local
  validation, descriptor inference, operation construction, and result-carrier assembly.
- `io.github.pho001.synaptik.model.tensor.Tensor` — remains the public receiver facade.

No generic scan utility, recurrent descriptor, recurrent attributes record, builder, facade,
manager, compiler adapter, or NN type is added.

## Exact public API and occurrence signatures

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

`RecurrentScanKind.signatures()` returns stable immutable lists with exactly these structural
contracts:

```text
RNN_TANH:       RecurrentDirection, 5..6 inputs, exactly 2 outputs
GRU_RESET_AFTER: RecurrentDirection, 5..6 inputs, exactly 2 outputs
LSTM:           RecurrentDirection, 6..7 inputs, exactly 3 outputs
```

Use `OperationSignature.inputRange(...)`. Each kind accepts the exact
`RecurrentDirection.class` only. Do not add a permissive signature, direction-specific kind,
no-attributes form, or second attributes implementation.

### Result records

Add exactly:

```java
public record RecurrentScanResult(Tensor outputs, Tensor finalHidden)

public record LstmRecurrentScanResult(
        Tensor outputs, Tensor finalHidden, Tensor finalCell)
```

Each compact constructor checks component nullability in declaration order, with the component
name as the null message, and retains each exact reference. The records perform no descriptor,
producer, operation, slot, storage, compiler, or execution validation for independently supplied
Tensors. They add no alternate constructor, factory, accessor alias, method, interface, or state.

### Tensor receiver methods

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

The receiver is `input`. Each facade method contains one return statement delegating directly to
the matching helper entry with the exact argument order. It does not read fields, validate,
construct descriptors or operations, or delegate through another public Tensor method.

There are exactly two public overloads for each of `rnnScan`, `gruScan`, and `lstmScan`; no alias,
static form, optional/null bias, default direction, generic recurrent method, descriptor argument,
or convenience method is added. The public Tensor declared-method inventory changes exactly from
202 to 208.

### Ordered inputs and outputs

The helper constructs exact immutable input snapshots in this order:

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

Output descriptor and canonical wrapper order is exactly:

```text
RNN/GRU: [outputs, finalHidden]
LSTM:    [outputs, finalHidden, finalCell]
```

## Fixed mathematical semantics

For one valid original-time coordinate, let row vector `x` be the input, `h` the previous hidden
state, and, for LSTM, `c` the previous cell state. Let `W` be `inputWeight`, `U` be
`hiddenWeight`, and let optional `b` be added only to the input projection. Matrix products,
elementwise arithmetic, sigmoid, and tanh have the represented-value meanings of the current
Model operations. These equations fix observable semantic association and gate packing; they do
not prescribe a decomposition, numerical kernel, accumulator widening, fusion, or backend
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
bias.

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

There is no recurrent-side bias, peephole, projection, configurable activation, or hidden
parameter convention. The scan is semantically equivalent to repeated application of these fixed
transitions but is not represented as repeated Tensor expressions.

## Static descriptors, valid lengths, and traversal

For receiver Shape `[time, batch, inputSize]` and hidden size `hiddenSize`:

| Role | Exact requirement |
|---|---|
| `input` | fully static rank 3 `[time, batch, inputSize]`; floating |
| `validLengths` | fully static rank 1 `INT64[batch]`; `requiresGrad == false` |
| `initialHidden` | fully static `[batch, hiddenSize]` |
| `initialCell` | LSTM only; fully static `[batch, hiddenSize]` |
| `inputWeight` | fully static `[gateCount * hiddenSize, inputSize]` |
| `hiddenWeight` | fully static `[gateCount * hiddenSize, hiddenSize]` |
| `bias` | optional; fully static `[gateCount * hiddenSize]` |
| `outputs` | `[time, batch, hiddenSize]` |
| `finalHidden` | `[batch, hiddenSize]` |
| `finalCell` | LSTM only; `[batch, hiddenSize]` |

`inputSize` and `hiddenSize` are positive. `time` and `batch` may be zero. Gate count is 1 for
RNN, 3 for GRU, and 4 for LSTM. The checked product `gateCount * hiddenSize` must be representable
as a Shape extent.

Input, initial states, weights, and optional bias have one exact common floating data type. No
promotion or conversion is implied. Outputs retain that exact type. Every output layout is
unresolved.

Every output has one common `requiresGrad` value equal to the logical OR of the request flags on
`input`, `initialHidden`, LSTM `initialCell`, `inputWeight`, `hiddenWeight`, and supplied `bias`.
`validLengths` is excluded. The flag records only gradient eligibility/request metadata; this task
does not make a backward-capable compile succeed.

At future execution, for each original batch row `b`, the complete runtime length vector is
validated with `0 <= L[b] <= time` before output representation mutation. Model neither reads nor
stores those scalar values beyond the ordinary `validLengths` Tensor input.

- `FORWARD` consumes original coordinates `0` through `L[b] - 1`.
- `REVERSE` consumes original coordinates `L[b] - 1` through `0`; it reverses only the valid
  prefix and never traverses the padded suffix.
- Each valid original coordinate stores the next hidden state after consuming that coordinate.
- Each padded original coordinate `t >= L[b]` stores the exact positive zero of the common data
  type.
- Final hidden, and LSTM final cell, is the state after the last consumed coordinate.
- A zero-length row has positive-zero output at every coordinate and returns the exact initial
  state values semantically.
- If `time == 0`, every future runtime length must be zero, the dense output is empty, and the
  initial states are preserved semantically.

These are semantic obligations for later executable adoption, not Model-time value checks or a
claim that a backend currently skips work.

## Construction validation and effects

The package-private helper performs validation in this observable order. Each step completes
before the next begins:

1. Null-check supplied parameters in public declaration order. For biased overloads, check
   `bias` before `direction`; for LSTM, check `initialCell` before weights. The receiver cannot be
   null during an instance call.
2. Validate `input`: exact floating data type, rank 3, fully static Shape; extract `time`, `batch`,
   and `inputSize`; allow zero `time` and `batch` and require positive `inputSize`.
3. Validate `validLengths`: exact `INT64`, `requiresGrad == false`, rank 1, fully static Shape,
   and exact extent `batch`.
4. Validate `initialHidden`: exact input data type, rank 2, fully static Shape, exact batch extent,
   and positive `hiddenSize`.
5. For LSTM, validate `initialCell`: exact input data type, rank 2, fully static Shape, and exact
   `[batch, hiddenSize]`.
6. Compute `gateCount * hiddenSize` with checked arithmetic. Overflow fails here.
7. Validate `inputWeight`: exact input data type, rank 2, fully static Shape, and exact
   `[gateCount * hiddenSize, inputSize]`.
8. Validate `hiddenWeight`: exact input data type, rank 2, fully static Shape, and exact
   `[gateCount * hiddenSize, hiddenSize]`.
9. When supplied, validate `bias`: exact input data type, rank 1, fully static Shape, and exact
   `[gateCount * hiddenSize]`.
10. Derive ordered output descriptors, create `Operation(kind, direction)`, form the exact ordered
    input snapshot, and invoke the existing factory-atomic canonical multi-output path.
11. Assemble the appropriate result record from the returned canonical wrappers in slot order.

Use stable, role-specific exception messages consistent with current Tensor expression helpers.
Null failures use `NullPointerException`; invalid type, gradient flag, rank, static extent,
dimension relation, or checked-size condition uses `IllegalArgumentException`. Do not catch,
translate, or suppress Tensor identifier exhaustion.

Every null/type/rank/static-Shape/size/overflow failure occurs before a Tensor identifier is
allocated and performs no mutation. Runtime length bounds cannot fail here because no value is
read. During factory output creation, output identifiers are allocated in slot order; if later
identifier allocation fails, earlier identifiers may remain consumed, no producer or result
carrier is published, and no partial result is returned. This preserves the existing
`TensorFactory.createDerivedOutputs` failure contract.

Descriptor construction should preserve exact immutable Shape references when their meaning is
already exact: final hidden may reuse `initialHidden.shape()`, final cell may reuse
`initialCell.shape()`, and the dense output should use the exact static `time`, `batch`, and
`hiddenSize` Dimension values obtained from the validated inputs. Do not mutate or normalize any
input Tensor, Shape, descriptor, producer, or caller collection.

## Identity, provenance, graph capture, and fail-closed adoption

Every successful call creates one fresh `Operation`, one fresh identity-distinct producer, one
fresh result carrier, and exactly two or three fresh output Tensors and Tensor identifiers in
output order. Two calls with equal operands and direction remain distinct occurrences.

All result components from one call:

- retain the same exact producer;
- expose the same exact operation, kind, direction object, immutable ordered input snapshot, and
  ordered output-descriptor snapshot through that producer;
- are the producer's canonical exact wrappers at indices 0, 1, and, for LSTM, 2;
- have distinct Tensor identifiers and the correct `producerOutputIndex`; and
- create no gate, projection, state-step, mask, zero-fill, or per-time intermediate Tensor.

The existing generic graph-capture path may therefore capture one ordinary multi-output node and
its ordinary input edges. This task does not change graph capture or promise a successful complete
compile.

Until Compiler 0006A explicitly adopts the family:

- a forward-only complete compile that reaches current inference dispatch rejects the unknown
  `RecurrentScanKind` as unsupported before planning or backend capability selection;
- a backward-capable request fails closed through the current closed autograd/signature coverage
  boundary before constructing derivative Tensors; and
- existing compiler branches and capability inventories remain unchanged, so unrelated operation
  families compile exactly as before.

No current backend advertises this family, and no execution, runtime validation, skipped-work,
lowering, or backend-support claim is made.

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

Expected documentation and planning:

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

## Maximum scope

This task may create or modify at most:

- 6 production Java files;
- 16 Model test files;
- 12 architecture-status, explanatory/API, glossary, capability, and Model planning files;
- 34 files total.

Documentation changes may update current-versus-planned status and explain the exact public
metadata contract. They must not change the architecture decision. `current-architecture-plan.md`
and ADR 0012 remain accurate navigation/history and are review-only.

If another file, type, public member, module, dependency, architecture test, conformance test,
integration test, build change, or broader documentation change is needed, stop and propose a
follow-up task.

## Acceptance criteria

- `RecurrentDirection` contains exactly `FORWARD` and `REVERSE`, implements `OperationAttrs`, and
  is the exact attributes type for every recurrent kind.
- `RecurrentScanKind` contains exactly `RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM` and exposes only
  the exact 5..6/2, 5..6/2, and 6..7/3 signatures defined above.
- The two records have exactly the specified components, check nulls in component order, and
  retain exact references without extra validation or methods.
- `Tensor` adds exactly the six specified receiver methods and no other public declaration; its
  declared public method count is exactly 208.
- Every kind, bias variant, ordered input snapshot, output order, direction, gate packing,
  equation, static Shape, exact data type, unresolved layout, and gradient-request rule matches
  this specification.
- RNN/GRU and LSTM results expose exactly two and three canonical producer wrappers respectively;
  all identity, shared-provenance, indexed-slot, freshness, and no-intermediate invariants pass.
- `time == 0` and `batch == 0` construct valid metadata when all other static requirements hold;
  positive `inputSize` and `hiddenSize` remain mandatory.
- Every construction-time failure class, validation order, no-identifier-allocation boundary, and
  factory identifier-exhaustion effect matches this specification.
- Tests prove that Model construction never reads length storage or values and accepts ordinary
  unbound runtime-input metadata satisfying the descriptor contract.
- Tests and documentation state zero-length, reverse-prefix, positive-zero padding, and final-state
  semantics without claiming current execution or skipped work.
- Existing Compiler, autograd, graph-capture, capability, Runtime, Engine, backend, and unrelated
  Model code remain byte-for-byte unchanged except for the explicitly listed Model files.
- Documentation clearly distinguishes the now-current Model construction surface from still-
  planned Compiler, execution, NN coordination, and BPTT support.
- Model capabilities list the family as constructible metadata and explicitly non-executable
  until downstream closed inventories adopt it.
- No compiler/backend/NN/Data/global-roadmap task is promoted or specified by this change; Model
  0026 remains Draft without a task specification.
- Every affected public or package-private type and method has detailed Javadoc covering inputs,
  constraints, nullability, outputs, failure conditions, provenance, and non-execution boundaries.
- A separate documentation-focused agent pass has finalized affected explanatory documentation,
  Javadoc, glossary impact, and planning status in this same overall change.

## Tests / validation

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

Then run the Model task tier once:

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
  count 208 are locked by Model tests rather than ad hoc reflection or bytecode commands;
- package/import compilation and Javadoc prove the public API shape;
- documentation links, heading anchors, code fences, final newlines, and trailing whitespace are
  checked against the final changed Markdown files and recorded with the checker or manual method
  used;
- the final changed-path inventory is exactly within the 34 allowed paths, with no Compiler,
  Runtime, Engine, backend, NN, Data, architecture-test, integration-test, conformance-test,
  Gradle, dependency, ADR, current-architecture-index, or global-roadmap change; and
- task/master status is synchronized only after all implementation, documentation, and validation
  evidence passes.

Repository-wide validation is deferred to CI or the next recorded cross-module recurrent-scan
capability checkpoint. This is one Model-module task; it changes no dependency, architecture
boundary, shared build configuration, or executable cross-module contract. Compiler 0006A owns
its focused capture/inference/preflight tests and later end-to-end execution adoption owns
integration and backend-conformance validation.

## Dependencies

- Accepted [ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)
  and completed [NN task 0021A](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)
  fix the architecture, surface, ordering, static descriptors, and layer ownership.
- Completed Model 0018K supplies exact family-owned `OperationSignature` validation.
- Completed Model 0018L and 0025 supply shared multi-output provenance and canonical output
  wrappers.
- Completed Model 0018N and existing Shape/data-type contracts supply exact floating type and
  static descriptor vocabulary.
- Completed Model 0019 and current NN recurrent-cell tasks supply the selected MATMUL,
  sigmoid/tanh, gate-packing, bias, and transition meanings.
- Model 0025D is the latest completed detailed Model task. Therefore 0025E is the first unfinished
  task in Model master-plan order and is the next executable Model frontier.

## Follow-up tasks

- Compiler 0006A must explicitly adopt one-node forward capture, descriptor inference, final
  validation, closed inventory coverage, and fail-closed backward handling for this family. It
  must not be specified or promoted in this task.
- Later concrete-backend and execution-layer tasks must choose exact capability truth, lowering,
  prepared-loop behavior, complete runtime length validation before output mutation, Engine
  binding/publication, and evidence for any skipped-work claim.
- Later NN 0021B/0022 and Data coordination may expose sequence/module conveniences while
  preserving existing static APIs and passing lengths as an ordinary Tensor.
- BPTT requires a separate compiler architecture decision covering saved state versus
  recomputation and exact derivative semantics.
- Dynamic Shapes, bidirectional composition, packed/compacted execution, broader recurrent
  variants, and serialization changes remain separate future decisions.
- Model 0026 remains the next Draft Model task after 0025E and receives no detailed specification
  here.

## Architecture impact

Expected impact: None.

This task implements the exact Model surface and ownership already authorized by
`ARCHITECTURE.md` and ADR 0012. The documentation pass changes planned/current status wording but
does not change an architecture rule. If implementation requires another signature, operation
shape, ownership rule, execution promise, or architecture edit beyond status reconciliation, stop
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
  of `OperationAttrs`, keeps operation-to-tensor dependency direction intact, and avoids a
  redundant one-field `RecurrentScanAttrs` record.
- One `RecurrentScanKind` owns all three fixed meanings because they share direction vocabulary,
  time-major/static descriptor rules, and one construction helper while retaining distinct enum
  identity and kind-specific signatures.
- Bias optionality is represented only by exact input-count ranges and two explicit overloads.
  `null`, an option wrapper, a boolean, or a second attributes type cannot create a hidden variant.
- The task extends current canonical multi-output factory behavior instead of adding a recurrent
  producer/result mechanism.
- No unresolved architecture or public-API decision remains for implementation.

## Known limitations

- The new Model metadata is intentionally non-executable until Compiler 0006A and downstream
  closed inventories adopt it.
- Every backward-capable compile remains unsupported; BPTT is not designed by this task.
- Only fully static Shapes, runtime `INT64[batch]` prefix lengths, one direction per occurrence,
  dense original-time-aligned output, and the three fixed cell equations are represented.
- Model construction cannot validate runtime length scalar bounds and does not promise physical
  skipped work.
- Current NN APIs are not migrated and no public runtime input-binding convenience is added.

## Validation evidence

Planning context `/root/model_0025e_planning` read the required repository instructions,
architecture contract and focused lifecycle/module/runtime/training explanations, ADR 0012,
planning guide and roadmap, Model/NN/Compiler master plans, completed prerequisite Model and NN
tasks, current Model source/tests/Javadocs/capabilities, and the current closed Compiler capture,
inference, autograd, and capability boundaries. It found no architecture or ordering conflict.

Planning-stage validation:

- A read-only Markdown checker validated this task and the Model master plan: two files and 209
  repository-local links, including heading anchors, with balanced fences, final newlines, and no
  trailing whitespace.
- `git diff --check` passed for tracked worktree changes, and
  `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md`
  returned the expected difference status 1 with no whitespace diagnostic for the new untracked
  specification.
- Exact-status checks found one `Ready` row in the Model master plan and this task as the only
  `Ready` detailed Model task. No stale 0025E Draft wording remains in the Model master plan.
- Model 0025D remains Complete; 0025E is the first unfinished Model task; Model 0026 and Compiler
  0006A remain Draft without task specifications.
- The planning context created or changed exactly the Model master plan and this new task. It did
  not touch the pre-existing concurrent CPU/backend, Data-master, or global-roadmap worktree
  changes.
- The future implementation maximum contains exactly 34 paths: 6 production Java files, 16 Model
  test files, and 12 directly affected documentation/planning files. It contains no Compiler,
  Runtime, Engine, backend, NN, Data, architecture-test, conformance-test, integration-test,
  Gradle, dependency, ADR, current-architecture-index, or global-roadmap change.

No Java test, Javadoc generation, or build command ran in the planning context because that pass
created planning only and did not implement task 0025E.

Implementation context `/root/model_0025e_implementation` validation after executable freeze:

- `./gradlew :modules:model:test` with the sixteen task-specified focused test filters passed 153
  tests with zero failures, errors, or skips. An earlier focused development run exposed two test-
  fixture ID-accounting mistakes; the fixtures were corrected before this final focused run.
- The one authoritative unfiltered `./gradlew :modules:model:test` passed 1,046 tests with zero
  failures, errors, or skips. Executable Java did not change afterward.
- Preliminary `./gradlew :modules:model:javadoc` passed after the production surface was complete.
  The documentation-focused context must run the final Javadoc gate after its independent review.
- External package-use compilation against `modules/model/build/classes/java/main` passed for all
  three result forms, both directions, and biased/unbiased public signatures. `javap` showed
  exactly two direction constants, three kind constants, the two- and three-component records,
  and exactly six Tensor scan overloads.
- A temporary compiler-package boundary probe compiled and ran after
  `./gradlew :modules:compiler:classes`: generic graph capture produced exactly one flat node with
  two ordered outputs; current captured inference rejected `RecurrentScanKind` as an unsupported
  operation kind; current autograd preflight rejected it as an unknown/unclassified kind before a
  derivative Tensor identifier was allocated. No Compiler source or test was changed.
- Imports and source inventories contain no downstream `RecurrentScanKind` adoption. No current
  capability provider advertises the family. A focused source scan found no callback, lambda,
  region/body value, nested graph, runtime length read, reflection/string dispatch, RNG, mask,
  dynamic Shape, storage, backend, Runtime, or Engine mechanism in the new production types.
- The implementation pass changes exactly 24 task paths: 6 production Java files, 16 Model test
  files, and 2 Model planning files. Global status review found the concurrent CPU/backend,
  Data-master, and global-roadmap changes preserved and outside this inventory; no source under
  Compiler, Planning, Prepare, Runtime, Engine, backend, NN, Data, architecture tests,
  conformance tests, or integration tests was changed by this context.
- New-file terminal-byte checks reported `0a` for every new task path, and `git diff --check`
  passed. The first untracked-file no-index loop used zsh's read-only `status` name and stopped
  before checking files; the corrected loop used a task-specific variable and passed all eight
  new files with the expected difference status and no whitespace diagnostics. Architecture tests
  and the repository-wide suite were not run because this task changes no dependency,
  authoritative architecture boundary, shared build configuration, or executable cross-module
  adoption; the planning guide defers those gates to the recurrent capability checkpoint or CI.
- Development-only compile/test commands were also explicit: initial Model compile/Javadoc and
  `:modules:model:testClasses` passed. The compiler probe's first autograd extension revealed the
  existing explicit-seed precondition, and its next run exposed the exact current
  `unknown or unclassified operation kind` diagnostic; after aligning the probe with those
  existing contracts, its final compile/run passed. These temporary `/tmp` sources changed no
  repository path and made no implementation decision.
- Documentation context `/root/model_0025e_docs` independently read the architecture contract,
  focused architecture explanations, ADR 0012, documentation rules and applicable profiles,
  planning guide and roadmap, Model task/master/capability contracts, NN 0021A and current cells,
  all six production files, all sixteen affected tests, shared multi-output provenance and graph
  capture, current Compiler inference/autograd boundaries, and every affected documentation page.
  It found no executable, API, semantic, provenance, fail-closed, or documentation blocker and
  made no executable Java or test change.
- Final `./gradlew :modules:model:javadoc` passed. Generated pages were inspected for both enums,
  both result records, and all six Tensor overloads, including ordered parameters, outputs,
  Shapes, nullability, failure documentation, and current-versus-future boundaries. No production
  Javadoc correction was required after the independent review.
- Independent `javap`, reflection, and external-package compilation/runtime probes confirmed the
  exact two direction constants, three kind constants, two-/three-component records, six public
  overloads, and all biased/unbiased forms. A compiler-package probe confirmed one flat captured
  node with two canonical outputs, current inference rejection as unsupported, and current
  autograd rejection as unknown/unclassified before formula construction.
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
  unchanged; global roadmap and concurrent CPU/backend/Data work were preserved outside scope.

## Implementation notes

- Implementation context `/root/model_0025e_implementation` added the exact closed recurrent
  semantic vocabulary, two exact result records, six Tensor receiver overloads, and one
  package-private field-free construction helper. One call creates one ordinary flat canonical
  multi-output producer and never reads runtime valid-length values.
- The helper validates caller-visible metadata in the specified order, preserves exact static
  Dimension and final-state Shape references where required, checks packed gate extents with
  overflow translation to `IllegalArgumentException`, and delegates output identity allocation
  atomically to `TensorFactory.createDerivedOutputs`.
- Focused tests lock all kinds, directions, bias forms, signatures, packing-derived Shapes,
  ordered inputs and outputs, canonical shared provenance, result records, zero time and batch,
  exact types, gradient-request OR, validation order, no-ID local failures, and partial allocation
  effects at identifier exhaustion. Existing public Tensor method-count locks now expect 208.
- Compiler, Planning, Prepare, Runtime, Engine, backend, NN, Data, architecture-test,
  conformance-test, integration-test, build, dependency, ADR, current-architecture-index, and
  global-roadmap sources remain unchanged by this implementation context. Temporary clean-room
  checks confirmed one-node graph capture and the current Compiler inference/autograd fail-closed
  boundaries without adding downstream adoption.
- Executable Java remained frozen after the successful authoritative Model test. Documentation
  context `/root/model_0025e_docs` independently finalized the ten additional authorized
  architecture/API/capability/glossary pages plus the Model task/master evidence and status. It
  confirmed that existing production Javadocs are complete and accurate without modification.

## Completion summary

- Completed changes: implemented fixed RNN-tanh, reset-after GRU, and LSTM Model metadata with
  exact forward/reverse attributes, static descriptor validation, canonical result carriers, and
  the six specified Tensor receiver methods.
- Files changed or created: exactly 6 production Java files, 16 Model test files, and 12
  documentation/planning files; all 34 are the task-authorized paths.
- Tests and validation: focused task matrix passed 153 tests; the single authoritative
  `./gradlew :modules:model:test` passed 1,046 tests with zero failures, errors, or skips;
  final Model Javadoc, generated-page inspection, external-use compilation/runtime, `javap`,
  reflection, compiler-boundary, import/inventory/capability/forbidden-mechanism, Markdown, exact-
  scope, status, newline, whitespace, and `git diff --check` validation passed.
- Documentation-agent review: complete in canonical context `/root/model_0025e_docs`.
- Documentation impact: reconciled the Tensor, Compile, and Training APIs; focused architecture
  status; glossary; Model capabilities; and task/master evidence while preserving the accepted
  architecture decision and downstream ownership boundaries.
- Javadoc review: all six affected production files were independently reviewed and generated;
  existing Javadocs remained accurate, detailed, and complete without a final source edit.
- Glossary impact: added the current fixed recurrent-scan and valid-length Tensor distinction from
  cumulative scans and static NN sequence containers.
- Unresolved issues: none for task 0025E.
- Follow-up required: Compiler 0006A and later execution/BPTT/NN tasks remain separately Draft or
  intentionally unspecified as recorded above.

Status: Complete
