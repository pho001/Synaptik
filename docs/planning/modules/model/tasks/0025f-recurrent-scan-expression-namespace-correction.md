# Task 0025F: Recurrent-Scan Expression Namespace Correction

## Status

Complete

## Goal

Correct the newly introduced fixed recurrent-scan construction surface before Compiler 0006A
adopts it. Ordinary neural-network callers should continue to use `RnnSequence`, `GruSequence`,
and `LstmSequence`; the fixed recurrent scan remains a low-level Model expression seam for later
NN and Compiler integration, but its six domain-heavy multi-input/multi-output entries must not
remain receiver methods on every `Tensor`.

Replace the package-private `TensorRecurrentScanExpressions` helper and six public Tensor receiver
overloads with one intentionally named public static namespace:

```text
RecurrentScan.rnn/gru/lstm(input, runtime lengths, explicit states, explicit parameters, direction)
  -> one immutable recurrent-scan producer occurrence
  -> existing typed canonical output carriers
```

The correction changes only API placement and names. It preserves the exact fixed operation
family, argument and producer-input order, validation and failure effects, descriptors, Tensor-ID
allocation, canonical output wrappers, one-producer provenance, mathematical meaning, and current
Compiler fail-closed behavior implemented by completed task 0025E.

## Motivation

The six 0025E receiver overloads place a stateful recurrent domain protocol on the general
`Tensor` instance surface. Each call needs a time-major input, runtime lengths, explicit carried
state, several packed parameter Tensors, optional bias, a direction attribute, and two or three
outputs. That API shape is unlike a natural single-output algebraic expression such as
`Tensor.linear(...)`, and ordinary users should not need to discover it while working with a
general Tensor.

A focused static namespace communicates that this is a low-level fixed operation constructor
without adding a generic scan body, widening the Tensor factory, or moving Tensor-aware
construction into the operation-semantic package. Keeping the namespace in `model.tensor` lets it
reuse the existing package-private canonical derived-output seam directly. The correction is
deliberately immediate and breaking: the provisional receiver methods were introduced only by
the preceding commit and have no repository consumer or stabilized released compatibility
contract.

## Scope

- Remove exactly the six public `rnnScan`, `gruScan`, and `lstmScan` receiver overloads, their
  Javadocs, and the now-unused recurrent-direction import from `Tensor`.
- Delete package-private `TensorRecurrentScanExpressions`.
- Add public final `io.github.pho001.synaptik.model.tensor.RecurrentScan` as an
  instance-field-free, stateless static expression namespace with a private no-argument
  constructor.
- Add exactly six public static methods to that namespace: biased and unbiased overloads of
  `rnn`, `gru`, and `lstm`, with `input` as the explicit first parameter and every remaining
  argument in the exact current order.
- Preserve the complete 0025E validation, descriptor, effect, identity, provenance, and semantic
  implementation while changing only the owning public type and entry names.
- Keep `RecurrentDirection` and `RecurrentScanKind` unchanged in
  `io.github.pho001.synaptik.model.operation.recurrent`.
- Keep `RecurrentScanResult` and `LstmRecurrentScanResult` unchanged in
  `io.github.pho001.synaptik.model.tensor` as typed carriers of canonical Tensor outputs.
- Restore the public Tensor declared-method inventory from 208 to 202 and remove only the
  recurrent names and recurrent-specific reflection allowances from the existing surface locks.
- Rename and refocus the recurrent construction test around `RecurrentScan`; preserve its full
  biased/unbiased, direction, descriptor, validation-order, Tensor-ID, and provenance coverage.
- Correct the directly affected Compiler inventory boundary test so it distinguishes the exact
  Compiler-supported closed production inventory from the three recurrent signatures deferred
  until Compiler 0006A, proves their exact union equals the complete Model inventory, and locks
  fail-closed rejection without Tensor-ID allocation.
- Amend the authoritative and focused architecture wording to select the low-level static
  namespace instead of Tensor receiver methods, without changing fixed-scan semantics, lifecycle
  ownership, or dependency direction.
- Finalize affected Javadocs, Tensor/Compile API explanations, glossary terminology, Model
  capabilities, and Model planning records through the required independent documentation pass.

## Out of scope

- changing `RecurrentDirection`, `RecurrentScanKind`, their constants, attributes/signatures,
  equations, gate packing, traversal, padding, static Shapes, data types, or output roles
- moving, renaming, extending, or changing the validation behavior of `RecurrentScanResult` or
  `LstmRecurrentScanResult`
- widening `TensorFactory.createDerivedOutputs`, moving it to another package, or adding a bridge,
  service, registry, provider, manager, builder, factory object, descriptor facade, or general
  derived-output API
- adding `scan(body)`, a callback, lambda, user-defined cell, Tensor body, callable block, nested
  graph, captured subgraph, loop intermediate representation, region, general control flow, or
  another identity domain
- retaining `rnnScan`, `gruScan`, or `lstmScan` as aliases, deprecated bridges, forwarding
  methods, package-private duplicates, or alternate public spellings
- adding default direction, nullable or optional bias, varargs, options, builder, instance method,
  or generic recurrent entry
- changing `Tensor.linear(...)`, `TensorLinearExpressions`, the NN `Linear` module, or any other
  Tensor operation; Linear remains a natural single-output algebraic Tensor expression
- changing the existing `RnnCell`, `GruCell`, `LstmCell`, `RnnSequence`, `GruSequence`,
  `LstmSequence`, bidirectional containers, `Module`, `ModuleFactory`, parameter/state ownership,
  or NN documentation/planning history
- redirecting current NN sequence containers to the fixed scan; their compact `long[]` static
  APIs and explicit static unrolling remain the general and reference path
- compiler production inference, graph finalization, autograd inventory, backpropagation through
  time (BPTT), Planning capability, Prepare, Runtime, Engine, backend, execution, binding, Data,
  Training, or serialization behavior
- editing Compiler 0006A, the Compiler/NN/Data/backend master plans, the global roadmap, completed
  task 0025E, completed NN 0021A, architecture-test source, dependency declarations, Gradle files,
  or unrelated documentation
- Model task 0026 or IEEE FLOAT16/mixed-precision work

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the fixed recurrent-scan,
  flat Tensor-producer, Model ownership, compiler, runtime, and training boundaries
- [ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)
- [Model capabilities](../capabilities.md)
- [Completed task 0025E](0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
- [Completed task 0025](0025-canonical-tensor-producer-outputs.md)
- [Completed task 0019D](0019d-linear-convenience.md)
- [Completed NN task 0021A](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)
- [Compiler master plan](../../compiler/master-plan.md), including Draft 0006A
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- The accepted fixed recurrent operation remains one ordinary flat, identity-distinct,
  multi-output `TensorProducer` occurrence and later one ordinary flat `CompiledNode`. API
  relocation must not add a body, region, nested graph, callback, or per-time producer.
- Model continues to own the fixed semantics and descriptor-visible construction only. It gains no
  compiler, runtime, backend, module, parameter, execution, or lifecycle state.
- `RecurrentScan` belongs in `model.tensor` because it constructs public Tensor expressions and
  must use package-private `TensorFactory.createDerivedOutputs`. The task must not widen that seam
  or create a cross-package bridge merely to place the namespace elsewhere.
- The `model.operation` packages remain independent of public Tensor construction. The semantic
  enums therefore stay in `model.operation.recurrent`, while the Tensor-aware namespace and
  typed Tensor-output carriers stay in `model.tensor`.
- `RecurrentScan` is a public final namespace, not an instantiable service. It has one private
  no-argument constructor, no declared fields or nested types, and only static behavior. Its six
  public entries are non-synchronized and non-varargs; invocation-local metadata makes concurrent
  calls independent except for the existing thread-safe JVM-wide Tensor-ID allocation effects.
- Every successful entry creates the same fresh operation, producer, result carrier, canonical
  wrappers, descriptors, and IDs in the same order as its 0025E receiver predecessor.
- Every failure occurs at the same validation stage with the same exception class, message, and
  Tensor-ID effect as 0025E. Making `input` explicit adds no new order: it is already the helper's
  first null check and remains first.
- No overload delegates through another public method. Each public entry null-checks in exact
  declaration order and reaches the one shared private recurrent construction path exactly as the
  corresponding completed helper method did.
- `Tensor` remains the public immutable tensor state and fluent algebra surface, but fixed
  recurrent scan is intentionally accessed through the focused namespace. This correction does
  not establish a general policy that every high-arity or multi-output operation needs a facade.
- Current graph capture can still observe the result producer as one flat occurrence. Current
  forward inference still rejects `RecurrentScanKind` as unsupported, and every backward-capable
  request still fails through the closed autograd inventory before constructing derivative
  Tensors. Compiler 0006A remains Draft and unchanged.
- Existing static NN sequence construction remains untouched. Later NN runtime-valid-length
  overloads may call this seam only after Compiler and a truthful execution path exist.
- The architecture amendment changes the selected Java entry placement only. Module ownership,
  dependency direction, graph identity, semantic meaning, and lifecycle boundaries remain
  unchanged; architecture-test source therefore remains review-only.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.operation.recurrent`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.datatype`

Packages added or changed:

- change only `io.github.pho001.synaptik.model.tensor` by replacing one package-private helper with
  one focused public static expression namespace and removing six Tensor receiver declarations;
- no package is added, moved, or widened.

Type placement:

- `io.github.pho001.synaptik.model.tensor.RecurrentScan` — owns public Tensor-aware construction,
  local validation, descriptor inference, operation assembly, canonical derived-output creation,
  and typed result assembly in the same package as the package-private factory seam.
- `io.github.pho001.synaptik.model.tensor.Tensor` — remains general public Tensor state and fluent
  algebra, with the six recurrent-specific receiver methods removed and every unrelated member
  unchanged.
- `io.github.pho001.synaptik.model.tensor.RecurrentScanResult` — remains the exact two-output
  RNN/GRU carrier because its components are public Tensor wrappers, not operation-semantic
  attributes.
- `io.github.pho001.synaptik.model.tensor.LstmRecurrentScanResult` — remains the exact three-output
  LSTM carrier for the same reason.
- `io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection` — remains the exact
  immutable direction attributes type.
- `io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind` — remains the three-kind
  fixed semantic family with unchanged signatures.

`TensorRecurrentScanExpressions` is removed. No alternate helper, facade, interface, provider,
registry, generic scan namespace, or recurrent package under `extensions/nn` is added.

## Exact public API

Add exactly this owning type:

```java
public final class RecurrentScan {
    private RecurrentScan() {
    }
}
```

The final class has no declared fields, implemented interfaces, superclass other than `Object`,
or nested types. Add exactly these six public methods; each is `static`, non-synchronized,
non-native, and non-varargs:

```java
public static RecurrentScanResult rnn(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

public static RecurrentScanResult rnn(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

public static RecurrentScanResult gru(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

public static RecurrentScanResult gru(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

public static LstmRecurrentScanResult lstm(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

public static LstmRecurrentScanResult lstm(
        Tensor input,
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)
```

There are exactly two overloads for each name. There is no `rnnScan`, `gruScan`, `lstmScan`,
`scan`, `apply`, default-direction, options, instance, nullable-bias, alias, or deprecated form on
`RecurrentScan`, `Tensor`, or another production type.

## Preserved occurrence contract

### Input and output order

Every namespace method preserves the exact 0025E operation inputs:

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

Output descriptor, Tensor-ID allocation, canonical wrapper, and result-component order remains:

```text
RNN/GRU: [outputs, finalHidden]
LSTM:    [outputs, finalHidden, finalCell]
```

### Validation and effects

For each overload, null checks occur in declaration order: `input`, `validLengths`,
`initialHidden`, then LSTM `initialCell`, then `inputWeight`, `hiddenWeight`, optional `bias`, and
`direction`. Each uses the parameter name as the `NullPointerException` message.

After null checks, preserve this exact sequence:

1. Validate floating, rank-three, fully static `input`; allow zero `time` and `batch`, require
   positive `inputSize`.
2. Validate exact non-gradient rank-one fully static `INT64[batch]` `validLengths` without reading
   storage or scalar values.
3. Validate exact-common-type rank-two fully static `initialHidden`, exact batch, and positive
   `hiddenSize`.
4. For LSTM, validate exact-common-type rank-two fully static `initialCell` with exact
   `[batch, hiddenSize]`.
5. Compute `gateCount * hiddenSize` with checked arithmetic and preserve the current overflow
   translation.
6. Validate exact-common-type fully static input and hidden weights with their current ranks and
   extents.
7. When supplied, validate exact-common-type rank-one fully static packed bias.
8. Derive the exact same unresolved output descriptors and common `requiresGrad` OR, excluding
   valid lengths.
9. Construct the same `Operation(kind, direction)`, immutable ordered input/output snapshots, and
   canonical multi-output producer through package-private
   `TensorFactory.createDerivedOutputs`.
10. Assemble the unchanged result record from canonical wrappers in producer-slot order.

All locally detectable invalid input fails before output ID allocation or mutation. During
factory creation, IDs are allocated in output order; if a later allocation exhausts the ID space,
earlier IDs remain consumed, no producer or result carrier is published, and no partial result is
returned. Exceptions, stable role-specific messages, exact Shape-reference reuse, input
non-mutation, and storage-free output behavior remain unchanged.

### Semantics, identity, and downstream boundaries

The RNN-tanh, reset-after GRU, and LSTM equations, gate orders, optional input-side bias,
valid-prefix direction, dense positive-zero padding, zero-length and zero-time meaning, common
floating type, static Shapes, and final-state semantics remain exactly those fixed by 0025E and
ADR 0012. This task selects no numerical algorithm, accumulator, gradient, lowering, execution,
or runtime-value inspection.

Each successful method call remains a fresh identity-distinct occurrence with exactly one
operation, one producer, one result carrier, and two or three canonical Tensor wrappers and IDs.
Equivalent calls remain distinct. No gate, projection, step, mask, padding, zero-fill, or static
unroll Tensor is created.

Generic graph capture still preserves the producer structurally as one flat node. Current
`CapturedGraphInference` continues to reject `RecurrentScanKind` as unsupported before planning,
and current closed autograd coverage continues to reject backward-capable requests before any
derivative Tensor construction. No current Planning capability provider or backend advertises the
family. No Compiler source, test, inventory, or Draft task is changed here.

## Affected files

Expected production source:

- delete `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRecurrentScanExpressions.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/RecurrentScan.java`
- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected Model tests:

- delete `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRecurrentScanExpressionTest.java`
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/RecurrentScanExpressionTest.java`
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

Expected Compiler boundary test:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`

Expected architecture, API, glossary, capability, and Model planning documentation:

- `ARCHITECTURE.md`
- `docs/design/decisions/0012-fixed-recurrent-scan-without-regions.md`
- `docs/architecture/module-boundaries.md`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/model/tasks/0025f-recurrent-scan-expression-namespace-correction.md`

No other file is expected.

## Maximum scope

This task may create, modify, or delete at most:

- 3 production Java paths;
- 15 Model test paths;
- 1 Compiler boundary-test path;
- 10 architecture, API, glossary, capability, and Model planning paths;
- 29 paths total.

The documentation pass found one current-tense receiver claim in the originally review-only
Training API. The coordinator authorized `docs/api/training-api.md` as the eighth documentation
path so the correction does not ship with a known false API statement. That scope amendment is
limited to the directly affected recurrent-scan paragraph.

The result records, recurrent enums, `TensorFactory`, `TensorLinearExpressions`, NN sources/tests,
Compiler production sources/other tests/master plan, architecture-test sources,
lifecycle/runtime/training focused architecture pages, unrelated public APIs, completed planning
history, global roadmap, Gradle, and
dependency declarations are review-only and remain unchanged. If another path, public member,
package, module, bridge, or behavior change is needed, stop and propose a separately justified
follow-up instead of expanding this correction.

## Acceptance criteria

- Public final `RecurrentScan` exists in `model.tensor`, has one private no-argument constructor,
  no declared fields or nested types, and exactly the six specified public static methods.
- `TensorRecurrentScanExpressions` no longer exists in production source or compiled output.
- `Tensor` has no declared `rnnScan`, `gruScan`, or `lstmScan` method and no recurrent imports; its
  public declared-method count is exactly 202 and every unrelated method remains unchanged.
- The six namespace entries have the exact names, return types, parameter order/types, static and
  visibility modifiers, and biased/unbiased split specified above; no alias or generic scan entry
  exists.
- Every existing fixed-scan kind, signature, direction, equation, gate order, Shape, type,
  gradient flag, output descriptor, ordered input/output, canonical wrapper, ID, freshness, and
  one-producer provenance assertion still passes through the namespace.
- Null/descriptor/overflow validation order, exception class/message, no-ID local-failure
  boundary, factory-exhaustion partial-consumption behavior, and non-mutation remain exact.
- Model construction still never reads or requires `validLengths` storage or values and still
  accepts zero time and zero batch under the existing descriptor contract.
- `RecurrentDirection`, `RecurrentScanKind`, `RecurrentScanResult`,
  `LstmRecurrentScanResult`, and `TensorFactory.createDerivedOutputs` are byte-for-byte unchanged
  after independent Javadoc review confirms they remain accurate.
- `Tensor.linear(...)`, its helper and tests, and the NN `Linear` module remain unchanged except
  for the one shared Tensor public-method-count assertion returning to 202.
- All NN cell, one-directional sequence, bidirectional sequence, module, factory, parameter,
  static-unroll, compact-output, final-state, and `long[]` valid-length contracts remain
  unchanged. No current NN implementation imports or calls `RecurrentScan`.
- Current generic capture and Compiler inference/autograd fail-closed behavior are documented and
  unchanged. The boundary inventory test explicitly separates the 128 supported signatures from
  the exact three recurrent signatures deferred until Compiler 0006A, proves their union equals
  the complete Model inventory, and checks every deferred boundary role rejects with the exact
  unknown/unclassified reason without allocating a Tensor ID. Compiler 0006A remains Draft
  without a detailed specification.
- `ARCHITECTURE.md`, ADR 0012, and module-boundary explanation consistently authorize the focused
  static namespace and no longer prescribe six Tensor receiver methods. The ADR records the
  correction without erasing the historical 0025E decision sequence.
- Tensor, Compile, and Training API documentation clearly distinguish the low-level Model
  namespace from ordinary NN sequence use, static unrolling, and future executable adoption.
- Model capabilities and glossary use `RecurrentScan.rnn/gru/lstm` and retain all current-versus-
  planned, dense-output, valid-length, fixed-family, and no-region boundaries.
- The documentation pass records reasoned no-change conclusions for lifecycle,
  runtime/prepare/backend, training graph, other public APIs, result-carrier and enum Javadocs,
  architecture tests, Compiler, NN, Data, other Model operations, Gradle, and dependencies.
- Model 0025F is Complete only after implementation and documentation evidence pass; Model 0026
  and Compiler 0006A remain Draft without detailed specifications.
- A separate documentation-focused agent pass has independently finalized affected Javadocs,
  explanatory/API documentation, glossary impact, architecture amendment, and planning status in
  the same overall change.

## Tests / validation

The implementation context runs focused tests while developing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.tensor.RecurrentScanExpressionTest \
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

Then run the affected-module task tier once after executable Java stabilizes:

```bash
./gradlew :modules:model:test
```

After the coordinator-authorized test-only Compiler boundary correction, run its focused test:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest
```

Because the authoritative public API contract changes and six methods are removed, run one
cross-repository compatibility checkpoint after the Model test succeeds:

```bash
./gradlew test
./gradlew :testing:architecture-tests:test
```

The separate documentation-focused context reuses all successful Java evidence unless it changes
executable Java afterward. It runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The implementation and documentation contexts also record these automated or justified manual
checks:

- Model tests lock the exact public `RecurrentScan` type/method surface, absence of recurrent
  Tensor receiver names, Tensor public count 202, old-helper absence, exact result records, and
  preserved occurrence/effect behavior;
- generated Javadoc and an external-package Java 26 compilation check prove all six namespace
  forms are usable without package-private access and the removed Tensor forms do not compile;
- source/import inventory confirms no production alias, generic scan body, widened derived-output
  seam, downstream Compiler/NN adoption, or unrelated Tensor/Linear change;
- a temporary Compiler-package probe may reuse the 0025E pattern to confirm one flat captured
  node, unsupported forward inference, and unknown/unclassified autograd rejection without
  editing Compiler source; record it only if used;
- documentation links, heading anchors, code fences, final newlines, carriage returns, and
  trailing whitespace are checked against all changed Markdown files;
- architecture contract, ADR, focused architecture explanation, Tensor/Compile APIs, glossary,
  capabilities, and Model planning status use the corrected surface consistently;
- the final task inventory is exactly within the 29 allowed paths and contains no Compiler path
  except the one authorized boundary test, and no NN, Data, backend, Runtime, Prepare, Engine,
  Training source, architecture-test source, Gradle, dependency, global-roadmap, completed-task, or
  unrelated worktree change; and
- the Model master plan has exactly one Ready row before implementation, and task/master status is
  synchronized only after every implementation, documentation, and validation criterion passes.

## Dependencies

- Completed [Model 0025E](0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
  supplies the exact fixed family, implementation, tests, Javadocs, and fail-closed evidence being
  relocated without semantic change.
- Completed [Model 0025](0025-canonical-tensor-producer-outputs.md) supplies the canonical indexed
  output wrappers and package-private factory-atomic multi-output seam.
- Accepted ADR 0012 and completed NN 0021A supply the fixed-family and no-region architecture;
  the explicit API-placement correction in this task amends only their Tensor-surface choice.
- The repository compatibility precedent permits atomic removal without aliases for newly
  introduced provisional APIs that have no stabilized released contract.
- Model 0025E is Complete, while Model 0026 is Draft without a specification. Inserting 0025F
  before 0026 makes this correction the first unfinished Model task and sole Ready Model frontier.
- Draft Compiler 0006A depends on the completed fixed Model family and must not begin until this
  correction is Complete, but it remains unchanged and unspecified here.

## Follow-up tasks

- Compiler 0006A remains the separate owner of forward inference, final validation, closed
  inventory adoption, and explicit fail-closed BPTT handling. When it becomes the active global
  frontier, its clean-context implementation must consume `RecurrentScan`-constructed producers
  without depending on the construction namespace itself.
- Later concrete-backend, Engine, and execution tasks remain responsible for truthful capability,
  prepared-loop lowering, complete runtime valid-length validation, input binding, publication,
  and execution.
- Later NN runtime-valid-length work may call `RecurrentScan` only after that executable path is
  available. Existing static sequence APIs and unrolling remain supported and unchanged.
- BPTT, dynamic Shapes, bidirectional fixed-scan composition, packed/compacted execution, broader
  cell variants, and general regions remain separate future decisions.
- Model 0026 remains Draft without a detailed task specification.

## Architecture impact

Expected impact: focused public API-placement amendment only.

The user-approved correction replaces the architecture contract's six Tensor receiver methods
with six static methods on `model.tensor.RecurrentScan`. The implementation must update
`ARCHITECTURE.md`, ADR 0012, and the directly affected module-boundary explanation in the same
change. Fixed semantics, Model ownership, package dependency direction, graph identity, Compiler
and lifecycle ownership, and execution boundaries do not change. No architecture-test source
change is required because no dependency or enforcement rule changes; existing architecture
tests must still pass.

If implementation requires a semantic, lifecycle, ownership, dependency, graph-region, factory-
visibility, or broader public-surface change beyond this exact amendment, stop and report the
conflict instead of extending the task.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are the clean-context implementation agent for Synaptik Model task 0025F in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/model/master-plan.md, and
docs/planning/modules/model/tasks/0025f-recurrent-scan-expression-namespace-correction.md
completely. Read the directly referenced ADR, focused architecture/API documentation, completed
0025E source/tests/Javadocs, TensorFactory producer seam, current NN sequence/cell/module
contracts, and Compiler fail-closed boundaries before editing.

Implement task 0025F exactly as specified. Preserve unrelated and concurrent worktree changes.
Do not implement out-of-scope Compiler, NN, Data, backend, execution, generic scan/body, Linear,
FLOAT16, roadmap, or unrelated Tensor work. Stop and report any architecture or maximum-scope
conflict.

The coordinator authorizes exactly one Compiler test-only boundary correction in
`FirstOrderGradientCoverageTest`: distinguish the supported and deferred recurrent signature
sets explicitly, prove their exact union equals the complete Model production inventory, and
assert direct fail-closed rejection without allocation. Do not change Compiler production or any
other Compiler test.

After Java implementation and the specified Model/repository validation, hand the diff and exact
test evidence to a separate documentation-focused agent/thread with clean context. That pass must
follow docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs,
architecture/ADR wording, Tensor/Compile APIs, glossary, capabilities, and planning status, record
the required no-change conclusions, reuse successful Java evidence unless it changes executable
behavior, and run final Javadoc/documentation validation.

At the end, update this task's validation evidence, local decisions, implementation notes,
completion summary, and final status. Mark Complete only after every acceptance criterion and the
documentation pass succeed.
```

## Local decisions

- The corrected public owner is `io.github.pho001.synaptik.model.tensor.RecurrentScan`. The name
  states the fixed domain concept without the redundant `Tensor` prefix used by package-private
  helpers and without implying a generic body-bearing scan abstraction.
- The method names are `rnn`, `gru`, and `lstm`. The namespace already supplies the scan context,
  so retaining the `Scan` suffix would be redundant. Bias is represented only by exact overload
  arity.
- `input` becomes the explicit first argument, matching the helper and producer-input order that
  0025E already implemented. All other arguments retain their exact current order.
- The namespace stays in `model.tensor` so it can call the package-private canonical
  derived-output factory seam. Widening that seam or adding a bridge would be a larger and less
  cohesive API change.
- Both result records stay in `model.tensor`: they are typed public carriers of Tensor wrappers,
  not semantic kinds or operation attributes. Moving them would create unrelated package churn
  and provide no construction or ownership benefit.
- No compatibility alias or deprecation period is selected because the receiver methods were
  introduced only in the immediately preceding Model task, have no downstream repository
  adoption, and are not part of a stabilized released compatibility contract.
- The coordinator-authorized Compiler test-only correction names the three deferred recurrent
  signature fingerprints explicitly. It does not filter discovered kinds or widen the supported
  inventory, so future missing Compiler adoption remains visible while Compiler 0006A stays
  Draft.
- `Tensor.linear(...)` remains unchanged. Linear is a natural single-output algebraic composition;
  this correction is about a domain-heavy explicit-state, multi-input/multi-output protocol, not a
  method-count threshold or blanket facade policy.

## Known limitations

- The fixed recurrent-scan metadata remains non-executable until Compiler 0006A and later
  lifecycle/backend work adopt it.
- Every backward-capable compile remains unsupported; BPTT is not designed here.
- Only the existing fully static Shape, runtime `INT64[batch]` valid-prefix, one-direction, dense
  original-time-aligned, three-cell-family contract is represented.
- The low-level namespace is not the ordinary NN user API. Existing NN sequence containers remain
  statically unrolled and accept Java `long[]` lengths until later executable evidence supports a
  deliberate runtime-length addition.

## Validation evidence

Planning context `/root/model_recurrent_scan_api_correction_planning` read the repository
instructions, authoritative and focused architecture contracts, ADR 0012, documentation rules and
planning profile, planning guide and roadmap, Model/Compiler/NN master plans, completed task
0025E, current recurrent production types, Tensor and TensorFactory construction seams, affected
Model tests and public-surface locks, Tensor/Compile/Training API documentation, glossary,
capabilities, NN cell/sequence/module/Linear contracts, Compiler fail-closed dispatch, and current
git state.

Planning-stage validation is recorded below. No Java test, Javadoc generation, or build command
is run by this planning-only context.

- The repository began at clean `HEAD` `e0ae213f`; committed Model 0025E is `390c2a77` and its
  actual public surface, tests, Javadocs, API documentation, and downstream fail-closed state were
  inspected rather than inferred from planning prose.
- Task ordering checks select 0025F immediately after Complete 0025E and before Draft 0026. The
  Model task directory and master plan contain exactly one Ready task/row, both for 0025F; Model
  0026 and Compiler 0006A have no detailed task specification.
- A read-only Markdown checker validated the Model master plan and this task: 2 files, 43
  generated heading IDs, 207 repository-local file links, no missing target or heading anchor,
  balanced fences, final newlines, no carriage return, and no trailing whitespace.
- `git diff --check` passed for the tracked Model-master change.
  `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0025f-recurrent-scan-expression-namespace-correction.md`
  returned the expected difference status with no whitespace diagnostic for the new task.
- The planning context created or changed exactly the Model master plan and this task. It did not
  touch, stage, revert, reformat, or incorporate CPU/backend, Data, global-roadmap, Compiler, NN,
  Java, test, API, glossary, or architecture implementation/documentation paths.
- The planning context initially scoped 27 paths: 3 production Java paths, 15 Model test paths,
  and 9 architecture/API/glossary/capability/Model-planning paths. The later coordinator-authorized
  Compiler boundary-test correction and Training API omission correction add one path each, for
  the exact final 29-path scope. The 14 existing Tensor
  surface locks affected by 0025E were reconciled to the renamed recurrent test plus 13 existing
  count/name locks; recurrent semantic/signature tests remain unchanged.
- The planning review found no unresolved design question. `model.tensor.RecurrentScan` is the
  only placement that keeps Tensor-aware construction public while directly preserving the
  package-private canonical-output factory seam and the operation-to-Tensor dependency boundary.

Implementation context `/root/model_0025f_implementation` read the required repository,
architecture, ADR, planning, Model/Compiler/NN boundary, source, test, Javadoc, and worktree
context before editing. It completed and froze executable Model source/tests before the
authoritative Model tier and changed no Model executable path afterward.

- `Tensor` lost exactly the six recurrent receiver overloads and recurrent imports; public final,
  field-free `model.tensor.RecurrentScan` now exposes exactly the specified six public static
  `rnn`, `gru`, and `lstm` biased/unbiased overloads with explicit `input` first. The old helper is
  absent from source and compiled output.
- The shared private construction path is text-identical to 0025E where semantics apply. Focused
  tests preserve declaration-order null checks, descriptor validation and messages, static
  Shapes, gate packing, gradient OR, ordered slots, canonical wrappers, IDs, failure effects,
  one-producer provenance, and capture boundaries.
- The 14-suite focused Model selection passed 144 tests with zero failures, errors, or skips.
  Exactly one authoritative `./gradlew :modules:model:test` then passed 1,046 tests in 129 suites
  with zero failures, errors, or skips. Preliminary `./gradlew :modules:model:javadoc` passed.
- The first repository checkpoint exposed one stale closed-inventory equality in
  `FirstOrderGradientCoverageTest`: committed Compiler support listed 128 signatures while the
  complete Model inventory now also contained the three recurrent signatures intentionally
  deferred until Compiler 0006A. The coordinator authorized only that directly affected Compiler
  test path. Its corrected six-test focused class passed, and the replacement `./gradlew test`
  passed, including 209 Compiler tests and 1,046 Model tests. The affected architecture-test task
  also passed.
- The Compiler boundary test names the exact three deferred fingerprints, proves the 128-row
  supported set is disjoint from them and their 131-row union equals the complete Model
  inventory, then checks every deferred kind/cardinality/output/input role returns fail-closed
  with the exact unknown/unclassified reason and consumes no Tensor ID. No filter can conceal a
  future missing adoption, and no Compiler production source or second Compiler test changed.
- `javap` showed exactly six public static methods on `RecurrentScan` and no recurrent receiver on
  `Tensor`. A Java 26 external-package probe compiled and ran all six namespace calls; a separate
  compile failed as required for removed `input.rnnScan(...)` with `cannot find symbol`.
- A temporary Compiler-package probe captured one flat LSTM node with six inputs and three
  outputs, observed unsupported forward inference, and observed exact unknown/unclassified
  autograd rejection for every output/input role without another Tensor-ID allocation.
- Source/import scans found no production alias, receiver spelling, generic body/region,
  downstream Compiler/NN namespace adoption, bridge, or widened factory seam. The result records,
  recurrent enums, `TensorFactory`, `TensorLinearExpressions`, and all NN sources remained
  byte-for-byte unchanged.
- Implementation changed exactly 21 authorized paths so far: 3 Model production paths, 15 Model
  test paths, the one coordinator-authorized Compiler boundary test, and the two Model planning
  paths. Concurrent CPU/backend work was neither touched nor incorporated. The clean
  documentation pass owns the remaining eight authorized architecture/API/glossary/capability
  paths, bringing the final maximum to exactly 29.

## Implementation notes

Canonical documentation context `/root/model_0025f_docs` independently read the repository and
architecture instructions, ADR 0012, documentation rules, General/API-and-Javadoc/Architecture/
Decision-record/Planning profiles, planning guide and roadmap, Model/Compiler plans, tasks 0025E
and 0025F, the complete implementation and test diff, rendered Javadocs, result carriers, enums,
factory seam, Compiler boundary, and relevant NN, Tensor, Compile, Training, glossary, capability,
and architecture contracts before finalization.

The review found no executable API, semantic, provenance, validation/effect, or fail-closed
defect. Removing Javadocs before comparison proved the new public namespace's executable body is
text-identical to the 0025E helper after only class/method visibility and name normalization. The
documentation pass changed no executable Java statement or test and therefore reused the frozen
successful Java tiers rather than rerunning them.

The initial seven-document scope omitted one current-tense Training API statement that still
claimed six Tensor receiver overloads. The coordinator authorized the minimal eighth
documentation path and final 29-path scope. Only that recurrent paragraph changed; its existing
Java `long[]`, static-unroll, compact-output, and future-adoption boundaries remain intact.

## Completion summary

- Completed changes: public final field-free `RecurrentScan` owns exactly six static biased or
  bias-free `rnn`, `gru`, and `lstm` constructors with explicit input first; `Tensor` has no
  recurrent receiver or alias; the old helper is absent; all 0025E semantics, validation,
  effects, descriptors, IDs, canonical outputs, and provenance remain exact. The directly
  affected Compiler test explicitly partitions its exact 128 supported signatures from the exact
  three deferred recurrent signatures and proves the 131-signature union is the complete Model
  inventory while preserving unknown/unclassified, no-allocation failure.
- Documentation and Javadocs: finalized `RecurrentScan` as an advanced low-level expression
  namespace, not a layer, module, execution service, registry, or general body. Updated the
  architecture contract, ADR 0012, module boundaries, Tensor/Compile/Training API references,
  glossary, and Model capabilities. The documents direct ordinary users to NN sequences, retain
  their static Java `long[]` unrolling, defer later NN delegation until Compiler/backend adoption,
  and explain why the single-output last-axis `Tensor.linear` composition and state-owning NN
  `Linear` remain unchanged. The result-carrier, enum, `TensorFactory`, and Linear Javadocs remain
  accurate and byte-for-byte unchanged; no `model.tensor` package-info file exists or requires a
  new package-level contract.
- Files changed or created: exactly 29 authorized paths—3 Model production, 15 Model tests, 1
  coordinator-authorized Compiler boundary test, 8 architecture/API/glossary/capability
  documents, and 2 Model planning paths. Nine concurrent CPU/backend paths were excluded and
  untouched.
- Reused Java evidence: focused Model 144 tests; authoritative Model 1,046 tests in 129 suites;
  focused Compiler boundary 6 tests; replacement root suite including Compiler 209 and Model
  1,046; and architecture tests all passed before executable freeze.
- Final documentation evidence: `./gradlew :modules:model:javadoc` passed after final Javadocs;
  rendered `RecurrentScan`, `Tensor`, both result records, and both recurrent enums were inspected.
  `javap` and reflection proved exactly six public static namespace methods, a public final
  field-free type with one private no-argument constructor, 202 public Tensor methods, no
  recurrent Tensor entry, two unchanged `linear` entries, exact result components, and absent old
  helper. A Java 26 external-package probe constructed all six forms; removed receiver use failed
  compilation with `cannot find symbol`.
- Final repository evidence: source/import/forbidden checks found no alias, general body/region,
  downstream Compiler/NN adoption, factory widening, or unrelated Tensor/Linear change. The 10
  changed Markdown files passed 656 link checks, 800 heading/explicit-anchor checks,
  balanced fences, final-newline, CRLF, and trailing-whitespace checks. Exact 29-path inventory,
  tracked and untracked whitespace checks, and `git diff --check` passed. Model 0025F and its
  master row are Complete; Compiler 0006A and Model 0026 remain Draft without specifications, and
  no Model task is Ready.
- Reasoned no-change conclusions: lifecycle, runtime/prepare/backend, training-graph, public and
  Runtime APIs, architecture-test source, Compiler production/other tests, NN source/tests, Data,
  other Model operations, Gradle, dependencies, and the global roadmap require no change because
  this correction only relocates Model expression construction and adds no execution or module
  boundary. Existing successful architecture tests remain applicable and were not repeated.
- Unresolved issues: none. Follow-up remains the already Draft Compiler 0006A forward-adoption
  task and later explicitly owned backend, NN runtime-length, Engine, and BPTT work.

Status: Complete
