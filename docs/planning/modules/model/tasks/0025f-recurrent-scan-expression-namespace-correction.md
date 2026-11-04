# Task 0025F: Recurrent-Scan Expression Namespace Correction

**Editorial reading key (current context, not a status update).**

This record concerns API placement: six provisional Tensor receiver overloads became
`RecurrentScan.rnn/gru/lstm`, not a general scan body. Ordinary NN callers use sequence containers;
advanced callers pass Tensor descriptions of input, runtime lengths, explicit states and parameters
to the static namespace. A producer is one identity-distinct operation occurrence; canonical
wrappers identify its ordered outputs. Construction validates descriptors, not length values, and
returns storage-free expressions: it neither executes recurrence nor owns result buffers. Model’s
semantic responsibility is not storage ownership.

The historical Compiler rejection and deferred execution statements describe this task’s scope, not
current support. Follow the [current fixed-scan
reference](../../../../reference/tensor/fixed-scans.md#fixed-recurrent-scan-expressions) for
construction, execution and NN-route distinctions. Model defines meaning; execution providers
realize supported routes without redefining it. For example, a row of length two in a three-step
input reverses only its first two positions under `REVERSE`; the third output position is positive
zero. Length zero preserves initial states semantically. These are value-level rules, not results
computed by the namespace.

## Status

Complete

## Goal

Correct the newly introduced fixed recurrent-scan construction surface before [Compiler](../../../../glossary.md#term-compiler) 0006A
adopts it. Ordinary neural-network callers should continue to use `RnnSequence`, `GruSequence`,
and `LstmSequence`; the fixed recurrent scan remains a low-level [Model](../../../../glossary.md#term-model) expression seam for later
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and Compiler integration, but its six domain-heavy multi-input/multi-output entries must not
remain receiver methods on every [`Tensor`](../../../../glossary.md#tensor).

Replace the package-private `TensorRecurrentScanExpressions` helper and six public Tensor receiver
overloads with one intentionally named public static namespace:

```text
RecurrentScan.rnn/gru/lstm(input, runtime lengths, explicit states, explicit parameters, direction)
  -> one immutable recurrent-scan producer occurrence
  -> existing typed canonical output carriers
```

The correction changes only [API](../../../../glossary.md#term-application-programming-interface-api) placement and names. It preserves the exact fixed [operation](../../../../glossary.md#operation)
family, [argument](../../../../glossary.md#term-argument) and [producer](../../../../glossary.md#term-producer)-input order, [validation](../../../../glossary.md#term-validation) and failure effects, [descriptors](../../../../glossary.md#term-descriptor), Tensor-ID
allocation, [canonical](../../../../glossary.md#term-canonical-form) output wrappers, one-producer [provenance](../../../../glossary.md#provenance), mathematical meaning, and current
Compiler [fail-closed behavior](../../../../glossary.md#admission-preflight-and-fail-closed-validation) implemented by completed task 0025E.

## Motivation

The six 0025E receiver overloads place a stateful recurrent domain protocol on the general
[`Tensor`](../../../../glossary.md#tensor) instance surface. Each call needs a [time-major](../../../../glossary.md#term-time-major) input, [runtime](../../../../glossary.md#term-runtime) lengths, explicit carried
state, several packed [parameter](../../../../glossary.md#term-parameter) Tensors, optional [bias](../../../../glossary.md#term-bias), a direction attribute, and two or three
outputs. That [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape) is unlike a natural single-output algebraic expression such as
`Tensor.linear(...)`, and ordinary users should not need to discover it while working with a
general Tensor.

A focused static namespace communicates that this is a low-level fixed [operation](../../../../glossary.md#operation) constructor
without adding a generic [scan body](../../../../glossary.md#backpropagation-through-time-and-recurrent-state), [widening](../../../../glossary.md#term-widening) the [Tensor factory](../../../../glossary.md#tensor-factory), or moving Tensor-aware
construction into the operation-semantic package. Keeping the namespace in `model.tensor` lets it
reuse the existing package-private [canonical](../../../../glossary.md#term-canonical-form) derived-output seam directly. The correction is
deliberately immediate and breaking: the provisional receiver methods were introduced only by
the preceding [commit](../../../../glossary.md#term-commit) and have no repository consumer or stabilized released compatibility
contract.

## [Scope](../../../../glossary.md#term-scope)

- Remove exactly the six public `rnnScan`, `gruScan`, and `lstmScan` receiver overloads, their
  Javadocs, and the now-unused recurrent-direction import from [`Tensor`](../../../../glossary.md#tensor).
- Delete package-private `TensorRecurrentScanExpressions`.
- Add public final `io.github.pho001.synaptik.model.tensor.RecurrentScan` as an
  instance-field-free, stateless static expression namespace with a private no-argument
  constructor.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- Add exactly six public static methods to that namespace: biased and unbiased overloads of
  `rnn`, `gru`, and `lstm`, with `input` as the explicit first parameter and every remaining
  argument in the exact current order.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Preserve the complete 0025E validation, descriptor, effect, identity, provenance, and semantic
  implementation while changing only the owning public type and entry names.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Keep `RecurrentDirection` and `RecurrentScanKind` unchanged in
  `io.github.pho001.synaptik.model.operation.recurrent`.
- Keep `RecurrentScanResult` and `LstmRecurrentScanResult` unchanged in
  `io.github.pho001.synaptik.model.tensor` as typed carriers of [canonical](../../../../glossary.md#term-canonical-form) Tensor outputs.
- Restore the public Tensor declared-method inventory from 208 to 202 and remove only the
  recurrent names and recurrent-specific reflection allowances from the existing surface locks.
- Rename and refocus the recurrent construction test around `RecurrentScan`; preserve its full
  biased/unbiased, direction, descriptor, validation-order, Tensor-ID, and provenance coverage.
- Correct the directly affected Compiler inventory boundary test so it distinguishes the exact
  Compiler-supported closed production inventory from the three recurrent signatures deferred
  until Compiler 0006A, proves their exact union equals the complete Model inventory, and locks
  fail-closed rejection without Tensor-ID allocation.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [signatures glossary](../../../../glossary.md#term-signature); [Model glossary](../../../../glossary.md#term-model); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Amend the authoritative and focused architecture wording to select the low-level static
  namespace instead of Tensor receiver methods, without changing fixed-scan semantics, lifecycle
  ownership, or dependency direction.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- Finalize affected Javadocs, Tensor/Compile API explanations, glossary terminology, Model
  capabilities, and Model planning records through the required independent documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- changing `RecurrentDirection`, `RecurrentScanKind`, their constants, attributes/signatures,
  equations, gate packing, traversal, padding, static Shapes, data types, or output roles<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [signatures glossary](../../../../glossary.md#term-signature); [padding glossary](../../../../glossary.md#padding); [Shapes glossary](../../../../glossary.md#shape))</span>
- moving, renaming, extending, or changing the [validation](../../../../glossary.md#term-validation) behavior of `RecurrentScanResult` or
  `LstmRecurrentScanResult`
- [widening](../../../../glossary.md#term-widening) `TensorFactory.createDerivedOutputs`, moving it to another package, or adding a bridge,
  service, registry, provider, manager, builder, factory object, descriptor facade, or general
  derived-output API<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [descriptor glossary](../../../../glossary.md#term-descriptor); [facade glossary](../../../../glossary.md#term-facade); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- adding `scan(body)`, a callback, lambda, user-defined cell, Tensor body, callable block, nested
  graph, captured subgraph, loop intermediate representation, region, general control flow, or
  another identity domain<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [intermediate representation glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity domain glossary](../../../../glossary.md#term-identity-domain))</span>
- retaining `rnnScan`, `gruScan`, or `lstmScan` as aliases, deprecated bridges, forwarding
  methods, package-private duplicates, or alternate public spellings<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- adding default direction, nullable or optional bias, varargs, options, builder, instance method,
  or generic recurrent entry<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- changing `Tensor.linear(...)`, `TensorLinearExpressions`, the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [`Linear`](../../../../glossary.md#term-linear-meanings) module, or any other
  Tensor operation; Linear remains a natural single-output algebraic Tensor expression<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- changing the existing `RnnCell`, `GruCell`, `LstmCell`, `RnnSequence`, `GruSequence`,
  `LstmSequence`, bidirectional containers, [`Module`](../../../../glossary.md#term-module), `ModuleFactory`, parameter/state ownership,
  or NN documentation/planning history<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [planning glossary](../../../../glossary.md#term-planning))</span>
- redirecting current NN sequence containers to the fixed scan; their compact `long[]` static
  APIs and explicit static unrolling remain the general and reference path<span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- compiler production inference, graph finalization, autograd inventory, backpropagation through
  time (BPTT), Planning capability, Prepare, Runtime, Engine, backend, execution, binding, Data,
  Training, or serialization behavior<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span><span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd); [backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [binding glossary](../../../../glossary.md#term-binding); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- editing Compiler 0006A, the Compiler/NN/Data/backend master plans, the global roadmap, completed
  task 0025E, completed NN 0021A, architecture-test source, dependency declarations, Gradle files,
  or unrelated documentation<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- [Model](../../../../glossary.md#term-model) task 0026 or [IEEE](../../../../glossary.md#term-ieee) [FLOAT16](../../../../glossary.md#term-float16)/mixed-[precision](../../../../glossary.md#term-precision) work

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the fixed recurrent-scan,
  flat Tensor-producer, Model ownership, compiler, runtime, and training boundaries<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [Model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)
- [Model capabilities](../capabilities.md)
- [Completed task 0025E](0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
- [Completed task 0025](0025-canonical-tensor-producer-outputs.md)
- [Completed task 0019D](0019d-linear-convenience.md)
- [Completed NN task 0021A](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Compiler master plan](../../compiler/master-plan.md), including Draft 0006A
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- The accepted fixed recurrent operation remains one ordinary flat, identity-distinct,
  multi-output <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`TensorProducer`](../../../../glossary.md#producer--tensorproducer) [occurrence](../../../../glossary.md#term-occurrence) and later one ordinary flat `CompiledNode`. API
  relocation must not add a body, region, nested graph, callback, or per-time producer.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [callback glossary](../../../../glossary.md#term-callback); [producer glossary](../../../../glossary.md#term-producer))</span>
- Model continues to own the fixed semantics and descriptor-visible construction only. It gains no
  compiler, runtime, backend, module, parameter, execution, or lifecycle state.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module); [parameter glossary](../../../../glossary.md#term-parameter); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `RecurrentScan` belongs in `model.tensor` because it constructs public Tensor expressions and
  must use package-private `TensorFactory.createDerivedOutputs`. The task must not widen that seam
  or create a cross-package bridge merely to place the namespace elsewhere.
- The `model.operation` packages remain independent of public Tensor construction. The semantic
  enums therefore stay in `model.operation.recurrent`, while the Tensor-aware namespace and
  typed Tensor-output carriers stay in `model.tensor`.
- `RecurrentScan` is a public final namespace, not an instantiable service. It has one private
  no-argument constructor, no declared fields or nested types, and only static behavior. Its six
  public entries are non-synchronized and non-varargs; invocation-local metadata makes concurrent
  calls independent except for the existing thread-safe JVM-wide Tensor-ID allocation effects.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [metadata glossary](../../../../glossary.md#term-metadata); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Every successful entry creates the same fresh operation, producer, result carrier, canonical
  wrappers, descriptors, and IDs in the same order as its 0025E receiver predecessor.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Every failure occurs at the same validation stage with the same exception class, message, and
  Tensor-ID effect as 0025E. Making <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`input` explicit adds no new order: it is already the helper's
  first null check and remains first.
- No overload delegates through another public method. Each public entry null-checks in exact
  declaration order and reaches the one shared private recurrent construction path exactly as the
  corresponding completed helper method did.
- [`Tensor`](../../../../glossary.md#tensor) remains the public immutable tensor state and fluent algebra surface, but fixed
  recurrent scan is intentionally accessed through the focused namespace. This correction does
  not establish a general policy that every high-arity or multi-output operation needs a facade.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [arity glossary](../../../../glossary.md#term-arity); [facade glossary](../../../../glossary.md#term-facade))</span>
- Current graph capture can still observe the result producer as one flat occurrence. Current
  forward inference still rejects <span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference))</span>`RecurrentScanKind` as unsupported, and every backward-capable
  request still fails through the closed autograd inventory before constructing derivative
  Tensors. Compiler 0006A remains Draft and unchanged.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Existing static NN sequence construction remains untouched. Later NN runtime-valid-length
  overloads may call this seam only after Compiler and a truthful execution path exist.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- The architecture amendment changes the selected Java entry placement only. Module ownership,
  dependency direction, graph identity, semantic meaning, and lifecycle boundaries remain
  unchanged; architecture-test source therefore remains review-only.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.operation.recurrent`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.datatype`

Packages added or changed:

- change only `io.github.pho001.synaptik.model.tensor` by replacing one package-private helper with
  one focused public static expression namespace and removing six Tensor receiver declarations;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- no package is added, moved, or widened.

Type placement:

- `io.github.pho001.synaptik.model.tensor.RecurrentScan` — owns public Tensor-aware construction,
  local validation, descriptor inference, operation assembly, canonical derived-output creation,
  and typed result assembly in the same package as the package-private factory seam.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [operation glossary](../../../../glossary.md#operation); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` — remains general public Tensor state and fluent
  algebra, with the six recurrent-specific receiver methods removed and every unrelated member
  unchanged.
- `io.github.pho001.synaptik.model.tensor.RecurrentScanResult` — remains the exact two-output
  RNN/GRU carrier because its components are public Tensor wrappers, not operation-semantic
  attributes.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.model.tensor.LstmRecurrentScanResult` — remains the exact three-output
  LSTM carrier for the same reason.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- `io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection` — remains the exact
  immutable direction attributes type.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind` — remains the three-kind
  fixed semantic family with unchanged signatures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>

`TensorRecurrentScanExpressions` is removed. No alternate helper, [facade](../../../../glossary.md#term-facade), interface, provider,
[registry](../../../../glossary.md#term-registry), generic scan namespace, or recurrent package under `extensions/nn` is added.

## Exact public [API](../../../../glossary.md#term-application-programming-interface-api)

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
`scan`, `apply`, default-direction, options, instance, nullable-[bias](../../../../glossary.md#term-bias), [alias](../../../../glossary.md#aliasing-and-arena), or deprecated form on
`RecurrentScan`, [`Tensor`](../../../../glossary.md#tensor), or another production type.

## Preserved [occurrence](../../../../glossary.md#term-occurrence) contract

### Input and output order

Every namespace method preserves the exact 0025E [operation](../../../../glossary.md#operation) inputs:

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

Output [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor)-ID allocation, [canonical](../../../../glossary.md#term-canonical-form) wrapper, and result-component order remains:

```text
RNN/GRU: [outputs, finalHidden]
LSTM:    [outputs, finalHidden, finalCell]
```

### [Validation](../../../../glossary.md#term-validation) and effects

For each overload, null checks occur in declaration order: `input`, `validLengths`,
`initialHidden`, then [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) `initialCell`, then `inputWeight`, `hiddenWeight`, optional [`bias`](../../../../glossary.md#term-bias), and
`direction`. Each uses the [parameter](../../../../glossary.md#term-parameter) name as the `NullPointerException` message.

After null checks, preserve this exact sequence:

1. Validate floating, [rank](../../../../glossary.md#term-rank)-three, fully static `input`; allow zero `time` and [`batch`](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), require
   positive `inputSize`.
2. Validate exact non-[gradient](../../../../glossary.md#term-gradient) [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) fully static `INT64[batch]` `validLengths` without reading
   storage or scalar values.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
3. Validate exact-common-type [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) fully static `initialHidden`, exact batch, and positive
   `hiddenSize`.
4. For LSTM, validate exact-common-type rank-two fully static `initialCell` with exact
   `[batch, hiddenSize]`.
5. Compute `gateCount * hiddenSize` with checked arithmetic and preserve the current overflow
   translation.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
6. Validate exact-common-type fully static input and hidden weights with their current ranks and
   extents.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
7. When supplied, validate exact-common-type rank-one fully static packed bias.
8. Derive the exact same unresolved output [descriptors](../../../../glossary.md#term-descriptor) and common `requiresGrad` OR, excluding
   valid lengths.
9. Construct the same `Operation(kind, direction)`, immutable ordered input/output snapshots, and
   canonical multi-output producer through package-private<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
   `TensorFactory.createDerivedOutputs`.
10. Assemble the unchanged result record from [canonical](../../../../glossary.md#term-canonical-form) wrappers in [producer](../../../../glossary.md#term-producer)-slot order.

All locally detectable invalid input fails before output ID allocation or mutation. During
factory creation, IDs are allocated in output order; if a later allocation exhausts the ID space,
earlier IDs remain consumed, no producer or result [carrier](../../../../glossary.md#term-storage-carrier) is published, and no partial result is
returned. Exceptions, stable role-specific messages, exact [Shape](../../../../glossary.md#shape)-reference reuse, input
non-mutation, and storage-free output behavior remain unchanged.

### Semantics, [identity](../../../../glossary.md#term-identity-meanings), and downstream boundaries

The [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)-[tanh](../../../../glossary.md#term-tanh), reset-after [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru), and [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) equations, gate orders, optional input-side [bias](../../../../glossary.md#term-bias),
valid-prefix direction, dense positive-zero [padding](../../../../glossary.md#padding), zero-length and zero-time meaning, common
floating type, static [Shapes](../../../../glossary.md#shape), and final-state semantics remain exactly those fixed by 0025E and
[ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012. This task selects no numerical algorithm, [accumulator](../../../../glossary.md#accumulator-type), [gradient](../../../../glossary.md#term-gradient), [lowering](../../../../glossary.md#lowering), execution,
or [runtime](../../../../glossary.md#term-runtime)-value inspection.

Each successful method call remains a fresh identity-distinct [occurrence](../../../../glossary.md#term-occurrence) with exactly one
[operation](../../../../glossary.md#operation), one [producer](../../../../glossary.md#term-producer), one result [carrier](../../../../glossary.md#term-storage-carrier), and two or three [canonical](../../../../glossary.md#term-canonical-form) [Tensor](../../../../glossary.md#tensor) wrappers and IDs.
Equivalent calls remain distinct. No gate, projection, step, [mask](../../../../glossary.md#term-boolean-mask), padding, zero-fill, or static
unroll Tensor is created.

Generic [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction) still preserves the producer structurally as one flat [node](../../../../glossary.md#node). Current
`CapturedGraphInference` continues to reject `RecurrentScanKind` as unsupported before [planning](../../../../glossary.md#term-planning),
and current closed [autograd](../../../../glossary.md#autograd) coverage continues to reject backward-capable requests before any
[derivative](../../../../glossary.md#term-derivative) Tensor construction. No current Planning capability provider or [backend](../../../../glossary.md#backend) advertises the
family. No [Compiler](../../../../glossary.md#term-compiler) source, test, inventory, or Draft task is changed here.

## Affected files

Expected production source:

- delete `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRecurrentScanExpressions.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/RecurrentScan.java`
- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected [Model](../../../../glossary.md#term-model) tests:

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

Expected [Compiler](../../../../glossary.md#term-compiler) boundary test:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`

Expected architecture, [API](../../../../glossary.md#term-application-programming-interface-api), glossary, capability, and Model [planning](../../../../glossary.md#term-planning) documentation:

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

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create, modify, or delete at most:

- 3 production Java paths;
- 15 [Model](../../../../glossary.md#term-model) test paths;
- 1 [Compiler](../../../../glossary.md#term-compiler) boundary-test path;
- 10 architecture, [API](../../../../glossary.md#term-application-programming-interface-api), glossary, capability, and Model [planning](../../../../glossary.md#term-planning) paths;
- 29 paths total.

The documentation pass found one current-tense receiver claim in the originally review-only
Training API. The coordinator authorized `docs/api/training-api.md` as the eighth documentation
path so the correction does not ship with a known false API statement. That scope amendment is
limited to the directly affected recurrent-scan paragraph.

The result records, recurrent enums, `TensorFactory`, `TensorLinearExpressions`, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) sources/tests,
Compiler production sources/other tests/master plan, architecture-test sources,
[lifecycle](../../../../glossary.md#lifecycle)/[runtime](../../../../glossary.md#term-runtime)/training focused architecture pages, unrelated public APIs, completed planning
history, global roadmap, [Gradle](../../../../glossary.md#term-gradle), and
[dependency](../../../../glossary.md#term-dependency) declarations are review-only and remain unchanged. If another path, public member,
package, [module](../../../../glossary.md#term-module), bridge, or behavior change is needed, stop and propose a separately justified
follow-up instead of expanding this correction.

## Acceptance criteria

- Public final `RecurrentScan` exists in `model.tensor`, has one private no-argument constructor,
  no declared fields or nested types, and exactly the six specified public static methods.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- `TensorRecurrentScanExpressions` no longer exists in production source or compiled output.
- [`Tensor`](../../../../glossary.md#tensor) has no declared `rnnScan`, `gruScan`, or `lstmScan` method and no recurrent imports; its
  public declared-method count is exactly 202 and every unrelated method remains unchanged.
- The six namespace entries have the exact names, return types, parameter order/types, static and
  visibility modifiers, and biased/unbiased split specified above; no alias or generic scan entry
  exists.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Every existing fixed-scan kind, signature, direction, equation, gate order, Shape, type,
  gradient flag, output descriptor, ordered input/output, canonical wrapper, ID, freshness, and
  one-producer provenance assertion still passes through the namespace.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [descriptor glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Null/descriptor/overflow validation order, exception class/message, no-ID local-failure
  boundary, factory-exhaustion partial-consumption behavior, and non-mutation remain exact.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Model](../../../../glossary.md#term-model) construction still never reads or requires `validLengths` storage or values and still
  accepts zero time and zero batch under the existing descriptor contract.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- `RecurrentDirection`, `RecurrentScanKind`, `RecurrentScanResult`,
  `LstmRecurrentScanResult`, and `TensorFactory.createDerivedOutputs` are byte-for-byte unchanged
  after independent Javadoc review confirms they remain accurate.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `Tensor.linear(...)`, its helper and tests, and the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [`Linear`](../../../../glossary.md#term-linear-meanings) module remain unchanged except
  for the one shared Tensor public-method-count assertion returning to 202.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- All NN cell, one-directional sequence, bidirectional sequence, module, factory, parameter,
  static-unroll, compact-output, final-state, and `long[]` valid-length contracts remain
  unchanged. No current NN implementation imports or calls `RecurrentScan`.
- Current generic capture and Compiler inference/autograd fail-closed behavior are documented and
  unchanged. The boundary inventory test explicitly separates the 128 supported signatures from
  the exact three recurrent signatures deferred until Compiler 0006A, proves their union equals
  the complete Model inventory, and checks every deferred boundary role rejects with the exact
  unknown/unclassified reason without allocating a Tensor ID. Compiler 0006A remains Draft
  without a detailed specification.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd); [fail-closed behavior glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- `ARCHITECTURE.md`, ADR 0012, and module-boundary explanation consistently authorize the focused
  static namespace and no longer prescribe six Tensor receiver methods. The ADR records the
  correction without erasing the historical 0025E decision sequence.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Tensor, Compile, and Training API documentation clearly distinguish the low-level Model
  namespace from ordinary NN sequence use, static unrolling, and future executable adoption.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- Model capabilities and glossary use `RecurrentScan.rnn/gru/lstm` and retain all current-versus-
  planned, dense-output, valid-length, fixed-family, and no-region boundaries.
- The documentation pass records reasoned no-change conclusions for lifecycle,
  runtime/prepare/backend, training graph, other public APIs, result-carrier and enum Javadocs,
  architecture tests, Compiler, NN, Data, other Model operations, Gradle, and dependencies.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [training graph glossary](../../../../glossary.md#training-graph); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Model 0025F is Complete only after implementation and documentation evidence pass; Model 0026
  and Compiler 0006A remain Draft without detailed specifications.
- A separate documentation-focused agent pass has independently finalized affected Javadocs,
  explanatory/API documentation, glossary impact, architecture amendment, and planning status in
  the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

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

Then [run](../../../../glossary.md#term-run) the affected-[module](../../../../glossary.md#term-module) task tier once after executable Java stabilizes:

```bash
./gradlew :modules:model:test
```

After the coordinator-authorized test-only [Compiler](../../../../glossary.md#term-compiler) boundary correction, run its focused test:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest
```

Because the authoritative public [API](../../../../glossary.md#term-application-programming-interface-api) contract changes and six methods are removed, run one
cross-repository compatibility [checkpoint](../../../../glossary.md#term-checkpoint) after the [Model](../../../../glossary.md#term-model) test succeeds:

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
  preserved occurrence/effect behavior;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- generated Javadoc and an external-package Java 26 compilation check prove all six namespace
  forms are usable without package-private access and the removed Tensor forms do not compile;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [compile glossary](../../../../glossary.md#compile))</span>
- source/import inventory confirms no production alias, generic scan body, widened derived-output
  seam, downstream Compiler/NN adoption, or unrelated Tensor/Linear change;<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [scan body glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- a temporary Compiler-package probe may reuse the 0025E pattern to confirm one flat captured
  node, unsupported forward inference, and unknown/unclassified autograd rejection without
  editing Compiler source; record it only if used;<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd))</span>
- documentation links, heading anchors, code fences, final newlines, carriage returns, and
  trailing whitespace are checked against all changed Markdown files;
- architecture contract, ADR, focused architecture explanation, Tensor/Compile APIs, glossary,
  capabilities, and Model planning status use the corrected surface consistently;<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [planning glossary](../../../../glossary.md#term-planning))</span>
- the final task inventory is exactly within the 29 allowed paths and contains no Compiler path
  except the one authorized boundary test, and no NN, Data, backend, Runtime, Prepare, Engine,
  Training source, architecture-test source, Gradle, dependency, global-roadmap, completed-task, or
  unrelated worktree change; and<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- the Model master plan has exactly one Ready row before implementation, and task/master status is
  synchronized only after every implementation, documentation, and validation criterion passes.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

## Dependencies

- Completed [Model 0025E](0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
  supplies the exact fixed family, implementation, tests, Javadocs, and fail-closed evidence being
  relocated without semantic change.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Completed [Model 0025](0025-canonical-tensor-producer-outputs.md) supplies the canonical indexed
  output wrappers and package-private factory-atomic multi-output seam.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Accepted ADR 0012 and completed NN 0021A supply the fixed-family and no-region architecture;
  the explicit API-placement correction in this task amends only their Tensor-surface choice.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The repository compatibility precedent permits atomic removal without aliases for newly
  introduced provisional APIs that have no stabilized released contract.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Model 0025E is Complete, while Model 0026 is Draft without a specification. Inserting 0025F
  before 0026 makes this correction the first unfinished Model task and sole Ready Model frontier.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Draft Compiler 0006A depends on the completed fixed Model family and must not begin until this
  correction is Complete, but it remains unchanged and unspecified here.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Follow-up tasks

- Compiler 0006A remains the separate owner of forward inference, final validation, closed
  inventory adoption, and explicit fail-closed BPTT handling. When it becomes the active global
  frontier, its clean-context implementation must consume <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [owner glossary](../../../../glossary.md#term-owner); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [closed inventory glossary](../../../../glossary.md#term-closed-inventory); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>`RecurrentScan`-constructed producers
  without depending on the construction namespace itself.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span>
- Later concrete-backend, Engine, and execution tasks remain responsible for truthful capability,
  prepared-loop lowering, complete runtime valid-length validation, input binding, publication,
  and execution.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication))</span>
- Later [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [runtime](../../../../glossary.md#term-runtime)-valid-length work may call `RecurrentScan` only after that executable path is
  available. Existing static sequence APIs and unrolling remain supported and unchanged.<span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- BPTT, dynamic Shapes, bidirectional fixed-scan composition, packed/compacted execution, broader
  cell variants, and general regions remain separate future decisions.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>
- [Model](../../../../glossary.md#term-model) 0026 remains Draft without a detailed task specification.

## Architecture impact

Expected impact: focused public [API](../../../../glossary.md#term-application-programming-interface-api)-placement amendment only.

The user-approved correction replaces the [architecture contract](../../../../glossary.md#architecture-contract)'s six [Tensor](../../../../glossary.md#tensor) receiver methods
with six static methods on `model.tensor.RecurrentScan`. The implementation must update
`ARCHITECTURE.md`, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012, and the directly affected [module](../../../../glossary.md#term-module)-boundary explanation in the same
change. Fixed semantics, [Model](../../../../glossary.md#term-model) [ownership](../../../../glossary.md#term-ownership-meanings), package [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [graph](../../../../glossary.md#graph) [identity](../../../../glossary.md#term-identity-meanings), [Compiler](../../../../glossary.md#term-compiler)
and [lifecycle](../../../../glossary.md#lifecycle) ownership, and execution boundaries do not change. No architecture-test source
change is required because no [dependency](../../../../glossary.md#term-dependency) or enforcement rule changes; existing architecture
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

- The corrected public [owner](../../../../glossary.md#term-owner) is `io.github.pho001.synaptik.model.tensor.RecurrentScan`. The name
  states the fixed domain concept without the redundant [`Tensor`](../../../../glossary.md#tensor) prefix used by package-private
  helpers and without implying a generic body-bearing scan abstraction.
- The method names are `rnn`, `gru`, and `lstm`. The namespace already supplies the scan context,
  so retaining the `Scan` suffix would be redundant. Bias is represented only by exact overload
  arity.<span data-glossary-navigation="true"> ([Bias glossary](../../../../glossary.md#term-bias); [arity glossary](../../../../glossary.md#term-arity))</span>
- `input` becomes the explicit first argument, matching the helper and producer-input order that
  0025E already implemented. All other arguments retain their exact current order.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [producer glossary](../../../../glossary.md#term-producer))</span>
- The namespace stays in `model.tensor` so it can call the package-private canonical
  derived-output factory seam. Widening that seam or adding a bridge would be a larger and less
  cohesive API change.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Widening glossary](../../../../glossary.md#term-widening); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Both result records stay in `model.tensor`: they are typed public carriers of Tensor wrappers,
  not semantic kinds or operation attributes. Moving them would create unrelated package churn
  and provide no construction or ownership benefit.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- No compatibility alias or deprecation period is selected because the receiver methods were
  introduced only in the immediately preceding Model task, have no downstream repository
  adoption, and are not part of a stabilized released compatibility contract.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [Model glossary](../../../../glossary.md#term-model))</span>
- The coordinator-authorized Compiler test-only correction names the three deferred recurrent
  signature fingerprints explicitly. It does not filter discovered kinds or widen the supported
  inventory, so future missing Compiler adoption remains visible while Compiler 0006A stays
  Draft.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [signature glossary](../../../../glossary.md#term-signature))</span>
- `Tensor.linear(...)` remains unchanged. Linear is a natural single-output algebraic composition;
  this correction is about a domain-heavy explicit-state, multi-input/multi-output protocol, not a
  method-count threshold or blanket facade policy.<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings); [facade glossary](../../../../glossary.md#term-facade))</span>

## Known limitations

- The fixed recurrent-scan metadata remains non-executable until Compiler 0006A and later
  lifecycle/backend work adopt it.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [lifecycle glossary](../../../../glossary.md#lifecycle); [backend glossary](../../../../glossary.md#backend))</span>
- Every backward-capable [compile](../../../../glossary.md#compile) remains unsupported; [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) is not designed here.
- Only the existing fully static [Shape](../../../../glossary.md#shape), [runtime](../../../../glossary.md#term-runtime) `INT64[batch]` valid-prefix, one-direction, dense
  original-time-aligned, three-cell-family contract is represented.
- The low-level namespace is not the ordinary NN user API. Existing NN sequence containers remain
  statically unrolled and accept Java <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>`long[]` lengths until later executable evidence supports a
  deliberate runtime-length addition.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/model_recurrent_scan_api_correction_planning` read the repository
instructions, authoritative and focused architecture contracts, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012, documentation rules and
planning profile, planning guide and roadmap, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plans, completed task
0025E, current recurrent production types, [Tensor](../../../../glossary.md#tensor) and TensorFactory construction seams, affected
Model tests and public-surface locks, Tensor/[Compile](../../../../glossary.md#compile)/Training [API](../../../../glossary.md#term-application-programming-interface-api) documentation, glossary,
capabilities, NN cell/sequence/[module](../../../../glossary.md#term-module)/[Linear](../../../../glossary.md#term-linear-meanings) contracts, Compiler [fail-closed](../../../../glossary.md#term-fail-closed) [dispatch](../../../../glossary.md#term-dispatch), and current
git state.

Planning-stage validation is recorded below. No Java test, [Javadoc](../../../../glossary.md#term-javadoc) generation, or build command
is [run](../../../../glossary.md#term-run) by this planning-only context.

- The repository began at clean `HEAD` `e0ae213f`; committed Model 0025E is `390c2a77` and its
  actual public surface, tests, Javadocs, API documentation, and downstream fail-closed state were
  inspected rather than inferred from planning prose.
- Task ordering checks select 0025F immediately after Complete 0025E and before Draft 0026. The
  Model task directory and master plan contain exactly one Ready task/row, both for 0025F; Model
  0026 and Compiler 0006A have no detailed task specification.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- A read-only Markdown checker validated the Model master plan and this task: 2 files, 43
  generated heading IDs, 207 repository-local file links, no missing target or heading anchor,
  balanced fences, final newlines, no carriage return, and no trailing whitespace.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- `git diff --check` passed for the tracked Model-master change.
  `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0025f-recurrent-scan-expression-namespace-correction.md`
  returned the expected difference status with no whitespace diagnostic for the new task.
- The planning context created or changed exactly the Model master plan and this task. It did not
  touch, stage, revert, reformat, or incorporate CPU/backend, Data, global-roadmap, Compiler, NN,
  Java, test, API, glossary, or architecture implementation/documentation paths.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- The planning context initially scoped 27 paths: 3 production Java paths, 15 Model test paths,
  and 9 architecture/API/glossary/capability/Model-planning paths. The later coordinator-authorized
  Compiler boundary-test correction and Training API omission correction add one path each, for
  the exact final 29-path scope. The 14 existing Tensor
  surface locks affected by 0025E were reconciled to the renamed recurrent test plus 13 existing
  count/name locks; recurrent semantic/signature tests remain unchanged.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [signature glossary](../../../../glossary.md#term-signature))</span>
- The planning review found no unresolved design question. `model.tensor.RecurrentScan` is the
  only placement that keeps Tensor-aware construction public while directly preserving the
  package-private canonical-output factory seam and the operation-to-Tensor dependency boundary.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [operation glossary](../../../../glossary.md#operation); [dependency glossary](../../../../glossary.md#term-dependency))</span>

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
  one-producer provenance, and capture boundaries.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Shapes glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- The 14-suite focused Model selection passed 144 tests with zero failures, errors, or skips.
  Exactly one authoritative `./gradlew :modules:model:test` then passed 1,046 tests in 129 suites
  with zero failures, errors, or skips. Preliminary `./gradlew :modules:model:javadoc` passed.
- The first repository [checkpoint](../../../../glossary.md#term-checkpoint) exposed one stale closed-inventory equality in
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
  future missing adoption, and no Compiler production source or second Compiler test changed.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- `javap` showed exactly six public static methods on `RecurrentScan` and no recurrent receiver on
  `Tensor`. A Java 26 external-package probe compiled and ran all six namespace calls; a separate
  compile failed as required for removed `input.rnnScan(...)` with `cannot find symbol`.
- A temporary Compiler-package probe captured one flat LSTM node with six inputs and three
  outputs, observed unsupported forward inference, and observed exact unknown/unclassified
  autograd rejection for every output/input role without another Tensor-ID allocation.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [node glossary](../../../../glossary.md#node); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd))</span>
- Source/import scans found no production alias, receiver spelling, generic body/region,
  downstream Compiler/NN namespace adoption, bridge, or widened factory seam. The result records,
  recurrent enums, <span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>`TensorFactory`, `TensorLinearExpressions`, and all NN sources remained
  byte-for-byte unchanged.
- Implementation changed exactly 21 authorized paths so far: 3 Model production paths, 15 Model
  test paths, the one coordinator-authorized Compiler boundary test, and the two Model planning
  paths. Concurrent CPU/backend work was neither touched nor incorporated. The clean
  documentation pass owns the remaining eight authorized architecture/API/glossary/capability
  paths, bringing the final maximum to exactly 29.

## Implementation notes

[Canonical](../../../../glossary.md#term-canonical-form) documentation context `/root/model_0025f_docs` independently read the repository and
architecture instructions, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012, documentation rules, General/[API](../../../../glossary.md#term-application-programming-interface-api)-and-[Javadoc](../../../../glossary.md#term-javadoc)/Architecture/
Decision-record/[Planning](../../../../glossary.md#term-planning) profiles, planning guide and roadmap, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) plans, tasks 0025E
and 0025F, the complete implementation and test diff, rendered Javadocs, result carriers, enums,
factory seam, Compiler boundary, and relevant [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [Tensor](../../../../glossary.md#tensor), [Compile](../../../../glossary.md#compile), Training, glossary, capability,
and architecture contracts before finalization.

The review found no executable API, semantic, [provenance](../../../../glossary.md#provenance), [validation](../../../../glossary.md#term-validation)/effect, or [fail-closed](../../../../glossary.md#term-fail-closed)
defect. Removing Javadocs before comparison proved the new public namespace's executable body is
text-identical to the 0025E helper after only class/method visibility and name [normalization](../../../../glossary.md#term-normalization). The
documentation pass changed no executable Java statement or test and therefore reused the frozen
successful Java tiers rather than rerunning them.

The initial seven-document [scope](../../../../glossary.md#term-scope) omitted one current-tense Training API statement that still
claimed six Tensor receiver overloads. The coordinator authorized the minimal eighth
documentation path and final 29-path scope. Only that recurrent paragraph changed; its existing
Java `long[]`, static-unroll, compact-output, and future-adoption boundaries remain intact.

## Completion summary

- Completed changes: public final field-free `RecurrentScan` owns exactly six static biased or
  bias-free <span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>`rnn`, `gru`, and `lstm` constructors with explicit input first; [`Tensor`](../../../../glossary.md#tensor) has no
  recurrent receiver or alias; the old helper is absent; all 0025E semantics, validation,
  effects, descriptors, IDs, canonical outputs, and provenance remain exact. The directly
  affected Compiler test explicitly partitions its exact 128 supported signatures from the exact
  three deferred recurrent signatures and proves the 131-signature union is the complete Model
  inventory while preserving unknown/unclassified, no-allocation failure.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [validation glossary](../../../../glossary.md#term-validation); [descriptors glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance); [Compiler glossary](../../../../glossary.md#term-compiler); [signatures glossary](../../../../glossary.md#term-signature); [Model glossary](../../../../glossary.md#term-model))</span>
- Documentation and Javadocs: finalized `RecurrentScan` as an advanced low-level expression
  namespace, not a layer, module, execution service, registry, or general body. Updated the
  architecture contract, ADR 0012, module boundaries, Tensor/Compile/Training API references,
  glossary, and Model capabilities. The documents direct ordinary users to NN sequences, retain
  their static Java <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [registry glossary](../../../../glossary.md#term-registry); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`long[]` unrolling, defer later NN delegation until Compiler/backend adoption,
  and explain why the single-output last-axis <span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling); [backend glossary](../../../../glossary.md#backend); [axis glossary](../../../../glossary.md#term-axis))</span>`Tensor.linear` composition and state-owning [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
  [`Linear`](../../../../glossary.md#term-linear-meanings) remain unchanged. The result-[carrier](../../../../glossary.md#term-storage-carrier), enum, `TensorFactory`, and Linear Javadocs remain
  accurate and byte-for-byte unchanged; no `model.tensor` package-info file exists or requires a
  new package-level contract.
- Files changed or created: exactly 29 authorized paths—3 Model production, 15 Model tests, 1
  coordinator-authorized Compiler boundary test, 8 architecture/API/glossary/capability
  documents, and 2 Model planning paths. Nine concurrent CPU/backend paths were excluded and
  untouched.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Reused Java evidence: focused Model 144 tests; authoritative Model 1,046 tests in 129 suites;
  focused Compiler boundary 6 tests; replacement root suite including Compiler 209 and Model
  1,046; and architecture tests all passed before executable freeze.
- Final documentation evidence: `./gradlew :modules:model:javadoc` passed after final Javadocs;
  rendered `RecurrentScan`, `Tensor`, both result records, and both recurrent enums were inspected.
  `javap` and reflection proved exactly six public static namespace methods, a public final
  field-free type with one private no-argument constructor, 202 public Tensor methods, no
  recurrent Tensor entry, two unchanged <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`linear` entries, exact result components, and absent old
  helper. A Java 26 external-package probe constructed all six forms; removed receiver use failed
  compilation with <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>`cannot find symbol`.
- Final repository evidence: source/import/forbidden checks found no alias, general body/region,
  downstream Compiler/NN adoption, factory widening, or unrelated Tensor/Linear change. The 10
  changed Markdown files passed 656 link checks, 800 heading/explicit-anchor checks,
  balanced fences, final-newline, CRLF, and trailing-whitespace checks. Exact 29-path inventory,
  tracked and untracked whitespace checks, and <span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>`git diff --check` passed. Model 0025F and its
  master row are Complete; Compiler 0006A and Model 0026 remain Draft without specifications, and
  no Model task is Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Reasoned no-change conclusions: lifecycle, runtime/prepare/backend, training-graph, public and
  Runtime APIs, architecture-test source, Compiler production/other tests, NN source/tests, Data,
  other Model operations, Gradle, dependencies, and the global roadmap require no change because
  this correction only relocates Model expression construction and adds no execution or module
  boundary. Existing successful architecture tests remain applicable and were not repeated.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [graph glossary](../../../../glossary.md#graph); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Unresolved issues: none. Follow-up remains the already Draft Compiler 0006A forward-adoption
  task and later explicitly owned backend, NN runtime-length, Engine, and BPTT work.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>

Status: Complete
