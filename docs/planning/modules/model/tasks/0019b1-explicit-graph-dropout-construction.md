# Task 0019B1: Explicit [Graph](../../../../glossary.md#graph) [Dropout](../../../../glossary.md#dropout--inverted-dropout) Construction

**Editorial reading key (current context, not a status update).**

This record’s completion concerns model construction, not execution or gradients; its deferred-layer
statements describe that historical scope. The caller passes a floating Tensor and opaque
`GraphRngState` to `Tensor.dropout`; the model records input occurrences and output descriptors, not
sampled values. A descriptor specifies an output’s type and shape. One producer describes dropped
output, auxiliary keep mask, and next state, but the public result returns only output and state.
Three wrappers/IDs do not mean three allocated buffers: these storage-free expressions neither
execute nor own result buffers.

For today’s boundary, follow [dropout
construction](../../../../reference/tensor/graph-randomness.md#explicit-state-dropout-construction),
which distinguishes model meaning from the described Metal execution and compiler-saved mask. Model
semantics defines dropout; providers realize supported routes without redefining it, and semantic
responsibility is not storage ownership. Follow [state
advancement](../../../../reference/tensor/graph-randomness.md#state-advancement-branching-and-edge-cases)
for replay limits: a two-by-three input requests six draws even at zero probability. Threading
`nextState` requests the following interval; reusing the original state requests the same interval,
not cross-provider sample identity.

## Status

Complete

## Goal

Add one [backend](../../../../glossary.md#backend)-independent, training-only [dropout](../../../../glossary.md#dropout--inverted-dropout) expression that consumes explicit
[`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), returns the dropped value and next state, and records a non-public [mask](../../../../glossary.md#term-boolean-mask) output for
later [compiler](../../../../glossary.md#term-compiler)-owned backward construction. Construction must expose no generator, perform no
[sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters), mutate no state, and implement no compiler, [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), or backend behavior.

Newcomer [mental model](../../../../glossary.md#term-mental-model):

```java
GraphRngState start = GraphRngState.initial(0x1234L, 0L);

DropoutResult first = activations.dropout(0.1d, start);
DropoutResult replay = activations.dropout(
        0.1d, GraphRngState.initial(0x1234L, 0L));
DropoutResult second = first.output().dropout(0.1d, first.nextState());
```

`first` and `replay` request the same abstract random interval. `second` requests the following
interval by threading `first.nextState()`. Separately constructed expressions remain distinct
[Tensor](../../../../glossary.md#tensor)/[producer](../../../../glossary.md#term-producer) occurrences even when their abstract values replay.

## Rationale and foundation audit

Completed tasks 0018K–0018L and 0019B make this [operation](../../../../glossary.md#operation) representable without an architecture
change. `OperationSignature.fixed(..., 2, 3)` accepts the [occurrence](../../../../glossary.md#term-occurrence); one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) can
retain ordered inputs and three ordered [descriptors](../../../../glossary.md#term-descriptor); `TensorFactory.createDerivedOutputs(...)`
creates one indexed [Tensor](../../../../glossary.md#tensor) wrapper for every output; and the package-private `GraphRngState.tensor()`
and constructor seams provide typed state input and output wrapping without a public Tensor
accessor.

The factory cannot create a descriptor-only output. It necessarily allocates three Tensor
wrappers and three Tensor IDs, one for each output slot. The slot-one [mask](../../../../glossary.md#term-boolean-mask) wrapper is created with
[provenance](../../../../glossary.md#provenance) and is then intentionally not retained by the public result. Its descriptor and output
position remain retained by the shared [producer](../../../../glossary.md#term-producer) reachable from output slots zero and two, which is
the current information boundary a later [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings)/[autograd](../../../../glossary.md#autograd) task needs to represent the
auxiliary value. If implementation instead needs a live mask Tensor reference, sibling-output
[registry](../../../../glossary.md#term-registry), public mask, [graph](../../../../glossary.md#graph)-local [identity](../../../../glossary.md#term-identity-meanings), or another [module](../../../../glossary.md#term-module), stop and report the conflict.

## [Scope](../../../../glossary.md#term-scope)

- Add `DropoutKind.DROPOUT` and [immutable](../../../../glossary.md#term-immutable) `DropoutAttrs(double probability)` in
  `model.operation.random`.
- Add exactly this public [Tensor](../../../../glossary.md#tensor) method and result [carrier](../../../../glossary.md#term-storage-carrier) in `model.tensor`:

  ```java
  public DropoutResult dropout(double probability, GraphRngState state)
  public record DropoutResult(Tensor output, GraphRngState nextState) {}
  ```

- Add one package-private, field-free `TensorDropoutExpressions` construction helper.
- Construct ordered inputs `[input, stateTensor]` and ordered outputs `[output, mask, nextState]`.
- Fix floating eligibility, descriptor propagation, validation order/messages, three-ID behavior,
  state advancement, replay/branching, special-value meaning, and training-only semantics.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation))</span>
- Add focused semantic, expression, API-inventory, signature-inventory, provenance, ID-allocation,
  validation, and immutability tests.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [signature glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalize public Javadocs, Tensor API, Compile API, glossary, capability/master/roadmap status, and
  this task's evidence through a mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters), [eager evaluation](../../../../glossary.md#term-eager-evaluation), storage reads/writes, [host](../../../../glossary.md#term-host) or [device](../../../../glossary.md#term-device) allocation, or
  `TensorRandoms` delegation
- a public mask accessor, public state Tensor accessor, sibling-output lookup, producer registry,
  state accessor, reset, split, fork, fold-in, jump, or copy API<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [Tensor glossary](../../../../glossary.md#tensor); [lookup glossary](../../../../glossary.md#term-lookup); [producer glossary](../../../../glossary.md#term-producer); [registry glossary](../../../../glossary.md#term-registry); [fork glossary](../../../../glossary.md#term-fork-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- a portable PRNG, key schedule, counter-to-bits function, uniform conversion, algorithm enum,
  generator interface, service, registry, framework, or statistical-quality claim<span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- compiler capture, gradient formulas, autograd traversal, backward graph construction, saved-value
  lifetime policy, common-subexpression policy, or inference rewrite implementation<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [backward graph glossary](../../../../glossary.md#backward-graph); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- runtime session or run state, prepare contracts, kernels, lowering, backend routes, conformance
  behavior, numerical execution algorithms, or cross-backend bitwise identity<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [run state glossary](../../../../glossary.md#run-state--runstate); [prepare glossary](../../../../glossary.md#prepare); [kernels glossary](../../../../glossary.md#kernel); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- attention dropout, an attention API change, generic graph-random distributions, Bernoulli API,
  shuffling, permutation, normal/uniform/integral/categorical sampling, or eager random changes<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#masks-padding-and-eager-evaluation))</span><span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [graph glossary](../../../../glossary.md#graph); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution); [permutation glossary](../../../../glossary.md#permutation))</span>
- a `training` flag, [inference](../../../../glossary.md#term-inference) mode, [module](../../../../glossary.md#term-module)/layer object, [parameter](../../../../glossary.md#term-parameter), [optimizer](../../../../glossary.md#term-optimizer), or training workflow
- Gradle, dependencies, architecture documents/tests, another module, or a detailed 0019C-or-later
  task specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Tasks 0018K](0018k-operation-signature-and-construction-hardening.md),
  [0018L](0018l-shared-multi-output-tensor-provenance.md), and
  [0019B](0019b-explicit-graph-rng-state-foundation.md)
- completed [Tensor](../../../../glossary.md#tensor)/TensorFactory/[provenance](../../../../glossary.md#provenance), typed-[scalar](../../../../glossary.md#term-scalar), [eager](../../../../glossary.md#term-eager-evaluation)-random, and [operation](../../../../glossary.md#operation)-family tasks
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>, [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>,
  [Runtime API](../../../../api/runtime-api.md), [Training API](../../../../api/training-api.md), and
  [glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-neutral Tensor expression and Operation semantics. It does not
  sample, execute, retain runtime state, choose a backend, or implement gradients.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Operation glossary](../../../../glossary.md#operation); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [runtime glossary](../../../../glossary.md#term-runtime); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and not [IR](../../../../glossary.md#term-intermediate-representation-ir). [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) remains an opaque wrapper
  around one storage-free Tensor occurrence, not a generator, runtime session, or Tensor metadata.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [occurrence glossary](../../../../glossary.md#term-occurrence); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- One [dropout](../../../../glossary.md#dropout--inverted-dropout) call creates one exact [`TensorProducer`](../../../../glossary.md#producer--tensorproducer); all three wrappers receive indexed
  `TensorProvenance` referencing that producer. The producer retains descriptors, not output
  Tensor references.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- The public output and private next-state Tensor wrapper keep the producer reachable. The mask
  wrapper is intentionally not publicly retained; the producer still describes slot one for later
  compiler-owned auxiliary-value construction.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compiler owns backward graph construction and any decision to retain/materialize the forward
  mask. Prepare/backend/runtime own sampling, state materialization, numerical implementation, and
  execution. Inference bypass is caller/compiler graph composition, not model execution mode.<span data-glossary-navigation="true"> ([backward graph glossary](../../../../glossary.md#backward-graph); [Prepare glossary](../../../../glossary.md#prepare); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [materialization glossary](../../../../glossary.md#materialization); [Inference glossary](../../../../glossary.md#term-inference); [graph glossary](../../../../glossary.md#graph); [model glossary](../../../../glossary.md#term-model))</span>
- No global/default/thread-local RNG, mutable static state, service locator, registry, reflection,
  backend handle, storage, or runtime object may enter these contracts.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [service locator glossary](../../../../glossary.md#term-service-locator); [registry glossary](../../../../glossary.md#term-registry))</span>
- If a public mask, exposed state Tensor, changed factory/provenance architecture, portable PRNG,
  another module, or architecture rule change is required, stop before implementation.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — public Tensor/state/result contracts and package-private
  multi-output construction.
- `io.github.pho001.synaptik.model.operation.random` — explicit graph-random semantics, distinct
  from eager host-data population.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host))</span>
- `io.github.pho001.synaptik.model.datatype` and `.shape` — output [descriptor](../../../../glossary.md#term-descriptor) facts.

Packages added or changed:

- No new package. Extend the established random [operation](../../../../glossary.md#operation) family and [tensor](../../../../glossary.md#tensor) expression package.

Type placement:

- `io.github.pho001.synaptik.model.operation.random.DropoutAttrs` — probability-bearing immutable
  semantic attributes beside graph RNG semantics.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- `io.github.pho001.synaptik.model.operation.random.DropoutKind` — focused one-constant dropout
  operation family.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- `io.github.pho001.synaptik.model.tensor.DropoutResult` — public value-plus-state result beside
  the Tensor and opaque state types it returns.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- `io.github.pho001.synaptik.model.tensor.TensorDropoutExpressions` — package-private construction
  helper with access to the state and factory seams.

Test placement:

- `model.operation.random.DropoutSemanticsTest` — attributes, probability boundary, exact kind,
  signature, and ideal meaning.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- `model.tensor.TensorDropoutExpressionTest` — public API, descriptors, provenance, state wrapping,
  hidden mask slot, validation, identities, and ID effects.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [provenance glossary](../../../../glossary.md#provenance); [mask glossary](../../../../glossary.md#term-boolean-mask); [validation glossary](../../../../glossary.md#term-validation))</span>

## Required contract

### [Operation](../../../../glossary.md#operation) attributes, kind, and [signature](../../../../glossary.md#term-signature)

Create:

```java
public record DropoutAttrs(double probability) implements OperationAttrs {}

public enum DropoutKind implements OperationKind {
    DROPOUT
}
```

`DropoutAttrs` accepts a [finite](../../../../glossary.md#term-finite) [binary64](../../../../glossary.md#floating-point-precision-and-rounding) drop probability numerically in `[0.0, 1.0)`. Positive
and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) are both accepted, retained bit-for-bit by the record, and mean zero [dropout](../../../../glossary.md#dropout--inverted-dropout).
[NaN](../../../../glossary.md#term-nan), either [infinity](../../../../glossary.md#term-infinity), negative finite values, and `1.0` or larger fail with:

```text
probability must be finite and in [0.0, 1.0): <value>
```

`DropoutKind.DROPOUT` accepts only `DropoutAttrs`, exactly two inputs, and exactly three outputs:

```java
OperationSignature.fixed(DropoutAttrs.class, 2, 3)
```

The operation is first-class because one stochastic [occurrence](../../../../glossary.md#term-occurrence) must preserve explicit state,
auxiliary-[mask](../../../../glossary.md#term-boolean-mask), and next-state semantics. It is not a [decomposition](../../../../glossary.md#term-decomposition), [eager](../../../../glossary.md#term-eager-evaluation) sampler, generator,
[kernel](../../../../glossary.md#kernel), or training-mode switch.

### Public [API](../../../../glossary.md#term-application-programming-interface-api) and result

Add to [`Tensor`](../../../../glossary.md#tensor):

```java
public DropoutResult dropout(double probability, GraphRngState state)
```

The receiver is named `input` inside the helper, the [scalar](../../../../glossary.md#term-scalar) [argument](../../../../glossary.md#term-argument) is `probability`, the state
argument is `state`, and `DropoutResult` record components are exactly `output` then `nextState`.
The public record's compact constructor null-checks `output` then `nextState`, with [parameter](../../../../glossary.md#term-parameter)-name
messages, and retains exact references. It is shallowly [immutable](../../../../glossary.md#term-immutable) and uses record value semantics;
it exposes no [mask](../../../../glossary.md#term-boolean-mask), [producer](../../../../glossary.md#term-producer), counter, or state Tensor.

### Ordered [producer](../../../../glossary.md#term-producer) inputs and outputs

One call creates exactly one producer with:

```text
operation: DropoutKind.DROPOUT + DropoutAttrs(probability)
inputs:
  0 input
  1 state.tensor()
outputs:
  0 public dropped output
  1 auxiliary keep mask
  2 next RNG state Tensor
```

Output zero and output two are returned through `DropoutResult`; output two is wrapped with the
existing package-private `new GraphRngState(tensor)` seam. Output one is never exposed. All three
factory-created wrappers have [provenance](../../../../glossary.md#provenance) indices matching their slots and share the exact producer.
The returned output and wrapped next-state [Tensor](../../../../glossary.md#tensor) therefore provide two reachable provenance paths
to the same producer and its slot-one [BOOL](../../../../glossary.md#data-type--datatype) [descriptor](../../../../glossary.md#term-descriptor).

Do not add a public [mask](../../../../glossary.md#term-boolean-mask) or retain it in `DropoutResult`. [Compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings)/[autograd](../../../../glossary.md#autograd) may later create
its own [graph value](../../../../glossary.md#graph-value) for producer slot one and retain that value for backward use. This task defines
the mask as the selected saved forward value but implements neither capture nor a [gradient](../../../../glossary.md#term-gradient) rule.

### Construction and [Tensor](../../../../glossary.md#tensor) IDs

`TensorDropoutExpressions.apply(input, probability, state)` performs all local [validation](../../../../glossary.md#term-validation) and
[descriptor](../../../../glossary.md#term-descriptor) construction, then calls `TensorFactory.createDerivedOutputs(...)` exactly once with
the exact ordered inputs and descriptors. It reads slots zero, one, and two from the [immutable](../../../../glossary.md#term-immutable)
result list, intentionally discards the local [mask](../../../../glossary.md#term-boolean-mask) reference after verifying construction by
position, wraps slot two, and returns slot zero plus that wrapper.

A successful call allocates exactly three fresh Tensor IDs in output order. It creates exactly
three fresh Tensor wrappers, three [provenance](../../../../glossary.md#provenance) values, one [producer](../../../../glossary.md#term-producer), one [operation](../../../../glossary.md#operation), one attributes
value, three descriptors, one public result record, and one [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) wrapper. It allocates
no storage and mutates neither input nor state. No claim is made about adjacency under concurrent
construction; tests that inspect allocation use an isolated factory-ID seam and assert the three
positions in one call.

All known validation occurs before factory delegation and consumes no ID. Identifier exhaustion
uses the existing factory contract: it may consume IDs for earlier output positions, returns no
partial list/result/state wrapper, and throws `tensor identifier space exhausted`. There is no
[rollback](../../../../glossary.md#term-rollback). This [identity](../../../../glossary.md#term-identity-meanings)-allocation effect is not [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)-state advancement or mutation.

### Output [descriptors](../../../../glossary.md#term-descriptor)

For input descriptor `(dataType, shape, layout, requiresGrad)`:

```text
slot 0 output:
  dataType: exact input floating data type
  shape: exact input Shape reference
  layout: unresolved
  requiresGrad: exact input requiresGrad
  label/storage: absent

slot 1 mask:
  dataType: BOOL
  shape: exact input Shape reference
  layout: unresolved
  requiresGrad: false
  label/storage: absent

slot 2 next state:
  dataType: INT64
  shape: Shape.of(2), structurally exact state shape
  layout: unresolved
  requiresGrad: false
  label/storage: absent
```

Only [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16) input are eligible through `DataType.isFloating()`. The output
preserves the exact type and [Shape](../../../../glossary.md#shape) reference and the input's `requiresGrad` flag; it deliberately
does not preserve resolved [layout](../../../../glossary.md#layout), label, or storage. The [mask](../../../../glossary.md#term-boolean-mask) never requires [gradients](../../../../glossary.md#term-gradient). The state
descriptor must satisfy every existing [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) wrapper [invariant](../../../../glossary.md#term-invariant).

### Ideal [dropout](../../../../glossary.md#dropout--inverted-dropout) and special-value meaning

For each logical input [element](../../../../glossary.md#term-element) at [row-major](../../../../glossary.md#term-row-major) logical position `i`, consume exactly one abstract
uniform draw `u_i` in `[0, 1)`. The [BOOL](../../../../glossary.md#data-type--datatype) keep [mask](../../../../glossary.md#term-boolean-mask) is true exactly when
`u_i >= probability`. The ideal result is:

```text
mask[i] ? input[i] / (1 - probability) : +0 of the input data type
```

This is [inverted dropout](../../../../glossary.md#dropout--inverted-dropout). A dropped value is [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) even when the input is [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero),
[NaN](../../../../glossary.md#term-nan), or an [infinity](../../../../glossary.md#term-infinity). A kept positive/negative zero retains its sign under the positive scale; a
kept NaN remains NaN without a [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign promise; and kept positive/[negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero)
retains its infinity sign. [Finite](../../../../glossary.md#term-finite) values use the ideal real scale followed by the result type's
eventual conforming [rounding](../../../../glossary.md#term-rounding). This task selects no finite-[precision](../../../../glossary.md#term-precision) [evaluation](../../../../glossary.md#term-evaluation) order, [fused](../../../../glossary.md#specialization-and-fusion) route,
[underflow](../../../../glossary.md#term-underflow)/[overflow](../../../../glossary.md#term-overflow) detail beyond the type's conforming arithmetic, or bitwise NaN behavior.

At either signed probability zero, every conforming draw keeps the value because draws are in
`[0,1)`. The result is numerical [identity](../../../../glossary.md#term-identity-meanings), including [signed zero](../../../../glossary.md#term-signed-zero), NaN classification, and infinity
sign, but the [operation](../../../../glossary.md#operation) still consumes one draw per element, creates fresh outputs, and advances
state. Do not canonicalize probability zero into a no-op at [model](../../../../glossary.md#term-model) construction.

### State advancement, replay, and dynamic [shapes](../../../../glossary.md#shape)

Let the input's bound logical [element](../../../../glossary.md#term-element) count be `N`. The [operation](../../../../glossary.md#operation) consumes abstract counter
positions `counter` through `counter + N - 1`, retains the unsigned 64-bit key unchanged, and
produces `(key, counter + N mod 2^64)`. The count is independent of probability, input values,
[mask](../../../../glossary.md#term-boolean-mask) outcomes, type, [layout](../../../../glossary.md#layout), and [backend](../../../../glossary.md#backend) vectorization.

- A fully static Shape uses the mathematical product of its static extents as `N`; when
  `Shape.knownElementCount()` is present, that value is exact. A product larger than signed
  `long` is not a model-construction rejection: later execution must advance by that product
  modulo `2^64` without requiring it to fit a signed [host](../../../../glossary.md#term-host) count.
- A dynamic or expression Shape records shape-dependent advancement. Compiler/prepare must bind
  all extents, and the conforming prepared execution advances by the resulting non-negative
  logical element count. Model construction neither binds nor stores <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>`N` in attributes.
- Any bound Shape containing a zero [extent](../../../../glossary.md#term-extent) has `N == 0`: output and mask are empty, the next state
  has the same key/counter value, and slot two is still a distinct Tensor/state occurrence.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- [Scalar](../../../../glossary.md#term-scalar) Shape has `N == 1`.
- Counter arithmetic is unsigned modulo `2^64`; wrap does not fail in [model](../../../../glossary.md#term-model).

Equal input values/Shape, probability, input state value, and the same conforming prepared
implementation path and configuration must replay equal mask, output, and next-state values.
Branching the same state into two [dropout](../../../../glossary.md#dropout--inverted-dropout) occurrences intentionally reuses the same counter
interval. Sequential non-overlap requires explicit `nextState` threading. Public objects are
[immutable](../../../../glossary.md#term-immutable)/[identity](../../../../glossary.md#term-identity-meanings)-bearing expression occurrences and are safe to share for construction, but
sharing does not serialize executions. No cross-backend, cross-route, cross-provider,
cross-version, or bitwise [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) identity is promised until a portable algorithm is selected.
The key/counter advancement result itself is portable semantic state.

### Training, [inference](../../../../glossary.md#term-inference), and backward boundary

[`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) always means training dropout. There is no [model](../../../../glossary.md#term-model)-level `training` [boolean](../../../../glossary.md#data-type--datatype) and no
inference behavior inside the [operation](../../../../glossary.md#operation). A caller or future [compiler](../../../../glossary.md#term-compiler) constructs inference by
using the original input and original state directly, so inference consumes no draw, advances no
state, and creates no dropout [producer](../../../../glossary.md#term-producer). A future compiler owns the rule that backward multiplies
the upstream [gradient](../../../../glossary.md#term-gradient) by the retained keep [mask](../../../../glossary.md#term-boolean-mask) and the same inverted scale; this task neither
adds that rule nor claims that backward [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) exists.

## [Validation](../../../../glossary.md#term-validation) order and side effects

The package-private helper validates in this exact order:

1. non-null `input`, else `NullPointerException("input")`;
2. floating input type, else `IllegalArgumentException` with
   `dropout input data type must be floating: <type>`;
3. construct `DropoutAttrs`, applying the exact probability message above;
4. non-null `state`, else `NullPointerException("state")`;
5. obtain the exact package-private state [Tensor](../../../../glossary.md#tensor);
6. construct output, [mask](../../../../glossary.md#term-boolean-mask), and fixed-state [descriptors](../../../../glossary.md#term-descriptor) and the [operation](../../../../glossary.md#operation);
7. delegate once to `createDerivedOutputs(...)`; and
8. wrap slot two, then construct `DropoutResult` from slots zero and two.

Thus input eligibility precedes probability, and probability precedes state nullity. Every local
failure occurs before [producer](../../../../glossary.md#term-producer)/ID allocation and has no Tensor-ID, state, storage, [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters), or
generator [side effect](../../../../glossary.md#term-side-effect). The already-validated [opaque](../../../../glossary.md#term-opaque) state needs no repeated [invariant](../../../../glossary.md#term-invariant) validation
before factory construction; the existing wrapper validates the generated next-state Tensor after
allocation. Unexpected wrapper failure would expose an implementation defect, not a public input
case, and does not justify [rollback](../../../../glossary.md#term-rollback) or [mutable](../../../../glossary.md#term-mutable) state.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/DropoutAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/DropoutKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/DropoutResult.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDropoutExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/GraphRngState.java`

Expected [model](../../../../glossary.md#term-model) tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/random/DropoutSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDropoutExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Mandatory inspected inventories/seams, changed only if a listed acceptance criterion proves a
stale contract that cannot be covered in the focused files above:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java` and
  `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java`,
  `TensorProvenance.java`, plus
  `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProducerTest.java` and
  `TensorProvenanceTest.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngStateAttrs.java`,
  `GraphRngKind.java`, and the existing `GraphRngStateSemanticsTest.java` plus
  `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/GraphRngStateTest.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java`,
  `OperationKind.java`, `OperationAttrs.java`, and `OperationSignature.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java`,
  `TensorRandoms.java`, plus `model/datatype/DataType.java` and `model/shape/Shape.java`
- `docs/api/runtime-api.md` and `docs/api/training-api.md`, for reasoned no-change conclusions

## Maximum [scope](../../../../glossary.md#term-scope)

This task may modify or create exactly the 19 expected paths above: six production, six test, and
seven documentation/[planning](../../../../glossary.md#term-planning) paths. The sixth test path is the authorized stale global [Tensor](../../../../glossary.md#tensor)
public-method inventory in `TensorMatmulExpressionTest`; its only permitted change is the expected
count from 162 to 163, with every [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-specific assertion preserved. Do not modify a mandatory
inspected seam merely to consume scope.

No Java path outside `modules/model`, [Gradle](../../../../glossary.md#term-gradle)/[dependency](../../../../glossary.md#term-dependency) file, architecture/focused-architecture
document, [architecture test](../../../../glossary.md#term-architecture-test), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)/[integration test](../../../../glossary.md#term-integration-test), another [module](../../../../glossary.md#term-module), or later task
specification may change. If a live auxiliary Tensor, [compiler](../../../../glossary.md#term-compiler) [API](../../../../glossary.md#term-application-programming-interface-api), another package concept, or
more than 19 paths is required, stop and propose a focused follow-up or architecture decision.

## [Javadoc](../../../../glossary.md#term-javadoc) and explanatory documentation requirements

- Document `DropoutAttrs`, `DropoutKind`, `DropoutResult`, the helper contract, and the Tensor
  method with probability, nullability, descriptors, three outputs, state advancement, ID effects,
  special values, replay boundary, training/inference meaning, and deferred execution/gradient
  ownership.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [nullability glossary](../../../../glossary.md#term-nullability); [descriptors glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Update [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) Javadoc from future tense to current state threading without adding a
  public accessor or algorithm promise.<span data-glossary-navigation="true"> ([state threading glossary](../../../../glossary.md#state-threading))</span>
- Tensor API must include the runnable construction example above, output-slot table, p=0 and empty
  behavior, branching versus threading, and honest current-versus-planned execution boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Compile API must explain that later capture/autograd can observe producer slot one even though
  no public mask Tensor is returned; it must not claim compiler capture or gradients are current.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [producer glossary](../../../../glossary.md#term-producer); [mask glossary](../../../../glossary.md#term-boolean-mask); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Glossary must define dropout/inverted dropout, auxiliary mask, state threading, and bounded
  replay at first use without duplicating the complete API guide.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [inverted dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [auxiliary mask glossary](../../../../glossary.md#auxiliary-mask); [bounded replay glossary](../../../../glossary.md#bounded-replay))</span>
- Runtime API and Training API are expected no-change: no executable/runtime or training-extension
  API is added. Record that conclusion. Architecture docs, conformance/integration guides, and
  other modules are also expected no-change because ownership/boundaries do not change.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact public [API](../../../../glossary.md#term-application-programming-interface-api) is `Tensor.dropout(double probability, GraphRngState state)` returning public
  `DropoutResult(Tensor output, GraphRngState nextState)` in `model.tensor`; no public mask or state
  Tensor accessor exists.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `DropoutAttrs` accepts and retains both signed zeros and every other [finite](../../../../glossary.md#term-finite) value in `[0,1)`, and
  rejects all other values with the exact selected message.
- `DropoutKind.DROPOUT` has exactly `OperationSignature.fixed(DropoutAttrs.class, 2, 3)` and no
  backend/execution knowledge.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- One [producer](../../../../glossary.md#term-producer) has exact inputs `[input, stateTensor]`, exact [descriptors](../../../../glossary.md#term-descriptor) `[output, mask,
  nextState]`, and all three factory wrappers have indices zero, one, and two with the same exact
  producer reference.
- Exactly three Tensor wrappers/IDs are allocated successfully. The mask wrapper is not exposed or
  retained in the public result, while its BOOL descriptor and slot remain producer-visible.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Output/mask/state descriptors, exact Shape references, gradient flags, absent layout/label/
  storage, and state-wrapper invariants match this specification.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout))</span>
- Floating inputs succeed; INT32, INT64, and BOOL fail before probability/state/ID work according
  to the selected order. Null/input/probability/state messages and no-ID behavior are exact.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- Ideal keep rule, inverted scaling, signed zero, NaN, infinity, p=0, scalar, empty, static,
  dynamic/bound-count, modulo advancement, replay, branching, threading, identity, and
  non-mutation semantics are documented and tested at the representation level possible now.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- No sampling, storage access, TensorRandoms call, random algorithm/framework, hidden global state,
  execution, gradient rule, compiler change, runtime state, backend route, or attention dropout is
  added or claimed.<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend route glossary](../../../../glossary.md#backend-route); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Tensor public-method count/name inventory, global operation-signature inventory, result/kind/attrs
  shapes, factory/provenance behavior, and GraphRngState public surface remain locked by tests.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance); [GraphRngState glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- One final `:modules:model:test` run occurs after executable Java stabilizes. The separate docs
  pass reuses that evidence and does not repeat it unless executable Java changes.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The documentation-focused pass finalizes all affected Javadocs/docs/glossary/planning evidence,
  runs Javadoc and documentation checks, and records no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Task/master/roadmap show 0019B1 Complete with tasks 0019 through 0019B. Tasks 0019C–0019E and
  later stay Draft without detailed specifications; no model task is Ready or Review needed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Ready glossary](../../../../glossary.md#term-ready))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During development, [run](../../../../glossary.md#term-run) focused [dropout](../../../../glossary.md#dropout--inverted-dropout) semantic/expression tests and directly affected
[signature](../../../../glossary.md#term-signature)/[API](../../../../glossary.md#term-application-programming-interface-api)/factory/[provenance](../../../../glossary.md#provenance) tests as needed. After executable Java stabilizes, run exactly one
final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

The first final run failed solely because the previously unlisted stale
`TensorMatmulExpressionTest` global count expected 162 methods. The explicitly authorized [scope](../../../../glossary.md#term-scope)
repair adds that nineteenth path, changes only the count to 163, and permits one necessary final
model-suite rerun. Record this reason and do not perform another model-suite rerun.

The separate clean-context documentation pass then reuses that result and runs after final
[Javadoc](../../../../glossary.md#term-javadoc)/documentation edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It must also [compile](../../../../glossary.md#compile)/run the newcomer example against current model classes; validate affected
local links and anchors, balanced fences, final newlines, trailing whitespace, exact path scope,
status/[dependency](../../../../glossary.md#term-dependency) coherence, no [Ready](../../../../glossary.md#term-ready) or Review-needed frontier after [checkpoint](../../../../glossary.md#term-checkpoint) completion, and
absence of a 0019C-or-later task file.

This task is the [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/dropout capability checkpoint because it closes the explicit-state plus first
state-consuming [operation](../../../../glossary.md#operation) sequence and exercises the shared multi-output foundation in production.
After both passes, run repository-wide:

```bash
./gradlew test
```

Also run affected architecture tests only if the final diff changes a dependency or boundary; such
a change is not expected and otherwise requires stopping. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and integration
tests remain deferred until an executable backend/[runtime](../../../../glossary.md#term-runtime) capability exists.

## Dependencies

- Task 0018K: [operation signature](../../../../glossary.md#operation-signature) and construction hardening — Complete.
- Task 0018L: shared multi-output [Tensor](../../../../glossary.md#tensor) [provenance](../../../../glossary.md#provenance) and factory seam — Complete.
- Tasks 0018N and 0018S: typed [scalar](../../../../glossary.md#term-scalar)/factory surface boundaries — Complete.
- Task 0019B: explicit [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) foundation and package-private wrapping seam — Complete.
- Existing floating DataType, Shape, TensorDescriptor, Tensor, and producer/provenance contracts —
  Complete.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer))</span>

## Follow-up tasks

- 0019C — sorting, argsort, and true multi-output top-K; remains Draft without a detailed spec.
- 0019D — public [linear](../../../../glossary.md#term-linear-meanings) convenience; remains Draft without a detailed spec.
- 0019E — [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention) without [dropout](../../../../glossary.md#dropout--inverted-dropout); remains Draft without a detailed spec.
- A later [compiler](../../../../glossary.md#term-compiler) task owns dropout [capture](../../../../glossary.md#term-capture-meanings), auxiliary-[mask](../../../../glossary.md#term-boolean-mask) retention, and backward construction.
- A later backend/runtime capability owns sampling and execution. A portable PRNG requires a
  separate explicit semantic decision before cross-backend bitwise replay can be promised.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Any later attention-dropout extension must consume/produce explicit state and is not part of
  initial 0019E.

## Architecture impact

Expected impact: None.

This task uses existing [model](../../../../glossary.md#term-model)-owned [Tensor](../../../../glossary.md#tensor)/[Operation](../../../../glossary.md#operation) semantics, explicit state, and multi-output
[provenance](../../../../glossary.md#provenance). It changes no [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), [compiler](../../../../glossary.md#term-compiler) boundary, or
[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend) responsibility. If implementation reveals otherwise, stop and report the
exact conflict before changing architecture documents or another module.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, completed tasks 0018K, 0018L, and 0019B, and task 0019B1 in full. Inspect every affected
source/test and mandatory inventory named by 0019B1.

Implement task 0019B1 exactly as specified, only in modules/model and the authorized paths. Stop
on an architecture, cross-module, live-mask, factory/provenance, or maximum-scope conflict. Do not
implement sampling, gradients, compiler/runtime/backend behavior, or another task. Run focused
tests while developing and exactly one final model suite after executable Java stabilizes.

Then hand the actual diff and recorded Java-test evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect the
final contracts, finalize Javadocs, Tensor/Compile API, glossary, planning/status/evidence and
no-change conclusions, run model Javadoc and documentation validation, and reuse the successful
Java evidence unless it changes executable behavior. Run the recorded checkpoint only after both
passes. Do not mark Complete until all acceptance criteria pass. Do not commit or push.
```

## Separate documentation [handoff](../../../../glossary.md#term-handoff)

The implementation agent must provide the documentation agent: this task file; the actual diff;
the exact final [model](../../../../glossary.md#term-model)-test command/result; affected public [API](../../../../glossary.md#term-application-programming-interface-api) and three-output/state semantics;
the no-algorithm, no-execution, [compiler](../../../../glossary.md#term-compiler)-owned-backward, and architecture boundaries; expected
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API and glossary changes; expected [Runtime](../../../../glossary.md#term-runtime)/Training/architecture no-change reviews;
and all [validation](../../../../glossary.md#term-validation) commands above. The docs agent must report its clean context ID and whether
executable Java changed after the reused test evidence.

## Local decisions

- The existing wrapper-per-descriptor factory seam is retained. Dropout creates all three indexed
  wrappers and IDs, discards the mask wrapper from its public result, and relies on the shared
  producer's slot-one descriptor for later compiler capture. No descriptor-only output or sibling
  registry is introduced.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask); [producer glossary](../../../../glossary.md#term-producer); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [registry glossary](../../../../glossary.md#term-registry))</span>
- The public result remains exactly `(output, nextState)`. The auxiliary keep mask is deliberately
  compiler-facing metadata rather than public Tensor API.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) probability is preserved by `DropoutAttrs` record semantics. Probability zero is not
  canonicalized away because eventual execution still consumes the operation's logical draw count.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## Known limitations

- Model construction represents dropout but cannot execute it. No current compiler captures the
  auxiliary slot, no backend samples it, and no gradient rule consumes it.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Replay of sampled values is bounded to the same conforming prepared implementation path and
  configuration because no portable PRNG/bitstream is selected.<span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- The factory creates a short-lived mask Tensor wrapper because current multi-output construction
  is wrapper-per-slot. Only its producer descriptor/position remains reachable publicly.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) evidence:

- Clean planning context inspected the authoritative/focused architecture, documentation/planning
  rules and profiles, model plans, completed foundation/random/state tasks, current source/tests,
  producer/factory/provenance seams, global signature/Tensor API inventories, APIs, and glossary.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [signature glossary](../../../../glossary.md#term-signature); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `TensorFactory.createDerivedOutputs(...)` was verified to validate one producer before allocating
  and then create one Tensor/provenance per descriptor. Therefore three dropout outputs require
  exactly three wrappers/IDs; a descriptor-only mask is not supported.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) was verified to provide package-private exact Tensor access and validated
  package-private wrapping, so no public accessor or cross-module change is required.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

Implementation evidence:

- The implementation context ran focused tests for the two dropout test classes, broader focused
  tests for ten affected classes, and focused final-refinement tests; each run reported<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  `BUILD SUCCESSFUL`.
- Its first final `./gradlew :modules:model:test` run reached 799 tests and failed only because the
  unlisted stale `TensorMatmulExpressionTest` inventory expected 162 public Tensor methods. The
  user authorized the exact 19-path repair, whose only change in that test is `162` to `163`.
- After that repair, the implementation context reran `./gradlew :modules:model:test`: `BUILD
  SUCCESSFUL` in 1s; 3 actionable tasks, 2 executed and 1 up-to-date. Executable production Java
  did not change after this successful run.

Documentation-agent evidence:

- Clean context `/root/task_0019b1_implementation/task_0019b1_docs` independently inspected the
  architecture contract and focused pages, documentation rules and General/API-Javadoc/Planning/
  Example profiles, planning guide/roadmap/capabilities/master plan, tasks 0018K/0018L/0019B/
  0019B1, the actual production and test diff, construction seams and inventories, Tensor/Compile/
  Runtime/Training APIs, and glossary.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The pass finalized all six production Javadocs/comments plus Tensor API, Compile API, glossary,
  capability/master/roadmap status and this evidence without changing executable Java.
- `./gradlew :modules:model:javadoc` passed after the final [Javadoc](../../../../glossary.md#term-javadoc) edit: `BUILD SUCCESSFUL` in 1s;
  2 actionable tasks, both executed. Generated pages for [`Tensor`](../../../../glossary.md#tensor), `GraphRngState`, `DropoutAttrs`,
  `DropoutKind`, and `DropoutResult` were inspected for the new contracts and resolved links.
- The documented `DropoutConstructionExample` compiled against current model classes and ran with
  exact output `2`, `3`, [`BOOL`](../../../../glossary.md#data-type--datatype), `true`, `false`, `false`.
- A targeted checker resolved every local Markdown file target and heading anchor in the seven
  affected documentation/planning files. Fence balance, final newlines, trailing whitespace,
  status/dependency coherence, absent 0019C-or-later specs, production imports/claims, and exact
  19-path scope were also checked. Final <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- `javap` confirmed the public `dropout(double, GraphRngState)`, `DropoutResult.output()` and
  `nextState()`, `GraphRngState.initial(long, long)`, `DropoutKind.DROPOUT`, and
  `DropoutAttrs.probability()` surface. Automated implementation tests remain the authoritative
  inventories for the full Tensor method count and operation signatures.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Runtime API and Training API remain unchanged because this task adds model construction only:
  there is no prepared execution, sampling, training-session, optimizer, or gradient API.
  Architecture pages, ADRs, and architecture tests remain unchanged because module ownership,
  dependency direction, and lifecycle boundaries are preserved. Backend conformance and
  integration tests remain unchanged because no backend or end-to-end execution behavior exists.
  Gradle/dependencies and other modules remain unchanged because the implementation uses only
  existing model contracts and Java dependencies.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

[Checkpoint](../../../../glossary.md#term-checkpoint) evidence:

- After the implementation and documentation passes, the coordinator ran `./gradlew test`:
  `BUILD SUCCESSFUL` in 1s; 36 actionable tasks, 2 executed and 34 up-to-date; configuration cache
  reused. No file or executable Java changed after the successful documentation validation and
  before this checkpoint.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- The completion sync changed only the four already-authorized planning paths. A final lightweight
  pass confirmed valid local links/anchors in those files, balanced fences, final newlines, no
  trailing whitespace, exactly 19 overall paths, 0019B1 Complete, 0019C–0019E Draft, no model
  Ready/Review-needed row, no later specification file, clean status scope, and a passing<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
  `git diff --check`. Java tests, Javadoc, the example, and the repository checkpoint were not
  repeated.

## Implementation notes

- Added `DropoutAttrs`, `DropoutKind.DROPOUT`, public `DropoutResult`, the package-private
  construction helper, and `Tensor.dropout(double, GraphRngState)` inside `modules/model`.
- One successful call records exact ordered inputs `[input, stateTensor]`, creates output/mask/state
  descriptors and three indexed wrappers under one producer, returns slots zero and two, and leaves
  slot one's auxiliary BOOL descriptor producer-visible without a public mask.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Added focused semantic/expression coverage and synchronized the global signature and Tensor
  public-surface inventories, including the authorized stale MATMUL inventory count repair.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Finalized current API documentation, glossary terminology, Javadocs, and planning status while
  retaining the explicit no-execution/no-gradient/no-algorithm boundaries.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [gradient glossary](../../../../glossary.md#term-gradient))</span>

## Completion summary

- Completed changes: explicit-state training-dropout model construction, three-output provenance,
  public value/next-state result, validation and inventory tests, Javadocs, API/glossary coverage,
  and synchronized planning evidence.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [model glossary](../../../../glossary.md#term-model); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly the six production, six test, and seven documentation/planning
  paths listed under Affected files.
- Tests and validation: final model suite passed in the implementation context; documentation
  validation passed in the clean documentation context; the coordinator's repository-wide
  `./gradlew test` [checkpoint](../../../../glossary.md#term-checkpoint) passed with 36 actionable tasks.
- Documentation-agent review: clean context
  `/root/task_0019b1_implementation/task_0019b1_docs`; executable Java was not changed.
- Documentation impact: Tensor and Compile API now describe current construction and the planned
  capture/autograd boundary. Runtime and Training API remain unchanged because no executable or
  training-extension API was added.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: all six authorized production paths reviewed and finalized.
- Glossary impact: added dropout/inverted dropout, auxiliary mask, state threading, and bounded
  replay distinctions.<span data-glossary-navigation="true"> ([inverted dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [auxiliary mask glossary](../../../../glossary.md#auxiliary-mask); [state threading glossary](../../../../glossary.md#state-threading); [bounded replay glossary](../../../../glossary.md#bounded-replay))</span>
- Unresolved issues: None.
- Follow-up required: None. Tasks 0019C–0019E remain Draft without detailed specifications; the
  next frontier requires a separate planning step.

Status: Complete
