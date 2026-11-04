# Task 0019B: Explicit [Graph RNG State](../../../../glossary.md#graph-rng-state--graphrngstate) Foundation

**Editorial reading key (current context, not a status update).**

This record covers the state foundation only; its dropout examples are prospective, not APIs
delivered by this task. The current [explicit-state dropout
description](../../../../reference/tensor/graph-randomness.md#explicit-state-dropout-construction)
explains the later caller/callee boundary: the caller supplies a floating Tensor, drop probability,
and opaque state; Tensor construction records value, auxiliary mask, and next-state output
descriptions, not sampled values or allocated result buffers.

A key names a caller-selected stream/domain; a counter names its next logical sample position. Both
preserve unsigned 64-bit patterns carried by Java `long`. Equal words request the same abstract
position, not equal object identities. For example, six-element dropout starting at counter zero
requests six positions; threading its next state starts at six, whereas reusing the original state
repeats the interval. See [state advancement and replay
boundaries](../../../../reference/tensor/graph-randomness.md#state-advancement-branching-and-edge-cases)
for empty inputs and bounded replay. Model semantics defines these meanings; providers realize
supported routes, not redefine them. Semantic responsibility is not storage ownership: the
storage-free state expression neither executes nor owns result buffers.

## Status

Complete

## Goal

Add the smallest public, [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) contract for explicit random-number-generator
([RNG](../../../../glossary.md#graph-rng-state--graphrngstate)) state in [Tensor](../../../../glossary.md#tensor) expression [graphs](../../../../glossary.md#graph). A caller must be able to create one state from explicit
64-bit key and counter bit patterns, thread that state through later state-consuming operations,
and replay the same graph from the same explicit state without a hidden process-global generator.

This task establishes the state value and its zero-input Tensor [producer](../../../../glossary.md#term-producer) only. Draft follow-up
0019B1 owns [dropout](../../../../glossary.md#dropout--inverted-dropout), state advancement by output [element](../../../../glossary.md#term-element) count, the genuine multi-output
[occurrence](../../../../glossary.md#term-occurrence), and its public result [carrier](../../../../glossary.md#term-storage-carrier). Separating those capabilities keeps the reusable state
contract reviewable before probability, masking, scaling, dynamic-count advancement, and backward
requirements are added.

Newcomer [mental model](../../../../glossary.md#term-mental-model):

```java
GraphRngState start = GraphRngState.initial(0x1234L, 0L);

// Draft task 0019B1 will make this state-threading sequence available:
// DropoutResult first = activations.dropout(0.1d, start);
// DropoutResult replay = activations.dropout(0.1d,
//         GraphRngState.initial(0x1234L, 0L));
// DropoutResult second = first.output().dropout(0.1d, first.nextState());
```

The equal key/counter pair requests replay from the same abstract stream position. The threaded
`nextState` requests the following non-overlapping counter interval. This foundation does not yet
[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), advance state, or promise a random bitstream.

## Rationale and split decision

The former broad 0019B combines two independently reviewable contracts:

1. an opaque public graph-state value, its exact key/counter representation, explicit creation,
   and Tensor provenance; and<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [graph glossary](../../../../glossary.md#graph); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
2. dropout probability/mask/scaling semantics, three producer outputs, dynamic state advancement,
   and a public value-plus-state result.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask); [producer glossary](../../../../glossary.md#term-producer))</span>

The combined change would normally require more than the [planning](../../../../glossary.md#term-planning) guide's 12–18-path guardrail
and would make the reusable state representation depend on one [distribution](../../../../glossary.md#term-distribution) [operation](../../../../glossary.md#operation). Task
0019B therefore remains the stable ID for the first [Ready](../../../../glossary.md#term-ready) [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)-state foundation. Task 0019B1 is a
broad Draft follow-up without a detailed specification. Established tasks 0019C–0019E keep their
IDs, order, and scopes.

## [Scope](../../../../glossary.md#term-scope)

- Add one public final [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) value in `model.tensor`.
- Add one public [immutable](../../../../glossary.md#term-immutable) `GraphRngStateAttrs(long key, long counter)` record and one
  `GraphRngKind.INITIAL_STATE` [operation](../../../../glossary.md#operation) kind in `model.operation.random`.
- Add exactly one public creation method:

  ```java
  public static GraphRngState initial(long key, long counter)
  ```

- Represent the state internally as one storage-free, unlabeled Tensor expression with exact
  descriptor <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>[`INT64`](../../../../glossary.md#data-type--datatype), `Shape.of(2)`, unresolved [layout](../../../../glossary.md#layout), and `requiresGrad == false`.
- Make `INITIAL_STATE` a zero-input, one-output producer. The state Tensor has output index zero
  in ordinary `TensorProvenance` and retains the exact [producer](../../../../glossary.md#term-producer) [descriptor](../../../../glossary.md#term-descriptor) reference.
- Interpret `key` and `counter` as unsigned 64-bit bit patterns carried by Java `long`; every bit
  pattern is valid and no signed ordering is assigned.
- Define the counter as the next abstract logical sample position and the key as caller-selected
  stream/domain identity. State-consuming operations keep the key and advance the counter modulo<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
  `2^64` by their specified logical draw count.
- Define public immutability, identity equality, ownership, portability, replay, branching, and
  serialization boundaries precisely without selecting a sampling algorithm.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [serialization glossary](../../../../glossary.md#term-serialization); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- Add focused semantic and public-state tests, and include existing global producer/provenance,
  factory-derived-output, operation-signature, and public Tensor API inventory tests in the
  authorized review scope.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [signature glossary](../../../../glossary.md#term-signature); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize public Javadocs, Tensor API, Compile API, glossary, capabilities, task evidence, master
  plan, and roadmap through a mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- dropout, Bernoulli masks, probability validation, inverted scaling, training/inference mode,
  mask exposure, or state advancement performed by an operation<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution); [masks glossary](../../../../glossary.md#term-boolean-mask); [validation glossary](../../../../glossary.md#term-validation); [inference glossary](../../../../glossary.md#term-inference); [operation glossary](../../../../glossary.md#operation))</span>
- a generic graph-random sampling API, random bits, uniform/normal/integral distributions,
  categorical sampling, shuffling, permutation, or attention dropout<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [permutation glossary](../../../../glossary.md#permutation))</span>
- a portable PRNG algorithm, key derivation, seed expansion, bitstream, floating conversion,
  statistical-quality guarantee, backend-independent numerical replay, or conformance tolerance<span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [seed glossary](../../../../glossary.md#term-seed); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- split/fold-in/jump/reset/copy/fork APIs, a distribution enum, pluggable algorithms, generator
  interfaces, factories, registries, services, dependency injection, or reflective discovery<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [distribution glossary](../../../../glossary.md#term-distribution); [dependency injection glossary](../../../../glossary.md#term-dependency-injection))</span>
- `RandomGenerator`, `TensorRandoms`, or [eager](../../../../glossary.md#term-eager-evaluation) [host](../../../../glossary.md#term-host)-data construction changes
- exposing the internal state Tensor as a general numerical Tensor, accepting an arbitrary
  caller-supplied `INT64[2]` [Tensor](../../../../glossary.md#tensor) as state, or adding state [metadata](../../../../glossary.md#term-metadata) to `TensorDescriptor`
- storage allocation, host-state mutation, device residency, runtime session state, prepared
  state, physical buffers, execution, kernels, lowering, or backend routes<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency); [runtime glossary](../../../../glossary.md#term-runtime); [buffers glossary](../../../../glossary.md#term-buffer); [kernels glossary](../../../../glossary.md#kernel); [lowering glossary](../../../../glossary.md#lowering))</span>
- graph capture, compiler serialization, common-subexpression policy, constant folding, gradient
  rules, autograd traversal, or backward graph construction<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [serialization glossary](../../../../glossary.md#term-serialization); [constant folding glossary](../../../../glossary.md#term-constant-folding); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [backward graph glossary](../../../../glossary.md#backward-graph))</span>
- Gradle, dependencies, architecture documents/tests, another module, or another detailed task
  specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership, Tensor as
  public mutable API state rather than IR, compiler-owned backward graph construction, and
  runtime/prepare/backend state boundaries<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [compiler glossary](../../../../glossary.md#term-compiler); [backward graph glossary](../../../../glossary.md#backward-graph); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Tasks 0018K](0018k-operation-signature-and-construction-hardening.md) and
  [0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018S](0018s-tensor-factory-surface-cleanup.md)
- completed [eager](../../../../glossary.md#term-eager-evaluation)-random tasks 0012F–0012I
- completed tasks 0019–0019A2
- [Tensor API](../../../../api/tensor-api.md), [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>,
  [Runtime API](../../../../api/runtime-api.md), [Training API](../../../../api/training-api.md),
  and [glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` may own this immutable backend-neutral semantic value and its public Tensor
  expression construction. The architecture already permits model-owned Tensor and Operation
  semantics; no architecture update is required.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [model glossary](../../../../glossary.md#term-model); [Operation glossary](../../../../glossary.md#operation))</span>
- [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) is a public model value that privately retains one Tensor expression. It is not
  Tensor metadata, a graph node/value, a runtime session, a generator service, or storage owner.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [runtime glossary](../../../../glossary.md#term-runtime); [owner glossary](../../../../glossary.md#term-owner))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state and not IR. The private retained Tensor supplies
  expression identity/provenance only; <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>`GraphRngState` does not expose storage attachment,
  publication, runtime residency, or graph-local IDs.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [residency glossary](../../../../glossary.md#residency))</span>
- The state Tensor participates normally in [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) input/output positions. This task's
  initializer is zero-input and one-output. Later state-consuming operations may receive the
  exact privately retained Tensor through package-private same-package mechanics.
- State construction must use the existing package-private derived Tensor construction seam. It
  must not directly allocate a [`TensorId`](../../../../glossary.md#tensorid), construct `TensorProducer`/`TensorProvenance`, attach
  storage, or add another factory.
- Compiler owns capture, serialization policy, CSE, gradient rules, and backward graph
  construction. Prepare/backend/runtime own materialization, sampling implementation, executable
  state, and execution.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [serialization glossary](../../../../glossary.md#term-serialization); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [Prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- No hidden process-global/default/thread-local generator, service locator, registry, reflection,
  backend handle, runtime state, storage, or mutable static state may enter model.<span data-glossary-navigation="true"> ([service locator glossary](../../../../glossary.md#term-service-locator); [registry glossary](../../../../glossary.md#term-registry))</span>
- If implementation requires exposing the retained Tensor publicly, accepting arbitrary Tensor
  state, changing TensorDescriptor, adding runtime/backend state, or selecting a portable PRNG
  algorithm, stop and update planning or request an architecture decision before editing.<span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [planning glossary](../../../../glossary.md#term-planning))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor expression values and the existing
  package-private derived-construction seam.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- `io.github.pho001.synaptik.model.operation` — owns generic [operation](../../../../glossary.md#operation) contracts.
- `io.github.pho001.synaptik.model.datatype` and `.shape` — own the fixed state [descriptor](../../../../glossary.md#term-descriptor).

Packages added or changed:

- Add `io.github.pho001.synaptik.model.operation.random` for explicit graph-random operation
  semantics. This is distinct from eager <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>`model.tensor.TensorRandoms` [host](../../../../glossary.md#term-host)-data population.
- Change the existing `model.tensor` package with one focused public state type and only the
  minimum Javadoc/inventory integration.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Type placement:

- `io.github.pho001.synaptik.model.tensor.GraphRngState` — public opaque graph-state value beside
  Tensor because it privately wraps one Tensor expression and must use the package-private
  construction seam.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- `io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs` — immutable raw key and
  counter semantic attributes for the initializer operation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.model.operation.random.GraphRngKind` — focused family containing only
  `INITIAL_STATE` in this task.

Test placement:

- `io.github.pho001.synaptik.model.operation.random.GraphRngStateSemanticsTest` — attribute/kind,
  unsigned-bit-pattern, signature, and API-shape contract.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>
- `io.github.pho001.synaptik.model.tensor.GraphRngStateTest` — public construction, descriptor,
  provenance, identity, immutability, no-storage, no-hidden-state, and ID-side-effect contract.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Required contract

### Public state type

Create:

```java
public final class GraphRngState {
    public static GraphRngState initial(long key, long counter);
}
```

The type has exactly one private final instance field containing the state [Tensor](../../../../glossary.md#tensor), one
package-private constructor used only by validated same-package expression construction, and no
public constructor. The constructor validates in this order: non-null Tensor; exact [`INT64`](../../../../glossary.md#data-type--datatype) [data
type](../../../../glossary.md#data-type--datatype); structural `Shape.of(2)` equality; unresolved [layout](../../../../glossary.md#layout); false [gradient](../../../../glossary.md#term-gradient) eligibility; absent
label; absent [host storage](../../../../glossary.md#host-storage); and present [provenance](../../../../glossary.md#provenance). Failures identify the first violated state
[invariant](../../../../glossary.md#term-invariant) and allocate no Tensor ID. It retains the exact Tensor reference after [validation](../../../../glossary.md#term-validation).

Expose exactly one package-private `tensor()` accessor returning that reference so the later
[dropout](../../../../glossary.md#dropout--inverted-dropout) helper can use it as an ordered [producer](../../../../glossary.md#term-producer) input and wrap the next-state output. There is no
public Tensor accessor, key/counter accessor, mutation method, storage method, split method, or
generic public wrapping method. The focused same-package test locks the constructor/accessor
visibility and validation order.

[`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) deliberately inherits ordinary object-[identity](../../../../glossary.md#term-identity-meanings) `equals` and [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests). Two calls
with equal key/counter attributes are distinct public state-expression occurrences with distinct
Tensor and producer identities, even though they request equivalent abstract stream positions.
The class is shallowly [immutable](../../../../glossary.md#term-immutable): its retained Tensor reference never changes and is not exposed,
so callers cannot use this type to mutate its storage association. It owns no [mutable](../../../../glossary.md#term-mutable) generator or
execution state.

### Key and counter representation

Create:

```java
public record GraphRngStateAttrs(long key, long counter)
        implements OperationAttrs {}
```

Every `long` bit pattern is valid. Interpret each component as an unsigned 64-bit word; Java's
signed decimal rendering and comparisons have no semantic meaning. Record equality and hashing
compare the exact 64 bits of both components. The record is [immutable](../../../../glossary.md#term-immutable), retains no [Tensor](../../../../glossary.md#tensor) or
service, and performs no [normalization](../../../../glossary.md#term-normalization), [seed](../../../../glossary.md#term-seed) expansion, hashing, or key derivation.

The key selects an abstract caller-owned stream/domain. Callers that require separated domains
must choose different keys explicitly; this task adds no derivation [API](../../../../glossary.md#term-application-programming-interface-api) or uniqueness promise.
The counter identifies the next abstract logical [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) position. A later [operation](../../../../glossary.md#operation) consumes a
documented number of consecutive positions, retains the key, and advances modulo `2^64`.
Branching one state into two operations intentionally reuses the same interval; sequential callers
thread `nextState` to avoid that reuse. Counter wrap is defined modularly and does not trigger a
[model](../../../../glossary.md#term-model) exception; avoiding a full-period wrap is caller responsibility.

### [Operation](../../../../glossary.md#operation) kind and [signature](../../../../glossary.md#term-signature)

Create:

```java
public enum GraphRngKind implements OperationKind {
    INITIAL_STATE
}
```

`INITIAL_STATE` accepts exactly `GraphRngStateAttrs`, exactly zero inputs, and exactly one output:

```java
OperationSignature.fixed(GraphRngStateAttrs.class, 0, 1)
```

Its semantics are the explicit key/counter state described above. It is not a source [lookup](../../../../glossary.md#term-lookup),
[host](../../../../glossary.md#term-host) allocation, random [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), [kernel](../../../../glossary.md#kernel), [runtime](../../../../glossary.md#term-runtime) initialization hook, or [serialization](../../../../glossary.md#term-serialization) [token](../../../../glossary.md#term-token).

### State [Tensor descriptor](../../../../glossary.md#tensor-descriptor) and [provenance](../../../../glossary.md#provenance)

`GraphRngState.initial(key, counter)` constructs exactly:

```text
operation: GraphRngKind.INITIAL_STATE + GraphRngStateAttrs(key, counter)
inputs: []
outputs: one
output descriptor:
  data type: INT64
  shape: Shape[2]
  layout: unresolved
  requiresGrad: false
label: absent
host storage: absent
provenance output index: 0
```

The two logical [INT64](../../../../glossary.md#data-type--datatype) [lanes](../../../../glossary.md#term-lane) are an [opaque](../../../../glossary.md#term-opaque) raw-word representation: lane zero is key and lane one
is counter. They are not signed numerical values and are not a public invitation to use ordinary
[Tensor](../../../../glossary.md#tensor) arithmetic on [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate). [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) rather than a bare Tensor is the public type-safe
boundary.

The result uses `TensorFactory.createDerived(...)` with the exact [descriptor](../../../../glossary.md#term-descriptor), empty label,
[operation](../../../../glossary.md#operation), and empty input list. [Producer](../../../../glossary.md#term-producer)/[signature](../../../../glossary.md#term-signature) [validation](../../../../glossary.md#term-validation) occurs before Tensor ID allocation.
The wrapper retains that exact result. No storage, random draw, state [buffer](../../../../glossary.md#term-buffer), or second ID is
allocated.

### Portability, replay, and [serialization](../../../../glossary.md#term-serialization) boundary

- Key/counter raw bits, fixed descriptor, input/output ordering, and future advancement counts are
  portable model semantics.<span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [descriptor glossary](../../../../glossary.md#term-descriptor); [model glossary](../../../../glossary.md#term-model))</span>
- Constructing the same graph with equal key/counter attributes requests deterministic replay
  from the same abstract stream position, but produces distinct expression identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- This task intentionally selects no PRNG, key schedule, counter-to-bits function, floating
  conversion, or backend bitstream. It therefore makes no cross-backend, cross-route,
  cross-provider, cross-version, or bitwise sample promise.<span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [backend glossary](../../../../glossary.md#backend); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Until a portable algorithm is selected, later dropout replay is bounded to the same conforming
  prepared implementation and configuration. State advancement itself remains portable.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- `GraphRngStateAttrs` is the semantic information a future graph serializer must preserve
  losslessly. This task defines no byte encoding, parser, stable enum token, schema version, or
  public serializer.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [schema glossary](../../../../glossary.md#term-schema))</span>
- [Eager](../../../../glossary.md#term-eager-evaluation) `TensorRandoms` remains unchanged: it consumes a caller-owned [JDK](../../../../glossary.md#term-java-development-kit-jdk) `RandomGenerator`
  immediately to create host-backed leaf data. <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [leaf glossary](../../../../glossary.md#term-leaf))</span>[`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) instead records storage-free graph
  semantics for later execution. Neither owns a hidden generator.

## [Validation](../../../../glossary.md#term-validation) order and side effects

`GraphRngState.initial(long, long)` accepts every primitive pair and has no null, range, signedness,
or storage validation failure. Construction order is:

1. create the fixed [descriptor](../../../../glossary.md#term-descriptor);
2. create exact attributes and [operation](../../../../glossary.md#operation);
3. call the existing single-output derived factory with an empty input list;
4. allocate exactly one [Tensor](../../../../glossary.md#tensor) ID inside that seam; and
5. validate and wrap the returned Tensor through the package-private constructor.

No state storage or random source is allocated at any step. If fixed descriptor or operation
construction unexpectedly fails, no ID is consumed. If Tensor ID space is exhausted, propagate
the existing `IllegalStateException` message `tensor identifier space exhausted`; allocate no
wrapper and perform no [rollback](../../../../glossary.md#term-rollback). An [`OutOfMemoryError`](../../../../glossary.md#memory-management-atomic-counters-and-complexity) may occur during ordinary object/list
allocation; no stronger rollback guarantee is added. Do not add validation that consumes an ID
before a known invalid condition.

## Future 0019B1 [dropout](../../../../glossary.md#dropout--inverted-dropout) contract fixed by this plan

Task 0019B1 remains Draft and has no detailed task file, but it must preserve these selected broad
decisions when it becomes [Ready](../../../../glossary.md#term-ready):

- Public [API](../../../../glossary.md#term-application-programming-interface-api):

  ```java
  public DropoutResult dropout(double probability, GraphRngState state)
  public record DropoutResult(Tensor output, GraphRngState nextState) {}
  ```

- `DropoutKind.DROPOUT` with `DropoutAttrs(double probability)`, exactly two ordered inputs
  `[input, stateTensor]`, and exactly three ordered producer outputs:
  0. dropped output Tensor;<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
  1. auxiliary BOOL mask, retained as a producer slot for compiler-owned backward construction
     but not exposed by <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [compiler glossary](../../../../glossary.md#term-compiler))</span>`DropoutResult`; and
  2. next [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)-state [Tensor](../../../../glossary.md#tensor) wrapped by [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate).
- The dropped output is floating-only, preserves exact input Shape/data type/requiresGrad, has
  unresolved layout, and has no label or storage. The mask is BOOL, same Shape, non-gradient,
  unresolved, unlabeled, and storage-free. The state output has the exact fixed state descriptor.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Probability is [finite](../../../../glossary.md#term-finite) [binary64](../../../../glossary.md#floating-point-precision-and-rounding) in `[0, 1)`. It is the drop probability. For each logical
  element, one abstract uniform draw selects keep when it lies in `[probability, 1)`; kept values
  scale by `1 / (1 - probability)` and dropped values become same-type positive zero. Probability
  zero still consumes one draw per element, preserving shape-based advancement.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- The operation always represents training dropout. Inference is caller/compiler composition that
  bypasses the operation and retains both the input and state; there is no <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Inference glossary](../../../../glossary.md#term-inference))</span>`training` flag.
- State advances by the bound logical [element](../../../../glossary.md#term-element) count modulo `2^64`, including dynamic Shapes at
  execution. Empty tensors consume zero draws, return an empty output/mask, and preserve the state
  words semantically while still producing a distinct next-state output occurrence.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Equal input state, Shape, probability, input values, and conforming prepared implementation
  determine equal mask/output/next-state values. Cross-backend bitwise masks are not promised
  without a later portable algorithm decision.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Compiler owns gradient rules and backward graph construction. The auxiliary forward mask is the
  selected saved value; model does not implement the gradient. Prepare/backend/runtime own
  sampling, state materialization, kernels, and execution.<span data-glossary-navigation="true"> ([backward graph glossary](../../../../glossary.md#backward-graph); [Prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [materialization glossary](../../../../glossary.md#materialization); [kernels glossary](../../../../glossary.md#kernel))</span>
- The implementation must use the existing shared multi-output factory seam, retain output indices
  exactly, test the discarded-public-but-retained-producer auxiliary mask slot, and update all
  global API/signature/provenance inventories in the same task.<span data-glossary-navigation="true"> ([auxiliary mask glossary](../../../../glossary.md#auxiliary-mask); [signature glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance))</span>
- No public mask, eager evaluation, TensorRandoms delegation, RNG algorithm framework, backend
  route, runtime session state, hidden mutation, or global generator is added.<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [backend route glossary](../../../../glossary.md#backend-route))</span>

These broad decisions may be refined for exact [validation](../../../../glossary.md#term-validation) messages and bounded paths only when
0019B1 is planned. A conflict with them requires updating this plan before implementation rather
than silently inventing a different state [model](../../../../glossary.md#term-model).

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngStateAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/GraphRngState.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java` (Javadoc family
  and boundary inventory only; no public method or executable behavior)<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Expected test Java:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/random/GraphRngStateSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/GraphRngStateTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
  (global production-family coverage if its current inventory requires the new family)
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
  (existing derived-output and ID-side-effect seam coverage; change only if an exact inventory
  assertion requires it)
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProducerTest.java`
  (review the zero-input/output-slot contract; change only for a reusable assertion)
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProvenanceTest.java`
  (review output-index contract; change only for a reusable assertion)
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
  (public Tensor API inventory must remain unchanged)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Expected documentation and [planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Required no-change reviews:

- `TensorRandoms` production and focused [eager](../../../../glossary.md#term-eager-evaluation)-random tests
- `docs/api/runtime-api.md` and `docs/api/training-api.md`
- graph records, TensorDescriptor/DataType/Shape contracts, architecture documents/tests,
  conformance/integration tests, Gradle, dependencies, other modules, and task 0019B1 or later
  detailed specs<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the exact 18 paths listed above: four production, seven
test, and seven documentation/[planning](../../../../glossary.md#term-planning) paths. Review-only files do not count unless changed; do
not substitute another path silently.

The normal limit is sufficient because this task adds one state type, one tiny [operation](../../../../glossary.md#operation) family,
and focused tests. Do not use the authorized inventory/test paths for unrelated cleanup. If a
public [Tensor](../../../../glossary.md#tensor) method, another production helper, TensorDescriptor change, multi-output operation,
another document, another [module](../../../../glossary.md#term-module), build change, or architecture update is required, stop and
propose a replanned task.

## Public [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) purpose, opaque Tensor-expression ownership, identity equality,
  shallow immutability, explicit construction, raw unsigned words, branching/threading, replay
  boundary, no public Tensor/storage mutation, no hidden generator, and no execution.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [Tensor glossary](../../../../glossary.md#tensor); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [shallow immutability glossary](../../../../glossary.md#term-shallow-immutability))</span>
- Document `initial(...)` parameters, accepted full bit domains, exact descriptor/provenance,
  result identity, absence of storage/draws, ID exhaustion, and bounded portability.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Document every `GraphRngStateAttrs` component/accessor and its raw-bit unsigned interpretation,
  record equality, no normalization, and future serialization boundary.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Document `GraphRngKind` and `INITIAL_STATE` as zero-input/one-output backend-neutral semantics,
  not a kernel, runtime initializer, service lookup, or stable serialized token.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup); [token glossary](../../../../glossary.md#term-token))</span>
- Update Tensor type Javadoc only enough to name the new operation family and distinguish the
  privately wrapped state Tensor from ordinary public numerical Tensor use.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Review TensorFactory/TensorProducer/TensorProvenance/TensorDescriptor/DataType/Shape/
  TensorRandoms Javadocs and record reasoned no-change conclusions unless the final implementation
  makes a current statement false.<span data-glossary-navigation="true"> ([TensorProducer glossary](../../../../glossary.md#producer--tensorproducer); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape))</span>

## Acceptance criteria

- Exactly the three selected production types are added; no additional public [graph](../../../../glossary.md#graph)-random [API](../../../../glossary.md#term-application-programming-interface-api).
- [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) is final, has no public constructor, privately retains one exact Tensor, exposes
  exactly one public static <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`initial(long, long)` method plus the exact package-private constructor
  and Tensor accessor, and inherits identity equality.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Every key/counter bit pattern is accepted and preserved exactly in `GraphRngStateAttrs`.
- `INITIAL_STATE` has the exact attrs class, zero-input/one-output signature, fixed state
  descriptor, producer output index zero, empty inputs/label/storage, and one fresh Tensor ID.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer))</span>
- Equal attributes create semantically equivalent stream positions but distinct state, Tensor,
  producer, and ID identities.
- No state storage, random draw, source lookup, generator retention, global/thread-local mutable
  state, service, registry, reflection, backend/runtime import, or algorithm promise exists.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [mutable glossary](../../../../glossary.md#term-mutable); [registry glossary](../../../../glossary.md#term-registry); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Eager](../../../../glossary.md#term-eager-evaluation) `TensorRandoms` production API, behavior, tests, source ownership, and Javadocs remain
  unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Tensor's public method inventory and TensorFactory's 31-public/35-total inventory remain
  unchanged. Operation signature and producer/provenance inventories include the new family only
  where their global coverage requires it.<span data-glossary-navigation="true"> ([Operation signature glossary](../../../../glossary.md#operation-signature); [provenance glossary](../../../../glossary.md#provenance))</span>
- Public Javadocs, Tensor API, Compile API, glossary, capabilities, task/master/roadmap status, and
  newcomer state-threading/replay example are finalized by a separate clean documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Runtime API and Training API remain unchanged with reasoned conclusions: this task adds no
  prepared/run state, execution, dropout, gradient, parameter, optimizer, or training workflow.<span data-glossary-navigation="true"> ([run state glossary](../../../../glossary.md#run-state--runstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [gradient glossary](../../../../glossary.md#term-gradient); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Architecture/Gradle/dependencies/tests outside model/other modules/later detailed specs remain
  unchanged with reasoned conclusions.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [model glossary](../../../../glossary.md#term-model); [modules glossary](../../../../glossary.md#term-module))</span>
- Task 0019B is Ready during planning and becomes Complete only after implementation and docs
  validation. Task 0019B1 and tasks 0019C–0019E remain Draft with no detailed follow-up spec.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Use focused tests while implementing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.random.GraphRngStateSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.GraphRngStateTest \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) test:

```bash
./gradlew :modules:model:test
```

The separate clean-context documentation pass reuses that evidence unless it changes executable
Java. After finalizing Javadocs and documentation, it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also must:

- compile and run one Java 26 public example that creates two replay-equivalent initial states and
  demonstrates the documented future threading sequence without claiming dropout exists;<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- inspect generated Javadoc for the exact three-type public surface and complete parameter/result/
  failure contracts;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- validate all local Markdown links/anchors, balanced fences, final newlines, and whitespace in
  changed documentation;
- inspect public/declared methods, fields, constructors, imports, and bytecode only where needed to
  prove the no-accessor/no-state/no-service contract;<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- confirm the raw key/counter attrs, fixed descriptor, empty inputs, output index zero, exact
  producer/descriptor references, identity inequality, and ID exhaustion behavior;<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- confirm Tensor/TensorFactory/TensorRandoms public inventories, global operation-signature
  coverage, and existing multi-output construction/provenance tests remain coherent;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [signature glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance))</span>
- confirm exactly the authorized paths changed and no architecture/focused architecture,
  runtime/training API, Gradle, architecture/conformance/integration test, other-module, or later
  detailed-task path changed; and<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- confirm no model task is currently Ready after completion. Task 0019B1 and 0019C–0019E remain
  Draft without detailed specifications.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Repository-wide validation is deferred to the selected-modern-[operation](../../../../glossary.md#operation)-family [checkpoint](../../../../glossary.md#term-checkpoint) after
task 0022 and [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [module](../../../../glossary.md#term-module) and no [dependency](../../../../glossary.md#term-dependency) or architecture boundary.

## Dependencies

- 0001–0002 — current [data type](../../../../glossary.md#data-type--datatype) and [Shape](../../../../glossary.md#shape) values.
- 0005–0007 — [operation](../../../../glossary.md#operation), attributes, and [Tensor descriptor](../../../../glossary.md#tensor-descriptor) foundations.
- 0011–0013 — public [Tensor](../../../../glossary.md#tensor) state, construction, and [provenance](../../../../glossary.md#provenance).
- 0018K — exact kind/attributes [signatures](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality).
- 0018L — shared [producer](../../../../glossary.md#term-producer) and indexed output provenance.
- 0018N — precedent for exact [immutable](../../../../glossary.md#term-immutable) semantic values distinct from Tensor storage.
- 0018S and 0012F–0012I — final [eager](../../../../glossary.md#term-eager-evaluation) `TensorRandoms` [ownership](../../../../glossary.md#term-ownership-meanings) and explicit-source boundary.

## Follow-up tasks

- 0019B1 — **required Draft follow-up**, explicit dropout construction using state input, auxiliary
  mask, and next-state output. It depends on 0019B and 0018K–0018L. No detailed spec exists yet.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [auxiliary mask glossary](../../../../glossary.md#auxiliary-mask))</span>
- 0019E — initial scaled dot-product attention remains independent and excludes dropout. A later
  attention-dropout extension must consume <span data-glossary-navigation="true"> ([scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention))</span>[`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) explicitly.
- Later graph sampling operations may reuse this state contract only after focused distribution
  semantics are selected; this task does not create those rows or specs.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [distribution glossary](../../../../glossary.md#term-distribution))</span>

## Architecture impact

Expected impact: None.

The existing architecture assigns [backend](../../../../glossary.md#backend)-independent [Tensor](../../../../glossary.md#tensor) and [Operation](../../../../glossary.md#operation) semantics to [model](../../../../glossary.md#term-model),
[compiler](../../../../glossary.md#term-compiler) transformations/backward construction to compiler, and executable state/[materialization](../../../../glossary.md#materialization)
to [prepare](../../../../glossary.md#prepare)/backend/[runtime](../../../../glossary.md#term-runtime). An [opaque](../../../../glossary.md#term-opaque) model value around a storage-free Tensor expression fits
those rules without changing [ownership](../../../../glossary.md#term-ownership-meanings) or [dependency direction](../../../../glossary.md#java-platform-and-build-tools).

If implementation requires a public raw-state Tensor, runtime session/generator state in model,
storage ownership, another [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or a fixed backend algorithm, stop and report the
exact architecture decision needed before editing code or architecture documents.

## Implementation prompt

Use this prompt in a separate clean-context agentic task/thread:

```text
You are a clean-context implementation agent in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, the focused architecture documents named by task 0019B,
documentation/planning rules, roadmap, model capabilities/master plan, task 0019B, completed
foundation/eager-random/provenance tasks named there, Tensor/Compile/Runtime/Training APIs,
glossary, and every affected or review-only source/test named by the task in full.

Implement docs/planning/modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md exactly.
Add only the explicit graph RNG state foundation; do not implement dropout or task 0019B1. Stay
within the exact authorized paths. Preserve TensorRandoms, Tensor/TensorFactory public surfaces,
architecture boundaries, and the no-hidden-generator/no-algorithm contract. Stop on architecture
or scope conflict. Do not commit or push.

Run focused tests as needed and exactly one final :modules:model:test after executable Java
stabilizes. Then hand the actual diff and exact evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass must independently finalize
affected Javadocs, Tensor/Compile APIs, glossary, capabilities/task/master/roadmap, the newcomer
example, and documentation validation; reuse Java evidence unless executable behavior changes.

Update this task's evidence, notes, completion summary, and status only after both passes. Leave
0019B1 and 0019C–0019E Draft without detailed specs.
```

Documentation-agent [handoff](../../../../glossary.md#term-handoff) must include the exact task path, final implementation diff, affected
public state/[operation](../../../../glossary.md#operation) contracts, the recorded final [model](../../../../glossary.md#term-model)-test result, unchanged [eager](../../../../glossary.md#term-eager-evaluation)-random and
[runtime](../../../../glossary.md#term-runtime)/training boundaries, all expected documentation paths, and every documentation/[scope](../../../../glossary.md#term-scope)/
status [validation](../../../../glossary.md#term-validation) above.

## Local decisions

- Planning split the broad frontier at the reusable semantic boundary: 0019B owns explicit state;
  0019B1 owns dropout.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- The state is a dedicated [opaque](../../../../glossary.md#term-opaque) public [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), not Tensor metadata, a bare numerical
  Tensor, an IR node, runtime state, or an eager host value.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [node glossary](../../../../glossary.md#node); [runtime glossary](../../../../glossary.md#term-runtime); [eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host))</span>
- Key and counter are two raw unsigned 64-bit words represented by Java `long`, with no seed
  expansion or algorithm selection.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- Initial state is a zero-input one-output Tensor producer. Later sampling operations thread its
  private Tensor as an ordinary producer input/output.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- State objects use expression identity equality. Equal attrs mean replay-equivalent positions,
  not the same expression occurrence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [mean glossary](../../../../glossary.md#term-mean); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Dropout will have a hidden auxiliary mask producer slot so compiler-owned backward construction
  can consume the exact forward selection without making the mask public.<span data-glossary-navigation="true"> ([auxiliary mask glossary](../../../../glossary.md#auxiliary-mask); [compiler glossary](../../../../glossary.md#term-compiler); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>

## Known limitations

- No [operation](../../../../glossary.md#operation) consumes or advances the state until task 0019B1.
- No public [checkpoint](../../../../glossary.md#term-checkpoint)/import/export or raw-state [Tensor](../../../../glossary.md#tensor) accessor is provided.
- No random algorithm or cross-backend bitstream is selected; replay guarantees are bounded as
  documented.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Counter wrap is modular and caller-managed.
- Graph capture, serialization, execution, gradients, and backend support remain separately owned
  and unimplemented by this task.<span data-glossary-navigation="true"> ([Graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [serialization glossary](../../../../glossary.md#term-serialization); [gradients glossary](../../../../glossary.md#term-gradient))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/plan_0019b`:

- Read the required architecture, lifecycle, training, module/dependency/runtime-boundary,
  documentation/planning, model capability/master/roadmap, relevant completed task, API, source,
  test, and inventory contracts.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Confirmed model may own backend-neutral state semantics and Tensor expression construction while
  compiler and prepare/backend/runtime retain their established responsibilities; no architecture
  update is required.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare))</span>
- Confirmed [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) supports zero inputs and indexed outputs through family [signatures](../../../../glossary.md#term-signature),
  `TensorFactory.createDerived(...)` constructs the required one-output [occurrence](../../../../glossary.md#term-occurrence), and
  `createDerivedOutputs(...)` remains available for Draft [dropout](../../../../glossary.md#dropout--inverted-dropout)'s three outputs.
- Confirmed `TensorRandoms` is [eager](../../../../glossary.md#term-eager-evaluation) caller-`RandomGenerator` host-data creation and must remain
  unchanged.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- Selected the split because state foundation and dropout are independently reviewable and the
  combined scope would exceed the normal task-size guardrail.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Created and synchronized exactly four planning-only paths: this specification,
  `capabilities.md`, the [model](../../../../glossary.md#term-model) master plan, and the roadmap. `git status --short` shows no Java,
  tests, Gradle, architecture/focused-architecture, API, other-module, or unrelated path.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- A targeted Ruby local-Markdown path check passed for all links in the four changed files.
  Targeted newline, trailing-whitespace, and backtick-fence checks also passed.
- Status scans found exactly one Ready frontier represented consistently by this task, the model
  master-plan row, and the roadmap row. No 0019B1 task file exists; 0019B1 and 0019C–0019E remain
  broad Draft rows, while 0019 through 0019A2 remain Complete.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Stale-status scans found no remaining statement that 0019B is Draft, that all 0019B–0019E tasks
  are Draft, or that no model task is Ready.
- `git diff --check` passed with no output after final planning edits. Created-file checks were
  covered separately because the new untracked specification is not included in ordinary diff
  output until staged.

Implementation context `/root/task_0019b_implementation`:

- Added exactly `GraphRngKind`, `GraphRngStateAttrs`, and [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), plus focused
  `GraphRngStateSemanticsTest`, `GraphRngStateTest`, and the required global
  `OperationSignatureTest` family entry. No executable path outside that selected set changed.
- Implemented the exact zero-input/one-output `INITIAL_STATE` signature; raw key/counter record;
  fixed storage-free `INT64 Shape[2]` state descriptor; output-index-zero provenance; one-ID
  construction; package-private validated wrapping; and identity-equality public boundary.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The required focused command selecting all seven test classes passed with `BUILD SUCCESSFUL`
  after the final executable refinement.
- Exactly one final `./gradlew :modules:model:test` then passed with `BUILD SUCCESSFUL in 1s`,
  three actionable tasks, one executed, and two up-to-date. Executable Java did not change after
  that run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>

Documentation-agent context `/root/task_0019b_implementation/task_0019b_docs`:

- Independently read the architecture contract and focused architecture pages, documentation and
  planning rules, General/API-Javadoc/Planning/Example profiles, task and directly related
  completed tasks, final source/tests, Tensor/Compile/Runtime/Training APIs, glossary,
  capabilities, master plan, roadmap, and foundational Tensor contracts.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile))</span>
- Finalized Javadocs for all three new production types and the Tensor family/boundary inventory.
  Finalized Tensor API, Compile API, glossary, capabilities, this task, master plan, and roadmap.
  The documentation distinguishes raw unsigned words, opaque state Tensor identity, replay,
  branching/threading, bounded portability, and future serialization from algorithms, sampling,
  execution, and still-Draft dropout.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [serialization glossary](../../../../glossary.md#term-serialization); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- Reused the focused and final model-test evidence because this pass changed only comments and
  documentation. `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL in 2s`, two
  actionable tasks executed, after final Javadoc edits.
- Compiled the public example with Java 26.0.1 and ran it successfully; two equal-word state
  initializers printed `false` for both reference and inherited equality. The documented dropout
  sequence is explicitly conceptual and identifies task 0019B1 as its future owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Inspected generated pages for `GraphRngState`, `GraphRngStateAttrs`, and `GraphRngKind`; parameter,
  result, failure, unsigned-word, identity, zero-input/one-output, and serialization boundaries are
  present. <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>`javap -p` and bytecode inspection confirmed the exact one-field/two-method wrapper,
  exact record and enum surfaces, validation order, fixed descriptor, empty inputs,<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
  `TensorFactory.createDerived(...)`, and no hidden accessor or service.
- A Java 26 reflection inventory check passed: Tensor remains 162 declared public methods,
  TensorFactory remains 31 public/35 total methods, TensorRandoms remains five public methods with
  no fields, GraphRngState has one public declared method, and `INITIAL_STATE` is exact zero-input/
  one-output. Raw `-1L` and `Long.MIN_VALUE` words were preserved. Focused tests supply the exact
  descriptor/provenance-reference, identity, validation-order/no-ID, and ID-exhaustion evidence.
- The first two temporary Ruby checker invocations were not valid checks: one exposed unavailable
  `Enumerator#filter_map` in the installed Ruby and one exposed an inaccurate repeated-hyphen slug
  rule. After correcting compatibility and GitHub-style anchors, the final invocation checked all
  seven documentation paths and passed local links, anchors, balanced fences, final newlines, and
  trailing whitespace.
- Final exact-scope, status, forbidden-import/path, later-spec, whitespace, and diff checks passed.
  The combined change uses 14 of the exact 18 authorized paths: four production, three tests, and
  seven documentation/planning paths. No architecture/focused-architecture, Runtime/Training API,
  Gradle/dependency, architecture/conformance/integration test, other-module, or later detailed
  task path changed. Task 0019B is Complete; 0019B1 and 0019C–0019E remain Draft without detailed
  specs; no model task is currently Ready.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>

Reasoned no-change reviews:

- TensorFactory, TensorProducer, TensorProvenance, TensorDescriptor, DataType, Shape, and
  TensorRandoms Javadocs remain accurate because the implementation reuses their existing seams
  without changing their public surface, ownership, validation, eager-random behavior, or
  producer/output-index contracts.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [producer glossary](../../../../glossary.md#term-producer))</span>
- Runtime API and Training API remain accurate because this task adds no prepared/run state,
  execution, dropout, gradient, parameter, optimizer, or training workflow.<span data-glossary-navigation="true"> ([run state glossary](../../../../glossary.md#run-state--runstate); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Architecture documents/ADRs/tests, backend conformance, integration tests, Gradle/dependencies,
  and other modules remain accurate because this is one backend-neutral model semantic value and
  expression occurrence with no dependency, lifecycle, backend, runtime, or build change.
- Task 0019B1 and later detailed specs remain absent because only the current frontier may have a
  detailed specification; the broad Draft rows already preserve future ownership.

## Implementation notes

- Used the existing package-private single-output derived-construction seam, so operation
  validation precedes the sole Tensor ID allocation and no storage or generator is introduced.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [validation glossary](../../../../glossary.md#term-validation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Kept state private behind [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate); equal attributes express replay-equivalent positions
  while ordinary object identity distinguishes occurrences.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>
- Documented the future state-threading example without adding or claiming a current [dropout](../../../../glossary.md#dropout--inverted-dropout) [API](../../../../glossary.md#term-application-programming-interface-api).

## Completion summary

- Completed changes: Added the explicit graph RNG state operation kind, raw attributes, opaque
  public wrapper, focused tests, global signature inventory, finalized Javadocs, API references,
  glossary term, and synchronized planning records.<span data-glossary-navigation="true"> ([graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [operation glossary](../../../../glossary.md#operation); [opaque glossary](../../../../glossary.md#term-opaque); [signature glossary](../../../../glossary.md#term-signature); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: Four production, three test, and seven documentation/planning paths,
  all within the exact authorized inventory.
- Tests and validation: Required focused tests and the final model suite passed in the
  implementation context; model Javadoc, Java 26 examples/surface checks, generated-Javadoc,
  Markdown, scope/status, import/bytecode, formatting, and final diff checks passed in the
  documentation context.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Documentation and Javadoc impact: Finalized all affected public contracts and recorded reasoned
  no-change conclusions for every required review-only area.
- Unresolved issues: None.
- Follow-up required: None. Draft task 0019B1 owns dropout and state advancement by a consuming
  operation.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>

Status: Complete
