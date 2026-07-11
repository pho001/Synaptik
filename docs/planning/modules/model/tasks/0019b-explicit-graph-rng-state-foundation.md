# Task 0019B: Explicit Graph RNG State Foundation

## Status

Complete

## Goal

Add the smallest public, backend-independent model contract for explicit random-number-generator
(RNG) state in Tensor expression graphs. A caller must be able to create one state from explicit
64-bit key and counter bit patterns, thread that state through later state-consuming operations,
and replay the same graph from the same explicit state without a hidden process-global generator.

This task establishes the state value and its zero-input Tensor producer only. Draft follow-up
0019B1 owns dropout, state advancement by output element count, the genuine multi-output
occurrence, and its public result carrier. Separating those capabilities keeps the reusable state
contract reviewable before probability, masking, scaling, dynamic-count advancement, and backward
requirements are added.

Newcomer mental model:

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
sample, advance state, or promise a random bitstream.

## Rationale and split decision

The former broad 0019B combines two independently reviewable contracts:

1. an opaque public graph-state value, its exact key/counter representation, explicit creation,
   and Tensor provenance; and
2. dropout probability/mask/scaling semantics, three producer outputs, dynamic state advancement,
   and a public value-plus-state result.

The combined change would normally require more than the planning guide's 12–18-path guardrail
and would make the reusable state representation depend on one distribution operation. Task
0019B therefore remains the stable ID for the first Ready RNG-state foundation. Task 0019B1 is a
broad Draft follow-up without a detailed specification. Established tasks 0019C–0019E keep their
IDs, order, and scopes.

## Scope

- Add one public final `GraphRngState` value in `model.tensor`.
- Add one public immutable `GraphRngStateAttrs(long key, long counter)` record and one
  `GraphRngKind.INITIAL_STATE` operation kind in `model.operation.random`.
- Add exactly one public creation method:

  ```java
  public static GraphRngState initial(long key, long counter)
  ```

- Represent the state internally as one storage-free, unlabeled Tensor expression with exact
  descriptor `INT64`, `Shape.of(2)`, unresolved layout, and `requiresGrad == false`.
- Make `INITIAL_STATE` a zero-input, one-output producer. The state Tensor has output index zero
  in ordinary `TensorProvenance` and retains the exact producer descriptor reference.
- Interpret `key` and `counter` as unsigned 64-bit bit patterns carried by Java `long`; every bit
  pattern is valid and no signed ordering is assigned.
- Define the counter as the next abstract logical sample position and the key as caller-selected
  stream/domain identity. State-consuming operations keep the key and advance the counter modulo
  `2^64` by their specified logical draw count.
- Define public immutability, identity equality, ownership, portability, replay, branching, and
  serialization boundaries precisely without selecting a sampling algorithm.
- Add focused semantic and public-state tests, and include existing global producer/provenance,
  factory-derived-output, operation-signature, and public Tensor API inventory tests in the
  authorized review scope.
- Finalize public Javadocs, Tensor API, Compile API, glossary, capabilities, task evidence, master
  plan, and roadmap through a mandatory separate clean-context documentation pass.

## Out of scope

- dropout, Bernoulli masks, probability validation, inverted scaling, training/inference mode,
  mask exposure, or state advancement performed by an operation
- a generic graph-random sampling API, random bits, uniform/normal/integral distributions,
  categorical sampling, shuffling, permutation, or attention dropout
- a portable PRNG algorithm, key derivation, seed expansion, bitstream, floating conversion,
  statistical-quality guarantee, backend-independent numerical replay, or conformance tolerance
- split/fold-in/jump/reset/copy/fork APIs, a distribution enum, pluggable algorithms, generator
  interfaces, factories, registries, services, dependency injection, or reflective discovery
- `RandomGenerator`, `TensorRandoms`, or eager host-data construction changes
- exposing the internal state Tensor as a general numerical Tensor, accepting an arbitrary
  caller-supplied `INT64[2]` Tensor as state, or adding state metadata to `TensorDescriptor`
- storage allocation, host-state mutation, device residency, runtime session state, prepared
  state, physical buffers, execution, kernels, lowering, or backend routes
- graph capture, compiler serialization, common-subexpression policy, constant folding, gradient
  rules, autograd traversal, or backward graph construction
- Gradle, dependencies, architecture documents/tests, another module, or another detailed task
  specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership, Tensor as
  public mutable API state rather than IR, compiler-owned backward graph construction, and
  runtime/prepare/backend state boundaries
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Tasks 0018K](0018k-operation-signature-and-construction-hardening.md) and
  [0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018S](0018s-tensor-factory-surface-cleanup.md)
- completed eager-random tasks 0012F–0012I
- completed tasks 0019–0019A2
- [Tensor API](../../../../api/tensor-api.md), [Compile API](../../../../api/compile-api.md),
  [Runtime API](../../../../api/runtime-api.md), [Training API](../../../../api/training-api.md),
  and [glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` may own this immutable backend-neutral semantic value and its public Tensor
  expression construction. The architecture already permits model-owned Tensor and Operation
  semantics; no architecture update is required.
- `GraphRngState` is a public model value that privately retains one Tensor expression. It is not
  Tensor metadata, a graph node/value, a runtime session, a generator service, or storage owner.
- `Tensor` remains public mutable API state and not IR. The private retained Tensor supplies
  expression identity/provenance only; `GraphRngState` does not expose storage attachment,
  publication, runtime residency, or graph-local IDs.
- The state Tensor participates normally in `TensorProducer` input/output positions. This task's
  initializer is zero-input and one-output. Later state-consuming operations may receive the
  exact privately retained Tensor through package-private same-package mechanics.
- State construction must use the existing package-private derived Tensor construction seam. It
  must not directly allocate a `TensorId`, construct `TensorProducer`/`TensorProvenance`, attach
  storage, or add another factory.
- Compiler owns capture, serialization policy, CSE, gradient rules, and backward graph
  construction. Prepare/backend/runtime own materialization, sampling implementation, executable
  state, and execution.
- No hidden process-global/default/thread-local generator, service locator, registry, reflection,
  backend handle, runtime state, storage, or mutable static state may enter model.
- If implementation requires exposing the retained Tensor publicly, accepting arbitrary Tensor
  state, changing TensorDescriptor, adding runtime/backend state, or selecting a portable PRNG
  algorithm, stop and update planning or request an architecture decision before editing.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor expression values and the existing
  package-private derived-construction seam.
- `io.github.pho001.synaptik.model.operation` — owns generic operation contracts.
- `io.github.pho001.synaptik.model.datatype` and `.shape` — own the fixed state descriptor.

Packages added or changed:

- Add `io.github.pho001.synaptik.model.operation.random` for explicit graph-random operation
  semantics. This is distinct from eager `model.tensor.TensorRandoms` host-data population.
- Change the existing `model.tensor` package with one focused public state type and only the
  minimum Javadoc/inventory integration.

Type placement:

- `io.github.pho001.synaptik.model.tensor.GraphRngState` — public opaque graph-state value beside
  Tensor because it privately wraps one Tensor expression and must use the package-private
  construction seam.
- `io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs` — immutable raw key and
  counter semantic attributes for the initializer operation.
- `io.github.pho001.synaptik.model.operation.random.GraphRngKind` — focused family containing only
  `INITIAL_STATE` in this task.

Test placement:

- `io.github.pho001.synaptik.model.operation.random.GraphRngStateSemanticsTest` — attribute/kind,
  unsigned-bit-pattern, signature, and API-shape contract.
- `io.github.pho001.synaptik.model.tensor.GraphRngStateTest` — public construction, descriptor,
  provenance, identity, immutability, no-storage, no-hidden-state, and ID-side-effect contract.

## Required contract

### Public state type

Create:

```java
public final class GraphRngState {
    public static GraphRngState initial(long key, long counter);
}
```

The type has exactly one private final instance field containing the state Tensor, one
package-private constructor used only by validated same-package expression construction, and no
public constructor. The constructor validates in this order: non-null Tensor; exact `INT64` data
type; structural `Shape.of(2)` equality; unresolved layout; false gradient eligibility; absent
label; absent host storage; and present provenance. Failures identify the first violated state
invariant and allocate no Tensor ID. It retains the exact Tensor reference after validation.

Expose exactly one package-private `tensor()` accessor returning that reference so the later
dropout helper can use it as an ordered producer input and wrap the next-state output. There is no
public Tensor accessor, key/counter accessor, mutation method, storage method, split method, or
generic public wrapping method. The focused same-package test locks the constructor/accessor
visibility and validation order.

`GraphRngState` deliberately inherits ordinary object-identity `equals` and `hashCode`. Two calls
with equal key/counter attributes are distinct public state-expression occurrences with distinct
Tensor and producer identities, even though they request equivalent abstract stream positions.
The class is shallowly immutable: its retained Tensor reference never changes and is not exposed,
so callers cannot use this type to mutate its storage association. It owns no mutable generator or
execution state.

### Key and counter representation

Create:

```java
public record GraphRngStateAttrs(long key, long counter)
        implements OperationAttrs {}
```

Every `long` bit pattern is valid. Interpret each component as an unsigned 64-bit word; Java's
signed decimal rendering and comparisons have no semantic meaning. Record equality and hashing
compare the exact 64 bits of both components. The record is immutable, retains no Tensor or
service, and performs no normalization, seed expansion, hashing, or key derivation.

The key selects an abstract caller-owned stream/domain. Callers that require separated domains
must choose different keys explicitly; this task adds no derivation API or uniqueness promise.
The counter identifies the next abstract logical sample position. A later operation consumes a
documented number of consecutive positions, retains the key, and advances modulo `2^64`.
Branching one state into two operations intentionally reuses the same interval; sequential callers
thread `nextState` to avoid that reuse. Counter wrap is defined modularly and does not trigger a
model exception; avoiding a full-period wrap is caller responsibility.

### Operation kind and signature

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

Its semantics are the explicit key/counter state described above. It is not a source lookup,
host allocation, random sample, kernel, runtime initialization hook, or serialization token.

### State Tensor descriptor and provenance

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

The two logical INT64 lanes are an opaque raw-word representation: lane zero is key and lane one
is counter. They are not signed numerical values and are not a public invitation to use ordinary
Tensor arithmetic on RNG state. `GraphRngState` rather than a bare Tensor is the public type-safe
boundary.

The result uses `TensorFactory.createDerived(...)` with the exact descriptor, empty label,
operation, and empty input list. Producer/signature validation occurs before Tensor ID allocation.
The wrapper retains that exact result. No storage, random draw, state buffer, or second ID is
allocated.

### Portability, replay, and serialization boundary

- Key/counter raw bits, fixed descriptor, input/output ordering, and future advancement counts are
  portable model semantics.
- Constructing the same graph with equal key/counter attributes requests deterministic replay
  from the same abstract stream position, but produces distinct expression identities.
- This task intentionally selects no PRNG, key schedule, counter-to-bits function, floating
  conversion, or backend bitstream. It therefore makes no cross-backend, cross-route,
  cross-provider, cross-version, or bitwise sample promise.
- Until a portable algorithm is selected, later dropout replay is bounded to the same conforming
  prepared implementation and configuration. State advancement itself remains portable.
- `GraphRngStateAttrs` is the semantic information a future graph serializer must preserve
  losslessly. This task defines no byte encoding, parser, stable enum token, schema version, or
  public serializer.
- Eager `TensorRandoms` remains unchanged: it consumes a caller-owned JDK `RandomGenerator`
  immediately to create host-backed leaf data. `GraphRngState` instead records storage-free graph
  semantics for later execution. Neither owns a hidden generator.

## Validation order and side effects

`GraphRngState.initial(long, long)` accepts every primitive pair and has no null, range, signedness,
or storage validation failure. Construction order is:

1. create the fixed descriptor;
2. create exact attributes and operation;
3. call the existing single-output derived factory with an empty input list;
4. allocate exactly one Tensor ID inside that seam; and
5. validate and wrap the returned Tensor through the package-private constructor.

No state storage or random source is allocated at any step. If fixed descriptor or operation
construction unexpectedly fails, no ID is consumed. If Tensor ID space is exhausted, propagate
the existing `IllegalStateException` message `tensor identifier space exhausted`; allocate no
wrapper and perform no rollback. An `OutOfMemoryError` may occur during ordinary object/list
allocation; no stronger rollback guarantee is added. Do not add validation that consumes an ID
before a known invalid condition.

## Future 0019B1 dropout contract fixed by this plan

Task 0019B1 remains Draft and has no detailed task file, but it must preserve these selected broad
decisions when it becomes Ready:

- Public API:

  ```java
  public DropoutResult dropout(double probability, GraphRngState state)
  public record DropoutResult(Tensor output, GraphRngState nextState) {}
  ```

- `DropoutKind.DROPOUT` with `DropoutAttrs(double probability)`, exactly two ordered inputs
  `[input, stateTensor]`, and exactly three ordered producer outputs:
  0. dropped output Tensor;
  1. auxiliary BOOL mask, retained as a producer slot for compiler-owned backward construction
     but not exposed by `DropoutResult`; and
  2. next RNG-state Tensor wrapped by `GraphRngState`.
- The dropped output is floating-only, preserves exact input Shape/data type/requiresGrad, has
  unresolved layout, and has no label or storage. The mask is BOOL, same Shape, non-gradient,
  unresolved, unlabeled, and storage-free. The state output has the exact fixed state descriptor.
- Probability is finite binary64 in `[0, 1)`. It is the drop probability. For each logical
  element, one abstract uniform draw selects keep when it lies in `[probability, 1)`; kept values
  scale by `1 / (1 - probability)` and dropped values become same-type positive zero. Probability
  zero still consumes one draw per element, preserving shape-based advancement.
- The operation always represents training dropout. Inference is caller/compiler composition that
  bypasses the operation and retains both the input and state; there is no `training` flag.
- State advances by the bound logical element count modulo `2^64`, including dynamic Shapes at
  execution. Empty tensors consume zero draws, return an empty output/mask, and preserve the state
  words semantically while still producing a distinct next-state output occurrence.
- Equal input state, Shape, probability, input values, and conforming prepared implementation
  determine equal mask/output/next-state values. Cross-backend bitwise masks are not promised
  without a later portable algorithm decision.
- Compiler owns gradient rules and backward graph construction. The auxiliary forward mask is the
  selected saved value; model does not implement the gradient. Prepare/backend/runtime own
  sampling, state materialization, kernels, and execution.
- The implementation must use the existing shared multi-output factory seam, retain output indices
  exactly, test the discarded-public-but-retained-producer auxiliary mask slot, and update all
  global API/signature/provenance inventories in the same task.
- No public mask, eager evaluation, TensorRandoms delegation, RNG algorithm framework, backend
  route, runtime session state, hidden mutation, or global generator is added.

These broad decisions may be refined for exact validation messages and bounded paths only when
0019B1 is planned. A conflict with them requires updating this plan before implementation rather
than silently inventing a different state model.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngStateAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/GraphRngState.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java` (Javadoc family
  and boundary inventory only; no public method or executable behavior)

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
  (public Tensor API inventory must remain unchanged)

Expected documentation and planning during implementation:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Required no-change reviews:

- `TensorRandoms` production and focused eager-random tests
- `docs/api/runtime-api.md` and `docs/api/training-api.md`
- graph records, TensorDescriptor/DataType/Shape contracts, architecture documents/tests,
  conformance/integration tests, Gradle, dependencies, other modules, and task 0019B1 or later
  detailed specs

## Maximum scope

This task may create or modify at most the exact 18 paths listed above: four production, seven
test, and seven documentation/planning paths. Review-only files do not count unless changed; do
not substitute another path silently.

The normal limit is sufficient because this task adds one state type, one tiny operation family,
and focused tests. Do not use the authorized inventory/test paths for unrelated cleanup. If a
public Tensor method, another production helper, TensorDescriptor change, multi-output operation,
another document, another module, build change, or architecture update is required, stop and
propose a replanned task.

## Public Javadoc requirements

- Document `GraphRngState` purpose, opaque Tensor-expression ownership, identity equality,
  shallow immutability, explicit construction, raw unsigned words, branching/threading, replay
  boundary, no public Tensor/storage mutation, no hidden generator, and no execution.
- Document `initial(...)` parameters, accepted full bit domains, exact descriptor/provenance,
  result identity, absence of storage/draws, ID exhaustion, and bounded portability.
- Document every `GraphRngStateAttrs` component/accessor and its raw-bit unsigned interpretation,
  record equality, no normalization, and future serialization boundary.
- Document `GraphRngKind` and `INITIAL_STATE` as zero-input/one-output backend-neutral semantics,
  not a kernel, runtime initializer, service lookup, or stable serialized token.
- Update Tensor type Javadoc only enough to name the new operation family and distinguish the
  privately wrapped state Tensor from ordinary public numerical Tensor use.
- Review TensorFactory/TensorProducer/TensorProvenance/TensorDescriptor/DataType/Shape/
  TensorRandoms Javadocs and record reasoned no-change conclusions unless the final implementation
  makes a current statement false.

## Acceptance criteria

- Exactly the three selected production types are added; no additional public graph-random API.
- `GraphRngState` is final, has no public constructor, privately retains one exact Tensor, exposes
  exactly one public static `initial(long, long)` method plus the exact package-private constructor
  and Tensor accessor, and inherits identity equality.
- Every key/counter bit pattern is accepted and preserved exactly in `GraphRngStateAttrs`.
- `INITIAL_STATE` has the exact attrs class, zero-input/one-output signature, fixed state
  descriptor, producer output index zero, empty inputs/label/storage, and one fresh Tensor ID.
- Equal attributes create semantically equivalent stream positions but distinct state, Tensor,
  producer, and ID identities.
- No state storage, random draw, source lookup, generator retention, global/thread-local mutable
  state, service, registry, reflection, backend/runtime import, or algorithm promise exists.
- Eager `TensorRandoms` production API, behavior, tests, source ownership, and Javadocs remain
  unchanged.
- Tensor's public method inventory and TensorFactory's 31-public/35-total inventory remain
  unchanged. Operation signature and producer/provenance inventories include the new family only
  where their global coverage requires it.
- Public Javadocs, Tensor API, Compile API, glossary, capabilities, task/master/roadmap status, and
  newcomer state-threading/replay example are finalized by a separate clean documentation pass.
- Runtime API and Training API remain unchanged with reasoned conclusions: this task adds no
  prepared/run state, execution, dropout, gradient, parameter, optimizer, or training workflow.
- Architecture/Gradle/dependencies/tests outside model/other modules/later detailed specs remain
  unchanged with reasoned conclusions.
- Task 0019B is Ready during planning and becomes Complete only after implementation and docs
  validation. Task 0019B1 and tasks 0019C–0019E remain Draft with no detailed follow-up spec.

## Tests / validation

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

After executable Java stabilizes, run exactly one final model test:

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
  demonstrates the documented future threading sequence without claiming dropout exists;
- inspect generated Javadoc for the exact three-type public surface and complete parameter/result/
  failure contracts;
- validate all local Markdown links/anchors, balanced fences, final newlines, and whitespace in
  changed documentation;
- inspect public/declared methods, fields, constructors, imports, and bytecode only where needed to
  prove the no-accessor/no-state/no-service contract;
- confirm the raw key/counter attrs, fixed descriptor, empty inputs, output index zero, exact
  producer/descriptor references, identity inequality, and ID exhaustion behavior;
- confirm Tensor/TensorFactory/TensorRandoms public inventories, global operation-signature
  coverage, and existing multi-output construction/provenance tests remain coherent;
- confirm exactly the authorized paths changed and no architecture/focused architecture,
  runtime/training API, Gradle, architecture/conformance/integration test, other-module, or later
  detailed-task path changed; and
- confirm no model task is currently Ready after completion. Task 0019B1 and 0019C–0019E remain
  Draft without detailed specifications.

Repository-wide validation is deferred to the selected-modern-operation-family checkpoint after
task 0022 and CI because this task changes one module and no dependency or architecture boundary.

## Dependencies

- 0001–0002 — current data type and Shape values.
- 0005–0007 — operation, attributes, and Tensor descriptor foundations.
- 0011–0013 — public Tensor state, construction, and provenance.
- 0018K — exact kind/attributes signatures and occurrence cardinality.
- 0018L — shared producer and indexed output provenance.
- 0018N — precedent for exact immutable semantic values distinct from Tensor storage.
- 0018S and 0012F–0012I — final eager `TensorRandoms` ownership and explicit-source boundary.

## Follow-up tasks

- 0019B1 — **required Draft follow-up**, explicit dropout construction using state input, auxiliary
  mask, and next-state output. It depends on 0019B and 0018K–0018L. No detailed spec exists yet.
- 0019E — initial scaled dot-product attention remains independent and excludes dropout. A later
  attention-dropout extension must consume `GraphRngState` explicitly.
- Later graph sampling operations may reuse this state contract only after focused distribution
  semantics are selected; this task does not create those rows or specs.

## Architecture impact

Expected impact: None.

The existing architecture assigns backend-independent Tensor and Operation semantics to model,
compiler transformations/backward construction to compiler, and executable state/materialization
to prepare/backend/runtime. An opaque model value around a storage-free Tensor expression fits
those rules without changing ownership or dependency direction.

If implementation requires a public raw-state Tensor, runtime session/generator state in model,
storage ownership, another module dependency, or a fixed backend algorithm, stop and report the
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

Documentation-agent handoff must include the exact task path, final implementation diff, affected
public state/operation contracts, the recorded final model-test result, unchanged eager-random and
runtime/training boundaries, all expected documentation paths, and every documentation/scope/
status validation above.

## Local decisions

- Planning split the broad frontier at the reusable semantic boundary: 0019B owns explicit state;
  0019B1 owns dropout.
- The state is a dedicated opaque public `GraphRngState`, not Tensor metadata, a bare numerical
  Tensor, an IR node, runtime state, or an eager host value.
- Key and counter are two raw unsigned 64-bit words represented by Java `long`, with no seed
  expansion or algorithm selection.
- Initial state is a zero-input one-output Tensor producer. Later sampling operations thread its
  private Tensor as an ordinary producer input/output.
- State objects use expression identity equality. Equal attrs mean replay-equivalent positions,
  not the same expression occurrence.
- Dropout will have a hidden auxiliary mask producer slot so compiler-owned backward construction
  can consume the exact forward selection without making the mask public.

## Known limitations

- No operation consumes or advances the state until task 0019B1.
- No public checkpoint/import/export or raw-state Tensor accessor is provided.
- No random algorithm or cross-backend bitstream is selected; replay guarantees are bounded as
  documented.
- Counter wrap is modular and caller-managed.
- Graph capture, serialization, execution, gradients, and backend support remain separately owned
  and unimplemented by this task.

## Validation evidence

Planning context `/root/plan_0019b`:

- Read the required architecture, lifecycle, training, module/dependency/runtime-boundary,
  documentation/planning, model capability/master/roadmap, relevant completed task, API, source,
  test, and inventory contracts.
- Confirmed model may own backend-neutral state semantics and Tensor expression construction while
  compiler and prepare/backend/runtime retain their established responsibilities; no architecture
  update is required.
- Confirmed `TensorProducer` supports zero inputs and indexed outputs through family signatures,
  `TensorFactory.createDerived(...)` constructs the required one-output occurrence, and
  `createDerivedOutputs(...)` remains available for Draft dropout's three outputs.
- Confirmed `TensorRandoms` is eager caller-`RandomGenerator` host-data creation and must remain
  unchanged.
- Selected the split because state foundation and dropout are independently reviewable and the
  combined scope would exceed the normal task-size guardrail.
- Created and synchronized exactly four planning-only paths: this specification,
  `capabilities.md`, the model master plan, and the roadmap. `git status --short` shows no Java,
  tests, Gradle, architecture/focused-architecture, API, other-module, or unrelated path.
- A targeted Ruby local-Markdown path check passed for all links in the four changed files.
  Targeted newline, trailing-whitespace, and backtick-fence checks also passed.
- Status scans found exactly one Ready frontier represented consistently by this task, the model
  master-plan row, and the roadmap row. No 0019B1 task file exists; 0019B1 and 0019C–0019E remain
  broad Draft rows, while 0019 through 0019A2 remain Complete.
- Stale-status scans found no remaining statement that 0019B is Draft, that all 0019B–0019E tasks
  are Draft, or that no model task is Ready.
- `git diff --check` passed with no output after final planning edits. Created-file checks were
  covered separately because the new untracked specification is not included in ordinary diff
  output until staged.

Implementation context `/root/task_0019b_implementation`:

- Added exactly `GraphRngKind`, `GraphRngStateAttrs`, and `GraphRngState`, plus focused
  `GraphRngStateSemanticsTest`, `GraphRngStateTest`, and the required global
  `OperationSignatureTest` family entry. No executable path outside that selected set changed.
- Implemented the exact zero-input/one-output `INITIAL_STATE` signature; raw key/counter record;
  fixed storage-free `INT64 Shape[2]` state descriptor; output-index-zero provenance; one-ID
  construction; package-private validated wrapping; and identity-equality public boundary.
- The required focused command selecting all seven test classes passed with `BUILD SUCCESSFUL`
  after the final executable refinement.
- Exactly one final `./gradlew :modules:model:test` then passed with `BUILD SUCCESSFUL in 1s`,
  three actionable tasks, one executed, and two up-to-date. Executable Java did not change after
  that run.

Documentation-agent context `/root/task_0019b_implementation/task_0019b_docs`:

- Independently read the architecture contract and focused architecture pages, documentation and
  planning rules, General/API-Javadoc/Planning/Example profiles, task and directly related
  completed tasks, final source/tests, Tensor/Compile/Runtime/Training APIs, glossary,
  capabilities, master plan, roadmap, and foundational Tensor contracts.
- Finalized Javadocs for all three new production types and the Tensor family/boundary inventory.
  Finalized Tensor API, Compile API, glossary, capabilities, this task, master plan, and roadmap.
  The documentation distinguishes raw unsigned words, opaque state Tensor identity, replay,
  branching/threading, bounded portability, and future serialization from algorithms, sampling,
  execution, and still-Draft dropout.
- Reused the focused and final model-test evidence because this pass changed only comments and
  documentation. `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL in 2s`, two
  actionable tasks executed, after final Javadoc edits.
- Compiled the public example with Java 26.0.1 and ran it successfully; two equal-word state
  initializers printed `false` for both reference and inherited equality. The documented dropout
  sequence is explicitly conceptual and identifies task 0019B1 as its future owner.
- Inspected generated pages for `GraphRngState`, `GraphRngStateAttrs`, and `GraphRngKind`; parameter,
  result, failure, unsigned-word, identity, zero-input/one-output, and serialization boundaries are
  present. `javap -p` and bytecode inspection confirmed the exact one-field/two-method wrapper,
  exact record and enum surfaces, validation order, fixed descriptor, empty inputs,
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
  specs; no model task is currently Ready.

Reasoned no-change reviews:

- TensorFactory, TensorProducer, TensorProvenance, TensorDescriptor, DataType, Shape, and
  TensorRandoms Javadocs remain accurate because the implementation reuses their existing seams
  without changing their public surface, ownership, validation, eager-random behavior, or
  producer/output-index contracts.
- Runtime API and Training API remain accurate because this task adds no prepared/run state,
  execution, dropout, gradient, parameter, optimizer, or training workflow.
- Architecture documents/ADRs/tests, backend conformance, integration tests, Gradle/dependencies,
  and other modules remain accurate because this is one backend-neutral model semantic value and
  expression occurrence with no dependency, lifecycle, backend, runtime, or build change.
- Task 0019B1 and later detailed specs remain absent because only the current frontier may have a
  detailed specification; the broad Draft rows already preserve future ownership.

## Implementation notes

- Used the existing package-private single-output derived-construction seam, so operation
  validation precedes the sole Tensor ID allocation and no storage or generator is introduced.
- Kept state private behind `GraphRngState`; equal attributes express replay-equivalent positions
  while ordinary object identity distinguishes occurrences.
- Documented the future state-threading example without adding or claiming a current dropout API.

## Completion summary

- Completed changes: Added the explicit graph RNG state operation kind, raw attributes, opaque
  public wrapper, focused tests, global signature inventory, finalized Javadocs, API references,
  glossary term, and synchronized planning records.
- Files changed or created: Four production, three test, and seven documentation/planning paths,
  all within the exact authorized inventory.
- Tests and validation: Required focused tests and the final model suite passed in the
  implementation context; model Javadoc, Java 26 examples/surface checks, generated-Javadoc,
  Markdown, scope/status, import/bytecode, formatting, and final diff checks passed in the
  documentation context.
- Documentation and Javadoc impact: Finalized all affected public contracts and recorded reasoned
  no-change conclusions for every required review-only area.
- Unresolved issues: None.
- Follow-up required: None. Draft task 0019B1 owns dropout and state advancement by a consuming
  operation.

Status: Complete
