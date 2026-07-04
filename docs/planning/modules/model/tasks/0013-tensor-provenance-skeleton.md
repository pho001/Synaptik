# Task 0013: Tensor Provenance Skeleton

## Status

Complete

## Goal

Add the smallest immutable model contract that records how a public expression tensor was
produced: one backend-independent `Operation` and an ordered snapshot of its input `Tensor`
objects. Attach that optional provenance to `Tensor` as stable origin metadata and add one
package-private factory path that future operation-family tasks can use without bypassing the
existing JVM-wide `TensorId` allocator.

The contract prepares later compiler-owned graph capture while preserving the architecture rule
that a public `Tensor` is not an IR node. It does not create a graph, assign graph-local identity,
infer descriptors, or implement any mathematical operation.

## Scope

- Add one public immutable `TensorProvenance` record with exactly `Operation operation` and
  ordered `List<Tensor> inputs` components.
- Validate non-null operation/list/elements, snapshot the list, preserve order, and allow empty or
  repeated inputs.
- Add one final `Optional<TensorProvenance>` field and one public `provenance()` accessor to
  `Tensor`.
- Extend the sole package-private Tensor constructor with one provenance optional while preserving
  all existing identity, descriptor, label, host-storage, synchronization, equality, hashing, and
  diagnostic behavior.
- Keep every existing public `TensorFactory` creation/population method leaf-producing with empty
  provenance.
- Add exactly one package-private `TensorFactory.createDerived(...)` method for future
  model-owned expression construction. It allocates identity through the existing allocator,
  attaches one already validated provenance object, and creates no storage.
- Add focused provenance tests and update only the existing exact-shape Tensor and TensorFactory
  tests required by the new contract.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through
  the required independent documentation pass during implementation.

## Out of scope

- any concrete `OperationKind`, family-specific attributes, arithmetic or other expression method,
  descriptor/data-type/shape/layout inference, broadcasting, arity validation, or result checking
- graph capture, graph traversal, topological sorting, cycle detection, common-subexpression
  elimination, `CompiledNode`, `GraphValue`, `CompiledGraphModel`, `NodeId`, `ValueId`, or
  `OperationId` creation or attachment
- producer/consumer indexes, graph membership, output index or multi-output producer grouping,
  publication bindings/plans, compile artifacts, or compiler entry points
- gradient rules, backward flags, gradient Tensor state, trainable role, autograd, optimizer, or
  training behavior
- provenance mutation, replacement, clearing, builder, registry, global lookup, weak references,
  serialization, reflection-based discovery, or a service locator
- storage allocation or attachment for derived tensors, typed access, eager evaluation, execution,
  runtime residency, prepared state, backend ownership/support, device state, kernels, or routes
- changing public factory overloads, eager leaf initializer semantics, ID allocation policy,
  existing operation contracts, graph contracts, dependencies, Gradle, architecture, packages, or
  another module
- implementing or creating the detailed specification for task 0014

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0004](0004-typed-identifiers.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0008](0008-graph-value-and-node-model.md)
- [Task 0009](0009-compiled-graph-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0012I](0012i-bernoulli-random-tensor-creation.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The legacy implementation stored an optional operation and mutable parent-tensor list inside an
internal mutable Tensor node, together with gradient, backward, runtime, and storage behavior. That
is evidence that public expression results need discoverable operation/input origin for graph
capture. It is not a design to copy.

The new contract retains only the selected capability as a new immutable value. It does not copy
legacy source, `TensorNode`, mutable graph setters, gradient rules, backward flags, runtime aliases,
package structure, or layer coupling.

## Architecture constraints

- `modules/model` owns public Tensor state, operation semantics, and the minimal provenance value.
- `Tensor` remains public mutable API state and must not become a `CompiledNode`, `GraphValue`, or
  other IR object.
- `TensorId` remains stable public Tensor identity. Provenance must contain no graph-local
  `NodeId`, `ValueId`, `OperationId`, or owning-graph identity.
- `Operation` remains a backend-independent semantic value and exposes no support, ownership,
  route, execution, compiler, runtime, or backend behavior.
- Provenance is immutable after Tensor construction. Tensor host-storage association remains the
  only mutable state in this task.
- Graph traversal, validation, snapshotting, ID assignment, compilation, and conversion into
  immutable graph records belong to a later compiler task.
- Package direction remains acyclic: `model.tensor` may depend on `model.operation`; operation must
  not depend on Tensor, provenance, or graph state.
- Public factory initializers remain provenance-free leaves. Only the new package-private derived
  path may attach provenance.
- Central `TensorFactory` identity allocation remains the only construction path used by future
  expression helpers; no second allocator, caller-supplied ID, registry, or service is introduced.
- Stop if implementation needs graph identity, output grouping, descriptor inference, operation-
  family validation, mutable provenance, another package/module, dependency, or architecture rule.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor state, TensorFactory identity
  construction, provenance, and same-package focused tests.
- `io.github.pho001.synaptik.model.operation` — supplies the immutable backend-independent
  `Operation` value referenced by provenance.
- `java.util` — supplies `List`, `Optional`, and defensive immutable snapshots.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorProvenance` — public immutable origin DTO because
  public Tensor inspection and a later compiler module must be able to read the semantic producer
  and ordered input tensors.
- `io.github.pho001.synaptik.model.tensor.Tensor` — retains optional provenance as stable metadata
  beside ID, descriptor, and label; host storage remains its sole mutable field.
- `io.github.pho001.synaptik.model.tensor.TensorFactory` — retains centralized identity allocation
  and gains one package-private derived-construction seam for later same-package expression code.
- `TensorProvenanceTest`, `TensorTest`, and `TensorFactoryTest` remain in the same package so they
  can verify package-private constructor/factory boundaries without widening production API.

## Required contract

### TensorProvenance record

Create exactly:

```java
public record TensorProvenance(Operation operation, List<Tensor> inputs)
```

The record has exactly those two components in that order and no additional instance/static state,
nested type, factory, builder, or convenience method. Its canonical constructor validates in this
order:

1. null `operation`: `NullPointerException`, message `operation`;
2. null `inputs`: `NullPointerException`, message `inputs`;
3. first null input in ascending order: `NullPointerException`, message `inputs[<index>]`;
4. snapshot the fully validated list with `List.copyOf(inputs)`.

An empty input list is valid for future zero-input semantic operations. Repeated Tensor references
are valid because an operation may consume the same expression in more than one ordered role.
Input order is semantic and must be preserved. The record retains the exact immutable `Operation`
reference and exact identity-bearing Tensor elements, but it never retains the caller's mutable
list object.

Record-generated `equals`, `hashCode`, and `toString` remain in use. Tensor elements have ordinary
object-identity equality, so provenance equality does not turn equal Tensor IDs into equal input
objects. Diagnostic text is not serialization, dispatch, graph identity, or a stable wire format.

Do not validate operation arity, kind-to-attribute compatibility, input descriptors, output
descriptor, data types, shapes, layouts, gradient intent, cycles, or graph-wide structure. Those
checks need operation-family or compiler context that this local origin DTO does not own.

### Tensor state and constructor

After this task, Tensor has exactly these five fields in this order:

```java
private final TensorId id;
private final TensorDescriptor descriptor;
private final Optional<String> label;
private final Optional<TensorProvenance> provenance;
private HostTensorStorage hostStorage;
```

The first four fields are final. Host storage remains the only mutable field. Replace the sole
package-private constructor with exactly:

```java
Tensor(
        TensorId id,
        TensorDescriptor descriptor,
        Optional<String> label,
        Optional<TensorProvenance> provenance,
        Optional<HostTensorStorage> hostStorage)
```

Validate container references in parameter order with `Objects.requireNonNull` and exact messages
`id`, `descriptor`, `label`, `provenance`, and `hostStorage`. Complete those null checks before
label normalization or storage semantic validation. Preserve the existing label normalization and
then the existing storage type/capacity/liveness behavior and messages unchanged.

Store the supplied non-null provenance optional through ordinary assignment. Treat `Optional` as a
value-based container: no API or test may promise its identity. A present result must contain the
exact immutable `TensorProvenance` reference supplied at construction. Provenance and host storage
are independent: package-private construction may represent a derived tensor with no storage or a
derived tensor with compatible host storage without changing either contract.

Add exactly one declared public method:

```java
public Optional<TensorProvenance> provenance()
```

It returns the immutable optional origin value without synchronization. Existing accessors and
storage methods retain their signatures and behavior; only the three storage association methods
remain synchronized. `equals` and `hashCode` remain inherited object identity. `toString()` remains
exactly metadata-only over Tensor type, ID, descriptor, and normalized label and must not include
provenance, operation, inputs, graph expansion, or storage facts.

### Leaf and derived factory construction

Every existing public TensorFactory method must continue to create a provenance-free leaf. Change
only the existing final constructor call in public `create(...)` so it passes
`Optional.empty()` in the provenance position while passing caller storage unchanged. Do not add a
public provenance parameter or overload and do not change validation, ID, allocation, import,
population, random-source, label, or storage behavior.

Add exactly one package-private method:

```java
static Tensor createDerived(
        TensorDescriptor descriptor,
        Optional<String> label,
        TensorProvenance provenance)
```

Validate `descriptor`, `label`, and `provenance` for null in that order with exact messages matching
the parameter names before allocating an ID. Then allocate exactly once with the existing private
`nextTensorId()`, invoke the new Tensor constructor with `Optional.of(provenance)` and
`Optional.empty()` host storage, and return that exact Tensor.

The method does not inspect the descriptor, operation, or inputs; normalize labels; infer gradient
intent; allocate/attach storage; copy tensors; traverse provenance; or validate semantic
compatibility. Existing Tensor construction remains the sole label semantic path. Consequently:

- null factory arguments consume no ID;
- a blank label consumes the allocated ID and fails through existing Tensor validation;
- allocator exhaustion occurs before Tensor construction and no ID is rolled back or reused;
- successful construction returns no host storage and the exact provenance reference.

The method is a package-private construction seam, not public graph capture, a compiler API, a
registry, or a service locator. Do not change `nextTensorId`, its fields, or exhaustion/concurrency
policy.

## Valid and invalid scenarios

| Scenario | Result |
|---|---|
| Empty provenance input list | Valid zero-input semantic origin |
| Same Tensor appears twice in inputs | Valid; both ordered roles are preserved |
| Caller mutates source list after construction | Provenance remains unchanged |
| Caller supplies a null input element | Invalid with indexed message |
| TensorFactory eager leaf creation | Tensor provenance is empty |
| Package-private derived creation | Exact provenance is present and host storage is empty |
| Derived creation with blank label | Invalid after ID allocation; ID is consumed |
| Tensor storage is attached/replaced/cleared | Provenance remains the same exact value |
| Input Tensor storage later changes or dies | Provenance input identity remains; no storage snapshot is implied |
| Two provenance values use equal Operation and same input objects | Record value equality may be true; this is not producer-occurrence identity |
| Two operations have equal semantic values | Equality does not perform common-subexpression elimination |
| Provenance would require graph-local IDs or cycle checks | Deferred to compiler-owned capture |

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProvenanceTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless an inconsistency requires stopping:

- `docs/api/compile-api.md` — graph capture and compiler entry points remain conceptual and
  unimplemented.
- `docs/api/training-api.md` — no gradient, trainable, optimizer, or session behavior changes.
- `docs/planning/modules/model/capabilities.md` — already selects minimal provenance and legacy
  capability evidence without prescribing this local representation.
- Existing `Operation`, `OperationKind`, `OperationAttrs`, `TensorId`, `TensorDescriptor`,
  `GraphValue`, `CompiledNode`, `CompiledGraphModel`, and `PublicationBinding` Javadocs.
- Focused architecture documentation and architecture tests.

## Maximum scope

At most three production files, three tests, and five documentation/planning files: eleven paths
total. The existing Tensor exact-shape test and TensorFactory exact-method test must change in the
same atomic buildable change because the constructor/field/method contracts evolve together.

Do not modify operation or graph production/tests, any other existing factory behavior test,
Gradle, `AGENTS.md`, `ARCHITECTURE.md`, focused architecture docs/tests, capabilities, another
module, or unrelated documentation. Do not create a task-0014 specification. Stop beyond eleven
paths or when an architecture, multi-output, capture, inference, or operation-family decision is
required.

## Javadoc requirements

- Document `TensorProvenance` as immutable expression-origin metadata, not IR, occurrence identity,
  executable behavior, or graph membership.
- Record/type/constructor/accessor Javadocs must explain exact Operation retention, ordered
  defensive list snapshot, exact Tensor-element identities, empty/repeated input acceptance,
  null/index failures, value equality limits, and deferred validation.
- Update Tensor type/constructor/accessor Javadocs for immutable optional provenance, exact
  reference retention, Optional value semantics, stable lifetime, lack of synchronization need,
  independence from storage mutation, and distinction from graph-local nodes/values/IDs.
- Preserve and explicitly document that host storage remains Tensor's sole mutable state, ordinary
  object equality remains, and diagnostic text omits provenance to avoid graph expansion and is not
  serialization.
- Update TensorFactory type/create Javadocs so public methods are clearly provenance-free eager
  leaves while the package-private derived seam owns no graph capture or semantic validation.
- Document `createDerived` parameters, null order, ID/failure effects, no-storage result, exact
  provenance retention, and all `@return`/`@throws` behavior.
- Review existing operation, descriptor, identity, graph-model, storage, and publication Javadocs;
  record why they remain accurate or stop on an out-of-scope discrepancy.

## Acceptance criteria

- Exactly one public `TensorProvenance` record exists with the two required components/order and no
  extra state/API. Constructor validation, list snapshot, ordering, empty/repeated inputs, record
  methods, and indexed failures match this task.
- Tensor has exactly five fields in the required order, with exactly the first four final and only
  host storage mutable. It retains exactly one package-private five-parameter constructor.
- Tensor declares exactly one new public `provenance()` accessor. It is not synchronized and
  returns empty or the exact underlying provenance reference using Optional value semantics.
- All existing Tensor metadata, label, storage compatibility/lifetime/synchronization, identity
  equality/hashing, and diagnostic behavior remain unchanged.
- Storage transitions, read-only storage, and late storage death do not alter provenance. Tensor
  diagnostic text remains unchanged and contains no operation/input/provenance detail.
- Existing public TensorFactory API and all eager initializer behavior remain unchanged and return
  provenance-free leaves.
- Exactly one package-private `createDerived` method exists with the required signature,
  validation order, centralized ID allocation, no-storage result, exact provenance, and failure
  side effects. No second allocator or public caller-supplied provenance API appears.
- No operation family, expression method, inference, traversal, graph record/ID, gradient,
  compiler, storage allocation, runtime, backend, dependency, package, or architecture behavior is
  introduced.
- Focused provenance/Tensor/factory tests and all existing model/repository tests pass.
- Reflection, `javap`, import, bytecode, package-direction, eleven-path, documentation, and status
  checks pass.
- A separate documentation-focused agent/thread finalizes affected Javadocs, Tensor API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for compile/training API, capabilities, architecture docs, and related contracts.
- Task 0013 becomes Complete only after both passes. The separately coordinated task 0013A remains
  the next Draft planning frontier without a detailed specification; task 0014 also remains Draft
  without a detailed specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must cover exact record/components/API, validation order/messages, mutable-source
snapshot, order, empty/repeated inputs, exact references, record equality limits, exact Tensor
fields/constructor/public methods, absent/present provenance, Optional semantics, storage
independence, unchanged diagnostics/equality, all public factory leaves, `createDerived` null/blank/
exhaustion/ID behavior, and allocator-state restoration after reflective boundary checks. Do not
weaken or remove existing Tensor/storage/factory allocator/concurrency/exhaustion coverage.

Manually inspect `javap -p -c -s` for all three affected production types; verify exact fields,
components, constructor/method visibility and signatures, final/synchronized modifiers,
`List.copyOf`, one `nextTensorId` call in `createDerived`, empty storage, empty public-leaf
provenance, and no extra state. Inspect imports/package direction for forbidden graph/compiler/
planning/runtime/prepare/backend/training dependencies. Validate generated Javadoc, links, anchors,
code fences, terminology, whitespace, exact eleven paths, synchronized planning status, and absence
of a task-0014 spec.

## Dependencies

- Task 0006 supplies the immutable backend-independent `Operation` value.
- Task 0011 supplies Tensor identity/descriptor/label/storage behavior and its package-private
  construction boundary.
- Task 0012 supplies the central JVM-wide ID allocator and public leaf construction path.
- Tasks 0008–0009 define the separate immutable graph model that provenance must not duplicate.

## Follow-up tasks

- Separately coordinated task 0013A owns full-value and identity-matrix tensor creation and remains
  the next Draft frontier without a detailed specification.
- The post-0013A model foundation checkpoint must explicitly decide whether task 0014 remains the
  next sequential frontier or a documented cross-module vertical slice should begin.
- Task 0014, if selected next, owns concrete elementwise operation kinds/attributes, descriptor
  semantics, public expression construction, and use of `createDerived`.
- A later compiler task owns provenance traversal, graph-local ID allocation, graph validation,
  immutable graph snapshotting, publication planning, and compile artifacts.
- Multi-output public producer grouping/output-slot semantics must be planned only when a concrete
  operation task requires them; this skeleton does not silently choose that architecture.

Do not create detailed follow-up specifications in this task.

## Architecture impact

Expected impact: None. The architecture already assigns minimal public Tensor origin metadata and
backend-independent Operation semantics to `modules/model`, while graph capture belongs to the
compiler. This task implements that boundary without changing module ownership or dependencies.

If implementation requires graph-local identity, compiler traversal, multi-output producer
identity, mutable provenance, or another module/package direction, stop and report the required
decision before editing architecture or expanding scope.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0004/0006/0007/0008/0009/0011/0012/0012I/0013, Tensor API,
Compile API, glossary, current Operation/Tensor/TensorFactory/graph contracts and focused tests,
and Java 26 Gradle configuration.

Implement task 0013 exactly. Add only TensorProvenance.java for the new production concept; modify
only Tensor.java and TensorFactory.java otherwise. Add TensorProvenanceTest and update TensorTest
and TensorFactoryTest only for the exact new state/constructor/accessor/package-private factory
surface. Preserve all existing eager initializer, allocator, storage, and operation/graph behavior.

TensorProvenance is exactly a public record of Operation plus an ordered immutable List<Tensor>.
Follow exact null/index validation, snapshot/order/reference, empty/repeated input, and record-value
rules. Tensor gets immutable Optional provenance, one non-synchronized accessor, and the exact new
constructor while host storage remains its sole mutable state. Public factory paths remain empty-
provenance leaves. Add exactly one package-private createDerived that uses the existing allocator,
attaches exact provenance, creates no storage, and performs no inference/traversal/semantic check.

Do not add operation families/expression methods, graph IDs/records/capture/traversal, descriptor
inference, output grouping, gradients/autograd, storage/evaluation, compiler/runtime/backend work,
dependencies/build/architecture changes, or later specs. Stop beyond eleven paths or on
architecture/multi-output uncertainty.

Run every specified focused/aggregate test, Javadoc, bytecode/import/manual, documentation/link/
whitespace/scope/status check. Then hand the actual diff and validation evidence to a separate
clean-context documentation agent/thread in the same change. It must independently inspect source,
tests, generated Javadoc, and evidence; finalize permitted Javadocs/Tensor API/glossary/planning;
record reasoned compile/training API, capability, architecture, and existing-contract no-change
conclusions; and rerun validation.

Update only this task, model master plan, and roadmap for planning status/evidence. Do not mark
0013 Complete until both passes and all validation succeed. Leave task 0014 Draft without a
detailed specification. Do not commit or push.
```

## Local decisions

- One `TensorProvenance` value contains only semantic operation plus ordered inputs; Tensor already
  owns output descriptor and identity, so duplicating them would invite inconsistency.
- Provenance is final optional Tensor metadata. A setter would permit origin changes/cycles during
  capture and repeat the legacy mutable-node coupling.
- Inputs are strong identity references in an immutable list snapshot. Weak references could lose
  the graph before compilation; copying Tensor objects would destroy expression identity.
- Empty and repeated inputs are valid local structures. Arity and graph-wide validity require
  future semantic/compiler context.
- Public factory methods remain leaves. One package-private `createDerived` centralizes ID
  allocation without exposing caller-supplied provenance or implementing expressions early.
- No output slot or producer occurrence ID is added. Current operation-family planning can model
  one public result per provenance; a real multi-output need must receive an explicit later design.

## Known limitations

- Provenance holds strong references to the reachable input expression DAG until tensors become
  unreachable; no pruning or weak-reference policy exists.
- This skeleton does not prove acyclicity, descriptor compatibility, operation arity, or graph
  validity and cannot compile or execute an expression.
- Record value equality is not producer-occurrence identity and must not be used as automatic
  common-subexpression elimination.
- Public factory output remains leaf-only. Concrete public derived tensors arrive with later
  operation-family tasks.
- Multi-output producer grouping/output slots, graph-local IDs, gradient rules, and capture
  snapshots are deferred to owning tasks.

## Validation evidence

Planning reviewed the architecture/model boundaries, lifecycle, current Operation/Tensor/factory
and immutable graph contracts, completed task chain, capability baseline, Tensor/Compile API,
glossary, exact-shape tests, and read-only legacy Tensor/TensorNode/traversal evidence. The proposed
contract stays in `model.tensor`, adds no dependency or architecture rule, preserves central ID
allocation, and fits eleven implementation/documentation paths.

- Implementation context `/root/implement_model_0013` added `TensorProvenance`, extended `Tensor`
  and `TensorFactory`, and updated only the three focused test classes. Independent documentation
  context `/root/implement_model_0013/review_model_0013_docs` then inspected the actual source,
  tests, generated Javadoc, XML reports, bytecode, build configuration, and complete workspace
  diff. It applied General plus API/Javadoc style to Java, Tensor API, and glossary work and
  Planning style to this task, the model master plan, and roadmap. Example format was reviewed but
  no executable example changed.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorProvenanceTest` — `BUILD SUCCESSFUL`; 6 tests, zero
  failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; 14 tests, zero failures,
  errors, or skips. The final test also proves that input storage may die and be cleared without
  changing retained provenance, and that all optional-container null checks precede label
  normalization.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` — `BUILD SUCCESSFUL`; 8 tests, zero
  failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 33 suites and 251
  tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` without Javadoc errors. Generated pages
  contain `TensorProvenance`, `Tensor.provenance()`, immutable origin/list ownership, sole mutable
  host storage, provenance-free public leaves, and the graph-capture boundary. Source review covers
  the package-private Tensor constructor and `createDerived`, which public/protected Javadoc does
  not render.
- `./gradlew test` — `BUILD SUCCESSFUL` for the repository; the validation run reported 36
  actionable tasks with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for `TensorProvenance`,
  `Tensor`, and `TensorFactory` confirmed exactly two final record fields and one `List.copyOf`,
  exactly five Tensor fields in required order with only storage mutable, the five-argument
  package-private constructor and direct unsynchronized provenance accessor, and only the three
  storage methods synchronized. Targeted `createDerived` bytecode contains descriptor/label/
  provenance null checks, one `nextTensorId()` call, `Optional.of(provenance)`, empty storage, and
  one Tensor construction, with no traversal, inference, or semantic validation.
- Import and package-direction scans found only the existing model/JDK dependencies plus the new
  permitted `model.tensor -> model.operation` edge. No graph, compiler, planning, runtime, prepare,
  backend, engine, trace, or training dependency was introduced; operation and storage packages do
  not depend back on Tensor provenance.
- The corrected local Markdown checker resolved 173 file targets and heading anchors across the
  five changed documentation/planning files with zero errors. Two preliminary checker invocations
  stopped before completing because one Ruby regular expression interpolated a heading quantifier
  and the installed Ruby lacks `Array#filter_map`; the compatible corrected checker produced the
  passing result. Backtick fence counts are even, no tilde fence is present, terminology/status
  checks found no stale planned-provenance claim, targeted trailing-whitespace scans found no
  matches, and `git diff --check` passed.
- Task-specific scope contains exactly the authorized eleven paths: three production files, three
  focused tests, Tensor API, glossary, this task, model master plan, and roadmap. A concurrent
  unrelated edit in `capabilities.md` adds only task-0013A factory-capability bullets; it was clean
  at documentation handoff, does not overlap provenance, was preserved without editing, and is
  excluded from task 0013's path count by coordinator direction. Concurrent task-0013A Draft
  planning in the already-authorized master plan and roadmap was also preserved and merged with
  task 0013 status. Task 0014 remains Draft and no task-0014 specification exists.
- `docs/api/compile-api.md` remains unchanged because task 0013 adds no capture entry point,
  compiler behavior, graph-local allocation, compile artifact, or publication plan.
  `docs/api/training-api.md` remains unchanged because no gradient, trainable, autograd, optimizer,
  or session behavior changed. `capabilities.md` already selected minimal provenance and needed no
  provenance edit; its unrelated concurrent task-0013A addition is described above.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests remain
  unchanged because provenance stays in the model, graph capture remains compiler-owned, and no
  module boundary, dependency rule, lifecycle, backend behavior, or end-to-end contract changed.
- Existing `Operation`, `OperationKind`, and `OperationAttrs` Javadocs remain accurate because
  provenance retains an Operation without changing semantic-kind/attribute validation or adding
  occurrence identity. `TensorId` and `TensorDescriptor` remain accurate because identity and
  logical descriptor semantics did not change. `GraphValue`, `CompiledNode`,
  `CompiledGraphModel`, and `PublicationBinding` remain accurate because provenance adds no
  graph-local ID, membership, capture, producer index, or publication context. `HostTensorStorage`
  and `MemorySegmentStorage` remain accurate because host storage ownership, sizing, liveness, and
  raw-memory behavior did not change.

## Implementation notes

- Added the exact public `TensorProvenance(Operation, List<Tensor>)` record with deterministic null
  checks, ordered immutable snapshot ownership, exact element/reference retention, empty/repeated
  input support, and record value semantics.
- Added final optional provenance to `Tensor`, a direct unsynchronized accessor, and the exact
  five-argument package-private constructor while retaining host storage as the sole mutable state,
  ordinary object equality, and provenance-free metadata diagnostics.
- Kept every public factory construction/population path provenance-free and added one
  package-private `createDerived` seam that reuses central ID allocation, attaches exact provenance,
  creates no storage, and performs no graph or semantic work.
- Finalized affected Javadocs, Tensor API, glossary, and planning status. The public reference now
  distinguishes implemented origin metadata from still-planned concrete expression construction
  and compiler capture.

## Completion summary

- Completed changes: Implemented and documented immutable Tensor provenance and the bounded
  derived-construction seam without turning Tensor into graph IR.
- Files changed or created: `TensorProvenance.java`, `Tensor.java`, `TensorFactory.java`, their
  three focused tests, Tensor API, glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused suites passed 6/6, 14/14, and 8/8; all 251 model tests, model
  Javadoc, root tests, bytecode/API/import/package checks, generated documentation, 173 Markdown
  link/anchor checks, fence/terminology/whitespace checks, scope/status checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0013/review_model_0013_docs` completed the independent pass using General,
  API/Javadoc, and Planning profiles; no executable example changed.
- Documentation impact: Tensor API and glossary now describe current immutable provenance, public
  factory leaves, the internal derived seam, record equality limits, and compiler/graph boundaries.
  Compile API, Training API, capabilities, architecture documents/tests, and unrelated contracts
  required no provenance-specific edit for the reasons recorded above.
- Javadoc review: `TensorProvenance`, affected `Tensor`, and affected `TensorFactory` contracts are
  final; related operation, identity, descriptor, graph, publication, and storage Javadocs remain
  accurate unchanged.
- Glossary impact: The implementation-status convention, Provenance, Tensor, Tensor factory, and
  Tensor-versus-graph-value distinction now reflect the implemented contract.
- Unresolved issues: None for task 0013. The preserved unrelated `capabilities.md` and task-0013A
  planning changes belong to their separate coordinated work.
- Follow-up required: None for task 0013. Task 0013A is the next Draft planning frontier without a
  detailed specification; task 0014 remains Draft without a detailed specification.

Status: Complete
