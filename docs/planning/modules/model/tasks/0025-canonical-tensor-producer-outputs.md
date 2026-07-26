# Task 0025: Canonical TensorProducer outputs

## Status

Complete

## Goal

Make every derived `TensorProducer` retain and return the canonical exact `Tensor` wrapper created
for each ordered output slot, including hidden auxiliary outputs omitted from public ergonomic
result carriers.

Complete producer, output wrappers, and indexed provenance as one factory-owned unpublished
occurrence. Preserve public Tensor methods, current result carriers, one shared producer,
output-index provenance, descriptor identity, and existing opaque Tensor-ID behavior.

This is the model prerequisite for Compiler task 0004's pre-capture Tensor-expression autograd.
It does not implement gradients or compiler behavior.

## Scope

- Add one private final immutable ordered output-Tensor snapshot to `TensorProducer`.
- Add exactly one public indexed retrieval method:

  ```java
  Tensor output(int outputIndex)
  ```

- Return the exact canonical wrapper for the requested slot; never reconstruct a wrapper.
- Keep `outputCount()` derived from `outputDescriptors().size()`.
- Refactor package-private derived construction so `TensorFactory` initiates one atomic producer
  construction that:
  - validates operation, ordered inputs, output descriptors, and signature before allocating an
    ID;
  - allocates one ID and creates one Tensor wrapper per output position;
  - creates each wrapper with indexed provenance pointing to the same producer under construction;
  - completes the producer's final output snapshot before any wrapper or producer is returned; and
  - returns the retained canonical wrappers from the producer.
- Preserve the existing single-output optional label and the unlabeled multi-output behavior.
- Preserve current validation order and ID-consumption rules as far as they are observable.
- Verify canonical hidden-output retrieval for dropout and batch-normalization training.
- Finalize affected Javadoc, Tensor API, Compile API, glossary, capabilities, task evidence,
  model master plan, and roadmap through the required separate documentation-focused pass.

## Out of scope

- any Tensor method, Tensor constructor signature, or ergonomic result-carrier signature change
- exposing the dropout mask or batch-normalization saved statistics in public result carriers
- adding `outputs()`, streams, iterators, arrays, maps, sibling registries, or name-based lookup to
  `TensorProducer`
- wrapper reconstruction from a descriptor, producer, provenance, or output index
- changing operation semantics, output descriptors, output order, gradient eligibility, labels,
  storage, or public expression validation
- `Tensor.gradient`, `Tensor.backward`, mutable gradient fields, gradient lifecycle, or a tape
- gradient rules, reverse traversal, accumulation, graph capture, graph phases, `NodeId`,
  `ValueId`, compiler state, or a compiler dependency in model
- weak-reference ownership, explicit cycle breaking, close behavior, finalizers, cleaners, or
  garbage-collection tests
- public producer builders, factories, registries, facades, interning, structural equality, or
  serialization identity
- Java module dependencies, Gradle changes, architecture tests, conformance tests, integration
  tests, or another module
- implementing or creating a detailed specification for Compiler task 0004

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially the model, compiler,
  training-graph, autograd, and compile-lifecycle sections
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0019B1](0019b1-explicit-graph-dropout-construction.md)
- [Task 0021C](0021c-batch-normalization-training-and-statistic-transition.md)
- [Task 0023F](0023f-scaled-dot-product-attention-weights-output.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns Tensor expression state, producer occurrence identity, TensorFactory
  construction, and provenance. It must not depend on compiler.
- A producer is pre-capture expression identity, not `CompiledNode`, graph membership, a
  canonicalization key, or a service.
- Tensor identity, descriptor, label, and provenance remain immutable. Its existing mutable
  borrowed host-storage association does not justify another mutable field.
- Tensor owns no gradient or backward lifecycle state.
- Every producer output slot has exactly one canonical wrapper. The exact wrapper returned by
  `TensorFactory` must be the same object returned by `producer.output(index)`.
- All output wrappers have indexed provenance pointing to that exact producer and retain the exact
  descriptor reference from the same slot.
- The intentional object cycle is:

  ```text
  Tensor -> TensorProvenance -> TensorProducer -> outputs -> Tensor
  ```

  It contains only ordinary Java references and immutable expression metadata. The complete
  occurrence must become externally reachable only after all final fields are assigned.
  Unreachable cycles remain eligible for ordinary garbage collection.
- The construction implementation may let wrappers refer to the producer while its constructor is
  active only when neither the producer nor any wrapper escapes the factory-owned construction
  boundary before the final output snapshot is assigned. Do not publish through static state,
  callbacks retained beyond construction, another thread, or caller-visible collections.
- Producer/signature validation remains before first ID allocation. Failure after ID allocation
  does not roll back or reuse IDs.
- Cross-module compiler access is satisfied by the one public indexed accessor. Do not broaden
  public Tensor or result-carrier APIs.
- If safe publication requires mutable output state, post-construction initialization, a registry,
  or another module dependency, stop and report the architecture conflict.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.tensor` — owns Tensor, producer, provenance, factory, result
  carriers, and expression construction.

Packages added or changed:

- No package is added.
- Only the existing tensor package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorProducer` — owns the final canonical output
  snapshot because one producer occurrence is the identity shared by every result position.
- `io.github.pho001.synaptik.model.tensor.TensorFactory` — remains the sole package-level derived
  construction boundary and ID allocator.
- `io.github.pho001.synaptik.model.tensor.TensorProvenance` — keeps its exact producer and
  output-index association unchanged.
- `io.github.pho001.synaptik.model.tensor.Tensor` — keeps its public declaration and existing
  immutable metadata plus mutable borrowed-storage behavior unchanged; Javadoc only may need cycle
  clarification.

Test placement:

- `TensorProducerTest` — exact field/method shape, indexed retrieval, canonical identity,
  immutability, bounds, and safe-publication observations.
- `TensorFactoryTest` — single/multi-output canonical construction, validation order, labels, and
  ID side effects.
- `TensorProvenanceTest` — exact producer/output/descriptor consistency.
- `TensorDropoutExpressionTest` and `TensorBatchNormTrainingExpressionTest` — exact retrieval of
  hidden same-occurrence auxiliaries.

## Required contracts

### Producer state and accessor

`TensorProducer` gains exactly:

```java
private final List<Tensor> outputs;
```

alongside its current final operation, inputs, and output descriptors. It exposes:

```java
public Tensor output(int outputIndex);
```

The method:

- rejects a negative index;
- rejects an index greater than or equal to `outputCount()`;
- reports the requested index and available output count in failures;
- returns the exact retained Tensor reference at that slot; and
- performs no allocation, traversal, inference, reconstruction, or mutation.

Do not expose the output list itself. `outputCount()` remains descriptor-derived; do not add a
count field.

### Factory-atomic cycle construction

Refactor only the package-private derived-construction path. The concrete constructor, helper, or
construction-local callback shape is an implementation-local decision. This is feasible because
`TensorProvenance` and `Tensor` validation consult only the producer's already-assigned output
count and descriptor snapshot while wrappers are constructed; neither needs the producer's output
snapshot before its final assignment. The selected implementation must satisfy this sequence:

1. `TensorFactory` validates its public/package-private arguments in the current order.
2. Producer construction snapshots and validates operation, inputs, output descriptors, and
   signature cardinality before requesting the first Tensor ID.
3. The producer has its operation, input snapshot, and descriptor snapshot available while it
   creates output wrappers in ascending slot order.
4. Each wrapper receives one newly allocated ID, its exact slot descriptor, current label policy,
   no storage, and `TensorProvenance(producer, slot)`.
5. No wrapper, producer, caller-owned collection, static field, or other thread can observe the
   producer before its immutable output snapshot is assigned.
6. The factory returns `producer.output(0)` for single-output construction and an immutable list
   assembled from `producer.output(0..outputCount-1)` for multi-output construction.

Construction-local callbacks or allocators must not be retained in producer state. Do not add a
mutable builder phase or initialize `outputs` after producer construction returns.

### Identity and lifecycle

- A current single-output expression still allocates one producer, one wrapper, one provenance,
  and one ID.
- A current N-output expression still allocates one producer, N wrappers, N provenance values, and
  N IDs in ascending output order.
- All public result carriers retain their current exact wrapper components.
- Keeping any public result wrapper alive may now keep its sibling wrappers alive through the
  producer. This is the intended cost of exact hidden-output access.
- When the whole occurrence is unreachable, the cycle owns no external resource and is ordinarily
  collectable. Do not add an unreliable forced-GC acceptance test.

### Failure and ID effects

- Null container/element, empty output, and signature-cardinality failures occur before the first
  ID request.
- Existing single-output blank-label failure continues to consume its allocated ID.
- Multi-output identifier exhaustion may consume IDs allocated for earlier slots, returns no
  partial list, and does not publish a partial producer.
- IDs remain opaque; tests must not introduce adjacency as a public contract beyond existing
  focused allocation-side-effect checks.

## Affected files

Expected production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProducerTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProvenanceTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDropoutExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormTrainingExpressionTest.java`

Expected documentation and planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Expected no-change reviews:

- `docs/api/training-api.md` — no training API changes.
- `ARCHITECTURE.md`, focused architecture pages, and ADR 0009 — already authorize and explain
  this prerequisite; implementation must not change architecture.
- `testing/architecture-tests/` — no module or dependency rule changes.
- compiler source/tests — Compiler 0004 remains Draft.
- `TensorScaledDotProductAttentionExpressionTest` and `TensorTopKExpressionTest` — their public
  carriers already expose every output and current factory/producer tests cover general visible
  multi-output identity; change only if a concrete regression is otherwise untestable.

## Maximum scope

This task may create or modify at most:

- the four production tensor-package files listed above;
- the five focused model tests listed above;
- the seven documentation/planning files listed above; and
- no Java file in another package or module.

The 16-path allowance is a cohesive foundation exception: producer state, factory construction,
indexed provenance, hidden-output examples, public contract documentation, and frontier status
must agree atomically. Do not use the allowance for operation-family changes, broad test cleanup,
or unrelated documentation.

If another production type, package, module, public result carrier, or Tensor method is needed,
stop and propose a follow-up or request an architecture decision.

The coordinating planning context later authorized three status-only consistency exceptions in
already-modified documentation: `docs/planning/modules/compiler/master-plan.md`,
`docs/api/training-api.md`, and `docs/design/notes/autograd-strategy.md`. Those corrections say
only that Model 0025 is Complete and Compiler 0004 remains Draft awaiting a dedicated planning
pass. They add no compiler or training behavior and do not expand Java scope.

## Acceptance criteria

- `TensorProducer` has exactly four private final instance fields: operation, inputs, output
  descriptors, and canonical outputs.
- Its public surface adds only `output(int)` to the current four accessors.
- `output(int)` validates both bounds and returns the exact retained wrapper without allocation.
- Every factory-created derived Tensor is identical to
  `tensor.provenance().orElseThrow().producer().output(outputIndex)`.
- Single-output expression behavior, optional label, ID count, exact descriptor, no storage, and
  provenance index zero remain unchanged.
- Multi-output factory results are immutable and each list element is the exact producer-retained
  wrapper at the same slot.
- Every output wrapper has the exact shared producer, matching output index, and exact slot
  descriptor.
- Dropout producer slot one returns the exact hidden BOOL mask wrapper from the original
  occurrence; it is not reconstructed or added to `DropoutResult`.
- Batch-normalization training producer slots three and four return the exact hidden saved mean
  and inverse-standard-deviation wrappers; they are not added to
  `BatchNormTrainingResult`.
- Producer/signature validation precedes first ID allocation; later failures preserve the
  documented non-rollback behavior and publish no partial result.
- Concurrently published successful derived results always expose a complete producer with every
  canonical slot and exact provenance relationship.
- The cycle uses only final producer state and current immutable provenance; no post-construction
  mutation, registry, weak-reference scheme, resource cleanup, or graph identity is added.
- Tensor methods and public ergonomic result-carrier signatures remain unchanged.
- No gradient, compiler, runtime, backend, dependency, Gradle, or architecture-test behavior is
  added.
- Javadocs define identity, indexed bounds, construction order, safe publication, retention cost,
  failure/ID effects, and absence of graph/gradient/runtime meaning.
- The separate documentation-focused pass finalizes all affected API, glossary, capabilities,
  task, master-plan, and roadmap text and records reasoned no-change conclusions.
- Task, master plan, and roadmap statuses agree only after implementation, tests, documentation
  review, and validation complete.
- Compiler 0004 remains Draft and has no detailed task specification.

## Tests / validation

Implementation-focused tests may be run while developing. After executable Java stabilizes, run
one final module suite:

```bash
./gradlew :modules:model:test
```

The separate documentation-focused pass then runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also checks affected Markdown links and anchors, balanced fences, final
newlines, terminology, exact status synchronization, exact allowed paths, and the absence of the
obsolete no-producer-output/current-placeholder design.

Repository-wide tests are deferred to CI because this task changes one module and no dependency,
build, or architecture boundary. Architecture tests are not run because dependency rules do not
change. The documentation pass reuses the final model-test evidence unless it changes executable
Java behavior or records a concrete reason.

Automated tests must cover:

- exact producer field and public-method shape;
- indexed bounds and canonical exact-reference retrieval;
- single- and multi-output factory construction;
- descriptor/provenance identity agreement;
- validation order and ID side effects;
- dropout and batch-normalization hidden outputs; and
- safe publication under concurrent successful construction or observation.

## Dependencies

- Task 0018L: shared multi-output producer and indexed provenance — Complete.
- Task 0019B1: hidden same-occurrence dropout mask — Complete.
- Task 0021C: hidden batch-normalization saved statistics — Complete.
- Task 0023F: explicit same-occurrence attention weights — Complete.
- Task 0024A: completed historical model-milestone documentation closure and immediate
  planning-order predecessor — Complete.
- ADR 0009: accepted pre-capture autograd architecture requiring canonical hidden-output wrappers.

## Follow-up tasks

- Compiler 0004 — implement fail-closed compiler-owned pre-capture Tensor-expression autograd,
  one phase-aware combined capture, validation, and combined exact optimization. Required.
- Compiler 0004A — extend exact-composition gradient formulas after the core pipeline is proved.
- Compiler 0004B — select derivative policies and add policy-dependent formulas.
- Compiler 0006 — later define explicit objectives, targets, seeds, and higher derivative order
  after the public compile/artifact boundary is stable.

Do not create detailed specifications for those follow-ups in this task.

## Architecture impact

Expected impact: None during implementation.

The coordinated planning change has already updated `ARCHITECTURE.md`, focused architecture
documentation, and ADR 0009. This task implements the model-owned prerequisite without changing
module ownership or dependency direction. Architecture tests require no update.

If implementation requires mutable producer outputs, Tensor gradient state, compiler imports,
another module, or a dependency-rule change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md,
docs/planning/modules/model/master-plan.md, docs/planning/roadmap.md,
docs/planning/modules/model/tasks/0018l-shared-multi-output-tensor-provenance.md,
docs/planning/modules/model/tasks/0019b1-explicit-graph-dropout-construction.md,
docs/planning/modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md,
ADR 0009, and this task specification.

Implement task 0025 exactly as specified. Do not implement Compiler 0004 or any out-of-scope item.
Stop on an architecture, safe-publication, or scope conflict. Do not commit or push.

After executable Java stabilizes and the model test suite passes, hand the actual diff and exact
test evidence to a separate documentation-focused agent/thread with clean context. That pass must
inspect final source/tests, finalize affected Javadocs, Tensor/Compile API, glossary, capabilities,
task/master/roadmap status and documentation validation in the same overall change, and must not
repeat successful Java tests unless executable behavior changes or a concrete risk requires it.

Do not mark 0025 Complete until both passes and all acceptance criteria succeed. Keep Compiler
0004 Draft without a detailed task specification.
```

## Local decisions

- `TensorProducer` owns exactly four private final fields. Its new `outputs` snapshot is assigned
  once in the constructor after wrappers are created in ascending slot order; there is no builder
  phase or post-construction mutation.
- One package-private five-argument producer constructor accepts the first-output label and a
  construction-local `Supplier<TensorId>`. The supplier is not retained. The existing
  package-private three-argument constructor delegates with an empty label and the factory's
  allocator, preserving its declaration for existing tensor-package callers.
- `TensorFactory.nextTensorId()` changed from private to package-private so both ordinary factory
  creation and the existing three-argument producer construction seam use the one JVM-wide
  allocator. This is the smallest truthful non-public collaboration: keeping it private would
  require changing callers outside this task, reflection, duplicate or non-global IDs, another
  production type or mutable state, or a redundant forwarding abstraction. Reflection confirms
  that the public factory API did not change.
- Single-output creation returns `producer.output(0)`. Multi-output creation returns an immutable
  list assembled only from `producer.output(index)`, so factory results and producer lookup cannot
  diverge by object identity.
- Producer/signature validation remains before the first ID request. Output wrappers then use the
  exact slot descriptor, output-zero label policy, no storage, and indexed provenance. The final
  output snapshot is assigned before factory return, preserving safe publication without a
  registry, weak reference, or mutable initialization phase.

## Known limitations

- Retaining any derived output may retain every sibling output, the producer, provenance, and
  producer inputs. This is the intentional cost of exact hidden-output identity.
- The immutable cycle owns no external resource and remains ordinarily garbage-collectable when
  unreachable, but this task deliberately adds no unreliable forced-garbage-collection test.
- Tensor IDs remain opaque. Allocation after the validation boundary is monotonic and
  non-rollback, but adjacency is not a public semantic contract.
- Hidden outputs are available through `TensorProducer.output(int)` only. Existing ergonomic
  result carriers remain unchanged.
- Compiler 0004 remains Draft without a detailed specification or implementation. This task
  supplies only its model prerequisite.

## Validation evidence

- Implementation context `/root/implement_model_0025` ran
  `./gradlew :modules:model:test`: `BUILD SUCCESSFUL in 2s`; three actionable tasks, two executed
  and one up-to-date. The suite covered exact field/public-method shape, indexed bounds, canonical
  single- and multi-output identity, descriptor/provenance agreement, validation and ID effects,
  concurrent successful publication, the hidden dropout mask, and hidden batch-normalization
  saved statistics.
- Documentation context `/root/implement_model_0025/finalize_model_0025_docs` ran
  `./gradlew :modules:model:javadoc`: `BUILD SUCCESSFUL in 2s`; two actionable tasks, both
  executed. Generated Javadoc includes the new indexed accessor and package-owned allocator seam.
- A targeted local Markdown validator resolved local paths and anchors and checked balanced
  fences and final newlines for all ten affected Markdown documents; all ten passed. A full stale
  status/design scan then returned no obsolete readiness, no-output, or planned-cycle wording.
- `javap -public` reports exactly five public `TensorProducer` methods: the four existing accessors
  plus `output(int)`. `javap -public` reports no `TensorFactory.nextTensorId()`, while
  `javap -private` reports it as package-private alongside the two existing package-private
  derived-construction seams.
- The final scope audit found exactly 19 coordinated task paths: 16 task-authorized paths and the
  three coordinator-approved status-only exceptions. It separately accounted for nine preserved
  pre-existing architecture paths and the six untouched untracked compiler prototypes.
- Task 0025, its model-master row, and its roadmap row all report Complete; Compiler 0004 remains
  Draft in the compiler master plan. `git diff --check` passed.

## Implementation notes

- `TensorProducer.output(int)` reports both the requested index and available output count for
  either bound failure and otherwise performs one indexed read of retained final state.
- Dropout output one is the exact hidden `BOOL` keep-mask wrapper from the original occurrence.
  Batch-normalization training outputs three and four are the exact hidden saved-mean and
  saved-inverse-standard-deviation wrappers. No carrier component was added.
- Existing single-output labels, output order, descriptor references, provenance indices,
  storage-free results, identifier count, and non-rollback behavior remain unchanged.
- Javadocs and API documentation now state the canonical identity, construction/validation order,
  safe-publication boundary, sibling-retention cost, ordinary cycle collection, failure/ID
  effects, and absence of graph, gradient, compiler, backend, runtime, or execution meaning.
- The implementation and substantive documentation stayed within the 16 task-authorized paths:
  four tensor production files, five focused tensor tests, and seven documentation/planning files.
  Three coordinator-approved status-only consistency exceptions corrected already-modified
  compiler-master, Training API, and autograd-strategy prose, for 19 coordinated paths total.

## Completion summary

- Completed changes: `TensorProducer` now owns and returns one canonical exact Tensor wrapper for
  every output position; `TensorFactory` constructs and publishes complete occurrences through
  that indexed identity; focused tests cover ordinary, hidden, failure, and concurrent-publication
  behavior; Javadocs, Tensor/Compile APIs, glossary, capability baseline, task, model master plan,
  and roadmap describe the final contract.
- Files changed or created: the four production, five test, and seven documentation/planning paths
  listed under [Affected files](#affected-files), with this task specification created as the
  sixteenth path, plus the three coordinator-approved status-only paths recorded under
  [Maximum scope](#maximum-scope).
- Public API: `TensorProducer.output(int)` is the only new public method. `TensorFactory`'s
  allocator is package-private, so its visibility change adds no public API. Tensor methods,
  constructors, descriptors, provenance components, operation signatures, and public result
  carriers are unchanged.
- Documentation no-change review: no Training API behavior changed because no training session,
  gradient publication, parameter, optimizer, or result-carrier contract changed; its only edit is
  the coordinator-approved Model 0025 status correction. `ARCHITECTURE.md`, the current/focused
  architecture pages, and ADR 0009 already authorize and explain this model prerequisite; no
  architecture decision changed. The autograd strategy note likewise receives status wording
  only.
- Validation no-change review: architecture tests need no update because module ownership and
  dependencies did not change. Backend-conformance and integration tests need no update because
  no backend or end-to-end behavior exists here. Repository-wide tests remain deferred to CI under
  the normal single-module validation tier.
- Compiler no-change review: compiler source and tests remain unchanged; Compiler 0004 remains
  Draft with no detailed task specification or implementation. The compiler master plan receives
  only the coordinator-approved status correction that completed Model 0025 does not
  automatically advance compiler work.
- Other no-change review: build and Gradle configuration, every other module, Tensor methods,
  descriptors, public carriers, operation semantics, input/output order, and label/storage policy
  remain unchanged. `TensorScaledDotProductAttentionExpressionTest` and
  `TensorTopKExpressionTest` need no update because their carriers already expose every output;
  the general visible multi-output identity cases remain covered by producer/factory tests.
- Unresolved issues: none.
- Required follow-up: Compiler 0004 remains the required Draft follow-up after a future planning
  step creates its detailed specification.

Status: Complete
