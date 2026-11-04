# Task 0025: [Canonical](../../../../glossary.md#term-canonical-form) [TensorProducer](../../../../glossary.md#producer--tensorproducer) outputs

**Editorial reading key (current context, not a status update).**

This completed historical task concerns canonical expression identity, not execution or gradients. A
producer is one operation occurrence; provenance identifies its zero-based output slot; a canonical
wrapper is that slot’s exact Tensor object, not an equivalent reconstruction. TensorFactory calls
package-private producer construction with input references and output descriptions, then returns
retained wrappers only after the occurrence is complete. Descriptors describe results; no computed
values cross this construction boundary. For example, dropout slot one retrieves the original hidden
BOOL mask wrapper, not sampled mask values. Follow [indexed
provenance](../../../../reference/tensor/identity.md#expression-producers-and-indexed-provenance)
for the current identity contract.

The historical four-field producer scope does not describe every current attachment: the current
reference also documents optional general-scan body containment. Model semantics defines operation
meaning; execution providers realize supported routes rather than redefine it. Semantic
responsibility is not storage ownership: these storage-free expressions neither execute nor own
result buffers. Retaining one wrapper can retain siblings without implying shared physical storage.
Follow [hidden-output
identity](../../../../reference/tensor/identity.md#multi-output-and-hidden-output-identity) for
dropout and batch-normalization examples without expanded public carriers.

## Status

Complete

## Goal

Make every derived [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) retain and return the [canonical](../../../../glossary.md#term-canonical-form) exact [`Tensor`](../../../../glossary.md#tensor) wrapper created
for each ordered output slot, including hidden auxiliary outputs omitted from public ergonomic
result carriers.

Complete [producer](../../../../glossary.md#term-producer), output wrappers, and indexed [provenance](../../../../glossary.md#provenance) as one factory-owned unpublished
[occurrence](../../../../glossary.md#term-occurrence). Preserve public Tensor methods, current result carriers, one shared producer,
output-index provenance, [descriptor](../../../../glossary.md#term-descriptor) [identity](../../../../glossary.md#term-identity-meanings), and existing [opaque](../../../../glossary.md#term-opaque) Tensor-ID behavior.

This is the [model](../../../../glossary.md#term-model) prerequisite for [Compiler](../../../../glossary.md#term-compiler) task 0004's [pre-capture Tensor-expression autograd](../../../../glossary.md#pre-capture-tensor-expression-autograd).
It does not implement [gradients](../../../../glossary.md#term-gradient) or compiler behavior.

## [Scope](../../../../glossary.md#term-scope)

- Add one private final [immutable](../../../../glossary.md#term-immutable) ordered output-[Tensor](../../../../glossary.md#tensor) [snapshot](../../../../glossary.md#term-snapshot) to [`TensorProducer`](../../../../glossary.md#producer--tensorproducer).
- Add exactly one public indexed retrieval method:

  ```java
  Tensor output(int outputIndex)
  ```

- Return the exact [canonical](../../../../glossary.md#term-canonical-form) wrapper for the requested slot; never reconstruct a wrapper.
- Keep `outputCount()` derived from `outputDescriptors().size()`.
- Refactor package-private derived construction so `TensorFactory` initiates one atomic producer
  construction that:<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
  - validates operation, ordered inputs, output descriptors, and signature before allocating an
    ID;<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptors glossary](../../../../glossary.md#term-descriptor); [signature glossary](../../../../glossary.md#term-signature))</span>
  - allocates one ID and creates one Tensor wrapper per output position;
  - creates each wrapper with indexed [provenance](../../../../glossary.md#provenance) pointing to the same [producer](../../../../glossary.md#term-producer) under construction;
  - completes the producer's final output snapshot before any wrapper or producer is returned; and
  - returns the retained canonical wrappers from the producer.
- Preserve the existing single-output optional label and the unlabeled multi-output behavior.
- Preserve current [validation](../../../../glossary.md#term-validation) order and ID-consumption rules as far as they are observable.
- Verify canonical hidden-output retrieval for [dropout](../../../../glossary.md#dropout--inverted-dropout) and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) training.
- Finalize affected Javadoc, Tensor API, Compile API, glossary, capabilities, task evidence,
  model master plan, and roadmap through the required separate documentation-focused pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- any [Tensor](../../../../glossary.md#tensor) method, Tensor constructor [signature](../../../../glossary.md#term-signature), or ergonomic result-[carrier](../../../../glossary.md#term-storage-carrier) signature change
- exposing the [dropout](../../../../glossary.md#dropout--inverted-dropout) [mask](../../../../glossary.md#term-boolean-mask) or [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) saved statistics in public result carriers
- adding `outputs()`, streams, iterators, arrays, maps, sibling registries, or name-based [lookup](../../../../glossary.md#term-lookup) to
  [`TensorProducer`](../../../../glossary.md#producer--tensorproducer)
- wrapper reconstruction from a [descriptor](../../../../glossary.md#term-descriptor), [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), or output index
- changing operation semantics, output descriptors, output order, gradient eligibility, labels,
  storage, or public expression validation<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [validation glossary](../../../../glossary.md#term-validation))</span>
- `Tensor.gradient`, `Tensor.backward`, [mutable](../../../../glossary.md#term-mutable) [gradient](../../../../glossary.md#term-gradient) fields, gradient [lifecycle](../../../../glossary.md#lifecycle), or a tape
- gradient rules, reverse traversal, accumulation, [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), [graph](../../../../glossary.md#graph) phases, [`NodeId`](../../../../glossary.md#nodeid),
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [compiler](../../../../glossary.md#term-compiler) state, or a compiler [dependency](../../../../glossary.md#term-dependency) in [model](../../../../glossary.md#term-model)
- weak-reference ownership, explicit cycle breaking, close behavior, finalizers, cleaners, or
  garbage-collection tests<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- public producer builders, factories, registries, facades, interning, structural equality, or
  serialization identity<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality); [serialization glossary](../../../../glossary.md#term-serialization); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Java module dependencies, Gradle changes, architecture tests, conformance tests, integration
  tests, or another module<span data-glossary-navigation="true"> ([Java module glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>
- implementing or creating a detailed specification for Compiler task 0004

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the model, compiler,
  training-graph, autograd, and compile-lifecycle sections<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [autograd glossary](../../../../glossary.md#autograd); [compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0019B1](0019b1-explicit-graph-dropout-construction.md)
- [Task 0021C](0021c-batch-normalization-training-and-statistic-transition.md)
- [Task 0023F](0023f-scaled-dot-product-attention-weights-output.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns Tensor expression state, producer occurrence identity, TensorFactory
  construction, and provenance. It must not depend on compiler.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- A [producer](../../../../glossary.md#term-producer) is pre-[capture](../../../../glossary.md#term-capture-meanings) expression [identity](../../../../glossary.md#term-identity-meanings), not `CompiledNode`, graph membership, a
  canonicalization key, or a service.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Tensor identity, descriptor, label, and provenance remain immutable. Its existing mutable
  borrowed host-storage association does not justify another mutable field.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [immutable glossary](../../../../glossary.md#term-immutable); [mutable glossary](../../../../glossary.md#term-mutable); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [host glossary](../../../../glossary.md#term-host))</span>
- [Tensor](../../../../glossary.md#tensor) owns no [gradient](../../../../glossary.md#term-gradient) or backward [lifecycle](../../../../glossary.md#lifecycle) state.
- Every producer output slot has exactly one [canonical](../../../../glossary.md#term-canonical-form) wrapper. The exact wrapper returned by
  `TensorFactory` must be the same object returned by `producer.output(index)`.
- All output wrappers have indexed provenance pointing to that exact producer and retain the exact
  descriptor reference from the same slot.
- The intentional object cycle is:

  ```text
  Tensor -> TensorProvenance -> TensorProducer -> outputs -> Tensor
  ```

  It contains only ordinary Java references and immutable expression metadata. The complete
  occurrence must become externally reachable only after all final fields are assigned.
  Unreachable cycles remain eligible for ordinary garbage collection.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [garbage collection glossary](../../../../glossary.md#term-garbage-collection-gc))</span>
- The construction implementation may let wrappers refer to the producer while its constructor is
  active only when neither the producer nor any wrapper escapes the factory-owned construction
  boundary before the final output snapshot is assigned. Do not publish through static state,
  callbacks retained beyond construction, another thread, or caller-visible collections.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Producer/signature validation remains before first ID allocation. Failure after ID allocation
  does not roll back or reuse IDs.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation))</span>
- Cross-module compiler access is satisfied by the one public indexed accessor. Do not broaden
  public Tensor or result-carrier APIs.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- If safe publication requires mutable output state, post-construction initialization, a registry,
  or another module dependency, stop and report the architecture conflict.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [registry glossary](../../../../glossary.md#term-registry); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.tensor` — owns Tensor, producer, provenance, factory, result
  carriers, and expression construction.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>

Packages added or changed:

- No package is added.
- Only the existing [tensor](../../../../glossary.md#tensor) package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorProducer` — owns the final canonical output
  snapshot because one producer occurrence is the identity shared by every result position.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [snapshot glossary](../../../../glossary.md#term-snapshot); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.model.tensor.TensorFactory` — remains the sole package-level derived
  construction boundary and ID allocator.
- `io.github.pho001.synaptik.model.tensor.TensorProvenance` — keeps its exact producer and
  output-index association unchanged.
- `io.github.pho001.synaptik.model.tensor.Tensor` — keeps its public declaration and existing
  immutable metadata plus mutable borrowed-storage behavior unchanged; Javadoc only may need cycle
  clarification.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [mutable glossary](../../../../glossary.md#term-mutable); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Test placement:

- `TensorProducerTest` — exact field/method shape, indexed retrieval, canonical identity,
  immutability, bounds, and safe-publication observations.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [publication glossary](../../../../glossary.md#term-publication))</span>
- `TensorFactoryTest` — single/multi-output canonical construction, validation order, labels, and
  ID side effects.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `TensorProvenanceTest` — exact [producer](../../../../glossary.md#term-producer)/output/[descriptor](../../../../glossary.md#term-descriptor) consistency.
- `TensorDropoutExpressionTest` and `TensorBatchNormTrainingExpressionTest` — exact retrieval of
  hidden same-occurrence auxiliaries.

## Required contracts

### [Producer](../../../../glossary.md#term-producer) state and accessor

[`TensorProducer`](../../../../glossary.md#producer--tensorproducer) gains exactly:

```java
private final List<Tensor> outputs;
```

alongside its current final [operation](../../../../glossary.md#operation), inputs, and output [descriptors](../../../../glossary.md#term-descriptor). It exposes:

```java
public Tensor output(int outputIndex);
```

The method:

- rejects a negative index;
- rejects an index greater than or equal to `outputCount()`;
- reports the requested index and available output count in failures;
- returns the exact retained [Tensor](../../../../glossary.md#tensor) reference at that slot; and
- performs no allocation, traversal, [inference](../../../../glossary.md#term-inference), reconstruction, or mutation.

Do not expose the output list itself. `outputCount()` remains descriptor-derived; do not add a
count field.

### Factory-[atomic](../../../../glossary.md#resource-transaction-and-rollback) cycle construction

Refactor only the package-private derived-construction path. The concrete constructor, helper, or
construction-local [callback](../../../../glossary.md#term-callback) [shape](../../../../glossary.md#shape) is an implementation-local decision. This is feasible because
`TensorProvenance` and [`Tensor`](../../../../glossary.md#tensor) [validation](../../../../glossary.md#term-validation) consult only the [producer](../../../../glossary.md#term-producer)'s already-assigned output
count and [descriptor](../../../../glossary.md#term-descriptor) [snapshot](../../../../glossary.md#term-snapshot) while wrappers are constructed; neither needs the producer's output
snapshot before its final assignment. The selected implementation must satisfy this sequence:

1. `TensorFactory` validates its public/package-private [arguments](../../../../glossary.md#term-argument) in the current order.
2. Producer construction snapshots and validates operation, inputs, output descriptors, and
   signature cardinality before requesting the first Tensor ID.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
3. The producer has its operation, input snapshot, and descriptor snapshot available while it
   creates output wrappers in ascending slot order.
4. Each wrapper receives one newly allocated ID, its exact slot descriptor, current label policy,
   no storage, and `TensorProvenance(producer, slot)`.
5. No wrapper, producer, caller-owned collection, static field, or other thread can observe the
   producer before its immutable output snapshot is assigned.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
6. The factory returns `producer.output(0)` for single-output construction and an immutable list
   assembled from `producer.output(0..outputCount-1)` for multi-output construction.

Construction-local callbacks or allocators must not be retained in producer state. Do not add a
[mutable](../../../../glossary.md#term-mutable) builder phase or initialize `outputs` after producer construction returns.

### [Identity](../../../../glossary.md#term-identity-meanings) and [lifecycle](../../../../glossary.md#lifecycle)

- A current single-output expression still allocates one producer, one wrapper, one provenance,
  and one ID.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- A current N-output expression still allocates one producer, N wrappers, N provenance values, and
  N IDs in ascending output order.
- All public result carriers retain their current exact wrapper components.
- Keeping any public result wrapper alive may now keep its sibling wrappers alive through the
  producer. This is the intended cost of exact hidden-output access.
- When the whole occurrence is unreachable, the cycle owns no external resource and is ordinarily
  collectable. Do not add an unreliable forced-GC acceptance test.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [GC glossary](../../../../glossary.md#term-garbage-collection-gc))</span>

### Failure and ID effects

- Null container/element, empty output, and signature-cardinality failures occur before the first
  ID request.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [signature glossary](../../../../glossary.md#term-signature); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- Existing single-output blank-label failure continues to consume its allocated ID.
- Multi-output identifier exhaustion may consume IDs allocated for earlier slots, returns no
  partial list, and does not publish a partial producer.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- IDs remain opaque; tests must not introduce adjacency as a public contract beyond existing
  focused allocation-side-effect checks.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>

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

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Expected no-change reviews:

- `docs/api/training-api.md` — no training [API](../../../../glossary.md#term-application-programming-interface-api) changes.
- `ARCHITECTURE.md`, focused architecture pages, and ADR 0009 — already authorize and explain
  this prerequisite; implementation must not change architecture.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- `testing/architecture-tests/` — no [module](../../../../glossary.md#term-module) or [dependency](../../../../glossary.md#term-dependency) rule changes.
- [compiler](../../../../glossary.md#term-compiler) source/tests — Compiler 0004 remains Draft.
- `TensorScaledDotProductAttentionExpressionTest` and `TensorTopKExpressionTest` — their public
  carriers already expose every output and current factory/producer tests cover general visible
  multi-output identity; change only if a concrete regression is otherwise untestable.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings); [regression glossary](../../../../glossary.md#term-regression))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- the four production [tensor](../../../../glossary.md#tensor)-package files listed above;
- the five focused [model](../../../../glossary.md#term-model) tests listed above;
- the seven documentation/[planning](../../../../glossary.md#term-planning) files listed above; and
- no Java file in another package or [module](../../../../glossary.md#term-module).

The 16-path allowance is a cohesive foundation exception: [producer](../../../../glossary.md#term-producer) state, factory construction,
indexed [provenance](../../../../glossary.md#provenance), hidden-output examples, public contract documentation, and frontier status
must agree atomically. Do not use the allowance for [operation](../../../../glossary.md#operation)-family changes, broad test cleanup,
or unrelated documentation.

If another production type, package, module, public result [carrier](../../../../glossary.md#term-storage-carrier), or Tensor method is needed,
stop and propose a follow-up or request an architecture decision.

The coordinating planning context later authorized three status-only consistency exceptions in
already-modified documentation: `docs/planning/modules/compiler/master-plan.md`,
`docs/api/training-api.md`, and `docs/design/notes/autograd-strategy.md`. Those corrections say
only that Model 0025 is Complete and [Compiler](../../../../glossary.md#term-compiler) 0004 remains Draft awaiting a dedicated planning
pass. They add no compiler or training behavior and do not expand Java scope.

## Acceptance criteria

- [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) has exactly four private final instance fields: operation, inputs, output
  descriptors, and canonical outputs.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptors glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Its public surface adds only `output(int)` to the current four accessors.
- `output(int)` validates both bounds and returns the exact retained wrapper without allocation.
- Every factory-created derived [Tensor](../../../../glossary.md#tensor) is identical to
  `tensor.provenance().orElseThrow().producer().output(outputIndex)`.
- Single-output expression behavior, optional label, ID count, exact descriptor, no storage, and
  provenance index zero remain unchanged.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Multi-output factory results are immutable and each list element is the exact producer-retained
  wrapper at the same slot.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [element glossary](../../../../glossary.md#term-element); [producer glossary](../../../../glossary.md#term-producer))</span>
- Every output wrapper has the exact shared producer, matching output index, and exact slot
  descriptor.
- Dropout producer slot one returns the exact hidden BOOL mask wrapper from the original
  occurrence; it is not reconstructed or added to <span data-glossary-navigation="true"> ([Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`DropoutResult`.
- Batch-normalization training producer slots three and four return the exact hidden saved mean
  and inverse-standard-deviation wrappers; they are not added to<span data-glossary-navigation="true"> ([Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [mean glossary](../../../../glossary.md#term-mean))</span>
  `BatchNormTrainingResult`.
- Producer/signature validation precedes first ID allocation; later failures preserve the
  documented non-rollback behavior and publish no partial result.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Concurrently published successful derived results always expose a complete producer with every
  canonical slot and exact provenance relationship.
- The cycle uses only final producer state and current immutable provenance; no post-construction
  mutation, registry, weak-reference scheme, resource cleanup, or graph identity is added.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tensor methods and public ergonomic result-[carrier](../../../../glossary.md#term-storage-carrier) [signatures](../../../../glossary.md#term-signature) remain unchanged.
- No gradient, compiler, runtime, backend, dependency, Gradle, or architecture-test behavior is
  added.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Javadocs define identity, indexed bounds, construction order, safe publication, retention cost,
  failure/ID effects, and absence of graph/gradient/runtime meaning.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- The separate documentation-focused pass finalizes all affected API, glossary, capabilities,
  task, master-plan, and roadmap text and records reasoned no-change conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task, master plan, and roadmap statuses agree only after implementation, tests, documentation
  review, and validation complete.
- [Compiler](../../../../glossary.md#term-compiler) 0004 remains Draft and has no detailed task specification.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused tests may be [run](../../../../glossary.md#term-run) while developing. After executable Java stabilizes, run
one final [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:model:test
```

The separate documentation-focused pass then runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also checks affected Markdown links and anchors, balanced fences, final
newlines, terminology, exact status [synchronization](../../../../glossary.md#term-synchronization), exact allowed paths, and the absence of the
obsolete no-[producer](../../../../glossary.md#term-producer)-output/current-placeholder design.

Repository-wide tests are deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one module and no [dependency](../../../../glossary.md#term-dependency),
build, or architecture boundary. Architecture tests are not run because dependency rules do not
change. The documentation pass reuses the final [model](../../../../glossary.md#term-model)-test evidence unless it changes executable
Java behavior or records a concrete reason.

Automated tests must cover:

- exact producer field and public-method [shape](../../../../glossary.md#shape);
- indexed bounds and [canonical](../../../../glossary.md#term-canonical-form) exact-reference retrieval;
- single- and multi-output factory construction;
- [descriptor](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance) [identity](../../../../glossary.md#term-identity-meanings) agreement;
- validation order and ID side effects;
- [dropout](../../../../glossary.md#dropout--inverted-dropout) and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) hidden outputs; and
- safe [publication](../../../../glossary.md#term-publication) under concurrent successful construction or observation.

## Dependencies

- Task 0018L: shared multi-output [producer](../../../../glossary.md#term-producer) and indexed [provenance](../../../../glossary.md#provenance) — Complete.
- Task 0019B1: hidden same-[occurrence](../../../../glossary.md#term-occurrence) [dropout](../../../../glossary.md#dropout--inverted-dropout) [mask](../../../../glossary.md#term-boolean-mask) — Complete.
- Task 0021C: hidden [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) saved statistics — Complete.
- Task 0023F: explicit same-occurrence attention [weights](../../../../glossary.md#term-weight) — Complete.
- Task 0024A: completed historical model-milestone documentation closure and immediate
  planning-order predecessor — Complete.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning))</span>
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009: accepted pre-[capture](../../../../glossary.md#term-capture-meanings) [autograd](../../../../glossary.md#autograd) architecture requiring [canonical](../../../../glossary.md#term-canonical-form) hidden-output wrappers.

## Follow-up tasks

- Compiler 0004 — implement fail-closed compiler-owned pre-capture Tensor-expression autograd,
  one phase-aware combined capture, validation, and combined exact optimization. Required.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [pre-capture Tensor-expression autograd glossary](../../../../glossary.md#pre-capture-tensor-expression-autograd); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0004A — extend exact-composition [gradient](../../../../glossary.md#term-gradient) formulas after the core pipeline is proved.
- Compiler 0004B — select [derivative](../../../../glossary.md#term-derivative) policies and add policy-dependent formulas.
- Compiler 0006 — later define explicit objectives, targets, seeds, and higher derivative order
  after the public compile/artifact boundary is stable.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [compile glossary](../../../../glossary.md#compile); [artifact glossary](../../../../glossary.md#term-artifact))</span>

Do not create detailed specifications for those follow-ups in this task.

## Architecture impact

Expected impact: None during implementation.

The coordinated [planning](../../../../glossary.md#term-planning) change has already updated `ARCHITECTURE.md`, focused architecture
documentation, and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009. This task implements the [model](../../../../glossary.md#term-model)-owned prerequisite without changing
[module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings) or [dependency direction](../../../../glossary.md#java-platform-and-build-tools). Architecture tests require no update.

If implementation requires [mutable](../../../../glossary.md#term-mutable) [producer](../../../../glossary.md#term-producer) outputs, [Tensor](../../../../glossary.md#tensor) [gradient](../../../../glossary.md#term-gradient) state, [compiler](../../../../glossary.md#term-compiler) imports,
another module, or a [dependency](../../../../glossary.md#term-dependency)-rule change, stop and report the conflict.

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

- [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) owns exactly four private final fields. Its new `outputs` snapshot is assigned
  once in the constructor after wrappers are created in ascending slot order; there is no builder
  phase or post-construction mutation.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- One package-private five-argument producer constructor accepts the first-output label and a
  construction-local <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [producer glossary](../../../../glossary.md#term-producer))</span>`Supplier<TensorId>`. The supplier is not retained. The existing
  package-private three-argument constructor delegates with an empty label and the factory's
  allocator, preserving its declaration for existing tensor-package callers.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>
- `TensorFactory.nextTensorId()` changed from private to package-private so both ordinary factory
  creation and the existing three-argument producer construction seam use the one JVM-wide
  allocator. This is the smallest truthful non-public collaboration: keeping it private would
  require changing callers outside this task, reflection, duplicate or non-global IDs, another
  production type or mutable state, or a redundant forwarding abstraction. Reflection confirms
  that the public factory API did not change.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Single-output creation returns `producer.output(0)`. Multi-output creation returns an immutable
  list assembled only from <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`producer.output(index)`, so factory results and producer lookup cannot
  diverge by object identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- Producer/signature validation remains before the first ID request. Output wrappers then use the
  exact slot descriptor, output-zero label policy, no storage, and indexed provenance. The final
  output snapshot is assigned before factory return, preserving safe publication without a
  registry, weak reference, or mutable initialization phase.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [publication glossary](../../../../glossary.md#term-publication); [registry glossary](../../../../glossary.md#term-registry); [weak reference glossary](../../../../glossary.md#weak-reference-and-weak-interning))</span>

## Known limitations

- Retaining any derived output may retain every sibling output, the producer, provenance, and
  producer inputs. This is the intentional cost of exact hidden-output identity.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The immutable cycle owns no external resource and remains ordinarily garbage-collectable when
  unreachable, but this task deliberately adds no unreliable forced-garbage-collection test.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Tensor IDs remain opaque. Allocation after the validation boundary is monotonic and
  non-rollback, but adjacency is not a public semantic contract.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [opaque glossary](../../../../glossary.md#term-opaque); [validation glossary](../../../../glossary.md#term-validation); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Hidden outputs are available through `TensorProducer.output(int)` only. Existing ergonomic
  result carriers remain unchanged.
- Compiler 0004 remains Draft without a detailed specification or implementation. This task
  supplies only its model prerequisite.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [model glossary](../../../../glossary.md#term-model))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/implement_model_0025` ran
  `./gradlew :modules:model:test`: `BUILD SUCCESSFUL in 2s`; three actionable tasks, two executed
  and one up-to-date. The suite covered exact field/public-method shape, indexed bounds, canonical
  single- and multi-output identity, descriptor/provenance agreement, validation and ID effects,
  concurrent successful publication, the hidden dropout mask, and hidden batch-normalization
  saved statistics.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [publication glossary](../../../../glossary.md#term-publication); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Documentation context `/root/implement_model_0025/finalize_model_0025_docs` ran
  `./gradlew :modules:model:javadoc`: `BUILD SUCCESSFUL in 2s`; two actionable tasks, both
  executed. Generated Javadoc includes the new indexed accessor and package-owned allocator seam.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- A targeted local Markdown validator resolved local paths and anchors and checked balanced
  fences and final newlines for all ten affected Markdown documents; all ten passed. A full stale
  status/design scan then returned no obsolete readiness, no-output, or planned-cycle wording.
- `javap -public` reports exactly five public [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) methods: the four existing accessors
  plus `output(int)`. `javap -public` reports no `TensorFactory.nextTensorId()`, while
  `javap -private` reports it as package-private alongside the two existing package-private
  derived-construction seams.
- The final scope audit found exactly 19 coordinated task paths: 16 task-authorized paths and the
  three coordinator-approved status-only exceptions. It separately accounted for nine preserved
  pre-existing architecture paths and the six untouched untracked compiler prototypes.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [compiler glossary](../../../../glossary.md#term-compiler); [prototypes glossary](../../../../glossary.md#term-prototype))</span>
- Task 0025, its model-master row, and its roadmap row all report Complete; Compiler 0004 remains
  Draft in the compiler master plan. <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`git diff --check` passed.

## Implementation notes

- `TensorProducer.output(int)` reports both the requested index and available output count for
  either bound failure and otherwise performs one indexed read of retained final state.
- [Dropout](../../../../glossary.md#dropout--inverted-dropout) output one is the exact hidden [`BOOL`](../../../../glossary.md#data-type--datatype) keep-mask wrapper from the original occurrence.
  Batch-normalization training outputs three and four are the exact hidden saved-mean and
  saved-inverse-standard-deviation wrappers. No carrier component was added.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [occurrence glossary](../../../../glossary.md#term-occurrence); [Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [mean glossary](../../../../glossary.md#term-mean); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Existing single-output labels, output order, descriptor references, provenance indices,
  storage-free results, identifier count, and non-rollback behavior remain unchanged.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Javadocs and API documentation now state the canonical identity, construction/validation order,
  safe-publication boundary, sibling-retention cost, ordinary cycle collection, failure/ID
  effects, and absence of graph, gradient, compiler, backend, runtime, or execution meaning.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The implementation and substantive documentation stayed within the 16 task-authorized paths:
  four tensor production files, five focused tensor tests, and seven documentation/planning files.
  Three coordinator-approved status-only consistency exceptions corrected already-modified
  compiler-master, Training API, and autograd-strategy prose, for 19 coordinated paths total.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning); [autograd glossary](../../../../glossary.md#autograd))</span>

## Completion summary

- Completed changes: [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) now owns and returns one canonical exact Tensor wrapper for
  every output position; <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Tensor glossary](../../../../glossary.md#tensor))</span>`TensorFactory` constructs and publishes complete occurrences through
  that indexed identity; focused tests cover ordinary, hidden, failure, and concurrent-publication
  behavior; Javadocs, Tensor/Compile APIs, glossary, capability baseline, task, model master plan,
  and roadmap describe the final contract.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [publication glossary](../../../../glossary.md#term-publication); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Files changed or created: the four production, five test, and seven documentation/planning paths
  listed under <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>[Affected files](#affected-files), with this task specification created as the
  sixteenth path, plus the three coordinator-approved status-only paths recorded under
  [Maximum scope](#maximum-scope)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>.
- Public [API](../../../../glossary.md#term-application-programming-interface-api): `TensorProducer.output(int)` is the only new public method. `TensorFactory`'s
  allocator is package-private, so its visibility change adds no public API. Tensor methods,
  constructors, descriptors, provenance components, operation signatures, and public result
  carriers are unchanged.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Documentation no-change review: no Training API behavior changed because no training session,
  gradient publication, parameter, optimizer, or result-carrier contract changed; its only edit is
  the coordinator-approved Model 0025 status correction. <span data-glossary-navigation="true"> ([training session glossary](../../../../glossary.md#training-session--trainingsession); [gradient glossary](../../../../glossary.md#term-gradient); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`ARCHITECTURE.md`, the current/focused
  architecture pages, and ADR 0009 already authorize and explain this model prerequisite; no
  architecture decision changed. The autograd strategy note likewise receives status wording
  only.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [autograd glossary](../../../../glossary.md#autograd))</span>
- Validation no-change review: architecture tests need no update because module ownership and
  dependencies did not change. Backend-conformance and integration tests need no update because
  no backend or end-to-end behavior exists here. Repository-wide tests remain deferred to CI under
  the normal single-module validation tier.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>
- Compiler no-change review: compiler source and tests remain unchanged; Compiler 0004 remains
  Draft with no detailed task specification or implementation. The compiler master plan receives
  only the coordinator-approved status correction that completed Model 0025 does not
  automatically advance compiler work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Other no-change review: build and Gradle configuration, every other module, Tensor methods,
  descriptors, public carriers, operation semantics, input/output order, and label/storage policy
  remain unchanged. <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`TensorScaledDotProductAttentionExpressionTest` and
  `TensorTopKExpressionTest` need no update because their carriers already expose every output;
  the general visible multi-output identity cases remain covered by producer/factory tests.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Unresolved issues: none.
- Required follow-up: Compiler 0004 remains the required Draft follow-up after a future planning
  step creates its detailed specification.

Status: Complete
