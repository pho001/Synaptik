# Task 0018L: Shared Multi-Output Tensor Provenance

## Status

Complete

## Goal

Represent every derived public Tensor as one indexed output of one immutable expression producer,
so a single backend-independent operation occurrence can honestly own one or several ordered
Tensor results before compiler capture.

Use one producer model for both single-output and multi-output expressions. A producer owns the
operation, ordered input Tensor references, and ordered output descriptors. Each result Tensor
owns provenance containing the exact shared producer reference and its zero-based output index.

This task provides the model foundation required by later top-K, explicit graph RNG, dropout,
normalization-statistics, and other genuine multi-output operations. It does not implement any of
those operations or compiler capture.

## Scope

- Add one public final non-record `TensorProducer` identity class in the existing tensor package.
- Give each producer exactly:
  - one immutable `Operation` reference;
  - one immutable ordered snapshot of input Tensor references; and
  - one immutable ordered snapshot of output `TensorDescriptor` references.
- Derive the actual output count from `outputDescriptors().size()`; do not store an independent
  count field.
- Validate the producer's final input and output counts through the operation's selected
  `OperationSignature`.
- Preserve empty and repeated input positions when the signature permits them.
- Require at least one output descriptor, permit repeated descriptor references, and retain every
  exact immutable descriptor reference in encounter order.
- Give `TensorProducer` ordinary object identity: two separately created producers remain
  different occurrences even when operation, inputs, and output descriptors are structurally
  equal.
- Replace the existing two-component `TensorProvenance(Operation, List<Tensor>)` record with a
  two-component `TensorProvenance(TensorProducer, int outputIndex)` record.
- Validate every provenance output index against its producer and provide derived compatibility
  accessors for `operation()`, `inputs()`, and `outputDescriptor()`.
- Require a derived Tensor's descriptor to be the exact descriptor reference selected by its
  provenance output index.
- Replace the existing package-private single-output `TensorFactory.createDerived` boundary with
  a producer-aware single-output boundary that receives the operation and ordered inputs.
- Add one package-private multi-output factory boundary that accepts an operation, ordered inputs,
  and ordered output descriptors, creates exactly one producer, and returns all indexed output
  Tensors in one immutable ordered list.
- Migrate every existing single-output expression helper to the new factory boundary without
  changing its public Tensor API, operation, input order, result descriptor, ID allocation,
  label, storage, or semantic behavior.
- Update focused provenance/factory/Tensor tests and direct test-local provenance construction for
  the intentional new model.
- Add focused test-local multi-output coverage without adding a production multi-output operation.
- Finalize affected Javadoc, Tensor API, Compile API, glossary, task evidence, master plan, and
  roadmap through the required targeted documentation pass.

## Out of scope

- `topK`, sort, argsort, dropout, RNG state, normalization statistics, or any other production
  multi-output operation or result record
- public `TopKResult`, `DropoutResult`, tuple, pair, destructuring, or general public multi-output
  collection API
- changing current public Tensor expression method signatures or fluent single-output syntax
- adding a public method to retrieve sibling output Tensor objects from a producer
- storing output Tensor references in `TensorProducer`
- storing an independent output-count field in addition to output descriptors
- storing output labels, host storage, values, gradients, publication state, or mutable result
  state in a producer
- adding `ProducerId`, `NodeId`, `ValueId`, graph membership, graph-local identity, or compiler
  state to Tensor, provenance, or producer
- turning `TensorProducer` into `CompiledNode`, graph IR, a graph builder, a capture session, a
  registry, a service, or a canonicalization key
- structural producer equality, producer interning, common-subexpression elimination, or merging
  separately invoked equal expressions
- compiler traversal, graph capture, `GraphValue` or `CompiledNode` creation, descriptor inference,
  publication binding, optimization, autograd, or backward construction
- backend ownership, lowering, kernel selection, prepare, runtime, execution, or device storage
- changing `Operation`, `OperationKind`, `OperationSignature`, `CompiledNode`, or any production
  operation-family contract
- changing unstack semantics; task 0018O owns its conversion to repeated scalar select
- dependencies, Gradle, architecture tests, another module, `ARCHITECTURE.md`, or focused
  architecture documentation
- creating a detailed task-0018M specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially public Tensor state,
  model-owned operation semantics, immutable graph state, and the rule that Tensor is not IR
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially “Multi-output provenance”
- [Model master plan](../master-plan.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0008](0008-graph-value-and-node-model.md)
- [Task 0009](0009-compiled-graph-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns public Tensor state, Tensor descriptors, operation semantics, expression
  provenance, and this pre-capture producer description.
- `Tensor` remains public mutable API state and is not an IR node. Its producer reference describes
  expression origin only and establishes no graph membership.
- `TensorProducer` is not `CompiledNode`. One producer may be captured into different nodes with
  different `NodeId` and `ValueId` values in separately compiled graphs.
- Producer identity is occurrence identity in the public expression model only. It is represented
  by the exact Java object reference, not a graph-local or globally allocated identifier.
- `Operation` remains backend-independent semantics. `TensorProducer` may consume its selected
  structural signature but must not add backend support, compiler policy, lowering, execution, or
  kernel metadata.
- Ordered output descriptors are immutable model facts already required by public result Tensors.
  They are not output Tensor objects, graph values, physical buffers, or compiler artifacts.
- Producer construction validates local structural counts only. It must not perform family-
  specific Shape or DataType inference, graph-wide validation, cycle traversal, or executable
  support checks.
- A producer must never retain its result Tensor objects. The direction remains result Tensor to
  provenance to producer, preventing a producer/output/provenance reference cycle.
- Single-output expressions and future multi-output expressions must use the same producer and
  indexed-provenance contracts. Do not retain a second legacy provenance representation or a
  nullable union of old and new state.
- If implementation cannot preserve exact existing single-output public behavior while completing
  this atomic migration, or if compiler/graph identity appears necessary in the producer, stop and
  report the design conflict before changing architecture or another module.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor state, descriptors, factories,
  expression helpers, producer identity, and provenance.

Packages added or changed:

- No package is added.
- The existing tensor package gains one cohesive producer contract and migrates its provenance and
  package-private expression-construction boundaries.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorProducer` — immutable expression-occurrence
  identity shared by all outputs because it composes operation, inputs, and ordered output
  descriptors before graph capture.
- `io.github.pho001.synaptik.model.tensor.TensorProvenance` — immutable association from one result
  Tensor to one exact producer output position.
- `io.github.pho001.synaptik.model.tensor.TensorFactory` — package-private atomic construction
  boundary for one or all outputs of a producer.
- `io.github.pho001.synaptik.model.tensor.Tensor` — enforces exact descriptor/provenance-position
  agreement while retaining its existing public state role.

Test placement:

- `io.github.pho001.synaptik.model.tensor.TensorProducerTest` — validates producer ownership,
  identity, counts, snapshots, and absence of graph/runtime state.
- Existing `TensorProvenanceTest`, `TensorFactoryTest`, and `TensorTest` validate indexed provenance,
  single- and multi-output construction, descriptor agreement, identity, and failure behavior.
- Existing expression tests remain behavior regression coverage; modify only tests that directly
  construct the replaced provenance contract or assert its exact API shape.

## Required contracts

### `TensorProducer`

Create one public final non-record class with exactly these three instance fields:

```java
private final Operation operation;
private final List<Tensor> inputs;
private final List<TensorDescriptor> outputDescriptors;
```

The class must:

- use a package-private constructor so producer creation remains inside the tensor expression-
  construction boundary;
- null-check `operation`, `inputs`, and `outputDescriptors` in that order;
- inspect input elements in ascending index order and reject a null element with message
  `inputs[index]`;
- snapshot inputs with `List.copyOf`, preserving order, empty input, and repeated exact Tensor
  references when permitted by the operation signature;
- reject an empty output-descriptor list with
  `IllegalArgumentException("outputDescriptors must not be empty")`;
- inspect output descriptor elements in ascending index order and reject a null element with
  message `outputDescriptors[index]`;
- snapshot output descriptors with `List.copyOf`, preserving order and repeated exact immutable
  descriptor references;
- after local collection validation, call
  `operation.signature().validateOccurrence(inputs.size(), outputDescriptors.size())` exactly once;
- retain the exact operation reference and exact list-element references;
- expose exactly `operation()`, `inputs()`, `outputDescriptors()`, and `outputCount()` as public
  accessors;
- derive `outputCount()` from `outputDescriptors.size()` and store no separate count;
- keep ordinary object identity equality and hashing by overriding neither `equals` nor `hashCode`;
- add no public factory, builder, mutator, producer identifier, output Tensor accessor, lookup,
  registry, traversal, or graph API; and
- contain no backend, compiler, runtime, storage, value, gradient, publication, or execution state.

The constructor may use temporary local validation state but must add no cached collection,
index, or mutable field. `toString()` is not a serialization or identity contract; do not add an
object-method override merely to expose inputs or descriptors recursively.

### `TensorProvenance`

Replace the current record with exactly these components and order:

```java
public record TensorProvenance(
        TensorProducer producer,
        int outputIndex) {
}
```

The canonical constructor must:

1. reject a null producer with `NullPointerException("producer")`;
2. reject a negative index with
   `IllegalArgumentException("outputIndex must be non-negative: " + outputIndex)`; and
3. reject an index greater than or equal to `producer.outputCount()` with a message identifying
   the index and available output count.

The record must retain the exact producer reference and primitive index. Add these derived public
accessors without adding record components or fields:

```java
Operation operation();
List<Tensor> inputs();
TensorDescriptor outputDescriptor();
```

They return the producer's exact operation, immutable inputs list, and exact descriptor at
`outputIndex`, respectively. Keeping `operation()` and `inputs()` as derived accessors preserves
the useful read surface of the current provenance contract while changing ownership to the
shared producer.

Record-generated equality and hashing must use exact producer identity through ordinary
`TensorProducer` object equality plus the output index. Two positions of one producer are unequal;
two separately invoked structurally equal producers remain unequal.

Do not add a compatibility constructor retaining the old `(Operation, List<Tensor>)` model, a null
producer sentinel, optional output index, output Tensor list, graph identity, traversal, or mutable
binding phase.

### Tensor descriptor agreement

The package-private `Tensor` constructor must preserve its existing component/null/label/storage
validation order and add one local provenance agreement check after non-null component validation
and before storage attachment:

```text
if provenance is present:
    provenance.outputDescriptor() must be the exact same reference as descriptor
```

Reject a mismatch with an `IllegalArgumentException` whose message clearly identifies descriptor
and provenance output-descriptor disagreement. Use reference identity, not structural equality,
because factory construction deliberately shares the one immutable descriptor object between the
producer slot and result Tensor.

Do not inspect producer inputs, traverse provenance, infer descriptors, reject repeated producer
descriptors, or change Tensor equality, hashing, labels, storage synchronization, or diagnostics.

### `TensorFactory` construction boundaries

Replace the existing package-private single-output method accepting a preconstructed
`TensorProvenance` with:

```java
static Tensor createDerived(
        TensorDescriptor descriptor,
        Optional<String> label,
        Operation operation,
        List<Tensor> inputs);
```

It must validate non-null arguments in declaration order before allocating a Tensor ID, create one
`TensorProducer(operation, inputs, List.of(descriptor))`, create provenance `(producer, 0)`, and
construct exactly one unlabeled-or-supplied-label, storage-free Tensor through the existing
package-private Tensor constructor. It must not copy or replace the descriptor, operation, or
input Tensor references beyond the producer's immutable list snapshot.

Add exactly one package-private multi-output method:

```java
static List<Tensor> createDerivedOutputs(
        Operation operation,
        List<Tensor> inputs,
        List<TensorDescriptor> outputDescriptors);
```

It must:

1. null-check arguments in declaration order;
2. construct exactly one `TensorProducer`, which owns element, snapshot, non-empty, and signature-
   cardinality validation;
3. allocate one fresh Tensor ID per ordered descriptor only after producer validation succeeds;
4. iterate the producer's immutable output-descriptor snapshot rather than the caller's list, and
   construct one storage-free, unlabeled Tensor per output position with the exact descriptor
   reference and `TensorProvenance(producer, index)`;
5. preserve output order; and
6. return one immutable ordered list containing every output Tensor.

Prevalidation failures consume no Tensor IDs. If identifier exhaustion occurs after some output
IDs have been allocated, consumed IDs are not rolled back or reused; no partial result list is
returned. Add no reset hook, caller-supplied ID, label list, output factory callback, generic tuple,
stream, array, map, or public multi-output method.

### Existing expression migration

Every current package-private expression helper that constructs:

```java
TensorProvenance provenance = new TensorProvenance(operation, inputs);
return TensorFactory.createDerived(descriptor, label, provenance);
```

must instead delegate once to the new producer-aware single-output factory with the same exact:

- result descriptor reference;
- label optional;
- operation reference;
- ordered input Tensor references; and
- externally observable ID, provenance, storage, and failure behavior, except for the intentional
  new producer and output-index provenance representation.

Do not refactor operation-specific validation, combine expression helpers, alter public methods,
or change operation semantics while performing this mechanical migration.

## Affected files

Expected core production files:

- new `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected mechanical production migration:

- current `Tensor*Expressions.java` files under
  `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/` that directly construct
  `TensorProvenance` or call the replaced package-private `createDerived` overload

Expected tests:

- new `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProducerTest.java`
- existing `TensorProvenanceTest.java`
- existing `TensorFactoryTest.java`
- existing `TensorTest.java`
- existing model tests that directly construct `TensorProvenance` or assert its exact record/API
  shape; modify no unrelated test behavior

Expected documentation and planning files:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Expected no-change reviews:

- `docs/planning/modules/model/capabilities.md` already records the selected producer, ordered
  output-descriptor, and indexed-result direction after this planning change;
- `ARCHITECTURE.md` and focused architecture documents already establish Tensor versus graph-node
  identity and require no architecture update.

## Maximum scope

This task may create or modify at most:

- one new production Java file and the three existing core tensor files listed above;
- only current tensor expression helpers that must mechanically migrate from direct provenance
  construction to the new single-output factory boundary;
- one new focused producer test and only existing model tests that directly construct provenance,
  exercise the changed factory/Tensor validation, or assert the changed API shape;
- the six implementation-time documentation/planning files listed above; and
- no file in another module, operation family, graph package, build configuration, architecture
  documentation, or architecture tests.

This is a documented atomic-migration exception to the normal 12–18-file guardrail. A partial
migration would leave two incompatible provenance models, require compiler capture to branch on
legacy versus shared origins, or allow output descriptors to disagree with result Tensors. Most
affected files receive one mechanical factory-call replacement. Do not use the larger path
allowance for unrelated cleanup, reformatting, public API expansion, or operation-specific changes.

If another production concept, another package, an operation-family change, graph/compiler code,
or a second provenance representation is required, stop and report the needed design decision.

## Javadoc requirements

- Document `TensorProducer` as immutable pre-capture expression-occurrence identity, including
  exact-reference ownership, ordered inputs and output descriptors, signature validation,
  identity equality, and its distinction from Tensor, `CompiledNode`, `NodeId`, and `ValueId`.
- Document why producer output descriptors are not output Tensor references and therefore create
  no producer/result/provenance cycle.
- Document `TensorProvenance` as one indexed output association and explain every component,
  derived accessor, validation, identity/equality behavior, and single-output index zero.
- Update Tensor Javadoc for exact descriptor/provenance-position agreement without implying graph
  membership or runtime residency.
- Update factory Javadocs for single- and multi-output validation, ownership, immutable result
  order, allocation/ID side effects, labels, and absence of storage.
- Correct affected expression-helper Javadocs only where they name the replaced factory signature
  or describe the old operation-and-input provenance shape.
- Explain the model to newcomers in Tensor API and glossary with one single-output example and one
  conceptual two-output example. Do not claim `topK` or compiler capture is implemented.
- Update Compile API only to explain what a future capture pass can observe from the now-current
  producer/output-index model. Keep compiler traversal and graph creation planned.

## Acceptance criteria

- `TensorProducer` is one public final non-record identity class with exactly operation, immutable
  ordered inputs, and immutable ordered output descriptors as instance fields.
- Producer construction validates component references, indexed elements, non-empty outputs, and
  final input/output counts through the exact selected `OperationSignature`.
- Producer lists preserve order, repeated permitted positions, exact immutable element references,
  and cannot be mutated through accessors.
- Producer output count is derived from the output descriptor list and has no independent field.
- Two separately created structurally equal producers remain unequal and have distinct references;
  ordinary identity hash codes are not treated as unique identifiers. Every output from one
  multi-output construction retains the exact same producer reference.
- `TensorProvenance` has exactly producer and outputIndex record components in order, rejects every
  invalid index, and derives operation, inputs, and output descriptor from the producer.
- A one-output expression has producer output count one and provenance index zero.
- A test-local two-output expression produces two fresh Tensors with one exact shared producer,
  indices zero and one, the exact ordered descriptor references, independent Tensor IDs, empty
  labels, empty storage, and an immutable result list.
- A derived Tensor rejects a provenance position whose descriptor is merely equal but not the exact
  descriptor reference stored for that producer output.
- Existing public Tensor expression signatures and fluent single-output syntax remain unchanged.
- Every current single-output helper preserves its exact operation, input order, descriptor,
  label, identity-allocation, storage-free result, and behavior while using producer index zero.
- No output Tensor reference, graph-local identity, compiler state, backend metadata, runtime
  state, storage, or executable behavior appears in producer or provenance.
- No production multi-output operation, public result wrapper, sibling-output lookup, producer
  registry, interning, or structural deduplication is added.
- All affected public contracts and methods have complete current Javadoc.
- The targeted clean-context documentation pass finalizes Javadoc, Tensor API, Compile API,
  glossary, planning evidence, links, and terminology without repeating successful Java tests
  unless executable behavior changes afterward.
- Task, master-plan row, and roadmap row have matching final status.
- Task 0018M remains Draft without a detailed specification.

## Tests / validation

During implementation, use focused producer, provenance, factory, and Tensor tests as needed.
After executable Java stabilizes, record one final module run:

```bash
./gradlew :modules:model:test
```

The targeted documentation pass then runs, after final Javadoc edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also checks local Markdown links and anchors, balanced fences, terminology,
and final newlines in the affected documentation. It reuses the recorded model-test result unless
it changes executable Java behavior.

Repository-wide `./gradlew test` is deferred to the foundation-contract capability checkpoint
after task 0018N. This task changes one module and no dependency or architecture boundary.

Automated tests must cover producer identity, exact API shape, list ownership, exact descriptor
references, signature cardinality, provenance indices, descriptor mismatch, single-output
compatibility, multi-output construction, and ID side effects. Do not require repeated manual
reflection, `javap`, or bytecode commands when ordinary compilation and automated tests prove the
same facts.

Final scope review must confirm:

- only allowed `modules/model` tensor-package Java files and documentation/planning files changed;
- no output Tensor reference or independent output-count field exists in `TensorProducer`;
- no old `(Operation, List<Tensor>)` provenance representation remains;
- no `NodeId`, `ValueId`, graph-package, compiler, backend, runtime, or service-locator import
  appears in producer or provenance;
- no Gradle, dependency, architecture, another-module, or operation-family file changed; and
- 0018L is `Complete` everywhere only after both passes, while 0018M has no task file.

## Dependencies

- Task 0007: immutable Tensor descriptors — Complete.
- Task 0008: graph values and multi-output-capable compiled nodes — Complete.
- Task 0009: immutable compiled graph model — Complete.
- Task 0011: public Tensor state — Complete.
- Task 0012: Tensor construction and identifier allocation — Complete.
- Task 0013: current single-output Tensor provenance — Complete and intentionally replaced by this
  unified producer/output-position model.
- Task 0018K: exact kind/attributes signatures and local occurrence cardinality — Complete.

## Follow-up tasks

- 0018O — normalize unstack into repeated scalar select rather than treating it as a genuine
  shared multi-output primitive.
- 0019B — use shared producer provenance for state-consuming/state-producing graph randomness and
  dropout.
- 0019C — add a typed public top-K result whose values and indices share one producer.
- 0021 — use shared producer provenance where normalization returns auxiliary statistics.

Do not create detailed specifications for these follow-ups in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns public Tensor state, immutable descriptors, operation semantics,
and expression provenance to `modules/model`, distinguishes Tensor from graph IR, and reserves
graph-local identities for compiled graph contexts. This task makes that existing provenance
model capable of representing multiple results without changing module ownership, dependencies,
lifecycle, or runtime visibility.

If implementation requires a graph-local identity on Tensor, a new cross-module dependency, or an
architecture rule change, stop and report the exact conflict before editing `ARCHITECTURE.md` or
focused architecture documentation.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, task 0018K, task 0018L, and the affected Tensor/provenance/factory/expression source and
tests in full.

Implement task 0018L exactly as specified. Stay inside modules/model and the explicitly allowed
documentation/planning files. Preserve every current public Tensor expression signature and
single-output semantic behavior. Stop on any scope or architecture conflict. Run the task-level
model validation once after executable code stabilizes.

Then hand the actual diff and recorded Java-test evidence to a separate clean-context
documentation-focused agent in the same change. That agent must inspect final source/tests,
finalize affected Javadocs, Tensor API, Compile API, glossary, task/master/roadmap status and
documentation validation, and must not repeat successful Java tests unless executable behavior
changed or it records a concrete reason.

Do not mark 0018L Complete until both passes succeed. Leave 0018M Draft without a detailed
specification. Do not commit or push.
```

## Local decisions

- `TensorProducer` is a final non-record identity class with a package-private constructor and
  exactly three final fields. Public accessors expose the exact operation and immutable ordered
  snapshots; no value-based object methods or producer identifier were added.
- Both factory seams construct the producer before allocating a Tensor ID. The single-output seam
  shares the supplied descriptor as producer slot zero and uses provenance index zero. The
  multi-output seam iterates the producer snapshot, assigns one ID per position, and returns an
  immutable list.
- Tensor descriptor/provenance agreement uses reference identity. This keeps one descriptor object
  authoritative for a producer slot and its result without broadening structural inference.
- Existing single-output helpers delegate operation and ordered inputs to the factory. Their public
  signatures and operation-specific validation remain unchanged. Current unstack continues to
  create independent one-output producers because task 0018O owns its semantic normalization.

## Known limitations

- No production operation currently declares multiple outputs, and no public multi-output result
  wrapper or sibling-result lookup exists. Focused package-local tests exercise the representation.
- Producer/provenance state is expression origin only. Compiler traversal, graph-node/value
  assignment, inference, optimization, gradients, backend lowering, runtime behavior, and
  execution remain unimplemented or separately owned.
- A producer intentionally retains exact input Tensor references and output descriptor references,
  but never output Tensor objects. It provides no cycle traversal, graph-wide validation,
  canonicalization, interning, or serialization identity.

## Validation evidence

- Implementation context `/root/task_0018l_implementation` first ran
  `./gradlew :modules:model:compileTestJava`; it failed with 49 expected atomic-migration compile
  errors from remaining test-local calls to the replaced provenance constructor and derived-factory
  signature. The implementation then migrated every directly affected test call.
- The same context reran `./gradlew :modules:model:compileTestJava` after that migration:
  `BUILD SUCCESSFUL in 1s`; 2 actionable tasks, with 1 executed and 1 up-to-date.
- The implementation context ran focused validation with
  `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest
  --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest`: `BUILD SUCCESSFUL in 5s`; 3 actionable tasks,
  with 1 executed and 2 up-to-date.
- Implementation context `/root/task_0018l_implementation` ran final
  `./gradlew :modules:model:test` after executable Java stabilized: `BUILD SUCCESSFUL in 1s`. XML
  reports contain 749 tests across 87 suites, with 0 failures, 0 errors, and 0 skipped tests.
- Clean documentation-focused context
  `/root/task_0018l_implementation/task_0018l_docs` independently reviewed `AGENTS.md`,
  `ARCHITECTURE.md`, the current architecture index, documentation rules and General,
  API/Javadoc, Planning, and Example profiles, planning guide, capability baseline, model master
  plan, roadmap, tasks 0018K and 0018L, final source/tests and actual diff, Tensor API, Compile API,
  and glossary.
- The documentation context changed only Javadocs and documentation after the successful model
  test. It therefore reused the implementation evidence and did not repeat Java tests, as required
  by the planning guide and documentation workflow.
- The documentation context ran `./gradlew :modules:model:javadoc` after final Javadoc edits:
  `BUILD SUCCESSFUL in 1s`; 2 actionable tasks executed and the configuration cache was reused.
- The runnable single-output producer example was compiled with
  `javac -cp modules/model/build/classes/java/main -d /tmp/synaptik-0018l-doc-example
  /tmp/SingleOutputProducerExample.java` and executed with the model classes. It printed the
  documented five lines: `1`, `0`, `true`, `true`, `true`.
- Targeted local Markdown validation over the six affected documentation/planning files passed
  after final edits: 447 local links, including 137 heading anchors, resolved; fenced code blocks
  were balanced; every file had a final newline; and no trailing whitespace was found.
- Final `git diff --check` passed with no output. Scope/status inspection confirmed the task's
  58-path combined inventory: every Java path is under the model tensor production/test package,
  and no architecture, build, architecture-test, conformance, integration, or other-module path
  changed. It also confirmed exactly three producer fields, no output Tensor or independent
  output-count field, no legacy provenance constructor or representation, no graph/compiler/
  backend/runtime imports in producer or provenance, synchronized 0018L `Complete` status, and no
  task 0018M specification.
- `docs/planning/modules/model/capabilities.md` was reviewed but not edited by implementation or
  documentation work: its pre-existing authorized planning diff already selects the exact
  producer/output-descriptor/indexed-result design. `ARCHITECTURE.md` and focused architecture
  documents remain unchanged because this task adds model-owned pre-capture provenance without
  changing module ownership, dependency direction, lifecycle, or graph/runtime identity.
- Repository-wide tests remain deferred to the foundation-contract checkpoint after task 0018N,
  as recorded before implementation.

## Implementation notes

- Added public final identity-based `TensorProducer` with exact operation, immutable ordered input
  Tensor references, immutable ordered output descriptor references, derived output count, and
  signature-cardinality validation.
- Replaced legacy operation/input provenance with exact producer plus zero-based output index.
  Compatibility accessors now derive operation, inputs, and selected descriptor from the producer;
  record equality combines producer identity with the index.
- Hardened package-private Tensor construction so a derived Tensor's descriptor must be the exact
  descriptor reference selected by provenance.
- Replaced the old derived factory seam with producer-aware single-output construction and added
  package-private atomic multi-output construction with immutable result order and documented ID
  side effects.
- Migrated all current expression helpers to the unified factory boundary. Existing operations,
  input order, descriptors, labels, storage-free results, public signatures, and semantic
  validation remain unchanged; current unstack outputs remain independent occurrences.
- Added focused producer, provenance, Tensor, factory, and expression regression coverage for
  exact API shape, snapshots, identity, cardinality, descriptor agreement, multi-output sharing,
  failure order, and ID consumption.
- Finalized affected Javadocs, Tensor API, Compile API, glossary, this task, model master plan, and
  roadmap. The references now distinguish producer output slots from unstack semantic coordinates
  and from graph-local `NodeId`/`ValueId` identities.

## Completion summary

- Completed changes: introduced unified identity-based producer and indexed provenance contracts,
  exact descriptor-slot agreement, single/multi-output package-private factory boundaries, and a
  complete migration of existing single-output expressions.
- Files changed or created: one new production and one new focused test file; the three core
  tensor contracts, current expression helpers, directly affected tensor tests, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.
- Tests and validation: reused the final 749-test/87-suite model run with zero failures, errors, or
  skips; model Javadoc, compiled/running documentation example, 447-link/137-anchor Markdown and
  fence/newline/whitespace checks, final scope/status audit, and `git diff --check` passed.
- Documentation-agent review: clean context
  `/root/task_0018l_implementation/task_0018l_docs` completed the independent API/Javadoc,
  Planning, General, and Example-profile review.
- Documentation impact: Tensor and Compile API references now explain shared producer identity,
  indexed output provenance, exact descriptor ownership, future capture observations, and current
  limitations. Architecture documentation requires no change because no architecture boundary
  changed.
- Javadoc review: all affected public and contract-relevant Javadocs were reviewed and finalized;
  stale helper wording about preconstructed provenance was corrected. No executable Java changed
  during the documentation pass.
- Glossary impact: added `TensorProducer`, updated provenance, Tensor, factory, and unstack
  distinctions, and kept compiler/runtime terminology aligned.
- Unresolved issues: None.
- Follow-up required: None for task 0018L. Task 0018M remains Draft without a detailed
  specification.

Status: Complete
