# Task 0018K: Operation Signature and Construction Hardening

## Status

Complete

## Goal

Make every constructed model `Operation` a valid pairing of one backend-independent
`OperationKind` and one accepted concrete `OperationAttrs` type, and make each kind expose a
compact occurrence signature that describes its permitted input and output counts.

Use that signature to reject a locally malformed `CompiledNode` without introducing a global
operation registry, reflective discovery, operand-aware inference, compiler policy, backend
metadata, or runtime behavior.

## Scope

- Add one public immutable `OperationSignature` value in the existing `model.operation` package.
- Represent one accepted attribute variant with:
  - the exact concrete `OperationAttrs` implementation class;
  - inclusive minimum and maximum input counts; and
  - inclusive minimum and maximum output counts.
- Extend `OperationKind` with a stable non-empty list of accepted signatures and a default exact
  attribute-class resolver.
- Require every current production kind family to declare all and only the variants in the
  [signature matrix](#current-production-signature-matrix).
- Keep signature declarations colocated with their owning kind family. Do not centralize the
  inventory in `Operation`, a map, a registry, or a discovery service.
- Harden the existing two-component `Operation` record so construction resolves and validates the
  kind/attributes pair immediately.
- Add a derived `Operation.signature()` accessor without storing signature as a third record
  component or additional instance field.
- Harden `CompiledNode` after its existing structural list validation so the final ordered input
  and output counts must be accepted by the operation signature.
- Preserve zero-input and multi-output graph-node capability when a future or test-local kind
  explicitly declares such a signature.
- Update current kind-family tests and test-local kind definitions for the intentional signature
  contract, and add focused signature and node-cardinality coverage.
- Finalize affected Javadoc, Tensor API, Compile API, glossary, task evidence, master plan, and
  roadmap through the required targeted documentation pass.

## Out of scope

- shared producer identity, Tensor output slots, or any change to `TensorProvenance`; task 0018L
  owns true multi-output public provenance
- adding a production multi-output kind merely to exercise the contract; focused tests use a
  test-local kind
- changing `Operation` record components, equality, hashing, or diagnostic value semantics
- a generic operation factory, builder, parser, serializer, schema generator, annotation
  processor, classpath scan, reflective plugin discovery, service locator, or mutable registry
- string-keyed kind or attribute lookup, class-name dispatch, a monolithic operation enum, or a
  root-package switch over every concrete production family
- Shape, DataType, layout, descriptor, axis-bound, broadcasting, tuple-depth, or
  operation-family numerical validation
- graph-wide value existence, producer uniqueness, topology, phase, descriptor agreement, cycle,
  publication, or inference validation
- public Tensor API additions, expression construction, provenance changes, graph capture, ID
  allocation, or operation canonicalization
- gradients, autograd traversal, backward graph construction, compiler transformations, planning,
  backend ownership, lowering, fusion, kernel selection, prepare, runtime, or execution
- changing the selected capability vocabulary; cleanup and renaming remain in tasks 0018O–0018S
- dependencies, Gradle, architecture tests, another module, or `ARCHITECTURE.md`
- creating a detailed task-0018L specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially `modules/model` ownership,
  backend-independent `Operation`, immutable `CompiledNode`, and runtime hot-path exclusions
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially “Valid operations and occurrence
  signatures”
- [Model master plan](../master-plan.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0008](0008-graph-value-and-node-model.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns these backend-independent semantics and local immutable graph contracts.
- `Operation` remains model-level semantic state and must contain no backend support, ownership,
  route, kernel, cost, fusion, storage, compiler service, or runtime state.
- `CompiledNode` remains immutable compile-time graph state and must not enter runtime hot paths.
- A signature describes structural occurrence counts only. It does not establish operand
  compatibility or executable support.
- Kind-to-signature declarations remain in each concrete kind family. The root operation package
  may provide generic validation mechanics but must not import or enumerate concrete families.
- Attribute matching uses an explicitly declared `Class<? extends OperationAttrs>` token and exact
  concrete-class equality. This is typed local validation, not classpath scanning, reflective
  discovery, class-name dispatch, or a registry.
- `Operation` continues to store exactly `kind` and `attrs`. A signature is derived from those two
  values so there is no independently supplied or stored second source of truth.
- `CompiledNode` validates only its own ordered collection sizes after preserving its existing
  null, snapshot, non-empty-output, and duplicate-output rules. Whole-graph validation remains in
  `CompiledGraphModel` and later compiler work.
- Public `Tensor` remains API state rather than IR. No graph-local identity is added to Tensor or
  provenance.
- If a valid current production kind/attributes pairing cannot be represented by this task's
  matrix, or implementation requires a central registry or cross-layer metadata, stop and report
  the design conflict before changing code.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — owns the generic signature, kind, attributes, and
  operation construction contracts.
- Existing `io.github.pho001.synaptik.model.operation.*` family packages — own their local accepted
  signature declarations.
- `io.github.pho001.synaptik.model.graph` — consumes the resolved operation signature only to
  validate one node's ordered input and output counts.

Packages added or changed:

- No package is added.
- Existing operation-family packages gain signature declarations but retain their present
  semantic ownership.

Type placement:

- `io.github.pho001.synaptik.model.operation.OperationSignature` — generic immutable structural
  contract for one accepted attributes class and inclusive occurrence-count bounds.
- `io.github.pho001.synaptik.model.operation.OperationKind` — exposes family-owned signature
  variants and resolves the exact variant for supplied attributes.
- `io.github.pho001.synaptik.model.operation.Operation` — validates and exposes its derived
  signature while retaining exactly its existing two record components.
- `io.github.pho001.synaptik.model.graph.CompiledNode` — validates occurrence counts after its
  existing local list validation.

Test placement:

- `io.github.pho001.synaptik.model.operation.OperationSignatureTest` — validates the new value,
  exact attribute matching, bounds, and fixed/bounded/variadic count behavior.
- Existing operation-family tests validate the matrix and replace obsolete “no project metadata”
  expectations with the exact intentional signature surface.
- Existing graph tests validate node-count enforcement and preserve test-only zero-input and
  multi-output coverage through explicit test-local signatures.

## Required contracts

### `OperationSignature`

Create this public record in `io.github.pho001.synaptik.model.operation`:

```java
public record OperationSignature(
        Class<? extends OperationAttrs> attributesType,
        int minimumInputs,
        int maximumInputs,
        int minimumOutputs,
        int maximumOutputs) {
    // Explicit documented constructor, accessors, matching, and count validation.
}
```

The record must:

- reject null `attributesType`;
- require `minimumInputs >= 0`;
- require `maximumInputs >= minimumInputs`;
- require `minimumOutputs >= 1`, matching the graph contract that a computation occurrence
  produces one or more values;
- require `maximumOutputs >= minimumOutputs`;
- treat `Integer.MAX_VALUE` as the real maximum possible Java `List.size()` and therefore as the
  inclusive upper bound for an effectively variadic position, not as a negative or hidden
  sentinel;
- retain the exact `Class` token and primitive bounds;
- match attributes by exact runtime class, not assignability, class name, package name, or
  reflection-based discovery;
- expose documented boolean queries for attribute, input-count, and output-count acceptance;
- expose one documented occurrence-validation method that rejects a count outside either range;
- use record-generated equality, hashing, and diagnostic text; and
- allocate no collection, map, registry entry, or cached mutable state per validation call.

Failure messages must identify the invalid field or count and the accepted inclusive range. Exact
wording is part of the focused tests only where needed to distinguish input from output failures;
do not create a broad message-format protocol.

### `OperationKind`

Keep `OperationKind` open to future family enums and retain `String name()`. Add:

```java
List<OperationSignature> signatures();

default OperationSignature signatureFor(OperationAttrs attrs) { ... }
```

The contract for `signatures()` requires a stable, immutable, non-empty ordered list. Each element
must be non-null, and no two variants for one kind may declare the same exact `attributesType`.
The list is family-owned semantic metadata; it is not a global registry and contains no operand,
backend, compiler, or execution facts.

`signatureFor` must:

1. reject null attributes;
2. inspect the family-provided variants in stable declaration order;
3. fail if the list is null, empty, contains null, or repeats an attributes class;
4. return the unique variant whose `attributesType` exactly equals `attrs.getClass()`; and
5. otherwise reject the pairing with an error that identifies the typed kind, actual attributes
   class, and accepted classes.

Production enums should create their immutable signature values and lists once during class
initialization. Resolving an operation must not allocate a new signature or signature list for
each Tensor expression. Intentional enum fields, constructors, and methods that implement this
contract replace the earlier temporary “no project state or behavior” restriction.

### `Operation`

Preserve exactly these record components and their order:

```java
public record Operation(OperationKind kind, OperationAttrs attrs)
```

After the existing ordered null checks, the canonical constructor must call the kind's signature
resolver. A wrong kind/attributes pair is rejected before the operation can enter Tensor
provenance or a graph. Do not copy, normalize, replace, or otherwise alter valid component
references.

Add one public derived accessor:

```java
OperationSignature signature();
```

It returns the stable family-owned signature resolved from the stored kind and attributes. It is
not a record component or stored field. Record-generated equality, hashing, and `toString()` remain
based only on `kind` and `attrs`.

### `CompiledNode`

Preserve exactly the existing four record components and all current collection ownership rules.
The canonical constructor validation order remains:

1. null component references;
2. indexed null input elements, then immutable input snapshot;
3. non-empty outputs;
4. indexed null and duplicate output elements, then immutable output snapshot; and
5. operation-signature validation of final `inputs.size()` and `outputs.size()`.

This order preserves useful structural failures before semantic count failures. Empty inputs and
multiple outputs are valid only when the supplied operation signature accepts those counts.
Repeated inputs remain valid positions. Outputs remain non-empty and unique within the node.

Do not add a stored signature, index, descriptor, producer, consumer, or graph validator.

## Current production signature matrix

`No attrs` below means the exact class of `NoOperationAttrs.INSTANCE`. Every listed output count is
exactly one in the current production inventory; bounded and multi-output behavior is nevertheless
implemented and tested for future kinds.

| Kind family / constants | Accepted attributes | Inputs | Outputs |
|---|---|---:|---:|
| `BinaryArithmeticKind.*` | `NoOperationAttrs` | exactly 2 | exactly 1 |
| `CastKind.CAST` | `CastAttrs` | exactly 1 | exactly 1 |
| `BinaryComparisonKind.*` | `NoOperationAttrs` | exactly 2 | exactly 1 |
| `BooleanLogicalKind.AND`, `OR` | `NoOperationAttrs` | exactly 2 | exactly 1 |
| `BooleanLogicalKind.NOT` | `NoOperationAttrs` | exactly 1 | exactly 1 |
| `ScalarElementwiseKind.MUL`, `POW`, `CLAMP_MIN`, `CLAMP_MAX` | `ScalarValueAttrs` | exactly 1 | exactly 1 |
| `ScalarElementwiseKind.CLAMP` | `ClampRangeAttrs` | exactly 1 | exactly 1 |
| `WhereSelectionKind.WHERE` | `NoOperationAttrs` | exactly 3 | exactly 1 |
| `UnaryElementwiseKind.*` | `NoOperationAttrs` | exactly 1 | exactly 1 |
| `AxisGatherKind.*` | `IndexAxisAttrs` | exactly 2 | exactly 1 |
| `AxisScatterKind.SCATTER_ADD`, `SCATTER_AXIS_ADD` | `IndexAxisAttrs` | exactly 3 | exactly 1 |
| `AxisScatterKind.SCATTER_ELEMENTS` | `ScatterElementsAttrs` | exactly 3 | exactly 1 |
| `GatherNdKind.GATHER_ND` | `GatherNdAttrs` | exactly 2 | exactly 1 |
| `ScatterNdKind.SCATTER_ND` | `ScatterNdAttrs` | exactly 3 | exactly 1 |
| `SelectKind.SELECT` | `SelectAttrs` | exactly 1 | exactly 1 |
| `AxisTransformKind.PERMUTE` | `PermutationAttrs` | exactly 1 | exactly 1 |
| `AxisTransformKind.EXPAND_DIMS`, `SQUEEZE` | `AxisTransformAttrs` | exactly 1 | exactly 1 |
| `ContiguousKind.CONTIGUOUS` | `NoOperationAttrs` | exactly 1 | exactly 1 |
| `PadKind.PAD` | `PadAttrs` | exactly 1 | exactly 1 |
| `ShapeTransformKind.RESHAPE`, `EXPAND` | `TargetShapeAttrs` | exactly 1 | exactly 1 |
| `SliceKind.SLICE` | `SliceAttrs` | exactly 1 | exactly 1 |
| `TensorCompositionKind.CONCAT`, `STACK` | `CompositionAxisAttrs` | 1 to `Integer.MAX_VALUE` | exactly 1 |
| `TensorCompositionKind.UNSTACK` | `UnstackOutputAttrs` | exactly 1 | exactly 1 |
| `TileKind.TILE` | `TileAttrs` | exactly 1 | exactly 1 |
| `WindowTransformKind.UNFOLD_AXIS` | `UnfoldAxisAttrs` | exactly 1 | exactly 1 |
| `WindowTransformKind.FOLD_AXIS` | `FoldAxisAttrs` | exactly 1 | exactly 1 |
| `WindowTransformKind.UNFOLD2D` | `Window2dAttrs` | exactly 1 | exactly 1 |
| `WindowTransformKind.FOLD2D` | `Fold2dAttrs` | exactly 1 | exactly 1 |
| `SoftmaxKind.SOFTMAX`, `LOG_SOFTMAX` | `SoftmaxAttrs` | exactly 1 | exactly 1 |
| `AggregateReductionKind.SUM`, `MEAN` | `NoOperationAttrs` or `AxisReductionAttrs` | exactly 1 | exactly 1 |
| `AggregateReductionKind.SUM`, `MEAN` masked form | `MaskedReductionAttrs` | exactly 2 | exactly 1 |
| `AggregateReductionKind.PROD`, `MIN`, `MAX`, `ALL`, `ANY` | `NoOperationAttrs` or `AxisReductionAttrs` | exactly 1 | exactly 1 |
| `AggregateReductionKind.ARG_MAX` | `ArgMaxAttrs` | exactly 1 | exactly 1 |
| `CumulativeSumKind.CUM_SUM` | `CumulativeSumAttrs` | exactly 1 | exactly 1 |

No production kind may accept an attributes implementation absent from this matrix. Task 0018K
does not remove provisional kinds; later reset tasks perform the selected vocabulary cleanup.

## Affected files

Expected core production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationSignature.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/NoOperationAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledNode.java`

Expected family production files:

- every current production `*Kind.java` below
  `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/` listed in the signature
  matrix; no attributes declaration or behavior changes are expected

Expected test files:

- new `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- existing `OperationKindTest`, `OperationTest`, `CompiledNodeTest`, and
  `CompiledGraphModelTest`
- existing operation-family `*KindTest.java` and `*SemanticsTest.java` files whose exact enum
  shape or composition assertions must reflect the new intentional signature contract
- existing `TensorTest`, `TensorFactoryTest`, and `TensorProvenanceTest` only to give their
  test-local kinds explicit signatures; production Tensor behavior must not change
- one focused production-signature coverage test may be added if it replaces duplicated mapping
  assertions and verifies every current production enum constant against the matrix

Expected documentation and planning files:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Expected no-change reviews:

- `docs/planning/modules/model/capabilities.md` already selects this exact hardening direction;
- focused architecture documents and `ARCHITECTURE.md` already assign these responsibilities to
  model and require no rule change.

## Maximum scope

This task may create or modify at most:

- one new production Java file;
- the five core existing production files listed above;
- every current production operation-kind enum listed by the matrix, and no other production
  family type;
- one new focused signature test plus only the existing tests identified above that require
  signature assertions or explicit test-local signatures; and
- the six documentation/planning files listed above.

This is an intentional exception to the normal 12–18-file guardrail. The invariant is atomic:
enabling constructor validation while leaving existing production families undeclared would make
valid current operations unconstructable, while allowing undeclared families through a fallback
would preserve the invalid-state gap that this task must close. Most family changes are a local
signature declaration, corresponding Javadoc correction, and automated test expectation.

Do not use the larger path allowance for unrelated refactors, formatting, semantic cleanup, or
Tensor behavior. If another production concept, attributes behavior change, module, build file,
or architecture document is required, stop and report the needed follow-up.

## Javadoc requirements

- Document `OperationSignature` as a structural occurrence contract, define every component,
  exact attributes-class matching, inclusive bounds, effective variadic maximum, failures, value
  semantics, and the distinction from Tensor rank and operand compatibility.
- Update `OperationKind` to explain family-owned signatures, exact attribute variants, stable
  immutable lists, and the absence of a global registry or executable metadata.
- Update `Operation` to state that kind/attributes compatibility is now enforced and that
  `signature()` is derived rather than stored.
- Update `CompiledNode` to explain signature-based local count validation while preserving the
  boundary with graph-wide and operand-aware validation.
- Correct every affected family Javadoc that currently says arity is only prose, the enum has no
  project metadata, or generic `Operation` does not enforce pairings.
- Review `OperationAttrs`, `NoOperationAttrs`, and related attributes Javadocs for accurate exact
  matching terminology; change only stale contract wording.
- Explain all new terms for a newcomer in Tensor API and glossary, including “operation
  signature”, “occurrence cardinality”, “fixed”, “bounded”, and “variadic”.
- Update Compile API only for the now-current `CompiledNode` local count validation. Do not claim
  compiler capture, descriptor inference, or runtime validation is implemented.

## Acceptance criteria

- `OperationSignature` has exactly the five specified record components in order and no mutable
  state.
- Signature construction enforces valid inclusive input/output ranges and permits fixed, bounded,
  zero-input, effectively variadic-input, and multi-output contracts.
- Attribute matching is exact-class based and uses no string comparison, classpath scan,
  annotation scan, service lookup, or global kind map.
- `OperationKind` retains `name()`, requires stable family-owned signature variants, and resolves
  one unique variant for an attributes object.
- A kind with missing, empty, null, duplicate, or nonmatching signature declarations fails closed;
  no permissive fallback signature exists.
- Every production kind constant and accepted attributes variant matches the complete matrix.
- Every intentionally wrong current cross-family pairing is rejected during `Operation`
  construction, including representative parameterless, cast, scalar, reduction, layout, index,
  and window cases.
- `Operation` remains a record with exactly `kind` and `attrs`, retains exact valid references,
  and keeps generated equality, hashing, and diagnostic text based only on those components.
- `Operation.signature()` returns the matching stable family-owned signature and adds no record
  component or instance field.
- `CompiledNode` preserves its record shape, snapshots, input repetition, output non-empty and
  uniqueness rules, then rejects input or output counts outside the signature.
- Focused tests prove exact unary, binary, ternary, bounded, variadic, zero-input, one-output, and
  multi-output behavior without introducing a production sample kind.
- Existing production Tensor expressions continue to construct their exact operation semantics
  and all model tests pass.
- No `Map` or registry of all operation kinds, no root switch over concrete families, and no
  reflective discovery appears in production.
- No backend, compiler, planning, prepare, runtime, execution, gradient, cost, fusion, kernel,
  storage, or device metadata is added.
- All affected public contracts and enum constants have complete current Javadoc.
- The targeted clean-context documentation pass finalizes Javadoc, Tensor API, Compile API,
  glossary, planning evidence, links, and terminology without rerunning successful Java tests
  unless executable behavior changed after the recorded run.
- Task, master-plan row, and roadmap row have matching final status.
- Task 0018L remains Draft without a detailed specification.

## Tests / validation

During implementation, use focused tests as needed. After executable Java code stabilizes, record
one final module run:

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

Repository-wide `./gradlew test` is deferred to the recorded foundation-contract capability
checkpoint after task 0018N. This task changes one module and no dependency or architecture
boundary.

Automated tests must cover the API-shape and signature matrix invariants. Do not require repeated
manual `javap`, reflection, or bytecode commands when the same facts are asserted in tests.

Final scope review must confirm:

- only `modules/model` Java files and the allowed documentation/planning files changed;
- `Operation` and `CompiledNode` record components did not change;
- every production `OperationKind` constant appears in signature coverage;
- no concrete family import appears in the root generic resolver;
- no Java/Gradle/module dependency or architecture file changed; and
- 0018K is `Complete` everywhere only after both passes, while 0018L has no task file.

## Dependencies

- Task 0005: `OperationKind`, `OperationAttrs`, and `NoOperationAttrs` — Complete.
- Task 0006: immutable two-component `Operation` — Complete.
- Task 0008: immutable `CompiledNode` occurrence with ordered inputs and outputs — Complete.
- Current production operation-family tasks through 0018J — Complete and inventoried by the
  signature matrix.

## Follow-up tasks

- 0018L — use the output-cardinality foundation for shared multi-output Tensor provenance without
  adding graph-local identity to Tensor.
- 0018O — normalize indexing taxonomy and replace independent UNSTACK semantics with repeated
  scalar select convenience.
- 0018P–0018V — perform the selected semantic cleanup and capability additions on top of validated
  signatures.
- 0019B and 0019C — use explicit multi-output signatures for graph RNG/dropout and top-K.

Do not create detailed specifications for these follow-ups in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns backend-independent operation semantics and immutable graph state
to `modules/model`. This task makes those existing local contracts stricter without changing
module ownership, dependency direction, lifecycle, or runtime visibility.

If implementation requires an architecture rule change, stop and report the exact conflict before
editing `ARCHITECTURE.md` or focused architecture documentation.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, task 0018K, and the affected operation/graph source and tests in full.

Implement task 0018K exactly as specified. Stay inside modules/model and the explicitly allowed
documentation/planning files. Stop on any scope or architecture conflict. Run the task-level model
validation once after executable code stabilizes.

Then hand the actual diff and recorded Java-test evidence to a separate clean-context
documentation-focused agent in the same change. That agent must inspect the final source/tests,
finalize affected Javadocs, Tensor API, Compile API, glossary, task/master/roadmap status and
documentation validation, and must not repeat successful Java tests unless executable behavior
changed or it records a concrete reason.

Do not mark 0018K Complete until both passes succeed. Leave 0018L Draft without a detailed
specification. Do not commit or push.
```

## Local decisions

- `OperationSignature.fixed(...)` and `inputRange(...)` provide readable declaration helpers while
  the five-component canonical record remains the complete value contract.
- Production and test-local enum kinds retain stable immutable signature lists in static final
  fields. Resolution therefore reuses family-owned values rather than allocating per operation.
- `Integer.MAX_VALUE` is the literal inclusive upper bound for an effectively variadic Java list;
  no sentinel translation or separate variadic flag is used.
- `Operation` resolves once during construction to reject an invalid pair and resolves again only
  through the derived `signature()` accessor. The signature remains absent from record state, so
  equality, hashing, and diagnostic text keep their two-component contract.
- `CompiledNode` performs signature-cardinality validation only after its existing input/output
  list checks and immutable snapshots, preserving the established failure order.

## Known limitations

- Signatures describe exact attributes classes and local occurrence counts only. They do not
  validate operand data types, Shapes, layouts, axes, descriptors, numerical policy, graph-wide
  relationships, compiler behavior, backend support, or executability.
- No current production kind is multi-output. Zero-input, bounded, effectively variadic, and
  multi-output contracts are covered with generic or test-local signatures; task 0018L still owns
  shared multi-output Tensor provenance.
- Public Tensor expression construction and `TensorProvenance` remain unchanged. A valid local
  signature does not establish shared producer identity or graph capture.

## Validation evidence

- Implementation context `/root` ran `./gradlew :modules:model:test` after executable Java and
  test changes stabilized: `BUILD SUCCESSFUL in 1s`. The generated report contains 743 tests in
  86 suites, with 0 failures, 0 errors, and 0 skipped tests. The documentation-focused pass reused
  this evidence and changed only Javadoc and documentation afterward.
- Clean documentation-focused context `/root/task_0018k_docs` independently reviewed
  `AGENTS.md`, `ARCHITECTURE.md`, the current architecture index, documentation rules and General,
  API/Javadoc, Planning, and Example profiles, the planning guide and roadmap, the capability
  baseline, model master plan, this task, affected source/tests and their actual diff, Tensor API,
  Compile API, and glossary.
- `/root/task_0018k_docs` ran `./gradlew :modules:model:javadoc` after final Javadoc edits:
  `BUILD SUCCESSFUL in 1s`; 2 actionable tasks executed and the configuration cache was reused.
- `/root/task_0018k_docs` ran the targeted local Markdown checker over the six affected
  documentation/planning files. It passed 413 local links, including 110 anchor links, and
  confirmed balanced fences and final newlines in all six files.
- `/root/task_0018k_docs` ran `git diff --check` on the final combined change: passed with no
  output.
- Final scope inspection confirmed that `Operation` remains a two-component record,
  `CompiledNode` remains a four-component record, every production kind is covered by the
  automated signature matrix, the generic operation resolver imports no concrete family, and no
  Gradle file, dependency, module boundary, `ARCHITECTURE.md`, focused architecture document,
  architecture test, backend-conformance test, integration test, or other module changed for this
  task. No task-0018L specification exists.
- `docs/planning/modules/model/capabilities.md` remains unchanged by task 0018K because its
  “Valid operations and occurrence signatures” section already selected exact family-owned
  attribute pairing and fixed/bounded/variadic cardinality. `ARCHITECTURE.md` and focused
  architecture documentation remain unchanged because the implementation only tightens existing
  model-owned semantic and immutable graph contracts without changing ownership, dependencies, or
  lifecycle rules.
- Repository-wide tests remain deferred to the foundation-contract checkpoint after task 0018N,
  as recorded before implementation. No Java test was repeated by the documentation-focused pass
  because it made no executable change after the successful module run.

## Implementation notes

- Added the five-component immutable `OperationSignature` record with validated inclusive bounds,
  exact runtime attributes-class matching, fixed and input-range declaration helpers, count
  queries, and occurrence validation.
- Extended `OperationKind` with stable family-owned signature variants and an exact resolver that
  fails closed for null, empty, malformed, duplicate, or incompatible declarations without a
  global registry or concrete-family switch.
- Hardened `Operation` construction against incompatible kind/attributes pairs and added the
  derived `signature()` accessor without changing record components or retaining signature state.
- Declared the complete task matrix in every current production kind family. Current production
  variants remain one-output; concat and stack accept one through `Integer.MAX_VALUE` inputs.
- Hardened `CompiledNode` to validate final local input/output counts after its established list
  checks and snapshots while preserving repeated inputs, explicitly declared zero-input kinds,
  and explicitly declared multi-output kinds.
- Added focused signature, resolver, operation, node-cardinality, and complete production-matrix
  coverage; updated existing test-local kinds to declare stable signatures.
- Finalized `OperationSignature`, `OperationKind`, `Operation`, `CompiledNode`, family-kind, and
  directly related attribute Javadocs. Corrected stale pairing and temporal claims in select,
  target-shape, permutation, and slice documentation without changing behavior.
- Updated Tensor API, Compile API, glossary, task evidence, model master plan, and roadmap. The
  reference material now defines operation signature, occurrence cardinality, fixed, bounded,
  and variadic, and separates local structural validation from operand-aware, graph-wide,
  compiler, backend, and runtime responsibilities.

## Completion summary

- Completed changes: added exact family-owned operation signatures, construction-time
  kind/attributes validation, derived signature access, complete production declarations, and
  local compiled-node occurrence-count validation.
- Files changed or created: new `OperationSignature.java`; updated core operation and
  `CompiledNode` contracts; updated every production operation-kind enum and directly affected
  attribute Javadocs; added `OperationSignatureTest`; updated focused operation, graph, and
  test-local-kind tests; finalized Tensor API, Compile API, glossary, this task, model master plan,
  and roadmap.
- Tests and validation: reused the final 743-test/86-suite model run with zero failures, errors, or
  skips; model Javadoc generation, 413-link/110-anchor Markdown checks, fence/final-newline checks,
  final scope checks, and `git diff --check` passed.
- Documentation-agent review: clean context `/root/task_0018k_docs` completed the required
  independent API/Javadoc and planning-profile review.
- Documentation impact: Tensor and Compile API references now explain the structural contract and
  its boundaries; planning evidence and synchronized status are final. Architecture and focused
  architecture documents require no change because ownership and lifecycle boundaries did not
  change.
- Javadoc review: all affected public and contract-relevant Javadocs were reviewed; stale exact
  pairing and already-implemented-convenience wording was corrected. No executable Java changed
  during the documentation pass.
- Glossary impact: added and aligned operation signature, occurrence cardinality, fixed, bounded,
  and variadic terminology and their distinction from Tensor rank and executable support.
- Unresolved issues: None.
- Follow-up required: None for task 0018K. Task 0018L remains the next Draft frontier and owns
  shared multi-output Tensor provenance.

Status: Complete
