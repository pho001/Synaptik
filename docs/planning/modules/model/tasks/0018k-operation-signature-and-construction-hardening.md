# Task 0018K: [Operation Signature](../../../../glossary.md#operation-signature) and Construction Hardening

**Editorial reading key (current context, not a status update).**

This task records historical construction hardening, not today’s operation inventory or provenance
scope. An `Operation` pairs a kind (calculation meaning) with attributes (fixed parameters); its
constructor calls the kind’s exact-class signature resolver. A signature permits input/output
position counts, not tensor axes or elements: binary addition takes two inputs even when each has a
million elements. Fixed means equal bounds, bounded means differing bounds, and effectively variadic
uses the inclusive `Integer.MAX_VALUE` ceiling. `CompiledNode` checks final list sizes after
structural checks, not operand compatibility or executability. Follow the [signature
foundation](../../../../reference/tensor/operation-foundation.md#operation-semantic-foundation) for
current variants.

The historical multi-output deferral does not describe current [producer
identity](../../../../reference/tensor/identity.md#expression-producers-and-indexed-provenance):
follow it for shared occurrences and indexed results. Callers supply operation descriptions and
Tensor references, not computed result values. Storage-free expressions neither execute nor own
result buffers. Model semantics defines meaning; execution providers realize supported routes
without redefining it. Semantic responsibility is not storage ownership, and shared producer
identity does not imply shared storage.

## Status

Complete

## Goal

Make every constructed [model](../../../../glossary.md#term-model) [`Operation`](../../../../glossary.md#operation) a valid pairing of one [backend](../../../../glossary.md#backend)-independent
[`OperationKind`](../../../../glossary.md#operationkind) and one accepted concrete [`OperationAttrs`](../../../../glossary.md#operationattrs) type, and make each kind expose a
compact [occurrence](../../../../glossary.md#term-occurrence) [signature](../../../../glossary.md#term-signature) that describes its permitted input and output counts.

Use that signature to reject a locally malformed `CompiledNode` without introducing a global
operation [registry](../../../../glossary.md#term-registry), reflective discovery, operand-aware [inference](../../../../glossary.md#term-inference), [compiler](../../../../glossary.md#term-compiler) policy, backend
[metadata](../../../../glossary.md#term-metadata), or [runtime](../../../../glossary.md#term-runtime) behavior.

## [Scope](../../../../glossary.md#term-scope)

- Add one public [immutable](../../../../glossary.md#term-immutable) `OperationSignature` value in the existing `model.operation` package.
- Represent one accepted attribute variant with:
  - the exact concrete [`OperationAttrs`](../../../../glossary.md#operationattrs) implementation class;
  - inclusive minimum and maximum input counts; and
  - inclusive minimum and maximum output counts.
- Extend [`OperationKind`](../../../../glossary.md#operationkind) with a stable non-empty list of accepted signatures and a default exact
  attribute-class resolver.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Require every current production kind family to declare all and only the variants in the
  [signature matrix](#current-production-signature-matrix)<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>.
- Keep signature declarations colocated with their owning kind family. Do not centralize the
  inventory in [`Operation`](../../../../glossary.md#operation), a map, a [registry](../../../../glossary.md#term-registry), or a discovery service.
- Harden the existing two-component `Operation` record so construction resolves and validates the
  kind/attributes pair immediately.
- Add a derived `Operation.signature()` accessor without storing signature as a third record
  component or additional instance field.
- Harden `CompiledNode` after its existing structural list validation so the final ordered input
  and output counts must be accepted by the operation signature.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation signature glossary](../../../../glossary.md#operation-signature))</span>
- Preserve zero-input and multi-output graph-node capability when a future or test-local kind
  explicitly declares such a signature.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node))</span>
- Update current kind-family tests and test-local kind definitions for the intentional signature
  contract, and add focused signature and node-cardinality coverage.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- Finalize affected Javadoc, Tensor API, Compile API, glossary, task evidence, master plan, and
  roadmap through the required targeted documentation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- shared [producer](../../../../glossary.md#term-producer) [identity](../../../../glossary.md#term-identity-meanings), [Tensor](../../../../glossary.md#tensor) output slots, or any change to `TensorProvenance`; task 0018L
  owns true multi-output public provenance<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- adding a production multi-output kind merely to exercise the contract; focused tests use a
  test-local kind
- changing [`Operation`](../../../../glossary.md#operation) record components, equality, hashing, or diagnostic value semantics
- a generic operation factory, builder, parser, serializer, schema generator, annotation
  processor, classpath scan, reflective plugin discovery, service locator, or mutable registry<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [classpath glossary](../../../../glossary.md#term-classpath); [plugin glossary](../../../../glossary.md#term-plugin); [service locator glossary](../../../../glossary.md#term-service-locator); [mutable glossary](../../../../glossary.md#term-mutable); [registry glossary](../../../../glossary.md#term-registry))</span>
- string-keyed kind or attribute lookup, class-name dispatch, a monolithic operation enum, or a
  root-package switch over every concrete production family<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Shape, DataType, layout, descriptor, axis-bound, broadcasting, tuple-depth, or
  operation-family numerical validation<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [axis glossary](../../../../glossary.md#term-axis); [broadcasting glossary](../../../../glossary.md#broadcasting); [validation glossary](../../../../glossary.md#term-validation))</span>
- graph-wide value existence, producer uniqueness, topology, phase, descriptor agreement, cycle,
  publication, or inference validation<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [topology glossary](../../../../glossary.md#term-topology); [publication glossary](../../../../glossary.md#term-publication); [inference glossary](../../../../glossary.md#term-inference))</span>
- public Tensor API additions, expression construction, provenance changes, graph capture, ID
  allocation, or operation canonicalization<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- gradients, autograd traversal, backward graph construction, compiler transformations, planning,
  backend ownership, lowering, fusion, kernel selection, prepare, runtime, or execution<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [backward graph glossary](../../../../glossary.md#backward-graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [backend ownership glossary](../../../../glossary.md#backend-ownership); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- changing the selected capability vocabulary; cleanup and renaming remain in tasks 0018O–0018S
- dependencies, [Gradle](../../../../glossary.md#term-gradle), architecture tests, another [module](../../../../glossary.md#term-module), or `ARCHITECTURE.md`
- creating a detailed task-0018L specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially `modules/model` ownership,
  backend-independent <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend))</span>[`Operation`](../../../../glossary.md#operation), [immutable](../../../../glossary.md#term-immutable) `CompiledNode`, and [runtime](../../../../glossary.md#term-runtime) hot-path exclusions
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially “Valid operations and occurrence
  signatures”<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [Model master plan](../master-plan.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0008](0008-graph-value-and-node-model.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns these [backend](../../../../glossary.md#backend)-independent semantics and local [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) contracts.
- [`Operation`](../../../../glossary.md#operation) remains model-level semantic state and must contain no backend support, ownership,
  route, kernel, cost, fusion, storage, compiler service, or runtime state.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [kernel glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- `CompiledNode` remains immutable [compile](../../../../glossary.md#compile)-time graph state and must not enter [runtime](../../../../glossary.md#term-runtime) hot paths.
- A signature describes structural occurrence counts only. It does not establish operand
  compatibility or executable support.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Kind-to-signature declarations remain in each concrete kind family. The root operation package
  may provide generic validation mechanics but must not import or enumerate concrete families.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Attribute matching uses an explicitly declared `Class<? extends OperationAttrs>` token and exact
  concrete-class equality. This is typed local validation, not classpath scanning, reflective
  discovery, class-name dispatch, or a registry.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [classpath glossary](../../../../glossary.md#term-classpath); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry))</span>
- `Operation` continues to store exactly `kind` and `attrs`. A signature is derived from those two
  values so there is no independently supplied or stored second source of truth.
- `CompiledNode` validates only its own ordered collection sizes after preserving its existing
  null, snapshot, non-empty-output, and duplicate-output rules. Whole-graph validation remains in<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
  [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) and later [compiler](../../../../glossary.md#term-compiler) work.
- Public [`Tensor`](../../../../glossary.md#tensor) remains API state rather than IR. No graph-local identity is added to Tensor or
  provenance.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- If a valid current production kind/attributes pairing cannot be represented by this task's
  matrix, or implementation requires a central registry or cross-layer metadata, stop and report
  the design conflict before changing code.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [metadata glossary](../../../../glossary.md#term-metadata))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — owns the generic signature, kind, attributes, and
  operation construction contracts.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [operation glossary](../../../../glossary.md#operation))</span>
- Existing `io.github.pho001.synaptik.model.operation.*` family packages — own their local accepted
  signature declarations.
- `io.github.pho001.synaptik.model.graph` — consumes the resolved operation signature only to
  validate one node's ordered input and output counts.<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature); [node glossary](../../../../glossary.md#node))</span>

Packages added or changed:

- No package is added.
- Existing operation-family packages gain signature declarations but retain their present
  semantic ownership.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Type placement:

- `io.github.pho001.synaptik.model.operation.OperationSignature` — generic immutable structural
  contract for one accepted attributes class and inclusive occurrence-count bounds.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `io.github.pho001.synaptik.model.operation.OperationKind` — exposes family-owned signature
  variants and resolves the exact variant for supplied attributes.
- `io.github.pho001.synaptik.model.operation.Operation` — validates and exposes its derived
  signature while retaining exactly its existing two record components.
- `io.github.pho001.synaptik.model.graph.CompiledNode` — validates occurrence counts after its
  existing local list validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

Test placement:

- `io.github.pho001.synaptik.model.operation.OperationSignatureTest` — validates the new value,
  exact attribute matching, bounds, and fixed/bounded/variadic count behavior.<span data-glossary-navigation="true"> ([variadic glossary](../../../../glossary.md#term-variadic))</span>
- Existing operation-family tests validate the matrix and replace obsolete “no project metadata”
  expectations with the exact intentional signature surface.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Existing graph tests validate node-count enforcement and preserve test-only zero-input and
  multi-output coverage through explicit test-local signatures.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>

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
  produces one or more values;<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- require `maximumOutputs >= minimumOutputs`;
- treat `Integer.MAX_VALUE` as the real maximum possible Java `List.size()` and therefore as the
  inclusive upper bound for an effectively variadic position, not as a negative or hidden
  sentinel;<span data-glossary-navigation="true"> ([variadic glossary](../../../../glossary.md#term-variadic))</span>
- retain the exact `Class` [token](../../../../glossary.md#term-token) and primitive bounds;
- match attributes by exact runtime class, not assignability, class name, package name, or
  reflection-based discovery;<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- expose documented [boolean](../../../../glossary.md#data-type--datatype) queries for attribute, input-count, and output-count acceptance;
- expose one documented [occurrence](../../../../glossary.md#term-occurrence)-[validation](../../../../glossary.md#term-validation) method that rejects a count outside either range;
- use record-generated equality, hashing, and diagnostic text; and
- allocate no collection, map, [registry](../../../../glossary.md#term-registry) entry, or cached [mutable](../../../../glossary.md#term-mutable) state per validation call.

Failure messages must identify the invalid field or count and the accepted inclusive range. Exact
wording is part of the focused tests only where needed to distinguish input from output failures;
do not create a broad message-format protocol.

### [`OperationKind`](../../../../glossary.md#operationkind)

Keep `OperationKind` open to future family enums and retain `String name()`. Add:

```java
List<OperationSignature> signatures();

default OperationSignature signatureFor(OperationAttrs attrs) { ... }
```

The contract for `signatures()` requires a stable, [immutable](../../../../glossary.md#term-immutable), non-empty ordered list. Each [element](../../../../glossary.md#term-element)
must be non-null, and no two variants for one kind may declare the same exact `attributesType`.
The list is family-owned semantic [metadata](../../../../glossary.md#term-metadata); it is not a global [registry](../../../../glossary.md#term-registry) and contains no operand,
[backend](../../../../glossary.md#backend), [compiler](../../../../glossary.md#term-compiler), or execution facts.

`signatureFor` must:

1. reject null attributes;
2. inspect the family-provided variants in stable declaration order;
3. fail if the list is null, empty, contains null, or repeats an attributes class;
4. return the unique variant whose `attributesType` exactly equals `attrs.getClass()`; and
5. otherwise reject the pairing with an error that identifies the typed kind, actual attributes
   class, and accepted classes.

Production enums should create their immutable [signature](../../../../glossary.md#term-signature) values and lists once during class
initialization. Resolving an [operation](../../../../glossary.md#operation) must not allocate a new signature or signature list for
each [Tensor](../../../../glossary.md#tensor) expression. Intentional enum fields, constructors, and methods that implement this
contract replace the earlier temporary “no project state or behavior” restriction.

### [`Operation`](../../../../glossary.md#operation)

Preserve exactly these record components and their order:

```java
public record Operation(OperationKind kind, OperationAttrs attrs)
```

After the existing ordered null checks, the [canonical](../../../../glossary.md#term-canonical-form) constructor must call the kind's [signature](../../../../glossary.md#term-signature)
resolver. A wrong kind/attributes pair is rejected before the operation can enter [Tensor](../../../../glossary.md#tensor)
[provenance](../../../../glossary.md#provenance) or a [graph](../../../../glossary.md#graph). Do not copy, normalize, replace, or otherwise alter valid component
references.

Add one public derived accessor:

```java
OperationSignature signature();
```

It returns the stable family-owned signature resolved from the stored kind and attributes. It is
not a record component or stored field. Record-generated equality, hashing, and `toString()` remain
based only on `kind` and `attrs`.

### `CompiledNode`

Preserve exactly the existing four record components and all current collection [ownership](../../../../glossary.md#term-ownership-meanings) rules.
The [canonical](../../../../glossary.md#term-canonical-form) constructor [validation](../../../../glossary.md#term-validation) order remains:

1. null component references;
2. indexed null input [elements](../../../../glossary.md#term-element), then [immutable](../../../../glossary.md#term-immutable) input [snapshot](../../../../glossary.md#term-snapshot);
3. non-empty outputs;
4. indexed null and duplicate output elements, then immutable output snapshot; and
5. [operation](../../../../glossary.md#operation)-[signature](../../../../glossary.md#term-signature) validation of final `inputs.size()` and `outputs.size()`.

This order preserves useful structural failures before semantic count failures. Empty inputs and
multiple outputs are valid only when the supplied [operation signature](../../../../glossary.md#operation-signature) accepts those counts.
Repeated inputs remain valid positions. Outputs remain non-empty and unique within the [node](../../../../glossary.md#node).

Do not add a stored signature, index, [descriptor](../../../../glossary.md#term-descriptor), [producer](../../../../glossary.md#term-producer), consumer, or [graph](../../../../glossary.md#graph) validator.

## Current production [signature](../../../../glossary.md#term-signature) [matrix](../../../../glossary.md#term-matrix-meanings)

`No attrs` below means the exact class of `NoOperationAttrs.INSTANCE`. Every listed output count is
exactly one in the current production inventory; bounded and multi-output behavior is nevertheless
implemented and tested for future kinds.

| Kind family / [constants](../../../../glossary.md#term-constant) | Accepted attributes | Inputs | Outputs |
|---|---|---:|---:|
| `BinaryArithmeticKind.*` | [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) | exactly 2 | exactly 1 |
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
| `AggregateReductionKind.SUM`, [`MEAN`](../../../../glossary.md#term-mean) | `NoOperationAttrs` or `AxisReductionAttrs` | exactly 1 | exactly 1 |
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
  matrix; no attributes declaration or behavior changes are expected<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

Expected test files:

- new `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- existing `OperationKindTest`, `OperationTest`, `CompiledNodeTest`, and
  `CompiledGraphModelTest`
- existing [operation](../../../../glossary.md#operation)-family `*KindTest.java` and `*SemanticsTest.java` files whose exact enum
  shape or composition assertions must reflect the new intentional signature contract<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- existing `TensorTest`, `TensorFactoryTest`, and `TensorProvenanceTest` only to give their
  test-local kinds explicit signatures; production Tensor behavior must not change<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- one focused production-signature coverage test may be added if it replaces duplicated mapping
  assertions and verifies every current production enum constant against the matrix<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Expected no-change reviews:

- `docs/planning/modules/model/capabilities.md` already selects this exact hardening direction;
- focused architecture documents and `ARCHITECTURE.md` already assign these responsibilities to
  model and require no rule change.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- one new production Java file;
- the five core existing production files listed above;
- every current production operation-kind enum listed by the matrix, and no other production
  family type;<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- one new focused signature test plus only the existing tests identified above that require
  signature assertions or explicit test-local signatures; and
- the six documentation/[planning](../../../../glossary.md#term-planning) files listed above.

This is an intentional exception to the normal 12–18-file guardrail. The [invariant](../../../../glossary.md#term-invariant) is [atomic](../../../../glossary.md#resource-transaction-and-rollback):
enabling constructor [validation](../../../../glossary.md#term-validation) while leaving existing production families undeclared would make
valid current operations unconstructable, while allowing undeclared families through a fallback
would preserve the invalid-state gap that this task must close. Most family changes are a local
[signature](../../../../glossary.md#term-signature) declaration, corresponding [Javadoc](../../../../glossary.md#term-javadoc) correction, and automated test expectation.

Do not use the larger path allowance for unrelated refactors, formatting, semantic cleanup, or
[Tensor](../../../../glossary.md#tensor) behavior. If another production concept, attributes behavior change, [module](../../../../glossary.md#term-module), build file,
or architecture document is required, stop and report the needed follow-up.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document `OperationSignature` as a structural occurrence contract, define every component,
  exact attributes-class matching, inclusive bounds, effective variadic maximum, failures, value
  semantics, and the distinction from Tensor rank and operand compatibility.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [variadic glossary](../../../../glossary.md#term-variadic); [Tensor glossary](../../../../glossary.md#tensor); [rank glossary](../../../../glossary.md#term-rank))</span>
- Update [`OperationKind`](../../../../glossary.md#operationkind) to explain family-owned signatures, exact attribute variants, stable
  immutable lists, and the absence of a global registry or executable metadata.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [immutable glossary](../../../../glossary.md#term-immutable); [registry glossary](../../../../glossary.md#term-registry); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Update [`Operation`](../../../../glossary.md#operation) to state that kind/attributes compatibility is now enforced and that
  `signature()` is derived rather than stored.
- Update `CompiledNode` to explain signature-based local count validation while preserving the
  boundary with graph-wide and operand-aware validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [graph glossary](../../../../glossary.md#graph))</span>
- Correct every affected family Javadoc that currently says arity is only prose, the enum has no
  project metadata, or generic <span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity))</span>`Operation` does not enforce pairings.
- Review [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and related attributes Javadocs for accurate exact
  matching terminology; change only stale contract wording.
- Explain all new terms for a newcomer in Tensor API and glossary, including “operation
  signature”, “occurrence cardinality”, “fixed”, “bounded”, and “variadic”.<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- Update [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) only for the now-current `CompiledNode` local count validation. Do not claim
  compiler capture, descriptor inference, or runtime validation is implemented.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Acceptance criteria

- `OperationSignature` has exactly the five specified record components in order and no mutable
  state.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- Signature construction enforces valid inclusive input/output ranges and permits fixed, bounded,
  zero-input, effectively variadic-input, and multi-output contracts.<span data-glossary-navigation="true"> ([Signature glossary](../../../../glossary.md#term-signature); [variadic glossary](../../../../glossary.md#term-variadic))</span>
- Attribute matching is exact-class based and uses no string comparison, classpath scan,
  annotation scan, service lookup, or global kind map.<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [`OperationKind`](../../../../glossary.md#operationkind) retains `name()`, requires stable family-owned signature variants, and resolves
  one unique variant for an attributes object.
- A kind with missing, empty, null, duplicate, or nonmatching signature declarations fails closed;
  no permissive fallback signature exists.
- Every production kind [constant](../../../../glossary.md#term-constant) and accepted attributes variant matches the complete [matrix](../../../../glossary.md#term-matrix-meanings).
- Every intentionally wrong current cross-family pairing is rejected during [`Operation`](../../../../glossary.md#operation)
  construction, including representative parameterless, cast, scalar, reduction, layout, index,
  and window cases.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [layout glossary](../../../../glossary.md#layout))</span>
- `Operation` remains a record with exactly `kind` and `attrs`, retains exact valid references,
  and keeps generated equality, hashing, and diagnostic text based only on those components.
- `Operation.signature()` returns the matching stable family-owned signature and adds no record
  component or instance field.
- `CompiledNode` preserves its record shape, snapshots, input repetition, output non-empty and
  uniqueness rules, then rejects input or output counts outside the signature.
- Focused tests prove exact unary, binary, ternary, bounded, variadic, zero-input, one-output, and
  multi-output behavior without introducing a production sample kind.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Existing production Tensor expressions continue to construct their exact operation semantics
  and all model tests pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>
- No `Map` or registry of all operation kinds, no root switch over concrete families, and no
  reflective discovery appears in production.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- No backend, compiler, planning, prepare, runtime, execution, gradient, cost, fusion, kernel,
  storage, or device metadata is added.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [gradient glossary](../../../../glossary.md#term-gradient); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [device glossary](../../../../glossary.md#term-device); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- All affected public contracts and enum constants have complete current [Javadoc](../../../../glossary.md#term-javadoc).
- The targeted clean-context documentation pass finalizes Javadoc, Tensor API, Compile API,
  glossary, planning evidence, links, and terminology without rerunning successful Java tests
  unless executable behavior changed after the recorded run.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run))</span>
- Task, master-plan row, and roadmap row have matching final status.
- Task 0018L remains Draft without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, use focused tests as needed. After executable Java code stabilizes, record
one final [module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test
```

The targeted documentation pass then runs, after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also checks local Markdown links and anchors, balanced fences, terminology,
and final newlines in the affected documentation. It reuses the recorded [model](../../../../glossary.md#term-model)-test result unless
it changes executable Java behavior.

Repository-wide `./gradlew test` is deferred to the recorded foundation-contract capability
[checkpoint](../../../../glossary.md#term-checkpoint) after task 0018N. This task changes one module and no [dependency](../../../../glossary.md#term-dependency) or architecture
boundary.

Automated tests must cover the [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) and [signature](../../../../glossary.md#term-signature) [matrix](../../../../glossary.md#term-matrix-meanings) invariants. Do not require repeated
manual `javap`, reflection, or [bytecode](../../../../glossary.md#term-bytecode) commands when the same facts are asserted in tests.

Final [scope](../../../../glossary.md#term-scope) review must confirm:

- only `modules/model` Java files and the allowed documentation/[planning](../../../../glossary.md#term-planning) files changed;
- [`Operation`](../../../../glossary.md#operation) and `CompiledNode` record components did not change;
- every production [`OperationKind`](../../../../glossary.md#operationkind) [constant](../../../../glossary.md#term-constant) appears in signature coverage;
- no concrete family import appears in the root generic resolver;
- no Java/[Gradle](../../../../glossary.md#term-gradle)/module dependency or architecture file changed; and
- 0018K is `Complete` everywhere only after both passes, while 0018L has no task file.

## Dependencies

- Task 0005: [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) — Complete.
- Task 0006: [immutable](../../../../glossary.md#term-immutable) two-component [`Operation`](../../../../glossary.md#operation) — Complete.
- Task 0008: immutable `CompiledNode` [occurrence](../../../../glossary.md#term-occurrence) with ordered inputs and outputs — Complete.
- Current production operation-family tasks through 0018J — Complete and inventoried by the
  signature matrix.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

## Follow-up tasks

- 0018L — use the output-cardinality foundation for shared multi-output Tensor provenance without
  adding graph-local identity to Tensor.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- 0018O — normalize indexing taxonomy and replace independent UNSTACK semantics with repeated
  scalar select convenience.<span data-glossary-navigation="true"> ([scalar select glossary](../../../../glossary.md#scalar-select))</span>
- 0018P–0018V — perform the selected semantic cleanup and capability additions on top of validated
  signatures.
- 0019B and 0019C — use explicit multi-output [signatures](../../../../glossary.md#term-signature) for [graph](../../../../glossary.md#graph) [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/[dropout](../../../../glossary.md#dropout--inverted-dropout) and top-K.

Do not create detailed specifications for these follow-ups in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics and [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) state
to `modules/model`. This task makes those existing local contracts stricter without changing
[module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), or [runtime](../../../../glossary.md#term-runtime) visibility.

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
  the five-component canonical record remains the complete value contract.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Production and test-local enum kinds retain stable immutable signature lists in static final
  fields. Resolution therefore reuses family-owned values rather than allocating per operation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [signature glossary](../../../../glossary.md#term-signature))</span>
- `Integer.MAX_VALUE` is the literal inclusive upper bound for an effectively variadic Java list;
  no sentinel translation or separate variadic flag is used.<span data-glossary-navigation="true"> ([variadic glossary](../../../../glossary.md#term-variadic))</span>
- [`Operation`](../../../../glossary.md#operation) resolves once during construction to reject an invalid pair and resolves again only
  through the derived `signature()` accessor. The signature remains absent from record state, so
  equality, hashing, and diagnostic text keep their two-component contract.
- `CompiledNode` performs signature-cardinality validation only after its existing input/output
  list checks and immutable snapshots, preserving the established failure order.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality); [validation glossary](../../../../glossary.md#term-validation))</span>

## Known limitations

- Signatures describe exact attributes classes and local occurrence counts only. They do not
  validate operand data types, Shapes, layouts, axes, descriptors, numerical policy, graph-wide
  relationships, compiler behavior, backend support, or executability.<span data-glossary-navigation="true"> ([Signatures glossary](../../../../glossary.md#term-signature); [occurrence glossary](../../../../glossary.md#term-occurrence); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [axes glossary](../../../../glossary.md#term-axis); [descriptors glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- No current production kind is multi-output. Zero-input, bounded, effectively variadic, and
  multi-output contracts are covered with generic or test-local signatures; task 0018L still owns
  shared multi-output Tensor provenance.<span data-glossary-navigation="true"> ([variadic glossary](../../../../glossary.md#term-variadic); [provenance glossary](../../../../glossary.md#provenance))</span>
- Public [Tensor](../../../../glossary.md#tensor) expression construction and `TensorProvenance` remain unchanged. A valid local
  signature does not establish shared producer identity or graph capture.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root` ran `./gradlew :modules:model:test` after executable Java and
  test changes stabilized: `BUILD SUCCESSFUL in 1s`. The generated report contains 743 tests in
  86 suites, with 0 failures, 0 errors, and 0 skipped tests. The documentation-focused pass reused
  this evidence and changed only Javadoc and documentation afterward.
- Clean documentation-focused context `/root/task_0018k_docs` independently reviewed
  `AGENTS.md`, `ARCHITECTURE.md`, the current architecture index, documentation rules and General,
  API/Javadoc, Planning, and Example profiles, the planning guide and roadmap, the capability
  baseline, model master plan, this task, affected source/tests and their actual diff, Tensor API,
  Compile API, and glossary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile))</span>
- `/root/task_0018k_docs` ran `./gradlew :modules:model:javadoc` after final [Javadoc](../../../../glossary.md#term-javadoc) edits:
  `BUILD SUCCESSFUL in 1s`; 2 actionable tasks executed and the configuration [cache](../../../../glossary.md#term-cache) was reused.
- `/root/task_0018k_docs` ran the targeted local Markdown checker over the six affected
  documentation/planning files. It passed 413 local links, including 110 anchor links, and
  confirmed balanced fences and final newlines in all six files.
- `/root/task_0018k_docs` ran `git diff --check` on the final combined change: passed with no
  output.
- Final [scope](../../../../glossary.md#term-scope) inspection confirmed that [`Operation`](../../../../glossary.md#operation) remains a two-component record,
  `CompiledNode` remains a four-component record, every production kind is covered by the
  automated signature matrix, the generic operation resolver imports no concrete family, and no
  Gradle file, dependency, module boundary, <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>`ARCHITECTURE.md`, focused architecture document,
  architecture test, backend-conformance test, integration test, or other module changed for this
  task. No task-0018L specification exists.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- `docs/planning/modules/model/capabilities.md` remains unchanged by task 0018K because its
  “Valid operations and occurrence signatures” section already selected exact family-owned
  attribute pairing and fixed/bounded/variadic cardinality. <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [variadic glossary](../../../../glossary.md#term-variadic); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>`ARCHITECTURE.md` and focused
  architecture documentation remain unchanged because the implementation only tightens existing
  model-owned semantic and immutable graph contracts without changing ownership, dependencies, or
  lifecycle rules.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Repository-wide tests remain deferred to the foundation-contract checkpoint after task 0018N,
  as recorded before implementation. No Java test was repeated by the documentation-focused pass
  because it made no executable change after the successful module run.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [run glossary](../../../../glossary.md#term-run))</span>

## Implementation notes

- Added the five-component [immutable](../../../../glossary.md#term-immutable) `OperationSignature` record with validated inclusive bounds,
  exact runtime attributes-class matching, fixed and input-range declaration helpers, count
  queries, and occurrence validation.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [occurrence glossary](../../../../glossary.md#term-occurrence); [validation glossary](../../../../glossary.md#term-validation))</span>
- Extended [`OperationKind`](../../../../glossary.md#operationkind) with stable family-owned signature variants and an exact resolver that
  fails closed for null, empty, malformed, duplicate, or incompatible declarations without a
  global registry or concrete-family switch.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- Hardened [`Operation`](../../../../glossary.md#operation) construction against incompatible kind/attributes pairs and added the
  derived `signature()` accessor without changing record components or retaining [signature](../../../../glossary.md#term-signature) state.
- Declared the complete task matrix in every current production kind family. Current production
  variants remain one-output; concat and stack accept one through <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`Integer.MAX_VALUE` inputs.
- Hardened `CompiledNode` to validate final local input/output counts after its established list
  checks and snapshots while preserving repeated inputs, explicitly declared zero-input kinds,
  and explicitly declared multi-output kinds.
- Added focused signature, resolver, operation, node-cardinality, and complete production-matrix
  coverage; updated existing test-local kinds to declare stable signatures.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- Finalized `OperationSignature`, `OperationKind`, `Operation`, `CompiledNode`, family-kind, and
  directly related attribute Javadocs. Corrected stale pairing and temporal claims in select,
  target-shape, permutation, and slice documentation without changing behavior.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [shape glossary](../../../../glossary.md#shape); [permutation glossary](../../../../glossary.md#permutation); [slice glossary](../../../../glossary.md#slice))</span>
- Updated Tensor API, Compile API, glossary, task evidence, model master plan, and roadmap. The
  reference material now defines operation signature, occurrence cardinality, fixed, bounded,
  and variadic, and separates local structural validation from operand-aware, graph-wide,
  compiler, backend, and runtime responsibilities.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [operation signature glossary](../../../../glossary.md#operation-signature); [variadic glossary](../../../../glossary.md#term-variadic); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>

## Completion summary

- Completed changes: added exact family-owned operation signatures, construction-time
  kind/attributes validation, derived signature access, complete production declarations, and
  local compiled-node occurrence-count validation.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [node glossary](../../../../glossary.md#node); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Files changed or created: new `OperationSignature.java`; updated core [operation](../../../../glossary.md#operation) and
  `CompiledNode` contracts; updated every production operation-kind enum and directly affected
  attribute Javadocs; added `OperationSignatureTest`; updated focused operation, graph, and
  test-local-kind tests; finalized Tensor API, Compile API, glossary, this task, model master plan,
  and roadmap.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: reused the final 743-test/86-suite model run with zero failures, errors, or
  skips; model Javadoc generation, 413-link/110-anchor Markdown checks, fence/final-newline checks,
  final scope checks, and <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: clean context `/root/task_0018k_docs` completed the required
  independent API/Javadoc and planning-profile review.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor and Compile API references now explain the structural contract and
  its boundaries; planning evidence and synchronized status are final. Architecture and focused
  architecture documents require no change because ownership and lifecycle boundaries did not
  change.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc review: all affected public and contract-relevant Javadocs were reviewed; stale exact
  pairing and already-implemented-convenience wording was corrected. No executable Java changed
  during the documentation pass.
- Glossary impact: added and aligned operation signature, occurrence cardinality, fixed, bounded,
  and variadic terminology and their distinction from Tensor rank and executable support.<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature); [cardinality glossary](../../../../glossary.md#term-cardinality); [variadic glossary](../../../../glossary.md#term-variadic); [rank glossary](../../../../glossary.md#term-rank))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0018K. Task 0018L remains the next Draft frontier and owns
  shared multi-output Tensor provenance.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>

Status: Complete
