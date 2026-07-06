# Task 0017L: Tensor Composition Expressions

## Status

Complete

## Goal

Add public concat, stack, and multi-result unstack expression construction to Tensor.

Concat and stack consume ordered non-empty Tensor inputs and return one fresh storage-free result.
Unstack returns an immutable ordered List of individually indexed fresh UNSTACK outputs. All
methods must validate locally provable Shape/type rules, preserve exact semantic input order,
retain gradient eligibility conservatively, and leave result layout unresolved. No values,
storage, gradients, graph grouping, compiler behavior, materialization, or execution is added.

## Scope

- Add exactly these public Tensor methods:
  - static `concat(int axis, Tensor... inputs)`
  - static `stack(int axis, Tensor... inputs)`
  - instance `List<Tensor> unstack(int axis)`
- Add one field-free package-private final TensorCompositionExpressions helper with exactly ten
  methods specified below.
- Snapshot varargs arrays defensively, require at least one input, reject indexed nulls, and
  preserve exact ordered Tensor references.
- Require exact matching DataType for concat/stack; do not promote or insert casts.
- Concat requires equal rank and structurally equal dimensions outside the normalized existing
  axis. Derive the concat extent with checked static sums or one locally unchanged dynamic extent
  plus only static-zero companions.
- Stack requires structurally identical Shapes, normalizes an insertion axis, inserts one static
  input-count dimension, and preserves exact first-input Dimension references.
- Result requiresGrad for concat/stack is true when any input descriptor requests it.
- Unstack requires a statically known selected extent no larger than Integer.MAX_VALUE, removes
  that axis, and returns one output per coordinate in increasing order.
- Give every unstack result exact UNSTACK/UnstackOutputAttrs(axis,index)/[input] provenance; add no
  producer grouping or graph output-slot state.
- Return immutable List output from unstack; zero selected extent returns an empty immutable list
  and consumes no Tensor IDs.
- Keep every result layout unresolved, labels/storage absent, and successful results fresh.
- Update TensorTest only for exact public API inventory and add one focused expression suite.
- Finalize Javadocs, Tensor/Compile API, glossary, task evidence, master plan, and roadmap through
  the mandatory independent documentation pass.

## Out of scope

- array-returning unstack overload, List-input concat/stack overload, two-input convenience,
  default axis, output-count parameter, caller-provided collection, stream, iterator, or builder
- empty concat/stack inputs, null elements, mixed DataTypes, implicit promotion/casts, broadcasting,
  concat rank promotion, or stack of merely broadcast-compatible Shapes
- affine/symbolic concat extent constraints beyond the exact locally provable dynamic-zero case
- public multi-output producer identity, TensorProvenance output slot/group, batch TensorFactory,
  CompiledNode/GraphValue creation, or grouping unstack results into one graph node
- decomposing stack into expandDims+concat or unstack into select; compiler may choose later
- resolved layout/view aliases, allocation, copying, value movement, storage access, residency,
  materialization, kernel/lowering, or execution
- gradients, concat/stack split, unstack scatter, autograd, compiler-generated operations,
  optimizer, training, compiler/planning/prepare/runtime/backend/engine/trace/ONNX behavior
- changing TensorCompositionKind, CompositionAxisAttrs, UnstackOutputAttrs, TensorProvenance,
  graph contracts, Shape/Dimension, TensorDescriptor, TensorFactory, Operation, completed tests, or
  another existing Java contract
- another helper/type/test, dependency, Gradle/build option, architecture change, another module,
  or task-0017M specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

Legacy provides static varargs concat/stack and array-returning instance unstack. Concat is a
first-class n-ary operation, stack decomposes to expandDims plus concat, and unstack decomposes to
one select per coordinate. It requires matching data types/shapes and static int dimensions while
also coupling model construction to gradients, storage, compiler, kernels, and runtime.

The new API retains static varargs concat/stack. It modernizes multi-result ownership by returning
an immutable List rather than a mutable Tensor array; capability parity does not require every
legacy overload. STACK and indexed UNSTACK remain explicit first-class model semantics from task
0017K, leaving decomposition to compiler policy.

Current TensorProvenance cannot identify one shared multi-output producer. Each unstack result
therefore records its own outputIndex, remains independently capturable, and does not claim
grouping. CompiledNode multi-output capability remains available for future explicit compiler
design but is not used or changed here.

## Architecture constraints

- Tensor is public API state, not IR. Methods create expression metadata through centralized
  TensorFactory.createDerived only.
- Ordered concat/stack inputs are represented in TensorProvenance, not semantic attributes.
- Operation semantics and normalized axis/output index come only from task 0017K.
- Local validation may inspect immutable descriptors and Shapes, but never values, storage,
  residency, backend capability, or old provenance.
- Result layout is unresolved because composition requires output materialization and cannot be
  represented as one input view geometry.
- Dynamic concat support is limited to an extent that is provably unchanged: exactly one dynamic
  selected dimension and only static-zero selected dimensions on every other input.
- Stack preserves identical dynamic symbols structurally. Unstack requires a static selected
  extent because public result count must be known immediately.
- Individual UNSTACK outputs are not grouped. No provenance/graph architecture is changed.
- Compiler owns capture/decomposition/canonicalization/backward graph; planning/backend prepare own
  materialization/lowering; runtime executes prepared work.
- Stop if implementation needs grouping, graph changes, value/storage access, dependencies, or an
  architecture decision.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` receives exactly three public methods.
- `io.github.pho001.synaptik.model.tensor.TensorCompositionExpressions` owns local ordered-input,
  axis, type, Shape, collection, descriptor, and provenance construction.
- `TensorCompositionExpressionTest` mirrors the package for helper/API and package-private checks.
- `TensorTest` changes only exact public API inventory/reflection assertions.

The three semantic contracts remain unchanged in model.operation.layout.

## Required contract

### Public Tensor methods

Add exactly:

```java
public static Tensor concat(int axis, Tensor... inputs) {
    return TensorCompositionExpressions.concat(axis, inputs);
}

public static Tensor stack(int axis, Tensor... inputs) {
    return TensorCompositionExpressions.stack(axis, inputs);
}

public List<Tensor> unstack(int axis) {
    return TensorCompositionExpressions.unstack(this, axis);
}
```

Each is one return and one matching helper call. concat/stack are public static non-synchronized;
unstack is public instance non-synchronized. Add no other public method or overload.

### Helper shape

Create one package-private final field-free class with private zero-argument constructor and
exactly these ten static methods:

```java
static Tensor concat(int axis, Tensor[] inputs)
static Tensor stack(int axis, Tensor[] inputs)
static List<Tensor> unstack(Tensor input, int axis)
private static List<Tensor> snapshotInputs(String operation, Tensor[] inputs)
private static int normalizeExistingAxis(String operation, int axis, int rank)
private static int normalizeInsertionAxis(int axis, int rank)
private static Shape concatShape(List<Tensor> inputs, Shape firstShape, int normalizedAxis)
private static Shape stackShape(Shape inputShape, int normalizedAxis, int inputCount)
private static Shape unstackShape(Shape inputShape, int normalizedAxis)
private static Tensor create(
        DataType dataType,
        Shape resultShape,
        boolean requiresGrad,
        Operation operation,
        List<Tensor> inputs)
```

Add no fields, nested type, alternate constructor, overload, cache, or extra method.

### Ordered input snapshot

snapshotInputs validates in exact order:

1. null-check inputs with exact message `inputs`;
2. reject empty array with IllegalArgumentException message
   `<operation> requires at least one input`;
3. clone the array exactly once;
4. inspect copied elements in order and null-check with exact message `inputs[<index>]`;
5. return one immutable List.copyOf(Arrays.asList(copy)) preserving exact references/order.

concat and stack each call snapshotInputs exactly once with operation text `concat` or `stack`.
Caller mutation after cloning cannot alter construction or provenance.

### Axis normalization

normalizeExistingAxis uses long arithmetic, adds rank once for negative raw axes, and returns
normalized [0,rank). On failure throw IndexOutOfBoundsException:

```text
<operation> axis <raw> is outside shape rank <rank>
```

It serves concat and unstack with operation names `concat` and `unstack`.

normalizeInsertionAxis adds rank+1 once for negative axes and accepts [0,rank]. On failure throw:

```text
stack axis <raw> is outside insertion range for shape rank <rank>
```

### Concat construction

concat snapshots inputs, reads first descriptor/Shape/DataType, normalizes one existing axis, then
validates every input in encounter order:

- exact DataType match, otherwise IllegalArgumentException
  `concat inputs must have matching data types: inputs[<i>] is <actual>, expected <expected>`;
- exact rank match, otherwise
  `concat inputs must have matching ranks: inputs[<i>] has <actual>, expected <expected>`;
- every non-concat Dimension structurally equals first input Dimension, otherwise
  `concat inputs differ at non-concat axis <axis>: inputs[<i>]`.

Compute requiresGrad as OR across descriptors. Call concatShape once, construct exact
Operation(CONCAT, new CompositionAxisAttrs(normalizedAxis)), and create once.

concatShape preserves exact first-input non-axis Dimension references. For selected dimensions:

- if all are static, checked-add every size starting from zero and create one StaticDimension;
- if exactly one is dynamic and every other selected dimension is static zero, retain that exact
  DynamicDimension reference;
- otherwise throw `cannot represent concat axis <axis> with dynamic extents`.

Single input and static-zero inputs are valid. Result rank is unchanged.

### Stack construction

stack snapshots inputs, reads first descriptor/Shape/DataType, normalizes insertion axis, then
validates every input in order:

- exact DataType match with analogous exact message using `stack`;
- Shape structural equality, otherwise
  `stack inputs must have identical shapes: inputs[<i>] differs from inputs[0]`.

Compute requiresGrad OR. stackShape inserts new StaticDimension(inputCount) at normalized result
axis and preserves exact first Shape Dimension references around it. Construct exact STACK plus
CompositionAxisAttrs and create once. One input is valid and remains explicit.

### Unstack construction

unstack null-checks input with message `input`, reads its descriptor/Shape, normalizes existing
axis, and requires selected Dimension be StaticDimension. Dynamic failure message:

```text
unstack axis <axis> must have a statically known dimension
```

Require size <= Integer.MAX_VALUE, otherwise:

```text
unstack axis <axis> size <size> exceeds maximum result count 2147483647
```

Call unstackShape once to remove selected Dimension and preserve all other exact references.
For size zero, return List.of() without Operation/Tensor/ID allocation. Otherwise create one
ArrayList sized to count and, for index 0 upward, construct exact UNSTACK plus
UnstackOutputAttrs(axis,index) and call create with List.of(input). Return List.copyOf(outputs).

All outputs retain input DataType/requiresGrad, share the immutable result Shape reference, have
unresolved layout, absent labels/storage, increasing outputIndex, and fresh IDs. If ID exhaustion
occurs mid-loop, already consumed IDs are not rolled back and no partial List is returned; current
factory has no batch reservation/rollback contract.

### Common result construction

create constructs one TensorDescriptor(dataType,resultShape,Optional.empty(),requiresGrad), one
TensorProvenance(operation,inputs), and calls TensorFactory.createDerived exactly once with absent
label. It never inspects input labels, layouts, old provenance, values, or storage.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCompositionExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCompositionExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most these ten paths may change. If another helper/type/test, overload, semantic/provenance/
graph edit, dependency, build/architecture change, or eleventh path is required, stop and report.
Do not create task 0017M.

## Javadoc requirements

- Document all three public methods and helper type/constructor/all ten methods.
- Explain ordered varargs ownership, validation order, exact type/Shape requirements, axis forms,
  dynamic concat-zero rule, checked arithmetic, stack insertion, and immutable List output.
- Explain individual UNSTACK outputIndex semantics, zero-size empty result, static count limit, no
  producer grouping, and possible partial ID consumption on exhaustion.
- Document result type/Shape/layout/eligibility/label/storage/operation/attrs/provenance/freshness.
- Explain unresolved layout/materialization boundary and no value/gradient/cross-layer behavior.
- Review semantic, provenance, graph, Shape/Dimension, descriptor/factory, and neighboring
  expression Javadocs; record unchanged reasons or stop on discrepancy.

## Acceptance criteria

- Tensor adds exactly three signatures and public method count rises 89 to 92.
- Public delegates and exact field-free ten-method helper match specification.
- Input snapshots, null/empty/order, axes, types, ranks/Shapes, messages, and ownership match.
- Concat static/dynamic-zero Shape rules and overflow behavior match.
- Stack inserts exact count and preserves exact first-input dimensions.
- Unstack returns immutable ordered outputs with exact indexed semantics; zero size returns empty.
- All data types and valid eligibility combinations retain expected metadata.
- Every result layout unresolved; labels/storage absent; IDs fresh except empty unstack.
- No grouping, values/storage, gradients, compiler/backend behavior, dependencies, or architecture
  changes.
- Independent docs review and all validation/status synchronization complete before Complete.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorCompositionExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact APIs/helper; ownership/order; null/empty; axes; all data types;
eligibility OR; concat static/zero/dynamic/extreme/overflow cases; stack scalar/static/dynamic and
shape/type failures; unstack scalar/dynamic/zero/count/output order/index/immutability; exact
Dimension references; unresolved layouts for all input layout kinds; provenance; storage
non-interference; freshness/nesting; ID effects/exhaustion; and no grouping state.

Inspect javap/reflection/imports/bytecode/source, generated Javadoc, executable examples,
Tensor/Compile API/glossary, links/anchors/fences/whitespace, exact ten paths, synchronized status,
and no task-0017M specification.

## Dependencies

- 0001/0002 supply DataType and Shape/Dimension.
- 0003/0007 supply unresolved TensorDescriptor construction.
- 0011–0013 supply Tensor, derived identity, and provenance.
- 0017K supplies composition kinds and attributes.

## Follow-up tasks

- 0017M remains Draft for unfold/fold semantics.
- Grouped multi-output provenance/capture requires a separate explicit future task if desired.
- Compiler owns decomposition/capture/canonicalization/backward graphs.
- Planning/backend prepare own materialization/lowering/kernels; ONNX extension owns mapping.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. Current one-provenance-per-Tensor and graph contracts remain unchanged.
Stop if implementation needs grouped producer identity, graph output slots, batch factory,
dependencies, or architecture changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0007/0011/0012/0013/0017K/0017L, Tensor API,
Compile API, Training API, glossary, current DataType/Dimension/Shape/TensorDescriptor/Tensor/
TensorFactory/TensorProvenance/Operation/TensorCompositionKind/CompositionAxisAttrs/
UnstackOutputAttrs contracts and focused expression tests, and Java 26 Gradle.

Implement task 0017L exactly. Modify Tensor.java and add package-private final
TensorCompositionExpressions.java. Update TensorTest only for exact three-method API expansion and
add TensorCompositionExpressionTest. Add exactly static concat(int,Tensor...), static
stack(int,Tensor...), and instance List<Tensor> unstack(int).

The field-free helper has exactly ten specified methods. Snapshot ordered non-empty varargs,
normalize axes, require exact data types and concat/stack Shape rules, derive checked concat and
inserted stack Shapes, OR gradient eligibility, and create exact unresolved CONCAT/STACK
provenance. Unstack requires a static int-sized axis extent, removes the axis, returns immutable
ordered individually indexed UNSTACK outputs, and uses no producer grouping. Zero extent returns
empty without IDs. Every created Tensor is fresh with no label/storage.

Do not modify semantic/provenance/graph/foundational contracts, add overloads/types/helpers,
promote/broadcast inputs, group multi-output producers, inspect/copy values/storage, derive resolved
layout, define gradients, or add compiler/planning/prepare/runtime/backend/ONNX behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on architecture
uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017L, model master plan, and roadmap only for status/evidence. Do not mark Complete until
both passes succeed. Leave 0017M Draft without a specification. Do not commit/push.
```

## Local decisions

- Public unstack returns immutable List<Tensor>, modernizing mutable legacy Tensor[] ownership.
- STACK/UNSTACK remain first-class semantics; compiler may decompose later.
- Each unstack output is independent and indexed; no producer grouping is introduced.
- Dynamic concat is accepted only when exactly one dynamic selected extent is combined solely with
  static zeros, making the result provably unchanged.
- All results use unresolved layout because composition requires materialization.
- Concat/stack eligibility is OR; unstack preserves input eligibility.
- Empty unstack returns no outputs/IDs; mid-loop exhaustion may consume IDs without rollback.

## Known limitations

- No grouped multi-output capture, affine dynamic concat, promotion/broadcasting, resolved layout,
  values, storage, gradients, compiler behavior, materialization, lowering, execution, or ONNX.

## Validation evidence

- Architecture/focused boundary docs, documentation/planning rules, roadmap, model capabilities and
  master plan, DataType/Shape/descriptor/Tensor/factory/provenance/graph contracts/tests, task
  0017K semantics, neighboring expressions, APIs/glossary, Java 26 build, and read-only legacy
  concat/stack/unstack builders/tests were reviewed.
- Planning explicitly resolves current provenance limits through individually indexed UNSTACK
  outputs and does not introduce grouped producer identity or graph output slots.
- `git diff --check` passed with no whitespace errors.
- This planning step changed exactly three logical paths: this new task plus existing master plan
  and roadmap. The shared working tree already contained the completed, independently validated
  nine-path uncommitted task-0017K change; those Java/API/glossary/task paths were preserved rather
  than treated as part of 0017L planning.
- Markdown structure check passed: 23 level-two sections, 16 balanced fences, no trailing
  whitespace, and final newline present.
- Every local Markdown link in task/master/roadmap resolves.
- Roadmap contains all 74 ordered task rows. 0017L is linked/Ready at row 65; 0017M remains Draft at
  row 66; no task-0017M specification exists.
- Task/master/roadmap consistently identify 0017L as next Ready frontier.
- Package/scope review confirms three Tensor public methods, one package-private helper, and one
  mirrored focused test fit the ten authorized implementation paths without provenance, graph,
  dependency, or architecture changes.
- No Gradle test was run for the planning delta because it changes no production/test code; the
  underlying completed 0017K diff already passed its recorded implementation/docs validation.
- Implementation context `/root/implement_model_0017l` completed the four Java source/test paths
  and handed the shared uncommitted diff to independent documentation context
  `/root/implement_model_0017l/docs_0017l`. The documentation pass used General plus API/Javadoc
  style for Java, Tensor API, Compile API, and glossary; General plus Planning style for this task,
  the model master plan, and roadmap; and Example format for the executable Tensor API example.
- The documentation context independently reviewed the final Tensor/helper source, both focused
  tests, generated Javadoc, DataType/Dimension/Shape/layout/descriptor/factory/provenance/
  operation/composition/graph contracts, neighboring expressions, all selected architecture and
  planning material, current APIs/glossary, and Java 26 build configuration. It finalized all
  three public method Javadocs, helper type/private constructor/all ten method Javadocs, Tensor and
  Compile API current/planned boundaries, glossary terminology, and planning evidence.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorCompositionExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed after documentation edits: 27 tests
  total, comprising 13 composition-expression tests and 14 Tensor contract tests, with no failure,
  error, or skip. Coverage includes every DataType and valid eligibility state, attached storage
  across resolved layout kinds, exact APIs/helper shape, validation order/messages, Shape rules,
  provenance, freshness, zero-result ID behavior, and exhaustion.
- `./gradlew :modules:model:test` passed after documentation edits. The 72 model suites report 601
  tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` passed without warnings. Generated `Tensor.html` was inspected
  for the exact three signatures, axis ranges, input ownership/validation, result metadata,
  immutable unstack ownership, individual output indexing, zero-size behavior, exhaustion, and
  no-grouping/cross-layer boundaries. The package-private helper's source Javadocs cover its type,
  private constructor, and all ten methods.
- `./gradlew test` passed after documentation edits with 36 actionable tasks: the first run had 1
  executed and 35 up-to-date, and the final synchronized-tree rerun had all 36 up-to-date.
  Architecture, backend-conformance, and integration sources required no task-local additions
  because no dependency rule, backend behavior, or executable end-to-end behavior changed.
- The Tensor API example was copied unchanged to `/tmp/TensorCompositionExpressionExample.java`,
  compiled with Java 26 against model classes, and executed. Its exact documented Shapes, kinds,
  attributes, provenance relationships, eligibility, and unresolved storage-free result checks
  matched the printed output.
- `javap -p -s` confirmed Tensor's exact three new public signatures and the final field-free
  helper's private zero-argument constructor plus exact ten static methods. Focused reflection
  tests confirm 92 public Tensor methods, static concat/stack, instance unstack, no helper fields or
  nested types, and exact visibility. Bytecode/source/import review confirmed one array clone and
  immutable snapshot, checked concat addition, zero-size early return, individually indexed output
  creation, one `createDerived` call per result, local model/JDK imports only, and no storage,
  graph, grouping, gradient, compiler, backend, or execution behavior.
- The targeted local Markdown validator resolved 360 links, including 106 heading anchors, across
  the six changed documentation/planning files. All fences are balanced; every authorized path has
  a final newline; trailing-whitespace search and `git diff --check` pass.
- Final inventory contains exactly the ten authorized 0017L paths: Tensor, the composition helper,
  TensorTest, the focused composition test, Tensor API, Compile API, glossary, this task, model
  master plan, and roadmap. Task 0017K remains the committed `da737077` baseline; no eleventh path,
  dependency/build/architecture change, or task-0017M specification exists.
- Training API remains accurate unchanged because no gradient object, optimizer, training session,
  or backward behavior was added. The capability baseline remains accurate unchanged because it
  already lists concat/stack/unstack at model and public-API layers without claiming compiler or
  executable parity. Semantic/provenance/graph/foundational contracts remain accurate unchanged
  because 0017L composes them without altering their representation or introducing grouped output
  identity. Architecture docs/tests, backend conformance, integration tests, Gradle configuration,
  dependencies, other modules, and neighboring expressions remain accurate unchanged because the
  task adds model-owned expression metadata only.
- Final status is synchronized as 0017L Complete in this task, the master-plan row/current status/
  decisions/notes, and roadmap frontier/table/status. Task 0017M remains Draft without a detailed
  specification.

## Implementation notes

- Added exactly static `Tensor.concat(int, Tensor...)`, static `Tensor.stack(int, Tensor...)`, and
  instance `Tensor.unstack(int)` delegating to one field-free package-private helper.
- Implemented ordered defensive varargs snapshots, existing/insertion axis normalization, exact
  type and Shape validation, checked static concat extents, the sole dynamic-zero concat case,
  stack dimension insertion, eligibility OR, and unresolved derived descriptors.
- Implemented immutable unstack Lists with static `int`-sized result counts, shared result Shape,
  individually indexed provenance, zero-size no-ID behavior, and documented mid-loop exhaustion
  effects without producer grouping.
- Finalized contract-accurate Javadocs, an executable Tensor API composition section, the Compile
  API compiler boundary, glossary distinctions, and synchronized planning status/evidence.

## Completion summary

- Completed changes: implemented and documented exact concat, stack, and immutable-list unstack
  model expression construction with local validation, Shape derivation, result metadata, and
  individually indexed provenance.
- Files changed or created: the exact ten authorized paths listed in Validation evidence; no
  architecture, build, dependency, other-module, semantic, provenance, graph, or later-task file.
- Tests and validation: focused 27/27, model 601/601 across 72 suites, model Javadoc, root tests,
  executable example, javap/reflection/bytecode/import/source/generated-doc checks, links/anchors,
  fences, whitespace, exact scope, and synchronized-status checks passed.
- Documentation-agent review: completed in clean context
  `/root/implement_model_0017l/docs_0017l` with General, API/Javadoc, Planning, and Example
  profiles.
- Documentation impact: Tensor/Compile API and glossary now describe composition construction as
  current while compiler capture/decomposition/grouping, gradients, materialization, lowering,
  ONNX mapping, and execution remain planned.
- Javadoc review: Tensor and helper contracts are final; related semantic, provenance, graph,
  foundational, and neighboring expression contracts remain accurate unchanged.
- Glossary impact: composition, Tensor, and provenance wording now distinguishes immutable ordered
  inputs and individual UNSTACK outputs from grouped graph producers.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for 0017L. Task 0017M remains a separate Draft frontier without a
  specification.

Status: Complete
