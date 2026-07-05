# Task 0016J: Softmax Tensor Expressions

## Status

Complete

## Goal

Add the public model-level expression boundary for one-axis softmax and log-softmax. Each method
must validate floating input metadata, normalize the selected axis, retain the exact logical Shape
and data type, record exact one-input provenance, and return a fresh storage-free Tensor without
evaluating exponentials, logarithms, reductions, or probabilities.

This task completes public expression construction for the semantic contracts introduced by task
0016I. It does not select a finite-precision algorithm, define a gradient rule, capture a compiled
graph, decompose the operation, or report backend support.

## Scope

- Add exactly `Tensor.softmax(int axis)` and `Tensor.logSoftmax(int axis)`.
- Make each public method delegate exactly once with the corresponding `SoftmaxKind`.
- Add one package-private final `TensorSoftmaxExpressions` construction boundary in the existing
  tensor package.
- Accept exactly FLOAT64, FLOAT32, and BFLOAT16 input.
- Reject INT32, INT64, and BOOL without conversion, promotion, or truthiness semantics.
- Normalize one positive or negative axis exactly once through the exact input Shape.
- Retain the exact input Shape and data type in a new unresolved-layout descriptor.
- Preserve the input descriptor's `requiresGrad` eligibility metadata without defining a gradient
  formula or backward operation.
- Construct exact `SoftmaxKind` and `SoftmaxAttrs` metadata.
- Record exact ordered one-input provenance `[input]` and create one fresh derived Tensor with no
  label or host storage.
- Update only the exact Tensor API surface assertion in `TensorTest` and add one focused expression
  test.
- Finalize Javadocs, Tensor API, Compile API current-expression inventory, glossary, task evidence,
  master plan, and roadmap through the required independent documentation pass during
  implementation.

## Out of scope

- another public Tensor method, overload, static factory, builder, alias, or detailed task-0017
  specification
- temperature, scale, mask, bias, causal mask, epsilon, stabilization toggle, output type,
  precision option, or approximation mode
- sparse, sampled, hierarchical, adaptive, masked, fused-attention, cross-entropy, NLL, loss, or
  optimizer API
- integral or BOOL input, implicit cast, promotion, accumulation data type, or caller-selected
  output type
- value or host-storage inspection, allocation, copy, materialization, maximum, subtraction,
  exponential, logarithm, sum, division, probability calculation, mutation, or execution
- numerical stability algorithm, subtract-maximum implementation, reduction order, precision,
  reproducibility, overflow, underflow, NaN, infinity, signed zero, empty-axis, or error policy
- preserving resolved layout, view construction, strides, offset, storage aliasing, or
  materialization policy
- gradient rule, Jacobian-vector product, backward operation, saved forward value, autograd
  expansion, gradient Tensor, optimizer, or training behavior
- graph capture, NodeId, ValueId, compiled graph records, compiler inference, canonicalization,
  decomposition, fusion, cost, planning ownership, prepare, runtime, backend, route, kernel, ONNX,
  or execution behavior
- changing `SoftmaxKind`, `SoftmaxAttrs`, `Operation`, `TensorDescriptor`, Shape, DataType,
  TensorFactory, TensorProvenance, existing expression helpers, dependencies, Gradle, architecture,
  or another module
- implementation or detailed specification of task 0017 or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016I](0016i-softmax-semantic-kinds-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly:

```java
Tensor softmax(int axis)
Tensor logSoftmax(int axis)
```

Both require floating input, accept positive or negative axes, preserve logical shape and data
type, and participate in legacy gradient construction. Legacy evidence covers FLOAT64, FLOAT32,
BFLOAT16, non-contiguous input, attention/loss composition, ONNX, CPU, Metal, and CUDA paths.

Legacy public builders decompose the meanings into maximum, subtraction, exponential, sum,
division, and logarithm expressions. The new model instead records the first-class SOFTMAX or
LOG_SOFTMAX semantic identity introduced by task 0016I. A later compiler owns whether to preserve,
canonicalize, or decompose that meaning, while concrete backend prepare owns executable routes.

For one normalization slice `[1, 2, 3]`, the requested ideal values are approximately:

| Public method | Semantic result |
|---|---|
| `softmax(axis)` | `[0.09003057, 0.24472847, 0.66524096]` |
| `logSoftmax(axis)` | `[-2.40760596, -1.40760596, -0.40760596]` |

The first row sums to approximately one, and exponentiating the second row reconstructs the
first. This task constructs metadata for those meanings but computes none of the values.

## Architecture constraints

- Public Tensor expression construction and backend-independent operation semantics belong to
  `modules/model`.
- `Tensor` remains public mutable API state and is not an IR node. Each result carries immutable
  descriptor/provenance metadata and no initial host storage.
- The helper performs local expression validation and composition only. It must not capture a
  graph, traverse provenance, infer graph-wide facts, decompose the kind, or select backend
  support.
- Validation order is deterministic: non-null input, non-null kind, floating input eligibility,
  exact Shape access, one axis normalization, attributes construction, then common
  descriptor/operation/provenance/factory construction.
- Both public kinds are valid at the helper boundary. No registry, compatibility map, or string
  dispatch is needed.
- The exact input Shape reference is retained because both meanings preserve every logical
  position, including dynamic and zero extents.
- Result layout is unresolved even when the input descriptor has a resolved layout. Compiler,
  prepare, and backend paths own layout and materialization decisions.
- Result data type and `requiresGrad` eligibility equal the input descriptor values. This task
  defines no gradient rule or executable backward computation.
- `SoftmaxAttrs` stores the normalized non-negative axis. The operation stores the exact requested
  SOFTMAX or LOG_SOFTMAX kind.
- Provenance contains exactly `[input]`. It stores no graph IDs, executable closure, or runtime
  state.
- Every valid invocation creates one fresh Tensor identity, including repeated identical requests.
  Model construction performs no common-subexpression elimination or canonicalization.
- Package direction remains `model.tensor -> model.operation.normalization` plus existing model
  foundations. No dependency leaves `modules/model`.
- Stop if implementation requires a semantic-contract change, another public method or type,
  value/storage access, numerical or gradient policy, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor methods and the package-private
  expression-construction helper.
- `io.github.pho001.synaptik.model.operation.normalization` — supplies completed softmax kinds and
  axis attributes.
- existing datatype, shape, descriptor, operation, and provenance packages supply immutable model
  contracts without modification.

No package is added or renamed.

Type placement:

- `Tensor.softmax(int)` and `Tensor.logSoftmax(int)` — public fluent expression surface on the
  existing Tensor API.
- `io.github.pho001.synaptik.model.tensor.TensorSoftmaxExpressions` — package-private stateless
  boundary that validates and composes both normalization expressions.
- `TensorSoftmaxExpressionTest` — same-package test of helper and public expression behavior.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor softmax(int axis)

public Tensor logSoftmax(int axis)
```

Both are public, non-static, non-synchronized instance methods returning Tensor. Each performs
exactly one delegation:

```java
TensorSoftmaxExpressions.apply(this, SoftmaxKind.SOFTMAX, axis)
TensorSoftmaxExpressions.apply(this, SoftmaxKind.LOG_SOFTMAX, axis)
```

Neither method validates, reads state, constructs metadata, accesses storage, decomposes the
operation, or allocates an identity itself.

### Construction helper

Add one package-private final `TensorSoftmaxExpressions` with:

- no field, nested type, interface, instance state, or public/protected member;
- one private zero-argument constructor;
- exactly one package-private static
  `apply(Tensor input, SoftmaxKind kind, int axis)` entry;
- one private static `validateFloatingInput(Tensor input)` method; and
- one private static
  `create(Tensor input, SoftmaxKind kind, Shape shape, SoftmaxAttrs attrs)` method.

Add no overload, generic normalization registry, operation-family helper, factory, service, cache,
state, or test hook.

### Validation and construction order

`apply` performs exactly this sequence:

1. `Objects.requireNonNull(input, "input")`;
2. `Objects.requireNonNull(kind, "kind")`;
3. validate that `input.descriptor().dataType().isFloating()` is true;
4. read exact `Shape inputShape = input.descriptor().shape()`;
5. normalize `axis` exactly once through `inputShape.normalizeAxis(axis)`;
6. create one `SoftmaxAttrs(normalizedAxis)`;
7. call `create(input, kind, inputShape, attrs)` exactly once.

Ineligible input fails at step 3 with `IllegalArgumentException` and exact message:

```text
input must have a floating data type, but was <dataType>
```

Invalid axes use the existing Shape failure type and exact message. Type validation precedes axis
validation, including for integral, BOOL, and scalar inputs. Failures before factory delegation
consume no Tensor identity.

`create` performs exactly this sequence:

1. construct one `TensorDescriptor` from the exact input data type, exact supplied Shape,
   `Optional.empty()` layout, and unchanged input `requiresGrad`;
2. construct one `Operation(kind, attrs)` retaining both exact references;
3. construct one `TensorProvenance(operation, List.of(input))`;
4. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its result.

Construction reads no input label, existing provenance, host storage, layout geometry,
dimensions, element count, or values. The result has no label or storage and has exact one-input
provenance.

### Shape, type, and semantic behavior

- FLOAT64, FLOAT32, and BFLOAT16 inputs of rank greater than zero are structurally eligible,
  including dynamic and zero extents.
- INT32, INT64, and BOOL are rejected before axis normalization.
- Every scalar axis is invalid because rank zero has no axis.
- Positive and negative caller axes normalize to the same stored non-negative axis when they
  address the same input position.
- The result descriptor retains the exact input Shape reference, exact floating data type, and
  exact gradient-eligibility value; it never retains a resolved input layout.
- SOFTMAX and LOG_SOFTMAX remain distinct operation identities with the same local descriptor and
  provenance rules.
- Every valid invocation is fresh and explicit. Repeated calls must not return the input, reuse a
  result, or silently rewrite LOG_SOFTMAX into other model operations.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSoftmaxExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSoftmaxExpressionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training API, capabilities, Softmax semantic
contracts/tests, descriptor/factory/provenance/Shape/DataType contracts, other expression families,
focused architecture, ADRs/tests, conformance/integration tests, and Gradle.

## Maximum scope

At most two production files, two tests, and six documentation/planning files: ten paths total.
Do not modify another existing Java contract or test. Stop beyond this scope or if implementation
requires semantic-contract changes, another public surface, value/storage behavior, numerical or
gradient policy, dependencies, or architecture changes. Do not create task 0017.

## Javadoc requirements

- Document both public Tensor methods and the package-private helper, constructor, entry, and
  private methods.
- Explain normalization axis and slice, probability, log-probability, shape preservation, and the
  relationship between SOFTMAX and LOG_SOFTMAX for a newcomer.
- Include the concrete `[1, 2, 3]` example, approximate values, sum-to-one interpretation, and
  exponentiation relationship.
- Document every parameter, accepted floating types, axis range and negative normalization,
  returned metadata, freshness, unresolved layout, absent label/storage, and one-input provenance.
- Document every caller-visible failure type and deterministic ordering, including exact
  non-floating failures and scalar/out-of-range axes.
- Explain that construction does not inspect values, calculate probabilities, select a stable
  algorithm, decompose operations, or define numerical, gradient, compiler, backend, or execution
  behavior.
- Review Tensor, descriptor, factory, provenance, Shape, DataType, Operation, and softmax semantic
  Javadocs. Record why unchanged contracts remain accurate or stop on an out-of-scope discrepancy.

## Acceptance criteria

- Tensor exposes exactly the two requested methods with exact signatures and one helper delegation
  each; no other public API changes.
- The helper has exactly the planned package visibility, finality, constructor, three methods,
  imports, and no state or extra API.
- All three floating types succeed; INT32, INT64, and BOOL fail with exact type/messages before
  axis validation.
- Positive and negative axes normalize exactly once; invalid/scalar axes preserve the existing
  Shape failure contract.
- Static, dynamic, and zero-extent Shapes are accepted when the axis exists. Exact Shape reference,
  input type, and gradient eligibility are retained; layout is unresolved.
- Exact SOFTMAX or LOG_SOFTMAX operation/attributes metadata and ordered `[input]` provenance are
  present. Result label and host storage are absent.
- Every valid call produces a fresh identity without returning or mutating input, inspecting
  values/storage, decomposing, canonicalizing, or executing normalization.
- `TensorTest` changes only for the deliberate two-method public API shape. One focused test owns
  all softmax expression behavior.
- Focused and aggregate tests, Javadoc, root tests, reflection/javap/bytecode/import/scope and
  documentation validation pass.
- A separate clean-context documentation-focused agent finalizes permitted Javadocs, Tensor API,
  Compile API, glossary, planning, examples, and no-change conclusions.
- 0016J becomes Complete only after both passes; task 0017 remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorSoftmaxExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact public/helper API shape; exact kind delegation; all three floating
types and three rejected types; positive/negative/scalar/out-of-range axes; static, dynamic,
zero-extent, unresolved/resolved-input-layout Shapes; exact descriptor facts; same Shape/type/
eligibility references or values; exact kind/attributes/provenance references; absent label and
storage; input non-mutation; freshness; failure order and identity side effects; and absence of
value, storage, graph, compiler, runtime, or backend behavior.

Manually inspect reflection, `javap -p -c -s`, source, imports, and bytecode for exact public
delegations, one axis normalization, validation/construction order, exact helper surface, one
descriptor/operation/provenance/factory path, no hidden state, no value/storage access, and no
cross-layer imports. Validate generated Javadoc, Tensor API/Compile API/glossary, newcomer example,
links/anchors/fences/whitespace, exact ten paths, synchronized statuses, and absence of a task-0017
specification.

## Dependencies

- 0001 supplies floating DataType categories and gradient eligibility.
- 0002 supplies exact immutable Shape retention and positive/negative axis normalization.
- 0006 supplies immutable generic Operation composition.
- 0007 supplies unresolved TensorDescriptor construction.
- 0011–0013 supply public Tensor metadata, central derived identity allocation, and immutable
  operation/input provenance.
- 0016I supplies exact SOFTMAX/LOG_SOFTMAX identities and normalized-axis attributes.

## Follow-up tasks

- Task 0017 remains Draft for layout and view operations. Its detailed decomposition must be
  planned only after 0016J is complete.
- Compiler tasks own capture, inference validation, canonicalization, optional decomposition,
  autograd expansion, and graph optimization.
- Backend/config/conformance tasks own stable finite-precision algorithms, storage traversal,
  fused/decomposed routes, kernels, and cross-backend parity.
- Loss and attention tasks may compose softmax semantics later without adding hidden behavior to
  this expression helper.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes existing model-owned public Tensor, descriptor,
operation, provenance, Shape, and factory contracts without changing module boundaries or
dependency direction. Stop if an architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0016I/0016J, Tensor API,
Compile API, Training API, glossary, current DataType/Shape/TensorDescriptor/Tensor/TensorFactory/
TensorProvenance/Operation/SoftmaxKind/SoftmaxAttrs contracts and tests, and Java 26 Gradle
configuration.

Implement task 0016J exactly. Modify Tensor.java and add package-private final
TensorSoftmaxExpressions.java for production. Update TensorTest only for the exact two-method API
surface and add TensorSoftmaxExpressionTest. Add exactly softmax(axis) and logSoftmax(axis), each
delegating once to the shared helper and exact kind.

The helper has exactly apply, validateFloatingInput, and create. Null-check input then kind, accept
only floating data types, retain the exact input Shape, normalize the axis once, construct exact
SoftmaxAttrs, preserve exact type and requiresGrad in an unresolved descriptor, create the exact
kind Operation and one-input provenance, and call createDerived once with no label/storage. Every
call is fresh.

Do not inspect or normalize values/storage, preserve resolved layout, convert/promote types,
select a numerical algorithm, decompose operations, define gradient rules, capture graphs, add
overloads, change existing contracts, or introduce compiler/runtime/backend behavior. Stop beyond
ten paths or on architecture uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record related-contract/capability/Training API/
architecture no-change conclusions, and rerun validation.

Update task 0016J, model master plan, and roadmap only for planning status/evidence. Do not mark
0016J Complete until both passes succeed. Leave task 0017 Draft without a specification. Do not
commit or push.
```

## Local decisions

- Provide exactly the two legacy public signatures and use one shared helper because both meanings
  have identical local validation, descriptor, and provenance rules.
- Accept only the three current floating types. Integral and BOOL normalization would introduce
  conversion semantics absent from the selected capability.
- Preserve exact input data type and `requiresGrad` eligibility. This is model metadata and does
  not copy legacy mutable gradient callbacks or claim a backward implementation.
- Retain exact input Shape and leave layout unresolved because normalization is shape-preserving
  while materialization belongs to later lifecycle stages.
- Record first-class SOFTMAX or LOG_SOFTMAX operations rather than reproducing legacy public
  decomposition. Compiler and backend owners later decide fused versus primitive representation.
- Use a dedicated tensor helper rather than extending aggregate or scan helpers. Normalization
  preserves all positions but depends on a complete axis slice.

## Known limitations

- No values or probabilities are calculated and no result storage exists at model construction.
- No finite-precision, numerical-edge, gradient, compiler, ONNX, backend, or kernel policy is
  implemented.
- Temperature, masking, scaling, fused attention, and loss composition remain outside this API.

## Validation evidence

Planning read the architecture contract and focused architecture explanations; documentation and
planning rules; roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011,
0012, 0013, and 0016I; current Tensor, descriptor, factory, provenance, Shape, DataType, softmax
semantic contracts/tests; Tensor/Compile/Training APIs; glossary; and Java 26 Gradle configuration.

The legacy branch was read directly. It confirms exactly one public method per kind, floating
eligibility, negative-axis normalization, shape/type preservation, mutable gradient callbacks,
stable mathematical intent, FLOAT64/FLOAT32/BFLOAT16, non-contiguous input, attention/loss
composition, ONNX, CPU, Metal, and CUDA evidence. Legacy decomposition, callbacks, operation
traits, storage, lowering, fusion, and kernels are excluded or assigned to later owners.

Planning selected two public methods, one three-method package-private helper, one focused test,
and no new package. Existing contracts suffice; no semantic, dependency, build, or architecture
change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three planning files.
- All 179 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: twelve in this task, two in the master plan, and zero in the
  roadmap.
- All 20 task-template headings are present, together with the focused Capability origin,
  Required contract, and Javadoc requirements sections.
- The `[1, 2, 3]` SOFTMAX and LOG_SOFTMAX examples were independently recalculated; displayed
  rounding matches, the unrounded SOFTMAX values sum to one, and exponentiated LOG_SOFTMAX values
  match them.
- Task, model master plan, and roadmap consistently identify 0016J as Ready and task 0017 as Draft.
- No detailed task-0017 specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0016j` added the two exact Tensor methods, the
  dedicated package-private helper, and the focused expression suite while changing `TensorTest`
  only for the deliberate public API expansion. Clean documentation context
  `/root/implement_model_0016j/review_model_0016j_docs` then independently read the required
  architecture, documentation, planning, historical task, API, glossary, Gradle, final source,
  final tests, generated Javadoc, and complete diff. It applied General, API/Javadoc, Planning,
  and Example profiles.
- Independent source and test inspection confirmed exactly one public delegation per method and
  exact SOFTMAX/LOG_SOFTMAX kind selection. The helper null-checks input then kind, validates
  floating type before axis, retains the exact Shape, normalizes the axis once, constructs one
  `SoftmaxAttrs`, then constructs exactly one unresolved descriptor, one Operation, one
  `[input]` provenance value, and one derived Tensor. Results retain exact type and
  `requiresGrad`, have no label or storage, remain fresh, and do not inspect values or preserve
  input layout.
- The submitted `Tensor.softmax`, `Tensor.logSoftmax`, and `TensorSoftmaxExpressions` Javadocs are
  complete without revision. They document every parameter, result, failure and validation-order
  condition; normalization slices; floating eligibility; exact Shape/type/eligibility retention;
  unresolved layout; absent label/storage; freshness and one-input provenance; the `[1, 2, 3]`
  approximate values; the sum-to-one and exponentiation relationships; and the numerical,
  gradient, compiler, backend, runtime, and execution boundaries.
- `docs/api/tensor-api.md` now treats both public expressions as current, adds them to the mental
  model and current expression inventory, and includes a complete Shape/axis/kind/provenance
  example whose stated output was checked against the focused tests and public contracts.
  `docs/api/compile-api.md` now accepts softmax expressions as current future-compiler inputs while
  keeping compiler entry, traversal, capture, normalization inference/canonicalization, optional
  decomposition, artifacts, and execution planned. `docs/glossary.md` synchronizes softmax,
  normalized-axis, provenance, Tensor, operation-family, and current-status distinctions.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorSoftmaxExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports 8 focused
  softmax tests plus 14 Tensor tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 58 XML suites report 449 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` includes both
  method signatures, parameter/result/failure sections, the ideal examples, floating/type/axis
  rules, exact metadata/provenance result, and deferred numerical/gradient/compiler/backend/
  runtime/execution behavior. Source review covers the package-private helper and private methods
  omitted from public generated output.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 actionable repository test tasks completed without
  failure.
- `javap -p -c -s` confirmed the exact stateless helper shape, private constructor, one
  package-private `apply`, two private methods, deterministic validation order, one
  `Shape.normalizeAxis` call, one descriptor/operation/provenance/factory path, and each public
  method's single exact-kind helper call. Import and source scans found only permitted model/JDK
  dependencies and no value/storage access, graph/compiler/planning/runtime/prepare/backend state,
  numerical calculation, decomposition, or hidden state.
- A local Markdown target-and-anchor validator resolved all 271 links across the six changed
  documentation/planning files. Backtick fences are balanced, targeted trailing-whitespace checks
  found no matches, generated-Javadoc links render, status terminology is synchronized, and
  `git diff --check` passed.
- Exact scope review confirmed the required ten paths: two production files, two tests, Tensor
  API, Compile API, glossary, this task, model master plan, and roadmap. Task 0017 remains Draft,
  and no task-0017 specification exists. No commit or push was performed.
- Training API remains accurate unchanged because this task adds no gradient formula, backward
  operation, autograd, optimizer, parameter, publication, or session behavior. `capabilities.md`
  remains accurate because it already inventories softmax/log-softmax and distinguishes model
  expression construction from later executable support.
- `SoftmaxKind`, `SoftmaxAttrs`, and `SoftmaxSemanticsTest` remain accurate unchanged because the
  new methods consume their existing first-class meanings and normalized-axis contract without
  changing semantic pairing or finite-precision policy. `DataType`, `Shape`, `TensorDescriptor`,
  `TensorFactory`, `TensorProvenance`, `Operation`, and the aggregate/scan and other expression
  contracts remain accurate because this task composes their existing validation, descriptor,
  identity, and provenance behavior without modifying those contracts.
- `ARCHITECTURE.md`, focused architecture documents, ADRs, and architecture tests remain accurate
  unchanged because the work stays inside the authorized model layer and changes no module
  boundary, dependency direction, lifecycle, or architecture rule. Backend-conformance and
  integration tests remain unchanged because no backend behavior or end-to-end execution exists.
  Root/model Java 26 Gradle configuration and other modules remain unchanged because no dependency,
  source set, language level, preview/incubator feature, build task, or cross-module behavior
  changed.

## Implementation notes

- Added `Tensor.softmax(int)` and `Tensor.logSoftmax(int)` as exact one-call delegations to the
  dedicated package-private construction boundary.
- Added local floating validation, one Shape axis normalization, exact metadata retention,
  unresolved layout, first-class kind/axis attributes, exact one-input provenance, and fresh
  storage-free derived construction without numerical or cross-layer behavior.
- Added the focused eight-test suite and expanded only Tensor's public API-shape assertion for the
  two methods.
- The documentation pass left complete production Javadocs unchanged and finalized Tensor API,
  Compile API, glossary, and synchronized planning status/evidence.

## Completion summary

- Completed changes: Implemented and documented public one-axis softmax and log-softmax Tensor
  expression construction with exact floating validation, axis normalization, metadata retention,
  and one-input provenance.
- Files changed or created: Exactly two production files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused suites passed 22/22; all 449 model tests across 58 suites, model
  Javadoc, root tests, bytecode/reflection/import/source/generated-documentation/manual checks,
  271 Markdown link/anchor checks, fence/whitespace checks, exact scope/status checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0016j/review_model_0016j_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, and glossary now present softmax/log-softmax
  public expressions as current while preserving planned numerical, gradient, compiler, backend,
  runtime, and execution boundaries.
- Javadoc review: Tensor and helper Javadocs are complete unchanged; all related foundational,
  semantic, descriptor, factory, provenance, aggregate/scan, and expression contracts remain
  accurate for the reasons recorded above.
- Glossary impact: Softmax/log-softmax, normalized-axis, provenance, Tensor, and operation-family
  status now reflect current public expression construction.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016J. Task 0017 remains Draft without a detailed
  specification.

Status: Complete
