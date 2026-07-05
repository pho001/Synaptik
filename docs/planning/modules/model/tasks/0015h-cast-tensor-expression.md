# Task 0015H: Cast Tensor Expression

## Status

Complete

## Goal

Expose the implemented `CAST` semantic family through one public, backend-independent
`Tensor.cast(DataType targetDataType)` expression method. Every non-null target data type must be
representable from every current source data type, including a target equal to the source. Every
successful call returns a fresh storage-free Tensor whose descriptor retains the exact input
Shape, changes only the requested data type and derived gradient eligibility, and whose provenance
records the exact input plus `CastKind.CAST` and `CastAttrs(targetDataType)`.

This task constructs explicit cast expression metadata only. It does not read or convert values,
define numerical conversion rules, allocate result storage, create gradient rules, capture a
compiled graph, remove redundant casts, or define backend support.

## Scope

- Add exactly one public instance method, `Tensor.cast(DataType targetDataType)`.
- Add one package-private final `TensorCastExpressions` helper with one package-private static
  `apply(Tensor, DataType)` entry and no other behavior.
- Require a non-null input and target data type with deterministic parameter-name messages.
- Accept every one of the 36 current source/target `DataType` pairs without compatibility lookup,
  implicit conversion, or backend-capability validation.
- Retain the input descriptor's exact immutable `Shape` reference.
- Leave result layout unresolved, even when the input layout is resolved or the target equals the
  source data type.
- Derive result gradient eligibility as
  `input.requiresGrad && sourceDataType.isFloating() && targetDataType.isFloating()`.
- Construct exactly one `Operation` from `CastKind.CAST` and one new
  `CastAttrs(targetDataType)`.
- Construct exactly one `TensorProvenance` with the exact ordered input list `[input]`.
- Delegate final identity-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Return a fresh explicit cast expression for same-type requests; do not return the input or
  eliminate the operation in the model API.
- Update the exact Tensor public-API reflection test and add one focused cast-expression test.
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.

## Out of scope

- eager value conversion, host-storage reads or writes, allocation, copy, materialization,
  mutation, aliasing, or output storage attachment
- numerical conversion policy, including rounding, truncation, saturation, overflow, underflow,
  precision loss, NaN, infinity, signed zero, BFLOAT16 bit conversion, BOOL zero/non-zero meaning,
  or invalid-value behavior
- returning the input for same-type requests, constant folding, identity elimination, cast-chain
  simplification, canonicalization, interning, common-subexpression elimination, or caching
- source/target compatibility tables, backend-support checks, conversion modes, rounding modes,
  attributes beyond the existing target data type, or operation registries
- resolved output layout, input-layout propagation, stride preservation, view preservation,
  materialization policy, or shape inference beyond exact reference retention
- gradient values, gradient callbacks, backward operations, cast-back rules, autograd expansion,
  optimizer behavior, training-root policy, or training execution
- changes to `DataType`, `CastKind`, `CastAttrs`, `Operation`, `TensorDescriptor`,
  `TensorProvenance`, `TensorFactory`, or another existing Java contract
- another public overload, static cast entry, alias, factory, builder, expression public type, or
  caller-supplied label
- graph traversal, graph IDs, cycle checks, graph capture, compiled graph construction,
  publication, compiler inference, optimization, or artifacts
- planning ownership, capability providers, backend selection, fusion, costs, lowering, kernels,
  prepare, runtime residency, execution, tracing, ONNX mapping, or engine behavior
- dependencies, Gradle changes, architecture changes, another module, unrelated refactors, or a
  detailed task-0016 specification

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
- [Task 0001](0001-data-type-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0015G](0015g-cast-semantic-kind-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes fluent `Tensor.cast(DataType targetType)`. Its
public builder accepts conversions among floating, BFLOAT16, integral, and BOOL data types,
preserves the input shape, disables gradient eligibility when either side is non-floating, and
supports floating-to-floating backward conversion. Legacy tests cover same-type requests, mixed
floating precision, BFLOAT16, integral/floating and BOOL/floating conversion, strided inputs,
expression chaining, gradients, ONNX mapping, and CPU and Metal execution routes.

Legacy returns the original Tensor when source and target types are equal. The new model does not
copy that eager canonicalization. A public call records the caller's explicit semantic request;
later compiler optimization owns removal of redundant casts and cast-chain simplification. This
preserves representability and keeps model construction deterministic and consistent with the
current expression APIs, while deliberately moving the optimization to its architectural owner.

Legacy code also installs Tensor-local gradient callbacks, performs storage conversion, carries
operation traits, and couples construction to executable routes. Those mechanisms are not copied.
This task retains only public model expression construction. Compiler autograd later owns backward
graph expansion; compiler optimization owns canonicalization; concrete backends and conformance
work own numerical conversion, lowering, storage access, and execution.

## Architecture constraints

- `Tensor` remains public mutable API state and must not become an IR node or executable value.
- `TensorCastExpressions` performs deterministic local model construction only. It must not read
  values or storage, traverse provenance, capture a graph, query backend support, or execute a
  conversion.
- `Operation` owns only the exact backend-independent `CAST` kind and target attributes. Source
  type belongs to the input descriptor and must not be duplicated in `CastAttrs`.
- `TensorProvenance` owns the exact one-input relationship. Its ordered list is exactly `[input]`.
- Result identity comes only from the existing package-private `TensorFactory.createDerived` seam.
  No second allocator, caller-supplied ID, registry, cache, or service is introduced.
- Every successful call creates a fresh, unlabeled, storage-free Tensor, including same-type casts.
  Model construction does not perform compiler-owned canonicalization.
- Every current source/target pair is representable. This does not promise executable support from
  every backend or define conversion results for edge values.
- The result Shape is the input descriptor's exact immutable reference. Cast does not broadcast,
  reshape, infer, reconstruct, or normalize dimensions.
- Result layout is always unresolved. A change of element width may invalidate input byte geometry,
  while even a same-type cast is a new logical value whose physical realization is not established
  by expression construction.
- Result gradient eligibility is true only when the input requested gradients and both source and
  target data types are floating. This is descriptor metadata, not a gradient-rule or backend
  differentiability guarantee.
- Package direction is `model.tensor -> model.operation.elementwise.cast`, plus existing
  `model.tensor -> model.operation` and `model.datatype`. The cast package must not import Tensor,
  and no package cycle may be introduced.
- Stop if implementation requires a foundational contract change, conversion policy, storage
  access, graph capture, gradient rule, backend query, dependency, or architecture decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns the public cast method, local result-descriptor
  construction, provenance, and package-private derived Tensor factory seam.
- `io.github.pho001.synaptik.model.operation` — supplies immutable `Operation` composition.
- `io.github.pho001.synaptik.model.operation.elementwise.cast` — supplies the exact `CAST` identity
  and immutable target-data-type attributes.
- `io.github.pho001.synaptik.model.datatype` — supplies the source and target type vocabulary and
  floating-category metadata.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the exact `cast` method and delegates all construction behavior.
- `io.github.pho001.synaptik.model.tensor.TensorCastExpressions` — package-private, stateless local
  cast-expression boundary colocated with Tensor, descriptors, provenance, and the package-private
  factory seam it must use.
- `TensorCastExpressionTest` — same-package focused test so it can verify helper shape and behavior
  without widening production visibility.

## Required contract

### Public Tensor surface

Add exactly this public instance method to `Tensor`:

```java
public Tensor cast(DataType targetDataType)
```

It delegates exactly once and returns the exact result:

```java
return TensorCastExpressions.apply(this, targetDataType);
```

The method is public, final only through the final enclosing class, non-static, non-synchronized,
and has no overload or alias. It performs no validation, descriptor construction, factory call, or
other work outside the one helper delegation.

### Helper shape

Create exactly one package-private final production helper:

```java
final class TensorCastExpressions {
    private TensorCastExpressions() {
    }

    static Tensor apply(Tensor input, DataType targetDataType) {
        // exact validation and construction contract below
    }
}
```

The helper has zero fields, zero nested types, one private zero-argument constructor, and exactly
one package-private static `apply` method. It declares no public or protected API, overload,
factory, cache, registry, compatibility table, or test hook.

### Validation and construction order

`apply` performs exactly this order:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(targetDataType, "targetDataType")`.
3. Read `sourceDataType` from `input.descriptor().dataType()` and read the exact input Shape.
4. Derive `requiresGrad` as input eligibility AND floating source AND floating target.
5. Create one `TensorDescriptor` from target type, exact input Shape reference,
   `Optional.empty()`, and derived eligibility.
6. Create one `CastAttrs` from the exact target reference.
7. Create one `Operation` from `CastKind.CAST` and the exact attributes value.
8. Create one `TensorProvenance` from that operation and `List.of(input)`.
9. Delegate exactly once to
   `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` and return its exact
   result.

No source/target pair is rejected after the two null checks. A same-type request follows all nine
steps and creates a fresh explicit cast expression; it does not return before descriptor or
identity construction. Failures before the factory call consume no Tensor ID. A valid request may
fail only if the central allocator is exhausted, after local immutable model values are built.

### Result descriptor and gradient eligibility

The result descriptor has exactly:

- `dataType == targetDataType`;
- `shape == input.descriptor().shape()` by reference;
- `layout().isEmpty()`; and
- `requiresGrad == input.descriptor().requiresGrad()
  && sourceDataType.isFloating() && targetDataType.isFloating()`.

All 36 current source/target pairs are valid. `TensorDescriptor` already prevents true gradient
eligibility on non-differentiable source tensors, but the helper still expresses the complete
source-and-target rule explicitly. Floating-to-floating casts retain an existing true request.
Floating-to-integral, floating-to-BOOL, integral-to-floating, BOOL-to-floating, integral-to-
integral, and BOOL-related casts produce false eligibility. No backward rule is created.

The exact input Shape reference is valid for scalar, zero-sized, static, and dynamic shapes.
Resolved input layout is never copied. The helper does not construct dense layout, preserve view
geometry, or access referenced element span.

### Operation and provenance

Every result has present provenance containing:

```java
new Operation(CastKind.CAST, new CastAttrs(targetDataType))
```

and exact ordered inputs:

```java
List.of(input)
```

The attributes retain the exact target enum reference. The operation does not duplicate the source
type, input, result descriptor, conversion policy, or backend fact. Repeated calls create fresh
Tensor, attributes, operation, and provenance values. A cast chain records only its immediate
input at each link, so `input.cast(A).cast(B)` remains two explicit expressions until a compiler
chooses a legal rewrite.

### Ownership and side effects

The input Tensor, descriptor, Shape, label, provenance, host-storage association, and storage
contents remain unchanged. The output has empty label and no host storage. The helper retains the
input only through immutable provenance and retains the input Shape through the result descriptor.
It never retains storage as output storage, reads memory, converts a value, or mutates either
Tensor.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCastExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCastExpressionTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Existing `DataType`, `TensorDescriptor`, `TensorProvenance`, `TensorFactory`, `Operation`,
  `CastKind`, and `CastAttrs` Javadocs/tests.
- Existing expression contracts, focused architecture documents, ADRs, architecture tests,
  backend-conformance tests, integration tests, and Gradle configuration.

## Maximum scope

At most two production files, two test files, and six documentation/planning files: ten paths
total.

`Tensor.java` and `TensorTest.java` may change only for the one exact public `cast` method, its
import/Javadocs, exact API-shape expectations, and non-synchronization assertions. Do not change
existing fields, constructor, metadata/storage behavior, expression methods, equality, hashing,
diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract,
source/target matrix, numerical policy, resolved-layout policy, storage access, graph/compiler
behavior, gradient rule, another documentation file, or more than ten paths, stop and propose a
follow-up or architecture decision. Do not create task 0016.

## Javadoc requirements

- Update Tensor type Javadoc only as needed to include explicit cast expression construction while
  preserving the distinctions among public Tensor state, provenance, graph IR, gradient
  eligibility, and executable values.
- `Tensor.cast` must explain the target role, acceptance of every current source/target pair,
  exact Shape retention, unresolved result layout, gradient-eligibility rule, fresh identity
  including same-type requests, absent label/storage, operation attributes, one-input provenance,
  and deferred numerical conversion, canonicalization, gradients, and execution.
- Document `targetDataType` with `@param`, the fresh result with `@return`, and exact null and
  identifier-exhaustion failures with `@throws`.
- Document the package-private helper, private constructor, and `apply` method with exact
  validation/construction order, reference ownership, source/target roles, ID side effects, and
  failure behavior.
- Explain why the model does not return the receiver for a same-type request and that compiler
  optimization owns redundant-cast removal.
- Explain that gradient eligibility is descriptor metadata only and does not create a cast-back
  gradient rule or promise backend differentiability.
- Review related foundational and expression Javadocs and record why they remain accurate or stop
  on an out-of-scope inconsistency.

## Acceptance criteria

- Tensor declares exactly one new `public Tensor cast(DataType)` method; no static form, overload,
  alias, or unrelated public API is added.
- The public method delegates once to `TensorCastExpressions.apply` and performs no other work.
- `TensorCastExpressions` has exactly the specified visibility, finality, constructor, method,
  fields, and nested-type shape.
- Null validation, source/shape reads, eligibility derivation, descriptor, attributes, operation,
  provenance, and factory delegation occur in the exact specified order.
- All 36 current source/target pairs succeed without compatibility or backend lookup.
- Every result has exact target type, exact input Shape reference, empty layout, derived gradient
  eligibility, empty label, no host storage, fresh identity, exact operation/attributes, and exact
  one-input provenance.
- Same-type requests return a fresh explicit CAST expression and consume one identity; they never
  return the input or disappear during model construction.
- Floating-to-floating casts preserve a true input request. Every cast with a non-floating source
  or target has false result eligibility. False input eligibility remains false for all targets.
- Scalar, zero-sized, fully static, dynamic, resolved-layout, unresolved-layout, leaf, and derived
  inputs are accepted without Shape reconstruction or storage access.
- Repeated calls and cast chains remain fresh and explicit, with each provenance value retaining
  only its exact immediate input.
- No input Tensor metadata, provenance, label, storage association, or storage contents are
  mutated or attached as output storage.
- No value conversion, numerical policy, gradient rule, graph state, backend fact, dependency,
  build option, or architecture change is added.
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.
- A separate documentation-focused agent finalizes Javadocs, Tensor API, Compile API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for Training API, capabilities, architecture, and related contracts.
- Task 0015H becomes Complete only after both passes. Task 0016 remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorCastExpressionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and signature;
- exact public `cast(DataType)` signature, visibility, instance form, non-synchronization, and one
  delegation;
- every one of the 36 current source/target pairs and exact result target type;
- exact Shape reference retention for scalar, zero-sized, static, and dynamic inputs;
- empty result layout for resolved, unresolved, and same-type inputs;
- exact gradient-eligibility matrix, including true and false floating inputs and every
  non-floating source/target category;
- exact `CastKind.CAST`, exact `CastAttrs` type and target, and exact one-input provenance;
- fresh same-type results, repeated calls, and immediate-input cast chains without
  canonicalization;
- empty label, absent storage, fresh identity, and unchanged input metadata/storage/content;
- helper/public null validation and no identity allocation for null target; and
- identifier-exhaustion propagation only after valid local construction.

Manually inspect `javap -p -c -s`, method bytecode, reflection, and imports for the exact Tensor
method descriptor, helper shape, sole delegation, validation/construction order, complete
eligibility expression, exact Shape reference, empty layout/label, fixed `CAST` and typed target
attributes, one-input provenance, and absence of synchronization on new public/helper entries.
Confirm no numeric or storage access, source/target matrix, backend lookup, conversion loop,
same-type early return, resolved layout, gradient rule, graph/compiler/runtime/backend type, cost,
fusion, route, registry, service, dependency, or build change appears. Validate generated Javadoc,
Tensor/Compile API status, glossary, links/anchors/fences/whitespace, exact ten-path scope,
synchronized statuses, and absence of a task-0016 specification.

## Dependencies

- Task 0001 supplies the exact six-value `DataType` vocabulary and floating/differentiable
  metadata.
- Task 0006 supplies immutable generic `Operation` composition.
- Task 0007 supplies `TensorDescriptor` and its differentiability validation.
- Task 0011 supplies public Tensor state and the exact API surface to extend.
- Task 0012 supplies centralized Tensor identity allocation through `TensorFactory`.
- Task 0013 supplies immutable ordered provenance and `TensorFactory.createDerived`.
- Task 0015G supplies exact `CastKind.CAST` semantics and `CastAttrs(targetDataType)`.

## Follow-up tasks

- Task 0016 remains Draft for reduction and scan operation semantics and Tensor expressions.
- Compiler tasks later own provenance traversal, graph capture, redundant-cast and cast-chain
  canonicalization, autograd expansion, optimization legality, and compile artifacts.
- Backend, ONNX, and conformance tasks later own mapping, supported conversion matrices, rounding,
  truncation/saturation, BOOL behavior, special values, lowering, storage access, and execution.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public Tensor state, backend-independent
Operation semantics, descriptors, and minimal provenance to `modules/model`. This task composes
those contracts and leaves compiler-owned canonicalization plus backend-owned conversion and
execution in their existing layers.

If implementation requires storage conversion, a gradient rule, graph capture, backend metadata,
another dependency, or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0006/0007/0011/0012/0013/0015G/0015H, Tensor API,
Compile API, Training API, glossary, current DataType/TensorDescriptor/Tensor/TensorFactory/
TensorProvenance/Operation/CastKind/CastAttrs contracts and tests, and Java 26 Gradle configuration.

Implement task 0015H exactly. Modify Tensor.java and add package-private final
TensorCastExpressions.java for production. Update TensorTest only for the exact one-method public
API surface and add TensorCastExpressionTest. Add exactly cast(DataType), delegating once to the
shared helper.

The helper has exactly one package-private static apply(input,targetDataType) method. Follow the
task's exact input/target/source/descriptor/attributes/operation/provenance/construction order and
messages. Accept all 36 current source/target pairs. Retain the exact input Shape, leave layout
unresolved, and set requiresGrad only when the input requested it and both types are floating.
Create exact CAST/CastAttrs semantics, exact one-input provenance, and delegate once to
TensorFactory.createDerived with no label/storage. Every valid call, including same-type cast,
returns a fresh explicit expression; compiler work later owns redundant-cast elimination.

Do not inspect or convert values/storage, return the input, add a compatibility matrix or numerical
policy, preserve resolved layout, define gradient rules, capture a graph, add overloads, change
existing contracts, or introduce compiler/runtime/backend behavior. Stop beyond ten paths or on
architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff and evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record
related-contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0015H, model master plan, and roadmap only for planning status/evidence. Do not mark
0015H Complete until both passes succeed. Leave 0016 Draft without a specification. Do not commit
or push.
```

## Local decisions

- The public method keeps the legacy-compatible fluent name and target-only signature:
  `input.cast(targetDataType)`. Source type remains an input-descriptor fact.
- All 36 current source/target pairs are representable. Executable conversion support and exact
  numerical results remain backend and conformance responsibilities rather than model validation.
- Same-type calls create fresh explicit expressions. Legacy returned the input, but that eagerly
  erased the requested semantic operation. The current architecture assigns redundant-operation
  canonicalization to compiler optimization, and all existing public expression methods preserve
  explicit caller requests as fresh Tensor state.
- The result retains the exact input Shape reference because cast is elementwise and changes no
  logical dimensions. It leaves layout unresolved because expression construction does not assert
  physical geometry or storage reuse, particularly across element-width changes.
- Gradient eligibility follows the legacy semantic boundary: it survives only an already-eligible
  floating-to-floating cast. The task creates no backward rule; compiler autograd later decides
  whether and how to cast gradients back.
- One single-entry package-private helper is sufficient because the family has one semantic kind
  and one public arity. A generic conversion framework would add abstraction without another
  current use.

## Known limitations

- The result records cast semantics and provenance but contains no converted values or storage.
- Numerical behavior for precision loss, overflow, truncation, BFLOAT16, BOOL, NaN, infinity, and
  signed zero remains unspecified until backend/conformance work.
- Gradient eligibility is metadata only; no autograd rule or cast-back expression is implemented.
- Same-type and redundant chained casts remain explicit until a future compiler optimization
  proves and performs their removal.
- No compiler capture, ONNX mapping, backend support, native storage conversion, or execution is
  implied.

## Validation evidence

Planning reviewed the architecture contract and focused module/dependency/lifecycle explanations;
documentation and planning rules; roadmap; model capabilities and master plan; tasks 0001, 0006,
0007, 0011, 0012, 0013, and 0015G; current DataType, TensorDescriptor, Tensor, TensorFactory,
TensorProvenance, Operation, CastKind, CastAttrs, and expression-helper source/tests; Tensor,
Compile, and Training APIs plus glossary; and Java 26 Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms fluent public
`Tensor.cast(DataType)`, all current source/target categories, exact shape preservation,
same-type input return, floating-only gradient propagation, strided inputs, expression chaining,
ONNX mapping, and CPU/Metal execution evidence. Legacy Tensor callbacks, storage conversion,
operation traits, backend routes, and same-type eager canonicalization are excluded from the model
expression task. The same-type difference is deliberate: explicit expression construction owns
requested semantics, while the compiler owns redundant-cast removal.

Planning selected one public method and one package-private single-entry helper. Existing data-type,
descriptor, provenance, and derived-factory contracts are sufficient; no new package, dependency,
foundation contract, or architecture rule is required.

Planning validation:

- `git diff --check` passed, and targeted trailing-whitespace inspection found no matches in the
  three changed planning paths.
- The canonical section scan found every required task-specification section, including exact
  public/helper contracts, package impact, bounded scope, validation, implementation handoff,
  decisions, limitations, and completion-evidence sections.
- Every local Markdown target linked from this task, the model master plan, and the roadmap
  resolves. Markdown fences are balanced in all three changed paths.
- Status inspection found 0015H `Ready` in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0016 remains `Draft` in both queues, and no
  task-0016 specification exists.
- Package inspection found no new package. The planned direction remains from `model.tensor` to
  existing operation, cast, and datatype contracts without a reverse dependency.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.

Implementation and independent documentation validation:

- The implementation pass changed exactly four Java/test paths: `Tensor.java`,
  `TensorCastExpressions.java`, `TensorTest.java`, and `TensorCastExpressionTest.java`.
  Independent source, test, diff, reflection, and bytecode inspection confirmed the exact public
  method/helper shapes, one public delegation, input-then-target null checks, source and exact Shape
  reads, complete gradient-eligibility expression, one unresolved descriptor, one `CastAttrs`,
  one `CAST` Operation, exact one-input provenance, and one `createDerived` delegation.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorCastExpressionTest` passed 9 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed 14 tests with zero failures, errors,
  or skips.
- `./gradlew :modules:model:test` passed all 362 tests across 48 suites with zero failures,
  errors, or skips. The matrix tests cover all 36 source/target pairs, exact Shape-reference
  retention, resolved/unresolved and scalar/zero/static/dynamic inputs, fresh same-type/repeated/
  chained expressions, gradient eligibility, storage/content non-observation, null ordering, and
  allocator exhaustion.
- `./gradlew :modules:model:javadoc` passed. Generated `Tensor.html` contains the public cast
  signature, fresh same-type contract, exact target/shape/result semantics, failure documentation,
  and compiler-owned redundant-cast boundary. The default public/protected Javadoc output omits the
  package-private helper, so its complete type, constructor, parameter, return, ownership, ordering,
  and failure Javadocs were reviewed directly in source.
- The independent documentation-focused context
  `/root/review_model_0015h_docs` applied General style, API/Javadoc style, Planning style, and
  Example format. It found the affected `Tensor` and `TensorCastExpressions` Javadocs complete
  without revision, then finalized the Tensor API, Compile API, glossary, this task, model master
  plan, and roadmap.
- The Tensor API now gives a newcomer-oriented public cast contract and complete compiled example,
  including all-pair representability, fresh same-type identity, exact Shape retention, unresolved
  layout, gradient eligibility, typed attributes, immediate-input provenance, storage ownership,
  and deferred numerical/compiler/backend behavior. `javac -cp
  modules/model/build/classes/java/main -d /private/tmp/synaptik-cast-doc
  CastExpressionExample.java` passed for the transient extracted example, and `java -cp
  modules/model/build/classes/java/main:/private/tmp/synaptik-cast-doc
  CastExpressionExample` printed all 12 documented lines exactly. The transient source file was
  removed after validation.
- The Compile API now inventories cast expression construction as current and keeps compiler entry,
  traversal, graph capture, redundant-cast/cast-chain canonicalization, optimization, and artifacts
  planned. No compiler behavior or callable compiler contract was added.
- The glossary now defines cast expression, updates implementation status, and aligns Cast,
  Tensor, provenance, operation-kind/attributes, and Tensor-versus-graph-value terminology without
  changing architecture authority.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for `Tensor` and
  `TensorCastExpressions` confirmed the exact method descriptors and modifiers, sole public
  delegation, zero-field helper, private constructor, validation/construction order, full
  short-circuit eligibility expression, exact Shape use, empty layout/label, typed operation and
  one-input provenance, one factory call, and absence of synchronization or same-type early return.
  Focused reflection tests independently confirm the same exact public/helper shapes.
- Import/package scans found only the planned model/JDK dependencies. Source and bytecode contain no
  value or storage access, conversion loop, source/target compatibility map, resolved output
  layout, gradient rule, graph/compiler/runtime/backend type, cost/fusion/route state, registry,
  service, cache, dependency, or build change.
- `./gradlew test` passed the complete repository lifecycle with all 36 actionable tasks
  successful. No architecture, backend-conformance, or integration test change is required because
  task 0015H changes no dependency rule, backend behavior, or end-to-end numerical execution.
- `git diff --check` passed. Targeted trailing-whitespace scans returned no matches. Fence counts
  are balanced in every changed Markdown file. The corrected local Markdown checker resolved 227
  file targets and heading anchors across the six changed documentation/planning files with zero
  errors. Two preliminary checker invocations stopped before repository validation because the
  first Ruby regular expression interpolated a heading quantifier and the installed Ruby lacks
  `Array#filter_map`; the compatible rerun produced the passing result.
- Exact scope inspection found only the authorized ten paths: two production files, two test files,
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Status is
  synchronized to Complete in this task and both queues. Task 0016 remains Draft and no task-0016
  specification exists. No commit or push was performed.
- The implementation handoff accurately reported that one combined diagnostic shell command was
  terminated after producing no output; its report inspection and Javadoc portions were rerun
  separately and succeeded. This documentation pass did not rely on that combined attempt and
  independently reran every required focused/aggregate test, Javadoc, root, bytecode, import,
  documentation, scope, and status validation.
- `docs/api/training-api.md` remains accurate unchanged because this task adds only immutable
  gradient-eligibility metadata, not a gradient rule, cast-back expression, trainable role,
  autograd expansion, optimizer, session, or execution behavior. The model capability baseline
  already inventories explicit cast and correctly separates model representation from executable
  parity, so it requires no task-status edit.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because cast
  construction stays in `modules/model`, preserves dependency direction, and adds no compiler,
  planning, prepare, runtime, backend, training, numerical, or end-to-end behavior. Java 26
  toolchain/release configuration also remains accurate and unchanged.
- Existing `DataType`, `TensorDescriptor`, `TensorFactory`, `TensorProvenance`, `Operation`,
  `CastKind`, and `CastAttrs` Javadocs remain accurate: their data-type categories,
  differentiability, descriptor eligibility, central identity allocation, exact provenance,
  generic semantic composition, cast identity, and target-only attributes are composed without
  changing those contracts. Existing expression-family Javadocs likewise remain accurate because
  cast adds a separate helper and public method without changing their validation or result rules.

## Implementation notes

- Added exactly one public fluent `Tensor.cast(DataType)` method and one package-private final,
  stateless, single-entry `TensorCastExpressions` helper.
- Every valid request creates a fresh unlabeled, storage-free explicit `CAST` expression. The
  helper accepts all 36 current pairs, retains the exact input Shape reference, leaves layout
  unresolved, and retains gradient eligibility only when the input already requests it and both
  types are floating.
- The result uses one target-only `CastAttrs`, one `CAST` Operation, exact one-input provenance,
  and the existing central derived-construction/identity path. No values or storage are accessed,
  and no conversion, canonicalization, gradient rule, graph capture, backend support, dependency,
  or execution behavior was introduced.
- Final documentation makes the public model behavior current while keeping numerical conversion,
  compiler canonicalization/autograd, backend lowering/support, and execution explicitly deferred.

## Completion summary

- Completed changes: Implemented and documented public explicit cast Tensor expression
  construction for every current source/target pair.
- Files changed or created: Exactly the two production, two test, and six documentation/planning
  paths listed under Affected files.
- Tests and validation: Focused suites passed 9/9 and 14/14; all 362 model tests across 48 suites,
  model Javadoc, root tests, the compiled documentation example, generated-Javadoc inspection,
  `javap`, reflection, import/forbidden-behavior scans, 227 Markdown link/anchor checks,
  fence/terminology/whitespace checks, exact-scope review, status checks, and `git diff --check`
  passed.
- Documentation-agent review: `/root/review_model_0015h_docs` independently completed the
  required clean-context pass using General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, glossary, task evidence, model master plan, and
  roadmap now describe cast expression construction as current while preserving all cross-layer
  boundaries.
- Javadoc review: Affected Tensor and helper Javadocs are complete without documentation-pass
  wording changes; all reviewed foundational and existing expression Javadocs remain accurate.
- Glossary impact: Cast expression and its distinction from value conversion, graph nodes,
  compiler canonicalization, gradient rules, storage, and executable backend support are current.
- Unresolved issues: None within task 0015H.
- Follow-up required: None for task 0015H. Task 0016 remains the next Draft planning frontier
  without a detailed specification.

Status: Complete
