# Task 0014F: Scalar Arithmetic and Clamp Tensor Expressions

## Status

Complete

## Goal

Expose scalar multiplication, scalar power, inclusive range clamp, lower-bound clamp, and
upper-bound clamp as public backend-independent Tensor expression methods. Each successful call
must validate a floating input, preserve the exact supplied binary64 parameter values, derive one
immutable unresolved-layout result descriptor, and return a fresh storage-free Tensor whose
provenance records the exact operation and input Tensor identity.

This task completes the public expression surface for the scalar elementwise semantics introduced
by task 0014E. It does not execute mathematics, inspect storage, canonicalize expressions, define
gradient rules, capture a compiled graph, or report backend support.

## Scope

- Add public `Tensor` overloads `mul(double)`, `pow(double)`, `clamp(double, double)`,
  `clampMin(double)`, and `clampMax(double)`.
- Add one package-private `TensorScalarExpressions` helper that owns local validation, typed
  attributes, immutable operation/provenance construction, and derived-Tensor creation.
- Accept only `BFLOAT16`, `FLOAT32`, and `FLOAT64` input Tensors.
- Preserve each caller-supplied Java `double` parameter unchanged in `ScalarValueAttrs` or
  `ClampRangeAttrs`, regardless of input data type.
- Preserve the exact input descriptor data type, exact immutable shape reference, and exact
  `requiresGrad` flag in a new unresolved-layout result descriptor.
- Represent `clamp` as one first-class `CLAMP` operation with `ClampRangeAttrs`, not as two public
  intermediate Tensors.
- Construct exactly one `TensorProvenance` with the exact input reference and delegate exactly once
  to `TensorFactory.createDerived` with no label and no storage.
- Update the exact Tensor public-API reflection test and add one focused scalar-expression test.
- Finalize affected Javadocs, Tensor API, Compile API current-status wording, glossary, task
  evidence, model master plan, and roadmap through the required independent documentation pass.

## Out of scope

- eager numerical execution, scalar application, constant folding, algebraic simplification,
  canonicalization, or returning an existing/eager Tensor for special parameter values
- quantizing, converting, normalizing, or caching scalar parameters as FLOAT32 or BFLOAT16 during
  public expression construction
- reading, copying, allocating, attaching, materializing, aliasing, or validating host storage
- integral or boolean scalar arithmetic, implicit Tensor casts, explicit cast insertion, result
  promotion, or non-floating Tensor support
- `float`, boxed `Number`, generic, static, reverse, in-place, variadic, or caller-kind overloads
- output labels, caller-supplied labels, expression strings, symbols, or serialization names
- resolved result layouts, input-layout preservation, view behavior, materialization policy,
  physical stride derivation, or backend routes
- changes to `ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`, `Operation`,
  `TensorProvenance`, `TensorFactory`, data-type, shape, or layout contracts
- generic kind-to-attributes validation, registries, factories, parsers, reflection discovery,
  services, visitors, string dispatch, or maps
- mathematical domain/range checks, finite-only policy, NaN/infinity/signed-zero behavior,
  rounding, precision, power edge cases, or clamp special-value execution semantics
- gradient values/rules, scalar-power differentiation, clamp boundary conventions, backward graph
  generation, autograd execution, optimizer, or training behavior
- graph traversal, graph IDs, cycle checks, capture, compiled graph construction, inference,
  common-subexpression elimination, or publication binding
- compiler optimization, planning ownership, backend support, lowering, fusion, kernels, runtime,
  prepare, execution, tracing, or engine behavior
- dependencies, Gradle changes, architecture changes, another module/family, or a detailed
  task-0015 specification

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
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0014D](0014d-unary-elementwise-tensor-expressions.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes public Tensor methods `mul(double)`,
`pow(double)`, `clamp(double, double)`, `clampMin(double)`, and `clampMax(double)`. They accept
floating Tensor inputs, preserve logical shape and result data type, and retain scalar parameters
for later lowering. Legacy backends choose double or cached float parameters according to the
prepared input data type.

Legacy public construction also performs eager semantic rewrites: multiply by zero/one/minus-one,
power by zero/one/minus-one/two, infinite clamp identities, and nested clamp collapse. Its range
clamp creates two intermediate public Tensors, builders attach gradient callbacks, descriptors
cache both double and float values, and operations contain fusion/cost/result/backend-adjacent
metadata.

The new model retains the five public capabilities, floating eligibility, shape/type preservation,
and caller scalar parameters but not those architectural couplings. Attributes store the original
binary64 parameters once. A later prepared backend may derive its required FLOAT32/BFLOAT16
representation under an explicit numerical contract. Compiler optimization owns rewrites,
compiler autograd owns gradient expansion, and backend conformance owns execution edge behavior.

Unlike legacy construction, `clamp(min, max)` records one `CLAMP` operation and one output Tensor.
This preserves the user's semantic request without exposing a lowering decision or unnecessary
intermediate identity. Later compiler/backend work may lower it into equivalent primitive
operations when appropriate.

## Architecture constraints

- `Tensor` remains public mutable API state and must not become an IR node.
- `TensorScalarExpressions` performs deterministic local model validation and construction only.
  It must not read values/storage, traverse provenance, capture a graph, execute mathematics, or
  inspect backend capability.
- `ScalarElementwiseKind` and its typed immutable attributes own backend-independent semantics.
  The helper must use only the valid pairings defined by task 0014E.
- `applyScalar` accepts only the four one-value kinds. It rejects `CLAMP` locally rather than
  constructing an incompatible `Operation`; this is a narrow helper invariant, not a generic
  registry or change to `Operation`.
- `applyClamp` always constructs exact `CLAMP` with `ClampRangeAttrs` and exposes no caller-selected
  kind.
- Every scalar/range parameter is retained as the original Java binary64 primitive. Expression
  construction does not consult input type to produce a cached or quantized value.
- Result data type and Shape are the exact input descriptor values. The result descriptor has
  `Optional.empty()` layout even for a resolved, fully static input.
- Result `requiresGrad` equals the input descriptor flag for all five methods. This is eligibility
  metadata only, not a promise of a gradient rule or differentiable execution.
- Result identity comes only from `TensorFactory.createDerived`. No second allocator, registry,
  cache, interning table, or service is introduced.
- Every successful call returns a storage-free Tensor with exactly one provenance input. It must
  not attach or alias input storage.
- `CLAMP` range ordering uses the completed `ClampRangeAttrs` constructor. The helper must not
  duplicate, weaken, translate, or normalize that validation.
- No valid call is canonicalized, including identity/infinite bounds, special powers, repeated
  clamps, or multiply by zero/one/minus-one.
- Package direction is `model.tensor -> model.operation.elementwise.scalar`, plus existing
  `model.tensor -> model.operation` and `model.datatype`. The operation package must not import
  Tensor and no package cycle may be introduced.
- Stop if implementation requires an existing contract change, scalar quantization, storage/value
  access, graph/compiler/autograd behavior, another dependency, or architecture decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor expression methods, local result
  descriptor construction, provenance, and derived Tensor creation.
- `io.github.pho001.synaptik.model.operation` — supplies immutable generic `Operation`.
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — supplies exact scalar/clamp
  kinds and attributes.
- `io.github.pho001.synaptik.model.datatype` — supplies the existing immutable `DataType` category
  query used for floating validation.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the five exact overloads and delegates shared behavior.
- `io.github.pho001.synaptik.model.tensor.TensorScalarExpressions` — package-private stateless
  construction boundary colocated with Tensor, descriptors, provenance, and the derived factory
  seam.
- `TensorScalarElementwiseTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor mul(double scalar)
public Tensor pow(double exponent)
public Tensor clamp(double minValue, double maxValue)
public Tensor clampMin(double minValue)
public Tensor clampMax(double maxValue)
```

`mul(double)` and `pow(double)` overload the existing Tensor-valued methods without changing them.
Each public method delegates exactly once and returns the helper's exact result:

| Tensor method | Helper call | Kind | Attributes |
|---|---|---|---|
| `mul(scalar)` | `applyScalar(this, MUL, scalar)` | `MUL` | `ScalarValueAttrs(scalar)` |
| `pow(exponent)` | `applyScalar(this, POW, exponent)` | `POW` | `ScalarValueAttrs(exponent)` |
| `clamp(min, max)` | `applyClamp(this, min, max)` | `CLAMP` | `ClampRangeAttrs(min, max)` |
| `clampMin(min)` | `applyScalar(this, CLAMP_MIN, min)` | `CLAMP_MIN` | `ScalarValueAttrs(min)` |
| `clampMax(max)` | `applyScalar(this, CLAMP_MAX, max)` | `CLAMP_MAX` | `ScalarValueAttrs(max)` |

No public method performs separate validation, attributes/descriptor/provenance construction,
canonicalization, storage access, or exception translation. No other overload is added.

### Package-private helper shape

Create exactly one package-private final non-record class:

```java
final class TensorScalarExpressions {
    private TensorScalarExpressions() {
    }

    static Tensor applyScalar(
            Tensor input,
            ScalarElementwiseKind kind,
            double value) {
        // exact contract below
    }

    static Tensor applyClamp(Tensor input, double minValue, double maxValue) {
        // exact contract below
    }

    private static Tensor create(
            Tensor input,
            DataType dataType,
            Operation operation) {
        // shared descriptor/provenance/factory path below
    }
}
```

The helper has no fields, nested types, public/protected members, overloads, cache, registry,
service, or caller-kind range method. Its constructor prevents instantiation. The two package-
private entry points make the one-value/range distinction explicit, while the private method
shares only construction after validation and typed operation creation.

### Scalar validation and construction order

`applyScalar` performs these steps exactly:

1. require non-null `input` then `kind`, with messages `input` and `kind`;
2. reject `ScalarElementwiseKind.CLAMP` with
   `IllegalArgumentException("CLAMP requires ClampRangeAttrs")`;
3. read the input descriptor's exact `DataType` and reject it when `isFloating()` is false with
   `IllegalArgumentException("input must be a floating data type, but was " + dataType)`;
4. create exactly one `ScalarValueAttrs(value)`, preserving the supplied primitive unchanged;
5. create exactly one `Operation(kind, attrs)`; and
6. delegate exactly once to private `create(input, dataType, operation)`.

The fixed enum currently makes `CLAMP` the only invalid one-value kind. Do not introduce a set,
map, switch registry, reflection, or generic family validator.

### Range validation and construction order

`applyClamp` performs these steps exactly:

1. require non-null `input` with message `input`;
2. read and validate the exact floating input `DataType` with the same failure as `applyScalar`;
3. create exactly one `ClampRangeAttrs(minValue, maxValue)`, allowing its exact strict-inversion
   validation and message to propagate;
4. create exactly one `Operation(ScalarElementwiseKind.CLAMP, attrs)`; and
5. delegate exactly once to private `create(input, dataType, operation)`.

Input type validation intentionally precedes range validation. A non-floating input with inverted
bounds reports the unsupported data type and constructs no range attributes or Tensor identity.
Do not catch or translate the completed attribute contract's failure.

### Shared derived construction

Private `create` performs exactly:

1. create one `TensorDescriptor` from the validated exact input data type, exact input Shape
   reference, `Optional.empty()` layout, and unchanged input `requiresGrad` flag;
2. create one `TensorProvenance(operation, List.of(input))`; and
3. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

All public/helper validation and immutable value construction occurs before ID allocation.
Failures consume no Tensor ID. Identity exhaustion occurs only through `createDerived` after local
model values have been constructed. Do not add a production ID-inspection hook.

### Parameter preservation

Every successful expression stores the caller's exact `double` in its attributes without casting
through `float`, BFLOAT16 conversion, normalization, or special-value replacement. Tests compare
raw bits for finite values, signed zero, infinities, and multiple NaN payloads. Record equality
still follows the standard attribute contract and is not raw-payload identity.

For a FLOAT32 or BFLOAT16 input, retaining binary64 parameters does not mean the eventual backend
calculates a wider output. It means the model preserves the semantic literal until a prepared
backend applies the later-defined input-type conversion and numerical contract.

### Result descriptor, provenance, and identity

Every result:

- retains the exact input `DataType` and exact immutable Shape reference;
- has empty layout for scalar, zero-sized, static, dynamic, and previously resolved inputs;
- retains the exact input `requiresGrad` value;
- has a fresh factory-assigned `TensorId`, empty label, and no host storage;
- contains the exact typed kind and immutable attributes described above; and
- has immutable provenance containing exactly the exact input reference.

Repeated equal calls are not interned. Chaining retains the immediately preceding result as the
next exact input. `clamp` creates one Tensor and one `CLAMP` provenance operation, not observable
`CLAMP_MIN`/`CLAMP_MAX` intermediates.

### No eager canonicalization or numerical inspection

Every valid call creates the requested expression even for zero, one, minus one, infinities, NaN,
signed zero, special exponents, equal bounds, unbounded ranges, repeated/nested clamps, or input
storage that would permit a rewrite. The helper must not read storage or provenance to return an
input, create eager constants, substitute parameterless unary/binary kinds, combine nested
parameters, or split range clamp. Compiler optimization may later perform valid rewrites over
immutable graph semantics.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScalarExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScalarElementwiseTest.java`

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
- Existing `DataType`, `Shape`, `TensorDescriptor`, `TensorProvenance`, `TensorFactory`,
  `Operation`, `ScalarElementwiseKind`, `ScalarValueAttrs`, and `ClampRangeAttrs` Javadocs/tests.
- Binary/unary expression helpers and tests, focused architecture documents, ADRs, architecture
  tests, backend-conformance tests, and integration tests.

## Maximum scope

At most two production files, two test files, and six documentation/planning files: ten paths
total. `docs/api/compile-api.md` is included from the start solely to keep its current public-Tensor
input status accurate; it must not add compiler behavior.

`Tensor.java` and `TensorTest.java` may change only for the five exact overloads, their Javadocs,
exact public API-shape expectations, and non-synchronization assertions. Do not change fields,
constructor, metadata/storage behavior, existing expressions, equality, hashing, diagnostics, or
unrelated tests.

If implementation needs another production/test concept, an existing foundation change, scalar
quantization, storage/value access, compiler/autograd behavior, another documentation file, or more
than ten paths, stop and propose a follow-up or architecture decision. Do not create task 0015.

## Javadoc requirements

- Update Tensor type Javadoc only as needed to include parameterized storage-free expressions
  without making Tensor an IR node or executable value.
- Every public method must explain its mathematical roles, floating-only input, exact binary64
  parameter preservation, exact type/shape and gradient-eligibility retention, unresolved result
  layout, fresh identity, storage absence, exact kind/attributes, one-input provenance, and
  deferral of scalar conversion, numerical execution, rewrites, and gradients.
- Document every primitive parameter with `@param`, the fresh derived Tensor with `@return`, and
  non-floating/range/identity-exhaustion failures with precise `@throws` tags.
- `clamp` must document inclusive lower/upper order, strict inversion failure, first-class single
  operation, and accepted NaN/infinity/signed-zero representation without promising execution
  behavior.
- Document the package-private helper, both package-private methods, private construction method,
  and private constructor with validation order, ownership, parameter retention, side effects,
  failures, and boundaries.
- Avoid five unexplained copies while keeping each public method complete in generated Javadoc.
- Review the completed scalar semantics and reused foundation Javadocs and record why they remain
  accurate, or stop on an out-of-scope inconsistency.

## Acceptance criteria

- Tensor declares exactly five new public overloads with the specified primitive signatures and
  Tensor return type; no other public API is added.
- Existing `mul(Tensor)` and `pow(Tensor)` remain unchanged and unambiguous.
- Every method maps to the exact kind/attributes and delegates once to the matching helper entry.
- `TensorScalarExpressions` has exactly the specified visibility, finality, constructor, two
  package-private methods, one private method, and zero-field/nested-type surface.
- Helper null/kind/type/range validation order, exception types, and messages are exact.
- All three floating data types are accepted; integral and boolean inputs fail before attributes or
  identity allocation.
- Every scalar/range parameter raw bit pattern is preserved unchanged for every floating input;
  no input-dependent conversion occurs.
- Scalar, zero-sized, static unresolved/resolved, and dynamic input shapes retain the exact Shape
  reference while result layout becomes empty.
- Result `requiresGrad` equals the input flag for all five methods.
- Every result is fresh, unlabeled, storage-free, and contains exact one-input provenance with the
  documented typed operation/attributes.
- `clamp` produces one `CLAMP` result and no observable intermediate Tensor.
- Special multipliers/exponents/bounds, repeated calls, and nested chains are not interned,
  canonicalized, split, collapsed, or evaluated.
- Input metadata, provenance, label, storage association, and storage contents remain unchanged.
- No numerical execution, gradient rules, graph/compiler state, backend facts, dependency, or
  architecture change is added.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/bytecode/scope checks,
  documentation links/formatting/example, and status synchronization pass.
- A separate clean-context documentation-focused agent finalizes Javadocs, Tensor API, Compile API
  status, glossary, task evidence, master plan, and roadmap and records related no-change
  conclusions.
- Task 0014F becomes Complete only after both passes. Task 0015 remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorScalarElementwiseTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and shape;
- exact five public overload descriptors, mappings, delegation, and unchanged Tensor overloads;
- exact `Operation` kind and attributes types/references for every method;
- raw-bit parameter preservation for ordinary finite values, infinities, signed zeros, and
  multiple NaN payloads across every floating input type;
- one first-class range operation with ordered exact bounds and no intermediate provenance;
- scalar, zero-sized, ordinary static, resolved-layout, and dynamic shapes;
- unchanged data type/Shape reference/gradient eligibility, empty layout/label/storage, exact input
  provenance, and fresh identity;
- null helper inputs/kinds, invalid scalar kind, all non-floating inputs, inverted clamp range,
  exact messages/order, and no ID consumption;
- zero/one/minus-one multiplication, special powers, infinite/NaN/equal/signed-zero bounds,
  repeated/nested clamps, and absence of canonicalization/domain inspection; and
- preservation of input metadata, provenance, labels, storage association, and contents.

Manually inspect `javap -p -c -s` and reflection for exact Tensor overload descriptors, helper
surface, one helper delegation per public method, validation order, raw value passage, exact
kind/attributes construction, and one factory call. Scan imports/dependencies for forbidden layers.
Confirm absence of scalar conversion, storage access, graph IDs, compiler/runtime/backend types,
gradient rules, canonicalization, registries, caches, and services. Validate generated Javadoc,
Tensor/Compile API current-versus-planned wording, newcomer example, glossary, links/anchors/
fences/whitespace, exact ten-path scope, synchronized statuses, and absence of a task-0015
specification.

## Dependencies

- Task 0001 supplies floating data-type classification.
- Task 0002 supplies immutable static/dynamic Shape values.
- Task 0007 supplies immutable result descriptors.
- Task 0011 supplies public Tensor state.
- Tasks 0012 and 0013 supply factory identity allocation, `createDerived`, and immutable
  provenance.
- Task 0014E supplies exact scalar/clamp kinds, one-value attributes, and range attributes.

## Follow-up tasks

- Task 0015 remains Draft for comparison, logical, selection, and cast semantics/expressions.
- Compiler optimization will later own scalar and clamp canonicalization.
- Compiler autograd will later own scalar-power and clamp gradient rules/boundary conventions.
- Prepare/backend/conformance work will later own scalar conversion and numerical execution edge
  behavior.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public Tensor expression semantics,
backend-independent Operation values, immutable attributes, descriptors, and provenance to
`modules/model`, while compiler, autograd, backend support, preparation, and execution remain in
their owning layers.

If implementation requires Tensor to become IR, scalar conversion policy in model state, resolved
physical layout, storage access, compiler/autograd behavior, backend facts, another dependency, or
an architecture change, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0006/0007/0011/0012/0013/0014D/0014E/0014F,
Tensor API, Compile API, Training API, glossary, current DataType/Shape/TensorDescriptor/Tensor/
TensorFactory/TensorProvenance/Operation/ScalarElementwiseKind/ScalarValueAttrs/ClampRangeAttrs
contracts and tests, and Java 26 Gradle configuration.

Implement task 0014F exactly. Modify Tensor.java and add package-private final
TensorScalarExpressions.java for production. Update TensorTest only for the exact five-overload API
surface and add TensorScalarElementwiseTest. Add exactly mul(double), pow(double),
clamp(double,double), clampMin(double), and clampMax(double), each delegating once to the exact
shared helper entry and semantic kind.

The helper has exactly applyScalar(input, kind, value), applyClamp(input, minValue, maxValue), and
private create(input, dataType, operation). Follow the task's exact null/kind/floating/range/
construction order and messages. Preserve caller double values unchanged, retain exact input data
type and Shape, create empty-layout descriptor with unchanged requiresGrad, construct exact typed
Operation and one-input provenance, and delegate once to TensorFactory.createDerived with no
label/storage. Every valid call is fresh; CLAMP is one first-class operation.

Do not convert scalars, inspect values/storage, execute mathematics, check domains, canonicalize,
split/collapse clamps, define gradients, capture a graph, add overloads, change existing contracts,
or introduce compiler/runtime/backend behavior. Stop beyond ten paths or on architecture
uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/example/link/whitespace/scope/status check. Then hand the actual diff/evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record
related-contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0014F, model master plan, and roadmap only for planning status/evidence. Do not mark
0014F Complete until both passes succeed. Leave 0015 Draft without a specification. Do not commit
or push.
```

## Local decisions

- Five fluent overloads preserve the selected legacy capability while using new immutable typed
  semantics and provenance.
- One package-private helper exposes separate one-value and range entries, preventing invalid
  CLAMP/`ScalarValueAttrs` composition without adding a global compatibility registry.
- Caller `double` parameters remain exact binary64 semantic literals for all floating input types.
  Input-dependent scalar conversion is deferred to prepared backend numerical contracts.
- Range input validates floating type before constructing `ClampRangeAttrs`; this keeps failure
  order deterministic and consumes no identity on either failure.
- `CLAMP` is first-class and creates one public result, avoiding legacy construction-level
  exposure of two lowering primitives.
- Result layout is unresolved even for static resolved inputs because expression construction does
  not decide materialization.
- Gradient eligibility propagates unchanged but does not define a gradient rule.
- No valid call is canonicalized. Typed immutable provenance gives future compiler optimization
  the correct boundary for rewrites.

## Known limitations

- Results have no computed values or host storage and require future compile/prepare/run support.
- Only floating input Tensors are accepted; integral/boolean scalar operations are not provided.
- Scalar conversion and numerical behavior for FLOAT32/BFLOAT16 execution remain undefined here.
- NaN, infinity, signed-zero, domain, overflow, and rounding behavior are represented but not
  executed or guaranteed by this model task.
- No gradient rule or clamp-boundary convention exists.
- Layout remains unresolved and repeated equivalent expressions receive distinct identities.
- Generic `Operation` still does not globally enforce kind-to-attributes compatibility.

## Validation evidence

Planning reviewed architecture/documentation rules; planning guide, capability baseline, master
plan, and roadmap; completed Tensor/provenance and operation tasks through 0014E; current Tensor,
TensorTest, TensorUnaryExpressions, TensorDescriptor, TensorFactory.createDerived,
TensorProvenance, DataType, Shape, Operation, ScalarElementwiseKind, ScalarValueAttrs, and
ClampRangeAttrs source/tests; and the read-only legacy public methods, builders, scalar operation
descriptors, canonicalization tests, CPU/accelerator lowering, and dtype-specific parameter use.

Planning confirmed that legacy execution selects float/double forms during lowering, while the
new model's completed attributes deliberately retain one exact binary64 semantic value. The
public expression can therefore remain backend-independent and requires no new conversion state,
dependency, existing contract change, or architecture change.

Planning validation:

- `git diff --check` passed, and the three changed planning files contain no trailing whitespace.
- The canonical section scan found every required task-specification section.
- The relative Markdown-target scan resolved every local `.md` link in this task, the model master
  plan, and the roadmap.
- Status inspection found task 0014F `Ready` exactly once in this specification, its model-master
  row, and its roadmap row.
- Scope inspection found exactly this new task plus the model master plan and roadmap changed; no
  Java, test, Gradle, AGENTS, architecture, API, glossary, or other module file changed during
  planning.
- No task-0015 specification exists; 0015 remains only a Draft queue entry.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0014d` added the five exact public Tensor
  overloads, package-private `TensorScalarExpressions`, the focused scalar-expression suite, and
  only the required Tensor public-API shape assertions. Independent documentation context
  `/root/implement_model_0014b` then performed the mandatory fresh reread of the architecture,
  focused architecture pages, documentation and planning rules, current APIs, glossary, completed
  task chain, actual diff/source/tests, generated reports, bytecode, and generated Javadoc. It
  applied General plus API/Javadoc style to Java and API reference review, Planning style to
  planning files, and Example format to the new Tensor API example.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorScalarElementwiseTest` passed, and an explicit fresh
  `--rerun-tasks` execution produced 7 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest` passed,
  and an explicit fresh `--rerun-tasks` execution produced 14 tests with zero failures, errors, or
  skips.
- `./gradlew :modules:model:test` passed, and an explicit fresh `--rerun-tasks` execution produced
  40 XML suites with 303 tests and zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` passed. An explicit fresh `--rerun-tasks` execution regenerated
  clean public Javadoc; generated `Tensor.html` contains all five overloads and their floating
  eligibility, exact binary64 attributes, type/shape/gradient retention, unresolved layout, fresh
  storage-free result, semantic kind/provenance, validation-order, and failure contracts. Standard
  public Javadoc omits the package-private helper; direct source review found its type, private
  constructor, both package-static entries, and private `create` contract complete, so the
  documentation pass made no Java edit.
- `./gradlew test` passed for the repository; the final run reported all 36 actionable tasks
  up-to-date and no failing task. The model suite immediately preceding it was freshly executed.
- The complete `ScalarExpressionExample` added to the Tensor API compiled with Java 26 against the
  generated model classes and printed the documented `FLOAT32`, exact-shape, unresolved-layout,
  gradient, label, storage, scalar `MUL`, raw negative-zero multiplier, first-class `CLAMP`, exact
  range bits, sole immediate input, and fresh-result values.
- `javap -p -c -s` confirmed exactly the five requested public descriptors and exactly one matching
  helper delegation per overload. It also confirmed a final zero-field helper with one private
  constructor, package-static `applyScalar` and `applyClamp`, private-static `create`, and no nested
  types. Helper bytecode preserves input/kind/CLAMP/type validation order, constructs exactly one
  typed attributes value and Operation, preserves exact type/Shape/gradient metadata with empty
  layout, records one exact input, and calls `createDerived` once with an empty label.
- Reflection and focused behavioral tests confirm exact API shape, visibility, finality, zero
  fields/nested types, non-synchronization, overload-to-kind mapping, raw-bit preservation for
  finite values, signed zeros, infinities, and NaN payloads, one first-class `CLAMP`, type-before-
  range validation, no identity allocation on failure, no canonicalization, fresh identity, and
  unchanged input metadata/storage/content.
- Import and source scans found only permitted same-module and JDK types and no storage access,
  scalar conversion, graph/compiler/planning/runtime/backend type, gradient rule, shape algebra,
  canonicalization, cache, registry, map, reflection, or service lookup in the helper.
- The local Markdown validator resolved 203 local file targets and heading anchors across the six
  changed documentation/planning files with zero errors. Fence counts are balanced (`78/0`,
  `4/0`, `0/0`, `2/0`, `0/0`, and `8/0` backtick/tilde fences respectively), targeted status and
  terminology scans are clean, and `git diff --check` passed.
- Final scope contains exactly the ten authorized paths: two production files, two test files,
  Tensor API, the pre-authorized Compile API current-status update, glossary, this task, model
  master plan, and roadmap. No Training API, capabilities, architecture/ADR/test, Gradle/build,
  dependency, existing scalar-semantics contract, another module, backend-conformance,
  integration-test, or task-0015 specification path changed.
- Task 0014F is synchronized as Complete in this specification, the model master plan, and the
  roadmap. Task 0015 remains the next Draft frontier without a detailed specification.
- `DataType`, `Shape`, `TensorDescriptor`, `TensorProvenance`, `TensorFactory.createDerived`,
  `Operation`, `ScalarElementwiseKind`, `ScalarValueAttrs`, and `ClampRangeAttrs` remain accurate
  because the helper composes their existing floating-category, immutable-shape,
  unresolved-descriptor, exact-reference, identity-allocation, typed-kind, exact-binary64, and
  range-validation boundaries without changing them. Existing binary/unary expression contracts
  and tests remain accurate and unchanged.
- `capabilities.md` requires no edit because it already selects these five public methods and
  distinguishes public model expression support from compiler/backend/runtime execution. The
  Training API requires no edit because propagating `requiresGrad` is eligibility metadata only;
  no gradient object, rule, autograd, optimizer, or training behavior changed.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, backend-conformance and
  integration tests, and build configuration require no edit because the implementation remains
  within model-owned public expression semantics and adds no module boundary, dependency,
  lifecycle, backend behavior, numerical execution, Java toolchain, preview/incubator, or
  end-to-end contract.

## Implementation notes

- Added exactly five public Tensor overloads, each delegating once to the matching scalar kind or
  range entry through one stateless package-private helper.
- The helper performs the specified null, kind, CLAMP-pairing, floating, and range validation;
  retains exact caller binary64 attributes; preserves exact input data type, Shape, and gradient
  eligibility in one unresolved descriptor; records exact one-input provenance; and delegates
  once to derived construction without value or storage access.
- Range clamp remains one `CLAMP` operation, and every valid call creates a fresh unlabeled,
  storage-free Tensor without conversion, decomposition, interning, or canonicalization.
- The independent documentation pass found all affected Javadocs complete; updated Tensor API,
  Compile API current-status wording, and existing glossary distinctions; and added a complete
  compiled scalar-expression example. No Java declaration, executable logic, or test changed
  during the documentation pass.

## Completion summary

- Completed changes: Implemented and documented all five floating scalar arithmetic and clamp
  Tensor expression methods with exact binary64 attributes and one-input provenance.
- Files changed or created: Exactly two production files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused scalar/clamp 7/7, Tensor 14/14, all 303 model tests across 40
  suites, regenerated model Javadoc, root tests, bytecode/reflection/import/absence checks, the
  compiled exact example, 203 local links/anchors, fences, terminology, whitespace, exact
  scope/status, and `git diff --check` passed.
- Documentation-agent review: Clean documentation context `/root/implement_model_0014b` completed
  the independent pass using General, API/Javadoc, Planning, and Example-format guidance.
- Documentation impact: Tensor API and glossary now describe current scalar expressions; the
  pre-authorized Compile API correction recognizes them without claiming compiler behavior.
- Javadoc review: Tensor type/all five methods and the helper type/constructor/methods are complete
  unchanged during the documentation pass; reused data type, shape, descriptor, provenance,
  factory, Operation, scalar kind/attribute, and binary/unary expression contracts remain accurate.
- Glossary impact: Existing implementation-status, `OperationKind`, Tensor, and common distinction
  entries now include scalar expression construction; no new reusable domain term was needed.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0014F. Task 0015 remains Draft without a specification.

Status: Complete
