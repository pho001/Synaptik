# Task 0019: Matmul Semantics and Tensor Expression

## Status

Complete

## Goal

Add one cohesive backend-independent model contract for vector, matrix, and batched matrix
multiplication. The task introduces the first-class `MATMUL` semantic kind and one public
`Tensor.matmul(Tensor right)` expression while preserving exact shape facts, selected numerical
meaning, and current producer/provenance behavior.

This task records storage-free metadata only. It does not multiply values, choose a matrix
kernel, construct gradients, capture or optimize a graph, or claim compiler, backend, runtime, or
end-to-end support.

## Why this is the first split frontier

The former broad task 0019 combined three independently reviewable contracts. MATMUL is the
primitive needed by both later linear and attention work and fits one model task. `linear` is an
explicit convenience composition whose weight and bias contract deserves separate public-API
review. Scaled dot-product attention is a named high-level semantic operation with distinct
masking, causal, scale, and numerical policies. Combining all three would exceed normal planning
granularity and make attention's later decisions block the primitive.

Task 0019 therefore owns MATMUL only. Tasks 0019D and 0019E remain Draft follow-ups without
detailed specifications. Existing tasks 0019A–0019C keep their stable identifiers and order.

## Mental model and examples

MATMUL treats the last two axes of each rank-two-or-greater operand as matrix axes. A rank-one
left operand is temporarily treated as `[1, K]` and that inserted result axis is removed. A
rank-one right operand is temporarily treated as `[K, 1]` and that inserted result axis is
removed. Leading axes are batch axes and broadcast right-aligned.

```text
[K]       @ [K]          -> []
[M, K]    @ [K]          -> [M]
[K]       @ [K, N]       -> [N]
[M, K]    @ [K, N]       -> [M, N]
[2, M, K] @ [1, K, N]    -> [2, M, N]
[K]       @ [B, K, N]    -> [B, N]
[B, M, K] @ [K]          -> [B, M]
```

For concrete rank-two values:

```text
[[1, 2, 3],      [[1, 2],       [[22, 28],
 [4, 5, 6]]   @   [3, 4],    =   [49, 64]]
                  [5, 6]]
```

Each output is the dot product of one left row and one right column. The example defines
mathematical meaning; this model task does not evaluate it.

## Scope

- Add exactly `MatmulKind.MATMUL` in a new linear-algebra operation package.
- Pair it only with `NoOperationAttrs.INSTANCE` and a fixed two-input, one-output signature.
- Add exactly `public Tensor matmul(Tensor right)`.
- Add one package-private, field-free `TensorMatmulExpressions` construction helper.
- Accept rank-one and higher operands under the exact vector/matrix/batch rules below.
- Derive exact static, named-symbolic, and expression-dimension result Shapes without runtime
  binding or graph-wide constraint solving.
- Accept same-category floating or signed-integral pairs through `DataTypePromotion.promoteNumeric`;
  reject BOOL and cross-category pairs without implicit casts.
- Fix result, accumulation, overflow, empty-contraction, and determinism policies below.
- Construct one fresh unresolved-layout result with exact ordered producer inputs and no label or
  storage.
- Add focused semantic and public-expression tests.
- Finalize Javadocs, Tensor/Compile API references, glossary, capabilities, and planning records
  through a mandatory separate clean-context documentation pass.

## Out of scope

- `linear`, scaled dot-product attention, attention weights, masks, causal behavior, dropout, RNG,
  sorting, top-K, activation, embedding, convolution, pooling, normalization, or losses
- transpose flags, alpha/beta coefficients, output data-type overrides, accumulator options,
  sparse/quantized/complex/unsigned values, FLOAT16, or general equation notation
- eager value reads, host-storage allocation, result storage, resolved result layout, or mutation
- compiler capture, graph-wide shape constraints, canonicalization, decomposition, fusion,
  autograd, gradient formulas, or backward operations
- capability providers, backend lowering, Basic Linear Algebra Subprograms (BLAS), kernels,
  tolerances, runtime execution, conformance, or integration support
- changes to `DataType`, `DataTypePromotion`, `Dimension`, `DimensionExpressions`, `Shape`,
  `ShapeBroadcast`, `TensorDescriptor`, producer/provenance/factory contracts, Gradle, another
  module, `ARCHITECTURE.md`, or focused architecture documentation

## Exact semantic and API contract

### Kind, attributes, and signature

Create exactly:

```java
public enum MatmulKind implements OperationKind {
    MATMUL
}
```

`MATMUL` declares one stable signature:

```java
OperationSignature.fixed(NoOperationAttrs.class, 2, 1)
```

It accepts only `NoOperationAttrs.INSTANCE`. Add no matmul attributes, transpose variants, aliases,
string dispatch, registry, backend support, or execution metadata.

Add exactly this public method:

```java
public Tensor matmul(Tensor right)
```

The receiver is the ordered left operand. The exact producer input order is `[this, right]`.

### Rank and matrix-axis rules

Both operands must have rank at least one. Define:

- left contraction axis: the final left axis;
- right contraction axis: the final right axis for rank one, otherwise the penultimate right axis;
- left row axis `M`: absent for rank one, otherwise the penultimate left axis;
- right column axis `N`: absent for rank one, otherwise the final right axis;
- left batch prefix: every left axis before its final two axes, or empty for rank one;
- right batch prefix: every right axis before its final two axes, or empty for rank one.

Broadcast the batch prefixes right-aligned. Form the output in this order:

1. broadcast batch dimensions;
2. append `M` when the left operand is not rank one;
3. append `N` when the right operand is not rank one.

Two vectors therefore produce `Shape.scalar()`. Every retained `M`, `N`, equal batch, singleton-
expanded opposing, and unpaired leading `Dimension` uses the exact input reference.

### Inner-dimension compatibility

If both contraction dimensions are static, their sizes must be equal. Reject a mismatch before
batch processing. Zero is valid when both are statically zero and denotes an empty contraction.

If either contraction dimension is unresolved, accept the request unless a static mismatch is
already proven. Structurally equal named or expression dimensions are accepted directly. Unequal
named, expression, static/unresolved, or unresolved/unresolved pairs carry a deferred equality
obligation: compiler shape validation or later binding must prove equal concrete sizes before
execution. No new constraint object is added because the contraction extent does not occur in the
result Shape and both exact input descriptors remain available through ordered provenance.

### Batch broadcasting and deferred obligations

For each aligned batch pair:

- structurally equal dimensions retain the exact left reference;
- a static singleton retains the exact opposing reference;
- an unpaired leading dimension retains its exact input reference;
- two unequal static non-singletons are rejected;
- one unresolved dimension paired with a static non-singleton accepts the static dimension as the
  exact result extent and defers the obligation that the unresolved extent bind to either one or
  that static size; and
- two unequal unresolved dimensions are rejected because the current model cannot derive one
  exact result extent without inventing a non-deterministic unknown dimension.

This is deliberately local MATMUL algebra. Do not broaden or change `ShapeBroadcast`.

Examples using `N`, `B`, and `C` as distinct named dimensions:

```text
[N, M, K] @ [1, K, P] -> [N, M, P]
[N, M, K] @ [4, K, P] -> [4, M, P] with deferred N in {1, 4}
[B, M, K] @ [B, K, P] -> [B, M, P]
[B, M, K] @ [C, K, P] -> rejected: exact batch result is not locally derivable
```

The same structural rules apply to `ExpressionDimension` values.

### Data type, result, and numerical policy

Use `DataTypePromotion.promoteNumeric(leftType, rightType)` unchanged:

- BFLOAT16, FLOAT32, and FLOAT64 pairs promote through the current floating hierarchy;
- INT32 and INT64 pairs promote through the current signed-integral hierarchy;
- floating/integral pairs and every BOOL pair are rejected; and
- no cast producer is inserted.

The promoted type is the result type. Floating accumulation uses FLOAT64 for a FLOAT64 result,
FLOAT32 for a FLOAT32 result, and FLOAT32 for a BFLOAT16 result followed by conversion to
BFLOAT16 output. The portable target is the sum of pairwise products over the contraction extent.
Backends may reassociate terms and use fused multiply-add, so bitwise equality, one traversal
order, and cross-backend identical rounding are not promised; later backend conformance must set
per-type error tolerances. IEEE-754 NaN, infinity, and signed-zero behavior follows the selected
floating multiply/add operations. An empty floating contraction produces positive zero.

Integral multiplication and accumulation mean the exact sum of products modulo `2^32` or `2^64`
in the promoted signed two's-complement result type. No widening, saturation, overflow exception,
or undefined overflow is selected. Modular reassociation is permitted and exact. An empty
integral contraction produces zero.

These are semantic policies, not a kernel, instruction, BLAS, loop, or storage design.

### Result metadata and provenance

Every successful call creates:

- one fresh `TensorDescriptor` with promoted result type, exact derived Shape,
  `Optional.empty()` layout, and `requiresGrad` equal to the logical OR of both inputs;
- one `Operation(MATMUL, NoOperationAttrs.INSTANCE)`;
- one fresh single-output `TensorProducer` retaining exact ordered inputs `[left, right]` and the
  exact result descriptor;
- one fresh unlabeled, storage-free Tensor with output index zero and one newly allocated ID.

The inputs, their descriptors, labels, storage, provenance, and IDs remain unchanged. Repeated
valid calls create distinct producer and Tensor identities even when operands are identical.

### Validation order, failures, and ID effects

`TensorMatmulExpressions.apply(left, right)` performs this exact order:

1. null-check `left`, then `right`, with the parameter name as message;
2. call `DataTypePromotion.promoteNumeric` on descriptor types;
3. require left rank at least one, then right rank at least one;
4. reject a proven static contraction mismatch;
5. process aligned batch dimensions from leading result batch axis to trailing;
6. build the result Shape, descriptor, and operation; and
7. delegate exactly once to `TensorFactory.createDerived`.

Use exact task-owned messages:

```text
left rank must be at least 1: 0
right rank must be at least 1: 0
matmul inner dimensions must match: left=<dimension>, right=<dimension>
cannot derive exact matmul batch dimension at result batch axis <axis>: left=<dimension>, right=<dimension>
cannot broadcast matmul batch dimensions at result batch axis <axis>: left=<dimension>, right=<dimension>
```

The first batch message is for unequal unresolved dimensions. The second is for unequal static
non-singletons. Existing `DataTypePromotion` messages remain unchanged.

Every failure through step 6 consumes no Tensor ID. Only the final factory delegation may allocate
an ID; exhaustion keeps the existing exact message and behavior. Tests must inspect the shared ID
counter only for this concrete no-ID-on-local-failure risk.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Planning guide](../../../planning-guide.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)

## Architecture constraints

- Work remains entirely inside `modules/model` plus its documentation/planning records.
- `Tensor` remains public mutable API state and is not graph IR.
- `Operation` and MATMUL record backend-independent meaning only and expose no support or route.
- Package direction remains `model.tensor -> model.operation.linalg`, datatype, and shape; the
  operation package must not import `Tensor` or graph state.
- Compiler later owns capture, graph-wide revalidation, symbolic binding, canonicalization, and
  gradients. Backend prepare owns lowering, algorithms, fusion, and kernel/BLAS route choice.
- No architecture, dependency, lifecycle, module-boundary, Gradle, or cross-module change.

## Legacy evidence

The read-only `legacy/pre-rewrite` branch confirms workloads for rank-two and batched MATMUL,
linear projections, attention products, BFLOAT16/FLOAT32/FLOAT64 execution, and backend route
selection. It also mixed front-end validation with backend result-type decisions and contains
architecture, package, lowering, execution, and optimization structures that are not carried
forward. It is test and workload evidence only, not design authority. This task is implemented
from current contracts.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.operation`

Package added:

- `io.github.pho001.synaptik.model.operation.linalg` — public backend-independent linear-algebra
  semantic kinds; no Tensor, compiler, backend, or execution behavior.

Type placement:

- `io.github.pho001.synaptik.model.operation.linalg.MatmulKind` — public typed MATMUL identity and
  family-owned fixed signature.
- `io.github.pho001.synaptik.model.tensor.TensorMatmulExpressions` — package-private local shape,
  type, descriptor, operation, and provenance construction boundary.
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public fluent API owner.

Tests mirror the production package whose package-private behavior they inspect.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/linalg/MatmulKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/linalg/MatmulSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — update only
  the exact public Tensor API inventory and method count for `matmul`.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java` —
  update only the shared exact public Tensor method count.

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless the implementation makes an existing claim inaccurate: Training API;
DataType/Shape/TensorDescriptor/producer/provenance/factory and related expression contracts;
architecture/ADRs/tests; conformance/integration; Gradle; other modules.

## Maximum scope

At most three production, four test, and seven documentation/planning files: exactly 14 paths.
`Tensor.java` changes only for the exact method, imports, and its Javadoc. The task adds no
attribute type and changes no existing source or test contract except the two explicitly listed
public API inventory/count updates required by the new method. Stop for a fifteenth path, another
production/test type, any other existing test edit, cross-module work, or architecture change.

## Javadoc and documentation requirements

- Document MATMUL's vector promotion/removal mental model, exact matrix axes, batch broadcasting,
  contraction, dynamic obligations, types, numerical policies, metadata, provenance, and layer
  boundaries.
- Document helper validation order, exact failures, construction order, and ID effects.
- Document `Tensor.matmul` parameters, result, Shape/type/gradient/layout/storage/provenance,
  freshness, failures, deferred constraints, and lack of execution/gradient/backend support.
- Update Tensor API with the full rank/Shape table and at least one concrete mathematical example.
- Update Compile API only to keep current model-expression versus planned capture boundaries
  accurate; do not claim compiler support.
- Add or revise glossary terms only when reusable terminology is introduced. At minimum review
  MATMUL, contraction dimension, batch dimension, and matrix multiplication.
- Keep capabilities, master plan, task, and roadmap synchronized. Record reasoned no-change
  conclusions for reviewed documents and contracts.

## Acceptance criteria

- Exactly one `MATMUL` kind exists with exact `NoOperationAttrs`, two-input, one-output signature.
- Exactly one public `matmul(Tensor)` method is added; the public Tensor method count becomes 157.
- Rank-one through batched combinations derive every specified output and preserve exact retained
  Dimension references.
- Static and symbolic/expression inner and batch compatibility follows the exact immediate versus
  deferred policy without changing `ShapeBroadcast` or inventing result constraints.
- All floating and integral ordered width pairs promote and record the selected result and
  accumulation/overflow policies; BOOL and cross-category pairs fail.
- Validation order, exact task-owned messages, and no-ID local failures match the specification.
- Results are fresh, storage-free, unlabeled, layout-unresolved, use gradient-request OR, and have
  exact operation/input/descriptor/output-index provenance; inputs remain unchanged.
- Tests validate contracts and metadata only; they do not pretend to execute MATMUL values.
- No linear, attention, dropout, gradient, compiler, backend, runtime, registry, dependency,
  Gradle, architecture, or other-module work lands.
- Focused tests, exactly one final model suite after Java stability, Javadoc, documentation, link,
  exact 14-path scope, status, formatting, and whitespace validation pass.
- A separate clean-context documentation pass finalizes all authorized documentation and records
  reused Java evidence plus no-change conclusions.
- Task 0019 becomes Complete only after all evidence is recorded; 0018V remains Complete; 0019D,
  0019E, existing 0019A–0019C, and every later task remain Draft without detailed specs.

## Tests / validation

Required focused command:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.linalg.MatmulSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.shape.ShapeBroadcastTest
```

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

The focused tests cover exact enum/signature/helper/public surfaces; all rank families; static,
named, expression, singleton, zero, and deferred Shape cases; type pairs; validation order/messages
and ID effects; descriptor/provenance/freshness; and the numerical contract as metadata without
eager evaluation.

The separate documentation pass runs after final Javadoc edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links and anchors, balanced fences, terminology, generated Javadoc,
final newlines, trailing whitespace, exact 14 paths, package placement, public method count,
synchronized statuses, one Ready frontier, 0018V Complete, all follow-ups Draft, and no unintended
later detailed specification. Manual reflection/source checks are justified only when the focused
automated surface tests do not cover a concrete risk.

Repository-wide validation is deferred to the selected-modern-operations checkpoint after task
0022 or CI. This single-module task changes no dependency, architecture boundary, shared Gradle
contract, or other module, and must not duplicate the completed 0018V repository checkpoint.

## Dependencies

- 0001–0002 and 0018M: current DataType promotion, Dimensions, Shapes, and symbolic extents.
- 0005–0007, 0011–0013, and 0018K–0018L: operation signatures, descriptors, Tensor, IDs, and
  shared producer/provenance construction.
- 0018N: exact typed-value boundary to preserve unchanged.
- 0018T, 0018U, and 0018V: selected same-category promotion and numerical-policy conventions.

## Follow-up tasks

- 0019D remains Draft for the exact `linear` convenience composition over MATMUL, transpose, and
  optional bias addition.
- 0019E remains Draft for one-output scaled dot-product attention without dropout or public
  attention-weight output.
- Compiler later owns capture, deferred constraint proof, canonicalization, and autograd.
- Backend/conformance/runtime/integration work later owns algorithms, tolerances, lowering,
  kernels, storage, and execution.

Do not create another detailed task specification during task 0019 implementation.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, dependency, lifecycle, focused-architecture,
cross-module, or scope change, stop and report the conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, current architecture,
documentation/planning rules, roadmap, model capabilities/master plan, completed foundation tasks
0001–0003, 0005–0008, 0013, relevant expression/softmax/reduction tasks, reset tasks 0018K–0018V,
and task 0019.

Implement docs/planning/modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md exactly
inside its 14 authorized paths. The two added existing test paths may change only for the exact
public Tensor API inventory/count from 156 to 157 and the one new matmul method as applicable.
Preserve current contracts. Stop on architecture uncertainty, scope overflow, another
type/test/document need, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and exact Java evidence to a separate clean-context documentation agent in the same overall
change. That agent finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and
documentation checks while reusing successful Java evidence. Synchronize status only after all
criteria pass; keep 0018V Complete and every follow-up Draft without detailed specs.
```

## Documentation-agent handoff

Give the separate clean-context documentation agent this task, the complete implementation diff,
exact focused/final model evidence and whether Java changed afterward, all selected Shape/type/
numerical/provenance policies, the seven authorized documentation paths, and required Javadoc,
Markdown, scope, and status validation.

The documentation agent independently reads AGENTS, architecture, documentation rules and the
General/API-Javadoc/Planning/Example profiles, this task, actual source/tests/generated Javadoc,
Tensor/Compile/Training APIs, glossary, capabilities/master/roadmap, and directly related
DataType/Shape/operation/provenance contracts. It finalizes Javadocs and documentation, checks the
mathematical example, and records reasoned no-change conclusions for Training API, foundational
contracts, architecture/ADRs/tests, conformance/integration, Gradle, other modules, and follow-ups.

It does not repeat successful Java tests unless executable Java changes, evidence is stale, or a
concrete recorded risk requires a rerun. It records the clean-context identifier, reused evidence,
files/topics reviewed, commands/results, glossary impact, limitations, and unresolved issues.

## Local decisions

- Kept MATMUL's batch algebra in the field-free package-private helper instead of changing shared
  `ShapeBroadcast`: MATMUL may defer an unresolved-versus-static singleton-or-equal obligation
  only when the exact static output extent is locally derivable.
- Retained exact input Dimension references wherever the result selects an existing row, column,
  equal batch, singleton-opposing, or unpaired batch extent. No new constraint or synthetic
  unknown Dimension is created.
- Used the existing `DataTypePromotion.promoteNumeric` boundary unchanged. Result and accumulator
  policy is documentation-level semantic meaning; model expression construction performs no
  numerical work.
- Used direct helper delegation from `Tensor.matmul` so the package-private helper surface remains
  exactly its two authored methods; this avoids a compiler-generated synthetic helper method that
  would violate the focused surface contract.
- Kept tasks 0019A–0019E and all later work Draft without detailed specifications. Task 0019A is
  the next Draft planning frontier after this completed task.

## Known limitations

- Unequal unresolved contraction dimensions are representable but require later equality proof.
- An unresolved batch dimension can pair with a known non-singleton extent only under a deferred
  singleton-or-equal obligation.
- Two unequal unresolved batch dimensions are rejected because no exact result Dimension can be
  selected locally.
- Model completion does not imply numeric execution or gradient support.

## Validation evidence

- Implementation context `/root/task_0019_implementation` ran the required focused command after
  executable Java stabilized:
  `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.linalg.MatmulSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.shape.ShapeBroadcastTest`.
  It passed 32 tests with zero failures and zero errors.
- The same implementation context then ran exactly one final `./gradlew :modules:model:test`.
  It passed 758 tests across 93 suites with zero failures and zero errors. An earlier focused run
  had exposed a compiler-generated synthetic method; the implementation replaced the method
  reference before both successful recorded runs. Executable Java did not change after them.
- Documentation context `/root/task_0019_implementation/task_0019_documentation` reused that Java
  evidence as required and did not rerun Java tests because it changed only Javadocs and Markdown.
  It independently reviewed all three production files, all four tests, generated Javadoc, Tensor,
  Compile, and Training API references, glossary, model capabilities/master plan, roadmap, and the
  directly related data-type promotion, Shape/Dimension, operation signature, descriptor,
  factory, producer, and provenance contracts.
- Documentation context ran `./gradlew :modules:model:javadoc` after the final Javadoc edits;
  it passed with two executed tasks. Generated pages for `MatmulKind` and `Tensor.matmul` contain
  the final semantic and accumulator text. The package-private helper is intentionally absent from
  public generated Javadoc and was reviewed directly in source.
- A targeted Ruby Markdown checker resolved 505 local links, including 145 heading anchors, across
  all seven changed documentation/planning files with zero errors. Two preliminary checker
  invocations failed in the validation script itself because the installed Ruby lacks
  `filter_map` and because the first heading-slug approximation mishandled slash-separated
  headings; the corrected checker then passed without requiring documentation changes.
- A targeted Ruby formatting check passed for all seven Markdown files: balanced backtick and
  tilde fences, final newlines, and no trailing whitespace. `git diff --check` passed.
- Exact-scope and source/generated-surface checks passed: exactly 14 changed or untracked paths;
  the three production, four test, and seven documentation/planning files match the authorized
  list; both new types match their planned packages; compiled `Tensor` has exactly 157 public
  method entries and no public constructor; and generated Javadoc contains `MatmulKind` and
  `Tensor.matmul`.
- Status checks confirmed task 0018V remains Complete, task 0019 and its master/roadmap rows are
  Complete, and tasks 0019A–0019E plus every later model task remain Draft. No detailed
  0019A–0019E specification exists; task 0019A is the next Draft frontier.
- The matrix example was recalculated exactly as four integer dot products: `22`, `28`, `49`, and
  `64`. Terminology checks found the three new glossary entries and consistent uses in Tensor API
  and Javadoc. A preliminary generated-page inspection used one incorrect output path; inspection
  against the actual generated paths then passed. No product file was changed by these validation
  script corrections.

## Implementation notes

- Added public `MatmulKind.MATMUL` with its sole immutable fixed
  `NoOperationAttrs`/two-input/one-output signature.
- Added the package-private field-free `TensorMatmulExpressions` helper and public
  `Tensor.matmul(Tensor right)`. Construction implements every vector/matrix/batch rank family,
  selected immediate and deferred Shape obligations, same-category numeric promotion, exact
  metadata/provenance, validation order/messages, and no-ID local failure boundary.
- Added focused semantic and expression tests, and changed only the authorized Tensor public API
  inventory/count assertions in the two existing tests. The public Tensor method count is 157.
- Finalized all three production Javadocs. Expanded the Tensor API with the complete rank/Shape
  table, exact dynamic obligations, numeric and metadata policies, and the independently checked
  `[[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]` example. Updated Compile API only
  to distinguish current model expressions from planned capture and validation.
- Added glossary entries for batch dimension, contraction dimension, and matrix
  multiplication/MATMUL. These are reusable terms introduced by the public contract.
- Reviewed Training API with no change: task 0019 adds no gradient rule, trainable role, or
  training workflow. Reviewed foundational datatype, Shape/Dimension, operation,
  TensorDescriptor/factory/producer/provenance contracts with no change because MATMUL composes
  them without altering their behavior. Architecture contract, focused architecture pages, ADRs,
  architecture tests, backend conformance, integration tests, Gradle, other modules, and Draft
  follow-ups also require no change because this task adds model metadata only and changes no
  boundary, dependency, backend/runtime behavior, or cross-module contract.

## Completion summary

- Completed changes: first-class MATMUL semantics, local vector/matrix/batch expression
  construction, public Tensor method, focused tests, complete Javadocs/API reference, glossary,
  capabilities, and synchronized planning records.
- Files changed or created: exactly the three production, four test, and seven
  documentation/planning paths listed under Affected files.
- Tests and validation: reused the successful 32-test focused run and 758-test/93-suite final model
  run from the implementation context; final Javadoc, Markdown, scope, status, and whitespace
  checks are recorded in Validation evidence.
- Documentation-agent review: clean context
  `/root/task_0019_implementation/task_0019_documentation` independently finalized the authorized
  documentation and Javadocs in the same overall change.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, model master plan,
  and roadmap now describe current MATMUL metadata and preserve planned compiler/backend/runtime
  boundaries. Training API and other reviewed areas require no change for the reasons above.
- Javadoc review: all three production paths document meaning, parameters/results, failure
  conditions, dynamic obligations, numerical policy, provenance, ID effects, and unsupported
  layers without claiming execution.
- Glossary impact: added reusable definitions for batch dimension, contraction dimension, and
  matrix multiplication/MATMUL.
- Known limitations: the accepted deferred obligations and metadata-only boundary remain as
  listed above.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
