# Task 0019D: Linear Convenience

## Status

Complete

## Goal

Add the conventional public linear-projection convenience as explicit composition over the
current transpose, MATMUL, and optional ADD expression builders. The receiver is the input,
weights use the common `[outFeatures, inFeatures]` orientation, and an optional bias is exactly
rank one `[outFeatures]`.

This task adds no `LINEAR` semantic kind. A caller can inspect the actual PERMUTE -> MATMUL chain,
and the bias form adds one final ADD producer. The model therefore preserves primitive expression
structure for later compiler inspection while leaving any profitable pattern fusion to backend
prepare.

## Rationale and mental model

A linear projection contracts the input's final feature axis with each row of the weight matrix.
The public weight orientation is convenient for layer parameters, while current MATMUL expects its
right operand in `[inFeatures, outFeatures]` orientation:

```text
input.linear(weight)
  = input.matmul(weight.transpose())

input.linear(weight, bias)
  = input.matmul(weight.transpose()).add(bias)
```

For input Shape `[batch, inFeatures]`, weight Shape `[outFeatures, inFeatures]`, and bias Shape
`[outFeatures]`:

```text
[batch, inFeatures] @ [inFeatures, outFeatures] -> [batch, outFeatures]
                                                    + [outFeatures]
                                                    -> [batch, outFeatures]
```

Rank-one input `[inFeatures]` produces `[outFeatures]`. Higher-rank input
`[..., inFeatures]` preserves every preceding input axis exactly and replaces only the final
feature extent with the exact weight `outFeatures` Dimension.

### Newcomer producer-chain example

The implementation and documentation pass must include a runnable metadata example equivalent to
this input and observation:

```java
Tensor input = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true));
Tensor weight = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(4, 3), Optional.empty(), true));
Tensor bias = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(4), Optional.empty(), true));

Tensor output = input.linear(weight, bias);
TensorProducer add = output.provenance().orElseThrow().producer();
Tensor product = add.inputs().getFirst();
TensorProducer matmul = product.provenance().orElseThrow().producer();
Tensor transposedWeight = matmul.inputs().get(1);
TensorProducer permute = transposedWeight.provenance().orElseThrow().producer();
```

The example must show Shape `[2, 4]`, final ADD inputs `[product, bias]`, MATMUL inputs
`[input, transposedWeight]`, PERMUTE input `[weight]`, permutation axes `[1, 0]`, and output index
zero at every link. It demonstrates storage-free expression structure only; it does not evaluate
numbers, capture a compiled graph, define gradients, or prove backend fusion.

## Scope

- Add exactly `public Tensor linear(Tensor weight)` and
  `public Tensor linear(Tensor weight, Tensor bias)`.
- Add one package-private, final, field-free `TensorLinearExpressions` helper in the tensor
  package.
- Treat the receiver as input and require input rank at least one.
- Require weight to have exact rank two and interpret it as
  `[outFeatures, inFeatures]`.
- Accept the existing same-category floating and signed-integral MATMUL/ADD domains through
  `DataTypePromotion.promoteNumeric`; insert no casts.
- Require optional bias to be non-null, exact rank one, and to have a Dimension structurally equal
  to the weight's exact `outFeatures` Dimension.
- Prevalidate every caller-controlled local semantic failure before creating the transpose or
  consuming any Tensor ID.
- After successful prevalidation, call the existing package-private transpose, MATMUL, and ADD
  helpers directly in exact semantic order.
- Lock exact result Shape/type/gradient/layout/label/storage metadata, producer identities,
  ordered inputs, output indices, wrapper count, and ID allocation order.
- Add focused model tests and update every existing global public-Tensor inventory/count test.
- Finalize public Javadocs, Tensor/Compile API documentation, glossary impact, capabilities, and
  planning records in a mandatory clean-context documentation pass.

## Out of scope

- a `LINEAR` operation kind, attributes, signature, result carrier, producer, compiler pass, or
  graph node
- a layer/module/parameter object, trainable-state ownership, initialization, parameter naming,
  optimizer integration, or serialization format
- scalar, rank-two, or higher-rank bias; bias broadcasting beyond the selected exact rank-one
  `[outFeatures]` contract; absent bias represented by null
- transpose flags on MATMUL, alpha/beta coefficients, output-type/accumulator options, casts,
  quantized/sparse/complex/unsigned tensors, or FLOAT16
- eager value access, storage allocation, result materialization, numeric execution, kernels,
  Basic Linear Algebra Subprograms (BLAS), tolerances, or backend support
- gradient formulas, backward operations, compiler capture, canonicalization, decomposition,
  common-subexpression elimination, fusion, lowering, prepare, runtime, conformance, or integration
- changes to transpose, MATMUL, ADD, `DataTypePromotion`, `Shape`, `ShapeBroadcast`, Dimension,
  descriptor, producer/provenance, factory, operation signatures, or operation kinds
- changes to Gradle, another module, `ARCHITECTURE.md`, focused architecture documentation, ADRs,
  architecture tests, backend conformance, or integration tests

## Exact public API and composition contract

Add exactly:

```java
public Tensor linear(Tensor weight)

public Tensor linear(Tensor weight, Tensor bias)
```

Argument names are part of the source/Javadoc contract. The receiver is called `input` inside the
helper. The two-argument form never treats null bias as absence; callers select the no-bias
overload instead.

After complete prevalidation, construction is exactly:

```java
Tensor transposedWeight = TensorPermutationExpressions.transpose(weight);
Tensor product = TensorMatmulExpressions.apply(input, transposedWeight);
return product; // no-bias overload
```

or:

```java
Tensor transposedWeight = TensorPermutationExpressions.transpose(weight);
Tensor product = TensorMatmulExpressions.apply(input, transposedWeight);
return TensorBinaryExpressions.apply(product, bias, BinaryArithmeticKind.ADD);
```

`TensorLinearExpressions` calls these existing package-private helpers directly. It must not call
the public fluent methods, recurse between the two public overloads, create an optional-bias
container, reproduce primitive producer construction, or add a new shared abstraction. The
existing helpers retain their own defensive validation; complete linear prevalidation guarantees
that no caller-controlled rank, type, contraction, bias, or broadcast failure remains after the
first intermediate ID is allocated.

## Shape and dynamic-dimension contract

Let:

- input Shape be `[..., inFeatures]`, with rank at least one;
- weight Shape be exactly `[outFeatures, weightInFeatures]`; and
- optional bias Shape be exactly `[biasOutFeatures]`.

Input and weight contraction follows current MATMUL compatibility unchanged:

- unequal static `inFeatures` and `weightInFeatures` fail locally;
- structurally equal static, named, expression, or same-identity constrained-unknown Dimensions
  are accepted;
- if either contraction Dimension is unresolved and no unequal static pair is proven, equality is
  deferred to later compiler validation or binding because the contraction extent is absent from
  the result Shape.

The result Shape contains every input Dimension except its final feature Dimension, followed by
the exact weight `outFeatures` Dimension reference. Thus:

```text
[K]       linear [N, K] -> [N]
[B, K]    linear [N, K] -> [B, N]
[B, T, K] linear [N, K] -> [B, T, N]
```

Every preserved leading input Dimension and final weight `outFeatures` Dimension is retained by
exact reference through current MATMUL shape algebra.

Bias is deliberately stricter than general broadcasting. Its sole Dimension must be structurally
equal to the exact weight `outFeatures` Dimension. Equal static sizes satisfy this through normal
Dimension equality. Named dimensions and expression dimensions must have equal structure;
identity-based constrained unknowns must reuse the same identity. An unresolved/static pair or
two structurally unequal unresolved Dimensions is rejected locally rather than creating a
constraint that current `ShapeBroadcast` cannot represent. A singleton bias is accepted only when
`outFeatures` is itself exactly the static singleton Dimension; it is not a general broadcast
escape hatch. This guarantees that the later ordinary ADD succeeds unchanged and produces a Shape
structurally equal to the product Shape with the exact same ordered Dimension references.
Ordinary ADD may construct a distinct outer Shape object.

Do not change `ShapeBroadcast`, synthesize a result Dimension, bind a dynamic extent, or introduce
a constraint object. The only deferred linear obligation is the already selected MATMUL
contraction equality.

## Data type and numerical policy

Do not narrow linear to floating-only. A convenience that is defined as existing primitive
composition retains those primitives' current numeric domains:

- input and weight use `DataTypePromotion.promoteNumeric(inputType, weightType)`;
- the no-bias result uses that promoted MATMUL type;
- the bias form promotes the MATMUL result type with the bias type through
  `DataTypePromotion.promoteNumeric(productType, biasType)`; and
- no cast producer is inserted.

Each promotion accepts only two floating types or two signed-integral types. Width promotion
remains BFLOAT16 < FLOAT32 < FLOAT64 or INT32 < INT64. BOOL and floating/integral category mixing
fail through current promotion messages. Consequently, a wider bias may widen only the final ADD
result type, exactly as literal composition does.

Numerical meaning is inherited, not restated as new linear semantics. MATMUL owns contraction,
accumulation, reassociation, empty-contraction, and integral modular-sum-of-products policy. ADD
owns elementwise promoted arithmetic and modular integral overflow. This task creates no separate
linear accuracy, accumulation, overflow, determinism, special-value, or backend contract.

Only floating descriptors can request gradients under the current model. The convenience does
not make weights trainable, create parameters, or define a gradient rule.

## Validation order, failures, and ID effects

Both helper entry points perform all local validation before calling a primitive construction
helper. Validation order is exact.

For `linear(input, weight)`:

1. null-check `input`, then `weight`, with the parameter name as message;
2. call `DataTypePromotion.promoteNumeric(inputType, weightType)` and retain `productType`;
3. require input rank at least one;
4. require weight rank exactly two; and
5. reject unequal static input-final and weight-axis-one contraction Dimensions.

For `linear(input, weight, bias)`:

1. null-check `input`, then `weight`, then `bias`, with the parameter name as message;
2. call `DataTypePromotion.promoteNumeric(inputType, weightType)` and retain `productType`;
3. require input rank at least one;
4. require weight rank exactly two;
5. reject unequal static input-final and weight-axis-one contraction Dimensions;
6. call `DataTypePromotion.promoteNumeric(productType, biasType)` and retain the final type only
   for validation consistency with ADD;
7. require bias rank exactly one; and
8. require the bias Dimension to equal the weight axis-zero `outFeatures` Dimension.

Use exact task-owned messages:

```text
input rank must be at least 1: <rank>
weight rank must be exactly 2: <rank>
linear input feature dimension must match weight in-features dimension: input=<dimension>, weight=<dimension>
bias rank must be exactly 1: <rank>
linear bias dimension must match weight out-features dimension: bias=<dimension>, weight=<dimension>
```

The contraction message is used only for unequal static Dimensions. The bias-dimension message is
used for every structural inequality, including static mismatch and unresolved combinations that
ordinary ADD cannot prove compatible. Existing `DataTypePromotion` messages remain unchanged.

Every failure above consumes no Tensor ID and creates no transpose, MATMUL, or ADD wrapper or
producer. In particular, invalid bias never leaves a partially created transpose or MATMUL
expression. Tests must inspect the shared factory ID counter only for this concrete boundary and
must restore it safely.

After validation, failures are limited to established primitive/factory behavior such as resolved
transpose-layout arithmetic or Tensor-ID exhaustion. A successful no-bias call allocates exactly
two IDs. A successful bias call allocates exactly three. If exhaustion occurs after an earlier
intermediate succeeds, already allocated IDs remain consumed and no rollback is added; this is the
existing factory contract, not a local validation failure.

## Exact metadata, wrappers, and provenance

### No-bias form

One call creates exactly two fresh Tensor wrappers and IDs in this order:

1. `transposedWeight`
   - exact type and `requiresGrad` from weight;
   - Shape `[weightInFeatures, outFeatures]` with exact reordered Dimensions;
   - resolved logical view layout with reordered strides and preserved offset when weight layout
     is resolved, otherwise unresolved layout;
   - empty label and storage;
   - one PERMUTE producer with `PermutationAttrs([1, 0])`, exact input `[weight]`, one descriptor,
     and provenance output index zero.
2. `product`, which is also the returned result
   - promoted input/weight type;
   - exact Shape `[..., outFeatures]`;
   - unresolved layout, empty label, and empty storage;
   - `requiresGrad = input.requiresGrad || weight.requiresGrad`;
   - one MATMUL producer with exact inputs `[input, transposedWeight]`, one descriptor, and output
     index zero.

There is no wrapper, ID, producer, operation, or provenance entry named LINEAR.

### Bias form

The first two wrappers are identical in meaning and allocation order to the no-bias form. The call
then creates exactly one additional wrapper and ID:

3. `sum`, which is the returned result
   - promoted product/bias type;
   - a Shape structurally equal to the product Shape and reusing its exact ordered Dimension
     references; the outer Shape object may differ under ordinary ADD construction;
   - unresolved layout, empty label, and empty storage;
   - `requiresGrad = input.requiresGrad || weight.requiresGrad || bias.requiresGrad`;
   - one ADD producer with exact inputs `[product, bias]`, one descriptor, and output index zero.

The final provenance therefore differs by overload: no-bias returns MATMUL provenance; bias
returns ADD provenance. In the bias form the MATMUL and PERMUTE remain reachable only through the
ordered intermediate inputs. Separate valid calls create identity-distinct wrappers and producers
even when all operands are the same. Inputs and all input metadata, storage, IDs, and provenance
remain unchanged.

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
- [MATMUL task](0019-matmul-semantics-and-tensor-expression.md)
- [Permute and transpose task](0017f-permute-and-transpose-tensor-expressions.md)
- [Binary arithmetic task](0014b-binary-arithmetic-tensor-expressions.md)

## Architecture constraints

- Work remains entirely in `modules/model` and its documentation/planning records.
- `Tensor` remains public mutable API state and is not graph intermediate representation.
- Primitive producers record backend-independent meaning only. No LINEAR operation is introduced.
- Package direction remains from `model.tensor` to existing datatype, shape, and operation
  contracts; no operation package imports Tensor or graph state.
- Compiler later owns capture, deferred contraction proof, canonicalization, decomposition, and
  gradients. Backend prepare owns pattern recognition, fusion, algorithms, and route selection.
- Runtime eventually executes only prepared work and does not inspect these model operations on
  its hot path.
- No architecture, dependency, lifecycle, focused-architecture, Gradle, or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.operation.elementwise.binary`

Packages added or changed:

- No package is added. The existing tensor package gains one package-private composition helper
  and two public methods on its established Tensor facade.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorLinearExpressions` — package-private owner of
  input-aware validation and explicit composition over three existing tensor-package helpers.
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public fluent API owner.

Tests mirror `model.tensor` because they inspect package-private helper and ID-allocation behavior.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — update the
  global exact public-name inventory and total count from 169 to 171.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — update only the global public Tensor method count from 169 to 171.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — update only the global public Tensor method count from 169 to 171.

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless the final diff makes an existing claim inaccurate: Training API;
transpose/MATMUL/ADD, DataTypePromotion, Shape/Dimension/ShapeBroadcast, TensorDescriptor,
producer/provenance/factory, and related operation contracts; architecture/ADRs/tests;
conformance/integration; Gradle; other modules.

## Maximum scope

At most two production, four test, and seven documentation/planning files: exactly 13 paths.
`Tensor.java` changes only for the two methods, their Javadocs, and any necessary local placement.
The three existing tests change only the explicit global inventory/count assertions described
above. No existing primitive helper changes. Stop for a fourteenth path, another production/test
type, any broader existing-test edit, architecture change, or cross-module work.

## Javadoc and documentation requirements

- Document both public overloads with conventional receiver/input and weight orientation, rank,
  type, Shape, bias, dynamic-dimension, metadata, freshness, exact composition, provenance, ID,
  nullability, failure, and unsupported-layer contracts.
- Document every parameter with `@param`, each result with `@return`, and every caller-visible
  failure with `@throws`; explain that null bias is invalid and selects no-bias only by overload.
- Document the helper's exact validation and construction order, no-ID failure boundary, direct
  package-private helper calls, intermediate metadata, and exhaustion limitation.
- Add a focused Tensor API section with Shape examples and the complete producer-chain example
  above. Explain why the final producer is MATMUL or ADD rather than LINEAR.
- Update Compile API only to list this current model composition accurately and preserve the
  planned compiler-capture/canonicalization/fusion boundary. Do not claim compiler support.
- Review the glossary for reusable `linear layer` / `linear projection`, `in-features`, and
  `out-features` terminology. Add one focused entry if needed; do not duplicate existing MATMUL,
  transpose, producer, or provenance definitions.
- Keep capabilities, task, master plan, and roadmap synchronized. Record reasoned no-change
  conclusions for every reviewed but unchanged contract.

## Acceptance criteria

- Exactly two public overloads exist with signatures and argument names
  `linear(Tensor weight)` and `linear(Tensor weight, Tensor bias)`; public Tensor count is 171.
- No LINEAR kind, attributes, signature, result carrier, producer, compiler pass, or backend/runtime
  behavior exists.
- Input rank, exact weight rank/orientation, contraction, exact bias rank/extent, promotion, and
  dynamic rules match this specification.
- The helper calls existing package-private transpose, MATMUL, and ADD helpers directly only after
  complete local prevalidation; invalid bias creates no partial chain and consumes no ID.
- No-bias creates exactly two wrappers/IDs in PERMUTE then MATMUL order and returns MATMUL
  provenance. Bias creates exactly three in PERMUTE, MATMUL, ADD order and returns ADD provenance.
- Every intermediate and final descriptor, layout distinction, label/storage absence, gradient
  eligibility, producer identity, ordered input, output descriptor, and provenance index matches
  this specification; inputs remain unchanged.
- Focused tests cover all floating and signed-integral promotions, BOOL/cross-category rejection,
  rank-one and higher-rank Shape derivation, zero/static/named/expression/unknown Dimensions,
  deferred contraction, strict bias equality, both layout states for transpose, exact messages,
  validation ordering, no-ID failures, freshness, and exact producer chains.
- All three pre-existing global public Tensor inventory/count tests are updated up front and no
  other inventory remains stale.
- Tests inspect metadata only and make no numerical execution, compiler, gradient, fusion,
  backend, or runtime claim.
- Focused tests, exactly one final model suite after Java stability, final model Javadoc,
  documentation/example, Markdown links/anchors, exact 13-path scope, status, formatting,
  whitespace, and `git diff --check` pass.
- A separate clean-context documentation pass finalizes all authorized documentation, reuses the
  successful Java evidence, and records no-change conclusions.
- Task 0019D becomes Complete only after all evidence is recorded. Tasks 0019 through 0019C1 remain
  Complete; 0019E and every later task remain Draft without a detailed 0019E specification.

## Tests / validation

Run focused validation while developing:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorPermutationExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.shape.ShapeBroadcastTest
```

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

The focused linear test owns exact public/helper surfaces, API count/name presence, validation
order/messages, ID effects, Shape/type/metadata, intermediate and final producer identity, wrapper
count/order, and freshness. Existing primitive suites protect the construction dependencies
without changing their contracts. Recurring public inventory invariants remain automated in the
three explicitly listed existing tests rather than manual reflection or bytecode checks.

The separate documentation pass runs after final Javadoc edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, terminology, the runnable Java 26 metadata example, generated Javadoc, exact 13 paths,
package placement, public method count/name inventory, synchronized task/master/roadmap status,
exactly one Ready frontier before implementation or none after completion, 0019 through 0019C1
Complete, 0019E/later Draft, and absence of a detailed 0019E specification.

Repository-wide validation is the selected-modern-operations checkpoint after task 0022 or CI.
This model-only convenience changes no dependency, architecture boundary, shared Gradle contract,
or other module and does not repeat the completed 0019C1 repository checkpoint.

## Dependencies

- 0017F: rank-two transpose as PERMUTE with current view-layout behavior.
- 0014B, 0018T, and 0018U: ADD construction and same-category numeric promotion/overflow policy.
- 0018K–0018N: operation signatures, shared producer/provenance, symbolic Dimensions, and typed
  numeric boundaries.
- 0019: current MATMUL rank, Shape, type, numerical, metadata, and deferred-contraction contract.
- 0019A–0019C1: completed sequential rows establishing 0019D as the sole Ready frontier.

## Follow-up tasks

- 0019E remains Draft for one-output scaled dot-product attention without dropout or a public
  attention-weight result.
- Compiler later owns capture, proof of deferred contraction equality, canonicalization,
  decomposition decisions, and autograd.
- Backend/conformance/runtime/integration work later owns fusion recognition, lowering, algorithms,
  tolerances, kernels, storage, and execution.

Do not create another detailed task specification during task 0019D implementation.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, dependency, lifecycle, focused-architecture,
cross-module, another-helper/type, or scope change, stop and report the conflict instead of editing
around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, current architecture,
documentation/planning rules, roadmap, model capabilities/master plan, completed provenance,
signature, symbolic-shape, typed-scalar, transpose, ADD/promotion, MATMUL, and current 0019A–0019C1
tasks, and task 0019D.

Implement docs/planning/modules/model/tasks/0019d-linear-convenience.md exactly inside its 13
authorized paths. Preserve existing primitive helpers and all architecture boundaries. Stop on
architecture uncertainty, scope overflow, another type/test/document need, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and exact Java evidence to a separate clean-context documentation agent in the same overall
change. That agent finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning, the
runnable producer-chain example, and documentation checks while reusing successful Java evidence.
Synchronize status only after every criterion passes; keep 0019–0019C1 Complete and 0019E/later
Draft without a detailed specification.
```

## Documentation-agent handoff

Give the separate clean-context documentation agent this task, the complete implementation diff,
exact focused/final model evidence and whether executable Java changed afterward, all selected
API/Shape/type/bias/validation/metadata/provenance/ID policies, the seven authorized documentation
paths, and required Javadoc, example, Markdown, scope, and status validation.

The documentation agent independently reads AGENTS, architecture, documentation rules and the
General/API-Javadoc/Planning/Example profiles, this task, final source/tests/generated Javadoc,
Tensor/Compile/Training APIs, glossary, capabilities/master/roadmap, and directly related
transpose/MATMUL/ADD, DataTypePromotion, Shape/Dimension/ShapeBroadcast, descriptor, factory,
producer, and provenance contracts. It finalizes Javadocs and documentation, compiles/runs the
metadata example, and records reasoned no-change conclusions for Training API, primitive and
foundational contracts, architecture/ADRs/tests, conformance/integration, Gradle, other modules,
and Draft follow-ups.

It does not repeat successful Java tests unless executable Java changes, evidence is stale, or a
concrete recorded risk requires a rerun. It records the clean-context identifier, reused evidence,
files/topics reviewed, commands/results, glossary impact, limitations, and unresolved issues.

## Local decisions

- Selected the conventional receiver/input API and `[outFeatures, inFeatures]` weight orientation
  already established by the model capability baseline.
- Retained both floating and signed-integral primitive domains rather than narrowing to floating:
  a pure convenience should not contradict the literal MATMUL-plus-ADD composition.
- Selected exact rank-one `[outFeatures]` bias with structural Dimension equality. Scalar,
  higher-rank, singleton-broadcast, and unresolved-but-unequal biases are rejected because they
  obscure the layer contract or require changing current Shape/broadcast constraints.
- Selected direct package-private helper calls after complete prevalidation. Public-method chaining
  would add dispatch/recursion ambiguity without changing semantics; reconstructing primitive
  producers would duplicate established contracts.
- Preserved the actual intermediate wrappers and provenance rather than inventing an idealized
  LINEAR origin. Resolved weight layout may remain resolved on transpose; MATMUL and ADD outputs
  are unresolved.

## Known limitations

- Unequal unresolved input/weight contraction Dimensions require later proof, as for MATMUL.
- Bias equality cannot be deferred unless the Dimensions are already structurally equal because
  current ordinary broadcasting carries no equality constraint.
- Construction may consume an earlier intermediate ID if Tensor-ID exhaustion occurs later; only
  caller-controlled local validation is guaranteed to be all-or-nothing before allocation.
- Model completion does not imply numerical execution, gradient construction, compiler capture,
  fusion, or backend support.

## Validation evidence

Implementation context: `/root/task_0019d_implementation`.

- The exact focused selection in this task passed with `BUILD SUCCESSFUL`: 60 tests across
  `TensorLinearExpressionTest`, `TensorPermutationExpressionTest`,
  `TensorMatmulExpressionTest`, `TensorBinaryArithmeticTest`, `DataTypePromotionTest`, and
  `ShapeBroadcastTest`; zero failures, errors, or skips.
- After executable Java stabilized, exactly one final `./gradlew :modules:model:test` passed with
  `BUILD SUCCESSFUL`. The generated reports contain 836 tests across 105 XML suites with zero
  failures, errors, or skips.
- No executable Java changed after either successful result. The separate documentation context
  changed only Javadocs in the two authorized production paths and the seven authorized
  documentation/planning paths, so it did not rerun Java tests.

Documentation context:
`/root/task_0018s_implementation/task_0018s_documentation`, independently restarted for 0019D and
applying the General, API/Javadoc, Planning, and Example profiles.

- Independently read repository instructions, the architecture contract and current architecture
  index, documentation/planning rules, roadmap, capabilities/master plan, this task, completed
  transpose/ADD/MATMUL contracts, the actual diff, final affected source/tests, Tensor/Compile/
  Training APIs, glossary, and generated test/Javadoc reports.
- Finalized both public `Tensor.linear` Javadocs and package-private `TensorLinearExpressions`
  Javadocs. They now cover conventional `[outFeatures, inFeatures]` weight orientation, exact
  rank-one bias, floating/integral promotion, dynamic contraction and strict bias rules, complete
  prevalidation, two/three ID order, intermediate metadata, primitive producers, Shape identity,
  failures, exhaustion, and model/compiler/backend/training boundaries.
- Corrected the clarified biased-Shape contract consistently: final ADD Shape is structurally
  equal to the MATMUL product Shape and reuses the exact ordered Dimension references, but the
  outer Shape objects may differ. No-bias returns the MATMUL product object directly.
- Finalized Tensor API with exact Shape examples and the complete producer-chain example; Compile
  API now lists current model composition without claiming compiler support; glossary defines
  linear projection, in-features, out-features, and the distinction from a stateful layer.
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; two tasks executed. Generated
  Tensor Javadoc contains exactly both public linear overloads and the final contract, while the
  package-private helper remains absent from public generated Javadoc.
- `javac -cp modules/model/build/classes/java/main -d
  /tmp/linear-producer-chain-example /tmp/LinearProducerChainExample.java` and the matching `java`
  command passed. Output was `Shape[2, 4]`, ADD/MATMUL/PERMUTE, exact ordered-input booleans,
  `[1, 0]`, output indices `0,0,0`, structural Shape equality `true`, and outer identity `false`.
- Reflection and `javap` confirmed exactly 171 public Tensor methods, exactly two public `linear`
  overloads with the required parameter types, and one package-private final helper with only its
  private constructor and two package-private static `apply` methods. Source/import scans confirm
  direct package-private PERMUTE, MATMUL, and ADD calls after prevalidation, no LINEAR kind or
  cross-layer import, and no change to shared primitive helpers.
- Targeted Markdown validation resolved all 562 local links and 155 GitHub-style fragments across
  the seven changed Markdown files. Fences are balanced; all existing changed paths are non-empty,
  end with a newline, and contain no trailing whitespace.
- Exact-scope validation found only the authorized thirteen paths: two production, four tests, and
  seven documentation/planning paths. Architecture, ADRs/tests, Gradle, conformance/integration,
  other modules, shared primitives, Training API, and later specifications are unchanged.
- Status/dependency validation found tasks 0019 through 0019C1 and 0019D Complete, 0019E and every
  later task Draft, no Ready model frontier, and no detailed 0019E-or-later specification.
- `git diff --check` passed with no output after the final combined edits. New untracked paths also
  passed explicit final-newline, whitespace, and fence checks.
- Training API needs no change because linear creates no parameter owner, initialization,
  optimizer integration, gradient formula, backward operation, or training workflow. Transpose,
  ADD/MATMUL, DataTypePromotion, Shape/Dimension/ShapeBroadcast, TensorDescriptor, factory,
  producer/provenance, architecture, build, conformance/integration, and other-module contracts
  remain accurate because 0019D composes them without modifying their behavior or boundaries.

## Implementation notes

Implemented the two public overloads as direct delegation to one new package-private stateless
helper. The helper fully prevalidates input/weight/bias, then calls existing transpose, MATMUL, and
ADD helpers in exact order. Documentation records actual primitive producers and the clarified
Dimension-reference versus outer-Shape-identity boundary; no executable behavior was changed by
the documentation pass.

## Completion summary

- Completed changes: added conventional weight-transposed MATMUL plus optional exact rank-one ADD
  bias as visible public linear composition without a LINEAR kind.
- Files changed or created: exactly the authorized thirteen production, test, API, glossary, and
  planning paths.
- Tests and validation: reused the passing 60-test focused selection and final 836-test/105-suite
  model run; model Javadoc, runnable Java 26 example, generated-page, reflection/javap/import/
  source, Markdown, exact-scope, status/dependency, terminology, fence/newline/whitespace, and
  `git diff --check` validations passed.
- Documentation-agent review: completed independently in the clean context named above.
- Documentation impact: Tensor and Compile APIs, glossary, capabilities, task, master plan, and
  roadmap now describe the current primitive composition and cross-layer limitations.
- Javadoc review: both public Tensor overloads and the package-private helper finalized; retained
  transpose, MATMUL, ADD, promotion, Shape, descriptor, factory, and provenance contracts reviewed
  unchanged.
- Glossary impact: added one focused linear-projection entry defining in-features/out-features and
  distinguishing model composition from a stateful linear layer.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
