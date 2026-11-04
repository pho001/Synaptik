# Task 0018I: [Scatter-ND](../../../../glossary.md#scatter-nd) Semantics

**Editorial reading key (current context, not a status update).**

This completed record concerns semantic identity and immutable parameters, not public expression
construction. `SCATTER_ND` names functional tuple scatter; `ScatterNdAttrs` carries a normalized
shared-leading-batch count and explicit reduction, not operands or tuple depth. Tuple depth is the
final indices extent: with data Shape `[2, 3]` and indices Shape `[2]`, one two-coordinate tuple
requires scalar updates `[]`, while the result remains `[2, 3]`. Follow the current [semantic
contract](../../../../reference/tensor/operation-indexing.md#scatter-nd-semantic-kind-and-attributes)
for these roles and reduction meanings.

In the current [expression
boundary](../../../../reference/tensor/scatter.md#scatter-nd-expressions), the caller invokes
`data.scatterNd` with indices and updates Tensor references; construction checks descriptions,
records those references, and returns a storage-free expression—not computed values or an owned
result buffer. Bounds and duplicate detection require values; `NONE` forbids duplicate targets.
Follow that section for validation and numerical rules, including exact untouched representations.
Model responsibility for meaning is not storage ownership; execution providers realize supported
routes without redefining semantics. These current descriptions do not expand this task’s historical
scope.

## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent semantic [identity](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) [parameters](../../../../glossary.md#term-parameter) for functional
tuple-index `SCATTER_ND`.

[Scatter-ND](../../../../glossary.md#scatter-nd) consumes ordered logical inputs `[data, indices, updates]`. The final indices [Dimension](../../../../glossary.md#dimension)
is [tuple depth](../../../../glossary.md#term-tuple-depth) `K`; each tuple addresses `K` data [axes](../../../../glossary.md#term-axis) after a shared leading [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix. Updates
have the same [Shape](../../../../glossary.md#shape) that [Gather-ND](../../../../glossary.md#gather-nd) would read:

```text
updates.shape == indices.shape[:-1]
                 + data.shape[batchDimensions + K:]
```

The conceptual result starts from `data`, applies the selected `ScatterReduction`, has exactly the
data Shape, and does not mutate data in place. This task defines meaning and intrinsic parameters
only; input-aware [validation](../../../../glossary.md#term-validation) and public [Tensor](../../../../glossary.md#tensor) construction remain task 0018J.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `ScatterNdKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind) with exactly `SCATTER_ND`.
- Add one public `ScatterNdAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly
  `int batchDimensions` followed by `ScatterReduction reduction`.
- Require non-negative normalized [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count and explicit non-null reduction in exact order.
- Reuse the completed `ScatterReduction` enum without changing or duplicating it.
- Define exact ordered `[data, indices, updates]` roles.
- Define final-indices-Dimension tuple depth, shared batch prefix, indexed data axes, untouched
  data suffix, exact updates-Shape formula, data-shaped result, and functional non-mutation.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [tuple depth glossary](../../../../glossary.md#term-tuple-depth); [axes glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape))</span>
- Document replacement/addition/multiplication/maximum/minimum through the shared reduction values.
- Document `NONE` duplicate [target](../../../../glossary.md#term-target) tuples as invalid according to the existing reduction contract.
- Pair `SCATTER_ND` explicitly with `ScatterNdAttrs` without adding a compatibility validator.
- Add one same-package focused structural, validation, value-semantics, distinction, and
  composition test.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Keep production in `io.github.pho001.synaptik.model.operation.index`.
- Finalize Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through the
  mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.scatterNd` methods, default overloads, expression helper, factory, or task-0018J
- [axis](../../../../glossary.md#term-axis) scatter, [Gather-ND](../../../../glossary.md#gather-nd) expression changes, [gather](../../../../glossary.md#gather) backward, [masks](../../../../glossary.md#term-boolean-mask), slices, or another family
- storing data, indices, updates, ranks, Shapes, tuple depth, data types, descriptors, layout,
  requiresGrad, provenance, label, storage, or backend facts<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [backend glossary](../../../../glossary.md#backend))</span>
- storing [tuple depth](../../../../glossary.md#term-tuple-depth) in attributes; it remains [occurrence](../../../../glossary.md#term-occurrence)-specific final indices [Shape](../../../../glossary.md#shape) data
- validating batch count against ranks, shared batch Dimensions, tuple depth, updates Shape, data
  and update types, index type, result descriptor, or reduction/type compatibility<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Dimensions glossary](../../../../glossary.md#dimension))</span>
- reading index/update values, normalizing index values, bounds checks, negative-index policy,
  duplicate detection, write/reduction execution, or in-place mutation
- implicit default reduction, null-as-`NONE`, default constructor, singleton, separate zero-batch
  kind, sentinel batch count, factory, builder, registry, parser, visitor, or map<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [registry glossary](../../../../glossary.md#term-registry))</span>
- numerical order, overflow, NaN, signed-zero, reproducibility, atomics, empty-domain behavior,
  gradient policy, or backend support<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- graph capture, compiler, planning, prepare, runtime, backend, engine, trace, ONNX implementation,
  training, gradients, or execution behavior<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- changing existing Java/tests, `ScatterReduction`, dependencies, Gradle, architecture, another
  module, or creating a task-0018J specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Task 0018F](0018f-gather-nd-tensor-expressions.md)
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch represents [ONNX](../../../../glossary.md#term-onnx)-style ScatterND with one `SCATTER_ND` [identity](../../../../glossary.md#term-identity-meanings), one
reduction, and a non-negative `batchDims`. Its public [API](../../../../glossary.md#term-application-programming-interface-api) exposes zero-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) replacement, zero-
batch explicit reduction, and explicit reduction plus batch count.

Legacy [Shape](../../../../glossary.md#shape) behavior uses the final indices [extent](../../../../glossary.md#term-extent) as [tuple depth](../../../../glossary.md#term-tuple-depth), requires shared batch
[Dimensions](../../../../glossary.md#dimension), and requires updates Shape equal to the [Gather-ND](../../../../glossary.md#gather-nd) result Shape. The conceptual result
always has data Shape. Replacement rejects duplicate tuple [targets](../../../../glossary.md#term-target); arithmetic reductions combine
duplicates. The current [model](../../../../glossary.md#term-model) uses [canonical](../../../../glossary.md#term-canonical-form) [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) Shape instead of legacy `[1]` where
the updates formula is scalar.

The new semantic value uses descriptive `batchDimensions`, explicit non-null reduction, and typed
[immutable](../../../../glossary.md#term-immutable) contracts. Legacy [mutable](../../../../glossary.md#term-mutable) Shapes, [graph](../../../../glossary.md#graph) builders, [gradient](../../../../glossary.md#term-gradient) callbacks, traits, [lowering](../../../../glossary.md#lowering),
[kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-neutral [operation](../../../../glossary.md#operation) meaning.
- `ScatterNdKind` identifies mathematical tuple-scatter semantics only, not an occurrence, Tensor,
  graph node, descriptor, executable, kernel, or backend route.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [descriptor glossary](../../../../glossary.md#term-descriptor); [kernel glossary](../../../../glossary.md#kernel))</span>
- Ordered inputs are semantically `[data, indices, updates]`; attributes store none of them.
- `ScatterNdAttrs.batchDimensions` is already normalized and non-negative. It stores no ranks and
  cannot prove input compatibility.
- [Tuple depth](../../../../glossary.md#term-tuple-depth) remains the final indices [Dimension](../../../../glossary.md#dimension) and must not be duplicated in attributes.
- The shared `ScatterReduction` is exact semantic vocabulary, not an algorithm or [backend route](../../../../glossary.md#backend-route).
- Result and updates [Shape](../../../../glossary.md#shape) formulas are explanatory contracts; these types inspect no Shape.
- Generic `Operation` remains an open kind/attributes pair and validates no family pairing, arity,
  ranks, Shapes, types, bounds, duplicates, gradients, or backend support.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Package direction remains `model.operation.index -> model.operation + java.base`.
- Stop if implementation requires Tensor, Shape, DataType, provenance, another production type,
  another test, dependency, architecture change, or cross-layer behavior.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.index.ScatterNdKind` — sole tuple-index scatter
  semantic identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs` — normalized batch count plus
  explicit shared reduction.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- `ScatterNdSemanticsTest` — same-package structural, [validation](../../../../glossary.md#term-validation), composition, and boundary test.

The existing index package remains cohesive and independent of public [Tensor](../../../../glossary.md#tensor) and [graph](../../../../glossary.md#graph) packages.

## Required contract

### Semantic kind

Create exactly:

```java
public enum ScatterNdKind implements OperationKind {
    SCATTER_ND
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) body,
[alias](../../../../glossary.md#aliasing-and-arena), [arity](../../../../glossary.md#term-arity), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count, [tuple depth](../../../../glossary.md#term-tuple-depth), reduction, [Shape](../../../../glossary.md#shape), type, result, cost, [fusion](../../../../glossary.md#term-fusion), route, or
[backend](../../../../glossary.md#backend) [metadata](../../../../glossary.md#term-metadata). Inherited `Enum.name()` satisfies `OperationKind.name()`.

[Javadoc](../../../../glossary.md#term-javadoc) must explain ordered inputs, functional non-mutation, final-[Dimension](../../../../glossary.md#dimension) tuple depth, batch
prefix, indexed [axes](../../../../glossary.md#term-axis), untouched suffix, updates formula, data-shaped result, reduction meaning,
duplicate-[target](../../../../glossary.md#term-target) boundary, and distinctions from [Gather-ND](../../../../glossary.md#gather-nd) and axis scatter.

### [Immutable](../../../../glossary.md#term-immutable) attributes

Create exactly:

```java
public record ScatterNdAttrs(
        int batchDimensions,
        ScatterReduction reduction) implements OperationAttrs
```

The record has exactly two components in that order, one [canonical](../../../../glossary.md#term-canonical-form) constructor, two explicit
documented accessors, and record-generated `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`. Add no [tuple
depth](../../../../glossary.md#term-tuple-depth), [rank](../../../../glossary.md#term-rank), [Shape](../../../../glossary.md#shape), input, result, default constructor, factory, builder, nested type, or extra [API](../../../../glossary.md#term-application-programming-interface-api).

Constructor [validation](../../../../glossary.md#term-validation) order and behavior are exact:

1. if `batchDimensions < 0`, throw `IllegalArgumentException` with exact message
   `batchDimensions must be non-negative: <batchDimensions>`;
2. require non-null `reduction` with `Objects.requireNonNull(reduction, "reduction")`;
3. otherwise retain both values unchanged.

Zero and `Integer.MAX_VALUE` [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) counts are structurally valid because no ranks are present.
Every reduction is structurally valid. Null never means `NONE`.

### Typed composition

Compose explicitly:

```java
ScatterNdAttrs attrs =
        new ScatterNdAttrs(1, ScatterReduction.ADD);
Operation operation =
        new Operation(ScatterNdKind.SCATTER_ND, attrs);
```

The exact attributes reference is retained. Generic [Operation](../../../../glossary.md#operation) does not enforce pairing. Add no
operation factory, default attributes, compatibility validator, or [matrix](../../../../glossary.md#term-matrix-meanings).

### [Shape](../../../../glossary.md#shape) terminology and examples

For data [rank](../../../../glossary.md#term-rank) `R`, indices rank `Q`, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count `B`, and final indices [extent](../../../../glossary.md#term-extent) `K`:

- `Q >= 1`;
- `0 <= B < Q` and `B <= R`;
- leading data and indices [Dimensions](../../../../glossary.md#dimension) `[0, B)` match;
- `1 <= K <= R - B`;
- each tuple indexes data [axes](../../../../glossary.md#term-axis) `[B, B + K)`;
- data axes `[B + K, R)` form the untouched suffix copied as one update [slice](../../../../glossary.md#slice);
- `updates.shape == indices.shape[0:Q-1] + data.shape[B+K:R]`;
- result Shape is exactly `data.shape`.

These are task-0018J input-aware rules, not constructor checks here.

Use these non-executable examples:

- data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, `K=2`, updates `[5, 4]`, result `[2, 3, 4]`;
- data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, `K=1`, updates `[2, 5, 4]`, result `[2, 3, 4]`;
- data `[2, 3]`, indices `[2]`, `B=0`, `K=2`, [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) updates `[]`, result `[2, 3]`.

No production code here stores operands, computes Shapes, reads values, or executes writes.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdAttrs.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/ScatterNdSemanticsTest.java`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most the eight paths above. Stop if implementation requires another production type/test,
existing Java edit, public [Tensor](../../../../glossary.md#tensor) behavior, [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api)/capabilities edit, [dependency](../../../../glossary.md#term-dependency), build or
architecture change, another [module](../../../../glossary.md#term-module), or a ninth path.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document every public type, [constant](../../../../glossary.md#term-constant), component/accessor, and [canonical](../../../../glossary.md#term-canonical-form) constructor.
- Define data, indices, updates, tuple depth, batch prefix, indexed axes, untouched suffix, target,
  reduction, and duplicate target before relying on them.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Include the exact formula and all three [Shape](../../../../glossary.md#shape) examples, including canonical [scalar](../../../../glossary.md#term-scalar) updates.
- Explain why [tuple depth](../../../../glossary.md#term-tuple-depth) is not stored and why ranks/Shapes are deferred to task 0018J.
- Explain exact [validation](../../../../glossary.md#term-validation) order, explicit reduction, and zero-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) convention.
- Explain `NONE` duplicate invalidity without promising detection here.
- Distinguish [Scatter-ND](../../../../glossary.md#scatter-nd) from [Gather-ND](../../../../glossary.md#gather-nd), [axis](../../../../glossary.md#term-axis) scatter, and in-place mutation.
- Promise no [gradients](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [backend](../../../../glossary.md#backend) support, bounds, numeric order, or execution.

## Acceptance criteria

- Exact one-[constant](../../../../glossary.md#term-constant) public `ScatterNdKind` implements [`OperationKind`](../../../../glossary.md#operationkind) with no project [API](../../../../glossary.md#term-application-programming-interface-api)/state.
- Exact two-component public `ScatterNdAttrs` implements [`OperationAttrs`](../../../../glossary.md#operationattrs) in required order.
- Negative-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) then null-reduction [validation](../../../../glossary.md#term-validation) uses exact exception types/messages.
- Every valid batch/reduction is retained; record value methods remain generated.
- Exact kind/attributes composition works for every reduction through unchanged [`Operation`](../../../../glossary.md#operation).
- Javadocs define roles, formulas, examples, reduction/duplicate boundaries, and deferred checks.
- No Tensor, Shape, DataType, descriptor, provenance, gradient, graph, compiler, backend, or
  execution behavior is introduced, and existing <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>`ScatterReduction` is unchanged.
- Tensor API, glossary, task/master/roadmap are independently finalized. Compile/Training API,
  capabilities, architecture, and related contracts receive reasoned no-change conclusions.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- All validation passes with exactly eight changed paths.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.index.ScatterNdSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact enum/record structure and interfaces; component order; zero, ordinary,
and maximum [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) values with every reduction; exact negative-first and null validation; generated
value semantics; exact composition/reference retention; distinctions from [Gather-ND](../../../../glossary.md#gather-nd)/[axis](../../../../glossary.md#term-axis) scatter;
and absence of extra kinds, APIs, state, or cross-layer imports.

Manually inspect `javap -p -c -s`, reflection/source/imports, generated [Javadoc](../../../../glossary.md#term-javadoc), Markdown links,
anchors, examples, fences, whitespace, exact eight-path [scope](../../../../glossary.md#term-scope), synchronized status, unchanged
`ScatterReduction`, and absence of a task-0018J specification.

## Dependencies

- 0005 and 0006 define semantic foundation and open [Operation](../../../../glossary.md#operation) pair.
- 0018E defines [Gather-ND](../../../../glossary.md#gather-nd) tuple/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) terminology and [immutable](../../../../glossary.md#term-immutable) batch [parameter](../../../../glossary.md#term-parameter) precedent.
- 0018G defines the shared exact reduction vocabulary and duplicate-[target](../../../../glossary.md#term-target) meaning.
- 0018H distinguishes completed one-[axis](../../../../glossary.md#term-axis) scatter from tuple-index scatter.

## Follow-up tasks

- 0018J: public [Scatter-ND](../../../../glossary.md#scatter-nd) [Tensor](../../../../glossary.md#tensor) expressions with type/[rank](../../../../glossary.md#term-rank)/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/tuple/[Shape](../../../../glossary.md#shape) [validation](../../../../glossary.md#term-validation).
- 0023: [compiler](../../../../glossary.md#term-compiler)-generated backward semantics.

Do not create a follow-up specification here.

## Architecture impact

Expected impact: None. Stop if a new rule or cross-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0018E/0018F/0018G/0018H/0018I, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation/GatherNdKind/
GatherNdAttrs/ScatterReduction and related index-family contracts/tests, and Java 26 Gradle.

Implement task 0018I exactly. Add only ScatterNdKind.java, ScatterNdAttrs.java, and
ScatterNdSemanticsTest.java under io.github.pho001.synaptik.model.operation.index.

ScatterNdKind contains exactly SCATTER_ND. ScatterNdAttrs contains exactly non-negative normalized
int batchDimensions then non-null ScatterReduction reduction, with exact validation order/messages
and explicit documented accessors. Reuse ScatterReduction unchanged. Document ordered
[data, indices, updates], final-Dimension tuple depth, shared batch prefix, indexed axes, untouched
suffix, updates formula/examples, exact data-shaped functional result, reduction meanings, and
invalid NONE duplicates without input-aware validation.

Do not add Tensor methods, Shape/DataType/result/provenance validation, new reduction types,
values/execution, gradients, graph/compiler/planning/runtime/backend behavior, dependencies,
build/architecture changes, existing Java edits, or later specs. Stop beyond eight paths or on
architecture uncertainty.

Run all validation, then hand actual diff/evidence to a separate clean-context docs agent. It must
inspect source/tests/Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record
Compile API/Training API/capability/architecture and related-contract no-change conclusions, and
rerun validation.

Update 0018I, master plan, and roadmap only for status/evidence. Do not mark Complete before both
passes. Leave 0018J Draft without a specification. Do not commit/push.
```

## Local decisions

- One kind represents tuple-index scatter; reduction and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count are explicit attributes.
- Structural [parameter](../../../../glossary.md#term-parameter) precedes reduction, matching [axis](../../../../glossary.md#term-axis)-scatter attribute style.
- [Tuple depth](../../../../glossary.md#term-tuple-depth) remains final indices [Shape](../../../../glossary.md#shape) data, not duplicated attributes.
- Shared `ScatterReduction` is reused unchanged; null is never a default.
- `NONE` duplicate tuples are invalid but detection is value-aware and deferred.
- Independent documentation review found both initial production Javadocs substantively complete.
  The final coordinating audit then requested two small source-Javadoc clarifications: define a
  target and duplicate target tuples locally, and explicitly name task 0018J as the owner of
  operand-dependent rank/Shape validation. No declaration or behavior changed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [target glossary](../../../../glossary.md#term-target); [owner glossary](../../../../glossary.md#term-owner); [rank glossary](../../../../glossary.md#term-rank); [validation glossary](../../../../glossary.md#term-validation))</span>

## Known limitations

- No ranks, [Shapes](../../../../glossary.md#shape), types, bounds, duplicates, [descriptors](../../../../glossary.md#term-descriptor), or [provenance](../../../../glossary.md#provenance) are validated/created.
- No numeric order, [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [ONNX](../../../../glossary.md#term-onnx), [backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), or execution behavior is defined.
- Public [Scatter-ND](../../../../glossary.md#scatter-nd) remains task 0018J.

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/implement_model_0018i` added exactly `ScatterNdKind`,
  `ScatterNdAttrs`, and `ScatterNdSemanticsTest`. Independent documentation context
  `/root/implement_model_0018i/review_model_0018i_docs` inspected the shared-tree diff, final
  source/test, generated Javadoc, related contracts, APIs, glossary, planning state, and Java 26
  build configuration before finalizing documentation in the same overall change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation pass applied General plus API/Javadoc style to the production Javadocs,
  Tensor API, and glossary; Planning style to this task, the model master plan, and roadmap; and
  Example format to the conceptual Shape examples. The final coordinating audit refined only the
  two production Javadocs to define target/duplicate-target terminology and explicitly name task
  0018J as the future owner of operand-dependent rank/Shape checks. Tensor API and glossary now
  explain current Scatter-ND semantics while public Tensor construction remains Draft task 0018J.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model); [Shape glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target); [owner glossary](../../../../glossary.md#term-owner); [rank glossary](../../../../glossary.md#term-rank); [Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, the focused
  current architecture, overview, lifecycle, module-boundary, and dependency explanations;
  documentation rules and General/API-Javadoc/Planning/Example profiles; planning guide and
  roadmap; model capabilities/master plan; tasks 0005, 0006, and 0018E–0018I; Tensor, Compile,
  and Training API references; glossary; Java 26 root/settings/model Gradle configuration; final
  implementation/tests; generated model Javadoc; and the actual complete diff.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Compile glossary](../../../../glossary.md#compile); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Related-contract review covered [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`Operation`](../../../../glossary.md#operation), `GatherNdKind`,
  `GatherNdAttrs`, `ScatterReduction`, `AxisScatterKind`, `ScatterElementsAttrs`, aggregate/scan
  reductions, their focused tests, and current Tensor/provenance boundaries. The new values
  compose the open semantic contracts without changing any existing declaration or behavior.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [provenance glossary](../../../../glossary.md#provenance))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.index.ScatterNdSemanticsTest` — `BUILD SUCCESSFUL`;
  XML reports 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 725 tests across
  84 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; both generated public pages contain the
  enum/constant, record components, canonical constructor, explicit accessors, exact failures,
  ordered roles, both formula notations, all three examples, reductions, duplicate boundary,
  family distinctions, and exclusions.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 root lifecycle tasks completed or were up-to-date
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed exactly one enum
  constant plus compiler-generated enum machinery. The record has exactly the two specified
  fields in order, negative-batch-first then non-null-reduction validation, two direct accessors,
  and generated record value methods. Focused reflection tests confirm the same API and state.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Source, import, and `jdeps` inspection found only the permitted local `OperationKind`/
  `OperationAttrs` contracts and `java.base`. No Tensor, DataType, Shape, layout, provenance,
  graph, compiler, planning, prepare, runtime, backend, training, or execution dependency or
  behavior was introduced. <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>`ScatterReduction` is unchanged.
- Targeted Markdown validation resolved 425 local links, including 131 heading anchors, across
  the five changed documentation/planning files with zero errors. Fence counts were 228, 4, 12,
  2, and 0 and therefore balanced. Trailing-whitespace scans found no matches, all eight paths
  have final newlines, and `git diff --check` passes.
- Final changed-path inventory contains exactly the eight authorized paths: two production
  contracts, one focused test, Tensor API, glossary, this task, model master plan, and roadmap.
  Task/master-plan/roadmap status is synchronized as Complete. Task 0018J remains Draft, and no
  task-0018J specification exists.
- Compile API remains accurate unchanged because this task adds no Tensor expression, graph
  capture, inference, validation, optimization, compile artifact, or engine behavior. Training
  API remains accurate unchanged because no gradient, autograd, parameter, optimizer,
  publication, session, or training execution behavior changed. The capability baseline already
  inventories Scatter-ND with batch dimensions, the shared five reductions, exact integral index
  types, and distinct support layers, so it required no edit.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [artifact glossary](../../../../glossary.md#term-artifact); [engine glossary](../../../../glossary.md#term-engine); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication); [dimensions glossary](../../../../glossary.md#dimension))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, and other modules remain
  accurate unchanged because the task stays inside model-owned semantics and changes no module
  boundary, dependency rule, backend behavior, executable end-to-end behavior, or build
  requirement. Existing Operation, Gather-ND, axis-scatter, ScatterReduction, aggregate/scan,
  Tensor, provenance, and other operation-family contracts remain accurate unchanged for the
  related-contract reasons above.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gather-ND glossary](../../../../glossary.md#gather-nd); [axis glossary](../../../../glossary.md#term-axis))</span>

## Implementation notes

- Added `ScatterNdKind` with exactly `SCATTER_ND` and no project-declared behavior or [metadata](../../../../glossary.md#term-metadata).
- Added `ScatterNdAttrs(int batchDimensions, ScatterReduction reduction)` with exact validation
  order, direct accessors, and generated record value semantics while reusing unchanged<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
  `ScatterReduction`.
- Added nine focused tests covering exact structure, retained values, exact failures, value
  semantics, typed composition, family distinctions, and dependency boundaries.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Finalized Tensor API, glossary terminology/inventories, and synchronized planning
  evidence/status through the independent documentation pass. A final coordinating audit added
  only the two source-Javadoc clarifications described above; declarations and behavior remain
  unchanged.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Completion summary

- Completed changes: Implemented and documented functional tuple-index `SCATTER_ND` semantics and
  immutable normalized batch-count plus explicit shared-reduction attributes without public
  Tensor or cross-layer behavior.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: Exactly the eight authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: Focused 9-test and all 725-model-test/84-suite runs, model Javadoc, root
  tests, javap/reflection/import/source/generated-page review, 425-link/131-anchor checks,
  fence/whitespace/newline checks, exact scope/status and no-0018J-spec checks, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018i/review_model_0018i_docs` completed the mandatory independent pass
  with General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now define current Scatter-ND roles, tuple/batch
  model, updates formula, examples, reduction meanings, duplicate boundary, and distinctions,
  while public Tensor construction and every value/executable concern remain planned. Compile
  API, Training API, capabilities, architecture/ADRs/tests, conformance/integration material,
  Gradle, dependencies, other modules, and later tasks remain accurate unchanged for the recorded
  reasons.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- Javadoc review: Both new public types, the enum constant, record components, canonical
  constructor, accessors, failures, formulas, examples, and exclusions are complete; no correction
  to declarations or behavior was required. The final wording explicitly defines target and
  duplicate target tuples and names task 0018J for operand-dependent rank/Shape validation.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form); [target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape))</span>
- Glossary impact: Added reusable [Scatter-ND](../../../../glossary.md#scatter-nd) terminology and synchronized current
  [`OperationKind`](../../../../glossary.md#operationkind)/[`OperationAttrs`](../../../../glossary.md#operationattrs) inventories without implying a public expression.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018I. Task 0018J remains Draft without a detailed
  specification.

Status: Complete
