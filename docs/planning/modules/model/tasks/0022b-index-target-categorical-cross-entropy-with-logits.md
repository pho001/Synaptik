# Task 0022B: Index-[Target](../../../../glossary.md#term-target) Categorical Cross-Entropy with Logits

**Editorial reading key (current context, not a status update).**

This completed task records model-only construction, not the extent of current execution support.
The caller invokes the logits Tensor’s loss method; `TensorLossExpressions` validates metadata and
delegates to the factory for one storage-free producer. Logits are unnormalized class scores; an
index target names one class per coordinate outside the class axis. Ordered inputs describe
`[logits, target]`; the exact typed ignore scalar is attributes metadata, not a third Tensor input.
Construction neither reads operand values nor executes or owns result buffers.

For today’s boundary, follow the [categorical-loss
reference](../../../../reference/tensor/statistics-losses.md#categorical-cross-entropy-with-logits-expressions):
it distinguishes current compiler differentiation and qualified execution routes from this
historical scope. Model semantics defines the loss; providers realize supported routes rather than
redefine it. Ignore matching precedes bounds and logits evaluation: target `-1` with exact INT32
ignore `-1` contributes positive zero even for NaN logits. `MEAN` counts only non-ignored targets;
an empty or all-ignored domain yields NaN. Deferred value checks and semantic responsibility do not
imply Model storage ownership.

**Mathematical reading illustration.** For $G\ge1$ groups, let $I$ be the subset of group positions
whose targets are not ignored, and let $\ell_g$ be each included group's finite real loss. When
$|I|>0$, $L=\sum_{g\in I}\ell_g/|I|$. For two groups with only the first included and $\ell_1=\log
2$, $L=\log 2$, not $(\log 2)/2$. No formula is evaluated for ignored logits. An empty $I$ uses the
specified NaN rule, not this positive-denominator illustration.

## Status

Complete

## Goal

Add one [backend](../../../../glossary.md#backend)-independent index-[target](../../../../glossary.md#term-target) categorical cross-entropy meaning for floating logits and
one [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype) class index per non-class [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). Preserve task 0022A's dense-target behavior,
add one exact typed ignore-index overload, and reuse explicit `NONE`, `SUM`, or [`MEAN`](../../../../glossary.md#term-mean) reduction.

```text
logits + class-axis-removed indices + optional exact typed ignore index
  -> one index-loss producer
     -> selected-class stable negative log-softmax or positive zero when ignored
        -> non-class Shape, sum scalar, or non-ignored-count mean scalar
```

[Model](../../../../glossary.md#term-model) construction validates [metadata](../../../../glossary.md#term-metadata) only. It reads no values and creates no public primitive
[decomposition](../../../../glossary.md#term-decomposition).

## [Scope](../../../../glossary.md#term-scope)

- Append `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` to `LossKind`.
- Add public [immutable](../../../../glossary.md#term-immutable) `IndexCategoricalCrossEntropyWithLogitsAttrs` containing normalized class
  axis, explicit reduction, and an internal optional exact typed ignore index.<span data-glossary-navigation="true"> ([class axis glossary](../../../../glossary.md#class-axis); [ignore index glossary](../../../../glossary.md#ignore-index))</span>
- Give the kind one fixed two-input/one-output [signature](../../../../glossary.md#term-signature) with ordered inputs `[logits, target]`.
- Broaden the existing method by exact [target](../../../../glossary.md#term-target) type, preserving its floating branch:

  ```java
  public Tensor categoricalCrossEntropyWithLogits(
          Tensor target, int classAxis, LossReduction reduction)
  ```

- Add exactly one public overload:

  ```java
  public Tensor categoricalCrossEntropyWithLogits(
          Tensor target,
          int classAxis,
          LossReduction reduction,
          ScalarValue ignoreIndex)
  ```

- Extend package-private `TensorLossExpressions` with target-type [dispatch](../../../../glossary.md#term-dispatch) and index construction.
- Fix Shape/type, ignore, denominator, empty/all-ignored, dynamic, bounds, numerical, metadata,
  provenance, identifier, diagnostic, and validation-order contracts.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- Add focused tests and change every exact public [Tensor](../../../../glossary.md#tensor) method-count inventory from 187 to 188.
- Finalize Javadocs, Tensor/Compile APIs, glossary, and planning records through the required clean
  documentation-focused handoff.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning); [handoff glossary](../../../../glossary.md#term-handoff))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- any change to the completed floating dense-target kind, formula, promotion, exact-shape rule,
  denominator, special values, diagnostics, or validation behavior<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- a second index-[target](../../../../glossary.md#term-target) name, public `Optional`, nullable or primitive ignore values, public attrs
  parameter, defaults, convenience aliases, parser, registry, or broad loss options<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [registry glossary](../../../../glossary.md#term-registry); [loss glossary](../../../../glossary.md#loss))</span>
- weights, masks, label smoothing, temperature, target broadcast/cast/promotion, one-hot conversion,
  sparse targets, negative-index wrapping, clamping, or a default class<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- probability-input or standalone negative-log-likelihood loss, binary cross entropy, or any other
  loss family<span data-glossary-navigation="true"> ([log glossary](../../../../glossary.md#term-log); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- construction-time value/storage reads, eager bounds checks/evaluation, algorithm selection,
  fixed traversal, or tolerances<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- [gradients](../../../../glossary.md#term-gradient)/[autograd](../../../../glossary.md#autograd), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), [prepare](../../../../glossary.md#prepare), [runtime](../../../../glossary.md#term-runtime), execution, [publication](../../../../glossary.md#term-publication), or training work
- changes to shared datatype, scalar, Shape, promotion, operation, factory, producer/provenance,
  architecture, dependencies, Gradle, another module, or tasks 0023 and later<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Exact semantic and public contract

### Kind, attributes, and [signature](../../../../glossary.md#term-signature)

`LossKind` has exactly this order:

```java
MEAN_SQUARED_ERROR,
DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
```

The new kind accepts only:

```java
OperationSignature.fixed(IndexCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1)
```

The attributes record has exactly these ordered components:

```java
int axis,
LossReduction reduction,
Optional<ScalarValue> ignoreIndex
```

Its constructor validates in component order:

1. negative [axis](../../../../glossary.md#term-axis) fails with `IllegalArgumentException("axis must be non-negative: " + axis)`;
2. null reduction fails with `NullPointerException("reduction")`;
3. null optional container fails with `NullPointerException("ignoreIndex")`;
4. a present non-[INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) value fails with:

   ```text
   ignoreIndex must have data type INT32 or INT64, but was <dataType>
   ```

The record retains exact references and no [Tensor](../../../../glossary.md#tensor), [Shape](../../../../glossary.md#shape), [target](../../../../glossary.md#term-target) type, count, denominator,
algorithm, [gradient](../../../../glossary.md#term-gradient), [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), or training state. Operand-aware helper
[validation](../../../../glossary.md#term-validation) additionally requires a present ignore value to have the exact target type.

### Existing-method [dispatch](../../../../glossary.md#term-dispatch) and new overload

After null checks and floating-logits [validation](../../../../glossary.md#term-validation), the existing three-[argument](../../../../glossary.md#term-argument) method dispatches on
the exact [target](../../../../glossary.md#term-target) [descriptor](../../../../glossary.md#term-descriptor) type:

- [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) selects the unchanged task-0022A dense construction;
- [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) selects index construction with `Optional.empty()`; and
- [BOOL](../../../../glossary.md#data-type--datatype) or a later unsupported type retains task 0022A's exact failure:

  ```text
  categoricalCrossEntropyWithLogits target must have a floating data type, but was <dataType>
  ```

Dispatch creates no intermediate [Tensor](../../../../glossary.md#tensor), [producer](../../../../glossary.md#term-producer), ID, cast, or one-hot value. The floating branch
must retain its exact dense kind/attrs, promotion, [Shape](../../../../glossary.md#shape) rules, messages, and validation order after
the dispatch point.

The new four-argument overload is index-only and represents its non-null [scalar](../../../../glossary.md#term-scalar) as
`Optional.of(ignoreIndex)`. A floating or BOOL target fails with:

```text
categoricalCrossEntropyWithLogits target must have data type INT32 or INT64 when ignoreIndex is present, but was <dataType>
```

A non-integral ignore value fails with the attrs message specified above; an integral value of the
wrong exact target type fails with the helper's target/ignore equality message specified below.
Both failures precede Shape access and factory delegation.

No Tensor method exposes `Optional`. This resolves the collision with the completed three-argument
[signature](../../../../glossary.md#term-signature) without changing source compatibility.

### [Shape](../../../../glossary.md#shape), [class axis](../../../../glossary.md#class-axis), and dynamic obligations

Logits must have [rank](../../../../glossary.md#term-rank) at least one. Normalize `classAxis` exactly once with
`logitsShape.normalizeAxis(classAxis)`; invalid [axes](../../../../glossary.md#term-axis) retain `Axis <axis> is outside shape rank
<rank>` and its current exception type.

[Index target](../../../../glossary.md#index-target) rank must equal `logits.rank() - 1`. [Target](../../../../glossary.md#term-target) axis `t` maps to logits axis `t` before
the class axis and `t + 1` at or after it. Structurally equal [Dimensions](../../../../glossary.md#dimension) pass; unequal static pairs
fail; unequal pairs involving an unresolved Dimension retain later equality obligations. No form
of [broadcasting](../../../../glossary.md#broadcasting) is accepted. Exact failures are:

```text
categoricalCrossEntropyWithLogits index target rank must equal logits rank minus one: logits=<logitsRank>, target=<targetRank>
categoricalCrossEntropyWithLogits index target dimension mismatch at target axis <targetAxis> (logits axis <logitsAxis>): logits=<logitsDimension>, target=<targetDimension>
```

`NONE` returns the exact target Shape object. That Shape is the per-target/non-class domain and is
constrained by the mapping above to equal the logits Shape with the class axis removed; [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
logits therefore require and return [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`. `SUM` and [`MEAN`](../../../../glossary.md#term-mean) return canonical
[scalar](../../../../glossary.md#term-scalar) Shape. No new result Shape or copied Dimension is created for `NONE`.

Let `S` be the eventual non-class group count and `C` the class [extent](../../../../glossary.md#term-extent). Without ignore, a
non-empty domain requires `C > 0`. Static `C == 0` is valid if a non-class zero extent proves
`S == 0`, and otherwise fails when all non-class extents are known non-zero with:

```text
categoricalCrossEntropyWithLogits class dimension must be positive when sample domain is non-empty: axis=<axis>, dimension=<dimension>
```

No-ignore unresolved cases retain `S == 0 || C > 0`. Local proof inspects Dimensions without
multiplying extents.

With ignore present, construction must not reject a structurally non-empty static `C == 0`
[occurrence](../../../../glossary.md#term-occurrence) solely from [metadata](../../../../glossary.md#term-metadata): it is valid when every target equals the ignore value. The later
obligation is `S == 0 || C > 0 || all targets equal ignoreIndex`; equivalently, when `C == 0`,
every target must match ignore. This value-dependent alternative is never proved by [model](../../../../glossary.md#term-model)
construction. Later [binding](../../../../glossary.md#term-binding)/execution compares ignore before bounds and rejects any non-matching
target because no class index is in range.

### Formula, ignore-before-bounds, and execution obligation

For non-class [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `g`, [target](../../../../glossary.md#term-target) `y[g]`, logits `z[g,c]`, class [extent](../../../../glossary.md#term-extent) `C`, and optional ignore
value `I`, compare the exact common [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) values before bounds or logits [evaluation](../../../../glossary.md#term-evaluation):

```text
if I is present and y[g] == I:
    loss[g] = +0
else:
    require 0 <= y[g] < C
    m[g] = max_c(z[g,c])
    lse[g] = m[g] + log(sum_c(exp(z[g,c] - m[g])))
    loss[g] = lse[g] - z[g,y[g]]
```

An ignore value may itself be outside `[0, C)`. A matching target is ignored; a non-matching
negative or `>= C` target is invalid and never wraps, clamps, selects a default, or produces zero.
For `C == 0`, every non-ignored target is therefore invalid, while an all-ignored non-empty domain
is valid.
Construction reads no values, including [eager](../../../../glossary.md#term-eager-evaluation) [constants](../../../../glossary.md#term-constant). A [compiler](../../../../glossary.md#term-compiler) may later reject provably
invalid constants; [backend](../../../../glossary.md#backend) preparation or [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) must safely prove/check every
non-ignored bound before indexed access. The future execution exception/status is intentionally
owned by that later contract and is not invented here. [Runtime](../../../../glossary.md#term-runtime) does not inspect [`Operation`](../../../../glossary.md#operation).

For logits `[1, 2, 3]`, `lse ~= 3.407605964`: target `2` gives approximately `0.407605964`, target
`0` gives `2.407605964`, and target `-1` with exact ignore `-1` gives [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) without
evaluating the [slice](../../../../glossary.md#slice). These are mathematical examples, not construction-time evaluation.

### Reduction, denominator, and empty/all-ignored behavior

Let `N` be the number of [targets](../../../../glossary.md#term-target) not equal to the optional ignore value; without ignore, `N = S`:

```text
NONE: one loss per group; ignored positions are +0      Shape = exact target Shape
SUM:  sum only non-ignored losses                       Shape = scalar
MEAN: sum only non-ignored losses divided by N          Shape = scalar
```

If `S == 0`, no target/logits value is evaluated, `NONE` is empty, `SUM` is [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), and
[`MEAN`](../../../../glossary.md#term-mean) is [NaN](../../../../glossary.md#term-nan). If `S > 0` but all targets are ignored, `NONE` contains positive zeros, `SUM` is
positive zero, and `MEAN` is NaN. Attributes store neither `S` nor `N`. `C == 1` is valid; target
zero with a [finite](../../../../glossary.md#term-finite) [logit](../../../../glossary.md#logit) has positive-zero [loss](../../../../glossary.md#loss).

### Types, computation, and special values

Logits accept only [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64). [Targets](../../../../glossary.md#term-target) accept only exact [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype). Every
reduction result retains exact logits type; target does not participate in floating promotion and
no cast is inserted. A present ignore [scalar](../../../../glossary.md#term-scalar) must exactly match target type. Task-owned failures:

```text
categoricalCrossEntropyWithLogits logits must have a floating data type, but was <dataType>
categoricalCrossEntropyWithLogits ignoreIndex data type must equal target data type: target=<targetType>, ignoreIndex=<ignoreType>
```

BFLOAT16/FLOAT32 use FLOAT32 max, exponential, summation, logarithm, subtraction, [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) sum, and
division; FLOAT64 uses FLOAT64. Final values round to logits format. Equal-or-wider intermediates,
compensation, vectorization, parallelization, [fusion](../../../../glossary.md#term-fusion), and [reassociation](../../../../glossary.md#term-reassociation) are allowed while exact
special-value classes remain stable. No narrower computation, traversal/tree, bitwise [identity](../../../../glossary.md#term-identity-meanings),
[NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign, or cross-[backend](../../../../glossary.md#backend) identical [finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding) is promised.

Ignored positions are exact [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) and do not evaluate [NaN](../../../../glossary.md#term-nan)/infinite logits. For a
non-ignored in-range target: any NaN [logit](../../../../glossary.md#logit) yields NaN; any positive-[infinity](../../../../glossary.md#term-infinity) logit makes
[log-softmax](../../../../glossary.md#softmax--log-softmax) indeterminate and yields NaN; an all-negative-infinity [slice](../../../../glossary.md#slice) yields NaN; with at least
one finite logit and other negative [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), selecting [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) yields [positive
infinity](../../../../glossary.md#nan-infinity-and-signed-zero) and selecting finite yields its finite stable [loss](../../../../glossary.md#loss). Exact-zero finite loss is positive
zero. Participating NaN propagates through reductions; otherwise participating positive infinity
yields positive infinity. Permitted reassociation may change finite rounding but not these classes.
Equal requests remain distinct [producers](../../../../glossary.md#term-producer); no interning, fixed order, or bitwise determinism exists.

### [Metadata](../../../../glossary.md#term-metadata), [provenance](../../../../glossary.md#provenance), and identifiers

Result metadata is exact logits type, selected [Shape](../../../../glossary.md#shape), unresolved [layout](../../../../glossary.md#layout), and
`requiresGrad = logits.requiresGrad()`. Integral [target](../../../../glossary.md#term-target) eligibility never contributes, even if its
[descriptor](../../../../glossary.md#term-descriptor) flag is true; this is metadata, not a [gradient](../../../../glossary.md#term-gradient) rule. Results are fresh, unlabeled, and
storage-free; inputs are unchanged.

Success creates exactly one attrs, [operation](../../../../glossary.md#operation), [producer](../../../../glossary.md#term-producer), descriptor, output-zero provenance,
wrapper, and [Tensor](../../../../glossary.md#tensor) ID. Provenance inputs are `[logits, target]`; ignore is attrs metadata, not an
input. No [softmax](../../../../glossary.md#softmax--log-softmax), [gather](../../../../glossary.md#gather), select, arithmetic, reduction, cast, or one-hot producer exists.
Every local failure precedes factory delegation and consumes no ID. Identifier exhaustion retains
`tensor identifier space exhausted` and current single-output behavior.

### [Validation](../../../../glossary.md#term-validation) order

Three-[argument](../../../../glossary.md#term-argument) helper:

1. null-check logits, [target](../../../../glossary.md#term-target), reduction;
2. require floating logits;
3. dispatch exact target type to unchanged dense, no-ignore index, or the unchanged dense
   unsupported-target failure;<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [ignore index glossary](../../../../glossary.md#ignore-index))</span>
4. selected branch completes all validation before one factory delegation.

Four-argument helper:

1. null-check logits, target, reduction, ignoreIndex;
2. require floating logits;
3. require exact [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) target using the ignore-present message;
4. require integral ignore, then exact ignore/target type equality;
5. read [Shapes](../../../../glossary.md#shape) and normalize [axis](../../../../glossary.md#term-axis) once;
6. validate [rank](../../../../glossary.md#term-rank)-minus-one and mapped [Dimensions](../../../../glossary.md#dimension) in target-axis order;
7. retain the ignore-present `S == 0 || C > 0 || all targets equal ignoreIndex` obligation without
   rejecting solely for static `C == 0`;
8. construct attrs with `Optional.of(ignoreIndex)`;
9. select exact target Shape for `NONE` or canonical scalar Shape otherwise, then construct the
   descriptor and exact operation;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scalar glossary](../../../../glossary.md#term-scalar); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
10. delegate once with `[logits, target]`.

The no-ignore branch follows steps 5-10 with `Optional.empty()`, except step 7 locally rejects a
definitely non-empty static `C == 0` domain and otherwise retains `S == 0 || C > 0`. Null messages
are `logits`, `target`, `reduction`, and `ignoreIndex`. The attrs constructor repeats its own
validation.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Dense categorical loss](0022a-dense-target-categorical-cross-entropy-with-logits.md)<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss))</span>
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [Producer provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([Producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Typed scalar values](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Indexing taxonomy](0018o-indexing-taxonomy-and-unstack-normalization.md)
- [Integral operations](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Integral reductions](0018u1-integral-reductions-and-arg-min-normalization.md)
- [Statistical reductions](0018v-multi-axis-and-statistical-reductions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work stays inside [model](../../../../glossary.md#term-model)-owned semantics, [Tensor](../../../../glossary.md#tensor) [metadata](../../../../glossary.md#term-metadata), pre-[capture](../../../../glossary.md#term-capture-meanings) [provenance](../../../../glossary.md#provenance), and docs.
- Attributes may consume foundational [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) but not Tensor, graph, compiler, training,
  runtime, prepare, engine, or backend types.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- Compiler owns capture, deferred proof, constant analysis, gradients, decomposition, and
  optimization. Backend prepare/prepared execution owns algorithms and safe bounds enforcement.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- The training extension may later consume the [loss](../../../../glossary.md#loss) Tensor but owns no [operation](../../../../glossary.md#operation) here.
- No architecture, ADR, architecture-test, dependency, Gradle, cross-module, conformance,
  integration, compiler, backend, runtime, or training change is authorized. Stop if current
  fixed-signature, descriptor, scalar, and single-output producer foundations are insufficient.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [signature glossary](../../../../glossary.md#term-signature); [descriptor glossary](../../../../glossary.md#term-descriptor); [scalar glossary](../../../../glossary.md#term-scalar); [producer glossary](../../../../glossary.md#term-producer))</span>

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.model.operation.loss`
- `io.github.pho001.synaptik.model.tensor`

Datatype, [Shape](../../../../glossary.md#shape), [operation](../../../../glossary.md#operation), and [provenance](../../../../glossary.md#provenance) foundations are consumed unchanged. No package is added,
moved, or renamed.

Type placement:

- `...operation.loss.LossKind` owns family [identity](../../../../glossary.md#term-identity-meanings)/[signature](../../../../glossary.md#term-signature).
- `...operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs` owns intrinsic axis, reduction,
  and optional exact integral ignore metadata.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `...tensor.TensorLossExpressions` owns package-private dispatch, validation, Shape, descriptor,
  and provenance construction.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `...tensor.Tensor` remains the public fluent logits [facade](../../../../glossary.md#term-facade).

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/IndexCategoricalCrossEntropyWithLogitsAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorLossExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (12):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/loss/LossSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`

The nine inventory-only tests change only 187 to 188. The existing dense categorical test also
changes its count lock and may lock unchanged floating [dispatch](../../../../glossary.md#term-dispatch) plus floating-[target](../../../../glossary.md#term-target) rejection by
the ignore overload; it must receive no unrelated changes.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Training/[Runtime](../../../../glossary.md#term-runtime) APIs; related [loss](../../../../glossary.md#loss), indexing, [Shape](../../../../glossary.md#shape), [scalar](../../../../glossary.md#term-scalar),
promotion, [signature](../../../../glossary.md#term-signature), factory, [producer](../../../../glossary.md#term-producer), and [provenance](../../../../glossary.md#provenance) contracts; architecture/ADRs/tests;
[conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); dependencies; and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 23 paths: four production, twelve tests, and seven documentation/[planning](../../../../glossary.md#term-planning) paths. The
cohesive capability exceeds the normal 18-path guardrail under the user's standing higher-path
authorization because [dispatch](../../../../glossary.md#term-dispatch), one new type/overload/focused test, the dense focused [regression](../../../../glossary.md#term-regression)
test, nine inventory-only count locks, and mandatory documentation must remain consistent
atomically. Stop for path 24, another public type/method/test/document, foundation edit, later
task, cross-[module](../../../../glossary.md#term-module), architecture/[Gradle](../../../../glossary.md#term-gradle),
or unrelated cleanup. If repository evidence changes an inventory path, update this [Ready](../../../../glossary.md#term-ready) task
without exceeding 23 before implementation.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Document kind, attrs, helper paths, and both Tensor forms: dispatch, roles, Shape mapping, axis,
  ignore-before-bounds, denominator, empty/all-ignored, computation/special values, metadata,
  provenance/IDs, failures, and lifecycle boundaries.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Apply the [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc profile with complete `@param`, `@return`, and expected `@throws` text.
- [Tensor](../../../../glossary.md#tensor) API adds exact overload/[dispatch](../../../../glossary.md#term-dispatch)/reduction tables, `[1,2,3]` examples, dynamic Shape and
  bounds obligations, and current-model versus planned execution/training boundaries.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Compile API lists current model metadata only; capture, proof, constant analysis, bounds,
  gradients, decomposition, lowering, and execution remain planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [constant glossary](../../../../glossary.md#term-constant); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Review glossary terms categorical cross entropy, logits, index target, class axis, ignore index,
  sample domain, reduction, and non-ignored denominator.<span data-glossary-navigation="true"> ([categorical cross entropy glossary](../../../../glossary.md#categorical-cross-entropy); [index target glossary](../../../../glossary.md#index-target); [class axis glossary](../../../../glossary.md#class-axis); [ignore index glossary](../../../../glossary.md#ignore-index); [sample domain glossary](../../../../glossary.md#sample-domain))</span>
- Keep task/master/roadmap/capabilities synchronized; keep 0022/0022A Complete and 0023-0024 Draft.
- Record reasoned no-change conclusions for Training/Runtime APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, dependencies, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact three-kind order and exact attrs-class fixed 2/1 [signatures](../../../../glossary.md#term-signature); existing signatures unchanged.
- Exact attrs components, [validation](../../../../glossary.md#term-validation)/messages, reference retention, equality/[hash](../../../../glossary.md#term-hash), and no extras.
- Existing method dispatches floating to unchanged dense and [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) to no-[ignore index](../../../../glossary.md#ignore-index).
- Exactly one new [ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue) overload; public count 188; no public Optional.
- Exact `[logits,target]` [provenance](../../../../glossary.md#provenance), attrs-held ignore, one [producer](../../../../glossary.md#term-producer)/output-zero/ID, no primitives.
- Exact logits/[target](../../../../glossary.md#term-target)/ignore types, no target promotion/cast, exact logits result type.
- Exact [rank](../../../../glossary.md#term-rank)-minus-one [Shape](../../../../glossary.md#shape) mapping, exact target Shape [identity](../../../../glossary.md#term-identity-meanings) for `NONE`, no broadcast, scalar
  reductions, no-ignore versus ignore-present class-domain obligations, and class-size zero/one
  rules.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Exact ignore-before-bounds, stable formula, execution-time bounds obligation, non-ignored mean,
  empty/all-ignored sum/mean, computation, and special-value classes.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean))</span>
- Exact validation order/messages and no-ID local failures.
- Unresolved [layout](../../../../glossary.md#layout), logits-only [gradient](../../../../glossary.md#term-gradient) eligibility, no label/storage, inputs unchanged.
- Exact 23-path/package scope; no execution, gradient, broad loss option, architecture, build, or
  later-task work.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [loss glossary](../../../../glossary.md#loss))</span>
- Separate clean documentation pass and all validation/evidence complete before status Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.loss.LossSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMeanSquaredErrorExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

Focused coverage must distinguish static `C == 0` construction in all three cases: no-ignore with
definitely non-empty `S` fails locally without consuming an ID; ignore-present with the same
[metadata](../../../../glossary.md#term-metadata) constructs and retains the all-[targets](../../../../glossary.md#term-target)-ignored alternative; and a definitely empty `S`
constructs with or without ignore. It must also lock `C == 1`, dynamic class/non-class obligations,
ignore-before-bounds examples, and the empty/all-ignored reduction semantics without [eager](../../../../glossary.md#term-eager-evaluation) reads.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate Markdown links/anchors/fences/final newlines/trailing whitespace, official primary
references, examples, exact 23 paths, package placement, [signatures](../../../../glossary.md#term-signature)/count 188, synchronized status,
and absence of later specs. Reuse successful Java evidence unless executable Java changes.

After completion, the coordinator runs the master-plan [loss](../../../../glossary.md#loss)-family capability [checkpoint](../../../../glossary.md#term-checkpoint) (full
repository tests, affected architecture tests, and final docs/deferred cross-task checks).

## Dependencies

- 0001-0002: exact types, [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension), empty/dynamic Shapes, [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization).
- 0005-0007, 0011-0013: [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor), factory, [provenance](../../../../glossary.md#provenance), [identity](../../../../glossary.md#term-identity-meanings).
- 0016I-0016J: stable [log-softmax](../../../../glossary.md#softmax--log-softmax) [slice](../../../../glossary.md#slice) semantics.
- 0018K: exact [signatures](../../../../glossary.md#term-signature); 0018L: [producer](../../../../glossary.md#term-producer)/output-index provenance.
- 0018N: exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue); 0018O: index/bounds and no-value-inspection precedent.
- 0018U-0018U1: integral domains; 0018V: numerical/empty/special-value precedent.
- 0022: loss vocabulary/helper; 0022A: completed public method, dense semantics, axis removal, and
  class/sample-domain precedent.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>

All dependencies are Complete.

## Follow-up tasks

- 0023 later owns selected [compiler](../../../../glossary.md#term-compiler)-generated backward semantics, not [autograd](../../../../glossary.md#autograd) traversal.
- Compiler, prepare/runtime, backends, conformance, and training later own capture/proof, constant
  analysis, bounds enforcement, lowering, execution, and coordination.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [capture glossary](../../../../glossary.md#term-capture-meanings); [constant glossary](../../../../glossary.md#term-constant); [lowering glossary](../../../../glossary.md#lowering))</span>
- 0024 remains the final [model](../../../../glossary.md#term-model) capability-selection audit.

Do not create a 0023 or later detailed specification during this task.

## Architecture impact

Expected impact: None. Stop if implementation needs a new [dependency](../../../../glossary.md#term-dependency), [graph](../../../../glossary.md#graph) [identity](../../../../glossary.md#term-identity-meanings), hidden
state, execution status, cross-layer type, architecture update, or different family [ownership](../../../../glossary.md#term-ownership-meanings).

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0022b-index-target-categorical-cross-entropy-with-logits.md.
Implement that task exactly. Preserve the completed dense branch. Do not add weights, smoothing,
sparse targets, gradients/autograd, compiler/backend/runtime/training behavior, or later tasks.
Stop on architecture, dependency, affected-file, maximum-scope, or completed-contract conflict.
Do not commit or push unless separately authorized.

After Java implementation and recorded focused/final tests, hand the diff and evidence to a
separate clean documentation-focused context following documentation-rules.md. It must inspect
source/tests, finalize Javadocs, Tensor/Compile APIs, glossary/planning, run documentation/scope/
status checks, and reuse valid Java evidence. Fill this task's decisions, limitations, evidence,
notes, completion summary, and status; do not mark Complete before every criterion passes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide this task, final diff/test evidence, [dispatch](../../../../glossary.md#term-dispatch) and overload, attrs/[signature](../../../../glossary.md#term-signature), [Shape](../../../../glossary.md#shape) mapping,
ignore-before-bounds, no-ignore versus ignore-present zero-class obligations, formula/denominator/
empty/special values, [metadata](../../../../glossary.md#term-metadata)/[provenance](../../../../glossary.md#provenance)/IDs, architecture and 23-path limits, expected docs,
references, and [validation](../../../../glossary.md#term-validation). Require independent source/test inspection and reasoned no-change
conclusions.

## Local decisions

- Existing three-[argument](../../../../glossary.md#term-argument) [dispatch](../../../../glossary.md#term-dispatch) preserves the dense call and avoids a colliding index [signature](../../../../glossary.md#term-signature).
- One [ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue) overload exposes ignore; Optional remains internal attrs [metadata](../../../../glossary.md#term-metadata).
- Ignore comparison precedes bounds/[evaluation](../../../../glossary.md#term-evaluation); [mean](../../../../glossary.md#term-mean) counts only non-ignored [targets](../../../../glossary.md#term-target).
- Static zero-class rejection remains local only without ignore; ignore-present construction keeps
  the all-targets-ignored alternative deferred because it cannot read values.
- Result type and eligibility come only from logits.

## Known limitations

- No construction-time values/bounds, future execution failure status, gradient, compiler,
  backend, runtime, or training implementation.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Dynamic equality and no-ignore `S == 0 || C > 0` remain later binding obligations. Ignore-present
  construction instead retains <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`S == 0 || C > 0 || all targets equal ignoreIndex`, whose final
  alternative requires value inspection at a later boundary.
- Mathematical examples establish meaning, not [backend](../../../../glossary.md#backend) [tolerance](../../../../glossary.md#term-tolerance) or bitwise algorithm.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation Java evidence was reused exactly as required; no executable Java changed after
  it. The implementation context's exact focused 12-class Gradle command listed under<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
  [Tests / validation](#tests--validation) passed with `BUILD SUCCESSFUL in 997ms` and three
  actionable tasks (two executed, one up-to-date). Its one final
  `./gradlew :modules:model:test` passed with `BUILD SUCCESSFUL in 1s`, 966 tests across 124
  suites, and three actionable tasks (one executed, two up-to-date). The documentation context did
  not rerun either Java test command.
- Clean documentation context `/root/docs_0022b_retry` applied the General, API/Javadoc, Planning,
  and Example profiles. It independently inspected the architecture contract, planning rules,
  final production and focused tests, dense-loss precedent, ScalarValue/Shape/signature/factory/
  producer/provenance contracts, Tensor/Compile/Training APIs, glossary, capabilities, master plan,
  roadmap, and actual diff/status.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [architecture contract glossary](../../../../glossary.md#architecture-contract); [loss glossary](../../../../glossary.md#loss); [ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [Shape glossary](../../../../glossary.md#shape); [signature glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile))</span>
- `./gradlew :modules:model:javadoc` passed after the final [Javadoc](../../../../glossary.md#term-javadoc)-only edits with
  `BUILD SUCCESSFUL in 1s`; two actionable tasks executed. A prior run before the last
  behavior-equivalent Javadoc cleanup also passed in 1s. Generated pages contain the new kind,
  attributes, both Tensor overloads, component accessors, parameters, returns, exceptions, and
  semantic text.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- A corrected read-only Ruby documentation check passed all seven changed Markdown files for local
  link targets, required anchors, balanced fences, and final newlines. Two preliminary checker
  attempts failed because of the checker's own Ruby interpolation and GitHub-slug approximation,
  not a repository defect; the final targeted anchor checks passed for the new categorical,
  class-axis, index-target, ignore-index, sample-domain, and Tensor API headings and links.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [axis glossary](../../../../glossary.md#term-axis); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- `javap -public` reported exactly 188 declared public Tensor methods and exactly the two intended
  categorical overloads. It also confirmed the exact attributes constructor and `axis()`,
  `reduction()`, and `ignoreIndex()` accessors. Source checks confirmed the four production types'
  packages and exact two receiver declarations.
- Primary-reference inspection confirmed the focused Tensor API example links directly to official
  PyTorch `CrossEntropyLoss` and [ONNX](../../../../glossary.md#term-onnx) `SoftmaxCrossEntropyLoss` documentation. The `[1, 2, 3]`
  dense/index/ignore calculations were independently matched to the focused semantic tests and are
  explicitly labeled mathematical rather than eager evaluation.<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Final scope checks reported exactly 23 paths: four production, twelve tests, and seven
  documentation/planning files. Package placement matches the task map; <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`LossKind` order and fixed
  signatures remain test-locked; Tensor public count is 188; 0022, 0022A, and 0022B are Complete;
  0023–0024 remain Draft; and no 0023/0024 detailed task specification exists.
- `git diff --check` passed. Final changed Markdown files have balanced fences, final newlines, and
  no trailing whitespace. Read-only inspection commands (`git status`, `git diff`, `rg`, `sed`,
  `cat`, `wc`, generated-Javadoc searches, and exact-scope inventories) completed without a
  repository blocker.
- Training API and Runtime API need no change: this task adds model expression metadata and
  explicitly does not add training coordination, publication, prepared execution, or runtime
  behavior. Compile API changed only to distinguish current compiler-visible model metadata from
  still-planned capture, proof, bounds, gradients, decomposition, lowering, and execution.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [publication glossary](../../../../glossary.md#term-publication); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Architecture, focused architecture documentation, ADRs, and architecture tests need no change
  because module ownership and dependency direction are unchanged. Backend conformance and
  integration tests need no change because no backend or end-to-end execution exists. Shared dense
  loss, ScalarValue, Shape, promotion, signature, factory, producer/provenance, and related
  indexing contracts remain accurate and unchanged; this task consumes them without redefining
  them. Dependencies, Gradle/build structure, other modules, and later tasks likewise need no
  change.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added the index categorical kind, exact attributes record, fixed signature, target-type dispatch,
  one exact <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [target glossary](../../../../glossary.md#term-target); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) ignore overload, index Shape/metadata construction, and focused semantic,
  validation-order, provenance, identifier, inventory, and regression coverage.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [regression glossary](../../../../glossary.md#term-regression))</span>
- Preserved the completed dense branch after target-type dispatch. No eager value read, primitive
  decomposition, cast, one-hot producer, gradient rule, execution path, or lifecycle-layer
  implementation was added.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [decomposition glossary](../../../../glossary.md#term-decomposition); [producer glossary](../../../../glossary.md#term-producer); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The documentation pass finalized all four affected production Javadocs, replaced the dense-only
  Tensor API section with a newcomer-readable dense/index contract, expanded Compile API current-
  versus-planned boundaries, added reusable glossary distinctions, synchronized planning status,
  and made no executable Java behavior change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: implemented exact INT32/INT64 index-target categorical cross entropy from
  logits with optional exact typed ignore metadata, stable selected-class meaning, explicit
  reductions, and preserved dense dispatch.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [target glossary](../../../../glossary.md#term-target); [categorical cross entropy glossary](../../../../glossary.md#categorical-cross-entropy); [metadata glossary](../../../../glossary.md#term-metadata); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Files changed or created: exactly 23 authorized paths (four production, twelve tests, seven
  documentation/planning).<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the passed exact focused and final model tests; final model Javadoc,
  generated-page checks, Markdown/link/anchor/fence/newline checks, examples/reference checks,
  package/signature/public-count/status/order/no-later-spec checks, exact-scope checks, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [signature glossary](../../../../glossary.md#term-signature); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: clean context `/root/docs_0022b_retry`; Complete.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, master plan, and
  roadmap finalized. Training/Runtime APIs and architecture documentation require no change for
  the model-only reasons recorded above.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: LossKind, IndexCategoricalCrossEntropyWithLogitsAttrs, TensorLossExpressions, and
  both Tensor receiver forms finalized; generated Javadoc passed.
- Glossary impact: categorical cross entropy, class axis, dense/index target, ignore index, loss,
  logit, reduction, and sample-domain/non-ignored-denominator distinctions synchronized.<span data-glossary-navigation="true"> ([class axis glossary](../../../../glossary.md#class-axis); [index target glossary](../../../../glossary.md#index-target); [ignore index glossary](../../../../glossary.md#ignore-index); [loss glossary](../../../../glossary.md#loss); [logit glossary](../../../../glossary.md#logit); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Unresolved issues: None within task scope. Deferred execution/compiler/training obligations are
  intentional known limitations, not incomplete implementation.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Follow-up required: the recorded post-0022B capability checkpoint and established later tasks
  only; no follow-up is required to complete 0022B.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

Status: Complete
