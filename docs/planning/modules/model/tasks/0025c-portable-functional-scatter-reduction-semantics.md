# Task 0025C: Portable Functional-Scatter Reduction Semantics

**Editorial reading key (current context, not a status update).**

This historical task records documentation-only forward semantics, not numerical execution or
gradient policy. The caller invokes `scatterElements` or `scatterNd` on base `data`, supplying index
and update Tensor references plus attributes. Model construction records descriptions and ordered
provenance, not copied scalar values or executed reductions; the storage-free result owns no result
buffer. Model owns meaning, Compiler chooses derivative expressions, and execution providers realize
supported routes rather than redefine semantics. Semantic responsibility is not storage ownership.

A target group contains the base once and every addressed update once; duplicates remain distinct.
For example, base two and two updates of three give integral `MUL` eighteen. ND tuples apply this
grouping separately to each suffix-slice scalar; an untouched coordinate preserves its exact base
representation. Crucially, the historical once-rounded exact floating-product target differs from
the [current Scatter Elements
rules](../../../../reference/tensor/scatter.md#scatter-elements-expressions): follow these for
permitted rounded multiplication trees, FLOAT32 working arithmetic for low types, and
rounding/DAZ/FTZ boundaries. Overflow before multiplication by zero can yield NaN; this is not an
algebraic-equivalence promise. Current extrema still propagate NaN and distinguish zero signs, while
integral multiplication wraps at fixed width. None of these forward rules chooses derivative
behavior.

**Mathematical reading illustration.** At one integral target before overflow, let base $b=2$ and
distinct addressed updates $u_1=u_2=3$. Multiplication includes the base once:
$r=b\prod_{j=1}^{2}u_j=18$. Duplicate destinations do not deduplicate update contributions. The same
real-number product is not permission to use exact-product-then-one-rounding for floating execution;
the current rounded-tree contract and special-value boundaries remain separate.

## Status

Complete

## Goal

Close the smallest [model](../../../../glossary.md#term-model)-owned forward-semantics prerequisite needed before [Compiler](../../../../glossary.md#term-compiler) task 0005C
can plan [gradients](../../../../glossary.md#term-gradient) for configurable functional scatter.

Make the current `SCATTER_ELEMENTS` and `SCATTER_ND` contracts state one portable represented-value
meaning for `ScatterReduction.MUL`, `MIN`, and `MAX`. The contract must fix base-value
participation, duplicate-[target](../../../../glossary.md#term-target) grouping, empty update groups, fixed data-type behavior, floating
special values, and signed-integral [overflow](../../../../glossary.md#term-overflow) without selecting a traversal, [atomic](../../../../glossary.md#resource-transaction-and-rollback) schedule,
[backend](../../../../glossary.md#backend) algorithm, or [derivative](../../../../glossary.md#term-derivative) convention.

The semantic relationship is:

```text
data value plus every update addressed to one logical result coordinate
  -> one model-owned candidate group
  -> MUL, MIN, or MAX represented value in the unchanged data type

fixed forward meaning
  -> later Compiler 0005C chooses derivative formulas and boundary policy separately
```

This is expected to be [Javadoc](../../../../glossary.md#term-javadoc) and explanatory documentation only. It adds no evaluator,
[operation](../../../../glossary.md#operation), attribute, [Tensor](../../../../glossary.md#tensor) method, helper behavior, test-only numerical implementation,
compiler formula, backend behavior, or execution.

## [Scope](../../../../glossary.md#term-scope)

- Finalize `ScatterReduction` [Javadoc](../../../../glossary.md#term-javadoc) for the exact `MUL`, `MIN`, and `MAX` contracts below.
- Finalize the existing axis-scatter and Scatter-ND kind/attribute Javadocs so both configurable
  families use that same contract.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- Finalize the existing public `Tensor.scatterElements` and `Tensor.scatterNd` Javadocs and their
  package-private construction-helper Javadocs without changing executable construction.
- Clarify the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), [Compile](../../../../glossary.md#compile) API, and glossary atomically.
- Record this task as `Complete` in the [model](../../../../glossary.md#term-model) master plan and roadmap after [validation](../../../../glossary.md#term-validation) passes.
- Make Draft Compiler 0005C depend on Model 0025C without creating its detailed task file or
  promoting it.
- Record the downstream compiler planning decision that a future Compiler 0005C may construct one
  separate stable <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>`ARGSORT` Tensor occurrence with exactly matching sort attributes as the
  ordinary backward formula for one-output <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`SORT`.
- Preserve [Compiler](../../../../glossary.md#term-compiler) 0005D, 0005E, and 0006 as concise `Draft` rows without detailed task files.
- During implementation, synchronize this task, the model master plan, compiler master plan, and
  roadmap only after all documentation and validation acceptance criteria pass.

## Fixed forward semantic contract

### Logical [target](../../../../glossary.md#term-target) groups

For one valid functional-scatter [occurrence](../../../../glossary.md#term-occurrence) and one logical result [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `c`, let `U(c)` be
the logical multiset of update [scalar](../../../../glossary.md#term-scalar) values whose indices address `c`.

- The base value `data[c]` participates exactly once.
- Every logical update coordinate that addresses `c` contributes its update value exactly once.
- Duplicate targets therefore contribute multiple members, including when their represented
  values are equal.
- `SCATTER_ELEMENTS` forms the target coordinate by replacing the selected coordinate of each
  logical indices/updates coordinate with its index value.
- `SCATTER_ND` applies each index tuple to the indexed data prefix. When one tuple addresses a
  suffix slice, each logical scalar in the matching update slice contributes to the corresponding
  scalar result coordinate; duplicate tuples create duplicate contributions at every matching
  suffix coordinate.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- If `U(c)` is empty, the result at `c` is the exact unchanged representation of `data[c]`. A
  zero-element update domain therefore leaves every base representation unchanged.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>

For `MUL`, `MIN`, and `MAX`, the result is a symmetric reduction over
`{data[c]} multiset-union U(c)`. Logical update encounter order, physical [layout](../../../../glossary.md#layout), [strides](../../../../glossary.md#term-stride),
[backend](../../../../glossary.md#backend) traversal, parallel grouping, and [atomic](../../../../glossary.md#resource-transaction-and-rollback) scheduling are not observable. This is an exact
grouping rule, not an implementation-order rule. Implementations may reassociate or parallelize
only while reproducing the selected abstract result and future [conformance](../../../../glossary.md#term-conformance) requirements.

### Semantics [matrix](../../../../glossary.md#term-matrix-meanings)

| Reduction | [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16) | [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype) | Empty `U(c)` |
|---|---|---|---|
| `MUL` | Abstract product in the unchanged result format, with the floating rules below | Exact-width [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) product modulo `2^32` or `2^64` | Exact base representation |
| `MIN` | [NaN](../../../../glossary.md#term-nan)-propagating numeric minimum; opposite signed zeros select [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) | Ordinary signed minimum | Exact base representation |
| `MAX` | NaN-propagating numeric maximum; opposite signed zeros select [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) | Ordinary signed maximum | Exact base representation |

[`BOOL`](../../../../glossary.md#data-type--datatype) remains ineligible for arithmetic scatter reductions under the current public construction
contract. Data and updates already have the same exact [`DataType`](../../../../glossary.md#data-type--datatype); this task adds no promotion,
widened output, [saturation](../../../../glossary.md#term-saturation), conversion, or accumulation attribute.

### Floating multiplication

For a non-empty update group, floating `MUL` follows the existing portable product meaning used by
ordinary reductions and cumulative product:

- any [NaN](../../../../glossary.md#term-nan) factor produces NaN, without a payload, sign, signaling, or source-factor promise;
- a group containing both any zero and any [infinity](../../../../glossary.md#term-infinity) produces NaN;
- otherwise, an infinity produces infinity and a zero produces zero, with sign determined by the
  parity of negative factors, including negative zero and negative infinity;<span data-glossary-navigation="true"> ([negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- otherwise, finite non-zero factors denote their exact mathematical product, rounded to the
  unchanged result format with round-to-nearest, ties-to-even;<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
- finite overflow produces signed infinity, while underflow, subnormal results, and signed zero
  follow that result-format rounding and multiplication sign; and<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [signed zero glossary](../../../../glossary.md#term-signed-zero); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- regrouping and an equal-or-wider implementation intermediate are lawful only when they conform
  to this abstract target and the future backend-conformance tolerance. Narrower accumulation,
  saturation, a model-visible promoted type, a fixed instruction sequence, NaN payload
  preservation, and a bitwise cross-backend algorithm are not selected.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [tolerance glossary](../../../../glossary.md#term-tolerance); [saturation glossary](../../../../glossary.md#term-saturation); [model glossary](../../../../glossary.md#term-model))</span>

[FLOAT64](../../../../glossary.md#term-float64) and [FLOAT32](../../../../glossary.md#term-float32) use their represented [IEEE](../../../../glossary.md#term-ieee)-754 [binary64](../../../../glossary.md#floating-point-precision-and-rounding) and [binary32](../../../../glossary.md#floating-point-precision-and-rounding) values and result
formats. [BFLOAT16](../../../../glossary.md#term-bfloat16) uses the value represented by its exact current 16-bit storage and rounds the
abstract result to BFLOAT16 with the existing [round-to-nearest, ties-to-even](../../../../glossary.md#floating-point-precision-and-rounding) conversion contract.

The exact-base rule for an empty `U(c)` is stronger than applying a synthetic multiplicative
[identity](../../../../glossary.md#term-identity-meanings): no arithmetic occurs, so an existing [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) or other base representation is not
canonicalized merely because no update addresses that [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix).

### Floating minimum and maximum

For a non-empty update group, floating `MIN` and `MAX` reuse completed [Model](../../../../glossary.md#term-model) 0025A's
represented-value [extrema](../../../../glossary.md#saturation-pole-and-extremum) meaning across the entire candidate group:

- if any candidate is NaN, the result is NaN, without a payload, sign, source-candidate, or
  bitwise-result promise;
- if both zero signs occur, `MIN` produces [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) and `MAX` produces positive zero,
  independent of how many occurrences contribute;<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) and unequal non-[NaN](../../../../glossary.md#term-nan) values use ordinary numeric order; and
- equal non-zero candidates produce that numeric value without promising which candidate's
  representation or identity is selected.

No bounded empty [identity](../../../../glossary.md#term-identity-meanings) is used because the base candidate is always present. When no update
addresses a [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), the exact-base rule applies instead.

### Signed-integral behavior

- [INT32](../../../../glossary.md#data-type--datatype) `MUL` is multiplication modulo `2^32`; [INT64](../../../../glossary.md#data-type--datatype) `MUL` is multiplication modulo `2^64`.
- Overflow wraps in the unchanged fixed width. It does not throw, saturate, widen, or become
  undefined.
- Modular multiplication is independent of candidate order and grouping.
- INT32 and INT64 `MIN`/`MAX` use ordinary signed order and cannot [overflow](../../../../glossary.md#term-overflow).
- Empty update groups preserve the exact base bit pattern rather than applying an [identity](../../../../glossary.md#term-identity-meanings).

### `NONE` and `ADD` no-change boundary

This task does not redefine the completed replacement or addition contracts:

- `NONE` still replaces an addressed base with its single exact update representation and requires
  unique targets. Duplicate targets remain invalid rather than first-write or last-write ordered.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `ADD` still combines the base and every addressed update, including duplicate accumulation.
  Existing fixed-width modular signed-integral addition and permitted floating reassociation/no
  bitwise-order guarantee remain unchanged.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- `SCATTER_ADD` remains the separate fixed-add Gather-compatible operation. This task changes
  neither its Shape relation nor its represented-value contract.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [operation glossary](../../../../glossary.md#operation); [Shape glossary](../../../../glossary.md#shape))</span>

No new general update order is introduced for `NONE` or `ADD`.

## Out of [scope](../../../../glossary.md#term-scope)

- any Java declaration, import, annotation, executable statement, validation branch, failure
  message, operation kind, signature, attribute component, Tensor method, helper method, data type,
  promotion rule, Shape rule, descriptor, provenance, or test behavior<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `SCATTER_ADD` semantics, `NONE` uniqueness, `ADD` numerical policy, index bounds, negative-index
  policy, duplicate detection, or value-aware validation
- a numerical-policy type, registry, service, selectable mode, generic evaluator, backend utility,
  runtime utility, or test-only fake evaluator<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- derivative formulas or policies for product zeros, infinities, NaNs, extrema ties, signed zero,
  duplicate sharing, discontinuities, or invalid forward occurrences<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- hidden sort outputs, a public sort result [carrier](../../../../glossary.md#term-storage-carrier), a multi-output `SORT`, changes to
  `Tensor.sort`/`argsort`, [producer](../../../../glossary.md#term-producer)-output changes, or [compiler](../../../../glossary.md#term-compiler) implementation
- graph capture, inference, canonicalization, constant folding, planning, lowering, execution,
  backend conformance implementation, runtime, prepare, engine, training, ONNX, or trace work<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [constant folding glossary](../../../../glossary.md#term-constant-folding); [planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [conformance glossary](../../../../glossary.md#term-conformance); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [trace glossary](../../../../glossary.md#trace))</span>
- changing `ARCHITECTURE.md`, an ADR, focused architecture documentation, module boundaries,
  dependencies, Gradle, Java 26 configuration, architecture tests, backend-conformance tests,
  integration tests, capabilities/audits, or another module's source/tests<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- creating or promoting Compiler 0005C, 0005D, 0005E, or 0006 task specifications
- unrelated documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model-owned operation
  semantics, compiler-owned gradient rules, and backend-owned execution<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Model capability and contract closure audit](../model-capability-contract-closure-audit.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>, especially I5-I9 and O1-O4
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Task 0018I](0018i-scatter-nd-semantics.md)
- [Task 0018J](0018j-scatter-nd-tensor-expression.md)
- [Task 0019C](0019c-sort-and-argsort.md)
- [Task 0025A](0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)
- [Compiler master plan](../../compiler/master-plan.md)
- [Completed Compiler task 0005A](../../compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Completed Compiler task 0005B](../../compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-independent forward operation semantics. Recording the fixed
  represented-value meaning in existing scatter semantics and public API documentation is inside
  the current architecture.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The model layer does not evaluate Tensor values. No executable helper may be added merely to
  make these rules unit-testable.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Compiler owns derivative policy and gradient-expression construction. This task must not select
  a cotangent formula or imply a policy for zeros, infinities, NaNs, extrema ties, duplicate
  targets, or discontinuities.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [derivative glossary](../../../../glossary.md#term-derivative); [gradient glossary](../../../../glossary.md#term-gradient); [cotangent glossary](../../../../glossary.md#cotangent); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [targets glossary](../../../../glossary.md#term-target))</span>
- Forward and generated gradient expressions later share this same model numerical contract.
  Compiler 0005C may choose local derivative conventions but may not create a separate forward
  scatter algebra.<span data-glossary-navigation="true"> ([local derivative glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Concrete backends later own evaluation. Backend-conformance tests must check the selected
  grouping and represented-value behavior when execution exists; this task implements no backend.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The order-independent logical group is necessary portability, not a prescribed implementation
  traversal. No storage layout, atomic ordering, sequential fold, tree shape, or kernel route is
  part of model semantics.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [shape glossary](../../../../glossary.md#shape); [kernel glossary](../../../../glossary.md#kernel))</span>
- Existing `SCATTER_ELEMENTS` and `SCATTER_ND` kinds, attributes, Tensor methods, validation,
  descriptors, provenance, and data-type eligibility remain exact.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- The downstream stable-ARGSORT decision is compiler planning only. It does not change Model
  0019C's separate one-output <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>`SORT` and `ARGSORT` occurrences or require a hidden forward output.
- If accurate implementation requires executable semantics, another type/member, changed
  validation, a derivative decision, or an architecture change, stop and report the exact
  conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.index` — owns shared scatter reduction vocabulary,
  axis-scatter semantics, Scatter-ND semantics, and their immutable attributes.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.model.tensor` — owns public Tensor method contracts and metadata-only
  expression construction.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata))</span>

Packages added, moved, or removed:

- None.

Type placement:

- No type is created or moved.
- Numerical meaning remains in `ScatterReduction` and the existing operation-family/public Tensor
  Javadocs. A new numerics package or policy object would create an unneeded abstraction and is
  forbidden.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## Affected files

Expected production Java, [Javadoc](../../../../glossary.md#term-javadoc)-only:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterReduction.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterElementsAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScatterNdExpressions.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification:

- [`DataType`](../../../../glossary.md#data-type--datatype), `BFloat16Bits`, `TensorDescriptor`, `TensorFactory`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer),
  `TensorProvenance`, [`Operation`](../../../../glossary.md#operation), and `OperationSignature` — their representations,
  construction, identity, and occurrence contracts remain exact.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Axis-scatter and Scatter-ND semantic/expression tests — they lock declarations, reduction
  eligibility, validation, descriptor, provenance, and non-execution behavior. They do not
  evaluate Tensor values and must not gain a fake evaluator.<span data-glossary-navigation="true"> ([Axis glossary](../../../../glossary.md#term-axis); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `AggregateReductionKind`, `CumulativeScanKind`, binary/scalar extrema, and their documentation —
  they provide the existing product/extrema meaning reused here and do not change.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- `OrderingKind`, `SortAttrs`, `TensorSortExpressions`, and sort tests — the downstream compiler
  planning decision consumes their existing stable one-output contracts without modifying them.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Model capabilities and both completed audits — historical selection and audit results remain
  unchanged; this focused prerequisite finalizes directly consumed Javadocs and APIs.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Training and Runtime APIs — no training request, optimizer, prepared state, runtime, or
  execution contract changes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture documents, ADRs, and architecture tests — ownership and
  dependency rules do not change.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Gradle](../../../../glossary.md#term-gradle) files — Java 26 and current [module](../../../../glossary.md#term-module) dependencies remain exact.
- Backend-conformance and integration tests — no backend or end-to-end execution exists in this
  task; future executable implementations must receive conformance coverage.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact fifteen paths listed under expected production Java and expected documentation
and [planning](../../../../glossary.md#term-planning) may change: eight [Javadoc](../../../../glossary.md#term-javadoc)-only Java paths and seven documentation/planning paths.

No test, [Gradle](../../../../glossary.md#term-gradle), architecture, capability/audit, sort Java, other Java, other [module](../../../../glossary.md#term-module),
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration path may change. If another path is required for accuracy,
stop and request a replanning decision rather than expanding scope.

## Acceptance criteria

- `MUL`, `MIN`, and `MAX` define one candidate group containing the base exactly once and every
  addressed logical update exactly once.
- Duplicate [target](../../../../glossary.md#term-target) entries remain distinct contributions, including equal represented values.
- [Scatter Elements](../../../../glossary.md#scatter-elements) and [Scatter-ND](../../../../glossary.md#scatter-nd) suffix-[slice](../../../../glossary.md#slice) grouping are both explicit.
- Empty update groups preserve the exact base representation without applying an identity,
  rounding, or NaN canonicalization.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [rounding glossary](../../../../glossary.md#term-rounding); [NaN glossary](../../../../glossary.md#term-nan); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Group meaning is independent of logical encounter order, physical layout, backend traversal,
  parallel grouping, or atomic schedule.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [backend glossary](../../../../glossary.md#backend); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16) `MUL` state exact product-format, rounding, overflow, underflow,
  zero, signed-zero, infinity, and NaN rules without selecting an executable algorithm or NaN
  payload.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) `MUL` uses unchanged-width [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) modular multiplication; integral
  `MIN`/`MAX` use signed order.
- Floating `MIN`/`MAX` reuse Model 0025A's NaN propagation, signed-zero selection, infinity order,
  and equal-candidate boundary.
- `NONE`, `ADD`, `SCATTER_ADD`, index bounds, negative indices, and duplicate detection retain
  their completed boundaries without a new order or behavior.
- No derivative formula, tie/subgradient sharing, zero-product case split, infinity/NaN
  convention, or generated compiler expression is selected.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>
- Every Java change is Javadoc-only. Declarations, imports, annotations, enum order, signatures,
  record components, executable statements, validation, messages, and bytecode-visible API remain
  unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [bytecode glossary](../../../../glossary.md#term-bytecode); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No evaluator, numerical-policy object, operation, attribute, Tensor method, helper behavior,
  data type, test change, backend behavior, or execution is added.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Tensor API, Compile API, glossary, task, model master plan, compiler master plan, and roadmap use
  consistent current/planned wording.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Compiler](../../../../glossary.md#term-compiler) [planning](../../../../glossary.md#term-planning) explicitly permits one separately constructed stable `ARGSORT` occurrence for
  one-output <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`SORT` backward only when kind/signature/axis/direction match exactly. It does not add
  a hidden sort output or public API. A missing or mismatched signature/attribute path must fail
  closed before partial backward construction.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- After implementation, [Model](../../../../glossary.md#term-model) 0025C is `Complete`; Compiler 0005C remains `Draft` without a task
  file and depends on Model 0025C plus Compiler 0005B; Compiler 0005D, 0005E, and 0006 remain
  `Draft` without task files.
- Exactly the fifteen authorized paths change. Markdown links/anchors, fences, final newlines,
  trailing whitespace, status/dependency consistency, and <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`git diff --check` pass.

## Tests / [validation](../../../../glossary.md#term-validation)

This task changes semantic contracts in Javadocs and explanatory documentation but no executable
Java behavior. One clean documentation-focused implementation context is the required
documentation pass. It must read the final source and tests directly and finalize the Javadocs
and documentation itself; do not create a redundant second documentation context merely to repeat
that work.

[Run](../../../../glossary.md#term-run) one final [model](../../../../glossary.md#term-model) [compile](../../../../glossary.md#compile)/test/[Javadoc](../../../../glossary.md#term-javadoc) command after all Javadocs are stable:

```bash
./gradlew :modules:model:compileJava :modules:model:test :modules:model:javadoc
```

The model tests are [regression](../../../../glossary.md#term-regression) evidence for unchanged declarations, attributes, validation,
public surfaces, [descriptors](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), and expression construction. They are not evidence that
[Tensor](../../../../glossary.md#tensor) values were evaluated. No focused numerical execution test is appropriate because Model
has no evaluator and [immutable](../../../../glossary.md#term-immutable) declaration members do not change.

Then run:

```bash
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

If the temporary Markdown validator is absent, create an equivalent validator outside the
repository. It must check repository-local [targets](../../../../glossary.md#term-target) and heading anchors, balanced fences, final
newlines, and trailing whitespace.

Required manual/source checks:

- inspect every Java diff hunk and confirm that only Javadoc comments changed;
- mechanically compare the eight Java files with Javadoc removed and confirm all remaining bytes
  are unchanged;
- inspect generated Javadoc for all affected public index-[operation](../../../../glossary.md#operation) types and Tensor methods;
- inspect the two package-private helper Javadocs in source;
- verify the exact fifteen-path ceiling and absence of test/[Gradle](../../../../glossary.md#term-gradle)/architecture/capability changes;
- verify Model 0025C status and [Compiler](../../../../glossary.md#term-compiler) 0005C [dependency](../../../../glossary.md#term-dependency)/status in all three [planning](../../../../glossary.md#term-planning) indexes;
- verify no Compiler 0005C–0005E or 0006 task file exists and that those rows remain Draft;
- verify the compiler master plan and roadmap contain the stable-ARGSORT planning decision, exact
  match guard, and fail-closed boundary without changing Model sort semantics;<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- verify no evaluator, numerical-policy type, registry, selectable mode, operation, attribute,
  Tensor method, data type, test, derivative convention, compiler implementation, backend
  behavior, dependency, or build change was added; and<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [data type glossary](../../../../glossary.md#data-type--datatype); [derivative glossary](../../../../glossary.md#term-derivative); [backend glossary](../../../../glossary.md#backend))</span>
- verify terminology, tables, links, anchors, fences, final newlines, trailing whitespace, and
  `git diff --check`.

Repository-wide and architecture validation are deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because this is one [module](../../../../glossary.md#term-module)'s
Javadoc plus explanatory/planning documentation and changes no executable behavior, dependency,
build configuration, or architecture boundary.

## Dependencies

- [Model](../../../../glossary.md#term-model) 0018G–0018J functional scatter semantics and public expression construction — Complete.
- Model 0018U–0018U1 integral [modular arithmetic](../../../../glossary.md#term-modular-arithmetic) and signed [extrema](../../../../glossary.md#saturation-pole-and-extremum) — Complete.
- Model 0019C stable separate one-output sort/argsort — Complete.
- Model 0025A portable floating comparison/extrema contract — Complete.
- Compiler 0005B reduction/scan/normalization gradient completion — Complete; Compiler 0005C is
  the downstream consumer whose planning remains blocked until this task is implemented.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [scan glossary](../../../../glossary.md#term-cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization); [gradient glossary](../../../../glossary.md#term-gradient); [planning glossary](../../../../glossary.md#term-planning))</span>

## Follow-up tasks

- Compiler 0005C — after Model 0025C is Complete, separately plan and implement compiler-owned
  layout/window/indexing/scatter/ordering/stochastic gradient completion. It must select every
  scatter derivative boundary and duplicate-sharing convention. For one-output <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative))</span>`SORT`, it may
  construct one separate stable `ARGSORT` Tensor occurrence with the exact same input, normalized
  axis, and direction, then route the cotangent through unique-target <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [occurrence glossary](../../../../glossary.md#term-occurrence); [normalized axis glossary](../../../../glossary.md#normalized-axis); [cotangent glossary](../../../../glossary.md#cotangent); [target glossary](../../../../glossary.md#term-target))</span>`SCATTER_ELEMENTS/NONE`.
  Preflight must fail closed if the exact SORT or ARGSORT signature/attributes cannot be matched.<span data-glossary-navigation="true"> ([Preflight glossary](../../../../glossary.md#term-preflight); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [signature glossary](../../../../glossary.md#term-signature))</span>
- Future concrete backends — when configurable scatter execution is implemented, add or update
  backend-conformance tests for duplicate groups, exact base preservation, zero-size update
  domains, floating zero signs/infinities/NaNs/overflow/underflow, modular integral overflow, and
  extrema signed-zero/NaN cases.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns forward [operation](../../../../glossary.md#operation) semantics to [Model](../../../../glossary.md#term-model), [gradient](../../../../glossary.md#term-gradient) rules to
[Compiler](../../../../glossary.md#term-compiler), and concrete [evaluation](../../../../glossary.md#term-evaluation) to [backends](../../../../glossary.md#backend). This task completes existing documentation without
changing [ownership](../../../../glossary.md#term-ownership-meanings) or dependencies.

If implementation requires an architecture change, executable model evaluation, another [module](../../../../glossary.md#term-module),
or a [derivative](../../../../glossary.md#term-derivative) decision, stop and report the exact conflict.

## Implementation prompt

Use this prompt in one separate clean documentation-focused task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, the focused model/compiler/module/dependency/lifecycle/backend
documents, documentation rules plus General, API/Javadoc, Planning, and Example profiles,
planning guide and roadmap, model capabilities/master plan and both completed audits, completed
Model 0018G-0018J/0018U-0018U1/0019C/0025A, completed Compiler 0005A/0005B, compiler master plan,
this task, all eight affected production files, the review-only helpers/tests, Tensor/Compile/
Training/Runtime APIs, glossary, and Java 26/module Gradle boundaries.

Implement Model 0025C exactly within its fifteen authorized paths. Change Java only in Javadoc.
Finalize the portable SCATTER_ELEMENTS and SCATTER_ND MUL/MIN/MAX represented-value semantics
without adding an evaluator, policy object, operation, attribute, Tensor method, test change,
derivative convention, compiler implementation, backend behavior, or execution. Preserve NONE,
ADD, SCATTER_ADD, value-aware index validation, and every declaration/behavior. Stop on any
scope, behavior, completed-contract, or architecture conflict.

This clean documentation-focused context itself is the required documentation pass; do not create
a redundant second pass. Run the specified model compile/test/Javadoc and documentation/scope/
status checks, inspect generated Javadoc, record exact evidence and no-change conclusions, then
mark Model 0025C Complete only if every acceptance criterion passes. Keep Compiler 0005C-0005E
and 0006 Draft without task files. Do not commit or push.
```

## Local decisions

- A target group is a logical multiset, not a traversal sequence. Base participates once and each
  addressed update participates once; duplicate targets remain distinct contributions.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- `MUL`, `MIN`, and `MAX` are symmetric abstract reductions, so storage layout, encounter order,
  atomic order, and tree grouping are not model-visible.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [model glossary](../../../../glossary.md#term-model))</span>
- Floating `MUL` reuses the existing product target rather than inventing sequential intermediate
  rounding. Integral <span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding))</span>`MUL` reuses fixed-width modular meaning.
- Floating extrema reuse Model 0025A across the complete candidate group. No identity is needed
  because every group contains its base.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- An empty update group bypasses arithmetic and preserves the exact base representation.
- `NONE`, `ADD`, `SCATTER_ADD`, bounds, negative-index policy, and duplicate detection do not
  change.
- Stable `ARGSORT` recomputation is a future compiler formula choice, not a hidden Model output or
  public sort API change.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Known limitations

- Index bounds, negative-index policy, and value-aware duplicate validation remain separately
  owned and unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- The completed contract will specify a mathematical floating target and exact special-value
  classes, not a sequential algorithm, NaN payload, instruction sequence, or universal bitwise
  backend implementation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend))</span>
- Compiler 0005C remains Draft and must still choose derivative routing at zeros, infinities,
  NaNs, extrema ties, duplicate targets, and ordering discontinuities.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [derivative glossary](../../../../glossary.md#term-derivative); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Numerical execution cannot be verified until a concrete backend exists. Future implementations
  require backend-conformance coverage.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/plan_model_scatter_semantics` read the required architecture, focused
architecture, documentation profiles, planning guide/roadmap, [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) plans, completed
scatter/[gather](../../../../glossary.md#gather)/indexing/[extrema](../../../../glossary.md#saturation-pole-and-extremum)/sort tasks, both Model audits, current APIs/glossary, production
contracts, helpers, and tests. It found no architecture or completed-contract conflict.

Planning-stage validation:

- A targeted Markdown validator passed for this task, both master plans, and the roadmap: four
  files and 550 repository-local links, including heading anchors, with balanced fences, final
  newlines, and no trailing whitespace.
- `git diff --check` passed.
- The union of tracked and untracked paths contains exactly this task, the Model master plan, the
  Compiler master plan, and the roadmap. No Java, test, Gradle, architecture, API, glossary,
  capability/audit, or other-module path changes in this planning-only diff.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module path glossary](../../../../glossary.md#term-module-path))</span>
- The future implementation scope contains exactly fifteen paths: eight existing Java files with
  Javadoc-only permission and seven API/glossary/planning files. It contains no test, Gradle,
  architecture, capability/audit, sort Java, or other-module path.
- Exact-status checks found this task as the only detailed task whose status line is [`Ready`](../../../../glossary.md#term-ready).
  Model/roadmap indexes agree; Compiler 0005C remains a Draft row dependent on Model 0025C and
  Compiler 0005B. Compiler 0005D, 0005E, and 0006 remain Draft.
- The Compiler task directory contains specifications only through completed 0005B. No 0005C,
  0005D, 0005E, or 0006 task file exists.
- The compiler master plan and roadmap both record the matching stable-ARGSORT allowance, reject
  hidden SORT outputs and public API changes, and require fail-closed exact kind/signature/axis/
  direction matching before partial backward construction.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [signature glossary](../../../../glossary.md#term-signature); [axis glossary](../../../../glossary.md#term-axis))</span>

No Java test or [Javadoc](../../../../glossary.md#term-javadoc)-generation command ran because this change creates planning only and does
not implement task 0025C.

Implementation context `/root/implement_model_0025c_scatter` independently read the governing
architecture, documentation profiles, planning contracts, completed related tasks and audits,
current APIs, exact affected source, helpers, tests, generated Javadoc, and Java 26 build boundary.
It found no architecture, behavior, or [scope](../../../../glossary.md#term-scope) conflict and completed the required documentation
pass without a second implementation context.

Implementation-stage validation:

- `./gradlew :modules:model:compileJava :modules:model:test :modules:model:javadoc` passed under
  OpenJDK 26.0.1. The Model suite reported 127 suites and 1,019 tests with no skips, failures, or
  errors.
- Generated Javadoc was inspected for all five public index-operation types and the affected
  public <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`Tensor.scatterElements`/`scatterNd` methods. Both package-private construction-helper
  Javadocs were inspected directly in source.
- Removing every Javadoc block mechanically from each of the eight before/after Java files
  produced identical remaining bytes. Independent `javac --release 26 -g:none` compilation of
  those before/after sources produced byte-for-byte identical class trees; declarations, imports,
  annotations, enum order, signatures, record components, validation, messages, and executable
  behavior are unchanged.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- The Markdown validator passed all 12 affected-or-linked Markdown files and 714 repository-local
  links, including heading anchors, balanced fences, final newlines, and trailing whitespace.
- The tracked/untracked union contains exactly the fifteen authorized paths: eight Javadoc-only
  Java files and seven API/glossary/planning files. No test, Gradle, architecture, capability/audit,
  sort Java, conformance/integration, other-module, or other path changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>
- Status/dependency checks agree that Model 0025C is Complete and Compiler 0005C remains Draft
  dependent on Model 0025C and Compiler 0005B. Compiler 0005C–0005E and 0006 have no task files and
  remain Draft. Compile API, compiler master plan, and roadmap retain the exact-match stable-
  ARGSORT allowance and fail-closed preflight boundary without a hidden SORT output or API change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Compile glossary](../../../../glossary.md#compile); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- `git diff --check` passed. No evaluator, numerical-policy type, registry, selectable mode,
  operation, attribute, Tensor method, data type, derivative convention, compiler implementation,
  backend behavior, dependency, or build change was added.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [Tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype); [derivative glossary](../../../../glossary.md#term-derivative); [backend glossary](../../../../glossary.md#backend))</span>

## Implementation notes

The shared `ScatterReduction` [Javadoc](../../../../glossary.md#term-javadoc) now owns the complete [target](../../../../glossary.md#term-target)-group and represented-value
contract. [Axis](../../../../glossary.md#term-axis)-scatter, [Scatter-ND](../../../../glossary.md#scatter-nd), public [Tensor](../../../../glossary.md#tensor), and package-private helper Javadocs specialize
that contract only for their [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and suffix-[slice](../../../../glossary.md#slice) mappings. Tensor [API](../../../../glossary.md#term-application-programming-interface-api) provides the full
reader-facing floating/integral [matrix](../../../../glossary.md#term-matrix-meanings); glossary names the [functional-scatter target group](../../../../glossary.md#functional-scatter-target-group);
[Compile](../../../../glossary.md#compile) API records preservation obligations and the future stable-ARGSORT allowance. No in-[scope](../../../../glossary.md#term-scope)
decision refinement changed the fixed task contract.

## Completion summary

Completed the portable functional-scatter represented-value contract for configurable
`SCATTER_ELEMENTS` and `SCATTER_ND` `MUL`, `MIN`, and `MAX`. Base participation, distinct
duplicate contributions, suffix-[slice](../../../../glossary.md#slice) [scalar](../../../../glossary.md#term-scalar) grouping, exact empty-group preservation, floating
product/[extrema](../../../../glossary.md#saturation-pole-and-extremum) edge cases, modular integral multiplication, signed extrema, and order
independence are now explicit without executable [Model](../../../../glossary.md#term-model) [evaluation](../../../../glossary.md#term-evaluation) or [derivative](../../../../glossary.md#term-derivative) policy.

Files changed: the exact eight [Javadoc](../../../../glossary.md#term-javadoc)-only Java files and seven documentation/[planning](../../../../glossary.md#term-planning) files
listed under [Affected files](#affected-files).

[Validation](../../../../glossary.md#term-validation): the final Model [compile](../../../../glossary.md#compile)/test/Javadoc command passed with 127 suites and 1,019 tests;
generated and source Javadocs were inspected; mechanically Javadoc-stripped source and no-debug
[bytecode](../../../../glossary.md#term-bytecode) were identical before/after; Markdown with 714 local links, exact fifteen-path [scope](../../../../glossary.md#term-scope),
status/[dependency](../../../../glossary.md#term-dependency)/spec-absence checks, Java 26/build boundaries, and `git diff --check` passed.

Documentation review: all affected Javadocs, [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), Compile API, glossary, this task, both
master plans, and roadmap required updates. Model capabilities/audits, Training/[Runtime](../../../../glossary.md#term-runtime) APIs,
related aggregate/[scan](../../../../glossary.md#term-cumulative-scan)/extrema/ordering contracts, architecture/ADRs/tests, [Gradle](../../../../glossary.md#term-gradle), dependencies,
other [modules](../../../../glossary.md#term-module), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration required no change because this task changes
only the existing Model forward documentation contract.

Unresolved issues: None within Model task 0025C.

Required follow-up: Draft [Compiler](../../../../glossary.md#term-compiler) 0005C must separately choose scatter, ordering, [layout](../../../../glossary.md#layout)/window,
and stochastic derivative policies while preserving this forward contract and the documented
stable-ARGSORT [fail-closed](../../../../glossary.md#term-fail-closed) planning boundary. Future executable backends require conformance
coverage.

Status: Complete
