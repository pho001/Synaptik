# Task 0018Q: [Masked Reduction](../../../../glossary.md#masked-reduction) Redesign

## Status

Complete

## Goal

Replace the provisional masked-reduction [axis](../../../../glossary.md#term-axis) mapper with one explicit, ordinary [broadcasting](../../../../glossary.md#broadcasting)
contract:

```text
caller mask Shape
  -> ordinary right-aligned broadcast to the input Shape
  -> one first-class masked SUM or MEAN occurrence
  -> remove the normalized reduction axis
```

Keep the public axis-removing `Tensor.sum(axis, mask)` and `Tensor.mean(axis, mask)` methods, but
make them construct one first-class two-input aggregate [occurrence](../../../../glossary.md#term-occurrence). They are not decompositions
into `where`, cast, multiplication, ordinary reduction, or division. Simplify
`MaskedReductionAttrs` to the smallest semantic [parameter](../../../../glossary.md#term-parameter) needed to distinguish that occurrence:
one normalized non-negative reduction axis and no [mask](../../../../glossary.md#term-boolean-mask)-axis mapping.

Masked-out values are excluded before aggregation, including masked-out [NaN](../../../../glossary.md#term-nan) and [infinity](../../../../glossary.md#term-infinity) values.
Masked sum over no selected values is the floating additive [identity](../../../../glossary.md#term-identity-meanings). Masked [mean](../../../../glossary.md#term-mean) over no
selected values is NaN, because the arithmetic mean of an empty selected set is undefined. The
task records those meanings in [model](../../../../glossary.md#term-model) contracts without implementing value [evaluation](../../../../glossary.md#term-evaluation), [gradients](../../../../glossary.md#term-gradient),
[compiler](../../../../glossary.md#term-compiler) [decomposition](../../../../glossary.md#term-decomposition), [lowering](../../../../glossary.md#lowering), or execution.

## [Mental model](../../../../glossary.md#term-mental-model)

The [mask](../../../../glossary.md#term-boolean-mask) must already say which input [axes](../../../../glossary.md#term-axis) it addresses through ordinary right [alignment](../../../../glossary.md#term-alignment).
Synaptik does not guess:

```text
input:                  [batch, time, features]
caller mask:                    [time]          addresses features, not time
explicitly expanded mask: [batch, time, 1]      addresses batch and time
```

For a `[batch, time]` mask intended to select values along the first two input axes, the caller
uses the existing [rank](../../../../glossary.md#term-rank)-editing [API](../../../../glossary.md#term-application-programming-interface-api) before the reduction:

```java
Tensor alignedMask = mask.expandDims(2); // [batch, time, 1]
Tensor result = input.mean(1, alignedMask);
```

The resulting [operation](../../../../glossary.md#operation) remains one masked [`MEAN`](../../../../glossary.md#term-mean) with exact [producer](../../../../glossary.md#term-producer) inputs `[input,
alignedMask]`. The expansion is explicit [provenance](../../../../glossary.md#provenance) rather than a hidden alignment decision
inside the reduction helper.

## Current problems

- `MaskedReductionAttrs(axis, maskInputAxes)` stores a bespoke ordered injection from every mask
  dimension to selected input axes. That representation is neither ordinary broadcasting nor a
  caller-visible reshape.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [dimension glossary](../../../../glossary.md#dimension); [axes glossary](../../../../glossary.md#term-axis); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- `TensorMaskedReductionExpressions` uses dynamic programming to prefer mappings that cover the
  reduced axis, then minimize positional displacement, then choose lexicographically. Shape
  repetition can therefore change which input axes a mask means without an explicit caller
  operation.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation))</span>
- The mapping accepts noncontiguous placements such as `[0, 2]`, so a mask Shape alone does not
  communicate its alignment under the ordinary model broadcasting contract.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [model glossary](../../../../glossary.md#term-model))</span>
- The provisional all-false masked-mean result is zero. Zero looks like observed data and hides
  that the selected count is zero.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean))</span>
- A seemingly simple decomposition is not a truthful replacement across the current model
  surface:<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition))</span>
  - `mask.cast(inputType).mul(input)` leaks masked-out [NaN](../../../../glossary.md#term-nan) and [infinity](../../../../glossary.md#term-infinity) through `NaN * 0` and
    `infinity * 0`;
  - `where(mask, input, input.mul(zero))` has the same leak in its false branch;
  - `TensorFactory.zeros`, `zerosLike`, [scalar](../../../../glossary.md#term-scalar), and `full` create eager host storage and reject
    dynamic non-scalar Shapes where applicable, so hiding them inside a fluent expression would
    add implicit materialization and ID allocation;<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [host storage glossary](../../../../glossary.md#host-storage); [materialization glossary](../../../../glossary.md#materialization))</span>
  - BOOL-to-floating cast value semantics, ordinary division by zero, and compiler gradient rules
    are not currently defined strongly enough to use <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>`mask.cast(...).sum(...)` and division as the
    public masked-mean contract; and
  - a lower-rank broadcast mask must be expanded to the input Shape before an ordinary reduction
    can count selections per output position, creating additional operations and gradient
    semantics that the current overload does not advertise.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Keeping both the mapper and an ordinary-broadcast variant would create two meanings for the same
  public request. The migration must remove the provisional contract atomically.

## Decision and rationale

### Selected design

1. Keep the public `sum(int, Tensor)` and `mean(int, Tensor)` overloads.
2. Keep `AggregateReductionKind.SUM` and [`MEAN`](../../../../glossary.md#term-mean) as the semantic kinds.
3. Replace `MaskedReductionAttrs(int axis, List<Integer> maskInputAxes)` with
   `MaskedReductionAttrs(int axis)`.
4. Keep the masked operation signatures as exact two-input, one-output variants distinguished by
   the `MaskedReductionAttrs` class.
5. Require the [mask](../../../../glossary.md#term-boolean-mask) [Shape](../../../../glossary.md#shape) to right-align and broadcast exactly to the input Shape under
   `ShapeBroadcast`. A broadcast result larger or higher-[rank](../../../../glossary.md#term-rank) than the input is rejected.
6. Require callers to use visible `reshape`, `expandDims`, or `expand` expressions when intended
   mask axes are not already expressed by ordinary right alignment.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
7. Define all-false masked sum as zero and all-false masked mean as [NaN](../../../../glossary.md#term-nan) for each output [slice](../../../../glossary.md#slice).
8. Preserve ordinary SUM and MEAN kinds, attributes, signatures, public methods, Shape rules, and
   current absence of numerical policy changes.

`MaskedReductionAttrs` remains necessary. `OperationKind.signatureFor` selects a [signature](../../../../glossary.md#term-signature) from
the attributes' exact concrete class before an [occurrence](../../../../glossary.md#term-occurrence)'s inputs are available, and one kind
cannot declare two variants with the same exact attributes class. Reusing `AxisReductionAttrs`
would therefore make the one-input ordinary form and two-input masked form structurally
ambiguous. One [axis](../../../../glossary.md#term-axis) component is the smallest explicit discriminator and semantic [parameter](../../../../glossary.md#term-parameter).

The two public methods remain first-class because current primitives cannot satisfy all of these
requirements together: no masked-out NaN/Inf leakage, static and dynamic Shapes, zero-sized axes,
no implicit [eager](../../../../glossary.md#term-eager-evaluation) storage, exact selected-count semantics, one stable [producer](../../../../glossary.md#term-producer) occurrence, and no
invented [gradient](../../../../glossary.md#term-gradient) promise. A misleading [decomposition](../../../../glossary.md#term-decomposition) would expose several ordinary operations
whose current contracts do not jointly define the masked [operation](../../../../glossary.md#operation)'s result.

### All-false [mean](../../../../glossary.md#term-mean) policy

The selected result is [NaN](../../../../glossary.md#term-nan) in the result's existing floating [data type](../../../../glossary.md#data-type--datatype):

- a masked mean is selected sum divided by selected count;
- a zero selected count has no arithmetic mean;
- NaN keeps the undefined result observable instead of silently fabricating zero;
- a value-dependent explicit failure would introduce an execution-time failure protocol not
  currently owned or represented by this model task; and<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- zero would preserve provisional legacy behavior but would be mathematically misleading and
  would make an empty selected set indistinguishable from a genuine zero mean.

The choice is consistent with primary library precedents without making their APIs authoritative
for Synaptik. [NumPy `mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)
defines the mean as sum divided by [element](../../../../glossary.md#term-element) count and accepts a broadcastable `where` [mask](../../../../glossary.md#term-boolean-mask).
[JAX `mean`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mean.html) likewise requires a
broadcast-compatible `where` mask, while
[JAX `nanmean`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmean.html) explicitly
returns NaN when `where` is false for every element.
[PyTorch `mean`](https://docs.pytorch.org/docs/stable/generated/torch.mean.html) returns NaN for an
empty [tensor](../../../../glossary.md#tensor), and [PyTorch `nanmean`](https://docs.pytorch.org/docs/stable/generated/torch.nanmean.html)
returns NaN when every reduced element is NaN. PyTorch's
[masked API](https://docs.pytorch.org/docs/stable/masked.html)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> remains [prototype](../../../../glossary.md#term-prototype) documentation, so
this task does not treat it as a stable direct contract.

For [gradient](../../../../glossary.md#term-gradient) semantics, the [BOOL](../../../../glossary.md#data-type--datatype) mask is non-[differentiable](../../../../glossary.md#derivative-and-chain-rule). A future [compiler](../../../../glossary.md#term-compiler) rule may give
excluded input positions zero contribution and selected positions the ordinary sum [derivative](../../../../glossary.md#term-derivative) or
`1 / selectedCount` for a non-empty mean [slice](../../../../glossary.md#slice). A zero-count mean slice has a NaN result and no
[finite](../../../../glossary.md#term-finite) derivative guarantee. This task records only that semantic implication; it adds no
gradient kind, backward expression, or compiler behavior.

### Rejected alternatives

| Alternative | Decision | Reason |
|---|---|---|
| Keep the ordered [injective](../../../../glossary.md#term-injective) mapper | Reject | It hides [axis](../../../../glossary.md#term-axis) placement and makes meaning depend on [heuristic](../../../../glossary.md#term-heuristic) DP/scoring. |
| Add a second ordinary-broadcast masked [API](../../../../glossary.md#term-application-programming-interface-api) | Reject | A transition would retain duplicate meanings and downstream compatibility burden. |
| Remove the public overloads | Reject | Masked exclusion, especially for [NaN](../../../../glossary.md#term-nan)/Inf and dynamic [Shapes](../../../../glossary.md#shape), is useful and not honestly expressible by the current primitive contracts. |
| Compose multiplication plus ordinary reduction | Reject | Masked-out NaN/Inf can leak through multiplication by zero. |
| Compose `where` with an input-derived zero | Reject | Input-derived multiplication/subtraction can produce NaN; factory zero values add hidden [eager](../../../../glossary.md#term-eager-evaluation) storage and do not cover dynamic full Shapes. |
| Compose `where`, cast, ordinary sum, and division | Reject | Current cast/division value contracts and [gradient](../../../../glossary.md#term-gradient) behavior are not sufficient, count Shape requires explicit expansion, and the sequence adds several [producers](../../../../glossary.md#term-producer) rather than one semantic [occurrence](../../../../glossary.md#term-occurrence). |
| Return zero for all-false [mean](../../../../glossary.md#term-mean) | Reject | It preserves provisional behavior but fabricates a valid-looking mean for an empty selected set. |
| Throw for all-false mean | Reject | It needs [runtime](../../../../glossary.md#term-runtime) value inspection and a cross-[backend](../../../../glossary.md#backend) execution failure contract outside this [model](../../../../glossary.md#term-model) task. |
| Add an all-false policy attribute | Reject | One portable NaN policy is sufficient; a caller policy would enlarge semantics and backend obligations without a current need. |
| Add `MaskedReductionKind` | Reject | Existing aggregate kinds plus the minimal attributes class already provide typed [identity](../../../../glossary.md#term-identity-meanings) without another family. |

## Before / after [API](../../../../glossary.md#term-application-programming-interface-api) and semantic tables

### Java surface

| Current contract | Final contract | Disposition |
|---|---|---|
| `MaskedReductionAttrs(int axis, List<Integer> maskInputAxes)` | `MaskedReductionAttrs(int axis)` | Replace atomically; remove mapping component/accessor/[validation](../../../../glossary.md#term-validation). |
| `AggregateReductionKind.SUM` masked [signature](../../../../glossary.md#term-signature): `MaskedReductionAttrs`, 2 -> 1 | unchanged signature | Keep first-class masked variant. |
| `AggregateReductionKind.MEAN` masked signature: `MaskedReductionAttrs`, 2 -> 1 | unchanged signature | Keep first-class masked variant. |
| `Tensor.sum(int axis, Tensor mask)` | same signature | Keep; change validation and semantics only. |
| `Tensor.mean(int axis, Tensor mask)` | same signature | Keep; change validation and semantics only. |
| `TensorMaskedReductionExpressions` DP/scoring mapper | one right-aligned broadcast proof | Delete resolver, compatibility method, mapping state, and score/tie behavior. |

No compatibility constructor, deprecated mapping accessor, [alias](../../../../glossary.md#aliasing-and-arena) attributes class, transitional
helper, or fallback mapper remains.

### [Shape](../../../../glossary.md#shape) and value semantics

| Case | Required final behavior |
|---|---|
| [Mask](../../../../glossary.md#term-boolean-mask) Shape | Must ordinary-right-align and broadcast exactly to the input Shape. |
| Mask has intended non-right-aligned [axes](../../../../glossary.md#term-axis) | Caller explicitly reshapes, inserts [dimensions](../../../../glossary.md#dimension), or expands before calling the reduction. |
| [Scalar](../../../../glossary.md#term-scalar) mask | Broadcasts to the complete input Shape. |
| Mask [singleton](../../../../glossary.md#term-singleton) | Expands under the existing `ShapeBroadcast` rule. |
| Equal dynamic or expression Dimension | Accepted under existing [structural equality](../../../../glossary.md#term-structural-equality). |
| Different symbols, unequal expressions, or symbolic/non-singleton static pair | Rejected as locally unprovable by `ShapeBroadcast`. |
| Mask would enlarge an input singleton or add a leading result axis | Rejected because the broadcast result is not exactly the input Shape. |
| Result Shape | Input Shape with the normalized reduction axis removed; unaffected Dimension references are retained. |
| Result type / eligibility | Exact floating input type and exact input `requiresGrad`; mask contributes neither. |
| Masked-out [finite](../../../../glossary.md#term-finite), [NaN](../../../../glossary.md#term-nan), or Inf input | Excluded before aggregation and cannot affect the result. |
| All-false sum [slice](../../../../glossary.md#slice) | Floating additive zero. |
| All-false [mean](../../../../glossary.md#term-mean) slice | NaN in the result floating type; [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)/bit pattern is not specified. |
| Static zero-sized reduction axis | Every sum slice is zero and every mean slice is NaN, because no [element](../../../../glossary.md#term-element) can be selected. |
| Dynamic reduction axis | Construction succeeds when Shape [broadcasting](../../../../glossary.md#broadcasting) is locally provable; [runtime](../../../../glossary.md#term-runtime) zero-size or all-false slices follow the same zero/NaN semantics. |
| Selected non-finite input | Participates in the requested ordinary aggregate; this task does not add a new selected-value accumulation or NaN policy. |

## [Scope](../../../../glossary.md#term-scope)

- Simplify `MaskedReductionAttrs` to exactly one `int axis` record component.
- Remove all mapping lists, list validation, immutable snapshots, accessors, examples, and terms
  from live production, tests, Javadocs, API references, and glossary documentation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Preserve exact non-negative normalized-[axis](../../../../glossary.md#term-axis) constructor [validation](../../../../glossary.md#term-validation) and value semantics.
- Preserve the existing masked SUM/[MEAN](../../../../glossary.md#term-mean) [signature](../../../../glossary.md#term-signature) class and exact two-input/one-output counts.
- Replace DP/scoring [alignment](../../../../glossary.md#term-alignment) with exactly one ordinary `ShapeBroadcast.broadcast(inputShape,
  maskShape)` call and directional equality to the input [Shape](../../../../glossary.md#shape).
- Preserve exact [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) and floating input requirements.
- Preserve positive/negative caller-axis normalization, axis removal, exact unaffected Dimension
  references, unresolved layout, exact type and input gradient eligibility, freshness, empty label,
  absent storage, and ordered <span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Dimension glossary](../../../../glossary.md#dimension); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>`[input, mask]` [producer](../../../../glossary.md#term-producer) inputs.
- Keep both public masked overload signatures with no [alias](../../../../glossary.md#aliasing-and-arena) or additional overload.
- Define excluded non-finite behavior and the all-false sum/mean policies in model semantics and
  Javadocs without value execution.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [model glossary](../../../../glossary.md#term-model))</span>
- Update focused semantic/expression tests and retain ordinary reduction [regression](../../../../glossary.md#term-regression) coverage.
- Finalize Tensor API, Compile API, glossary, capability baseline, task evidence, master plan, and
  roadmap through the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- full/all-[axes](../../../../glossary.md#term-axis) [masked reduction](../../../../glossary.md#masked-reduction), `keepDimensions`, multiple axes, weights, numeric masks, nullable
  masks, policy parameters, or another masked operation family<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [masks glossary](../../../../glossary.md#term-boolean-mask); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- masked product, extrema, boolean reductions, arg reduction, loss, normalization, pooling,
  attention, indexing, or generic masking<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [boolean glossary](../../../../glossary.md#data-type--datatype); [loss glossary](../../../../glossary.md#loss); [normalization glossary](../../../../glossary.md#term-normalization); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- value or storage inspection, eager result calculation, mask materialization, selected-count
  calculation, division, allocation, mutation, or output storage<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [materialization glossary](../../../../glossary.md#materialization))</span>
- an implementation [decomposition](../../../../glossary.md#term-decomposition) into `where`, cast, multiplication, ordinary reduction,
  division, constants, or expand
- a new zero-like expression, scalar conversion policy, division-by-zero policy, or ordinary
  SUM/MEAN numerical policy<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean))</span>
- gradient implementation, backward semantic kinds, autograd traversal, optimizer, or training
  execution<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- compiler capture, canonicalization, decomposition, inference, planning, prepare, runtime,
  backend lowering, kernels, conformance execution, tracing, ONNX, or engine work<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [inference glossary](../../../../glossary.md#term-inference); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance); [ONNX glossary](../../../../glossary.md#term-onnx); [engine glossary](../../../../glossary.md#term-engine))</span>
- changes to `ShapeBroadcast`, [`Shape`](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension) expression semantics, ordinary
  `AxisReductionAttrs`, [operation](../../../../glossary.md#operation)-[signature](../../../../glossary.md#term-signature) mechanics, [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) foundations,
  `TensorFactory`, [scalar](../../../../glossary.md#term-scalar) [constants](../../../../glossary.md#term-constant), or another completed operation family
- dependencies, Gradle, Java version, architecture contract, focused architecture docs,
  architecture tests, another module, unrelated refactoring, or a detailed task-0018R-or-later
  specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership of public
  Tensor and operation semantics and later-layer ownership of gradients, lowering, and execution<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation); [gradients glossary](../../../../glossary.md#term-gradient); [lowering glossary](../../../../glossary.md#lowering))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0015E](0015e-where-selection-semantic-kind.md)
- [Task 0015F](0015f-where-selection-tensor-expression.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Task 0016F](0016f-masked-reduction-semantics-and-axis-mapping.md)
- [Task 0016F1](0016f1-masked-sum-and-mean-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0018M](0018m-symbolic-extent-expressions.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work remains entirely in `modules/model` plus explanatory/[planning](../../../../glossary.md#term-planning) documentation.
- `AggregateReductionKind` continues to identify backend-independent semantics and contains no
  backend, compiler, cost, fusion, route, kernel, storage, or execution metadata.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `MaskedReductionAttrs` stores only intrinsic normalized semantic state. It contains no Tensor,
  Shape, mask, mapping, broadcast plan, score, policy, or producer occurrence.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [mask glossary](../../../../glossary.md#term-boolean-mask); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `TensorMaskedReductionExpressions` performs local deterministic Shape validation and metadata
  construction only. It reads no storage or values and creates no hidden expression subgraph.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- The masked [operation](../../../../glossary.md#operation) remains one [producer](../../../../glossary.md#term-producer) with exact ordered inputs `[input, mask]` and one
  output descriptor. Tensor remains public API state, not graph IR.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- `ShapeBroadcast` remains the sole local right-aligned rule. The helper consumes it without
  changing symbolic equality, singleton behavior, or Shape APIs.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- Model records the all-false result meaning. Compiler owns gradient construction; backend prepare
  and concrete backends own lowering and execution algorithms that must preserve that meaning.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering))</span>
- Ordinary one-input SUM/[MEAN](../../../../glossary.md#term-mean) forms and every other aggregate kind remain unchanged.
- Stop if implementation requires a new operation family, zero-expression contract, runtime
  failure protocol, dependency, another module, or architecture change.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.reduction` — owns aggregate kinds and the simplified
  masked axis parameter.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- `io.github.pho001.synaptik.model.tensor` — owns public overloads, local validation, descriptor
  construction, producer inputs, and derived Tensor creation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `io.github.pho001.synaptik.model.shape` — supplies ordinary right-aligned local broadcasting,
  immutable Dimensions, Shapes, and axis normalization.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [immutable glossary](../../../../glossary.md#term-immutable); [Dimensions glossary](../../../../glossary.md#dimension); [Shapes glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `io.github.pho001.synaptik.model.datatype` — supplies floating and exact [BOOL](../../../../glossary.md#data-type--datatype) category checks.

Packages added or moved: None.

Type placement remains unchanged:

- `MaskedReductionAttrs` remains in the reduction semantic package because it distinguishes the
  two-input masked occurrence and carries its one intrinsic axis parameter.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `TensorMaskedReductionExpressions` remains package-private beside Tensor, descriptors,
  producer/provenance construction, and the central derived factory seam.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>

## Required contracts

### Simplified attributes

Replace the record with exactly:

```java
public record MaskedReductionAttrs(int axis) implements OperationAttrs {
    public MaskedReductionAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    @Override
    public int axis() {
        return axis;
    }
}
```

It has one component, one public [canonical](../../../../glossary.md#term-canonical-form) constructor, one explicit accessor, and generated
`equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`. It has no list, mapping, secondary constructor, helper,
factory, nested type, field beyond the record component, or compatibility [API](../../../../glossary.md#term-application-programming-interface-api). Every non-negative
`int`, including `Integer.MAX_VALUE`, remains structurally valid because input-[rank](../../../../glossary.md#term-rank) [validation](../../../../glossary.md#term-validation)
belongs to [Tensor](../../../../glossary.md#tensor) construction.

The exact constructor failure remains:

```text
axis must be non-negative: <axis>
```

### Aggregate semantics and [signatures](../../../../glossary.md#term-signature)

`AggregateReductionKind.SUM` and [`MEAN`](../../../../glossary.md#term-mean) retain these variants in stable order:

```text
NoOperationAttrs       exactly 1 input -> exactly 1 output
AxisReductionAttrs     exactly 1 input -> exactly 1 output
MaskedReductionAttrs   exactly 2 inputs -> exactly 1 output
```

No other aggregate kind accepts `MaskedReductionAttrs`. No `OperationSignature`, [`OperationKind`](../../../../glossary.md#operationkind),
or [`Operation`](../../../../glossary.md#operation) mechanics change.

The masked input roles are exactly `[input, mask]`. False excludes the corresponding broadcast
input position before aggregation. Excluded [NaN](../../../../glossary.md#term-nan) and [infinity](../../../../glossary.md#term-infinity) values do not participate. SUM over
an empty selected set is zero; MEAN over an empty selected set is NaN. These are family semantics,
not additional attributes.

### Public [Tensor](../../../../glossary.md#tensor) surface

Keep exactly:

```java
public Tensor sum(int axis, Tensor mask)
public Tensor mean(int axis, Tensor mask)
```

Each remains public, instance, non-static, and non-synchronized and delegates once to
`TensorMaskedReductionExpressions.apply(this, mask, exactKind, axis)`. No new method is added, so
the current exact Tensor public method count remains unchanged.

### Helper surface

The final package-private helper remains a final non-record class with no fields or nested types,
one private zero-[argument](../../../../glossary.md#term-argument) constructor, and exactly these three static methods:

```java
static Tensor apply(Tensor input, Tensor mask, AggregateReductionKind kind, int axis)
private static Shape reduceShape(Shape inputShape, int normalizedAxis)
private static Tensor create(
        Tensor input,
        Tensor mask,
        AggregateReductionKind kind,
        int normalizedAxis,
        Shape resultShape)
```

Delete `resolveMapping`, `compatible`, all DP arrays, candidate reconstruction, displacement
arithmetic, lexicographic tie logic, and mapping collections. Add no replacement resolver or
public/package-private broadcast helper.

### [Validation](../../../../glossary.md#term-validation), failure order, and [identity](../../../../glossary.md#term-identity-meanings) effects

`apply` performs exactly:

1. null-check `input`, [`mask`](../../../../glossary.md#term-boolean-mask), and `kind`, in that order, with [parameter](../../../../glossary.md#term-parameter)-name messages;
2. require exactly SUM or [MEAN](../../../../glossary.md#term-mean), preserving
   `kind must be SUM or MEAN, but was <kind>`;
3. require floating input, preserving
   `input must be a floating data type, but was <type>`;
4. require exact [BOOL](../../../../glossary.md#data-type--datatype) mask, preserving
   `mask must have BOOL data type, but was <type>`;
5. read the input and mask [Shapes](../../../../glossary.md#shape);
6. normalize [`axis`](../../../../glossary.md#term-axis) exactly once through `inputShape.normalizeAxis(axis)`;
7. call `ShapeBroadcast.broadcast(inputShape, maskShape)` exactly once;
8. require the returned Shape to equal `inputShape`; otherwise throw
   `IllegalArgumentException` with exact message
   `mask shape <maskShape> must broadcast exactly to input shape <inputShape>, but produced <broadcastShape>`;
9. derive the axis-removing result Shape once; and
10. construct the result through `create`.

Do not catch or rewrite an incompatibility thrown by `ShapeBroadcast`; its existing diagnostic
remains the failure for an aligned incompatible [Dimension](../../../../glossary.md#dimension) pair. The explicit equality failure
covers an otherwise valid symmetric broadcast that would enlarge the input Shape or add a leading
axis.

Every failure through step 8 occurs before attributes, [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), [producer](../../../../glossary.md#term-producer), or [Tensor](../../../../glossary.md#tensor)
identity construction and consumes no Tensor ID. `create` constructs, in order:

1. `new MaskedReductionAttrs(normalizedAxis)`;
2. one unresolved `TensorDescriptor` with exact input type, result Shape, and input
   `requiresGrad`;
3. one `Operation(kind, attrs)`;
4. one `TensorFactory.createDerived(descriptor, Optional.empty(), operation,
   List.of(input, mask))` call.

The factory creates the one-output producer and index-zero [provenance](../../../../glossary.md#provenance). Identifier exhaustion can
occur only at that final factory boundary after local [immutable](../../../../glossary.md#term-immutable) values are valid. The result is
fresh, unlabeled, storage-free, and unresolved; no input or mask state is mutated.

### [Shape](../../../../glossary.md#shape) rules and examples

The accepted [mask](../../../../glossary.md#term-boolean-mask) relationship is directional even though `ShapeBroadcast` itself is symmetric:

```text
broadcast(inputShape, maskShape) == inputShape
```

Examples for input `[batch, time, features]`:

| Mask Shape | Result | Reason |
|---|---|---|
| `[]` | accepted | [Scalar](../../../../glossary.md#term-scalar) broadcasts to every input position. |
| `[features]` | accepted | Ordinary trailing-[axis](../../../../glossary.md#term-axis) [alignment](../../../../glossary.md#term-alignment). |
| `[time, 1]` | accepted | Time aligns to input axis 1; [singleton](../../../../glossary.md#term-singleton) expands over [features](../../../../glossary.md#term-feature). |
| `[batch, time, 1]` | accepted | Explicit intended [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/time alignment. |
| `[batch, time]` | rejected in general | Right alignment compares `time` to [`features`](../../../../glossary.md#term-software-feature); caller uses `expandDims(2)`. |
| `[2, time, features]` against input leading singleton | rejected if it enlarges input | A mask may broadcast to the input; it may not change the input domain. |

Static zero extents follow existing singleton rules. Equal named or exact expression [dimensions](../../../../glossary.md#dimension)
are accepted. A static mask singleton may align to any input Dimension. Different symbols,
distinct unknowns, unequal expressions, and symbolic/non-singleton static pairs remain locally
unprovable.

The result removes the normalized input axis, uses [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()` for [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) input,
and retains every unaffected Dimension reference in order.

### No hidden [decomposition](../../../../glossary.md#term-decomposition)

The implementation must not construct or call:

- `Tensor.where`;
- `Tensor.cast`;
- `Tensor.mul`, `Tensor.div`, ordinary `sum`, or ordinary [`mean`](../../../../glossary.md#term-mean);
- `TensorFactory.scalar`, `zeros`, `zerosLike`, `full`, or another [eager](../../../../glossary.md#term-eager-evaluation) [constant](../../../../glossary.md#term-constant) path;
- `Tensor.expand`, `expandDims`, or `reshape` on behalf of the caller; or
- a new zero, count, or division semantic kind.

Exact [producer](../../../../glossary.md#term-producer) inspection must show one [operation](../../../../glossary.md#operation), inputs `[input, mask]`, one output [descriptor](../../../../glossary.md#term-descriptor),
and [provenance](../../../../glossary.md#provenance) output index zero.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/MaskedReductionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMaskedReductionExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/reduction/MaskedReductionAttrsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMaskedReductionTest.java`

Documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `TensorTest`, `TensorNumericReductionTest`, `OperationSignatureTest`, `ReductionSemanticsTest`,
  `ShapeBroadcastTest`, `TensorWhereSelectionTest`, `TensorCastExpressionTest`,
  `TensorBinaryArithmeticTest`, `TensorFactoryConstantTest`, `TensorFactoryFullIdentityTest`,
  `ScalarValueTest`, `TensorProducerTest`, and `TensorProvenanceTest`.
- `ShapeBroadcast`, ordinary reduction helpers and attributes, `TensorFactory`, `TensorConstants`,
  [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) foundations, and their Javadocs.
- Training API, focused architecture docs, ADRs, architecture tests, backend-conformance tests,
  integration tests, Gradle configuration, dependencies, and other modules.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most four production files, two focused tests, and seven documentation/[planning](../../../../glossary.md#term-planning) files: thirteen
paths total.

This does not exceed the planning guide's 12–18-file guardrail. The thirteen paths form the
smallest cohesive [atomic](../../../../glossary.md#resource-transaction-and-rollback) migration: attributes, kind semantics, public/helper Javadocs and
behavior, focused tests, and every current explanatory/planning statement must agree in one
change. Splitting them would temporarily retain a mapping [API](../../../../glossary.md#term-application-programming-interface-api) with ordinary broadcast behavior or
publish conflicting all-false semantics.

Do not modify a review-only path merely to spend the allowance. Stop before a fourteenth path,
another production concept, ordinary reduction behavior, ShapeBroadcast change, [dependency](../../../../glossary.md#term-dependency),
build file, another [module](../../../../glossary.md#term-module), or architecture document.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Rewrite `MaskedReductionAttrs` type, constructor, component, and accessor Javadocs for the one-
  axis contract and remove every mapping claim.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Update `AggregateReductionKind` type/SUM/MEAN Javadocs for ordinary broadcasting, exclusion of
  masked non-finite values, zero all-false sum, and NaN all-false mean without changing constants
  or signature declarations.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [finite glossary](../../../../glossary.md#term-finite); [NaN glossary](../../../../glossary.md#term-nan); [constants glossary](../../../../glossary.md#term-constant); [signature glossary](../../../../glossary.md#term-signature))</span>
- Update both public Tensor overloads and every helper member for exact right-aligned validation,
  explicit caller alignment, result/provenance facts, validation order, ID effects, all-false
  semantics, and deferred gradients/execution.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [alignment glossary](../../../../glossary.md#term-alignment); [provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) must replace the mapping example with a complete explicit `expandDims(2)` example,
  show the resulting mask producer and masked reduction producer boundary, and explain what the
  result proves.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [producer glossary](../../../../glossary.md#term-producer); [masked reduction glossary](../../../../glossary.md#masked-reduction))</span>
- Tensor API and glossary must include masked-out NaN/Inf exclusion, static zero-axis and dynamic
  all-false behavior, and the NaN mean policy without promising a NaN payload.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Compile API must describe the current first-class two-input occurrence and explicit broadcast
  validation without claiming capture, decomposition, gradient generation, lowering, or
  execution.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [occurrence glossary](../../../../glossary.md#term-occurrence); [capture glossary](../../../../glossary.md#term-capture-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Capability baseline must move from an undecided possible composition to the selected
  first-class contract and implementation status only after implementation finishes.
- Completed tasks 0016F/0016F1 remain unchanged historical records of the provisional design.
- Record reasoned no-change conclusions for ordinary SUM/[MEAN](../../../../glossary.md#term-mean), `where`, cast, binary arithmetic,
  factory constants, typed scalar values, Shape/ShapeBroadcast, producer/provenance, Training API,
  architecture/ADRs/tests, conformance/integration, Gradle/dependencies, and other modules.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- `MaskedReductionAttrs` has exactly one `int axis` component and no mapping API or collection
  state.
- Its exact non-negative validation, generated value semantics, and SUM/MEAN operation composition
  pass focused tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation))</span>
- SUM and MEAN retain exactly ordinary no-attrs, ordinary axis-attrs, and masked two-input
  signatures; all other aggregate signatures remain unchanged.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- The two public masked method [signatures](../../../../glossary.md#term-signature) remain exact and no overload/[alias](../../../../glossary.md#aliasing-and-arena) is added or removed.
- The helper has exactly the three specified methods, no fields/nested types, and no resolver,
  compatibility method, DP state, score, candidate, mapping, or synthetic helper.
- Null, kind, type, axis, ShapeBroadcast, directional-result, and ID-exhaustion failures occur in
  the specified order with the specified messages and ID effects.
- Mask broadcasting is exactly ordinary right-aligned local broadcasting whose result must equal
  input Shape; mask alignment never enlarges the input domain.<span data-glossary-navigation="true"> ([Mask glossary](../../../../glossary.md#term-boolean-mask); [broadcasting glossary](../../../../glossary.md#broadcasting); [Shape glossary](../../../../glossary.md#shape); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Scalar, equal-rank, lower-rank trailing, explicit singleton-axis, static zero-extent, equal
  dynamic-symbol, equal expression, and same-unknown cases are covered.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [extent glossary](../../../../glossary.md#term-extent); [symbol glossary](../../../../glossary.md#term-symbol))</span>
- Non-right-aligned `[batch, time]` masks, input-enlarging masks, incompatible static dimensions,
  different symbols, unequal expressions, and distinct unknowns are rejected until the caller
  explicitly transforms the mask.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>
- A test proves `mask.expandDims(2)` makes `[batch, time]` explicit as `[batch, time, 1]` and that
  the masked operation's ordered second producer input is that exact transformed Tensor.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Result Shape, type, eligibility, unresolved layout, freshness, label/storage absence, producer,
  provenance order, output descriptor reference, and output index remain exact.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Live Java and current [API](../../../../glossary.md#term-application-programming-interface-api)/glossary text contain no `maskInputAxes`, mapping resolver, mapping
  preference, positional displacement, or lexicographic alignment contract.
- Current semantics state masked-out NaN/Inf exclusion, zero all-false sum, and NaN all-false mean
  for static, zero-sized, and runtime dynamic cases.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Ordinary SUM/[MEAN](../../../../glossary.md#term-mean) declarations, helpers, signatures, public behavior, and tests remain unchanged.
- No `where`, cast, multiplication, division, ordinary-reduction, constant-factory, or hidden
  expand/reshape composition is added.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- No value execution, gradient implementation, compiler/backend behavior, dependency, build, or
  architecture change is added.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and final model tests, model Javadoc, runnable example, documentation validation,
  thirteen-path scope, synchronized status, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass.
- The independent documentation pass completes in the same overall change and reuses successful
  Java-test evidence unless executable Java changes afterward.
- 0018Q becomes Complete only after both passes. Task 0018R and every later task remain Draft
  without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) the focused contract set:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMaskedReductionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.shape.ShapeBroadcastTest
```

After executable Java stabilizes, record one final [module](../../../../glossary.md#term-module) run:

```bash
./gradlew :modules:model:test
```

Automated tests must cover the exact record/helper/public surfaces, [signature](../../../../glossary.md#term-signature) variants,
validation order/messages, pre-factory ID non-consumption, ID exhaustion, accepted/rejected
broadcast [matrix](../../../../glossary.md#term-matrix-meanings), explicit [rank](../../../../glossary.md#term-rank) editing, [Dimension](../../../../glossary.md#dimension) reference retention, [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance)
[identity](../../../../glossary.md#term-identity-meanings), freshness, and ordinary-reduction [regression](../../../../glossary.md#term-regression). Do not repeat manual reflection or
[bytecode](../../../../glossary.md#term-bytecode) checks when stable automated assertions cover the [invariant](../../../../glossary.md#term-invariant).

The separate documentation-focused pass receives the final diff and [model](../../../../glossary.md#term-model)-test evidence. After
final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also:

- compiles and runs the new explicit-alignment Tensor API example against the final Java 26 model
  classes;<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- checks generated Javadoc for the simplified record, right-aligned overloads, and [NaN](../../../../glossary.md#term-nan) policy;
- validates all changed local Markdown links and GitHub-style anchors;
- checks the six official external documentation URLs used by the decision rationale;
- checks balanced fences, final newlines, trailing whitespace, terminology, and authority
  boundaries;
- verifies exactly the thirteen authorized paths and no live mapping vocabulary;
- confirms no executable Java changed after the reused model-test result, or reruns that result if
  it did;
- confirms task/master/roadmap/capability synchronization, 0018R Draft status, and absence of any
  detailed 0018R-or-later task; and<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- confirms no architecture, [Gradle](../../../../glossary.md#term-gradle), [dependency](../../../../glossary.md#term-dependency), other-module, [commit](../../../../glossary.md#term-commit), or push change.

Repository-wide `./gradlew test` remains deferred to the recorded public-surface cleanup [checkpoint](../../../../glossary.md#term-checkpoint)
after task 0018S. This task changes one module and no dependency, build, or architecture boundary.

## Dependencies

- Tasks 0015E–0015F — current first-class conditional selection and its exact limitations —
  Complete.
- Tasks 0016A–0016B — aggregate kinds, ordinary attributes, and ordinary SUM/MEAN expressions —
  Complete.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean))</span>
- Tasks 0016F–0016F1 — provisional masked mapping semantics and public expressions replaced by
  this cleanup — Complete historical prerequisites.
- Task 0018K — exact kind/attributes [signatures](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality) — Complete.
- Task 0018L — [producer](../../../../glossary.md#term-producer)/output-index [provenance](../../../../glossary.md#provenance) — Complete.
- Tasks 0018M–0018M1 — symbolic [Dimension](../../../../glossary.md#dimension) equality and first dynamic-[shape](../../../../glossary.md#shape) adoption — Complete.
- Task 0018N — exact typed [scalar](../../../../glossary.md#term-scalar) values and factory-boundary evidence — Complete.

All dependencies are Complete. No architecture or package decision blocks this task.

## Follow-up tasks

- 0018R remains Draft for negative-step slicing and window public-contract cleanup.
- 0018V remains Draft for multi-axis and statistical reductions; it must preserve this selected
  mask contract unless a separately planned capability explicitly changes it.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Compiler tasks later own masked operation capture, gradient-rule selection, optional legal
  decomposition, and validation of arbitrary captured occurrences.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [operation glossary](../../../../glossary.md#operation); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [validation glossary](../../../../glossary.md#term-validation))</span>
- Backend and conformance tasks later own selected-value execution, accumulation, exact NaN/Inf
  behavior within ordinary selected aggregates, all-false results, and cross-backend parity.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [NaN glossary](../../../../glossary.md#term-nan))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics, public [Tensor](../../../../glossary.md#tensor)
construction, [Shape](../../../../glossary.md#shape) values, and [provenance](../../../../glossary.md#provenance) to `modules/model`, [compiler](../../../../glossary.md#term-compiler)-owned [gradient](../../../../glossary.md#term-gradient) expansion
to compiler, and executable [lowering](../../../../glossary.md#lowering) to backend [prepare](../../../../glossary.md#prepare)/concrete backends. This redesign removes
[model](../../../../glossary.md#term-model)-local [heuristic](../../../../glossary.md#term-heuristic) meaning while remaining inside those [owners](../../../../glossary.md#term-owner).

If implementation requires an architecture rule change, [dependency](../../../../glossary.md#term-dependency), [runtime](../../../../glossary.md#term-runtime) value failure
protocol, or another [module](../../../../glossary.md#term-module), stop and report the exact conflict before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0015E/0015F/0016A–0016F1/0018K/0018L/0018M/0018N/0018Q, relevant Tensor/Compile/
Training API and glossary sections, and every affected or review-only source/test named by task
0018Q in full.

Implement task 0018Q exactly. Atomically simplify MaskedReductionAttrs to one normalized axis,
delete mapping/DP/scoring behavior, require one ordinary right-aligned broadcast whose result is
exactly the input Shape, retain the two public first-class SUM/MEAN overloads and exact
[input, mask] producer, and document zero all-false sum plus NaN all-false mean. Callers explicitly
reshape/expand masks; add no hidden composition or compatibility bridge. Preserve ordinary
SUM/MEAN and every architecture boundary. Stay within the exact thirteen paths, stop on scope or
architecture conflict, and do not commit or push.

Run the focused contract command and final model suite after executable Java stabilizes. Then
hand the actual diff and exact evidence to a separate clean-context documentation-focused agent
in the same overall change. That agent must independently finalize Javadocs, Tensor/Compile APIs,
glossary, capability/task/master/roadmap status, official links, the runnable example, and all
specified documentation/scope checks without repeating successful Java tests unless executable
behavior changes or a concrete risk is recorded.

Mark 0018Q Complete only after both passes succeed. Leave 0018R and every later task Draft without
a detailed specification.
```

## Local decisions

- Keep first-class public masked SUM/MEAN because the current primitive surface cannot express
  safe masked exclusion and selected-count mean for every current Shape without hidden eager
  constants or undefined value/gradient behavior.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [Shape glossary](../../../../glossary.md#shape); [eager glossary](../../../../glossary.md#term-eager-evaluation); [constants glossary](../../../../glossary.md#term-constant); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Retain one-component `MaskedReductionAttrs` because exact attributes class selects the masked
  two-input signature before occurrence inputs are available.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Use ordinary `ShapeBroadcast` once and require its result to equal the input Shape. This reuses
  the model's canonical symbolic/singleton rules while making direction explicit.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [canonical glossary](../../../../glossary.md#term-canonical-form); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Require explicit caller rank editing rather than inferring intended axes. The transformed mask
  becomes visible producer/provenance input.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axes glossary](../../../../glossary.md#term-axis); [mask glossary](../../../../glossary.md#term-boolean-mask); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Choose NaN for an all-false masked mean. Zero remains the all-false sum identity; runtime failure
  and caller-selected policies are rejected.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [identity glossary](../../../../glossary.md#term-identity-meanings); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Preserve input gradient eligibility as metadata and exclude mask eligibility. This does not
  promise a compiler rule, especially for NaN zero-count slices.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Preserve completed tasks 0016F/0016F1 unchanged as historical evidence; current APIs and plans
  describe the replacement.

## Known limitations

- The task constructs semantic metadata only; no masked result values exist until later compiler,
  backend, runtime, and engine work.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>
- Only [axis](../../../../glossary.md#term-axis)-removing floating SUM and [MEAN](../../../../glossary.md#term-mean) with exact [BOOL](../../../../glossary.md#data-type--datatype) [masks](../../../../glossary.md#term-boolean-mask) are represented.
- Dynamic [broadcasting](../../../../glossary.md#broadcasting) remains conservative: only relationships already provable by
  `ShapeBroadcast` are accepted.
- NaN payload, selected-value accumulation precision/order, signed zero, selected NaN propagation,
  determinism, and backend algorithm remain unspecified with the ordinary reduction numerical
  policy.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [precision glossary](../../../../glossary.md#term-precision); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
- No gradient operation or rule is implemented. The zero-count mean slice intentionally has no
  finite derivative guarantee.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation); [slice glossary](../../../../glossary.md#slice); [finite glossary](../../../../glossary.md#term-finite); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Explicit mask rank editing creates its own visible producer; this task does not canonicalize or
  fuse it.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [producer glossary](../../../../glossary.md#term-producer))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/plan_0018q` read the repository instructions and [architecture contract](../../../../glossary.md#architecture-contract);
documentation rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/Planning/Example profiles; planning guide and
roadmap; [model](../../../../glossary.md#term-model) capability baseline and master plan; completed tasks 0015E/0015F, 0016A–0016F1,
0018K/0018L/0018M/0018N/0018P; relevant [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training API and glossary sections; and the
current masked attributes, aggregate [signatures](../../../../glossary.md#term-signature), masked/public/ordinary reduction helpers,
ShapeBroadcast, where, cast, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic, factory [constant](../../../../glossary.md#term-constant), typed [scalar](../../../../glossary.md#term-scalar), [producer](../../../../glossary.md#term-producer),
[provenance](../../../../glossary.md#provenance), dynamic-[Shape](../../../../glossary.md#shape), and focused-test contracts.

Planning found no architecture conflict. The existing aggregate kind, exact attribute-class
signature mechanism, Tensor producer model, and ShapeBroadcast contract support the selected
minimal first-class design. Thirteen implementation paths remain within the planning guide's
normal guardrail.

Planning consulted only official primary documentation for the cross-library comparison: NumPy
[mean](../../../../glossary.md#term-mean), JAX mean/nanmean, and PyTorch mean/nanmean/masked API pages linked above. The comparison is
used only to support the [NaN](../../../../glossary.md#term-nan) empty-mean decision; Synaptik semantics remain explicitly selected
here.

Planning-only validation:

- the task specification contains every canonical planning section, final decisions, exact
  before/after surfaces, validation/failure order, package impact, bounded scope, implementation
  handoff, limitations, and completion placeholders; the section check found all required
  headings with no duplicate level-two heading;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- task 0018Q is synchronized as Ready in this task, model master plan, and roadmap, while 0018R
  and every later task remain Draft;<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- no detailed task specification exists for 0018R or later;
- the local Markdown validator resolved all 318 local links and anchors across the four changed
  planning files;
- direct HTTP checks returned status 200 for all six official NumPy, JAX, and PyTorch URLs used
  above;
- fence, final-newline, and trailing-whitespace validation passed across all four changed files;
- final planning scope is exactly this task, the capability baseline, model master plan, and
  roadmap; no Java, test, API, glossary, architecture, Gradle, dependency, other-module, or later
  task path changed; and<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- `git diff --check` passed with no output. No Gradle or Javadoc task ran because this planning-
  only change modifies no Java, executable behavior, or public Javadoc.

Implementation and independent documentation evidence:

- Implementation context `/root/task_0018q_implementation` simplified
  `MaskedReductionAttrs` to exactly one normalized axis, retained the SUM/MEAN masked signature
  variants, replaced the mapper with one <span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis))</span>`ShapeBroadcast.broadcast(inputShape, maskShape)` call
  plus equality to the input Shape, and preserved one exact `[input, mask]` producer. It changed
  only the four production and two focused-test paths authorized by this task.
- The first development-focused run failed only because four new assertions or fixtures did not
  yet match the intended test setup. Those tests were corrected without changing production
  behavior. After a final distinct-unknown rejection assertion was added, the implementation
  context reran the exact five-class focused command from this task: <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`BUILD SUCCESSFUL` in 977 ms.
- The implementation context then ran `./gradlew :modules:model:test`: `BUILD SUCCESSFUL` in 1 s;
  720 tests across 88 XML suites, with zero failures, errors, or skips. No executable production
  Java changed after that result. The later focused-test assertion did not change production
  behavior, and both the focused command and final model suite were rerun after it.
- Independent documentation context
  `/root/task_0018q_implementation/task_0018q_docs` read the repository instructions and
  architecture contract; documentation rules and General/API-Javadoc/Planning/Example profiles;
  planning guide and roadmap; model capability baseline and master plan; task 0018Q and its
  directly relevant completed semantic, expression, signature, provenance, Shape, and scalar
  predecessors; final affected source/tests and review-only contracts; Tensor/Compile/Training
  APIs; glossary; generated Javadoc; and the actual combined diff.
- The documentation pass finalized all four production Javadocs, Tensor API, Compile API,
  glossary, capability baseline, this task, model master plan, and roadmap. The Tensor API now
  contains a complete Java 26 example in which `mask.expandDims(2)` creates the visible rank-edit
  producer and the masked operation consumes that exact Tensor as its second producer input.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [operation glossary](../../../../glossary.md#operation))</span>
- `./gradlew :modules:model:javadoc` passed: `BUILD SUCCESSFUL` in 1 s. Generated pages for
  `MaskedReductionAttrs`, `AggregateReductionKind`, and `Tensor` show the one-component record,
  ordinary right-aligned overload contract, excluded non-finite positions, zero all-false sum,
  NaN all-false mean, and unspecified NaN payload. No removed mapping vocabulary appears in those
  generated contracts.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- `javac -cp modules/model/build/classes/java/main -d /tmp/masked-reduction-example
  /tmp/MaskedReductionExpressionExample.java` and the matching `java` command passed against the
  final Java 26 model classes. Output was exactly
  `alignedMaskShape=Shape[batch, time, 1]`, `sumShape=Shape[batch, 4]`,
  `meanShape=Shape[batch, 4]`, `axis=1`, `rankEditInput=true`, `orderedInputs=true`, and
  `metadata=true`.
- The local Markdown validator resolved 449 local links, including 112 anchor links, across the
  seven changed documentation/planning files with zero failures. Direct `curl -L` checks returned
  HTTP 200 for all six official NumPy, JAX, and PyTorch URLs in the rationale.
- Fence, final-newline, trailing-whitespace, terminology, and authority-boundary checks passed.
  Current Java, Tensor API, Compile API, and glossary contain no live mapper, mapping-preference,
  displacement, lexicographic-alignment, or <span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>`maskInputAxes` contract. Completed tasks 0016F and
  0016F1 remain unchanged historical records.
- Final scope is exactly the thirteen authorized paths: four production files, two focused tests,
  Tensor API, Compile API, glossary, capability baseline, this task, model master plan, and
  roadmap. Status is synchronized as 0018Q Complete; 0018R and every later task remain Draft, and
  no detailed 0018R-or-later specification exists. No architecture, ADR, architecture-test,
  conformance, integration, Gradle, dependency, other-module, commit, or push change occurred.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [commit glossary](../../../../glossary.md#term-commit))</span>
- Ordinary SUM/MEAN declarations, helpers, signatures, public behavior, and tests remain unchanged.
  `where`, cast, binary arithmetic, factory constants, typed scalar values, Shape/ShapeBroadcast,
  and producer/provenance foundations remain accurate unchanged because this task consumes their
  existing contracts without changing them. Training API remains accurate because no gradient,
  optimizer, or training behavior was added. Architecture and focused explanatory documents,
  ADRs/tests, backend conformance, integration, Gradle/dependencies, and other modules remain
  accurate unchanged because the work is model-owned semantic metadata and local Tensor
  construction only.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [backend glossary](../../../../glossary.md#backend); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `git diff --check` passed on the final combined change.

## Implementation notes

- Simplified the masked attributes record atomically; no compatibility constructor, accessor,
  alias, or fallback mapper remains.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Replaced heuristic placement with the existing ordinary right-aligned Shape rule and a
  directional equality check that prevents the mask from enlarging the input domain.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic); [Shape glossary](../../../../glossary.md#shape); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Retained the public overloads as one first-class masked SUM/MEAN occurrence each. Callers use
  visible <span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`reshape`, `expandDims`, or `expand` expressions when right alignment does not express
  their intended axes.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [axes glossary](../../../../glossary.md#term-axis))</span>
- Preserved exact input type and gradient-eligibility metadata, axis removal, unaffected Dimension
  references, unresolved layout, freshness, and output-index-zero provenance without adding value
  evaluation or a gradient promise.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [Dimension glossary](../../../../glossary.md#dimension); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Finalized zero all-false sum and NaN all-false mean semantics, including static zero-sized axes
  and runtime zero-sized or all-false dynamic slices, without specifying a NaN payload or backend
  algorithm.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [runtime glossary](../../../../glossary.md#term-runtime); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend))</span>

## Completion summary

- Completed changes: Replaced masked-reduction axis mapping with explicit ordinary
  broadcast-to-input validation and one-axis attributes while preserving first-class masked
  SUM/MEAN producers.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation); [MEAN glossary](../../../../glossary.md#term-mean))</span>
- Files changed or created: Four production Java files, two focused tests, Tensor API, Compile
  API, glossary, capability baseline, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: The exact focused five-class command, all 720 model tests across 88
  suites, model Javadoc, compiled and executed Java 26 example, generated-Javadoc inspection,
  449-link/112-anchor Markdown validation, six official-URL checks, formatting/terminology/scope/
  status checks, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Independent clean-context review completed in the same overall
  change and reused the final implementation-test evidence without rerunning Java tests.
- Documentation impact: Tensor and Compile APIs, glossary, capability baseline, task evidence,
  model master plan, and roadmap were finalized; Training API and architecture documentation
  remain accurate unchanged for the recorded ownership reasons.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: All four affected production contracts were finalized and generated successfully.
- Glossary impact: Removed the live mapper term and documented ordinary alignment, explicit caller
  rank editing, excluded non-finite positions, and zero/NaN empty-selection semantics.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [rank glossary](../../../../glossary.md#term-rank); [finite glossary](../../../../glossary.md#term-finite); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018Q. Plan task 0018R separately before implementation.

Status: Complete
