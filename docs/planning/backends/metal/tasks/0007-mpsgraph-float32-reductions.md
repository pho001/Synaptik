# Task 0007: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32) SUM/[MEAN](../../../../glossary.md#term-mean) Reduction Foundation

## Reader context

This probe separated three questions: could MPSGraph represent the reduction, did its computed
values satisfy Model's numerical rule, and could Metal publish the resulting shape? API and
shape feasibility did not overcome the cancellation counterexample. The implementation-shaped
sections below remain requirements of the blocked proposal.

Axes and output Shape describe which input terms belong together; the listed floating-point
words are actual test values. A rank-zero result is one stored scalar, not an empty tensor.
Its proposed local publication would copy four bytes into detached host storage, without
authorizing scalar CPU/Metal transfers. Compiler-generated gradient shapes likewise did not prove
an executable backward route.
The [current numerical domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
describe the later reduction implementation independently of this exact-contract failure.

## Status

Blocked

The implementation/probe pass is complete and [fail-closed](../../../../glossary.md#term-fail-closed). No production code, [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture),
tests, or probe [artifacts](../../../../glossary.md#term-artifact) remain changed.

## Blocking evidence

The current [Model](../../../../glossary.md#term-model) contract requires [FLOAT32](../../../../glossary.md#term-float32) `SUM` to perform exact real addition followed by one
[rounding](../../../../glossary.md#term-rounding) to FLOAT32; [`MEAN`](../../../../glossary.md#term-mean) is that exact sum divided by the positive reduction count, then rounded.
A real [shape](../../../../glossary.md#shape)-specialized `MPSGraphExecutable`, bound directly to the [scalar](../../../../glossary.md#term-scalar) output [buffer](../../../../glossary.md#term-buffer), was [run](../../../../glossary.md#term-run)
eight times with input raw FLOAT32 values `[1.0e20f, 1.0f, 1.0f, -1.0e20f]`. Every run returned:

- `SUM`: [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), bits `0x00000000`, instead of `2.0f`, bits `0x40000000`;
- `MEAN`: positive zero, bits `0x00000000`, instead of `0.5f`, bits `0x3f000000`.

This is a reproducible semantic failure, not [tolerance](../../../../glossary.md#term-tolerance) noise or [publication](../../../../glossary.md#term-publication) error. The direct
scalar destination and repeated identical result isolate the observed [MPSGraph](../../../../glossary.md#term-mpsgraph) reduction
semantics from a [host](../../../../glossary.md#term-host)-copy fallback. Ordinary [API](../../../../glossary.md#term-application-programming-interface-api) feasibility was otherwise proved during the
reverted attempt: the current [SDK](../../../../glossary.md#term-sdk) selectors are `reductionSumWithTensor:axes:name:` and
`meanOfTensor:axes:name:`, keep-[dimensions](../../../../glossary.md#dimension) can be represented by an exact reshape, and the typed
[schema](../../../../glossary.md#term-schema) plus [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) path compiled. None of that evidence satisfies the numerical contract.

The task therefore remains Blocked rather than weakening Model semantics, introducing a hidden
fallback, or silently authorizing custom [kernels](../../../../glossary.md#kernel). Unblocking requires an exact bounded
implementation, such as separately authorized custom kernels, or an explicit Model contract
change. Neither is authorized here.

## Change class

Class C. This was a bounded [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend) extension after the blocked 0006 [unary](../../../../glossary.md#term-unary) experiment. It
would have owned typed reduction [lowering](../../../../glossary.md#lowering), route [identity](../../../../glossary.md#term-identity-meanings), Metal resources, [lifecycle](../../../../glossary.md#lifecycle) evidence,
and the backend-local [scalar](../../../../glossary.md#term-scalar) [publication](../../../../glossary.md#term-publication) seam; it did not alter shared transfer policy.

## Goal

Add one typed whole-[partition](../../../../glossary.md#partition) [MPSGraph](../../../../glossary.md#term-mpsgraph) route for `AggregateReductionKind.SUM` and [`MEAN`](../../../../glossary.md#term-mean) over
static [canonical](../../../../glossary.md#term-canonical-form)-[contiguous](../../../../glossary.md#term-contiguous) [`FLOAT32`](../../../../glossary.md#term-float32) values. Admit full reduction (`NoOperationAttrs.INSTANCE`),
one-[axis](../../../../glossary.md#term-axis) reduction (`AxisReductionAttrs`), ordered [multi-axis reduction](../../../../glossary.md#multi-axis-reduction)
(`MultiAxisReductionAttrs`), and [binding](../../../../glossary.md#term-binding)-aware `SUM_TO_SHAPE` (`SumToShapeAttrs`). Preserve exact
axis order as semantic [metadata](../../../../glossary.md#term-metadata) while passing a [normalized axis](../../../../glossary.md#normalized-axis) set to MPSGraph. Support
`keepDimensions` exactly, including [rank](../../../../glossary.md#term-rank)-0 [scalar](../../../../glossary.md#term-scalar) results. Do not admit masked, statistical,
product, [extrema](../../../../glossary.md#saturation-pole-and-extremum), logical, arg-extrema, [scan](../../../../glossary.md#term-cumulative-scan), [normalization](../../../../glossary.md#term-normalization), or any other reduction family.

The typed [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) prerequisites exist without reopening shared architecture: Model has
the required records, Compiler infers exact output [Shapes](../../../../glossary.md#shape) and expands SUM_TO_SHAPE [gradients](../../../../glossary.md#term-gradient), and
`0005` is the architectural predecessor. Blocked 0006 is not a [dependency](../../../../glossary.md#term-dependency). The real numerical
probe above nevertheless prevents this route from becoming production capability.

## [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) contract readout

- `AggregateReductionKind.SUM` and [`MEAN`](../../../../glossary.md#term-mean) use [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) for all [axes](../../../../glossary.md#term-axis), typed
  `AxisReductionAttrs(axis, keepDimensions)` for one axis, and typed
  `MultiAxisReductionAttrs(axes, keepDimensions)` for multiple axes. Axes are normalized,
  non-negative, distinct, and immutable before lowering.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [lowering glossary](../../../../glossary.md#lowering))</span>
- `SumToShapeAttrs(targetShape)` carries the exact target Shape. Compiler right-aligns source and
  target; leading source axes reduce, target extent one reduces an aligned source axis, and equal
  extents preserve it. Other pairs fail at binding. The output descriptor is exactly targetShape.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [extent glossary](../../../../glossary.md#term-extent); [binding glossary](../../../../glossary.md#term-binding); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Compiler [inference](../../../../glossary.md#term-inference) computes [rank](../../../../glossary.md#term-rank) changes and rank-0 [Shape](../../../../glossary.md#shape) precisely. `ReductionGradientRules`
  restores ordinary SUM/MEAN gradients with <span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient))</span>`expandDims` when axes are removed and then `expand`
  to the source Shape; `SUM_TO_SHAPE` backward directly calls `gradient.expand(sourceShape)`.
  Every nontrivial admitted reduction backward contains an EXPAND path. Metal 0008 may implement
  that affine operation independently, but it cannot make the blocked reduction forward route
  executable or establish a complete Metal-only reduction backward path.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [operation glossary](../../../../glossary.md#operation))</span>
- Compiler-generated gradient graphs were intended as a shape/contract gate only. No executable
  Metal-only backward claim survives this task's numerical failure.<span data-glossary-navigation="true"> ([graphs glossary](../../../../glossary.md#graph))</span>
- No generic attribute bag is permitted. The backend schema must distinguish the four typed forms
  and carry only validated axes, keep-dims, and right-aligned target geometry.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [schema glossary](../../../../glossary.md#term-schema); [carry glossary](../../../../glossary.md#term-carry))</span>

## [MPSGraph](../../../../glossary.md#term-mpsgraph) [lowering](../../../../glossary.md#lowering) and geometry

The reverted feasibility pass confirmed the documented typed MPSGraph selectors
`reductionSumWithTensor:axes:name:` and `meanOfTensor:axes:name:`. Keep-[dimensions](../../../../glossary.md#dimension) can be expressed
by reshaping the reduced result to the exact inferred retained-[axis](../../../../glossary.md#term-axis) [Shape](../../../../glossary.md#shape). A future replacement
would still build one [graph](../../../../glossary.md#graph) for the complete maximal [Metal](../../../../glossary.md#term-metal)-owned [partition](../../../../glossary.md#partition), feed [tensors](../../../../glossary.md#tensor) in stable
order, and [target](../../../../glossary.md#term-target) tensors in stable order. For ordinary reductions, lowering would supply the
normalized selected axes. For full reduction, it would supply every input axis. For
SUM_TO_SHAPE, it would derive the exact reduced-axis set from the bound source/target pair, reduce
those axes, and retain equal aligned axes; when no axis reduces, the route would be a typed
[identity](../../../../glossary.md#term-identity-meanings)/copy result rather than an accidental reduction.

Admit fully static positive input Shapes of [rank](../../../../glossary.md#term-rank) `1..16`, [FLOAT32](../../../../glossary.md#term-float32), [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous),
zero-offset, non-[view](../../../../glossary.md#term-view-meanings) [layouts](../../../../glossary.md#layout), and equal `requiresGrad` flags. An output may have rank `0..16` and
must be canonical dense-contiguous with its exact inferred Shape. All [element](../../../../glossary.md#term-element) and byte arithmetic
is checked. Any unresolved Shape/layout, zero [extent](../../../../glossary.md#term-extent), view/strided layout, unsupported attrs,
wrong kind/type, malformed axis set, or partition [topology](../../../../glossary.md#term-topology) fails closed before preparation.

## [Rank](../../../../glossary.md#term-rank)-0 [materialization](../../../../glossary.md#materialization) and transfer boundary

Rank-0 is a [Metal](../../../../glossary.md#term-metal)-owned output capability, not a transfer expansion. The implementation must add a
[backend](../../../../glossary.md#backend)-local [scalar](../../../../glossary.md#term-scalar) materialization path that downloads exactly four [FLOAT32](../../../../glossary.md#term-float32) bytes from the
assigned Metal output [buffer](../../../../glossary.md#term-buffer) into [detached](../../../../glossary.md#detached-model-results) [canonical](../../../../glossary.md#term-canonical-form) [host](../../../../glossary.md#term-host) bytes (and exposes the scalar through the
existing Metal [publication](../../../../glossary.md#term-publication)/materialization adapter). It must validate scalar [Shape](../../../../glossary.md#shape), FLOAT32 type,
canonical zero-offset [layout](../../../../glossary.md#layout), byte count four, context, openness, and current-thread access before
copying. The real [CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine) scenario must [compile](../../../../glossary.md#compile), [run](../../../../glossary.md#term-run), publish, and materialize a scalar
SUM/[MEAN](../../../../glossary.md#term-mean) result without a CPU [partition](../../../../glossary.md#partition) or CPU representation.

Do **not** widen CPU↔Metal transfer. The existing cross-[owner](../../../../glossary.md#term-owner) contract remains positive static
rank `1..16` canonical [contiguous](../../../../glossary.md#term-contiguous) FLOAT32 only; rank-0 values cannot be inserted into a mixed-owner
schedule and must fail before backend analysis. Do not silently widen rank, [data type](../../../../glossary.md#data-type--datatype), conversion,
[heap](../../../../glossary.md#term-heap) staging, or transfer directions. If the existing public publication adapter cannot represent
rank-0 without changing a shared contract, stop and mark this task Blocked rather than adding a
shim or generic representation.

## Typed versioned [ABI](../../../../glossary.md#term-application-binary-interface-abi) and [identity](../../../../glossary.md#term-identity-meanings)

Keep native ABI version 4 and its exact thirteen-[symbol](../../../../glossary.md#term-symbol) export set unchanged; reduction [operation](../../../../glossary.md#operation)
identity is encoded in the existing typed whole-[partition](../../../../glossary.md#partition) [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) and requires no new native
export. Add typed operation values `SUM` and [`MEAN`](../../../../glossary.md#term-mean), a typed reduction-form discriminator, and
bounded fixed-width fields for keep-dims, [axis](../../../../glossary.md#term-axis) count/axes, and [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank)/[dimensions](../../../../glossary.md#dimension). No strings,
reflection, raw hidden dimensions, or generic maps. Reject malformed records consistently in Java
and native [validation](../../../../glossary.md#term-validation).

Bump the Java/native node schema, candidate schema, workload [fingerprint](../../../../glossary.md#term-fingerprint) schema, and route-policy
version (proposed reduction schema `v2`, candidate/fingerprint `v3`, policy `v3`; final numbers are
implementation-owned but must be monotonic and recorded). Old decisions must [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). Fingerprints
include kind, typed attrs, normalized axes, keep-dims, source/output [descriptors](../../../../glossary.md#term-descriptor) and [layouts](../../../../glossary.md#layout),
right-aligned target geometry, ordered [topology](../../../../glossary.md#term-topology), numerical policy, and ABI schema.

## Numerical and special-value gates

The controlling gate is exact, not an implementation-selected [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) [tolerance](../../../../glossary.md#term-tolerance): [Model](../../../../glossary.md#term-model) [FLOAT32](../../../../glossary.md#term-float32) `SUM`
means exact real addition followed by one result-format [rounding](../../../../glossary.md#term-rounding), and [`MEAN`](../../../../glossary.md#term-mean) means that exact sum
divided by the positive reduction count followed by result-format rounding. [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed
zero](../../../../glossary.md#term-signed-zero), empty-domain, and count semantics remain those of the current Model contract. The real
counterexample in **Blocking evidence** fails this gate for both operations, so no production
route may be admitted. Any future restart must first prove the [cancellation](../../../../glossary.md#term-cancellation) [vector](../../../../glossary.md#term-vector) and then the
full, single-[axis](../../../../glossary.md#term-axis), multi-axis, keep-[dimensions](../../../../glossary.md#dimension), and SUM_TO_SHAPE [matrices](../../../../glossary.md#term-matrix-meanings) without relaxed math,
[flush-to-zero](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow), narrower accumulation, [reassociation](../../../../glossary.md#term-reassociation) that changes the required result, or a
different [data type](../../../../glossary.md#data-type--datatype).

## [Lifecycle](../../../../glossary.md#lifecycle) and implementation boundaries

Analysis validates the complete maximal [partition](../../../../glossary.md#partition) and declares exact feed/[target](../../../../glossary.md#term-target) [buffers](../../../../glossary.md#term-buffer) and the
[MPSGraph](../../../../glossary.md#term-mpsgraph) address workspace before assignment. Finalization creates one reusable [shape](../../../../glossary.md#shape)-specialized
MPSGraph executable transactionally and transfers it through [Prepare](../../../../glossary.md#prepare). [Runtime](../../../../glossary.md#term-runtime) cold [binding](../../../../glossary.md#term-binding) checks
live typed [Metal](../../../../glossary.md#term-metal) representations once; hot execution makes one [synchronous](../../../../glossary.md#term-synchronous) native submission into
assigned outputs. Repeated runs use fresh [run](../../../../glossary.md#term-run)-owned buffers and reuse the [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable).
No [CPU](../../../../glossary.md#term-cpu) fallback, route retry, [graph](../../../../glossary.md#graph) inspection, per-[node](../../../../glossary.md#node) invocation, hidden output copy, or [Engine](../../../../glossary.md#term-engine)
[lookup](../../../../glossary.md#term-lookup) is allowed. Candidate generation and authenticated compatibility follow 0005 conventions.

## Acceptance gates

- Capability admits exactly SUM/MEAN and the four typed attr forms under the stated FLOAT32,
  static rank/layout/topology constraints; all unsupported cases fail closed.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rank glossary](../../../../glossary.md#term-rank); [layout glossary](../../../../glossary.md#layout); [topology glossary](../../../../glossary.md#term-topology); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Full, single-axis, multi-axis, keep-dims, SUM_TO_SHAPE, rank-0, rank-1, and rank-16 output
  Shapes have exact descriptors and represented layout; axis and target validation is deterministic.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Shapes glossary](../../../../glossary.md#shape); [descriptors glossary](../../../../glossary.md#term-descriptor); [target glossary](../../../../glossary.md#term-target); [validation glossary](../../../../glossary.md#term-validation))</span>
- Java/native typed schema agrees; ABI remains v4 with exactly thirteen exports; candidate,
  fingerprint, and route-policy versions are bumped and stale decisions reject.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- Fake-native coverage exercises every mapping, malformed records, dimensions/axes, rollback,
  close/rejection, exact declarations, reuse, and rank-0 materialization binding.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [rollback glossary](../../../../glossary.md#term-rollback); [materialization glossary](../../../../glossary.md#materialization); [binding glossary](../../../../glossary.md#term-binding))</span>
- Real native CPU-free Engine evidence covers full/single/multi-axis SUM and MEAN, keep-dims,
  SUM_TO_SHAPE, rank-0 materialization, repeated runs, two independent prepared sessions, direct
  publication, close/rejection, and no CPU integration. A separate negative test proves rank-0
  CPU↔Metal transfer is rejected and rank-1..16 transfer behavior is unchanged.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Compiler-generated SUM/MEAN forward and autograd graphs are built, and their inferred gradient
  Shape/operation contracts may be inspected as a non-execution gate in a future restart.
  Executable Metal-only backward remains unavailable while the reduction route is blocked; Metal
  0008's affine scope does not change that result.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [graphs glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation, Javadocs, native schema audit, lifecycle review, architecture checks, and an
  independent Class C review pass; no production claim is made before all gates pass.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Dependencies and review

Depends on [Metal](../../../../glossary.md#term-metal) 0005, current [Model](../../../../glossary.md#term-model) reduction contracts, [Compiler](../../../../glossary.md#term-compiler) [autograd](../../../../glossary.md#autograd)/[inference](../../../../glossary.md#term-inference), [Engine](../../../../glossary.md#term-engine)
0017, Compiler 0006B7, [Prepare](../../../../glossary.md#prepare) 0008, and [Runtime](../../../../glossary.md#term-runtime) 0016. It does not depend on Metal 0006; both 0006
and 0007 are independently Blocked by exact semantic failures. Metal 0008 may proceed with
independent affine transforms, but it does not unblock reduction execution or claim Metal-only
reduction backward. A future 0007 restart must specifically review the replacement numerical
route, typed attrs [schema](../../../../glossary.md#term-schema), [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [publication](../../../../glossary.md#term-publication), no-[widening](../../../../glossary.md#term-widening) transfer behavior, candidate bump,
transactional cleanup, and truthful [gradient](../../../../glossary.md#term-gradient) [scope](../../../../glossary.md#term-scope).

## [Validation](../../../../glossary.md#term-validation)

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Reduction*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
git diff --check
```

## Result

Blocked. Across eight real executions, [MPSGraph](../../../../glossary.md#term-mpsgraph) returned [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) for the [cancellation](../../../../glossary.md#term-cancellation) input
whose [Model](../../../../glossary.md#term-model)-exact results are `SUM = 2.0f` and `MEAN = 0.5f`. Selector, keep-[dimensions](../../../../glossary.md#dimension), typed
[schema](../../../../glossary.md#term-schema), and [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [API](../../../../glossary.md#term-application-programming-interface-api) feasibility were proved but cannot overcome the numerical failure. No
production, native, test, or probe change remains; the repository is clean [fail-closed](../../../../glossary.md#term-fail-closed).
