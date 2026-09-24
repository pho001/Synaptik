# Task 0007: MPSGraph FLOAT32 SUM/MEAN Reduction Foundation

## Status

Blocked

The implementation/probe pass is complete and fail-closed. No production code, native code,
tests, or probe artifacts remain changed.

## Blocking evidence

The current Model contract requires FLOAT32 `SUM` to perform exact real addition followed by one
rounding to FLOAT32; `MEAN` is that exact sum divided by the positive reduction count, then rounded.
A real shape-specialized `MPSGraphExecutable`, bound directly to the scalar output buffer, was run
eight times with input raw FLOAT32 values `[1.0e20f, 1.0f, 1.0f, -1.0e20f]`. Every run returned:

- `SUM`: positive zero, bits `0x00000000`, instead of `2.0f`, bits `0x40000000`;
- `MEAN`: positive zero, bits `0x00000000`, instead of `0.5f`, bits `0x3f000000`.

This is a reproducible semantic failure, not tolerance noise or publication error. The direct
scalar destination and repeated identical result isolate the observed MPSGraph reduction
semantics from a host-copy fallback. Ordinary API feasibility was otherwise proved during the
reverted attempt: the current SDK selectors are `reductionSumWithTensor:axes:name:` and
`meanOfTensor:axes:name:`, keep-dimensions can be represented by an exact reshape, and the typed
schema plus rank-zero path compiled. None of that evidence satisfies the numerical contract.

The task therefore remains Blocked rather than weakening Model semantics, introducing a hidden
fallback, or silently authorizing custom kernels. Unblocking requires an exact bounded
implementation, such as separately authorized custom kernels, or an explicit Model contract
change. Neither is authorized here.

## Change class

Class C. This was a bounded Metal backend extension after the blocked 0006 unary experiment. It
would have owned typed reduction lowering, route identity, Metal resources, lifecycle evidence,
and the backend-local scalar publication seam; it did not alter shared transfer policy.

## Goal

Add one typed whole-partition MPSGraph route for `AggregateReductionKind.SUM` and `MEAN` over
static canonical-contiguous `FLOAT32` values. Admit full reduction (`NoOperationAttrs.INSTANCE`),
one-axis reduction (`AxisReductionAttrs`), ordered multi-axis reduction
(`MultiAxisReductionAttrs`), and binding-aware `SUM_TO_SHAPE` (`SumToShapeAttrs`). Preserve exact
axis order as semantic metadata while passing a normalized axis set to MPSGraph. Support
`keepDimensions` exactly, including rank-0 scalar results. Do not admit masked, statistical,
product, extrema, logical, arg-extrema, scan, normalization, or any other reduction family.

The typed Model and Compiler prerequisites exist without reopening shared architecture: Model has
the required records, Compiler infers exact output Shapes and expands SUM_TO_SHAPE gradients, and
`0005` is the architectural predecessor. Blocked 0006 is not a dependency. The real numerical
probe above nevertheless prevents this route from becoming production capability.

## Model and Compiler contract readout

- `AggregateReductionKind.SUM` and `MEAN` use `NoOperationAttrs` for all axes, typed
  `AxisReductionAttrs(axis, keepDimensions)` for one axis, and typed
  `MultiAxisReductionAttrs(axes, keepDimensions)` for multiple axes. Axes are normalized,
  non-negative, distinct, and immutable before lowering.
- `SumToShapeAttrs(targetShape)` carries the exact target Shape. Compiler right-aligns source and
  target; leading source axes reduce, target extent one reduces an aligned source axis, and equal
  extents preserve it. Other pairs fail at binding. The output descriptor is exactly targetShape.
- Compiler inference computes rank changes and rank-0 Shape precisely. `ReductionGradientRules`
  restores ordinary SUM/MEAN gradients with `expandDims` when axes are removed and then `expand`
  to the source Shape; `SUM_TO_SHAPE` backward directly calls `gradient.expand(sourceShape)`.
  Every nontrivial admitted reduction backward contains an EXPAND path. Metal 0008 may implement
  that affine operation independently, but it cannot make the blocked reduction forward route
  executable or establish a complete Metal-only reduction backward path.
- Compiler-generated gradient graphs were intended as a shape/contract gate only. No executable
  Metal-only backward claim survives this task's numerical failure.
- No generic attribute bag is permitted. The backend schema must distinguish the four typed forms
  and carry only validated axes, keep-dims, and right-aligned target geometry.

## MPSGraph lowering and geometry

The reverted feasibility pass confirmed the documented typed MPSGraph selectors
`reductionSumWithTensor:axes:name:` and `meanOfTensor:axes:name:`. Keep-dimensions can be expressed
by reshaping the reduced result to the exact inferred retained-axis Shape. A future replacement
would still build one graph for the complete maximal Metal-owned partition, feed tensors in stable
order, and target tensors in stable order. For ordinary reductions, lowering would supply the
normalized selected axes. For full reduction, it would supply every input axis. For
SUM_TO_SHAPE, it would derive the exact reduced-axis set from the bound source/target pair, reduce
those axes, and retain equal aligned axes; when no axis reduces, the route would be a typed
identity/copy result rather than an accidental reduction.

Admit fully static positive input Shapes of rank `1..16`, FLOAT32, canonical dense-contiguous,
zero-offset, non-view layouts, and equal `requiresGrad` flags. An output may have rank `0..16` and
must be canonical dense-contiguous with its exact inferred Shape. All element and byte arithmetic
is checked. Any unresolved Shape/layout, zero extent, view/strided layout, unsupported attrs,
wrong kind/type, malformed axis set, or partition topology fails closed before preparation.

## Rank-0 materialization and transfer boundary

Rank-0 is a Metal-owned output capability, not a transfer expansion. The implementation must add a
backend-local scalar materialization path that downloads exactly four FLOAT32 bytes from the
assigned Metal output buffer into detached canonical host bytes (and exposes the scalar through the
existing Metal publication/materialization adapter). It must validate scalar Shape, FLOAT32 type,
canonical zero-offset layout, byte count four, context, openness, and current-thread access before
copying. The real CPU-free Engine scenario must compile, run, publish, and materialize a scalar
SUM/MEAN result without a CPU partition or CPU representation.

Do **not** widen CPU↔Metal transfer. The existing cross-owner contract remains positive static
rank `1..16` canonical contiguous FLOAT32 only; rank-0 values cannot be inserted into a mixed-owner
schedule and must fail before backend analysis. Do not silently widen rank, data type, conversion,
heap staging, or transfer directions. If the existing public publication adapter cannot represent
rank-0 without changing a shared contract, stop and mark this task Blocked rather than adding a
shim or generic representation.

## Typed versioned ABI and identity

Keep native ABI version 4 and its exact thirteen-symbol export set unchanged; reduction operation
identity is encoded in the existing typed whole-partition node schema and requires no new native
export. Add typed operation values `SUM` and `MEAN`, a typed reduction-form discriminator, and
bounded fixed-width fields for keep-dims, axis count/axes, and target rank/dimensions. No strings,
reflection, raw hidden dimensions, or generic maps. Reject malformed records consistently in Java
and native validation.

Bump the Java/native node schema, candidate schema, workload fingerprint schema, and route-policy
version (proposed reduction schema `v2`, candidate/fingerprint `v3`, policy `v3`; final numbers are
implementation-owned but must be monotonic and recorded). Old decisions must fail closed. Fingerprints
include kind, typed attrs, normalized axes, keep-dims, source/output descriptors and layouts,
right-aligned target geometry, ordered topology, numerical policy, and ABI schema.

## Numerical and special-value gates

The controlling gate is exact, not an implementation-selected ULP tolerance: Model FLOAT32 `SUM`
means exact real addition followed by one result-format rounding, and `MEAN` means that exact sum
divided by the positive reduction count followed by result-format rounding. NaN, infinity, signed
zero, empty-domain, and count semantics remain those of the current Model contract. The real
counterexample in **Blocking evidence** fails this gate for both operations, so no production
route may be admitted. Any future restart must first prove the cancellation vector and then the
full, single-axis, multi-axis, keep-dimensions, and SUM_TO_SHAPE matrices without relaxed math,
flush-to-zero, narrower accumulation, reassociation that changes the required result, or a
different data type.

## Lifecycle and implementation boundaries

Analysis validates the complete maximal partition and declares exact feed/target buffers and the
MPSGraph address workspace before assignment. Finalization creates one reusable shape-specialized
MPSGraph executable transactionally and transfers it through Prepare. Runtime cold binding checks
live typed Metal representations once; hot execution makes one synchronous native submission into
assigned outputs. Repeated runs use fresh run-owned buffers and reuse the prepared executable.
No CPU fallback, route retry, graph inspection, per-node invocation, hidden output copy, or Engine
lookup is allowed. Candidate generation and authenticated compatibility follow 0005 conventions.

## Acceptance gates

- Capability admits exactly SUM/MEAN and the four typed attr forms under the stated FLOAT32,
  static rank/layout/topology constraints; all unsupported cases fail closed.
- Full, single-axis, multi-axis, keep-dims, SUM_TO_SHAPE, rank-0, rank-1, and rank-16 output
  Shapes have exact descriptors and represented layout; axis and target validation is deterministic.
- Java/native typed schema agrees; ABI remains v4 with exactly thirteen exports; candidate,
  fingerprint, and route-policy versions are bumped and stale decisions reject.
- Fake-native coverage exercises every mapping, malformed records, dimensions/axes, rollback,
  close/rejection, exact declarations, reuse, and rank-0 materialization binding.
- Real native CPU-free Engine evidence covers full/single/multi-axis SUM and MEAN, keep-dims,
  SUM_TO_SHAPE, rank-0 materialization, repeated runs, two independent prepared sessions, direct
  publication, close/rejection, and no CPU integration. A separate negative test proves rank-0
  CPU↔Metal transfer is rejected and rank-1..16 transfer behavior is unchanged.
- Compiler-generated SUM/MEAN forward and autograd graphs are built, and their inferred gradient
  Shape/operation contracts may be inspected as a non-execution gate in a future restart.
  Executable Metal-only backward remains unavailable while the reduction route is blocked; Metal
  0008's affine scope does not change that result.
- Documentation, Javadocs, native schema audit, lifecycle review, architecture checks, and an
  independent Class C review pass; no production claim is made before all gates pass.

## Dependencies and review

Depends on Metal 0005, current Model reduction contracts, Compiler autograd/inference, Engine
0017, Compiler 0006B7, Prepare 0008, and Runtime 0016. It does not depend on Metal 0006; both 0006
and 0007 are independently Blocked by exact semantic failures. Metal 0008 may proceed with
independent affine transforms, but it does not unblock reduction execution or claim Metal-only
reduction backward. A future 0007 restart must specifically review the replacement numerical
route, typed attrs schema, rank-zero publication, no-widening transfer behavior, candidate bump,
transactional cleanup, and truthful gradient scope.

## Validation

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

Blocked. Across eight real executions, MPSGraph returned positive zero for the cancellation input
whose Model-exact results are `SUM = 2.0f` and `MEAN = 0.5f`. Selector, keep-dimensions, typed
schema, and rank-zero API feasibility were proved but cannot overcome the numerical failure. No
production, native, test, or probe change remains; the repository is clean fail-closed.
