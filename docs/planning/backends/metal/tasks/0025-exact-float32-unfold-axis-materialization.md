# Task 0025: Exact FLOAT32 UNFOLD_AXIS Materialization

## Status

Ready

Implementation is complete from exact clean planning revision
`c54ccbde0dc8defd4def1e25c3fd2898a1d66f4e`; the task remains the sole Ready Metal frontier for
independent lean Class C review. Metal 0024 is Complete at implementation
`a947e574732273bee4469d42afe8935082d53109` plus documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e`; its final independent review returned `APPROVE` with
zero findings.

## Change class

Class C — this adds a materializing, potentially overlapping window transform to both profile
matrices, evolves the private native node schema and every route identity, and exposes it through
public Engine execution. One smallest one-run real-device smoke is permitted for the concrete
undocumented raw-bit movement risk. Selector, Shape, rank, axis, size, step, profile, optimization,
context, and repetition matrices are prohibited.

## Coordination metadata

- Depends on: Metal 0024 Complete; Model 0017M–0017N, 0018R, and 0023D; Compiler 0005C; CPU
  0006A1; Config 0006; Engine 0018; current completed shared lifecycle and opaque-backend
  contracts.
- Conflicts with: any Metal capability, schema, native-preflight, candidate/codec,
  ingress/publication, materialization, or public Engine scope; any blocked Metal 0016–0018
  restart; two- or three-dimensional window, fold, pad, convolution, or pooling work; Model 0026;
  or shared-layer mutation.
- Parallel group: None.
- Common base revision: `c54ccbde0dc8defd4def1e25c3fd2898a1d66f4e`.
- Integration order: one optionality-free one-run movement smoke before production, schema/identity,
  capability/analysis, documented native selector composition, focused public proof,
  documentation, independent review.
- Integration validation: focused Metal/conformance/real-Engine checkpoint, native export audit,
  architecture checks, Javadoc, one final build, Markdown/diff validation.
- Shared-document integration owner: Metal 0025 implementer.

## Goal

Add only canonical positive-rank `FLOAT32` `WindowTransformKind.UNFOLD_AXIS` with exact
`UnfoldAxisAttrs(axis, size, step)`, identically under `STRICT_IEEE` and `ACCELERATOR`.

The operation materializes general-axis windows. It copies source representations exactly, may
repeat one source cell in several output positions when windows overlap, never aliases or mutates
the source, performs no arithmetic on values, and has no padding or dilation meaning.

## Exact capability and Shape domain

Admit one occurrence exactly when all of the following hold:

- kind and attributes are exactly `UNFOLD_AXIS + UnfoldAxisAttrs`; there is one ordered input and
  one output;
- input and output are both `FLOAT32`, have equal `requiresGrad`, and have fully static Shapes with
  strictly positive extents;
- input rank `R` is `1..15`, output rank is `R + 1` and at most 16, and normalized axis `a` is in
  `[0,R)`;
- input and output are canonical dense-contiguous, zero-offset, non-view values; the output is a
  fresh materialized value;
- window size `S` is in `1..16`, a deliberate fixed selector-expansion bound, step `T` is positive,
  and `S <= D` for selected input extent `D`;
- every subtract, multiply, add, element-count, byte-count, selector start/end, and native integer
  conversion is checked before native resource creation; and
- position count and output Shape are exactly

```text
P = floor((D - S) / T) + 1
output = input[0:a] + [P] + input[a+1:R] + [S]
```

For output coordinate `y[0..R]`, the exact source coordinate is:

```text
x[i] = y[i]                 when i != a
x[a] = y[a] * T + y[R]
```

The `S <= 16` bound limits native expansion to at most 16 slices, 16 singleton-axis insertions, and
one concatenation per semantic node. It is capability scope, not a Model limit, and must be present
in capability, Java preflight, native preflight, workload identity, and tests. Do not silently widen
it or conflate it with the independent rank-16 bound.

A valid canonical result may feed or be fed by any currently admitted same-profile operation whose
exact canonical state and type requirements hold. Caller and exact splat input retain current
behavior. Direct publication uses the existing canonical FLOAT32 target path. CPU/Metal transfer
remains canonical positive-rank FLOAT32 only and needs no widening.

Reject wrong attributes, scalar input, size zero or above 16, non-positive step, size above the
selected extent, rank overflow, dynamic or zero extents, wrong result Shape, wrong type, mismatched
`requiresGrad`, affine views, noncanonical layout, byte/element overflow, malformed topology, and
every unlisted window kind before native creation.

## Order, overlap, tails, and padding

Output order is row-major in the exact Shape above. The appended final coordinate is window offset
`k=0..S-1`; the selected-axis output coordinate is window position `p=0..P-1`. Therefore source
coordinate `p*T+k` is selected in increasing `k` within each output position.

When `T < S`, windows overlap. The same source representation is copied independently to every
corresponding output coordinate; no reduction, accumulation, deduplication, alias, or ordering-
dependent arithmetic exists. When `T > S`, gaps are not represented. Any suffix that cannot form a
complete window is omitted by the floor formula. There is no padding, dilation, wrap, clamp,
implicit zero, image layout, or ceil mode.

Every output bit must equal the addressed input bit, including both zero signs, subnormals,
infinities, and quiet or signaling NaN payloads. The input remains unchanged.

## Exact documented selector mapping

The active Command Line Tools macOS 27.0 SDK is authoritative. Its
`MPSGraphTensorShapeOps.h` declares the category from macOS 11.0 and documents:

- `sliceTensor:starts:ends:strides:name:` as TensorFlow-style half-open strided slicing;
- `expandDimsOfTensor:axis:name:` as inserting one size-one dimension, available from macOS 12.3;
  and
- `concatTensors:dimension:name:` as concatenating its ordered tensor list along one dimension.

The existing backend already relies on the expand-dimensions selector, so this task introduces no
new deployment-floor claim. Lower one semantic node as follows, with all arrays and arithmetic
created only during shape-specialized native compilation:

1. For each `k` in ascending `0..S-1`, create rank-`R` `starts`, `ends`, and `strides` arrays.
2. On every `i != a`, use `start=0`, `end=inputExtent[i]`, and `stride=1`.
3. On axis `a`, use `start=k`, `stride=T`, and
   `end=k + (P - 1) * T + 1`. The documented half-open rule yields exactly `P` values
   `k, k+T, ..., k+(P-1)T`.
4. Append a singleton final dimension to each slice with
   `expandDimsOfTensor:axis:R:name:`.
5. Concatenate the tensors in ascending `k` order with
   `concatTensors:dimension:R:name:`. The result Shape and coordinate mapping are exactly the Model
   contract above.

No undocumented selector behavior is used for type, Shape, mapping, overlap, tail, or padding.
Native creation must independently rederive `P`, authenticate the exact input/output Shapes and
canonical value states, prove every selector integer representable, and reject a malformed record
before graph compilation.

## Minimal device evidence policy

The SDK documents the structural selector mapping but does not promise preservation of every
`FLOAT32` payload bit through the composed slice/expand/concat path. That is the only concrete
undocumented behavior risk.

Before production edits, run at most one disposable Objective-C program, once, using one context,
one graph, one executable, optimization level zero, and one execution. Use canonical rank-two
input `[2,6]`, `axis=1`, `size=3`, and `step=2`, producing `[2,2,3]`. This single geometry exercises
an unaffected leading axis, overlap at source coordinate two, and an omitted incomplete tail.
Include signed zeros, a subnormal, infinities, and quiet/signaling NaN payloads; compare output raw
bits against the formula, verify the input unchanged, and remove source and binary immediately.

Do not probe size/rank/axis/step/profile/context/optimization/repetition combinations, invalid
geometry, padding, folds, im-to-column selectors, or performance. A failed raw-bit cell blocks the
task without production changes; a pass authorizes only this exact documented composition.

## Native schema, identity, lifecycle, and cache

- Retain native ABI version 4, every status/signature, and exactly thirteen exports.
- Evolve fixed 160-byte node schema `10 -> 11`; retain operation wires `1..18` and append only
  `UNFOLD_AXIS=19`.
- Append typed attribute discriminator `WINDOW_AXIS=6`. Encode exactly three semantic fields:
  normalized axis in the existing axis cell, positive size in `attributeValues[0]`, and positive
  step in `attributeValues[1]`. Set `attributeCount=3`, auxiliary to zero, and every remaining
  attribute/padding cell to zero. Native and Java validators must treat this as one closed typed
  form rather than generic integers.
- Reject unknown wire `20+`, attribute `7+`, stale schema, nonzero sentinel, size above 16, wrong
  value state/type/rank/Shape, and any malformed attribute topology before creating a graph
  resource.
- Advance workload, exact-policy, candidate, compatibility, route-policy, and codec identity
  versions together from 11 to 12. Identity covers profile, schema, operation, axis/size/step,
  ordered topology, descriptors/states, selector-expansion bound, feeds/targets, ABI, and splats.
  Version-eleven or corrupt decisions reject or miss safely; no migration reader or alias remains.
- Keep the existing whole-partition `MPSGRAPH` candidate, direct targets, exact binding order,
  leases, rollback, reuse, concurrency, close/quiescence, cleanup, context/session isolation, and
  candidate authentication. No workspace, custom kernel, late route, retry, host result, or CPU
  fallback is added.

## Compiler and backward boundary

Compiler production remains unchanged. Current `LayoutInference` owns the exact Shape formula, and
`LayoutGradientRules` defines the floating input cotangent as
`gradient.foldAxis(axis, originalInputExtent, step)`.

Metal 0025 adds forward `UNFOLD_AXIS` only. It does not admit `FOLD_AXIS`, whose semantics start from
represented positive zero and add overlapping contributions, nor does it claim that MPSGraph
floating accumulation satisfies either numerical profile. A forward occurrence may retain
`requiresGrad=true`, but a compiled backward graph containing `FOLD_AXIS` must fail owner
selection/preparation before Metal native creation unless another explicit owner handles it. Do not
add a decomposition, cross-owner transfer, seeded-gradient success claim, training path, or
compiler special case.

## Deferred window workstream findings

The same active SDK exposes `MPSGraphImToColOps.h` selectors from macOS 14.0, but current
documentation is insufficient for an exact Synaptik task:

- `imToColWithSourceTensor:descriptor:name:` documents rank-four source plus kernel, stride,
  dilation, asymmetric padding, and layout, but not its output rank/axis order, exact padding sample
  representation, terminal ceil-grid behavior, or a configurable padding value. Model
  `UNFOLD2D` requires canonical rank-three `[N,C*kH*kW,outH*outW]`, exact channel/kernel/spatial
  order, explicit floor/ceil behavior, and either represented positive zero or an arbitrary exact
  typed padding scalar.
- `colToImWithSourceTensor:outputShape:descriptor:name:` documents a rank-four source, while Model
  `FOLD2D` consumes rank-three canonical columns. It does not document overlap contribution order,
  represented FLOAT32 accumulation, padding exclusion, or terminal ceil-tail exclusion.
- The active SDK provides no corresponding documented three-dimensional im-to-column/column-to-
  image selector for current `UNFOLD3D`/`FOLD3D` semantics.
- General `FOLD_AXIS` needs exact overlap addition and represented order. Neither the structural
  slice composition above nor current SDK documentation supplies that numerical contract.

Therefore `FOLD_AXIS`, both `UNFOLD2D` variants, `FOLD2D`, `UNFOLD3D`, and `FOLD3D` remain closed.
After 0025, the next safe window slice is a documentation/evidence decision for direct positive-zero
`UNFOLD2D`; it must first establish exact output mapping, floor/ceil tail sampling, and positive-zero
padding without claiming typed padding or folds. No implementation task for that slice is Ready
now.

## Non-goals

No `FOLD_AXIS`; two- or three-dimensional unfold/fold; `PAD`; convolution or pooling; overlap
addition; FLOAT64/BFLOAT16/FLOAT16/integral/BOOL window; scalar/dynamic/zero-extent/view form;
window size above 16; padding/dilation/ceil mode; asymmetric window; output alias; transfer widening;
custom kernel; host result; compiler/shared production change; backward/training claim; new export,
dependency, route, cache migration, performance claim, or selector matrix.

## Files and symbols

Expected production scope is limited to existing Metal-private files:

- `MetalCapabilityProvider`, `MetalNativeApi` — exact both-profile occurrence and Java ABI
  preflight;
- `MetalMpsGraphProgram` — schema 11, wire 19, typed `WINDOW_AXIS`, and 160-byte encoding;
- `MetalNegPartitionPreparer` — canonical state/topology validation and exact node lowering;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec` — version-twelve
  identity cutover;
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — schema validation and documented
  slice/expand/concat construction; and
- existing package documentation where the private program vocabulary is enumerated.

Reuse existing focused test classes. Add no window-specific test class when an existing capability,
raw-ABI, native program, prepared execution, candidate/codec, conformance, or public Engine method
can defend the observable contract. Shared Model, Compiler, Planning, Prepare, Runtime, Config,
Trace, Engine, Training, and transfer production remain read-only.

## Lean proof and validation

Permanent proof must be contract-minimal:

- extend the existing capability method for one valid overlapping/tail occurrence under both
  profiles and grouped exclusions for wrong kind/attrs/type/rank/Shape/layout/size/profile state;
- extend existing raw-schema tests for wire 19, `WINDOW_AXIS`, exact 160-byte cells, and malformed
  or stale-schema rejection;
- extend one existing native program test with the same compact mapping geometry and exact raw-bit
  output/input-preservation assertion;
- update existing identity/codec tests for axis/size/step separation and version-eleven rejection;
- extend one existing maximal-partition conformance test so `UNFOLD_AXIS` composes with a currently
  supported canonical consumer; and
- extend one existing CPU-free public Engine integration method for exact Shape and raw-bit output.

Use current generic lifecycle, candidate, codec, malformed-record, target publication, reuse,
concurrency, close, ABI/export, architecture, and documentation coverage. Do not duplicate those
matrices for one wire. Run focused commands while implementation stabilizes, then the full build
exactly once. Independent lean Class C review reuses successful evidence and runs no device work
unless it identifies one new concrete risk.

## Acceptance and review

- Only the exact bounded canonical FLOAT32 `UNFOLD_AXIS` occurrence is advertised and executable
  identically under both profiles.
- Shape and every output coordinate match the documented slice/expand/concat construction;
  overlapping source bits repeat exactly, incomplete tails disappear, no padding exists, and input
  storage is unchanged.
- Schema 11/wire 19/attribute 6 and version-twelve identities fail closed while ABI 4, thirteen
  exports, existing ingress/publication, and FLOAT32 transfer remain exact.
- Compiler production is unchanged; generated `FOLD_AXIS` backward remains unsupported by Metal.
- The one bounded smoke, focused reused proof, final validation, synchronized current docs, clean
  diff, and independent Class C `APPROVE` with zero findings close the task.

## Implementation checkpoint for independent review

Implementation began from exact clean planning revision
`c54ccbde0dc8defd4def1e25c3fd2898a1d66f4e`, correcting the preplanning common-base metadata
`aa191ca469010d081150e97dcddd504ec626dd9e`. Before any production edit, one disposable
Objective-C program compiled against the production frameworks and ran exactly once on the M3 Max
with one context, graph, executable at optimization level zero, and execution. Its canonical
`[2,6]`, axis-one, size-three, step-two slice/expand/concat run reported
`task0025 UNFOLD_AXIS smoke passed: exact overlap/tail raw bits and unchanged input`. The corpus
covered signed zeros, positive and negative subnormals, infinities, and quiet/signaling NaN
payloads. The exact `[2,2,3]` output proved overlap and omitted-tail mapping, the input was
unchanged, and the source and executable were removed immediately. No Shape, rank, axis, size,
step, profile, context, optimization, repetition, invalid-geometry, padding, fold, or matrix probe
ran.

The implementation adds only bounded canonical FLOAT32 `UNFOLD_AXIS` to both profile matrices.
Schema 11 retains the 160-byte ABI-v4 record and wires `1..18`, appends `UNFOLD_AXIS=19` and
`WINDOW_AXIS=6`, and carries normalized axis, size, and step in the closed typed form. Java and
native preflight independently rederive the floor count and exact Shape, require canonical
rank-`1..15` input and output, size `1..16`, positive step, representable selector values, and
fresh canonical FLOAT32 output. Native lowering creates half-open strided slices in ascending
window-offset order, appends one final singleton dimension to each, and concatenates along that
dimension. Workload, exact-policy, candidate, compatibility, route-policy, and codec identities
are version twelve and reject version eleven. ABI 4, thirteen exports, existing
ingress/publication, canonical FLOAT32 transfer, shared production, compiler behavior, and the
unsupported fold/backward boundary are unchanged.

Worker evidence passed:

- `native/metal-macos-arm64/build.sh`, an exact thirteen-symbol `nm -gU` export audit, and
  Foundation, Metal, and MetalPerformanceShadersGraph linkage audit;
- named focused capability, exact 160-byte schema/native-malformed, native raw-bit mapping/input-
  preservation, axis/size/step identity, version-eleven codec rejection, maximal-partition
  composition, and CPU-free both-profile public Engine tests;
- `:testing:architecture-tests:test` and `:backends:metal:javadoc`; and
- the single final `./gradlew build` invocation: `BUILD SUCCESSFUL`, 87 actionable tasks, eight
  executed and 79 up-to-date.

The first independent review returned `BLOCK` with two remediation findings. Remediation began from
exact clean implementation checkpoint `44edd86092509348e1e72cf7f0f4c3b13d141fa8` and does not claim
review approval. Java now owns a distinct fixed
`MetalMpsGraphProgram.MAX_SELECTOR_EXPANSION=16` constant, native owns the matching fixed
`SYNAPTIK_MAX_SELECTOR_EXPANSION=16U` constant, and capability, Java schema/preflight, and native
preflight no longer reuse rank or an unnamed literal for the selector-expansion policy. Focused
capability and raw/native proofs use selected extent 17 with exact size-17 output geometry, so size
17 fails solely at that cap. The capability proof now independently covers the typed-attribute and
occurrence-cardinality boundaries, both FLOAT32 positions, scalar and rank-16 input, exact Shape,
input and output layout, `requiresGrad`, size/selected-extent, step-derived position count, and both
profile states; layout and gradient mismatch cases do not mask one another. Raw validation also
covers zero size and zero step.

Remediation evidence passed without rerunning the already successful full build:

- `native/metal-macos-arm64/build.sh`; and
- the two named `MetalCapabilityProviderTest` and `MetalMpsGraphRawAbiNativeTest` remediation
  methods plus `:backends:metal:javadoc` in one focused Gradle invocation: `BUILD SUCCESSFUL`, 20
  actionable tasks, four executed and 16 up-to-date.

The exact 32-path checkpoint is:

- production under
  `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/`:
  `MetalCapabilityProvider.java`, `MetalMpsGraphProgram.java`, `MetalNativeApi.java`,
  `MetalNegPartitionPreparer.java`, `MetalNegRouteCandidateGenerator.java`,
  `MetalNegTuningBatch.java`, `MetalNegTuningCodec.java`, and `package-info.java`; plus
  `native/metal-macos-arm64/src/synaptik_metal_foundation.m`;
- tests under `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/`:
  `MetalCapabilityProviderTest.java`,
  `MetalMpsGraphAbsNativeTest.java`, `MetalMpsGraphAffineSchemaTest.java`,
  `MetalMpsGraphBinaryNativeTest.java`, `MetalMpsGraphIndexingNativeTest.java`,
  `MetalMpsGraphRawAbiNativeTest.java`, `MetalMpsGraphReductionNativeTest.java`, and
  `MetalNegRouteCandidateGeneratorTest.java`; plus
  `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java`
  and
  `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.java`;
- documentation: `ARCHITECTURE.md`, `docs/api/compile-api.md`, `docs/api/public-api.md`,
  `docs/api/tensor-api.md`, `docs/architecture/contracts/backend-execution.md`,
  `docs/architecture/module-boundaries.md`, `docs/backend-guide/metal-backend.md`,
  `docs/backend-guide/partition-preparer.md`, `docs/glossary.md`,
  `docs/planning/backends/metal/master-plan.md`,
  `docs/planning/backends/metal/tasks/0025-exact-float32-unfold-axis-materialization.md`,
  `docs/planning/roadmap.md`, and `native/metal-macos-arm64/README.md`.

Task 0025 remains Ready for independent lean Class C review. This checkpoint makes no review
result or approval claim.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
- [Prepare and run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)

## Architecture impact

Impact: None. The implementation extends only the existing backend-private typed MPSGraph program
and canonical materialization route. Shared Model, Compiler, Planning, Prepare, Runtime, Config,
Trace, Engine, Training, and transfer production remain unchanged.
