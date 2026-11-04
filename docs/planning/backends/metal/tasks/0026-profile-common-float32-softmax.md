# Task 0026: Profile-Common [Canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) [SOFTMAX](../../../../glossary.md#softmax--log-softmax)

## Status

Blocked

## [Planning](../../../../glossary.md#term-planning) decision

The broad todo `Implementovat Metal 0018 softmax a normalizace` becomes one new serial [Metal](../../../../glossary.md#term-metal) task,
not a restart or renumbering of historical blocked Metal 0018. This task implements only first-class
`SoftmaxKind.SOFTMAX`. It does not implement `LOG_SOFTMAX`, [layer normalization](../../../../glossary.md#layer-normalization), [RMS normalization](../../../../glossary.md#rms-normalization),
or [batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training).

The smallest coherent [slice](../../../../glossary.md#slice) is [canonical](../../../../glossary.md#term-canonical-form) static positive-[Shape](../../../../glossary.md#shape) [`FLOAT32`](../../../../glossary.md#term-float32) [SOFTMAX](../../../../glossary.md#softmax--log-softmax) forward with the
same [finite](../../../../glossary.md#term-finite) policy under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator), plus only the existing [Compiler](../../../../glossary.md#term-compiler)-generated
SOFTMAX [gradient](../../../../glossary.md#term-gradient) [graph](../../../../glossary.md#graph) under `ACCELERATOR`. There is no [backend](../../../../glossary.md#backend) gradient [operation](../../../../glossary.md#operation), selector,
[fusion](../../../../glossary.md#term-fusion), rewrite, or semantic kind.

There is deliberately no selected-[axis](../../../../glossary.md#term-axis) upper bound. A width limit such as 16 would be arbitrary:
[Model](../../../../glossary.md#term-model) defines no such limit, the direct [MPSGraph](../../../../glossary.md#term-mpsgraph) selector performs no selector-list expansion, and
the existing `MAX_SELECTOR_EXPANSION=16` belongs only to `UNFOLD_AXIS`. Admit every strictly
positive selected [extent](../../../../glossary.md#term-extent) whose complete static Shape, [element](../../../../glossary.md#term-element) count, byte count, and native integer
conversions pass the existing checked Metal geometry. [Rank](../../../../glossary.md#term-rank) remains bounded to 1..16 by the current
Metal [ABI](../../../../glossary.md#term-application-binary-interface-abi) [dimension](../../../../glossary.md#dimension) table.

## Change class

Class C — this adds a floating numerical selector to both [graph](../../../../glossary.md#graph)-wide profile [matrices](../../../../glossary.md#term-matrix-meanings), evolves the
private [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) and all route [identities](../../../../glossary.md#term-identity-meanings), and proves public [Engine](../../../../glossary.md#term-engine) forward execution plus
[accelerator](../../../../glossary.md#term-accelerator) generated-[gradient](../../../../glossary.md#term-gradient) closure. One smallest one-[run](../../../../glossary.md#term-run) real-[device](../../../../glossary.md#term-device) numerical gate is required
before production edits because the active [SDK](../../../../glossary.md#term-sdk) documents the [SOFTMAX](../../../../glossary.md#softmax--log-softmax) [operation](../../../../glossary.md#operation) and [axis](../../../../glossary.md#term-axis) but gives no
[finite](../../../../glossary.md#term-finite) accuracy bound. No [Shape](../../../../glossary.md#shape), width, axis, profile, optimization, context, or repetition matrix
is permitted.

## Coordination [metadata](../../../../glossary.md#term-metadata)

- Depends on: [Metal](../../../../glossary.md#term-metal) 0025 Complete at final reviewed revision
  `f88066e3ad0547987bb03b2d18ed2813f97de223`; Model 0016I–0016J; Compiler 0005B/0005C; CPU 0007E
  only as independent numerical-policy precedent; Config 0006; Engine 0018; current completed Metal
  lifecycle, profile, schema, and opaque-route contracts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [schema glossary](../../../../glossary.md#term-schema); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- [Planning](../../../../glossary.md#term-planning) base: exact clean reviewed Metal 0025 revision
  `f88066e3ad0547987bb03b2d18ed2813f97de223`. Implementation begins only from the committed revision
  containing this plan.
- Conflicts with: any Metal capability, schema, native-preflight, candidate/codec, materialization,
  or public Engine scope; any blocked Metal 0016–0018 restart; another normalization, loss,
  attention, fusion, or shared semantic task.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Parallel group: None.
- Integration order: one fail-closed direct-selector numerical gate; schema/identity; capability and
  Java/native preflight; native lowering; focused forward and generated-gradient composition;
  current documentation; independent lean Class C review.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [identity glossary](../../../../glossary.md#term-identity-meanings); [lowering glossary](../../../../glossary.md#lowering); [gradient glossary](../../../../glossary.md#term-gradient); [lean glossary](../../../../glossary.md#proof-assistant-and-model-checking-tools))</span>
- Integration validation: focused Metal/conformance/real-Engine proof, native export/link audit,
  architecture checks, Javadoc, one final build, Markdown/diff validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Metal 0026 implementer.

## [Model](../../../../glossary.md#term-model) and profile contract

Model already owns the complete public meaning:

- `SoftmaxKind.SOFTMAX` is one first-class one-input/one-output [operation](../../../../glossary.md#operation) paired exactly with
  `SoftmaxAttrs`;
- `SoftmaxAttrs.axis()` is already normalized, non-negative, and must be in range for the concrete
  input Shape;
- the output preserves the exact logical [Shape](../../../../glossary.md#shape), floating [data type](../../../../glossary.md#data-type--datatype), and `requiresGrad` eligibility;
- for one selected-[axis](../../../../glossary.md#term-axis) [slice](../../../../glossary.md#slice), the ideal result is
  `exp(x_i) / sum_j(exp(x_j))`, preserving every logical position; and
- Model deliberately defines no temperature, mask, scale, epsilon, algorithm switch, reduction
  order, finite-precision decomposition, or NaN/infinity/empty-axis policy.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [epsilon glossary](../../../../glossary.md#epsilon); [precision glossary](../../../../glossary.md#term-precision); [decomposition glossary](../../../../glossary.md#term-decomposition); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>

`STRICT_IEEE` means that exact current per-operation Model contract; it is not a universal
correct-[rounding](../../../../glossary.md#term-rounding) promise. [`ACCELERATOR`](../../../../glossary.md#term-accelerator) adds no [normalization](../../../../glossary.md#term-normalization) result freedom. The architecture's
profile table explicitly leaves normalization unrelaxed, so this task uses one identical forward
capability and [finite](../../../../glossary.md#term-finite) policy under both profiles. For every fixed [occurrence](../../../../glossary.md#term-occurrence), [Metal](../../../../glossary.md#term-metal) strict
capability is therefore a subset of Metal accelerator capability without [widening](../../../../glossary.md#term-widening) [SOFTMAX](../../../../glossary.md#softmax--log-softmax) results.
Any future accelerator-only [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), approximation, or special-value envelope requires a preceding
Model/architecture task and is not part of 0026.

## Exact capability domain

Admit one [occurrence](../../../../glossary.md#term-occurrence) exactly when all of the following hold:

- kind and attributes are exactly `SoftmaxKind.SOFTMAX + SoftmaxAttrs`;
- there is one ordered input and one fresh output;
- both [descriptors](../../../../glossary.md#term-descriptor) are [`FLOAT32`](../../../../glossary.md#term-float32), have the same exact [Shape](../../../../glossary.md#shape) and `requiresGrad`, and have canonical
  dense-contiguous, zero-offset, non-view layouts;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- the Shape is fully static, [rank](../../../../glossary.md#term-rank) 1..16, and every [extent](../../../../glossary.md#term-extent) is strictly positive;
- the [normalized axis](../../../../glossary.md#normalized-axis) is in `[0, rank)`; and
- checked [element](../../../../glossary.md#term-element) count, byte count, [dimension](../../../../glossary.md#dimension)-table cells, and every native `NSUInteger`/`NSInteger`
  conversion are representable before native resource creation.

The selected extent may be 1, 17, or any other positive representable value. Do not use
`MAX_SELECTOR_EXPANSION`, add a [softmax](../../../../glossary.md#softmax--log-softmax) width [constant](../../../../glossary.md#term-constant), or infer a width cap from the one-[run](../../../../glossary.md#term-run) probe.
The probe establishes one concrete undocumented numerical property of the direct selector; the
[SDK](../../../../glossary.md#term-sdk)'s [axis](../../../../glossary.md#term-axis)-generic selector declaration and shape-specialized [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) establish structural
applicability.

Reject `LOG_SOFTMAX`; wrong kind or attributes; wrong [arity](../../../../glossary.md#term-arity); non-FLOAT32 or cross-type descriptors;
[scalar](../../../../glossary.md#term-scalar), dynamic, zero-extent, rank-above-16, axis-out-of-range, Shape-mismatched, [layout](../../../../glossary.md#layout)-mismatched,
or `requiresGrad`-mismatched occurrences; affine [views](../../../../glossary.md#term-view-meanings); [aliases](../../../../glossary.md#aliasing-and-arena); [overflow](../../../../glossary.md#term-overflow); and every unlisted
[normalization](../../../../glossary.md#term-normalization) family before native creation.

## [Metal](../../../../glossary.md#term-metal)-private [finite](../../../../glossary.md#term-finite) numerical policy

For every structurally admitted [slice](../../../../glossary.md#slice) whose represented inputs are all finite, define the Task 0026
acceptance [oracle](../../../../glossary.md#term-test-oracle) as follows:

1. Convert each exact [FLOAT32](../../../../glossary.md#term-float32) input value to [binary64](../../../../glossary.md#floating-point-precision-and-rounding).
2. Find the slice maximum in binary64.
3. Evaluate `StrictMath.exp(x_i - maximum)` in binary64 and accumulate the terms with a compensated
   binary64 sum in increasing selected-axis coordinate order.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
4. Divide each binary64 exponential by that sum and narrow once to FLOAT32 with Java's
   round-to-nearest-ties-to-even conversion.<span data-glossary-navigation="true"> ([round-to-nearest glossary](../../../../glossary.md#floating-point-precision-and-rounding); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even))</span>
5. Require each Metal output to be finite, in `[0.0f, 1.0f]`, and at ordered FLOAT32 distance at
   most four result ULPs from that independently calculated reference value.

The four-[ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) bound matches the completed [CPU](../../../../glossary.md#term-cpu) FLOAT32 [softmax](../../../../glossary.md#softmax--log-softmax) accuracy envelope but does not require
CPU's private three-pass algorithm, compensation strategy, raw-bit result, or rejection machinery.
It is a Metal-private executable policy, not new [Model](../../../../glossary.md#term-model) semantics and not a cross-[backend](../../../../glossary.md#backend) equality
claim. The input [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) must remain unchanged and the output must be fresh [canonical](../../../../glossary.md#term-canonical-form) storage.
Cellwise oracle comparison is the complete accuracy gate; do not add a redundant independently tuned
sum-to-one [epsilon](../../../../glossary.md#epsilon).

[Signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) and [subnormal](../../../../glossary.md#term-subnormal) inputs are finite and remain inside this finite gate. A reference result
may narrow to a subnormal or zero. No [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) permission is added. [NaN](../../../../glossary.md#term-nan) and either [infinity](../../../../glossary.md#term-infinity) are
outside the advertised finite accuracy envelope because Model has not defined their SOFTMAX edge
policy. Task 0026 neither value-scans nor rejects them, freezes their observed [MPSGraph](../../../../glossary.md#term-mpsgraph) result, or
claims [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)/sign behavior; [descriptor](../../../../glossary.md#term-descriptor) capability has no value predicate. All permanent numerical
acceptance data is finite. Zero selected [extent](../../../../glossary.md#term-extent) and every other zero extent are structurally
rejected.

## One smallest real-[device](../../../../glossary.md#term-device) gate

The active macOS 27.0 [SDK](../../../../glossary.md#term-sdk)'s `MPSGraphActivationOps.h` declares
`softMaxWithTensor:axis:name:` but states no [finite](../../../../glossary.md#term-finite) error bound. That is one concrete undocumented
property necessary for the advertised four-[ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) behavior, so exactly one disposable real-device [run](../../../../glossary.md#term-run)
is mandatory before production edits.

Use one process, [Metal](../../../../glossary.md#term-metal) context/queue, [MPSGraph](../../../../glossary.md#term-mpsgraph), direct [SOFTMAX](../../../../glossary.md#softmax--log-softmax) [node](../../../../glossary.md#node), executable, and execution. Mirror
production rather than creating a favorable probe mode: leave the [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) [descriptor](../../../../glossary.md#term-descriptor) at its
production default optimization level, set and read back
`MPSGraphReducedPrecisionFastMathNone` where available, [compile](../../../../glossary.md#compile) one direct [target](../../../../glossary.md#term-target), and set
`MPSGraphOptionsNone` on the executable. Use one [canonical](../../../../glossary.md#term-canonical-form) `[4,8]` [FLOAT32](../../../../glossary.md#term-float32) input and [axis](../../../../glossary.md#term-axis) one with
these four frozen finite slices:

```text
[1, 2, 3, -1, 0.5, -2, 4, 0]
[999997, 999998, 999999, 1000000, 1000001, 1000002, 1000003, 1000004]
[0, -80, -90, -100, -103, -104, -110, -120]
[+0, -0, +minSubnormal, -minSubnormal, +maxFinite, +maxFinite, -maxFinite, 1]
```

Compare all 32 cells to the independent policy above, require finite `[0,1]` results and unchanged
input [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), then remove probe source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) immediately. This one execution covers ordinary
finite values, a large common offset, strongly skewed [subnormal](../../../../glossary.md#term-subnormal)/zero probability results, signed
zeros, subnormal inputs, equal maxima, and the complete finite range. Do not probe another width,
axis, [rank](../../../../glossary.md#term-rank), profile, optimization level, context, repetition, [NaN](../../../../glossary.md#term-nan), or [infinity](../../../../glossary.md#term-infinity).

If any cell fails, mark Task 0026 `Blocked`, retain only exact evidence, remove the probe, and make no
production change. Do not narrow capability to the probe width, weaken the [tolerance](../../../../glossary.md#term-tolerance), move SOFTMAX
to [accelerator](../../../../glossary.md#term-accelerator)-only, decompose it, or introduce fallback.

## Blocker evidence

The mandatory gate ran from exact clean [planning](../../../../glossary.md#term-planning) revision
`c5e5472600ede66673d589902e94dc3eb66f4678`. One disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) source compiled against
the active macOS 27.0 [SDK](../../../../glossary.md#term-sdk), then its executable ran exactly once on the M3 Max. It created one [Metal](../../../../glossary.md#term-metal)
[device](../../../../glossary.md#term-device)/queue, one `[4,8]` [FLOAT32](../../../../glossary.md#term-float32) [graph](../../../../glossary.md#graph) with one direct [axis](../../../../glossary.md#term-axis)-one
`softMaxWithTensor:axis:name:` [target](../../../../glossary.md#term-target), one [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) at the production default optimization
level with `MPSGraphReducedPrecisionFastMathNone` set/read and `MPSGraphOptionsNone`, and one
[synchronous](../../../../glossary.md#term-synchronous) execution.

The [run](../../../../glossary.md#term-run) stopped at the first failing output, flattened index 18: the third [slice](../../../../glossary.md#slice)'s input `-90.0f`
with slice maximum `0.0f`. [MPSGraph](../../../../glossary.md#term-mpsgraph) returned [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) `0x00000000`; the frozen
[binary64](../../../../glossary.md#floating-point-precision-and-rounding) max-shift/compensated [oracle](../../../../glossary.md#term-test-oracle) narrowed to positive FLOAT32 [subnormal](../../../../glossary.md#term-subnormal) bits `0x0008ec28`.
Their ordered FLOAT32 distance is 584,744 ULPs, exceeding the four-[ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) limit. The first 18 cells
had passed before this failure. A [host](../../../../glossary.md#term-host)-only Java check using exact `StrictMath.exp`, the specified
compensated order, and one FLOAT32 [narrowing](../../../../glossary.md#term-narrowing) independently confirmed expected bits `0x0008ec28`;
it performed no Metal compilation or execution. Because the numerical cell failed, the full
classification/input-preservation [scan](../../../../glossary.md#term-cumulative-scan) did not complete and makes no success claim.

The disposable Objective-C source/executable and host-only Java oracle source/class were removed
immediately. No second device run, [Shape](../../../../glossary.md#shape)/width/axis/profile/optimization/context/repetition [matrix](../../../../glossary.md#term-matrix-meanings),
production edit, test edit, or retained probe [artifact](../../../../glossary.md#term-artifact) exists. Task 0026 is therefore `Blocked`.
Unblocking requires a separately planned exact replacement route, such as a proven custom [kernel](../../../../glossary.md#kernel),
or a preceding [Model](../../../../glossary.md#term-model)/architecture numerical-contract decision. This task may not be retried by
narrowing to width eight, adding an arbitrary width cap, weakening four ULPs, admitting [SOFTMAX](../../../../glossary.md#softmax--log-softmax)
only under [`ACCELERATOR`](../../../../glossary.md#term-accelerator), decomposing it, or adding fallback.

## [Compiler](../../../../glossary.md#term-compiler)-owned [gradient](../../../../glossary.md#term-gradient) closure

Compiler production remains unchanged. `ReductionGradientRules` already expands first-order
[SOFTMAX](../../../../glossary.md#softmax--log-softmax) input contribution from the exact forward output `y` and incoming gradient `g` as:

```text
y * (g - sum(g * y, axis, keepDimensions=true))
```

Under [`ACCELERATOR`](../../../../glossary.md#term-accelerator), current [Metal](../../../../glossary.md#term-metal) capability already owns the required [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) `MUL`,
`SUB`, and keep-[dimensions](../../../../glossary.md#dimension) `SUM` occurrences, including their existing [operation](../../../../glossary.md#operation)-local profile
semantics. After adding profile-common SOFTMAX forward, an explicitly seeded first-order [graph](../../../../glossary.md#graph) in
the admitted canonical domain must therefore form one maximal Metal [MPSGraph](../../../../glossary.md#term-mpsgraph) [partition](../../../../glossary.md#partition) and execute
without [CPU](../../../../glossary.md#term-cpu) [ownership](../../../../glossary.md#term-ownership-meanings) or transfer.

This is composition of the compiler-generated graph, not a direct softmax-gradient [backend](../../../../glossary.md#backend)
semantic. Although the [SDK](../../../../glossary.md#term-sdk) declares `softMaxGradientWithIncomingGradient:sourceTensor:axis:name:`,
Task 0026 must not call it, add a gradient wire, recognize/fuse the formula, save hidden forward
state, or change Compiler. Strict Metal does not own [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic or SUM, so Task 0026 claims
strict SOFTMAX forward only, not a strict all-Metal [backward graph](../../../../glossary.md#backward-graph). Accelerator gradient results are
governed [node](../../../../glossary.md#node)-by-node by existing SOFTMAX, binary, and reduction contracts; no new [fused](../../../../glossary.md#specialization-and-fusion)-gradient
accuracy promise is introduced.

## Native [schema](../../../../glossary.md#term-schema) and [identity](../../../../glossary.md#term-identity-meanings) cutover

- Retain native ABI version 4, every C signature/status, the fixed 160-byte node record, and exactly
  thirteen exports.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [signature glossary](../../../../glossary.md#term-signature))</span>
- Evolve [node](../../../../glossary.md#node) schema `11 -> 12`; retain [operation](../../../../glossary.md#operation) wires `1..19` and append only `SOFTMAX=20`.
- Reuse typed `AXIS=3`. Encode one semantic [axis](../../../../glossary.md#term-axis) field with `attributeCount=1`, absent second-input
  sentinel, zero auxiliary, and every attribute-value/padding cell zero. Add no attribute kind.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- Define Java `NodeKind.SOFTMAX(20, 1, AXIS, CANONICAL, false)` and lower through the existing
  axis-node constructor. Java and native schema/preflight independently authenticate kind,
  profile, FLOAT32 types, canonical states, one-input topology, equal positive Shapes, rank, axis,
  and all zero/sentinel cells before graph construction.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [FLOAT32 glossary](../../../../glossary.md#term-float32); [canonical glossary](../../../../glossary.md#term-canonical-form); [topology glossary](../../../../glossary.md#term-topology); [Shapes glossary](../../../../glossary.md#shape); [rank glossary](../../../../glossary.md#term-rank); [graph glossary](../../../../glossary.md#graph))</span>
- Native [lowering](../../../../glossary.md#lowering) calls only
  `[graph softMaxWithTensor:first axis:(NSInteger)node.axis name:nil]` and verifies the exact output
  Shape before compilation.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Reject unknown wire `21+`, stale schema 11, wrong discriminator/count/sentinel/padding, profile,
  type/state/topology/Shape/rank/axis, and overflow before resource creation.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- Advance workload signature, exact default policy, candidate schema, compatibility schema,
  route-policy, and tuning-codec identities together from version 12 to 13. No alias or migration
  reader remains. Identity must distinguish normalized SOFTMAX axis in addition to existing
  profile, schema, topology, descriptors/states, feeds/targets, ABI, and splat facts.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [alias glossary](../../../../glossary.md#aliasing-and-arena); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [descriptors glossary](../../../../glossary.md#term-descriptor); [targets glossary](../../../../glossary.md#term-target))</span>
- Keep the existing whole-partition MPSGraph candidate, direct target publication, binding order,
  leases, rollback, reuse, concurrency, close/quiescence, cleanup, and session isolation. Add no
  workspace, custom kernel, late route, retry, host result, or CPU fallback.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [quiescence glossary](../../../../glossary.md#term-quiescence); [kernel glossary](../../../../glossary.md#kernel); [host glossary](../../../../glossary.md#term-host); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Exact production and proof touchpoints

Expected [Metal](../../../../glossary.md#term-metal)-private production [scope](../../../../glossary.md#term-scope):

- `MetalCapabilityProvider.java`;
- `MetalMpsGraphProgram.java`;
- `MetalNativeApi.java`;
- `MetalNegPartitionPreparer.java` — handle [SOFTMAX](../../../../glossary.md#softmax--log-softmax) before its trailing generic [axis](../../../../glossary.md#term-axis)-transform cast;
- `MetalNegRouteCandidateGenerator.java`, `MetalNegTuningBatch.java`, and
  `MetalNegTuningCodec.java` for the version-13 cutover;
- Metal `package-info.java`; and
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`.

[Compiler](../../../../glossary.md#term-compiler), [Model](../../../../glossary.md#term-model), Config, [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Backend](../../../../glossary.md#backend) Contract, [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), Training, [CPU](../../../../glossary.md#term-cpu),
and transfer production are read-only.

[Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) permanent proof:

- extend `MetalCapabilityProviderTest` with one common-profile valid occurrence, one positive
  selected-width-17 occurrence proving there is no selector-expansion cap, and independent exact
  exclusions for LOG_SOFTMAX/kind, type, rank/axis/Shape/layout/gradient state;<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- extend `MetalMpsGraphRawAbiNativeTest` for schema 12, wire 20, AXIS 3, exact 160-byte cells,
  malformed/stale rejection, and native/Java parity;
- add one focused `MetalMpsGraphSoftmaxNativeTest` as the numerical owner for the frozen finite
  corpus, four-ULP oracle, <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [finite glossary](../../../../glossary.md#term-finite); [ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>`[0,1]` classification, output freshness, and input preservation;
- update current [schema](../../../../glossary.md#term-schema) assertions without duplicating generic [lifecycle](../../../../glossary.md#lifecycle) [matrices](../../../../glossary.md#term-matrix-meanings);
- extend `MetalNegRouteCandidateGeneratorTest` for axis-separated identity and version-12
  rejection;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- extend one backend-conformance maximal-region test with the accelerator generated-gradient
  formula; and<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- extend one CPU-free public Engine integration method for both-profile forward plus one
  ACCELERATOR explicitly seeded first-order gradient. Assert numerical outputs/gradient and Metal
  ownership; do not add a direct gradient route test or claim strict backward closure.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Documentation at implementation

After executable behavior stabilizes, synchronize only current claims in `ARCHITECTURE.md`, [Tensor](../../../../glossary.md#tensor)
and [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) references, [backend](../../../../glossary.md#backend)-execution and [module](../../../../glossary.md#term-module)-boundary contracts, [Metal](../../../../glossary.md#term-metal) backend and
[partition](../../../../glossary.md#partition)-preparer guides, glossary, Metal native README, Metal package documentation, this task,
Metal master plan, and roadmap. Do not edit [Model](../../../../glossary.md#term-model)'s semantic plan to present the Metal-private
[finite](../../../../glossary.md#term-finite) policy as public semantics.

## Explicit exclusions

- `SoftmaxKind.LOG_SOFTMAX`: the active SDK exposes no direct log-softmax selector; SOFTMAX followed
  by LOG is not the first-class stable log-probability contract, and its generated gradient needs
  currently unsupported EXP.<span data-glossary-navigation="true"> ([SDK glossary](../../../../glossary.md#term-sdk); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [LOG glossary](../../../../glossary.md#term-log); [gradient glossary](../../../../glossary.md#term-gradient); [EXP glossary](../../../../glossary.md#term-expx))</span>
- Layer normalization and RMS normalization: their exact trailing normalized Shape, epsilon,
  population/mean-square accumulation, constant-slice, NaN/infinity, signed-zero, and overflow
  contracts are not established by selector headers; generated gradients also require unsupported
  scalar arithmetic and RSQRT closure.<span data-glossary-navigation="true"> ([Layer normalization glossary](../../../../glossary.md#layer-normalization); [RMS normalization glossary](../../../../glossary.md#rms-normalization); [normalized Shape glossary](../../../../glossary.md#normalized-shape); [epsilon glossary](../../../../glossary.md#epsilon); [mean glossary](../../../../glossary.md#term-mean); [constant glossary](../../../../glossary.md#term-constant); [slice glossary](../../../../glossary.md#slice); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Batch normalization: inference is a wider five-input arbitrary-channel slice and training has
  five outputs, while current Metal preparation admits one output per node.<span data-glossary-navigation="true"> ([Batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [inference glossary](../../../../glossary.md#term-inference); [channel glossary](../../../../glossary.md#term-channel); [Metal glossary](../../../../glossary.md#term-metal); [node glossary](../../../../glossary.md#node))</span>
- BFLOAT16, FLOAT64, FLOAT16, integral/BOOL, dynamic, scalar, zero-extent, rank-above-16, affine-view,
  noncanonical, aliased, or cross-type forms.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [extent glossary](../../../../glossary.md#term-extent); [rank glossary](../../../../glossary.md#term-rank); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Masks, scales, temperature, bias, causal forms, losses, attention, sparse/adaptive variants,
  decomposed-softmax recognition, fusion, direct gradient selector, hidden saved state, or training
  API.<span data-glossary-navigation="true"> ([Masks glossary](../../../../glossary.md#term-boolean-mask); [bias glossary](../../../../glossary.md#term-bias); [losses glossary](../../../../glossary.md#loss); [fusion glossary](../../../../glossary.md#term-fusion); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Model/Compiler/shared semantic changes, accelerator normalization relaxation, DAZ/FTZ permission,
  special-value promise, custom kernel, workspace, transfer widening, retry, fallback, or
  performance claim.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [accelerator glossary](../../../../glossary.md#term-accelerator); [normalization glossary](../../../../glossary.md#term-normalization); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [kernel glossary](../../../../glossary.md#kernel); [widening glossary](../../../../glossary.md#term-widening))</span>

## Acceptance and review

- The one mandatory probe either passes the exact finite gate before production or blocks the task
  without production changes.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
- Capability and execution admit exactly the canonical FLOAT32 SOFTMAX domain under both profiles,
  with no selected-axis upper bound beyond positive checked geometry and the independent rank-16
  ABI limit.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Direct finite forward output satisfies the frozen four-ULP Metal-private oracle, remains finite
  and in <span data-glossary-navigation="true"> ([ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp); [Metal glossary](../../../../glossary.md#term-metal); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>`[0,1]`, preserves input bits, and publishes fresh [canonical](../../../../glossary.md#term-canonical-form) storage.
- ACCELERATOR explicit seeded backward closes as one Metal partition through the unchanged Compiler
  formula and current MUL/SUB/SUM routes. No backend gradient semantic or strict backward claim is
  added.<span data-glossary-navigation="true"> ([ACCELERATOR glossary](../../../../glossary.md#term-accelerator); [partition glossary](../../../../glossary.md#partition); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Schema 12/wire 20/AXIS 3 and version-13 identities fail closed while ABI 4, 160-byte records,
  thirteen exports, ingress/publication, FLOAT32 transfer, and lifecycle remain exact.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- LOG_SOFTMAX and every general [normalization](../../../../glossary.md#term-normalization) remain closed.
- Focused proof, native export/link audit, architecture checks, Javadoc, one final build,
  synchronized current documentation, clean diff, and independent lean Class C <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [lean glossary](../../../../glossary.md#proof-assistant-and-model-checking-tools))</span>`APPROVE` with zero
  findings close the task.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Model softmax semantics](../../../modules/model/tasks/0016i-softmax-semantic-kinds-and-attributes.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- [Model softmax Tensor expressions](../../../modules/model/tasks/0016j-softmax-tensor-expressions.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compiler softmax inference and gradients](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- [CPU stable softmax precedent](../../../backends/cpu/tasks/0007e-portable-stable-softmax-and-log-softmax-coverage.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture impact

Expected impact: None. This task extends only the existing [backend](../../../../glossary.md#backend)-private typed [MPSGraph](../../../../glossary.md#term-mpsgraph) program,
capability, native [lowering](../../../../glossary.md#lowering), [identities](../../../../glossary.md#term-identity-meanings), and tests. Stop and replan if implementation requires a
new public/shared semantic, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module), resource kind, profile relaxation, or fallback.
