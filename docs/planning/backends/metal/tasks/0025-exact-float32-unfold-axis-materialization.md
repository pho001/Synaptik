# Task 0025: Exact [FLOAT32](../../../../glossary.md#term-float32) UNFOLD_AXIS [Materialization](../../../../glossary.md#materialization)

## Status

Complete

Implementation began from exact clean [planning](../../../../glossary.md#term-planning) revision
`c54ccbde0dc8defd4def1e25c3fd2898a1d66f4e` and landed at
`44edd86092509348e1e72cf7f0f4c3b13d141fa8`. The first independent review's selector-cap and
capability-proof findings were remediated at `d0a947fa953bddb2714c1917c7114ac345c530b7`; the second
wrong-kind proof finding was remediated at `f88066e3ad0547987bb03b2d18ed2813f97de223`. Final
independent [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) Class C review of `f88066e3ad0547987bb03b2d18ed2813f97de223` returned `APPROVE`
with zero findings.

## Change class

Class C — this adds a materializing, potentially overlapping [window transform](../../../../glossary.md#window-transform) to both profile
[matrices](../../../../glossary.md#term-matrix-meanings), evolves the private native [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) and every route [identity](../../../../glossary.md#term-identity-meanings), and exposes it through
public [Engine](../../../../glossary.md#term-engine) execution. One smallest one-[run](../../../../glossary.md#term-run) real-[device](../../../../glossary.md#term-device) [smoke](../../../../glossary.md#term-smoke-test) is permitted for the concrete
undocumented raw-bit movement risk. Selector, [Shape](../../../../glossary.md#shape), [rank](../../../../glossary.md#term-rank), [axis](../../../../glossary.md#term-axis), size, step, profile, optimization,
context, and repetition matrices are prohibited.

## Coordination [metadata](../../../../glossary.md#term-metadata)

- Depends on: Metal 0024 Complete; Model 0017M–0017N, 0018R, and 0023D; Compiler 0005C; CPU
  0006A1; Config 0006; Engine 0018; current completed shared lifecycle and opaque-backend
  contracts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend))</span>
- Conflicts with: any Metal capability, schema, native-preflight, candidate/codec,
  ingress/publication, materialization, or public Engine scope; any blocked Metal 0016–0018
  restart; two- or three-dimensional window, fold, pad, convolution, or pooling work; Model 0026;
  or shared-layer mutation.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- Parallel group: None.
- Common base revision: `c54ccbde0dc8defd4def1e25c3fd2898a1d66f4e`.
- Integration order: one optionality-free one-run movement smoke before production, schema/identity,
  capability/analysis, documented native selector composition, focused public proof,
  documentation, independent review.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [smoke glossary](../../../../glossary.md#term-smoke-test); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Integration validation: focused Metal/conformance/real-Engine checkpoint, native export audit,
  architecture checks, Javadoc, one final build, Markdown/diff validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): [Metal](../../../../glossary.md#term-metal) 0025 implementer.

## Goal

Add only [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) [`FLOAT32`](../../../../glossary.md#term-float32) `WindowTransformKind.UNFOLD_AXIS` with exact
`UnfoldAxisAttrs(axis, size, step)`, identically under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator).

The [operation](../../../../glossary.md#operation) materializes general-[axis](../../../../glossary.md#term-axis) windows. It copies source representations exactly, may
repeat one source cell in several output positions when windows overlap, never [aliases](../../../../glossary.md#aliasing-and-arena) or mutates
the source, performs no arithmetic on values, and has no [padding](../../../../glossary.md#padding) or [dilation](../../../../glossary.md#term-dilation) meaning.

## Exact capability and [Shape](../../../../glossary.md#shape) domain

Admit one [occurrence](../../../../glossary.md#term-occurrence) exactly when all of the following hold:

- kind and attributes are exactly `UNFOLD_AXIS + UnfoldAxisAttrs`; there is one ordered input and
  one output;
- input and output are both [`FLOAT32`](../../../../glossary.md#term-float32), have equal `requiresGrad`, and have fully static Shapes with
  strictly positive extents;
- input [rank](../../../../glossary.md#term-rank) `R` is `1..15`, output rank is `R + 1` and at most 16, and [normalized axis](../../../../glossary.md#normalized-axis) `a` is in
  `[0,R)`;
- input and output are canonical dense-contiguous, zero-offset, non-view values; the output is a
  fresh materialized value;<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- window size `S` is in `1..16`, a deliberate fixed selector-expansion bound, step `T` is positive,
  and `S <= D` for selected input [extent](../../../../glossary.md#term-extent) `D`;
- every subtract, multiply, add, element-count, byte-count, selector start/end, and native integer
  conversion is checked before native resource creation; and
- position count and output Shape are exactly

```text
P = floor((D - S) / T) + 1
output = input[0:a] + [P] + input[a+1:R] + [S]
```

For output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `y[0..R]`, the exact source coordinate is:

```text
x[i] = y[i]                 when i != a
x[a] = y[a] * T + y[R]
```

The `S <= 16` bound limits native expansion to at most 16 slices, 16 [singleton](../../../../glossary.md#term-singleton)-[axis](../../../../glossary.md#term-axis) insertions, and
one concatenation per semantic [node](../../../../glossary.md#node). It is capability [scope](../../../../glossary.md#term-scope), not a [Model](../../../../glossary.md#term-model) limit, and must be present
in capability, Java [preflight](../../../../glossary.md#term-preflight), native preflight, workload [identity](../../../../glossary.md#term-identity-meanings), and tests. Do not silently widen
it or conflate it with the independent rank-16 bound.

A valid [canonical](../../../../glossary.md#term-canonical-form) result may feed or be fed by any currently admitted same-profile [operation](../../../../glossary.md#operation) whose
exact canonical state and type requirements hold. Caller and exact splat input retain current
behavior. Direct [publication](../../../../glossary.md#term-publication) uses the existing canonical FLOAT32 [target](../../../../glossary.md#term-target) path. [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) transfer
remains canonical positive-rank FLOAT32 only and needs no [widening](../../../../glossary.md#term-widening).

Reject wrong attributes, [scalar](../../../../glossary.md#term-scalar) input, size zero or above 16, non-positive step, size above the
selected extent, rank [overflow](../../../../glossary.md#term-overflow), dynamic or zero extents, wrong result Shape, wrong type, mismatched
`requiresGrad`, affine [views](../../../../glossary.md#term-view-meanings), noncanonical [layout](../../../../glossary.md#layout), byte/[element](../../../../glossary.md#term-element) overflow, malformed [topology](../../../../glossary.md#term-topology), and
every unlisted window kind before native creation.

## Order, overlap, tails, and [padding](../../../../glossary.md#padding)

Output order is [row-major](../../../../glossary.md#term-row-major) in the exact [Shape](../../../../glossary.md#shape) above. The appended final [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) is window offset
`k=0..S-1`; the selected-[axis](../../../../glossary.md#term-axis) output coordinate is window position `p=0..P-1`. Therefore source
coordinate `p*T+k` is selected in increasing `k` within each output position.

When `T < S`, windows overlap. The same source representation is copied independently to every
corresponding output coordinate; no reduction, accumulation, deduplication, [alias](../../../../glossary.md#aliasing-and-arena), or ordering-
dependent arithmetic exists. When `T > S`, gaps are not represented. Any suffix that cannot form a
complete window is omitted by the floor formula. There is no padding, [dilation](../../../../glossary.md#term-dilation), wrap, clamp,
implicit zero, image [layout](../../../../glossary.md#layout), or ceil mode.

Every output bit must equal the addressed input bit, including both zero signs, [subnormals](../../../../glossary.md#term-subnormal),
[infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and quiet or [signaling NaN](../../../../glossary.md#term-signaling-nan) payloads. The input remains unchanged.

## Exact documented selector mapping

The active Command Line Tools macOS 27.0 [SDK](../../../../glossary.md#term-sdk) is authoritative. Its
`MPSGraphTensorShapeOps.h` declares the category from macOS 11.0 and documents:

- `sliceTensor:starts:ends:strides:name:` as TensorFlow-style [half-open](../../../../glossary.md#term-half-open-interval) strided slicing;
- `expandDimsOfTensor:axis:name:` as inserting one size-one dimension, available from macOS 12.3;
  and
- `concatTensors:dimension:name:` as concatenating its ordered [tensor](../../../../glossary.md#tensor) list along one [dimension](../../../../glossary.md#dimension).

The existing [backend](../../../../glossary.md#backend) already relies on the expand-dimensions selector, so this task introduces no
new deployment-floor claim. Lower one semantic [node](../../../../glossary.md#node) as follows, with all arrays and arithmetic
created only during [shape](../../../../glossary.md#shape)-specialized native [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names):

1. For each `k` in ascending `0..S-1`, create [rank](../../../../glossary.md#term-rank)-`R` `starts`, `ends`, and [`strides`](../../../../glossary.md#term-stride) arrays.
2. On every `i != a`, use `start=0`, `end=inputExtent[i]`, and `stride=1`.
3. On [axis](../../../../glossary.md#term-axis) `a`, use `start=k`, `stride=T`, and
   `end=k + (P - 1) * T + 1`. The documented half-open rule yields exactly `P` values
   `k, k+T, ..., k+(P-1)T`.
4. Append a [singleton](../../../../glossary.md#term-singleton) final dimension to each [slice](../../../../glossary.md#slice) with
   `expandDimsOfTensor:axis:R:name:`.
5. Concatenate the tensors in ascending `k` order with
   `concatTensors:dimension:R:name:`. The result Shape and coordinate mapping are exactly the Model
   contract above.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

No undocumented selector behavior is used for type, Shape, mapping, overlap, tail, or [padding](../../../../glossary.md#padding).
Native creation must independently rederive `P`, authenticate the exact input/output Shapes and
[canonical](../../../../glossary.md#term-canonical-form) value states, prove every selector integer representable, and reject a malformed record
before [graph](../../../../glossary.md#graph) compilation.

## Minimal [device](../../../../glossary.md#term-device) evidence policy

The [SDK](../../../../glossary.md#term-sdk) documents the structural selector mapping but does not promise preservation of every
[`FLOAT32`](../../../../glossary.md#term-float32) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) bit through the composed [slice](../../../../glossary.md#slice)/expand/concat path. That is the only concrete
undocumented behavior risk.

Before production edits, [run](../../../../glossary.md#term-run) at most one disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) program, once, using one context,
one [graph](../../../../glossary.md#graph), one executable, optimization level zero, and one execution. Use [canonical](../../../../glossary.md#term-canonical-form) [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
input `[2,6]`, `axis=1`, `size=3`, and `step=2`, producing `[2,2,3]`. This single geometry exercises
an unaffected leading [axis](../../../../glossary.md#term-axis), overlap at source [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) two, and an omitted incomplete tail.
Include signed zeros, a [subnormal](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and quiet/[signaling NaN](../../../../glossary.md#term-signaling-nan) payloads; compare output [raw
bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) against the formula, verify the input unchanged, and remove source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) immediately.

Do not probe size/[rank](../../../../glossary.md#term-rank)/axis/step/profile/context/optimization/repetition combinations, invalid
geometry, [padding](../../../../glossary.md#padding), folds, im-to-column selectors, or performance. A failed raw-bit cell blocks the
task without production changes; a pass authorizes only this exact documented composition.

## Native [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle), and [cache](../../../../glossary.md#term-cache)

- Retain native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, every status/[signature](../../../../glossary.md#term-signature), and exactly thirteen exports.
- Evolve fixed 160-byte [node](../../../../glossary.md#node) schema `10 -> 11`; retain [operation](../../../../glossary.md#operation) wires `1..18` and append only
  `UNFOLD_AXIS=19`.
- Append typed attribute discriminator `WINDOW_AXIS=6`. Encode exactly three semantic fields:
  normalized axis in the existing axis cell, positive size in <span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [axis glossary](../../../../glossary.md#term-axis))</span>`attributeValues[0]`, and positive
  step in `attributeValues[1]`. Set `attributeCount=3`, auxiliary to zero, and every remaining
  attribute/padding cell to zero. Native and Java validators must treat this as one closed typed
  form rather than generic integers.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- Reject unknown wire `20+`, attribute `7+`, stale schema, nonzero sentinel, size above 16, wrong
  value state/type/rank/Shape, and any malformed attribute topology before creating a graph
  resource.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [topology glossary](../../../../glossary.md#term-topology); [graph glossary](../../../../glossary.md#graph))</span>
- Advance workload, exact-policy, candidate, compatibility, route-policy, and codec identity
  versions together from 11 to 12. Identity covers profile, schema, operation, axis/size/step,
  ordered topology, descriptors/states, selector-expansion bound, feeds/targets, ABI, and splats.
  Version-eleven or corrupt decisions reject or miss safely; no migration reader or alias remains.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [descriptors glossary](../../../../glossary.md#term-descriptor); [targets glossary](../../../../glossary.md#term-target); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Keep the existing whole-[partition](../../../../glossary.md#partition) [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) candidate, direct targets, exact binding order,
  leases, rollback, reuse, concurrency, close/quiescence, cleanup, context/session isolation, and
  candidate authentication. No workspace, custom kernel, late route, retry, host result, or CPU
  fallback is added.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [quiescence glossary](../../../../glossary.md#term-quiescence); [authentication glossary](../../../../glossary.md#term-authentication); [kernel glossary](../../../../glossary.md#kernel); [host glossary](../../../../glossary.md#term-host); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## [Compiler](../../../../glossary.md#term-compiler) and backward boundary

Compiler production remains unchanged. Current `LayoutInference` owns the exact [Shape](../../../../glossary.md#shape) formula, and
`LayoutGradientRules` defines the floating input [cotangent](../../../../glossary.md#cotangent) as
`gradient.foldAxis(axis, originalInputExtent, step)`.

[Metal](../../../../glossary.md#term-metal) 0025 adds forward `UNFOLD_AXIS` only. It does not admit `FOLD_AXIS`, whose semantics start from
represented [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) and add overlapping contributions, nor does it claim that [MPSGraph](../../../../glossary.md#term-mpsgraph)
floating accumulation satisfies either numerical profile. A forward [occurrence](../../../../glossary.md#term-occurrence) may retain
`requiresGrad=true`, but a compiled [backward graph](../../../../glossary.md#backward-graph) containing `FOLD_AXIS` must fail [owner](../../../../glossary.md#term-owner)
selection/preparation before Metal native creation unless another explicit owner handles it. Do not
add a [decomposition](../../../../glossary.md#term-decomposition), cross-owner transfer, seeded-[gradient](../../../../glossary.md#term-gradient) success claim, training path, or
compiler special case.

## Deferred window workstream findings

The same active [SDK](../../../../glossary.md#term-sdk) exposes `MPSGraphImToColOps.h` selectors from macOS 14.0, but current
documentation is insufficient for an exact Synaptik task:

- `imToColWithSourceTensor:descriptor:name:` documents rank-four source plus kernel, stride,
  dilation, asymmetric padding, and layout, but not its output rank/axis order, exact padding sample
  representation, terminal ceil-grid behavior, or a configurable padding value. Model<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [layout glossary](../../../../glossary.md#layout); [axis glossary](../../../../glossary.md#term-axis); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
  `UNFOLD2D` requires [canonical](../../../../glossary.md#term-canonical-form) [rank](../../../../glossary.md#term-rank)-three `[N,C*kH*kW,outH*outW]`, exact channel/kernel/spatial
  order, explicit floor/ceil behavior, and either represented positive zero or an arbitrary exact
  typed padding scalar.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [spatial glossary](../../../../glossary.md#term-spatial-axes); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- `colToImWithSourceTensor:outputShape:descriptor:name:` documents a rank-four source, while [Model](../../../../glossary.md#term-model)
  `FOLD2D` consumes rank-three canonical columns. It does not document overlap contribution order,
  represented FLOAT32 accumulation, padding exclusion, or terminal ceil-tail exclusion.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- The active SDK provides no corresponding documented three-dimensional im-to-column/column-to-
  image selector for current `UNFOLD3D`/`FOLD3D` semantics.
- General `FOLD_AXIS` needs exact overlap addition and represented order. Neither the structural
  slice composition above nor current SDK documentation supplies that numerical contract.

Therefore `FOLD_AXIS`, both `UNFOLD2D` variants, `FOLD2D`, `UNFOLD3D`, and `FOLD3D` remain closed.
After 0025, the next safe window [slice](../../../../glossary.md#slice) is a documentation/evidence decision for direct positive-zero
`UNFOLD2D`; it must first establish exact output mapping, floor/ceil tail [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters), and positive-zero
[padding](../../../../glossary.md#padding) without claiming typed padding or folds. No implementation task for that slice is [Ready](../../../../glossary.md#term-ready)
now.

## Non-goals

No `FOLD_AXIS`; two- or three-dimensional unfold/fold; [`PAD`](../../../../glossary.md#padding); [convolution](../../../../glossary.md#term-channels-first-convolution-layers) or [pooling](../../../../glossary.md#term-pooling); overlap
addition; [FLOAT64](../../../../glossary.md#term-float64)/[BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16)/integral/[BOOL](../../../../glossary.md#data-type--datatype) window; [scalar](../../../../glossary.md#term-scalar)/dynamic/zero-[extent](../../../../glossary.md#term-extent)/[view](../../../../glossary.md#term-view-meanings) form;
window size above 16; [padding](../../../../glossary.md#padding)/[dilation](../../../../glossary.md#term-dilation)/ceil mode; asymmetric window; output [alias](../../../../glossary.md#aliasing-and-arena); transfer [widening](../../../../glossary.md#term-widening);
custom [kernel](../../../../glossary.md#kernel); [host](../../../../glossary.md#term-host) result; [compiler](../../../../glossary.md#term-compiler)/shared production change; backward/training claim; new export,
[dependency](../../../../glossary.md#term-dependency), route, [cache](../../../../glossary.md#term-cache) migration, performance claim, or selector [matrix](../../../../glossary.md#term-matrix-meanings).

## Files and symbols

Expected production [scope](../../../../glossary.md#term-scope) is limited to existing [Metal](../../../../glossary.md#term-metal)-private files:

- `MetalCapabilityProvider`, `MetalNativeApi` — exact both-profile occurrence and Java ABI
  preflight;<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- `MetalMpsGraphProgram` — [schema](../../../../glossary.md#term-schema) 11, wire 19, typed `WINDOW_AXIS`, and 160-byte encoding;
- `MetalNegPartitionPreparer` — [canonical](../../../../glossary.md#term-canonical-form) state/[topology](../../../../glossary.md#term-topology) [validation](../../../../glossary.md#term-validation) and exact [node](../../../../glossary.md#node) [lowering](../../../../glossary.md#lowering);
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec` — version-twelve
  identity cutover;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — schema validation and documented
  slice/expand/concat construction; and<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- existing package documentation where the private program vocabulary is enumerated.

Reuse existing focused test classes. Add no window-specific test class when an existing capability,
raw-[ABI](../../../../glossary.md#term-application-binary-interface-abi), native program, [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution), candidate/[codec](../../../../glossary.md#term-codec), [conformance](../../../../glossary.md#term-conformance), or public [Engine](../../../../glossary.md#term-engine) method
can defend the observable contract. Shared [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Config,
[Trace](../../../../glossary.md#trace), Engine, Training, and transfer production remain read-only.

## [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) proof and [validation](../../../../glossary.md#term-validation)

Permanent proof must be contract-minimal:

- extend the existing capability method for one valid overlapping/tail occurrence under both
  profiles and grouped exclusions for wrong kind/attrs/type/rank/Shape/layout/size/profile state;<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [rank glossary](../../../../glossary.md#term-rank); [layout glossary](../../../../glossary.md#layout))</span>
- extend existing raw-[schema](../../../../glossary.md#term-schema) tests for wire 19, `WINDOW_AXIS`, exact 160-byte cells, and malformed
  or stale-schema rejection;
- extend one existing native program test with the same compact mapping geometry and exact raw-bit
  output/input-preservation assertion;
- update existing [identity](../../../../glossary.md#term-identity-meanings)/[codec](../../../../glossary.md#term-codec) tests for [axis](../../../../glossary.md#term-axis)/size/step separation and version-eleven rejection;
- extend one existing maximal-[partition](../../../../glossary.md#partition) [conformance](../../../../glossary.md#term-conformance) test so `UNFOLD_AXIS` composes with a currently
  supported canonical consumer; and<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- extend one existing [CPU](../../../../glossary.md#term-cpu)-free public [Engine](../../../../glossary.md#term-engine) integration method for exact [Shape](../../../../glossary.md#shape) and raw-bit output.

Use current generic [lifecycle](../../../../glossary.md#lifecycle), candidate, codec, malformed-record, [target](../../../../glossary.md#term-target) [publication](../../../../glossary.md#term-publication), reuse,
[concurrency](../../../../glossary.md#term-concurrency), close, [ABI](../../../../glossary.md#term-application-binary-interface-abi)/export, architecture, and documentation coverage. Do not duplicate those
[matrices](../../../../glossary.md#term-matrix-meanings) for one wire. [Run](../../../../glossary.md#term-run) focused commands while implementation stabilizes, then the full build
exactly once. Independent lean Class C review reuses successful evidence and runs no [device](../../../../glossary.md#term-device) work
unless it identifies one new concrete risk.

## Acceptance and review

- Only the exact bounded [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) `UNFOLD_AXIS` occurrence is advertised and executable
  identically under both profiles.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Shape and every output coordinate match the documented slice/expand/concat construction;
  overlapping source bits repeat exactly, incomplete tails disappear, no padding exists, and input
  storage is unchanged.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [slice glossary](../../../../glossary.md#slice); [padding glossary](../../../../glossary.md#padding))</span>
- Schema 11/wire 19/attribute 6 and version-twelve identities fail closed while ABI 4, thirteen
  exports, existing ingress/publication, and FLOAT32 transfer remain exact.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [publication glossary](../../../../glossary.md#term-publication))</span>
- [Compiler](../../../../glossary.md#term-compiler) production is unchanged; generated `FOLD_AXIS` backward remains unsupported by [Metal](../../../../glossary.md#term-metal).
- The one bounded smoke, focused reused proof, final validation, synchronized current docs, clean
  diff, and independent Class C <span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test); [validation glossary](../../../../glossary.md#term-validation))</span>`APPROVE` with zero findings close the task.

## Implementation [checkpoint](../../../../glossary.md#term-checkpoint) for independent review

Implementation began from exact clean [planning](../../../../glossary.md#term-planning) revision
`c54ccbde0dc8defd4def1e25c3fd2898a1d66f4e`, correcting the preplanning common-base [metadata](../../../../glossary.md#term-metadata)
`aa191ca469010d081150e97dcddd504ec626dd9e`. Before any production edit, one disposable
[Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) program compiled against the production frameworks and ran exactly once on the M3 Max
with one context, [graph](../../../../glossary.md#graph), executable at optimization level zero, and execution. Its [canonical](../../../../glossary.md#term-canonical-form)
`[2,6]`, [axis](../../../../glossary.md#term-axis)-one, size-three, step-two [slice](../../../../glossary.md#slice)/expand/concat [run](../../../../glossary.md#term-run) reported
`task0025 UNFOLD_AXIS smoke passed: exact overlap/tail raw bits and unchanged input`. The corpus
covered signed zeros, positive and negative [subnormals](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and quiet/[signaling NaN](../../../../glossary.md#term-signaling-nan)
payloads. The exact `[2,2,3]` output proved overlap and omitted-tail mapping, the input was
unchanged, and the source and executable were removed immediately. No [Shape](../../../../glossary.md#shape), [rank](../../../../glossary.md#term-rank), axis, size,
step, profile, context, optimization, repetition, invalid-geometry, [padding](../../../../glossary.md#padding), fold, or [matrix](../../../../glossary.md#term-matrix-meanings) probe
ran.

The implementation adds only bounded canonical [FLOAT32](../../../../glossary.md#term-float32) `UNFOLD_AXIS` to both profile matrices.
[Schema](../../../../glossary.md#term-schema) 11 retains the 160-byte [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4 record and wires `1..18`, appends `UNFOLD_AXIS=19` and
`WINDOW_AXIS=6`, and carries [normalized axis](../../../../glossary.md#normalized-axis), size, and step in the closed typed form. Java and
native [preflight](../../../../glossary.md#term-preflight) independently rederive the floor count and exact Shape, require canonical
rank-`1..15` input and output, size `1..16`, positive step, representable selector values, and
fresh canonical FLOAT32 output. Native [lowering](../../../../glossary.md#lowering) creates [half-open](../../../../glossary.md#term-half-open-interval) strided slices in ascending
window-offset order, appends one final [singleton](../../../../glossary.md#term-singleton) [dimension](../../../../glossary.md#dimension) to each, and concatenates along that
dimension. Workload, exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings)
are version twelve and reject version eleven. ABI 4, thirteen exports, existing
ingress/[publication](../../../../glossary.md#term-publication), canonical FLOAT32 transfer, shared production, [compiler](../../../../glossary.md#term-compiler) behavior, and the
unsupported fold/backward boundary are unchanged.

Worker evidence passed:

- `native/metal-macos-arm64/build.sh`, an exact thirteen-[symbol](../../../../glossary.md#term-symbol) `nm -gU` export audit, and
  Foundation, Metal, and MetalPerformanceShadersGraph linkage audit;<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- named focused capability, exact 160-byte schema/native-malformed, native raw-bit mapping/input-
  preservation, axis/size/step identity, version-eleven codec rejection, maximal-partition
  composition, and CPU-free both-profile public Engine tests;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [partition glossary](../../../../glossary.md#partition); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- `:testing:architecture-tests:test` and `:backends:metal:javadoc`; and
- the single final `./gradlew build` invocation: `BUILD SUCCESSFUL`, 87 actionable tasks, eight
  executed and 79 up-to-date.

The first independent review returned `BLOCK` with two remediation findings. Remediation began from
exact clean implementation checkpoint `44edd86092509348e1e72cf7f0f4c3b13d141fa8` and does not claim
review approval. Java now owns a distinct fixed
`MetalMpsGraphProgram.MAX_SELECTOR_EXPANSION=16` [constant](../../../../glossary.md#term-constant), native owns the matching fixed
`SYNAPTIK_MAX_SELECTOR_EXPANSION=16U` constant, and capability, Java schema/preflight, and native
preflight no longer reuse rank or an unnamed literal for the selector-expansion policy. Focused
capability and raw/native proofs use selected [extent](../../../../glossary.md#term-extent) 17 with exact size-17 output geometry, so size
17 fails solely at that cap. The capability proof now independently covers the typed-attribute and
[occurrence](../../../../glossary.md#term-occurrence)-[cardinality](../../../../glossary.md#term-cardinality) boundaries, both FLOAT32 positions, [scalar](../../../../glossary.md#term-scalar) and rank-16 input, exact Shape,
input and output [layout](../../../../glossary.md#layout), `requiresGrad`, size/selected-extent, step-derived position count, and both
profile states; layout and [gradient](../../../../glossary.md#term-gradient) mismatch cases do not [mask](../../../../glossary.md#term-boolean-mask) one another. Raw [validation](../../../../glossary.md#term-validation) also
covers zero size and zero step.

Remediation evidence passed without rerunning the already successful full build:

- `native/metal-macos-arm64/build.sh`; and
- the two named `MetalCapabilityProviderTest` and `MetalMpsGraphRawAbiNativeTest` remediation
  methods plus `:backends:metal:javadoc` in one focused [Gradle](../../../../glossary.md#term-gradle) invocation: `BUILD SUCCESSFUL`, 20
  actionable tasks, four executed and 16 up-to-date.

The second independent re-review returned `BLOCK` for one remaining wrong-kind proof gap.
Remediation began from exact clean checkpoint `d0a947fa953bddb2714c1917c7114ac345c530b7`
and again makes no approval claim. The existing capability method now constructs the valid
`FOLD_AXIS + FoldAxisAttrs(axis=1, outputSize=6, step=2)` occurrence from canonical FLOAT32
`[2,2,3]` input to canonical FLOAT32 `[2,6]` output with matching `requiresGrad`, then proves its
rejection under both profiles solely because the kind is not `UNFOLD_AXIS`. Its single focused test
invocation passed: `BUILD SUCCESSFUL`, 19 actionable tasks, two executed and 17 up-to-date. No
production code changed and the already-successful broader validation was not rerun.

Final independent [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) Class C re-review then inspected exact remediation revision
`f88066e3ad0547987bb03b2d18ed2813f97de223` and returned `APPROVE` with zero findings. This closes
Task 0025 without further code, test, [device](../../../../glossary.md#term-device), or documentation changes.

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

Task 0025 is Complete at implementation `44edd86092509348e1e72cf7f0f4c3b13d141fa8`, first
remediation `d0a947fa953bddb2714c1917c7114ac345c530b7`, and final remediation/reviewed revision
`f88066e3ad0547987bb03b2d18ed2813f97de223`; final independent lean Class C review returned
`APPROVE` with zero findings.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
- [Prepare and run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Architecture impact

Impact: None. The implementation extends only the existing [backend](../../../../glossary.md#backend)-private typed [MPSGraph](../../../../glossary.md#term-mpsgraph) program
and [canonical](../../../../glossary.md#term-canonical-form) [materialization](../../../../glossary.md#materialization) route. Shared [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Config,
[Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), Training, and transfer production remain unchanged.
