# Task 0024: Exact [INT32](../../../../glossary.md#data-type--datatype) SCATTER_ELEMENTS Replacement

## Status

Complete

Completed at implementation `a947e574732273bee4469d42afe8935082d53109` from exact clean base
`ad07e7e5d029f0472f1d05e82f2e869899d7dbf7`, followed by documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e`. Final independent [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) Class C review returned
`APPROVE` with zero findings after both [checkpoints](../../../../glossary.md#term-checkpoint). [Metal](../../../../glossary.md#term-metal) 0023 remains Complete at implementation
`9a7911c9447eeb5125ceb8a4d78e55349ec18b97` plus evidence
`80e6cedda96df2a03f284d551e6e34046eaaa2f3`. Metal 0025 is the sole [Ready](../../../../glossary.md#term-ready) successor.

## Change class

Class C — three-input functional update execution, duplicate-[target](../../../../glossary.md#term-target) [validation](../../../../glossary.md#term-validation), native [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache)
[identity](../../../../glossary.md#term-identity-meanings) evolution, and public [Engine](../../../../glossary.md#term-engine) execution cross the native boundary. One minimal real-[device](../../../../glossary.md#term-device)
[smoke](../../../../glossary.md#term-smoke-test) for the only undocumented raw-bit risk and one independent final review are mandatory. Broad
selector, [Shape](../../../../glossary.md#shape), type, profile, invalid-index, context, optimization, or repetition [matrices](../../../../glossary.md#term-matrix-meanings) are
prohibited.

## Coordination [metadata](../../../../glossary.md#term-metadata)

- Depends on: Metal 0023; Model 0018G–0018H, 0018O, and 0025C; Compiler 0005C; CPU 0006B1;
  Config 0006; Engine 0018; current completed shared lifecycle/opaque-backend contracts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [opaque glossary](../../../../glossary.md#term-opaque); [backend glossary](../../../../glossary.md#backend))</span>
- Conflicts with: any Metal capability, schema, native-preflight, candidate/codec,
  ingress/publication, materialization, or public Engine scope; any blocked Metal 0016–0018 restart;
  arithmetic scatter, Scatter-ND, Model 0026, or shared-layer mutation.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- Parallel group: None.
- Common base revision: `ad07e7e5d029f0472f1d05e82f2e869899d7dbf7`.
- Integration order: schema/identity, capability/analysis, native validation/lowering, prepared
  diagnostics, public proof, documentation, independent review.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering))</span>
- Integration validation: focused Metal/conformance/real-Engine checkpoint, native export audit,
  architecture checks, Javadoc, one final build, Markdown/diff validation.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): [Metal](../../../../glossary.md#term-metal) 0024 implementer.

## Goal

Add only [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) [`FLOAT32`](../../../../glossary.md#term-float32) `SCATTER_ELEMENTS` with canonical [`INT32`](../../../../glossary.md#data-type--datatype) indices and
`ScatterReduction.NONE`, identically under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator).

The [operation](../../../../glossary.md#operation) consumes exact ordered inputs `[data, indices, updates]`, returns one new canonical
data-shaped `FLOAT32` value, and never mutates an input. Each logical update [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) replaces
its selected [axis](../../../../glossary.md#term-axis) with the corresponding non-negative index and copies the exact update bits to
that unique [target](../../../../glossary.md#term-target). Unaddressed output coordinates preserve exact base-data bits. Invalid bounds or
overlapping targets fail before selector [dispatch](../../../../glossary.md#term-dispatch) or any target write.

## Verified contract and selector prerequisites

- [Model](../../../../glossary.md#term-model) 0018G/0018H define same-[rank](../../../../glossary.md#term-rank) functional `SCATTER_ELEMENTS`, normalized axis, explicit
  reduction, exact ordered roles, equal indices/update Shape, data-shaped fresh result, exact type,
  and <span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis))</span>`data.requiresGrad || updates.requiresGrad`; Model 0018O fixes its current [canonical](../../../../glossary.md#term-canonical-form) taxonomy.
- Model 0018I/0018J define separate tuple-index `SCATTER_ND`; Model 0023B defines separate
  Gather-compatible fixed-add <span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather))</span>`SCATTER_ADD`; neither is an alias or dependency of the selected
  operation.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [dependency glossary](../../../../glossary.md#term-dependency); [operation glossary](../../../../glossary.md#operation))</span>
- Model 0025C keeps `NONE` as replacement with unique-[target](../../../../glossary.md#term-target) validity and no arithmetic reduction.
- [Compiler](../../../../glossary.md#term-compiler) 0005C owns [fail-closed](../../../../glossary.md#term-fail-closed) scatter [preflight](../../../../glossary.md#term-preflight) and exact formulas. For
  `SCATTER_ELEMENTS/NONE`, the base cotangent is replacement scatter with zero updates, the update
  cotangent is <span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent))</span>`GATHER_ELEMENTS`, and indices are non-[differentiable](../../../../glossary.md#derivative-and-chain-rule).
- [CPU](../../../../glossary.md#term-cpu) 0006B1 is the diagnostic [oracle](../../../../glossary.md#term-test-oracle): full bounds [validation](../../../../glossary.md#term-validation) precedes `NONE` duplicate validation
  and every output write; update positions are row-major ordinals.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>
- Metal 0023 supplies exact canonical FLOAT32/INT32 caller and splat ingress, local FLOAT32
  publication, pre-dispatch index scanning, schema 9/wires through 17, and version-ten identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [publication glossary](../../../../glossary.md#term-publication); [dispatch glossary](../../../../glossary.md#term-dispatch); [schema glossary](../../../../glossary.md#term-schema))</span>
- Config 0006 and Engine 0018 carry the immutable profile. Shared Runtime, Prepare, Backend
  Contract, Trace, and Engine lifecycle ownership needs no change.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [carry glossary](../../../../glossary.md#term-carry); [immutable glossary](../../../../glossary.md#term-immutable); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

The active [SDK](../../../../glossary.md#term-sdk) `MPSGraphScatterNDOps.h` documents the macOS 12.3 selector:

```objc
- (MPSGraphTensor *) scatterAlongAxis:(NSInteger) axis
                       withDataTensor:(MPSGraphTensor *) dataTensor
                        updatesTensor:(MPSGraphTensor *) updatesTensor
                        indicesTensor:(MPSGraphTensor *) indicesTensor
                                 mode:(MPSGraphScatterMode) mode
                                 name:(NSString * _Nullable) name;
```

It documents matching updates/indices [Shapes](../../../../glossary.md#shape), data matching except at the [axis](../../../../glossary.md#term-axis), [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype)
indices, and `MPSGraphScatterModeSet`. That valid type/Shape/[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) mapping is authoritative and
needs no [matrix](../../../../glossary.md#term-matrix-meanings). The same header says out-of-bounds updates are skipped and defines no portable
replacement winner for overlaps; both behaviors are unreachable because [Metal](../../../../glossary.md#term-metal) prevalidates bounds
and uniqueness.

No semantic or architecture prerequisite remains open for this exact subset.

## Exact supported domain

Every admitted [occurrence](../../../../glossary.md#term-occurrence) has:

- kind `AxisScatterKind.SCATTER_ELEMENTS`, exact `ScatterElementsAttrs(axis, NONE)`, three ordered
  inputs and one output;
- `[data FLOAT32, indices INT32, updates FLOAT32] -> FLOAT32`;
- [canonical](../../../../glossary.md#term-canonical-form) dense, zero-offset, fully static [rank](../../../../glossary.md#term-rank) `1..16` values with positive extents;
- equal data/indices/updates rank, exact indices/update Shape, and data matching every non-axis
  extent; the positive selected update extent may differ from data;<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent))</span>
- result [Shape](../../../../glossary.md#shape) exactly data Shape and [normalized axis](../../../../glossary.md#normalized-axis) in `[0, rank)`;
- result [gradient](../../../../glossary.md#term-gradient) eligibility exactly the data/update OR and non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) indices;
- canonical inputs on consuming edges, with no implicit [materialization](../../../../glossary.md#materialization).

Data and updates may reuse the same read-only logical value when Shapes permit; existing feed
deduplication remains legal. Output is a distinct [graph value](../../../../glossary.md#graph-value)/destination; add no in-place or [alias](../../../../glossary.md#aliasing-and-arena)
optimization. Canonical local [FLOAT32](../../../../glossary.md#term-float32) [producers](../../../../glossary.md#term-producer) may feed data/updates, and the result may feed a
compatible canonical consumer or [target](../../../../glossary.md#term-target). Indices use current caller/splat [INT32](../../../../glossary.md#data-type--datatype) ingress.

Capability is identical under both profiles. The [normative](../../../../glossary.md#term-normative) profile table gives scatter/fold no
relaxation; [`ACCELERATOR`](../../../../glossary.md#term-accelerator) adds no result and remains a monotonic superset.

For logical update [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `u` and `i = indices[u]`:

```text
target(u)[d] = i when d == axis, otherwise u[d]
result[target(u)] = raw bits of updates[u]
result[x] = raw bits of data[x] when no update addresses x
```

Unique targets make update order irrelevant. [Row-major](../../../../glossary.md#term-row-major) order exists only for diagnostics. Preserve
[signed zero](../../../../glossary.md#term-signed-zero), [subnormals](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and [NaN](../../../../glossary.md#term-nan) sign/[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)/quiet-or-signaling bits as movement; inputs
remain byte-for-byte unchanged. Use only the data-taking selector with
`MPSGraphScatterModeSet`; do not synthesize replacement with arithmetic, [masks](../../../../glossary.md#term-boolean-mask), one-hot, reduction,
or a custom [kernel](../../../../glossary.md#kernel).

## Invalid-index and overlap contract

Negative indices and indices `>= data.shape[axis]` are invalid; never wrap, clamp, skip, or pad.
Two updates overlap when their complete derived [target](../../../../glossary.md#term-target) [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) match. Under `NONE`, that is
invalid. Equal index values at different non-[axis](../../../../glossary.md#term-axis) coordinates are not duplicates.

[Run](../../../../glossary.md#term-run) [validation](../../../../glossary.md#term-validation) order is exact:

1. current [ABI](../../../../glossary.md#term-application-binary-interface-abi) counts, handle arrays, resource sizes, and typed [bindings](../../../../glossary.md#term-binding);
2. retained indexing [descriptors](../../../../glossary.md#term-descriptor) in stable [node](../../../../glossary.md#node) order;
3. for each scatter, every index in [row-major](../../../../glossary.md#term-row-major) ordinal order for bounds;
4. only after that occurrence's complete bounds pass, uniqueness, selecting the smallest later
   duplicate ordinal and the smallest earlier ordinal for that target;<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
5. only after every descriptor passes, construct [tensor](../../../../glossary.md#tensor) data, [dispatch](../../../../glossary.md#term-dispatch), or expose a target write.

Exact [CPU](../../../../glossary.md#term-cpu)-parity messages are:

```text
SCATTER_ELEMENTS index at logical position <ordinal> for data axis <axis> is out of bounds: value=<value>, extent=<extent>
SCATTER_ELEMENTS duplicate target at logical update position <later>; first addressed at logical update position <first>
```

Native may return the existing index-domain status. Java rescans retained indices only on that
exceptional status in the same order; unmatched rejection is an internal failure. Every target and
input byte remains unchanged on failure. Valid-path uniqueness uses checked executable-owned
scratch or another bounded primitive representation, has no per-run [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) object [graph](../../../../glossary.md#graph), and
is not quadratic. Executable-owned scratch is rebuilt per invocation under the existing
synchronized resource run; allocation/size [overflow](../../../../glossary.md#term-overflow) rejects creation without fallback.

## Typed boundary, [schema](../../../../glossary.md#term-schema), and [identity](../../../../glossary.md#term-identity-meanings)

Reuse exact [FLOAT32](../../../../glossary.md#term-float32) data/update ingress, [INT32](../../../../glossary.md#data-type--datatype) index ingress, and local FLOAT32 [publication](../../../../glossary.md#term-publication). Native
typing authenticates `[FLOAT32, INT32, FLOAT32]`; conflicting value reuse fails. [CPU](../../../../glossary.md#term-cpu)↔[Metal](../../../../glossary.md#term-metal) transfer
remains [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) FLOAT32 only.

Advance [backend](../../../../glossary.md#backend)-local [node](../../../../glossary.md#node) schema `9 -> 10`:

- append [operation](../../../../glossary.md#operation) wire `SCATTER_ELEMENTS = 18` and reuse `AXIS = 3`;
- retain the 160-byte record; reinterpret offset 28 as typed operation-specific `auxiliary`:
  reduction keep-dimensions for reductions, third stable input for Scatter Elements, zero otherwise;<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [Scatter Elements glossary](../../../../glossary.md#scatter-elements))</span>
- rename Java/native internal `reserved` terminology atomically;
- encode data/indices/updates/result as first/second/auxiliary/output, axis count one, normalized
  axis, and no payload;<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- reject every wrong arity/state/type/rank/Shape/axis/reduction/unused field and every old/unknown
  schema or wire before graph compilation.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>

Native calls the documented selector once with Set mode. Do not use [shape](../../../../glossary.md#shape)-initializing/dynamic-[axis](../../../../glossary.md#term-axis)
or [Scatter-ND](../../../../glossary.md#scatter-nd) overloads, reflection, string [dispatch](../../../../glossary.md#term-dispatch), or fallback. [ABI](../../../../glossary.md#term-application-binary-interface-abi) remains `4` with exactly
thirteen exports and unchanged [signatures](../../../../glossary.md#term-signature)/[ownership](../../../../glossary.md#term-ownership-meanings).

Advance workload signature, exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec)
identities `10 -> 11` together. [Hash](../../../../glossary.md#term-hash) the exact three ordered edges, kind/reduction/axis,
types/Shapes/[layout](../../../../glossary.md#layout) states/output [metadata](../../../../glossary.md#term-metadata), schema 10, ABI 4, and profile/common policy. Old bytes
[fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation); add no shim, [alias](../../../../glossary.md#aliasing-and-arena), dual decoder, or migration.

## [Compiler](../../../../glossary.md#term-compiler) and execution boundary

Compiler production remains unchanged. Capability is [occurrence](../../../../glossary.md#term-occurrence)-based and phase-agnostic, so an
ordinary Compiler-generated `SCATTER_ELEMENTS/NONE` occurrence may use this row. That is not full
backward support: update cotangents require unsupported `GATHER_ELEMENTS`, and arithmetic scatter is
excluded. Make no seeded-[gradient](../../../../glossary.md#term-gradient), training, [optimizer](../../../../glossary.md#term-optimizer), tape, or all-[Metal](../../../../glossary.md#term-metal) backward claim.

This task also excludes SCATTER_ADD, [Scatter-ND](../../../../glossary.md#scatter-nd), every arithmetic reduction, [INT64](../../../../glossary.md#data-type--datatype), non-[FLOAT32](../../../../glossary.md#term-float32)
data, [BOOL](../../../../glossary.md#data-type--datatype) consumption, [scalar](../../../../glossary.md#term-scalar)/zero/dynamic/[view](../../../../glossary.md#term-view-meanings) forms, [broadcasting](../../../../glossary.md#broadcasting), [slice](../../../../glossary.md#slice) updates, atomics,
relaxed results, output [aliasing](../../../../glossary.md#term-aliasing), transfer [widening](../../../../glossary.md#term-widening)/fallback, split execution, shared-layer
production edits, public [API](../../../../glossary.md#term-application-programming-interface-api)/configuration, dependencies, new exports, benchmarks, or tuning.
Arithmetic scatter needs separate exact [rounding](../../../../glossary.md#term-rounding)/order/[NaN](../../../../glossary.md#term-nan)/zero/[subnormal](../../../../glossary.md#term-subnormal) proof; Scatter-ND needs
separate tuple/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/suffix geometry.

## Files and symbols

Expected existing [owners](../../../../glossary.md#term-owner) only:

- `MetalCapabilityProvider`, `MetalBackendRuntime`, `MetalMpsGraphProgram`;
- `MetalNegPartitionPreparer`, `MetalNegPreparedExecutable`, `MetalNativeApi`;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`;
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`;
- focused existing [Metal](../../../../glossary.md#term-metal), [conformance](../../../../glossary.md#term-conformance), and [Engine](../../../../glossary.md#term-engine) [integration test](../../../../glossary.md#term-integration-test) owners;
- affected current Metal/API/glossary/architecture-status/planning/Javadoc documents after behavior
  stabilizes.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Add no production package or shared-[module](../../../../glossary.md#term-module) file. Stop and mark Blocked if implementation requires a
new shared abstraction, module edge, public [API](../../../../glossary.md#term-application-programming-interface-api), export/[ABI](../../../../glossary.md#term-application-binary-interface-abi) version, fallback, relaxed result, or
unresolved collision rule.

## [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) proof and [validation](../../../../glossary.md#term-validation)

Before production edits, [run](../../../../glossary.md#term-run) one disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) valid-selector [smoke](../../../../glossary.md#term-smoke-test) for the only concrete
undocumented risk: exact Set-mode bit movement. Use one [device](../../../../glossary.md#term-device)/context/[graph](../../../../glossary.md#graph)/executable/optimization/
execution, one positive-[rank](../../../../glossary.md#term-rank) [FLOAT32](../../../../glossary.md#term-float32)/[INT32](../../../../glossary.md#data-type--datatype) case with unique valid [targets](../../../../glossary.md#term-target), and a minimal corpus that
proves addressed update plus unaddressed base bits including [signed zero](../../../../glossary.md#term-signed-zero), one [subnormal](../../../../glossary.md#term-subnormal), and
signaling-[NaN](../../../../glossary.md#term-nan) payloads. Assert unchanged inputs. Do not probe documented mappings, OOB, duplicates,
rank/[Shape](../../../../glossary.md#shape)/[axis](../../../../glossary.md#term-axis)/type variations, contexts, optimizations, or repetition. Remove every [artifact](../../../../glossary.md#term-artifact). A
failure makes this task Blocked before production edits.

Reuse generic [lifecycle](../../../../glossary.md#lifecycle), close, [ABI](../../../../glossary.md#term-application-binary-interface-abi)/export, typed-ingress, [publication](../../../../glossary.md#term-publication), [codec](../../../../glossary.md#term-codec), and [conformance](../../../../glossary.md#term-conformance) tests.
Add only tests that defend plausible scatter defects:

- exact capability/profile acceptance and nearby exclusions;
- third-edge/[schema](../../../../glossary.md#term-schema)/native [preflight](../../../../glossary.md#term-preflight), wrong reduction/type/Shape/state/axis, stale schema/[identity](../../../../glossary.md#term-identity-meanings);
- valid [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) plus exact bounds/duplicate diagnostics and unchanged sentinels/inputs;
- third-edge/axis/reduction [cache](../../../../glossary.md#term-cache) [authentication](../../../../glossary.md#term-authentication) and stale-byte rejection;
- one [backend](../../../../glossary.md#backend)-conformance case;
- one real CPU-free public Engine test containing valid raw-bit replacement and both value-aware
  failure classes through the production ABI.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>

No same-path [parameter](../../../../glossary.md#term-parameter) rows, source-text assertions, [mock](../../../../glossary.md#term-mock) echoes, invalid-input cross-product,
performance test, or duplicate lifecycle suite.

Worker validation:

```bash
native/metal-macos-arm64/build.sh
./gradlew :backends:metal:test
./gradlew :testing:backend-conformance:test --tests '*Metal*Scatter*'
./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
./gradlew :backends:metal:javadoc
./gradlew build
python3 /tmp/validate_synaptik_markdown.py

git diff --check
```

Run the full build once after executable and documentation changes stabilize. If the temporary
Markdown validator is absent, use an equivalent external validator for local targets/anchors,
unique effective anchors, balanced fences, final newlines, and trailing whitespace.

## Acceptance and review

- Only exact [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32)/[INT32](../../../../glossary.md#data-type--datatype)/FLOAT32 `SCATTER_ELEMENTS/NONE` is advertised/executable under
  both profiles; valid unique replacement preserves all bits and inputs.
- Bounds then uniqueness complete before dispatch/write with exact deterministic messages and
  unchanged failure targets; selector skip/collision behavior is unreachable.
- Schema 10/wire 18/typed third edge and version-eleven identities fail closed while ABI 4,
  thirteen exports, existing typed ingress/publication, and FLOAT32-only transfer remain exact.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [publication glossary](../../../../glossary.md#term-publication))</span>
- Shared production is unchanged and no complete backward claim appears.
- The one disposable smoke, focused permanent proof, final validation, current documentation, clean
  diff, and independent Class C <span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test); [validation glossary](../../../../glossary.md#term-validation))</span>`APPROVE` with zero findings close the task.

Independent review inspects selector/Set use, third-edge [schema](../../../../glossary.md#term-schema), pre-[dispatch](../../../../glossary.md#term-dispatch) bounds/uniqueness,
diagnostic ordering, scratch [lifetime](../../../../glossary.md#term-lifetime)/complexity, unchanged [targets](../../../../glossary.md#term-target)/inputs, no [alias](../../../../glossary.md#aliasing-and-arena) path, profile
monotonicity, [identity](../../../../glossary.md#term-identity-meanings) cutover, typed boundaries, [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) [device](../../../../glossary.md#term-device) evidence, backward boundary, and diff.
It reuses successful evidence and reruns executable work only for a concrete identified risk.

## Implementation [checkpoint](../../../../glossary.md#term-checkpoint) for independent review

Implementation began from exact clean revision
`ad07e7e5d029f0472f1d05e82f2e869899d7dbf7`. Before any production edit, one disposable
[Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) program compiled against the production frameworks and ran exactly once on the M3 Max.
Its one Set-mode data-taking [Scatter Elements](../../../../glossary.md#scatter-elements) execution reported
`task0024 Set smoke passed: exact addressed/unaddressed raw bits and unchanged inputs`. The corpus
covered [signed zero](../../../../glossary.md#term-signed-zero), one [subnormal](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and signaling-[NaN](../../../../glossary.md#term-nan) payloads in addressed updates and
unaddressed base cells. The source and executable were removed immediately; no bounds, duplicate,
[Shape](../../../../glossary.md#shape)/type/profile/context/optimization, or repetition probe ran.

The implementation adds only [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) [FLOAT32](../../../../glossary.md#term-float32)/[INT32](../../../../glossary.md#data-type--datatype)/FLOAT32
`SCATTER_ELEMENTS/NONE` to both profile [matrices](../../../../glossary.md#term-matrix-meanings). [Schema](../../../../glossary.md#term-schema) 10 appends wire 18 and carries updates in
the renamed typed auxiliary cell without changing the 160-byte record. Native creation authenticates
types, state, rank, Shape, [axis](../../../../glossary.md#term-axis), [topology](../../../../glossary.md#term-topology), and stable index feeds. Each synchronized executable [run](../../../../glossary.md#term-run)
validates every resource, completes [row-major](../../../../glossary.md#term-row-major) bounds, rebuilds complete [target](../../../../glossary.md#term-target) [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) in
bounded executable-owned primitive scratch, sorts them for nonquadratic uniqueness, and performs
no [tensor](../../../../glossary.md#tensor)-data construction, [dispatch](../../../../glossary.md#term-dispatch), or target write until every [descriptor](../../../../glossary.md#term-descriptor) passes. Java's
exceptional rescan reproduces the exact bounds and smallest-later/smallest-earlier duplicate
messages and preserves an unmatched native failure. Version-eleven workload, exact-policy,
candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) reject older bytes. [ABI](../../../../glossary.md#term-application-binary-interface-abi) 4, thirteen
exports, FLOAT32-only transfer, shared production, and the incomplete backward boundary are
unchanged.

Worker evidence passed:

- the [native bridge](../../../../glossary.md#term-native-bridge) build, an exact thirteen-[symbol](../../../../glossary.md#term-symbol) `nm -gU` export audit, and Foundation,
  Metal, and MetalPerformanceShadersGraph linkage audit;
- focused capability, schema/native preflight, real native raw-bit/failure-sentinel, prepared
  diagnostic, route-identity, backend-conformance, and CPU-free public Engine methods;<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- architecture tests and [Metal](../../../../glossary.md#term-metal) [Javadoc](../../../../glossary.md#term-javadoc); and
- the single final `./gradlew build` invocation: `BUILD SUCCESSFUL`, 87 actionable tasks, eight
  executed and 79 up-to-date.

The exact 34-path checkpoint is:

- production under
  `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/`:
  `MetalCapabilityProvider.java`, `MetalMpsGraphProgram.java`, `MetalNativeApi.java`,
  `MetalNegPartitionPreparer.java`, `MetalNegPreparedExecutable.java`,
  `MetalNegRouteCandidateGenerator.java`, `MetalNegTuningBatch.java`, `MetalNegTuningCodec.java`,
  and `package-info.java`; plus
  `native/metal-macos-arm64/src/synaptik_metal_foundation.m`;
- tests under `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/`:
  `MetalCapabilityProviderTest.java`, `MetalMpsGraphAbsNativeTest.java`,
  `MetalMpsGraphAffineSchemaTest.java`, `MetalMpsGraphBinaryNativeTest.java`,
  `MetalMpsGraphIndexingNativeTest.java`, `MetalMpsGraphRawAbiNativeTest.java`,
  `MetalMpsGraphReductionNativeTest.java`, `MetalNegPreparedExecutionTest.java`, and
  `MetalNegRouteCandidateGeneratorTest.java`; plus
  `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java`
  and
  `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.java`;
- documentation: `ARCHITECTURE.md`, `docs/api/compile-api.md`, `docs/api/public-api.md`,
  `docs/api/tensor-api.md`, `docs/architecture/contracts/backend-execution.md`,
  `docs/architecture/module-boundaries.md`, `docs/backend-guide/metal-backend.md`,
  `docs/backend-guide/partition-preparer.md`, `docs/glossary.md`,
  `docs/planning/backends/metal/master-plan.md`,
  `docs/planning/backends/metal/tasks/0024-exact-int32-scatter-elements-replacement.md`,
  `docs/planning/roadmap.md`, and `native/metal-macos-arm64/README.md`.

Implementation `a947e574732273bee4469d42afe8935082d53109`, documentation remediation
`aa191ca469010d081150e97dcddd504ec626dd9e`, and final independent [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) Class C `APPROVE` with zero
findings close this task.

## Architecture impact

Expected impact: None. This extends the existing [backend](../../../../glossary.md#backend)-private route and [ownership](../../../../glossary.md#term-ownership-meanings). [Planning](../../../../glossary.md#term-planning) ran
no [device](../../../../glossary.md#term-device) probe or production edit because documented type/[Shape](../../../../glossary.md#shape)/valid mapping is authoritative and
the single undocumented movement risk belongs to the implementation gate above.
