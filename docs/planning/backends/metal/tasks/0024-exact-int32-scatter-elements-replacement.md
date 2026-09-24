# Task 0024: Exact INT32 SCATTER_ELEMENTS Replacement

## Status

Ready

Metal 0023 is Complete at implementation `9a7911c9447eeb5125ceb8a4d78e55349ec18b97`
plus evidence `80e6cedda96df2a03f284d551e6e34046eaaa2f3`; final independent lean Class C
review returned `APPROVE` with zero findings. The dependencies below are Complete, no declared
conflict is active, and this is the sole Ready Metal frontier.

## Change class

Class C — three-input functional update execution, duplicate-target validation, native schema/cache
identity evolution, and public Engine execution cross the native boundary. One minimal real-device
smoke for the only undocumented raw-bit risk and one independent final review are mandatory. Broad
selector, Shape, type, profile, invalid-index, context, optimization, or repetition matrices are
prohibited.

## Coordination metadata

- Depends on: Metal 0023; Model 0018G–0018H, 0018O, and 0025C; Compiler 0005C; CPU 0006B1;
  Config 0006; Engine 0018; current completed shared lifecycle/opaque-backend contracts.
- Conflicts with: any Metal capability, schema, native-preflight, candidate/codec,
  ingress/publication, materialization, or public Engine scope; any blocked Metal 0016–0018 restart;
  arithmetic scatter, Scatter-ND, Model 0026, or shared-layer mutation.
- Parallel group: None.
- Common base revision: N/A; this work is serial after the planning checkpoint.
- Integration order: schema/identity, capability/analysis, native validation/lowering, prepared
  diagnostics, public proof, documentation, independent review.
- Integration validation: focused Metal/conformance/real-Engine checkpoint, native export audit,
  architecture checks, Javadoc, one final build, Markdown/diff validation.
- Shared-document integration owner: Metal 0024 implementer.

## Goal

Add only canonical positive-rank `FLOAT32` `SCATTER_ELEMENTS` with canonical `INT32` indices and
`ScatterReduction.NONE`, identically under `STRICT_IEEE` and `ACCELERATOR`.

The operation consumes exact ordered inputs `[data, indices, updates]`, returns one new canonical
data-shaped `FLOAT32` value, and never mutates an input. Each logical update coordinate replaces
its selected axis with the corresponding non-negative index and copies the exact update bits to
that unique target. Unaddressed output coordinates preserve exact base-data bits. Invalid bounds or
overlapping targets fail before selector dispatch or any target write.

## Verified contract and selector prerequisites

- Model 0018G/0018H define same-rank functional `SCATTER_ELEMENTS`, normalized axis, explicit
  reduction, exact ordered roles, equal indices/update Shape, data-shaped fresh result, exact type,
  and `data.requiresGrad || updates.requiresGrad`; Model 0018O fixes its current canonical taxonomy.
- Model 0018I/0018J define separate tuple-index `SCATTER_ND`; Model 0023B defines separate
  Gather-compatible fixed-add `SCATTER_ADD`; neither is an alias or dependency of the selected
  operation.
- Model 0025C keeps `NONE` as replacement with unique-target validity and no arithmetic reduction.
- Compiler 0005C owns fail-closed scatter preflight and exact formulas. For
  `SCATTER_ELEMENTS/NONE`, the base cotangent is replacement scatter with zero updates, the update
  cotangent is `GATHER_ELEMENTS`, and indices are non-differentiable.
- CPU 0006B1 is the diagnostic oracle: full bounds validation precedes `NONE` duplicate validation
  and every output write; update positions are row-major ordinals.
- Metal 0023 supplies exact canonical FLOAT32/INT32 caller and splat ingress, local FLOAT32
  publication, pre-dispatch index scanning, schema 9/wires through 17, and version-ten identities.
- Config 0006 and Engine 0018 carry the immutable profile. Shared Runtime, Prepare, Backend
  Contract, Trace, and Engine lifecycle ownership needs no change.

The active SDK `MPSGraphScatterNDOps.h` documents the macOS 12.3 selector:

```objc
- (MPSGraphTensor *) scatterAlongAxis:(NSInteger) axis
                       withDataTensor:(MPSGraphTensor *) dataTensor
                        updatesTensor:(MPSGraphTensor *) updatesTensor
                        indicesTensor:(MPSGraphTensor *) indicesTensor
                                 mode:(MPSGraphScatterMode) mode
                                 name:(NSString * _Nullable) name;
```

It documents matching updates/indices Shapes, data matching except at the axis, INT32/INT64
indices, and `MPSGraphScatterModeSet`. That valid type/Shape/coordinate mapping is authoritative and
needs no matrix. The same header says out-of-bounds updates are skipped and defines no portable
replacement winner for overlaps; both behaviors are unreachable because Metal prevalidates bounds
and uniqueness.

No semantic or architecture prerequisite remains open for this exact subset.

## Exact supported domain

Every admitted occurrence has:

- kind `AxisScatterKind.SCATTER_ELEMENTS`, exact `ScatterElementsAttrs(axis, NONE)`, three ordered
  inputs and one output;
- `[data FLOAT32, indices INT32, updates FLOAT32] -> FLOAT32`;
- canonical dense, zero-offset, fully static rank `1..16` values with positive extents;
- equal data/indices/updates rank, exact indices/update Shape, and data matching every non-axis
  extent; the positive selected update extent may differ from data;
- result Shape exactly data Shape and normalized axis in `[0, rank)`;
- result gradient eligibility exactly the data/update OR and non-differentiable indices;
- canonical inputs on consuming edges, with no implicit materialization.

Data and updates may reuse the same read-only logical value when Shapes permit; existing feed
deduplication remains legal. Output is a distinct graph value/destination; add no in-place or alias
optimization. Canonical local FLOAT32 producers may feed data/updates, and the result may feed a
compatible canonical consumer or target. Indices use current caller/splat INT32 ingress.

Capability is identical under both profiles. The normative profile table gives scatter/fold no
relaxation; `ACCELERATOR` adds no result and remains a monotonic superset.

For logical update coordinate `u` and `i = indices[u]`:

```text
target(u)[d] = i when d == axis, otherwise u[d]
result[target(u)] = raw bits of updates[u]
result[x] = raw bits of data[x] when no update addresses x
```

Unique targets make update order irrelevant. Row-major order exists only for diagnostics. Preserve
signed zero, subnormals, infinities, and NaN sign/payload/quiet-or-signaling bits as movement; inputs
remain byte-for-byte unchanged. Use only the data-taking selector with
`MPSGraphScatterModeSet`; do not synthesize replacement with arithmetic, masks, one-hot, reduction,
or a custom kernel.

## Invalid-index and overlap contract

Negative indices and indices `>= data.shape[axis]` are invalid; never wrap, clamp, skip, or pad.
Two updates overlap when their complete derived target coordinates match. Under `NONE`, that is
invalid. Equal index values at different non-axis coordinates are not duplicates.

Run validation order is exact:

1. current ABI counts, handle arrays, resource sizes, and typed bindings;
2. retained indexing descriptors in stable node order;
3. for each scatter, every index in row-major ordinal order for bounds;
4. only after that occurrence's complete bounds pass, uniqueness, selecting the smallest later
   duplicate ordinal and the smallest earlier ordinal for that target;
5. only after every descriptor passes, construct tensor data, dispatch, or expose a target write.

Exact CPU-parity messages are:

```text
SCATTER_ELEMENTS index at logical position <ordinal> for data axis <axis> is out of bounds: value=<value>, extent=<extent>
SCATTER_ELEMENTS duplicate target at logical update position <later>; first addressed at logical update position <first>
```

Native may return the existing index-domain status. Java rescans retained indices only on that
exceptional status in the same order; unmatched rejection is an internal failure. Every target and
input byte remains unchanged on failure. Valid-path uniqueness uses checked executable-owned
scratch or another bounded primitive representation, has no per-run Objective-C object graph, and
is not quadratic. Executable-owned scratch is rebuilt per invocation under the existing
synchronized resource run; allocation/size overflow rejects creation without fallback.

## Typed boundary, schema, and identity

Reuse exact FLOAT32 data/update ingress, INT32 index ingress, and local FLOAT32 publication. Native
typing authenticates `[FLOAT32, INT32, FLOAT32]`; conflicting value reuse fails. CPU↔Metal transfer
remains canonical positive-rank FLOAT32 only.

Advance backend-local node schema `9 -> 10`:

- append operation wire `SCATTER_ELEMENTS = 18` and reuse `AXIS = 3`;
- retain the 160-byte record; reinterpret offset 28 as typed operation-specific `auxiliary`:
  reduction keep-dimensions for reductions, third stable input for Scatter Elements, zero otherwise;
- rename Java/native internal `reserved` terminology atomically;
- encode data/indices/updates/result as first/second/auxiliary/output, axis count one, normalized
  axis, and no payload;
- reject every wrong arity/state/type/rank/Shape/axis/reduction/unused field and every old/unknown
  schema or wire before graph compilation.

Native calls the documented selector once with Set mode. Do not use shape-initializing/dynamic-axis
or Scatter-ND overloads, reflection, string dispatch, or fallback. ABI remains `4` with exactly
thirteen exports and unchanged signatures/ownership.

Advance workload signature, exact-policy, candidate, compatibility, route-policy, and codec
identities `10 -> 11` together. Hash the exact three ordered edges, kind/reduction/axis,
types/Shapes/layout states/output metadata, schema 10, ABI 4, and profile/common policy. Old bytes
fail closed; add no shim, alias, dual decoder, or migration.

## Compiler and execution boundary

Compiler production remains unchanged. Capability is occurrence-based and phase-agnostic, so an
ordinary Compiler-generated `SCATTER_ELEMENTS/NONE` occurrence may use this row. That is not full
backward support: update cotangents require unsupported `GATHER_ELEMENTS`, and arithmetic scatter is
excluded. Make no seeded-gradient, training, optimizer, tape, or all-Metal backward claim.

This task also excludes SCATTER_ADD, Scatter-ND, every arithmetic reduction, INT64, non-FLOAT32
data, BOOL consumption, scalar/zero/dynamic/view forms, broadcasting, slice updates, atomics,
relaxed results, output aliasing, transfer widening/fallback, split execution, shared-layer
production edits, public API/configuration, dependencies, new exports, benchmarks, or tuning.
Arithmetic scatter needs separate exact rounding/order/NaN/zero/subnormal proof; Scatter-ND needs
separate tuple/batch/suffix geometry.

## Files and symbols

Expected existing owners only:

- `MetalCapabilityProvider`, `MetalBackendRuntime`, `MetalMpsGraphProgram`;
- `MetalNegPartitionPreparer`, `MetalNegPreparedExecutable`, `MetalNativeApi`;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`;
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`;
- focused existing Metal, conformance, and Engine integration test owners;
- affected current Metal/API/glossary/architecture-status/planning/Javadoc documents after behavior
  stabilizes.

Add no production package or shared-module file. Stop and mark Blocked if implementation requires a
new shared abstraction, module edge, public API, export/ABI version, fallback, relaxed result, or
unresolved collision rule.

## Lean proof and validation

Before production edits, run one disposable Objective-C valid-selector smoke for the only concrete
undocumented risk: exact Set-mode bit movement. Use one device/context/graph/executable/optimization/
execution, one positive-rank FLOAT32/INT32 case with unique valid targets, and a minimal corpus that
proves addressed update plus unaddressed base bits including signed zero, one subnormal, and
signaling-NaN payloads. Assert unchanged inputs. Do not probe documented mappings, OOB, duplicates,
rank/Shape/axis/type variations, contexts, optimizations, or repetition. Remove every artifact. A
failure makes this task Blocked before production edits.

Reuse generic lifecycle, close, ABI/export, typed-ingress, publication, codec, and conformance tests.
Add only tests that defend plausible scatter defects:

- exact capability/profile acceptance and nearby exclusions;
- third-edge/schema/native preflight, wrong reduction/type/Shape/state/axis, stale schema/identity;
- valid prepared execution plus exact bounds/duplicate diagnostics and unchanged sentinels/inputs;
- third-edge/axis/reduction cache authentication and stale-byte rejection;
- one backend-conformance case;
- one real CPU-free public Engine test containing valid raw-bit replacement and both value-aware
  failure classes through the production ABI.

No same-path parameter rows, source-text assertions, mock echoes, invalid-input cross-product,
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

- Only exact canonical FLOAT32/INT32/FLOAT32 `SCATTER_ELEMENTS/NONE` is advertised/executable under
  both profiles; valid unique replacement preserves all bits and inputs.
- Bounds then uniqueness complete before dispatch/write with exact deterministic messages and
  unchanged failure targets; selector skip/collision behavior is unreachable.
- Schema 10/wire 18/typed third edge and version-eleven identities fail closed while ABI 4,
  thirteen exports, existing typed ingress/publication, and FLOAT32-only transfer remain exact.
- Shared production is unchanged and no complete backward claim appears.
- The one disposable smoke, focused permanent proof, final validation, current documentation, clean
  diff, and independent Class C `APPROVE` with zero findings close the task.

Independent review inspects selector/Set use, third-edge schema, pre-dispatch bounds/uniqueness,
diagnostic ordering, scratch lifetime/complexity, unchanged targets/inputs, no alias path, profile
monotonicity, identity cutover, typed boundaries, lean device evidence, backward boundary, and diff.
It reuses successful evidence and reruns executable work only for a concrete identified risk.

## Architecture impact

Expected impact: None. This extends the existing backend-private route and ownership. Planning ran
no device probe or production edit because documented type/Shape/valid mapping is authoritative and
the single undocumented movement risk belongs to the implementation gate above.
