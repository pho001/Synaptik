# Task 0019: Exact Profile-Qualified [FLOAT32](../../../../glossary.md#term-float32) ABS

## Status

Complete

Readiness verification: [Metal](../../../../glossary.md#term-metal) 0015, [Model](../../../../glossary.md#term-model) 0027, Config 0006, and [Engine](../../../../glossary.md#term-engine) 0018 are Complete. Metal
0016 is Blocked at its unchanged three-[operation](../../../../glossary.md#operation) gate; it made no production, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), or
test change. Its mandatory Apple M3 Max probe proved the exact `ABS` [slice](../../../../glossary.md#slice) across every level,
[Shape](../../../../glossary.md#shape), context, and [topology](../../../../glossary.md#term-topology) form. The executable baseline remains schema 5 with version-six
identities at clean revision `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514`. No conflicting Metal
write task is active. Historical Metal 0006 remains Blocked and is not a [dependency](../../../../glossary.md#term-dependency).

## Change class

Class C — this changes profile-qualified [Metal](../../../../glossary.md#term-metal) capability, typed native [schema](../../../../glossary.md#term-schema), whole-[partition](../../../../glossary.md#partition)
[lowering](../../../../glossary.md#lowering), prepared compatibility, and public [Engine](../../../../glossary.md#term-engine) execution across the native resource boundary.
Fresh implementation proof and independent review are mandatory.

## Goal

Add exactly `UnaryElementwiseKind.ABS` for [canonical](../../../../glossary.md#term-canonical-form) [`FLOAT32`](../../../../glossary.md#term-float32) [Metal](../../../../glossary.md#term-metal) execution under both
`STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator). The [operation](../../../../glossary.md#operation) has the same exact result contract in both profiles;
this task adds no relaxed [unary](../../../../glossary.md#term-unary) result, approximation, or profile-specific arithmetic.

## [Scope](../../../../glossary.md#term-scope)

- Require one [`FLOAT32`](../../../../glossary.md#term-float32) input and output, `NoOperationAttrs.INSTANCE`, fully static positive [rank](../../../../glossary.md#term-rank)
  `1..16`, equal [Shape](../../../../glossary.md#shape) and `requiresGrad`, checked geometry, and canonical dense-contiguous,
  zero-offset non-view descriptors.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Require exact magnitude bits for every non-NaN input, including every subnormal and normal
  boundary; require <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [subnormal glossary](../../../../glossary.md#term-subnormal))</span>`-0 -> +0`, both infinities to positive infinity, and NaN classification under
  the existing Model NaN representation freedom.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Model glossary](../../../../glossary.md#term-model))</span>
- `ABS` consumes only canonical state and produces canonical state. A strict affine view must pass
  through [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) before `ABS`; [canonical](../../../../glossary.md#term-canonical-form) `ABS` output may feed another admitted [operation](../../../../glossary.md#operation).
- Preserve [MPSGraph](../../../../glossary.md#term-mpsgraph) as the sole `ABS` route. The custom singleton route remains strict NEG-only.
  Preserve transactional finalization, context leases, direct targets, run isolation, synchronous
  submission, reuse, concurrency, close rejection, and canonical-only transfer.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [leases glossary](../../../../glossary.md#term-lease); [targets glossary](../../../../glossary.md#term-target); [run glossary](../../../../glossary.md#term-run); [synchronous glossary](../../../../glossary.md#term-synchronous); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>

### Exact profile [matrices](../../../../glossary.md#term-matrix-meanings)

These are the complete [Metal](../../../../glossary.md#term-metal) [occurrence](../../../../glossary.md#term-occurrence) matrices after 0019. A kind omitted from a row is rejected
before native resource creation.

| Requested profile | Admitted [canonical](../../../../glossary.md#term-canonical-form) arithmetic | Admitted [layout](../../../../glossary.md#layout) operations | Numerical rule |
|---|---|---|---|
| `STRICT_IEEE` | `NEG`, `ABS` | `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) | Existing exact per-[operation](../../../../glossary.md#operation) [Model](../../../../glossary.md#term-model) results only |
| [`ACCELERATOR`](../../../../glossary.md#term-accelerator) | [Tensor](../../../../glossary.md#tensor) `ADD`, `SUB`, `MUL`, `DIV`; `ABS` | None | Existing bounded [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) result sets; exact `ABS` with no relaxation |

Strict rejects all tensor binary operations. Accelerator rejects NEG and all six layout operations.
Both profiles reject [`EXP`](../../../../glossary.md#term-expx), [`SIGMOID`](../../../../glossary.md#term-sigmoid), every other [unary](../../../../glossary.md#term-unary) operation, [scalar](../../../../glossary.md#term-scalar) arithmetic, reductions,
[MATMUL](../../../../glossary.md#matrix-multiplication--matmul), comparisons, [extrema](../../../../glossary.md#saturation-pole-and-extremum), scans, [BOOL](../../../../glossary.md#data-type--datatype)/WHERE, and every unlisted family. Java capability,
analysis, and native-create [preflight](../../../../glossary.md#term-preflight) enforce this matrix; the private native [schema](../../../../glossary.md#term-schema) validates the
authenticated [topology](../../../../glossary.md#term-topology), [Shape](../../../../glossary.md#shape), and state without becoming the profile-policy authority.

### Reused and fresh evidence

The pre-production selector/numerical gate is reused from 0016 rather than rerun solely because the
failed broad task was split. At executable base `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514`, the
Apple M3 Max probe disabled reduced-[precision](../../../../glossary.md#term-precision) fast math and covered optimization levels `0` and
`1`; [Shapes](../../../../glossary.md#shape) `[128]`, `[2,4,4,4]`, and [rank](../../../../glossary.md#term-rank) 16; two fresh contexts; alone, repeated-[node](../../../../glossary.md#node)/[fan-out](../../../../glossary.md#term-fan-out),
and direct-[target](../../../../glossary.md#term-target) forms; three independently compiled executables per cell; and eight runs per
executable. The `ABS` [slice](../../../../glossary.md#slice) passed all 108 executables and 864 runs with exact non-[NaN](../../../../glossary.md#term-nan) bits and all
input-preservation, [canary](../../../../glossary.md#term-canary), Shape, direct-target, feed/target [permutation](../../../../glossary.md#permutation), and no-[DAZ](../../../../glossary.md#term-daz) controls. The
probe source and binaries were removed.

Reuse is valid only while the [MPSGraph](../../../../glossary.md#term-mpsgraph) selector, probe [oracle](../../../../glossary.md#term-test-oracle)/corpus, relevant native executable
baseline, Apple M3 Max target, and [SDK](../../../../glossary.md#term-sdk)/[toolchain](../../../../glossary.md#term-toolchain) remain unchanged. A change to any of those, a
missing raw summary, or a review finding about coverage requires a fresh disposable gate before
production edits. [Planning](../../../../glossary.md#term-planning)-only descendants of the recorded base do not invalidate it. Regardless
of reuse, all [backend](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance), native, [identity](../../../../glossary.md#term-identity-meanings), and real [Engine](../../../../glossary.md#term-engine) evidence below is fresh after
0019 implementation against the newly built [dylib](../../../../glossary.md#term-dylib).

### [Schema](../../../../glossary.md#term-schema) and [identity](../../../../glossary.md#term-identity-meanings) evolution

- Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, statuses, create [signature](../../../../glossary.md#term-signature), and exactly thirteen exports.
- Advance the fixed 160-byte [node](../../../../glossary.md#node) schema `5 -> 6`; retain wires `NEG=1`, [tensor](../../../../glossary.md#tensor) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `2..5`, and
  affine/contiguous <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>`6..11`; append only `ABS=12`. Wires `13`, `14`, and every other unknown wire
  remain rejected.
- Wire `ABS=12` has attribute kind `NONE`, one input, `second_input = UINT32_MAX`, zero attribute
  count/payload/reserved cells, equal input/output Shape, canonical input state, and canonical
  output state. Java and native validation must agree.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation))</span>
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`6 -> 7`. Include requested profile, node schema 6, ordered ABS topology/edges,
  descriptors, states, targets, geometry, ABI, and splats. Version-six, cross-profile, malformed,
  corrupt, foreign-session, and unknown-wire data reject or miss safely.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [descriptors glossary](../../../../glossary.md#term-descriptor); [targets glossary](../../../../glossary.md#term-target))</span>

## Non-goals

No [`EXP`](../../../../glossary.md#term-expx), [`SIGMOID`](../../../../glossary.md#term-sigmoid), `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `SQRT`, `RSQRT`, `RELU`, [`TANH`](../../../../glossary.md#term-tanh), other [unary](../../../../glossary.md#term-unary) kind,
relaxed unary semantics, [scalar](../../../../glossary.md#term-scalar) arithmetic, reduction, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), comparison, [extrema](../../../../glossary.md#saturation-pole-and-extremum), [scan](../../../../glossary.md#term-cumulative-scan), new [data
type](../../../../glossary.md#data-type--datatype), [rank](../../../../glossary.md#term-rank) zero, dynamic [Shape](../../../../glossary.md#shape), [view](../../../../glossary.md#term-view-meanings) input, custom unary [kernel](../../../../glossary.md#kernel), [host](../../../../glossary.md#term-host) rewrite, value-dependent
[dispatch](../../../../glossary.md#term-dispatch), transfer [widening](../../../../glossary.md#term-widening), backward/training claim, packaging, discovery, or performance claim.
Never authorize the historically observed gross `RELU(NaN) -> +0` or `TANH(NaN) -> +1` results.
[Model](../../../../glossary.md#term-model) 0026 remains Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model
  owns result sets and profile selection stays cold.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — exact `ABS` is valid in both profiles; unlisted [unary](../../../../glossary.md#term-unary) operations gain no relaxation.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> — Metal
  owns truthful capability, lowering, native integration, and resources.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> —
  routes and resources are fixed before direct hot execution.

If exact `ABS` needs a shared contract change, new [ABI](../../../../glossary.md#term-application-binary-interface-abi) export, custom [kernel](../../../../glossary.md#kernel), [host](../../../../glossary.md#term-host) rewrite, fallback,
or value-dependent [dispatch](../../../../glossary.md#term-dispatch), stop and report the blocker.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0015; [Model](../../../../glossary.md#term-model) 0027; Config 0006; [Engine](../../../../glossary.md#term-engine) 0018; not Metal 0016
- Conflicts with: any Metal 0016 restart; Metal 0017–0018; every Metal capability, preparation,
  native schema, candidate/codec, materialization, or public Metal Engine integration scope<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514` plus this [planning](../../../../glossary.md#term-planning)-only authorization
- Integration order: Serial; after 0015 and before Metal 0017–0018
- Integration [validation](../../../../glossary.md#term-validation): exact both-profile Metal ABS [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, and `MetalNegPreparationPlan` — exact
  two-profile occurrence matrix, topology/state validation, lowering, and retained profile.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [topology glossary](../../../../glossary.md#term-topology); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering))</span>
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, and the native foundation — schema
  six, wire <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`ABS=12`, Java [preflight](../../../../glossary.md#term-preflight), [MPSGraph](../../../../glossary.md#term-mpsgraph) selector, and [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4 stability.
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and decision
  compatibility — version-seven identities and stale rejection.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- Focused Metal schema/native/execution/candidate tests, Metal backend conformance, and the real
  explicit-composition Metal Engine integration test — exact positive and fail-closed proof.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [integration test glossary](../../../../glossary.md#term-integration-test); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Metal package Javadocs, backend/native guides, this brief, Metal master plan, roadmap, focused API
  and glossary status — current capability and exclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

1. The recorded 0016 `ABS` gate is explicitly accepted under the reuse policy or a required fresh
   replacement passes before production edits; no disposable probe artifact remains.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
2. The exact profile matrices above are exhaustive in capability, analysis, preparation, and
   native-create preflight. Every excluded occurrence rejects before native resource creation.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [preflight glossary](../../../../glossary.md#term-preflight); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
3. [Schema](../../../../glossary.md#term-schema) six appends only wire `ABS=12`; ABI v4 retains exactly thirteen exports. Java and native
   validation agree on records, states, Shapes, sentinels, and topology.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [validation glossary](../../../../glossary.md#term-validation); [Shapes glossary](../../../../glossary.md#shape); [topology glossary](../../../../glossary.md#term-topology))</span>
4. Version-seven workload/candidate/compatibility/route/codec identities are deterministic and
   reject version six, cross-profile, malformed, corrupt, stale-session, and unknown data.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
5. Fresh real-dylib backend tests prove exact raw bits for both profiles, every semantic bucket,
   strict and accelerator compositions, reuse, concurrency, independent contexts, direct targets,
   and close rejection; negative tests prove the full closed matrix.<span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [backend glossary](../../../../glossary.md#backend); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [accelerator glossary](../../../../glossary.md#term-accelerator); [concurrency glossary](../../../../glossary.md#term-concurrency); [targets glossary](../../../../glossary.md#term-target))</span>
6. A fresh CPU-free public Engine test selects each profile in turn, executes and materializes
   observable exact <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>`ABS` results including signed zero/subnormals/infinities/NaNs, composes only
   with that profile's admitted operations, reuses a session, opens an independent session, and
   proves close rejection. It must not reduce the claim to compile or not-throw behavior.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [subnormals glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [compile glossary](../../../../glossary.md#compile))</span>
7. Runtime/Trace stay profile-free; transfer, lifecycle, public API, and Model semantics do not
   change. Focused validation and independent Class C review pass with no unresolved finding.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
git diff --check
```

Inspect the accepted or fresh [oracle](../../../../glossary.md#term-test-oracle) summary, exact exports/[schema](../../../../glossary.md#term-schema), generated [Metal](../../../../glossary.md#term-metal) Javadocs,
materialized [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) under both profiles, complete negative [matrix](../../../../glossary.md#term-matrix-meanings), and absence of new
[Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) profile state. Validate changed Markdown against the clean baseline for links,
anchors, fences, final newlines, and whitespace.

Integration/repository validation: the serial task owns the exact both-profile Metal ABS [checkpoint](../../../../glossary.md#term-checkpoint);
[CI](../../../../glossary.md#term-continuous-integration-ci) owns the next full repository build unless implementation changes a [dependency](../../../../glossary.md#term-dependency) or shared build
contract.

## Documentation and review impact

Update affected [Metal](../../../../glossary.md#term-metal) Javadocs, Metal/native guides, current capability/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary status, this
brief, master plan, and roadmap. Independent Class C review must inspect evidence reuse validity,
exact raw-bit semantics, both complete profile [matrices](../../../../glossary.md#term-matrix-meanings), every [unary](../../../../glossary.md#term-unary) exclusion, gross [ReLU](../../../../glossary.md#term-relu)/[tanh](../../../../glossary.md#term-tanh)
exclusions, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings) evolution, [lifecycle](../../../../glossary.md#lifecycle), real [Engine](../../../../glossary.md#term-engine) evidence, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence, and
exact changed-path [scope](../../../../glossary.md#term-scope). The review reuses successful worker tests unless it changes executable
behavior or identifies a concrete stale-evidence risk.

## Result

Implementation `a6d1796d` plus mixed-[owner](../../../../glossary.md#term-owner) test remediation `bcb717a6` completed the task.
Gate reuse was accepted after comparing `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514`
with the clean implementation base: only [planning](../../../../glossary.md#term-planning) documents had changed. The selector,
[oracle](../../../../glossary.md#term-test-oracle)/corpus, native executable/build baseline, Apple M3 Max [target](../../../../glossary.md#term-target), and [SDK](../../../../glossary.md#term-sdk)/[toolchain](../../../../glossary.md#term-toolchain) remained
unchanged, so 0016's 108-executable/864-[run](../../../../glossary.md#term-run) raw ABS result remained valid.

Both profile [matrices](../../../../glossary.md#term-matrix-meanings) now admit exact [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) ABS through [MPSGraph](../../../../glossary.md#term-mpsgraph) only. [Schema](../../../../glossary.md#term-schema) 6 appends
`ABS=12`; wires `1..11`, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, all statuses, and exactly thirteen exports remain unchanged.
Workload, exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) are version
seven; stale version-six, cross-profile, foreign-session, malformed, corrupt, and unknown-wire
inputs [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

Independent [validation](../../../../glossary.md#term-validation) passed the native ABI-v4 exact-thirteen-export audit, 86 [Metal](../../../../glossary.md#term-metal) tests, four
Metal [conformance](../../../../glossary.md#term-conformance) tests, eight real-[dylib](../../../../glossary.md#term-dylib) [Engine](../../../../glossary.md#term-engine) tests, nine architecture tests, the full 87-task
build, and planning/documentation/diff checks. Independent Class C final review returned
`APPROVE` with zero findings. No limitation, unresolved issue, or follow-up remains.

Status: Complete
