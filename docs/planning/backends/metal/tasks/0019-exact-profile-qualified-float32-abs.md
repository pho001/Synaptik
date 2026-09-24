# Task 0019: Exact Profile-Qualified FLOAT32 ABS

## Status

Complete

Readiness verification: Metal 0015, Model 0027, Config 0006, and Engine 0018 are Complete. Metal
0016 is Blocked at its unchanged three-operation gate; it made no production, schema, identity, or
test change. Its mandatory Apple M3 Max probe proved the exact `ABS` slice across every level,
Shape, context, and topology form. The executable baseline remains schema 5 with version-six
identities at clean revision `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514`. No conflicting Metal
write task is active. Historical Metal 0006 remains Blocked and is not a dependency.

## Change class

Class C — this changes profile-qualified Metal capability, typed native schema, whole-partition
lowering, prepared compatibility, and public Engine execution across the native resource boundary.
Fresh implementation proof and independent review are mandatory.

## Goal

Add exactly `UnaryElementwiseKind.ABS` for canonical `FLOAT32` Metal execution under both
`STRICT_IEEE` and `ACCELERATOR`. The operation has the same exact result contract in both profiles;
this task adds no relaxed unary result, approximation, or profile-specific arithmetic.

## Scope

- Require one `FLOAT32` input and output, `NoOperationAttrs.INSTANCE`, fully static positive rank
  `1..16`, equal Shape and `requiresGrad`, checked geometry, and canonical dense-contiguous,
  zero-offset non-view descriptors.
- Require exact magnitude bits for every non-NaN input, including every subnormal and normal
  boundary; require `-0 -> +0`, both infinities to positive infinity, and NaN classification under
  the existing Model NaN representation freedom.
- `ABS` consumes only canonical state and produces canonical state. A strict affine view must pass
  through `CONTIGUOUS` before `ABS`; canonical `ABS` output may feed another admitted operation.
- Preserve MPSGraph as the sole `ABS` route. The custom singleton route remains strict NEG-only.
  Preserve transactional finalization, context leases, direct targets, run isolation, synchronous
  submission, reuse, concurrency, close rejection, and canonical-only transfer.

### Exact profile matrices

These are the complete Metal occurrence matrices after 0019. A kind omitted from a row is rejected
before native resource creation.

| Requested profile | Admitted canonical arithmetic | Admitted layout operations | Numerical rule |
|---|---|---|---|
| `STRICT_IEEE` | `NEG`, `ABS` | `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, `CONTIGUOUS` | Existing exact per-operation Model results only |
| `ACCELERATOR` | Tensor `ADD`, `SUB`, `MUL`, `DIV`; `ABS` | None | Existing bounded binary DAZ/FTZ result sets; exact `ABS` with no relaxation |

Strict rejects all tensor binary operations. Accelerator rejects NEG and all six layout operations.
Both profiles reject `EXP`, `SIGMOID`, every other unary operation, scalar arithmetic, reductions,
MATMUL, comparisons, extrema, scans, BOOL/WHERE, and every unlisted family. Java capability,
analysis, and native-create preflight enforce this matrix; the private native schema validates the
authenticated topology, Shape, and state without becoming the profile-policy authority.

### Reused and fresh evidence

The pre-production selector/numerical gate is reused from 0016 rather than rerun solely because the
failed broad task was split. At executable base `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514`, the
Apple M3 Max probe disabled reduced-precision fast math and covered optimization levels `0` and
`1`; Shapes `[128]`, `[2,4,4,4]`, and rank 16; two fresh contexts; alone, repeated-node/fan-out,
and direct-target forms; three independently compiled executables per cell; and eight runs per
executable. The `ABS` slice passed all 108 executables and 864 runs with exact non-NaN bits and all
input-preservation, canary, Shape, direct-target, feed/target permutation, and no-DAZ controls. The
probe source and binaries were removed.

Reuse is valid only while the MPSGraph selector, probe oracle/corpus, relevant native executable
baseline, Apple M3 Max target, and SDK/toolchain remain unchanged. A change to any of those, a
missing raw summary, or a review finding about coverage requires a fresh disposable gate before
production edits. Planning-only descendants of the recorded base do not invalidate it. Regardless
of reuse, all backend, conformance, native, identity, and real Engine evidence below is fresh after
0019 implementation against the newly built dylib.

### Schema and identity evolution

- Keep native ABI version 4, statuses, create signature, and exactly thirteen exports.
- Advance the fixed 160-byte node schema `5 -> 6`; retain wires `NEG=1`, tensor binary `2..5`, and
  affine/contiguous `6..11`; append only `ABS=12`. Wires `13`, `14`, and every other unknown wire
  remain rejected.
- Wire `ABS=12` has attribute kind `NONE`, one input, `second_input = UINT32_MAX`, zero attribute
  count/payload/reserved cells, equal input/output Shape, canonical input state, and canonical
  output state. Java and native validation must agree.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities `6 -> 7`. Include requested profile, node schema 6, ordered ABS topology/edges,
  descriptors, states, targets, geometry, ABI, and splats. Version-six, cross-profile, malformed,
  corrupt, foreign-session, and unknown-wire data reject or miss safely.

## Non-goals

No `EXP`, `SIGMOID`, `RECIPROCAL`, `LOG`, `SQRT`, `RSQRT`, `RELU`, `TANH`, other unary kind,
relaxed unary semantics, scalar arithmetic, reduction, MATMUL, comparison, extrema, scan, new data
type, rank zero, dynamic Shape, view input, custom unary kernel, host rewrite, value-dependent
dispatch, transfer widening, backward/training claim, packaging, discovery, or performance claim.
Never authorize the historically observed gross `RELU(NaN) -> +0` or `TANH(NaN) -> +1` results.
Model 0026 remains Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model
  owns result sets and profile selection stays cold.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — exact `ABS` is valid in both profiles; unlisted unary operations gain no relaxation.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — Metal
  owns truthful capability, lowering, native integration, and resources.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  routes and resources are fixed before direct hot execution.

If exact `ABS` needs a shared contract change, new ABI export, custom kernel, host rewrite, fallback,
or value-dependent dispatch, stop and report the blocker.

## Dependencies and integration

- Depends on: Metal 0015; Model 0027; Config 0006; Engine 0018; not Metal 0016
- Conflicts with: any Metal 0016 restart; Metal 0017–0018; every Metal capability, preparation,
  native schema, candidate/codec, materialization, or public Metal Engine integration scope
- Parallel group: None
- Common base revision: `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514` plus this planning-only authorization
- Integration order: Serial; after 0015 and before Metal 0017–0018
- Integration validation: exact both-profile Metal ABS checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, and `MetalNegPreparationPlan` — exact
  two-profile occurrence matrix, topology/state validation, lowering, and retained profile.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, and the native foundation — schema
  six, wire `ABS=12`, Java preflight, MPSGraph selector, and ABI-v4 stability.
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and decision
  compatibility — version-seven identities and stale rejection.
- Focused Metal schema/native/execution/candidate tests, Metal backend conformance, and the real
  explicit-composition Metal Engine integration test — exact positive and fail-closed proof.
- Metal package Javadocs, backend/native guides, this brief, Metal master plan, roadmap, focused API
  and glossary status — current capability and exclusions.

## Acceptance criteria

1. The recorded 0016 `ABS` gate is explicitly accepted under the reuse policy or a required fresh
   replacement passes before production edits; no disposable probe artifact remains.
2. The exact profile matrices above are exhaustive in capability, analysis, preparation, and
   native-create preflight. Every excluded occurrence rejects before native resource creation.
3. Schema six appends only wire `ABS=12`; ABI v4 retains exactly thirteen exports. Java and native
   validation agree on records, states, Shapes, sentinels, and topology.
4. Version-seven workload/candidate/compatibility/route/codec identities are deterministic and
   reject version six, cross-profile, malformed, corrupt, stale-session, and unknown data.
5. Fresh real-dylib backend tests prove exact raw bits for both profiles, every semantic bucket,
   strict and accelerator compositions, reuse, concurrency, independent contexts, direct targets,
   and close rejection; negative tests prove the full closed matrix.
6. A fresh CPU-free public Engine test selects each profile in turn, executes and materializes
   observable exact `ABS` results including signed zero/subnormals/infinities/NaNs, composes only
   with that profile's admitted operations, reuses a session, opens an independent session, and
   proves close rejection. It must not reduce the claim to compile or not-throw behavior.
7. Runtime/Trace stay profile-free; transfer, lifecycle, public API, and Model semantics do not
   change. Focused validation and independent Class C review pass with no unresolved finding.

## Validation

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

Inspect the accepted or fresh oracle summary, exact exports/schema, generated Metal Javadocs,
materialized raw bits under both profiles, complete negative matrix, and absence of new
Runtime/Trace profile state. Validate changed Markdown against the clean baseline for links,
anchors, fences, final newlines, and whitespace.

Integration/repository validation: the serial task owns the exact both-profile Metal ABS checkpoint;
CI owns the next full repository build unless implementation changes a dependency or shared build
contract.

## Documentation and review impact

Update affected Metal Javadocs, Metal/native guides, current capability/API/glossary status, this
brief, master plan, and roadmap. Independent Class C review must inspect evidence reuse validity,
exact raw-bit semantics, both complete profile matrices, every unary exclusion, gross ReLU/tanh
exclusions, schema/identity evolution, lifecycle, real Engine evidence, Runtime/Trace absence, and
exact changed-path scope. The review reuses successful worker tests unless it changes executable
behavior or identifies a concrete stale-evidence risk.

## Result

Implementation `a6d1796d` plus mixed-owner test remediation `bcb717a6` completed the task.
Gate reuse was accepted after comparing `9f7ad2d3e3d0fbdd690f200c14887fd4a158b514`
with the clean implementation base: only planning documents had changed. The selector,
oracle/corpus, native executable/build baseline, Apple M3 Max target, and SDK/toolchain remained
unchanged, so 0016's 108-executable/864-run raw ABS result remained valid.

Both profile matrices now admit exact canonical FLOAT32 ABS through MPSGraph only. Schema 6 appends
`ABS=12`; wires `1..11`, ABI v4, all statuses, and exactly thirteen exports remain unchanged.
Workload, exact-policy, candidate, compatibility, route-policy, and codec identities are version
seven; stale version-six, cross-profile, foreign-session, malformed, corrupt, and unknown-wire
inputs fail closed.

Independent validation passed the native ABI-v4 exact-thirteen-export audit, 86 Metal tests, four
Metal conformance tests, eight real-dylib Engine tests, nine architecture tests, the full 87-task
build, and planning/documentation/diff checks. Independent Class C final review returned
`APPROVE` with zero findings. No limitation, unresolved issue, or follow-up remains.

Status: Complete
