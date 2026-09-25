# Task 0040: Profile-Common Exact Canonical BFLOAT16 Gather

## Status

Ready

This is the sole Ready Metal frontier. Its prerequisites are implemented and reviewed, its conflicts
are inactive, and its executable work is authorized from exact clean base
`b495e4e43ce5bdcd962256f9bfc726301f3d1549`. This planning-only authorization runs no probe,
build, test, or implementation.

## Change class

Class C — implementation changes a backend capability, private native node/type schema, caller
storage admission, local materialization, and cache identity while preserving ABI and cross-owner
transfer boundaries. It requires one clean implementation context and independent lean Class C
review.

## Goal

Deliver one exact profile-common Metal execution path for canonical positive-rank BFLOAT16 data
`GATHER` with canonical INT32 indices. Preserve selected raw 16-bit representations from public
caller ingress through Metal-local chaining and detached host publication, without BFLOAT16
arithmetic, conversion, CPU fallback, or CPU/Metal transfer widening.

## Scope

- Admit exactly `AxisGatherKind.GATHER` with `IndexAxisAttrs`, ordered `[data, indices]`, and one
  output. Data and output are BFLOAT16; indices are INT32.
- Require fully static Shapes with strictly positive extents and checked byte geometry. Data and
  indices ranks are `1..16`; normalized axis is in the data rank; output rank is `1..16` and its
  Shape is `data[:axis] + indices.shape + data[axis + 1:]`.
- Require canonical dense-contiguous, zero-offset, non-view layouts for both inputs and output.
  Data/output `requiresGrad` values remain equal metadata; no backward execution is claimed.
- Copy the exact selected BFLOAT16 raw short. Signed zeros, subnormals, ordinary normals,
  infinities, and NaN sign/payload/signaling representations are never widened, rounded, decoded,
  canonicalized, or recreated numerically.
- Advertise the identical occurrence and represented-bit behavior under `STRICT_IEEE` and
  `ACCELERATOR`; profile identity remains distinct and cold.
- Admit caller-owned raw-short BFLOAT16 host storage through Metal upload without conversion.
  Admit a locally produced canonical BFLOAT16 Gather output as data for another BFLOAT16 Gather in
  the same Metal partition.
- Publish a Metal-local canonical BFLOAT16 target directly as canonical row-major big-endian
  two-byte elements through the owning Metal integration and existing `HostTensorValue` contract.
- Reuse current complete INT32 prevalidation, stable node/ordinal failure order, no-dispatch and
  no-target-write failure behavior, and exact public bounds exception reconstruction.

### Native schema and identity cutover

- Advance node schema `11 -> 12` and append only type-specific `BFLOAT16_GATHER=20` with two
  inputs, `AXIS=3`, and canonical output. Keep FLOAT32 `GATHER=16` unchanged.
- Add inferred local value type `BFLOAT16=4`, width two, mapped to `MPSDataTypeBFloat16` only on its
  documented available runtime. A type-specific operation wire is required because ABI v4 carries
  no per-value type table.
- Keep ABI v4, the fixed 160-byte node record, exactly thirteen exports, attribute wires `0..6`,
  existing status/state/reduction wires, and every C signature unchanged. Add no attribute wire 7.
- Advance workload, exact-policy, candidate, compatibility, route-policy, and codec identities
  from 12 to 13. Add BFLOAT16 workload type identity 4. Reject schema 11 and identity 12; add no
  dual reader, migration shim, alias, or fallback.
- Infer BFLOAT16/INT32/BFLOAT16 for wire 20 in Java and native preflight, reject conflicting value
  reuse, and lower wire 20 through the existing batch-dimensions-zero Gather selector. Retain the
  existing MPSGRAPH route; custom NEG remains unrelated.

### One mandatory lean device gate

The installed headers establish the BFLOAT16 representation and generic Gather structure, but not
the selector's concrete raw-bit movement. Before production edits, run exactly one disposable graph,
one compile, and one execution with one rank-one BFLOAT16 feed and one reverse INT32 permutation.
The small raw-short feed is exactly:

```text
0000 8000 0001 8001 007f 807f 3f80 bf80 4120 c120
7f80 ff80 7fc1 7fe5 ffc2 ffe6 7f81 7fa5 ff82 ffa6
```

This covers positive/negative zero, minimum/maximum subnormal, ordinary normals, infinities, and
multiple positive/negative quiet-NaN and signaling-NaN payloads. The single execution must produce
the exact reverse raw-short sequence, preserve both input buffers byte-for-byte, and preserve
prefilled target guard bytes. Do not enumerate all 65,536 patterns, repeat contexts/runs, or add a
Shape/axis/operation matrix.

If compilation, execution, any raw word, either input, or a guard fails, remove the probe, mark this
task Blocked, and make no production change. Do not narrow the corpus, compare numerically,
canonicalize NaNs, widen through FLOAT32, substitute `CONTIGUOUS`, or fall back to host/CPU work.
Remove every disposable probe artifact after either outcome.

## Non-goals

BFLOAT16 arithmetic, unary/reduction/matrix operations, affine/view input or publication,
`CONTIGUOUS`, window operations, casts, mixed precision, scalar or logical-splat constants,
INT64/BFLOAT16 indices, zero/dynamic/scalar Shapes, noncanonical layouts, in-place output,
backward/training, custom kernels, host result repair, retry/fallback, CPU production changes,
BFLOAT16 CPU/Metal transfer, public API additions, packaging/discovery, or performance claims.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle), and
  [compile ownership and mixed-owner routing](../../../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)

If implementation needs broader Model semantics, BFLOAT16 conversion/arithmetic, a new ABI/export,
shared type tables, noncanonical publication, cross-owner BFLOAT16 transfer, Compiler rewriting,
Runtime/Trace profile state, or different invalid-index semantics, stop and report the conflict.

## Dependencies and integration

- Depends on: Metal 0023, 0025, and 0038 Complete; current Model BFLOAT16 representation and
  GATHER semantics; Compiler indexing capture; CPU 0006A2 raw-bit oracle; Config 0006; Engine 0018
- Conflicts with: every concurrent Metal capability/schema/native/type/ingress/publication/
  identity/transfer scope and any restart that would consume the same next schema values
- Parallel group: None
- Common base revision: `b495e4e43ce5bdcd962256f9bfc726301f3d1549`
- Integration order: documentation-first gate; one disposable smoke; schema/type/identity;
  capability/preparation; ingress/publication; focused proof; synchronized docs; lean review
- Integration validation: Metal 0040 Class C checkpoint, including exactly one final full build
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- Metal capability/preparation/schedule/runtime: `MetalCapabilityProvider`,
  `MetalNegPartitionPreparer`, `MetalNegPreparedScheduleAssembler`, `MetalBackendRuntime`, and
  `MetalBackendIntegration`.
- Schema/native/error handling: `MetalMpsGraphProgram`, `MetalNativeApi.MpsGraphExecutableAbi`,
  `MetalNegPreparedExecutable`, and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`.
- Identity: `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, and `MetalNegTuningCodec`.
- Focused Metal capability, raw-ABI, prepared execution, candidate/codec, native indexing,
  conformance, and public Engine integration tests.
- After behavior passes: affected Javadocs/package information, `ARCHITECTURE.md`, the owning
  scoped contracts, current Metal/backend/API guides, native README, glossary impact, this brief,
  Metal master plan, and repository roadmap.

## Acceptance criteria

1. The one mandatory smoke uses only the listed representative vector and one graph/compile/run;
   exact reversed output, unchanged inputs, and unchanged guards pass, and artifacts are removed.
2. Both profiles admit exactly the canonical BFLOAT16+INT32 Gather domain with identical bit
   semantics; every named exclusion fails before native resource creation.
3. Caller raw-short BFLOAT16 ingress, same-partition BFLOAT16 Gather chaining, and Metal-local
   big-endian BFLOAT16 publication preserve every tested raw word without conversion.
4. Existing complete INT32 validation still precedes tensor-data construction, dispatch, and target
   writes; one focused BFLOAT16 invalid-index test proves unchanged inputs/target and diagnostics.
5. Schema 12 adds only wire 20 and type 4/width two. ABI v4, 160 bytes, thirteen exports, attributes
   `0..6`, and FLOAT32 wire 16 remain exact; Java/native validation agree.
6. All six identity versions are 13, BFLOAT16 workload type is 4, and stale schema-11/version-12,
   cross-profile, cross-type, malformed, and foreign decisions reject without compatibility shims.
7. CPU/Metal transfer remains positive-rank canonical FLOAT32-only; no BFLOAT16 mixed-owner graph,
   arithmetic, affine, constant, or backward route is advertised.
8. Focused Metal/native/conformance/public Engine/Javadoc/architecture checks, one final full build,
   Markdown/status/frontier/diff checks, and independent lean Class C review pass.

## Validation

Worker validation, after the single disposable smoke and production implementation:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
./gradlew build
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Run the full build exactly once after focused checks stabilize. Permanent tests use one compact
representative case per observable contract, reuse existing lifecycle/index-validation coverage,
and add no exhaustive bit corpus, Shape/axis matrix, source-text assertion, or duplicate smoke.
The independent reviewer inspects the final diff and reuses successful executable evidence unless
a concrete finding changes executable behavior.

## Documentation and review impact

This planning change owns only Task 0040, Task 0039's consumed-value correction, the Metal master
plan, and the repository roadmap. Implementation must update current architecture/capability/API/
native/Javadoc status only after behavior passes. No new glossary term is expected; the targeted
review must confirm that conclusion. Independent lean Class C review covers schema/type parity,
raw-bit semantics, ingress/publication ownership, transfer non-widening, stale cutover, tests,
documentation, and validation evidence without adding a second probe or broad rerun.

## Result

Planning-only authorization from the recorded clean base. No probe, implementation, build, or test
has run; executable acceptance remains open.
