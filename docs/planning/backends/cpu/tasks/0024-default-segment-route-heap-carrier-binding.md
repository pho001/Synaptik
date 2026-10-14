# CPU 0024 — Bind heap inputs to the prepared default segment route

## Status and launch verification

Complete — launched as the authorized CPU frontier on 2026-10-14. The user approved a separate
CPU repair after Checkpoint 0004 exposed this pre-existing failure. CPU 0022 and the ordinary
Engine/CPU route were present at `b85bf3a2`; no CPU writer was active at launch, and CPU 0023
remains an unrelated Draft.

An ordinary `TensorFactory.fromFlatArray` FLOAT64 heap leaf reproduced the failure without any
Checkpoint code: `Engine.compute(leaf.contiguous())` rejected
`bufferSelections[0] is incompatible with prepared executable`. At launch, Checkpoint 0004 could
not complete its direct restored-Tensor acceptance until this was repaired. This task was
independent of its checkpoint format and ran as the one CPU frontier.

## Goal and class

Class C — fix CPU cold binding of caller-owned primitive-array heap storage to an already
prepared portable route, without moving route selection into Runtime or the element hot path.
Make ordinary heap leaves and checkpoint-restored heap leaves directly executable through the
public Engine CPU path while preserving native-segment and explicitly selected array-carrier
routes.

## Scope and non-goals

- Diagnose the exact mismatch between CPU's default `MEMORY_SEGMENT` carrier pattern and
  `CpuBufferRepresentation.argument()`'s observable primitive-array classification. Correct
  CPU-owned compatibility/binding for the prepared carrier form, including composed partition
  and selected materialization paths where relevant. Keep preparation and generated identity
  truthful; do not merely bypass a check or copy all heap inputs to native memory.
- Use a CPU-private cold-binding choice or equivalent proved approach. Preserve exact borrowed
  segment/array identity and lifetime, logical type, size, alignment, read-only and overlap
  checks. Keep existing explicitly specialized array routes and their no-copy hot-loop shape.
- Add focused CPU representation/executable regressions and public Engine integration tests for
  direct heap `CONTIGUOUS` execution. Cover FLOAT64/FLOAT32/BFLOAT16/FLOAT16/INT32/INT64/BOOL,
  rank-zero and empty shape where supported, exact signed-zero/NaN/low/integer/BOOL bits, and
  native versus heap source behavior. Include a nonzero heap-slice offset or explain precisely
  why the prepared route rejects it. A failed incompatibility must still fail closed.
- Do not change Checkpoint schema, NN/Model semantics, Runtime route ownership, backend identity,
  Engine public API, OpenBLAS qualification, or tuning policy. A minimal cross-module regression
  test is allowed under `testing/integration-tests`; the coordinator owns the follow-up
  Checkpoint 0004 direct-execution assertion after the CPU repair.

## Contracts and files

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [`backend-execution.md` — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes):
  CPU owns carrier and route realization within one backend.
- [`runtime-prepare-engine.md` — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle), and
  [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  no Runtime or Engine carrier-specific bypass.
- Start with `CpuBufferRepresentation`, `CpuBufferArgument`, `CpuPreparedExecutable`,
  `CpuPreparedPartitionExecutable`, `CpuPartitionPreparer`, and their focused tests. Change only
  the minimal owning CPU classes once the mismatch is traced. Add/update Javadocs and the
  focused CPU guide if its carrier contract changes.

## Acceptance and validation

- The public CPU Engine computes directly from ordinary and checkpoint-restored heap leaves;
  no native test mirror is needed. Output bytes equal represented input bits for exact movement.
  The existing native-segment path and explicit primitive-array specialization still work.
- Binding chooses only a form compatible with the prepared kernel and actual storage, before
  invocation. No per-element carrier dispatch, new copy, generic fallback with hidden overhead,
  or unannounced generated-code shape/identity change. Prove safety for read-only inputs,
  alignment, offsets, overlap, resource lifetime, and multi-unit/materialized execution where
  the affected binding path applies.
- If generated bytecode changes, compare the changed specialized loop with optimal direct Java
  and provide proportionate semantic, structural, and bounded performance evidence. Otherwise
  explicitly record why the generated body and compatibility identity remain unchanged.
- Run focused CPU representation/executable tests, affected public integration/conformance tests,
  then `./gradlew :backends:cpu:check :backends:cpu:javadoc` once after executable edits. A clean
  independent Class C review checks the actual binding and documentation diff; the coordinator
  reruns direct Checkpoint acceptance and one combined `./gradlew test javadoc` gate after
  integration. `git diff --check` must pass.

## Dependencies and integration

- Depends on: completed CPU 0022 and existing CPU/Engine execution; Checkpoint 0004 supplied
  reproduction but is not a code dependency.
- Conflicts with: CPU 0023 and any concurrent CPU carrier/preparation/generated-identity writer;
  Checkpoint 0004 remained unfinalized until this repair and its direct regression pass.
- Parallel group: None.
- Common base revision: N/A — serial CPU work from `b85bf3a2` plus disjoint in-progress
  Checkpoint 0004 changes; no concurrent write executor at launch.
- Integration order: CPU repair and independent review, then Checkpoint direct acceptance and
  closure.
- Integration validation: affected CPU and public integration tests, Checkpoint direct
  round-trip, and one combined repository `test javadoc` gate.
- Shared-document integration owner: coordinator; worker owns only focused CPU source/tests,
  focused CPU guide/Javadocs, and disjoint public integration test.

## Result

CPU cold binding now passes the exact selected segment to a prepared `MEMORY_SEGMENT` route,
including matching primitive-array heap storage and nonzero slices. Explicit array routes retain
their classified direct arguments; no copy, generated body, schema, or compatibility identity
changed. The first independent review found a P1 mixed segment/array alias check: overlapping
heap reads and writes could escape rejection. The fix compares exact selected segments and
accessed spans across both prepared forms; a clean independent re-review approved it with no
remaining code finding.

Changed files: `CpuBufferRepresentation`, `CpuBufferArgument`, `CpuPreparedExecutable`,
`CpuPreparedPartitionExecutable`, their focused representation/executable tests,
`EngineCpuHeapContiguousIntegrationTest`, and `docs/backend-guide/cpu-backend.md`. CPU Javadocs
were updated for the cold-binding and lifetime contract; no public API or architecture edit was
needed for this CPU repair.

Validation: focused `CpuPreparedExecutableTest` (61), `CpuPreparedPartitionExecutableTest` (4),
`CpuNativeRepresentationTest` (2), and public `EngineCpuHeapContiguousIntegrationTest` (3)
passed with zero failures; `./gradlew :backends:cpu:check :backends:cpu:javadoc` passed after the
overlap fix. The combined source-matched Metal checkpoint
`./gradlew test javadoc --max-workers=2 --console=plain` also passed (105 tasks: 1 executed,
104 up-to-date) with `SYNAPTIK_METAL_TEST_LIBRARY` set to the local dylib. The earlier
unconfigured run's 15 Metal failures required that environment variable, not a CPU code change.
No unresolved CPU 0024 issue or follow-up remains; CPU 0023 stays Draft.

Status: Complete
