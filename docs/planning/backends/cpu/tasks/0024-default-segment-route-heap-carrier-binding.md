# [CPU](../../../../glossary.md#term-cpu) 0024 — Bind [heap](../../../../glossary.md#term-heap) inputs to the prepared default segment route

## Reader context — historical integration boundary

Here a ‘heap leaf’ is a source Tensor whose current host values occupy a primitive Java array, not a
computed expression waiting to execute. Its immediate Engine caller may be application code or
Checkpoint/NN code supplying restored/model storage. Engine asks CPU to borrow that storage, and CPU
cold binding chooses the argument form required by the already prepared route. Passing an exact
array-backed `MemorySegment` is not copying values to native memory or transferring array ownership.
Closing a borrowed wrapper does not close the original storage. Mixed array/segment aliases must
still be compared physically, because two wrappers can address overlapping bytes of the same array.

Current reading: [Supported lifecycle integration](../../../../backend-guide/cpu-backend.md#supported-lifecycle-integration); [Metadata binding versus produced-value checks](../../../../backend-guide/cpu-backend.md#binding-metadata-and-validating-produced-values).

## Status and launch verification

Complete — launched as the authorized [CPU](../../../../glossary.md#term-cpu) frontier on 2026-10-14. The user approved a separate
CPU repair after [Checkpoint](../../../../glossary.md#term-checkpoint) 0004 exposed this pre-existing failure. CPU 0022 and the ordinary
[Engine](../../../../glossary.md#term-engine)/CPU route were present at `b85bf3a2`; no CPU writer was active at launch, and CPU 0023
remains an unrelated Draft.

An ordinary `TensorFactory.fromFlatArray` [FLOAT64](../../../../glossary.md#term-float64) [heap](../../../../glossary.md#term-heap) [leaf](../../../../glossary.md#term-leaf) reproduced the failure without any
Checkpoint code: `Engine.compute(leaf.contiguous())` rejected
`bufferSelections[0] is incompatible with prepared executable`. At launch, Checkpoint 0004 could
not complete its direct restored-[Tensor](../../../../glossary.md#tensor) acceptance until this was repaired. This task was
independent of its checkpoint format and ran as the one CPU frontier.

## Goal and class

Class C — fix [CPU](../../../../glossary.md#term-cpu) cold [binding](../../../../glossary.md#term-binding) of caller-owned primitive-array [heap](../../../../glossary.md#term-heap) storage to an already
prepared portable route, without moving route selection into [Runtime](../../../../glossary.md#term-runtime) or the [element](../../../../glossary.md#term-element) [hot path](../../../../glossary.md#term-hot-path).
Make ordinary heap leaves and [checkpoint](../../../../glossary.md#term-checkpoint)-restored heap leaves directly executable through the
public [Engine](../../../../glossary.md#term-engine) CPU path while preserving native-segment and explicitly selected array-[carrier](../../../../glossary.md#term-storage-carrier)
routes.

## [Scope](../../../../glossary.md#term-scope) and non-goals

- Diagnose the exact mismatch between [CPU](../../../../glossary.md#term-cpu)'s default `MEMORY_SEGMENT` [carrier](../../../../glossary.md#term-storage-carrier) pattern and
  `CpuBufferRepresentation.argument()`'s observable primitive-array classification. Correct
  CPU-owned compatibility/binding for the prepared carrier form, including composed partition
  and selected materialization paths where relevant. Keep preparation and generated identity
  truthful; do not merely bypass a check or copy all heap inputs to native memory.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [partition glossary](../../../../glossary.md#partition); [materialization glossary](../../../../glossary.md#materialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [heap glossary](../../../../glossary.md#term-heap); [native memory glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Use a CPU-private cold-binding choice or equivalent proved approach. Preserve exact borrowed
  segment/array identity and lifetime, logical type, size, alignment, read-only and overlap
  checks. Keep existing explicitly specialized array routes and their no-copy hot-loop shape.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lifetime glossary](../../../../glossary.md#term-lifetime); [alignment glossary](../../../../glossary.md#term-alignment); [shape glossary](../../../../glossary.md#shape))</span>
- Add focused CPU representation/executable regressions and public Engine integration tests for
  direct heap <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) execution. Cover FLOAT64/FLOAT32/BFLOAT16/FLOAT16/INT32/INT64/BOOL,
  rank-zero and empty shape where supported, exact signed-zero/NaN/low/integer/BOOL bits, and
  native versus heap source behavior. Include a nonzero heap-slice offset or explain precisely
  why the prepared route rejects it. A failed incompatibility must still fail closed.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [slice glossary](../../../../glossary.md#slice); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Do not change Checkpoint schema, NN/Model semantics, Runtime route ownership, backend identity,
  Engine public API, OpenBLAS qualification, or tuning policy. A minimal cross-module regression
  test is allowed under <span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [schema glossary](../../../../glossary.md#term-schema); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend identity glossary](../../../../glossary.md#backend-identity--backendid); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [OpenBLAS qualification glossary](../../../../glossary.md#openblas-qualification); [module glossary](../../../../glossary.md#term-module); [regression glossary](../../../../glossary.md#term-regression))</span>`testing/integration-tests`; the coordinator owns the follow-up
  Checkpoint 0004 direct-execution assertion after the CPU repair.

## Contracts and files

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [`backend-execution.md` — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>:
  CPU owns carrier and route realization within one backend.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- [`runtime-prepare-engine.md` — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>, and
  [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  no Runtime or Engine carrier-specific bypass.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Start with `CpuBufferRepresentation`, `CpuBufferArgument`, `CpuPreparedExecutable`,
  `CpuPreparedPartitionExecutable`, `CpuPartitionPreparer`, and their focused tests. Change only
  the minimal owning CPU classes once the mismatch is traced. Add/update Javadocs and the
  focused CPU guide if its carrier contract changes.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- The public CPU Engine computes directly from ordinary and checkpoint-restored heap leaves;
  no native test mirror is needed. Output bytes equal represented input bits for exact movement.
  The existing native-segment path and explicit primitive-array specialization still work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [heap glossary](../../../../glossary.md#term-heap); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Binding chooses only a form compatible with the prepared kernel and actual storage, before
  invocation. No per-element carrier dispatch, new copy, generic fallback with hidden overhead,
  or unannounced generated-code shape/identity change. Prove safety for read-only inputs,
  alignment, offsets, overlap, resource lifetime, and multi-unit/materialized execution where
  the affected binding path applies.<span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding); [kernel glossary](../../../../glossary.md#kernel); [element glossary](../../../../glossary.md#term-element); [carrier glossary](../../../../glossary.md#term-storage-carrier); [dispatch glossary](../../../../glossary.md#term-dispatch); [shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings); [alignment glossary](../../../../glossary.md#term-alignment); [resource lifetime glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- If generated bytecode changes, compare the changed specialized loop with optimal direct Java
  and provide proportionate semantic, structural, and bounded performance evidence. Otherwise
  explicitly record why the generated body and compatibility identity remain unchanged.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Run focused CPU representation/executable tests, affected public integration/conformance tests,
  then <span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [conformance glossary](../../../../glossary.md#term-conformance))</span>`./gradlew :backends:cpu:check :backends:cpu:javadoc` once after executable edits. A clean
  independent Class C review checks the actual binding and documentation diff; the coordinator
  reruns direct Checkpoint acceptance and one combined `./gradlew test javadoc` gate after
  integration. `git diff --check` must pass.

## Dependencies and integration

- Depends on: completed CPU 0022 and existing CPU/Engine execution; Checkpoint 0004 supplied
  reproduction but is not a code dependency.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Conflicts with: CPU 0023 and any concurrent CPU carrier/preparation/generated-identity writer;
  Checkpoint 0004 remained unfinalized until this repair and its direct regression pass.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [identity glossary](../../../../glossary.md#term-identity-meanings); [regression glossary](../../../../glossary.md#term-regression))</span>
- Parallel group: None.
- Common base revision: N/A — serial [CPU](../../../../glossary.md#term-cpu) work from `b85bf3a2` plus disjoint in-progress
  Checkpoint 0004 changes; no concurrent write executor at launch.
- Integration order: CPU repair and independent review, then Checkpoint direct acceptance and
  closure.
- Integration validation: affected CPU and public integration tests, Checkpoint direct
  round-trip, and one combined repository <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>`test javadoc` gate.
- Shared-document integration owner: coordinator; worker owns only focused CPU source/tests,
  focused CPU guide/Javadocs, and disjoint public integration test.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [integration test glossary](../../../../glossary.md#term-integration-test))</span>

## Result

[CPU](../../../../glossary.md#term-cpu) cold [binding](../../../../glossary.md#term-binding) now passes the exact selected segment to a prepared `MEMORY_SEGMENT` route,
including matching primitive-array [heap](../../../../glossary.md#term-heap) storage and nonzero slices. Explicit array routes retain
their classified direct [arguments](../../../../glossary.md#term-argument); no copy, generated body, [schema](../../../../glossary.md#term-schema), or compatibility [identity](../../../../glossary.md#term-identity-meanings)
changed. The first independent review found a P1 mixed segment/array [alias](../../../../glossary.md#aliasing-and-arena) check: overlapping
heap reads and writes could escape rejection. The fix compares exact selected segments and
accessed spans across both prepared forms; a clean independent re-review approved it with no
remaining code finding.

Changed files: `CpuBufferRepresentation`, `CpuBufferArgument`, `CpuPreparedExecutable`,
`CpuPreparedPartitionExecutable`, their focused representation/executable tests,
`EngineCpuHeapContiguousIntegrationTest`, and `docs/backend-guide/cpu-backend.md`. CPU Javadocs
were updated for the cold-binding and [lifetime](../../../../glossary.md#term-lifetime) contract; no public [API](../../../../glossary.md#term-application-programming-interface-api) or architecture edit was
needed for this CPU repair.

[Validation](../../../../glossary.md#term-validation): focused `CpuPreparedExecutableTest` (61), `CpuPreparedPartitionExecutableTest` (4),
`CpuNativeRepresentationTest` (2), and public `EngineCpuHeapContiguousIntegrationTest` (3)
passed with zero failures; `./gradlew :backends:cpu:check :backends:cpu:javadoc` passed after the
overlap fix. The combined source-matched [Metal](../../../../glossary.md#term-metal) [checkpoint](../../../../glossary.md#term-checkpoint)
`./gradlew test javadoc --max-workers=2 --console=plain` also passed (105 tasks: 1 executed,
104 up-to-date) with `SYNAPTIK_METAL_TEST_LIBRARY` set to the local [dylib](../../../../glossary.md#term-dylib). The earlier
unconfigured [run](../../../../glossary.md#term-run)'s 15 Metal failures required that environment variable, not a CPU code change.
No unresolved CPU 0024 issue or follow-up remains; CPU 0023 stays Draft.

Status: Complete
