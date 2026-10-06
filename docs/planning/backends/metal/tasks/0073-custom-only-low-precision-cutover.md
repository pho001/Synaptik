# Task 0073: Custom-Only Metal Low-Precision Cutover

## Status

Complete — the BFLOAT16/FLOAT16 (16-bit floating-point) cutover, integrated validation, and final
independent Class C rereview are recorded below.

## Change class

Class C — fixed backend route and Prepare behavior, native application binary interface (ABI),
prepared-resource lifetime, Trace diagnostic transfer object (DTO), and cross-module conformance
change together.

## Goal

Make every Metal partition containing BFLOAT16 or FLOAT16 use one fixed `CUSTOM_PROGRAM` candidate,
including exact raw-preserving movement; remove the unused machine-bound certificate and vendor
qualification path without changing admitted capability or numerical semantics.

## Scope

- Retain the existing custom low-precision arithmetic and exact raw-word `RESHAPE`, simple
  `PERMUTE`, materializing `CONTIGUOUS`, `SLICE`, `CONCAT`, and `TILE` execution. Reject low-precision
  MPSGraph partition selection in Java and native code. Low-valued operations use custom steps;
  any FLOAT32-only node may use the existing internal MPSGraph boundary policy in the same
  `CUSTOM_PROGRAM` partition, whether independent or after an explicit low-to-FLOAT32 cast.
  Never substitute BFLOAT16 with
  FLOAT16 or fuse generated-pointwise/anchor steps anywhere in that partition. Leave eligible
  FLOAT32 routes unchanged.
- Remove certificate DTOs/store/evidence, environment identity and export, vendor probes, and
  qualification scripts. Advance the native ABI from 6 to 7 with exactly thirteen exports and
  coordinated Metal workload, exact-policy, candidate, compatibility, route, and codec identities
  from 28 to 29. Keep program schema 19 and existing wire allocations.
- Emit only selected-route, ordered feed-then-target logical dtype, and profile facts in
  low-precision PREPARE trace; remove `accumulatorDtype` and its constants rather than replacing
  them with another working-type field. Retain authenticated cold selection, exact resource
  ownership, fail-closed identities, and session-scoped tuning.
- Reconcile focused Metal/native, Trace, backend-conformance, explanatory documentation, Javadocs,
  ADR 0025, and planning evidence against the final implementation diff.

## Non-goals

No new operation or gradient domain, Model numerical rule, FLOAT32 route change, schema/wire
renumbering, runtime fallback/retry, general cache, public release, Developer ID signing, or
notarization. The already-dirty `extensions/training` cleanup and its tests are a separate change;
Task 0073 neither owns nor validates them as a cutover result.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants): Model
  floors, fixed custom low route, ABI/identity, trace, lifecycle, and unchanged FLOAT32 policy.
- [Backend execution — Low-precision capability evidence](../../../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend):
  capability, candidate, native ABI, and resource boundaries.
- [Foundational modules — `modules/trace`](../../../../architecture/contracts/foundational-modules.md#modulestrace)
  and [Model low-precision contract and active ACCELERATOR extension](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract-and-active-accelerator-extension):
  typed diagnostics and unchanged allowed results.
- [Runtime/Prepare/Engine — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle):
  cold route and prepared-resource ownership.
- [ADR 0025](../../../../design/decisions/0025-custom-only-metal-low-precision.md) records the
  decision; it does not supersede the authoritative contracts. If they conflict, stop and resolve
  the contract before changing implementation.

## Dependencies and integration

- Depends on: completed Metal 0071 and Trace 0003; active Model low-precision and schema-19
  contracts; accepted ADR 0025. Metal 0053 remains independently Blocked.
- Frontier verification: user authorized documenting the already-running uncommitted cutover as a
  special Metal frontier at `8278a34a0395fa67ee0f673107d189e9d40ad970`. This is not a new
  `Ready` launch or permission for concurrent writes; prerequisite surfaces are present in the
  current tree, and completion gates remain open.
- Conflicts with: concurrent edits to Metal routes, native ABI/package, Trace low-precision DTOs,
  conformance identity, or their shared contracts/docs; any resumed Metal 0053 production work.
- Parallel group: None.
- Common base revision: N/A — one already-dirty worktree; initial HEAD is the revision above.
- Integration order: stabilize scoped implementation and documentation, run the final matrix,
  obtain independent targeted Class C review, then remediate/revalidate any findings.
- Integration validation: the native/package, focused modules, real-device integration,
  architecture, Javadoc, documentation, and repository-wide gates below, once on the final diff.
- Shared-document integration owner: Metal 0073 coordinator; serialize master-plan, roadmap, root
  contract, and explanatory-document reconciliation.

## Files and symbols

- `backends/metal/src/main/java/.../metal/` — `MetalNegPartitionPreparer`,
  `MetalNegRouteCandidateGenerator`, `MetalNegPreparationPlan`, `MetalNegPartitionFinalizer`,
  `MetalProgramExecutableResource`, `MetalTraceProducer`, `MetalNativeApi`, context/tuning/codec,
  and removed certificate/MPSGraph-resource classes; focused Metal tests and removed evidence.
- `native/metal-macos-arm64/` — foundation ABI/export, build/package verification, removed probes,
  qualification script, and native documentation.
- `modules/trace/src/main/java/.../trace/` — `LowPrecisionTraceMetadata` and removed certificate
  DTOs; focused Trace test.
- `testing/backend-conformance/` — low-precision identity allocations and tests, certificate-schema
  removal, and provider-ledger regression; `testing/integration-tests/` — existing Metal public
  Engine cases relevant to low arithmetic and raw movement.
- `ARCHITECTURE.md`, the owning backend-execution and foundational-modules contracts,
  `docs/architecture/`, `docs/backend-guide/`, `docs/developer-guide/debugging-trace.md`,
  `docs/api/`, `docs/user-guide/`, `docs/glossary.md`, ADRs 0023–0025, and Metal planning status —
  review only affected claims/links in the final dirty diff.

## Acceptance criteria

- Both low types expose only `CUSTOM_PROGRAM` for every admitted arithmetic/raw partition;
  explicit low MPSGraph forcing fails before resource creation. Existing FLOAT32 selection and
  admitted/unsupported low occurrence answers remain unchanged.
- A BFLOAT16 or FLOAT16 feed cast explicitly to FLOAT32 followed by FLOAT32 NEG retains that one
  custom candidate, plans a fixed custom cast and internal MPSGraph boundary NEG, and creates from
  the verified native package. Independent FLOAT32-only nodes have the same existing step policy;
  no generated-pointwise or anchor fusion occurs anywhere in the low-containing partition.
- A fresh native package verifies ABI 7, exactly thirteen exports, schema 19, stable wires, and
  identity 29; old identity/certificate inputs fail closed without aliases or hidden environment
  lookup. Prepared resources still close once across success, failure, and repeated/concurrent runs.
- PREPARE trace contains only selected custom route, ordered logical boundary dtypes, and profile;
  neither arithmetic nor raw-preserving programs expose an accumulator or working-type field, and
  certificate/vendor fields are absent. Model's FLOAT32 low-arithmetic guarantee remains unchanged.
  Trace remains backend-independent.
- Focused native/Metal/Trace/conformance and real-device public Engine regressions pass without
  relevant skips; affected Javadocs and documents reflect the actual diff. Independent Class C
  review of code, ABI/resource lifetime, tests, contract/ADR, and documentation has no unresolved
  findings. Neither prior P12/P13 evidence nor this brief substitutes for those final gates.

## Validation

The scoped and repository-wide validation below was reported against the integrated cutover on an
Apple-silicon host. The repository-wide build required the one-off CPU Test worker override
recorded in `Result`; the stock 512 MB CPU worker did not pass.

```bash
./native/metal-macos-arm64/build.sh
/usr/bin/codesign --force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
./gradlew :modules:trace:test :testing:backend-conformance:test :testing:architecture-tests:test
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --rerun-tasks
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*MetalIntegrationTest' --rerun-tasks
./gradlew :modules:trace:javadoc :backends:metal:javadoc
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew build
git diff --check
```

Also verify exact native exports and old-ABI/identity rejection through package verification and
focused tests; verify the Trace DTO, producer, and focused tests no longer reference
`accumulatorDtype`, `FLOAT32_ACCUMULATOR`, or `NO_ACCUMULATOR`; check changed Markdown links/anchors
and new-file whitespace after final edits.
The integration owner runs the repository build once because ABI, architecture, and several modules
change. The independent reviewer reuses passing executable evidence unless code changes or a
concrete stale-evidence risk warrants rerun.

## Documentation and review impact

Use the [documentation rules](../../../../developer-guide/documentation-rules.md) and planning
profile for this brief; the separate Class C review must inspect affected backend, trace, native,
contract, ADR, guides, glossary terminology, Javadocs, links, and final diff. Record any needed
corrections and review outcome in the same overall change. Training documentation is outside scope.

## Result

Integrated evidence on 2026-10-02: a fresh build, ad-hoc signing, local package, and package
verification passed with native ABI 7, thirteen exports, and identity 29. The focused Trace,
backend-conformance, and architecture suites passed 36, 30, and 9 tests respectively. The initial
Metal run passed 321 tests; the final full build included 323 Metal tests with 10 optional skips.
Metal public-Engine integration reported 89 tests with one skip. The repository-wide full build
passed 3,850 tests with 39 skips and zero failures or errors. These totals include unrelated work
in the shared dirty tree; they are not attributed solely to Task 0073.

The passing full build used the one-off `/private/tmp/synaptik-task0073-cpu-heap.init.gradle` to
increase only the CPU Test heap to 2 GB and set `forkEvery=1`. The stock 512 MB CPU worker ran out
of memory in the unrelated `CpuGeneratedCoverageCheckpointTest`; an unmodified default full build
is not claimed to pass. Earlier scoped evidence also passed the BFLOAT16/FLOAT16 cast-to-FLOAT32
then FLOAT32 NEG regressions, the packaged-native selector without a skip, Metal Javadoc with 76
warnings outside the repaired overloads, and a targeted check of 468 Markdown links in seven files.

Independent Class C review resolved route, Javadoc, and trace wording. The final narrow rereview
returned `APPROVE` with zero unresolved findings; it checked ADR 0023 provider-evidence authority
and the glossary observer `Error` wording against source and tests. No Task-0073 follow-up remains.
