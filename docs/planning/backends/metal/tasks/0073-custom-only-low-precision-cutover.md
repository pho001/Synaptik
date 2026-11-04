# Task 0073: Custom-Only [Metal](../../../../glossary.md#term-metal) Low-[Precision](../../../../glossary.md#term-precision) Cutover

## Reader context

A whole-partition route is not an individual native step. A BFLOAT16/FLOAT16-containing partition is fixed to `CUSTOM_PROGRAM`; a separately typed FLOAT32 step can still use an approved internal MPSGraph boundary. That neither lets MPSGraph consume low-valued data nor substitutes FLOAT16 for BFLOAT16.

Metal preparation owns routing, the executable resource owns native lifetime, and trace reports logical boundary types rather than an inferred accumulator policy. Removing the old certificate path did not remove Model working-precision rules. Test totals and the heap override below retain their stated shared-tree limits; they are not a fresh stock-build claim.

- [Current low-containing partition route](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Explains whole route versus internal FLOAT32 steps.
- [Current shared original-carrier implementation](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains) — Locates later source evolution without reviving certificates.

## Status

Complete — the [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) (16-bit [floating-point](../../../../glossary.md#term-floating-point)) cutover, integrated [validation](../../../../glossary.md#term-validation), and final
independent Class C rereview are recorded below.

## Change class

Class C — fixed [backend route](../../../../glossary.md#backend-route) and [Prepare](../../../../glossary.md#prepare) behavior, native [application binary interface](../../../../glossary.md#term-application-binary-interface-abi) ([ABI](../../../../glossary.md#application-binary-interface-and-native-bridge)),
prepared-[resource lifetime](../../../../glossary.md#ownership-borrowing-and-lease), [Trace](../../../../glossary.md#trace) diagnostic transfer object ([DTO](../../../../glossary.md#data-transfer-object--dto)), and cross-[module](../../../../glossary.md#term-module) [conformance](../../../../glossary.md#term-conformance)
change together.

## Goal

Make every [Metal](../../../../glossary.md#term-metal) [partition](../../../../glossary.md#partition) containing [BFLOAT16](../../../../glossary.md#term-bfloat16) or [FLOAT16](../../../../glossary.md#term-float16) use one fixed `CUSTOM_PROGRAM` candidate,
including exact raw-preserving movement; remove the unused machine-bound [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) and vendor
[qualification](../../../../glossary.md#term-qualification) path without changing admitted capability or numerical semantics.

## [Scope](../../../../glossary.md#term-scope)

- Retain the existing custom low-[precision](../../../../glossary.md#term-precision) arithmetic and exact raw-word `RESHAPE`, simple
  `PERMUTE`, materializing [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), [`SLICE`](../../../../glossary.md#slice), `CONCAT`, and `TILE` execution. Reject low-precision
  MPSGraph partition selection in Java and native code. Low-valued operations use custom steps;
  any FLOAT32-only node may use the existing internal MPSGraph boundary policy in the same<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [partition glossary](../../../../glossary.md#partition); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [FLOAT32 glossary](../../../../glossary.md#term-float32); [node glossary](../../../../glossary.md#node))</span>
  `CUSTOM_PROGRAM` partition, whether independent or after an explicit low-to-FLOAT32 cast.
  Never substitute BFLOAT16 with
  FLOAT16 or fuse generated-pointwise/anchor steps anywhere in that partition. Leave eligible
  FLOAT32 routes unchanged.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Remove certificate DTOs/store/evidence, environment identity and export, vendor probes, and
  qualification scripts. Advance the native ABI from 6 to 7 with exactly thirteen exports and
  coordinated Metal workload, exact-policy, candidate, compatibility, route, and codec identities
  from 28 to 29. Keep program schema 19 and existing wire allocations.<span data-glossary-navigation="true"> ([certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [identity glossary](../../../../glossary.md#term-identity-meanings); [qualification glossary](../../../../glossary.md#term-qualification); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [Metal glossary](../../../../glossary.md#term-metal); [codec glossary](../../../../glossary.md#term-codec); [schema glossary](../../../../glossary.md#term-schema))</span>
- Emit only selected-route, ordered feed-then-target logical dtype, and profile facts in
  low-precision PREPARE trace; remove <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [dtype glossary](../../../../glossary.md#data-type--datatype); [PREPARE glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace))</span>`accumulatorDtype` and its constants rather than replacing
  them with another working-type field. Retain authenticated cold selection, exact resource
  ownership, fail-closed identities, and session-scoped tuning.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [ownership glossary](../../../../glossary.md#term-ownership); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Reconcile focused Metal/native, Trace, backend-conformance, explanatory documentation, Javadocs,
  ADR 0025, and planning evidence against the final implementation diff.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [planning glossary](../../../../glossary.md#term-planning))</span>

## Non-goals

No new [operation](../../../../glossary.md#operation) or [gradient](../../../../glossary.md#term-gradient) domain, [Model](../../../../glossary.md#term-model) numerical rule, [FLOAT32](../../../../glossary.md#term-float32) route change, [schema](../../../../glossary.md#term-schema)/wire
renumbering, [runtime](../../../../glossary.md#term-runtime) fallback/retry, general [cache](../../../../glossary.md#term-cache), public [release](../../../../glossary.md#term-release), Developer ID signing, or
[notarization](../../../../glossary.md#term-notarization). The already-dirty `extensions/training` cleanup and its tests are a separate change;
Task 0073 neither owns nor validates them as a cutover result.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants): Model
  floors, fixed custom low route, ABI/identity, trace, lifecycle, and unchanged FLOAT32 policy.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [identity glossary](../../../../glossary.md#term-identity-meanings); [trace glossary](../../../../glossary.md#trace); [lifecycle glossary](../../../../glossary.md#lifecycle); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- [Backend execution — Low-precision capability evidence](../../../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [precision glossary](../../../../glossary.md#term-precision))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>:
  capability, candidate, native ABI, and resource boundaries.
- [Foundational modules — `modules/trace`](../../../../architecture/contracts/foundational-modules.md#modulestrace)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  and [Model low-precision contract and active ACCELERATOR extension](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract)<span data-glossary-navigation="true"> ([ACCELERATOR glossary](../../../../glossary.md#term-accelerator))</span>:
  typed diagnostics and unchanged allowed results.
- [Runtime/Prepare/Engine — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>
  and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle):
  cold route and prepared-resource ownership.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership))</span>
- [ADR 0025](../../../../design/decisions/0025-custom-only-metal-low-precision.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span> records the
  decision; it does not supersede the authoritative contracts. If they conflict, stop and resolve
  the contract before changing implementation.

## Dependencies and integration

- Depends on: completed Metal 0071 and Trace 0003; active Model low-precision and schema-19
  contracts; accepted ADR 0025. Metal 0053 remains independently Blocked.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Trace glossary](../../../../glossary.md#trace); [Model glossary](../../../../glossary.md#term-model); [precision glossary](../../../../glossary.md#term-precision); [schema glossary](../../../../glossary.md#term-schema); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Frontier verification: user authorized documenting the already-running uncommitted cutover as a
  special Metal frontier at `8278a34a0395fa67ee0f673107d189e9d40ad970`. This is not a new
  [`Ready`](../../../../glossary.md#term-ready) launch or permission for concurrent writes; prerequisite surfaces are present in the
  current tree, and completion gates remain open.
- Conflicts with: concurrent edits to Metal routes, native ABI/package, Trace low-precision DTOs,
  conformance identity, or their shared contracts/docs; any resumed Metal 0053 production work.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [conformance glossary](../../../../glossary.md#term-conformance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None.
- Common base revision: N/A — one already-dirty worktree; initial HEAD is the revision above.
- Integration order: stabilize scoped implementation and documentation, run the final matrix,
  obtain independent targeted Class C review, then remediate/revalidate any findings.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Integration validation: the native/package, focused modules, real-device integration,
  architecture, Javadoc, documentation, and repository-wide gates below, once on the final diff.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [modules glossary](../../../../glossary.md#term-module); [device glossary](../../../../glossary.md#term-device); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration owner: Metal 0073 coordinator; serialize master-plan, roadmap, root
  contract, and explanatory-document reconciliation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

## Files and symbols

- `backends/metal/src/main/java/.../metal/` — `MetalNegPartitionPreparer`,
  `MetalNegRouteCandidateGenerator`, `MetalNegPreparationPlan`, `MetalNegPartitionFinalizer`,
  `MetalProgramExecutableResource`, `MetalTraceProducer`, `MetalNativeApi`, context/tuning/codec,
  and removed certificate/MPSGraph-resource classes; focused Metal tests and removed evidence.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [Metal glossary](../../../../glossary.md#term-metal))</span>
- `native/metal-macos-arm64/` — foundation ABI/export, build/package verification, removed probes,
  qualification script, and native documentation.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- `modules/trace/src/main/java/.../trace/` — `LowPrecisionTraceMetadata` and removed certificate
  DTOs; focused Trace test.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- `testing/backend-conformance/` — low-precision identity allocations and tests, certificate-schema
  removal, and provider-ledger regression; <span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema); [regression glossary](../../../../glossary.md#term-regression))</span>`testing/integration-tests/` — existing Metal public
  Engine cases relevant to low arithmetic and raw movement.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- `ARCHITECTURE.md`, the owning [backend](../../../../glossary.md#backend)-execution and foundational-[modules](../../../../glossary.md#term-module) contracts,
  `docs/architecture/`, `docs/backend-guide/`, `docs/developer-guide/debugging-trace.md`,
  `docs/api/`, `docs/user-guide/`, `docs/glossary.md`, ADRs 0023–0025, and Metal planning status —
  review only affected claims/links in the final dirty diff.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Acceptance criteria

- Both low types expose only `CUSTOM_PROGRAM` for every admitted arithmetic/raw partition;
  explicit low MPSGraph forcing fails before resource creation. Existing FLOAT32 selection and
  admitted/unsupported low occurrence answers remain unchanged.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [FLOAT32 glossary](../../../../glossary.md#term-float32); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- A BFLOAT16 or FLOAT16 feed cast explicitly to FLOAT32 followed by FLOAT32 NEG retains that one
  custom candidate, plans a fixed custom cast and internal MPSGraph boundary NEG, and creates from
  the verified native package. Independent FLOAT32-only nodes have the same existing step policy;
  no generated-pointwise or anchor fusion occurs anywhere in the low-containing partition.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [nodes glossary](../../../../glossary.md#node); [pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- A fresh native package verifies ABI 7, exactly thirteen exports, schema 19, stable wires, and
  identity 29; old identity/certificate inputs fail closed without aliases or hidden environment
  lookup. Prepared resources still close once across success, failure, and repeated/concurrent runs.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- PREPARE trace contains only selected custom route, ordered logical boundary dtypes, and profile;
  neither arithmetic nor raw-preserving programs expose an accumulator or working-type field, and
  certificate/vendor fields are absent. Model's FLOAT32 low-arithmetic guarantee remains unchanged.
  Trace remains backend-independent.<span data-glossary-navigation="true"> ([PREPARE glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace); [dtypes glossary](../../../../glossary.md#data-type--datatype); [accumulator glossary](../../../../glossary.md#accumulator-type); [Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend))</span>
- Focused native/Metal/Trace/conformance and real-device public Engine regressions pass without
  relevant skips; affected Javadocs and documents reflect the actual diff. Independent Class C
  review of code, ABI/resource lifetime, tests, contract/ADR, and documentation has no unresolved
  findings. Neither prior P12/P13 evidence nor this brief substitutes for those final gates.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [conformance glossary](../../../../glossary.md#term-conformance); [device glossary](../../../../glossary.md#term-device); [Engine glossary](../../../../glossary.md#term-engine); [resource lifetime glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## [Validation](../../../../glossary.md#term-validation)

The scoped and repository-wide validation below was reported against the integrated cutover on an
Apple-silicon [host](../../../../glossary.md#term-host). The repository-wide build required the one-off [CPU](../../../../glossary.md#term-cpu) Test worker override
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

Also verify exact native exports and old-[ABI](../../../../glossary.md#term-application-binary-interface-abi)/[identity](../../../../glossary.md#term-identity-meanings) rejection through package verification and
focused tests; verify the [Trace](../../../../glossary.md#trace) [DTO](../../../../glossary.md#data-transfer-object--dto), [producer](../../../../glossary.md#term-producer), and focused tests no longer reference
`accumulatorDtype`, `FLOAT32_ACCUMULATOR`, or `NO_ACCUMULATOR`; check changed Markdown links/anchors
and new-file whitespace after final edits.
The integration [owner](../../../../glossary.md#term-owner) runs the repository build once because ABI, architecture, and several [modules](../../../../glossary.md#term-module)
change. The independent reviewer reuses passing executable evidence unless code changes or a
concrete stale-evidence risk warrants rerun.

## Documentation and review impact

Use the [documentation rules](../../../../developer-guide/documentation-rules.md) and [planning](../../../../glossary.md#term-planning)
profile for this brief; the separate Class C review must inspect affected [backend](../../../../glossary.md#backend), [trace](../../../../glossary.md#trace), native,
contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), guides, glossary terminology, Javadocs, links, and final diff. Record any needed
corrections and review outcome in the same overall change. Training documentation is outside [scope](../../../../glossary.md#term-scope).

## Result

Integrated evidence on 2026-10-02: a fresh build, [ad-hoc signing](../../../../glossary.md#term-ad-hoc-signing), local package, and package
verification passed with native [ABI](../../../../glossary.md#term-application-binary-interface-abi) 7, thirteen exports, and [identity](../../../../glossary.md#term-identity-meanings) 29. The focused [Trace](../../../../glossary.md#trace),
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and architecture suites passed 36, 30, and 9 tests respectively. The initial
[Metal](../../../../glossary.md#term-metal) [run](../../../../glossary.md#term-run) passed 321 tests; the final full build included 323 Metal tests with 10 optional skips.
Metal public-[Engine](../../../../glossary.md#term-engine) integration reported 89 tests with one skip. The repository-wide full build
passed 3,850 tests with 39 skips and zero failures or errors. These totals include unrelated work
in the shared dirty tree; they are not attributed solely to Task 0073.

The passing full build used the one-off `/private/tmp/synaptik-task0073-cpu-heap.init.gradle` to
increase only the [CPU](../../../../glossary.md#term-cpu) Test [heap](../../../../glossary.md#term-heap) to 2 GB and set `forkEvery=1`. The stock 512 MB CPU worker ran [out
of memory](../../../../glossary.md#term-out-of-memory-oom) in the unrelated `CpuGeneratedCoverageCheckpointTest`; an unmodified default full build
is not claimed to pass. Earlier scoped evidence also passed the [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) cast-to-[FLOAT32](../../../../glossary.md#term-float32)
then FLOAT32 NEG regressions, the packaged-native selector without a skip, Metal [Javadoc](../../../../glossary.md#term-javadoc) with 76
warnings outside the repaired overloads, and a targeted check of 468 Markdown links in seven files.

Independent Class C review resolved route, Javadoc, and trace wording. The final narrow rereview
returned `APPROVE` with zero unresolved findings; it checked [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0023 provider-evidence authority
and the glossary [observer](../../../../glossary.md#term-observer) `Error` wording against source and tests. No Task-0073 follow-up remains.
