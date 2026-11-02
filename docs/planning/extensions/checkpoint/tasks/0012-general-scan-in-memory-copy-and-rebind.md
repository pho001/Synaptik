# Task Checkpoint0012: General-scan in-memory copy and rebind

## Status

Complete — main accepted implementation, independent review and scoped validation on 2026-10-23,
with the unrelated native-environment root exception recorded below; unrestricted root is NOT green.
Historical Ready/launch verification: approved NN0030, Complete Compiler0012/Checkpoint0008–0011
and low-level join, accepted 274-path base/CPU0036, matching HEAD/origin, empty index and inactive
conflicts verified before the clean executor launched. Checkpoint/NN/Compiler frontiers: None.

## Change class

Class C — scoped-body/ownership and cross-consumer boundary; clean executor and independent clean code/docs review mandatory.

## Goal and scope

Make existing `GraphCheckpoint.copy(Source)` and `rebind(Source,List<Tensor>)` preserve general
SCAN body attachments while normalizing outer IDs. Body IDs remain independently scoped.
Compiler0012 already owns complete cold reconstruction; Checkpoint must not clone bodies or
maintain a body opcode inventory/reconstruction table. Preserve applicable Model dtypes for exact
Compiler-reconstructible signatures, independently of CPU support; not blanket all-PURE admission.

Keep disk save/load closed. Add only minimal compatibility guards in existing NN owners so
Checkpoint widening cannot accidentally open general-scan acquisition/compute/train. Reject through
ordinary public Model provenance and `ScanKind.SCAN` before copy/setup; no new Compiler edge.

## Non-goals

No new production type/package/public API, dependency, wire/version, architecture policy or ADR;
no NN qualification/workflow widening, fixed-scan BPTT, padding, backend/numerical/runtime change,
T-unrolling, per-step/per-compute/train source reconstruction, benchmark or native/device opt-in.
Do not reopen Compiler0012 or require CPU0036 performance qualification for cold copying.
STOP if a new type, policy, missing contract or architecture decision becomes necessary.

## Contracts

Read AGENTS.md, affected source/tests and these exact headings via the root scope index, not predecessor histories.

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core lifecycle`,
  `Core invariants`, `Scope-indexed normative contracts`, `Module-ownership routing`, `Dependency rules`
  (actual Checkpoint/NN edges), `Testing requirements`.
- [Extensions owner](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint):
  `extensions/checkpoint`, `Generic graph source and values`, `Canonical graph-model version 1`,
  `Canonical graph-model version 2`, `Canonical graph-model version 3`; also
  [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)
  for the independently closed NN acquisition gate and unchanged fixed-scan training rejection.
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  `General differentiable scan`, `Source body, attachments and scoped identities`,
  `Canonical residual ports and demand`, `Traversal, empty domains and declared qualification`.
- [Compiler owner](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
  `Compile lifecycle`, `Bounded general-scan cold reconstruction` — exact flat/body signatures,
  complete preflight and sole reconstruction ownership, not backend eligibility.

## Dependencies and integration

- Depends on: approved NN0030 decision; Complete Compiler0012 and Checkpoint0008–0011;
  accepted Model/Planning/Compiler/Prepare/CPU low-level scan join. CPU0036 is accepted baseline,
  not an additional performance gate. Required predecessor reviews/validation are recorded Complete.
- Conflicts with: Checkpoint graph normalization/wire/API, Compiler cold reconstruction, scan
  contracts, NN acquisition/lifecycle and shared-document writers; inactive and serialized.
- Parallel group: None; source and documentation writers serialize. Read-only review may overlap tests.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  274 accepted dirty paths, including CPU0036 Complete; preserve all unrelated work.
  Baseline: `/private/tmp/synaptik-cp0012-baseline.P4pqr7/accepted` and `snapshot.json`;
  evidence root: `/private/tmp/synaptik-cp0012-evidence.bMKlxo`.
- Integration order: clean implementation → independent review/docs finalization → final affected
  module gates → root attempts with recorded environment exception → main scoped acceptance.
- Integration validation: combined root command below, including architecture and Engine consumers;
  no successful standalone filtered main suites first, and no duplicate module runs.
- Shared-document integration owner: main; exclusively delegates targeted current-fact/Javadoc
  finalization to the independent reviewer. Main alone owns master/roadmap/result integration.

## Files and symbols

- Checkpoint production directory: `modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph`.
  `GraphSource.java`: `capture` preserves attachments with five-component `CompiledNode`,
  remapping only outer ports.
- Same directory: `GraphArtifactCodec.java` explicit independent wire eligibility; `GraphCheckpoint.java`
  affected facade Javadoc. `GraphValueStorage.java` read-only ownership/storage reference.
- Same package under `modules/checkpoint/src/test/java`: `GraphCheckpointTest.java`,
  `GraphArtifactCodecTest.java`, `GraphArtifactV3Test.java`; new focused
  `GraphCheckpointGeneralScanTest.java` in this same test package.
- NN production directory: `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module`.
  `Model.java`: `graph` before `GraphCheckpoint.copy`;
  `OwnedModelSource.java`: `acquireDefinition`, `acquireComposition` and provenance inventory.
  Minimal early guards only; adoption after copy is too late. Do not broaden fixed-only
  `containsScan`/`rejectScanTraining`. `ModelExecutionState.java` is read-only lifecycle reference.
- Same package under `extensions/nn/src/test/java`: `OwnedModelSourceTest.java` and focused existing
  lifecycle/training/runtime-length tests as needed: preserve every affected acquisition route.
- Integration package `io.github.pho001.synaptik.testing.integration` under
  `testing/integration-tests/src/test/java`: `GraphCheckpointEngineIntegrationTest.java`,
  bounded copy/rebind-to-public-Engine regression, not NN workflow qualification.
- Read-only Model `CompiledNode`, `CompiledScanBody`, `ScanBody`/`Scan`; Compiler
  `GraphCompilationPort`, `SourceGraphReconstruction` and its focused tests. Do not edit Compiler.

## Acceptance criteria

- Capture once; preserve full descriptors, occurrences, sharing, declared unused inputs/formals,
  repeated actual-role positions, role aliases, canonical final-carry/emission/residual ports and
  root order. Outer NodeId/ValueId normalization never rewrites/resolves local body IDs.
- Compiler reconstructs fresh formals/body per occurrence and every canonical port once. Reusing
  an immutable template never merges scopes or retains original body/formals in copied provenance.
  Both traversal flags, T=0 and constituent empty-domain semantics survive; no time unrolling.
- Copy owns detached heap state once per named identity, preserves raw bits/roles/aliases and
  storage-free prototypes, outlives original native state and mutates no caller binding/storage.
  Rebind is metadata-only, retains named state, permits repeated/produced exact-descriptor inputs
  with opaque upstream provenance, and returns immutable ordinary roots under existing guards.
- Test outer remapping with coincident independent body IDs, multiple carries/emissions/residuals,
  tied roles, multiple occurrences, unused formals, detached ownership and template isolation.
  Include valid low dtype and CPU-unsupported reconstructible bodies (e.g. low TANH/F32 RELU);
  reject unsupported exact body signatures without importing backend predicates or all-PURE support.
- Full unsupported-body preflight precedes host access/copy, including late failures and T=0;
  retain existing source/API, geometry, alias, storage, null/replacement and failure-order guards.
- `GraphArtifactCodec` independently rejects general bodies and malformed attachments before
  host admission/import/copy or file effects. Keep explicit opcodes 1..22 and byte-compatible
  v1/v2/v3 selection/readers. No general-body save/load via widened Compiler admission.
- Split obsolete `generalScanConsumersRejectMissingNormalizedAttachmentBeforeAnyHostReadOrFileEffect`:
  positive copy/metadata-only rebind tests plus independent SAVE rejection using its real host-read
  spy. Test unreadable/absent/dead state, unchanged existing target, absent target and no staging
  files/host effects; retain positive spy control and malformed/unknown wire rejection coverage.
- NN early GENERAL rejection covers graph overloads, defineTensor/acquireDefinition and
  acquireComposition as applicable, before copy/storage/setup effects using Model-only provenance.
  No general-scan owned Model reaches compute/configure/train; fixed inference/training guard and
  static routes remain unchanged. Test final-state/residual-only roots, heads, T=0 and unreadable state.
- Existing Engine consumers still work; new supported F32/F64 copy/rebind regression matches direct
  ordinary scan results, while low/CPU-unsupported cases remain cold-only rather than execution promises.

## Validation

Recorded final module gate after independent Javadoc finalization; outcomes are in Result:

```bash
./gradlew :extensions:checkpoint:test :extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc --max-workers=2 --console=plain
```

Recorded normal root command; a remaining-gate attempt appended `-x :backends:metal:test`.
Both exited 1 on missing native configuration, not CP0012 failures; no third attempt was run.

```bash
./gradlew test javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false
```

Required consumer XML, fresh/cached outcomes and root limitations are recorded below.
No timings, benchmarking, native/device opt-in or build/dependency edits; future NN native join stays separate.
Check links/anchors, size, whitespace, baseline preservation and `git diff --check`.

## Documentation and review impact

Independent clean reviewer inspects actual stabilized diff/tests and finalizes affected Javadocs,
`docs/api/graph-checkpoint-api.md`, and targeted root/scoped current-implementation facts and
explanations under main's exclusive delegation. No architecture policy changes/ADR are expected;
stop on a new decision. Follow documentation-rules plus general/API/architecture profiles as
applicable; targeted glossary search, no new terminology expected. Source/doc writers serialize;
read-only review may overlap executor tests. Reuse successful tests unless executable code changes.
Completion summary: changes/files, validations, documentation/Javadoc/glossary, unresolved issues/follow-up;
`Status: Complete` or `Status: Incomplete` with specific follow-up.

## Result

Status: Complete — implementation/scoped acceptance only; unrestricted repository root is NOT green.

- Delivered attachment-preserving outer normalization, Compiler-owned cold reconstruction,
  independent closed v1/v2/v3 wire admission and NN early GENERAL rejection; no new API/type/edge.
- Changed 20 task paths: checkpoint.graph `GraphSource`, `GraphArtifactCodec`, `GraphCheckpoint`,
  `GraphCheckpointTest`, new `GraphCheckpointGeneralScanTest`; nn.module `Model`, `OwnedModelSource`,
  `OwnedModelSourceTest`; `GraphCheckpointEngineIntegrationTest`; `GraphCompilationPort` Javadoc;
  root, graph-checkpoint API, compiler-autograd/extensions-training contracts, module-boundaries,
  general-differentiable-scan explanation, glossary; this brief, Checkpoint master and roadmap.
- Independent clean code/docs review Complete, pass/no findings (2026-10-23 11:29:53 UTC).
  Seven targeted documents and affected Javadocs finalized; existing glossary distinctions clarified,
  no new terminology/policy/ADR. Compiler executable inventory/logic and CPU0036 stayed unchanged.
- Final module command above: 52 Checkpoint + 526 NN = 578 fresh tests, zero failures/errors/skips;
  both fresh Javadocs had zero warnings. Development evidence reused; no duplicate filtered reruns.
- Normal root command above: exit 1, 5m8s, only four unchanged `MetalExpNativeTest`/
  `MetalSigmoidNativeTest` cases failed at `Path.of(null)` with unset `SYNAPTIK_METAL_TEST_LIBRARY`,
  before library opening. Fresh CPU: 1214 reported, 1176 executed, 38 skipped, zero failures.
- Same root command plus `-x :backends:metal:test`: exit 1, 19s; 15 unchanged cases in
  `EngineScalarMetalIntegrationTest`, `EngineExpMetalIntegrationTest`, `EngineSigmoidMetalIntegrationTest`
  failed on the same missing environment before native opening. Normal unit tasks, architecture
  and conformance passed/reused; all Javadocs passed/reused, with two benchmark incubator warnings.
- Fresh required XML: `CheckpointDependencyContractTest` 1 pass; `GraphCheckpointEngineIntegrationTest`
  7 passes; architecture 25 and conformance 59 passes, no failures/skips in those suites.
- Current reports: 4762 reported = 4507 passed + 236 skipped + 19 environment failures, zero errors;
  not all fresh. Root reused module/CPU successes; 13 stale pre-relocation reports excluded.
  No third root, benchmarks, native/device qualification, unrelated source fixes or build changes.
- Evidence: `/private/tmp/synaptik-cp0012-evidence.bMKlxo/{final-validation.json,root-test-results.json}`;
  exact commands/tasks/logs and fresh/cached breakdown retained there. Source freeze unchanged;
  274-path baseline preserved outside 20 authorized task paths, HEAD=origin at base, index empty.
- No unresolved CP0012 acceptance issue. Separate follow-up: native test harness configuration/gating
  for the 19 unrelated failures; no fix/qualification here. NN workflow/native join remains separate,
  body persistence stays closed, Checkpoint/NN/Compiler frontiers None; no successor task or launch.
