# Task CPU0035: General-scan emission cleanup and specialized oracle

## Status

Complete — main accepted bounded cleanup, independent final review, checkpoint and the single
bounded campaign on 2026-10-23. Performance parity was not achieved; no successor is launched.

## Change class

Class C — generated hot-loop/dataflow and staged compatibility identity, without a semantic,
capability, public API, resource, lifecycle or dependency change. One clean implementation context
and one independent final targeted code/documentation review; no additional planning/review layer.

## Goal

Remove proved redundant staged scan bytecode and compare the actual generated kernel with an
equally specialized clean Java reference: FP64 `next=emit=h+xStep*w`, static B=4/H=8, runtime long T,
complete final-carry and stacked-emission outputs. Keep the existing generic oracle and history.

## Scope and non-goals

- In existing generation-only owners, remove unused factor locals/stores, redundant zero address
  arithmetic and alias/local copies where a dominating equivalent binding is proved. Forward
  shaped traversal uses its long iteration directly, without copying it into a second time local.
- Preserve exact operation order, represented sites, empty/reverse/multi-carry/view lifetimes,
  scalar-initializer→dense transitions and all unproved safe fallbacks. No new production type.
- No public API/capability/numerical/resource/lifecycle/dependency change, time-to-int or baked-T
  specialization, generic optimizer, new abstraction, while/if expansion, low-type, vector,
  parallel, OpenBLAS, padding, persistence or NN workflow expansion. No GSD or legacy copying.
- CPU0034 Status/Result, metrics and artifacts remain untouched; no inherited timing qualification.

## Contracts

Read `AGENTS.md`, this brief, affected source/focused tests, and only these root/scoped headings:

- [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) (CPU).
- Backend execution: [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes),
  [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
  [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity).
- Scan owner: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  especially [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)
  and [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).

No architecture policy change is authorized. Stop for missing/ambiguous contracts, an unexpected
architecture decision or scope expansion; do not invent a new rule or ADR to continue.

## Dependencies and integration

- Depends on: CPU0034 Complete, including its accepted CPU0033/low-level join and Compiler0012 baseline.
- Conflicts with: any scan emitter/IR/staged identity, oracle/harness or shared-document writer;
  inactive. Checkpoint/NN and unrelated Draft successors stay held.
- Parallel group: None; serial implementation/review closed; no active executor or concurrent writer.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all 266
  accepted dirty paths, including CPU0034; these are not merged predecessors. Archive:
  `/private/tmp/synaptik-cpu0035-baseline.ktXxoA/accepted`, sibling `snapshot.json`.
- Integration order: main launch recheck → clean implementation → independent final Class C
  code/docs review/corrections → main targeted checkpoint and authorized bounded campaign → acceptance.
- Integration validation: main's exact conformance/Engine commands below, one bounded evidence
  campaign after review/authorization, combined diff/document checks; no root/native/full matrix.
- Shared-document integration owner: main; independent serial reviewer may exclusively finalize
  explanatory/Javadoc and descriptive current-v3 facts; main owns planning/closure.

Historical Ready verification (2026-10-23): CPU0034 Status/Result and master/roadmap agreed Complete,
including reviews/gates; user explicitly authorized CPU0035 from frontier None. HEAD/origin matched,
index was empty, all 266 snapshot hashes matched and conflicts were inactive. Main verified launch;
implementation/review/gates are now closed on frozen source, and CPU frontier returns to None.

## Files and symbols

Java paths: `modules/backend/cpu/src/{main,test}/java/io/github/pho001/synaptik/backend/cpu/`.

- Main `internal/codegen/emit/CpuGeneralScanEmitter.java`: `Shaped` locals/bindings,
  `emit`, `load`, `copyDense`, `offset`, `frameOffset`, coordinate loops and post-step proof.
- Main same directory `CpuGeneralScanReductionEmitter.java`, `CpuGeneralScanMatmulEmitter.java`:
  consumers of shared coordinates/addresses/factors, especially MATMUL `stride`.
- Main `internal/ir/CpuGeneralScanIr.java`: `bakedSpecializationIdentity`, staged Javadoc;
  `internal/cache/CpuKernelSpecialization.java`: both compatibility/class-identity streams.
- Test `internal/codegen/emit/CpuGeneralScanGeneratedKernelTest.java`,
  `CpuGeneralScanCleanJavaOracle.java`, `CpuGeneralScanTimingTest.java`: structure, matched oracle,
  existing bounded timing seam; test `internal/cache/CpuKernelSpecializationTest.java` and
  `CpuGeneratedKernelArtifactStoreTest.java`: identity/rejection/reuse. Focused existing lowering,
  preparation, execution and capability tests; new stale-v2 resources in existing emit test resources.
- Review current [CPU scan explanation](../../../../backend-guide/cpu-backend.md#bounded-float32float64-general-scan)
  and [general-scan explanation](../../../../architecture/general-differentiable-scan.md), plus only
  current identity facts in root Core invariants and owning backend contract (not broad rule edits).

## Acceptance criteria

1. Generated structure proves removed locals/stores/zero-adds/copies were redundant; forward
   time index aliases long iteration. Static B/H induction and widening stay valid, and runtime
   LONG T/index/addresses remain long. No hidden allocation, boxing, reflection, dispatch or helper
   calls; generated algorithm/hot-loop/dataflow matches the clean specialized Java oracle.
2. Allocate/initialize factors only where consumed. Baseline MATMUL `stride` read
   `state.factors[origin]` for ANY scalar formal: proved fixed scalar operands need effective stride
   zero, while evolving scalar→dense carries keep initialized dynamic factors through load/copy
   AND MATMUL consumers. No uninitialized local or dense read from a physical one-cell initializer.
3. F32/F64 focused execution/structure covers T=0/1/multiple, empty axes/nonempty SUM from empty
   steps, forward/reverse, broadcast/SUM/MATMUL, fixed scalar operands and scalar→dense carries,
   tied/multiple carries, previous-state saves, forwarded/transposed/chained next views and final
   publication. Preserve old/new versions, raw empty final carry, demanded histories and fallbacks.
4. Advance only the sole baked `scan-shaped-v2` and `scan-matmul-v2` payloads to v3 in BOTH existing
   specialization streams when affected (shared Shaped cleanup is expected to affect both).
   Keep global envelope/family 71, geometry-free canonical v1 projections, unrelated/pointwise/fixed
   keys and bytes unchanged. No shadow version payload. Preserve historical v1/v2 provenance and
   add current v3 assertions rather than relabeling old goldens as new evidence.
5. Reject actual stale v1 AND v2 classes at definition and persisted compatibility seams, including
   valid/checksummed same-current-path artifacts; preserve old-key entries and compatible runtime-T
   reuse. Keep existing v1 fixtures. Verify source geometry/spec/identity before freezing actual v2
   fixtures from `/private/tmp/synaptik-cpu0034-FgGCiI/measurement-v2/measured-fork-1/production.class`
   and `/private/tmp/synaptik-cpu0034-FgGCiI/structure/matmul-0-0.class`; do not fake old bytes by
   patching new classes. Keep fixture hashes/identity provenance, including differing case geometry.
6. Add a test-only B4/H8 FP64 Java oracle alongside unchanged generic `broadcastForwardFloat64`.
   Runtime T is long (prefer the same entry ABI or clearly equivalent borrowed geometry), never
   compile-time 8/int; B/H are static. Same native access/layout, multiply-then-add order, emission
   carry backing and complete final copy, no shortcut output or primitive-array fast path. Check
   full outputs, including T=0 initial-carry publication, and class/identity/oracle disassembly.
7. Required independent final review/docs, worker validation, main checkpoint and the single bounded
   campaign are recorded with exact outcomes/limits before Complete; preserve all unrelated edits.

## Validation

Worker: focused regression/structure during development; no opt-in timing during ordinary tests.
The existing evidenceRoot property retains classes/identities/javap without enabling performance:

```bash
./gradlew -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0035-evidence.6sqRuQ :backends:cpu:test --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuGeneralScanLoweringTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuGeneralScanPreparationTest' --tests '*CpuGeneralScanCapabilityTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew :backends:cpu:test :backends:cpu:javadoc
git diff --check
```

Final module/Javadoc passed once on stabilized source; reviewer reused focused tests. Main checkpoint:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
```

Completed campaign used the existing bounded main/harness and main's fresh guarded launcher/analyzer
at `/private/tmp/synaptik-cpu0035-evidence.6sqRuQ`, targeting the new matched-ABI
`CpuGeneralScanCleanJavaOracle::broadcastForwardFloat64B4H8`. Original CPU0034 artifacts/generic
oracle remain unchanged. Main ran the following once after review and source-matched gates:

```bash
CPU0035_REVIEW_APPROVED=yes zsh /private/tmp/synaptik-cpu0035-evidence.6sqRuQ/run-bounded.zsh
```

Protocol: 3 sequential fresh JVMs, 100 warmup/300 measured alternating pairs each, 10000 calls/side,
T=8 runtime geometry/B=4/H=8; actual generated class versus equally specialized oracle. Shared native
inputs and distinct equally shared native outputs, zero workspace. Retain all primary pairs, full
output-byte checks, class/identity/javap, JFR/LogCompilation and clock/block correlation; distinguish
cold generation/definition/first entry from warmed costs. No jcmd, forced compilation, pilot, retry,
extra matrix, Engine/binding subtraction or universal parity/causal-v2-speedup claim.

## Follow-up

Required separate performance follow-up: a separately authorized narrow loop-canonicalization
experiment in the existing emitter, starting with javac-style loop control; keep final-tail sharing
separate. The remaining ~3× gap has no proven cause. This closure creates or launches no successor,
runner or measurement and grants no new authorization; Checkpoint/NN and other gates stay held.

## Documentation and review impact

Independent final Class C review finalized affected emitter/oracle/timing Javadocs, current backend/
general-scan explanations and descriptive v3 facts in root/owning backend contract. Documentation
rules and applicable general/API-Javadoc/backend/architecture profiles governed the targeted pass;
glossary/link/anchor/diff review found no new terminology, policy, dependency or ADR requirement.

## Result

Status: Complete
Main accepted 2026-10-23 after independent final APPROVE, zero findings.
Changed three emitters, IR/cache, generated/oracle/timing/cache tests, actual v2 fixtures/provenance,
four targeted documents and these three planning files; no new production type/API/capability/policy.
Generated code bytes/locals 377/77 → 297/61; specialized oracle 215/28. Runtime LONG/static B/H and
conditional-zero empty/singleton/scalar/MATMUL fallbacks remain; generic oracle unchanged.
Only staged payloads advance to v3 in both streams; schema/family 71, canonical v1 and unrelated
identities/bytes stay frozen. Actual old v2 fixture equality and v1/v2 rejection/reuse are proved.
Commands above passed: 114 focused; final CPU/Javadoc 5m27s, 216 suites, 1211 reported/1173 executed/
38 skipped, zero failures/errors; 94 preexisting Javadoc warnings. Conformance 7/7, Engine 17/17, no skips.
Independent docs/Javadocs finalized; no new glossary terms/dependency/ADR. Source remained frozen.
Evidence/changed-path list: [final-validation.json](/private/tmp/synaptik-cpu0035-evidence.6sqRuQ/final-validation.json).
Class SHA `870c954f4d83597980d5ea9d5c704a89c9cd8a459fbf35ca85cadc14ce01014e`;
key `295312b55b6f922dc77232e71af1d6dc22e49de755266b8bc15bd542d23556af`.
One campaign: 3 fresh sequential JVMs, 100 warmup/300 measured pairs each, 10000 calls/side;
all 900 primary pairs/2400 blocks independently audited, all output bytes equal before/after.
Primary per-JVM medians (generated ns / oracle ns; ratio of medians):
1: 386.875 / 125.125; 3.091908091908092. 2: 379.8083 / 124.37295; 3.053785409126341.
3: 385.44795 / 125.67085; 3.0671229644742595. Both normal-entry C2, zero target deopts/inlines.
Optional ±100ms guard qualifies 281/fork; last 19 lack post-window tail only; ALL 300 remain primary.
Parity NOT achieved (~3× gap). Bottom-tested generated/top-tested javac loops and deeper generated
C2 loop tree are observed, not proven vectorization/cause/assembly. Reviewer accepted bounded cleanup
without a timing threshold; separate performance follow-up above is required, not launched. CPU0034 generic-oracle
timings remain historical, not causal v2→v3 speedup evidence. No commits/pushes or successor authorization.
