# Task CPU0035: General-scan emission cleanup and specialized [oracle](../../../../glossary.md#term-test-oracle)

## Reader context — historical integration boundary

The specialized Java oracle is a test reference with matching fixed geometry and demanded output
work, not another production executor. CPU removes only proved redundant local/address work from
existing generated Scan code; required old/new carry lifetimes and saved sites still determine
storage. A shorter Class-File or fewer locals alone does not establish faster machine code. The
result below explicitly retains the roughly threefold gap against this more specialized reference,
while preserving the older generic oracle and its historical observations. Read those comparisons as
different evidence boundaries, not a causal before/after speedup.

Current reading: [Retained static Scan resources](../../../../backend-guide/cpu-backend.md#retained-static-scan-qualification-and-resources).

## Status

Complete — main accepted bounded cleanup, independent final review, [checkpoint](../../../../glossary.md#term-checkpoint) and the single
bounded campaign on 2026-10-23. Performance parity was not achieved; no successor is launched.

## Change class

Class C — generated hot-loop/[dataflow](../../../../glossary.md#term-dataflow) and staged compatibility [identity](../../../../glossary.md#term-identity-meanings), without a semantic,
capability, public [API](../../../../glossary.md#term-application-programming-interface-api), resource, [lifecycle](../../../../glossary.md#lifecycle) or [dependency](../../../../glossary.md#term-dependency) change. One clean implementation context
and one independent final targeted code/documentation review; no additional [planning](../../../../glossary.md#term-planning)/review layer.

## Goal

Remove proved redundant staged scan [bytecode](../../../../glossary.md#term-bytecode) and compare the actual generated [kernel](../../../../glossary.md#kernel) with an
equally specialized clean Java reference: [FP64](../../../../glossary.md#term-fp64) `next=emit=h+xStep*w`, static B=4/H=8, [runtime](../../../../glossary.md#term-runtime) long T,
complete final-[carry](../../../../glossary.md#term-carry) and stacked-emission outputs. Keep the existing generic [oracle](../../../../glossary.md#term-test-oracle) and history.

## [Scope](../../../../glossary.md#term-scope) and non-goals

- In existing generation-only owners, remove unused factor locals/stores, redundant zero address
  arithmetic and alias/local copies where a dominating equivalent binding is proved. Forward
  shaped traversal uses its long iteration directly, without copying it into a second time local.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [binding glossary](../../../../glossary.md#term-binding))</span>
- Preserve exact operation order, represented sites, empty/reverse/multi-carry/view lifetimes,
  scalar-initializer→dense transitions and all unproved safe fallbacks. No new production type.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [carry glossary](../../../../glossary.md#term-carry); [view glossary](../../../../glossary.md#term-view-meanings); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- No public API/capability/numerical/resource/lifecycle/dependency change, time-to-int or baked-T
  specialization, generic optimizer, new abstraction, while/if expansion, low-type, vector,
  parallel, OpenBLAS, padding, persistence or NN workflow expansion. No GSD or legacy copying.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency); [specialization glossary](../../../../glossary.md#term-specialization); [optimizer glossary](../../../../glossary.md#term-optimizer); [vector glossary](../../../../glossary.md#term-vector); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [padding glossary](../../../../glossary.md#padding); [persistence glossary](../../../../glossary.md#term-persistence); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- CPU0034 Status/Result, [metrics](../../../../glossary.md#machine-learning-and-hyperparameters) and [artifacts](../../../../glossary.md#term-artifact) remain untouched; no inherited timing [qualification](../../../../glossary.md#term-qualification).

## Contracts

Read `AGENTS.md`, this brief, affected source/focused tests, and only these root/scoped headings:

- [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> ([CPU](../../../../glossary.md#term-cpu)).
- [Backend](../../../../glossary.md#backend) execution: [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes),
  [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
  [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>.
- Scan [owner](../../../../glossary.md#term-owner): [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  especially [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
  and [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).

No architecture policy change is authorized. Stop for missing/ambiguous contracts, an unexpected
architecture decision or [scope](../../../../glossary.md#term-scope) expansion; do not invent a new rule or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) to continue.

## Dependencies and integration

- Depends on: CPU0034 Complete, including its accepted CPU0033/low-level join and Compiler0012 baseline.
- Conflicts with: any scan emitter/IR/staged identity, oracle/harness or shared-document writer;
  inactive. Checkpoint/NN and unrelated Draft successors stay held.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Parallel group: None; serial implementation/review closed; no active executor or concurrent writer.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all 266
  accepted dirty paths, including CPU0034; these are not merged predecessors. Archive:
  `/private/tmp/synaptik-cpu0035-baseline.ktXxoA/accepted`, sibling `snapshot.json`.
- Integration order: main launch recheck → clean implementation → independent final Class C
  code/docs review/corrections → main targeted checkpoint and authorized bounded campaign → acceptance.
- Integration validation: main's exact conformance/Engine commands below, one bounded evidence
  campaign after review/authorization, combined diff/document checks; no root/native/full matrix.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Shared-document integration owner: main; independent serial reviewer may exclusively finalize
  explanatory/Javadoc and descriptive current-v3 facts; main owns planning/closure.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

Historical [Ready](../../../../glossary.md#term-ready) verification (2026-10-23): CPU0034 Status/Result and master/roadmap agreed Complete,
including reviews/gates; user explicitly authorized CPU0035 from frontier None. HEAD/origin matched,
index was empty, all 266 [snapshot](../../../../glossary.md#term-snapshot) [hashes](../../../../glossary.md#term-hash) matched and conflicts were inactive. Main verified launch;
implementation/review/gates are now closed on frozen source, and [CPU](../../../../glossary.md#term-cpu) frontier returns to None.

## Files and symbols

Java paths: `modules/backend/cpu/src/{main,test}/java/io/github/pho001/synaptik/backend/cpu/`.

- Main `internal/codegen/emit/CpuGeneralScanEmitter.java`: `Shaped` locals/[bindings](../../../../glossary.md#term-binding),
  `emit`, `load`, `copyDense`, `offset`, `frameOffset`, [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) loops and post-step proof.
- Main same directory `CpuGeneralScanReductionEmitter.java`, `CpuGeneralScanMatmulEmitter.java`:
  consumers of shared coordinates/addresses/factors, especially MATMUL <span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>[`stride`](../../../../glossary.md#term-stride).
- Main `internal/ir/CpuGeneralScanIr.java`: `bakedSpecializationIdentity`, staged [Javadoc](../../../../glossary.md#term-javadoc);
  `internal/cache/CpuKernelSpecialization.java`: both compatibility/class-[identity](../../../../glossary.md#term-identity-meanings) streams.
- Test `internal/codegen/emit/CpuGeneralScanGeneratedKernelTest.java`,
  `CpuGeneralScanCleanJavaOracle.java`, `CpuGeneralScanTimingTest.java`: structure, matched oracle,
  existing bounded timing seam; test <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>`internal/cache/CpuKernelSpecializationTest.java` and
  `CpuGeneratedKernelArtifactStoreTest.java`: identity/rejection/reuse. Focused existing lowering,
  preparation, execution and capability tests; new stale-v2 resources in existing emit test resources.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- Review current [CPU scan explanation](../../../../backend-guide/cpu-backend.md#bounded-float32float64-general-scan)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  and [general-scan explanation](../../../../architecture/general-differentiable-scan.md), plus only
  current identity facts in root Core invariants and owning backend contract (not broad rule edits).<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Acceptance criteria

1. Generated structure proves removed locals/stores/zero-adds/copies were redundant; forward
   time index aliases long iteration. Static B/H induction and widening stay valid, and runtime
   LONG T/index/addresses remain long. No hidden allocation, boxing, reflection, dispatch or helper
   calls; generated algorithm/hot-loop/dataflow matches the clean specialized Java oracle.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [widening glossary](../../../../glossary.md#term-widening); [runtime glossary](../../../../glossary.md#term-runtime); [dispatch glossary](../../../../glossary.md#term-dispatch); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
2. Allocate/initialize factors only where consumed. Baseline [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [`stride`](../../../../glossary.md#term-stride) read
   `state.factors[origin]` for ANY scalar formal: proved fixed scalar operands need effective stride
   zero, while evolving scalar→dense carries keep initialized dynamic factors through load/copy
   AND MATMUL consumers. No uninitialized local or dense read from a physical one-cell initializer.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
3. F32/F64 focused execution/structure covers T=0/1/multiple, empty axes/nonempty SUM from empty
   steps, forward/reverse, broadcast/SUM/MATMUL, fixed scalar operands and scalar→dense carries,
   tied/multiple carries, previous-state saves, forwarded/transposed/chained next views and final
   publication. Preserve old/new versions, raw empty final carry, demanded histories and fallbacks.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [axes glossary](../../../../glossary.md#term-axis); [views glossary](../../../../glossary.md#term-view-meanings); [publication glossary](../../../../glossary.md#term-publication); [carry glossary](../../../../glossary.md#term-carry))</span>
4. Advance only the sole baked `scan-shaped-v2` and `scan-matmul-v2` payloads to v3 in BOTH existing
   specialization streams when affected (shared Shaped cleanup is expected to affect both).
   Keep global envelope/family 71, geometry-free canonical v1 projections, unrelated/pointwise/fixed
   keys and bytes unchanged. No shadow version payload. Preserve historical v1/v2 provenance and
   add current v3 assertions rather than relabeling old goldens as new evidence.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [canonical glossary](../../../../glossary.md#term-canonical-form); [pointwise glossary](../../../../glossary.md#term-pointwise); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [provenance glossary](../../../../glossary.md#provenance))</span>
5. Reject actual stale v1 AND v2 classes at definition and persisted compatibility seams, including
   valid/checksummed same-current-path artifacts; preserve old-key entries and compatible runtime-T
   reuse. Keep existing v1 fixtures. Verify source geometry/spec/identity before freezing actual v2
   fixtures from <span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`/private/tmp/synaptik-cpu0034-FgGCiI/measurement-v2/measured-fork-1/production.class`
   and `/private/tmp/synaptik-cpu0034-FgGCiI/structure/matmul-0-0.class`; do not fake old bytes by
   patching new classes. Keep fixture hashes/identity provenance, including differing case geometry.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
6. Add a test-only B4/H8 [FP64](../../../../glossary.md#term-fp64) Java [oracle](../../../../glossary.md#term-test-oracle) alongside unchanged generic `broadcastForwardFloat64`.
   Runtime T is long (prefer the same entry ABI or clearly equivalent borrowed geometry), never
   compile-time 8/int; B/H are static. Same native access/layout, multiply-then-add order, emission
   carry backing and complete final copy, no shortcut output or primitive-array fast path. Check
   full outputs, including T=0 initial-carry publication, and class/identity/oracle disassembly.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [compile glossary](../../../../glossary.md#compile); [layout glossary](../../../../glossary.md#layout))</span>
7. Required independent final review/docs, worker validation, main checkpoint and the single bounded
   campaign are recorded with exact outcomes/limits before Complete; preserve all unrelated edits.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: focused [regression](../../../../glossary.md#term-regression)/structure during development; no opt-in timing during ordinary tests.
The existing evidenceRoot property retains classes/[identities](../../../../glossary.md#term-identity-meanings)/javap without enabling performance:

```bash
./gradlew -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0035-evidence.6sqRuQ :backends:cpu:test --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuGeneralScanLoweringTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuGeneralScanPreparationTest' --tests '*CpuGeneralScanCapabilityTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew :backends:cpu:test :backends:cpu:javadoc
git diff --check
```

Final [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) passed once on stabilized source; reviewer reused focused tests. Main [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
```

Completed campaign used the existing bounded main/harness and main's fresh guarded launcher/analyzer
at `/private/tmp/synaptik-cpu0035-evidence.6sqRuQ`, targeting the new matched-[ABI](../../../../glossary.md#term-application-binary-interface-abi)
`CpuGeneralScanCleanJavaOracle::broadcastForwardFloat64B4H8`. Original CPU0034 [artifacts](../../../../glossary.md#term-artifact)/generic
[oracle](../../../../glossary.md#term-test-oracle) remain unchanged. Main ran the following once after review and source-matched gates:

```bash
CPU0035_REVIEW_APPROVED=yes zsh /private/tmp/synaptik-cpu0035-evidence.6sqRuQ/run-bounded.zsh
```

Protocol: 3 sequential fresh JVMs, 100 [warmup](../../../../glossary.md#term-warmup)/300 measured alternating pairs each, 10000 calls/side,
T=8 [runtime](../../../../glossary.md#term-runtime) geometry/B=4/H=8; actual generated class versus equally specialized oracle. Shared native
inputs and distinct equally shared native outputs, zero workspace. Retain all primary pairs, full
output-byte checks, class/[identity](../../../../glossary.md#term-identity-meanings)/javap, [JFR](../../../../glossary.md#term-java-flight-recorder-jfr)/LogCompilation and clock/block correlation; distinguish
cold generation/definition/first entry from warmed costs. No jcmd, forced [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), pilot, retry,
extra [matrix](../../../../glossary.md#term-matrix-meanings), [Engine](../../../../glossary.md#term-engine)/[binding](../../../../glossary.md#term-binding) subtraction or universal parity/causal-v2-speedup claim.

## Follow-up

Required separate performance follow-up: a separately authorized narrow loop-[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form)
experiment in the existing emitter, starting with javac-style loop control; keep final-tail sharing
separate. The remaining ~3× gap has no proven cause. This closure creates or launches no successor,
runner or measurement and grants no new authorization; [Checkpoint](../../../../glossary.md#term-checkpoint)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and other gates stay held.

## Documentation and review impact

Independent final Class C review finalized affected emitter/[oracle](../../../../glossary.md#term-test-oracle)/timing Javadocs, current [backend](../../../../glossary.md#backend)/
general-scan explanations and descriptive v3 facts in root/owning backend contract. Documentation
rules and applicable general/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/backend/architecture profiles governed the targeted pass;
glossary/link/anchor/diff review found no new terminology, policy, [dependency](../../../../glossary.md#term-dependency) or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) requirement.

## Result

Status: Complete
Main accepted 2026-10-23 after independent final APPROVE, zero findings.
Changed three emitters, [IR](../../../../glossary.md#term-intermediate-representation-ir)/[cache](../../../../glossary.md#term-cache), generated/[oracle](../../../../glossary.md#term-test-oracle)/timing/cache tests, actual v2 fixtures/[provenance](../../../../glossary.md#provenance),
four targeted documents and these three [planning](../../../../glossary.md#term-planning) files; no new production type/[API](../../../../glossary.md#term-application-programming-interface-api)/capability/policy.
Generated code bytes/locals 377/77 → 297/61; specialized oracle 215/28. [Runtime](../../../../glossary.md#term-runtime) LONG/static B/H and
conditional-zero empty/[singleton](../../../../glossary.md#term-singleton)/[scalar](../../../../glossary.md#term-scalar)/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) fallbacks remain; generic oracle unchanged.
Only staged payloads advance to v3 in both streams; [schema](../../../../glossary.md#term-schema)/family 71, [canonical](../../../../glossary.md#term-canonical-form) v1 and unrelated
[identities](../../../../glossary.md#term-identity-meanings)/bytes stay frozen. Actual old v2 [fixture](../../../../glossary.md#term-fixture) equality and v1/v2 rejection/reuse are proved.
Commands above passed: 114 focused; final [CPU](../../../../glossary.md#term-cpu)/[Javadoc](../../../../glossary.md#term-javadoc) 5m27s, 216 suites, 1211 reported/1173 executed/
38 skipped, zero failures/errors; 94 preexisting Javadoc warnings. [Conformance](../../../../glossary.md#term-conformance) 7/7, [Engine](../../../../glossary.md#term-engine) 17/17, no skips.
Independent docs/Javadocs finalized; no new glossary terms/[dependency](../../../../glossary.md#term-dependency)/[ADR](../../../../glossary.md#term-architecture-decision-record-adr). Source remained frozen.
Evidence/changed-path list: [final-validation.json](/private/tmp/synaptik-cpu0035-evidence.6sqRuQ/final-validation.json)<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>.
Class SHA `870c954f4d83597980d5ea9d5c704a89c9cd8a459fbf35ca85cadc14ce01014e`;
key `295312b55b6f922dc77232e71af1d6dc22e49de755266b8bc15bd542d23556af`.
One campaign: 3 fresh sequential JVMs, 100 [warmup](../../../../glossary.md#term-warmup)/300 measured pairs each, 10000 calls/side;
all 900 primary pairs/2400 blocks independently audited, all output bytes equal before/after.
Primary per-[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) medians (generated ns / oracle ns; ratio of medians):
1: 386.875 / 125.125; 3.091908091908092. 2: 379.8083 / 124.37295; 3.053785409126341.
3: 385.44795 / 125.67085; 3.0671229644742595. Both normal-entry [C2](../../../../glossary.md#term-c2), zero [target](../../../../glossary.md#term-target) deopts/inlines.
Optional ±100ms [guard](../../../../glossary.md#term-guard) qualifies 281/[fork](../../../../glossary.md#term-fork-meanings); last 19 lack post-window tail only; ALL 300 remain primary.
Parity NOT achieved (~3× gap). Bottom-tested generated/top-tested javac loops and deeper generated
C2 loop tree are observed, not proven vectorization/cause/assembly. Reviewer accepted bounded cleanup
without a timing threshold; separate performance follow-up above is required, not launched. CPU0034 generic-oracle
timings remain historical, not causal v2→v3 speedup evidence. No [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/pushes or successor authorization.
