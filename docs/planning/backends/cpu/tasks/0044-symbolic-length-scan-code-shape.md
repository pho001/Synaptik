# Task CPU0044: Symbolic-length Scan code-shape optimization

## Status

Complete — MAIN accepted the bounded four-fix v4 repair and sole comparison on 2026-10-27 13:11 UTC. Java parity remains unresolved; CPU source/authority frontiers None. All four bodies consumed; no fifth/retry or automatic follow-up.

## Change class

Class C — generated hot-loop/dataflow and baked artifact compatibility; clean executor and
independent targeted code/documentation review mandatory. MAIN alone accepts the final result.

## Goal

Remove the four verified combined symbolic-B/runtime-length redundancies in the repair amendment
below, then compare retained v3/repaired v4 with unchanged Java in one bounded body. Earlier
layout/dataflow work and its negative v3 evidence remain historical, not fresh execution instructions.

## Scope and non-goals

One existing CPU capability, homogeneous F32/F64, both traversals and actual Compiler reverse.
No new production type/package, configuration, public API, ABI, Runtime handoff, dependency,
execution owner, numeric policy or architecture decision. Six core Scan semantic sections stay frozen.
No unrelated refactor, global carrier-policy change, algorithm/arity/layout change, unsafe access,
oracle weakening, broad timing campaign, default JIT-policy change, native build or Git mutation.
New named types, semantic ambiguity or actual contract conflicts require stop/report to MAIN.

## Contracts

Read the root scope index, then these exact headings; no full predecessor histories:

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core invariants`,
  `Scope-indexed normative contracts`, `Dependency rules`, `Testing requirements`.
- [Backend owner](../../../../architecture/contracts/backend-execution.md): `Concrete backend modules`,
  `Performance evidence and optimization tooling`, `CPU backend routes`,
  `Accepted ordinary symbolic-batch CPU contract` (checked access, identity and entry-cost rules).
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md): `Canonical residual ports and demand`,
  `Reverse scan and exact primal bindings`, `Traversal, empty domains and declared qualification`,
  `Qualification, performance and migration gates`; `Accepted symbolic-batch scan` →
  `Actual first-order closure and arithmetic`, `Empty domains and numeric resources`,
  `Qualification and mechanism gates`; `Length-aware whole-batch scan` → `Header, bindings and source admission`,
  `Bounded row-role analysis`, `Activity, raw identity and retained histories`,
  `First-order length-mode reverse`, `Runtime preflight and implementation gates`.

Prepare0015's [accepted Result](../../../modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
supplies acceptance behind older pending receipts; no semantic waiver or receipt cleanup is selected.
Existing append-only cache rules authorize baked-version advancement, not frozen changed bytes under old keys.

## Dependencies and integration

- Depends on: CPU0043; Prepare0015; CPU0041; CPU0042 — Complete, accepted source/evidence/reviews/gates.
- Conflicts with: combined Scan lowering/emission/resources/identities, CPU harness and shared docs; serialize writers.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`, empty index,
  plus accepted dirty inventory `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0044-repair-request-FodFPm.json` (2487 files; copies retained externally).
- Integration order: MAIN Ready/base acceptance → new clean repair executor → source/harness/protocol freeze
  → independent prelaunch approval → separate MAIN measured-launch authorization → sole repair body
  → independent final source/evidence/docs review and final owning gates → MAIN acceptance/maps.
- Integration validation: focused regression + final CPU test/Javadoc below; MAIN reuses recorded owning/targeted
  evidence, audits preservation/identities/links and accepts once. No root/conformance/integration rerun
  absent cross-module or semantic change; such a change requires scope approval, not silent expansion.
- Shared-document integration owner: MAIN; exclusive final docs delegation to independent reviewer after freeze.

Historical launch record (execution now Complete). Repair planning verification 2026-10-27:
all 2487 supplied inventory hashes matched before edits;
current CPU0043, Prepare0015 and CPU0041/0042 Status/Result confirm Complete with accepted reviews/gates.
User “ok, pojdme to zkusit opravit a zmerit” authorizes only the amendment below, not v2 rollback.
Per current handoff all prior agents are closed and conflicts inactive; serial writer, Parallel group None,
MAIN integration owner. CPU0044 alone was reopened Ready; MAIN acceptance is recorded below.
HEAD/origin/empty-index facts come from the supplied snapshot/handoff; this planner runs no Git commands.
MAIN Ready acceptance 2026-10-27: independently read amendment/metadata and prior full brief;
accepted four fixes + ONE new body subject to later freeze/review/separate measured authorization.
Complete dependencies/reviews, unchanged HEAD/origin/index, protected 2487-file source base and all
1496 old CPU0044 artifacts, prior 3289 historical files and inactive writers verified. In progress
for different clean executor launch ONLY; MAIN snapshots after this planning freeze. No measurement yet.

Execution/admission and protocol records below are historical; the current accepted outcome is in Result.

## Historical validation and bounded measurements

Original protocol for consumed slots 1–2 (unchanged); the explicit third-body amendment below extends only the total budget.

At most TWO short fresh-JVM test bodies, each <=60s including cold setup/checks, <=120s total.
Failed starts consume slots; no retries, five forks/matrix, warm-to-stability or extra JIT campaign.
Before slot 1 freeze precise variant arrangement/call counts/order/log flags for both slots; worker may
choose a compact valid arrangement, not adapt it to results. Fixed 100 warmup + 300 measured pairs,
100 calls/side/pair, alternating per-pair order; one F32 family only, B7 then B32, no timing in normal suites.
Slot 1 compares original + layout-only + unchanged optimal Java BEFORE full dataflow repair.
Slot 2 compares final with SAME Java and baseline witness confirmation using comparable entry wrappers.
Retained old ClassFiles are usable only after exact typed argument/order/pack/ABI mapping verification;
separate classloading/setup from kernel dataflow. Independent writable carriers, before/after output/raw/
poison checks outside timing. Deadline checks stop admitting work; disclose censoring/overshoot and missing windows.
Correlate timestamps/call counts with actual generated/oracle normal C2 nmethod availability and OSR;
report cold/startup separately from windows where BOTH normal entries exist. Counts alone prove no C2 state.
Disclose MethodHandle/wrapper/clock/admission costs; do not subtract empty-call estimates as kernel timing.

External per-task `probe.init.gradle` targets only `:backends:cpu` Test `test`: from FIRST launch set
`include('**/CpuGeneralScanTimingTest.class')`, `maxParallelForks=1`, `forkEvery=1`, disable task output
up-to-date/cache reuse only there; per-worker UnlockDiagnosticVMOptions/LogCompilation/PrintCompilation/
PrintInlining, `-XX:LogFile=<run-root>/hotspot_%p.log`. No global JAVA_TOOL_OPTIONS, --rerun-tasks,
repository build edits or compiler-policy flags. Validate actual PID/options/log and JUnit 1 executed,
0 skipped/failures/errors each slot; empty candidate workers are not a valid launch.

Exact command templates from root (RUN is `layout-only` then `final`; external init forwards existing properties):

```bash
./gradlew -I /private/tmp/synaptik-cpu0044.grql2o/probe.init.gradle :backends:cpu:test --tests '*.CpuGeneralScanTimingTest.symbolicLengthCodeShapeCpu0044' -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0044.grql2o/RUN -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
./gradlew :backends:cpu:test --tests '*.CpuSymbolicScanGeneratedKernelTest' --tests '*.CpuGeneralScanLoweringTest' --tests '*.CpuGeneralScanExecutionTest' --tests '*.CpuKernelSpecializationTest' --tests '*.CpuGeneratedKernelArtifactStoreTest' -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0044.grql2o/regression -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
./gradlew :backends:cpu:test :backends:cpu:javadoc -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0044.grql2o/validation -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
javap -c -p -v /private/tmp/synaptik-cpu0044.grql2o/ACTUAL-MEASURED.class
```

Record expanded commands, versions/hashes/counts/skips/limits externally. Final CPU tests once after
executable freeze; Javadoc once after final Javadoc edits. Review reuses tests, never reruns for duplication.

### Third-body amendment — user-approved 2026-10-27

Historical authorization below; all three slots are now consumed. Current disposition is in Result.

At most THREE fresh-JVM measured bodies, each <=60s, <=180s cumulative; two consumed. ONE slot remains;
no fourth, retry or adaptive warmup; failed starts consume slots. Original protocols/results stay immutable.
Slot 3: retained measured layout-only v2 SHA-256 `94c5476ee0b08c496d628d0789c37bd8138bdd4bb1f53f831ae4a63eeb03ef30`
versus frozen final v3 in SAME JVM; F32 false/T4/F3/H2, B7 then B32, existing inputs/L=row%5/joint seeds.
Fixed 100 warmup + 300 measured pairs x100 calls/side, balanced interleaving, unchanged Java plus direct v2/v3 metric.
EACH class authenticates OWN numeric pack/scratch/cuts/both identities; explicit semantic boundary mapping,
never normalize v3 facts to v2. Checked accesses unchanged; setup outside timing; timestamp normal-C2
availability and wrapper context. All other checks/logging/deadline/code-comparison requirements above apply.
Before MAIN's release below, Goodall remains repo-frozen. External `third-slot-protocol.txt` under the evidence root
must freeze harness/commands before independent prelaunch ABI/harness review, THEN MAIN launch authorization.
No production change authorized. Reuse passed source-matched full CPU suite/Javadoc if only opt-in performance
harness changes; independent review covers that diff. Performance pending: slot 2 B32 ~2.74x Java after normal
C2 versus slot 1 v1/v2 ~1.81x postnormal; cross-JVM causal attribution unproved (MAIN report, not a new result).
Size exception: retain original protocols/receipts intact plus bounded amendment and final Result; below 25 KB, same atomic task.

MAIN verification 2026-10-27: full brief/amendment, explicit user approval, unchanged dependencies/frontier/conflicts and CPU/Javadoc PASS accepted; two slots used. Goodall released ONLY for opt-in third diagnostic harness/untimed ABI preflight; v3 production frozen. Exact frozen `third-slot-protocol.txt` + harness require independent APPROVE, then separate MAIN measured-launch authorization. This is NOT measured launch.

## Repair amendment — user-authorized 2026-10-27

This amendment supersedes frozen-production/await-user instructions only for this continuation.
All three old bodies are consumed/exhausted; protocols and Result remain intact. No v2 rollback,
task cascade or broad performance work. Planner writes only this brief, CPU master and roadmap.
Evidence root: `/private/tmp/synaptik-cpu0044-repair.FodFPm`. Protect old v3/v2/oracle/harness roots,
sources, ClassFiles and identities before any executor tests; direct all new outputs here and audit
protected hashes afterward. Old timing methods/oracle are immutable.

### Repair scope and falsifiable acceptance

Existing production owners only: `codegen/emit/CpuGeneralScanEmitter` (`Shaped.arithmeticCell/load`,
initial copies and symbolic bases), `CpuCarrierEmitter`, `CpuClassFileKernelGenerator` (combined
layout reservation), `ir/CpuGeneralScanIr` and `cache/CpuKernelSpecialization`; all `.java` under
`modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`. No lowering/reduction-emitter rewrite. Focus existing generation,
execution and specialization/artifact tests; existing `CpuSymbolicLengthScanFixture` and timing class
may support a NEW opt-in `symbolicLengthRepairCpu0044` method only; do not edit old timing methods.

1. Remove already-available scalar local self-move: retained v3 BCI925 `fload 82` /927 `fstore 82`,
   traced to `arithmeticCell` ~1163 and `load` ~1311. Preserve operand/local availability and typed sites.
2. Initial dW/dU/db splat loads 6+4+2 → 1+1+1 for the witness, under each ORIGINAL NONEMPTY
   initialization domain. Cache actual input values, never infer literal zero. Preserve arbitrary/raw
   initials, T=0/B=0/all-zero activity and required shared initialization WITHOUT an activity guard.
   Proof is stable borrowed storage plus writable-overlap rejection in
   `executable/CpuGeneralScanPreparedExecutable.bindCompatible` ~156 and post-producer recheck ~233;
   read-only proof target, not edit scope. Never cache evolving carry as invariant.
3. Omit dead 4B products at v3 BCI54..70, locals208/212, and unused layout initialization locals25/28.
   Generator currently calls `prepareSegmentLayouts(...)` first: use the existing generator/carrier
   method seam to reserve fixed layout slot indices but omit initialization ONLY for combined constant-layout access;
   no global carrier policy. Remove unused geometry without moving expensive work into hot loops;
   no generic dead-code-elimination framework.
4. Share equal-coefficient sequence/cut/publication original-time products within a dominating
   time/activity lifetime: six → two witness products/iteration. Prove symbolic coefficient equality,
   never equality at sample B; retain definitions dominating final/empty exits and activity scopes.

No new type/API/config/schema/numeric pack/ABI/carrier/numerics/algorithm, T=4 baking, unsafe access
or broad rewrite. Preserve v3 cuts/scratch and canonical `:symbolic-length-v1`, schema/family71.
Advance ONLY `|scan-symbolic-length-v3=` → `|scan-symbolic-length-v4=` in BOTH full identity streams;
prove their complete difference is the baked version alone, with ordered semantic boundary mapping.
Reject stale combined artifacts; preserve every noncombined identity AND generated byte exactly.

Use the unchanged specialized Java code-equivalence oracle, arithmetic and raw/special-value tests;
both dtypes/traversals, actual Compiler reverse, B0/1, T0, empty features, arbitrary initial splats,
all-zero/mixed activity, poison padding/seeds and demanded/fanout histories must retain semantics.
Add scalar-variant and unequal-coefficient/lifetime negative cases. Inspect ACTUAL generated ClassFiles:
assert no dead products/self-assignment, one actual scalar memory load per nonempty initializer and
equal-base product sharing. Retain javap/BCI mappings and labeled Java-equivalent loop analysis; check
hidden helpers/allocation/boxing/reflection/dispatch. Smaller bytecode alone is not acceptance.

Contracts remain exactly those listed above. Initializer/binding proofs specifically use
[Empty domains and numeric resources](../../../../architecture/contracts/recurrent-scan.md#empty-domains-and-numeric-resources),
[Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
[Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories),
[First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
[Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)
and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
Required initial copy is not body hoisting. Contract ambiguity/changed semantics requires stop/report.

### Sole new measurement and freeze gates

Latest user authorization grants ONE NEW fresh-JVM measured body <=60s including cold setup/checks:
fourth overall, <=240s cumulative authorized ceiling. Failed start consumes it. No fifth, retry,
adaptive warmup, compiler-policy change or further campaign. Reuse the third harness PATTERN only:
new method/root, retained v3 authenticated against its OWN trusted metadata, never v2 pack mapping.
Compare retained v3 SHA-256 `a34c1c727792f90362a19a43e48d9d8b468c38470d69f710dcf91d4523135e0e`
with repaired v4 in SAME JVM and UNCHANGED Java oracle
`57d30a13da1a58836a1a89463295543c05ec2ff45d208c945d3fbb89242351f9`.
F32 false, T4/F3/H2, B7 then32, L=row%5, existing inputs/joint seeds; 100 warmup +300 measured
pairs ×100 calls/side. Each pair contains balanced interleaved v3/J3 and v4/J4 groups with fixed
alternating group and side order. Independent fixtures/packs/scratch; expected ABI identical, proved
by FULL both identity streams (version-only difference) and ordered semantic mapping, not a hash label.

Before measurement freeze exact source/harness/ClassFile/metadata hashes, expanded command, pair/group/
side order, flags, deadline and analysis windows in external `repair-protocol.txt`. Independent targeted
Class C source/ABI/harness/protocol reviewer must APPROVE that exact freeze BEFORE separate MAIN
measured-launch authorization. Ready acceptance or executor launch is NOT measured-launch approval.
Retain all raw samples, uptime brackets, PID/options/logs and JUnit (1 executed, 0 skipped/errors/failures).
Correctness/raw/poison checks stay outside timing; no empty-call subtraction. Qualify normal C2 and
wrapper compilation state with timestamps; report cold/censored/missing windows and entry-cost limits.
Production preflight is excluded and must be disclosed; fixed-T/literal-specialized Java is narrower
than the generated entry. No parity guarantee or acceptance merely from structural savings.

External `probe.init.gradle` must include target class FROM FIRST START, maxParallelForks=1,
forkEvery=1, cache/up-to-date reuse false only for this Test task; retain original diagnostic flags,
per-PID log and performance-property forwarding, no global/JIT policy edits or retiming old tests.
Freeze and review expanded repair command before release:

```bash
./gradlew -I /private/tmp/synaptik-cpu0044-repair.FodFPm/probe.init.gradle :backends:cpu:test --tests '*.CpuGeneralScanTimingTest.symbolicLengthRepairCpu0044' -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0044-repair.FodFPm/comparison -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
```

Executor runs the earlier focused test selection with NEW root `/private/tmp/synaptik-cpu0044-repair.FodFPm/regression`
and all three performance flags false. After executable freeze run final `:backends:cpu:test` once;
after final Javadoc edits run `:backends:cpu:javadoc` once, using the earlier final command with NEW
root `/private/tmp/synaptik-cpu0044-repair.FodFPm/validation` and all performance flags false.
No full-root/conformance/integration rerun absent a semantic/boundary change (which first needs approval).
Independent final source/evidence/docs review reuses successful source-matched gates. MAIN separately
accepts or withholds performance after the sole body; no faster-than-v3 or near-Java result is presumed.

Documentation impact: planning only now. Executor/reviewer must review affected implementation Javadoc,
update changed invariants or justify existing accuracy, and perform targeted glossary/link checks.
Only after acceptance may narrowly scoped baked-version receipts be integrated by MAIN/delegated
reviewer as permitted below; no new semantics, ADR or architecture decision. Atomic size exception (>15KB,
and >25KB with closeout): retain immutable protocols/negative Result with this repair; splitting would
obscure the exhausted budget and separate the accepted result from its qualification limits.

## Documentation and review impact

Independent source/evidence/Javadoc review APPROVE, no open P1/P2. Fermat completed the accepted
baked-v4 receipts in `ARCHITECTURE.md`, `docs/architecture/contracts/backend-execution.md` and
`docs/backend-guide/cpu-backend.md`; MAIN approved that scope and released writing on 2026-10-27
13:16:26. Final Javadoc remains current; no further Java edit or test rerun. Glossary impact: none;
no new term/meaning, numerical or architecture decision. This closeout changes only this brief,
CPU master and roadmap; MAIN retains final integration/scope checks.

## Result

Status: Complete — bounded four-fix repair and one comparison, MAIN accepted 2026-10-27 13:11 UTC.
Broader unchanged-Java parity remains UNRESOLVED; no automatic successor, rollback or extra budget.

- Removed scalar self-move; initial dW/dU/db scalar loads 6+4+2→1+1+1 under original nonempty initialization domains; omitted dead 4B/layout initialization while reserving slots; shared six→two equal-coefficient original-time products with dominating activity/lifetimes. Arbitrary/raw initials, required empty-activity shared initialization and evolving carries remain correct.
- Production file group: `CpuGeneralScanEmitter`, `CpuCarrierEmitter`, `CpuClassFileKernelGenerator`, `CpuGeneralScanIr`, `CpuKernelSpecialization`. Test/harness group: `CpuSymbolicScanGeneratedKernelTest`, specialization/artifact-store tests and new timing/preflight methods in `CpuGeneralScanTimingTest`; old timing/oracle methods unchanged.
- Combined baked v4 in both full identity streams; canonical symbolic-length-v1, numeric pack/ABI, schema/family71 and v3 cuts/scratch unchanged. All 23 freshly regenerated noncombined key/byte rows exact; checked access preserved.
- [MAIN acceptance](/private/tmp/synaptik-cpu0044-repair.FodFPm/MAIN-acceptance.txt), [accepted handoff](/private/tmp/synaptik-cpu0044-repair.FodFPm/accepted-handoff-receipt.json), [exact freeze](/private/tmp/synaptik-cpu0044-repair.FodFPm/freeze-manifest.json) SHA-256 `a94a38d629ba735268f2e123253cab0271c1551d6b52040aeb973ff60645b034`; independent source/prelaunch and evidence-validity APPROVE, no open P1/P2.
- [Final analysis](/private/tmp/synaptik-cpu0044-repair.FodFPm/comparison/final-analysis.json) SHA-256 `82dab25350c21d94a218585c3ed87642a23c0420c7fc2a9028c80be3a62a5e64`: PID61708, complete pre-persistence body 1.844793209s, no censor/overshoot; JUnit 1 executed/0 skips/failures/errors, 3200 sides/320000 timed calls/32 untimed calls. Frozen F32 false/T4/F3/H2, B7 then32 protocol; retained v3/repaired v4/unchanged Java in one JVM, independent packs/scratch.

All 300 pairs per B meet the conservative common-live-normal-C2 rule; medians below are ns/call and paired ratios, not ratios reconstructed from medians.

| B | Pairs | v3 ns/call | v4 ns/call | Java J4 ns/call | Paired v4/v3 | Paired v4/J4 |
|---|---|---|---|---|---|---|
| 7 | 300 | 1134.795 | 760.205 | 552.295 | 0.6717569725 | 1.3757149954 |
| 32 | 300 | 4836.665 | 3080.835 | 1750.210 | 0.6376664359 | 1.7612377137 |

- v4 elapsed time is 32.82%/36.23% lower than v3 in these cases. B32 after the last wrapper transition: n229, paired v4/v3=0.6383742972, v4/J4=1.7645602459, consistent with the primary result.
- Limits: one bounded observation, not steady-state/cross-platform qualification or individual-fix causal proof. Normal nmethod availability does not prove every invocation used it. Wrapper transitions, MH/spreader versus direct entry, deadline clocks, runtime-T versus fixed-T/literal Java and excluded production binding/preflight remain disclosed; no empty-call subtraction. Java has 80000 timed calls/B versus 40000 per generated variant/B. Remaining gap is not explained or declared closed.
- Gates reused, not rerun by this planner: 139 focused PASS; `:backends:cpu:test` 1285 executed/50 skipped/0 failures/errors (222 suites). Full run covers unchanged production/harness plus PRIOR census test, NOT all final test source; later optional-output null/blank guard has separate final no-property census PASS (1 test). Untimed ABI preflight PASS. Final `:backends:cpu:javadoc` PASS, 94 existing warnings. Exact expanded commands/source receipts: [commands](/private/tmp/synaptik-cpu0044-repair.FodFPm/commands.txt), [full-run source freeze](/private/tmp/synaptik-cpu0044-repair.FodFPm/validation-execution-source-freeze.json).
- Preservation: all 1496 old CPU0044 + prior 3289 evidence files unchanged; HEAD/origin/index untouched. Historical protocols and the v3 Result below remain verbatim. All FOUR bodies are consumed; no fifth/retry, new measurement, production task, task cascade or v2 rollback authorized.
- Documentation/Javadoc: independent accepted baked-version receipts above completed; glossary unchanged, current Javadoc accurate. Planning-only closeout; no authority/semantic/source/test/JVM/Git changes by planner. CPU source/authority frontiers None; broader parity is an unresolved target, not required follow-up within this completed scope.

The following disposition is HISTORICAL, before the repair authorization; preserve it verbatim.

### Historical v3 disposition — three bodies consumed

Status: Incomplete
Follow-up required: user disposition on returning dataflow to measured layout-only v2 (recommended) versus a separately authorized targeted JIT fix; neither revert nor new work is authorized.

- Correctness APPROVE; MAIN withholds/rejects performance acceptance. v3 production remains frozen; all three slots consumed, no fourth/retry or new production change.
- Implemented combined-only constant layouts, SUM-tail/cut and activity-gated scalar reuse, with baked v2/v3 identities; canonical v1 and retained modes preserved.
- Affected implementation: CpuCarrierEmitter/CpuGeneralScanEmitter/CpuGeneralScanReductionEmitter, CpuGeneralScanLowering/CpuGeneralScanIr/CpuKernelSpecialization and focused CPU generation/cache/semantic/harness tests; this disposition changes only task/master/roadmap.
- [Actual code comparison](/private/tmp/synaptik-cpu0044.grql2o/code-comparison.txt) retains original/layout-only/final ClassFiles, both identities, javap and labeled Java-equivalent per-loop analysis; [third-body addendum](/private/tmp/synaptik-cpu0044.grql2o/code-comparisonaddendum-third.txt) records the same-JVM comparison.
- Bytecode differences: constant getstatic replaces layout-local loads; scalar reuse removes an inner-loop read; final bias SUM retains one distinct carry ADD without its 8-byte F32 cut/pass. Structural savings did not deliver speed.
- User-authorized third body: [frozen report](/private/tmp/synaptik-cpu0044.grql2o/v2-v3-ab/cpu0044-third-report.txt), [final analysis](/private/tmp/synaptik-cpu0044.grql2o/v2-v3-ab/final-analysis.json); PID40178, exact approved hashes match, JUnit 1 executed/0 skipped/failures/errors.
- JUnit testcase 2.281s, suite 2.292s; complete uncensored pre-persistence body 2.265493541s. Same-JVM F32 false/T4/F3/H2, B7 then B32; own v2/v3 numeric pack/cuts/scratch/identities, unchanged Java and fixed balanced protocol.
- Conservative windows start after ALL normal C2 entries + >=2ms: B7 n=227 paired v3/v2=1.3945194, v2=847.92ns, v3=1179.58ns, Java~589ns; raw all-300 B7 ratio=1.41115 is distinct.
- B32 n=300 paired v3/v2=1.50146818, v2=3160.83ns, v3=4752.915ns, Java~1718ns; paired v2/Java=1.8415829, v3/Java=2.77449699. Confirmed regression is bounded to these measured cases, not universal parity evidence.
- Normal C2 installed Java/V2/V3 at 1.429/1.721/2.135s. B7 overlaps harness compilation transitions; B32 follows final harness recompile, so the result is not dismissed as transient-only.
- Total C2 nmethod allocation size v2=21392/v3=23560 bytes, not isolated instruction-code size; size/logs do not establish machine-level cause. MethodHandle/spreader versus direct-Java entry costs still differ.
- Parfit independent harness/untimed ABI preflight APPROVE; final post-run evidence review Complete, no evidence-validity blocker, concurs ~1.395x/~1.501x slower v3 and withheld performance acceptance.
- Source-matched owning evidence reused: CPU 1281 executed/46 skipped, 222 suites, 0 failures/errors; Javadoc PASS with 94 unrelated warnings. Exact commands remain in Validation/external evidence; no rerun for this planning disposition.
- MAIN preservation audit: frozen source hashes and all 3289 historical files unchanged after slot 3. Original protocols/results and complete predecessor histories remain intact.
- Documentation impact: these three planning records only; no root/backend-guide version receipts, authority, numerical/Scan semantics, architecture decision or glossary change. Source Javadoc correctness review already approved.
- Required next step is a user decision, then explicit MAIN scope/acceptance; performance is NOT accepted, task is NOT Complete, and no successor or extra measurement is authorized.
