# Task CPU0041: Symbolic Scan reverse carrier specialization

## Status

Complete — MAIN accepted the bounded fix on 2026-10-26 after final independent approval and gates.
Current CPU source/test frontier: None; no automatic successor or further measurement authorized.

## Change class

Class C — generated reverse hot-loop/dataflow and checked foreign-memory access invariants.
Clean implementation and independent clean targeted source/Javadoc/documentation review required.

## Goal

Establish the source-level cause of repeated reverse dH checks, then implement the smallest
evidence-backed correction in existing CPU owners. dH is the reverse carry read by parameter
gradient dW and input gradient dX contractions. Preserve optimal specialized clean-Java semantics
and checked Foreign Function & Memory (FFM) access; measure actual final phase/code improvement.

## Scope

- First make a controlled source-level cause test in the existing emitter/carrier dataflow,
  holding arithmetic, workload, layout and compiler policy fixed; record hypothesis/control/result.
  Use that evidence before selecting the minimal production correction, not a predetermined rewrite.
- Compare generated carrier selection, reads, contributor loops and dataflow with the unchanged
  specialized Java oracle; retain exact saved sites, contraction order and complete actual reverse.
- Add focused semantic/structural regression and identity invalidation/pins; inspect final bytecode
  and actual optimized code for dH checks, backedges, calls and avoidable overhead.
- If symbolic bytes change, advance only the baked symbolic payload v2→v3 in both existing
  identity streams, rejecting stale v1/v2 artifacts. Preserve canonical symbolic v1/family 71.

## Non-goals

No new production type/abstraction, supported API, Runtime handoff, unsafe route, unchecked access,
arithmetic reassociation, alignment-layout change, resource/lifetime weakening, or broad T/frame
rewrite. A final local alone is not proof. No forced compiler/inlining/tier flags, retuned protocol,
broader B/shape/Engine timing matrix, five-fork campaign, native/root build, commit/push or GSD.
Static shaped/MATMUL v4 and pointwise/fixed/ordinary keys/bytes remain frozen; no NN migration,
length-mode, backend/dtype, geometry-memo, route or topology expansion. Stop on required expansion.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
- Backend execution [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- Recurrent scan [Accepted symbolic-batch scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan),
  including **Symbolic templates, roles and ports**, **Actual first-order closure and arithmetic**,
  **Empty domains and numeric resources**, and **Qualification and mechanism gates**.

These current contracts govern; stop on architectural ambiguity rather than inventing a mechanism.

## Dependencies and integration

- Depends on: [CPU0040](0040-symbolic-scan-phase-code-diagnostic.md#result),
  [CPU0039](0039-symbolic-scan-fixed-geometry-specialization.md#result),
  [CPU0037](0037-general-scan-dense-read-views.md#result),
  [Prepare0014](../../../modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result); all Complete.
- Conflicts with: Scan emitter/carrier/IR/identity, admission/resources/binding, shared contracts/docs,
  and every performance task; serialize these scopes, no competing writer or performance run.
- Parallel group: None; serial writer handoffs need no worktree; read-only audit may overlap.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted dirty work, not HEAD ownership or merged predecessors; index empty.
- Integration order: MAIN launch recheck → clean cause test/minimal implementation → independent
  targeted review/corrections → source/Javadoc freeze → worker final CPU gate → MAIN named join
  and final phase/code capture → MAIN acceptance/status integration; reuse successful evidence.
- Integration validation: CPU0041 named conformance/Engine join and single final phase/code
  diagnostic below; no root/native gate unless newly approved actual scope requires one.
- Shared-document integration owner: MAIN; exclusive serial delegation for contract-version text,
  explanatory docs and planning closure; final independent reviewer finalizes affected docs/Javadoc.

Admission verified 2026-10-26: predecessor current Status/Result records include final independent
approvals, required validation and MAIN acceptance. CPU0040 is diagnostic-only, not a solved gap.
Latest user approval selects CPU0041 alone. MAIN confirms sole planning writer/inactive conflicts,
2475-file baseline and HEAD/origin/index; planner independently compared all baseline hashes with
no changed/missing/new paths before edits. Snapshot:
`/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0041-pre-plan.json`.
MAIN rechecks this Ready frontier, conflicts and baseline before handing off the sole writer.

## Files and symbols

Production paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `codegen/emit/CpuGeneralScanEmitter.java` — `Shaped.emitSymbolic`, carrier/base binding and reads.
- `codegen/emit/CpuGeneralScanMatmulEmitter.java` — `emit`, ordered operand reads/contributor control.
- `ir/CpuGeneralScanIr.java` — `Stages` Javadoc and `bakedSpecializationIdentity`; both existing
  specialization streams consume this payload. Inspect the existing carrier seam; no broad refactor.

Tests share `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`:

- `CpuSymbolicScanGeneratedKernelTest`, `CpuGeneralScanGeneratedKernelTest` — actual reverse,
  FFM/empty/carry semantics, generated structure and retained static pins.
- `CpuSymbolicScanTimingMethodologyTest` — narrow v3 identity/pin and new structural regression
  edits allowed; preserve historical pins and v1 reconstruction, adapting its version handling
  correctly. No measurement/oracle/protocol change; this file is not frozen wholesale.
- MAIN's existing `CpuSymbolicScanConformanceTest` and `EngineCpuSymbolicScanIntegrationTest`
  are the named boundary witnesses; extend only for a concrete same-contract regression if needed.

Four files in that test directory are immutable; SHA-256 pins:

```text
CpuSymbolicScanCleanJavaOracle.java  8d6abed866ee4984b3286b1d36e03ea72f447c33a1da162dea6547c2d17530a1
CpuSymbolicScanTimingTest.java       506a5a6bb949bcac8e9e4c9bc16a1f4421f816477a691d92384858f59fe443a9
CpuSymbolicScanPhaseOracle.java      e9524585331c553325940e0f980405812e75aa9d51cfd84c15e092a8b1fbf940
CpuSymbolicScanPhaseDiagnosticTest.java 43dea472afc9ad58898077be3e5c91faaf204b4e89cb3129c50749fe22995c62
```

## Acceptance criteria

- Retain the controlled cause test and evidence connecting the minimal fix to the dH dataflow.
  CPU0040's report `/private/tmp/synaptik-cpu0040-main.P3w6A4/report.txt` is the starting evidence:
  generated reverse dW/dX dH.get BCIs 570/763 retain type/bounds/session checks and +1 contributor
  backedges versus direct +4 without inner FFM guards; helpers inline, no measured C2 invalidation.
  Packed-time causality is unproved: checked dH uses row*128+column*4; frame-dependent X is guard-free.
- Preserve actual Compiler-generated reverse outputs, saved bits, both traversals/seeds, B=0/1,
  T=0, mixed-B reuse/isolation, shared gradients, aliases, checked bounds and session/lifetime safety.
  Reuse existing matrix coverage; add the precise regression, not a new performance matrix.
- Class-File verification and structural inspection show the intended carrier/read/loop change,
  unchanged arithmetic and no hidden allocation, boxing, reflection, helper dispatch or fallback.
  Explain how any hoisted facts remain valid for the admitted carriers/ranges/aliases/lifetimes.
- Both symbolic identity streams bind changed bytes/meaning; retain historical v1/v2 pins and
  pin actual v3 keys/compatibility/Class-Files when advanced. All static/ordinary witnesses pass
  unchanged. If static bytes must change, stop for scope review, never update their pins to pass.
- Final actual code and all three phase results are reported honestly, including residual gap,
  regression, incomplete C2/decode coverage or censorship. Correctness alone does not close the
  requested performance issue; do not label a partial result parity or silently mark it solved.

## Validation

Configured JDK 26; worker focused development checks, then ONE final affected CPU test/Javadoc
gate after source/Javadoc freeze. Preserve XML/logs before performance replaces task output.

```bash
./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuSymbolicScanTimingMethodologyTest' --tests '*CpuSymbolicScanPhaseDiagnosticTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --console=plain
./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test :backends:cpu:javadoc -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --console=plain
JDK_JAVA_OPTIONS='--add-modules=jdk.incubator.vector' ./gradlew --no-daemon --no-parallel --max-workers=1 :testing:backend-conformance:test --tests '*CpuSymbolicScanConformanceTest' :testing:integration-tests:test --tests '*EngineCpuSymbolicScanIntegrationTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
```

MAIN owns the third command once and the final fresh measured test JVM. At most ONE development
C2 capture is allowed only if the source cause needs it; worker coordinates with MAIN BEFORE
launch. No concurrent performance tasks, automatic retry, extra fork or adaptive extension.
MAIN creates an external `CPU0041_EVIDENCE` directory and targeted compile-command file for actual
post-fix generated identities, unchanged Java phases/original joint oracle and effective wrappers.

```bash
JDK_JAVA_OPTIONS="--add-modules=jdk.incubator.vector -XX:+UnlockDiagnosticVMOptions -XX:+LogCompilation -XX:LogFile=$CPU0041_EVIDENCE/hotspot-%p.xml -XX:CompileCommandFile=$CPU0041_EVIDENCE/compile-commands.txt" ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --rerun --tests '*CpuSymbolicScanPhaseDiagnosticTest.boundedForwardReverseAndJointDiagnostic' -Dsynaptik.cpu.dynamicBatch.performance=true -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
```

Keep fixed 30s body/three rows, 3000 startup pairs + 1000 warm pairs + 3×300 paired windows per
row; normal compiler policy, no global PrintAssembly. Record every forward/reverse/joint window,
median, entry/admission/setup costs and source/command hashes. CPU0040 medians G/D were
1.000082190×/3.694822522×/2.208052723× respectively, diagnostic not universal targets.
Capture addressed MachCode/BCIs and actual normal C2 version/inline/invalidation versus windows;
use the existing external LLVM decoder offline, no install or extra JVM. Report code improvement
and capture limitations, not exact cycle attribution. Stop when the authorized budget is consumed.
Validate four immutable hashes, scoped diff, links/anchors, brief size and `git diff --check`.

## Documentation and review impact

Peirce independently approved final source/tests/Javadoc/docs and the final evidence audit with
no findings/P1/P2; successful executable evidence was reused. MAIN coordinated the three current
symbolic-v3 paragraphs in root Core invariants, backend CPU routes and the general-scan explanation.
Existing Javadoc remains accurate with no Java comment edits; final generation passed afterward.
Targeted guide/glossary review required no broader rewrite, glossary change or ADR. Planning/general
profiles govern this closure; no architecture redesign, new type/API/Runtime handoff or unsafe route.

## Result

Status: Complete — MAIN accepted 2026-10-26; Peirce final source/docs and evidence audits APPROVE, zero findings/P1/P2.
Changed: `CpuGeneralScanEmitter` immutable entry receivers for demanded port-backed cuts,
`CpuGeneralScanIr` symbolic v3, two existing symbolic generated/methodology tests, three version-doc
paragraphs and this brief/master/roadmap: exactly 10 paths against the pre-plan snapshot.
No MATMUL-emitter/new type/API/Runtime/numerics/resource change; checked FFM remains intact.
Both streams reject symbolic v1/v2; canonical v1/family 71/static v4 and four immutable files retained.
Controlled source/bytecode comparison isolates receiver substitution/dead copies, not an internal C2 phi proof.
Focused: initial 53 PASS; final 64 first yielded 62 PASS/1 skip/1 test-only WrongMethodTypeException;
corrected void `invokeExact` lambda targeted 1 PASS gives 63 PASS/1 skip combined, then full gate green.
[CPU gate](</private/tmp/synaptik-cpu0041-worker.jRJgxS/gate-handoff.txt>): ONE final command, 5m25s, 222 suites,
1270 tests = 1229 PASS/41 opt-in skips/0 failures/errors; Javadoc PASS, 94 warnings, no warning repair.
MAIN named join: 6s, two fresh conformance/Engine tests PASS, zero skips/failures/errors.
Final fresh JVM 52779: timestamp 2026-10-26T05:28:22.535Z, build 18s, body 14.778689s/30s;
all 18×300 windows complete, zero censorship/overshoot. One DEV + one FINAL measured JVM; worker zero.
Final median G/D microseconds: F 218.134720/218.042363333 (1.000423572×),
R 173.630833333/173.064443333 (1.003272712×), J 393.047083333/388.821110 (1.010868683×).
Final normal C2 4060 dW/dX has ordered four-way guard-free backedges matching Java 3940/joint 4108;
all five current normal main-code intervals have zero unknown words; global 796 unknown bytes remain.
Fixed T4/F32/H32 B64 native-scalar gap resolved: bounded observed near-Java, not universal speed parity,
exact cycle attribution, sampled-PC proof or blanket decode success; protocol/oracles unchanged.
Full 2476-file freeze matched before/after gates; HEAD/origin/empty index unchanged. Docs/Javadoc final;
no glossary/ADR change, unresolved issue or required follow-up. CPU frontier None; no automatic successor.
[Final report](</private/tmp/synaptik-cpu0041-main.F8vADG/result.txt>) links exact commands, raw metrics, code addresses and coverage evidence.
