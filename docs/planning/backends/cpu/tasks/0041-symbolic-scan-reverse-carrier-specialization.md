# Task CPU0041: [Symbolic Scan](../../../../glossary.md#symbolic-scan) reverse [carrier](../../../../glossary.md#term-storage-carrier) [specialization](../../../../glossary.md#term-specialization)

## Reader context — historical integration boundary

The correction concerns the storage receiver used by generated reverse contractions: dH, the reverse
carry, supplies values consumed in dW and dX. CPU proves a demanded port-backed cut has a stable
entry receiver and uses that fact without removing checked Foreign Function and Memory access. The
composition still lends source storage and Runtime still owns run outputs/workspaces; no unsafe
pointer or new execution owner appears. Source intervention and final code observations support this
bounded correction, while the near-Java result belongs only to the named workload/host and does not
establish a universal JIT or speed guarantee.

Current reading: [Symbolic-batch Scan execution](../../../../backend-guide/cpu-backend.md#symbolic-scan-cpu-execution).

## Status

Complete — MAIN accepted the bounded fix on 2026-10-26 after final independent approval and gates.
Current [CPU](../../../../glossary.md#term-cpu) source/test frontier: None; no automatic successor or further measurement authorized.

## Change class

Class C — generated reverse hot-loop/[dataflow](../../../../glossary.md#term-dataflow) and checked foreign-memory access invariants.
Clean implementation and independent clean targeted source/[Javadoc](../../../../glossary.md#term-javadoc)/documentation review required.

## Goal

Establish the source-level cause of repeated reverse dH checks, then implement the smallest
evidence-backed correction in existing [CPU](../../../../glossary.md#term-cpu) [owners](../../../../glossary.md#term-owner). dH is the reverse [carry](../../../../glossary.md#term-carry) read by [parameter](../../../../glossary.md#term-parameter)
[gradient](../../../../glossary.md#term-gradient) dW and input gradient dX contractions. Preserve optimal specialized clean-Java semantics
and checked Foreign Function & Memory ([FFM](../../../../glossary.md#term-ffm)) access; measure actual final phase/code improvement.

## [Scope](../../../../glossary.md#term-scope)

- First make a controlled source-level cause test in the existing emitter/carrier dataflow,
  holding arithmetic, workload, layout and compiler policy fixed; record hypothesis/control/result.
  Use that evidence before selecting the minimal production correction, not a predetermined rewrite.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [dataflow glossary](../../../../glossary.md#term-dataflow); [layout glossary](../../../../glossary.md#layout); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compare generated carrier selection, reads, contributor loops and dataflow with the unchanged
  specialized Java oracle; retain exact saved sites, contraction order and complete actual reverse.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- Add focused semantic/structural regression and identity invalidation/pins; inspect final bytecode
  and actual optimized code for dH checks, backedges, calls and avoidable overhead.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [identity glossary](../../../../glossary.md#term-identity-meanings); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- If symbolic bytes change, advance only the baked symbolic payload v2→v3 in both existing
  identity streams, rejecting stale v1/v2 artifacts. Preserve canonical symbolic v1/family 71.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [artifacts glossary](../../../../glossary.md#term-artifact); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>

## Non-goals

No new production type/abstraction, supported [API](../../../../glossary.md#term-application-programming-interface-api), [Runtime](../../../../glossary.md#term-runtime) [handoff](../../../../glossary.md#term-handoff), unsafe route, unchecked access,
[arithmetic reassociation](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks), [alignment](../../../../glossary.md#term-alignment)-[layout](../../../../glossary.md#layout) change, resource/[lifetime](../../../../glossary.md#term-lifetime) weakening, or broad T/frame
rewrite. A final local alone is not proof. No forced [compiler](../../../../glossary.md#term-compiler)/[inlining](../../../../glossary.md#term-inlining)/tier flags, retuned protocol,
broader B/[shape](../../../../glossary.md#shape)/[Engine](../../../../glossary.md#term-engine) timing [matrix](../../../../glossary.md#term-matrix-meanings), five-[fork](../../../../glossary.md#term-fork-meanings) campaign, native/root build, [commit](../../../../glossary.md#term-commit)/push or GSD.
Static shaped/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) v4 and [pointwise](../../../../glossary.md#term-pointwise)/fixed/ordinary keys/bytes remain frozen; no [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) migration,
length-mode, [backend](../../../../glossary.md#backend)/[dtype](../../../../glossary.md#data-type--datatype), geometry-[memo](../../../../glossary.md#cache-and-memoization), route or [topology](../../../../glossary.md#term-topology) expansion. Stop on required expansion.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
- [Backend](../../../../glossary.md#backend) execution [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- Recurrent scan [Accepted symbolic-batch scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>,
  including **Symbolic templates, roles and ports**, **Actual first-order closure and arithmetic**,
  **Empty domains and numeric resources**, and **[Qualification](../../../../glossary.md#term-qualification) and mechanism gates**.

These current contracts govern; stop on architectural ambiguity rather than inventing a mechanism.

## Dependencies and integration

- Depends on: [CPU0040](0040-symbolic-scan-phase-code-diagnostic.md#result),
  [CPU0039](0039-symbolic-scan-fixed-geometry-specialization.md#result),
  [CPU0037](0037-general-scan-dense-read-views.md#result),
  [Prepare0014](../../../modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result); all Complete.
- Conflicts with: Scan emitter/carrier/IR/identity, admission/resources/binding, shared contracts/docs,
  and every performance task; serialize these scopes, no competing writer or performance run.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run))</span>
- Parallel group: None; serial writer handoffs need no worktree; read-only audit may overlap.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted dirty work, not HEAD ownership or merged predecessors; index empty.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Integration order: MAIN launch recheck → clean cause test/minimal implementation → independent
  targeted review/corrections → source/Javadoc freeze → worker final CPU gate → MAIN named join
  and final phase/code capture → MAIN acceptance/status integration; reuse successful evidence.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Integration validation: CPU0041 named conformance/Engine join and single final phase/code
  diagnostic below; no root/native gate unless newly approved actual scope requires one.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [scope glossary](../../../../glossary.md#term-scope))</span>
- Shared-document integration owner: MAIN; exclusive serial delegation for contract-version text,
  explanatory docs and planning closure; final independent reviewer finalizes affected docs/Javadoc.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

[Admission](../../../../glossary.md#term-admission) verified 2026-10-26: predecessor current Status/Result records include final independent
approvals, required [validation](../../../../glossary.md#term-validation) and MAIN acceptance. CPU0040 is diagnostic-only, not a solved gap.
Latest user approval selects CPU0041 alone. MAIN confirms sole [planning](../../../../glossary.md#term-planning) writer/inactive conflicts,
2475-file baseline and HEAD/origin/index; planner independently compared all baseline [hashes](../../../../glossary.md#term-hash) with
no changed/missing/new paths before edits. [Snapshot](../../../../glossary.md#term-snapshot):
`/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0041-pre-plan.json`.
MAIN rechecks this [Ready](../../../../glossary.md#term-ready) frontier, conflicts and baseline before handing off the sole writer.

## Files and symbols

Production paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `codegen/emit/CpuGeneralScanEmitter.java` — `Shaped.emitSymbolic`, [carrier](../../../../glossary.md#term-storage-carrier)/base [binding](../../../../glossary.md#term-binding) and reads.
- `codegen/emit/CpuGeneralScanMatmulEmitter.java` — `emit`, ordered operand reads/contributor control.
- `ir/CpuGeneralScanIr.java` — `Stages` [Javadoc](../../../../glossary.md#term-javadoc) and `bakedSpecializationIdentity`; both existing
  specialization streams consume this payload. Inspect the existing carrier seam; no broad refactor.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>

Tests share `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`:

- `CpuSymbolicScanGeneratedKernelTest`, `CpuGeneralScanGeneratedKernelTest` — actual reverse,
  FFM/empty/carry semantics, generated structure and retained static pins.<span data-glossary-navigation="true"> ([FFM glossary](../../../../glossary.md#term-ffm); [carry glossary](../../../../glossary.md#term-carry))</span>
- `CpuSymbolicScanTimingMethodologyTest` — narrow v3 identity/pin and new structural regression
  edits allowed; preserve historical pins and v1 reconstruction, adapting its version handling
  correctly. No measurement/oracle/protocol change; this file is not frozen wholesale.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [regression glossary](../../../../glossary.md#term-regression); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- MAIN's existing `CpuSymbolicScanConformanceTest` and `EngineCpuSymbolicScanIntegrationTest`
  are the named boundary witnesses; extend only for a concrete same-contract regression if needed.

Four files in that test directory are [immutable](../../../../glossary.md#term-immutable); [SHA-256](../../../../glossary.md#term-sha-256) pins:

```text
CpuSymbolicScanCleanJavaOracle.java  8d6abed866ee4984b3286b1d36e03ea72f447c33a1da162dea6547c2d17530a1
CpuSymbolicScanTimingTest.java       506a5a6bb949bcac8e9e4c9bc16a1f4421f816477a691d92384858f59fe443a9
CpuSymbolicScanPhaseOracle.java      e9524585331c553325940e0f980405812e75aa9d51cfd84c15e092a8b1fbf940
CpuSymbolicScanPhaseDiagnosticTest.java 43dea472afc9ad58898077be3e5c91faaf204b4e89cb3129c50749fe22995c62
```

## Acceptance criteria

- Retain the controlled cause test and evidence connecting the minimal fix to the dH dataflow.
  CPU0040's report <span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow))</span>`/private/tmp/synaptik-cpu0040-main.P3w6A4/report.txt` is the starting evidence:
  generated reverse dW/dX dH.get BCIs 570/763 retain type/bounds/session checks and +1 contributor
  backedges versus direct +4 without inner FFM guards; helpers inline, no measured C2 invalidation.
  Packed-time causality is unproved: checked dH uses row*128+column*4; frame-dependent X is [guard](../../../../glossary.md#term-guard)-free.
- Preserve actual Compiler-generated reverse outputs, saved bits, both traversals/seeds, B=0/1,
  T=0, mixed-B reuse/isolation, shared gradients, aliases, checked bounds and session/lifetime safety.
  Reuse existing matrix coverage; add the precise regression, not a new performance matrix.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [lifetime glossary](../../../../glossary.md#term-lifetime); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [regression glossary](../../../../glossary.md#term-regression))</span>
- Class-File verification and structural inspection show the intended carrier/read/loop change,
  unchanged arithmetic and no hidden allocation, boxing, reflection, helper dispatch or fallback.
  Explain how any hoisted facts remain valid for the admitted carriers/ranges/aliases/lifetimes.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Both symbolic identity streams bind changed bytes/meaning; retain historical v1/v2 pins and
  pin actual v3 keys/compatibility/Class-Files when advanced. All static/ordinary witnesses pass
  unchanged. If static bytes must change, stop for scope review, never update their pins to pass.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [scope glossary](../../../../glossary.md#term-scope))</span>
- Final actual code and all three phase results are reported honestly, including residual gap,
  regression, incomplete C2/decode coverage or censorship. Correctness alone does not close the
  requested performance issue; do not label a partial result parity or silently mark it solved.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#term-residual-meanings))</span>

## [Validation](../../../../glossary.md#term-validation)

Configured [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26; worker focused development checks, then ONE final affected [CPU](../../../../glossary.md#term-cpu) test/[Javadoc](../../../../glossary.md#term-javadoc)
gate after source/Javadoc freeze. Preserve XML/[logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) before performance replaces task output.

```bash
./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuSymbolicScanTimingMethodologyTest' --tests '*CpuSymbolicScanPhaseDiagnosticTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --console=plain
./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test :backends:cpu:javadoc -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --console=plain
JDK_JAVA_OPTIONS='--add-modules=jdk.incubator.vector' ./gradlew --no-daemon --no-parallel --max-workers=1 :testing:backend-conformance:test --tests '*CpuSymbolicScanConformanceTest' :testing:integration-tests:test --tests '*EngineCpuSymbolicScanIntegrationTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
```

MAIN owns the third command once and the final fresh measured test [JVM](../../../../glossary.md#term-java-virtual-machine-jvm). At most ONE development
[C2](../../../../glossary.md#term-c2) [capture](../../../../glossary.md#term-capture-meanings) is allowed only if the source cause needs it; worker coordinates with MAIN BEFORE
launch. No concurrent performance tasks, automatic retry, extra [fork](../../../../glossary.md#term-fork-meanings) or adaptive extension.
MAIN creates an external `CPU0041_EVIDENCE` directory and targeted [compile](../../../../glossary.md#compile)-command file for actual
post-fix generated [identities](../../../../glossary.md#term-identity-meanings), unchanged Java phases/original joint [oracle](../../../../glossary.md#term-test-oracle) and effective wrappers.

```bash
JDK_JAVA_OPTIONS="--add-modules=jdk.incubator.vector -XX:+UnlockDiagnosticVMOptions -XX:+LogCompilation -XX:LogFile=$CPU0041_EVIDENCE/hotspot-%p.xml -XX:CompileCommandFile=$CPU0041_EVIDENCE/compile-commands.txt" ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --rerun --tests '*CpuSymbolicScanPhaseDiagnosticTest.boundedForwardReverseAndJointDiagnostic' -Dsynaptik.cpu.dynamicBatch.performance=true -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
```

Keep fixed 30s body/three rows, 3000 startup pairs + 1000 warm pairs + 3×300 paired windows per
row; normal [compiler](../../../../glossary.md#term-compiler) policy, no global PrintAssembly. Record every forward/reverse/joint window,
[median](../../../../glossary.md#term-median), entry/[admission](../../../../glossary.md#term-admission)/setup costs and source/command [hashes](../../../../glossary.md#term-hash). CPU0040 medians G/D were
1.000082190×/3.694822522×/2.208052723× respectively, diagnostic not universal [targets](../../../../glossary.md#term-target).
Capture addressed MachCode/BCIs and actual normal C2 version/inline/invalidation versus windows;
use the existing external LLVM decoder offline, no install or extra JVM. Report code improvement
and capture limitations, not exact cycle attribution. Stop when the authorized budget is consumed.
Validate four [immutable](../../../../glossary.md#term-immutable) hashes, scoped diff, links/anchors, brief size and `git diff --check`.

## Documentation and review impact

Peirce independently approved final source/tests/[Javadoc](../../../../glossary.md#term-javadoc)/docs and the final evidence audit with
no findings/P1/P2; successful executable evidence was reused. MAIN coordinated the three current
symbolic-v3 paragraphs in root Core invariants, [backend](../../../../glossary.md#backend) [CPU](../../../../glossary.md#term-cpu) routes and the general-scan explanation.
Existing Javadoc remains accurate with no Java comment edits; final generation passed afterward.
Targeted guide/glossary review required no broader rewrite, glossary change or [ADR](../../../../glossary.md#term-architecture-decision-record-adr). [Planning](../../../../glossary.md#term-planning)/general
profiles govern this closure; no architecture redesign, new type/[API](../../../../glossary.md#term-application-programming-interface-api)/[Runtime](../../../../glossary.md#term-runtime) [handoff](../../../../glossary.md#term-handoff) or unsafe route.

## Result

Status: Complete — MAIN accepted 2026-10-26; Peirce final source/docs and evidence audits APPROVE, zero findings/P1/P2.
Changed: `CpuGeneralScanEmitter` [immutable](../../../../glossary.md#term-immutable) entry receivers for demanded port-backed cuts,
`CpuGeneralScanIr` symbolic v3, two existing symbolic generated/methodology tests, three version-doc
paragraphs and this brief/master/roadmap: exactly 10 paths against the pre-plan [snapshot](../../../../glossary.md#term-snapshot).
No [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-emitter/new type/[API](../../../../glossary.md#term-application-programming-interface-api)/[Runtime](../../../../glossary.md#term-runtime)/numerics/resource change; checked [FFM](../../../../glossary.md#term-ffm) remains intact.
Both streams reject symbolic v1/v2; [canonical](../../../../glossary.md#term-canonical-form) v1/family 71/static v4 and four immutable files retained.
Controlled source/[bytecode](../../../../glossary.md#term-bytecode) comparison isolates receiver substitution/dead copies, not an internal [C2](../../../../glossary.md#term-c2) [phi](../../../../glossary.md#term-phi) proof.
Focused: initial 53 PASS; final 64 first yielded 62 PASS/1 skip/1 test-only WrongMethodTypeException;
corrected void `invokeExact` lambda targeted 1 PASS gives 63 PASS/1 skip combined, then full gate green.
[CPU gate](</private/tmp/synaptik-cpu0041-worker.jRJgxS/gate-handoff.txt>): ONE final command, 5m25s, 222 suites,
1270 tests = 1229 PASS/41 opt-in skips/0 failures/errors; [Javadoc](../../../../glossary.md#term-javadoc) PASS, 94 warnings, no warning repair.
MAIN named join: 6s, two fresh [conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) tests PASS, zero skips/failures/errors.
Final fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) 52779: timestamp 2026-10-26T05:28:22.535Z, build 18s, body 14.778689s/30s;
all 18×300 windows complete, zero censorship/overshoot. One DEV + one FINAL measured JVM; worker zero.
Final [median](../../../../glossary.md#term-median) G/D microseconds: F 218.134720/218.042363333 (1.000423572×),
R 173.630833333/173.064443333 (1.003272712×), J 393.047083333/388.821110 (1.010868683×).
Final normal C2 4060 dW/dX has ordered four-way [guard](../../../../glossary.md#term-guard)-free backedges matching Java 3940/joint 4108;
all five current normal main-code intervals have zero unknown words; global 796 unknown bytes remain.
Fixed T4/[F32](../../../../glossary.md#floating-point-precision-and-rounding)/H32 B64 native-[scalar](../../../../glossary.md#term-scalar) gap resolved: bounded observed near-Java, not universal speed parity,
exact cycle attribution, sampled-PC proof or blanket decode success; protocol/oracles unchanged.
Full 2476-file freeze matched before/after gates; HEAD/origin/empty index unchanged. Docs/Javadoc final;
no glossary/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) change, unresolved issue or required follow-up. [CPU](../../../../glossary.md#term-cpu) frontier None; no automatic successor.
[Final report](</private/tmp/synaptik-cpu0041-main.F8vADG/result.txt>) links exact commands, raw [metrics](../../../../glossary.md#machine-learning-and-hyperparameters), code addresses and coverage evidence.
