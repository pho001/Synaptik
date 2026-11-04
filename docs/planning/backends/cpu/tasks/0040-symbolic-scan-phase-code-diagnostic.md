# Task CPU0040: [Symbolic Scan](../../../../glossary.md#symbolic-scan) phase and optimized-code diagnostic

## Status

Complete — MAIN accepted the bounded diagnostic only after final independent approval.
The original performance issue remains unresolved. [CPU](../../../../glossary.md#term-cpu) production source/test frontiers: None;
no additional measurement, production fix, [commit](../../../../glossary.md#term-commit)/push or successor is authorized.

## Change class

Class C conservatively — test-only generated hot-path evidence, no production/[invariant](../../../../glossary.md#term-invariant) change.
Clean implementation and independent targeted review; MAIN owns measurement/shared documents.

## Goal

Localize the generated/optimal-Java gap for [TANH](../../../../glossary.md#term-tanh) T4/[F32](../../../../glossary.md#floating-point-precision-and-rounding)/H32, B64 only: forward, actual [Compiler](../../../../glossary.md#term-compiler)
reverse and joint work, then actual optimized code. Prior joint 2.351869× is context, not a [target](../../../../glossary.md#term-target)/cause.

## [Scope](../../../../glossary.md#term-scope) and non-goals

Add at most two test-only types in existing `internal.codegen.emit`: `CpuSymbolicScanPhaseOracle`
and `CpuSymbolicScanPhaseDiagnosticTest`. Reuse one `CpuSymbolicScanTimingTest.PreparedFamily`
and its `Fixture.generatedStep(0/1)` in every phase/joint row, with no regeneration or [rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction).
Extract literal `forward`/`reverse` methods from [immutable](../../../../glossary.md#term-immutable) `CpuSymbolicScanCleanJavaOracle.t4f32h32`;
retain native [FLOAT32](../../../../glossary.md#term-float32) [scalar](../../../../glossary.md#term-scalar) [MemorySegment](../../../../glossary.md#term-memorysegment) work, dynamic B, literal 1f/0f, represented saved
[activations](../../../../glossary.md#term-activation), separate public/saved writes, contributor/producing-site order and dW before dX.
No production, existing [oracle](../../../../glossary.md#term-test-oracle)/timing/methodology, build flag, [API](../../../../glossary.md#term-application-programming-interface-api), [ABI](../../../../glossary.md#term-application-binary-interface-abi), [identity](../../../../glossary.md#term-identity-meanings), route, resource,
numerics or architecture edits; no installation, extra [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), [contraction](../../../../glossary.md#contraction-dimension)/other-[shape](../../../../glossary.md#shape)/B/packing/[Engine](../../../../glossary.md#term-engine)
campaign, speculative fix or forced [compiler](../../../../glossary.md#term-compiler) policy.
Packed [runtime](../../../../glossary.md#term-runtime)-long T remains frozen; phase results cannot prove a fixed-time-fact cause.

## Contracts

- Root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>
  and [Core invariants](../../../../../ARCHITECTURE.md#core-invariants): existing [ownership](../../../../glossary.md#term-ownership-meanings)/[identity](../../../../glossary.md#term-identity-meanings)/semantics.
- [Backend](../../../../glossary.md#backend) [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>: passive evidence and optimal Java [oracle](../../../../glossary.md#term-test-oracle).
- [Accepted symbolic-batch scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>,
  all four subsections: actual reverse, saved sites, numeric resources and qualification gates.
Stop for any contract ambiguity or required production expansion; no new contract/ADR is selected.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Dependencies and integration

- Depends on: [CPU0039](0039-symbolic-scan-fixed-geometry-specialization.md#result),
  [Prepare0014](../../../modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result),
  [CPU0037](0037-general-scan-dense-read-views.md#result); all Complete with final reviews/gates.
- Conflicts with: generated Scan/fixture/oracle/timing/protocol/shared-document writers;
  inactive, previous agents closed, exclusive serial writer only.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [fixture glossary](../../../../glossary.md#term-fixture); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Parallel group: None; implementation then independent review, no concurrent write tasks.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted uncommitted 2472-file baseline, not merged predecessors.
- Integration order: Ready verification → clean test implementation/focused checks → independent
  fairness/Javadoc approval → MAIN source/protocol freeze → one measured JVM/offline decode → closure.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [Javadoc glossary](../../../../glossary.md#term-javadoc); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Integration validation: focused semantic/methodology and single diagnostic commands below,
  source pins/scope, complete samples and correlated optimized-code audit.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Shared-document integration owner: MAIN; exclusive serial delegation of this brief, CPU master
  and roadmap current sections. No completed predecessor rewrite.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu))</span>

[Admission](../../../../glossary.md#term-admission) verified 2026-10-25: approval, Complete predecessor Status/Result/reviews/gates,
stable contracts/sole test frontier, HEAD/origin and empty index checked.
All 2472 [hashes](../../../../glossary.md#term-hash) match `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0040-pre-plan.json`;
no competing writer. MAIN confirmed address-tagged MachCode [capture](../../../../glossary.md#term-capture-meanings) and local LLVM decoding.

## Files and symbols

Test base: `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`.
Only add the two named types; original `CpuSymbolicScanTimingTest` (PreparedFamily/[Fixture](../../../../glossary.md#term-fixture)),
`CpuSymbolicScanCleanJavaOracle` and `CpuSymbolicScanTimingMethodologyTest` remain read-only.
Frozen [SHA-256](../../../../glossary.md#term-sha-256): [oracle](../../../../glossary.md#term-test-oracle) `8d6abed866ee4984b3286b1d36e03ea72f447c33a1da162dea6547c2d17530a1`;
timing `506a5a6bb949bcac8e9e4c9bc16a1f4421f816477a691d92384858f59fe443a9`.
MAIN's parser, print-command file, [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), bytes and decoded assembly live outside the repository.

## Acceptance and frozen protocol

- Before timing, poison/check repeated phase composition against immutable joint Java: raw
  all-five outputs and both public/saved histories; exact paired footprints and no workspace.
  Check generated phase/composition with existing numerical comparisons and exact represented
  saved-site integrity; independent reverse consumes pre-established valid saved history.
  Inputs/seeds/constants stay unchanged; no timed activation replay, extra fixture/setup copy,
  reset, allocation or check. Mandatory oracle final-carry <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [integrity glossary](../../../../glossary.md#term-integrity); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [constants glossary](../../../../glossary.md#term-constant); [activation glossary](../../../../glossary.md#term-activation); [fixture glossary](../../../../glossary.md#term-fixture); [carry glossary](../../../../glossary.md#term-carry))</span>`MemorySegment.copy` remains timed.
- One fresh measured JVM, one nonpreemptive 30s whole-test-body admission cap including setup/checks/diagnostics. Fixed
  row order forward → reverse → joint; each row 3000 alternating G/D startup pairs, then 1000
  warmup pairs, then three paired 300-call windows in G-D/D-G/G-D order: 18 windows total.
  Check the deadline before each logical invocation: one phase entry, or the entire joint work
  (two generated entries versus one immutable Java call). Do not add a mid-joint clock check.
  An admitted whole call may finish after the deadline; censor its window and report overshoot.
  Admit no new logical call at/after the deadline. Report every count/censorship; no partial
  successful means, retries or adaptive warmup. This clarifies granularity without adding calls
  or budget; counts/order/warmup/shapes/flags stay frozen. Read-only post-checks remain mandatory.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [warmup glossary](../../../../glossary.md#term-warmup); [shapes glossary](../../../../glossary.md#shape))</span>
- Forward/reverse use the same two retained generated methods and extracted Java phases;
  joint invokes those same generated methods versus the immutable original joint Java control.
  No extra split-joint timing row. Report all windows/medians, one spreader/entry per phase versus
  one Java call, two generated entries versus one joint Java call, and common Runnable/clock costs.
  Setup separate; not pure-loop/Engine costs. Never equate phase sums with the original joint control.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Keep normal compiler/inlining policy: no dontinline, forced tier, -Xbatch, compileonly or retune.
  Record actual inline context and normal-entry C2 availability/invalidation versus sample boundaries;
  split Java may inline unlike the original joint. Inspect effective compiled wrappers when inlined;
  differing compilation contexts preclude exact causal percentages or a parity claim.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [inlining glossary](../../../../glossary.md#term-inlining); [C2 glossary](../../../../glossary.md#term-c2); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- MAIN [captures](../../../../glossary.md#term-capture-meanings) LogCompilation plus targeted `print,*Generated_HASH*::invoke` for forward
  `0016fd0174374d5acb608175a2056a418d7d4859969e5c9d0e9b8d1b0a257a2c` and reverse
  `77709bd97049d2dbf630dfe7317943010421212f500ea0caba55f36b2b4f18c9`, original [oracle](../../../../glossary.md#term-test-oracle)
  `t4f32h32`, and `print,*CpuSymbolicScanPhase*::*` for only the new test classes/wrappers.
  No global PrintAssembly. Preserve full addressed [MachCode] bytes/BCI comments and offline-decode
  AArch64 using <span data-glossary-navigation="true"> ([BCI glossary](../../../../glossary.md#term-bytecode-index-bci); [AArch64 glossary](../../../../glossary.md#term-aarch64))</span>`/opt/homebrew/Cellar/llvm/23.1.1/bin/llvm-mc`; no hsdis/install/extra [capture](../../../../glossary.md#term-capture-meanings) [JVM](../../../../glossary.md#term-java-virtual-machine-jvm).
- Audit actual matching code/version/addresses for loops, addressing, checks, calls and spills;
  distinguish observed instructions, C2 availability and hypotheses, not assumed executed assembly
  or exact cycle attribution. If capture/decode/version coverage fails, report the limitation and
  incomplete optimized-code diagnosis; do not expand the budget. Disclose diagnostic-output overhead.

## [Validation](../../../../glossary.md#term-validation)

Completed focused validation, retained as evidence only; do not repeat successful tests:

```bash
./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --tests '*CpuSymbolicScanPhaseDiagnosticTest' --tests '*CpuSymbolicScanTimingMethodologyTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --console=plain
```

MAIN froze exact source [hashes](../../../../glossary.md#term-hash), external evidence path/print directives and command before launch.
`CPU0040_EVIDENCE` is `/private/tmp/synaptik-cpu0040-main.P3w6A4`; print file [targets](../../../../glossary.md#term-target) only the above.
Only `boundedForwardReverseAndJointDiagnostic` uses the existing performance gate; semantic checks
are always on. MAIN alone ran exactly once (evidence only, no repeat authorization):

```bash
JDK_JAVA_OPTIONS="--add-modules=jdk.incubator.vector -XX:+UnlockDiagnosticVMOptions -XX:+LogCompilation -XX:LogFile=$CPU0040_EVIDENCE/hotspot-%p.xml -XX:CompileCommandFile=$CPU0040_EVIDENCE/compile-commands.txt" ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --rerun --tests '*CpuSymbolicScanPhaseDiagnosticTest.boundedForwardReverseAndJointDiagnostic' -Dsynaptik.cpu.dynamicBatch.performance=true -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
```

Reuse CPU0039's 1264-test [CPU](../../../../glossary.md#term-cpu) and named two-test [conformance](../../../../glossary.md#term-conformance)/integration evidence for frozen production;
no full CPU/root/native rerun. Validate pins/[scope](../../../../glossary.md#term-scope)/links/anchors/size and `git diff --check`;
archive focused XML before timing replaces it. Review does not repeat tests.

## Documentation and review impact

Independent clean fairness/[Javadoc](../../../../glossary.md#term-javadoc) review must approve final executable tests before the [run](../../../../glossary.md#term-run):
literal extraction, raw composition, saved history, footprints, retained methods, [compiler](../../../../glossary.md#term-compiler)-context
limits, counts/cap and [capture](../../../../glossary.md#term-capture-meanings) [provenance](../../../../glossary.md#provenance). Document all new test method inputs/results/failures;
existing Javadocs remain accurate. [Planning](../../../../glossary.md#term-planning)/general profiles apply; no public guide/contract/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)
or glossary meaning changes. MAIN records review and measured limitations in this brief only.

Boyle final core fairness/Javadoc APPROVE, zero open P1/P2, covers the current two-test freeze
and exact console/comment-only [admission](../../../../glossary.md#term-admission)-granularity clarification. No functional/protocol change;
successful focused tests are reused, not repeated. MAIN re-pinned final sources before measurement.

## Result

Status: Complete — MAIN accepted bounded diagnostic only after Boyle final independent APPROVE,
2026-10-25 22:35 UTC, zero open P1/P2. The original performance issue remains unresolved.
Changed [scope](../../../../glossary.md#term-scope): two named new test types plus this brief/[CPU](../../../../glossary.md#term-cpu) master/roadmap; production frozen.
Boyle final APPROVE, 0 open P1/P2. [Oracle](../../../../glossary.md#term-test-oracle) SHA `e9524585331c553325940e0f980405812e75aa9d51cfd84c15e092a8b1fbf940`;
diagnostic SHA `43dea472afc9ad58898077be3e5c91faaf204b4e89cb3129c50749fe22995c62`.
Focused command above: 9 passed/1 performance-gated skip/0 failures/errors; exact wording-only
diff reviewed, no duplicate test [run](../../../../glossary.md#term-run). Worker evidence: `/private/tmp/synaptik-cpu0040-worker.ylTk7h/handoff.txt`.
MAIN command above: BUILD PASS 24s, 22 tasks/2 executed/20 up-to-date; CPU test freshly executed.
`/private/tmp/synaptik-cpu0040-main.P3w6A4/measurement-command.txt` records its exact expanded command.
`probe.xml`/`probe.log`: 1 test/0 skips/failures/errors at 2026-10-25T22:23:45.615Z; testcase 19.022s,
body 19.015134s/30s; all 18 windows/300 calls complete, no overshoot/censorship; frozen startup/[warmup](../../../../glossary.md#term-warmup).
[Median](../../../../glossary.md#term-median) G/D microseconds (ratio): forward 214.601387/214.583750 (1.000082190), reverse
625.304583/169.238057 (3.694822522), joint 839.323470/380.119307 (2.208052723).
The [sample analysis](</private/tmp/synaptik-cpu0040-main.P3w6A4/analysis.json>) retains every [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); observed excess localizes to reverse, not universal parity,
exact phase percentages or a cause. Split/original-joint [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) contexts are not interchangeable.
Actual measured [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) [log](../../../../glossary.md#term-log): `hotspot-pid32214.xml`; normal [C2](../../../../glossary.md#term-c2) 3968/3969/4061/4062/4111 were [ready](../../../../glossary.md#term-ready) before
matching samples, no later invalidation/deopt; original-joint 4108 invalidation was startup-only.
`caller-inline-audit.json`: generated callers fail inline as too large; reverse [FFM](../../../../glossary.md#term-ffm) helpers inline.
Availability/[inlining](../../../../glossary.md#term-inlining) is not executed-instruction proof; targeted diagnostic output adds overhead.
2475-file pre-probe comparison changed=[]; HEAD/origin/index and nine old source/static pins unchanged.
New test Javadocs finalized; no guide/architecture/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary change or Java rerun.
The [optimized-code report](</private/tmp/synaptik-cpu0040-main.P3w6A4/report.txt>) owns exact addresses/bytes/BCIs.
All five current normal C2 main-code intervals have zero unknown words; reverse 4061/4062 fully
decode 3260/1468 words. Global parser status stays incomplete: 9956 unknown raw bytes across
[C1](../../../../glossary.md#term-c1)/stale blocks plus current stub data; current 3969/4111's 100/76 unknown bytes are 8-byte
trampoline [target](../../../../glossary.md#term-target) literals, not missing main instructions. No complete-whole-blob claim.
Generated reverse dW/dX has +1 contributor/backedge, repeated dH type/bounds/session guards and
SP-relative accesses; split 4062 and measured original joint 4111 have +4 ordered [scalar](../../../../glossary.md#term-scalar) FMUL/FADD
and no FFM guards inside those backedges. Contractions remain scalar, not [SIMD](../../../../glossary.md#simd) [reassociation](../../../../glossary.md#term-reassociation).
Direct [derivative](../../../../glossary.md#term-derivative) vectorization versus generated scalar is secondary/unquantified; forward both
have four-way [guard](../../../../glossary.md#term-guard)-free contributor loops, consistent with the observed near-1× phase result.
Packed T/frame causality is NOT proved: checked dH reads at BCIs 570/763 use row*128+col*4,
not frame/time; frame-dependent X is already guard-free inside dW. No exact cycle attribution.
Smallest unapproved follow-up: source-level test of existing dH [carrier](../../../../glossary.md#term-storage-carrier)/read-proof [shape](../../../../glossary.md#shape) before any fix;
retain checked FFM, arithmetic order, storage/[alias](../../../../glossary.md#aliasing-and-arena) and [lifetime](../../../../glossary.md#term-lifetime). No broad T rewrite/new route/Unsafe.
No required work remains in this diagnostic. Original performance parity remains incomplete;
CPU source/test frontiers None; no new Ready production task, fix, probe, [commit](../../../../glossary.md#term-commit) or push is authorized.
