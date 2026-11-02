# Task CPU0055: Verified Scan versus Java measurement

## Status

Complete — MAIN accepted on 2026-10-28 after independent actual-result APPROVE, no open P1/P2.
The one preflight and three measured slots are closed. Performance parity remains unresolved;
CPU0053 stays Cancelled and unchanged. Result retains the wrapper false-negative and continuation.

## Change class

Class C conservatively: generated-code/application binary interface (ABI) comparability and
entry-inclusive performance evidence. Clean off-tree launcher worker; independent prelaunch
and actual-result review are mandatory. No production change.

## Goal and scope

Measure the remaining symbolic-length actual reverse Scan gap on the accepted current source:
verify current bytes and each invocation tuple, then observe the fixed four-target schedule.
Witness: FLOAT32, T4/F3/H2, original forward traversal and actual Compiler-generated opposite
reverse, B7 then B32, length[row]=row%5, joint final-carry/emission seeds. Target indices are
0=retained G4, 1=current G5, 2=equivalent runtime-T/scalar Java E, 3=unchanged literal Java J.
G4 is a diagnostic control, not a production-compatible cache artifact.

## Non-goals

No repository source/test/build change, optimization, new API/capability, architecture/numerical
policy change, native rebuild, Git mutation, commit or push. Do not edit identity stores, replace
historical pins, reopen old selectors/budgets, reset markers or alter historical inputs/results.
No retry, replacement, adaptive warmup, extra campaign, causal or universal-parity claim.
CPU0048–0052 stay Draft; no other CPU source/test/authority/diagnostic frontier is authorized.

## Contracts

Read [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
and [Core invariants](../../../../../ARCHITECTURE.md#core-invariants).
Backend owner: [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) and
[Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
Scan owner: [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic) and
[Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories).
Keep observation separate from production admission; stop on missing/ambiguous contracts.

## Dependencies and integration

- Depends on: CPU0045, Compiler0016, CPU0047 and accepted [CPU0054 Result](0054-scan-classfile-and-abi-verification.md#result) Complete; existing successful semantic/module evidence.
- Conflicts with: any source/class/fixture/oracle writer, repository build or concurrent measurement.
- Parallel group: None; one clean off-tree writer, serial JVM bodies, read-only reviews only.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty source `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0055-base.json`, unchanged from `cpu0054-accepted.json` in that directory. The measured Scan remains local, not published by CPU0047.
- Integration order: MAIN final Ready/frontier release → clean launcher preparation/compile only → independent source/protocol/compile-evidence APPROVE → MAIN executable release → at most one untimed preflight → three serial fresh measured JVMs → independent actual-result review → MAIN closure.
- Integration validation: CPU0055 frozen-input/ABI/output/schedule/disposition checkpoint below, then MAIN preservation audit and shared-document Markdown checks; no repeated module/full suite without a concrete new risk.
- Shared-document integration owner: MAIN; worker writes only the new off-tree root, not this brief/master/roadmap.

Ready verification (MAIN, 2026-10-28): CPU master/roadmap synchronized to CPU0055 Ready as the
sole observational frontier; CPU0053 Cancelled is preserved. CPU0045, Compiler0016, CPU0047 and
CPU0054 are Complete with required review evidence. MAIN's 2,495-file baseline/source/evidence
audit passed: source unchanged from CPU0054 accepted, HEAD=origin/main at the recorded revision,
index empty. No active competing benchmark or writer; the idle Gradle daemon remains untouched.
MAIN accepted the bounded scope/protocol, including direct-main orchestration. Final dependency,
base and conflict verification is complete before worker launch. All historical budgets stay
closed; CPU0055 is a distinct finite grant and authorizes no other CPU frontier.

## Files and artifacts

Read these unchanged files in `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`:

- `CpuGeneralScanTimingTest.java`: `cpu0045Body`, check/batch/admit/mark/output protocol.
- `CpuSymbolicLengthScanFixture.java`: `Family`, numeric resources and captured/prebound actions.
- `CpuSymbolicLengthScanEquivalentJavaOracle.java` and `CpuSymbolicLengthScanCleanJavaOracle.java`: `reverseFloat32` and forward saved-site setup.

Read-only protocol inputs: `/private/tmp/synaptik-scan-continuation.ajgLjq/` launcher source,
`protocol.txt` and runtime classpath; [CPU0053](0053-scan-java-continuation-measurement.md) only for
the unresolved campaign protocol, never execution authority. Accepted verification root:
`/private/tmp/synaptik-scan-identity-verification.MFkNvn/`, especially `probe/` current/historical
classes, identities and tuple records, `analysis/classfile-comparison.json`, comparison script
and `independent-review.txt`. Reuse these results; do not rerun the causal metadata control.

Write only `/private/tmp/synaptik-scan-measurement-verified.LLDIqd/`: new main launcher
`src/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/Cpu0055VerifiedScanMeasurement.java`,
isolated classes, runtime-classpath, protocol/freeze/commands/compile evidence and new
`preflight`, `repeat-1`, `repeat-2`, `repeat-3` slots. All slot work/tmp/jit/output directories
exist before JVM launch. Compile only the off-tree launcher against existing pinned classes;
no repository compilation/configuration, Gradle invocation, source overlay or new runtime type.

## Acceptance criteria

1. Freeze current source/runtime classes, JDK executables/release, classpath order/origins,
   launcher, protocol and historical inputs with fresh hashes. Preserve MAIN's accepted baseline;
   MAIN owns complete pre/post preservation audits rather than duplicating them in the worker.
2. Every body saves freshly generated current G5 bytes, both identity streams and facts before
   comparison assertions. Require exact SHA-256
   `30477241d7603cfec1c2f46079739a5ac10c2b1292dba3fb58aeaa2b05884330`.
   Also repeat CPU0054's complete-byte comparison to historical v5
   `b119b85deb07068761d9755185b4d68758ddb90420b08eb7d0d7b19b1f79f02c`:
   replace exactly its one generated binary-name occurrence only; all 17,302 bytes must match.
   Keep actual names/identities distinct and compare current streams to accepted current0054
   streams. Never use CPU0053's simplistic v5→v4 identity-string equality as ABI proof.
3. Before any G4 invocation, independently check ClassFile validity/static entry descriptor,
   19 ordered semantic slots/dtypes, each target's exact 202-long numeric pack and spans at
   B7/B32 against retained G4 mapping. Own scratch must be 8*B bytes (56/256), range [0,2*B),
   T=4 and explicit B. Retain scalar bits, length facts and independent writable carriers/packs.
   G4 SHA-256 stays `beee86e2474d5a276a731726eff6b895c13d814b9f5788db65a58045ee60428c`;
   use only the existing authenticated hidden-class diagnostic seam, never cache admission.
   Any unproved tuple or mismatch stops the campaign; no adapter, dropped target or repair.
4. Preserve E/J source and loaded-class pins from CPU0053's frozen protocol. Mechanically retain
   its timed actions, batch/admit/mark body, witness, rotations and raw side-row format. Generated
   G4/G5 keep identical MethodHandle/spreader/Runnable capture; E/J remain prebound direct calls.
   Only current authentication, fresh paths and standalone orchestration/disposition may change.
5. Fresh cold ABI and ordinary raw-output checks precede any timing in each body; a separate
   untimed preflight is permitted at most once and has zero warmup/clocked calls. Use the existing
   forward/direct-forward saved-site comparisons, five raw-equal cotangents, inactive dX +0,
   empty-row raw final seed and unchanged input checks. Verify latest measured outputs before
   rerunning checks; retain the existing after-timing NaN-poisoned-padding check and hashes.
6. Three serial fresh JVMs only. Per B: 100 warmup + 300 measured rounds, 100 calls/target/round,
   target=(round+position)%4. Ceiling: 320,000 clocked calls/JVM, 960,000 total including warmup.
   No JIT-driven extensions, extra calls/batches or retries. Untimed setup/check calls remain
   bounded by the original body and deadline, separately counted, never hidden as extra warmup.
7. Each body has a 60,000,000,000 ns admission deadline including cold checks and persistence.
   Retain per-call admission and partial-batch records; an in-flight call may overrun. Use
   CREATE_NEW launch/body markers and outputs. Failure, incomplete schedule, kill or persistence
   overrun consumes that opportunity and is failed/censored, never replaced. A preflight failure
   stops timing. Keep the exact exit code, report, disposition and post-persistence override;
   MAIN records externally killed/missing dispositions. No silently successful partial result.
   A wrapper-only false negative may be reclassified from retained evidence only after independent
   review: require the exact child PID, complete matching JIT XML and all existing exit/body/raw
   guards. Keep the original false rejection, bind the corrected driver/admission separately, and
   release only the three still-unused measured slots. Do not relax any numerical or ABI guard.
8. Retain every warmup/measured `side B=...` record with round/warm/position/target, begin/end
   uptime and nanoTime, elapsed and actual calls; retain unchanged `phase=...` brackets,
   before/latest/after-poison correctness hashes, cold compile/generate/define times, loaded
   names/origins and fresh per-slot JIT XML. Zero exit is necessary, not sufficient for success.
9. Report per-JVM/B absolute latencies and paired G5/E, G5/J ratios; G4 is control. Preserve all
   raw observations, including nonstationary/censored cases. Correlate JIT windows only when
   normal-nmethod/phase evidence qualifies; otherwise report raw timings only. Do not subtract
   harness cost or infer MemorySegment/code-size/JIT causality. Direct-main replaces a Gradle
   worker only as orchestration; disclose it and do not pool historical timings.

## Validation

Worker first prepares and compiles only; independent review inspects source, protocol, fresh
freeze and compile evidence before MAIN releases any launcher execution. Exact javac/Java
executable, classpath, arguments, environment and working directories are frozen in the artifact
root's `commands.json`/`commands.txt`; results retain the actual child command and exit.
The preparation command is `node <root>/launcher/run.mjs compile`; the single retained preflight
used `node <root>/launcher/run.mjs preflight`. The separately reviewed corrected evidence driver
may run only `node <root>/launcher/continue.mjs repeat-1`, then `repeat-2` and `repeat-3`, once each.
It preserves the frozen Java command and requires a separate hash-bound MAIN release. No compile
or preflight repetition is allowed. Work/tmp/jit/output paths precede every JVM launch.

Use JDK26.0.1, vector module, `-XX:+TieredCompilation`, diagnostic `LogCompilation`, per-slot
`hotspot_%p.log`, UTF-8, US locale, slot tmpdir and assertions as frozen in the command file. No explicit
Xmx/Xms, heap/threshold/compiler tuning or injected JVM options: preserve heap ergonomics. Record
actual VM arguments/version/pid, working/tmp paths and standalone exit/disposition. These are
standalone checks, **not JUnit validation**; no JUnit report is expected or claimed.

Reuse accepted CPU0054 raw correctness and successful existing affected-module evidence; fresh
pre-timing guards above remain mandatory. Independent actual-result review inspects artifacts,
counts, commands, limitations and Result without rerunning successful tests. MAIN alone checks
final baseline/history preservation and shared-document links/anchors/whitespace, including
`git diff --check`; no full repository/build/native validation is warranted for observation only.

## Documentation and review impact

Planning/General profiles apply. MAIN owns final brief Result, CPU master and roadmap integration;
no separate plan-review cycle. Launcher Javadoc must explain inputs, bounded schedule, ownership,
failure/censoring and orchestration departure. Documentation impact: no API/implementation
Javadoc, architecture or glossary change; existing behavior, contracts and domain terms remain
unchanged. Required independent prelaunch and actual-result reviews are execution gates.

## Result

Status: Complete

Direct JDK26 javac passed (initial failed compilation retained); standalone Java preflight and
three serial measured JVMs exited 0, complete/uncensored. All 9,600 side rows satisfy the schedule:
960,000 clocked calls including 240,000 warmup and 720,000 measured calls; 192 functional calls
include the single preflight. Raw/NaN/ABI checks and current G5 pins passed. No extra benchmark JVM/retry.
The old wrapper falsely rejected `hotspot_pid9930.log`; its failed exit/censor records remain.
Independent review confirmed the passed preflight; corrected exact-PID/log/failure guards and a
separate hash-bound MAIN release admitted only the three unused slots. No preflight was repeated.

Paired-median ratios across three JVMs: B7 G5/E 1.1789–1.1948, G5/J 1.3987–1.4360;
B32 G5/E 1.3770–1.3869, G5/J 1.7380–1.7779. G5 per-call medians span respectively
765.415–865.840 ns and 2944.170–3501.250 ns. Evidence and exact commands: artifact root
`commands.json`, `launcher/*review*`, `repeat-{1,2,3}/` and `raw-summary.json`; MAIN's
`summarize.mjs` validates schedule, exits, dispositions and computes each JVM separately.
These are entry-inclusive raw observations, not isolated-loop, JIT-cause, universal parity or
historical speedup claims. Java/bytecode remained unchanged; standalone checks are not JUnit.

MAIN preservation: 3,945 old inputs/evidence unchanged; HEAD/origin/index unchanged, only this
brief/master/roadmap changed. Existing module evidence reused; no production/build/API/Javadoc,
architecture/glossary, native or Git mutation. Timing budget is closed; CPU0048–0052 remain Draft.
Independent actual-result APPROVE, no P1/P2; 396 local links/anchors and whitespace checks passed.
