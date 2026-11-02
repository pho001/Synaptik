# Task CPU0057: Scan body versus equivalent Java offline audit

## Status

Complete — MAIN accepted the independently reviewed offline audit on 2026-10-28.
User approved only this offline diagnostic. No implementation or additional measurement is authorized.

## Change class

Class C conservatively: generated-code hot-path evidence; clean audit and independent targeted report review mandatory.

## Goal

Compare current symbolic-length Scan computation stage by stage with optimal direct equivalent Java:
source intent → ClassFile facts → existing just-in-time compiler (JIT) evidence, without inferring speed from code size.
Use F32, T=4/F=3/H=2, actual Compiler-generated reverse at B=7 and B=32; retain runtime T,
actual scalar operands, checked unaligned access, represented arithmetic and the same semantic algorithm.

## Scope and non-goals

Read-only current source, retained ClassFiles/decompilations and existing JIT logs; `javap` and offline parsers allowed.
No generator/source/oracle changes, witness execution, new workload/benchmark/JIT run, build, test, Gradle, commit or push.
Do not use literal Java as the primary oracle, or slow equivalent Java to match generated code.
Do not reopen closed measurement budgets or audit predecessor histories. CPU0055 earlier logs only if identity correlation remains unresolved.

## Contracts

- [ARCHITECTURE.md — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
- [Backend execution — CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- [Backend execution — Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
- [Recurrent scan — Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- [Recurrent scan — Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic).
- [Recurrent scan — Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories).

Read these headings, AGENTS.md and affected source/tests; stop on contract ambiguity.

## Dependencies and integration

- Depends on: CPU0056 Complete; MAIN confirms no repository change since acceptance; baseline verification complete.
- Conflicts with: source/artifact writers and new benchmarks; inactive at supplied admission; reverify before launch.
- Parallel group: None; source/ClassFile and JIT readers may run in parallel read-only, with disjoint off-tree outputs;
  no isolated worktrees needed and no concurrent repository writers.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty base.
  Snapshot `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0057-base.json`: 2497 files, index empty; preserve unrelated work.
  MAIN captured the 23-file source/artifact manifest as `inputs.json` using `inputs-check.mjs` in the new evidence root;
  historical 4136 unchanged. Reuse this baseline; do not repeat the historical audit.
- Integration order: plan verified → source/ClassFile + JIT read-only audit → independent report review → MAIN closure.
- Integration validation: offline evidence/identity/parser/link/whitespace checkpoint below; reuse semantic evidence, no executable reruns.
- Shared-document integration owner: MAIN alone owns master, roadmap and brief Result; independent review context solely writes durable report after audit.

Launch verification (MAIN, 2026-10-28): sole authorized frontier is CPU0057 offline observation;
source/test, architecture and measurement frontiers remain None. CPU0056 is independently reviewed Complete.
HEAD/origin/index match the base; the 2497-file accepted tree differs only by this brief and synchronized
master/roadmap updates. Contracts and 23 named source/artifact hashes are stable; 4,136 historical inputs
are unchanged. No competing writer/measurement is active. Analysts read repository/retained artifacts
only and own separate off-tree outputs; MAIN owns shared plans, with serial report-review write handoff.

## Files and symbols

- Source root `modules/backend/cpu/src/`; emitter package `io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`.
  Read-only `main/java/` owners: `CpuGeneralScanEmitter`, `CpuGeneralScanMatmulEmitter`,
  `CpuGeneralScanReductionEmitter`, `CpuCarrierEmitter` (all `.java` in that package).
- Primary reference: `test/java/` + that package + `CpuSymbolicLengthScanEquivalentJavaOracle.java`, `reverseFloat32`;
  inspect focused identity/semantic tests only as necessary, without execution. Use loaded E ClassFile retained by CPU0056.
- Current v5 SHA-256: `30477241d7603cfec1c2f46079739a5ac10c2b1292dba3fb58aeaa2b05884330`.
  CPU0054 analysis current-v5 and CPU0056 measured `reverse.class` share this pin.
  Old v5 decompile differs only in generated name, previously proved; preserve that provenance distinction.
- Immutable artifact root `/private/tmp/synaptik-cpu0045.ICUu7M/`:
  `decompiled/v5.java`, `decompiled/equivalent-java.java`, `v5-entry.javap.txt`, `equivalent-entry.javap.txt`.
- Immutable `/private/tmp/synaptik-scan-identity-verification.MFkNvn/analysis/current-v5.javap.txt`.
- Immutable `/private/tmp/synaptik-scan-binding.D2bvwa/`: `summary.json`, `measured-{1,2,3}/report.txt`,
  retained classes and `measured-*-hotspot_pid*.log`; resolve exact E/class/log paths read-only and record hashes.
- New evidence only: `/private/tmp/synaptik-scan-body-audit.MvTjuv/`; never overwrite retained inputs.
  Source/ClassFile reader owns `source-audit.md`; JIT reader owns `jit-audit.md`; preserve `inputs.json` and `inputs-check.mjs`.
- Durable output: `docs/planning/backends/cpu/0057-scan-body-java-audit.md` (independent reviewer sole writer).

## Acceptance criteria

- Record source/artifact hashes and generated/E loaded-class identities; separate decompiled aliases from measured class identities.
- Provide a source/ClassFile comparison matrix for initial copies, activation adjoint, dX, dW, dU, db and dh.
  For each stage cite source lines and method descriptors/bytecode offsets; quantify exact extra address,
  geometry and length loads, loops/branches and temporary reads/writes, including frequency/domain and zero differences.
  Distinguish eliminated source constructs from emitted instructions; do not call bytecode locals hardware spills.
- Explain harmless javac/decompiler differences separately from necessary runtime/semantic differences:
  runtime T/B, scalar operands, activity gates, raw identity, saved sites, contributor order and checked access.
- Label every conclusion bytecode fact, plausible cost, or proven existing JIT evidence; attach precise evidence locations.
  Correlate existing methods, inlining decisions, compiled native methods (nmethods) and deoptimizations by JVM/class/method/compile ID.
  State absent assembly or executed-check information explicitly; no invented bounds-check/spill attribution,
  causal speed claim from code size, or universal parity claim. Do not treat missing log events as proof of absence.
- Give bounded actionable recommendations: one preferred proven code-shape opportunity if supported, with exact site,
  semantic constraints and why direct Java avoids it; otherwise state the precise missing evidence. Cost/speed remains unproved
  unless existing evidence proves it. Recommendations grant neither implementation nor further measurement approval.
- Independent reviewer checks actual sources/artifacts and evidence claims, resolves findings, writes the durable report,
  and records approval/limitations. MAIN alone accepts closure and synchronizes shared planning records.

## Validation and documentation impact

Use document/source/artifact SHA-256 manifests (`shasum -a 256`), reproducible offline parser commands/checks
and recorded outcomes; `javap -c -p -v <retained-class-path>` may inspect bytes without loading/executing witnesses.
Check local links/anchors, `wc -lc` against planning-guide brief guardrails, trailing whitespace and scoped `git diff --check`;
check new/untracked Markdown explicitly. Preserve input hashes; validate report evidence links and parser counts.
Reuse passed semantic evidence and its limits; no module/root validation because no executable behavior changes.
Apply general/planning documentation profiles and documentation-rules; reviewer performs targeted glossary search.
No architecture, public API, Javadoc or glossary-term changes: only diagnostic evidence/report and planning records.

## Result

MAIN accepted independent APPROVE, no open P1/P2. The [combined report](../0057-scan-body-java-audit.md)
compares initialization, activation adjoint, dX/dW/dU/db/dh and existing JIT evidence. Only this brief,
that report, CPU master and roadmap changed; source/oracles/ClassFiles/architecture remain unchanged.
Current G/E have 952/660 instructions and 2147/1406 code bytes. In the admitted F32 witness,
represented arithmetic, length reads, tensor/work-bank accesses and body traversal counts match;
four generated pack loads occur once per invocation, not inside the body loops.
Proved code-shape opportunity: two accesses add a constant-zero scratch base, 44/244 additions
at B7/B32. Inactive dW/dU paths advance cursors 340/1340 times; E skips those address expressions,
but G saves active-path multiplications. Neither observation proves native cost or a speed gain.
Existing logs: 34 target tasks, 28 successful C2 including six normal-entry versions; all normal
tasks have no inline failures. No assembly/sample-to-version timestamps prove executed checks,
spills or measured-window C2 readiness. The timing-gap cause therefore remains unresolved.
Independent review corrected activation load/store wording while confirming extra-local deltas.
Offline source/JIT count/identity checks and report links/examples/whitespace passed; commands and
evidence are in the report and `/private/tmp/synaptik-scan-body-audit.MvTjuv/review-final.txt`.
MAIN's combined documentation/preservation checkpoint passed: 23 pinned inputs and 4,136 protected
historical/source inputs unchanged, HEAD/origin/index unchanged. Existing semantic evidence reused;
no builds/tests/witness/JIT/benchmark runs, implementation, commit or push were performed.
Documentation profiles/glossary/Javadoc reviewed; no API, terminology or Javadoc change needed.
Preferred possible successor is bounded zero-base omission with proper identity/validation handling;
it is not authorized here. All frontiers close; no further work is required for this offline scope.

Status: Complete
