# Task CPU0054: Verify current Scan ClassFile and historical ABI

## Status

Complete — MAIN accepted the untimed verification on 2026-10-28 after independent actual-result
review APPROVE, with no open P1/P2 findings. CPU0053 and all performance slots stay closed.

## Change class

Class C conservatively: generated-code, identity and diagnostic entry-ABI verification.
Clean probe context and independent targeted actual-result review; no production change.

## Goal and scope

Retain the actual current symbolic-length FLOAT32 reverse ClassFile before any comparison gate.
Explain the CPU0053 expected `b119b85…` versus observed `3047724…` mismatch using actual bytes,
not a guessed identity update. Compare current and retained v5 after normalizing only the generated
binary name, inspect instructions, and check descriptor/slot/pack/span/scratch compatibility with
retained v4 at T4/F3/H2, B7 and B32. Verify outputs against the existing equivalent and literal Java
references, including poisoned inactive padding, with independent resources and bounded calls.

## Non-goals

No benchmark, warmup, measured loop, performance ratio, JIT-causality claim, cache pin replacement,
production/source-test/build edit, numerical policy change, native rebuild, commit or push.
Do not resume or modify CPU0053's failed slot, frozen inputs or historical evidence.

## Contracts

Read [root authority index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
and [Core invariants](../../../../../ARCHITECTURE.md#core-invariants).
Backend owner: [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
and [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
Scan owner: [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic)
and [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories).
An old artifact may run only through the existing explicit diagnostic seam after its exact ABI is
proved; it must not enter production cache admission.

## Dependencies and integration

- Depends on: Compiler0016 and CPU0047 Complete; cancelled CPU0053's retained failed preflight
  explicitly supplies the unresolved evidence, not a renewed timing grant.
- Conflicts with: any source/class/fixture writer or measurement.
- Parallel group: None. MAIN handles read-only baseline/byte analysis and shared documents;
  the clean worker exclusively writes its new off-tree probe and outputs.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1` plus accepted
  dirty source snapshot `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0054-base.json`.
- Integration order: verified brief → clean untimed probe → byte/ABI/output analysis → independent
  actual-result review → MAIN documentation/closure.
- Integration validation: compile/run only the off-tree probe with installed JDK26.0.1 and
  vector module; actual ClassFiles, normalized disassembly/byte comparison, exact numeric packs,
  span/slot records, raw-output checks, source/evidence preservation and Markdown checks.
- Shared-document integration owner: MAIN.

Ready verification: MAIN checked open HEAD/origin/index, current source and failed evidence,
existing successful semantic gates, cancelled measurement status and absence of active workers.
The latest request permits this verification, not a production fix or timing campaign.

## Files and artifacts

Write only `/private/tmp/synaptik-scan-identity-verification.MFkNvn/` for probe code, compiled
classes, copied actual artifacts, invocation records and analysis. Reuse read-only classpath
`/private/tmp/synaptik-scan-continuation.ajgLjq/runtime-classpath.txt` and its fresh 1,204-input
manifest. Read existing CPU `CpuSymbolicLengthScanFixture`, both Java oracles, current generator
and IR; retained v4 ABI data and CPU0045 v5 artifacts; Compiler0016 identity review as context.
No new repository Java type or runtime abstraction is introduced.

## Acceptance and validation

1. Persist actual current ClassFile, both identity streams and facts before assertions can stop
   the probe. Report deterministic hashes and concrete binary names.
2. Compare complete current/retained v5 bytes after replacing only the old generated binary name;
   keep actual `javap -c -p` output. If more differs, report it rather than treating a hash update
   as proof. Separate source-graph identity changes from emitted instructions and arithmetic.
3. Check old/current descriptors, semantic argument order, numeric packs and exact spans/scratch
   separately for B7/B32. Keep old/current identities distinct. If G4 ABI is not proved, do not run it.
4. Run compatible targets once per ordinary/poisoned case only; no warmup or clocked batch.
   Preserve output equality and exact inactive/padding facts. Save diagnostics even on failure.
5. Independent review checks actual artifacts/analysis and limitations without rerunning passing
   executable checks. No full-module rerun; this task changes no executable repository source.
6. If straightforward, one metadata-only controlled generation may restore only Compiler0016's
   six former rank-case SUM_TO_SHAPE calls in an off-tree copy of the current gradient rule.
   Keep current lexical-binding API, compile to an isolated overlay, do no numeric target work,
   and compare the resulting artifact with historical v5. This is a causal diagnostic, never an
   installed rule or source fix; preserve the exact copied source/diff and invocation.

## Documentation and review impact

Planning profile for this brief/master/roadmap; detailed evidence off-tree. Probe Javadoc explains
its bounded diagnostic protocol. Existing API/architecture/glossary semantics remain unchanged.

## Result

Status: Complete

The actual current ClassFile is `30477241d7603cfec1c2f46079739a5ac10c2b1292dba3fb58aeaa2b05884330`;
retained v5 is `b119b85deb07068761d9755185b4d68758ddb90420b08eb7d0d7b19b1f79f02c`.
Both are 17,302 bytes. Replacing exactly one generated binary-name occurrence makes complete
ClassFile bytes identical; actual normalized `javap -c -p` has the same 952 instructions.
Both identity streams differ only in their one 64-character canonical-IR fingerprint; all other
identity/geometry bytes are unchanged. No arithmetic, loop, helper, allocation or dataflow change
is hidden by the binary-name normalization.

The off-tree control restored only six former rank-case SUM_TO_SHAPE calls in the current
gradient rule, retained its lexical binding/normalizer, and regenerated the complete historical
v5 bytes exactly. This confirms Compiler0016's identity effect in this witness, not just an inference.
Old G4 and current G5 have the same 19-segment entry, exact 202-long packs, semantic slots and
spans at B7/B32; scratch is respectively 56/256 bytes. Four targets G4/G5/equivalent Java/literal
Java produced raw-equal five cotangents in ordinary and NaN-poisoned inactive-padding cases.
Inactive dX +0, empty-row raw seed, read-only inputs and independent writable resources passed.

Direct JDK26.0.1 javac/java: 0/0 for each probe; 48/48 functional invocations (32 forward setup,
16 reverse), zero warmup/clocked calls. The causal control generated once with zero numeric calls.
Exact commands/results: artifact root `probe/commands.txt`, `probe/report.txt`, `probe/control/`;
MAIN analysis scripts and `analysis/*comparison.json` retain complete-byte/identity/disassembly checks.
Repository changes: this new brief, CPU master and roadmap only. No production, API/Javadoc,
architecture, glossary, build, cache pin, historical evidence, commit or push changed. This bounded
verification makes no speed, JIT, general-shape or cross-backend claim; timings remain stopped.

Independent review checked actual classes, identities, disassembly, ABI guards, saved binary
inputs/outputs, probe/control sources and commands, and this Result: APPROVE, no P1/P2 findings.
It reran no compilation, generation, functional test or measurement. MAIN's preservation audit
confirmed unchanged source, HEAD/origin/index and historical evidence; only these three planning
documents changed. Existing API/implementation Javadoc remains accurate and unchanged. Markdown
links/anchors and whitespace passed; no follow-up is required within this verification scope.
