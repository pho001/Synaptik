# Task CPU0054: Verify current Scan ClassFile and historical [ABI](../../../../glossary.md#term-application-binary-interface-abi)

## Reader context

The probe in this record compares three different questions: generated-code identity, call
compatibility and computed values. A changed binary class name can change a whole-file hash
without changing arithmetic instructions; matching instructions alone cannot prove that a
caller supplies the correct argument order or scratch span. The off-tree probe owns separate
writable resources for each diagnostic target and calls old code only through the explicit
diagnostic seam. It does not admit an old ClassFile into the production artifact store.

The [current generated-byte explanation](../../../../backend-guide/cpu-backend.md#generated-class-byte-persistence-and-evidence)
separates class bytes, hidden-class reuse and JIT code. The [current Scan composition overview](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
locates body reuse without extending this witness's ABI or numerical evidence to other shapes.

## Status

Complete — MAIN accepted the untimed verification on 2026-10-28 after independent actual-result
review APPROVE, with no open P1/P2 findings. CPU0053 and all performance slots stay closed.

## Change class

Class C conservatively: generated-code, [identity](../../../../glossary.md#term-identity-meanings) and diagnostic entry-[ABI](../../../../glossary.md#term-application-binary-interface-abi) verification.
Clean probe context and independent targeted actual-result review; no production change.

## Goal and [scope](../../../../glossary.md#term-scope)

Retain the actual current symbolic-length [FLOAT32](../../../../glossary.md#term-float32) reverse ClassFile before any comparison gate.
Explain the CPU0053 expected `b119b85…` versus observed `3047724…` mismatch using actual bytes,
not a guessed [identity](../../../../glossary.md#term-identity-meanings) update. Compare current and retained v5 after normalizing only the generated
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) name, inspect instructions, and check [descriptor](../../../../glossary.md#term-descriptor)/slot/pack/span/scratch compatibility with
retained v4 at T4/F3/H2, B7 and B32. Verify outputs against the existing equivalent and literal Java
references, including poisoned inactive [padding](../../../../glossary.md#padding), with independent resources and bounded calls.

## Non-goals

No benchmark, [warmup](../../../../glossary.md#term-warmup), measured loop, performance ratio, [JIT](../../../../glossary.md#term-just-in-time-compilation-jit)-causality claim, [cache](../../../../glossary.md#term-cache) pin replacement,
production/source-test/build edit, numerical policy change, native rebuild, [commit](../../../../glossary.md#term-commit) or push.
Do not resume or modify CPU0053's failed slot, frozen inputs or historical evidence.

## Contracts

Read [root authority index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
and [Core invariants](../../../../../ARCHITECTURE.md#core-invariants).
[Backend](../../../../glossary.md#backend) [owner](../../../../glossary.md#term-owner): [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>,
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
and [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
Scan owner: [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic)
and [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>.
An old [artifact](../../../../glossary.md#term-artifact) may [run](../../../../glossary.md#term-run) only through the existing explicit diagnostic seam after its exact [ABI](../../../../glossary.md#term-application-binary-interface-abi) is
proved; it must not enter production [cache](../../../../glossary.md#term-cache) [admission](../../../../glossary.md#term-admission).

## Dependencies and integration

- Depends on: Compiler0016 and CPU0047 Complete; cancelled CPU0053's retained failed preflight
  explicitly supplies the unresolved evidence, not a renewed timing grant.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- Conflicts with: any source/class/[fixture](../../../../glossary.md#term-fixture) writer or measurement.
- Parallel group: None. MAIN handles read-only baseline/byte analysis and shared documents;
  the clean worker exclusively writes its new off-tree probe and outputs.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1` plus accepted
  dirty source snapshot <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0054-base.json`.
- Integration order: verified brief → clean untimed probe → byte/ABI/output analysis → independent
  actual-result review → MAIN documentation/closure.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Integration validation: compile/run only the off-tree probe with installed JDK26.0.1 and
  vector module; actual ClassFiles, normalized disassembly/byte comparison, exact numeric packs,
  span/slot records, raw-output checks, source/evidence preservation and Markdown checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [vector glossary](../../../../glossary.md#term-vector); [module glossary](../../../../glossary.md#term-module))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

[Ready](../../../../glossary.md#term-ready) verification: MAIN checked open HEAD/origin/index, current source and failed evidence,
existing successful semantic gates, cancelled measurement status and absence of active workers.
The latest request permits this verification, not a production fix or timing campaign.

## Files and [artifacts](../../../../glossary.md#term-artifact)

Write only `/private/tmp/synaptik-scan-identity-verification.MFkNvn/` for probe code, compiled
classes, copied actual artifacts, invocation records and analysis. Reuse read-only [classpath](../../../../glossary.md#term-classpath)
`/private/tmp/synaptik-scan-continuation.ajgLjq/runtime-classpath.txt` and its fresh 1,204-input
[manifest](../../../../glossary.md#term-manifest). Read existing [CPU](../../../../glossary.md#term-cpu) `CpuSymbolicLengthScanFixture`, both Java oracles, current generator
and [IR](../../../../glossary.md#term-intermediate-representation-ir); retained v4 [ABI](../../../../glossary.md#term-application-binary-interface-abi) data and CPU0045 v5 artifacts; Compiler0016 [identity](../../../../glossary.md#term-identity-meanings) review as context.
No new repository Java type or [runtime](../../../../glossary.md#term-runtime) abstraction is introduced.

## Acceptance and [validation](../../../../glossary.md#term-validation)

1. Persist actual current ClassFile, both identity streams and facts before assertions can stop
   the probe. Report deterministic hashes and concrete binary names.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
2. Compare complete current/retained v5 bytes after replacing only the old generated binary name;
   keep actual `javap -c -p` output. If more differs, report it rather than treating a hash update
   as proof. Separate source-graph identity changes from emitted instructions and arithmetic.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [graph glossary](../../../../glossary.md#graph))</span>
3. Check old/current descriptors, semantic argument order, numeric packs and exact spans/scratch
   separately for B7/B32. Keep old/current identities distinct. If G4 ABI is not proved, do not run it.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [argument glossary](../../../../glossary.md#term-argument); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [run glossary](../../../../glossary.md#term-run))</span>
4. Run compatible targets once per ordinary/poisoned case only; no warmup or clocked batch.
   Preserve output equality and exact inactive/padding facts. Save diagnostics even on failure.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [warmup glossary](../../../../glossary.md#term-warmup); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [padding glossary](../../../../glossary.md#padding))</span>
5. Independent review checks actual artifacts/analysis and limitations without rerunning passing
   executable checks. No full-module rerun; this task changes no executable repository source.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [module glossary](../../../../glossary.md#term-module))</span>
6. If straightforward, one metadata-only controlled generation may restore only Compiler0016's
   six former rank-case SUM_TO_SHAPE calls in an off-tree copy of the current gradient rule.
   Keep current lexical-binding API, compile to an isolated overlay, do no numeric target work,
   and compare the resulting artifact with historical v5. This is a causal diagnostic, never an
   installed rule or source fix; preserve the exact copied source/diff and invocation.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [rank glossary](../../../../glossary.md#term-rank); [gradient glossary](../../../../glossary.md#term-gradient); [binding glossary](../../../../glossary.md#term-binding); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [compile glossary](../../../../glossary.md#compile))</span>

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning) profile for this brief/master/roadmap; detailed evidence off-tree. Probe [Javadoc](../../../../glossary.md#term-javadoc) explains
its bounded diagnostic protocol. Existing [API](../../../../glossary.md#term-application-programming-interface-api)/architecture/glossary semantics remain unchanged.

## Result

Status: Complete

The actual current ClassFile is `30477241d7603cfec1c2f46079739a5ac10c2b1292dba3fb58aeaa2b05884330`;
retained v5 is `b119b85deb07068761d9755185b4d68758ddb90420b08eb7d0d7b19b1f79f02c`.
Both are 17,302 bytes. Replacing exactly one generated [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-name [occurrence](../../../../glossary.md#term-occurrence) makes complete
ClassFile bytes identical; actual normalized `javap -c -p` has the same 952 instructions.
Both [identity](../../../../glossary.md#term-identity-meanings) streams differ only in their one 64-character [canonical](../../../../glossary.md#term-canonical-form)-[IR](../../../../glossary.md#term-intermediate-representation-ir) [fingerprint](../../../../glossary.md#term-fingerprint); all other
identity/geometry bytes are unchanged. No arithmetic, loop, helper, allocation or [dataflow](../../../../glossary.md#term-dataflow) change
is hidden by the binary-name [normalization](../../../../glossary.md#term-normalization).

The off-tree control restored only six former [rank](../../../../glossary.md#term-rank)-case SUM_TO_SHAPE calls in the current
[gradient](../../../../glossary.md#term-gradient) rule, retained its lexical [binding](../../../../glossary.md#term-binding)/normalizer, and regenerated the complete historical
v5 bytes exactly. This confirms Compiler0016's identity effect in this witness, not just an [inference](../../../../glossary.md#term-inference).
Old G4 and current G5 have the same 19-segment entry, exact 202-long packs, semantic slots and
spans at B7/B32; scratch is respectively 56/256 bytes. Four [targets](../../../../glossary.md#term-target) G4/G5/equivalent Java/literal
Java produced raw-equal five cotangents in ordinary and [NaN](../../../../glossary.md#term-nan)-poisoned inactive-[padding](../../../../glossary.md#padding) cases.
Inactive dX +0, empty-row raw [seed](../../../../glossary.md#term-seed), read-only inputs and independent writable resources passed.

Direct JDK26.0.1 javac/java: 0/0 for each probe; 48/48 functional invocations (32 forward setup,
16 reverse), zero [warmup](../../../../glossary.md#term-warmup)/clocked calls. The causal control generated once with zero numeric calls.
Exact commands/results: [artifact](../../../../glossary.md#term-artifact) root `probe/commands.txt`, `probe/report.txt`, `probe/control/`;
MAIN analysis scripts and `analysis/*comparison.json` retain complete-byte/identity/disassembly checks.
Repository changes: this new brief, [CPU](../../../../glossary.md#term-cpu) master and roadmap only. No production, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc),
architecture, glossary, build, [cache](../../../../glossary.md#term-cache) pin, historical evidence, [commit](../../../../glossary.md#term-commit) or push changed. This bounded
verification makes no speed, [JIT](../../../../glossary.md#term-just-in-time-compilation-jit), general-[shape](../../../../glossary.md#shape) or cross-[backend](../../../../glossary.md#backend) claim; timings remain stopped.

Independent review checked actual classes, identities, disassembly, [ABI](../../../../glossary.md#term-application-binary-interface-abi) guards, saved binary
inputs/outputs, probe/control sources and commands, and this Result: APPROVE, no P1/P2 findings.
It reran no [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), generation, functional test or measurement. MAIN's preservation audit
confirmed unchanged source, HEAD/origin/index and historical evidence; only these three [planning](../../../../glossary.md#term-planning)
documents changed. Existing API/implementation Javadoc remains accurate and unchanged. Markdown
links/anchors and whitespace passed; no follow-up is required within this verification [scope](../../../../glossary.md#term-scope).
