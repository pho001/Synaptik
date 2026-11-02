# Task CPU0063: Clean Scan Java with generated entry ABI

## Status and goal

Complete; bounded J4 experiment and independent review accepted on 2026-10-28. MAIN verified accepted CPU0062, base/empty index, dependencies and inactive conflicts before launch. Class A: align J4 with unchanged E, no production change. CPU0062 slots remain closed; subsequent generator work required and received separate CPU0064 authorization.

## Scope and contracts

Write only in `/private/tmp/synaptik-scan-java-abi-lab.K8KBvk`; sources/old roots immutable. No generator/production edits, build/suite, commit/push; off-tree javac only.
- Root: [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
- Backend: [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling), [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- Scan: [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic), [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories). Stop on ambiguity/conflict.

## Dependencies and integration

- Depends on: CPU0062 Complete/accepted; CPU0060/Prepare0015 accepted; CPU0061 restored.
- Conflicts with: repository source/oracle/fixture/artifact mutation and any concurrent performance campaign.
- Parallel group: None; sole OFFTREEJAVAONLY frontier after MAIN authorization; production closed.
- Common base revision: HEAD=origin `07a281edd3086efde2853211b71b48288db027e1`, empty index; `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0063-request.json` (2505 files), captured after CPU0062 closure.
- Integration order: MAIN Ready/base/conflicts → preparation → MAIN/reviewer frozen-compile/protocol approval → runs → independent result review → MAIN integration.
- Integration validation: preservation/source/protocol/raw/JIT/report checkpoint below.
- Shared-document integration owner: MAIN (Ready/master/roadmap/manifest); worker supplies evidence.

## Read-only inputs and source acceptance

G7: `/private/tmp/synaptik-cpu0060.uNRs5H/current-evidence/classfiles/FLOAT32-false-1.class`, SHA-256 `fe48de1f10b6dc257c548d716fad0a1ccf97653bc1899a3e3499beb986b008f9`. J3: `/private/tmp/synaptik-scan-java-lab.T7IrZN/ScanJ3.java` + classes. E: CPU test emit `CpuSymbolicLengthScanEquivalentJavaOracle.reverseFloat32`; existing `CpuSymbolicLengthScanFixture.Family` + unchanged `preboundEquivalentReverseCpu0045`. CPU0062 measurement mechanics read-only; no new framework/store.
- J4: G7's 19 segments + pack + start/end ABI; FLOAT32 F3/H2, T=`pack[0]`, B=`(int)pack[3]`, carrier scalar initializers/derivative, existing `sourceReverse=false`; no B/T baking.
- Inline E: initialization order, int row loops, shared rowFrame/sequenceFrame, direct addresses, work → dX → dW → dU → db → dH, early time guard. No E call/helper/allocation; preserve checked unaligned accesses, represented float products/adds, per-time carry ADD, singleton raw SUM, empty/padding/activity.
- Prove nonalias/dataflow legality; compare matched blocks with actual G7/J3/E; retain readable source/diff, carrier mapping, ClassFiles/pins. Stop if proof fails.

## Frozen validation and bounded measurement

- G7/J3/J4/E, E index 3 only; four independent fixtures/packs, disjoint outputs/scratch. Separate static-final typed G/J handles/wrappers and fixed 100-call batches per target; E capture unchanged. Hidden J3/J4 like G7; no polymorphic callsite.
- B7/B32 T4/F3/H2 mixed: (1000 warm + 300 measure) × 100 calls × four targets × two B = 1.04M calls/10400 samples (2400 measured)/JVM, excluding sanity. Four-position rotation exactly balanced (1000/300 divisible by four).
- In each JVM: B0/B1 zero, B1 active, B7 all-zero raw sanity; E-first, latest five cotangents, poison/inactive/empty raw +0 and inputs/packs before overwrite. Preserve failures.
- MAIN/reviewer preapprove frozen compile/sources/diffs/proofs, commands/JDK/options/classpath/pins and protocol before runs. At most TWO short serial JVMs ≤120s each; no retry/full suite/Gradle/extra sanity JVM/forced JIT/agents/adaptive warmup. Distinct grant, not CPU0062 reopening.
- Review raw + normal JIT evidence, no extra launch. Report all latencies/windows/ratios, failures and JIT limits; entry-inclusive E has an ABI limitation, no guaranteed parity/causality. No filtering or automatic generator port, even on a win.
- MAIN checks preservation/index/base, diff/whitespace, links/anchors and ≤45 lines/~5KB; no production test/Javadoc gate.

## Documentation, review and completion

Documentation impact: none in production; API/Javadoc unchanged; general/planning profiles, no glossary change. Independent targeted experiment review, not production Class C. Source/diffs, ClassFiles and two matched raw/JIT measurements retained in the scoped off-tree root. Reviewed whole-loop alignment justified the separately authorized [CPU0064 port](0064-scan-clean-java-generator-port.md#result), not a universal parity claim. No repository source changes or functional suites in this experiment. Status: Complete; no required experiment follow-up.
