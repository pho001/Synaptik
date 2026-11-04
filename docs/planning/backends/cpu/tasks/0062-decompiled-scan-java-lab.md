# Task CPU0062: Decompiled [Scan](../../../../glossary.md#term-cumulative-scan) Java lab

## Reader context

J0–J3 are off-tree Java programs compiled from the retained decompile, not changes to CPU's
production generator. The launcher invokes their hidden classes through the generated entry
ABI: nineteen segments plus packed geometry and range. Each target has its own writable
fixtures, so matching input values do not imply shared scratch or output ownership.
E is a separately prebound direct Java reference. A J0/G7 difference can therefore expose a
translation effect before any of the later manual optimizations are considered.

For the actual code-generation participants, see [common MATMUL binding](../../../../backend-guide/cpu-backend.md#common-rank-two-matmul-binding).
For why smaller ClassFiles do not imply faster machine code, see [generated-byte evidence](../../../../backend-guide/cpu-backend.md#generated-class-byte-persistence-and-evidence).
Nothing in this lab's completed status means its manual code was installed.

## Status

Complete; MAIN accepted independent prelaunch/result approval on 2026-10-28. Both slots/frontier closed.

## Change class and goal

Class A off-tree Java experiment: optimize decompiled G7 against unchanged Java E. Clean implementation/review; no production change or parity guarantee.

## [Scope](../../../../glossary.md#term-scope) and non-goals

Write Java/evidence only under `/private/tmp/synaptik-scan-java-lab.T7IrZN`; repository: brief/master/roadmap only. No production/generator/[oracle](../../../../glossary.md#term-test-oracle)/format/[cache](../../../../glossary.md#term-cache) changes, port, [Gradle](../../../../glossary.md#term-gradle), functional suites or [publication](../../../../glossary.md#term-publication); off-tree javac only.

## Contracts

- [Root Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
- [Backend](../../../../glossary.md#backend) [owner](../../../../glossary.md#term-owner): [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling), [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- [Scan](../../../../glossary.md#term-cumulative-scan) owner: [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic), [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>. Stop on ambiguity/conflict; no new architecture.

## Dependencies and integration

- Depends on: CPU0060/Prepare0015 Complete; CPU0061 Cancelled/restored v7.
- Conflicts with: source/[artifact](../../../../glossary.md#term-artifact) mutation or concurrent measurement.
- Parallel group: None; serial execution/review.
- Common base revision: HEAD=origin `07a281edd3086efde2853211b71b48288db027e1`, empty index; request `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0062-request.json` (2503 files); prior audit unchanged.
- Integration order: MAIN [Ready](../../../../glossary.md#term-ready) verification → preparation → preapproval → measurement → result review → MAIN integration. Dependencies/base/inactive conflicts verified before launch.
- Integration [validation](../../../../glossary.md#term-validation): source/raw-output/preservation/report below; no [module](../../../../glossary.md#term-module) gate.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; worker supplies evidence. Older grants remain closed.

## Files and symbols

Read-only G7: `/private/tmp/synaptik-cpu0060.uNRs5H/current-evidence/classfiles/FLOAT32-false-1.class`, [SHA-256](../../../../glossary.md#term-sha-256) `fe48de1f10b6dc257c548d716fad0a1ccf97653bc1899a3e3499beb986b008f9`; sibling `../../decompiled/FLOAT32-false-1.java` is the actual decompile.
Read-only [CPU](../../../../glossary.md#term-cpu) test emit package: `CpuSymbolicLengthScanEquivalentJavaOracle.reverseFloat32` (E), `CpuSymbolicLengthScanFixture.Family` and `preboundEquivalentReverseCpu0045`; request-identical source. E ClassFile pin is recorded in lab `input-pins.json`.
Adapt `/private/tmp/synaptik-cpu0061.CFjsUt/measurement` read-only; not its old selectors, second mechanism or retention gate. Family setup permits no source writes.

## Acceptance criteria

- Six [targets](../../../../glossary.md#term-target): G7; J0 exact decompile, class rename only; J1 derive 8*B/12*B; J2 move inactive-time [guard](../../../../glossary.md#term-guard) before setup; J3 move dH last with nonalias/[dataflow](../../../../glossary.md#term-dataflow) proof; E unchanged. No fixed B/T; F3/H2 already baked.
- Keep J0–J3 sources, ClassFiles and adjacent diffs. Load [prototypes](../../../../glossary.md#term-prototype) as hidden classes like G7; no [kernel](../../../../glossary.md#kernel) helpers. Preserve actual [scalars](../../../../glossary.md#term-scalar), checked unaligned memory access, float product/sum sites, per-time [carry](../../../../glossary.md#term-carry) ADD, [padding](../../../../glossary.md#padding)/raw state.
- F3/H2, T4, B7/B32, mixed lengths and actual roles; same 19 segments + packed + start/end [ABI](../../../../glossary.md#term-application-binary-interface-abi) for G7/J. Six independent fixtures/packs with disjoint outputs/scratch.
- TYPED entry only: separate static-final invokeExact wrapper and fixed-kind 100-call loop per target, no shared polymorphic callsite. E uses unchanged prebound action. Report entry-inclusive costs and ABI limitations, not pure-body parity.
- E is index 5 ONLY (not old `i%3==2`). E-first raw baseline, latest/poison checks before overwrite, all five outputs and [immutable](../../../../glossary.md#term-immutable) inputs/packs. B0/B1/all-zero/mixed sanity inside measured [JVM](../../../../glossary.md#term-java-virtual-machine-jvm); no new-T claim or functional suite.

## [Validation](../../../../glossary.md#term-validation)

MAIN/reviewer preapprove compiled sources, diffs, nonalias/[dataflow](../../../../glossary.md#term-dataflow) proof, E selector and protocol. Freeze off-tree commands/[JDK](../../../../glossary.md#term-java-development-kit-jdk)/options/[classpath](../../../../glossary.md#term-classpath)/pins and balanced rotating order in a small local report.
At most TWO serial measured JVMs, 120 seconds each, no retry/agent/forced [JIT](../../../../glossary.md#term-just-in-time-compilation-jit)/adaptive [warmup](../../../../glossary.md#term-warmup)/native campaign. Each: B7/B32 × six [targets](../../../../glossary.md#term-target) × (1000 warmup + 300 measured rounds) × 100 calls = 1.56M calls, 15600 [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) (3600 measured). Second confirms the same six unchanged, without favorable case/window filtering. Preserve failures.
Report all latencies/ratios, early/middle/late, normal JIT readiness/limits, raw checks, commands and source/ClassFile pins. Separate J0/G7 translation effect from adjacent changes; no old retention gate/[runtime](../../../../glossary.md#term-runtime) policy.
MAIN verifies request preservation except three [planning](../../../../glossary.md#term-planning) files, empty index/base, links/anchors, scoped diff/whitespace and ≤70 lines/6KB. No [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) gate for unchanged production.

## Documentation and review impact

Documentation impact: none; production [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc)/contracts unchanged. [Planning](../../../../glossary.md#term-planning) profile; no new glossary term or review launch.

## Result

Status: Complete (bounded experiment, not parity). Lab sources/diffs, `analysis-results.json`, protocol and pins are under the [scope](../../../../glossary.md#term-scope) root; freeze `f3130c1f6526ddd907d1fb2df9dea3cb5230bb3cbcaab8bc5cfba837d9fa3e13`.
Both JVMs passed (5.026/5.078s): each 15600 [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), 1.56M calls, 31200 raw events, zero errors. Review confirmed frozen inputs/persisted outputs. No functional suites/generator edits.
J3 was best manual across runs/windows/positions: J3/J2 ~.964 at B7 and .929–.934 at B32, yet 20–22% slower than G7 and 37–66% slower than E. J0 was 23–29% slower than G7 despite smaller [bytecode](../../../../glossary.md#term-bytecode). Nothing is ported.
[Kernel](../../../../glossary.md#kernel)/wrapper/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [C2](../../../../glossary.md#term-c2) completed before measurements; body calls were not inlined. B7 overlaps background logging [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), B32 has no logged compilation events. This supports compiled-path readiness, not native causation or universal parity. Follow-up: separate same-[ABI](../../../../glossary.md#term-application-binary-interface-abi) clean-loop Java experiment; old slots stay closed.
