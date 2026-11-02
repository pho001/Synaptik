# Task CPU0053: Fresh Scan versus Java continuation measurement

## Status

Cancelled — the user stopped this measurement campaign on 2026-10-28 after the sole untimed
preflight rejected a historical generated-class pin. No measured body started. Retain the failed
preflight and all preparation evidence; no retry or replacement is authorized.

## Change class

Class C conservatively: generated/direct performance comparability and frozen evidence review.
Clean launcher implementation and independent targeted review; no production change is authorized.

## Goal and scope

Measure the remaining entry-inclusive symbolic-length actual reverse Scan gap against equivalent
optimal Java and unchanged literal Java. Reuse current independent fixtures, but create a fresh
off-tree launcher and evidence root rather than reopening or modifying CPU0045.
Witness: FLOAT32, original forward traversal/opposite reverse, T4/F3/H2, B7 then B32,
length[row]=row%5 and joint seeds. Targets in fixed order: retained G4, current G5,
runtime-T/scalar equivalent Java E and unchanged literal Java J.

## Non-goals

No production/test-source/build-file edit, optimization, new capability, numerical policy,
native rebuild, tuning, repeated old selector, deleted marker or rewritten prior evidence.
No pure-inner-loop, causal or universal parity claim. CPU0048–0052 remain queued, not completed.

## Contracts

Read [root authority/scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
and [Core invariants](../../../../../ARCHITECTURE.md#core-invariants).
Backend contract: [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
Scan contract: [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic)
and [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories).

## Dependencies and integration

- Depends on: CPU0045 and CPU0047 Complete; successful existing semantic/structural gates.
- Conflicts with: any Scan/fixture/oracle/class writer or concurrent measurement.
- Parallel group: None for measured bodies. Read-only publication review and off-tree preparation
  are disjoint; neither may start a benchmark while publication validation runs.
- Common base revision: published CPU0047 `07a281edd3086efde2853211b71b48288db027e1` plus the retained
  accepted local Scan source in `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0047-accepted.json`;
  record a fresh exact source/class manifest before timing. The measured Scan remains local
  uncommitted work, not a capability introduced by the CPU0047 commit.
- Integration order: isolate/publish CPU0047 → approve frozen launcher → one untimed preflight
  if needed → three serial fresh JVMs → independent result review → MAIN closure.
- Integration validation: exact opt-in off-tree task, raw output/JUnit/JIT XML, source/class
  pins, output equality, fixed schedule/call counts and protected historical-evidence hashes.
- Shared-document integration owner: MAIN; workers do not edit roadmap/master or this brief.

Ready verification: MAIN checked completed predecessor results and current source owners,
unchanged accepted source bytes, no competing writer/measurement, and the user's explicit change
of priority ahead of remaining structural repairs. Historical measurement budgets stay closed.

## Files and artifacts

Write only `/private/tmp/synaptik-scan-continuation.ajgLjq/`: new Java test launcher, external
Gradle init, protocol/manifest and fresh preflight/repeat-1/2/3 output directories.
Read existing CPU `CpuGeneralScanTimingTest`, `CpuSymbolicLengthScanFixture`,
`CpuSymbolicLengthScanEquivalentJavaOracle`, `CpuSymbolicLengthScanCleanJavaOracle`, generator/IR
and retained v4 ABI resources. No repository Java type is added.

## Acceptance and validation

1. Pin JDK 26.0.1, normal tiered JVM, vector module, exact current source/classes and regenerated
   G5 bytes. Preserve historical literal/E oracle and G4 bytes; independent packs/resources.
2. Three serial JVMs; each B gets 100 warmup + 300 measured rounds, 100 calls/target/round,
   target=(round+position)%4. Ceiling 320,000 clocked calls/JVM and 960,000 total.
3. Sixty seconds/body including cold checks/persistence; at most one separate untimed preflight.
   No adaptive warmup, retry, replacement or extra campaign. Fail/censor honestly if a slot fails.
4. Create evidence/XML directories before JVM start; use CREATE_NEW markers and new output paths.
   Verify outputs outside timed batches before/after, retain raw targets/rounds and JIT brackets.
5. Report G5/E and G5/J per JVM/B with absolute latency; G4 is a control. Separate cold setup,
   entry costs and any qualifying JIT windows. Missing phase evidence means raw timings only.
   Do not attribute a ratio to MemorySegment, code size or a JIT mechanism without proof.

Launcher command and hash freeze are recorded in the artifact protocol before MAIN release.
Reuse existing semantic/module validation; no repeat of full passing suites or historical entries.

## Documentation and review impact

Planning profile for this brief/master/roadmap; detailed evidence stays off-tree. Independent
prelaunch and actual-result review are required. No API/Javadoc/architecture/glossary change:
production behavior and terminology are unchanged; explain benchmark limitations in Result.

## Result

Status: Incomplete
Follow-up required: only if separately requested, adapt the comparison to the actual current
Compiler reverse, prove generated ClassFile/ABI equivalence, and obtain a fresh preflight release.

CPU0047 alone was committed/pushed as `07a281edd3086efde2853211b71b48288db027e1`: 15 isolated
paths, clean-base CPU 49, conformance 16, Engine 2 and Javadoc passed; independent publication
review approved. Other accepted Scan/Compiler changes stayed local and were not discarded.
The off-tree CPU0053 launcher compiled, and independent prelaunch review approved its fixed
schedule/path/ownership configuration. All 1,204 fresh manifest inputs and 2,306 historical
evidence files matched. No production Java or repository build file changed for this diagnostic.

The single untimed preflight failed before target execution at launcher line 95: expected
historical G5 SHA-256 `b119b85deb07068761d9755185b4d68758ddb90420b08eb7d0d7b19b1f79f02c`,
observed freshly generated SHA-256 `30477241d7603cfec1c2f46079739a5ac10c2b1292dba3fb58aeaa2b05884330`.
JUnit: one failed preflight, no skip; body about 350 ms, validation failure rather than timeout.
Measured slots used: 0/3; clocked comparison calls: 0/960,000. No new performance result exists.

Compiler0016 had already changed reverse-graph metadata by removing redundant same-Shape
SUM_TO_SHAPE nodes. Its retained noncombined disassembly evidence shows only binary-name changes,
but that does not prove combined length-mode ClassFile/ABI equivalence for this failed preflight.
Reusing the historical full-class pin was an invalid launcher assumption, not evidence of a
numerical/backend failure. Do not simply replace the pin or weaken identity/packing checks.
The user declined another untimed slot and requested stop/summary; all workers are closed.
Evidence: `/private/tmp/synaptik-scan-continuation.ajgLjq/`, especially `preflight/report.txt`,
`preflight/disposition.properties`, `preflight/junit-xml/`, frozen launcher and protocol.
