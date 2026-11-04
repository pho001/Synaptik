# Task CPU0027: Unit [topology](../../../../glossary.md#term-topology) adjacency ordering

## Reader context — historical integration boundary

Shared Prepare supplies node/port edges; CPU must also order its own contracted computation units,
whose membership can change during fusion trials. This repair builds unit-level producer/consumer
adjacency for the current invocation, then uses stable Kahn ordering: units become ready only after
their direct producer units, with original node order breaking ties. That adjacency is not a second
owner of the shared graph and cannot be cached across changed membership. The resulting dependency
indices describe the finalized CPU recipe; Runtime does not repeat topological analysis. The one
reconstructed LSTM observation below is cold-preparation evidence, not a training-throughput
guarantee.

Current reading: [Partition and computation-unit boundaries](../../../../backend-guide/cpu-backend.md#general-partition-dag-and-bounded-pointwise-fusion).

## Status

Complete — independent code/docs review APPROVE, zero findings; main accepted the unchanged source and focused integration gate on 2026-10-20. [Ready](../../../../glossary.md#term-ready) dependencies, base/contracts and inactive conflicts were verified before launch.

## Change class

Class C — conservative [CPU](../../../../glossary.md#term-cpu) cold-analysis [topology](../../../../glossary.md#term-topology) change affects generated-unit order and resource-index [alignment](../../../../glossary.md#term-alignment); preserve all existing contracts.

## Goal

Replace repeated [producer](../../../../glossary.md#term-producer) scans in `CpuPartitionDagDecomposer.finish/topological` with invocation-local adjacency and stable Kahn ordering. Units are [CPU](../../../../glossary.md#term-cpu) computation groups; their dependencies are direct producer-unit indices in the final order.

## [Scope](../../../../glossary.md#term-scope)

- Build [identity](../../../../glossary.md#term-identity-meanings) `CompiledNode` → current-unit membership and minimum original [node](../../../../glossary.md#node) ordinal once per invocation. In one `PartitionDag.edges()` pass build unique inter-unit dependencies, consumer adjacency and indegrees; ignore intra-unit edges.
- Use a priority queue of ready units keyed by minimum original node ordinal. Reuse the same [dependency](../../../../glossary.md#term-dependency) facts for `finish`'s sorted final `Unit.dependencies`; no second edge/member scan to emit Units.
- Every [`topological`](../../../../glossary.md#term-topological-order)/`finish` invocation sees its current [topology](../../../../glossary.md#term-topology). Keep `producers/hasEdge` and horizontal/[mutable](../../../../glossary.md#term-mutable) contraction trials fresh; no retained [cache](../../../../glossary.md#term-cache) across topology changes.
- Prefer private implementation in the existing decomposer. One small package-private `CpuUnitTopology` in `io.github.pho001.synaptik.backend.cpu.internal.lowering` plus `CpuUnitTopologyTest` is allowed only as a justified structural-test seam; no broad abstraction or production instrumentation.

## Non-goals

No [partition](../../../../glossary.md#partition) cap, [fusion](../../../../glossary.md#term-fusion)-budget/policy, route/default, [API](../../../../glossary.md#term-application-programming-interface-api), numerics, [schema](../../../../glossary.md#term-schema)/[artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings), resource/[lifecycle](../../../../glossary.md#lifecycle), [dependency](../../../../glossary.md#term-dependency) or architecture change. No legacy copying, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) production edits, full-repository suite, [JMH](../../../../glossary.md#term-java-microbenchmark-harness-jmh) or five-[fork](../../../../glossary.md#term-fork-meanings) campaign.

## Contracts

Start with the root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>, then read these exact headings; stop on ambiguity/conflict:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>, [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Backend](../../../../glossary.md#backend) execution [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span> and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes): deterministic [CPU](../../../../glossary.md#term-cpu)-owned [lowering](../../../../glossary.md#lowering); no new backend or route.
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine) [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare) and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle): exact declarations before slot assignment; unchanged finalization and [ownership](../../../../glossary.md#term-ownership-meanings).

## Dependencies and integration

- Depends on: CPU0026 Complete, independently reviewed and accepted; its CPU0025/Compiler0009/NN0029 prerequisites and shared gate are Complete.
- Conflicts with: [CPU](../../../../glossary.md#term-cpu) [lowering](../../../../glossary.md#lowering)/[fusion](../../../../glossary.md#term-fusion)/unit-order/resource-index changes and shared [planning](../../../../glossary.md#term-planning) edits; all inactive, no other active writer per main.
- Parallel group: None; serial execution, no concurrent writes.
- Common base revision: `2090e0a6e32485756a3dfd0d545b5dbce523cfe6` (`HEAD == origin/main`).
- Integration order: one clean implementation context → one independent targeted code/docs review → main closure; no further planning/review prerequisite to launch.
- Integration [validation](../../../../glossary.md#term-validation): main accepts current worker evidence, focused CPU [conformance](../../../../glossary.md#term-conformance)/Fluent integration below once, links/anchors and `git diff --check`; no root gate for unchanged boundaries.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; owns this brief's final status/Result and CPU master/roadmap closure. Worker/reviewer return evidence without editing shared plans.

[Ready](../../../../glossary.md#term-ready) verification 2026-10-20: actual CPU0026 Status/Result confirms Complete/APPROVE and acceptance; base verified, [scope](../../../../glossary.md#term-scope)/contracts stable, conflicts inactive, ID free, sole CPU0027 frontier synchronized. CPU0023 stays Draft. This planning context launches nothing.
Preserve existing CPU0026 closure, dirty [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master, `THIRD_PARTY_NOTICES.md`, and user-owned `CpuOrdinaryPointwiseCastCleanJavaOracle.java`; no [commit](../../../../glossary.md#term-commit)/push. Recheck base/conflicts before implementation launch.

## Files and symbols

[CPU](../../../../glossary.md#term-cpu) paths below are under `modules/backend/cpu/src/`, package prefix `io/github/pho001/synaptik/backend/cpu/`:

- `main/java/.../internal/lowering/CpuPartitionDagDecomposer.java`: `finish`, [`topological`](../../../../glossary.md#term-topological-order), `Unit`; inspect [`producers`](../../../../glossary.md#term-producer), `hasEdge`, `containsNode`, `horizontal` and callers only for freshness/order preservation.
- `test/java/.../internal/lowering/CpuPartitionDagDecomposerTest.java`: [topology](../../../../glossary.md#term-topology), barriers, repeatability and port coverage; optional same-package `CpuUnitTopology.java`/`CpuUnitTopologyTest.java` as above.
- `test/java/.../CpuInternalPackageInventoryTest.java`: register the one new package-private helper in the existing exact internal inventory.
- Read-only boundaries: `CpuPartitionPreparer.analyze/analyzeLongPartition`, `modules/prepare/.../analysis/PartitionDag.java`; existing `testing/backend-conformance` CPU tests and `testing/integration-tests/.../FluentModelBuilderIntegrationTest.java`.

## Acceptance criteria

1. Test-only old scan-based [oracle](../../../../glossary.md#term-test-oracle) and new ordering yield exactly equal ordered memberships and sorted [dependency](../../../../glossary.md#term-dependency) indices for chains, diamonds, independent roots, interleaved memberships and newly ready lower-ordinal units; include contraction-created cycles and identical `IllegalArgumentException` rejection.
2. [Identity](../../../../glossary.md#term-identity-meanings) membership is preserved; deduplicate only inter-unit dependency pairs, never [graph port](../../../../glossary.md#term-graph-port) occurrences. Cover repeated input ports, multiple outputs and nonzero [producer](../../../../glossary.md#term-producer) output ports; intra-unit edges do not become self-dependencies.
3. Large sparse [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) uses a counting edge iterable at the narrow internal seam to assert exactly one edge traversal for [topology](../../../../glossary.md#term-topology) construction and no additional traversal during ordering/dependency emission; assert exact visits, not wall-clock thresholds. Keep test-only oracle inputs bounded. No shared [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag) change for instrumentation.
4. Existing one/eight/nine-[node](../../../../glossary.md#node), repeated-analysis, [publication](../../../../glossary.md#term-publication)/[fan-out](../../../../glossary.md#term-fan-out), affine/numerical/state barriers, tail [fusion](../../../../glossary.md#term-fusion), mixed-family/workspace, multi-output and repeated-port tests remain passing with precise order unchanged. No contraction legality, [materialization](../../../../glossary.md#materialization), route, resource declaration, fan-out, publication or numerical change.
5. Complete worker/[module](../../../../glossary.md#term-module) and focused integration gates below; rerun the identical CPU0026 public probe once after [CPU](../../../../glossary.md#term-cpu) classes rebuild, retaining separate evidence. Report completed/[censored](../../../../glossary.md#term-censored) phases honestly; a timeout leaves [latency](../../../../glossary.md#term-latency) unproved and requires an explicit limitation/follow-up, never a fabricated success or broadened investigation.

## [Validation](../../../../glossary.md#term-validation)

Commands verified against current [Gradle](../../../../glossary.md#term-gradle) mappings/build scripts and test classes; [run](../../../../glossary.md#term-run) from repository root with [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26. Worker runs focused tests during development, then [CPU](../../../../glossary.md#term-cpu) tests/[Javadoc](../../../../glossary.md#term-javadoc) once after executable changes stabilize:

```bash
./gradlew :backends:cpu:test --tests '*CpuPartitionDagDecomposerTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

If the helper is added, include `--tests '*CpuUnitTopologyTest'` in the focused command. Main runs this bounded integration gate once after review, or reuses an unchanged successful worker run:

```bash
./gradlew :testing:backend-conformance:test --tests '*Cpu*ConformanceTest' :testing:integration-tests:test --tests '*FluentModelBuilderIntegrationTest'
```

After the CPU rebuild, reuse the unchanged old launcher/harness via importlib; override both output directory and its precomputed [classpath](../../../../glossary.md#term-classpath) entry. This command creates only fresh external evidence:

```bash
probe_out=$(mktemp -d /private/tmp/cpu0027-probe.XXXXXX)
python3 -u - "$probe_out" > "$probe_out/controller.log" 2>&1 <<'PY'
import importlib.util, os, pathlib, sys
spec = importlib.util.spec_from_file_location('cpu0027_probe', '/private/tmp/cpu0026-diagnostic.IUaC7S/run_followup.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.OUT = pathlib.Path(sys.argv[1])
m.CP = os.pathsep.join([str(m.OUT / 'classes'), *m.CP.split(os.pathsep)[1:]])
sys.exit(m.main())
PY
```

The existing `ColdPreparationProbe.java` retains `FLOAT32[3,1,2]`, [seed](../../../../glossary.md#term-seed) 42, [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) widths 3/2, last step and [linear](../../../../glossary.md#term-linear-meanings)(1), reconstructed values/[objective](../../../../glossary.md#term-objective)/[targets](../../../../glossary.md#term-target). Keep profiling/configuration and the preinstalled external ≤180-second [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) watchdog; poll at ≤60 seconds, reap the child, never overwrite old evidence. Record rebuilt source/class [hashes](../../../../glossary.md#term-hash), flags/classpath, phases, exit and new evidence path. Inspect `exit.json`/phase [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings): launcher exit 0 alone does not prove preparation completed. Old forward [prepare](../../../../glossary.md#prepare) was 11.412469833 s; combined prepare was [censored](../../../../glossary.md#term-censored) >168 s, not a completed duration. This is a direct public [Engine](../../../../glossary.md#term-engine) probe, not actual cached `model.train` or a historical [regression](../../../../glossary.md#term-regression)/speedup proof. Do not restart the diagnosis or add a benchmark [matrix](../../../../glossary.md#term-review-matrix).

Finish with `git diff --check`, new-file whitespace, local links/anchors, protected files and [scope](../../../../glossary.md#term-scope)/status checks. No full repository/[JMH](../../../../glossary.md#term-java-microbenchmark-harness-jmh)/native build campaign.

## Documentation and review impact

- Apply local write-page guidance and general/[planning](../../../../glossary.md#term-planning) profiles; no Pages upload. Update meaningful affected implementation [Javadoc](../../../../glossary.md#term-javadoc) for invocation-local facts, stable tie-break, [dependency](../../../../glossary.md#term-dependency) order and cycle failure, documenting every input/result/failure; review unchanged public Javadoc explicitly.
- One independent clean code/docs review checks actual diff/tests, contract/order/freshness/complexity and resource-index preservation; finalizes Javadoc and any directly affected [CPU](../../../../glossary.md#term-cpu) explanation. No architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/glossary change expected because semantics/terms/boundaries stay unchanged; record targeted glossary/no-change reasoning. Reuse successful tests; regenerate Javadoc only if final Javadoc changes.
- Main closes shared plans after review. Completion evidence must list changes/files, exact commands/results/counts, probe limits, documentation/Javadoc impact, unresolved issues/follow-up and `Status: Complete` or `Status: Incomplete` with `Follow-up required: <specific follow-up>`.

## Result

- Replaced repeated readiness scans in `CpuPartitionDagDecomposer.finish/topological` with package-private `CpuUnitTopology`: [identity](../../../../glossary.md#term-identity-meanings) membership/minimum ordinals, one edge pass, unique unit dependencies, priority-queue Kahn ordering, and reused final [dependency](../../../../glossary.md#term-dependency)/member facts. Existing [mutable](../../../../glossary.md#term-mutable) contraction checks remain unchanged; no route, resource, numerical, [schema](../../../../glossary.md#term-schema), [API](../../../../glossary.md#term-application-programming-interface-api) or architecture change.
- Changed four [CPU](../../../../glossary.md#term-cpu) files: the decomposer, new `CpuUnitTopology.java`, new `CpuUnitTopologyTest.java`, and one inventory-test registration. Independent documentation review added two explanatory sentences in [CPU backend guide](../../../../backend-guide/cpu-backend.md#general-partition-dag-and-bounded-pointwise-fusion)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>, distinguishing contracted-unit adjacency from the shared [node](../../../../glossary.md#node)/port [graph](../../../../glossary.md#graph).
- Focused `./gradlew :backends:cpu:test --tests '*CpuPartitionDagDecomposerTest' --tests '*CpuUnitTopologyTest'`: 20 passed. Seven new tests compare the former scan [oracle](../../../../glossary.md#term-test-oracle), 256 bounded randomized contractions, newly-ready order, cycles, current identity membership, repeated ports/nonzero outputs, and a 20,000-node sparse graph with exactly 39,998 edge visits in one traversal, unchanged after dependency emission. No wall-clock assertions or production counters.
- Final `./gradlew :backends:cpu:test :backends:cpu:javadoc`: exit 0, 5m12s, 207 classes / 1,086 tests / 1,056 passed / 30 skipped / zero failures/errors. Initial full-[run](../../../../glossary.md#term-run) failure was solely the missing helper inventory entry; registration and final full rerun are green. [Final module log](</private/tmp/cpu0027-validation.hY3JCu/cpu-module.log>)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [log glossary](../../../../glossary.md#term-log))</span>.
- [Javadoc](../../../../glossary.md#term-javadoc) generated after final implementation comments; generation emitted 94 existing-code/incubator warnings, none for the changed files, and the final invocation was UP-TO-DATE. Reviewed public Javadoc remains accurate; no public API or glossary meaning changed.
- Main integration command `./gradlew :testing:backend-conformance:test --tests '*Cpu*ConformanceTest' :testing:integration-tests:test --tests '*FluentModelBuilderIntegrationTest'`: exit 0, 3s, both test tasks executed; 22 CPU [conformance](../../../../glossary.md#term-conformance) and 3 Fluent integration tests passed, zero skips/failures/errors. Includes existing recurrent compute/train coverage. No root/native/[JMH](../../../../glossary.md#term-java-microbenchmark-harness-jmh) campaign.
- Reused identical CPU0026 harness/driver [hashes](../../../../glossary.md#term-hash) in a fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/output root [cpu0027-probe.ofYWB4](</private/tmp/cpu0027-probe.ofYWB4>), with the same [fixture](../../../../glossary.md#term-fixture)/configuration/[JFR](../../../../glossary.md#term-java-flight-recorder-jfr) and 180-second watchdog. [Phase log](</private/tmp/cpu0027-probe.ofYWB4/probe.log>), [exit](</private/tmp/cpu0027-probe.ofYWB4/exit.json>) and [metadata/class hashes](</private/tmp/cpu0027-probe.ofYWB4/metadata.json>)<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span> retained; child exit 0, no watchdog timeout, total 3.797928417s, no lingering child.
- Completed public preparation: forward 0.261195583s versus old 11.412469833s; combined seeded 2.853043000s versus old [censored](../../../../glossary.md#term-censored) >168s. These are single diagnostic observations with profiling, not statistical speedup, cached `model.train` [latency](../../../../glossary.md#term-latency), a historical [regression](../../../../glossary.md#term-regression) proof or universal preparation bounds. No new [censoring](../../../../glossary.md#benchmark-measurement-vocabulary) or benchmark [matrix](../../../../glossary.md#term-review-matrix).
- Independent clean code/docs review APPROVE, zero findings; checked exact ordering/dependency remapping, mutable-trial freshness, workspace [alignment](../../../../glossary.md#term-alignment), contracts, code/Javadoc and targeted glossary entries, then reused worker and integration evidence without rerunning tests. Source [SHA-256](../../../../glossary.md#term-sha-256) prefixes remained `4fa3399fe1d5f324` (decomposer), `b2ab73ba2a3c08d6` ([topology](../../../../glossary.md#term-topology)), `a9c1e36ed11476e7` (new test), `09050d874d0c9ac5` (inventory).
- Main verified protected user-file hashes, source stability, links/anchors/new-file whitespace and `git diff --check`; only CPU implementation/tests, targeted guide and this task's [planning](../../../../glossary.md#term-planning) records changed. Existing CPU0026 closure/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) risk and unrelated user changes are preserved. No [commit](../../../../glossary.md#term-commit)/push. No unresolved issue or required follow-up; CPU0023 stays Draft and no new frontier is authorized.

Status: Complete
