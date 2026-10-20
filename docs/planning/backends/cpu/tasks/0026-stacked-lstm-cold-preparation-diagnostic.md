# Task CPU0026: Stacked LSTM cold preparation diagnostic

## Status

Complete — Pasteur final binding APPROVE, zero findings; main accepted the diagnostic on 2026-10-20. Any fix requires separate authorization.

## Change class

Class A — observation/report only, no behavior or contract changes. User nevertheless requires a clean executor and one independent targeted result review.

## Goal

Locate cold cost in one reconstructed three-step stacked long short-term memory (LSTM) training graph: composition, forward/backward compile and CPU preparation. Diagnose, do not fix.

## Scope

- Use exactly `Model.builder().input(DataType.FLOAT32, 3, 1, 2).seed(42).lstm(3).lstm(2).lastTimeStep().linear(1).build()`: time/batch/features `[3,1,2]`, hidden widths 3 then 2, output `[1,1]`.
- Structure/seed/target: observed in main's prior 17:53 tool snapshot; user authorized diagnosis 2026-10-20, not fixture confirmation. Reconstructed input `.1f,-.2f,.3f,.4f,-.2f,.6f`, contiguous FLOAT32, `requiresGrad=false`; target `[1,1]=.2f` likewise. Old input bytes unknown; no byte-identity claim or forensics.
- Reuse current integration objective `error=prediction.sub(target); error.mul(error).sum()` and `new Sgd(.01)` as reconstruction policy. Current two-step/one-hidden-lane coverage is not the interrupted structure.
- Read `/private/tmp/nn0029-frozen-handoff.txt` lines 1–110 only: exit 130 and one `CpuPartitionDagDecomposer.topological` sample prove neither failure nor regression.
- Harness via `apply_patch` in fresh `/private/tmp`; public APIs, existing classpath/minimal recorded compilation, no source/build edits.

## Non-goals

No production fix, kernel/default/route/timer/trace API/dependency/contract/native/NN/codec change; no numerical qualification, certification, matrix, five-fork/JMH campaign or full test/Javadoc rerun. Graph persistence is separate future work, not Ready or next implementation.

## Contracts

Read the root scope index, then these exact headings; stop on ambiguity or conflict:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules).
- Runtime/Prepare/Engine: [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle), [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
- Backend execution: [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules), [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- Compiler/autograd: [Training graph model](../../../../architecture/contracts/compiler-autograd.md#training-graph-model), [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle).
- Extensions/training: [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models), [`modules/training`](../../../../architecture/contracts/extensions-training.md#modulestraining). Observation preserves these owners; no architecture update needed.

## Dependencies and integration

- Depends on: CPU0025, Compiler0009, NN0029 — Complete, independently reviewed and integrated.
- Conflicts with: CPU lowering/preparation, Engine compile/prepare, Training/NN fixture semantics and shared planning edits; none active at verification.
- Parallel group: None.
- Common base revision: `2090e0a6e32485756a3dfd0d545b5dbce523cfe6` (`main == origin/main`).
- Integration order: clean executor → one targeted independent result review → main.
- Integration validation: source/evidence review, Markdown links/anchors, `git diff --check`, scope/protected hashes; no root suite.
- Shared-document integration owner: main; executor writes only this brief's Result/status; main owns master/roadmap/risk links.

Main Ready launch verification 2026-10-20: actual brief/frontier, Complete predecessors/reviews/root gate, base, inactive conflicts, contracts/public APIs and protected hashes verified. ID free; sole CPU diagnostic frontier. Planning launched nothing.
Preserve dirty `THIRD_PARTY_NOTICES.md` SHA-256 `8efe7e3c1f83bd5b30b913d447fa3ee95f9c0ee60485c30de7c0008638e7037f` and CPU test `internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java` SHA-256 `57ce4dc701a83d2ed01b96c725197a72f5aa752fe5e200f3a354f9d273f8f842`.

## Files and symbols

Java paths below use each module's `src/main/java/io/github/pho001/synaptik/` prefix (tests use `src/test/java`):

- `testing/integration-tests` test `testing/integration/FluentModelBuilderIntegrationTest.java` — values/objective and recurrent case.
- `extensions/nn`: `nn/module/{ModelBuilder,Model,ModelExecutionState}.java` — recurrence, `forward`, `configureOpen`; `modules/training`: `training/TrainingSession.java` — open's seed/targets/compile/prepare.
- `modules/engine`: `engine/{Engine,CompiledGraph}.java` — public compile overloads, prepare and metadata.
- `modules/backend/cpu`: `backend/cpu/internal/prepare/CpuPartitionPreparer.java` — `analyze`, `analyzeLongPartition`, finalization; `backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java` — `decompose`, `topological`, `producers`, `hasEdge`. Nested scans remain a hypothesis.
- CPU test `backend/cpu/internal/lowering/CpuPartitionDagDecomposerTest.java` — one/eight/nine nodes, stable analysis, repeated ports/multiple outputs.

## Acceptance and bounded execution

1. Log revision/dirty scope, JDK/JVM flags, CPU configuration, harness/hash, classpath, fixture and commands externally. No tuning/production cache mutation; fresh JVM ≠ cold OS caches.
2. ONE cold JVM, external hard timeout **180 seconds from launch**. Install watchdog first; emit phase start/end and `System.nanoTime` durations; poll at ≤60 seconds. Record exit, last phase and completed timings. Timeout means censored/incomplete phase, never successful completion.
3. Time composition, `Engine.compile(outputs)`/prepare, objective/seed/ordered-target setup, then `Engine.compile(outputs,cotangentSeeds,targets)`/prepare separately. Match TrainingSession seed and parameter admission. This direct probe is NOT an actual `model.train` measurement: actual `configureTraining` performs compile/prepare; `train` reuses the session. Record deviations and timer boundaries; no duplicate backward compile just to exercise both paths.
4. Within the run use Java Flight Recorder (JFR) and/or time-separated `jcmd` samples for CPU analysis/finalization/shared-prepare attribution. Whole-prepare elapsed time is exact at the public boundary; sampled subphase attribution is not exact duration. Mark unreached/indistinguishable stages. No extra training benchmark required.
5. Correlate time/stacks, source and available graph/partition/node/edge/unit counts with provenance. Separate observation from inferred complexity; recommend fix symbols/validation only. One sample proves no regression; no historical timing baseline.
6. Stop on a clear source-backed cause. At most ONE follow-up, same fixture and ≤180 seconds, only after a concrete finding and one recorded question; no repeat matrix or fix experiment. Insufficient evidence ends with uncertainty and a specific next decision.
7. One independent review checks actual logs/diff, source attribution, reconstruction/timeout limits and recommendation; reuses execution evidence, no extra formal cycle or run.

## Validation and documentation impact

Run `git diff --check`, `wc -lc docs/planning/backends/cpu/tasks/0026-stacked-lstm-cold-preparation-diagnostic.md`; check new-file whitespace, links/anchors, protected hashes and `git status --short`. Brief ≤100 lines / approximately 8 KB.
General/planning profiles and write-page guidance apply locally, no Pages upload. Review relevant Javadocs; API/Javadoc/glossary semantics unchanged, no Java/Javadoc generation. Logs/harness/JFR stay external, no extra report.

## Result

- Evidence root: `/private/tmp/cpu0026-diagnostic.IUaC7S`; [compact evidence](</private/tmp/cpu0026-diagnostic.IUaC7S/evidence-summary.txt>), [commands/flags/classpath](</private/tmp/cpu0026-diagnostic.IUaC7S/followup-controller.log>), [phase log](</private/tmp/cpu0026-diagnostic.IUaC7S/followup/probe.log>), [class/source hashes](</private/tmp/cpu0026-diagnostic.IUaC7S/followup/metadata.json>), [artifact hashes](</private/tmp/cpu0026-diagnostic.IUaC7S/SHA256SUMS>).
- Fixture exactly as Scope; values are reconstruction, not original bytes. Direct public Engine probe, **not actual cached `model.train`**: `model.forward(input)`, squared-error sum, public shared-native parameter admission, eight ordered `trainingParameters()` targets, native scalar-one leaf followed by `.contiguous()` cotangent; `Sgd(.01)` constructed, no update/run. Session-private optimizer buffers/binding and NN cached-output guards are excluded.
- Two launches total: setup PID 52465/driver 52455 exited 1 in 0.506859 s (missing `tools/tuning`, no performance baseline). Sole follow-up question: does the same fixture with complete existing classpath spend cold cost in CPU topology analysis? Corrected PID 52578/driver 52573 is the first valid measurement; no further launch.
- Commands: `python3 -u /private/tmp/cpu0026-diagnostic.IUaC7S/run_probe.py` then `run_followup.py` at the same root, redirected to `controller.log`/`followup-controller.log`; each contains exact harness-only `javac` and `java` commands. JDK 26.0.1+8-34, Vector module/native access, profile JFR; `Engine.standard()` CPU-only/default capacity one, automatic optional OpenBLAS, no tuning/cache configuration changes.
- Completed durations (seconds, `System.nanoTime`): build 0.055986333; forward composition 0.005544291; Engine open 0.104390042; forward compile 0.064433000; forward prepare 11.412469833; forward close 0.000130792; objective/native-admission/seed/targets 0.006067959; combined seeded compile 0.055150792.
- Combined prepare **censored**, not completed: began at main +11.720231792 s, unfinished at external launch deadline 180 s (~168.1 s in phase). Watchdog installed before spawn, SIGKILL exact PID at 180.004173 s (4.173 ms scheduling overshoot), reaped exit -9 at 180.066071 s; controller exited 0. Later explicit stop found no such process. No abandoned child or unbounded Engine.close.
- [Valid JFR](</private/tmp/cpu0026-diagnostic.IUaC7S/followup/finalized-first-chunk.jfr>) covers 130 s, main samples 08:13:05.589–08:15:15.923 UTC. Failed/timed-out `jcmd` and nondumping SIGQUIT supply **no** attribution; `exit.jfr` is empty. Finalized first chunk recovered from JVM repository before timeout; final active tail unavailable. Expansion command and compact sample histogram retained externally; regenerable 1,010,993,043-byte JSON removed after summary, binary/logs/harness retained.
- Sampled cause: three distinct late 30-second windows wholly in combined prepare contain 7,791/7,791 main samples in `topological`, 7,759 also in `hasEdge`; repeated chain `containsNode → hasEdge → producers → topological → finish → decompose → CpuPartitionPreparer.analyzeLongPartition → GraphPreparation.prepareInternal`. This is sustained CPU **analysis**, not evidence of finalization/kernel/optimizer cost; sampled subphase durations are not exact.
- Source attribution: `CpuPartitionPreparer.analyze/analyzeLongPartition` lines 151–173; `CpuPartitionDagDecomposer.finish/topological/producers/hasEdge/containsNode` lines 837–895. Every remaining candidate recomputes producer units by all-unit/full-edge/member scans. Inferred worst-case scan work `O(U^3 E K)` (units/edge occurrences/member bound), **not measured scaling**; existing-class bytecode confirms this call structure, no full source rebuild claimed.
- Available counts: raw forward provenance 192 producers/282 input-port occurrences/237 produced-input ports/23 leaves; objective 195/287/241/24. These are pre-optimization, repeated ports counted, **not combined compiled nodes/edges/units**. Public bindable inputs 23 forward/25 combined, targets 8; exact optimized graph/partition/unit counts unavailable. Combined slot assignment/finalization/schedule/run not established as reached; forward internal stages not separately timed.
- Minimal recommendation only: CPU-owned `finish/topological` builds identity node→current-unit membership and deduplicated dependencies/adjacency once from `PartitionDag.edges`, uses stable Kahn ordering by minimum original ordinal, reuses dependencies for emitted Units. Preserve contraction freshness (`horizontal/producers/hasEdge`), tie-breaking, port/multi-output semantics, fusion barriers and resource ownership; no global cache or contract change.
- Separately authorized fix validation: existing `CpuPartitionDagDecomposerTest` one/eight/nine, repeated analysis, fan-out/publication/tail/mixed-family/workspace/repeated-port/nonzero-output cases; add large sparse topology/order equivalence and deterministic operation-count evidence, rerun this same bounded public probe and affected CPU tests. No fix implemented or speedup claimed.
- Limits: fresh JVM ≠ cold OS caches; later phases inherit earlier JIT/class/artifact state and profiling overhead. No historical baseline/performance-regression conclusion, numerical qualification, actual training latency, exact sampled subphase timing or final-tail attribution.
- Only this brief's Status/Result changed by executor; main-owned dirty plans/roadmap untouched. Protected SHA-256 values above unchanged. Documentation impact: none beyond diagnostic result; reviewed API/implementation Javadocs remain accurate, no API/behavior/glossary semantics change; local write-page/general/planning guidance used.
- Final review/acceptance: Pasteur binding APPROVE, zero findings; main accepted 2026-10-20 after independently reproducing the offline JFR totals (7,791/7,791 `topological`, 7,759 `hasEdge`, three late 30-second windows). The 130-second recording does not attribute the missing tail. Main verified unchanged source/full snapshot during review, protected hashes and HEAD/main/origin base above. No additional JVM run.
- Validation: `git diff --check`, brief size, local links/anchors/whitespace, protected hashes and dirty scope passed; main's repository link check found no new errors. No suite/Javadoc/JMH/Gradle/native rerun for report-only closure. Main closes this brief, CPU master, NN risk link and roadmap; CPU frontier None, CPU0023 Draft. Brief exceeds ~8 KB to retain execution evidence/limits, remains below 100 lines/15 KB. Adjacency-once/stable-Kahn fix requires separate authorization; no successor task created.
Status: Complete
