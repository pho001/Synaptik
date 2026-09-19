# Benchmarking

## What you will learn

This guide defines the evidence expected from future Synaptik benchmarks and separates
benchmarking from the two implemented generic tuning transactions, their still-later Engine
composition, planning cost, and runtime profiling. The
`tools/benchmarks` project exists structurally, but no harness, `BenchmarkReport`, or workload is
implemented.

## Prerequisites and terms

Read the [performance evidence and tuning boundary](../architecture/performance-evidence-and-tuning.md)
and the [performance glossary entries](../glossary.md#benchmark-report--benchmarking) before
designing a harness.

- **Benchmarking** runs a fixed reproducible workload and reports measurements without changing
  production settings.
- A **benchmark report** is the rich immutable evidence from one recorded benchmark run.
- **Phase-1 workload tuning** reuses or measures caller-supplied exact compatible local workloads
  and records selected backend-owned decisions in an explicit persistent workload cache.
- **Phase-2 complete-plan tuning** authenticates a reusable selected decision or checks and
  measures every member of one caller-supplied bounded complete-plan batch.
- **Model autotuning** is the broader workflow that composes Phase 1 and Phase 2 around a model,
  representative inputs, target, policy, and later production preparation.
- **Runtime profiling** passively observes actual prepared execution.

## Mental model

```text
correctness tests -> is the result valid?
benchmark report  -> how did this fixed workload behave here?
Phase-1 tuning    -> which supplied complete local candidate has the lowest median elapsed time?
Phase-2 tuning    -> which supplied valid complete plan has the lowest median elapsed time?
model autotuning  -> how are those transactions composed for a model and production preparation?
runtime profiling -> what happened during actual prepared execution?
```

A benchmark answers only the second question. It never substitutes for unit, conformance, or
integration tests and never installs the fastest measured setting.

## Current tuning capabilities

### Phase 1: local workload transaction

`tools/tuning` currently provides a generic caller-supplied cold workload tuner and a reusable,
bounded persistent workload cache. The caller supplies stable model and representative-profile
evidence identities, original occurrence contexts and weights, backend-owned compatibility and
candidate identities, complete candidate enumeration, a decision codec, and one complete
candidate execution action on equivalent representative inputs.

The tuner loads and validates the complete explicit cache before candidate enumeration or
execution. It deduplicates exact compatible occurrences in encounter order. Compatible persistent
hits reuse the backend-decoded decision without enumerating or executing candidates. For each
miss, the tuner validates the complete batch against the budget, performs the configured complete
warmup and timed executions, calculates each candidate's integer-middle median, and retains the
first encountered candidate when medians tie. A successfully encoded decision must decode back to
an equal compatible decision before same-directory atomic cache publication. Returned rich
evidence retains raw measured samples separately; cache-hit evidence has an empty candidate list
and retains only the compact stored winner summary.

Engine supplies a bounded public CPU Phase-1 composition for one representative input set and at most
one local workload. A compatible hit executes no trial. A miss uses the configured warmup and
timed-sample counts, chooses the lowest integer-middle median with encounter-order ties, and then
freshly prepares production state. The backend retains semantic, compatibility, candidate,
decision, route, and resource ownership. Model extraction and multiple-occurrence aggregation
remain planned.

### Phase 2: complete-plan transaction

`tools/tuning` also provides a generic bounded complete-plan transaction. The caller supplies
model, representative-profile, target, and policy fingerprints; an exact decision-empty Prepare
handoff; a complete backend-owned candidate batch; a correctness collaboration; a fresh complete
measurement action; and an explicit model-plan-cache path. The concrete producer owns candidate
meaning, legality, stable order and identity, compatibility, reuse scope, selected decisions, and
decision codecs. The caller owns representative inputs, preparation, execution, canonical
publication copying, correctness-reference bytes, aggregate correctness-byte enforcement, and
cleanup.

On a measured transaction with `N` candidates, `W` warmups per candidate, and `S` timed samples
per candidate, the tool checks the exact whole-transaction execution count before running work:

```text
N * (1 + W + S)
    1 = one correctness execution per candidate
    W = complete untimed warmups per candidate
    S = complete timed samples per candidate
```

All `N` correctness actions finish before the first warmup or sample. The first candidate captures
an opaque reference; every later candidate must exactly match it. A clean mismatch stops the
transaction without timing, a result, or cache publication. Caller execution and cleanup failures
remain failures rather than being reclassified as mismatches. After correctness succeeds, each
candidate receives its complete warmups and samples. Selection uses the smallest integer-middle
median elapsed nanoseconds and retains the first candidate on a tie.

Reuse behavior is producer-declared. `SESSION` is a strict no-filesystem transaction: the cache
path is retained in the request but is never inspected or published. `PERSISTENT` validates the
entire bounded, checksummed cache before candidate enumeration. An exact-key entry is a hit only if
the current producer decodes it as compatible. Such a hit performs no enumeration, correctness,
warmup, or timing. A persistent miss is published only after the selected decision encodes within
the bound and decodes back to an equal compatible decision.

The persistent cache is compact: it contains opaque compatibility-key material, the selected
decision and winner identities, and the winner's minimum/median/maximum/count summary. The
returned evidence is deliberately richer. A measured result contains ordered correctness actions
and every raw timing sample; a cache-hit result identifies its source and selected summary but has
no fabricated candidate measurements. Neither artifact serializes an executable, Runtime state,
representative publication bytes, or the opaque correctness reference.

The generic Phase-2 transaction is current, but its public Engine/Config composition is not.
Current CPU complete-plan batches declare `SESSION`, so they neither read nor write the persistent
model-plan cache. A later Engine-owned adapter must join the CPU producer with Engine's existing
exact correctness primitive, freshly prepare every trial and the selected production decision,
and apply the existing fallback policy. The tools layer does not perform that composition itself.

## Current declarative Config request

`modules/config` now provides `ModelAutotuningConfig`, an immutable request facade separate from
the operational tool-local `WorkloadTuningRequest`. Possessing the Config value means tuning was
requested; there is no enabled flag, disabled sentinel, implicit default, or default cache path.
It stores exactly five user-owned inputs: the minimum-median-elapsed-nanoseconds objective, a
four-count sampling budget, one opaque schema-versioned representative-profile identity, a
fallback policy, and an explicit workload-cache path.

The Config value stores identity and policy, not work or execution state. Its representative
profile is an immutable byte snapshot that identifies evidence; actual representative inputs,
Shapes, model fingerprint, occurrences, backend candidate batches, cache contents, and execution
resources do not enter Config. Constructing the facade performs no cache I/O, measurement,
selection, preparation, Engine orchestration, or Runtime work.

Engine maps the objective and four budget counts explicitly, copies the profile identity, and
passes the exact cache path. `ModelAutotuningRequest` separately supplies live representative
Tensors and a caller-defined model identity. That identity labels evidence only; it is not a cache
key or proof of equivalent model behavior. `REQUIRE_TUNED_RESULT` is strict, while
`ALLOW_SAFE_HEURISTIC` permits abandoning a tuning
transaction that cannot yield a complete result and continuing through ordinary safe heuristic
preparation. Safe fallback does not make corrupt or incompatible cache data a hit, admit a
candidate, relax numerical requirements, accept a partial result, or suppress an unrelated
preparation failure. `TUNED` carries evidence; `SAFE_HEURISTIC_FALLBACK` carries none.

## Benchmark workloads and reports

Future benchmark suites may contain four fixed workload levels:

- one operation occurrence;
- a representative operation-family workload;
- a complete model workload; and
- end-to-end compile, prepare, and run workloads.

For every report, record the exact commit, model or workload identity, JDK, operating system,
hardware, native-library versions, warmup policy, measurement iterations, input data types and
shapes, backend and supplied route/configuration, thread settings, lifecycle boundary, and summary
statistics. Keep raw samples or equivalent distribution evidence when the selected reporting
policy requires it.

Measurement evidence remains richer than any later compact planning-cost profile, workload tuning
cache, or model-plan record. A production artifact may retain selected values and compatibility
identity; the report retains the environment, workload, candidates, samples, and statistics that
justify them.

## Complete conceptual example

### Inputs and initial state

Assume a future operation benchmark measures matrix multiplication `[64, 128] × [128, 32]` on one
recorded CPU configuration. The fixed logical work is:

```text
output values:          64 × 32 = 2,048
multiply contributions: 64 × 128 × 32 = 262,144
```

### Procedure and intermediate evidence

The harness warms up according to the report policy, measures only the chosen lifecycle stage,
and writes a report containing the fixed inputs, environment, supplied CPU route parameters, raw
or aggregated samples, and summary statistic.

### Result and interpretation

Running the same report definition on two commits permits a performance comparison. The faster
result does not update a CPU route threshold, vector strategy, thread count, workload tuning
cache, or model-plan cache. The current explicit tuning tools may instead run a caller-supplied
bounded Phase-1 or Phase-2 transaction and, when the producer permits persistent reuse, publish
their own compact cache entries. They do not mutate settings as a consequence of this benchmark
report.

## Workload-tuning candidate spaces

Workload tuning belongs to `tools/tuning`, not the benchmark suite. A concrete backend owns typed,
version-controlled candidate generators beside its routes. An operation family selects the
appropriate generator, but cache reuse uses a canonical workload signature that also includes
semantics and attributes, data types, shapes, layouts, relevant policies, and target
compatibility. A single family-wide thread or vector value is not a valid general cache contract.

Hardware and supported JDK Vector API species constrain physical vector lanes. A CPU candidate
generator may return complete valid species/strategy, unroll, tile, parallelism, and OpenBLAS
thread configurations. It cannot promise an arbitrary lane count or expose private knobs through
a generic parameter map.

## Current validation

```bash
./gradlew :tools:benchmarks:build
```

Today this validates only the empty project structure. The [benchmarks master plan](../planning/tools/benchmarks/master-plan.md)
governs later harness and reporting work.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| A result cannot be reproduced | Environment or workload inputs were omitted. | Record the complete workload and environment. |
| Compile time is mixed into run time | Lifecycle boundaries were not isolated. | Report compile, prepare, and run measurements separately. |
| A benchmark changes a later run's settings | Reporting was confused with tuning. | Remove side effects; use the explicit model-autotuning workflow. |
| One vector or thread value is applied everywhere | Operation family was incorrectly used as a universal cache key. | Key reuse by canonical workload signature and keep candidate vocabulary backend-owned. |
| Faster code changes numerical behavior | Performance was accepted without correctness evidence. | Run reference and conformance tests before interpreting speed. |
| Complete plans are timed before comparison | Correctness and timing were mixed. | Complete every Phase-2 correctness action before the first warmup or timed sample. |
| A session-only CPU plan creates a cache file | Reuse scope was ignored. | Treat `SESSION` as strict no-I/O; persistent reuse requires a producer that declares and authenticates it. |

## Limitations and boundaries

The benchmark harness and report, public Engine/Config Phase-2 composition, planning-cost model,
and concrete runtime-profile payloads remain planned. The generic caller-supplied Phase-1 and
Phase-2 transactions, bounded persistent cache formats, immutable Phase-1 Config request, and
bounded CPU/Engine Phase-1 composition are current. Model extraction, multiple-occurrence
aggregation, broader graph/plan generation, persistent CPU complete-plan reuse, concurrent cache
writers, cache migration, and inspection remain deferred. Config owns policy inputs only; it does
not own the runner, search algorithm, cache behavior, live discovery, or mutable evidence. No
benchmark or tuning action runs in the Runtime hot path.

## Related documentation

- [Performance discipline in `AGENTS.md`](../../AGENTS.md)
- [Performance evidence and tuning](../architecture/performance-evidence-and-tuning.md)
- [CPU kernel strategy](../design/notes/cpu-kernel-strategy.md)
- [Benchmark master plan](../planning/tools/benchmarks/master-plan.md)
- [Tuning master plan](../planning/tools/tuning/master-plan.md)
