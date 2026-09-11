# Benchmarking

## What you will learn

This guide defines the evidence expected from future Synaptik benchmarks and separates
benchmarking from workload tuning, the broader model-autotuning roadmap, planning cost, and
runtime profiling. The
`tools/benchmarks` project exists structurally, but no harness, `BenchmarkReport`, or workload is
implemented.

## Prerequisites and terms

Read the [performance evidence and tuning boundary](../architecture/performance-evidence-and-tuning.md)
and the [performance glossary entries](../glossary.md#benchmark-report--benchmarking) before
designing a harness.

- **Benchmarking** runs a fixed reproducible workload and reports measurements without changing
  production settings.
- A **benchmark report** is the rich immutable evidence from one recorded benchmark run.
- **Workload tuning** currently reuses or measures caller-supplied exact compatible workloads and
  records selected backend-owned decisions in an explicit persistent cache.
- **Model autotuning** is the broader two-phase roadmap: model-guided workload extraction and
  reuse followed by bounded comparison of complete valid model plans.
- **Runtime profiling** passively observes actual prepared execution.

## Mental model

```text
correctness tests -> is the result valid?
benchmark report  -> how did this fixed workload behave here?
workload tuning   -> which supplied complete local candidate has the lowest median elapsed time?
model autotuning  -> which compatible local results and complete plan best meet the objective?
runtime profiling -> what happened during actual prepared execution?
```

A benchmark answers only the second question. It never substitutes for unit, conformance, or
integration tests and never installs the fastest measured setting.

## Current workload-tuning capability

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

This is a composition contract, not an out-of-the-box CPU or model API. There is no supported CPU
adapter or Engine integration. The caller must supply complete execution and stable identities,
while the backend retains semantic, compatibility, candidate-generation, decision-validation,
route, and resource ownership. The current tool does not extract workloads from a model or perform
the second graph/plan phase of model autotuning.

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

A later outer composition layer may depend on both modules. It can map the objective and four
budget counts explicitly, construct the tool-local profile fingerprint from a Config accessor
copy, and pass the exact requested path. That layer must separately supply the model fingerprint,
occurrences, representative execution, and eligible candidates. It also owns fallback control
flow: `REQUIRE_TUNED_RESULT` is strict, while `ALLOW_SAFE_HEURISTIC` permits abandoning a tuning
transaction that cannot yield a complete result and continuing through ordinary safe heuristic
preparation. Safe fallback does not make corrupt or incompatible cache data a hit, admit a
candidate, relax numerical requirements, accept a partial result, or suppress an unrelated
preparation failure. This mapping and control flow are planned, not current Engine behavior.

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
cache, or model plan. The current explicit workload tuner may instead run a caller-supplied
bounded local search and publish its own compact cache entry. It does not mutate settings as a
consequence of this benchmark report, and it does not yet produce a model plan.

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

## Limitations and boundaries

The benchmark harness and report, full two-phase model-autotuning workflow, planning-cost model,
and concrete runtime-profile payloads remain planned. The generic caller-supplied workload tuner,
bounded persistent workload cache, and separate immutable Config request facade are current, but
their supported CPU/Engine composition, model extraction, complete graph/plan search, concurrent
writers, cache migration, and inspection remain deferred. Config owns request inputs only; it does
not own the runner, search algorithm, cache behavior, live discovery, or mutable evidence. No
benchmark runs in the runtime hot path.

## Related documentation

- [Performance discipline in `AGENTS.md`](../../AGENTS.md)
- [Performance evidence and tuning](../architecture/performance-evidence-and-tuning.md)
- [CPU kernel strategy](../design/notes/cpu-kernel-strategy.md)
- [Benchmark master plan](../planning/tools/benchmarks/master-plan.md)
- [Tuning master plan](../planning/tools/tuning/master-plan.md)
