# Task 0043: Reproducible Metal Route Benchmark

## Status

Ready

Metal 0042 is Complete at `9feb2505705263b6efb417d606678c606c2b9598` after final independent
Class C review returned `APPROVE` with zero findings. Implementation must start from the exact clean
planning revision containing this brief and record that revision before changing this status.

## Change class

Class C — this adds direct report-only tooling composition over public Compiler, Prepare, Runtime,
Trace, and Metal collaborations so one fixed workload can pin both backend-owned routes. It changes
no production lifecycle, public production API, capability, numerical contract, native ABI, cache,
or route-selection policy. Use one clean implementation context and a fresh independent targeted
review after the focused validation below. Reserve the repository-wide build for the final program
checkpoint; this task must not run it.

## Goal

Add the smallest fixed reproducible Metal benchmark: one positive static canonical no-grad
`FLOAT32` singleton `NEG` workload of Shape `[1_048_576]`, measured through both current
`MetalLocalWorkloadTuning` candidates as a report-only observation.

The benchmark enumerates and pins the complete current candidate pair, but never selects a winner,
invokes `Engine.prepareTuned(...)`, uses `MetalCompletePlanTuning`, reads or writes a tuning cache,
changes a production setting, or applies a performance threshold. Its JSON is explicitly ineligible
for a production decision and is not model-autotuning evidence.

## Fixed benchmark contract

### Workload and compile boundary

Use exactly one graph:

```text
input: canonical contiguous no-grad FLOAT32 [1_048_576]
output = NEG(input)
profile = STRICT_IEEE
optimization = disabled
providers/snapshots = the one explicitly opened Metal integration only
```

The input corpus is versioned and deterministic. Its prefix contains positive/negative zero,
minimum and maximum subnormals, minimum normals, ordinary finite values, infinities, and quiet and
signaling NaN payloads. Remaining elements use one documented deterministic finite generator. The
expected output is computed without floating arithmetic by XORing every input raw word with
`0x80000000`.

Compile through `GraphCompilationPort`, require exactly one Metal-owned partition, project it through
`GraphPreparation`, and ask the retained `MetalLocalWorkloadTuning` collaboration for its handoff.
Require exactly two candidates in the documented stable order. Candidate identity remains opaque;
the report may retain its public canonical bytes in hexadecimal but must not decode private route
wire values or publish session compatibility bytes.

### Separate route attestation

Route attestation is an untimed preflight with a separately opened traced integration:

1. compile and project the exact fixed workload;
2. enumerate the exact two candidates;
3. freshly prepare and run each trial once;
4. exact-byte-check its complete output; and
5. require one successful `PREPARE` and one successful `RUN` trace outcome with the same route.

Candidate zero must attest `CUSTOM_KERNEL`; candidate one must attest `GRAPH_EXECUTABLE`. Every
preparation must report `NOT_QUERIED` and native success. Capture only the closed Trace DTO facts;
fixed stream-local backend/device IDs are not machine identity. Close the entire traced integration
before timed setup. None of its durations enters the report.

Timed work uses a fresh ordinary `MetalBackendIntegration.open(configuration)` with no observer.
Recompile and re-enumerate the fixed pair, then require its public candidate identity bytes to match
the attested pair. The ordinary overload is the zero-trace fast path; no traced integration,
producer, event, clock read, or callback is allowed inside timed execution.

### Preparation and timed run boundary

Upload the fixed caller input once before warmup. Freshly prepare both trial candidates through
`GraphPreparation.prepare(...)` and the integration's schedule assembler. Record graph compile,
candidate enumeration, and each route's single prepare duration separately.

The retained hot measurement is exactly:

```text
PreparedExecutionRunner.run(prepared, borrowedInput) + RunResult.close()
```

It includes fresh Runtime output/workspace allocation, the synchronous native invocation and wait,
and run-owned cleanup. It excludes graph compile, candidate enumeration, route preparation, caller
upload, tracing, output download, checksum, JSON construction, and integration cleanup. A prepare
duration is a descriptive cold observation in fixed enumeration order, not a comparative statistic.

### Correctness and anti-optimization

Before warmup, run each ordinary untraced prepared route once, download the complete canonical
output, and compare every raw word with the expected sign-bit-flipped corpus. Both routes must
produce the same complete bytes and SHA-256.

After every retained timed batch, perform a separate untimed materializing run and repeat the
complete raw-bit comparison. Retain its SHA-256 in the report. The timed result itself remains
unmaterialized so host export is outside the hot boundary; synchronous native calls, result closure,
and the observed materializing runs prevent a dead-result benchmark. Any mismatch fails the whole
command before a successful report.

## Profiles and sampling

Profiles are normalized with `Locale.ROOT` and limited to:

| Profile | Paired warmup rounds | Paired measurement rounds | Retained batch floor |
|---|---:|---:|---:|
| `smoke` | 2 | 2 | 0 ns |
| `baseline` | 4 | 8 | 25,000,000 ns per route batch |

Each paired round alternates candidate order: `A,B`, then `B,A`, repeating. Thus both routes occupy
each position equally without a random-order or Shape matrix. Each baseline batch uses the hardened
CPU benchmark's adaptive iteration calculation and a ceiling of 1,000,000 executions. No execution,
sample, round, retry, outlier, or failure is discarded or replaced.

Retain, per route, every sample's iteration count, raw batch nanoseconds, integer normalized
per-run nanoseconds, complete post-batch checksum, and minimum/median/maximum normalized summary.
State that one fresh JVM invocation is the independent unit and within-process samples and paired
rounds are correlated.

Baseline adds and validates exact JVM arguments:

```text
-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch
--enable-native-access=ALL-UNNAMED
```

Smoke requires native access but does not claim controlled evidence.

## Native configuration and metadata

Register public Gradle task `:tools:benchmarks:metalBenchmark`; do not wire it into `build`, `check`,
or another lifecycle task. Reuse the existing explicit environment contract
`SYNAPTIK_METAL_TEST_LIBRARY`. The benchmark requires a nonblank absolute regular-file path and
constructs `MetalBackendConfiguration` from it. It performs no library discovery, extraction,
packaging, build, signing, or fallback.

Every successful report records:

- schema and benchmark identity, profile, `eligibleForProductionDecision=false`,
  `autotuningEvidence=false`, and fixed-enumeration/no-winner selection mode;
- exact fixed workload/generator schema, profile, descriptor, Shape, element count, expected SHA-256,
  timing boundary, warmups, rounds, floor, iteration ceiling, no-retry rule, and actual round order;
- Java/VM versions, exact JVM arguments, OS name/version/architecture, available processors, and
  observed initial/maximum heap;
- normalized real dylib path, byte size, SHA-256, and benchmark class SHA-256;
- compile/enumeration/elapsed durations and both complete raw route rows;
- candidate index and public identity hex, compatibility schema and `SESSION` scope, attested route,
  successful PREPARE/RUN/native facts, `NOT_QUERIED`, prepare duration, correctness hashes, raw
  samples, normalized samples, and distribution.

`baseline` additionally requires nonblank caller-supplied properties for exact base revision, source
identity, Git tree object SHA-1, harness-source SHA-256, host/model identity, Metal device identity,
macOS product/build identity, native build identity including Xcode/clang/SDK, power state, thermal
state, and a positive fork number. Caller-supplied values are labeled as such; an honest explicit
`unknown` is preferable to fabrication where a host fact is unavailable. Trace IDs are never used
as hardware facts.

## Failure and output contract

A successful invocation writes exactly one JSON document to standard output and exits zero. Missing
or invalid metadata/JVM flags, wrong host/library configuration, open or ABI failure, changed
partition/candidate identity/count/order, route-attestation mismatch, non-success trace fact,
correctness mismatch, iteration ceiling, or retained-floor violation fails nonzero. It must not
silently skip Metal, downgrade a mandatory route to `unavailable`, fabricate a row, or emit a
successful partial report.

The harness writes no report, cache, baseline, manifest, or other retained result artifact. A caller
may capture standard output externally. The task retains only command and validation facts in this
brief.

## Output schema

Top-level schema version `1` contains:

```text
schema, benchmark, profile,
eligibleForProductionDecision=false, autotuningEvidence=false,
selectionMode=fixed-candidate-enumeration-no-winner,
source, environment, nativeLibrary, protocol, workload,
compileNanos, candidateEnumerationNanos, elapsedNanos, routes[2]
```

Each route row contains:

```text
candidateIndex, candidateIdentityHex, compatibilitySchema, reuseScope,
attestation { route, prepareStatus, runStatus, cacheStatus, nativeStatus },
prepareNanos,
correctness { expectedSha256, precheckSha256, postBatchSha256[] },
sampleIterations[], repeatedBatchNanos[], repeatedRunNanos[],
distribution { minimum, median, maximum }
```

The schema contains no winner, recommendation, speedup claim, tuning objective, cache path,
persistent decision, route/default mutation, or performance gate.

## Files and ownership

Implementation scope:

- `tools/benchmarks/build.gradle.kts`
- `tools/benchmarks/src/main/java/io/github/pho001/synaptik/tools/benchmarks/MetalRouteBenchmark.java`
- `docs/developer-guide/benchmarking.md`
- `docs/backend-guide/metal-backend.md`
- `docs/architecture/performance-evidence-and-tuning.md`
- this task, `docs/planning/backends/metal/master-plan.md`,
  `docs/planning/tools/benchmarks/master-plan.md`, and `docs/planning/roadmap.md`

Do not change production Metal/Engine/Compiler/Prepare/Runtime/Trace sources, native sources or ABI,
capability/conformance tests, CPU benchmark behavior/schema/task, tuning tooling, caches, or
architecture authority. No permanent benchmark unit test is required: the actual smoke/baseline
commands exercise the supported surface and their JSON/failure validation is the proof.

## CI boundary

- Default and portable CI may compile and document the benchmark through
  `:tools:benchmarks:build`; it never runs `metalBenchmark` and needs no dylib.
- An opt-in dedicated Apple-silicon macOS runner with an explicitly built/provided dylib may run
  `smoke` to validate schema, routes, correctness, and exit status. It must not gate on timing.
- `baseline` belongs only on a controlled local or dedicated host. Shared or virtualized CI must not
  publish, compare, or gate on its timings.
- A future route/default decision requires a separate pre-registered multi-fork/target/statistical
  protocol. This task neither defines nor runs that matrix.

## Acceptance

- `:tools:benchmarks:build` and `:tools:benchmarks:javadoc` pass without loading a native library;
  existing CPU benchmark behavior and schema remain unchanged.
- The exact smoke command emits one valid schema-1 JSON document and exits zero with one fixed
  `[1_048_576]` singleton NEG workload and exactly two rows attested as `CUSTOM_KERNEL` and
  `GRAPH_EXECUTABLE`.
- Both rows use the identical graph, input, profile, and descriptor; both pass complete exact raw-bit
  precheck and every post-batch check; all raw counts/durations/checksums are retained.
- Timed execution uses only the separately opened ordinary no-trace integration. Attestation is
  untimed, reports `NOT_QUERIED`, and matches the ordinary candidate identities.
- The exact metadata-complete baseline command emits eight retained samples per route, validates
  fixed JVM flags, and every retained raw batch reaches 25 ms with no retry or discard.
- A focused failure invocation proves a missing required baseline metadata field fails nonzero
  before native benchmark work and emits no successful report.
- No `prepareTuned`, winner, complete-plan tuning, tools/tuning dependency, cache path or I/O,
  production mutation, native/capability change, timing threshold, silent skip, or retained result
  artifact exists.
- Documentation states the exact measurement boundary, metadata, commands, CI limits, correlation,
  and non-decision status without presenting observed timing as a route recommendation.
- Planning/Markdown links, fences, final newlines, trailing whitespace, and `git diff --check` pass.
- Run exactly one actual `smoke` and one actual metadata-complete `baseline` on the configured
  Apple-silicon Metal device, validate both JSON documents structurally and semantically without
  retaining them in the repository, and run no Shape/operation/fork matrix.
- Do not run the repository-wide build. After focused build/Javadocs, the two actual invocations,
  JSON validation, failure validation, docs/diff checks, and a clean implementation commit, set
  status to `Review needed` for fresh independent Class C review.

## Commands

From repository root:

```bash
./native/metal-macos-arm64/build.sh
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew -q :tools:benchmarks:metalBenchmark -Pprofile=smoke
```

The baseline command uses the same environment/task with `-Pprofile=baseline` and all required
`-Pbenchmark...` metadata properties named above. Exact observed values and the focused failure
command are recorded only after implementation; generated JSON remains untracked.
