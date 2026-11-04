# Task 0043: Reproducible [Metal](../../../../glossary.md#term-metal) Route Benchmark

## Status

Complete

Implemented from exact clean [planning](../../../../glossary.md#term-planning) base `1d8f8cb03cb631ab59c25bbfa025369f4d2547e9`.
Initial independent Class C review returned `BLOCK` for two P2 findings: the attestation collector
did not honor the [observer](../../../../glossary.md#term-observer)'s concurrent-[callback](../../../../glossary.md#term-callback) contract, and generator [schema](../../../../glossary.md#term-schema) 1 did not fully
specify its corpus. Both were remediated from exact clean base
`c3e12e7de949b8d3f2c5dad5f45b8086cfc8fedb` at
`77e6091b2a452faa04fa2b674bc295ddc7be88b7`; updated focused and actual-[device](../../../../glossary.md#term-device) evidence superseded
the stale original reports. The same reviewer then returned `APPROVE` with zero remaining findings.
The repository-wide build remains deliberately reserved for the final program [checkpoint](../../../../glossary.md#term-checkpoint).

## Change class

Class C — this adds direct report-only tooling composition over public [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime),
[Trace](../../../../glossary.md#trace), and [Metal](../../../../glossary.md#term-metal) collaborations so one fixed workload can pin both [backend](../../../../glossary.md#backend)-owned routes. It changes
no production [lifecycle](../../../../glossary.md#lifecycle), public production [API](../../../../glossary.md#term-application-programming-interface-api), capability, numerical contract, native [ABI](../../../../glossary.md#term-application-binary-interface-abi), [cache](../../../../glossary.md#term-cache),
or route-selection policy. Use one clean implementation context and a fresh independent targeted
review after the focused [validation](../../../../glossary.md#term-validation) below. Reserve the repository-wide build for the final program
[checkpoint](../../../../glossary.md#term-checkpoint); this task must not [run](../../../../glossary.md#term-run) it.

## Goal

Add the smallest fixed reproducible [Metal](../../../../glossary.md#term-metal) benchmark: one positive static [canonical](../../../../glossary.md#term-canonical-form) no-grad
[`FLOAT32`](../../../../glossary.md#term-float32) [singleton](../../../../glossary.md#term-singleton) `NEG` workload of [Shape](../../../../glossary.md#shape) `[1_048_576]`, measured through both current
`MetalLocalWorkloadTuning` candidates as a report-only observation.

The benchmark enumerates and pins the complete current candidate pair, but never selects a winner,
invokes `Engine.prepareTuned(...)`, uses `MetalCompletePlanTuning`, reads or writes a tuning [cache](../../../../glossary.md#term-cache),
changes a production setting, or applies a performance threshold. Its [JSON](../../../../glossary.md#term-json) is explicitly ineligible
for a production decision and is not [model](../../../../glossary.md#term-model)-[autotuning](../../../../glossary.md#autotuning--model-autotuning) evidence.

## Fixed benchmark contract

### Workload and [compile](../../../../glossary.md#compile) boundary

Use exactly one [graph](../../../../glossary.md#graph):

```text
input: canonical contiguous no-grad FLOAT32 [1_048_576]
output = NEG(input)
profile = STRICT_IEEE
optimization = disabled
providers/snapshots = the one explicitly opened Metal integration only
```

Input generator [schema](../../../../glossary.md#term-schema) 2 is fully reconstructible. Zero-based indices `0..15` contain these ordered
raw [binary32](../../../../glossary.md#floating-point-precision-and-rounding) words:

```text
0x00000000 0x80000000 0x00000001 0x80000001
0x007fffff 0x807fffff 0x00800000 0x80800000
0x3f800000 0xbf800000 0x7f800000 0xff800000
0x7fc12345 0xffc54321 0x7fa12345 0xffa54321
```

For each zero-based index `i` in `16..1_048_575`, compute signed 64-bit integers
`r = ((i * 17) % 101) - 50`, convert the exact integer `r` and exact integer `101` to binary32,
divide under Java's strict [IEEE 754](../../../../glossary.md#term-ieee-754) binary32 [round-to-nearest, ties-to-even](../../../../glossary.md#floating-point-precision-and-rounding) semantics, and retain
`Float.floatToRawIntBits(r / 101.0f)`. The nonnegative dividend makes Java remainder `%` lie in
`0..100`; therefore `r` lies in `-50..50` and both integer-to-binary32 conversions are exact.
Expected output is computed without floating arithmetic by XORing every input raw word with
`0x80000000`. The report carries this schema version, every sentinel word, the indexed formula,
integer/remainder domain, conversion, denominator raw word, division [rounding](../../../../glossary.md#term-rounding), and raw encoding.

Compile through `GraphCompilationPort`, require exactly one [Metal](../../../../glossary.md#term-metal)-owned [partition](../../../../glossary.md#partition), project it through
[`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), and ask the retained `MetalLocalWorkloadTuning` collaboration for its [handoff](../../../../glossary.md#term-handoff).
Require exactly two candidates in the documented stable order. Candidate [identity](../../../../glossary.md#term-identity-meanings) remains [opaque](../../../../glossary.md#term-opaque);
the report may retain its public [canonical](../../../../glossary.md#term-canonical-form) bytes in hexadecimal but must not decode private route
wire values or publish session compatibility bytes.

### Separate route attestation

Route attestation is an untimed [preflight](../../../../glossary.md#term-preflight) with a separately opened traced integration:

1. [compile](../../../../glossary.md#compile) and project the exact fixed workload;
2. enumerate the exact two candidates;
3. freshly [prepare](../../../../glossary.md#prepare) and [run](../../../../glossary.md#term-run) each trial once;
4. exact-byte-check its complete output; and
5. require one successful `PREPARE` and one successful `RUN` [trace](../../../../glossary.md#trace) outcome with the same route.

Candidate zero must attest `CUSTOM_KERNEL`; candidate one must attest `GRAPH_EXECUTABLE`. Every
preparation must report `NOT_QUERIED` and native success. The [observer](../../../../glossary.md#term-observer) honors its concurrent-[callback](../../../../glossary.md#term-callback)
contract through a synchronized collector. After each candidate's materializing run closes, the
harness obtains one [immutable](../../../../glossary.md#term-immutable) collector [snapshot](../../../../glossary.md#term-snapshot) before validating the appended count, unchanged
prefix, PREPARE/RUN order, indices, and route-specific facts; a candidate can never consume facts
from a live list or another candidate. [Capture](../../../../glossary.md#term-capture-meanings) only the closed Trace [DTO](../../../../glossary.md#data-transfer-object--dto) facts; fixed stream-local
[backend](../../../../glossary.md#backend)/[device](../../../../glossary.md#term-device) IDs are not machine [identity](../../../../glossary.md#term-identity-meanings). Close the entire traced integration before timed
setup. None of its durations enters the report.

Timed work uses a fresh ordinary `MetalBackendIntegration.open(configuration)` with no observer.
Recompile and re-enumerate the fixed pair, then require its public candidate identity bytes to match
the attested pair. The ordinary overload is the zero-trace fast path; no traced integration,
[producer](../../../../glossary.md#term-producer), event, clock read, or callback is allowed inside timed execution.

### Preparation and timed [run](../../../../glossary.md#term-run) boundary

Upload the fixed caller input once before [warmup](../../../../glossary.md#term-warmup). Freshly [prepare](../../../../glossary.md#prepare) both trial candidates through
`GraphPreparation.prepare(...)` and the integration's schedule assembler. Record [graph](../../../../glossary.md#graph) [compile](../../../../glossary.md#compile),
candidate enumeration, and each route's single prepare duration separately.

The retained hot measurement is exactly:

```text
PreparedExecutionRunner.run(prepared, borrowedInput) + RunResult.close()
```

It includes fresh [Runtime](../../../../glossary.md#term-runtime) output/workspace allocation, the [synchronous](../../../../glossary.md#term-synchronous) native invocation and wait,
and run-owned cleanup. It excludes graph compile, candidate enumeration, route preparation, caller
upload, tracing, output download, [checksum](../../../../glossary.md#term-checksum), [JSON](../../../../glossary.md#term-json) construction, and integration cleanup. A prepare
duration is a descriptive cold observation in fixed enumeration order, not a comparative [statistic](../../../../glossary.md#term-statistic).

### Correctness and anti-optimization

Before [warmup](../../../../glossary.md#term-warmup), [run](../../../../glossary.md#term-run) each ordinary untraced prepared route once, download the complete [canonical](../../../../glossary.md#term-canonical-form)
output, and compare every raw word with the expected sign-bit-flipped corpus. Both routes must
produce the same complete bytes and [SHA-256](../../../../glossary.md#term-sha-256).

After every retained timed [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), perform a separate untimed materializing run and repeat the
complete raw-bit comparison. Retain its SHA-256 in the report. The timed result itself remains
unmaterialized so [host](../../../../glossary.md#term-host) export is outside the hot boundary; [synchronous](../../../../glossary.md#term-synchronous) native calls, result closure,
and the observed materializing runs prevent a dead-result benchmark. Any mismatch fails the whole
command before a successful report.

## Profiles and [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters)

Profiles are normalized with `Locale.ROOT` and limited to:

| Profile | Paired [warmup](../../../../glossary.md#term-warmup) rounds | Paired measurement rounds | Retained [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) floor |
|---|---:|---:|---:|
| [`smoke`](../../../../glossary.md#term-smoke-test) | 2 | 2 | 0 ns |
| `baseline` | 4 | 8 | 25,000,000 ns per route batch |

Each paired round alternates candidate order: `A,B`, then `B,A`, repeating. Thus both routes occupy
each position equally without a random-order or [Shape](../../../../glossary.md#shape) [matrix](../../../../glossary.md#term-matrix-meanings). Each baseline batch uses the hardened
[CPU](../../../../glossary.md#term-cpu) benchmark's adaptive iteration calculation and a ceiling of 1,000,000 executions. No execution,
[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), round, retry, [outlier](../../../../glossary.md#term-outlier), or failure is discarded or replaced.

Retain, per route, every sample's iteration count, raw batch nanoseconds, integer normalized
per-[run](../../../../glossary.md#term-run) nanoseconds, complete post-batch [checksum](../../../../glossary.md#term-checksum), and minimum/[median](../../../../glossary.md#term-median)/maximum normalized summary.
State that one fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) invocation is the independent unit and within-process samples and paired
rounds are correlated.

Baseline adds and validates exact JVM [arguments](../../../../glossary.md#term-argument):

```text
-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch
--enable-native-access=ALL-UNNAMED
```

Smoke requires native access but does not claim controlled evidence.

## Native configuration and [metadata](../../../../glossary.md#term-metadata)

Register public [Gradle](../../../../glossary.md#term-gradle) task `:tools:benchmarks:metalBenchmark`; do not wire it into `build`, `check`,
or another [lifecycle](../../../../glossary.md#lifecycle) task. Reuse the existing explicit environment contract
`SYNAPTIK_METAL_TEST_LIBRARY`. The benchmark requires a nonblank absolute regular-file path and
constructs `MetalBackendConfiguration` from it. It performs no library discovery, extraction,
packaging, build, signing, or fallback.

Every successful report records:

- [schema](../../../../glossary.md#term-schema) and benchmark [identity](../../../../glossary.md#term-identity-meanings), profile, `eligibleForProductionDecision=false`,
  `autotuningEvidence=false`, and fixed-enumeration/no-winner selection mode;
- exact fixed workload/generator schema, profile, descriptor, Shape, element count, expected SHA-256,
  timing boundary, warmups, rounds, floor, iteration ceiling, no-retry rule, and actual round order;<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [element glossary](../../../../glossary.md#term-element))</span>
- Java/VM versions, exact JVM arguments, OS name/version/architecture, available processors, and
  observed initial/maximum heap;<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [arguments glossary](../../../../glossary.md#term-argument); [heap glossary](../../../../glossary.md#term-heap))</span>
- normalized real [dylib](../../../../glossary.md#term-dylib) path, byte size, [SHA-256](../../../../glossary.md#term-sha-256), and benchmark class SHA-256;
- [compile](../../../../glossary.md#compile)/enumeration/elapsed durations and both complete raw route rows;
- candidate index and public identity hex, compatibility schema and `SESSION` scope, attested route,
  successful PREPARE/RUN/native facts, <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [PREPARE glossary](../../../../glossary.md#prepare); [RUN glossary](../../../../glossary.md#term-run))</span>`NOT_QUERIED`, prepare duration, correctness hashes, raw
  samples, normalized samples, and distribution.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [distribution glossary](../../../../glossary.md#term-distribution))</span>

`baseline` additionally requires nonblank caller-supplied properties for exact base revision, source
identity, Git tree object SHA-1, harness-source SHA-256, [host](../../../../glossary.md#term-host)/[model](../../../../glossary.md#term-model) identity, [Metal](../../../../glossary.md#term-metal) [device](../../../../glossary.md#term-device) identity,
macOS product/build identity, native build identity including Xcode/clang/[SDK](../../../../glossary.md#term-sdk), power state, thermal
state, and a positive [fork](../../../../glossary.md#term-fork-meanings) number. Caller-supplied values are labeled as such; an honest explicit
`unknown` is preferable to fabrication where a host fact is unavailable. [Trace](../../../../glossary.md#trace) IDs are never used
as hardware facts.

## Failure and output contract

A successful invocation writes exactly one [JSON](../../../../glossary.md#term-json) document to standard output and exits zero. Missing
or invalid [metadata](../../../../glossary.md#term-metadata)/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) flags, wrong [host](../../../../glossary.md#term-host)/library configuration, open or [ABI](../../../../glossary.md#term-application-binary-interface-abi) failure, changed
[partition](../../../../glossary.md#partition)/candidate [identity](../../../../glossary.md#term-identity-meanings)/count/order, route-attestation mismatch, non-success [trace](../../../../glossary.md#trace) fact,
correctness mismatch, iteration ceiling, or retained-floor violation fails nonzero. It must not
silently skip [Metal](../../../../glossary.md#term-metal), downgrade a mandatory route to `unavailable`, fabricate a row, or emit a
successful partial report.

The harness writes no report, [cache](../../../../glossary.md#term-cache), baseline, [manifest](../../../../glossary.md#term-manifest), or other retained result [artifact](../../../../glossary.md#term-artifact). A caller
may [capture](../../../../glossary.md#term-capture-meanings) standard output externally. The task retains only command and [validation](../../../../glossary.md#term-validation) facts in this
brief.

## Output [schema](../../../../glossary.md#term-schema)

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

The schema contains no winner, recommendation, speedup claim, tuning [objective](../../../../glossary.md#term-objective), [cache](../../../../glossary.md#term-cache) path,
persistent decision, route/default mutation, or performance gate.

## Files and [ownership](../../../../glossary.md#term-ownership-meanings)

Implementation [scope](../../../../glossary.md#term-scope):

- `tools/benchmarks/build.gradle.kts`
- `tools/benchmarks/src/main/java/io/github/pho001/synaptik/tools/benchmarks/MetalRouteBenchmark.java`
- `docs/developer-guide/benchmarking.md`
- `docs/backend-guide/metal-backend.md`
- `docs/architecture/performance-evidence-and-tuning.md`
- this task, `docs/planning/backends/metal/master-plan.md`,
  `docs/planning/tools/benchmarks/master-plan.md`, and `docs/planning/roadmap.md`

Do not change production [Metal](../../../../glossary.md#term-metal)/[Engine](../../../../glossary.md#term-engine)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) sources, native sources or [ABI](../../../../glossary.md#term-application-binary-interface-abi),
capability/[conformance](../../../../glossary.md#term-conformance) tests, [CPU](../../../../glossary.md#term-cpu) benchmark behavior/[schema](../../../../glossary.md#term-schema)/task, tuning tooling, caches, or
architecture authority. No permanent benchmark [unit test](../../../../glossary.md#term-unit-test) is required: the actual [smoke](../../../../glossary.md#term-smoke-test)/baseline
commands exercise the supported surface and their [JSON](../../../../glossary.md#term-json)/failure [validation](../../../../glossary.md#term-validation) is the proof.

## [CI](../../../../glossary.md#term-continuous-integration-ci) boundary

- Default and portable CI may [compile](../../../../glossary.md#compile) and document the benchmark through
  `:tools:benchmarks:build`; it never runs `metalBenchmark` and needs no [dylib](../../../../glossary.md#term-dylib).
- An opt-in dedicated Apple-silicon macOS runner with an explicitly built/provided dylib may [run](../../../../glossary.md#term-run)
  [`smoke`](../../../../glossary.md#term-smoke-test) to validate [schema](../../../../glossary.md#term-schema), routes, correctness, and exit status. It must not gate on timing.
- `baseline` belongs only on a controlled local or dedicated host. Shared or virtualized CI must not
  publish, compare, or gate on its timings.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- A future route/default decision requires a separate pre-registered multi-fork/target/statistical
  protocol. This task neither defines nor runs that matrix.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [target glossary](../../../../glossary.md#term-target); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

## Acceptance

- `:tools:benchmarks:build` and `:tools:benchmarks:javadoc` pass without loading a native library;
  existing CPU benchmark behavior and schema remain unchanged.<span data-glossary-navigation="true"> ([native library glossary](../../../../glossary.md#term-native-library); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The exact [smoke](../../../../glossary.md#term-smoke-test) command emits one valid [schema](../../../../glossary.md#term-schema)-1 [JSON](../../../../glossary.md#term-json) document and exits zero with one fixed
  `[1_048_576]` [singleton](../../../../glossary.md#term-singleton) NEG workload and exactly two rows attested as `CUSTOM_KERNEL` and
  `GRAPH_EXECUTABLE`.
- Both rows use the identical graph, input, profile, and descriptor; both pass complete exact raw-bit
  precheck and every post-batch check; all raw counts/durations/checksums are retained.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [descriptor glossary](../../../../glossary.md#term-descriptor); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Timed execution uses only the separately opened ordinary no-trace integration. Attestation is
  untimed, reports <span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>`NOT_QUERIED`, and matches the ordinary candidate [identities](../../../../glossary.md#term-identity-meanings).
- The exact metadata-complete baseline command emits eight retained samples per route, validates
  fixed JVM flags, and every retained raw batch reaches 25 ms with no retry or discard.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- A focused failure invocation proves a missing required baseline metadata field fails nonzero
  before native benchmark work and emits no successful report.
- No `prepareTuned`, winner, complete-plan tuning, tools/tuning dependency, cache path or I/O,
  production mutation, native/capability change, timing threshold, silent skip, or retained result
  artifact exists.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [cache glossary](../../../../glossary.md#term-cache); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Documentation states the exact measurement boundary, metadata, commands, CI limits, correlation,
  and non-decision status without presenting observed timing as a route recommendation.<span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>
- [Planning](../../../../glossary.md#term-planning)/Markdown links, fences, final newlines, trailing whitespace, and `git diff --check` pass.
- [Run](../../../../glossary.md#term-run) exactly one actual `smoke` and one actual [metadata](../../../../glossary.md#term-metadata)-complete `baseline` on the configured
  Apple-silicon Metal device, validate both JSON documents structurally and semantically without
  retaining them in the repository, and run no Shape/operation/fork matrix.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [device glossary](../../../../glossary.md#term-device); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [fork glossary](../../../../glossary.md#term-fork-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Do not run the repository-wide build. After focused build/Javadocs, the two actual invocations,
  JSON validation, failure validation, docs/diff checks, and a clean implementation commit, set
  status to <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [commit glossary](../../../../glossary.md#term-commit))</span>`Review needed` for fresh independent Class C review.

## Implementation evidence

- The remediation changes no native source, ABI, or build input, so the native bridge was not
  rebuilt. Both updated invocations used the existing explicitly configured dylib with observed
  SHA-256 <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [native bridge glossary](../../../../glossary.md#term-native-bridge); [dylib glossary](../../../../glossary.md#term-dylib))</span>`8aebd8773d33e12b0348cf07b5d3789fc3631ab1d3bbf522691e8b97eb329190`.
- `./gradlew :tools:benchmarks:build :tools:benchmarks:javadoc` passed with 31 actionable tasks:
  three executed and 28 up to date. Javadoc emitted only the existing incubating Vector-module
  warnings. No native library was loaded by that command.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Vector glossary](../../../../glossary.md#term-vector); [module glossary](../../../../glossary.md#term-module); [native library glossary](../../../../glossary.md#term-native-library))</span>
- The one updated smoke invocation passed on Apple M3 Max and emitted one parsed schema-1 JSON
  document with benchmark-class SHA-256<span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test); [schema glossary](../../../../glossary.md#term-schema); [JSON glossary](../../../../glossary.md#term-json))</span>
  `5aedbaf2e731974fa0c4e4797ef47aacb7cbe835a5b87ab6d38d649ebe70d28a`. Parsed assertions proved
  generator schema 2's exact 16-word sentinel array, complete indexed tail formula/conversion/
  rounding fields, unchanged input/oracle hashes, Shape <span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [oracle glossary](../../../../glossary.md#term-test-oracle); [Shape glossary](../../../../glossary.md#shape))</span>`[1_048_576]`, routes `CUSTOM_KERNEL` and
  `GRAPH_EXECUTABLE`, two [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) per route, `SESSION`, `NOT_QUERIED`, and exact raw-bit [SHA-256](../../../../glossary.md#term-sha-256)
  `d95c57cdfc2aca246e868f7f050233768417ba2c1e5967c799afa05ea6c8bb02` for both prechecks and all
  four post-batch checks.
- The one updated [metadata](../../../../glossary.md#term-metadata)-complete baseline invocation below passed on MacBook Pro `Mac15,9`,
  Apple M3 Max 40-core GPU/Metal 4, macOS 26.6.2 build 25G83, AC power, with no thermal or
  performance warning recorded. The unchanged native build used Apple clang 21.0.0
  (<span data-glossary-navigation="true"> ([GPU glossary](../../../../glossary.md#term-gpu); [Metal glossary](../../../../glossary.md#term-metal))</span>`clang-2100.3.34.2`) and macOS [SDK](../../../../glossary.md#term-sdk) 27.0.
- Baseline parsed as generator schema 2 and eight samples for each attested route, retaining every
  execution. Custom sample iterations were `[42,42,42,53,53,53,53,53]`; [graph](../../../../glossary.md#graph) iterations were
  `[36,36,45,45,57,57,57,57]`. The minimum raw route-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) durations were respectively
  `27,109,542` ns and `25,979,209` ns, so all 16 retained batches met the 25 ms floor. Every
  precheck and post-batch checksum matched the fixed expected SHA-256.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum))</span>
- Smoke and baseline reported the same updated benchmark-class hash. Baseline reported the required
  heap/JIT/native-access arguments, remediation base revision above, staged executable tree OID<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [heap glossary](../../../../glossary.md#term-heap); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [arguments glossary](../../../../glossary.md#term-argument))</span>
  `1eeede489969c95b969ba91660ec8d06163b41fd`, harness-source SHA-256
  `df6d3b514ff5d791632fea1daaae1a903881fcc545c785972bca8173a34d2989`, [fork](../../../../glossary.md#term-fork-meanings) `1`, and every
  required caller-labeled machine/native fact.
- The original missing-metadata validation remains applicable because remediation did not change
  the Gradle task: it failed during task creation on <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`benchmarkBaseRevision`, before Java/native
  work, and emitted no successful report. The two updated successful JSON documents were parsed
  and validated in memory and were not retained in the repository.
- Markdown/link/fence/newline/trailing-whitespace [validation](../../../../glossary.md#term-validation) and `git diff --check` passed for the
  affected documentation. No full build, native rebuild, additional fork, Shape, operation, route,
  or performance matrix ran.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Same-reviewer Class C rereview at remediation
  `77e6091b2a452faa04fa2b674bc295ddc7be88b7` returned `APPROVE` with zero remaining findings.

## Commands

From repository root:

```bash
./native/metal-macos-arm64/build.sh
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew -q :tools:benchmarks:metalBenchmark -Pprofile=smoke
```

The exact baseline invocation used:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew -q :tools:benchmarks:metalBenchmark -Pprofile=baseline \
  -PbenchmarkBaseRevision=c3e12e7de949b8d3f2c5dad5f45b8086cfc8fedb \
  "-PbenchmarkSourceIdentity=Task0043 P2 remediation staged executable tree before evidence-only status update" \
  -PbenchmarkGitTreeObjectSha1=1eeede489969c95b969ba91660ec8d06163b41fd \
  -PbenchmarkHarnessSourceSha256=df6d3b514ff5d791632fea1daaae1a903881fcc545c785972bca8173a34d2989 \
  "-PbenchmarkHostIdentity=MacBook Pro Mac15,9; Apple M3 Max; 64 GB; 16 cores" \
  "-PbenchmarkMetalDeviceIdentity=Apple M3 Max; 40-core GPU; Metal 4" \
  "-PbenchmarkOsBuild=macOS 26.6.2; build 25G83" \
  "-PbenchmarkNativeBuildIdentity=Apple clang 21.0.0 clang-2100.3.34.2; macOS SDK 27.0" \
  "-PbenchmarkPowerState=AC power; battery 100% charged" \
  "-PbenchmarkThermalState=no thermal or performance warning recorded" \
  -PbenchmarkFork=1
```

The focused failure invocation omitted all `-Pbenchmark...` values from that command. Generated
[JSON](../../../../glossary.md#term-json) remained standard output only and is untracked.
