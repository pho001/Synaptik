# Task 0010A: Automatic [OpenBLAS](../../../../glossary.md#term-openblas) Discovery and Internal Composition Foundation

## Reader context — historical integration boundary

Discovery answers ‘which exact library could be loaded?’, not ‘which implementation should execute
this graph?’. In this milestone the CPU-local composition/test code calls discovery, the
exact-loading provider opens each candidate, and a separate session owns the successful library
until close or a later explicit transfer. Immutable attempt metadata does not own that handle. An
application is not required to invoke this internal seam. A loaded library is not yet a qualified
route, and a failed exact override is not permission to try a different library. The future-Engine
wording below records the stage at which this foundation was written.

Current reading: [Current OpenBLAS route](../../../../backend-guide/cpu-backend.md#current-narrow-openblas-matmul-route); [Supported lifecycle integration](../../../../backend-guide/cpu-backend.md#supported-lifecycle-integration).

## Status

Complete

## Goal

Add a bounded [CPU](../../../../glossary.md#term-cpu)-owned cold discovery and [lifetime](../../../../glossary.md#term-lifetime) foundation for the already implemented
[OpenBLAS](../../../../glossary.md#term-openblas) route. A caller of the unsupported CPU-internal composition seam chooses disabled,
automatic, exact-name, or exact-absolute-path loading. Automatic mode tries one fixed
platform-specific list of ordinary loader names and stable non-versioned installation paths,
stops at the first complete provider load, and otherwise records an unavailable result so CPU
composition can retain the portable route.

The low-level provider remains an exact name/path loader. Discovery does not make a loaded [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)
qualified, does not enable the OpenBLAS route by itself, and does not replace the current explicit
[qualification](../../../../glossary.md#term-qualification), expected-storage, single-thread, and complete-cost facts. The result [metadata](../../../../glossary.md#term-metadata) is
[immutable](../../../../glossary.md#term-immutable); a separate internal session owns the [mutable](../../../../glossary.md#term-mutable) provider lifetime. This internal split is
actionable before [Engine](../../../../glossary.md#term-engine) exists because CPU-local tests and the existing opt-in native [checkpoint](../../../../glossary.md#term-checkpoint)
can exercise it directly. Future Config, CPU composition-[API](../../../../glossary.md#term-application-programming-interface-api), and Engine tasks will define the
supported intent translation and [lifecycle](../../../../glossary.md#lifecycle) boundary without exposing these temporary internal
types as supported API; Engine must not reach into the CPU internal package directly.

## [Scope](../../../../glossary.md#term-scope)

- Add one [immutable](../../../../glossary.md#term-immutable) [CPU](../../../../glossary.md#term-cpu)-private discovery request with exactly four modes:
  `DISABLED`, `AUTOMATIC`, `EXACT_NAME`, and `EXACT_ABSOLUTE_PATH`.
- Add one immutable CPU-private discovery result containing the requested mode, the optional
  snapshotted platform classification used only by automatic mode, the exact ordered attempts, the selected exact
  name/path when one load succeeds, and a closed typed status of `DISABLED`, `UNAVAILABLE`, or
  `LOADED`. It retains no provider, native handle, segment, thread state, route plan, cost, or
  mutable exception object.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- Add one field-free CPU-private discovery operation and a deterministic package-private loader
  seam. Production loading delegates each candidate unchanged to<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
  `OpenBlasLibrary.open(String)` or `OpenBlasLibrary.open(Path)` and wraps the resulting library,
  borrowed invocation, and close action as one owned resource. Ordinary tests substitute a
  native-free owned resource or a typed load failure. The seam's helper interfaces/records remain
  nested implementation details of the four planned top-level types; they do not add another
  source file or public type.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Add one CPU-private composition session that owns exactly the first successfully loaded
  [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), exposes the immutable discovery result and an optional existing
  `CpuOpenBlasInvocation` view, and closes only that owned provider lifetime. Disabled and
  unavailable sessions own no provider and expose no invocation.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- In automatic mode only, snapshot operating-system family and architecture once during the
  explicit cold discovery call. The snapshot is input to deterministic candidate generation and
  is never reread during preparation, finalization, binding, or execution. Disabled and exact
  modes perform no platform lookup and record no platform snapshot.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [binding glossary](../../../../glossary.md#term-binding); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Preserve explicit override. An exact-name or exact-path request attempts only that selection;
  it never continues into automatic candidates after failure. Invalid explicit values fail request
  construction. A valid override that cannot load yields `UNAVAILABLE`, so later composition can
  retain portable execution.
- Preserve portable fallback. Disabled mode performs zero load attempts. Automatic mode records
  each failed bounded candidate and returns `UNAVAILABLE` after the final failure. Neither case
  throws merely because no OpenBLAS installation is usable.
- Extend the existing CPU [OpenBLAS](../../../../glossary.md#term-openblas) native [checkpoint](../../../../glossary.md#term-checkpoint) so its single [argument](../../../../glossary.md#term-argument) is either `--auto` or
  the existing exact absolute path. In automatic mode, close the discovery session after recording
  its exact selected name/path, reopen that exact selection through the provider, and run the
  existing isolated numerical/thread checkpoint. This bounded double load keeps production
  discovery free of thread control and avoids a temporary provider-handle escape from the session.
  The checkpoint may continue to install thread count one temporarily and restore it; production
  discovery itself never queries or mutates thread state.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Add focused CPU tests for the complete request, candidate ordering, stop/fallback, immutable
  metadata, loader delegation, session ownership, failure summaries, and no-route-enablement
  boundaries. All ordinary tests remain native-free.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Update the CPU internal-package inventory, OpenBLAS package Javadoc, CPU backend guide,
  glossary impact, task/master plan, and roadmap through the required separate clean
  documentation-focused pass after executable code stabilizes.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [backend glossary](../../../../glossary.md#backend))</span>

## Exact discovery and composition contract

### Request [validation](../../../../glossary.md#term-validation) and precedence

| Mode | Required value | Attempts | Failure outcome |
|---|---|---|---|
| `DISABLED` | No name or path | None | `DISABLED`; portable remains available. |
| `AUTOMATIC` | No name or path | The exact platform table below | First complete load wins; exhaustion yields `UNAVAILABLE`. |
| `EXACT_NAME` | One nonblank name | That exact unchanged name only | `UNAVAILABLE`; do not try automatic candidates. |
| `EXACT_ABSOLUTE_PATH` | One absolute path | That exact unchanged path only | `UNAVAILABLE`; do not try automatic candidates. |

The request snapshots its value. A null mode or null required exact value raises
`NullPointerException`; a blank exact name, relative exact path, or name/path component that
disagrees with the mode raises `IllegalArgumentException`. It reads no environment variable,
configuration file, command output, package-manager database, [classpath](../../../../glossary.md#term-classpath) service, or global
[registry](../../../../glossary.md#term-registry). Future public Config may express equivalent intent, but this task adds no Config type or
public [CPU](../../../../glossary.md#term-cpu) builder.

Explicit override has precedence because it is a distinct request mode, not the first entry in an
automatic list. A failed explicit override falls back to portable execution rather than silently
substituting another installed [binary](../../../../glossary.md#term-binary-descriptor-and-metadata). The caller may make a later separate automatic request.

### Bounded automatic candidate table

Candidate [identity](../../../../glossary.md#term-identity-meanings) distinguishes exact loader names from exact absolute paths. Candidate
generation uses only the snapshotted operating-system family and normalized architecture [token](../../../../glossary.md#term-token);
the fixed tables contain no duplicate typed selection:

| Platform [snapshot](../../../../glossary.md#term-snapshot) | Ordered automatic candidates |
|---|---|
| macOS, [`aarch64`](../../../../glossary.md#term-aarch64) or [`arm64`](../../../../glossary.md#native-packaging-and-platform-vocabulary) | name `openblas`; name `libopenblas.dylib`; path `/opt/homebrew/opt/openblas/lib/libopenblas.dylib`; path `/usr/local/opt/openblas/lib/libopenblas.dylib` |
| Other macOS architecture | name `openblas`; name `libopenblas.dylib`; path `/usr/local/opt/openblas/lib/libopenblas.dylib`; path `/opt/homebrew/opt/openblas/lib/libopenblas.dylib` |
| Linux | name `openblas`; name `libopenblas.so.0`; name `libopenblas.so` |
| Windows | name `openblas`; name `libopenblas.dll`; name `openblas.dll` |
| Other or unrecognized | name `openblas` |

This table is intentionally small and reviewable. Absolute candidates are stable non-versioned
Homebrew `opt` locations evidenced by the current macOS [checkpoint](../../../../glossary.md#term-checkpoint); discovery passes them directly
to the provider whether or not they exist. It performs no directory listing, recursive search,
glob, `PATH`/loader-path inspection, home-directory probing, package-manager invocation, or
versioned-file enumeration. Adding another location later requires a reviewed task and tests for
its exact order.

The production automatic-mode snapshot reads `os.name` and `os.arch` once in the cold entry, using
the empty string when a property is absent. It trims and lowercases both with `Locale.ROOT`.
Operating-system values beginning with `mac` or `darwin` classify as macOS, those beginning with
`linux` as Linux, and those beginning with `windows` as Windows; every other value is
unrecognized. Only the exact normalized architecture tokens `aarch64` and `arm64` select the
macOS ARM row. Tests inject exact snapshot strings rather than mutating global system properties.
The normalized tokens remain diagnostic [metadata](../../../../glossary.md#term-metadata), not [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) compatibility or [qualification](../../../../glossary.md#term-qualification)
evidence. Failure to read a property propagates rather than becoming [OpenBLAS](../../../../glossary.md#term-openblas) unavailability.

### Attempts and [immutable](../../../../glossary.md#term-immutable) result [metadata](../../../../glossary.md#term-metadata)

Every attempt records:

- its zero-based position;
- the exact typed name/path selection passed to the provider;
- `LOADED` or `LOAD_OR_BIND_FAILURE`; and
- for failure only, immutable diagnostic strings for the thrown type and nullable message.

The nested checked loader failure carries those two diagnostic strings. The production adapter
copies them from the caught `OpenBlasLoadException`; it does not replace them with the nested
failure's own type or retain the provider exception after the discovery call.

The result defensively snapshots the ordered attempt list and validates all status relationships.
It does not retain a `Throwable`, provider, [lookup](../../../../glossary.md#term-lookup), native address, file contents, [mutable](../../../../glossary.md#term-mutable) list, or
live platform query. A loaded result has exactly one successful final attempt and names that same
selection. A disabled result has no attempts or selection. An unavailable result has one or more
failed attempts, except that an automatic platform table is never empty under this contract.

The native-free loader seam reports an ordinary attempt failure through one nested checked
failure type. Its production adapter converts only `OpenBlasLoadException` into that type. Because
the provider preserves its original cause after its own broad failure translation, the adapter
first walks that cause chain and rethrows any `VirtualMachineError` or `ThreadDeath`; those fatal
failures never become portable unavailability. Request-[validation](../../../../glossary.md#term-validation) defects and unrelated [CPU](../../../../glossary.md#term-cpu)
programming failures also propagate rather than becoming failed attempts. If production
adaptation fails after a provider opened, discovery closes that handle before propagating the
adaptation failure, suppressing a distinct cleanup failure on the primary failure.

### Session [ownership](../../../../glossary.md#term-ownership-meanings) and route boundary

The internal discovery session is the composition-[lifetime](../../../../glossary.md#term-lifetime) [owner](../../../../glossary.md#term-owner), not the [immutable](../../../../glossary.md#term-immutable) result. It
has these rules:

- one production loaded session strongly owns exactly one [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) and the existing
  borrowed <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>`CpuOpenBlasInvocation` adapter; the native-free test seam supplies an equivalent fake
  invocation plus one observable close action;
- disabled and unavailable sessions own neither;
- close atomically claims the session once, propagates any unchecked close failure without a
  retry on a later close, and never restores a thread count because this task never changes one;
- composition must keep a loaded session open until every prepared executable borrowing its
  invocation is quiescent, and must not race close with provider use; and<span data-glossary-navigation="true"> ([prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [borrowing glossary](../../../../glossary.md#term-borrowing); [quiescent glossary](../../../../glossary.md#term-quiescent); [race glossary](../../../../glossary.md#term-race))</span>
- result metadata remains readable after session close, while the borrowed invocation observes
  the provider's existing closed-state behavior.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>

A `LOADED` result means only that the exact provider selection loaded and bound the current four
required symbols. It proves neither 32-bit-`blasint` behavior nor numerical compatibility,
thread-state isolation, route eligibility, or profitability. Until [CPU](../../../../glossary.md#term-cpu) 0010D automates installed-
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [qualification](../../../../glossary.md#term-qualification), current composition may pair the session with a caller-supplied
`OpenBlasRouteConfig.QUALIFIED` [snapshot](../../../../glossary.md#term-snapshot) only when the caller's existing [checkpoint](../../../../glossary.md#term-checkpoint) evidence
actually covers the selected binary. Otherwise it supplies `OpenBlasRouteConfig.DISABLED` and
portable remains selected. This task does not weaken `OpenBlasRouteConfig.complete()` or infer
qualification from discovery.

Current `SINGLE_THREAD` remains unchanged. Discovery neither calls `threadCount()` nor
`setThreadCount(...)`, creates a lock, installs a worker budget, or coordinates another [OpenBLAS](../../../../glossary.md#term-openblas)
owner. CPU 0010C owns that later thread-state and shared-budget capability.

## Out of [scope](../../../../glossary.md#term-scope)

- Treating successful loading or four-symbol binding as ABI, numerical, determinism, thread,
  target, or performance qualification.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [binding glossary](../../../../glossary.md#term-binding); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [target glossary](../../../../glossary.md#term-target); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Binary content/version fingerprinting, resolved-loader-path recovery, persistent target
  identity, invalidation, optional <span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`openblas_get_config`/`openblas_get_corename`, or [CPU](../../../../glossary.md#term-cpu) 0010D.
- Any OpenBLAS provider public/source/test change. The provider remains an exact name/path loader
  and retains its current four required symbols, 32-bit integer descriptors, GEMM, thread-control,
  lifecycle, and failure contracts.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [descriptors glossary](../../../../glossary.md#term-descriptor); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Changing current OpenBLAS route eligibility, representation forms, cost equations, native
  invocation, prepared execution, no-late-fallback behavior, or portable MATMUL semantics.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Transpose, two-input or output materialization, batch, broadcast, epilogue, packing, or CPU
  0010B.<span data-glossary-navigation="true"> ([Transpose glossary](../../../../glossary.md#term-transpose); [materialization glossary](../../../../glossary.md#materialization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- Multithreaded OpenBLAS, shared CPU permits, global-state serialization/restoration, concurrent
  GEMM policy, tuning candidates, or CPU 0010C.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>
- Model autotuning, workload signatures, persistent tuning evidence, measurement, cache lookup or
  mutation, benchmarking, or CPU 0010E/0016.<span data-glossary-navigation="true"> ([Model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [signatures glossary](../../../../glossary.md#term-signature); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- Downloading, installing, updating, bundling, extracting, or copying an OpenBLAS binary; mutating
  the filesystem, loader configuration, environment, system properties, or package-manager state.
- Arbitrary filesystem search, directory enumeration, wildcard matching, subprocess discovery,
  background refresh, watcher, retry loop, time-based cache, singleton, service locator,<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
  `ServiceLoader`, reflection, annotation scanning, or hidden global [registry](../../../../glossary.md#term-registry).
- Public Config, Engine, Tensor, Compiler, Training, Prepare, Runtime, backend-contract, or Trace
  API changes; another <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid); public [backend](../../../../glossary.md#backend) registration; or end-to-end application use.
- Discovery or provider selection during backend analysis, finalization, cold Runtime binding,
  execution, or any hot path. The session must already exist before those phases consume it.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Architecture, ADR, module dependency, Gradle dependency, shared-build, backend-conformance, or
  integration-test changes.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially explicit composition, CPU
  routes, OpenBLAS provider ownership, staged preparation, runtime prohibitions, and performance
  evidence<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering))</span>
- [ADR 0006: No runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)<span data-glossary-navigation="true"> ([service locator glossary](../../../../glossary.md#term-service-locator))</span>
- [ADR 0008: Performance evidence and tuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [ADR 0010: Staged preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run resources](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU task 0010](0010-narrow-openblas-blas-compatible-native-route.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading/binding](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [Provider 0002 GEMM invocation](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- [Provider 0003 thread control/checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [Config master plan](../../../modules/config/master-plan.md)
- [Engine master plan](../../../modules/engine/master-plan.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Benchmark master plan](../../../tools/benchmarks/master-plan.md)

## Architecture constraints

- Engine remains the future outer composition root. This task provides an unsupported CPU-
  internal seam only and creates no supported public composition API or reverse CPU-to-Engine
  dependency.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [composition root glossary](../../../../glossary.md#term-composition-root); [CPU glossary](../../../../glossary.md#term-cpu); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- CPU owns OpenBLAS discovery policy because the provider must remain a low-level leaf. Every
  actual load still delegates one exact candidate unchanged to the provider.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Discovery is explicit and cold. Backend analysis receives only already-resolved immutable facts;
  finalization receives only an already-borrowed invocation; Runtime receives only prepared work.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [immutable glossary](../../../../glossary.md#term-immutable); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- A portable plan is always established independently for every currently supported occurrence.
  Discovery absence never removes portable support, and a selected native plan still never falls
  back after analysis.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `LOADED` and `QUALIFIED` remain separate facts. Only CPU-owned qualification can authorize a
  provider for route selection.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
- The internal request, result, attempts, selections, and platform snapshot are immutable typed
  values. Do not use <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`Map<String,Object>`, string [dispatch](../../../../glossary.md#term-dispatch), a [registry](../../../../glossary.md#term-registry), or reflection.
- The live session is explicitly caller-owned and separate from immutable discovery metadata. It
  does not become a hidden process-global cache or prepared/runtime resource.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [cache glossary](../../../../glossary.md#term-cache))</span>
- No architecture rule, module edge, or provider API changes. If implementation needs resolved
  operating-system library paths, optional provider metadata symbols, a public Config type, Engine
  lifecycle, or shared global coordination to satisfy this task, stop and report the exact gap.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.provider.openblas`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Packages added or changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Type placement:

- `CpuOpenBlasDiscoveryRequest` — immutable disabled/automatic/exact override intent beside the
  route whose provider it selects.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `CpuOpenBlasDiscoveryResult` — immutable typed platform, attempt, selection, and outcome facts;
  it owns no live resource.
- `CpuOpenBlasDiscovery` — stateless cold candidate generation and exact-loader delegation, with a
  package-private native-free test seam whose nested checked failure distinguishes expected
  load/bind absence from programming and fatal failures.
- `CpuOpenBlasDiscoverySession` — explicit CPU-internal composition owner for the selected provider
  lifetime and existing borrowed invocation adapter; its nested owned-resource form also permits
  a close-observable native-free fake in ordinary tests.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lifetime glossary](../../../../glossary.md#term-lifetime); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>

All four types remain package-private. `CpuCapabilityProvider` remains the sole supported public
[CPU](../../../../glossary.md#term-cpu) [API](../../../../glossary.md#term-application-programming-interface-api). A future CPU composition task may deliberately wrap, move, replace, or change the
visibility of this seam behind a supported CPU-owned [facade](../../../../glossary.md#term-facade) when Config and [Engine](../../../../glossary.md#term-engine) exist. That
cross-package boundary is future work; this task does not make these types temporarily accessible
to another CPU package or [module](../../../../glossary.md#term-module).

## Affected files

Expected production and test paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscoveryRequest.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscoveryResult.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscovery.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscoverySession.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscoveryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasNativeCheckpoint.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0010a-automatic-openblas-discovery-and-internal-composition-foundation.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the 13 exact paths above. No provider, Config, [Engine](../../../../glossary.md#term-engine),
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [Gradle](../../../../glossary.md#term-gradle), settings, or other
source/test/documentation path may change. If another path is required, stop and propose a
follow-up or architecture clarification before editing it.

## Acceptance criteria

1. The four package-private CPU-owned types implement the exact request/result/discovery/session
   split, and every immutable value defensively snapshots its members and validates all status
   relationships.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
2. Disabled mode performs no loader or platform [lookup](../../../../glossary.md#term-lookup) and returns a resource-free `DISABLED`
   session. Exact modes also perform no platform lookup. Only automatic mode snapshots production
   platform properties once.
3. Exact-name and exact-path modes delegate only the exact supplied value once. They never append
   automatic candidates; load/bind failure returns a resource-free `UNAVAILABLE` result.
4. Automatic platform classification and candidate ordering are exactly the rules and table above
   for macOS arm64, other macOS, Linux, Windows, and unrecognized platforms. Tests prove exact
   normalization, absence handling, and first-success stop.<span data-glossary-navigation="true"> ([arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [normalization glossary](../../../../glossary.md#term-normalization))</span>
5. The production adapter calls only `OpenBlasLibrary.open(String)` or
   `OpenBlasLibrary.open(Path)`. No provider code or public API changes, and no alternate native
   loading path exists in CPU.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
6. Failed attempts retain only immutable typed selection plus failure type/message strings. A
   loaded result identifies the final successful attempt exactly and cannot be constructed with
   inconsistent attempt metadata. The production adapter converts <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>`OpenBlasLoadException` only;
   it rethrows `VirtualMachineError` or `ThreadDeath` found in that exception's cause chain, and
   unrelated failures propagate.
7. A production loaded session owns exactly one provider and existing invocation adapter; the
   native-free seam proves the same one-close ownership with a fake. Close is idempotent, exposes
   the provider's ordinary closed behavior to the borrowed invocation, and performs no thread
   query, setter, restoration, or filesystem mutation.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [idempotent glossary](../../../../glossary.md#term-idempotent); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
8. Disabled/unavailable composition maps to no invocation and cannot accidentally make
   `OpenBlasRouteConfig.complete()` true. Loaded composition also does not synthesize
   `QUALIFIED`, cost, storage, or thread facts.
9. The existing portable default, current `SINGLE_THREAD` route, exact 0010 eligibility/cost
   equations, direct/one-copy execution, provider contract, and no-late-fallback behavior remain
   unchanged under focused regression tests.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
10. The opt-in [CPU](../../../../glossary.md#term-cpu) native [checkpoint](../../../../glossary.md#term-checkpoint) accepts `--auto`, reports the exact selected name/path,
    closes the discovery session, and reopens that exact selection for the existing numerical and
    thread-restoration assertions without requiring a version-specific Cellar path. Its existing
    single absolute-path argument remains exact and exclusive.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
11. Source and bytecode/API-shape scans confirm no new supported public CPU type, static mutable
    discovery state, singleton, registry, <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [mutable glossary](../../../../glossary.md#term-mutable); [singleton glossary](../../../../glossary.md#term-singleton); [registry glossary](../../../../glossary.md#term-registry))</span>`ServiceLoader`, filesystem enumeration, process launch,
    network/download/install code, provider mutation, or discovery reference in Runtime/hot
    execution.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
12. `CpuInternalPackageInventoryTest` records exactly the four new production types. The new
    focused test remains in the ordinary test inventory; unrelated production inventory and
    generated-kernel schemas remain unchanged.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [schemas glossary](../../../../glossary.md#term-schema))</span>
13. A distinct clean documentation-focused agent reviews the final code/test diff, finalizes
    affected Javadoc, the OpenBLAS package summary, CPU guide, glossary impact, task/master plan,
    and roadmap under the required profiles, and records reasoned no-change conclusions for
    provider docs/API, public Config/Engine/Tensor/Compile/Training APIs, architecture/ADRs, shared
    Prepare/Runtime, backend conformance, integration, other modules, and generated artifacts.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Engine glossary](../../../../glossary.md#term-engine); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
14. Focused CPU tests, CPU Javadoc, the opt-in automatic native checkpoint, documentation checks,
    exact-path scope, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass. Repository-wide validation remains deferred to
    the 0010A–0010E capability checkpoint or CI because this task changes one module and no shared
    contract or dependency.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused new tests as needed, then record one final affected-[module](../../../../glossary.md#term-module) run
after executable code stabilizes:

```bash
./gradlew :backends:cpu:test
```

All ordinary tests use the package-private fake loader and current fake invocation seam. They need
no installed [OpenBLAS](../../../../glossary.md#term-openblas) library, native-access permission, system-property mutation, or platform-
specific [host](../../../../glossary.md#term-host).

On a host with a compatible installed OpenBLAS [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), [compile](../../../../glossary.md#compile) and run the existing isolated
[checkpoint](../../../../glossary.md#term-checkpoint) in automatic mode:

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  --auto
```

The command must print the exact automatic selection and the existing pass/restoration result. If
automatic discovery is unavailable on the implementation host, the task remains `Incomplete`
until the cause is recorded and either the fixed candidate table is corrected within [scope](../../../../glossary.md#term-scope) or the
required environment becomes available. The exact-path checkpoint remains available for diagnosis
but does not substitute for the automatic-mode acceptance evidence.

The documentation-focused pass reuses the final Java/native evidence unless it changes executable
Java or tests, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It also renders and inspects the affected [Javadoc](../../../../glossary.md#term-javadoc), validates local Markdown links and anchors,
heading order/uniqueness, balanced fences, LF/final newlines, terminology, the exact 13-path
ceiling, internal/public inventory, final `0010A Complete` plus `0010B`–`0010E Draft` ordering,
unchanged
`0011 Blocked`, and absence of any detailed 0010B–0010E task file.

Repository-wide validation is deferred to the 0010A–0010E capability checkpoint or [CI](../../../../glossary.md#term-continuous-integration-ci) unless the
implementation changes a repository-wide contract. [Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) and integration remain
unchanged because no [Engine](../../../../glossary.md#term-engine)/public [lifecycle](../../../../glossary.md#lifecycle) or executable route semantics are added.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0010 is `Complete` and supplies the route, provider-free invocation adapter, explicit
  qualification/config facts, finalization, and native checkpoint.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- The [OpenBLAS](../../../../glossary.md#term-openblas) provider milestone is `Complete` and supplies exact name/path loading, the required
  four-symbol binding, GEMM, thread control, and caller-owned lifecycle without discovery.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [binding glossary](../../../../glossary.md#term-binding); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Current CPU package inventory and documentation contracts are stable.
- No Config, Engine, Prepare, Runtime, tuning, benchmark, or new provider prerequisite is required
  for this internal foundation.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Follow-up tasks

- CPU 0010B broadens exact OpenBLAS MATMUL representations through bounded two-input and output
  materialization while retaining the current non-transposed provider call; batching,
  broadcasting, and epilogues remain later evidence-driven work.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [materialization glossary](../../../../glossary.md#materialization); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- CPU 0010C adds explicit composition-owned OpenBLAS thread-state coordination and one CPU thread
  budget. It may consume the discovery session but must not turn it into a singleton or provider
  policy owner.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [owner glossary](../../../../glossary.md#term-owner))</span>
- CPU 0010D separates installed-binary qualification and stable target fingerprinting from this
  loading result. It is the first task allowed to turn discovered binary evidence into an
  automatically compatible qualification fact.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [qualification glossary](../../../../glossary.md#term-qualification); [target glossary](../../../../glossary.md#term-target))</span>
- CPU 0010E adds typed OpenBLAS candidates and compatible tuning-evidence consumption with
  measurement/cache mutation still owned by <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>`tools/tuning`.
- Future Config work defines supported declarative intent; a future CPU composition-API task wraps
  or replaces this package-private seam behind a supported CPU-owned boundary; and Engine 0001
  owns outer application lifecycle, explicit registration, and user-visible portable fallback.
  Engine must not import the internal package or expose these temporary types.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [CPU](../../../../glossary.md#term-cpu) 0011 remains `Blocked` independently on a concrete Intel use case and supported oneMKL ABI
  evidence. Completing 0010A does not satisfy or bypass those gates.<span data-glossary-navigation="true"> ([oneMKL glossary](../../../../glossary.md#term-onemkl); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>

## Architecture impact

Expected impact: None. The task implements the already-authorized rule that [CPU](../../../../glossary.md#term-cpu) owns route and
provider-composition policy while the provider remains an exact loader and [Engine](../../../../glossary.md#term-engine) remains the
future outer [composition root](../../../../glossary.md#term-composition-root). It adds no [module](../../../../glossary.md#term-module) edge, supported public [API](../../../../glossary.md#term-application-programming-interface-api), [service locator](../../../../glossary.md#term-service-locator),
global [registry](../../../../glossary.md#term-registry), [Runtime](../../../../glossary.md#term-runtime) behavior, or architecture rule.

If implementation requires a provider API change, public Config or Engine type, resolved native-
library path API, cross-module resource [lifecycle](../../../../glossary.md#lifecycle), global coordinator, or discovery after the
explicit composition phase, stop and report the conflict instead of expanding this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/backends/cpu/tasks/0010a-automatic-openblas-discovery-and-internal-composition-foundation.md
in full. Implement that task exactly within its 13-path ceiling. Do not use a GSD workflow. Stop
for any architecture, provider, public-API, or scope conflict instead of inventing a boundary.

After executable code and the final CPU/native validation stabilize, hand the exact diff and
recorded evidence to a distinct clean documentation-focused agent. That pass must follow
documentation-rules.md, finalize all affected Javadocs/explanatory/planning documentation and
glossary impact in the same overall change, and must not repeat successful Java tests unless it
changes executable behavior or records a concrete reason. Update the task with implementation
notes, exact evidence, completion summary, and final status only after that pass succeeds.
```

## Local decisions

- The task is [`Ready`](../../../../glossary.md#term-ready) before Engine because its request, result, loader, and lifetime seam are
  package-private CPU implementation details with native-free tests. Public request translation
  and application lifecycle remain deferred rather than approximated.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifetime glossary](../../../../glossary.md#term-lifetime); [CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Automatic discovery is bounded candidate probing, not arbitrary filesystem search. The only
  absolute automatic candidates are the two stable non-versioned Homebrew `opt` paths evidenced
  by the current macOS installation model.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Exact override is exclusive. Failure never silently changes the requested library; portable is
  the fallback.
- Immutable discovery metadata and live provider ownership are separate types. This avoids
  describing a closeable provider handle as an immutable result.<span data-glossary-navigation="true"> ([Immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Loading and qualification remain separate. Existing explicit checkpoint-qualified facts can be
  paired manually during the transition, while CPU 0010D owns automatic qualification and target
  fingerprinting.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [target glossary](../../../../glossary.md#term-target))</span>

## Known limitations

- Automatic discovery covers only the fixed names and two stable macOS paths above. It does not
  find versioned files or nonstandard installations outside the operating-system loader search
  configuration; exact override remains available.
- The JDK name-based loader does not expose the resolved binary path. This task therefore cannot
  create a stable persistent binary fingerprint from a successful name load. CPU 0010D must
  either establish an exact identity mechanism or classify such results as session-only for
  persistent compatibility.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [CPU glossary](../../../../glossary.md#term-cpu); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Successful discovery proves only loading and required-symbol binding. Current native routing
  still needs explicitly associated qualification, single-thread, expected-storage, and cost
  facts.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [binding glossary](../../../../glossary.md#term-binding); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- No public Config or Engine lifecycle exists. The foundation is intentionally usable only by
  CPU-local composition/tests until those owners are implemented.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The explicit session coordinates no other handle, class loader, engine, or arbitrary native
  consumer and makes no thread-state isolation guarantee.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a08f99-6951-7ad1-825f-08912ba2f037` ran
  `./gradlew :backends:cpu:test`: passed 189 suites and 958 tests with zero failures or errors and
  28 existing conditional skips. `CpuOpenBlasDiscoveryTest` passed all 16 tests. Executable Java
  and tests did not change afterward; documentation context
  `01a08fd1-151c-7111-b094-a1140cf7ed04` therefore reused this evidence without repeating the
  suite.
- The implementation context ran `./gradlew :backends:cpu:testClasses`: passed. It then ran the
  specified `CpuOpenBlasNativeCheckpoint --auto` command with native access enabled. Automatic
  discovery selected `/opt/homebrew/opt/openblas/lib/libopenblas.dylib`, the checkpoint reopened
  that exact selection, confirmed OpenBLAS 0.3.34, exercised FLOAT32/FLOAT64 direct and one-copy
  prepared routes, and reported <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`CPU OpenBLAS native checkpoint passed; restored thread count 16`.
- The implementation context's Class-File inspection confirmed that all four new top-level types
  are package-private and `CpuOpenBlasDiscovery` is field-free. Its `git diff --check` passed.
- Documentation context `01a08fd1-151c-7111-b094-a1140cf7ed04` applied the General,
  API/Javadoc, Backend Guide, Planning, and Example profiles, inspected all eight executable paths
  and the complete diff, and changed only Javadoc plus the five authorized documentation/planning
  paths. No executable statement or test behavior changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation context ran `./gradlew :backends:cpu:javadoc`: passed with two incubating-
  module warnings and 92 unrelated pre-existing missing-<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`@param` warnings in
  `CpuPartitionLowering.LoweredPartition` and `CpuPartitionPreparationPlan` records. Generated
  package and discovery/request/result/session pages were inspected for request precedence,
  immutable diagnostics, failure contracts, route non-enablement, ownership, idempotent close,
  post-close metadata, borrowed invocation, and thread/lifecycle boundaries.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [idempotent glossary](../../../../glossary.md#term-idempotent); [metadata glossary](../../../../glossary.md#term-metadata); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The documentation context ran
  `python3 /tmp/validate_synaptik_markdown.py docs/backend-guide/cpu-backend.md docs/glossary.md docs/planning/backends/cpu/tasks/0010a-automatic-openblas-discovery-and-internal-composition-foundation.md docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md`:
  passed with `validated 5 Markdown files`, covering local links and anchors, unique headings,
  balanced fences, LF/final newlines, and trailing whitespace.
- Final manual checks confirmed the exact 13 authorized worktree paths; four new package-private
  production types; immutable automatic attempt metadata; exclusive exact overrides; no provider,
  route, configuration, Engine, Prepare, Runtime, generated-artifact, architecture, dependency, or
  Gradle change; synchronized 0010A <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [artifact glossary](../../../../glossary.md#term-artifact); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`Complete`; 0010B–0010E `Draft`; 0011 `Blocked`; and no
  detailed 0010B–0010E task file. The final combined
  `git status --short -uall`/tracked-plus-untracked path check reported exactly those 13 paths, and
  `git diff --check` passed with no output. A separate `rg` trailing-whitespace scan and final-byte
  newline check passed for all five untracked Java files, which Git's unstaged diff does not cover.
- No-change conclusions: provider API/documentation remain accurate because CPU delegates exact
  selections unchanged and adds no provider behavior; public Config, Engine, Tensor, Compile, and
  Training APIs expose no discovery or route activation; shared Prepare and Runtime receive no
  discovery work or lifetime; architecture/ADRs and architecture tests need no update because
  ownership, lifecycle boundaries, and module dependencies are unchanged; backend-conformance and
  integration need no update because executable route semantics and public end-to-end composition
  are unchanged; generated artifacts and schemas are unaffected; and other modules, Gradle, root
  settings, native packaging, and environment configuration remain unchanged.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [activation glossary](../../../../glossary.md#term-activation); [lifetime glossary](../../../../glossary.md#term-lifetime); [conformance glossary](../../../../glossary.md#term-conformance); [schemas glossary](../../../../glossary.md#term-schema))</span>

## Implementation notes

- Added immutable disabled/automatic/exact-name/exact-path request and result contracts, bounded
  platform classification and candidate ordering, immutable attempt diagnostics, and exact loader
  delegation. No deviation from the specified table or precedence was required.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Added a separate caller-owned session that retains one loaded provider-equivalent resource,
  exposes the borrowed invocation without qualifying a route, and atomically claims close once.
  Disabled and unavailable sessions remain resource-free.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Extended the isolated [CPU](../../../../glossary.md#term-cpu) [checkpoint](../../../../glossary.md#term-checkpoint) with `--auto`; it closes discovery ownership and reopens
  only the selected exact name/path before the existing qualification-style route checks and
  caller-coordinated thread restoration.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Finalized implementation-contract Javadocs, the OpenBLAS package summary, CPU guide, glossary
  terminology, task evidence, master-plan frontier, and roadmap status without changing executable
  behavior.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>

## Completion summary

- Completed changes: implemented the exact bounded CPU-private OpenBLAS discovery request, result,
  operation, and lifetime session; extended the native checkpoint; and finalized Javadocs,
  explanatory documentation, glossary terminology, and planning status.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [operation glossary](../../../../glossary.md#operation); [lifetime glossary](../../../../glossary.md#term-lifetime); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Files changed or created: exactly the 13 authorized paths—four new discovery production types,
  OpenBLAS package documentation, discovery tests, native checkpoint, CPU package inventory, CPU
  backend guide, glossary, this task, CPU master plan, and roadmap.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Tests and validation performed: reused the implementation context's passing 189-suite/958-test
  CPU run, passing test-class compilation, automatic native checkpoint, Class-File visibility/
  field scan, and whitespace result; the documentation context passed CPU Javadoc, generated-page
  inspection, five-file Markdown validation, exact-scope/status/frontier/manual checks, and final<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check`.
- Documentation-agent review: clean documentation context
  `01a08fd1-151c-7111-b094-a1140cf7ed04` independently reviewed the final code/tests and finalized
  all affected Javadocs and documentation. It made no executable behavior change and did not
  duplicate the stable Java suite or native checkpoint.
- Documentation impact: the CPU guide and glossary now distinguish bounded loading evidence,
  immutable metadata, session ownership, later qualification, route selection, and public
  composition. Task, master plan, and roadmap consistently close 0010A without advancing 0010B.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Javadoc review: every new production type and explicit constructor/method documents purpose,
  inputs, results, failures, immutability, ownership, lifecycle, concurrency, and thread-state
  boundaries as applicable; the changed checkpoint entry points document their exact inputs and
  failures.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Glossary impact: added the reusable OpenBLAS discovery-result/session distinction and clarified
  that CPU discovery delegates to the unchanged exact-loading provider.
- Unresolved issues: None.
- Follow-up required: None for 0010A. [CPU](../../../../glossary.md#term-cpu) 0010B is the next [planning](../../../../glossary.md#term-planning) frontier and remains `Draft`;
  CPU 0010C–0010E remain `Draft`, and CPU 0011 remains `Blocked`.

Status: Complete
