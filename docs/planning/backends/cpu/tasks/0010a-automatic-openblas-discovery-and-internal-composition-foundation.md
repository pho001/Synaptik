# Task 0010A: Automatic OpenBLAS Discovery and Internal Composition Foundation

## Status

Complete

## Goal

Add a bounded CPU-owned cold discovery and lifetime foundation for the already implemented
OpenBLAS route. A caller of the unsupported CPU-internal composition seam chooses disabled,
automatic, exact-name, or exact-absolute-path loading. Automatic mode tries one fixed
platform-specific list of ordinary loader names and stable non-versioned installation paths,
stops at the first complete provider load, and otherwise records an unavailable result so CPU
composition can retain the portable route.

The low-level provider remains an exact name/path loader. Discovery does not make a loaded binary
qualified, does not enable the OpenBLAS route by itself, and does not replace the current explicit
qualification, expected-storage, single-thread, and complete-cost facts. The result metadata is
immutable; a separate internal session owns the mutable provider lifetime. This internal split is
actionable before Engine exists because CPU-local tests and the existing opt-in native checkpoint
can exercise it directly. Future Config, CPU composition-API, and Engine tasks will define the
supported intent translation and lifecycle boundary without exposing these temporary internal
types as supported API; Engine must not reach into the CPU internal package directly.

## Scope

- Add one immutable CPU-private discovery request with exactly four modes:
  `DISABLED`, `AUTOMATIC`, `EXACT_NAME`, and `EXACT_ABSOLUTE_PATH`.
- Add one immutable CPU-private discovery result containing the requested mode, the optional
  snapshotted platform classification used only by automatic mode, the exact ordered attempts, the selected exact
  name/path when one load succeeds, and a closed typed status of `DISABLED`, `UNAVAILABLE`, or
  `LOADED`. It retains no provider, native handle, segment, thread state, route plan, cost, or
  mutable exception object.
- Add one field-free CPU-private discovery operation and a deterministic package-private loader
  seam. Production loading delegates each candidate unchanged to
  `OpenBlasLibrary.open(String)` or `OpenBlasLibrary.open(Path)` and wraps the resulting library,
  borrowed invocation, and close action as one owned resource. Ordinary tests substitute a
  native-free owned resource or a typed load failure. The seam's helper interfaces/records remain
  nested implementation details of the four planned top-level types; they do not add another
  source file or public type.
- Add one CPU-private composition session that owns exactly the first successfully loaded
  `OpenBlasLibrary`, exposes the immutable discovery result and an optional existing
  `CpuOpenBlasInvocation` view, and closes only that owned provider lifetime. Disabled and
  unavailable sessions own no provider and expose no invocation.
- In automatic mode only, snapshot operating-system family and architecture once during the
  explicit cold discovery call. The snapshot is input to deterministic candidate generation and
  is never reread during preparation, finalization, binding, or execution. Disabled and exact
  modes perform no platform lookup and record no platform snapshot.
- Preserve explicit override. An exact-name or exact-path request attempts only that selection;
  it never continues into automatic candidates after failure. Invalid explicit values fail request
  construction. A valid override that cannot load yields `UNAVAILABLE`, so later composition can
  retain portable execution.
- Preserve portable fallback. Disabled mode performs zero load attempts. Automatic mode records
  each failed bounded candidate and returns `UNAVAILABLE` after the final failure. Neither case
  throws merely because no OpenBLAS installation is usable.
- Extend the existing CPU OpenBLAS native checkpoint so its single argument is either `--auto` or
  the existing exact absolute path. In automatic mode, close the discovery session after recording
  its exact selected name/path, reopen that exact selection through the provider, and run the
  existing isolated numerical/thread checkpoint. This bounded double load keeps production
  discovery free of thread control and avoids a temporary provider-handle escape from the session.
  The checkpoint may continue to install thread count one temporarily and restore it; production
  discovery itself never queries or mutates thread state.
- Add focused CPU tests for the complete request, candidate ordering, stop/fallback, immutable
  metadata, loader delegation, session ownership, failure summaries, and no-route-enablement
  boundaries. All ordinary tests remain native-free.
- Update the CPU internal-package inventory, OpenBLAS package Javadoc, CPU backend guide,
  glossary impact, task/master plan, and roadmap through the required separate clean
  documentation-focused pass after executable code stabilizes.

## Exact discovery and composition contract

### Request validation and precedence

| Mode | Required value | Attempts | Failure outcome |
|---|---|---|---|
| `DISABLED` | No name or path | None | `DISABLED`; portable remains available. |
| `AUTOMATIC` | No name or path | The exact platform table below | First complete load wins; exhaustion yields `UNAVAILABLE`. |
| `EXACT_NAME` | One nonblank name | That exact unchanged name only | `UNAVAILABLE`; do not try automatic candidates. |
| `EXACT_ABSOLUTE_PATH` | One absolute path | That exact unchanged path only | `UNAVAILABLE`; do not try automatic candidates. |

The request snapshots its value. A null mode or null required exact value raises
`NullPointerException`; a blank exact name, relative exact path, or name/path component that
disagrees with the mode raises `IllegalArgumentException`. It reads no environment variable,
configuration file, command output, package-manager database, classpath service, or global
registry. Future public Config may express equivalent intent, but this task adds no Config type or
public CPU builder.

Explicit override has precedence because it is a distinct request mode, not the first entry in an
automatic list. A failed explicit override falls back to portable execution rather than silently
substituting another installed binary. The caller may make a later separate automatic request.

### Bounded automatic candidate table

Candidate identity distinguishes exact loader names from exact absolute paths. Candidate
generation uses only the snapshotted operating-system family and normalized architecture token;
the fixed tables contain no duplicate typed selection:

| Platform snapshot | Ordered automatic candidates |
|---|---|
| macOS, `aarch64` or `arm64` | name `openblas`; name `libopenblas.dylib`; path `/opt/homebrew/opt/openblas/lib/libopenblas.dylib`; path `/usr/local/opt/openblas/lib/libopenblas.dylib` |
| Other macOS architecture | name `openblas`; name `libopenblas.dylib`; path `/usr/local/opt/openblas/lib/libopenblas.dylib`; path `/opt/homebrew/opt/openblas/lib/libopenblas.dylib` |
| Linux | name `openblas`; name `libopenblas.so.0`; name `libopenblas.so` |
| Windows | name `openblas`; name `libopenblas.dll`; name `openblas.dll` |
| Other or unrecognized | name `openblas` |

This table is intentionally small and reviewable. Absolute candidates are stable non-versioned
Homebrew `opt` locations evidenced by the current macOS checkpoint; discovery passes them directly
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
The normalized tokens remain diagnostic metadata, not binary compatibility or qualification
evidence. Failure to read a property propagates rather than becoming OpenBLAS unavailability.

### Attempts and immutable result metadata

Every attempt records:

- its zero-based position;
- the exact typed name/path selection passed to the provider;
- `LOADED` or `LOAD_OR_BIND_FAILURE`; and
- for failure only, immutable diagnostic strings for the thrown type and nullable message.

The nested checked loader failure carries those two diagnostic strings. The production adapter
copies them from the caught `OpenBlasLoadException`; it does not replace them with the nested
failure's own type or retain the provider exception after the discovery call.

The result defensively snapshots the ordered attempt list and validates all status relationships.
It does not retain a `Throwable`, provider, lookup, native address, file contents, mutable list, or
live platform query. A loaded result has exactly one successful final attempt and names that same
selection. A disabled result has no attempts or selection. An unavailable result has one or more
failed attempts, except that an automatic platform table is never empty under this contract.

The native-free loader seam reports an ordinary attempt failure through one nested checked
failure type. Its production adapter converts only `OpenBlasLoadException` into that type. Because
the provider preserves its original cause after its own broad failure translation, the adapter
first walks that cause chain and rethrows any `VirtualMachineError` or `ThreadDeath`; those fatal
failures never become portable unavailability. Request-validation defects and unrelated CPU
programming failures also propagate rather than becoming failed attempts. If production
adaptation fails after a provider opened, discovery closes that handle before propagating the
adaptation failure, suppressing a distinct cleanup failure on the primary failure.

### Session ownership and route boundary

The internal discovery session is the composition-lifetime owner, not the immutable result. It
has these rules:

- one production loaded session strongly owns exactly one `OpenBlasLibrary` and the existing
  borrowed `CpuOpenBlasInvocation` adapter; the native-free test seam supplies an equivalent fake
  invocation plus one observable close action;
- disabled and unavailable sessions own neither;
- close atomically claims the session once, propagates any unchecked close failure without a
  retry on a later close, and never restores a thread count because this task never changes one;
- composition must keep a loaded session open until every prepared executable borrowing its
  invocation is quiescent, and must not race close with provider use; and
- result metadata remains readable after session close, while the borrowed invocation observes
  the provider's existing closed-state behavior.

A `LOADED` result means only that the exact provider selection loaded and bound the current four
required symbols. It proves neither 32-bit-`blasint` behavior nor numerical compatibility,
thread-state isolation, route eligibility, or profitability. Until CPU 0010D automates installed-
binary qualification, current composition may pair the session with a caller-supplied
`OpenBlasRouteConfig.QUALIFIED` snapshot only when the caller's existing checkpoint evidence
actually covers the selected binary. Otherwise it supplies `OpenBlasRouteConfig.DISABLED` and
portable remains selected. This task does not weaken `OpenBlasRouteConfig.complete()` or infer
qualification from discovery.

Current `SINGLE_THREAD` remains unchanged. Discovery neither calls `threadCount()` nor
`setThreadCount(...)`, creates a lock, installs a worker budget, or coordinates another OpenBLAS
owner. CPU 0010C owns that later thread-state and shared-budget capability.

## Out of scope

- Treating successful loading or four-symbol binding as ABI, numerical, determinism, thread,
  target, or performance qualification.
- Binary content/version fingerprinting, resolved-loader-path recovery, persistent target
  identity, invalidation, optional `openblas_get_config`/`openblas_get_corename`, or CPU 0010D.
- Any OpenBLAS provider public/source/test change. The provider remains an exact name/path loader
  and retains its current four required symbols, 32-bit integer descriptors, GEMM, thread-control,
  lifecycle, and failure contracts.
- Changing current OpenBLAS route eligibility, representation forms, cost equations, native
  invocation, prepared execution, no-late-fallback behavior, or portable MATMUL semantics.
- Transpose, two-input or output materialization, batch, broadcast, epilogue, packing, or CPU
  0010B.
- Multithreaded OpenBLAS, shared CPU permits, global-state serialization/restoration, concurrent
  GEMM policy, tuning candidates, or CPU 0010C.
- Model autotuning, workload signatures, persistent tuning evidence, measurement, cache lookup or
  mutation, benchmarking, or CPU 0010E/0016.
- Downloading, installing, updating, bundling, extracting, or copying an OpenBLAS binary; mutating
  the filesystem, loader configuration, environment, system properties, or package-manager state.
- Arbitrary filesystem search, directory enumeration, wildcard matching, subprocess discovery,
  background refresh, watcher, retry loop, time-based cache, singleton, service locator,
  `ServiceLoader`, reflection, annotation scanning, or hidden global registry.
- Public Config, Engine, Tensor, Compiler, Training, Prepare, Runtime, backend-contract, or Trace
  API changes; another `BackendId`; public backend registration; or end-to-end application use.
- Discovery or provider selection during backend analysis, finalization, cold Runtime binding,
  execution, or any hot path. The session must already exist before those phases consume it.
- Architecture, ADR, module dependency, Gradle dependency, shared-build, backend-conformance, or
  integration-test changes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially explicit composition, CPU
  routes, OpenBLAS provider ownership, staged preparation, runtime prohibitions, and performance
  evidence
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0006: No runtime service locator](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [ADR 0008: Performance evidence and tuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [ADR 0010: Staged preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run resources](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU task 0010](0010-narrow-openblas-blas-compatible-native-route.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading/binding](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)
- [Provider 0002 GEMM invocation](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)
- [Provider 0003 thread control/checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)
- [Config master plan](../../../modules/config/master-plan.md)
- [Engine master plan](../../../modules/engine/master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Benchmark master plan](../../../tools/benchmarks/master-plan.md)

## Architecture constraints

- Engine remains the future outer composition root. This task provides an unsupported CPU-
  internal seam only and creates no supported public composition API or reverse CPU-to-Engine
  dependency.
- CPU owns OpenBLAS discovery policy because the provider must remain a low-level leaf. Every
  actual load still delegates one exact candidate unchanged to the provider.
- Discovery is explicit and cold. Backend analysis receives only already-resolved immutable facts;
  finalization receives only an already-borrowed invocation; Runtime receives only prepared work.
- A portable plan is always established independently for every currently supported occurrence.
  Discovery absence never removes portable support, and a selected native plan still never falls
  back after analysis.
- `LOADED` and `QUALIFIED` remain separate facts. Only CPU-owned qualification can authorize a
  provider for route selection.
- The internal request, result, attempts, selections, and platform snapshot are immutable typed
  values. Do not use `Map<String,Object>`, string dispatch, a registry, or reflection.
- The live session is explicitly caller-owned and separate from immutable discovery metadata. It
  does not become a hidden process-global cache or prepared/runtime resource.
- No architecture rule, module edge, or provider API changes. If implementation needs resolved
  operating-system library paths, optional provider metadata symbols, a public Config type, Engine
  lifecycle, or shared global coordination to satisfy this task, stop and report the exact gap.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.provider.openblas`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Packages added or changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Type placement:

- `CpuOpenBlasDiscoveryRequest` — immutable disabled/automatic/exact override intent beside the
  route whose provider it selects.
- `CpuOpenBlasDiscoveryResult` — immutable typed platform, attempt, selection, and outcome facts;
  it owns no live resource.
- `CpuOpenBlasDiscovery` — stateless cold candidate generation and exact-loader delegation, with a
  package-private native-free test seam whose nested checked failure distinguishes expected
  load/bind absence from programming and fatal failures.
- `CpuOpenBlasDiscoverySession` — explicit CPU-internal composition owner for the selected provider
  lifetime and existing borrowed invocation adapter; its nested owned-resource form also permits
  a close-observable native-free fake in ordinary tests.

All four types remain package-private. `CpuCapabilityProvider` remains the sole supported public
CPU API. A future CPU composition task may deliberately wrap, move, replace, or change the
visibility of this seam behind a supported CPU-owned facade when Config and Engine exist. That
cross-package boundary is future work; this task does not make these types temporarily accessible
to another CPU package or module.

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

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0010a-automatic-openblas-discovery-and-internal-composition-foundation.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the 13 exact paths above. No provider, Config, Engine,
Prepare, Runtime, backend-conformance, integration, architecture, ADR, Gradle, settings, or other
source/test/documentation path may change. If another path is required, stop and propose a
follow-up or architecture clarification before editing it.

## Acceptance criteria

1. The four package-private CPU-owned types implement the exact request/result/discovery/session
   split, and every immutable value defensively snapshots its members and validates all status
   relationships.
2. Disabled mode performs no loader or platform lookup and returns a resource-free `DISABLED`
   session. Exact modes also perform no platform lookup. Only automatic mode snapshots production
   platform properties once.
3. Exact-name and exact-path modes delegate only the exact supplied value once. They never append
   automatic candidates; load/bind failure returns a resource-free `UNAVAILABLE` result.
4. Automatic platform classification and candidate ordering are exactly the rules and table above
   for macOS arm64, other macOS, Linux, Windows, and unrecognized platforms. Tests prove exact
   normalization, absence handling, and first-success stop.
5. The production adapter calls only `OpenBlasLibrary.open(String)` or
   `OpenBlasLibrary.open(Path)`. No provider code or public API changes, and no alternate native
   loading path exists in CPU.
6. Failed attempts retain only immutable typed selection plus failure type/message strings. A
   loaded result identifies the final successful attempt exactly and cannot be constructed with
   inconsistent attempt metadata. The production adapter converts `OpenBlasLoadException` only;
   it rethrows `VirtualMachineError` or `ThreadDeath` found in that exception's cause chain, and
   unrelated failures propagate.
7. A production loaded session owns exactly one provider and existing invocation adapter; the
   native-free seam proves the same one-close ownership with a fake. Close is idempotent, exposes
   the provider's ordinary closed behavior to the borrowed invocation, and performs no thread
   query, setter, restoration, or filesystem mutation.
8. Disabled/unavailable composition maps to no invocation and cannot accidentally make
   `OpenBlasRouteConfig.complete()` true. Loaded composition also does not synthesize
   `QUALIFIED`, cost, storage, or thread facts.
9. The existing portable default, current `SINGLE_THREAD` route, exact 0010 eligibility/cost
   equations, direct/one-copy execution, provider contract, and no-late-fallback behavior remain
   unchanged under focused regression tests.
10. The opt-in CPU native checkpoint accepts `--auto`, reports the exact selected name/path,
    closes the discovery session, and reopens that exact selection for the existing numerical and
    thread-restoration assertions without requiring a version-specific Cellar path. Its existing
    single absolute-path argument remains exact and exclusive.
11. Source and bytecode/API-shape scans confirm no new supported public CPU type, static mutable
    discovery state, singleton, registry, `ServiceLoader`, filesystem enumeration, process launch,
    network/download/install code, provider mutation, or discovery reference in Runtime/hot
    execution.
12. `CpuInternalPackageInventoryTest` records exactly the four new production types. The new
    focused test remains in the ordinary test inventory; unrelated production inventory and
    generated-kernel schemas remain unchanged.
13. A distinct clean documentation-focused agent reviews the final code/test diff, finalizes
    affected Javadoc, the OpenBLAS package summary, CPU guide, glossary impact, task/master plan,
    and roadmap under the required profiles, and records reasoned no-change conclusions for
    provider docs/API, public Config/Engine/Tensor/Compile/Training APIs, architecture/ADRs, shared
    Prepare/Runtime, backend conformance, integration, other modules, and generated artifacts.
14. Focused CPU tests, CPU Javadoc, the opt-in automatic native checkpoint, documentation checks,
    exact-path scope, and `git diff --check` pass. Repository-wide validation remains deferred to
    the 0010A–0010E capability checkpoint or CI because this task changes one module and no shared
    contract or dependency.

## Tests / validation

During implementation, run focused new tests as needed, then record one final affected-module run
after executable code stabilizes:

```bash
./gradlew :backends:cpu:test
```

All ordinary tests use the package-private fake loader and current fake invocation seam. They need
no installed OpenBLAS library, native-access permission, system-property mutation, or platform-
specific host.

On a host with a compatible installed OpenBLAS binary, compile and run the existing isolated
checkpoint in automatic mode:

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  --auto
```

The command must print the exact automatic selection and the existing pass/restoration result. If
automatic discovery is unavailable on the implementation host, the task remains `Incomplete`
until the cause is recorded and either the fixed candidate table is corrected within scope or the
required environment becomes available. The exact-path checkpoint remains available for diagnosis
but does not substitute for the automatic-mode acceptance evidence.

The documentation-focused pass reuses the final Java/native evidence unless it changes executable
Java or tests, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It also renders and inspects the affected Javadoc, validates local Markdown links and anchors,
heading order/uniqueness, balanced fences, LF/final newlines, terminology, the exact 13-path
ceiling, internal/public inventory, final `0010A Complete` plus `0010B`–`0010E Draft` ordering,
unchanged
`0011 Blocked`, and absence of any detailed 0010B–0010E task file.

Repository-wide validation is deferred to the 0010A–0010E capability checkpoint or CI unless the
implementation changes a repository-wide contract. Backend-conformance and integration remain
unchanged because no Engine/public lifecycle or executable route semantics are added.

## Dependencies

- CPU 0010 is `Complete` and supplies the route, provider-free invocation adapter, explicit
  qualification/config facts, finalization, and native checkpoint.
- The OpenBLAS provider milestone is `Complete` and supplies exact name/path loading, the required
  four-symbol binding, GEMM, thread control, and caller-owned lifecycle without discovery.
- Current CPU package inventory and documentation contracts are stable.
- No Config, Engine, Prepare, Runtime, tuning, benchmark, or new provider prerequisite is required
  for this internal foundation.

## Follow-up tasks

- CPU 0010B broadens exact OpenBLAS MATMUL representations through bounded two-input and output
  materialization while retaining the current non-transposed provider call; batching,
  broadcasting, and epilogues remain later evidence-driven work.
- CPU 0010C adds explicit composition-owned OpenBLAS thread-state coordination and one CPU thread
  budget. It may consume the discovery session but must not turn it into a singleton or provider
  policy owner.
- CPU 0010D separates installed-binary qualification and stable target fingerprinting from this
  loading result. It is the first task allowed to turn discovered binary evidence into an
  automatically compatible qualification fact.
- CPU 0010E adds typed OpenBLAS candidates and compatible tuning-evidence consumption with
  measurement/cache mutation still owned by `tools/tuning`.
- Future Config work defines supported declarative intent; a future CPU composition-API task wraps
  or replaces this package-private seam behind a supported CPU-owned boundary; and Engine 0001
  owns outer application lifecycle, explicit registration, and user-visible portable fallback.
  Engine must not import the internal package or expose these temporary types.
- CPU 0011 remains `Blocked` independently on a concrete Intel use case and supported oneMKL ABI
  evidence. Completing 0010A does not satisfy or bypass those gates.

## Architecture impact

Expected impact: None. The task implements the already-authorized rule that CPU owns route and
provider-composition policy while the provider remains an exact loader and Engine remains the
future outer composition root. It adds no module edge, supported public API, service locator,
global registry, Runtime behavior, or architecture rule.

If implementation requires a provider API change, public Config or Engine type, resolved native-
library path API, cross-module resource lifecycle, global coordinator, or discovery after the
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

- The task is `Ready` before Engine because its request, result, loader, and lifetime seam are
  package-private CPU implementation details with native-free tests. Public request translation
  and application lifecycle remain deferred rather than approximated.
- Automatic discovery is bounded candidate probing, not arbitrary filesystem search. The only
  absolute automatic candidates are the two stable non-versioned Homebrew `opt` paths evidenced
  by the current macOS installation model.
- Exact override is exclusive. Failure never silently changes the requested library; portable is
  the fallback.
- Immutable discovery metadata and live provider ownership are separate types. This avoids
  describing a closeable provider handle as an immutable result.
- Loading and qualification remain separate. Existing explicit checkpoint-qualified facts can be
  paired manually during the transition, while CPU 0010D owns automatic qualification and target
  fingerprinting.

## Known limitations

- Automatic discovery covers only the fixed names and two stable macOS paths above. It does not
  find versioned files or nonstandard installations outside the operating-system loader search
  configuration; exact override remains available.
- The JDK name-based loader does not expose the resolved binary path. This task therefore cannot
  create a stable persistent binary fingerprint from a successful name load. CPU 0010D must
  either establish an exact identity mechanism or classify such results as session-only for
  persistent compatibility.
- Successful discovery proves only loading and required-symbol binding. Current native routing
  still needs explicitly associated qualification, single-thread, expected-storage, and cost
  facts.
- No public Config or Engine lifecycle exists. The foundation is intentionally usable only by
  CPU-local composition/tests until those owners are implemented.
- The explicit session coordinates no other handle, class loader, engine, or arbitrary native
  consumer and makes no thread-state isolation guarantee.

## Validation evidence

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
  prepared routes, and reported `CPU OpenBLAS native checkpoint passed; restored thread count 16`.
- The implementation context's Class-File inspection confirmed that all four new top-level types
  are package-private and `CpuOpenBlasDiscovery` is field-free. Its `git diff --check` passed.
- Documentation context `01a08fd1-151c-7111-b094-a1140cf7ed04` applied the General,
  API/Javadoc, Backend Guide, Planning, and Example profiles, inspected all eight executable paths
  and the complete diff, and changed only Javadoc plus the five authorized documentation/planning
  paths. No executable statement or test behavior changed.
- The documentation context ran `./gradlew :backends:cpu:javadoc`: passed with two incubating-
  module warnings and 92 unrelated pre-existing missing-`@param` warnings in
  `CpuPartitionLowering.LoweredPartition` and `CpuPartitionPreparationPlan` records. Generated
  package and discovery/request/result/session pages were inspected for request precedence,
  immutable diagnostics, failure contracts, route non-enablement, ownership, idempotent close,
  post-close metadata, borrowed invocation, and thread/lifecycle boundaries.
- The documentation context ran
  `python3 /tmp/validate_synaptik_markdown.py docs/backend-guide/cpu-backend.md docs/glossary.md docs/planning/backends/cpu/tasks/0010a-automatic-openblas-discovery-and-internal-composition-foundation.md docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md`:
  passed with `validated 5 Markdown files`, covering local links and anchors, unique headings,
  balanced fences, LF/final newlines, and trailing whitespace.
- Final manual checks confirmed the exact 13 authorized worktree paths; four new package-private
  production types; immutable automatic attempt metadata; exclusive exact overrides; no provider,
  route, configuration, Engine, Prepare, Runtime, generated-artifact, architecture, dependency, or
  Gradle change; synchronized 0010A `Complete`; 0010B–0010E `Draft`; 0011 `Blocked`; and no
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
  settings, native packaging, and environment configuration remain unchanged.

## Implementation notes

- Added immutable disabled/automatic/exact-name/exact-path request and result contracts, bounded
  platform classification and candidate ordering, immutable attempt diagnostics, and exact loader
  delegation. No deviation from the specified table or precedence was required.
- Added a separate caller-owned session that retains one loaded provider-equivalent resource,
  exposes the borrowed invocation without qualifying a route, and atomically claims close once.
  Disabled and unavailable sessions remain resource-free.
- Extended the isolated CPU checkpoint with `--auto`; it closes discovery ownership and reopens
  only the selected exact name/path before the existing qualification-style route checks and
  caller-coordinated thread restoration.
- Finalized implementation-contract Javadocs, the OpenBLAS package summary, CPU guide, glossary
  terminology, task evidence, master-plan frontier, and roadmap status without changing executable
  behavior.

## Completion summary

- Completed changes: implemented the exact bounded CPU-private OpenBLAS discovery request, result,
  operation, and lifetime session; extended the native checkpoint; and finalized Javadocs,
  explanatory documentation, glossary terminology, and planning status.
- Files changed or created: exactly the 13 authorized paths—four new discovery production types,
  OpenBLAS package documentation, discovery tests, native checkpoint, CPU package inventory, CPU
  backend guide, glossary, this task, CPU master plan, and roadmap.
- Tests and validation performed: reused the implementation context's passing 189-suite/958-test
  CPU run, passing test-class compilation, automatic native checkpoint, Class-File visibility/
  field scan, and whitespace result; the documentation context passed CPU Javadoc, generated-page
  inspection, five-file Markdown validation, exact-scope/status/frontier/manual checks, and final
  `git diff --check`.
- Documentation-agent review: clean documentation context
  `01a08fd1-151c-7111-b094-a1140cf7ed04` independently reviewed the final code/tests and finalized
  all affected Javadocs and documentation. It made no executable behavior change and did not
  duplicate the stable Java suite or native checkpoint.
- Documentation impact: the CPU guide and glossary now distinguish bounded loading evidence,
  immutable metadata, session ownership, later qualification, route selection, and public
  composition. Task, master plan, and roadmap consistently close 0010A without advancing 0010B.
- Javadoc review: every new production type and explicit constructor/method documents purpose,
  inputs, results, failures, immutability, ownership, lifecycle, concurrency, and thread-state
  boundaries as applicable; the changed checkpoint entry points document their exact inputs and
  failures.
- Glossary impact: added the reusable OpenBLAS discovery-result/session distinction and clarified
  that CPU discovery delegates to the unchanged exact-loading provider.
- Unresolved issues: None.
- Follow-up required: None for 0010A. CPU 0010B is the next planning frontier and remains `Draft`;
  CPU 0010C–0010E remain `Draft`, and CPU 0011 remains `Blocked`.

Status: Complete
