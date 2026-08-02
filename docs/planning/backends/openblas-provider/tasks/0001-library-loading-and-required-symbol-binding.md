# Task 0001: Library Loading and Required Symbol Binding

## Status

Complete

## Goal

Replace the OpenBLAS provider placeholder with the smallest usable native-interoperability
foundation: a caller-owned, closeable OpenBLAS library handle that loads one caller-specified
native library and binds the exact FLOAT32/FLOAT64 general matrix multiplication (GEMM) and
thread-control symbols required by the next provider tasks.

The foundation reports loading or binding failure without choosing a CPU route or fallback. It
does not invoke a native function yet.

## Rationale and mental model

```text
caller-selected library name or absolute path
  -> JDK 26 Foreign Function and Memory library lookup
  -> resolve and bind the complete required symbol set
  -> caller-owned OpenBlasLibrary
     -> later GEMM and thread-control provider APIs
  -> close the lookup lifetime
```

The current module contains only a marker. Loading without binding would not prove that the
library can support the first planned provider capabilities, while adding GEMM calls or thread
mutation would combine separately testable native contracts. Loading and complete required-symbol
binding are therefore one atomic foundation. A missing library or any missing required symbol
fails the whole open operation and releases its partial lookup lifetime.

## Scope

- Remove the placeholder `OpenBlasProviderModule` marker.
- Add one public final `OpenBlasLibrary` lifetime owner with explicit name-based and absolute-
  path-based factories.
- Add one public final `OpenBlasLoadException` so the CPU backend can distinguish provider loading
  failure while retaining ownership of any later fallback decision.
- Use the JDK 26 Foreign Function and Memory (FFM) API to create one shared arena, load the exact
  caller-specified library, resolve the complete required symbol set, and create downcall method
  handles with fixed descriptors.
- Require exactly `cblas_sgemm`, `cblas_dgemm`, `openblas_set_num_threads`, and
  `openblas_get_num_threads` for a successful open.
- Retain the arena and bound handles privately for later package-colocated GEMM and thread-control
  APIs; expose no `MethodHandle`, `MemorySegment`, `SymbolLookup`, `Linker`, or `Arena` publicly.
- Add one narrow package-private native-access seam used only to test loading, symbol-failure,
  cleanup, and lifetime behavior without requiring an installed OpenBLAS library.
- Add focused tests for exact public shape, input/failure/lifetime behavior, exact symbols and
  descriptors, partial-failure cleanup, and concurrency boundaries.
- Finalize affected Javadocs, the CPU backend guide, glossary impact, and planning records through
  the required separate clean documentation pass.

## Out of scope

- invoking GEMM or proving GEMM numerical results
- setting or querying the OpenBLAS thread count
- FLOAT16, BFLOAT16, complex, batched, transposed, column-major, vector, or non-GEMM routines
- optional symbols, symbol aliases, Fortran BLAS entry points, or fallback to another BLAS
  implementation
- selecting a default library, probing platform-specific candidate filenames, scanning library
  paths, or reading a system property, environment variable, config object, registry, or service
  locator
- deciding whether OpenBLAS is available, preferred, profitable, or safe for a workload
- CPU route selection, fallback policy, thread-count selection, workload classification,
  calibration, benchmarking, model autotuning, caches, or candidate generation
- Tensor, Model, Compiler, Planning, Prepare, Runtime, Engine, Trace, Backend Contract, Config, or
  concrete CPU behavior
- prepared execution, buffer ownership, residency, allocation, transfer, materialization, or
  backend orchestration
- bundling, downloading, installing, extracting, or redistributing an OpenBLAS binary
- a global singleton, static eager load, registry, manager, provider discovery mechanism, or
  service locator
- a Gradle dependency or native packaging change, root build change, architecture-contract
  change, ADR, or architecture-test change
- detailed task specifications for task 0002 or later work

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - OpenBLAS provider
  - CPU backend routes
  - Dependency rules
  - Performance evidence and optimization tooling
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [ADR 0008: Separate performance evidence and model-autotuning concerns](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)

## Architecture constraints

- `backends/openblas-provider` remains a low-level JDK/native-interoperability leaf used only in
  the dependency direction `backends/cpu -> backends/openblas-provider`.
- The provider owns only loading, symbol binding, later GEMM calls, and later low-level thread
  control. It does not interpret configuration or decide ownership, route, fallback, or tuning
  policy.
- The public surface contains no Tensor, graph, operation, backend identity, compile, planning,
  prepare, runtime, engine, residency, buffer, workspace, or configuration type.
- The module keeps zero project dependencies and uses only JDK APIs.
- Loading is explicit and cold. Class initialization performs no native lookup, and ordinary unit
  tests do not require OpenBLAS to be installed.
- Native handles and their arena lifetime remain encapsulated. No raw native address or FFM handle
  becomes a cross-module contract.
- `legacy/pre-rewrite` is capability evidence only. Do not copy its source structure, singleton
  loading, property/environment policy, optional-symbol policy, fallback behavior, or runtime
  coupling into the new provider.
- If implementation requires a new dependency direction, platform discovery policy, global
  lifecycle, or another public abstraction, stop and request a planning or architecture decision.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.provider.openblas` — replaces its marker with the deliberate
  public library-lifetime surface and package-private FFM binding implementation.

No package is added. The module root package remains appropriate because this first surface is the
provider's small public entry point and its package-private native binding state. Do not add
`service`, `manager`, `registry`, `util`, `config`, `runtime`, or `internal` subpackages.

## Exact public and package-private API intent

The public source surface is exactly:

```java
package io.github.pho001.synaptik.backend.provider.openblas;

import java.nio.file.Path;

public final class OpenBlasLibrary implements AutoCloseable {
    public static OpenBlasLibrary open(String libraryName);

    public static OpenBlasLibrary open(Path absoluteLibraryPath);

    public boolean isOpen();

    @Override
    public void close();
}

public final class OpenBlasLoadException extends IllegalStateException {
    // Package-private construction only; callers catch the public failure type.
}
```

`OpenBlasLibrary` has no public constructor, subclassing point, default/singleton accessor,
availability query, library-discovery method, operation method, thread method, or exposed native
handle. `OpenBlasLoadException` has no public constructor or additional public member beyond those
inherited from `IllegalStateException`.

Package-private implementation intent:

- `OpenBlasNativeAccess` is the single narrow test seam, with exactly
  `OpenBlasNativeBindings open(String libraryName)` and
  `OpenBlasNativeBindings open(Path absoluteLibraryPath)`. It accepts the already validated
  explicit name or absolute path and returns one complete binding set, or throws. It contains no
  config, fallback, candidate, platform, route, or cache vocabulary.
- `FfmOpenBlasNativeAccess` is the sole production implementation. It owns `Arena.ofShared()`,
  `SymbolLookup.libraryLookup(...)`, `Linker.nativeLinker()`, exact symbol lookup, descriptor
  binding, and partial-failure closure. It is final, has no mutable global state, and exposes only
  the two package-private interface methods.
- `OpenBlasNativeBindings` is a package-private final immutable carrier retaining exactly the
  shared `Arena` and four `MethodHandle` values in symbol order: SGEMM, DGEMM, set thread count,
  and get thread count. Its constructor validates every reference, its package-private accessors
  retain exact references for later provider classes, and its package-private idempotent close
  operation owns the arena lifecycle.
- `OpenBlasLibrary` has exactly two package-private test overloads,
  `open(String, OpenBlasNativeAccess)` and `open(Path, OpenBlasNativeAccess)`, which apply the same
  public input/failure/lifetime behavior while accepting a deterministic fake. The production
  overloads pass a private immutable `FfmOpenBlasNativeAccess` instance. The seam must not become
  public, reflective, string-dispatched, globally mutable, or reusable as a generic native plugin
  abstraction.
- `OpenBlasLoadException` has exactly one package-private constructor accepting the stable message
  and original cause. Failure-list construction and suppression stay in the package-private
  loading implementation rather than adding public exception state.

Do not add a second public result, availability DTO, optional handle, provider facade, registry,
or abstract base class.

## Native library ownership, lifetime, and failure semantics

- Each successful `open` returns a distinct Java lifetime owner and retains one complete binding
  set. The object owns its FFM arena and library lookup lifetime; it does not own, install, delete,
  or modify the library file.
- The arena is shared so later provider calls may use the immutable bound handles from multiple
  Java threads. No per-call memory or mutable execution state is introduced in this task.
- `close()` is idempotent and safe for multiple sequential or concurrent close attempts. The first
  close ends this object's FFM lookup lifetime; later closes do nothing.
- `isOpen()` is a local lifecycle observation only. It does not probe the operating system,
  re-resolve symbols, report route availability, or promise that a later call cannot fail.
- Use after close is rejected by later operation APIs; task 0001 itself has no native operation.
  Closing must never reopen or replace a binding set.
- A caller must not race `close()` with a later GEMM or thread-control invocation. This task does
  not add reference counting, active-call tracking, synchronization around native calls, or a
  global shutdown protocol.
- If library lookup, linker creation, symbol lookup, or handle binding fails, `open` throws one
  `OpenBlasLoadException` that identifies the caller-supplied name or path and retains the original
  cause. The partial arena is closed before the exception escapes. A cleanup failure is suppressed
  on the primary loading/binding failure.
- Missing symbols are collected in required-list order and reported together. A partially bound
  library is never returned.
- One failed `open` does not poison later opens and is not cached globally.
- The JDK and operating-system loader may share or retain an underlying process library across
  multiple handles. Closing this Java owner ends only its own arena/lookup lifetime and does not
  promise physical unload while another owner or the platform keeps the binary resident.

## Platform and library resolution policy boundary

- `open(String)` passes one nonblank caller-specified operating-system library name to the JDK
  name-based lookup. A caller may explicitly pass a platform name such as `openblas`,
  `libopenblas.so`, `libopenblas.dylib`, or an equivalent installed name; the provider chooses none
  of them.
- `open(Path)` requires a non-null absolute path and passes that exact path to the JDK path-based
  lookup. It does not normalize, canonicalize, search sibling directories, append extensions, or
  derive a platform name.
- Input validation rejects a null or blank name and a null or non-absolute path before invoking
  the package-private native seam.
- CPU/config/Engine composition later owns where an explicit name/path comes from and whether a
  failed open causes a scalar/Vector route or terminal failure. This provider neither reads nor
  interprets those sources.
- There is no automatic macOS/Linux/Windows branch, bundled-native lookup, environment fallback,
  system-property convention, loader lookup, or default lookup in this task.

## Required symbols and ABI assumptions

A successful task-0001 open binds all four symbols:

| Symbol | Required C shape | Task-0001 reason |
|---|---|---|
| `cblas_sgemm` | `void(int,int,int,int,int,int,float,address,int,address,int,float,address,int)` | Required by the next FLOAT32 GEMM task. |
| `cblas_dgemm` | `void(int,int,int,int,int,int,double,address,int,address,int,double,address,int)` | Required by the next FLOAT64 GEMM task. |
| `openblas_set_num_threads` | `void(int)` | Required by the later low-level thread-control task. |
| `openblas_get_num_threads` | `int(void)` | Required to observe/validate that later thread-control operation. |

The implementation uses the native linker's C ABI with `ValueLayout.JAVA_INT`,
`ValueLayout.JAVA_FLOAT`, `ValueLayout.JAVA_DOUBLE`, and `ValueLayout.ADDRESS` in the exact order
above. This selects the ordinary OpenBLAS CBLAS ABI where `blasint` is 32 bits. An OpenBLAS build
whose C interface uses 64-bit integers (`INTERFACE64`/ILP64) is unsupported by this task even if it
exports the same names.

No minimum OpenBLAS release is claimed. Compatibility is structural: the supplied binary must be
loadable on the current JDK native linker and export the four symbols with the specified ABI.
Task 0001 does not call `openblas_get_config`, parse a version/build string, inspect the threading
backend, or validate numerical behavior. Optional BFLOAT16 symbols from legacy evidence are not
required because the current Model supports BFLOAT16 semantics but no current CPU/OpenBLAS route
contract justifies their ABI or output policy.

## Concurrency and thread-safety boundaries

- A successfully opened `OpenBlasLibrary` is an immutable handle owner except for its one-way open
  to closed lifecycle.
- Immutable native bindings and the shared arena are suitable for later concurrent read/use from
  multiple threads while the owner remains open.
- `isOpen()` and idempotent `close()` are thread-safe.
- Different `OpenBlasLibrary` objects have distinct Java lifecycle state. They do not imply
  distinct native OpenBLAS process state.
- OpenBLAS thread count is library/process-global native state rather than per-handle state. The
  later thread-control task must define mutation/coordination semantics; task 0001 only binds the
  symbols and performs no mutation.
- No lock, executor, thread-local, active-call counter, or mutable global cache is introduced.

## Affected files

Production:

- remove `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasProviderModule.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLoadException.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasNativeAccess.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/FfmOpenBlasNativeAccess.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasNativeBindings.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/package-info.java`

Tests:

- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryPublicShapeTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasAbiContractTest.java`

Explanatory documentation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`

Planning:

- this task
- `docs/planning/backends/openblas-provider/master-plan.md`
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `ARCHITECTURE.md`, focused architecture and
performance documents, ADR 0008, CPU strategy note, documentation rules/profiles, planning guide,
OpenBLAS/CPU/Prepare/Runtime/Backend Contract/Config master plans, completed Prepare 0003, root and
module build files, current provider/CPU source inventory, architecture tests, and current JDK 26
and OpenBLAS primary API/header evidence.

## Maximum scope

At most 15 paths:

- 7 provider production paths, counting removal of the marker;
- 3 provider test paths;
- 2 explanatory documentation paths; and
- 3 planning paths.

No Gradle file, root/settings file, authoritative architecture contract, architecture explanation,
ADR, architecture test, backend-conformance test, integration test, CPU source/test, or other
module path may change. If implementation requires another public type, another symbol, native
packaging, a JVM-launch configuration change, or more than 15 paths, stop and propose a bounded
follow-up or planning correction.

## Acceptance criteria

- [x] The marker is removed and the public surface is exactly `OpenBlasLibrary` and
      `OpenBlasLoadException` with the members specified above.
- [x] No class initialization or ordinary construction path performs eager global native loading.
- [x] Name and path factories validate inputs before native access and apply no implicit lookup
      policy.
- [x] A successful open binds exactly the four required symbols with exact descriptors and retains
      one shared arena for the complete handle lifetime.
- [x] Missing library, restricted native access, linker failure, missing symbols, and binding
      failure become `OpenBlasLoadException`; partial lifetime cleanup preserves the primary cause
      and suppresses cleanup failure.
- [x] Missing required symbols are reported together in required-list order; no partial handle is
      returned.
- [x] `isOpen()` and idempotent concurrent `close()` satisfy the specified local lifecycle without
      probing, reloading, or global caching.
- [x] The package-private seam is used only for deterministic native-boundary tests and does not
      become a generic abstraction or public API.
- [x] Ordinary unit tests pass without an installed OpenBLAS and without silently skipping their
      loading, failure, ABI-shape, cleanup, lifecycle, or concurrency assertions.
- [x] Tests lock the standard 32-bit-`blasint` symbol descriptors and the absence of optional BF16,
      version, platform-discovery, config, fallback, route, tuning, and invocation behavior.
- [x] Production imports remain limited to `java.base` APIs and the module retains zero project
      dependencies.
- [x] No Tensor or Model, Compiler, Planning, Prepare, Runtime, Engine, Trace, Backend Contract,
      Config, CPU, registry, manager, or service-locator type enters the provider surface or
      implementation.
- [x] Every production type/method/constructor has complete Javadoc for ownership, lifetime,
      concurrency, parameters, return value, and failures where applicable.
- [x] A separate clean documentation-focused agent finalizes Javadocs, CPU guide status/boundary,
      glossary impact, and planning evidence in the same overall change.
- [x] Task 0001 remains the only detailed unfinished OpenBLAS provider task until it is Complete.
- [x] All focused, documentation, exact-scope, status, link/anchor/fence/whitespace, and diff checks
      pass before the task is marked Complete.

## Tests / validation

Implementation-focused tests must cover:

- exact public and package-private class/member visibility;
- exact four-symbol ordered inventory and FFM descriptor shapes;
- null/blank name and null/relative path rejection before the seam is called;
- successful fake loading by name and path with exact reference retention;
- library-load, missing-symbol, linker/binding, and partial-cleanup failure translation;
- stable missing-symbol ordering and no returned partial binding;
- no eager or cached global load and a fresh independent result for each open;
- open/closed observation and repeated/concurrent close behavior;
- shared-lifetime intent without a close-versus-call guarantee; and
- source/import/build checks for the leaf boundary and excluded mechanisms.

Run focused tests while implementation stabilizes. After executable Java stabilizes, run exactly
one final affected-module command:

```bash
./gradlew :backends:openblas-provider:test
```

The clean documentation pass reuses that successful test evidence unless it changes executable
Java or records a concrete stale-evidence risk. It runs:

```bash
./gradlew :backends:openblas-provider:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/backend-guide/cpu-backend.md \
  docs/glossary.md \
  docs/planning/backends/openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md \
  docs/planning/backends/openblas-provider/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

It must also inspect generated Javadocs and verify exact public/package-private surface, symbol
order/descriptors, package placement, zero project dependencies, exact 15-path maximum, task/
master/roadmap status synchronization, preserved Prepare history, absence of a detailed OpenBLAS
0002 task, relative links, anchors, fences, final newlines, and trailing whitespace.

Repository-wide tests and native numerical validation are deferred to the OpenBLAS provider
capability checkpoint after the GEMM and thread-control tasks, or to continuous integration. No
installed-library integration test is part of ordinary task validation; a later native checkpoint
must use an explicitly supplied compatible library and explicitly enabled JDK native access.

## Dependencies

- Prepare 0003 and the shared Prepare milestone — Complete
- Runtime milestone through 0014 — Complete
- Compiler and Planning milestones — Complete
- Backend Contract selected milestone — Complete
- JDK 26 toolchain and stable `java.lang.foreign` API — current root build baseline
- OpenBLAS C header contract for the four required symbols — external ABI evidence, not a Java or
  project dependency

No unfinished repository task blocks implementation. OpenBLAS need not be installed for ordinary
unit validation.

## Follow-up tasks

- OpenBLAS provider 0002 (Draft row only): add validated row-major FLOAT32/FLOAT64 GEMM invocation
  over caller-owned `MemorySegment` storage using this exact library lifetime and bindings.
- OpenBLAS provider 0003 (Draft row only): add low-level thread-count query/control and the native
  provider capability checkpoint, including explicit installed-library validation.
- CPU 0001 and later CPU route work remain Draft and consume the completed provider without moving
  route or fallback policy into it.

Do not create detailed specifications for these follow-ups in this task.

## Architecture impact

Expected impact: None.

This task implements the loading and binding responsibilities already assigned to the low-level
OpenBLAS provider and adds no module edge or architecture rule. If the implementation needs the
provider to interpret config, choose a platform candidate, own fallback, expose native handles,
or depend on CPU or a shared lifecycle module, stop and report the conflict rather than editing
architecture.

## Documentation impact

The separate documentation pass must:

- finalize Javadoc for every new production declaration, including ownership, close behavior,
  concurrency, ABI boundary, input constraints, results, and failures;
- update the CPU backend guide from placeholder-only status to the exact implemented loading/
  binding foundation while keeping GEMM invocation, thread control, CPU routes, and Engine work
  clearly planned;
- add or update a glossary entry only if the new public library-handle/lifetime distinction is a
  reusable project term; otherwise record a reasoned no-change conclusion;
- preserve architecture documents because module ownership and dependency rules do not change;
  and
- record reasoned no-change conclusions for Tensor/Compile/Training APIs, Config, Prepare,
  Runtime, CPU implementation, architecture tests, backend conformance, integration, Gradle, and
  native packaging.

## Implementation prompt

```text
You are a clean-context implementation agent working in the Synaptik repository. Do not commit or
push.

Read in full: AGENTS.md; ARCHITECTURE.md; docs/planning/planning-guide.md;
docs/developer-guide/documentation-rules.md; the General, API/Javadoc, Backend Guide, and Planning
documentation profiles; docs/planning/backends/openblas-provider/master-plan.md; and
docs/planning/backends/openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md.
Read the directly referenced architecture/performance documents, CPU strategy and master plan,
completed Prepare 0003 boundary, current provider source/test/build inventory, root Java 26 build
conventions, relevant architecture tests, and current JDK 26/OpenBLAS primary ABI references named
by the task. Treat `legacy/pre-rewrite` as read-only capability evidence only; do not copy or adapt
its architecture, source structure, loading policy, singleton, fallback behavior, or coupling.

Implement task 0001 exactly within its fifteen authorized paths. Add only explicit caller-directed
name/path loading, fail-closed binding of the exact four required symbols, caller-owned closeable
lifetime, the typed load failure, and the narrowly justified package-private native test seam.
Preserve the provider's zero-project-dependency leaf boundary. Add no native invocation, default or
platform lookup policy, config/fallback/route/tuning behavior, singleton/cache/registry/manager,
CPU or shared-lifecycle code, Gradle/native packaging change, architecture change, or later task.
Stop on an architecture conflict, ABI uncertainty beyond the recorded 32-bit-blasint boundary,
scope overflow, or need for another public abstraction.

Run focused tests while developing, then exactly one final
./gradlew :backends:openblas-provider:test after executable Java stabilizes. Hand the actual diff
and exact test evidence to a separate clean-context documentation-focused agent in the same
overall change. That agent must inspect final source/tests and independently finalize Javadocs,
the CPU backend guide, glossary impact, task/master/roadmap evidence and status, and all specified
documentation/scope checks without repeating successful Java tests unless executable behavior
changes or a concrete stale-evidence risk is recorded. Mark task 0001 Complete only after every
criterion passes. Do not create a detailed task 0002 specification.
```

## Local decisions

- Loading and complete required-symbol binding are one task because either part alone cannot
  produce a valid foundation for the immediately planned GEMM and thread-control APIs.
- The provider accepts only an explicit caller-supplied name or absolute path. This preserves the
  provider's loading responsibility without importing configuration or fallback policy.
- FLOAT32/FLOAT64 GEMM and get/set thread-count symbols are required together. Optional BFLOAT16
  and version/build-query symbols are excluded until a current route or compatibility consumer
  justifies them.
- The public handle owns lifetime but exposes no native mechanism. Later package-colocated
  provider operations consume its package-private immutable bindings.
- The package-private native seam is justified solely because ordinary unit tests must prove
  failure, cleanup, and lifecycle behavior without assuming OpenBLAS is installed.
- Standard 32-bit `blasint` is the only selected ABI. No heuristic version or ILP64 detection is
  invented.

## Known limitations

- No native function is invoked, so this task does not prove numerical correctness or that the
  supplied binary's symbols actually obey their advertised ABI.
- ILP64/`INTERFACE64`, BFLOAT16, complex, batched, and non-GEMM builds/routes are unsupported.
- The provider does not discover a library or supply a default; later composition must provide an
  explicit name or absolute path.
- Real loading requires the deployment JVM to permit restricted native access. Task 0001 does not
  change Gradle or application launch policy.
- Separate Java handle owners may refer to one underlying process library and one global OpenBLAS
  thread-count state.
- Closing cannot safely race a later native invocation; active-call coordination remains outside
  this foundation.

## Validation evidence

- Implementation context `/root/openblas_0001_impl` ran the single final executable command
  `./gradlew :backends:openblas-provider:test`: passed 3 suites and 21 tests with 0 skipped,
  0 failures, and 0 errors (`OpenBlasLibraryPublicShapeTest` 3,
  `OpenBlasLibraryTest` 9, and `OpenBlasAbiContractTest` 9). Documentation context
  `/root/openblas_0001_docs` reused this evidence because it changed Javadoc and Markdown only;
  executable Java did not change afterward, so it did not rerun Java tests.
- Documentation context `/root/openblas_0001_docs` ran
  `./gradlew :backends:openblas-provider:javadoc`: passed. Inspection of the generated package,
  `OpenBlasLibrary`, and `OpenBlasLoadException` pages confirmed the two-type public surface,
  name/path factories, local lifecycle, caller ownership, failure contracts, and absence of
  package-private native handles from the public API.
- Documentation context `/root/openblas_0001_docs` ran
  `python3 /tmp/validate_synaptik_markdown.py docs/backend-guide/cpu-backend.md docs/glossary.md docs/planning/backends/openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md docs/planning/backends/openblas-provider/master-plan.md docs/planning/roadmap.md`:
  the first run identified an incorrect generated glossary anchor in the CPU guide. After that
  link was corrected, the same five-file command passed, including relative links, anchors,
  fences, final newlines, and trailing whitespace. It passed again after the final backend-guide
  profile refinement and evidence synchronization.
- Documentation context `/root/openblas_0001_docs` ran `git diff --check` after each final
  documentation/evidence refinement: every run passed, including the final combined diff.
- Manual source/test/generated-output checks confirmed package placement; the exact public and
  package-private surface; ordered `cblas_sgemm`, `cblas_dgemm`,
  `openblas_set_num_threads`, and `openblas_get_num_threads` inventory; exact 32-bit-`blasint`
  FFM descriptors; shared-arena ownership; ordered missing-symbol aggregation; partial cleanup
  with suppression; typed failure translation; fresh owners; atomic idempotent close; and no
  native invocation, platform discovery, fallback, route, configuration, or cache behavior.
- Manual `javap -public` inspection confirmed exactly the two public final types and the two
  factories, `isOpen()`, and `close()` members. `javap -private` confirmed the narrow native-access
  seam and ordered binding carrier without exposing either type publicly. Generated Javadoc
  contains only the package page and the two public type pages.
- The provider build remains JDK-only with zero project dependencies. The final change contains
  exactly the 15 authorized paths: 7 production paths including marker removal, 3 tests, 2
  explanatory documents, and 3 planning documents. Task/master/roadmap statuses are synchronized
  as Complete/In progress/In progress, the roadmap's Prepare history is preserved, task 0001 is
  the only detailed OpenBLAS task, and no task-0002 specification exists.
- No-change conclusions: Tensor, Compile, and Training APIs do not expose or consume the low-level
  provider; Config gains no native selection or fallback policy; Prepare and Runtime gain no
  provider handle or behavior; CPU production remains its marker and receives no route or
  execution code; architecture contracts/explanations and architecture tests remain accurate
  because ownership and dependency rules did not change; backend-conformance and integration
  tests remain premature because no provider invocation or CPU/Engine execution exists; Gradle
  remains unchanged because JDK 26 FFM needs no project dependency; and native packaging remains
  unchanged because the caller must supply an installed library name or absolute path.

## Implementation notes

- `OpenBlasNativeAccess` remains the exact two-method package-private test seam, and
  `FfmOpenBlasNativeAccess` is its stateless production implementation. The production path owns
  one shared arena, resolves every required symbol before reporting ordered omissions, binds the
  four fixed descriptors, and closes partial state before propagating failure.
- `OpenBlasNativeBindings` retains the arena and four handles in required-symbol order and uses an
  atomic one-way lifecycle. `OpenBlasLibrary` applies public validation and typed failure
  translation, returns a fresh owner per open, and uses its own atomic one-way lifecycle to make
  sequential and concurrent close attempts idempotent.
- The CPU guide and glossary distinguish the current loading/binding lifetime from planned GEMM
  invocation, thread mutation, CPU route choice, fallback, and native packaging.

## Completion summary

- Completed changes: replaced the marker with explicit name/path loading, complete four-symbol
  binding, typed failure reporting, and a fresh caller-owned closeable lookup lifetime.
- Files changed or created: exactly the 15 authorized provider production/test, CPU guide,
  glossary, task, master-plan, and roadmap paths listed in this specification.
- Tests and validation: the implementation context's final provider test passed 3 suites/21 tests;
  the documentation context passed provider Javadoc, five-file Markdown validation, generated-
  page/manual surface and scope checks, and `git diff --check`.
- Documentation-agent review: clean context `/root/openblas_0001_docs` independently reviewed the
  final source/tests and finalized all affected Javadocs and documentation without executable
  Java changes or duplicate test execution.
- Documentation impact: the CPU guide now describes the current loading/binding foundation and
  its planned boundaries; task/master/roadmap evidence and statuses are synchronized.
- Javadoc review: every production declaration documents ownership, lifetime, concurrency,
  parameters, results, failures, ABI assumptions, and current non-invocation boundary.
- Glossary impact: added the reusable OpenBLAS library-handle/lifetime distinction and linked it
  to the CPU guide.
- Unresolved issues: None.
- Follow-up required: None for task 0001. Task 0002 remains the next Draft row and has no detailed
  specification.

Status: Complete
