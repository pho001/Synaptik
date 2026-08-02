# Task 0003: Thread Control and Native Provider Checkpoint

## Status

Complete

## Goal

Complete the low-level OpenBLAS provider milestone by adding direct thread-count query and control
to the existing caller-owned `OpenBlasLibrary`, then validate the complete loading, binding,
thread-control, FLOAT32 GEMM, and FLOAT64 GEMM boundary against one explicitly supplied compatible
native library.

The Java API exposes native state without choosing a thread count or a CPU route. Ordinary tests
remain deterministic and native-free. The real-native checkpoint is an explicit, isolated command
that accepts one absolute library path from its caller and restores the observed prior thread count
before exit.

## Rationale and mental model

```text
CPU or another explicit caller
  -> chooses a positive thread count and coordinates shared-state use
  -> existing open OpenBlasLibrary
     -> query openblas_get_num_threads()
     -> invoke openblas_set_num_threads(count)
     -> existing SGEMM/DGEMM calls

explicit native checkpoint process
  -> caller supplies one compatible absolute library path
  -> open two Java owners for that exact library
  -> capture shared native thread count
  -> set 1, observe 1 through both owners
  -> run one fixed SGEMM and one fixed DGEMM case
  -> restore the captured count in finally
```

Synaptik must conservatively treat OpenBLAS thread control as mutable library/process state, not
state owned by one Java handle. A scoped Java lease or close-time restoration would suggest
isolation that the provider cannot guarantee across another handle, class loader, native caller,
or OpenBLAS consumer. The provider therefore exposes only low-level read and write operations.
CPU later owns thread selection, coordination, and any application-level restoration policy.

Treating the count as library/process-global is Synaptik's conservative integration rule. It is
inferred from OpenBLAS's context-free global utility symbols and documentation that the setter is
not a per-BLAS-call tuning control. It is not an OpenBLAS guarantee that independently loaded
binary copies, loader namespaces, or arbitrary native consumers share one coordinated state.

The checkpoint belongs in this task because fake handles can prove Java validation and exact ABI
forwarding but cannot prove that a supplied binary implements the bound ABI or produces the two
selected numerical results. Keeping the checkpoint opt-in avoids making an installed OpenBLAS
binary, native-access privilege, or platform-specific library location an ordinary test
prerequisite.

## Scope

- Add exactly `int threadCount()` and `void setThreadCount(int threadCount)` to the existing
  public final `OpenBlasLibrary`.
- Invoke only the already-bound `openblas_get_num_threads` and `openblas_set_num_threads` handles
  through one package-private final field-free helper.
- Validate the owner-open state before setter argument validation.
- Require a requested thread count greater than zero and reject a non-positive getter result.
- Use exact typed `MethodHandle.invokeExact` calls and stable failure translation.
- Document shared native-state, multiple-owner, lifecycle, concurrency, close-race, and
  restoration boundaries.
- Add deterministic fake-handle unit tests that require no native access or installed library.
- Add one test-source checkpoint launcher that consumes one caller-supplied absolute path, opens
  the same compatible library twice, verifies shared observation, restores the prior count in a
  `finally` path, and checks the exact SGEMM/DGEMM cases and tolerances below.
- Run the provider capability checkpoint, including repository and architecture tests, only when
  a caller or continuous-integration job supplies a compatible library and explicitly enables
  native access.
- Finalize affected Javadocs, package documentation, CPU guide, glossary impact, and planning
  evidence through the required separate clean documentation pass.

## Out of scope

- choosing a thread count, deriving one from hardware or workload facts, or interpreting config
- per-GEMM thread settings, thread-local settings, `openblas_set_num_threads_local`, OpenMP policy,
  affinity, processor count, parallel-mode, build-config, version, or additional native symbols
- a scoped thread-count lease, automatic restoration, compare-and-set, transaction, lock,
  executor, active-call counter, reference count, or provider-owned global coordinator
- pretending that separate Java owners, class loaders, or lookup arenas isolate native state
- restoring a thread count from `OpenBlasLibrary.close()` or retaining a previous/requested count
  in a Java owner
- CPU capability reporting, route selection, fallback, candidate generation, workload
  classification, tuning, cache behavior, preparation, storage, execution, or tracing
- Tensor, Model, Compiler, Planning, Prepare, Runtime, Engine, Trace, Backend Contract, Config, or
  concrete CPU implementation changes
- broad numerical conformance, performance, determinism, exceptional-value, NaN, infinity,
  signed-zero, rounding-mode, or cross-platform guarantees
- transpose, column-major, batched, strided, BFLOAT16, FLOAT16, complex, integral, sparse,
  quantized, or mixed-precision native cases
- native discovery, default names, platform filename probing, environment-variable or system-
  property lookup, library-path scanning, fallback, download, installation, extraction,
  packaging, bundling, or redistribution
- silently skipped native tests in the ordinary Gradle test task
- Gradle, root/settings, dependency, architecture-contract, ADR, architecture-test,
  backend-conformance, integration-test, CPU task, or generated-output changes
- a detailed CPU task or any later task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - OpenBLAS provider
  - CPU backend routes
  - Concrete backend modules
  - Dependency rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [OpenBLAS provider master plan](../master-plan.md)
- [CPU backend master plan](../../cpu/master-plan.md)
- [Task 0001: Library loading and required symbol binding](0001-library-loading-and-required-symbol-binding.md)
- [Task 0002: FLOAT32/FLOAT64 row-major GEMM invocation](0002-float32-float64-row-major-gemm-invocation.md)

Primary external contracts:

- [JDK 26 `Linker`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/Linker.html)
  defines native-linker downcalls as restricted operations, ABI-specific function descriptors,
  and exact method-handle carriers.
- [JDK 26 restricted methods](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/doc-files/RestrictedMethods.html)
  defines `--enable-native-access=ALL-UNNAMED` for this non-modular Gradle project and the
  `--illegal-native-access` launch policy used by the checkpoint.
- [JDK 26 `Arena`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/Arena.html)
  defines shared-arena cross-thread access, closure, and the failure possible when closure races a
  downcall.
- [JDK 26 `MethodHandle`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/invoke/MethodHandle.html)
  defines exact invocation typing and propagation of a target throwable.
- [OpenBLAS extension functions](https://www.openmathlib.org/OpenBLAS/docs/extensions/#utility-functions)
  identifies `openblas_get_num_threads` and `openblas_set_num_threads` as OpenBLAS utility
  functions.
- [OpenBLAS runtime thread-control documentation](https://github.com/OpenMathLib/OpenBLAS#setting-the-number-of-threads-at-runtime)
  documents `void openblas_set_num_threads(int)` and warns that the setting is library-wide rather
  than a per-BLAS-call tuning control.
- [OpenBLAS CBLAS user-manual example](https://www.openmathlib.org/OpenBLAS/docs/user_manual/#call-cblas-interface)
  confirms direct CBLAS GEMM use; task 0002's pinned `cblas.h` remains the exact signature and
  constant evidence.

## Architecture constraints

- `backends/openblas-provider` remains a low-level JDK/native leaf with zero project dependencies.
- Dependency direction remains `backends/cpu -> backends/openblas-provider`, never the reverse.
- The provider may expose loading, binding, GEMM, and thread control only. It must not interpret
  configuration or decide backend ownership, route, fallback, profitability, or tuning policy.
- The public surface contains no Tensor, operation, graph, backend identity, compile, planning,
  prepare, runtime, engine, storage, residency, or configuration type.
- `OpenBlasLibrary` remains the only public lifetime owner. Thread operations consume its existing
  package-private bindings without exposing native handles.
- Shared native state is not modeled as immutable prepared state or per-run state. CPU must later
  coordinate any policy before invoking provider calls.
- Class initialization and ordinary tests perform no native lookup or invocation.
- The checkpoint consumes an explicit caller argument. Production and ordinary tests must not
  discover or obtain a native library from an environment variable, system property, default
  filename, platform branch, registry, cache, or service locator.
- `legacy/pre-rewrite` remains read-only capability evidence and supplies no architecture,
  lifecycle, singleton, discovery, fallback, or API design.
- If implementation requires another public type, another native symbol, a project dependency,
  provider-owned global coordination, or CPU policy, stop and request a planning or architecture
  decision.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.provider.openblas` — extends the one public lifetime owner
  with two direct thread-control calls and adds one package-private stateless invocation helper.

No package is added.

Type placement:

- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary` — owns the two public
  methods because the already-bound handles are valid only inside this caller-owned open lookup
  lifetime.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasThreadControl` — package-private
  final, field-free helper that owns exact invocation, validation, result checking, and throwable
  translation without becoming policy or another native seam.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeCheckpoint` — test-source-only
  command-line checkpoint in the same package so it can exercise the public surface while
  remaining outside production and ordinary JUnit discovery.

Tests remain in the production package because they inject exact fake handles through the
existing package-private binding seam and lock the helper's visibility and shape.

## Exact public and package-private contracts

Add exactly these public methods to `OpenBlasLibrary`:

```java
public int threadCount();

public void setThreadCount(int threadCount);
```

`threadCount()` reports the positive count returned by the already-bound
`openblas_get_num_threads` handle. `setThreadCount(int)` passes one validated positive 32-bit C
`int` unchanged to the already-bound `openblas_set_num_threads` handle. Neither method caches,
normalizes, caps, derives, or otherwise interprets a value.

Do not add a result/lease type, overload, static convenience, default, availability query,
previous-value return, callback, lock API, public exception, or separate provider facade.

`OpenBlasThreadControl` is one package-private final type with no instance fields, public member,
or mutable static field. Its exact package-private operations are:

```java
static int threadCount(OpenBlasNativeBindings bindings);

static void setThreadCount(OpenBlasNativeBindings bindings, int threadCount);
```

The helper accepts only the already-checked exact bindings and the primitive setter input. It is
not a test seam, policy abstraction, global coordinator, or generic native utility.

## Validation, invocation, and failure contracts

### Query order

1. `OpenBlasLibrary.threadCount()` acquires bindings through the existing `bindings()` operation.
   A closed owner fails first with the existing exact
   `IllegalStateException("OpenBLAS library is closed")`.
2. The helper obtains the exact `getNumThreads()` handle and invokes it with typed
   `(int) handle.invokeExact()`.
3. If the native return is less than or equal to zero, fail with
   `IllegalStateException("OpenBLAS returned non-positive thread count: " + result)`.
4. Otherwise return the exact positive result.

### Setter order

1. `OpenBlasLibrary.setThreadCount(...)` acquires bindings through the existing `bindings()`
   operation. A closed owner therefore fails before argument validation.
2. Reject `threadCount <= 0` with exact
   `IllegalArgumentException("threadCount must be positive: " + threadCount)`.
3. Obtain the exact `setNumThreads()` handle and invoke it with typed
   `handle.invokeExact(threadCount)`.
4. Return normally without querying, caching, or claiming the effective count. A caller that
   needs confirmation performs a separate `threadCount()` call and accepts the shared-state race
   between those operations.

For each exact native invocation, catch `Error` first and rethrow it unchanged. Catch every other
`Throwable` and wrap it with the original cause in an `IllegalStateException` whose exact message
is:

- `OpenBLAS get thread count invocation failed`; or
- `OpenBLAS set thread count invocation failed`.

The non-positive getter result is checked after successful invocation and is not wrapped as an
invocation throwable. Setter input validation occurs outside the invocation catch. No new public
failure type is added.

## Shared state, lifecycle, concurrency, and restoration

- The thread count is mutable OpenBLAS library/process state. It is not a field of
  `OpenBlasLibrary`, a per-handle preference, a per-GEMM argument, a per-run resource, or an
  immutable prepared decision.
- As Synaptik's conservative integration rule, multiple Java owners of the same loaded OpenBLAS
  binary must be treated as potentially observing and mutating the same state. A successful setter
  through one may be visible through another. The provider does not claim coordination or shared
  state across independently loaded binary copies, loader namespaces, or arbitrary native users.
- Concurrent getters and setters are permitted as direct native calls, but the provider supplies
  no deterministic winner or ordering beyond each Java invocation boundary. A later observation
  may reflect any competing native mutation; a read-then-write or write-then-read sequence is not
  atomic.
- The provider does not serialize thread control with GEMM. Changing the count while any OpenBLAS
  call is active has no provider guarantee and must be prevented by the coordinating caller.
- `close()` remains local lifetime cleanup. It does not reset, restore, or otherwise mutate the
  OpenBLAS thread count, because another owner or native consumer may still use the library.
- Closing one Java owner does not make shared state private to another owner and does not promise
  physical library unload.
- A caller must not race `close()` with a thread query, setter, or GEMM call on the same owner.
  The operation may fail at the stable owner-open check or through the JDK/native lifetime. No
  successful-call guarantee, active-call tracking, or new synchronization is added.
- Restoration is caller-owned coordination: capture a positive count, prevent concurrent users,
  apply temporary changes, and restore in `finally` through a still-open owner. The provider
  stores no previous value and cannot guarantee restoration against independent native writers.
- Future CPU prepare/configuration owns the chosen thread count and when it is safe to install or
  restore that choice. This task does not make thread control part of Runtime hot-path discovery
  or per-invocation route selection.

## Explicit real-native checkpoint contract

### Preconditions

- A Java 26 runtime matching the repository toolchain is available.
- The caller supplies one absolute path to a compatible shared OpenBLAS binary. The binary must
  load on the current operating-system ABI and export the four task-0001 symbols under the
  ordinary 32-bit-`blasint` interface.
- The command runs in an isolated process with no concurrent OpenBLAS consumer and with enough
  permission to load the supplied file.
- The caller, CI image, or machine administrator installs/provides the binary. Repository code
  does not locate, download, unpack, build, package, or redistribute it.
- Native access is explicitly enabled for `ALL-UNNAMED`; `--illegal-native-access=deny` ensures
  that an accidental unenabled restricted operation fails instead of producing only a warning.

### Checkpoint launcher

`OpenBlasNativeCheckpoint.main(String[] args)` is test source, not a JUnit test and not production
API. It accepts exactly one argument, requires it to form an absolute `Path`, and passes it
directly to `OpenBlasLibrary.open(Path)`. It reads no environment variable or system property and
performs no alternate lookup.

Argument validation occurs before native access. A count other than one fails with
`IllegalArgumentException("expected exactly one absolute OpenBLAS library path argument")`.
After `Path.of(args[0])`, a non-absolute result fails with
`IllegalArgumentException("OpenBLAS library path must be absolute: " + path)`. An invalid path
syntax retains the JDK `InvalidPathException` and its diagnostics.

The launcher must:

1. open two owners against the exact supplied path;
2. query a positive original thread count;
3. set the count to `1` through the first owner;
4. require both owners to report `1`, proving shared observation for this supplied binary;
5. run the exact SGEMM and DGEMM cases below while the count is `1`;
6. in `finally`, restore the original positive count through a still-open owner and verify it
   through both owners before either closes; and
7. close matrix arenas and both library owners on every path.

If checkpoint work fails and restoration also fails, retain the original throwable and suppress
the distinct restoration throwable. If only restoration or verification fails, the checkpoint
fails. A repeated identical throwable must not be self-suppressed. The launcher prints one concise
success line including the restored count and otherwise exits nonzero through the uncaught
failure.

### Native numerical cases

Both precisions use this dense row-major case:

```text
A[2,3] = [1, 2, 3,
          4, 5, 6]

B[3,2] = [ 7,  8,
           9, 10,
          11, 12]

C initial[2,2] = [1, 2,
                  3, 4]

alpha = 0.5
beta  = 2.0

C expected = [31.0, 36.0,
              75.5, 85.0]
```

The checkpoint allocates distinct native A, B, and C segments with the precision's natural
alignment, initializes them explicitly, invokes the existing public `sgemm` or `dgemm`, and
checks every output element. It accepts a value when:

```text
abs(actual - expected) <= absoluteTolerance
    + relativeTolerance * abs(expected)
```

The exact tolerances are:

| Call | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| SGEMM | `1.0e-4f` | `1.0e-5f` |
| DGEMM | `1.0e-12` | `1.0e-12` |

These small finite cases use exactly representable inputs and expected values. The nonzero
tolerances avoid turning the checkpoint into a claim about one implementation's accumulation or
instruction sequence while remaining tight enough to catch wrong ABI carriers, layout,
transpose, leading dimensions, alpha/beta forwarding, or output placement. No other case is
covered by this checkpoint, and the existing task-0002 GEMM input contract is not narrowed by this
selected evidence case.

### Commands

Ordinary native-free validation first:

```bash
./gradlew :backends:openblas-provider:test \
  :backends:openblas-provider:testClasses
```

Then launch the explicitly supplied checkpoint from the repository root. `<JDK26>` and
`<ABSOLUTE_OPENBLAS_LIBRARY>` are caller substitutions, not production lookup mechanisms:

```bash
<JDK26>/bin/java \
  --enable-native-access=ALL-UNNAMED \
  --illegal-native-access=deny \
  -cp backends/openblas-provider/build/classes/java/main:backends/openblas-provider/build/classes/java/test \
  io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_LIBRARY>
```

On Windows, use `;` rather than `:` between the two class-path entries and pass an absolute path
to a compatible DLL. Do not replace the argument with discovery, an environment variable, or a
system property in production or ordinary tests.

## Affected files

Production:

- modify `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasThreadControl.java`
- modify `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/package-info.java`

Tests and checkpoint:

- modify `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryPublicShapeTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasThreadControlTest.java`
- modify `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasAbiContractTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasNativeCheckpoint.java`

Explanatory documentation:

- modify `docs/backend-guide/cpu-backend.md`
- modify `docs/glossary.md` only if the final public/shared-state distinction changes reusable
  terminology; otherwise record the reasoned no-change conclusion

Planning:

- this task
- modify `docs/planning/backends/openblas-provider/master-plan.md`
- modify `docs/planning/roadmap.md`

`FfmOpenBlasNativeAccess`, `OpenBlasNativeAccess`, `OpenBlasNativeBindings`,
`OpenBlasGemmInvocation`, `OpenBlasLoadException`, completed tasks 0001/0002, provider/CPU build
files, root/settings files, architecture documents/tests, conformance/integration projects, CPU
source/tests, and generated output are review-only and must remain byte-for-byte unchanged.

## Maximum scope

At most 12 paths:

- 3 provider production paths;
- 4 provider test/checkpoint paths;
- at most 2 explanatory documentation paths; and
- 3 planning paths.

If the glossary needs no edit, the final implementation scope is at most 11 paths. No completed
task byte may change. If implementation needs Gradle configuration, another public member/type,
another native symbol or seam, another file, or more than 12 paths, stop and propose a bounded
planning correction or follow-up.

## Acceptance criteria

- [x] `OpenBlasLibrary` exposes exactly the two specified additional public methods and no new
      public type, overload, lease, result, previous-value, callback, or failure surface.
- [x] Both methods perform the existing owner-open check first and delegate to one package-private
      final field-free helper over the existing handles.
- [x] Setter validation rejects every non-positive value with the exact specified type/message
      before native invocation, but after the owner-open check.
- [x] Getter returns an exact positive native value and rejects a non-positive result with the
      exact specified `IllegalStateException` message.
- [x] The getter and setter use exact typed `invokeExact`, rethrow every `Error` unchanged, and
      wrap every other invocation throwable with the exact operation-specific message and cause.
- [x] Deterministic fake tests prove exact call counts/values, validation order, invalid getter
      result handling, checked/runtime/wrong-type/Error behavior, closed-owner precedence,
      multiple-owner shared-handle observation, and the unsupported close-race boundary without
      requiring an installed library.
- [x] Java owners retain only local lookup lifetime. No prior/current thread count is stored, and
      `close()` performs no native query, setter, or restoration.
- [x] Javadocs and package documentation describe Synaptik's conservative library/process-global
      treatment, potentially shared mutation across owners, no deterministic winner, non-atomic
      read/write sequences, GEMM coordination, caller-owned restoration, and close races without
      claiming coordination across independently loaded copies or arbitrary native consumers.
- [x] Ordinary tests perform no native load/invocation, use no skip/assumption for a missing
      library, and require no native-access launch option.
- [x] The test-source checkpoint accepts exactly one explicit absolute path and contains no name
      selection, platform probing, property/environment lookup, download, packaging, fallback, or
      alternate candidate.
- [x] With a supplied compatible library and the exact native-access launch flags, the checkpoint
      verifies positive query, cross-owner shared observation at count `1`, the exact SGEMM/DGEMM
      values within the specified tolerances, and restoration/verification of the original count
      in `finally`.
- [x] Restoration failure preserves any earlier distinct primary failure and never attempts Java
      self-suppression; both owners and all checkpoint arenas close on every path.
- [x] Production remains JDK-only with zero project dependencies and contains no CPU, config,
      route, fallback, tuning, Tensor, Prepare, Runtime, discovery, or global-coordination logic.
- [x] Existing task-0001 loading/binding and task-0002 GEMM contracts/tests remain unchanged except
      for the narrowly authorized public-shape and ABI-test extensions.
- [x] CPU code, Gradle/root/settings, architecture files/tests, backend conformance, integration,
      native packaging, completed task files, and generated outputs remain unchanged.
- [x] A separate clean documentation-focused agent finalizes Javadocs, CPU guide, glossary impact,
      task/master/roadmap evidence, and documentation validation in the same overall change.
- [x] Task 0003 remained `Ready` until implementation, final module validation, supplied native
      checkpoint, capability checkpoint, and clean documentation passes all succeeded; task,
      master plan, and roadmap then moved to Complete and CPU became the next Draft planning
      frontier.
- [x] Exact scope, package placement, public/package-private surface, links, anchors, fences,
      final newlines, trailing whitespace, status/history, no-later-spec, and final diff checks
      pass.

## Tests / validation

Focused development tests must cover:

- exact public methods and package-private helper visibility/stateless shape;
- exact existing getter/setter handle types and exact invocation carriers;
- closed-owner precedence over setter validation;
- `Integer.MIN_VALUE`, `-1`, and `0` rejection plus positive boundary/value forwarding;
- positive getter returns and zero/negative native-result rejection;
- checked throwable, runtime throwable, wrong method type, and unchanged `Error` behavior for
  both calls;
- sequential and concurrent fake calls with deterministic coordination, including two owners
  backed by one fake global thread-count cell;
- absence of cached per-owner count, automatic close restoration, synchronization, another
  native seam/symbol, project dependency, and forbidden vocabulary; and
- checkpoint argument validation, exact matrix data/expected values/tolerances, restoration
  suppression logic, explicit path forwarding, and absence from ordinary JUnit discovery.

After executable Java stabilizes, the implementation context runs exactly one final ordinary
affected-module command:

```bash
./gradlew :backends:openblas-provider:test
```

It then compiles the checkpoint if necessary and runs the exact explicit real-native command from
the checkpoint contract with the caller/CI-supplied compatible library. Missing native input is a
blocked checkpoint, not a skipped passing test and not permission to discover a library.

Because this task closes the provider capability milestone, the implementation context also runs
the capability checkpoint after the native command succeeds:

```bash
./gradlew test :testing:architecture-tests:test
```

The clean documentation-focused pass reuses successful executable evidence unless it changes
executable Java or records a concrete stale-evidence risk. After final Javadocs and Markdown it
runs:

```bash
./gradlew :backends:openblas-provider:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/backend-guide/cpu-backend.md \
  docs/glossary.md \
  docs/planning/backends/openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md \
  docs/planning/backends/openblas-provider/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

If the glossary is unchanged, omit it from the Markdown command and record the no-change reason.
The documentation pass must inspect generated Javadocs and verify exact surface/package placement,
shared-state/lifecycle/restoration wording, zero project dependencies, exact path maximum,
unchanged completed tasks, synchronized Ready-to-Complete status history, no later provider or CPU
task specification, direct external links, anchors, fences, final newlines, and trailing
whitespace.

## Dependencies

- [OpenBLAS provider 0001](0001-library-loading-and-required-symbol-binding.md) — Complete; supplies
  the explicit library lifetime and exact already-bound thread/GEMM handles.
- [OpenBLAS provider 0002](0002-float32-float64-row-major-gemm-invocation.md) — Complete; supplies
  the exact public dense row-major SGEMM/DGEMM calls checked by the native milestone.
- Prepare 0003, Runtime 0014, Compiler 0006, Planning 0006, and Backend Contract 0004 — Complete;
  establish downstream boundaries without becoming provider dependencies.
- JDK 26 FFM and restricted-method launch contracts — current root toolchain baseline.
- One caller/CI-supplied compatible OpenBLAS shared library — required only to complete the
  explicit native checkpoint, not ordinary unit tests or production discovery.

No CPU implementation task may begin before this provider task and checkpoint are Complete.

## Follow-up tasks

- CPU backend 0001 remains a Draft row only. A later separate planning step may make it Ready
  after this task closes the provider milestone. CPU will own capability truth, thread-count
  selection/coordination, route/fallback policy, normalization, storage, preparation, and
  execution.
- No later OpenBLAS provider task is planned by this milestone. A new symbol, precision, layout,
  or lifecycle capability requires a separately justified future master-plan row and task.

Do not create a CPU task file or another provider task file in this change.

## Architecture impact

Expected impact: None.

This task implements thread control already assigned to the low-level provider and validates its
already planned native surface. It adds no project edge, backend identity, route, fallback, or
shared lifecycle rule. If implementation requires provider-owned global coordination, config
interpretation, another native symbol, or a dependency outside the JDK, stop and report the
conflict rather than editing architecture.

## Documentation impact

The separate documentation-focused pass must:

- finalize `OpenBlasLibrary`, `OpenBlasThreadControl`, and package Javadocs for exact thread-count
  values, validation, failures, the conservative shared-state inference, multiple owners,
  lifecycle, concurrency, restoration, and CPU-policy boundaries;
- update the CPU backend guide to mark loading/binding, low-level GEMM, thread control, and the
  explicit native provider checkpoint current while keeping CPU capability, route, fallback,
  storage, prepare, and execution work planned;
- update the glossary only if the final change introduces or changes a reusable project term;
  otherwise record why the existing OpenBLAS library-handle/GEMM entries and ordinary shared-
  state language are sufficient;
- synchronize task 0003, provider master plan, and roadmap evidence/status only after every
  implementation, native, checkpoint, and documentation gate succeeds;
- preserve architecture documentation because ownership and dependency rules do not change; and
- record reasoned no-change conclusions for Tensor, Compile, and Training APIs; Model capability
  and operation contracts; Config, Prepare, Runtime, Backend Contract, Trace, CPU implementation;
  architecture/tests; backend-conformance/integration; Gradle/root/settings; native packaging;
  completed tasks; generated outputs; and other modules.

## Implementation prompt

```text
You are a clean-context implementation agent working in the Synaptik repository. Do not commit or
push.

Read in full: AGENTS.md; ARCHITECTURE.md; docs/planning/planning-guide.md;
docs/developer-guide/documentation-rules.md; the General, API/Javadoc, Backend Guide, Example, and
Planning documentation profiles; docs/planning/backends/openblas-provider/master-plan.md;
completed OpenBLAS tasks 0001 and 0002; and task 0003 at
docs/planning/backends/openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md.
Read the directly referenced architecture pages and official JDK 26/OpenBLAS sources, current
provider source/tests/Javadocs/build, CPU placeholder/build/master-plan boundary, root/settings
Java 26 configuration, roadmap frontier, and relevant architecture tests. Treat legacy only as
read-only capability evidence, never design authority.

Implement task 0003 exactly within its twelve-path ceiling. Add only the two public methods, one
package-private field-free helper, deterministic fake tests, and the explicit argument-driven
test-source native checkpoint with the exact validation, failure, shared-state, lifecycle,
restoration, numerical, tolerance, and launch contracts in the task. Use only already-bound
handles. Add no symbol, discovery/property/environment lookup, automatic restoration, scoped
lease, global coordinator, CPU/config/route/fallback behavior, Gradle/native packaging, project
dependency, architecture change, or later task. Stop on architecture conflict, scope overflow,
need for another public abstraction/symbol/seam, or ambiguity that changes the recorded contract.

Run the one final ordinary provider test command after executable Java stabilizes, then the exact
native checkpoint with the caller-supplied compatible absolute library path, then the repository/
architecture capability checkpoint. A missing supplied library blocks completion; do not discover
one or silently skip. Hand the actual diff and exact evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass reuses successful executable
evidence and independently finalizes Javadocs, package documentation, CPU guide, glossary impact,
task/master/roadmap, generated-Javadoc, Markdown/link/surface/scope/status/history checks, and
git diff --check without rerunning successful Java tests unless executable behavior changes or a
concrete stale-evidence risk is recorded. Keep task 0003 Ready until every gate passes; only then
mark the provider milestone Complete. Do not create a CPU task file.
```

## Local decisions

- Keep query and setter on `OpenBlasLibrary`; their handles and validity are inseparable from its
  existing caller-owned lookup lifetime, and no independent public thread-control abstraction has
  a current consumer.
- Use one field-free package-private helper, matching task 0002's exact-invocation boundary without
  broadening the native-access test seam.
- Accept positive 32-bit counts only. Zero and negative values cannot express a CPU thread
  candidate and would delegate undocumented normalization policy to the native library.
- Reject a non-positive getter result rather than exposing an invalid count as usable provider
  state. This remains a low-level ABI sanity check, not an availability or fallback decision.
- Do not return the prior value from the setter: the already-bound OpenBLAS setter returns `void`,
  and synthesizing a return with a preceding getter would falsely imply one atomic operation.
- Do not add a scoped restoration object or close-time reset. Java lookup owners do not own the
  shared native setting and cannot exclude independent writers.
- Use one explicit non-JUnit test-source launcher rather than a Gradle property or environment-
  gated test. A missing library is then a visible checkpoint precondition, never a skipped
  ordinary test.
- Verify two owners and restore in an isolated process because that is the smallest honest check
  of the shared-state contract without claiming coordination with arbitrary native consumers.
- Use one nontrivial alpha/beta row-major case per precision. It exercises both inputs, existing
  output, layout, leading dimensions, and scalar forwarding while keeping expected results
  independently reviewable.

## Known limitations

- The provider cannot make a query/set pair atomic or coordinate another Java class loader,
  native caller, OpenMP runtime, or application component using the same OpenBLAS binary.
- `close()` cannot safely race a thread-control or GEMM invocation and does not undo shared native
  state.
- The setter is not a per-call tuning control. A coordinating caller must install a choice only
  at a safe lifecycle point and restore it explicitly when required.
- The task supports only the already-bound ordinary 32-bit-`blasint` getter/setter symbols. It
  does not bind local-thread, version, build-config, processor-count, or parallel-mode functions.
- The native checkpoint proves only one supplied binary, process, ABI, shared-state observation,
  and two small finite GEMM cases. It does not establish general numerical conformance,
  determinism, performance, packaging compatibility, or support for another OpenBLAS build.
- Completing the task requires a caller/CI-supplied compatible shared library and native-access
  permission. Ordinary development remains native-free, but the provider milestone cannot be
  marked Complete without the explicit checkpoint evidence.

## Validation evidence

- During development, implementation context `/root/task_0003_impl` first ran
  `./gradlew :backends:openblas-provider:test`: 48 tests ran with one failure in
  `OpenBlasThreadControlTest.rejectsNonPositiveGetterResults`. Production compiled; the test fake
  incorrectly used `MethodHandle.bindTo` with a primitive `int`. After correcting only that fake
  handle construction, the same command passed 48 tests with zero failures. Two checkpoint tests
  were then added before the final recorded module run below.
- Implementation context `/root/task_0003_impl` ran the final ordinary command
  `./gradlew :backends:openblas-provider:test`: passed 5 suites and 50 tests with zero skips,
  failures, or errors (`OpenBlasAbiContractTest` 9, `OpenBlasGemmInvocationTest` 17,
  `OpenBlasLibraryPublicShapeTest` 5, `OpenBlasLibraryTest` 9, and
  `OpenBlasThreadControlTest` 10). Clean documentation context
  `/root/task_0003_impl/openblas_0003_docs` reused this evidence because it changed only Javadoc
  and Markdown; executable Java and tests did not change afterward.
- Clean validation context `/root/task_0003_native_resume` ran `brew --prefix openblas` exactly
  once and obtained `/opt/homebrew/opt/openblas`. From that prefix it derived the Darwin arm64
  candidate `/opt/homebrew/opt/openblas/lib/libopenblas.dylib`, resolved the real regular-file
  target `/opt/homebrew/Cellar/openblas/0.3.33/lib/libopenblasp-r0.3.33.dylib`, and supplied that
  target as the checkpoint's sole argument. The target was absolute, `stat` reported Regular File,
  mode `-r--r--r--`, and size 22,337,424 bytes, `file` reported a Mach-O 64-bit dynamically linked
  shared library for arm64, and `nm -gU` found `_cblas_sgemm`, `_cblas_dgemm`,
  `_openblas_get_num_threads`, and `_openblas_set_num_threads`.
- Clean validation context `/root/task_0003_native_resume` used
  `/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home/bin/java`, Java
  26.0.1 arm64, and ran
  `/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home/bin/java --enable-native-access=ALL-UNNAMED --illegal-native-access=deny -cp backends/openblas-provider/build/classes/java/main:backends/openblas-provider/build/classes/java/test io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeCheckpoint /opt/homebrew/Cellar/openblas/0.3.33/lib/libopenblasp-r0.3.33.dylib`.
  It exited 0 with `OpenBLAS native checkpoint passed; restored thread count 16`, covering positive
  query, shared observation through two owners, the exact SGEMM/DGEMM cases, and restoration plus
  cross-owner verification in `finally`.
- Only after the native command succeeded, clean validation context
  `/root/task_0003_native_resume` ran `./gradlew test :testing:architecture-tests:test`: passed
  with `BUILD SUCCESSFUL` in 768 ms and 54 actionable tasks (2 executed, 52 up-to-date). The
  provider test task executed and the architecture-test task was up-to-date. No executable Java
  changed afterward, so this remains the final capability-checkpoint evidence.
- Documentation context `/root/task_0003_impl/openblas_0003_docs` ran
  `./gradlew :backends:openblas-provider:javadoc`: passed. Generated package and
  `OpenBlasLibrary` pages were inspected for the exact two-method addition, positive-value and
  failure contracts, conservative shared-state wording, multiple-owner limits, non-atomic and
  no-winner concurrency, GEMM coordination, caller-owned restoration, local close lifetime, and
  close-race boundary. `OpenBlasThreadControl` remains package-private and therefore is not a
  generated public page.
- The exact five-file Markdown command
  `python3 /tmp/validate_synaptik_markdown.py docs/backend-guide/cpu-backend.md docs/glossary.md docs/planning/backends/openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md docs/planning/backends/openblas-provider/master-plan.md docs/planning/roadmap.md`
  passed relative-link, heading-anchor, fence, final-newline, and trailing-whitespace checks.
  Glossary validation is included because the reusable `OpenBlasLibrary` entry previously said
  thread-control invocation was absent and required correction.
- Manual source, test, build, generated-output, and planning checks confirmed the exact public
  surface and package-private final field-free helper, exact package placement, four existing
  native symbols, zero project dependencies, exactly twelve authorized changed paths, unchanged
  completed task files 0001/0002, synchronized Complete task/milestone/project-area status,
  preserved roadmap history, CPU as the next Draft planning frontier, and absence of a later
  OpenBLAS-provider or CPU task specification.
- `git diff --check` passed on the final combined change.
- Clean completion-documentation context
  `/root/task_0003_native_resume/openblas_0003_completion_docs` independently inspected the final
  twelve-path implementation, tests, Javadocs, explanatory documentation, planning records, and
  supplied native/capability-checkpoint evidence. It changed only this task, the provider master
  plan, and the roadmap to synchronize completion; no executable Java or Javadoc changed, so it
  reused both the 5-suite/50-test provider result and the later native and repository/architecture
  checkpoint results without rerunning Java or Javadoc.
- Completion-documentation context
  `/root/task_0003_native_resume/openblas_0003_completion_docs` ran
  `python3 /tmp/validate_synaptik_markdown.py docs/planning/backends/openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md docs/planning/backends/openblas-provider/master-plan.md docs/planning/roadmap.md`:
  passed with `validated 3 Markdown files`. The exact twelve-path scope, Complete task/master/
  roadmap status, next-Draft CPU frontier, public/helper surface, package placement, zero project
  dependencies, preserved history, and absence of later provider/CPU specifications also passed.
  Completed tasks 0001 and 0002 remained byte-for-byte unchanged with SHA-256 values
  `4e058be28272b85937b199cffc37c77396f80b99ae8ad84d939af937d13e0d54` and
  `1fdb5fa6d552ec869b3be0c41e9acc6d61f7bbf4c9af13f02b79f55abe9b166b`, respectively.
- Completion-documentation context
  `/root/task_0003_native_resume/openblas_0003_completion_docs` ran `git diff --check` after the
  planning-only synchronization: passed with no output. It reran the three-file Markdown and
  whitespace checks after recording this evidence; both remained successful.
- No-change conclusions: Tensor, Compile, and Training APIs and Model capabilities/operation
  contracts do not expose or consume this low-level native state; Config, Prepare, Runtime,
  Backend Contract, Trace, and CPU implementation gain no policy, handle, route, storage, or
  execution behavior. Architecture contracts/explanations and architecture tests remain accurate
  because ownership and dependency direction do not change. Backend conformance and integration
  remain premature without a CPU execution path. Gradle/root/settings remain unchanged because
  the provider is still JDK-only and the checkpoint is an explicit launcher. Native packaging,
  generated outputs, completed tasks, and every other module remain unchanged. No further
  Javadoc, CPU-guide, glossary, or other explanatory-document edit was needed in this completion
  pass because the native run changed no API, behavior, terminology, example, or architecture;
  the earlier clean documentation pass had already finalized those contracts.

## Implementation notes

- Added exactly `threadCount()` and `setThreadCount(int)` to the existing public lifetime owner.
  Both acquire the checked bindings first and delegate to one package-private final field-free
  helper using the already-bound exact handles.
- Added deterministic fake-handle coverage for value forwarding, validation order, native-result
  checking, throwable translation, multiple-owner observation, concurrency boundaries, no close
  restoration, and checkpoint argument/suppression behavior.
- Added the test-source argument-driven native checkpoint with explicit restoration and the fixed
  SGEMM/DGEMM cases. It performs no discovery and is not an ordinary JUnit test.
- Finalized the affected public/helper/package Javadocs, CPU guide, glossary entry, and planning
  records without changing executable Java or tests in the documentation pass.

## Completion summary

- Completed changes: implemented the exact direct positive thread-count query/control surface,
  package-private invocation helper, deterministic tests, and explicit native checkpoint launcher;
  finalized all affected documentation and planning evidence.
- Files changed or created: exactly twelve authorized paths—three provider production paths, four
  provider test/checkpoint paths, the CPU guide, glossary, this task, provider master plan, and
  roadmap.
- Tests and validation: reused the implementation context's passing 5-suite/50-test provider run,
  the clean native checkpoint that restored thread count 16, and the later passing repository/
  architecture checkpoint with 54 actionable tasks; the documentation contexts passed provider
  Javadoc, Markdown, generated-page, surface/scope/status/history/later-specification,
  completed-task, and whitespace checks.
- Documentation-agent review: clean context `/root/task_0003_impl/openblas_0003_docs`
  independently finalized Javadocs, package documentation, CPU guide, glossary impact, and
  planning evidence. Clean completion context
  `/root/task_0003_native_resume/openblas_0003_completion_docs` independently reviewed the final
  twelve-path change and synchronized only the three planning paths after the executable gates.
  Neither context changed executable Java or duplicated successful Java-test execution.
- Documentation impact: the CPU guide now teaches direct thread control and its coordination,
  restoration, close, and checkpoint boundaries; task, master plan, and roadmap now synchronize
  the Complete provider milestone and next Draft CPU planning frontier.
- Javadoc review: `OpenBlasLibrary`, `OpenBlasThreadControl`, and package documentation cover exact
  values/failures, conservative potential sharing, explicit exclusions, concurrency, GEMM
  coordination, caller-owned restoration, local lifetime, and close races.
- Glossary impact: updated the reusable `OpenBlasLibrary` entry because its prior statement that
  thread-control invocation was absent became stale; no new reusable term was introduced.
- Unresolved issues: None for task 0003 or the selected OpenBLAS provider milestone.
- Follow-up required: None for this task. CPU is the next Draft planning frontier; a separate
  planning step may create the first detailed CPU task without extending this change.

Status: Complete
