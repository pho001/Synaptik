# Task 0003: Thread Control and Native Provider [Checkpoint](../../../../glossary.md#term-checkpoint)

## Status

Complete

## Goal

Complete the low-level [OpenBLAS](../../../../glossary.md#term-openblas) provider milestone by adding direct thread-count query and control
to the existing caller-owned [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), then validate the complete loading, [binding](../../../../glossary.md#term-binding),
thread-control, [FLOAT32](../../../../glossary.md#term-float32) [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm), and [FLOAT64](../../../../glossary.md#term-float64) GEMM boundary against one explicitly supplied compatible
[native library](../../../../glossary.md#term-native-library).

The Java [API](../../../../glossary.md#term-application-programming-interface-api) exposes native state without choosing a thread count or a [CPU](../../../../glossary.md#term-cpu) route. Ordinary tests
remain deterministic and native-free. The real-native [checkpoint](../../../../glossary.md#term-checkpoint) is an explicit, isolated command
that accepts one absolute library path from its caller and restores the observed prior thread count
before exit.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

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

Synaptik must conservatively treat [OpenBLAS](../../../../glossary.md#term-openblas) thread control as [mutable](../../../../glossary.md#term-mutable) library/process state, not
state owned by one Java handle. A scoped Java [lease](../../../../glossary.md#term-lease) or close-time restoration would suggest
isolation that the provider cannot guarantee across another handle, class loader, native caller,
or OpenBLAS consumer. The provider therefore exposes only low-level read and write operations.
[CPU](../../../../glossary.md#term-cpu) later owns thread selection, coordination, and any application-level restoration policy.

Treating the count as library/process-global is Synaptik's conservative integration rule. It is
inferred from OpenBLAS's context-free global utility symbols and documentation that the setter is
not a per-[BLAS](../../../../glossary.md#term-blas)-call tuning control. It is not an OpenBLAS guarantee that independently loaded
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) copies, loader namespaces, or arbitrary native consumers share one coordinated state.

The [checkpoint](../../../../glossary.md#term-checkpoint) belongs in this task because fake handles can prove Java [validation](../../../../glossary.md#term-validation) and exact [ABI](../../../../glossary.md#term-application-binary-interface-abi)
forwarding but cannot prove that a supplied binary implements the bound ABI or produces the two
selected numerical results. Keeping the checkpoint opt-in avoids making an installed OpenBLAS
binary, native-access privilege, or platform-specific library location an ordinary test
prerequisite.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `int threadCount()` and `void setThreadCount(int threadCount)` to the existing
  public final [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary).
- Invoke only the already-bound `openblas_get_num_threads` and `openblas_set_num_threads` handles
  through one package-private final field-free helper.
- Validate the [owner](../../../../glossary.md#term-owner)-open state before setter [argument](../../../../glossary.md#term-argument) [validation](../../../../glossary.md#term-validation).
- Require a requested thread count greater than zero and reject a non-positive getter result.
- Use exact typed `MethodHandle.invokeExact` calls and stable failure translation.
- Document shared native-state, multiple-owner, lifecycle, concurrency, close-race, and
  restoration boundaries.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [concurrency glossary](../../../../glossary.md#term-concurrency); [race glossary](../../../../glossary.md#term-race))</span>
- Add deterministic fake-handle unit tests that require no native access or installed library.
- Add one test-source checkpoint launcher that consumes one caller-supplied absolute path, opens
  the same compatible library twice, verifies shared observation, restores the prior count in a<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  `finally` path, and checks the exact [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/[DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) cases and tolerances below.
- Run the provider capability checkpoint, including repository and architecture tests, only when
  a caller or continuous-integration job supplies a compatible library and explicitly enables
  native access.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>
- Finalize affected Javadocs, package documentation, CPU guide, glossary impact, and planning
  evidence through the required separate clean documentation pass.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- choosing a thread count, deriving one from hardware or workload facts, or interpreting config
- per-[GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) thread settings, thread-local settings, `openblas_set_num_threads_local`, OpenMP policy,
  affinity, processor count, parallel-mode, build-config, version, or additional native symbols
- a scoped thread-count lease, automatic restoration, compare-and-set, transaction, lock,
  executor, active-call counter, reference count, or provider-owned global coordinator<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [compare-and-set glossary](../../../../glossary.md#term-compare-and-set-cas); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- pretending that separate Java [owners](../../../../glossary.md#term-owner), class loaders, or [lookup](../../../../glossary.md#term-lookup) [arenas](../../../../glossary.md#term-arena) isolate native state
- restoring a thread count from `OpenBlasLibrary.close()` or retaining a previous/requested count
  in a Java owner<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- CPU capability reporting, route selection, fallback, candidate generation, workload
  classification, tuning, cache behavior, preparation, storage, execution, or tracing<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Tensor, Model, Compiler, Planning, Prepare, Runtime, Engine, Trace, Backend Contract, Config, or
  concrete CPU implementation changes<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [Trace glossary](../../../../glossary.md#trace); [Backend glossary](../../../../glossary.md#backend))</span>
- broad numerical conformance, performance, determinism, exceptional-value, NaN, infinity,
  signed-zero, rounding-mode, or cross-platform guarantees<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- transpose, column-major, batched, strided, BFLOAT16, FLOAT16, complex, integral, sparse,
  quantized, or mixed-precision native cases<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [column-major glossary](../../../../glossary.md#term-column-major); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [precision glossary](../../../../glossary.md#term-precision))</span>
- native discovery, default names, platform filename probing, environment-variable or system-
  property lookup, library-path scanning, fallback, download, installation, extraction,
  packaging, bundling, or redistribution
- silently skipped native tests in the ordinary [Gradle](../../../../glossary.md#term-gradle) test task
- Gradle, root/settings, dependency, architecture-contract, ADR, architecture-test,
  backend-conformance, integration-test, CPU task, or generated-output changes<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- a detailed [CPU](../../../../glossary.md#term-cpu) task or any later task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - [OpenBLAS](../../../../glossary.md#term-openblas) provider
  - [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [Dependency](../../../../glossary.md#term-dependency) rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [OpenBLAS provider master plan](../master-plan.md)
- [CPU backend master plan](../../cpu/master-plan.md)
- [Task 0001: Library loading and required symbol binding](0001-library-loading-and-required-symbol-binding.md)<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [binding glossary](../../../../glossary.md#term-binding))</span>
- [Task 0002: FLOAT32/FLOAT64 row-major GEMM invocation](0002-float32-float64-row-major-gemm-invocation.md)<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [row-major glossary](../../../../glossary.md#term-row-major); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>

Primary external contracts:

- [JDK 26 `Linker`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/Linker.html)<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
  defines native-linker downcalls as restricted operations, ABI-specific function descriptors,
  and exact method-handle carriers.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- [JDK 26 restricted methods](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/doc-files/RestrictedMethods.html)
  defines `--enable-native-access=ALL-UNNAMED` for this non-modular [Gradle](../../../../glossary.md#term-gradle) project and the
  `--illegal-native-access` launch policy used by the [checkpoint](../../../../glossary.md#term-checkpoint).
- [JDK 26 `Arena`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/Arena.html)<span data-glossary-navigation="true"> ([Arena glossary](../../../../glossary.md#term-arena))</span>
  defines shared-arena cross-thread access, closure, and the failure possible when closure races a
  downcall.<span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall))</span>
- [JDK 26 `MethodHandle`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/invoke/MethodHandle.html)<span data-glossary-navigation="true"> ([MethodHandle glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture))</span>
  defines exact invocation typing and propagation of a target throwable.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- [OpenBLAS extension functions](https://www.openmathlib.org/OpenBLAS/docs/extensions/#utility-functions)
  identifies `openblas_get_num_threads` and `openblas_set_num_threads` as OpenBLAS utility
  functions.
- [OpenBLAS runtime thread-control documentation](https://github.com/OpenMathLib/OpenBLAS#setting-the-number-of-threads-at-runtime)
  documents `void openblas_set_num_threads(int)` and warns that the setting is library-wide rather
  than a per-BLAS-call tuning control.<span data-glossary-navigation="true"> ([BLAS glossary](../../../../glossary.md#term-blas))</span>
- [OpenBLAS CBLAS user-manual example](https://www.openmathlib.org/OpenBLAS/docs/user_manual/#call-cblas-interface)<span data-glossary-navigation="true"> ([CBLAS glossary](../../../../glossary.md#term-cblas))</span>
  confirms direct CBLAS GEMM use; task 0002's pinned `cblas.h` remains the exact signature and
  constant evidence.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [constant glossary](../../../../glossary.md#term-constant))</span>

## Architecture constraints

- `backends/openblas-provider` remains a low-level [JDK](../../../../glossary.md#term-java-development-kit-jdk)/native [leaf](../../../../glossary.md#term-leaf) with zero project dependencies.
- [Dependency direction](../../../../glossary.md#java-platform-and-build-tools) remains `backends/cpu -> backends/openblas-provider`, never the reverse.
- The provider may expose loading, binding, GEMM, and thread control only. It must not interpret
  configuration or decide backend ownership, route, fallback, profitability, or tuning policy.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [backend ownership glossary](../../../../glossary.md#backend-ownership))</span>
- The public surface contains no Tensor, operation, graph, backend identity, compile, planning,
  prepare, runtime, engine, storage, residency, or configuration type.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [backend identity glossary](../../../../glossary.md#backend-identity--backendid); [compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [residency glossary](../../../../glossary.md#residency))</span>
- [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) remains the only public lifetime owner. Thread operations consume its existing
  package-private bindings without exposing native handles.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [owner glossary](../../../../glossary.md#term-owner))</span>
- Shared native state is not modeled as immutable prepared state or per-run state. CPU must later
  coordinate any policy before invoking provider calls.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [run state glossary](../../../../glossary.md#run-state--runstate); [CPU glossary](../../../../glossary.md#term-cpu); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Class initialization and ordinary tests perform no native [lookup](../../../../glossary.md#term-lookup) or invocation.
- The checkpoint consumes an explicit caller argument. Production and ordinary tests must not
  discover or obtain a native library from an environment variable, system property, default
  filename, platform branch, registry, cache, or service locator.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [argument glossary](../../../../glossary.md#term-argument); [native library glossary](../../../../glossary.md#term-native-library); [registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- `legacy/pre-rewrite` remains read-only capability evidence and supplies no architecture,
  lifecycle, singleton, discovery, fallback, or API design.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [singleton glossary](../../../../glossary.md#term-singleton); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- If implementation requires another public type, another native symbol, a project dependency,
  provider-owned global coordination, or CPU policy, stop and request a planning or architecture
  decision.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.provider.openblas` — extends the one public lifetime owner
  with two direct thread-control calls and adds one package-private stateless invocation helper.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [owner glossary](../../../../glossary.md#term-owner))</span>

No package is added.

Type placement:

- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary` — owns the two public
  methods because the already-bound handles are valid only inside this caller-owned open lookup
  lifetime.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasThreadControl` — package-private
  final, field-free helper that owns exact invocation, validation, result checking, and throwable
  translation without becoming policy or another native seam.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeCheckpoint` — test-source-only
  command-line checkpoint in the same package so it can exercise the public surface while
  remaining outside production and ordinary JUnit discovery.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [JUnit glossary](../../../../glossary.md#term-junit))</span>

Tests remain in the production package because they inject exact fake handles through the
existing package-private [binding](../../../../glossary.md#term-binding) seam and lock the helper's visibility and [shape](../../../../glossary.md#shape).

## Exact public and package-private contracts

Add exactly these public methods to [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary):

```java
public int threadCount();

public void setThreadCount(int threadCount);
```

`threadCount()` reports the positive count returned by the already-bound
`openblas_get_num_threads` handle. `setThreadCount(int)` passes one validated positive 32-bit C
`int` unchanged to the already-bound `openblas_set_num_threads` handle. Neither method caches,
normalizes, caps, derives, or otherwise interprets a value.

Do not add a result/[lease](../../../../glossary.md#term-lease) type, overload, static convenience, default, availability query,
previous-value return, [callback](../../../../glossary.md#term-callback), lock [API](../../../../glossary.md#term-application-programming-interface-api), public exception, or separate provider [facade](../../../../glossary.md#term-facade).

`OpenBlasThreadControl` is one package-private final type with no instance fields, public member,
or [mutable](../../../../glossary.md#term-mutable) static field. Its exact package-private operations are:

```java
static int threadCount(OpenBlasNativeBindings bindings);

static void setThreadCount(OpenBlasNativeBindings bindings, int threadCount);
```

The helper accepts only the already-checked exact [bindings](../../../../glossary.md#term-binding) and the primitive setter input. It is
not a test seam, policy abstraction, global coordinator, or generic native utility.

## [Validation](../../../../glossary.md#term-validation), invocation, and failure contracts

### Query order

1. `OpenBlasLibrary.threadCount()` acquires [bindings](../../../../glossary.md#term-binding) through the existing `bindings()` operation.
   A closed owner fails first with the existing exact<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [owner glossary](../../../../glossary.md#term-owner))</span>
   `IllegalStateException("OpenBLAS library is closed")`.
2. The helper obtains the exact `getNumThreads()` handle and invokes it with typed
   `(int) handle.invokeExact()`.
3. If the native return is less than or equal to zero, fail with
   `IllegalStateException("OpenBLAS returned non-positive thread count: " + result)`.
4. Otherwise return the exact positive result.

### Setter order

1. `OpenBlasLibrary.setThreadCount(...)` acquires [bindings](../../../../glossary.md#term-binding) through the existing `bindings()`
   operation. A closed owner therefore fails before argument validation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [owner glossary](../../../../glossary.md#term-owner); [argument glossary](../../../../glossary.md#term-argument))</span>
2. Reject `threadCount <= 0` with exact
   `IllegalArgumentException("threadCount must be positive: " + threadCount)`.
3. Obtain the exact `setNumThreads()` handle and invoke it with typed
   `handle.invokeExact(threadCount)`.
4. Return normally without querying, caching, or claiming the effective count. A caller that
   needs confirmation performs a separate `threadCount()` call and accepts the shared-state race
   between those operations.<span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race))</span>

For each exact native invocation, catch `Error` first and rethrow it unchanged. Catch every other
`Throwable` and wrap it with the original cause in an `IllegalStateException` whose exact message
is:

- `OpenBLAS get thread count invocation failed`; or
- `OpenBLAS set thread count invocation failed`.

The non-positive getter result is checked after successful invocation and is not wrapped as an
invocation throwable. Setter input [validation](../../../../glossary.md#term-validation) occurs outside the invocation catch. No new public
failure type is added.

## Shared state, [lifecycle](../../../../glossary.md#lifecycle), [concurrency](../../../../glossary.md#term-concurrency), and restoration

- The thread count is [mutable](../../../../glossary.md#term-mutable) [OpenBLAS](../../../../glossary.md#term-openblas) library/process state. It is not a field of
  [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), a per-handle preference, a per-GEMM argument, a per-run resource, or an
  immutable prepared decision.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [argument glossary](../../../../glossary.md#term-argument); [run glossary](../../../../glossary.md#term-run); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- As Synaptik's conservative integration rule, multiple Java owners of the same loaded OpenBLAS
  binary must be treated as potentially observing and mutating the same state. A successful setter
  through one may be visible through another. The provider does not claim coordination or shared
  state across independently loaded binary copies, loader namespaces, or arbitrary native users.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Concurrent getters and setters are permitted as direct native calls, but the provider supplies
  no deterministic winner or ordering beyond each Java invocation boundary. A later observation
  may reflect any competing native mutation; a read-then-write or write-then-read sequence is not
  atomic.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- The provider does not serialize thread control with GEMM. Changing the count while any OpenBLAS
  call is active has no provider guarantee and must be prevented by the coordinating caller.
- `close()` remains local lifetime cleanup. It does not reset, restore, or otherwise mutate the
  OpenBLAS thread count, because another owner or native consumer may still use the library.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [owner glossary](../../../../glossary.md#term-owner))</span>
- Closing one Java owner does not make shared state private to another owner and does not promise
  physical library unload.
- A caller must not [race](../../../../glossary.md#term-race) `close()` with a thread query, setter, or GEMM call on the same owner.
  The operation may fail at the stable owner-open check or through the JDK/native lifetime. No
  successful-call guarantee, active-call tracking, or new synchronization is added.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Restoration is caller-owned coordination: capture a positive count, prevent concurrent users,
  apply temporary changes, and restore in <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>`finally` through a still-open owner. The provider
  stores no previous value and cannot guarantee restoration against independent native writers.
- Future CPU prepare/configuration owns the chosen thread count and when it is safe to install or
  restore that choice. This task does not make thread control part of Runtime hot-path discovery
  or per-invocation route selection.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Explicit real-native [checkpoint](../../../../glossary.md#term-checkpoint) contract

### Preconditions

- A Java 26 [runtime](../../../../glossary.md#term-runtime) matching the repository [toolchain](../../../../glossary.md#term-toolchain) is available.
- The caller supplies one absolute path to a compatible shared OpenBLAS binary. The binary must
  load on the current operating-system ABI and export the four task-0001 symbols under the
  ordinary 32-bit-<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>`blasint` interface.
- The command runs in an isolated process with no concurrent OpenBLAS consumer and with enough
  permission to load the supplied file.
- The caller, CI image, or machine administrator installs/provides the binary. Repository code
  does not locate, download, unpack, build, package, or redistribute it.<span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>
- Native access is explicitly enabled for `ALL-UNNAMED`; `--illegal-native-access=deny` ensures
  that an accidental unenabled restricted operation fails instead of producing only a warning.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

### [Checkpoint](../../../../glossary.md#term-checkpoint) launcher

`OpenBlasNativeCheckpoint.main(String[] args)` is test source, not a [JUnit](../../../../glossary.md#term-junit) test and not production
[API](../../../../glossary.md#term-application-programming-interface-api). It accepts exactly one [argument](../../../../glossary.md#term-argument), requires it to form an absolute `Path`, and passes it
directly to `OpenBlasLibrary.open(Path)`. It reads no environment variable or system property and
performs no alternate [lookup](../../../../glossary.md#term-lookup).

Argument [validation](../../../../glossary.md#term-validation) occurs before native access. A count other than one fails with
`IllegalArgumentException("expected exactly one absolute OpenBLAS library path argument")`.
After `Path.of(args[0])`, a non-absolute result fails with
`IllegalArgumentException("OpenBLAS library path must be absolute: " + path)`. An invalid path
syntax retains the [JDK](../../../../glossary.md#term-java-development-kit-jdk) `InvalidPathException` and its diagnostics.

The launcher must:

1. open two [owners](../../../../glossary.md#term-owner) against the exact supplied path;
2. query a positive original thread count;
3. set the count to `1` through the first [owner](../../../../glossary.md#term-owner);
4. require both owners to report `1`, proving shared observation for this supplied [binary](../../../../glossary.md#term-binary-descriptor-and-metadata);
5. [run](../../../../glossary.md#term-run) the exact [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) and [DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) cases below while the count is `1`;
6. in `finally`, restore the original positive count through a still-open owner and verify it
   through both owners before either closes; and
7. close [matrix](../../../../glossary.md#term-matrix-meanings) [arenas](../../../../glossary.md#term-arena) and both library owners on every path.

If checkpoint work fails and restoration also fails, retain the original throwable and suppress
the distinct restoration throwable. If only restoration or verification fails, the checkpoint
fails. A repeated identical throwable must not be self-suppressed. The launcher prints one concise
success line including the restored count and otherwise exits nonzero through the uncaught
failure.

### Native numerical cases

Both precisions use this dense [row-major](../../../../glossary.md#term-row-major) case:

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

The [checkpoint](../../../../glossary.md#term-checkpoint) allocates distinct native A, B, and C segments with the [precision](../../../../glossary.md#term-precision)'s natural
[alignment](../../../../glossary.md#term-alignment), initializes them explicitly, invokes the existing public `sgemm` or `dgemm`, and
checks every output [element](../../../../glossary.md#term-element). It accepts a value when:

```text
abs(actual - expected) <= absoluteTolerance
    + relativeTolerance * abs(expected)
```

The exact tolerances are:

| Call | Absolute [tolerance](../../../../glossary.md#term-tolerance) | Relative tolerance |
|---|---:|---:|
| [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) | `1.0e-4f` | `1.0e-5f` |
| [DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) | `1.0e-12` | `1.0e-12` |

These small [finite](../../../../glossary.md#term-finite) cases use exactly representable inputs and expected values. The nonzero
tolerances avoid turning the checkpoint into a claim about one implementation's accumulation or
instruction sequence while remaining tight enough to catch wrong [ABI](../../../../glossary.md#term-application-binary-interface-abi) carriers, [layout](../../../../glossary.md#layout),
[transpose](../../../../glossary.md#term-transpose), leading [dimensions](../../../../glossary.md#dimension), alpha/beta forwarding, or output placement. No other case is
covered by this checkpoint, and the existing task-0002 [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) input contract is not narrowed by this
selected evidence case.

### Commands

Ordinary native-free [validation](../../../../glossary.md#term-validation) first:

```bash
./gradlew :backends:openblas-provider:test \
  :backends:openblas-provider:testClasses
```

Then launch the explicitly supplied [checkpoint](../../../../glossary.md#term-checkpoint) from the repository root. `<JDK26>` and
`<ABSOLUTE_OPENBLAS_LIBRARY>` are caller substitutions, not production [lookup](../../../../glossary.md#term-lookup) mechanisms:

```bash
<JDK26>/bin/java \
  --enable-native-access=ALL-UNNAMED \
  --illegal-native-access=deny \
  -cp backends/openblas-provider/build/classes/java/main:backends/openblas-provider/build/classes/java/test \
  io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_LIBRARY>
```

On Windows, use `;` rather than `:` between the two class-path entries and pass an absolute path
to a compatible [DLL](../../../../glossary.md#term-dll). Do not replace the [argument](../../../../glossary.md#term-argument) with discovery, an environment variable, or a
system property in production or ordinary tests.

## Affected files

Production:

- modify `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasThreadControl.java`
- modify `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/package-info.java`

Tests and [checkpoint](../../../../glossary.md#term-checkpoint):

- modify `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryPublicShapeTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasThreadControlTest.java`
- modify `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasAbiContractTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasNativeCheckpoint.java`

Explanatory documentation:

- modify `docs/backend-guide/cpu-backend.md`
- modify `docs/glossary.md` only if the final public/shared-state distinction changes reusable
  terminology; otherwise record the reasoned no-change conclusion

[Planning](../../../../glossary.md#term-planning):

- this task
- modify `docs/planning/backends/openblas-provider/master-plan.md`
- modify `docs/planning/roadmap.md`

`FfmOpenBlasNativeAccess`, `OpenBlasNativeAccess`, `OpenBlasNativeBindings`,
`OpenBlasGemmInvocation`, `OpenBlasLoadException`, completed tasks 0001/0002, provider/[CPU](../../../../glossary.md#term-cpu) build
files, root/settings files, architecture documents/tests, [conformance](../../../../glossary.md#term-conformance)/integration projects, CPU
source/tests, and generated output are review-only and must remain byte-for-byte unchanged.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 12 paths:

- 3 provider production paths;
- 4 provider test/[checkpoint](../../../../glossary.md#term-checkpoint) paths;
- at most 2 explanatory documentation paths; and
- 3 [planning](../../../../glossary.md#term-planning) paths.

If the glossary needs no edit, the final implementation scope is at most 11 paths. No completed
task byte may change. If implementation needs [Gradle](../../../../glossary.md#term-gradle) configuration, another public member/type,
another native [symbol](../../../../glossary.md#term-symbol) or seam, another file, or more than 12 paths, stop and propose a bounded
planning correction or follow-up.

## Acceptance criteria

- [x] [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) exposes exactly the two specified additional public methods and no new
      public type, overload, lease, result, previous-value, callback, or failure surface.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [callback glossary](../../../../glossary.md#term-callback))</span>
- [x] Both methods perform the existing owner-open check first and delegate to one package-private
      final field-free helper over the existing handles.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- [x] Setter validation rejects every non-positive value with the exact specified type/message
      before native invocation, but after the owner-open check.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [x] Getter returns an exact positive native value and rejects a non-positive result with the
      exact specified `IllegalStateException` message.
- [x] The getter and setter use exact typed `invokeExact`, rethrow every `Error` unchanged, and
      wrap every other invocation throwable with the exact operation-specific message and cause.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- [x] Deterministic fake tests prove exact call counts/values, validation order, invalid getter
      result handling, checked/runtime/wrong-type/Error behavior, closed-owner precedence,
      multiple-owner shared-handle observation, and the unsupported close-race boundary without
      requiring an installed library.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [race glossary](../../../../glossary.md#term-race))</span>
- [x] Java owners retain only local [lookup](../../../../glossary.md#term-lookup) [lifetime](../../../../glossary.md#term-lifetime). No prior/current thread count is stored, and
      `close()` performs no native query, setter, or restoration.
- [x] Javadocs and package documentation describe Synaptik's conservative library/process-global
      treatment, potentially shared mutation across owners, no deterministic winner, non-atomic
      read/write sequences, GEMM coordination, caller-owned restoration, and close races without
      claiming coordination across independently loaded copies or arbitrary native consumers.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- [x] Ordinary tests perform no native load/invocation, use no skip/assumption for a missing
      library, and require no native-access launch option.
- [x] The test-source checkpoint accepts exactly one explicit absolute path and contains no name
      selection, platform probing, property/environment lookup, download, packaging, fallback, or
      alternate candidate.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [x] With a supplied compatible library and the exact native-access launch flags, the checkpoint
      verifies positive query, cross-owner shared observation at count `1`, the exact SGEMM/DGEMM
      values within the specified tolerances, and restoration/verification of the original count
      in <span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>`finally`.
- [x] Restoration failure preserves any earlier distinct primary failure and never attempts Java
      self-suppression; both owners and all checkpoint arenas close on every path.<span data-glossary-navigation="true"> ([arenas glossary](../../../../glossary.md#term-arena))</span>
- [x] Production remains JDK-only with zero project dependencies and contains no CPU, config,
      route, fallback, tuning, Tensor, Prepare, Runtime, discovery, or global-coordination logic.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [CPU glossary](../../../../glossary.md#term-cpu); [Tensor glossary](../../../../glossary.md#tensor); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [x] Existing task-0001 loading/binding and task-0002 GEMM contracts/tests remain unchanged except
      for the narrowly authorized public-shape and ABI-test extensions.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [shape glossary](../../../../glossary.md#shape); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- [x] CPU code, Gradle/root/settings, architecture files/tests, backend conformance, integration,
      native packaging, completed task files, and generated outputs remain unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [x] A separate clean documentation-focused agent finalizes Javadocs, CPU guide, glossary impact,
      task/master/roadmap evidence, and documentation validation in the same overall change.
- [x] Task 0003 remained [`Ready`](../../../../glossary.md#term-ready) until implementation, final module validation, supplied native
      checkpoint, capability checkpoint, and clean documentation passes all succeeded; task,
      master plan, and roadmap then moved to Complete and CPU became the next Draft planning
      frontier.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [planning glossary](../../../../glossary.md#term-planning))</span>
- [x] Exact scope, package placement, public/package-private surface, links, anchors, fences,
      final newlines, trailing whitespace, status/history, no-later-spec, and final diff checks
      pass.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development tests must cover:

- exact public methods and package-private helper visibility/stateless [shape](../../../../glossary.md#shape);
- exact existing getter/setter handle types and exact invocation carriers;
- closed-[owner](../../../../glossary.md#term-owner) precedence over setter validation;
- `Integer.MIN_VALUE`, `-1`, and `0` rejection plus positive boundary/value forwarding;
- positive getter returns and zero/negative native-result rejection;
- checked throwable, [runtime](../../../../glossary.md#term-runtime) throwable, wrong method type, and unchanged `Error` behavior for
  both calls;
- sequential and concurrent fake calls with deterministic coordination, including two owners
  backed by one fake global thread-count cell;
- absence of cached per-owner count, automatic close restoration, synchronization, another
  native seam/symbol, project dependency, and forbidden vocabulary; and<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [symbol glossary](../../../../glossary.md#term-symbol); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- checkpoint argument validation, exact matrix data/expected values/tolerances, restoration
  suppression logic, explicit path forwarding, and absence from ordinary JUnit discovery.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [JUnit glossary](../../../../glossary.md#term-junit))</span>

After executable Java stabilizes, the implementation context runs exactly one final ordinary
affected-[module](../../../../glossary.md#term-module) command:

```bash
./gradlew :backends:openblas-provider:test
```

It then compiles the [checkpoint](../../../../glossary.md#term-checkpoint) if necessary and runs the exact explicit real-native command from
the checkpoint contract with the caller/[CI](../../../../glossary.md#term-continuous-integration-ci)-supplied compatible library. Missing native input is a
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
shared-state/[lifecycle](../../../../glossary.md#lifecycle)/restoration wording, zero project dependencies, exact path maximum,
unchanged completed tasks, synchronized [Ready](../../../../glossary.md#term-ready)-to-Complete status history, no later provider or [CPU](../../../../glossary.md#term-cpu)
task specification, direct external links, anchors, fences, final newlines, and trailing
whitespace.

## Dependencies

- [OpenBLAS provider 0001](0001-library-loading-and-required-symbol-binding.md)<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span> — Complete; supplies
  the explicit library lifetime and exact already-bound thread/GEMM handles.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- [OpenBLAS provider 0002](0002-float32-float64-row-major-gemm-invocation.md) — Complete; supplies
  the exact public dense row-major SGEMM/DGEMM calls checked by the native milestone.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- Prepare 0003, Runtime 0014, Compiler 0006, Planning 0006, and Backend Contract 0004 — Complete;
  establish downstream boundaries without becoming provider dependencies.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Backend glossary](../../../../glossary.md#backend))</span>
- [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 [FFM](../../../../glossary.md#term-ffm) and restricted-method launch contracts — current root [toolchain](../../../../glossary.md#term-toolchain) baseline.
- One caller/CI-supplied compatible OpenBLAS shared library — required only to complete the
  explicit native checkpoint, not ordinary unit tests or production discovery.<span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

No [CPU](../../../../glossary.md#term-cpu) implementation task may begin before this provider task and [checkpoint](../../../../glossary.md#term-checkpoint) are Complete.

## Follow-up tasks

- CPU backend 0001 remains a Draft row only. A later separate planning step may make it Ready
  after this task closes the provider milestone. CPU will own capability truth, thread-count
  selection/coordination, route/fallback policy, normalization, storage, preparation, and
  execution.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning); [Ready glossary](../../../../glossary.md#term-ready); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- No later OpenBLAS provider task is planned by this milestone. A new symbol, precision, layout,
  or lifecycle capability requires a separately justified future master-plan row and task.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [symbol glossary](../../../../glossary.md#term-symbol); [precision glossary](../../../../glossary.md#term-precision); [layout glossary](../../../../glossary.md#layout); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

Do not create a [CPU](../../../../glossary.md#term-cpu) task file or another provider task file in this change.

## Architecture impact

Expected impact: None.

This task implements thread control already assigned to the low-level provider and validates its
already planned native surface. It adds no project edge, [backend identity](../../../../glossary.md#backend-identity--backendid), route, fallback, or
shared [lifecycle](../../../../glossary.md#lifecycle) rule. If implementation requires provider-owned global coordination, config
interpretation, another native [symbol](../../../../glossary.md#term-symbol), or a [dependency](../../../../glossary.md#term-dependency) outside the [JDK](../../../../glossary.md#term-java-development-kit-jdk), stop and report the
conflict rather than editing architecture.

## Documentation impact

The separate documentation-focused pass must:

- finalize [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), `OpenBlasThreadControl`, and package Javadocs for exact thread-count
  values, validation, failures, the conservative shared-state inference, multiple owners,
  lifecycle, concurrency, restoration, and CPU-policy boundaries;<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [inference glossary](../../../../glossary.md#term-inference); [lifecycle glossary](../../../../glossary.md#lifecycle); [concurrency glossary](../../../../glossary.md#term-concurrency); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- update the CPU backend guide to mark loading/binding, low-level GEMM, thread control, and the
  explicit native provider checkpoint current while keeping CPU capability, route, fallback,
  storage, prepare, and execution work planned;<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [binding glossary](../../../../glossary.md#term-binding); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [prepare glossary](../../../../glossary.md#prepare))</span>
- update the glossary only if the final change introduces or changes a reusable project term;
  otherwise record why the existing OpenBLAS library-handle/GEMM entries and ordinary shared-
  state language are sufficient;<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- synchronize task 0003, provider master plan, and roadmap evidence/status only after every
  implementation, native, checkpoint, and documentation gate succeeds;
- preserve architecture documentation because [ownership](../../../../glossary.md#term-ownership-meanings) and [dependency](../../../../glossary.md#term-dependency) rules do not change; and
- record reasoned no-change conclusions for Tensor, Compile, and Training APIs; Model capability
  and operation contracts; Config, Prepare, Runtime, Backend Contract, Trace, CPU implementation;
  architecture/tests; backend-conformance/integration; Gradle/root/settings; native packaging;
  completed tasks; generated outputs; and other modules.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

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

- Keep query and setter on [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary); their handles and validity are inseparable from its
  existing caller-owned lookup lifetime, and no independent public thread-control abstraction has
  a current consumer.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Use one field-free package-private helper, matching task 0002's exact-invocation boundary without
  broadening the native-access test seam.
- Accept positive 32-bit counts only. Zero and negative values cannot express a CPU thread
  candidate and would delegate undocumented normalization policy to the native library.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [normalization glossary](../../../../glossary.md#term-normalization); [native library glossary](../../../../glossary.md#term-native-library))</span>
- Reject a non-positive getter result rather than exposing an invalid count as usable provider
  state. This remains a low-level ABI sanity check, not an availability or fallback decision.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Do not return the prior value from the setter: the already-bound [OpenBLAS](../../../../glossary.md#term-openblas) setter returns `void`,
  and synthesizing a return with a preceding getter would falsely imply one atomic operation.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [operation glossary](../../../../glossary.md#operation))</span>
- Do not add a scoped restoration object or close-time reset. Java lookup owners do not own the
  shared native setting and cannot exclude independent writers.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
- Use one explicit non-JUnit test-source launcher rather than a Gradle property or environment-
  gated test. A missing library is then a visible checkpoint precondition, never a skipped
  ordinary test.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit); [Gradle glossary](../../../../glossary.md#term-gradle); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Verify two owners and restore in an isolated process because that is the smallest honest check
  of the shared-state contract without claiming coordination with arbitrary native consumers.
- Use one nontrivial alpha/beta row-major case per precision. It exercises both inputs, existing
  output, layout, leading dimensions, and scalar forwarding while keeping expected results
  independently reviewable.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [precision glossary](../../../../glossary.md#term-precision); [layout glossary](../../../../glossary.md#layout); [dimensions glossary](../../../../glossary.md#dimension); [scalar glossary](../../../../glossary.md#term-scalar))</span>

## Known limitations

- The provider cannot make a query/set pair atomic or coordinate another Java class loader,
  native caller, OpenMP runtime, or application component using the same OpenBLAS binary.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [runtime glossary](../../../../glossary.md#term-runtime); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- `close()` cannot safely race a thread-control or GEMM invocation and does not undo shared native
  state.<span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- The setter is not a per-call tuning control. A coordinating caller must install a choice only
  at a safe lifecycle point and restore it explicitly when required.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The task supports only the already-bound ordinary 32-bit-`blasint` getter/setter symbols. It
  does not bind local-thread, version, build-config, processor-count, or parallel-mode functions.
- The native checkpoint proves only one supplied binary, process, ABI, shared-state observation,
  and two small finite GEMM cases. It does not establish general numerical conformance,
  determinism, performance, packaging compatibility, or support for another OpenBLAS build.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [finite glossary](../../../../glossary.md#term-finite); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Completing the task requires a caller/CI-supplied compatible shared library and native-access
  permission. Ordinary development remains native-free, but the provider milestone cannot be
  marked Complete without the explicit checkpoint evidence.<span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- During development, implementation context `/root/task_0003_impl` first ran
  `./gradlew :backends:openblas-provider:test`: 48 tests ran with one failure in
  `OpenBlasThreadControlTest.rejectsNonPositiveGetterResults`. Production compiled; the test fake
  incorrectly used `MethodHandle.bindTo` with a primitive `int`. After correcting only that fake
  handle construction, the same command passed 48 tests with zero failures. Two checkpoint tests
  were then added before the final recorded module run below.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module); [run glossary](../../../../glossary.md#term-run))</span>
- Implementation context `/root/task_0003_impl` ran the final ordinary command
  `./gradlew :backends:openblas-provider:test`: passed 5 suites and 50 tests with zero skips,
  failures, or errors (`OpenBlasAbiContractTest` 9, `OpenBlasGemmInvocationTest` 17,
  `OpenBlasLibraryPublicShapeTest` 5, `OpenBlasLibraryTest` 9, and
  `OpenBlasThreadControlTest` 10). Clean documentation context
  `/root/task_0003_impl/openblas_0003_docs` reused this evidence because it changed only Javadoc
  and Markdown; executable Java and tests did not change afterward.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Clean validation context `/root/task_0003_native_resume` ran `brew --prefix openblas` exactly
  once and obtained `/opt/homebrew/opt/openblas`. From that prefix it derived the Darwin arm64
  candidate <span data-glossary-navigation="true"> ([arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>`/opt/homebrew/opt/openblas/lib/libopenblas.dylib`, resolved the real regular-file
  target <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`/opt/homebrew/Cellar/openblas/0.3.33/lib/libopenblasp-r0.3.33.dylib`, and supplied that
  target as the checkpoint's sole argument. The target was absolute, <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`stat` reported Regular File,
  mode `-r--r--r--`, and size 22,337,424 bytes, `file` reported a Mach-O 64-bit dynamically linked
  shared library for arm64, and <span data-glossary-navigation="true"> ([Mach-O glossary](../../../../glossary.md#term-mach-o))</span>`nm -gU` found `_cblas_sgemm`, `_cblas_dgemm`,
  `_openblas_get_num_threads`, and `_openblas_set_num_threads`.
- Clean validation context `/root/task_0003_native_resume` used
  `/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home/bin/java`, Java
  26.0.1 arm64, and ran
  `/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home/bin/java --enable-native-access=ALL-UNNAMED --illegal-native-access=deny -cp backends/openblas-provider/build/classes/java/main:backends/openblas-provider/build/classes/java/test io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeCheckpoint /opt/homebrew/Cellar/openblas/0.3.33/lib/libopenblasp-r0.3.33.dylib`.
  It exited 0 with `OpenBLAS native checkpoint passed; restored thread count 16`, covering positive
  query, shared observation through two owners, the exact SGEMM/DGEMM cases, and restoration plus
  cross-owner verification in <span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [owner glossary](../../../../glossary.md#term-owner))</span>`finally`.
- Only after the native command succeeded, clean validation context
  `/root/task_0003_native_resume` ran `./gradlew test :testing:architecture-tests:test`: passed
  with `BUILD SUCCESSFUL` in 768 ms and 54 actionable tasks (2 executed, 52 up-to-date). The
  provider test task executed and the architecture-test task was up-to-date. No executable Java
  changed afterward, so this remains the final capability-checkpoint evidence.
- Documentation context `/root/task_0003_impl/openblas_0003_docs` ran
  `./gradlew :backends:openblas-provider:javadoc`: passed. Generated package and
  [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) pages were inspected for the exact two-method addition, positive-value and
  failure contracts, conservative shared-state wording, multiple-owner limits, non-atomic and
  no-winner concurrency, GEMM coordination, caller-owned restoration, local close lifetime, and
  close-race boundary. <span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [lifetime glossary](../../../../glossary.md#term-lifetime); [race glossary](../../../../glossary.md#term-race))</span>`OpenBlasThreadControl` remains package-private and therefore is not a
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
  OpenBLAS-provider or CPU task specification.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
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
  Completed tasks 0001 and 0002 remained byte-for-byte unchanged with SHA-256 values<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>
  `4e058be28272b85937b199cffc37c77396f80b99ae8ad84d939af937d13e0d54` and
  `1fdb5fa6d552ec869b3be0c41e9acc6d61f7bbf4c9af13f02b79f55abe9b166b`, respectively.
- Completion-documentation context
  `/root/task_0003_native_resume/openblas_0003_completion_docs` ran `git diff --check` after the
  planning-only synchronization: passed with no output. It reran the three-file Markdown and
  whitespace checks after recording this evidence; both remained successful.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
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
  the earlier clean documentation pass had already finalized those contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Implementation notes

- Added exactly `threadCount()` and `setThreadCount(int)` to the existing public lifetime owner.
  Both acquire the checked bindings first and delegate to one package-private final field-free
  helper using the already-bound exact handles.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [owner glossary](../../../../glossary.md#term-owner); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Added deterministic fake-handle coverage for value forwarding, validation order, native-result
  checking, throwable translation, multiple-owner observation, concurrency boundaries, no close
  restoration, and checkpoint argument/suppression behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [concurrency glossary](../../../../glossary.md#term-concurrency); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [argument glossary](../../../../glossary.md#term-argument))</span>
- Added the test-source argument-driven native checkpoint with explicit restoration and the fixed
  SGEMM/DGEMM cases. It performs no discovery and is not an ordinary JUnit test.<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [JUnit glossary](../../../../glossary.md#term-junit))</span>
- Finalized the affected public/helper/package Javadocs, CPU guide, glossary entry, and planning
  records without changing executable Java or tests in the documentation pass.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: implemented the exact direct positive thread-count query/control surface,
  package-private invocation helper, deterministic tests, and explicit native checkpoint launcher;
  finalized all affected documentation and planning evidence.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly twelve authorized paths—three provider production paths, four
  provider test/checkpoint paths, the CPU guide, glossary, this task, provider master plan, and
  roadmap.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tests and validation: reused the implementation context's passing 5-suite/50-test provider run,
  the clean native checkpoint that restored thread count 16, and the later passing repository/
  architecture checkpoint with 54 actionable tasks; the documentation contexts passed provider
  Javadoc, Markdown, generated-page, surface/scope/status/history/later-specification,
  completed-task, and whitespace checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `/root/task_0003_impl/openblas_0003_docs`
  independently finalized Javadocs, package documentation, CPU guide, glossary impact, and
  planning evidence. Clean completion context
  `/root/task_0003_native_resume/openblas_0003_completion_docs` independently reviewed the final
  twelve-path change and synchronized only the three planning paths after the executable gates.
  Neither context changed executable Java or duplicated successful Java-test execution.
- Documentation impact: the CPU guide now teaches direct thread control and its coordination,
  restoration, close, and checkpoint boundaries; task, master plan, and roadmap now synchronize
  the Complete provider milestone and next Draft CPU planning frontier.
- [Javadoc](../../../../glossary.md#term-javadoc) review: [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), `OpenBlasThreadControl`, and package documentation cover exact
  values/failures, conservative potential sharing, explicit exclusions, concurrency, GEMM
  coordination, caller-owned restoration, local lifetime, and close races.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Glossary impact: updated the reusable `OpenBlasLibrary` entry because its prior statement that
  thread-control invocation was absent became stale; no new reusable term was introduced.
- Unresolved issues: None for task 0003 or the selected [OpenBLAS](../../../../glossary.md#term-openblas) provider milestone.
- Follow-up required: None for this task. CPU is the next Draft planning frontier; a separate
  planning step may create the first detailed CPU task without extending this change.

Status: Complete
