# Task 0002: FLOAT32/FLOAT64 Row-Major GEMM Invocation

## Status

Complete

## Goal

Add the smallest usable general matrix multiplication (GEMM) invocation surface to the existing
caller-owned `OpenBlasLibrary`: validated FLOAT32 and FLOAT64 calls for dense row-major,
non-transposed `A[m,k]`, `B[k,n]`, and `C[m,n]` native memory segments.

The provider forwards the exact caller-supplied scalars to the already-bound OpenBLAS CBLAS
symbols and mutates only the logical `C` output. It does not select a CPU route, allocate or copy
matrix storage, interpret Tensor metadata, or promise numerical results.

## Rationale and mental model

```text
caller-owned open OpenBlasLibrary
  -> validate m, n, k and caller-owned native A/B/C segments
  -> derive fixed row-major/no-transpose CBLAS arguments
  -> invoke the already-bound cblas_sgemm or cblas_dgemm handle exactly
  -> caller still owns A, B, C, and the library lifetime

future CPU prepare/execution
  -> owns batching, broadcasting, transposition, layout conversion, packing, and loops
  -> calls this provider only for one already-normalized dense GEMM
```

Dense row-major no-transpose GEMM is the concrete first native building block for current MATMUL,
linear, scaled dot-product attention, and convolution lowering. A broader provider surface would
move CPU-owned normalization and route policy into the leaf. A separate public invocation
abstraction has no current consumer and would separate calls from the `OpenBlasLibrary` lifetime
they require, so the two methods belong directly on that public final owner.

## Scope

- Add exactly two public `void` methods to `OpenBlasLibrary`, one for FLOAT32 SGEMM and one for
  FLOAT64 DGEMM.
- Support only dense row-major, non-transposed `A[m,k]`, `B[k,n]`, and `C[m,n]` matrices.
- Use the standard 32-bit-`blasint` C ABI already selected and bound by task 0001.
- Fix `CblasRowMajor = 101` and `CblasNoTrans = 111` for both operands.
- Derive `lda = max(1, k)`, `ldb = max(1, n)`, and `ldc = max(1, n)` internally.
- Validate library lifetime, dimensions, segment references and properties, alignment, required
  spans, and output/input overlap in the exact order and with the stable messages below.
- Preserve every raw FLOAT32/FLOAT64 `alpha` and `beta` value unchanged, including NaN payloads
  where the Java carrier retains them, infinities, and both signed zeros.
- Invoke the existing exact typed `OpenBlasNativeBindings.sgemm()` or `dgemm()` handle with
  `invokeExact` through one field-free package-private helper.
- Translate every non-`Error` throwable from the native-handle invocation into one stable
  `IllegalStateException` while retaining the original cause; rethrow every `Error` unchanged.
- Add deterministic tests using exact typed fake method handles; ordinary tests must not require
  or load an installed OpenBLAS library.
- Finalize affected Javadocs, package documentation, CPU guide, glossary, and planning records in
  the required separate clean documentation pass.

## Out of scope

- offsets, strides, caller-supplied leading dimensions, transpose flags, column-major input, or
  non-dense matrix views
- batched GEMM, broadcasting, vector/matrix rank adaptation, grouping, tiling, packing, layout
  conversion, transpose materialization, or provider-owned loops
- FLOAT16, BFLOAT16, complex, integral, sparse, quantized, or mixed-precision GEMM
- a result object, allocation, copy, storage ownership, segment closure, or retained segment view
- accepting heap segments or automatically copying them to native memory
- CPU capability reporting, lowering, route choice, fallback, workload classification, tuning,
  cache behavior, thread-count choice, or thread-control invocation
- Tensor, Model, Compiler, Planning, Prepare, Runtime, Engine, Trace, Backend Contract, Config, or
  concrete CPU implementation changes
- numerical tolerance, accumulation, reassociation, fused-operation, NaN propagation, signed-zero,
  determinism, or cross-platform result guarantees for OpenBLAS
- a new public failure type, public invocation abstraction, callback, registry, manager, service,
  generic native seam, or exposed `MethodHandle`
- synchronization, active-call accounting, reference counting, close/call serialization, or a
  successful-call guarantee when `close()` races an invocation
- OpenBLAS thread control, compatible installed-library validation, native access launch policy,
  or the provider capability checkpoint retained by task 0003
- Gradle, native packaging, dependency, architecture-contract, ADR, architecture-test,
  backend-conformance, integration-test, or CPU task changes
- a detailed task-0003 specification

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
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)
- [OpenBLAS provider master plan](../master-plan.md)
- [Task 0001: Library loading and required symbol binding](0001-library-loading-and-required-symbol-binding.md)

Primary external contracts:

- [JDK 26 `MemorySegment`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/MemorySegment.html)
  defines native versus heap segments, physical native addresses, byte sizes, scopes,
  current-thread accessibility, read-only state, and actual backing-region overlap.
- [JDK 26 `Linker`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/Linker.html)
  defines exact downcall method handles and `MemorySegment` address arguments.
- [JDK 26 `MethodHandle`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/invoke/MethodHandle.html)
  requires the `invokeExact` call-site descriptor to match the handle's method type exactly and
  specifies that any throwable from the underlying operation propagates through the handle call.
- [OpenBLAS `cblas.h` at the reviewed revision](https://github.com/OpenMathLib/OpenBLAS/blob/c3db185d6c779e5ea222c3ff550bf0082d7e0c09/cblas.h)
  defines `CblasRowMajor = 101`, `CblasNoTrans = 111`, and the ordered SGEMM/DGEMM C signatures.
  The implementation remains constrained to task 0001's ordinary 32-bit-`blasint` ABI; the pinned
  source is durable ABI evidence rather than a minimum-version promise.
- The official [OpenBLAS user manual's CBLAS interface example](https://www.openmathlib.org/OpenBLAS/docs/user_manual/#call-cblas-interface)
  confirms that C callers include `cblas.h` and invoke `cblas_dgemm`. It is usage evidence only
  and does not establish a minimum compatible OpenBLAS version or supersede the pinned header.

## Architecture constraints

- `backends/openblas-provider` remains the low-level native-interoperability leaf assigned to
  library loading, symbol binding, GEMM calls, and thread control.
- The dependency direction remains `backends/cpu -> backends/openblas-provider`; the provider
  gains no project dependency and imports no CPU or lifecycle-layer type.
- Planning selects CPU ownership. Future CPU prepare owns route selection and every normalization
  step needed to present one dense row-major no-transpose invocation to this provider.
- `OpenBlasLibrary` remains the caller-owned native lookup lifetime. GEMM methods check and consume
  its package-private bindings; no handle or arena crosses the public boundary.
- Caller segments remain caller-owned. The provider retains, allocates, copies, reinterprets, and
  closes no caller memory, and it mutates only the logical `C` output through the native call.
- Ordinary unit tests use the existing package-private `OpenBlasNativeAccess` and
  `OpenBlasNativeBindings` path with exact typed fake method handles. Do not add another seam.
- The implementation introduces no synchronization or mutable call state on the invocation hot
  path.
- `legacy/pre-rewrite` is capability evidence only. Do not copy its API, source structure,
  lifecycle coupling, fallback, discovery, or execution policy.
- If implementation needs a new module edge, CPU behavior, an independent public abstraction,
  another native symbol, or lifecycle coordination beyond the recorded close-race boundary, stop
  and request a planning or architecture decision.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.provider.openblas` — extends the deliberate public lifetime
  owner with two inseparable low-level calls and adds one package-private stateless invocation
  helper beside the existing bindings.

No package is added.

Type placement:

- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary` — remains the public final
  owner because every invocation requires and is bounded by this caller-owned open lifetime.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasGemmInvocation` — package-private
  final, field-free helper that owns validation, derived CBLAS arguments, exact invocation, and
  invocation-failure translation without becoming a public abstraction or test seam.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeBindings` — unchanged; its
  existing package-private exact SGEMM/DGEMM accessors already supply the required handles.

Tests remain in the production package because they must inject exact handles through the existing
package-private loading/binding boundary and lock the helper's visibility and stateless shape.

## Exact public and package-private API intent

Add exactly these public methods to the existing public final `OpenBlasLibrary`:

```java
public void sgemm(
        int m,
        int n,
        int k,
        float alpha,
        MemorySegment a,
        MemorySegment b,
        float beta,
        MemorySegment c);

public void dgemm(
        int m,
        int n,
        int k,
        double alpha,
        MemorySegment a,
        MemorySegment b,
        double beta,
        MemorySegment c);
```

Both methods first call the existing package-private `bindings()` check and then delegate to
`OpenBlasGemmInvocation`. They return normally with no result object. Do not add overloads,
offsets, transpose/layout flags, leading-dimension parameters, generic numeric dispatch, or a new
public exception.

`OpenBlasGemmInvocation` is one package-private final type with no instance fields or externally
mutable state. Static methods may be used if they keep construction and surface minimal. It must
not expose or accept a native-access abstraction; it accepts the already-checked exact bindings
and invocation inputs.

## GEMM semantics and fixed CBLAS mapping

For both precisions, the logical operation is:

```text
C[m,n] <- alpha * (A[m,k] x B[k,n]) + beta * C[m,n]
```

All matrices are dense row-major and non-transposed from byte offset zero of their supplied
segments. The native argument order is exactly:

| Position | CBLAS argument | Value |
|---:|---|---|
| 1 | `Order` | `CblasRowMajor` (`101`) |
| 2 | `TransA` | `CblasNoTrans` (`111`) |
| 3 | `TransB` | `CblasNoTrans` (`111`) |
| 4 | `M` | caller `m` |
| 5 | `N` | caller `n` |
| 6 | `K` | caller `k` |
| 7 | `alpha` | caller raw FLOAT32/FLOAT64 value unchanged |
| 8 | `A` | caller segment `a` |
| 9 | `lda` | `max(1, k)` |
| 10 | `B` | caller segment `b` |
| 11 | `ldb` | `max(1, n)` |
| 12 | `beta` | caller raw FLOAT32/FLOAT64 value unchanged |
| 13 | `C` | caller segment `c` |
| 14 | `ldc` | `max(1, n)` |

The helper uses exact typed `invokeExact`; it must not use `invoke`, `invokeWithArguments`,
reflection, arrays, boxing, or string dispatch for the downcall.

## Validation order, types, and stable messages

Validation is observable and must occur in this exact order for both methods:

1. `OpenBlasLibrary` acquires bindings through `bindings()`. A closed owner fails with the
   existing `IllegalStateException("OpenBLAS library is closed")` before any argument validation.
2. Validate `m`, `n`, and `k` in that order. Each is the standard LP64 32-bit `blasint` carrier
   and must be non-negative. Failure is `IllegalArgumentException` with exactly
   `<name> must be non-negative: <value>`.
3. Apply `Objects.requireNonNull` to `a`, `b`, and `c` in that order with exact parameter-name
   messages `a`, `b`, and `c`.
4. Require native `a`, `b`, and `c` in that order using `MemorySegment.isNative()`. Failure is
   `IllegalArgumentException("<role> must be a native memory segment")`.
5. Require live scopes for `a`, `b`, and `c` in that order using `scope().isAlive()`. Failure is
   `IllegalStateException("<role> scope is not alive")`.
6. Require current-thread accessibility for `a`, `b`, and `c` in that order using
   `isAccessibleBy(Thread.currentThread())`. Failure is
   `IllegalStateException("<role> is not accessible by the current thread")`.
7. Require writable `c` using `!c.isReadOnly()`. Failure is
   `IllegalArgumentException("c must be writable")`. `a` and `b` may be read-only.
8. Require each base physical address in `a`, `b`, and `c` order to be aligned modulo the element
   size: four bytes for SGEMM and eight bytes for DGEMM. Failure is
   `IllegalArgumentException("<role> address must be aligned to <bytes> bytes")`.
9. Derive `lda`, `ldb`, and `ldc`, then compute checked required element and byte spans in
   `A`, `B`, and `C` order using `long`. Any `Math.addExact` or `Math.multiplyExact` failure is
   translated to `IllegalArgumentException("<role> required byte span overflows long", cause)`,
   where the cause is the original `ArithmeticException`; no incidental `Math` message escapes
   as the provider's message.
10. Compare `byteSize()` coverage in `A`, `B`, and `C` order. Failure is
    `IllegalArgumentException("<role> requires at least <required> bytes, but segment has <actual>")`.
11. Reject logical output overlap in `C`-with-`A`, then `C`-with-`B` order with exact messages
    `c must not overlap a` and `c must not overlap b`. Use `MemorySegment.asOverlappingSlice` on
    the required ranges so actual native backing-region overlap is detected for independently
    supplied segments and slices. A zero required span never overlaps. `A` and `B` may overlap
    because both are read-only inputs to the operation.

Only after every validation succeeds may zero-output handling or native invocation occur.
Validation failures occur outside the native-invocation `try` block and retain their exact types
and messages.

### Required-span formulas

For row-major storage with logical `rows`, `columns`, and derived `leadingDimension`:

```text
requiredElements = 0                                      if rows == 0 or columns == 0
requiredElements = (rows - 1) * leadingDimension + columns otherwise
requiredBytes    = requiredElements * elementBytes
```

Apply the formula as follows:

| Role | Rows | Columns | Leading dimension | Element bytes |
|---|---:|---:|---:|---:|
| SGEMM `A` | `m` | `k` | `max(1, k)` | 4 |
| SGEMM `B` | `k` | `n` | `max(1, n)` | 4 |
| SGEMM `C` | `m` | `n` | `max(1, n)` | 4 |
| DGEMM `A` | `m` | `k` | `max(1, k)` | 8 |
| DGEMM `B` | `k` | `n` | `max(1, n)` | 8 |
| DGEMM `C` | `m` | `n` | `max(1, n)` | 8 |

The implementation must use checked `long` arithmetic even though public dimensions are
non-negative `int` values. This locks the provider contract and avoids relying on incidental
limits of the current dimension carrier.

## Zero-dimension and scalar behavior

- After complete validation, if `m == 0 || n == 0`, return without invoking the native handle
  because there are no output elements. The contract still rejects read-only `c` and every other
  invalid argument so no-op calls have the same stable preconditions.
- If `k == 0` while `m > 0 && n > 0`, invoke native GEMM. Required `A` and `B` spans are zero,
  required `C` is `m * n` elements, `lda` is one, and OpenBLAS receives the exact `alpha` and
  `beta`; this permits the CBLAS operation `C <- beta * C` for the empty product term.
- Do not claim whether native code dereferences a zero-span `A` or `B`. The provider validates
  their references, native kind, scopes, accessibility, alignment, and zero-byte coverage before
  invocation.
- Accept every raw `alpha` and `beta` carrier value without finiteness checks, canonicalization,
  or special-value policy. Fake-handle tests verify raw-bit forwarding for ordinary finite values,
  infinities, both signed zeros, and NaN payloads where feasible without asserting OpenBLAS
  numerical behavior.

## Segment ownership and overlap contract

- `a`, `b`, and `c` are non-null native `MemorySegment` values whose scopes are alive and whose
  memory is accessible by the current calling thread.
- `a` and `b` may be read-only. `c` must be writable, including for calls that later return as an
  output-empty no-op.
- Base addresses must be aligned to the physical element width. Byte size must cover the exact
  checked logical row-major range beginning at offset zero.
- `c` must not overlap the required `a` or `b` range. `a` and `b` may overlap each other.
- Temporary slices returned by JDK overlap inspection are non-owning validation views only; the
  provider retains no slice and allocates, copies, reinterprets, or closes no caller memory.
- The caller owns all segment lifetimes and must keep them valid for the complete native call. The
  provider mutates only the logical `c` range through OpenBLAS and makes no promise about bytes
  outside the required ranges.

## Invocation failure and concurrency semantics

- `OpenBlasLibrary.sgemm` and `dgemm` perform the existing open-owner check before argument
  validation and delegation.
- Around only the exact `invokeExact` call, catch `Error` first and rethrow it unchanged. This
  preserves every VM-fatal, linkage, assertion, and other `Error` without swallowing or wrapping
  it.
- Then catch `Throwable` and wrap it in `IllegalStateException` with exactly
  `OpenBLAS sgemm invocation failed` or `OpenBLAS dgemm invocation failed`, retaining the original
  cause. This includes checked throwables, `RuntimeException`, and `WrongMethodTypeException`
  thrown by invocation.
- Provider validation failures occur before that `try` block and therefore retain their specified
  type, message, and overflow cause.
- No new public exception type is added.
- Concurrent calls while the owner remains open are permitted, subject to caller-managed
  nonconflicting segment access and the behavior of the supplied OpenBLAS library. The provider
  adds no call synchronization.
- A caller must not race `close()` with invocation. There is no lock, active-call counter,
  reference count, or successful-call guarantee. A race may fail at the stable owner-open check
  or later through a JDK/native lifetime failure.
- This task does not invoke or coordinate OpenBLAS thread-control functions. Different
  `OpenBlasLibrary` owners may still refer to one process-global native thread state.

## Affected files

Production:

- modify `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasGemmInvocation.java`
- modify `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/package-info.java`

Tests:

- modify `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryPublicShapeTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasGemmInvocationTest.java`
- modify `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasAbiContractTest.java`

Explanatory documentation:

- modify `docs/backend-guide/cpu-backend.md`
- modify `docs/glossary.md`

Planning:

- this task
- modify `docs/planning/backends/openblas-provider/master-plan.md`
- modify `docs/planning/roadmap.md`

`OpenBlasNativeBindings`, `OpenBlasNativeAccess`, and `FfmOpenBlasNativeAccess` are review-only:
the required exact handles, accessors, descriptors, and deterministic injection path already
exist. No source or test elsewhere is authorized.

## Maximum scope

At most 11 paths:

- 3 provider production paths;
- 3 provider test paths;
- 2 explanatory documentation paths; and
- 3 planning paths.

No task-0001 byte may change. No Gradle, root/settings, architecture contract/explanation, ADR,
architecture test, backend-conformance test, integration test, CPU source/test, provider binding/
access source, generated output, or other module path may change. If implementation requires
another public type or method, another seam or symbol, another file, or more than 11 paths, stop
and propose a bounded planning correction or follow-up.

## Acceptance criteria

- [x] `OpenBlasLibrary` exposes exactly the two specified additional public `void` methods with
      no overload, offset, transpose, leading-dimension, result, or new public-failure surface.
- [x] Each method performs the existing bindings/open check before any argument validation and
      delegates to one field-free package-private `OpenBlasGemmInvocation` helper.
- [x] SGEMM and DGEMM validate every condition in the exact recorded order with the exact stable
      messages and exception types, including stable overflow translation with cause.
- [x] The fixed constants, derived leading dimensions, exact fourteen-argument order, and
      precision-specific typed `invokeExact` calls match the existing task-0001 ABI descriptors.
- [x] Required byte spans use the checked row-major formula and validate A, B, and C coverage in
      order for ordinary, padded-by-leading-dimension, zero, and overflow cases.
- [x] Actual required native regions enforce C-with-A then C-with-B non-overlap; A/B overlap is
      accepted, disjoint slices are accepted, overlapping slices are rejected, and zero spans do
      not overlap.
- [x] `m == 0 || n == 0` returns only after complete validation and makes no native call; `k == 0`
      with positive output geometry invokes native GEMM with zero A/B spans and exact scalars.
- [x] Raw FLOAT32/FLOAT64 scalar bits are forwarded unchanged for finite values, infinities,
      signed zeros, and NaN payloads where feasible; production imposes no result promise.
- [x] Invocation rethrows every `Error` unchanged and wraps every non-`Error` throwable from
      `invokeExact` in the exact operation-specific `IllegalStateException` with cause.
- [x] Caller lifetime, ownership, read-only input, writable output, alignment, close-race, and
      concurrent-call boundaries are fully documented in Javadoc and package documentation.
- [x] Deterministic fake exact typed handles record all fourteen arguments and optionally perform
      small in-Java arithmetic only to prove Java-side call occurrence and forwarding, never
      OpenBLAS numerical correctness.
- [x] Ordinary tests never load OpenBLAS, skip on missing OpenBLAS, or require native-access JVM
      enablement.
- [x] Tests lock exact public/package-private shape, imports, zero project dependencies, excluded
      mechanisms, and the narrow update replacing the prior source assertion that no
      `invokeExact` exists.
- [x] No allocation, copy, retained slice, caller-memory closure, call synchronization,
      thread-control invocation, config, fallback, route, Tensor, Prepare, Runtime, or CPU
      behavior is added.
- [x] `OpenBlasNativeBindings`, native-access/binder sources, Gradle files, architecture files and
      tests, conformance/integration projects, CPU source, and task 0001 remain unchanged.
- [x] A separate clean documentation-focused agent finalizes Javadocs, the CPU guide, glossary,
      this task, master plan, and roadmap in the same overall change without duplicating a
      successful final Java test run.
- [x] Task 0002 remains `Ready` until all implementation, documentation, and validation criteria
      pass. Task 0003 remains a Draft row and has no detailed task file; no CPU task is Ready.
- [x] Exact scope, status/order/dependency, generated-Javadoc, Markdown link/anchor/fence,
      whitespace, and final diff checks pass.

## Tests / validation

Focused development tests may run while executable Java stabilizes. Tests must cover:

- both precision-specific public methods and exact public return/parameter shape;
- the package-private helper being final and field-free with no public surface;
- constants `101`, `111`, all fourteen ordered arguments, derived `lda`/`ldb`/`ldc`, and exact
  SGEMM/DGEMM method types;
- the complete validation order, exception classes, exact messages, and overflow causes;
- native/heap, live/closed, current-thread/inaccessible, read-only/writable, aligned/misaligned,
  undersized/exact/oversized, and overlap/disjoint segment cases;
- ordinary spans and all `m == 0`, `n == 0`, combined-zero, and positive-output `k == 0` cases;
- raw scalar-bit forwarding and exact call/no-call counts;
- invocation checked throwable, runtime throwable, wrong method type, and `Error` behavior;
- owner close-before-call and the documented unsupported close race, without a flaky race outcome
  assertion;
- caller ownership: no allocation, copy, retained slice, reinterpretation, segment close, or
  output replacement; and
- source/import/build exclusions, exact changed surface, and no installed-library access.

After executable Java stabilizes, the implementation context runs exactly one final affected-
module command:

```bash
./gradlew :backends:openblas-provider:test
```

The clean documentation-focused pass receives and reuses that exact evidence unless it changes
executable Java or records a concrete stale-evidence risk. After finalizing Javadocs and Markdown,
it runs:

```bash
./gradlew :backends:openblas-provider:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/backend-guide/cpu-backend.md \
  docs/glossary.md \
  docs/planning/backends/openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md \
  docs/planning/backends/openblas-provider/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

It must inspect generated Javadocs and manually or through stable automated assertions verify the
exact public/package-private surface, argument order/types, package placement, zero project
dependencies, exact 11-path maximum, unchanged task 0001, synchronized task/master/roadmap
statuses, task 0003 remaining Draft without a file, no CPU task, relative links, anchors, fences,
final newlines, and trailing whitespace.

Repository-wide tests, architecture tests, backend conformance, integration tests, and real-native
numerical validation are deferred to task 0003's explicitly supplied installed-library provider
capability checkpoint or continuous integration. This task changes one leaf module without a
dependency, architecture, shared-build, or cross-module executable change.

## Dependencies

- [OpenBLAS provider 0001](0001-library-loading-and-required-symbol-binding.md) — Complete; supplies
  caller-directed loading, the caller-owned shared lookup lifetime, exact SGEMM/DGEMM bindings,
  and the deterministic package-private injection path.
- Prepare 0003, Runtime 0014, Compiler 0006, Planning 0006, and Backend Contract 0004 — Complete;
  establish the downstream ownership boundaries without becoming provider dependencies.
- JDK 26 `java.lang.foreign` API and Java 26 toolchain — current root build baseline.
- OpenBLAS CBLAS header contract — external ABI evidence only, not a Java/project dependency.

No installed OpenBLAS library is required for ordinary implementation or test validation.

## Follow-up tasks

- OpenBLAS provider 0003 remains a Draft row only. It owns low-level thread-count query/control,
  explicitly supplied installed-library numerical/ABI validation, JDK native-access enablement,
  process/global thread-state semantics, and the provider capability checkpoint.
- CPU 0001 and later CPU work remain Draft. CPU will own capability truth, route/fallback policy,
  batching, broadcasting, transpose/layout conversion, packing, loops, storage, execution, and
  use of this low-level GEMM surface.

Do not create a task-0003 file or a CPU task in this change.

## Architecture impact

Expected impact: None.

This task implements GEMM calls already assigned to the OpenBLAS provider, preserves the provider
as a zero-project-dependency leaf, and leaves CPU route selection and execution ownership
unchanged. If implementation needs the provider to interpret Tensor/config/layout/route facts,
depend on another project module, coordinate active calls, or expose a native handle, stop and
report the conflict rather than editing architecture.

## Documentation impact

The separate documentation-focused pass must:

- finalize `OpenBlasLibrary`, `OpenBlasGemmInvocation`, and package Javadocs for exact matrix
  geometry, scalars, segments, validation/failures, ownership, mutation, concurrency, close race,
  zero dimensions, and numerical non-guarantees;
- update the CPU backend guide to distinguish the current low-level GEMM provider call from the
  still-planned CPU route, normalization, storage, execution, and fallback work;
- update the glossary only for the reusable GEMM/provider distinction and dense row-major
  vocabulary justified by the final change, or record a reasoned no-change conclusion;
- synchronize task 0002/master/roadmap evidence and statuses only after implementation and the
  clean documentation pass finish; task 0002 remained `Ready` until those gates passed;
- preserve architecture documents because no ownership or dependency rule changes; and
- record reasoned no-change conclusions for Tensor, Compile, and Training APIs; Model
  capabilities/master plan and the MATMUL/linear/attention/convolution semantic contracts;
  Prepare, Runtime, Backend Contract, Config, CPU implementation; architecture/tests;
  backend-conformance/integration; Gradle/root/settings; native packaging; and other modules.

## Implementation prompt

```text
You are a clean-context implementation agent working in the Synaptik repository. Do not commit or
push.

Read in full: AGENTS.md; ARCHITECTURE.md; docs/planning/planning-guide.md;
docs/developer-guide/documentation-rules.md; the General, API/Javadoc, Backend Guide, and Planning
documentation profiles; docs/planning/backends/openblas-provider/master-plan.md; completed task
0001; and task 0002 at
docs/planning/backends/openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md.
Read the task's directly referenced architecture and official JDK 26/OpenBLAS sources, final
provider source/tests/generated Javadocs/build inventory, CPU placeholder/build and master-plan
boundary, focused current MATMUL/linear/attention/convolution contracts, relevant architecture
tests, root/settings Java 26 build, and Prepare/Runtime/Backend Contract/Config master-plan
boundaries. Treat legacy only as read-only capability evidence, never design authority.

Implement task 0002 exactly within its eleven authorized paths. Add only the two public calls and
one package-private field-free helper over the existing bindings/seam, with the exact semantics,
validation order/messages, span/overlap rules, zero behavior, scalar forwarding, invokeExact
failure policy, ownership, and concurrency boundaries in the task. Do not modify task 0001,
bindings/access/binder sources, Gradle, architecture, CPU code, or later tasks. Stop on an
architecture conflict, scope overflow, need for another public abstraction/symbol/seam, or any
ambiguity that would change the recorded contract.

Use focused tests while developing, then run exactly one final
./gradlew :backends:openblas-provider:test after executable Java stabilizes. Hand the actual diff
and exact test evidence to a separate clean-context documentation-focused agent in the same
overall change. That pass reuses the Java evidence and independently finalizes Javadocs, package
documentation, CPU guide, glossary, task/master/roadmap, provider Javadoc, Markdown/link/surface/
scope/status checks, and git diff --check without rerunning successful Java tests unless it changes
executable behavior or records a concrete stale-evidence risk. Keep task 0002 Ready until every
criterion passes; only then record evidence/completion and mark it Complete. Leave task 0003 Draft
without a file and every CPU task Draft.
```

## Local decisions

- The public calls remain on `OpenBlasLibrary` because invocation is inseparable from its
  caller-owned lookup lifetime and no independent public abstraction has a current consumer.
- The provider accepts one already-normalized dense row-major no-transpose matrix product. CPU
  owns every higher-level shape, layout, batching, packing, and loop concern.
- One package-private field-free helper keeps validation and exact invocation cohesive without
  broadening the existing native seam or making the lifetime owner a multi-concept implementation.
- Required spans describe only the logical row-major regions passed to native code. Overlap is
  checked on those actual backing regions with non-owning JDK views; A/B aliasing is safe at this
  boundary because both are inputs.
- Complete validation precedes output-empty return, which keeps failure behavior uniform. A
  positive-output empty contraction still calls GEMM to preserve `beta * C` semantics.
- Scalars are raw ABI inputs. The provider forwards rather than interprets them and promises no
  OpenBLAS numerical policy.
- Invocation catches `Error` separately and unchanged, then wraps every other throwable. Moving
  validation outside that catch keeps provider validation failures stable.
- Close/call coordination is deliberately caller-owned, matching task 0001's lifetime contract;
  no hot-path synchronization or active-call state is justified.

## Known limitations

- Only FLOAT32/FLOAT64 dense row-major non-transposed GEMM with 32-bit `blasint` dimensions is
  supported.
- The API starts each matrix at segment byte offset zero and exposes no stride, offset, transpose,
  column-major, batch, or packing control.
- Heap segments are rejected; callers must provide suitably aligned native memory with sufficient
  spatial, temporal, and thread-access bounds.
- The provider does not define or verify OpenBLAS numerical accuracy, rounding, exceptional-value
  behavior, determinism, or performance.
- Ordinary tests prove Java validation and ABI forwarding with fake handles only. Real native ABI
  and numerical validation remains task 0003 work with an explicitly supplied compatible library.
- A caller racing `close()` with invocation has no successful-call guarantee.
- Thread-count choice and process-global OpenBLAS state remain outside this task.

## Validation evidence

- The implementation context ran exactly one final
  `./gradlew :backends:openblas-provider:test` after executable Java stabilized. It passed four
  suites and 39 tests with zero skips, failures, or errors: `OpenBlasAbiContractTest` 9,
  `OpenBlasGemmInvocationTest` 17, `OpenBlasLibraryPublicShapeTest` 4, and
  `OpenBlasLibraryTest` 9. Executable Java and tests did not change afterward.
- Clean documentation context `/root/openblas_0002_docs` reused that evidence without rerunning
  Java tests because it changed only Javadoc and Markdown after executable stabilization.
- The documentation context ran `./gradlew :backends:openblas-provider:javadoc`; it passed, and
  generated package and `OpenBlasLibrary` pages were inspected for the finalized geometry,
  ownership, zero-dimension, concurrency, close-race, failure, and numerical boundaries.
- The five-file `python3 /tmp/validate_synaptik_markdown.py ...` command passed all relative-link,
  heading-anchor, fence, final-newline, and trailing-whitespace checks.
- Stable source, reflection/bytecode, build, planning, and diff checks confirmed the exact public
  and package-private shapes and argument carriers/order, helper placement, zero project
  dependencies, exact eleven-path maximum, unchanged task 0001, synchronized Complete status,
  task 0003 Draft with no task file, every CPU task Draft, and no forbidden scope expansion.
- `git diff --check` passed on the final combined change.

## Implementation notes

- Added exactly public `void sgemm(...)` and `void dgemm(...)` to `OpenBlasLibrary`; both acquire
  the existing checked bindings before delegating.
- Added one package-private final, field-free `OpenBlasGemmInvocation`. It implements the recorded
  validation order and messages, checked row-major spans, required-region overlap checks, zero
  behavior, exact scalar/ABI forwarding, and throwable policy without another native seam.
- Deterministic fake-handle tests cover both typed call sites, exact ABI shape and values,
  validation/failure order, spans and overflow, overlap, zero dimensions, scalar families,
  lifecycle ownership, and invocation throwable translation without installed OpenBLAS.
- Finalized public/helper/package Javadocs and the CPU guide. Added a reusable glossary distinction
  between low-level GEMM invocation and backend-independent MATMUL semantics.
- No Tensor, Compile, or Training API changes are justified: the provider accepts native segments
  only and does not expose model, compiler, or extension semantics. Model capabilities/master
  planning and current MATMUL, linear, attention, and convolution contracts remain accurate
  because they describe semantic expressions, not backend lowering or execution.
- No Prepare, Runtime, Backend Contract, Config, or CPU implementation changes are justified:
  ownership, route normalization, storage, execution, and fallback remain future CPU work. No
  architecture or architecture-test change is justified because module ownership and dependency
  direction are unchanged. Backend-conformance/integration and real-native numerical validation
  remain deferred to task 0003's provider checkpoint. Gradle/root/settings, native packaging, and
  every other module remain unchanged because the leaf keeps its JDK-only dependency surface.

## Completion summary

- Completed the exact two-method low-level FLOAT32/FLOAT64 row-major GEMM surface and its
  package-private validation/invocation helper.
- Added focused deterministic coverage and finalized affected Javadocs, package documentation,
  CPU guide, glossary terminology, and synchronized planning evidence.
- Changed exactly the eleven authorized paths: three provider production files, three provider
  tests, the CPU guide, glossary, this task, provider master plan, and roadmap.
- Reused the final 39-test provider evidence; provider Javadoc, five-file Markdown validation,
  generated-page/manual surface and scope checks, and `git diff --check` all passed.
- Unresolved issues: none for task 0002. Follow-up: task 0003 remains Draft and owns thread
  control plus explicitly supplied installed-library ABI/numerical validation; all CPU work
  remains Draft.

Status: Complete
