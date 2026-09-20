# Task 0001: Metal Capability, Storage, and Native Foundation

## Status

Complete

## Goal

Replace the Metal marker with the smallest truthful foundation that later Metal preparation can
use without claiming executable tensor work:

- one explicitly constructed `metal` capability provider that rejects null queries and reports
  no supported operation occurrence;
- one backend-private Java Foreign Function and Memory (FFM) bridge to a versioned Objective-C
  C ABI on macOS arm64;
- one closeable Metal device context that owns the default `MTLDevice` and one
  `MTLCommandQueue` but submits no command;
- run-owned Metal buffer and workspace representations backed by opaque `MTLBuffer` handles;
- exact bounded host upload and download access for buffer-foundation validation; and
- explicit, tested allocation, concurrency, lifetime, cleanup, and native-failure behavior.

The task proves a non-executing storage and native-resource foundation. MPSGraph lowering,
executable creation, prepared schedules, operation capability, and public Engine composition all
remain absent.

## Rationale and mental model

```text
Planning capability query
  -> MetalCapabilityProvider(metal)
  -> false until task 0002 delivers a complete prepared execution path

caller-supplied native-library path
  -> FFM symbol binding
  -> Metal device context (MTLDevice + MTLCommandQueue)
  -> fresh run-owned MTLBuffer representation
  -> task 0001 Java owner proves cleanup without prepared Runtime integration
  -> future task 0002 prepared use lets Runtime orchestrate per-run cleanup
  -> Metal representation always owns physical release and checked byte access
```

Capability is semantic ownership truth, not a report that Metal hardware or a native library is
present. A device context or successfully allocated buffer therefore cannot make
`MetalCapabilityProvider.supports(...)` return `true`. Task 0002 is the first task allowed to
advertise an operation, and only after it supplies the complete analysis, finalization,
representation, schedule, and execution path for that exact occurrence.

## Scope

- Remove `MetalBackendModule` and add public final `MetalCapabilityProvider` implementing the
  existing Planning `BackendCapabilityProvider` contract.
- Define `METAL_BACKEND_ID` as one public immutable `BackendId` whose exact value is `metal`;
  every provider returns that exact constant.
- Make `supports(query)` throw `NullPointerException("query")` for null and return `false` for
  every non-null `OperationCapabilityQuery` without inspecting its operation, data type, Shape,
  layout, attributes, or device state.
- Add root and internal Metal package documentation that distinguishes the delivered foundation
  from executable Metal support.
- Add one package-private injectable native API seam so deterministic unit tests can prove Java
  ownership and failure behavior without a Metal host. Its production implementation uses only
  JDK 26 FFM and a caller-supplied absolute native-library path; it performs no classpath scan,
  `ServiceLoader` lookup, environment search, default-name search, extraction, or global caching.
- Implement the exact version-1 foundation ABI defined in
  [Exact native ABI contract](#exact-native-abi-contract). It contains only version query,
  default device/queue context creation and release, buffer allocation and release, and bounded
  host upload/download calls. Java code must not expose Objective-C object pointers, raw
  addresses, or Metal framework types outside the package-private bridge.
- Add the Objective-C implementation under `native/metal-macos-arm64`, using automatic reference
  counting and the system Foundation and Metal frameworks. The native build script must write
  only beneath that directory's ignored `build/` directory and must not publish a binary.
- Create exactly one default device and one command queue per Java device context. If either is
  unavailable, creation fails without publishing a partial context and releases everything
  acquired so far.
- Allocate every buffer/workspace as a fresh shared-storage `MTLBuffer` with an exact non-negative
  logical byte size. A logical zero-byte representation uses the exact version-1 ABI rule: one
  private physical byte of shared-storage backing while every exposed bound and copy rule remains
  at logical size zero.
- Make buffer and workspace representations package-private implementations of Runtime's nominal
  `BufferRepresentation` and `WorkspaceRepresentation` roles. They retain no Tensor, operation,
  graph, route, slot, Engine, or public storage state.
- Keep one context owner reference plus child resource leases. `close()` on the context marks it
  closed before releasing its owner reference, rejects new allocation/access, is thread-safe and
  idempotent, and releases the native device/queue context exactly once after the last child has
  closed. It does not close child resources behind their Java wrappers; later prepared use may
  let Runtime orchestrate those wrappers, but task 0001 has no prepared integration.
- Make each buffer/workspace close thread-safe and idempotent. The first close marks the resource
  closed before native release, releases its native handle at most once, then releases its context
  lease. A native release failure propagates once and is never retried; the context lease must
  still be released, with a distinct context cleanup failure suppressed on the original failure.
- Serialize close against allocation and byte access at the narrow resource/context state gate.
  A call admitted before close either completes or propagates its exact native failure; a call
  admitted after close begins fails before an FFM downcall. Separate open resources may be used
  concurrently, and no process-global lock is introduced.
- Provide package-private exact-range upload and download only on the buffer representation.
  Validate source/destination segment liveness, current-thread accessibility, destination
  writability for downloads, non-negative offsets/counts, checked range arithmetic, and both
  Java-segment and logical-buffer bounds before the native call. Zero-byte access at a valid end
  position is permitted. Workspace bytes are not exposed to Java callers.
- Translate native status values into one package-private unchecked failure type with stable
  operation and status facts. Do not encode route, operation family, data-type capability,
  fallback, or user-facing Engine exception policy in that type.
- Add deterministic tests using the injected fake native API for provider truth, exact handle
  ownership, zero-byte geometry, independent allocations, child leases, reverse partial-failure
  cleanup, idempotent/concurrent close, close-versus-access behavior, range checks, and exact
  primary/suppressed failures.
- Add an opt-in native integration test that, when given the freshly built dylib on macOS arm64,
  creates a real device/queue context, allocates distinct zero/nonzero buffers and a workspace,
  round-trips bounded bytes, verifies out-of-range rejection, closes every resource, and proves
  post-close rejection. The task cannot be completed on the current eligible macOS arm64 host by
  skipping this test.
- Finalize affected Javadocs, the Metal backend guide, glossary impact, and planning evidence in a
  separate clean documentation-focused context after implementation tests stabilize.

## Exact native ABI contract

ABI version is the unsigned 32-bit value `1`. The dylib exports exactly these foundation symbols
with C linkage and default visibility; the Objective-C source includes the standard integer and
size definitions it needs directly, so this task adds no header file:

```c
uint32_t synaptik_metal_foundation_abi_version(void);

int32_t synaptik_metal_context_create(void **out_context);
int32_t synaptik_metal_context_release(void *context);

int32_t synaptik_metal_buffer_create(
        void *context,
        uint64_t logical_byte_size,
        void **out_buffer);
int32_t synaptik_metal_buffer_release(void *buffer);

int32_t synaptik_metal_buffer_upload(
        void *buffer,
        uint64_t buffer_offset,
        const void *source,
        uint64_t byte_count);
int32_t synaptik_metal_buffer_download(
        void *buffer,
        uint64_t buffer_offset,
        void *destination,
        uint64_t byte_count);
```

The version function returns `1` and has no status result or side effect. Every other function
returns exactly one of these stable signed 32-bit values:

| Value | C name | Meaning |
|---:|---|---|
| `0` | `SYNAPTIK_METAL_STATUS_OK` | The requested operation completed successfully. |
| `1` | `SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT` | A required handle, output pointer, or nonempty-copy pointer is null, or another scalar argument violates the rules below. |
| `2` | `SYNAPTIK_METAL_STATUS_NO_DEVICE` | `MTLCreateSystemDefaultDevice()` returned no device. |
| `3` | `SYNAPTIK_METAL_STATUS_NO_COMMAND_QUEUE` | The selected device could not create the context's command queue. |
| `4` | `SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED` | A context box, buffer box, or shared `MTLBuffer` could not be allocated. |
| `5` | `SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS` | An upload/download offset or count is outside the logical buffer range, including overflow-safe rejection. |
| `6` | `SYNAPTIK_METAL_STATUS_COPY_FAILED` | A validated shared buffer exposes no usable `contents` pointer for a nonempty copy. |
| `7` | `SYNAPTIK_METAL_STATUS_INTERNAL_ERROR` | An Objective-C exception or another unexpected native foundation failure was contained. |

The native source may use a private enum or constants for these names, but it exports no status-
string function. Java maps values `0` through `7` exactly. Any other returned integer fails
closed as an unknown native status, retains the raw integer for package-private diagnostics, and
is never treated as success, retried, remapped to `INTERNAL_ERROR`, or used for fallback.

### Handles, outputs, and FFM carriers

- `void *` context and buffer values are opaque single-consumption native handles. A context
  handle denotes one ARC-owned box retaining exactly one `id<MTLDevice>` and one
  `id<MTLCommandQueue>`. A buffer handle denotes one ARC-owned box retaining exactly one
  `id<MTLBuffer>` plus its logical byte size.
- The bridge creates handles with retained bridging ownership and consumes that ownership exactly
  once in the matching release function. Handles have no numeric identity, serialization,
  equality, public accessor, raw-address API, or validity query.
- Java FFM maps `uint32_t` and `int32_t` to `ValueLayout.JAVA_INT`, `uint64_t` to
  `ValueLayout.JAVA_LONG`, and every `void *`, `const void *`, or `void **` position to
  `ValueLayout.ADDRESS` carried as a `MemorySegment`. Java compares the version result through
  `Integer.toUnsignedLong(result)` and requires `1L`. Java rejects a negative `long` before using
  it as an unsigned size, offset, or count.
- Raw context/buffer carrier segments and the downcall handles remain package-private inside
  `MetalNativeApi`/`MetalDeviceContext`; neither representation exposes them through a public or
  Runtime interface.
- For `context_create` and `buffer_create`, the native function first validates that the output
  pointer itself is non-null, then stores null into `*out_context` or `*out_buffer` before any
  allocation or framework call. It publishes one non-null retained handle only immediately
  before returning `OK`. Every failure leaves the output cell null and releases all partial ARC
  state before return.
- Java allocates each output cell as one `ADDRESS` value initialized to `MemorySegment.NULL`,
  invokes the downcall, and accepts a handle only when status is `OK` and the resulting value is
  non-null. `OK` plus null is an internal bridge-contract failure. Non-`OK` plus non-null is also
  a bridge-contract failure, and Java invokes the matching release exactly once before reporting
  it so a malformed implementation cannot leak the published handle.

### Null, zero, range, storage, and ownership rules

- `context_create(NULL)` and `buffer_create(..., NULL)` return `INVALID_ARGUMENT` and have no
  side effect. `buffer_create(NULL, ..., out_buffer)` first nulls a valid output cell, then returns
  `INVALID_ARGUMENT`.
- `context_release(NULL)`, `buffer_release(NULL)`, upload/download with a null buffer, and a
  nonempty upload/download with a null source/destination return `INVALID_ARGUMENT`.
- A live context is required for `buffer_create`. Java child leases ensure the native context is
  not released before all buffers created from it are released. The native ABI has no child
  registry and does not close, enumerate, or validate outstanding buffers during context release.
- Every buffer uses `MTLResourceStorageModeShared`. A positive logical size requests that exact
  physical length. A logical size of zero is valid and creates a fresh buffer box whose logical
  size is zero over a one-byte shared `MTLBuffer`; the private physical byte is never addressable
  through this ABI.
- Upload/download accept a zero byte count. For a zero-byte copy, the source/destination may be
  null, `buffer_offset` may equal the logical size, and no `contents` access or `memcpy` occurs.
  An offset greater than logical size still returns `RANGE_OUT_OF_BOUNDS`.
- For a nonempty copy, source/destination must be non-null, `buffer_offset` must be no greater
  than logical size, and `byte_count` must be no greater than
  `logical_size - buffer_offset`. This subtraction form is mandatory so addition cannot wrap.
  Only after those checks may native code obtain `contents` and copy exactly `byte_count` bytes.
- The native Objective-C implementation uses automatic reference counting (ARC). Context and
  buffer boxes own their Metal objects strongly. No autoreleased object, error string, callback,
  block, command buffer, or framework object crosses the C ABI.
- Native release functions consume one live non-null handle exactly once and return `OK` after
  relinquishing that retained box. After release, any reuse of that handle—including another
  release—is outside the ABI contract and must never be attempted by Java. Native releases are
  intentionally not an idempotency service and keep no released-handle registry.
- Java context/buffer/workspace wrappers supply thread-safe, idempotent, once-only `close()` on
  top of those single-consumption native handles. The first Java close owns the one native
  release; later or concurrent Java closes are inert. This distinction must be explicit in
  Javadoc and tests.

### Production resolution and validation order

Given a caller-supplied absolute dylib path, production performs this exact cold sequence before
any context creation:

1. validate that the `Path` is absolute, open one shared lookup arena, and create the library
   `SymbolLookup`;
2. resolve only `synaptik_metal_foundation_abi_version`, construct its downcall handle, invoke it
   once, and require unsigned value `1`;
3. resolve and construct downcalls, in order, for
   `synaptik_metal_context_create`, `synaptik_metal_context_release`,
   `synaptik_metal_buffer_create`, `synaptik_metal_buffer_release`,
   `synaptik_metal_buffer_upload`, and `synaptik_metal_buffer_download`;
4. only after all six non-version symbols have resolved may Java call
   `synaptik_metal_context_create`.

An invalid path, lookup failure, missing version symbol, version mismatch, missing later symbol,
or downcall-construction failure closes the lookup arena, creates no context, and exposes no
partially usable bridge. There is no symbol alias, optional foundation symbol, lazy resolution,
default library name, environment/classpath search, registry, callback, error-string ownership,
retry, or fallback. Version 1 contains no MPSGraph type/function, command submission,
synchronization call, operation or data-type field, route, executable, trace payload, or Engine
contract.

## Out of scope

- any `true` Metal capability answer, including metadata-only, zero-element, FLOAT32, BFLOAT16,
  or FLOAT16 occurrences
- Model `FLOAT16`; mixed-precision semantics; data-type promotion; accumulation rules; or any
  inference from two-byte storage width
- BFLOAT16 operation, device, numerical-contract, MPSGraph, or custom-kernel eligibility
- MPSGraph graph construction, compilation, executable creation, tensor-data binding, command
  submission, synchronization, result publication, or operation execution
- custom Metal shaders, libraries, pipelines, kernel routes, fusion, optimizer routes, or route
  selection
- Metal partition analysis, lowering, Prepare analysis/finalization, prepared representation
  plans, `PreparedExecutable`, `PreparedBufferTransfer`, schedule assembly, or Runtime execution
- automatic host/device materialization, device-to-device transfer, hidden coherence, residency
  policy, pooling, reuse, aliasing, persistent prepared resources, or public device storage
- `BackendAvailabilitySnapshot` production, stable device tokens, device selection, refresh, or a
  public Metal integration/factory
- Engine changes, generic backend registration, mixed-owner schedule composition, Metal/CPU
  fallback, or a hypothetical backend registry
- Trace payload types or emission; current Trace backend payload work remains Draft
- native binary packaging, classpath resource extraction, signing, notarization, distribution,
  Intel macOS support, iOS support, or non-Apple platforms
- performance benchmarks or throughput/latency claims; this task executes no tensor operation
- changes to shared modules, architecture contracts/explanations, ADRs, dependency rules,
  architecture tests, backend-conformance tests, integration tests, root Gradle configuration, or
  `backends/metal/build.gradle.kts`
- a detailed specification for Metal task 0002 or any later task

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/planning`
  - `modules/runtime`
  - `modules/prepare`
  - Concrete backend modules
  - Metal backend
  - Prepare lifecycle
  - Run lifecycle
  - Dependency rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Metal backend guide](../../../../backend-guide/metal-backend.md)
- [Metal strategy note](../../../../design/notes/metal-backend-strategy.md)

## Architecture constraints

- Planning chooses backend ownership only. Task 0001 advertises no Metal ownership candidate and
  contains no MPSGraph/custom route decision.
- A capability answer is independent of native availability. Native foundation success cannot
  substitute for a complete prepared execution path.
- Metal owns its physical representations, native integration, access mechanics, resource
  lifetime, and future transfers. Runtime retains per-run logical ownership, validity/residency,
  cleanup orchestration, and prepared-schedule execution.
- Caller inputs are not wrapped as Metal storage in this task. Every Metal buffer/workspace
  created here is run-owned foundation state; later prepared creator callbacks will allocate it.
- The representations implement Runtime's current nominal `BufferRepresentation` and
  `WorkspaceRepresentation` interfaces, but task 0001 does not place them in a
  `PreparedRepresentationPlan` or `RunState`. Future prepared use lets Runtime orchestrate
  per-run cleanup. The Metal wrappers are independently idempotent and thread-safe so that later
  orchestration and current direct foundation tests share one physical lifetime contract.
- A future immutable prepared recipe may be reused concurrently, so the context and creators must
  support isolated resources for concurrent runs. No mutable buffer is shared between runs.
- The command queue is context-owned native foundation state. No command buffer is created or
  submitted in this task, and no synchronization or execution guarantee is claimed.
- Shared Prepare receives no Metal type. Metal analysis/finalization remain absent until task
  0002 and must later use the existing staged contracts without moving Metal lowering/storage
  into `modules/prepare`.
- Runtime receives no operation, graph, MPSGraph object, native route choice, or backend lookup.
- Metal does not depend on Engine. Engine remains the only composition root, and this task adds no
  composition API in either direction.
- MPSGraph and custom Metal kernels remain Metal-only and cannot become CPU-internal routes.
- Model task 0026 is the sole prerequisite for future true IEEE-754 binary16 `FLOAT16` semantics.
  It is not a dependency of this fail-closed current-type foundation. No task-0001 name, constant,
  buffer width, ABI field, test, or documentation may advertise FLOAT16.
- BFLOAT16 remains a distinct current logical type. A buffer can hold arbitrary bytes, but that
  fact establishes no BFLOAT16 operation, device, numerical, or route capability.
- The legacy branch is read-only behavioral evidence. No legacy package layout, runtime fallback,
  buffer registry, Engine coupling, training bridge, or source is copied.
- If implementation requires a shared-contract, dependency, architecture, Engine, or route
  change, stop and report the exact conflict instead of expanding this task.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.contract` — stable `BackendId` only.
- `io.github.pho001.synaptik.planning.capability` — current capability provider/query contract.
- `io.github.pho001.synaptik.runtime.resource` — nominal buffer/workspace lifecycle roles.

Packages added or changed:

- `io.github.pho001.synaptik.backend.metal` — deliberate public task-0001 capability surface and
  package documentation; the marker is removed.
- `io.github.pho001.synaptik.backend.metal.internal` — package-private native ABI, device-context,
  physical-representation, byte-access, and lifetime implementation. Keeping these coupled
  foundation roles together avoids widening an unproved public or cross-package Metal API.

Type placement:

- `io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider` — public Planning
  collaboration because the concrete backend owns capability truth.
- `io.github.pho001.synaptik.backend.metal.internal.MetalNativeApi` — package-private injectable
  C-ABI/FFM seam, including its production implementation and unchecked native failure type.
- `io.github.pho001.synaptik.backend.metal.internal.MetalDeviceContext` — package-private owner of
  one native device/queue context and child-resource leases.
- `io.github.pho001.synaptik.backend.metal.internal.MetalBufferRepresentation` — package-private
  Runtime buffer role over one opaque native buffer with bounded host byte access.
- `io.github.pho001.synaptik.backend.metal.internal.MetalWorkspaceRepresentation` —
  package-private Runtime scratch role over one opaque native buffer without logical validity or
  public byte access.

No `api`, public `storage`, public `native`, route, kernel, prepare, service, manager, registry, or
generic backend package is added.

## Affected files

Expected production and native paths:

- remove `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalBackendModule.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/package-info.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNativeApi.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalDeviceContext.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalBufferRepresentation.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalWorkspaceRepresentation.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/package-info.java`
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
- `native/metal-macos-arm64/build.sh`
- `native/metal-macos-arm64/README.md`

Expected test paths:

- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProviderTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalFoundationTest.java`

Expected documentation and planning paths:

- `docs/backend-guide/metal-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/metal/tasks/0001-metal-capability-storage-and-native-foundation.md`
- `docs/planning/backends/metal/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create, remove, or modify at most the 18 paths listed above:

- 8 Metal Java production paths, counting marker removal;
- 3 native-layout paths;
- 2 Metal test paths;
- 2 explanatory documentation paths; and
- 3 planning paths.

This is the smallest cohesive Java/native/test/documentation change that proves the bridge and
resource lifecycle together. If another production type, test file, build file, shared module,
architecture path, or documentation path is needed, stop and update this specification before
implementation; do not create a second detailed Metal task.

## Acceptance criteria

- `MetalBackendModule` is removed, and the exact public task-0001 Metal Java surface is
  `MetalCapabilityProvider` plus inherited and ordinary JDK members.
- `METAL_BACKEND_ID.value()` is exactly `metal`; `backendId()` returns the exact constant;
  construction performs no discovery, loading, allocation, registration, or caching.
- Every non-null query returns `false`, null fails exactly as specified, and tests prove that
  current data types—including BFLOAT16—do not change the answer. No FLOAT16 symbol or claim is
  introduced.
- The ABI version, exact seven exported symbols, C signatures, status integers, opaque-handle
  carriers, output-cell protocol, null/zero/range rules, ARC/shared-storage ownership, and
  single-consumption native release semantics match the exact ABI section without additions.
- Production resolves the version symbol first, validates value `1`, resolves the remaining six
  symbols in the specified order, and only then creates a context. Missing symbols, version
  mismatch, absent device/queue, allocation/range/copy failure, and unknown native status fail
  closed without leaking an owned handle.
- The Objective-C bridge owns one default device and queue per context and one retained
  `MTLBuffer` per resource. It submits no command and imports no MPSGraph framework.
- Buffer/workspace creation returns fresh handles, preserves exact logical byte size including
  zero, and never reuses one mutable representation across positions or concurrent runs.
- Context owner/child leases, close-before-release visibility, idempotency, concurrent close,
  deferred context release, allocation/access rejection after close, and exact once-only native
  release are automated and pass.
- Partial failures preserve the exact primary failure and attach only distinct cleanup failures
  in deterministic order; cleanup attempts all resources the Java owner acquired.
- Buffer upload/download validates complete Java/native logical ranges before the downcall,
  preserves exact bytes including zero-length ranges, and performs no Tensor conversion,
  data-type interpretation, route selection, command submission, or hidden copy outside the
  explicitly requested host access.
- Workspace storage exposes no byte-access API and carries no logical-buffer validity.
- Production Metal code contains no Engine import, registry, service locator, `ServiceLoader`,
  reflection-based discovery, raw `Object` API, unchecked generic access, public native handle,
  operation switch, MPSGraph/custom-kernel code, FLOAT16/BFLOAT16 capability logic, or fallback.
- `backends/metal/build.gradle.kts`, its current direct project dependencies, root build logic,
  shared modules, architecture/ADR paths, and architecture/conformance/integration tests are
  unchanged.
- Focused Metal tests pass on an ordinary host without loading native code; the opt-in native
  integration case passes against the freshly built dylib on the current macOS arm64 host.
- Public and internal Java declarations have meaningful Javadoc covering purpose, ownership,
  lifetime, thread safety, nullability, parameters, results, and failures.
- The Metal guide states that only the fail-closed provider and storage/native foundation are
  current; it does not claim an executable operation, Engine integration, MPSGraph route,
  FLOAT16, BFLOAT16, synchronization, or performance.
- The glossary's existing backend-capability-provider and physical-runtime-representation entries
  are updated with the bounded current Metal state: a fail-closed provider plus package-private
  Metal buffer/workspace representations. No new glossary heading is added merely for Objective-C
  or Metal framework names.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, the Metal guide,
  glossary impact, links, and planning evidence in the same overall change.
- Javadoc, local Markdown links/anchors/fences, exact path scope, package/type placement,
  task/master/roadmap status synchronization, native-source/build-script scope, and
  `git diff --check` all pass.

## Tests / validation

Implementation development may run focused test classes. After executable Java and native code
stabilize, the implementation context runs exactly one final ordinary module suite:

```bash
./gradlew :backends:metal:test
```

On the current macOS arm64 host, it must then build the native foundation and run the explicit
native integration case with the dylib path supplied by the test contract:

```bash
./native/metal-macos-arm64/build.sh
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test --tests '*MetalFoundationTest.nativeFoundationRoundTrip'
```

The implementation pass must also inspect the built library's exported `synaptik_metal_*`
symbols and linked Foundation/Metal frameworks. This is concrete ABI evidence, not a benchmark.

Documentation-focused pass, after final Javadocs and explanatory text:

```bash
./gradlew :backends:metal:javadoc
git diff --check
```

Validate every local Markdown link and heading anchor in the task, Metal master plan, roadmap,
Metal guide, native README, and any changed glossary section. Validate the exact 18-path limit,
the task/master/roadmap `Ready` synchronization before implementation and final synchronization
afterward, no Metal 0002+ specification, and the package/type map.

Repository-wide Java validation is deferred to the Metal executable/conformance milestone after
task 0002 or continuous integration. This task changes neither a shared contract, dependency,
module boundary, nor Gradle configuration. Run `:testing:architecture-tests:test` only if the
implementation unexpectedly changes a dependency/build boundary; that condition first requires
stopping and updating this specification. Backend conformance and end-to-end integration tests
are deferred because task 0001 advertises and executes no Model operation.

The documentation pass reuses the successful implementation/native test evidence and does not
rerun Java tests unless it changes executable behavior or records a concrete stale-evidence risk.

## Dependencies

- Complete Backend Contract identity, device, availability, and requirement value contracts.
- Complete Planning capability provider/query and ownership contracts. Task 0001 consumes only
  the provider/query boundary and never becomes eligible.
- Complete Runtime physical representation, run ownership, prepared origin, validity, transfer,
  publication, and cleanup contracts. This task implements only the two nominal representation
  roles and leaves their prepared use to task 0002.
- Complete Prepare staged analysis/finalization and schedule orchestration contracts. This task
  changes none of them.
- Complete Trace event envelope/correlation foundation as a preserved leaf. No backend payload or
  trace emission contract is stable or introduced here.
- Current JDK 26 FFM support and the repository's macOS arm64 Xcode Command Line Tools with
  Foundation and Metal frameworks.
- Model task 0026 is not a dependency. It gates every future FLOAT16 semantic or capability
  claim, while this task remains current-type-neutral and fail-closed.

## Follow-up tasks

- Metal 0002 remains Draft. It may become Ready only after task 0001 completes and a fresh
  planning pass defines exact MPSGraph operation coverage plus the complete backend-owned
  analysis, finalization, representation, transfer/materialization, schedule, and execution path.
- Metal 0002 is the first task allowed to change `supports(...)` to `true`; each advertised row
  must have a matching complete executable path. It must not infer FLOAT16 or BFLOAT16 support
  from this task's storage.
- Metal 0003 and 0004 remain concise Draft master-plan rows without detailed specifications.
- Later Engine composition must be planned from the then-current explicit composition contract.
  Task 0001 deliberately adds no generic registry or mixed-owner Engine contract.

## Architecture impact

Expected impact: None.

The task implements responsibilities already assigned to concrete Metal: capability truth,
physical storage/access, resource lifetime, and native integration. It consumes current inward
contracts and adds no dependency, authority, module boundary, or route decision. If implementation
requires an architecture or shared-contract change, stop and report the rule, affected modules,
and required decision instead of editing architecture in this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/metal/master-plan.md, and
docs/planning/backends/metal/tasks/0001-metal-capability-storage-and-native-foundation.md.
Read the directly referenced Planning, Runtime, Prepare, Backend Contract, Metal documentation,
native-layout, documentation-profile, and Java 26/macOS arm64 contracts needed by the task.

Implement Metal task 0001 exactly as specified. Do not add operation capability, MPSGraph or
custom-kernel execution, Prepare/finalization work, Engine composition, FLOAT16/BFLOAT16 claims,
automatic discovery, binary packaging, or any out-of-scope change. Stop and report an
architecture, dependency, native-ABI, or scope conflict instead of inventing a new boundary.

After implementation and the recorded ordinary/native validation, hand the exact diff and test
evidence to a separate documentation-focused agent/thread with clean context. That pass must
follow docs/developer-guide/documentation-rules.md, independently finalize Javadocs, the Metal
guide, glossary impact, planning evidence, and documentation validation in the same overall
change, and must not repeat successful Java tests unless executable behavior changes or a
concrete stale-evidence risk is recorded.

Update this task with local decisions, known limitations, validation evidence, implementation
notes, completion summary, and synchronized final status. Do not mark Complete before every
acceptance criterion and the documentation pass succeed.
```

## Local decisions

- Kept the only public task-0001 type as `MetalCapabilityProvider`; device, FFM, buffer, workspace,
  status, and failure details remain package-private.
- Used one context owner reference plus child leases rather than closing child wrappers from
  context close. This preserves wrapper ownership and delays native prerequisite release until
  the last child closes.
- Used one private physical byte for a logical zero-byte native buffer, as required by ABI
  version 1, while preserving a zero-byte Java-visible range.
- Kept host byte upload/download as package-private validation access on buffers only. Workspace
  storage exposes no byte access or logical-buffer validity.
- Kept capability independent from native availability. Successful library loading, device
  creation, or allocation never changes the provider's fail-closed answer.

## Known limitations

- The foundation is macOS arm64 only and requires a caller-supplied absolute dylib path; it does
  not package, sign, notarize, publish, discover, or extract a native library.
- The sandboxed native invocation reached the bridge but could not see the default Metal device
  and returned `NO_DEVICE`. The coordinating main agent therefore performed the required exact
  opt-in round trip outside the sandbox, where it passed.
- There is no supported operation, prepared Runtime integration, Engine composition, MPSGraph or
  custom-kernel route, command submission, synchronization guarantee, FLOAT16 or BFLOAT16 support
  claim, or performance claim.
- Backend conformance, integration, and repository-wide validation remain deferred as specified:
  this task advertises and executes no Model operation and changes no shared or build boundary.

## Validation evidence

- The corrective implementation context made access admission atomic with context close while
  moving admitted native actions outside the context monitor. Its focused
  `MetalFoundationTest` run passed all 11 tests with the native opt-in case skipped, and its
  current final `./gradlew :backends:metal:test` passed. This documentation follow-up reused that
  evidence and did not rerun Java tests.
- Implementation-context native evidence was reused: `./native/metal-macos-arm64/build.sh`
  passed; the dylib was inspected as arm64 Mach-O; `nm` showed exactly the seven required
  `synaptik_metal_*` exports; and `otool` showed Foundation and Metal framework links.
- The implementation context's sandboxed native invocation reached the native bridge and returned
  `NO_DEVICE` because the sandbox hid the default Metal device. This is recorded as an environment
  limitation, not successful integration evidence.
- The coordinating main agent, not this documentation agent, ran the exact opt-in command outside
  the sandbox:

  ```bash
  env SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew :backends:metal:test --tests '*MetalFoundationTest.nativeFoundationRoundTrip'
  ```

  It passed with `BUILD SUCCESSFUL`, one test executed, and 18 tasks up-to-date.
- This mandatory separate clean-context documentation-focused completion pass read the
  architecture contract, documentation rules and selected profiles, planning guide and roadmap,
  Metal plan/task, complete changed Java/native/test surface, and directly relevant Planning and
  Runtime contracts. It finalized Javadocs, the backend guide, native README, the two relevant
  existing glossary entries, and synchronized planning evidence without changing executable
  behavior.
- `./gradlew :backends:metal:javadoc` ran exactly once after Javadocs stabilized and passed:
  `BUILD SUCCESSFUL`, 10 actionable tasks, 2 executed and 8 up-to-date.
- After the corrective executable/Javadoc change, this same documentation-focused agent reread
  the final `MetalDeviceContext` and `MetalFoundationTest`, clarified the admission-versus-action
  concurrency boundary in the Metal guide and native README, updated the Java evidence above,
  and reran `./gradlew :backends:metal:javadoc` exactly once for the follow-up. It passed with
  `BUILD SUCCESSFUL`, 10 actionable tasks, one executed and nine up-to-date. The glossary, master
  plan, and roadmap required no correction because none describes context-monitor scope.
- The native ABI and Objective-C source did not change in the correction, so the earlier
  successful unsandboxed real-device evidence remains current.
- The final documentation checks validated local Markdown targets and heading anchors in every
  changed Markdown file; balanced fences; final newlines; no trailing whitespace; the exact 18
  changed paths; no Metal 0002-or-later task specification; task/master/roadmap status agreement;
  package/type placement; and `git diff --check`. All passed.
- The public surface remains exactly `MetalCapabilityProvider` plus inherited and ordinary JDK
  members. The package/type map matches the master plan: the provider and root package summary
  are in `io.github.pho001.synaptik.backend.metal`; all native/resource owners and their package
  summary are in `io.github.pho001.synaptik.backend.metal.internal`; both tests mirror those
  packages.
- No architecture, shared-module, dependency, Gradle, architecture-test, backend-conformance, or
  integration-test path changed. No repeated Java suite, repository-wide suite, architecture
  suite, conformance suite, integration suite, benchmark, or performance measurement was run by
  this documentation pass because the task changes no such boundary and claims no execution.

## Implementation notes

- Replaced the marker class with an immutable provider whose stable identity is `metal`, whose
  null behavior follows the Planning contract, and whose non-null support result is always false.
- Added the version-1 seven-symbol Objective-C ABI, JDK 26 FFM binding, one device/queue context,
  fresh shared-storage buffer/workspace wrappers, exact checked buffer byte access, child leases,
  and deterministic once-only cleanup and failure suppression.
- Added deterministic fake-native tests plus the opt-in real-native resource round trip. Tests
  cover fail-closed capability, logical geometry, independent handles, concurrency, close/access
  ordering, range validation, partial cleanup, and known/unknown native status facts.
- Documentation review found no defect requiring a Java/native correction. Javadoc changes were
  contract-only, and the backend guide explicitly labels the future prepare/execution lifecycle
  as conceptual rather than implemented.

## Completion summary

- Completed changes: delivered the fail-closed Metal capability provider and the non-executing
  native/storage foundation with exact ownership, concurrency, range, ABI, and failure behavior.
- Files changed or created: exactly the 18 paths listed in [Affected files](#affected-files),
  including removal of `MetalBackendModule`; no scope expansion occurred.
- Tests and validation: reused the corrective 11-test focused run and current final Metal module
  run, the unchanged native build/ABI/link inspection, and coordinator-run unsandboxed one-test
  round trip; this follow-up completed its required single Metal Javadoc run and all
  documentation/scope/status/whitespace checks.
- Documentation-agent review: the mandatory separate clean-context documentation pass is complete
  and changed no executable behavior.
- Documentation impact: finalized the Metal backend guide, native implementation/build/ABI guide,
  task evidence, master plan, and roadmap.
- Javadoc review: finalized every changed public and internal contract-relevant Java declaration,
  including purpose, ownership, lifecycle, concurrency, nullability, parameters, results, and
  caller-visible failures.
- Glossary impact: updated only the existing backend-capability-provider and physical-runtime-
  representation entries; no gratuitous Objective-C or Metal heading was added.
- Architecture/shared/build/test impact: none; existing contracts and validation deferrals remain
  accurate, so those paths required no change.
- Unresolved issues: None within task 0001.
- Follow-up required: Metal 0002 remains a separate Draft planning frontier with no detailed task
  specification; it must define the first complete truthful executable route before any supported
  operation claim.

Status: Complete
