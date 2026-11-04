# Task 0001: [Metal](../../../../glossary.md#term-metal) Capability, Storage, and Native Foundation

## Status

Complete

## Goal

Replace the [Metal](../../../../glossary.md#term-metal) marker with the smallest truthful foundation that later Metal preparation can
use without claiming executable [tensor](../../../../glossary.md#tensor) work:

- one explicitly constructed `metal` capability provider that rejects null queries and reports
  no supported operation occurrence;<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- one backend-private Java Foreign Function and Memory (FFM) bridge to a versioned Objective-C
  C ABI on macOS arm64;<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Foreign Function and Memory glossary](../../../../glossary.md#application-binary-interface-and-native-bridge); [FFM glossary](../../../../glossary.md#term-ffm); [Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>
- one closeable Metal [device context](../../../../glossary.md#device-submission-synchronization-and-execution-context) that owns the default `MTLDevice` and one
  `MTLCommandQueue` but submits no command;
- [run](../../../../glossary.md#term-run)-owned Metal [buffer](../../../../glossary.md#term-buffer) and workspace representations backed by [opaque](../../../../glossary.md#term-opaque) `MTLBuffer` handles;
- exact bounded [host](../../../../glossary.md#term-host) upload and download access for buffer-foundation [validation](../../../../glossary.md#term-validation); and
- explicit, tested allocation, [concurrency](../../../../glossary.md#term-concurrency), [lifetime](../../../../glossary.md#term-lifetime), cleanup, and native-failure behavior.

At task 0001's completion frontier, the delivered work was a non-executing storage and
native-resource foundation. [MPSGraph](../../../../glossary.md#term-mpsgraph) [lowering](../../../../glossary.md#lowering), executable creation, prepared schedules, [operation](../../../../glossary.md#operation)
capability, and public [Engine](../../../../glossary.md#term-engine) composition were absent. Metal tasks 0002–0003 later added executable
routes, and Engine task 0015 later added explicit single-[owner](../../../../glossary.md#term-owner) Metal composition.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

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

Capability is semantic [ownership](../../../../glossary.md#term-ownership-meanings) truth, not a report that [Metal](../../../../glossary.md#term-metal) hardware or a [native library](../../../../glossary.md#term-native-library) is
present. A [device context](../../../../glossary.md#device-submission-synchronization-and-execution-context) or successfully allocated [buffer](../../../../glossary.md#term-buffer) therefore cannot make
`MetalCapabilityProvider.supports(...)` return `true`. Task 0002 is the first task allowed to
advertise an [operation](../../../../glossary.md#operation), and only after it supplies the complete analysis, finalization,
representation, schedule, and execution path for that exact [occurrence](../../../../glossary.md#term-occurrence).

## [Scope](../../../../glossary.md#term-scope)

- Remove `MetalBackendModule` and add public final `MetalCapabilityProvider` implementing the
  existing Planning <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>[`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) contract.
- Define `METAL_BACKEND_ID` as one public [immutable](../../../../glossary.md#term-immutable) [`BackendId`](../../../../glossary.md#backend-identity--backendid) whose exact value is [`metal`](../../../../glossary.md#term-metal);
  every provider returns that exact constant.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Make `supports(query)` throw `NullPointerException("query")` for null and return `false` for
  every non-null [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) without inspecting its operation, data type, Shape,
  layout, attributes, or device state.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [device glossary](../../../../glossary.md#term-device))</span>
- Add root and internal Metal package documentation that distinguishes the delivered foundation
  from executable Metal support.
- Add one package-private injectable native API seam so deterministic unit tests can prove Java
  ownership and failure behavior without a Metal host. Its production implementation uses only
  JDK 26 FFM and a caller-supplied absolute native-library path; it performs no classpath scan,<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [host glossary](../../../../glossary.md#term-host); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [FFM glossary](../../../../glossary.md#term-ffm); [classpath glossary](../../../../glossary.md#term-classpath); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
  `ServiceLoader` [lookup](../../../../glossary.md#term-lookup), environment search, default-name search, extraction, or global caching.
- Implement the exact version-1 foundation [ABI](../../../../glossary.md#term-application-binary-interface-abi) defined in
  [Exact native ABI contract](#exact-native-abi-contract). It contains only version query,
  default device/queue context creation and release, buffer allocation and release, and bounded
  host upload/download calls. Java code must not expose Objective-C object pointers, raw
  addresses, or Metal framework types outside the package-private bridge.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>
- Add the [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) implementation under `native/metal-macos-arm64`, using automatic reference
  counting and the system Foundation and Metal frameworks. The native build script must write
  only beneath that directory's ignored <span data-glossary-navigation="true"> ([automatic reference counting glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>`build/` directory and must not publish a [binary](../../../../glossary.md#term-binary-descriptor-and-metadata).
- Create exactly one default device and one command queue per Java device context. If either is
  unavailable, creation fails without publishing a partial context and releases everything
  acquired so far.<span data-glossary-navigation="true"> ([command queue glossary](../../../../glossary.md#term-command-queue); [device context glossary](../../../../glossary.md#device-submission-synchronization-and-execution-context); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Allocate every [buffer](../../../../glossary.md#term-buffer)/workspace as a fresh shared-storage `MTLBuffer` with an exact non-negative
  logical byte size. A logical zero-byte representation uses the exact version-1 ABI rule: one
  private physical byte of shared-storage backing while every exposed bound and copy rule remains
  at logical size zero.
- Make buffer and workspace representations package-private implementations of [Runtime](../../../../glossary.md#term-runtime)'s [nominal](../../../../glossary.md#term-nominal)
  `BufferRepresentation` and `WorkspaceRepresentation` roles. They retain no Tensor, operation,
  graph, route, slot, Engine, or public storage state.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Keep one context [owner](../../../../glossary.md#term-owner) reference plus child resource [leases](../../../../glossary.md#term-lease). `close()` on the context marks it
  closed before releasing its owner reference, rejects new allocation/access, is thread-safe and
  idempotent, and releases the native device/queue context exactly once after the last child has
  closed. It does not close child resources behind their Java wrappers; later prepared use may
  let Runtime orchestrate those wrappers, but task 0001 has no prepared integration.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [idempotent glossary](../../../../glossary.md#term-idempotent))</span>
- Make each buffer/workspace close thread-safe and idempotent. The first close marks the resource
  closed before native release, releases its native handle at most once, then releases its context
  lease. A native release failure propagates once and is never retried; the context lease must
  still be released, with a distinct context cleanup failure suppressed on the original failure.
- Serialize close against allocation and byte access at the narrow resource/context state gate.
  A call admitted before close either completes or propagates its exact native failure; a call
  admitted after close begins fails before an FFM downcall. Separate open resources may be used
  concurrently, and no process-global lock is introduced.<span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall))</span>
- Provide package-private exact-range upload and download only on the buffer representation.
  Validate source/destination segment liveness, current-thread accessibility, destination
  writability for downloads, non-negative offsets/counts, checked range arithmetic, and both
  Java-segment and logical-buffer bounds before the native call. Zero-byte access at a valid end
  position is permitted. Workspace bytes are not exposed to Java callers.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>
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
  skipping this test.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test); [dylib glossary](../../../../glossary.md#term-dylib); [arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>
- Finalize affected Javadocs, the Metal backend guide, glossary impact, and planning evidence in a
  separate clean documentation-focused context after implementation tests stabilize.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Exact native [ABI](../../../../glossary.md#term-application-binary-interface-abi) contract

ABI version is the unsigned 32-bit value `1`. The [dylib](../../../../glossary.md#term-dylib) exports exactly these foundation symbols
with C linkage and default visibility; the [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) source includes the standard integer and
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

The version function returns `1` and has no status result or [side effect](../../../../glossary.md#term-side-effect). Every other function
returns exactly one of these stable signed 32-bit values:

| Value | C name | Meaning |
|---:|---|---|
| `0` | `SYNAPTIK_METAL_STATUS_OK` | The requested [operation](../../../../glossary.md#operation) completed successfully. |
| `1` | `SYNAPTIK_METAL_STATUS_INVALID_ARGUMENT` | A required handle, output pointer, or nonempty-copy pointer is null, or another [scalar](../../../../glossary.md#term-scalar) [argument](../../../../glossary.md#term-argument) violates the rules below. |
| `2` | `SYNAPTIK_METAL_STATUS_NO_DEVICE` | `MTLCreateSystemDefaultDevice()` returned no [device](../../../../glossary.md#term-device). |
| `3` | `SYNAPTIK_METAL_STATUS_NO_COMMAND_QUEUE` | The selected device could not create the context's [command queue](../../../../glossary.md#term-command-queue). |
| `4` | `SYNAPTIK_METAL_STATUS_ALLOCATION_FAILED` | A context box, [buffer](../../../../glossary.md#term-buffer) box, or shared `MTLBuffer` could not be allocated. |
| `5` | `SYNAPTIK_METAL_STATUS_RANGE_OUT_OF_BOUNDS` | An upload/download offset or count is outside the logical buffer range, including [overflow](../../../../glossary.md#term-overflow)-safe rejection. |
| `6` | `SYNAPTIK_METAL_STATUS_COPY_FAILED` | A validated shared buffer exposes no usable `contents` pointer for a nonempty copy. |
| `7` | `SYNAPTIK_METAL_STATUS_INTERNAL_ERROR` | An Objective-C exception or another unexpected native foundation failure was contained. |

The native source may use a private enum or [constants](../../../../glossary.md#term-constant) for these names, but it exports no status-
string function. Java maps values `0` through `7` exactly. Any other returned integer fails
closed as an unknown native status, retains the raw integer for package-private diagnostics, and
is never treated as success, retried, remapped to `INTERNAL_ERROR`, or used for fallback.

### Handles, outputs, and [FFM](../../../../glossary.md#term-ffm) carriers

- `void *` context and buffer values are opaque single-consumption native handles. A context
  handle denotes one ARC-owned box retaining exactly one <span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>`id<MTLDevice>` and one
  `id<MTLCommandQueue>`. A [buffer](../../../../glossary.md#term-buffer) handle denotes one [ARC](../../../../glossary.md#term-arc)-owned box retaining exactly one
  `id<MTLBuffer>` plus its logical byte size.
- The bridge creates handles with retained bridging ownership and consumes that ownership exactly
  once in the matching release function. Handles have no numeric identity, serialization,
  equality, public accessor, raw-address API, or validity query.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [release glossary](../../../../glossary.md#term-release); [identity glossary](../../../../glossary.md#term-identity-meanings); [serialization glossary](../../../../glossary.md#term-serialization); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Java FFM maps [`uint32_t`](../../../../glossary.md#index-mappings-and-cost-dominance) and `int32_t` to `ValueLayout.JAVA_INT`, [`uint64_t`](../../../../glossary.md#index-mappings-and-cost-dominance) to
  `ValueLayout.JAVA_LONG`, and every `void *`, `const void *`, or `void **` position to
  `ValueLayout.ADDRESS` carried as a [`MemorySegment`](../../../../glossary.md#term-memorysegment). Java compares the version result through
  `Integer.toUnsignedLong(result)` and requires `1L`. Java rejects a negative `long` before using
  it as an unsigned size, offset, or count.
- Raw context/buffer [carrier](../../../../glossary.md#term-storage-carrier) segments and the [downcall](../../../../glossary.md#term-downcall) handles remain package-private inside
  `MetalNativeApi`/`MetalDeviceContext`; neither representation exposes them through a public or
  Runtime interface.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- For `context_create` and `buffer_create`, the native function first validates that the output
  pointer itself is non-null, then stores null into `*out_context` or `*out_buffer` before any
  allocation or framework call. It publishes one non-null retained handle only immediately
  before returning `OK`. Every failure leaves the output cell null and releases all partial ARC
  state before return.<span data-glossary-navigation="true"> ([releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Java allocates each output cell as one `ADDRESS` value initialized to `MemorySegment.NULL`,
  invokes the downcall, and accepts a handle only when status is `OK` and the resulting value is
  non-null. `OK` plus null is an internal bridge-contract failure. Non-`OK` plus non-null is also
  a bridge-contract failure, and Java invokes the matching release exactly once before reporting
  it so a malformed implementation cannot leak the published handle.

### Null, zero, range, storage, and [ownership](../../../../glossary.md#term-ownership-meanings) rules

- `context_create(NULL)` and `buffer_create(..., NULL)` return `INVALID_ARGUMENT` and have no
  side effect. <span data-glossary-navigation="true"> ([side effect glossary](../../../../glossary.md#term-side-effect))</span>`buffer_create(NULL, ..., out_buffer)` first nulls a valid output cell, then returns
  `INVALID_ARGUMENT`.
- `context_release(NULL)`, `buffer_release(NULL)`, upload/download with a null buffer, and a
  nonempty upload/download with a null source/destination return `INVALID_ARGUMENT`.
- A live context is required for `buffer_create`. Java child leases ensure the native context is
  not released before all buffers created from it are released. The native ABI has no child
  registry and does not close, enumerate, or validate outstanding buffers during context release.<span data-glossary-navigation="true"> ([leases glossary](../../../../glossary.md#term-lease); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [registry glossary](../../../../glossary.md#term-registry))</span>
- Every [buffer](../../../../glossary.md#term-buffer) uses `MTLResourceStorageModeShared`. A positive logical size requests that exact
  physical length. A logical size of zero is valid and creates a fresh buffer box whose logical
  size is zero over a one-byte shared `MTLBuffer`; the private physical byte is never addressable
  through this ABI.
- Upload/download accept a zero byte count. For a zero-byte copy, the source/destination may be
  null, `buffer_offset` may equal the logical size, and no `contents` access or `memcpy` occurs.
  An offset greater than logical size still returns `RANGE_OUT_OF_BOUNDS`.
- For a nonempty copy, source/destination must be non-null, `buffer_offset` must be no greater
  than logical size, and `byte_count` must be no greater than
  `logical_size - buffer_offset`. This subtraction form is mandatory so addition cannot wrap.
  Only after those checks may native code obtain <span data-glossary-navigation="true"> ([native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture))</span>`contents` and copy exactly `byte_count` bytes.
- The native Objective-C implementation uses automatic reference counting (ARC). Context and
  buffer boxes own their Metal objects strongly. No autoreleased object, error string, callback,
  block, command buffer, or framework object crosses the C ABI.<span data-glossary-navigation="true"> ([Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [automatic reference counting glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [ARC glossary](../../../../glossary.md#term-arc); [Metal glossary](../../../../glossary.md#term-metal); [callback glossary](../../../../glossary.md#term-callback); [command buffer glossary](../../../../glossary.md#term-command-buffer))</span>
- Native [release](../../../../glossary.md#term-release) functions consume one live non-null handle exactly once and return `OK` after
  relinquishing that retained box. After release, any reuse of that handle—including another
  release—is outside the ABI contract and must never be attempted by Java. Native releases are
  intentionally not an idempotency service and keep no released-handle registry.<span data-glossary-navigation="true"> ([releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Java context/buffer/workspace wrappers supply [thread-safe](../../../../glossary.md#term-thread-safe), [idempotent](../../../../glossary.md#term-idempotent), once-only `close()` on
  top of those single-consumption native handles. The first Java close owns the one native
  release; later or concurrent Java closes are inert. This distinction must be explicit in
  Javadoc and tests.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

### Production resolution and [validation](../../../../glossary.md#term-validation) order

Given a caller-supplied absolute [dylib](../../../../glossary.md#term-dylib) path, production performs this exact cold sequence before
any context creation:

1. validate that the `Path` is absolute, open one shared [lookup](../../../../glossary.md#term-lookup) [arena](../../../../glossary.md#term-arena), and create the library
   `SymbolLookup`;
2. resolve only `synaptik_metal_foundation_abi_version`, construct its downcall handle, invoke it
   once, and require unsigned value `1`;
3. resolve and construct downcalls, in order, for
   `synaptik_metal_context_create`, `synaptik_metal_context_release`,
   `synaptik_metal_buffer_create`, `synaptik_metal_buffer_release`,
   `synaptik_metal_buffer_upload`, and `synaptik_metal_buffer_download`;
4. only after all six non-version symbols have resolved may Java call
   `synaptik_metal_context_create`.

An invalid path, lookup failure, missing version [symbol](../../../../glossary.md#term-symbol), version mismatch, missing later symbol,
or [downcall](../../../../glossary.md#term-downcall)-construction failure closes the lookup arena, creates no context, and exposes no
partially usable bridge. There is no symbol [alias](../../../../glossary.md#aliasing-and-arena), optional foundation symbol, [lazy](../../../../glossary.md#term-lazy-evaluation) resolution,
default library name, environment/[classpath](../../../../glossary.md#term-classpath) search, [registry](../../../../glossary.md#term-registry), [callback](../../../../glossary.md#term-callback), error-string [ownership](../../../../glossary.md#term-ownership-meanings),
retry, or fallback. Version 1 contains no [MPSGraph](../../../../glossary.md#term-mpsgraph) type/function, command submission,
[synchronization](../../../../glossary.md#term-synchronization) call, [operation](../../../../glossary.md#operation) or data-type field, route, executable, [trace payload](../../../../glossary.md#trace-payload--tracepayload), or [Engine](../../../../glossary.md#term-engine)
contract.

## Out of [scope](../../../../glossary.md#term-scope)

- any `true` Metal capability answer, including metadata-only, zero-element, FLOAT32, BFLOAT16,
  or FLOAT16 occurrences<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [element glossary](../../../../glossary.md#term-element); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- [Model](../../../../glossary.md#term-model) [`FLOAT16`](../../../../glossary.md#term-float16); mixed-precision semantics; data-type promotion; accumulation rules; or any
  inference from two-byte storage width<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [inference glossary](../../../../glossary.md#term-inference))</span>
- [BFLOAT16](../../../../glossary.md#term-bfloat16) [operation](../../../../glossary.md#operation), [device](../../../../glossary.md#term-device), numerical-contract, [MPSGraph](../../../../glossary.md#term-mpsgraph), or custom-[kernel](../../../../glossary.md#kernel) eligibility
- MPSGraph graph construction, compilation, executable creation, tensor-data binding, command
  submission, synchronization, result publication, or operation execution<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [binding glossary](../../../../glossary.md#term-binding); [synchronization glossary](../../../../glossary.md#term-synchronization); [publication glossary](../../../../glossary.md#term-publication))</span>
- custom Metal shaders, libraries, pipelines, kernel routes, fusion, optimizer routes, or route
  selection<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Metal partition analysis, lowering, Prepare analysis/finalization, prepared representation
  plans, <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [Prepare glossary](../../../../glossary.md#prepare))</span>[`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), [`PreparedBufferTransfer`](../../../../glossary.md#prepared-buffer-transfer--preparedbuffertransfer), schedule assembly, or [Runtime](../../../../glossary.md#term-runtime) execution
- automatic host/device materialization, device-to-device transfer, hidden coherence, residency
  policy, pooling, reuse, aliasing, persistent prepared resources, or public device storage<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [residency glossary](../../../../glossary.md#residency); [pooling glossary](../../../../glossary.md#term-pooling); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot) production, stable device tokens, device selection, refresh, or a
  public Metal integration/factory<span data-glossary-navigation="true"> ([tokens glossary](../../../../glossary.md#term-token))</span>
- Engine changes, generic backend registration, mixed-owner schedule composition, Metal/CPU
  fallback, or a hypothetical backend registry<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu); [registry glossary](../../../../glossary.md#term-registry))</span>
- [Trace payload](../../../../glossary.md#trace-payload--tracepayload) types or emission; current [Trace](../../../../glossary.md#trace) [backend](../../../../glossary.md#backend) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) work remains Draft
- native binary packaging, classpath resource extraction, signing, notarization, distribution,
  Intel macOS support, iOS support, or non-Apple platforms<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [classpath glossary](../../../../glossary.md#term-classpath); [notarization glossary](../../../../glossary.md#term-notarization); [distribution glossary](../../../../glossary.md#term-distribution))</span>
- performance benchmarks or [throughput](../../../../glossary.md#term-throughput)/[latency](../../../../glossary.md#term-latency) claims; this task executes no [tensor](../../../../glossary.md#tensor) operation
- changes to shared modules, architecture contracts/explanations, ADRs, dependency rules,
  architecture tests, backend-conformance tests, integration tests, root Gradle configuration, or<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
  `backends/metal/build.gradle.kts`
- a detailed specification for [Metal](../../../../glossary.md#term-metal) task 0002 or any later task

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/planning`
  - `modules/runtime`
  - `modules/prepare`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend)
  - [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle)
  - [Run](../../../../glossary.md#term-run) lifecycle
  - [Dependency](../../../../glossary.md#term-dependency) rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011: Per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership))</span>
- [Metal backend guide](../../../../backend-guide/metal-backend.md)
- [Metal strategy note](../../../../design/notes/metal-backend-strategy.md)

## Architecture constraints

- Planning chooses backend ownership only. Task 0001 advertises no Metal ownership candidate and
  contains no MPSGraph/custom route decision.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend ownership glossary](../../../../glossary.md#backend-ownership); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- A capability answer is independent of native availability. Native foundation success cannot
  substitute for a complete prepared execution path.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Metal owns its physical representations, native integration, access mechanics, resource
  lifetime, and future transfers. Runtime retains per-run logical ownership, validity/residency,
  cleanup orchestration, and prepared-schedule execution.<span data-glossary-navigation="true"> ([resource lifetime glossary](../../../../glossary.md#ownership-borrowing-and-lease); [run glossary](../../../../glossary.md#term-run); [residency glossary](../../../../glossary.md#residency))</span>
- Caller inputs are not wrapped as Metal storage in this task. Every Metal buffer/workspace
  created here is run-owned foundation state; later prepared creator callbacks will allocate it.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- The representations implement [Runtime](../../../../glossary.md#term-runtime)'s current [nominal](../../../../glossary.md#term-nominal) `BufferRepresentation` and
  `WorkspaceRepresentation` interfaces, but task 0001 does not place them in a
  [`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan) or [`RunState`](../../../../glossary.md#run-state--runstate). Future prepared use lets Runtime orchestrate
  per-run cleanup. The Metal wrappers are independently idempotent and thread-safe so that later
  orchestration and current direct foundation tests share one physical lifetime contract.<span data-glossary-navigation="true"> ([idempotent glossary](../../../../glossary.md#term-idempotent); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- A future immutable prepared recipe may be reused concurrently, so the context and creators must
  support isolated resources for concurrent runs. No mutable buffer is shared between runs.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- The command queue is context-owned native foundation state. No command buffer is created or
  submitted in this task, and no synchronization or execution guarantee is claimed.<span data-glossary-navigation="true"> ([command queue glossary](../../../../glossary.md#term-command-queue); [command buffer glossary](../../../../glossary.md#term-command-buffer); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Shared Prepare receives no Metal type. Metal analysis/finalization remain absent until task
  0002 and must later use the existing staged contracts without moving Metal lowering/storage
  into <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering))</span>`modules/prepare`.
- Runtime receives no [operation](../../../../glossary.md#operation), [graph](../../../../glossary.md#graph), [MPSGraph](../../../../glossary.md#term-mpsgraph) object, native route choice, or [backend](../../../../glossary.md#backend) [lookup](../../../../glossary.md#term-lookup).
- Metal does not depend on Engine. Engine remains the only composition root, and this task adds no
  composition API in either direction.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [composition root glossary](../../../../glossary.md#term-composition-root); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- MPSGraph and custom [Metal](../../../../glossary.md#term-metal) [kernels](../../../../glossary.md#kernel) remain Metal-only and cannot become [CPU](../../../../glossary.md#term-cpu)-internal routes.
- [Model](../../../../glossary.md#term-model) task 0026 is the sole prerequisite for future true [IEEE](../../../../glossary.md#term-ieee)-754 [binary16](../../../../glossary.md#floating-point-precision-and-rounding) [`FLOAT16`](../../../../glossary.md#term-float16) semantics.
  It is not a dependency of this fail-closed current-type foundation. No task-0001 name, constant,
  buffer width, ABI field, test, or documentation may advertise FLOAT16.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [constant glossary](../../../../glossary.md#term-constant); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- BFLOAT16 remains a distinct current logical type. A buffer can hold arbitrary bytes, but that
  fact establishes no BFLOAT16 operation, device, numerical, or route capability.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [device glossary](../../../../glossary.md#term-device))</span>
- The legacy branch is read-only behavioral evidence. No legacy package layout, runtime fallback,
  buffer registry, Engine coupling, training bridge, or source is copied.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- If implementation requires a shared-contract, dependency, architecture, Engine, or route
  change, stop and report the exact conflict instead of expanding this task.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.contract` — stable [`BackendId`](../../../../glossary.md#backend-identity--backendid) only.
- `io.github.pho001.synaptik.planning.capability` — current capability provider/query contract.
- `io.github.pho001.synaptik.runtime.resource` — [nominal](../../../../glossary.md#term-nominal) [buffer](../../../../glossary.md#term-buffer)/workspace [lifecycle](../../../../glossary.md#lifecycle) roles.

Packages added or changed:

- `io.github.pho001.synaptik.backend.metal` — deliberate public task-0001 capability surface and
  package documentation; the marker is removed.
- `io.github.pho001.synaptik.backend.metal.internal` — package-private native ABI, device-context,
  physical-representation, byte-access, and lifetime implementation. Keeping these coupled
  foundation roles together avoids widening an unproved public or cross-package Metal API.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [device glossary](../../../../glossary.md#term-device); [lifetime glossary](../../../../glossary.md#term-lifetime); [widening glossary](../../../../glossary.md#term-widening); [Metal glossary](../../../../glossary.md#term-metal); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Type placement:

- `io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider` — public Planning
  collaboration because the concrete backend owns capability truth.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- `io.github.pho001.synaptik.backend.metal.internal.MetalNativeApi` — package-private injectable
  C-ABI/FFM seam, including its production implementation and unchecked native failure type.<span data-glossary-navigation="true"> ([FFM glossary](../../../../glossary.md#term-ffm))</span>
- `io.github.pho001.synaptik.backend.metal.internal.MetalDeviceContext` — package-private owner of
  one native device/queue context and child-resource leases.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [leases glossary](../../../../glossary.md#term-lease))</span>
- `io.github.pho001.synaptik.backend.metal.internal.MetalBufferRepresentation` — package-private
  Runtime buffer role over one opaque native buffer with bounded host byte access.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [opaque glossary](../../../../glossary.md#term-opaque); [host glossary](../../../../glossary.md#term-host))</span>
- `io.github.pho001.synaptik.backend.metal.internal.MetalWorkspaceRepresentation` —
  package-private Runtime scratch role over one opaque native buffer without logical validity or
  public byte access.

No `api`, public `storage`, public `native`, route, [kernel](../../../../glossary.md#kernel), [prepare](../../../../glossary.md#prepare), service, manager, [registry](../../../../glossary.md#term-registry), or
generic [backend](../../../../glossary.md#backend) package is added.

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

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/metal-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/metal/tasks/0001-metal-capability-storage-and-native-foundation.md`
- `docs/planning/backends/metal/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create, remove, or modify at most the 18 paths listed above:

- 8 [Metal](../../../../glossary.md#term-metal) Java production paths, counting marker removal;
- 3 native-[layout](../../../../glossary.md#layout) paths;
- 2 Metal test paths;
- 2 explanatory documentation paths; and
- 3 [planning](../../../../glossary.md#term-planning) paths.

This is the smallest cohesive Java/native/test/documentation change that proves the bridge and
resource [lifecycle](../../../../glossary.md#lifecycle) together. If another production type, test file, build file, shared [module](../../../../glossary.md#term-module),
architecture path, or documentation path is needed, stop and update this specification before
implementation; do not create a second detailed Metal task.

## Acceptance criteria

- `MetalBackendModule` is removed, and the exact public task-0001 [Metal](../../../../glossary.md#term-metal) Java surface is
  `MetalCapabilityProvider` plus inherited and ordinary [JDK](../../../../glossary.md#term-java-development-kit-jdk) members.
- `METAL_BACKEND_ID.value()` is exactly `metal`; `backendId()` returns the exact constant;
  construction performs no discovery, loading, allocation, registration, or caching.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Every non-null query returns `false`, null fails exactly as specified, and tests prove that
  current data types—including BFLOAT16—do not change the answer. No FLOAT16 symbol or claim is
  introduced.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- The ABI version, exact seven exported symbols, C signatures, status integers, opaque-handle
  carriers, output-cell protocol, null/zero/range rules, ARC/shared-storage ownership, and
  single-consumption native release semantics match the exact ABI section without additions.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [signatures glossary](../../../../glossary.md#term-signature); [opaque glossary](../../../../glossary.md#term-opaque); [ARC glossary](../../../../glossary.md#term-arc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [release glossary](../../../../glossary.md#term-release))</span>
- Production resolves the version [symbol](../../../../glossary.md#term-symbol) first, validates value `1`, resolves the remaining six
  symbols in the specified order, and only then creates a context. Missing symbols, version
  mismatch, absent device/queue, allocation/range/copy failure, and unknown native status fail
  closed without leaking an owned handle.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- The [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) bridge owns one default [device](../../../../glossary.md#term-device) and queue per context and one retained
  `MTLBuffer` per resource. It submits no command and imports no [MPSGraph](../../../../glossary.md#term-mpsgraph) framework.
- Buffer/workspace creation returns fresh handles, preserves exact logical byte size including
  zero, and never reuses one mutable representation across positions or concurrent runs.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Context owner/child leases, close-before-release visibility, idempotency, concurrent close,
  deferred context release, allocation/access rejection after close, and exact once-only native
  release are automated and pass.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [leases glossary](../../../../glossary.md#term-lease))</span>
- Partial failures preserve the exact primary failure and attach only distinct cleanup failures
  in deterministic order; cleanup attempts all resources the Java owner acquired.
- Buffer upload/download validates complete Java/native logical ranges before the downcall,
  preserves exact bytes including zero-length ranges, and performs no Tensor conversion,
  data-type interpretation, route selection, command submission, or hidden copy outside the
  explicitly requested host access.<span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall); [Tensor glossary](../../../../glossary.md#tensor); [host glossary](../../../../glossary.md#term-host))</span>
- Workspace storage exposes no byte-access [API](../../../../glossary.md#term-application-programming-interface-api) and carries no logical-[buffer validity](../../../../glossary.md#buffer-validity).
- Production Metal code contains no [Engine](../../../../glossary.md#term-engine) import, [registry](../../../../glossary.md#term-registry), [service locator](../../../../glossary.md#term-service-locator), `ServiceLoader`,
  reflection-based discovery, raw `Object` API, unchecked generic access, public native handle,
  operation switch, MPSGraph/custom-kernel code, FLOAT16/BFLOAT16 capability logic, or fallback.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [kernel glossary](../../../../glossary.md#kernel))</span>
- `backends/metal/build.gradle.kts`, its current direct project dependencies, root build logic,
  shared modules, architecture/ADR paths, and architecture/conformance/integration tests are
  unchanged.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Focused Metal tests pass on an ordinary host without loading native code; the opt-in native
  integration case passes against the freshly built dylib on the current macOS arm64 host.<span data-glossary-navigation="true"> ([native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [dylib glossary](../../../../glossary.md#term-dylib); [arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>
- Public and internal Java declarations have meaningful Javadoc covering purpose, ownership,
  lifetime, thread safety, nullability, parameters, results, and failures.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [lifetime glossary](../../../../glossary.md#term-lifetime); [nullability glossary](../../../../glossary.md#term-nullability); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- At task 0001 completion, the Metal guide stated that only the fail-closed provider and
  storage/native foundation were current; it claimed no executable operation, Engine integration,
  MPSGraph route, FLOAT16, BFLOAT16, synchronization, or performance.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- The glossary's existing backend-capability-provider and physical-runtime-representation entries
  were updated with the bounded task-0001 state: a fail-closed provider plus package-private Metal
  buffer/workspace representations. No new glossary heading was added merely for Objective-C
  or Metal framework names.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- A separate clean documentation-focused agent pass finalizes affected Javadocs, the Metal guide,
  glossary impact, links, and planning evidence in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Javadoc, local Markdown links/anchors/fences, exact path scope, package/type placement,
  task/master/roadmap status synchronization, native-source/build-script scope, and<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` all pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development may [run](../../../../glossary.md#term-run) focused test classes. After executable Java and [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture)
stabilize, the implementation context runs exactly one final ordinary [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:metal:test
```

On the current macOS [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) [host](../../../../glossary.md#term-host), it must then build the native foundation and run the explicit
native integration case with the [dylib](../../../../glossary.md#term-dylib) path supplied by the test contract:

```bash
./native/metal-macos-arm64/build.sh
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test --tests '*MetalFoundationTest.nativeFoundationRoundTrip'
```

The implementation pass must also inspect the built library's exported `synaptik_metal_*`
symbols and linked Foundation/[Metal](../../../../glossary.md#term-metal) frameworks. This is concrete [ABI](../../../../glossary.md#term-application-binary-interface-abi) evidence, not a benchmark.

Documentation-focused pass, after final Javadocs and explanatory text:

```bash
./gradlew :backends:metal:javadoc
git diff --check
```

Validate every local Markdown link and heading anchor in the task, Metal master plan, roadmap,
Metal guide, native README, and any changed glossary section. Validate the exact 18-path limit,
the task/master/roadmap [`Ready`](../../../../glossary.md#term-ready) [synchronization](../../../../glossary.md#term-synchronization) before implementation and final synchronization
afterward, no Metal 0002+ specification, and the package/type map.

Repository-wide Java validation is deferred to the Metal executable/[conformance](../../../../glossary.md#term-conformance) milestone after
task 0002 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes neither a shared contract, [dependency](../../../../glossary.md#term-dependency),
module boundary, nor [Gradle](../../../../glossary.md#term-gradle) configuration. Run `:testing:architecture-tests:test` only if the
implementation unexpectedly changes a dependency/build boundary; that condition first requires
stopping and updating this specification. [Backend](../../../../glossary.md#backend) conformance and end-to-end integration tests
are deferred because task 0001 advertises and executes no [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation).

The documentation pass reuses the successful implementation/native test evidence and does not
rerun Java tests unless it changes executable behavior or records a concrete stale-evidence risk.

## Dependencies

- Complete [Backend](../../../../glossary.md#backend) Contract [identity](../../../../glossary.md#term-identity-meanings), [device](../../../../glossary.md#term-device), availability, and requirement value contracts.
- Complete Planning capability provider/query and ownership contracts. Task 0001 consumes only
  the provider/query boundary and never becomes eligible.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Complete Runtime physical representation, run ownership, prepared origin, validity, transfer,
  publication, and cleanup contracts. This task implements only the two nominal representation
  roles and leaves their prepared use to task 0002.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [nominal glossary](../../../../glossary.md#term-nominal))</span>
- Complete Prepare staged analysis/finalization and schedule orchestration contracts. This task
  changes none of them.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- Complete Trace event envelope/correlation foundation as a preserved leaf. No backend payload or
  trace emission contract is stable or introduced here.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Trace event envelope glossary](../../../../glossary.md#trace-event-envelope); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [trace glossary](../../../../glossary.md#trace))</span>
- Current JDK 26 FFM support and the repository's macOS arm64 Xcode Command Line Tools with
  Foundation and Metal frameworks.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [FFM glossary](../../../../glossary.md#term-ffm); [arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Model task 0026 is not a dependency. It gates every future FLOAT16 semantic or capability
  claim, while this task remains current-type-neutral and fail-closed.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency); [FLOAT16 glossary](../../../../glossary.md#term-float16); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Follow-up tasks

- Metal 0002 remains Draft. It may become Ready only after task 0001 completes and a fresh
  planning pass defines exact MPSGraph operation coverage plus the complete backend-owned
  analysis, finalization, representation, transfer/materialization, schedule, and execution path.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [planning glossary](../../../../glossary.md#term-planning); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [operation glossary](../../../../glossary.md#operation); [backend glossary](../../../../glossary.md#backend); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Metal](../../../../glossary.md#term-metal) 0002 is the first task allowed to change `supports(...)` to `true`; each advertised row
  must have a matching complete executable path. It must not infer FLOAT16 or BFLOAT16 support
  from this task's storage.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Metal 0003 and 0004 remain concise Draft master-plan rows without detailed specifications.
- Later Engine composition must be planned from the then-current explicit composition contract.
  Task 0001 deliberately adds no generic registry or mixed-owner Engine contract.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [registry glossary](../../../../glossary.md#term-registry); [owner glossary](../../../../glossary.md#term-owner))</span>

## Architecture impact

Expected impact: None.

The task implements responsibilities already assigned to concrete [Metal](../../../../glossary.md#term-metal): capability truth,
physical storage/access, [resource lifetime](../../../../glossary.md#ownership-borrowing-and-lease), and native integration. It consumes current inward
contracts and adds no [dependency](../../../../glossary.md#term-dependency), authority, [module](../../../../glossary.md#term-module) boundary, or route decision. If implementation
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
  status, and failure details remain package-private.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [FFM glossary](../../../../glossary.md#term-ffm); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Used one context owner reference plus child leases rather than closing child wrappers from
  context close. This preserves wrapper ownership and delays native prerequisite release until
  the last child closes.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [leases glossary](../../../../glossary.md#term-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [release glossary](../../../../glossary.md#term-release))</span>
- Used one private physical byte for a logical zero-byte native buffer, as required by ABI
  version 1, while preserving a zero-byte Java-visible range.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Kept host byte upload/download as package-private validation access on buffers only. Workspace
  storage exposes no byte access or logical-buffer validity.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [validation glossary](../../../../glossary.md#term-validation); [buffer validity glossary](../../../../glossary.md#buffer-validity))</span>
- Kept capability independent from native availability. Successful library loading, device
  creation, or allocation never changes the provider's fail-closed answer.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Known limitations

- The foundation is macOS arm64 only and requires a caller-supplied absolute dylib path; it does
  not package, sign, notarize, publish, discover, or extract a native library.<span data-glossary-navigation="true"> ([arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [dylib glossary](../../../../glossary.md#term-dylib); [native library glossary](../../../../glossary.md#term-native-library))</span>
- The sandboxed native invocation reached the bridge but could not see the default Metal device
  and returned <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [device glossary](../../../../glossary.md#term-device))</span>`NO_DEVICE`. The coordinating main agent therefore performed the required exact
  opt-in round trip outside the sandbox, where it passed.<span data-glossary-navigation="true"> ([round trip glossary](../../../../glossary.md#term-round-trip))</span>
- At task 0001's completion frontier, there was no supported operation, prepared Runtime
  integration, Engine composition, MPSGraph or custom-kernel route, command submission,
  synchronization guarantee, FLOAT16 or BFLOAT16 support claim, or performance claim. Later Metal
  route tasks and Engine task 0015 superseded only those execution/composition absences.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [kernel glossary](../../../../glossary.md#kernel); [synchronization glossary](../../../../glossary.md#term-synchronization); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Backend conformance, integration, and repository-wide validation were deferred as specified:
  this task advertised and executed no Model operation and changed no shared or build boundary.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [validation glossary](../../../../glossary.md#term-validation); [Model glossary](../../../../glossary.md#term-model))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The corrective implementation context made access admission atomic with context close while
  moving admitted native actions outside the context monitor. Its focused<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
  `MetalFoundationTest` run passed all 11 tests with the native opt-in case skipped, and its
  current final <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`./gradlew :backends:metal:test` passed. This documentation follow-up reused that
  evidence and did not rerun Java tests.
- Implementation-context native evidence was reused: `./native/metal-macos-arm64/build.sh`
  passed; the dylib was inspected as arm64 Mach-O; <span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [Mach-O glossary](../../../../glossary.md#term-mach-o))</span>`nm` showed exactly the seven required
  `synaptik_metal_*` exports; and `otool` showed Foundation and [Metal](../../../../glossary.md#term-metal) framework links.
- The implementation context's sandboxed native invocation reached the [native bridge](../../../../glossary.md#term-native-bridge) and returned
  `NO_DEVICE` because the sandbox hid the default Metal device. This is recorded as an environment
  limitation, not successful integration evidence.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
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
  behavior.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [planning glossary](../../../../glossary.md#term-planning); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- `./gradlew :backends:metal:javadoc` ran exactly once after Javadocs stabilized and passed:
  `BUILD SUCCESSFUL`, 10 actionable tasks, 2 executed and 8 up-to-date.
- After the corrective executable/Javadoc change, this same documentation-focused agent reread
  the final <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`MetalDeviceContext` and `MetalFoundationTest`, clarified the admission-versus-action
  concurrency boundary in the Metal guide and native README, updated the Java evidence above,
  and reran <span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency))</span>`./gradlew :backends:metal:javadoc` exactly once for the follow-up. It passed with
  `BUILD SUCCESSFUL`, 10 actionable tasks, one executed and nine up-to-date. The glossary, master
  plan, and roadmap required no correction because none describes context-monitor scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- The native ABI and Objective-C source did not change in the correction, so the earlier
  successful unsandboxed real-device evidence remains current.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>
- The final documentation checks validated local Markdown targets and heading anchors in every
  changed Markdown file; balanced fences; final newlines; no trailing whitespace; the exact 18
  changed paths; no Metal 0002-or-later task specification; task/master/roadmap status agreement;
  package/type placement; and <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check`. All passed.
- At task 0001's completion frontier, the public surface was exactly
  `MetalCapabilityProvider` plus inherited and ordinary JDK members. The then-current package map
  placed the provider and root package summary in <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`io.github.pho001.synaptik.backend.metal` and
  native/resource owners plus their package summary in<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
  `io.github.pho001.synaptik.backend.metal.internal`; tests mirrored those packages. Engine task
  0015 later added public Metal configuration/integration and moved implementation owners beside
  that facade as package-private types without changing task 0001's capability domain.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade))</span>
- No architecture, shared-module, dependency, Gradle, architecture-test, backend-conformance, or
  integration-test path changed. No repeated Java suite, repository-wide suite, architecture
  suite, conformance suite, integration suite, benchmark, or performance measurement was run by
  this documentation pass because the task changes no such boundary and claims no execution.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Replaced the marker class with an [immutable](../../../../glossary.md#term-immutable) provider whose stable [identity](../../../../glossary.md#term-identity-meanings) is [`metal`](../../../../glossary.md#term-metal), whose
  null behavior follows the Planning contract, and whose non-null support result is always false.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Added the version-1 seven-symbol Objective-C ABI, JDK 26 FFM binding, one device/queue context,
  fresh shared-storage buffer/workspace wrappers, exact checked buffer byte access, child leases,
  and deterministic once-only cleanup and failure suppression.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [FFM glossary](../../../../glossary.md#term-ffm); [binding glossary](../../../../glossary.md#term-binding); [device glossary](../../../../glossary.md#term-device); [buffer glossary](../../../../glossary.md#term-buffer); [leases glossary](../../../../glossary.md#term-lease))</span>
- Added deterministic fake-native tests plus the opt-in real-native resource round trip. Tests
  cover fail-closed capability, logical geometry, independent handles, concurrency, close/access
  ordering, range validation, partial cleanup, and known/unknown native status facts.<span data-glossary-navigation="true"> ([round trip glossary](../../../../glossary.md#term-round-trip); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [concurrency glossary](../../../../glossary.md#term-concurrency); [validation glossary](../../../../glossary.md#term-validation))</span>
- Documentation review found no defect requiring a Java/native correction. Javadoc changes were
  contract-only, and the backend guide explicitly labels the future prepare/execution lifecycle
  as conceptual rather than implemented.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Completion summary

- Completed changes: delivered the fail-closed Metal capability provider and the non-executing
  native/storage foundation with exact ownership, concurrency, range, ABI, and failure behavior.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [Metal glossary](../../../../glossary.md#term-metal); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Files changed or created: exactly the 18 paths listed in [Affected files](#affected-files),
  including removal of `MetalBackendModule`; no [scope](../../../../glossary.md#term-scope) expansion occurred.
- Tests and validation: reused the corrective 11-test focused run and current final Metal module
  run, the unchanged native build/ABI/link inspection, and coordinator-run unsandboxed one-test
  round trip; this follow-up completed its required single Metal Javadoc run and all
  documentation/scope/status/whitespace checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [module glossary](../../../../glossary.md#term-module); [round trip glossary](../../../../glossary.md#term-round-trip); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation-agent review: the mandatory separate clean-context documentation pass is complete
  and changed no executable behavior.
- Documentation impact: finalized the Metal backend guide, native implementation/build/ABI guide,
  task evidence, master plan, and roadmap.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: finalized every changed public and internal contract-relevant Java declaration,
  including purpose, ownership, lifecycle, concurrency, nullability, parameters, results, and
  caller-visible failures.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [nullability glossary](../../../../glossary.md#term-nullability); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Glossary impact: updated only the existing backend-capability-provider and physical-runtime-
  representation entries; no gratuitous Objective-C or Metal heading was added.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>
- Architecture/shared/build/test impact: none; existing contracts and validation deferrals remain
  accurate, so those paths required no change.
- Unresolved issues: None within task 0001.
- Follow-up required: Metal 0002 remains a separate Draft planning frontier with no detailed task
  specification; it must define the first complete truthful executable route before any supported
  operation claim.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [operation glossary](../../../../glossary.md#operation))</span>

Status: Complete
