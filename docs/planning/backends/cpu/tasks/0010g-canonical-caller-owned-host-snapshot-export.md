# Task 0010G: [Canonical](../../../../glossary.md#term-canonical-form) Caller-Owned [Host](../../../../glossary.md#term-host) [Snapshot](../../../../glossary.md#term-snapshot) Export

## Reader context — historical integration boundary

‘Caller-owned host snapshot’ names the result of a copy, not ownership of the source buffer. At the
CPU service-provider boundary, Engine's result-materialization code is the caller of
`copyToCanonicalHostBytes(...)`; ordinary application code receives Engine's detached
`HostTensorValue`. Engine pairs one publication representation borrowed from an open Runtime result
with that same occurrence's final descriptor. The descriptor describes logical shape and addressing;
the representation contains the current values. CPU synchronously reads those values into a fresh
mutable `byte[]` owned by its immediate caller, retaining neither array nor source. Engine owns
result-close coordination and the outward immutable wrapper. The source remains leased, must stay
accessible and unmodified throughout copying, and is not closed by export; the detached result can
survive source closure. This explains the later consumer without changing the future-Engine
assertions below.

For a concrete address calculation, assume a valid INT32 source with at least seven elements,
logical shape `[2,2]`, element offset 1 and element strides `[3,2]`. For integer coordinates
$i,j\in\{0,1\}$ the source element index is $a(i,j)=1+3i+2j$. Row-major traversal therefore reads
$a(0,0)=1$, $a(0,1)=3$, $a(1,0)=4$ and $a(1,1)=6$, in that order. Four INT32 elements require
$4\times4=16$ output bytes. Those bytes are encoded big-endian; source gaps are not copied, and this
storage-index calculation makes no floating-point arithmetic claim.

Current reading: [Canonical host snapshot ownership](../../../../backend-guide/cpu-backend.md#canonical-caller-owned-host-snapshots).

## Status

Complete

## Goal

Add the smallest supported [CPU](../../../../glossary.md#term-cpu) integration [operation](../../../../glossary.md#operation) that copies one CPU [publication](../../../../glossary.md#term-publication)
representation, [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) from an open [Runtime](../../../../glossary.md#term-runtime) [`RunResult`](../../../../glossary.md#run-result--runresult), through its exact resolved logical
`TensorDescriptor` into a fresh caller-owned `byte[]`.

The returned [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) is independent of source [layout](../../../../glossary.md#layout) and storage [lifetime](../../../../glossary.md#term-lifetime): logical [elements](../../../../glossary.md#term-element) appear
in [canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) order, and every multi-byte element uses [big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) [byte order](../../../../glossary.md#text-encoding-and-file-boundaries).
The operation supports all six current [Model](../../../../glossary.md#term-model) data types and exposes no CPU-private type, storage
[lease](../../../../glossary.md#term-lease), [arena](../../../../glossary.md#term-arena), [`MemorySegment`](../../../../glossary.md#term-memorysegment), Runtime coordinate, [Engine](../../../../glossary.md#term-engine) type, or [graph](../../../../glossary.md#graph) [identity](../../../../glossary.md#term-identity-meanings).

## [Scope](../../../../glossary.md#term-scope)

- Add exactly this public instance method to `CpuBackendIntegration`:

  ```java
  public byte[] copyToCanonicalHostBytes(
          BufferRepresentation representation,
          TensorDescriptor descriptor,
          long maximumBytes)
  ```

  The method is supported Engine-facing SPI, not an ordinary application facade. It returns a new
  mutable array owned exclusively by the caller; CPU retains neither the array nor a view of it.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [facade glossary](../../../../glossary.md#term-facade); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Delegate from the supported adapter to the existing CPU-private composition owner, preserving
  the adapter's open-lifetime guard and keeping physical access under <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lifetime glossary](../../../../glossary.md#term-lifetime); [guard glossary](../../../../glossary.md#term-guard))</span>`.internal`.
- Add one focused [CPU](../../../../glossary.md#term-cpu)-private [host](../../../../glossary.md#term-host)-[snapshot](../../../../glossary.md#term-snapshot) exporter under `internal.memory`. It accepts only the
  two current supported CPU buffer implementations, <span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>`CpuBorrowedBuffer` and `CpuNativeBuffer`,
  through the nominal public <span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal))</span>`BufferRepresentation` boundary. It must reject an arbitrary
  `BufferRepresentation` and an external or test subclass of `CpuBufferRepresentation` rather
  than trusting a cast or nominal marker.
- Interpret the supplied descriptor as the logical view of that exact publication occurrence.
  Validate every intrinsic fact that current CPU representations can prove: current supported
  concrete representation class, open/current-thread-accessible storage, exact CPU data type,
  complete-element byte geometry, direct carrier compatibility, and sufficient physical element
  capacity for the descriptor's referenced span. Current representations do not retain Shape,
  layout, publication, or graph provenance, so identity of the descriptor/representation pairing
  is intentionally supplied by the future Engine caller and cannot be re-proved here.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [publication occurrence glossary](../../../../glossary.md#publication-occurrence); [carrier glossary](../../../../glossary.md#term-storage-carrier); [referenced span glossary](../../../../glossary.md#referenced-element-span); [layout glossary](../../../../glossary.md#layout); [publication glossary](../../../../glossary.md#term-publication); [graph glossary](../../../../glossary.md#graph); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Require a fully static [Shape](../../../../glossary.md#shape) and a present resolved `LayoutDescriptor`. Use the descriptor's
  non-negative element offset and strides exactly. Do not infer a default layout, repair a
  descriptor, or silently substitute dense geometry.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride))</span>
- Traverse logical [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) in [row-major](../../../../glossary.md#term-row-major) order: the final [axis](../../../../glossary.md#term-axis) changes fastest. For coordinate
  `c`, read source [element](../../../../glossary.md#term-element) index
  `storageOffset + sum(c[axis] * stride[axis])` with checked `long` arithmetic. Rank zero reads
  exactly the element at <span data-glossary-navigation="true"> ([Rank glossary](../../../../glossary.md#term-rank))</span>`storageOffset`. A Shape with any zero extent has zero logical elements,
  performs no carrier access, and returns a fresh empty array.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- Accept positive and zero source strides. Repeated physical addresses, including broadcast
  zero-stride layouts, are valid reads and produce repeated canonical logical elements. Negative
  strides are unsupported by the current Model <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>`LayoutDescriptor` constructor and must never be
  manufactured or separately interpreted. Source injectivity and source/source overlap are not
  preconditions because export is read-only; the detached destination is dense and injective.<span data-glossary-navigation="true"> ([injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [detached glossary](../../../../glossary.md#detached-model-results); [injective glossary](../../../../glossary.md#term-injective))</span>
- Compute `elementCount` from the fully static Shape with the Model's zero-before-product rule.
  Compute canonical byte count as `Math.multiplyExact(elementCount,
  descriptor.dataType().byteWidth())`. A non-zero [overflow](../../../../glossary.md#term-overflow) is an `ArithmeticException`; a
  zero-element Shape produces zero bytes even when unused other extents would otherwise overflow.
- Require `maximumBytes >= 0`. Reject when [canonical](../../../../glossary.md#term-canonical-form) byte count exceeds `maximumBytes`, before
  allocation or source access. Independently reject a canonical byte count greater than
  `Integer.MAX_VALUE`, the explicit portable [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) `byte[]` index/length ceiling for this API,
  before allocation or source access. A lower implementation-specific allocation limit or memory
  exhaustion may still surface as the JVM's original <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`OutOfMemoryError`](../../../../glossary.md#memory-management-atomic-counters-and-complexity).
- Define canonical output encoding exactly:

  | [Data type](../../../../glossary.md#data-type--datatype) | Bytes per logical element | Canonical bytes |
  |---|---:|---|
  | [`FLOAT64`](../../../../glossary.md#term-float64) | 8 | [Big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) `Double.doubleToRawLongBits(value)`; preserve every raw [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), including [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and [signed zero](../../../../glossary.md#term-signed-zero). |
  | [`FLOAT32`](../../../../glossary.md#term-float32) | 4 | Big-endian `Float.floatToRawIntBits(value)`; preserve every raw payload, including NaN payload/sign, infinities, and signed zero. |
  | [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | 2 | Big-endian unsigned [view](../../../../glossary.md#term-view-meanings) of the exact stored 16-bit `short`; no [widening](../../../../glossary.md#term-widening), [rounding](../../../../glossary.md#term-rounding), [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), or [NaN](../../../../glossary.md#term-nan) conversion. |
  | [`INT64`](../../../../glossary.md#data-type--datatype) | 8 | Big-endian [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) 64-bit pattern. |
  | [`INT32`](../../../../glossary.md#data-type--datatype) | 4 | Big-endian two's-complement 32-bit pattern. |
  | [`BOOL`](../../../../glossary.md#data-type--datatype) | 1 | The exact canonical byte `0` or `1`; any other stored byte fails instead of being normalized. |

  CPU storage access remains native-order represented-value access, matching current CPU
  kernels; canonical output encoding is always big-endian and therefore platform-independent.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel))</span>
- Read segment-backed elements with the current CPU convention: the corresponding unaligned Java
  primitive `ValueLayout` in native [byte order](../../../../glossary.md#text-encoding-and-file-boundaries), and `JAVA_BYTE` for BOOL. Read observable primitive
  arrays through their matching carrier and carrier-relative byte offset. Require all address,
  element-to-byte, array-index, and destination-index conversions to be checked.
- Add a proved dense fast path for a `DENSE_CONTIGUOUS`, zero-offset descriptor. It may use a
  simple linear source-address loop and a raw bulk copy only where raw bytes are already identical
  to the canonical contract. Every type must still use the same raw-bit/canonical-BOOL rules as
  general traversal. The general logical odometer remains the semantic oracle. No benchmark gate
  is required because this is cold host export, outside generated kernels and Runtime execution.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Validate all non-racy metadata and capacity preconditions before allocating or reading source
  elements. The new array remains private until successful return, so any validation, access,
  allocation, or copy failure exposes no partial payload. Do not cache, publish, retain, close,
  mutate, transfer, or change validity/ownership of the source representation.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [cache glossary](../../../../glossary.md#term-cache); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Keep the call [synchronous](../../../../glossary.md#term-synchronous). `CpuBackendIntegration` may serve independent calls concurrently,
  but one call adds no lock around caller/Runtime storage. The caller must keep the adapter and
  publication lease open, keep the representation accessible to the calling thread, prevent
  source mutation, and prevent result/representation closure for the complete call. Concurrent
  mutation has no atomic-snapshot guarantee; racing closure/access may surface the original JDK or
  CPU access failure. Future Engine 0004 owns serialization against result closure.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Add focused CPU tests and finalize the supported-API guide, CPU backend guide, glossary impact,
  Javadocs, and planning evidence through the mandatory clean documentation-focused pass.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

### [Validation](../../../../glossary.md#term-validation) and failure precedence

The implementation and [Javadoc](../../../../glossary.md#term-javadoc) must preserve this deterministic order for conditions [CPU](../../../../glossary.md#term-cpu) can
check without racing external closure:

1. reject a closed `CpuBackendIntegration` with the existing `CPU backend integration is closed`
   failure before inspecting arguments;
2. reject null `representation`, then null [`descriptor`](../../../../glossary.md#term-descriptor);
3. reject a negative `maximumBytes`;
4. reject a non-static [Shape](../../../../glossary.md#shape), then an unresolved [layout](../../../../glossary.md#layout);
5. compute the checked logical [element](../../../../glossary.md#term-element) count and [canonical](../../../../glossary.md#term-canonical-form) byte count;
6. reject the caller byte limit, then the `Integer.MAX_VALUE` array ceiling;
7. reject an unsupported concrete representation;
8. reject a closed or current-thread-inaccessible CPU representation;
9. reject representation/descriptor data-type disagreement, incomplete-element representation
   geometry, or insufficient capacity/span, in that order;
10. obtain and validate the already-classified direct CPU [argument](../../../../glossary.md#term-argument) before allocating the result;
11. allocate, traverse, validate each [BOOL](../../../../glossary.md#data-type--datatype) byte at encounter time, and encode.

Stable messages are required for the new deliberate failures:

- `maximumBytes must be non-negative: <value>`;
- `host snapshot requires a fully static shape: <shape>`;
- `host snapshot requires a resolved layout`;
- `canonical byte count exceeds maximumBytes: required=<required>, maximum=<maximum>`;
- `canonical byte count exceeds JVM byte[] limit: <required>`;
- `representation is not a supported CPU buffer representation`;
- `CPU representation is closed` for the existing closed-state [guard](../../../../glossary.md#term-guard), or
  `CPU representation is not accessible to current thread` after that guard succeeds;
- `CPU representation data type differs from descriptor: representation=<actual>, descriptor=<expected>`;
- `CPU representation byte size is not a complete element multiple`;
- `CPU representation capacity is smaller than descriptor span: capacity=<capacity>, required=<span>`;
- `BOOL representation contains non-canonical byte at logical index <index>: <unsignedValue>`.

Null failures are `NullPointerException`; closed/inaccessible [lifecycle](../../../../glossary.md#lifecycle) failures are
`IllegalStateException`; every new negative-limit, unsupported-shape/layout/representation,
type/geometry/capacity, byte-limit/array-ceiling, or BOOL-content failure is
`IllegalArgumentException`; checked numeric [overflow](../../../../glossary.md#term-overflow) is `ArithmeticException`. Public Javadoc
must document each applicable failure category.

Existing lower-level closed/accessibility or checked-arithmetic failures retain their established
exception types and messages. A [race](../../../../glossary.md#term-race) after validation is explicitly outside deterministic
precedence.

## Out of [scope](../../../../glossary.md#term-scope)

- Engine 0004 or any Engine production/test/documentation implementation; no later detailed task
  specification may be created here.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- A new public [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) record, [immutable](../../../../glossary.md#term-immutable) byte wrapper, `ByteBuffer`, typed Java arrays, scalar
  decoding, <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>`HostTensorStorage`, [Tensor](../../../../glossary.md#tensor) construction, or ordinary-user [CPU](../../../../glossary.md#term-cpu) [API](../../../../glossary.md#term-application-programming-interface-api).
- Exposure of CPU internal representations, direct carriers, [`MemorySegment`](../../../../glossary.md#term-memorysegment), [`Arena`](../../../../glossary.md#term-arena), storage or
  result leases, Runtime buffer/result indices, prepared slots, <span data-glossary-navigation="true"> ([leases glossary](../../../../glossary.md#term-lease); [Runtime glossary](../../../../glossary.md#term-runtime); [buffer glossary](../../../../glossary.md#term-buffer))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [`TensorId`](../../../../glossary.md#tensorid), or graph
  identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Runtime selection, publication lookup, representation transfer/materialization, fallback,
  validity mutation, ownership transfer, or cleanup.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [lookup glossary](../../../../glossary.md#term-lookup); [materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Dynamic or binding-dependent Shapes, unresolved layouts, negative strides, non-CPU
  representations, conversion between data types, BOOL truthiness normalization, or floating NaN
  canonicalization.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization); [NaN glossary](../../../../glossary.md#term-nan); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Generated code, Class-File emitters, generator schema, operation IR, kernel specialization,
  route selection, OpenBLAS/provider behavior, worker orchestration, hot-path changes, tuning, or
  performance claims/benchmarks.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [operation glossary](../../../../glossary.md#operation); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Model, Compiler, Planning, Prepare, Runtime, Engine, Config, Trace, Backend Contract, provider,
  build/dependency, architecture-contract, ADR, architecture-test, backend-conformance, or
  integration-test changes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A cross-backend canonical-host-export contract. CPU 0010G is the demonstrated concrete CPU
  boundary needed by Engine; another backend may later require its own supported adapter.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [host glossary](../../../../glossary.md#term-host))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially concrete-backend physical access,
  Engine composition, Runtime result leases, and dependency rules.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [leases glossary](../../../../glossary.md#term-lease); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- [ADR 0011: Per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0005A partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [kernel glossary](../../../../glossary.md#kernel))</span>
- [CPU 0005B universal access plans](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [CPU 0006 static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
- [CPU 0008E bounded materialization](0008e-bounded-multi-input-materialization-and-representation-reuse.md)<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- [CPU 0009G closure checkpoint](0009g-final-support-correctness-hygiene-and-inventory-checkpoint.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [CPU 0010F lifecycle integration](0010f-supported-cpu-lifecycle-integration-adapter.md)
- [Runtime 0015 leased publication access](../../../modules/runtime/tasks/0015-leased-publication-representation-access.md)<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- [Engine 0003 typed binding and result metadata](../../../modules/engine/tasks/0003-typed-logical-input-binding-and-published-result-access.md)<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Concrete CPU owns physical representation compatibility, access, and copying. Runtime exposes
  only the borrowed representation while its result lease is open; Engine later pairs that exact
  occurrence with its final descriptor and owns the ordinary host-value facade.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lease glossary](../../../../glossary.md#term-lease); [Engine glossary](../../../../glossary.md#term-engine); [occurrence glossary](../../../../glossary.md#term-occurrence); [descriptor glossary](../../../../glossary.md#term-descriptor); [host glossary](../../../../glossary.md#term-host); [facade glossary](../../../../glossary.md#term-facade))</span>
- Engine remains the composition root and ordinary API owner. CPU must not depend on Engine, and
  no CPU type appears in the future ordinary Engine signature.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [owner glossary](../../../../glossary.md#term-owner); [signature glossary](../../../../glossary.md#term-signature))</span>
- The new operation is cold result export after Runtime publication, not prepared execution work.
  It does not enter a generated kernel, <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [publication glossary](../../../../glossary.md#term-publication); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [kernel glossary](../../../../glossary.md#kernel))</span>[`PreparedSchedule`](../../../../glossary.md#prepared-schedule--preparedschedule), [`RunState`](../../../../glossary.md#run-state--runstate), or [Runtime](../../../../glossary.md#term-runtime) [hot path](../../../../glossary.md#term-hot-path).
- Current Model layout geometry is authoritative: static extents, non-negative strides and offset,
  referenced span, rank-zero scalar behavior, and zero-element behavior are consumed rather than
  redefined.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [layout glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [referenced span glossary](../../../../glossary.md#referenced-element-span); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element))</span>
- Current CPU representations provide enough intrinsic metadata to prove concrete CPU class,
  type, byte-capacity, carrier, liveness, and current-thread access compatibility. They do not
  contain descriptor or publication provenance. The future Engine call site must obtain the
  representation and descriptor from the same exact publication occurrence while its lease is
  held; CPU 0010G must not add a graph/runtime identity solely to re-prove that pairing.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [carrier glossary](../../../../glossary.md#term-storage-carrier); [liveness glossary](../../../../glossary.md#term-liveness); [provenance glossary](../../../../glossary.md#provenance); [publication occurrence glossary](../../../../glossary.md#publication-occurrence); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- No module boundary, dependency, shared contract, or architecture rule changes. If one becomes
  necessary, stop and report rather than expanding this task.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing supported package changed:

- `io.github.pho001.synaptik.backend.cpu` — add the one host-copy operation to the existing
  Engine-facing <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [operation glossary](../../../../glossary.md#operation); [Engine glossary](../../../../glossary.md#term-engine))</span>`CpuBackendIntegration`; add no constructor or ordinary-user type.

Existing [CPU](../../../../glossary.md#term-cpu)-private packages changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — the existing
  composition owner delegates to the focused exporter and preserves lifecycle precedence.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — add one stateless focused exporter
  that owns checked CPU representation access, logical-layout traversal, and canonical encoding.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>

Test packages:

- `io.github.pho001.synaptik.backend.cpu.spi` — distinct-package supported-surface and black-box
  behavior without `.internal` imports.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — exact native/borrowed carrier,
  validation, traversal, encoding, and failure evidence.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [carrier glossary](../../../../glossary.md#term-storage-carrier); [validation glossary](../../../../glossary.md#term-validation))</span>

No package or [module](../../../../glossary.md#term-module) is added.

## Affected files

Expected production files:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendComposition.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuHostSnapshotExporter.java` (new)

Expected test files:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuBackendIntegrationHostSnapshotTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuHostSnapshotExporterTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
  (existing; add exactly the expected production-source entry
  `internal/memory/CpuHostSnapshotExporter.java`)

Expected explanatory documentation finalized in the same overall change:

- `docs/api/public-api.md`
- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if the documentation pass finds a reusable new or changed term; an
  evidence-backed no-change conclusion is otherwise required

[Planning](../../../../glossary.md#term-planning)/status files:

- `docs/planning/backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most thirteen paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production | 3 | Two existing delegation/[API](../../../../glossary.md#term-application-programming-interface-api) paths plus one new focused exporter |
| CPU tests | 4 | Two existing tests plus two focused new tests |
| Explanatory documentation | 3 | Public API, CPU [backend](../../../../glossary.md#backend) guide, and conditional glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **13** | **3 + 4 + 3 + 3** |

If another production, test, documentation, build, or architecture path is required, stop and
revise the plan before implementation. An unchanged conditional glossary path need not be spent;
unused capacity is not headroom for unrelated work.

## Acceptance criteria

- `CpuBackendIntegration` remains public, final, privately constructed, and `AutoCloseable`; its
  public declarations are exactly the existing seven plus
  `copyToCanonicalHostBytes(BufferRepresentation, TensorDescriptor, long)`. No public/protected
  signature or Javadoc exposes <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`.internal`, [`MemorySegment`](../../../../glossary.md#term-memorysegment), an [Engine](../../../../glossary.md#term-engine) type, or a new [CPU](../../../../glossary.md#term-cpu) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication).
- A distinct-package test compiles and calls the new method using only supported public CPU,
  Model, and Runtime types. Ordinary users still obtain CPU through Engine and are not instructed
  to construct the integration SPI directly.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- All six current data types are exhaustively covered on matching heap carriers and segment-backed
  representations. Tests lock big-endian canonical encoding, raw FLOAT64/FLOAT32 NaN payloads and
  signed zeros, raw BFLOAT16 NaN/zero/infinity/ordinary patterns, signed integral boundaries, and
  exact BOOL <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [canonical glossary](../../../../glossary.md#term-canonical-form); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [NaN glossary](../../../../glossary.md#term-nan); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [infinity glossary](../../../../glossary.md#term-infinity); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`0`/`1` bytes.
- Canonical bytes follow row-major logical coordinates for dense, non-zero-offset dense,
  positive-strided, zero-stride/broadcast, rank-zero, and zero-element descriptors. General
  traversal and the dense fast path produce identical bytes for the same logical values.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [stride glossary](../../../../glossary.md#term-stride); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Static/resolved preconditions, checked element/byte/address arithmetic, caller limit, JVM array
  ceiling, complete-element byte geometry, representation type/capacity/carrier/accessibility,
  and descriptor span are enforced in the specified order before source element access.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- A descriptor span equal to capacity succeeds. A larger span fails. Zero-element descriptors
  accept zero capacity and perform no source read regardless of their otherwise unused offset.
  Extra representation capacity is permitted and never exported.
- Positive and zero strides are accepted. Repeated source addresses are read once per logical
  occurrence and repeated in the result. No source injectivity or overlap restriction is added.
  No negative stride can enter through the supported Model descriptor.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- An arbitrary [nominal](../../../../glossary.md#term-nominal) `BufferRepresentation` and a non-supported subclass of the internal base
  are rejected with the stable unsupported-representation failure. The exact current borrowed and
  native CPU representations are accepted while open and accessible.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- BOOL validation fails on the first non-canonical logical value in row-major encounter order and
  reports its logical index and unsigned byte. It never normalizes another byte to true.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Every successful call returns a fresh array. Repeated calls and aliased publication occurrences
  are neither cached nor deduplicated. Mutating one returned array cannot affect source storage or
  another returned array.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- Export does not close, retain, mutate, transfer, or change validity/ownership of the source and
  performs no provider, route, worker, schedule, graph, or Runtime operation. Result/adapter
  closure after return cannot affect the detached bytes.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Concurrent calls over independently valid representations succeed without shared mutable
  exporter state. Tests and Javadocs state that callers must prevent concurrent source mutation or
  closure and that Engine 0004 will serialize materialization against result closure.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [materialization glossary](../../../../glossary.md#materialization))</span>
- Any failure returns no array and exposes no partial payload. JVM/JDK failures caused by a genuine
  allocation/access race propagate without translation.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [race glossary](../../../../glossary.md#term-race))</span>
- No generated source/emitter/schema, OpenBLAS/provider behavior, module dependency, build file,
  shared Java contract, architecture test, backend conformance, or integration test changes.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [architecture test glossary](../../../../glossary.md#term-architecture-test); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- The existing exhaustive production-source allowlist in `CpuInternalPackageInventoryTest`
  receives exactly `internal/memory/CpuHostSnapshotExporter.java`. Its assertion remains a fixed
  exhaustive inventory; do not weaken it or replace it with dynamic discovery.
- A separate clean documentation-focused pass finalizes Javadocs, the two explanatory guides,
  glossary impact, links, examples, and planning evidence before the task becomes <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>`Complete`.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation context `01a0a4db-17b7-76c3-99ce-5c5358b14586` already ran the focused command
after the six planned Java paths stabilized; all 14 focused tests passed. Preserve that evidence
and do not rerun the focused tests for the one-line exhaustive-inventory correction. Add exactly
`internal/memory/CpuHostSnapshotExporter.java` to the existing fixed expected set in
`CpuInternalPackageInventoryTest`, then rerun only the full [CPU](../../../../glossary.md#term-cpu) suite:

```bash
./gradlew :backends:cpu:test
```

Focused tests must cover the complete acceptance [matrix](../../../../glossary.md#term-matrix-meanings), including all six types; [heap](../../../../glossary.md#term-heap) and
segment carriers; [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) and native CPU representations; dense/general equivalence; offset,
positive/zero [strides](../../../../glossary.md#term-stride), [scalar](../../../../glossary.md#term-scalar), and empty [Shapes](../../../../glossary.md#shape); raw floating/[BFLOAT16](../../../../glossary.md#term-bfloat16) patterns; [BOOL](../../../../glossary.md#data-type--datatype) rejection;
every validation-precedence edge; exact-limit/equal-capacity success; [overflow](../../../../glossary.md#term-overflow), limit, [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) ceiling,
capacity, wrong-type, unsupported-representation, closed, and inaccessible failures; fresh-array
[identity](../../../../glossary.md#term-identity-meanings); no source mutation/closure; post-copy independence; and concurrent independent calls.

Documentation-focused context reuses the successful CPU test evidence unless it changes
executable Java, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also records:

- `javap -public` confirmation of the exact eight-method `CpuBackendIntegration` surface and no
  `.internal`, [`MemorySegment`](../../../../glossary.md#term-memorysegment), [Engine](../../../../glossary.md#term-engine), storage-[lease](../../../../glossary.md#term-lease), or new [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) type in public [descriptors](../../../../glossary.md#term-descriptor);
- successful [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) of a distinct-package [fixture](../../../../glossary.md#term-fixture) against declared dependencies;
- Markdown local-link/anchor, heading-order, balanced-fence, trailing-whitespace, and final-newline
  checks for the six changed Markdown files, including the glossary entry justified by the
  documentation pass;
- exact at-most-thirteen-path allowlist and confirmation that Java changes are CPU-only;
- scans proving no Engine import/dependency, build change, generated-code/schema edit, Runtime
  selection, reflection, service locator, provider/tuning work, or new ordinary CPU construction;<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [Runtime glossary](../../../../glossary.md#term-runtime); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- source inspection confirming native-order unaligned source access, big-endian output, checked
  traversal/address arithmetic, limit-before-access behavior, and no source ownership/validity
  mutation; and<span data-glossary-navigation="true"> ([big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- synchronized CPU task/master/roadmap status, Engine 0004 still `Blocked`, no Engine 0004 or
  later detailed task specification, and preservation of every pre-existing unrelated worktree
  change.

Repository-wide validation is deferred to Engine 0008/[CI](../../../../glossary.md#term-continuous-integration-ci). [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) is not [run](../../../../glossary.md#term-run) because
this is a CPU-specific export [SPI](../../../../glossary.md#term-service-provider-interface-spi) rather than a shared backend behavior contract. Architecture and
integration tests are not run because no [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module) boundary, shared architecture rule,
or end-to-end ordinary Engine behavior changes.

## Dependencies

- [CPU 0010F](0010f-supported-cpu-lifecycle-integration-adapter.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> — Complete; supplies the
  supported lifecycle owner and current checked CPU representation implementations.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner))</span>
- [Runtime 0015](../../../modules/runtime/tasks/0015-leased-publication-representation-access.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span> —
  Complete; supplies exact borrowed publication-representation access while the result lease is
  open.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [publication glossary](../../../../glossary.md#term-publication); [lease glossary](../../../../glossary.md#term-lease))</span>
- [Engine 0003](../../../modules/engine/tasks/0003-typed-logical-input-binding-and-published-result-access.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span> —
  Complete; supplies exact ordered publication metadata and final resolved descriptors for the
  later Engine caller.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Current [Model](../../../../glossary.md#term-model) [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `LayoutDescriptor`, and `TensorDescriptor`
  contracts, plus current CPU representation and carrier/access conventions.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>

All dependencies are complete and every contract decision required for implementation is fixed.

## Follow-up tasks

- [Engine](../../../../glossary.md#term-engine) 0004 remains `Blocked` pending its own subsequent clean planning pass. CPU 0010G now
  satisfies its CPU prerequisite, but this task does not create or edit the Engine specification
  for lazy per-publication <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [publication glossary](../../../../glossary.md#term-publication))</span>[`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) [materialization](../../../../glossary.md#materialization).
- Engine 0005 and later work remain Draft. This task creates no later specification and does not
  advance optional CPU vendor/tuning rows.

## Architecture impact

Expected impact: None.

This task realizes the existing [ownership](../../../../glossary.md#term-ownership-meanings) split: [Runtime](../../../../glossary.md#term-runtime) lends the [nominal](../../../../glossary.md#term-nominal) [publication](../../../../glossary.md#term-publication)
representation, [CPU](../../../../glossary.md#term-cpu) validates and reads its concrete physical storage, and future [Engine](../../../../glossary.md#term-engine) code
will own publication selection, close coordination, and the ordinary [immutable](../../../../glossary.md#term-immutable) [host](../../../../glossary.md#term-host) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication). No
[dependency](../../../../glossary.md#term-dependency) or [module](../../../../glossary.md#term-module) boundary changes. If implementation requires a shared [API](../../../../glossary.md#term-application-programming-interface-api), Engine
dependency, build edit, or architecture change, stop and report the exact gap.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository on CPU task 0010G.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and
docs/planning/backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md in full.
Read the directly referenced Runtime 0015 and Engine 0003 contracts and inspect the current final
CPU representations, integration adapter, Model descriptor/layout/data-type contracts, source
access conventions, tests, and documentation before editing.

Implement task 0010G exactly as specified within its thirteen-path ceiling. Add only the supported
CpuBackendIntegration.copyToCanonicalHostBytes(BufferRepresentation, TensorDescriptor, long)
operation, CPU-private delegation/exporter, focused CPU tests, the exact existing-inventory entry
`internal/memory/CpuHostSnapshotExporter.java`, and authorized documentation. Keep the inventory
assertion fixed and exhaustive; do not weaken it or replace it with dynamic discovery.
Preserve raw represented bits, canonical big-endian row-major output, checked validation order,
detached ownership, and the result/adapter lifetime boundary. Do not add Engine code, shared
contracts, build/dependency changes, generated code, provider/tuning work, backend conformance,
later task specs, commits, or pushes. Stop if an architecture or scope conflict appears.

Reuse implementation context `01a0a4db-17b7-76c3-99ce-5c5358b14586`'s successful 14/14 focused
test evidence. After the one-line inventory allowlist update, rerun only the full CPU suite. Then
hand the diff and exact test evidence to a distinct clean documentation-focused context. That pass must follow
docs/developer-guide/documentation-rules.md, finalize affected Javadocs, public/CPU guides,
glossary impact, and planning evidence, run Javadoc and documentation checks, and not repeat
successful Java tests unless it changes executable behavior or records a concrete risk. Mark the
task Complete only after every gate passes.
```

## Local decisions

- The return value is a [detached](../../../../glossary.md#detached-model-results) `byte[]`; no CPU-owned payload type is justified because future
  Engine owns the immutable <span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Engine glossary](../../../../glossary.md#term-engine); [immutable glossary](../../../../glossary.md#term-immutable))</span>[`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) wrapper and can defensively encapsulate these bytes.
- The method name states both action and format. `copy` avoids implying a live [view](../../../../glossary.md#term-view-meanings); [`Canonical`](../../../../glossary.md#term-canonical-form)
  and `HostBytes` distinguish logical [row-major](../../../../glossary.md#term-row-major) [big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) output from raw [CPU](../../../../glossary.md#term-cpu) storage bytes.
- Big-endian is the fixed canonical external byte order already selected by the Engine master
  plan. Native order remains only the source-representation interpretation.<span data-glossary-navigation="true"> ([byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
- BOOL fails on non-canonical storage. Normalization would hide a corrupt CPU representation and
  would not preserve the exact current CPU BOOL contract.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [Normalization glossary](../../../../glossary.md#term-normalization))</span>
- A dense zero-offset linear fast path is included; general logical traversal remains the oracle.
  No platform-dependent multi-byte raw-copy shortcut is required.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Current CPU metadata proves safe physical compatibility but not descriptor provenance. Adding
  provenance or Runtime/graph identity to CPU storage would exceed the smallest boundary and
  duplicate the exact pairing that future Engine already owns.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- No backend-conformance row is added because the canonical bytes are a CPU adapter contract, not
  a shared backend SPI. No benchmark gate applies to cold caller-requested export.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- The existing exhaustive production-source inventory is part of this capability's test scope.
  Adding its one required literal exporter entry preserves the assertion's fixed fail-closed
  purpose; weakening it or deriving its expected set dynamically would remove that protection.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Known limitations

- Only fully static, resolved descriptors and the two current supported CPU representation
  implementations are accepted.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The result is bounded by `Integer.MAX_VALUE` bytes and available heap. There is no streaming,
  chunked, mapped, or caller-supplied destination form.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- The method does not synchronize source mutation or closure. Future Engine 0004 must serialize
  access to one result lease; callers at this integration SPI must coordinate directly.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lease glossary](../../../../glossary.md#term-lease); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- The supplied descriptor/representation publication pairing is trusted at this layer because
  current CPU buffers intentionally carry no publication or graph provenance.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [buffers glossary](../../../../glossary.md#term-buffer); [carry glossary](../../../../glossary.md#term-carry); [graph glossary](../../../../glossary.md#graph); [provenance glossary](../../../../glossary.md#provenance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context: this clean [CPU](../../../../glossary.md#term-cpu) 0010G planning pass.

- Inspected current Runtime 0015 source, Javadocs, tests, and completion record: publication access
  returns the exact borrowed <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>`BufferRepresentation` only while the result lease is open, checks
  closure before index, and performs no copy or physical access.
- Inspected current CPU integration and representation source/tests: the supported adapter has
  seven public methods; `CpuBorrowedBuffer` and `CpuNativeBuffer` are the current concrete buffer
  implementations; their base retains exact <span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>[`DataType`](../../../../glossary.md#data-type--datatype), byte size, segment, accessibility, and a
  once-classified array/segment argument, but no descriptor or publication provenance.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Inspected Model contracts: exactly six data types exist; Shapes admit rank zero and zero extents;
  resolved layouts permit only non-negative strides/offset, calculate checked referenced span,
  and allow zero-stride views; <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [rank glossary](../../../../glossary.md#term-rank); [strides glossary](../../../../glossary.md#term-stride); [referenced span glossary](../../../../glossary.md#referenced-element-span); [views glossary](../../../../glossary.md#term-view-meanings))</span>`TensorDescriptor` reconstructs a present [layout](../../../../glossary.md#layout) against its [Shape](../../../../glossary.md#shape).
- Inspected CPU affine/access/copy evidence: current CPU represented-bit movement preserves raw
  FLOAT64/FLOAT32/BFLOAT16/integral bits and canonical BOOL, uses native-order unaligned segment
  access, supports positive/zero strides, and treats zero-element access as empty.<span data-glossary-navigation="true"> ([represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element))</span>
- Inspected Engine planning decisions: future host values require big-endian canonical row-major
  bytes, a caller byte limit, fresh uncached snapshots, and close/materialization serialization.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [row-major glossary](../../../../glossary.md#term-row-major); [materialization glossary](../../../../glossary.md#materialization); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- No architecture, dependency, shared-contract, or unresolved API decision was found. Runtime
  0015, CPU 0010F, and Engine 0003 are Complete, so this task is Ready.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Java tests were not [run](../../../../glossary.md#term-run), as required for this planning-only pass.
- [Scope](../../../../glossary.md#term-scope)-correction evidence: implementation context `01a0a4db-17b7-76c3-99ce-5c5358b14586`
  completed the six originally planned Java paths and its focused command passed 14/14 tests. Its
  full CPU suite ran 1,010 tests with exactly one failure and no other reported failure: the
  existing exhaustive `CpuInternalPackageInventoryTest` expected-source set omitted the required
  new `internal/memory/CpuHostSnapshotExporter.java`. The actual and expected inventories otherwise
  matched. This correction therefore adds that existing test as the fourth CPU test path, raises
  the task ceiling from 12 to 13, and authorizes only the one literal expected entry; it does not
  weaken or dynamically derive the inventory assertion.
- Final implementation evidence reused from clean context
  `01a0a4db-17b7-76c3-99ce-5c5358b14586`: the focused CPU host-snapshot/public matrix passed
  14/14 tests; after the exact inventory correction, the final<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
  `./gradlew :backends:cpu:test` ran 1,010 tests with zero failures, zero errors, and 28 skips.
  That context also passed the exact seven-Java-path and `git diff` checks. Documentation context
  `01a0a4ec-8d2e-7a41-a944-cdcc1ca43802` changed no executable Java token and therefore did not
  rerun either successful Java test command.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>
- Documentation context `01a0a4ec-8d2e-7a41-a944-cdcc1ca43802` independently reviewed the
  architecture contract; current architecture plan; documentation rules and General,
  API/Javadoc, Backend Guide, Planning, and Example profiles; planning guide and roadmap; CPU,
  Runtime 0015, and Engine 0003 plans; all seven task-owned Java paths; the public/CPU APIs;
  glossary; generated Javadocs; and the final diff.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Backend glossary](../../../../glossary.md#backend))</span>
- `./gradlew :backends:cpu:javadoc` completed successfully. The first sandboxed invocation could
  not create the Gradle wrapper lock under the user cache; the permitted rerun completed with the
  two expected incubating-Vector notices and existing unrelated missing-<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [cache glossary](../../../../glossary.md#term-cache); [Vector glossary](../../../../glossary.md#term-vector))</span>`@param` warnings in
  `CpuPartitionLowering.LoweredPartition` and `CpuPartitionPreparationPlan`. No warning names a
  0010G path. Generated pages for `CpuBackendIntegration`, `CpuBackendComposition`, and
  `CpuHostSnapshotExporter` were inspected and contain the final ownership, layout, encoding,
  size, failure, and concurrency contracts.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- `javap -classpath backends/cpu/build/classes/java/main -public
  io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration` reports exactly eight public
  methods: `open`, `capabilityProvider`, `availabilitySnapshot`, `preparations`,
  `scheduleAssembler`, `borrow`, `copyToCanonicalHostBytes`, and `close`. Its descriptors expose
  no `.internal`, [`MemorySegment`](../../../../glossary.md#term-memorysegment), [Engine](../../../../glossary.md#term-engine), storage-[lease](../../../../glossary.md#term-lease), or new [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) type.
- A distinct-package [fixture](../../../../glossary.md#term-fixture) compiled successfully with `javac` against the declared CPU, Model,
  Runtime, Backend Contract, Compiler, Prepare, Planning, Config, Trace, and OpenBLAS-provider
  output dependencies. The fixture imports only supported CPU, Model, and Runtime types and calls
  the exact three-argument copy method.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Comment-stripped SHA-256 values for the three production files were unchanged before and after
  documentation finalization: <span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>`0294b776cea786c840bfbf846ba8238e56f0ba3000a3c9791ae24721c3ac7f24`
  for `CpuBackendIntegration`, `bbda20b3131a3c4953a63596ebcef1f61c49c4151aefcf9e2c1e846137e82ee4`
  for `CpuBackendComposition`, and
  `d5f69e71271c84d7119c828388f5680f4a2dac4c5e9c750ee997c0f21cd78adb`
  for `CpuHostSnapshotExporter`. This proves the documentation pass changed [Javadoc](../../../../glossary.md#term-javadoc) only in Java.
- Source/mechanism inspection confirmed exact-class acceptance for `CpuBorrowedBuffer` and
  `CpuNativeBuffer`, native-order unaligned source reads, fixed big-endian output, raw floating
  and BFLOAT16 bits, strict BOOL, checked count/address arithmetic, caller-limit and JVM-ceiling
  checks before representation access, resolved offsets with positive/zero strides, and no
  source close, validity mutation, ownership transfer, Runtime selection, provider action, or
  generated-code path.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- `python3 /tmp/validate_synaptik_markdown.py docs/api/public-api.md
  docs/backend-guide/cpu-backend.md docs/glossary.md
  docs/planning/backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md
  docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md` passed all six changed
  Markdown paths for local targets, effective anchors, balanced fences, LF/final newlines, and
  trailing whitespace. The separate heading-order scan also passed.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- The final task-attributable union is exactly 13 paths: three CPU production Java files, four CPU
  test files, three explanatory documents, and three planning/status documents. Java changes are
  exactly the seven CPU paths. Scans found no Engine import or dependency, build edit, generated
  source/schema edit, provider change, Runtime selection, service locator, production reflection,
  ordinary CPU construction guidance, architecture-test source, backend-conformance source, or
  integration-test source change.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [service locator glossary](../../../../glossary.md#term-service-locator); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- CPU task 0010G, the CPU master row, and the roadmap now agree on `Complete`. [Runtime](../../../../glossary.md#term-runtime) 0015 remains
  `Complete`; Engine 0004 remains `Blocked` pending its own subsequent clean planning pass;
  Engine 0005–0008 remain `Draft`; and no Engine 0004 or later detailed task specification exists.
  The pre-existing Runtime 0015 and Engine/Runtime frontier files remain otherwise preserved.
- Final `git diff --check` passed.

## Implementation notes

Implementation context `01a0a4db-17b7-76c3-99ce-5c5358b14586` implemented the exact three
production and four test paths, including the corrected fixed exhaustive inventory entry. Its
focused and final [CPU](../../../../glossary.md#term-cpu) suites passed. Clean documentation context
`01a0a4ec-8d2e-7a41-a944-cdcc1ca43802` retained the implementation, finalized Javadocs and the
public [API](../../../../glossary.md#term-application-programming-interface-api)/CPU guide/glossary boundary, synchronized task/master/roadmap status, and completed the
non-executable [validation](../../../../glossary.md#term-validation) without changing Java [tokens](../../../../glossary.md#term-token).

## Completion summary

- Completed changes: added the supported Engine-facing CPU snapshot-copy SPI, CPU-private
  delegation/exporter, exact validation and six-type canonical encoding, focused tests, and the
  fixed exhaustive production-source inventory entry.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Files changed or created: exactly the 13 paths authorized under [Maximum scope](#maximum-scope)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>:
  three CPU production files, four CPU tests, Public API, CPU backend guide, glossary, this task,
  CPU master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend))</span>
- Tests and [validation](../../../../glossary.md#term-validation): reused implementation context
  `01a0a4db-17b7-76c3-99ce-5c5358b14586`'s passing 14/14 focused tests and final 1,010-test CPU
  suite with zero failures/errors and 28 skips. Documentation context passed CPU Javadoc,
  generated-page inspection, exact eight-method <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, distinct-package fixture compilation,
  executable-token preservation, six-file Markdown validation, heading/scope/import/dependency/
  mechanism/status checks, and final <span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [token glossary](../../../../glossary.md#term-token); [dependency glossary](../../../../glossary.md#term-dependency))</span>`git diff --check`.
- Documentation-agent review: clean context `01a0a4ec-8d2e-7a41-a944-cdcc1ca43802` completed the
  mandatory independent General, API/Javadoc, Backend Guide, Planning, and Example review without
  rerunning stable Java tests.
- Documentation impact: Public API and CPU backend guide now distinguish the current inward CPU
  copy SPI from planned ordinary Engine materialization and explain layout traversal, encoding,
  byte limits, failures, ownership, lifetime, concurrency, and unsupported adjacent behavior.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [layout glossary](../../../../glossary.md#layout); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Javadoc review: all three affected production contracts now give complete parameter, return,
  failure, ownership, lifecycle, raw-bit, layout, and concurrency details. Generated output was
  inspected, and comment-stripped hashes prove no executable-token change.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: added [`Canonical host snapshot`](../../../../glossary.md#canonical-host-snapshot) because this is a reusable new boundary term
  consumed by CPU 0010G and planned Engine 0004; the entry distinguishes detached canonical bytes
  from physical storage, transfer, conversion, synchronization, and cross-backend materialization.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- No-change conclusions: `ARCHITECTURE.md`, focused architecture pages, ADRs, module dependencies,
  Gradle files, shared Runtime/Prepare/Model/Compiler contracts, Engine source/specifications,
  provider/generated-code paths, architecture tests, backend conformance, integration tests, and
  other modules require no change because 0010G realizes the existing CPU-owned physical-copy and
  Engine-composition boundary. Runtime API and Runtime 0015 records remain accurate and were
  preserved. The Engine master plan remains unedited under this pass's strict path limit: its
  0004 row remains deliberately Blocked, and its subsequent clean planning pass owns refreshing
  the prerequisite prose while creating the detailed Ready task.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Unresolved issues: None within [CPU](../../../../glossary.md#term-cpu) 0010G.
- Follow-up required: None for CPU 0010G. [Engine](../../../../glossary.md#term-engine) 0004 [planning](../../../../glossary.md#term-planning) remains separate work.

Status: Complete
