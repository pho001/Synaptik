# Task 0010G: Canonical Caller-Owned Host Snapshot Export

## Status

Complete

## Goal

Add the smallest supported CPU integration operation that copies one CPU publication
representation, borrowed from an open Runtime `RunResult`, through its exact resolved logical
`TensorDescriptor` into a fresh caller-owned `byte[]`.

The returned payload is independent of source layout and storage lifetime: logical elements appear
in canonical row-major coordinate order, and every multi-byte element uses big-endian byte order.
The operation supports all six current Model data types and exposes no CPU-private type, storage
lease, arena, `MemorySegment`, Runtime coordinate, Engine type, or graph identity.

## Scope

- Add exactly this public instance method to `CpuBackendIntegration`:

  ```java
  public byte[] copyToCanonicalHostBytes(
          BufferRepresentation representation,
          TensorDescriptor descriptor,
          long maximumBytes)
  ```

  The method is supported Engine-facing SPI, not an ordinary application facade. It returns a new
  mutable array owned exclusively by the caller; CPU retains neither the array nor a view of it.
- Delegate from the supported adapter to the existing CPU-private composition owner, preserving
  the adapter's open-lifetime guard and keeping physical access under `.internal`.
- Add one focused CPU-private host-snapshot exporter under `internal.memory`. It accepts only the
  two current supported CPU buffer implementations, `CpuBorrowedBuffer` and `CpuNativeBuffer`,
  through the nominal public `BufferRepresentation` boundary. It must reject an arbitrary
  `BufferRepresentation` and an external or test subclass of `CpuBufferRepresentation` rather
  than trusting a cast or nominal marker.
- Interpret the supplied descriptor as the logical view of that exact publication occurrence.
  Validate every intrinsic fact that current CPU representations can prove: current supported
  concrete representation class, open/current-thread-accessible storage, exact CPU data type,
  complete-element byte geometry, direct carrier compatibility, and sufficient physical element
  capacity for the descriptor's referenced span. Current representations do not retain Shape,
  layout, publication, or graph provenance, so identity of the descriptor/representation pairing
  is intentionally supplied by the future Engine caller and cannot be re-proved here.
- Require a fully static Shape and a present resolved `LayoutDescriptor`. Use the descriptor's
  non-negative element offset and strides exactly. Do not infer a default layout, repair a
  descriptor, or silently substitute dense geometry.
- Traverse logical coordinates in row-major order: the final axis changes fastest. For coordinate
  `c`, read source element index
  `storageOffset + sum(c[axis] * stride[axis])` with checked `long` arithmetic. Rank zero reads
  exactly the element at `storageOffset`. A Shape with any zero extent has zero logical elements,
  performs no carrier access, and returns a fresh empty array.
- Accept positive and zero source strides. Repeated physical addresses, including broadcast
  zero-stride layouts, are valid reads and produce repeated canonical logical elements. Negative
  strides are unsupported by the current Model `LayoutDescriptor` constructor and must never be
  manufactured or separately interpreted. Source injectivity and source/source overlap are not
  preconditions because export is read-only; the detached destination is dense and injective.
- Compute `elementCount` from the fully static Shape with the Model's zero-before-product rule.
  Compute canonical byte count as `Math.multiplyExact(elementCount,
  descriptor.dataType().byteWidth())`. A non-zero overflow is an `ArithmeticException`; a
  zero-element Shape produces zero bytes even when unused other extents would otherwise overflow.
- Require `maximumBytes >= 0`. Reject when canonical byte count exceeds `maximumBytes`, before
  allocation or source access. Independently reject a canonical byte count greater than
  `Integer.MAX_VALUE`, the explicit portable JVM `byte[]` index/length ceiling for this API,
  before allocation or source access. A lower implementation-specific allocation limit or memory
  exhaustion may still surface as the JVM's original `OutOfMemoryError`.
- Define canonical output encoding exactly:

  | Data type | Bytes per logical element | Canonical bytes |
  |---|---:|---|
  | `FLOAT64` | 8 | Big-endian `Double.doubleToRawLongBits(value)`; preserve every raw payload, including NaN payload/sign, infinities, and signed zero. |
  | `FLOAT32` | 4 | Big-endian `Float.floatToRawIntBits(value)`; preserve every raw payload, including NaN payload/sign, infinities, and signed zero. |
  | `BFLOAT16` | 2 | Big-endian unsigned view of the exact stored 16-bit `short`; no widening, rounding, canonicalization, or NaN conversion. |
  | `INT64` | 8 | Big-endian two's-complement 64-bit pattern. |
  | `INT32` | 4 | Big-endian two's-complement 32-bit pattern. |
  | `BOOL` | 1 | The exact canonical byte `0` or `1`; any other stored byte fails instead of being normalized. |

  CPU storage access remains native-order represented-value access, matching current CPU
  kernels; canonical output encoding is always big-endian and therefore platform-independent.
- Read segment-backed elements with the current CPU convention: the corresponding unaligned Java
  primitive `ValueLayout` in native byte order, and `JAVA_BYTE` for BOOL. Read observable primitive
  arrays through their matching carrier and carrier-relative byte offset. Require all address,
  element-to-byte, array-index, and destination-index conversions to be checked.
- Add a proved dense fast path for a `DENSE_CONTIGUOUS`, zero-offset descriptor. It may use a
  simple linear source-address loop and a raw bulk copy only where raw bytes are already identical
  to the canonical contract. Every type must still use the same raw-bit/canonical-BOOL rules as
  general traversal. The general logical odometer remains the semantic oracle. No benchmark gate
  is required because this is cold host export, outside generated kernels and Runtime execution.
- Validate all non-racy metadata and capacity preconditions before allocating or reading source
  elements. The new array remains private until successful return, so any validation, access,
  allocation, or copy failure exposes no partial payload. Do not cache, publish, retain, close,
  mutate, transfer, or change validity/ownership of the source representation.
- Keep the call synchronous. `CpuBackendIntegration` may serve independent calls concurrently,
  but one call adds no lock around caller/Runtime storage. The caller must keep the adapter and
  publication lease open, keep the representation accessible to the calling thread, prevent
  source mutation, and prevent result/representation closure for the complete call. Concurrent
  mutation has no atomic-snapshot guarantee; racing closure/access may surface the original JDK or
  CPU access failure. Future Engine 0004 owns serialization against result closure.
- Add focused CPU tests and finalize the supported-API guide, CPU backend guide, glossary impact,
  Javadocs, and planning evidence through the mandatory clean documentation-focused pass.

### Validation and failure precedence

The implementation and Javadoc must preserve this deterministic order for conditions CPU can
check without racing external closure:

1. reject a closed `CpuBackendIntegration` with the existing `CPU backend integration is closed`
   failure before inspecting arguments;
2. reject null `representation`, then null `descriptor`;
3. reject a negative `maximumBytes`;
4. reject a non-static Shape, then an unresolved layout;
5. compute the checked logical element count and canonical byte count;
6. reject the caller byte limit, then the `Integer.MAX_VALUE` array ceiling;
7. reject an unsupported concrete representation;
8. reject a closed or current-thread-inaccessible CPU representation;
9. reject representation/descriptor data-type disagreement, incomplete-element representation
   geometry, or insufficient capacity/span, in that order;
10. obtain and validate the already-classified direct CPU argument before allocating the result;
11. allocate, traverse, validate each BOOL byte at encounter time, and encode.

Stable messages are required for the new deliberate failures:

- `maximumBytes must be non-negative: <value>`;
- `host snapshot requires a fully static shape: <shape>`;
- `host snapshot requires a resolved layout`;
- `canonical byte count exceeds maximumBytes: required=<required>, maximum=<maximum>`;
- `canonical byte count exceeds JVM byte[] limit: <required>`;
- `representation is not a supported CPU buffer representation`;
- `CPU representation is closed` for the existing closed-state guard, or
  `CPU representation is not accessible to current thread` after that guard succeeds;
- `CPU representation data type differs from descriptor: representation=<actual>, descriptor=<expected>`;
- `CPU representation byte size is not a complete element multiple`;
- `CPU representation capacity is smaller than descriptor span: capacity=<capacity>, required=<span>`;
- `BOOL representation contains non-canonical byte at logical index <index>: <unsignedValue>`.

Null failures are `NullPointerException`; closed/inaccessible lifecycle failures are
`IllegalStateException`; every new negative-limit, unsupported-shape/layout/representation,
type/geometry/capacity, byte-limit/array-ceiling, or BOOL-content failure is
`IllegalArgumentException`; checked numeric overflow is `ArithmeticException`. Public Javadoc
must document each applicable failure category.

Existing lower-level closed/accessibility or checked-arithmetic failures retain their established
exception types and messages. A race after validation is explicitly outside deterministic
precedence.

## Out of scope

- Engine 0004 or any Engine production/test/documentation implementation; no later detailed task
  specification may be created here.
- A new public payload record, immutable byte wrapper, `ByteBuffer`, typed Java arrays, scalar
  decoding, `HostTensorStorage`, Tensor construction, or ordinary-user CPU API.
- Exposure of CPU internal representations, direct carriers, `MemorySegment`, `Arena`, storage or
  result leases, Runtime buffer/result indices, prepared slots, `ValueId`, `TensorId`, or graph
  identities.
- Runtime selection, publication lookup, representation transfer/materialization, fallback,
  validity mutation, ownership transfer, or cleanup.
- Dynamic or binding-dependent Shapes, unresolved layouts, negative strides, non-CPU
  representations, conversion between data types, BOOL truthiness normalization, or floating NaN
  canonicalization.
- Generated code, Class-File emitters, generator schema, operation IR, kernel specialization,
  route selection, OpenBLAS/provider behavior, worker orchestration, hot-path changes, tuning, or
  performance claims/benchmarks.
- Model, Compiler, Planning, Prepare, Runtime, Engine, Config, Trace, Backend Contract, provider,
  build/dependency, architecture-contract, ADR, architecture-test, backend-conformance, or
  integration-test changes.
- A cross-backend canonical-host-export contract. CPU 0010G is the demonstrated concrete CPU
  boundary needed by Engine; another backend may later require its own supported adapter.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially concrete-backend physical access,
  Engine composition, Runtime result leases, and dependency rules.
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0011: Per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../../planning-guide.md)
- [CPU master plan](../master-plan.md)
- [CPU 0005A partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [CPU 0005B universal access plans](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [CPU 0006 static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)
- [CPU 0008E bounded materialization](0008e-bounded-multi-input-materialization-and-representation-reuse.md)
- [CPU 0009G closure checkpoint](0009g-final-support-correctness-hygiene-and-inventory-checkpoint.md)
- [CPU 0010F lifecycle integration](0010f-supported-cpu-lifecycle-integration-adapter.md)
- [Runtime 0015 leased publication access](../../../modules/runtime/tasks/0015-leased-publication-representation-access.md)
- [Engine 0003 typed binding and result metadata](../../../modules/engine/tasks/0003-typed-logical-input-binding-and-published-result-access.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Concrete CPU owns physical representation compatibility, access, and copying. Runtime exposes
  only the borrowed representation while its result lease is open; Engine later pairs that exact
  occurrence with its final descriptor and owns the ordinary host-value facade.
- Engine remains the composition root and ordinary API owner. CPU must not depend on Engine, and
  no CPU type appears in the future ordinary Engine signature.
- The new operation is cold result export after Runtime publication, not prepared execution work.
  It does not enter a generated kernel, `PreparedSchedule`, `RunState`, or Runtime hot path.
- Current Model layout geometry is authoritative: static extents, non-negative strides and offset,
  referenced span, rank-zero scalar behavior, and zero-element behavior are consumed rather than
  redefined.
- Current CPU representations provide enough intrinsic metadata to prove concrete CPU class,
  type, byte-capacity, carrier, liveness, and current-thread access compatibility. They do not
  contain descriptor or publication provenance. The future Engine call site must obtain the
  representation and descriptor from the same exact publication occurrence while its lease is
  held; CPU 0010G must not add a graph/runtime identity solely to re-prove that pairing.
- No module boundary, dependency, shared contract, or architecture rule changes. If one becomes
  necessary, stop and report rather than expanding this task.

## Package impact

Existing supported package changed:

- `io.github.pho001.synaptik.backend.cpu` — add the one host-copy operation to the existing
  Engine-facing `CpuBackendIntegration`; add no constructor or ordinary-user type.

Existing CPU-private packages changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — the existing
  composition owner delegates to the focused exporter and preserves lifecycle precedence.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — add one stateless focused exporter
  that owns checked CPU representation access, logical-layout traversal, and canonical encoding.

Test packages:

- `io.github.pho001.synaptik.backend.cpu.spi` — distinct-package supported-surface and black-box
  behavior without `.internal` imports.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — exact native/borrowed carrier,
  validation, traversal, encoding, and failure evidence.

No package or module is added.

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

Planning/status files:

- `docs/planning/backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum scope

This task may create or modify at most thirteen paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| CPU production | 3 | Two existing delegation/API paths plus one new focused exporter |
| CPU tests | 4 | Two existing tests plus two focused new tests |
| Explanatory documentation | 3 | Public API, CPU backend guide, and conditional glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **13** | **3 + 4 + 3 + 3** |

If another production, test, documentation, build, or architecture path is required, stop and
revise the plan before implementation. An unchanged conditional glossary path need not be spent;
unused capacity is not headroom for unrelated work.

## Acceptance criteria

- `CpuBackendIntegration` remains public, final, privately constructed, and `AutoCloseable`; its
  public declarations are exactly the existing seven plus
  `copyToCanonicalHostBytes(BufferRepresentation, TensorDescriptor, long)`. No public/protected
  signature or Javadoc exposes `.internal`, `MemorySegment`, an Engine type, or a new CPU payload.
- A distinct-package test compiles and calls the new method using only supported public CPU,
  Model, and Runtime types. Ordinary users still obtain CPU through Engine and are not instructed
  to construct the integration SPI directly.
- All six current data types are exhaustively covered on matching heap carriers and segment-backed
  representations. Tests lock big-endian canonical encoding, raw FLOAT64/FLOAT32 NaN payloads and
  signed zeros, raw BFLOAT16 NaN/zero/infinity/ordinary patterns, signed integral boundaries, and
  exact BOOL `0`/`1` bytes.
- Canonical bytes follow row-major logical coordinates for dense, non-zero-offset dense,
  positive-strided, zero-stride/broadcast, rank-zero, and zero-element descriptors. General
  traversal and the dense fast path produce identical bytes for the same logical values.
- Static/resolved preconditions, checked element/byte/address arithmetic, caller limit, JVM array
  ceiling, complete-element byte geometry, representation type/capacity/carrier/accessibility,
  and descriptor span are enforced in the specified order before source element access.
- A descriptor span equal to capacity succeeds. A larger span fails. Zero-element descriptors
  accept zero capacity and perform no source read regardless of their otherwise unused offset.
  Extra representation capacity is permitted and never exported.
- Positive and zero strides are accepted. Repeated source addresses are read once per logical
  occurrence and repeated in the result. No source injectivity or overlap restriction is added.
  No negative stride can enter through the supported Model descriptor.
- An arbitrary nominal `BufferRepresentation` and a non-supported subclass of the internal base
  are rejected with the stable unsupported-representation failure. The exact current borrowed and
  native CPU representations are accepted while open and accessible.
- BOOL validation fails on the first non-canonical logical value in row-major encounter order and
  reports its logical index and unsigned byte. It never normalizes another byte to true.
- Every successful call returns a fresh array. Repeated calls and aliased publication occurrences
  are neither cached nor deduplicated. Mutating one returned array cannot affect source storage or
  another returned array.
- Export does not close, retain, mutate, transfer, or change validity/ownership of the source and
  performs no provider, route, worker, schedule, graph, or Runtime operation. Result/adapter
  closure after return cannot affect the detached bytes.
- Concurrent calls over independently valid representations succeed without shared mutable
  exporter state. Tests and Javadocs state that callers must prevent concurrent source mutation or
  closure and that Engine 0004 will serialize materialization against result closure.
- Any failure returns no array and exposes no partial payload. JVM/JDK failures caused by a genuine
  allocation/access race propagate without translation.
- No generated source/emitter/schema, OpenBLAS/provider behavior, module dependency, build file,
  shared Java contract, architecture test, backend conformance, or integration test changes.
- The existing exhaustive production-source allowlist in `CpuInternalPackageInventoryTest`
  receives exactly `internal/memory/CpuHostSnapshotExporter.java`. Its assertion remains a fixed
  exhaustive inventory; do not weaken it or replace it with dynamic discovery.
- A separate clean documentation-focused pass finalizes Javadocs, the two explanatory guides,
  glossary impact, links, examples, and planning evidence before the task becomes `Complete`.

## Tests / validation

Implementation context `01a0a4db-17b7-76c3-99ce-5c5358b14586` already ran the focused command
after the six planned Java paths stabilized; all 14 focused tests passed. Preserve that evidence
and do not rerun the focused tests for the one-line exhaustive-inventory correction. Add exactly
`internal/memory/CpuHostSnapshotExporter.java` to the existing fixed expected set in
`CpuInternalPackageInventoryTest`, then rerun only the full CPU suite:

```bash
./gradlew :backends:cpu:test
```

Focused tests must cover the complete acceptance matrix, including all six types; heap and
segment carriers; borrowed and native CPU representations; dense/general equivalence; offset,
positive/zero strides, scalar, and empty Shapes; raw floating/BFLOAT16 patterns; BOOL rejection;
every validation-precedence edge; exact-limit/equal-capacity success; overflow, limit, JVM ceiling,
capacity, wrong-type, unsupported-representation, closed, and inaccessible failures; fresh-array
identity; no source mutation/closure; post-copy independence; and concurrent independent calls.

Documentation-focused context reuses the successful CPU test evidence unless it changes
executable Java, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also records:

- `javap -public` confirmation of the exact eight-method `CpuBackendIntegration` surface and no
  `.internal`, `MemorySegment`, Engine, storage-lease, or new payload type in public descriptors;
- successful compilation of a distinct-package fixture against declared dependencies;
- Markdown local-link/anchor, heading-order, balanced-fence, trailing-whitespace, and final-newline
  checks for the six changed Markdown files, including the glossary entry justified by the
  documentation pass;
- exact at-most-thirteen-path allowlist and confirmation that Java changes are CPU-only;
- scans proving no Engine import/dependency, build change, generated-code/schema edit, Runtime
  selection, reflection, service locator, provider/tuning work, or new ordinary CPU construction;
- source inspection confirming native-order unaligned source access, big-endian output, checked
  traversal/address arithmetic, limit-before-access behavior, and no source ownership/validity
  mutation; and
- synchronized CPU task/master/roadmap status, Engine 0004 still `Blocked`, no Engine 0004 or
  later detailed task specification, and preservation of every pre-existing unrelated worktree
  change.

Repository-wide validation is deferred to Engine 0008/CI. Backend conformance is not run because
this is a CPU-specific export SPI rather than a shared backend behavior contract. Architecture and
integration tests are not run because no dependency, module boundary, shared architecture rule,
or end-to-end ordinary Engine behavior changes.

## Dependencies

- [CPU 0010F](0010f-supported-cpu-lifecycle-integration-adapter.md) — Complete; supplies the
  supported lifecycle owner and current checked CPU representation implementations.
- [Runtime 0015](../../../modules/runtime/tasks/0015-leased-publication-representation-access.md) —
  Complete; supplies exact borrowed publication-representation access while the result lease is
  open.
- [Engine 0003](../../../modules/engine/tasks/0003-typed-logical-input-binding-and-published-result-access.md) —
  Complete; supplies exact ordered publication metadata and final resolved descriptors for the
  later Engine caller.
- Current Model `DataType`, `Shape`, `Dimension`, `LayoutDescriptor`, and `TensorDescriptor`
  contracts, plus current CPU representation and carrier/access conventions.

All dependencies are complete and every contract decision required for implementation is fixed.

## Follow-up tasks

- Engine 0004 remains `Blocked` pending its own subsequent clean planning pass. CPU 0010G now
  satisfies its CPU prerequisite, but this task does not create or edit the Engine specification
  for lazy per-publication `HostTensorValue` materialization.
- Engine 0005 and later work remain Draft. This task creates no later specification and does not
  advance optional CPU vendor/tuning rows.

## Architecture impact

Expected impact: None.

This task realizes the existing ownership split: Runtime lends the nominal publication
representation, CPU validates and reads its concrete physical storage, and future Engine code
will own publication selection, close coordination, and the ordinary immutable host payload. No
dependency or module boundary changes. If implementation requires a shared API, Engine
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

- The return value is a detached `byte[]`; no CPU-owned payload type is justified because future
  Engine owns the immutable `HostTensorValue` wrapper and can defensively encapsulate these bytes.
- The method name states both action and format. `copy` avoids implying a live view; `Canonical`
  and `HostBytes` distinguish logical row-major big-endian output from raw CPU storage bytes.
- Big-endian is the fixed canonical external byte order already selected by the Engine master
  plan. Native order remains only the source-representation interpretation.
- BOOL fails on non-canonical storage. Normalization would hide a corrupt CPU representation and
  would not preserve the exact current CPU BOOL contract.
- A dense zero-offset linear fast path is included; general logical traversal remains the oracle.
  No platform-dependent multi-byte raw-copy shortcut is required.
- Current CPU metadata proves safe physical compatibility but not descriptor provenance. Adding
  provenance or Runtime/graph identity to CPU storage would exceed the smallest boundary and
  duplicate the exact pairing that future Engine already owns.
- No backend-conformance row is added because the canonical bytes are a CPU adapter contract, not
  a shared backend SPI. No benchmark gate applies to cold caller-requested export.
- The existing exhaustive production-source inventory is part of this capability's test scope.
  Adding its one required literal exporter entry preserves the assertion's fixed fail-closed
  purpose; weakening it or deriving its expected set dynamically would remove that protection.

## Known limitations

- Only fully static, resolved descriptors and the two current supported CPU representation
  implementations are accepted.
- The result is bounded by `Integer.MAX_VALUE` bytes and available heap. There is no streaming,
  chunked, mapped, or caller-supplied destination form.
- The method does not synchronize source mutation or closure. Future Engine 0004 must serialize
  access to one result lease; callers at this integration SPI must coordinate directly.
- The supplied descriptor/representation publication pairing is trusted at this layer because
  current CPU buffers intentionally carry no publication or graph provenance.

## Validation evidence

Planning context: this clean CPU 0010G planning pass.

- Inspected current Runtime 0015 source, Javadocs, tests, and completion record: publication access
  returns the exact borrowed `BufferRepresentation` only while the result lease is open, checks
  closure before index, and performs no copy or physical access.
- Inspected current CPU integration and representation source/tests: the supported adapter has
  seven public methods; `CpuBorrowedBuffer` and `CpuNativeBuffer` are the current concrete buffer
  implementations; their base retains exact `DataType`, byte size, segment, accessibility, and a
  once-classified array/segment argument, but no descriptor or publication provenance.
- Inspected Model contracts: exactly six data types exist; Shapes admit rank zero and zero extents;
  resolved layouts permit only non-negative strides/offset, calculate checked referenced span,
  and allow zero-stride views; `TensorDescriptor` reconstructs a present layout against its Shape.
- Inspected CPU affine/access/copy evidence: current CPU represented-bit movement preserves raw
  FLOAT64/FLOAT32/BFLOAT16/integral bits and canonical BOOL, uses native-order unaligned segment
  access, supports positive/zero strides, and treats zero-element access as empty.
- Inspected Engine planning decisions: future host values require big-endian canonical row-major
  bytes, a caller byte limit, fresh uncached snapshots, and close/materialization serialization.
- No architecture, dependency, shared-contract, or unresolved API decision was found. Runtime
  0015, CPU 0010F, and Engine 0003 are Complete, so this task is Ready.
- Java tests were not run, as required for this planning-only pass.
- Scope-correction evidence: implementation context `01a0a4db-17b7-76c3-99ce-5c5358b14586`
  completed the six originally planned Java paths and its focused command passed 14/14 tests. Its
  full CPU suite ran 1,010 tests with exactly one failure and no other reported failure: the
  existing exhaustive `CpuInternalPackageInventoryTest` expected-source set omitted the required
  new `internal/memory/CpuHostSnapshotExporter.java`. The actual and expected inventories otherwise
  matched. This correction therefore adds that existing test as the fourth CPU test path, raises
  the task ceiling from 12 to 13, and authorizes only the one literal expected entry; it does not
  weaken or dynamically derive the inventory assertion.
- Final implementation evidence reused from clean context
  `01a0a4db-17b7-76c3-99ce-5c5358b14586`: the focused CPU host-snapshot/public matrix passed
  14/14 tests; after the exact inventory correction, the final
  `./gradlew :backends:cpu:test` ran 1,010 tests with zero failures, zero errors, and 28 skips.
  That context also passed the exact seven-Java-path and `git diff` checks. Documentation context
  `01a0a4ec-8d2e-7a41-a944-cdcc1ca43802` changed no executable Java token and therefore did not
  rerun either successful Java test command.
- Documentation context `01a0a4ec-8d2e-7a41-a944-cdcc1ca43802` independently reviewed the
  architecture contract; current architecture plan; documentation rules and General,
  API/Javadoc, Backend Guide, Planning, and Example profiles; planning guide and roadmap; CPU,
  Runtime 0015, and Engine 0003 plans; all seven task-owned Java paths; the public/CPU APIs;
  glossary; generated Javadocs; and the final diff.
- `./gradlew :backends:cpu:javadoc` completed successfully. The first sandboxed invocation could
  not create the Gradle wrapper lock under the user cache; the permitted rerun completed with the
  two expected incubating-Vector notices and existing unrelated missing-`@param` warnings in
  `CpuPartitionLowering.LoweredPartition` and `CpuPartitionPreparationPlan`. No warning names a
  0010G path. Generated pages for `CpuBackendIntegration`, `CpuBackendComposition`, and
  `CpuHostSnapshotExporter` were inspected and contain the final ownership, layout, encoding,
  size, failure, and concurrency contracts.
- `javap -classpath backends/cpu/build/classes/java/main -public
  io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration` reports exactly eight public
  methods: `open`, `capabilityProvider`, `availabilitySnapshot`, `preparations`,
  `scheduleAssembler`, `borrow`, `copyToCanonicalHostBytes`, and `close`. Its descriptors expose
  no `.internal`, `MemorySegment`, Engine, storage-lease, or new payload type.
- A distinct-package fixture compiled successfully with `javac` against the declared CPU, Model,
  Runtime, Backend Contract, Compiler, Prepare, Planning, Config, Trace, and OpenBLAS-provider
  output dependencies. The fixture imports only supported CPU, Model, and Runtime types and calls
  the exact three-argument copy method.
- Comment-stripped SHA-256 values for the three production files were unchanged before and after
  documentation finalization: `0294b776cea786c840bfbf846ba8238e56f0ba3000a3c9791ae24721c3ac7f24`
  for `CpuBackendIntegration`, `bbda20b3131a3c4953a63596ebcef1f61c49c4151aefcf9e2c1e846137e82ee4`
  for `CpuBackendComposition`, and
  `d5f69e71271c84d7119c828388f5680f4a2dac4c5e9c750ee997c0f21cd78adb`
  for `CpuHostSnapshotExporter`. This proves the documentation pass changed Javadoc only in Java.
- Source/mechanism inspection confirmed exact-class acceptance for `CpuBorrowedBuffer` and
  `CpuNativeBuffer`, native-order unaligned source reads, fixed big-endian output, raw floating
  and BFLOAT16 bits, strict BOOL, checked count/address arithmetic, caller-limit and JVM-ceiling
  checks before representation access, resolved offsets with positive/zero strides, and no
  source close, validity mutation, ownership transfer, Runtime selection, provider action, or
  generated-code path.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/public-api.md
  docs/backend-guide/cpu-backend.md docs/glossary.md
  docs/planning/backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md
  docs/planning/backends/cpu/master-plan.md docs/planning/roadmap.md` passed all six changed
  Markdown paths for local targets, effective anchors, balanced fences, LF/final newlines, and
  trailing whitespace. The separate heading-order scan also passed.
- The final task-attributable union is exactly 13 paths: three CPU production Java files, four CPU
  test files, three explanatory documents, and three planning/status documents. Java changes are
  exactly the seven CPU paths. Scans found no Engine import or dependency, build edit, generated
  source/schema edit, provider change, Runtime selection, service locator, production reflection,
  ordinary CPU construction guidance, architecture-test source, backend-conformance source, or
  integration-test source change.
- CPU task 0010G, the CPU master row, and the roadmap now agree on `Complete`. Runtime 0015 remains
  `Complete`; Engine 0004 remains `Blocked` pending its own subsequent clean planning pass;
  Engine 0005–0008 remain `Draft`; and no Engine 0004 or later detailed task specification exists.
  The pre-existing Runtime 0015 and Engine/Runtime frontier files remain otherwise preserved.
- Final `git diff --check` passed.

## Implementation notes

Implementation context `01a0a4db-17b7-76c3-99ce-5c5358b14586` implemented the exact three
production and four test paths, including the corrected fixed exhaustive inventory entry. Its
focused and final CPU suites passed. Clean documentation context
`01a0a4ec-8d2e-7a41-a944-cdcc1ca43802` retained the implementation, finalized Javadocs and the
public API/CPU guide/glossary boundary, synchronized task/master/roadmap status, and completed the
non-executable validation without changing Java tokens.

## Completion summary

- Completed changes: added the supported Engine-facing CPU snapshot-copy SPI, CPU-private
  delegation/exporter, exact validation and six-type canonical encoding, focused tests, and the
  fixed exhaustive production-source inventory entry.
- Files changed or created: exactly the 13 paths authorized under [Maximum scope](#maximum-scope):
  three CPU production files, four CPU tests, Public API, CPU backend guide, glossary, this task,
  CPU master plan, and roadmap.
- Tests and validation: reused implementation context
  `01a0a4db-17b7-76c3-99ce-5c5358b14586`'s passing 14/14 focused tests and final 1,010-test CPU
  suite with zero failures/errors and 28 skips. Documentation context passed CPU Javadoc,
  generated-page inspection, exact eight-method `javap`, distinct-package fixture compilation,
  executable-token preservation, six-file Markdown validation, heading/scope/import/dependency/
  mechanism/status checks, and final `git diff --check`.
- Documentation-agent review: clean context `01a0a4ec-8d2e-7a41-a944-cdcc1ca43802` completed the
  mandatory independent General, API/Javadoc, Backend Guide, Planning, and Example review without
  rerunning stable Java tests.
- Documentation impact: Public API and CPU backend guide now distinguish the current inward CPU
  copy SPI from planned ordinary Engine materialization and explain layout traversal, encoding,
  byte limits, failures, ownership, lifetime, concurrency, and unsupported adjacent behavior.
- Javadoc review: all three affected production contracts now give complete parameter, return,
  failure, ownership, lifecycle, raw-bit, layout, and concurrency details. Generated output was
  inspected, and comment-stripped hashes prove no executable-token change.
- Glossary impact: added `Canonical host snapshot` because this is a reusable new boundary term
  consumed by CPU 0010G and planned Engine 0004; the entry distinguishes detached canonical bytes
  from physical storage, transfer, conversion, synchronization, and cross-backend materialization.
- No-change conclusions: `ARCHITECTURE.md`, focused architecture pages, ADRs, module dependencies,
  Gradle files, shared Runtime/Prepare/Model/Compiler contracts, Engine source/specifications,
  provider/generated-code paths, architecture tests, backend conformance, integration tests, and
  other modules require no change because 0010G realizes the existing CPU-owned physical-copy and
  Engine-composition boundary. Runtime API and Runtime 0015 records remain accurate and were
  preserved. The Engine master plan remains unedited under this pass's strict path limit: its
  0004 row remains deliberately Blocked, and its subsequent clean planning pass owns refreshing
  the prerequisite prose while creating the detailed Ready task.
- Unresolved issues: None within CPU 0010G.
- Follow-up required: None for CPU 0010G. Engine 0004 planning remains separate work.

Status: Complete
