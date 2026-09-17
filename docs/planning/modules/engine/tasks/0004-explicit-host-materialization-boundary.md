# Task 0004: Explicit Host Materialization Boundary

## Status

Complete

## Goal

Add the minimal ordinary Engine API that lazily materializes one selected publication occurrence
from an open `RunResult` into a detached immutable `HostTensorValue`. The caller selects the exact
`RunResult.Publication` object and supplies a non-negative byte limit. The result carries logical
type and Shape metadata plus canonical row-major big-endian bytes and remains readable after the
originating result or Engine closes.

This task composes two completed inward capabilities without widening them: Runtime 0015 lends the
exact result-indexed representation while its lease is open, and CPU 0010G validates that CPU
representation against the final descriptor and copies it into fresh canonical bytes.

## Required reading

Read these files and contracts in the clean implementation context rather than relying only on
this specification:

- root `AGENTS.md`, [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), and the
  [current architecture plan](../../../../architecture/current-architecture-plan.md);
- the [planning guide](../../../planning-guide.md), [roadmap](../../../roadmap.md),
  [Engine master plan](../master-plan.md), and Engine task
  [0003](0003-typed-logical-input-binding-and-published-result-access.md);
- Runtime task
  [0015](../../runtime/tasks/0015-leased-publication-representation-access.md), its final
  `runtime.run.RunResult`, and its result/state/publication tests and Javadocs;
- CPU task
  [0010G](../../../backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md), final
  `CpuBackendIntegration.copyToCanonicalHostBytes(...)`, the CPU-private implementation as
  evidence only, and its public/private tests and Javadocs;
- every current Engine production and test file plus the existing ordinary Engine integration
  test and focused Engine architecture test;
- Model `DataType`, `Shape`, `LayoutDescriptor`, and `TensorDescriptor` contracts; and
- the [documentation rules](../../../../developer-guide/documentation-rules.md), General,
  API/Javadoc, Planning, and Example profiles, and the affected API, backend-guide, and glossary
  sections identified below.

## Scope

- Add one Engine-owned immutable host-value type in the ordinary facade package.
- Add one explicit lazy per-occurrence materialization method to ordinary `RunResult`.
- Select only the exact publication object created by that same result and map its existing dense
  occurrence index directly to Runtime 0015.
- Validate the public byte budget and descriptor-derived logical counts before physical access.
- Serialize materialization against close and against another materialization on the same result.
- Delegate physical validation, logical-layout traversal, and six-type canonical encoding to CPU
  0010G through the existing Engine-owned composition seam.
- Add focused surface, semantic, ownership, failure, lifecycle, concurrency, and real CPU
  integration coverage.
- Finalize all affected Javadocs and explanatory documentation in a distinct clean
  documentation-focused pass in the same overall change.

### Exact ordinary public API

All ordinary types remain in `io.github.pho001.synaptik.engine`. These are complete declared
public method inventories for the changed/new types; inherited `Object` methods are not facade
methods.

```java
public final class RunResult implements AutoCloseable {
    public int resultCount();
    public List<RunResult.Publication> publications();
    public HostTensorValue materialize(
            RunResult.Publication publication,
            long maximumBytes);
    public boolean isClosed();
    public void close();
}

public final class HostTensorValue {
    public DataType dataType();
    public Shape shape();
    public long elementCount();
    public long byteSize();
    public ByteBuffer bytes();
}
```

`HostTensorValue` has exactly one constructor, with package-private visibility and signature
`HostTensorValue(DataType dataType, Shape shape, byte[] canonicalBytes)`. It is used only by Engine
implementation and defensively copies `canonicalBytes`. It has no public or protected constructor,
field, nested type, factory, close method, typed-array method, scalar decoder, descriptor accessor,
layout accessor, mutation method, or other declared public member. `RunResult` gains exactly the
one overload shown. `Engine`, `CompiledGraph`,
`PreparedExecution`, `RunResult.Publication`, `RunResult.Role`, and every advanced public type gain
no new ordinary public signature.

Ordinary signatures may name only Engine, Model, and JDK types. They expose no Runtime,
Prepare, Compiler, CPU, other backend, advanced handle, `.internal`, `BufferRepresentation`,
`MemorySegment`, arena, storage, slot, `ValueId`, or backend identity.

### Host value semantics

- `dataType()` is the exact final publication descriptor's non-null Model `DataType`.
- `shape()` is the exact immutable final publication descriptor's non-null `Shape` reference.
  `requiresGrad` and source layout are deliberately absent because the payload is a detached value
  in canonical logical order, not another Tensor descriptor or physical view.
- `elementCount()` is the checked fully static logical Shape count. Rank zero is one. Any zero
  extent produces zero before multiplying other extents. A non-zero product overflow remains
  `ArithmeticException`.
- `byteSize()` is the checked product of `elementCount()` and `dataType().byteWidth()` and equals
  the private byte-array length. It is a `long` even though this first payload is additionally
  bounded by `Integer.MAX_VALUE`.
- The package-private constructor rejects null arguments in declaration order with messages
  `dataType`, `shape`, and `canonicalBytes`; recomputes the checked static element/byte counts from
  the metadata; validates exact byte-array length with the same backend-count-mismatch failure
  specified below; and defensively clones its input. Constructor failures return no value.
- `bytes()` returns a fresh `ByteBuffer` object on every call. Each buffer has position zero,
  limit/capacity equal to `byteSize()`, explicit `ByteOrder.BIG_ENDIAN`, is read-only, exposes no
  mutable backing array, and views the immutable private clone. Buffer position/limit changes in
  one returned view do not affect the value or later views.
- Encoding is the CPU 0010G contract for all six current types: raw FLOAT64/FLOAT32 bits,
  represented BFLOAT16 bits, and two's-complement INT64/INT32 are big-endian; BOOL is exactly one
  byte `0` or `1`. Logical elements are dense row-major in the destination regardless of the
  resolved source layout.
- `HostTensorValue` deliberately keeps `Object` reference identity. It does not override
  `equals`, `hashCode`, or `toString`; equal metadata and bytes do not make two snapshots equal.
- It owns no closeable resource and retains no result, Engine, Runtime, CPU, representation,
  descriptor, storage, or arena reference. Every accessor remains usable concurrently and after
  result or Engine closure.

### Selection, mapping, and validation precedence

`RunResult.materialize(publication, maximumBytes)` must perform exactly this ordered protocol:

1. Enter the existing AdvancedEngine lifecycle admission before inspecting arguments. If Engine
   closure has begun, preserve `IllegalStateException("advanced engine is closed")` ahead of every
   result, selector, limit, descriptor, Runtime, or CPU failure.
2. Acquire the originating registered result owner's lifecycle monitor and hold it through the
   complete synchronous Runtime lookup, CPU copy, and `HostTensorValue` construction. If result
   closure has begun, throw `IllegalStateException("run result is closed")` before inspecting the
   publication or byte limit.
3. Reject null `publication` with `NullPointerException` and message `publication`.
4. Require the exact publication object's private originating-result reference to be this method
   receiver. Otherwise throw
   `IllegalArgumentException("publication does not belong to this run result")`. Equality of
   metadata, Tensor ID, descriptor, role, derivative order, target position, or index is never an
   ownership substitute. A publication from another run, including the same prepared execution or
   an equal-looking alias occurrence, is foreign.
5. Reject `maximumBytes < 0` with
   `IllegalArgumentException("maximumBytes must be non-negative: " + maximumBytes)`.
6. Using the selected publication's exact final descriptor, reject a non-static Shape with
   `IllegalArgumentException("host snapshot requires a fully static shape: " + shape)`, then an
   unresolved layout with
   `IllegalArgumentException("host snapshot requires a resolved layout")`.
7. Compute the Shape count and canonical byte count with the Model zero-before-product rule and
   checked `long` multiplication. Arithmetic overflow remains the original `ArithmeticException`.
8. Reject a canonical byte count above `maximumBytes` with CPU 0010G's exact
   `canonical byte count exceeds maximumBytes: required=<required>, maximum=<maximum>` message,
   then reject a count above `Integer.MAX_VALUE` with
   `canonical byte count exceeds JVM byte[] limit: <required>`.
9. Pass the publication's dense `index()` directly to the exact inward Runtime result's
   `publicationRepresentation(int)`. Do not search by Tensor ID, role, descriptor equality,
   publication equality, Runtime identity, or graph identity. Pass that exact returned object,
   the same publication descriptor, and the caller's same `maximumBytes` to the composition copy
   method exactly once.
10. Require a non-null backend byte array whose length exactly equals the already checked canonical
    byte count. Null fails with `NullPointerException("canonicalBytes")`; a wrong length fails with
    `IllegalStateException("backend canonical byte count does not match descriptor: expected=" +
    byteCount + ", actual=" + bytes.length)`. Construct the detached immutable value only after
    that check.

The Engine validations are deliberate defense at the ordinary boundary. CPU 0010G repeats the
descriptor/count/limit checks and remains authoritative for concrete CPU class, liveness,
accessibility, data type, element geometry, capacity/span, carrier access, BOOL validity, layout
traversal, and canonical encoding. Preserve all original unchecked CPU/JDK failures, including
`OutOfMemoryError`; do not translate them into a new exception family. A failed call returns no
partial `HostTensorValue`, does not cache bytes, and leaves the still-open result usable unless a
concurrent close subsequently owns cleanup.

### Lifecycle, concurrency, and failure protocol

- Reuse the existing AdvancedEngine lifecycle lock and active-operation count. Add one narrow
  package-private ordinary materialization entry; do not nest a public advanced call, add another
  Engine gate/registry, or add a generic callback/service abstraction.
- Admission covers validation, exact publication mapping, Runtime borrowed access, CPU copying,
  defensive payload construction, and return readiness. Unlike construction of a new Engine-owned
  handle or lease, an admitted materialization that began while Engine was open may complete and
  return its independent snapshot after another thread starts Engine closure. Engine close waits
  for that admission to finish before closing results and then the CPU composition.
- The registered result owner's existing monitor is the one per-result serialization point.
  Materialization holds it for the complete copy. Result close obtains the same monitor before
  marking closure begun, so a close that loses the monitor race waits for the admitted copy; a
  materialization that loses observes `run result is closed` and performs no inward access.
- Concurrent materializations on one result serialize, including calls for distinct occurrences.
  Materializations on different results may proceed concurrently under separate result monitors.
  Every successful call invokes CPU copying and constructs a fresh independent value; no result,
  publication, representation, alias, or byte cache is added.
- Maintain lock order: Engine admission is acquired before the result monitor; the result monitor
  is released before decrementing Engine active-operation state. Engine close waits for active
  operations with the Engine lock released and calls result close without holding that lock.
  Materialization must not acquire the Engine lifecycle lock while holding the result monitor.
- `AdvancedRunResult.close()` keeps its current once-only cleanup, wait-uninterruptibly,
  interrupt-restoration, exact retained failure, Runtime-result-first, reverse-borrow-wrapper,
  and unregister behavior. It adds only the monitor exclusion required above. No copy owns or
  closes the borrowed Runtime representation.
- A materialization failure is rethrown by exact unchecked object identity after Engine admission
  is released. It triggers no result/backend cleanup itself. A concurrent explicit or Engine close
  then performs its ordinary cleanup and reports its own retained cleanup failure to its close
  caller. Distinct cleanup failures must never replace or self-suppress a materialization failure.
- A completed value is entirely independent. Closing the result or Engine during or after return
  cannot invalidate its metadata or byte views.

### Delegation boundary

Extend the package-private `EngineBackendComposition` with exactly:

```java
byte[] copyToCanonicalHostBytes(
        BufferRepresentation representation,
        TensorDescriptor descriptor,
        long maximumBytes);
```

`CpuEngineBackendComposition` delegates it exactly once to its owned
`CpuBackendIntegration.copyToCanonicalHostBytes(...)`. The three existing Engine test
compositions implement the method explicitly; production has no default/fallback implementation.
This is inward package-private composition, so its Runtime and Model types do not escape the
ordinary API. Engine imports no CPU `.internal` type and performs no physical access, conversion,
layout traversal, backend switch, discovery, or transfer.

## Out of scope

- `Tensor.execute`, `Tensor.backward`, Engine one-shot forward convenience, scalar-objective
  backward convenience, implicit targets/seeds, mutable Tensor gradient state, or Engine tasks
  0005–0008.
- A cross-backend host-materialization abstraction, shared backend SPI, backend selection,
  multi-backend schedule composition, transfer, fallback, or another concrete backend.
- Dynamic or binding-dependent Shapes, unresolved layouts, layout inference, conversion,
  normalization, physical-span exposure, typed arrays, scalar decoding, streaming, chunking,
  caller-supplied destinations, caching, mapped output, or `ByteBuffer`-backed ownership.
- A public descriptor, Runtime result/index, representation, storage, `MemorySegment`, arena,
  backend, slot, `TensorId` lookup, role lookup, `ValueId`, graph identity, alias query, or
  individual representation lease.
- Changes to Model, Compiler, Planning, Prepare, Runtime, CPU implementation, Config, Trace,
  Backend Contract, build files/dependencies, architecture rules/docs, ADRs, generated code,
  provider/tuning code, backend conformance, or unrelated integration behavior.
- Creating a detailed Engine 0005-or-later task specification.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Engine composition, Runtime
  result/resource ownership, concrete-backend physical access, lifecycle, and dependency rules.
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)

## Architecture constraints

- Engine remains the outer public lifecycle/composition owner. It selects the exact ordinary
  publication occurrence, coordinates close, enforces the caller byte budget defensively, and
  owns the immutable outward payload.
- Runtime lends only its retained nominal representation under the exact whole-state result lease.
  It gains no Engine/CPU knowledge and performs no host copy or lifetime extension.
- CPU owns all concrete representation compatibility and physical copying. Engine uses only the
  supported root-package integration method; CPU remains independent of Engine.
- No dependency direction, module boundary, authoritative architecture rule, shared Runtime/CPU
  contract, or build structure changes. If implementation needs one, stop and report the exact
  conflict rather than editing outside this task.

## Package impact

Existing public package changed:

- `io.github.pho001.synaptik.engine` remains the deliberate ordinary facade and gains the one
  immutable `HostTensorValue` plus the materialization method on its existing `RunResult`.

Existing package-private composition in that package gains only the exact CPU-copy delegation
described above. No package or module is added or moved.

Type placement:

- `io.github.pho001.synaptik.engine.HostTensorValue` — Engine owns the ordinary detached result
  payload and lifecycle-independent public metadata/byte view.
- `io.github.pho001.synaptik.engine.RunResult` — already owns the ordered publication occurrences
  and is therefore the only ordinary object that can authenticate an exact occurrence and bound
  access to its open lease.
- `io.github.pho001.synaptik.engine.AdvancedRunResult` and `AdvancedEngine` — remain private
  lifecycle machinery for the existing admission/result gate; no advanced public method is added.

## Affected files

Expected Engine production/Javadoc paths (eight):

1. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/HostTensorValue.java` (new).
2. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RunResult.java`.
3. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedRunResult.java`.
4. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`.
5. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineBackendComposition.java`.
6. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/CpuEngineBackendComposition.java`.
7. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java` (Javadoc only).
8. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java` (Javadoc only).

Expected test paths (five):

9. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java`.
10. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineStandardCompositionTest.java`
    (implement the expanded fake composition contract and retain existing tests).
11. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/AdvancedEngineLifecycleTest.java`
    (implement the expanded fake composition contract and retain advanced behavior).
12. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`.
13. `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineTypedLifecycleIntegrationTest.java`.

Expected explanatory documentation paths finalized by the clean documentation pass (six):

14. `docs/api/public-api.md`.
15. `docs/api/compile-api.md`.
16. `docs/api/runtime-api.md`.
17. `docs/api/training-api.md`.
18. `docs/backend-guide/cpu-backend.md`.
19. `docs/glossary.md`.

Planning/status paths (four):

20. This task.
21. `docs/planning/modules/engine/master-plan.md`.
22. `docs/planning/modules/runtime/master-plan.md`.
23. `docs/planning/roadmap.md`.

Review without modification: `ARCHITECTURE.md`, focused architecture/ADR files, Model descriptor
contracts, Runtime 0015 source/tests/docs, CPU 0010G source/tests/docs, Engine build files,
architecture tests, backend-conformance tests, and other modules. Record reasoned no-change
conclusions.

## Maximum scope

This task may create or modify at most the 23 paths listed above: eight Engine production/Javadoc,
five focused tests, six explanatory documents, and four planning/status documents. This exceeds
the normal guardrail because the cohesive public boundary necessarily changes the result/payload,
existing shared lifecycle owner, required composition collaboration and all three compile-time
fake implementations, plus one real CPU integration fixture and every currently stale API status
reference. It does not authorize a second production module or unrelated refactor. If a 24th path,
another production type, a build/dependency edit, or an inward contract change is needed, stop and
replan before editing.

## Acceptance criteria

- The exact public signatures, constructor visibility, finality, declared-method inventory, and
  ordinary-signature import boundary above are automated in the distinct-package public-shape
  test. `HostTensorValue` has identity semantics and no close/resource API.
- The exact publication object from the receiver result is required. Null, foreign-result,
  equal-looking, repeated-role, and closed-result cases follow the specified precedence without
  Tensor ID, role, descriptor-equality, or index search.
- Static Shape count, byte count, caller limit, JVM ceiling, result-length defense, zero-element,
  and metadata semantics follow the exact contract. All failure messages and exception categories
  selected above are locked by focused tests.
- Returned byte access is fresh-view, read-only, big-endian, position-zero, exact-size, non-array-
  exposing, and immune to mutation of the backend-returned array after construction. Separate
  successful calls produce identity-distinct values with independent bytes.
- The exact descriptor, dense occurrence index, Runtime representation object, and caller limit
  cross the inward chain once and in order. Aliased Runtime occurrences are not merged. Repeated
  materialization invokes copying again and does not cache.
- Focused fake-composition tests cover forward and gradient occurrences, duplicate/aliased inward
  representations, exact mapping, representative non-dense resolved descriptors, empty values,
  repeated calls, malformed backend byte count, and original unchecked failure identity without
  duplicating CPU's physical traversal/encoding implementation.
- Real `Engine.standard()` integration materializes known output values for all six current data
  types through supported static resolved CONTIGUOUS CPU execution and verifies their canonical
  bytes. It also proves repeated/aliased gradient occurrences as currently feasible, fresh
  snapshots, limit failure, post-result-close value readability, and no inward type import in the
  ordinary fixture. Broader offset/strided/zero-stride and raw special-value encoding remain
  covered by the unchanged CPU 0010G tests rather than reimplemented in Engine tests.
- Bounded-latch tests, with no timing sleeps, prove same-result serialization, different-result
  concurrency, result-close waiting, result-close winning, Engine-close waiting, admitted-copy
  completion, closed precedence, no use-after-close, no deadlock, and cleanup after copy success or
  failure. Existing result/Engine cleanup order, exact-once behavior, retained failures, and
  interrupt restoration remain passing.
- Advanced public behavior is unchanged; `AdvancedRunResult` still exposes only count/lifecycle
  and no materialization or representation access.
- The documentation-focused pass finalizes meaningful Javadocs for all eight production paths,
  current Public/Compile/Runtime/Training/CPU-guide wording, one complete ordinary materialization
  example, and glossary entries for `HostTensorValue`, ordinary Engine `RunResult`, publication
  occurrence, and canonical host snapshot. It distinguishes current CPU-only Engine support from
  any cross-backend promise.
- No architecture, dependency, build, Runtime, CPU implementation, Model, Compiler, Prepare,
  backend-conformance, or later-task change is present. Engine 0004 is `Complete`; 0005–0008 stay
  `Draft` and have no detailed specifications.

## Tests / validation

Implementation-focused validation after executable Java stabilizes:

```bash
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
```

The implementation context records exact suite/test counts and passes the executable evidence to
the documentation context. Do not rerun Runtime or CPU suites: their completed 0015/0010G evidence
is unchanged, and this task modifies neither module. Do not run backend conformance because no
shared backend contract changes.

The distinct clean documentation-focused pass reuses successful Java evidence unless it changes
executable behavior, then runs:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

Also validate generated Javadoc; the automated exact public surface; a distinct-package ordinary
fixture using only Engine, Model, and JDK imports; changed Markdown local links/anchors, heading
order, balanced fences, trailing whitespace, LF/final newlines; exact at-most-23-path allowlist;
no forbidden imports/build edits/later specs; task/master/roadmap/Runtime-frontier status; and
preservation of all pre-existing unrelated worktree changes. Repository-wide tests remain
deferred to Engine 0008/CI because dependencies and architecture rules do not change.

## Documentation handoff

Primary profiles are Planning for this task/master/roadmap, API/Javadoc for Java contracts and API
references, Backend Guide for the existing CPU SPI's new current consumer, General for glossary
entries, and Example for the ordinary materialization walkthrough.

The implementation agent may draft Javadoc but must hand the final Java/test diff, exact public
surface, successful test evidence, dirty-worktree baseline, lifecycle/validation decisions, and
the unchanged Runtime 0015/CPU 0010G evidence to a distinct clean documentation-focused context.
That pass must independently inspect the behavior and finalize:

- `HostTensorValue` immutability, metadata, byte order/view, identity, independence, and no-close
  contract;
- `RunResult.materialize` ownership, validation precedence, lifetime, serialization, failures,
  no-cache semantics, and CPU-only current support;
- lifecycle and composition Javadocs changed by the implementation;
- Public/Runtime API current ordinary flow and Compile/Training references that currently call
  numerical access future or metadata-only;
- CPU backend-guide wording that currently says ordinary Engine materialization is absent; and
- glossary distinctions among publication occurrence, Runtime result lease, ordinary Engine
  result, canonical host snapshot, and `HostTensorValue`.

The complete example must use `Engine.standard()`, an existing supported static resolved Tensor
and CONTIGUOUS output, exact publication-object selection, an explicit byte limit, try-with-
resources for Engine/result, read-only big-endian byte inspection, and post-close value reading.
It must state that the value is detached and CPU-only today, not a transfer, Tensor, typed array,
cache, training workflow, or cross-backend contract.

## Dependencies

- Engine [0003](0003-typed-logical-input-binding-and-published-result-access.md) — Complete;
  owns ordinary publication occurrences, exact descriptors/indices, Runtime leases, input-wrapper
  cleanup, and Engine/result lifecycle.
- Runtime
  [0015](../../runtime/tasks/0015-leased-publication-representation-access.md) — Complete; returns
  the exact retained occurrence representation only while its result lease is open.
- CPU
  [0010G](../../../backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md) —
  Complete; validates/copies one exact CPU representation plus static resolved descriptor into
  fresh bounded canonical bytes for all six current types.
- Current Model `DataType`, `Shape`, `LayoutDescriptor`, and `TensorDescriptor` — implemented and
  sufficient.

All prerequisites are complete. No architecture or inward-contract dependency remains.

## Follow-up tasks

- Engine 0005 remains Draft: Engine-owned one-shot forward convenience may consume this completed
  explicit materialization boundary after 0004 implementation is Complete.
- Engine 0006 remains Draft: scalar-objective backward convenience stays Engine-owned and keeps
  explicit targets.
- Engine 0007–0008 remain Draft. Do not create their detailed specifications in this task.
- A future concrete backend requires its own demonstrated composition/copy design before ordinary
  materialization can claim cross-backend support; this task creates no speculative abstraction.

## Architecture impact

Expected impact: None.

The task realizes existing ownership: Engine composes and exposes public lifecycle/value access,
Runtime lends the selected representation under its lease, and CPU performs concrete physical
copying. It changes no module edge or architecture rule. Stop and report if implementation proves
otherwise.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik on Engine task 0004. Do not use GSD,
commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, the Engine master plan, and
docs/planning/modules/engine/tasks/0004-explicit-host-materialization-boundary.md in full. Read
the final Engine 0003, Runtime 0015, and CPU 0010G contracts and inspect all affected current
source, tests, Javadocs, API documentation, and the dirty worktree before editing.

Implement task 0004 exactly within its 23-path ceiling. Preserve every pre-existing change. Add
only the exact ordinary HostTensorValue/materialize surface, existing-gate lifecycle coordination,
package-private composition delegation, focused tests, real CPU integration, and authorized
documentation. Do not add Tensor execution/backward convenience, cross-backend abstractions,
transfer/fallback, dynamic descriptors, caching, typed arrays, ByteBuffer ownership, build or
architecture changes, or later task specs. Stop on architecture or scope conflict.

Run the task-tier Java validation once after executable code stabilizes. Then hand the same diff
and exact evidence to a distinct clean documentation-focused context, which must follow the
documentation rules and finalize Javadocs, explanatory docs, glossary impact, examples, and
documentation validation without repeating successful Java suites unless executable behavior
changes. Mark Complete only after every implementation and documentation gate passes.
```

## Local decisions

- Choose one immutable class rather than a record because public record construction and
  array-based generated equality/accessors would violate Engine ownership and immutability.
- Choose a fresh read-only `ByteBuffer` view rather than returning `byte[]`; the class privately
  clones CPU's fresh array so neither the backend producer nor caller can mutate its snapshot.
- Keep only type, Shape, and checked counts. The detached canonical payload has no source layout,
  physical span, gradient flag, Tensor identity, or resource lifetime.
- Authenticate publication selection by exact private originating-result identity and then use its
  stored dense index. No descriptive field is an authority token.
- Use the current result monitor for copy/close exclusion and the current Engine operation counter
  for Engine-close quiescence. No pin, reference count, future, cancellation, or new lock is needed.
- Keep CPU 0010G's validations as concrete-backend defense while repeating public metadata and
  byte-limit checks in Engine. The duplicate checks are intentional boundary validation, not
  duplicate encoding logic.
- CPU 0010G proved the dense zero-offset fast path while preserving the general layout oracle; no
  Engine question or work remains for that optimization.

## Known limitations

- Current ordinary materialization works only through the fixed CPU-only standard composition and
  fully static resolved publication descriptors supported by CPU 0010G.
- One payload is limited to `Integer.MAX_VALUE` bytes and available heap. There is no streaming,
  mapped, chunked, caller-destination, or off-heap result.
- Calls on one result serialize and may block result/Engine closure for the duration of the copy.
- The payload is canonical bytes, not a Tensor, host storage association, typed array, scalar
  object, persistence format, transfer request, or training abstraction.

## Validation evidence

Planning context: `01a0a4f6-d1f9-7b61-9090-27a25b034228`.

- Read the required architecture, planning, Engine/Runtime/CPU master/task, implementation,
  test, Model descriptor, API/Javadoc, and documentation-profile contracts. No architecture,
  dependency, or inward-contract conflict was found.
- Confirmed final Runtime 0015 returns the identical indexed representation only while its
  non-thread-safe whole-state lease is open and checks closure before bounds.
- Confirmed final CPU 0010G accepts the nominal Runtime representation plus exact static resolved
  descriptor and limit, supports the two current concrete CPU representations and all six types,
  returns fresh canonical bytes, and preserves the deterministic validation order.
- Confirmed Engine 0003 already retains the exact outward occurrence index/descriptor and Runtime
  result behind one registered `AdvancedRunResult`, and its Engine gate/result monitor can provide
  the required admission and close serialization without a new abstraction.
- Confirmed CPU 0010G's dense zero-offset fast path is implemented and validated; the former Engine
  master-plan question is closed.
- Java tests were not run because this was a planning-only pass.

## Implementation notes

- Implementation context `01a0a502-9546-7de3-b34f-882f918f8e97` added the exact ordinary
  `HostTensorValue` and `RunResult.materialize(...)` surface, existing-gate/result-monitor
  lifecycle coordination, CPU composition delegation, five focused tests, and real CPU coverage.
- The implementation changed no build, dependency, architecture, Runtime, CPU, Model, Compiler,
  Prepare, backend-conformance, or later-task path. It retained the exact 23-path ceiling.
- Documentation context `01a0a510-07fe-7bd1-816b-8e7318b5387b` independently finalized all eight
  affected Engine production Javadocs and all six authorized explanatory documents. The ordinary
  example uses only Engine, Model, and JDK imports and reads the detached value after result,
  Engine, and arena closure.

## Completion summary

- Completed changes: one exact publication occurrence can be copied lazily into a fresh immutable
  `HostTensorValue` with checked static metadata and canonical row-major big-endian bytes. Exact
  occurrence authentication, byte-budget validation, CPU-only support, no-cache semantics,
  ownership, failure precedence, and close/concurrency behavior are documented and tested.
- Files changed or created: exactly the 23 paths enumerated under Affected files. The
  documentation pass changed Javadocs only in production paths 1–8, explanatory paths 14–19, and
  planning/status paths 20–23; tests 9–13 remained byte-for-byte unchanged.
- Reused validation: `:modules:engine:test` passed 5 suites and 30 tests;
  `EngineTypedLifecycleIntegrationTest` passed 1 suite and 4 tests; and
  `EngineCompositionContractTest` passed 1 suite and 1 test from Gradle cache. Runtime 0015 and
  CPU 0010G remained unchanged with their previously completed suites.
- Documentation validation: `:modules:engine:javadoc`, generated-page inspection,
  `git diff --check`, exact `javap` and reflection surface checks, and the distinct-package
  ordinary materialization fixture passed. Markdown local-link/anchor, heading, fence, whitespace,
  LF/final-newline checks; exact task/unrelated-dirty-path scope; forbidden build/architecture and
  later-spec absence; synchronized statuses; and preservation hashes passed.
- Documentation impact: Public, Compile, Runtime, Training, CPU backend, and glossary references
  now distinguish current CPU-only fully static resolved materialization from future transfer,
  cross-backend, one-shot, persistence, and training coordination. Architecture, build,
  dependencies, and other modules required no change.
- Unresolved issues: none for task 0004. Engine 0005–0008 remain Draft and are separate follow-up
  work; no later task specification was created.

Status: Complete
