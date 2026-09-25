# Task 0048: Persistent immutable Metal splats and no-general-pool closure

## Status

Review needed

Implementation from sole-Ready planning commit
`08b9e7ea7782bc0165cb6b45c409afdb59673c2e` is complete and locally proved on the current
Apple-silicon host. Blocked Task 0047 remains independent; this task adds no hosted CI.

## Change class

Class C — this task changes backend prepared-resource ownership, concurrent run/result lifetime,
failure timing, cleanup suppression, transfer/materialization compatibility, and trace finalization.
Use one implementation context and require an independent lifecycle-focused review before marking
Complete.

## Goal

Remove the measured per-run allocation, host fill, upload, and release of immutable Metal scalar
splat inputs. Allocate each source-owned splat once per `PreparedExecution`, publish only read-only
fresh run bindings, and release the native buffer exactly once after the prepared owner and every
already-issued binding have closed.

Close the buffer-pooling todo honestly by making a deliberate no-general-pool decision. Mutable
outputs can outlive the synchronous call through `RunResult`; concurrent runs require distinct
output identity. MPSGraph address workspaces are mutable pointer arrays populated during cold bind.
Pooling either resource without a later async/result-lifetime contract would require authenticated
exclusive leases, result-close return, reset/validity rules, capacity and eviction bounds, context
close deferral, and failure recovery. This task therefore keeps outputs and workspaces fresh per run
rather than silently narrowing or pretending that persistent constants constitute a general pool.

## Measured structural waste

`MetalNegPreparedScheduleAssembler.createSplatBuffer` currently allocates one native buffer and one
full-size confined host segment, performs an O(elements) Java raw-bit fill, uploads the complete
payload, and later releases the native buffer for every splat on every run.

- The existing custom splat test observes four buffer creations and two uploads across two runs:
  one splat and one output per run.
- The existing positive-rank MPSGraph fixture has six splats and eight targets. Each run creates
  fourteen route-internal buffers and performs six uploads. Across two runs this is twenty-eight
  creations and twelve uploads, excluding the one caller buffer.
- The same fixture already covers `+0.0f`, `-0.0f`, infinities, quiet NaN, and signaling NaN raw
  bits, so no optimization matrix is needed.

After this task, the custom fixture must observe one preparation-time splat create/upload and one
fresh output create per run: three creations and one upload across two runs. The six-splat fixture
must observe six preparation-time creates/uploads and only eight output creates with zero uploads
per run: twenty-two creations and six uploads across two runs.

These native-call counts are deterministic structural evidence. The existing
`MetalRouteBenchmark` has one caller-backed input and no splat, so running it cannot measure this
change; timing it would be noise from an unchanged path. Do not modify or run that benchmark for
this task.

## Ownership and lifecycle contract

### Source ownership

Metal analysis retains one source-splat fact aligned with the existing feed order. A splat is
physically prepared by a partition only when that partition is the first ordered consumer of the
producer-free logical constant. A later Metal partition sharing that value reuses the one source
representation. If another backend is the source owner, Metal retains the ordinary writable
transfer destination and receives the constant through the existing transfer path.

The fact changes only physical ownership. It does not change capability, route selection,
workload/candidate identity, logical memory requirements, scalar bits, or schedule order.

### Prepared splat resource and run binding

Add one package-private `MetalPreparedSplatResource` per source-owned splat. It implements
`PreparedResource`, owns the exact Metal context, byte extent, scalar type/raw bits, and one
uploaded `MetalBufferRepresentation`, and never mutates that buffer after publication.

Every creator call returns a fresh read-only `BufferRepresentation` binding object. The binding
owns one child lease, starts valid through the existing `InitializedBuffer` path, and closes only
its lease. The resource owner rejects new bindings after close. Owner close releases its ownership
reference; if an already-issued run binding remains in an open `RunState`/`RunResult`, physical
release is deferred until the last binding closes. Exactly one thread claims the native release.
Repeated close is harmless.

Construct a binding object before incrementing its active count so an allocation `Error` cannot
leak a lease. All mutable state transitions are synchronized or otherwise linearizable. The
underlying buffer's existing context child lease keeps context cleanup correct.

This is not cross-execution caching: a fresh preparation gets fresh splat resources, and closing one
preparation cannot affect another.

### Finalization and rollback

The partition finalizer validates all finalized assignments before native work. It acquires the
selected custom pipeline or MPSGraph executable first, then source-owned splats in stable feed
order. Each splat is allocated, filled with exact raw bits, and uploaded before the final PREPARE
success callback.

Return resources in physical acquisition order:

```text
[route resource, source splat 0, source splat 1, ...]
```

On allocation, fill, upload, resource construction, executable construction, finalizer-result, or
PREPARE-success observer failure, keep the original throwable primary and attempt cleanup of the
current buffer and all prior resources in exact reverse acquisition order. Add distinct cleanup
throwables as suppressed in encounter order; never self-suppress.

### Schedule and executable

Replace per-run splat creation with `InitializedBuffer(resource::newRunBinding)` for the exact
source representation. Composite schedules find exactly one source resource for a shared logical
constant; later Metal consumers bind the same run representation. Non-source Metal representations
remain ordinary `CreatedBuffer` destinations.

Cold executable binding unwraps a persistent splat binding only for a read selection, authenticates
context, byte extent, scalar type/raw bits, and compares unwrapped physical identity for input/output
alias rejection. Outputs reject persistent bindings. Custom and MPSGraph hot invocation retain only
the underlying direct buffers/handles, so the hot path gains no wrapper cast or allocation.

MPSGraph error rescan continues to download exact input buffers. Outputs and address workspaces
remain fresh and isolated for repeated and concurrent runs.

### Transfer and materialization

Metal-to-CPU transfer and canonical publication materialization accept a live persistent splat
binding as a read source by unwrapping its authenticated underlying buffer. CPU-to-Metal upload and
all writable destinations reject it. Existing ordinary writable `MetalBufferRepresentation`
behavior is unchanged.

### Close and concurrency

A prepared execution may close after synchronous execution while returned results remain open.
Resource owner close must therefore prevent new bindings yet retain physical storage under existing
child leases. With two open results, closing the first binding does not release the buffer; closing
the last binding releases it exactly once. New runs after prepared close are rejected before
creating a binding or output.

Concurrent runs share only immutable constant bytes and receive distinct binding identities,
outputs, and address workspaces. The current custom and MPSGraph route resources already serialize
native invocation, but correctness must not depend on that serialization: shared splat reads are
immutable.

### Failure observation

A setup failure after binding creation closes that run binding but not the still-open prepared
owner; a later run can succeed. If owner close has no outstanding binding, a native release failure
is observed by `PreparedExecution.close()` or the last synchronous lease release. If release is
deferred under a result-held binding, the last `RunResult` cleanup observes it and existing
attempt-all/suppression rules apply.

## Preserved boundaries

- Public Java API shape is unchanged.
- Native ABI v4, thirteen exports, node schema 11, local value types, and wires are unchanged.
- Capability, route policy, exact numerics, and numerical-profile behavior are unchanged.
- Metal candidate/workload/codec identities remain version 12. Scalar type/raw bits are already in
  the workload signature; source storage lifetime is not a route semantic.
- Trace adds no event or payload. Splat failure emits one failed PREPARE and no RUN; success is
  emitted only after all resources exist.
- Tuning trial and selected preparations each own and close their own splats. No cache, registry,
  intern table, context-global reuse, retry, fallback, or telemetry is added.

## Explicit no-general-pool decision

This task closes the requested pooling/persistent-constant optimization with the following result:

- immutable source splats are persisted because exact bits, read-only access, preparation identity,
  and lifetime can be proven;
- mutable outputs are not pooled because ownership extends into `RunResult` and concurrent results
  require distinct writable resources;
- mutable MPSGraph address workspaces are not pooled because cold binding writes per-run addresses;
- a general pool remains unauthorized until a separate async/result-lifetime contract defines
  exclusive return, capacity/eviction, reset, close deferral, and failure semantics from evidence.

No follow-up pool placeholder, matrix, or partial implementation belongs in this task.

## Dependencies and integration

- Depends on: Task 0046 Complete at independently approved implementation `4aad1ab6`; Runtime 0016
  prepared-resource lifecycle; Prepare 0006 finalizer ownership; Engine 0010 prepared ownership;
  current Metal finalization/schedule/trace/tuning implementation
- Independent blocker retained: Task 0047 remains Blocked on a GPU-capable hosted runner
- Conflicts with: every concurrent Metal preparation/finalization/executable/schedule/buffer/transfer,
  trace/tuning identity, backend/runtime/API/decision documentation, Metal test, master, or roadmap
  edit
- Parallel group: None
- Common base revision: `a9adf4d707c9472283ad0621647bee22efc7ff82`
- Integration order: implemented from sole-Ready planning commit `08b9e7ea`; now Review needed
  while 0047 remains independently Blocked; no Metal task is Ready
- Integration validation: focused fake-native proof, exactly two explicit real-device no-skip tests,
  Javadocs/architecture/docs/diff checks, then independent Class C review; do not run the repository
  full build until final program verification
- Shared-document integration owner: Task 0048 implementation owner

## Files and responsibilities

Production and contract:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedResource.java` —
  clarify owner close versus an already-issued backend child lease; no API-shape change
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalPreparedSplatResource.java`
  — prepared immutable buffer owner, fresh read-only bindings, exact close/failure behavior
- `MetalDeviceContext.java` and `MetalBufferRepresentation.java` — correct run-only ownership wording
- `MetalNegPreparationPlan.java` and `MetalNegPartitionPreparer.java` — aligned first-consumer source
  fact
- `MetalNegPartitionFinalizer.java` — ordered acquisition, exact splat initialization, reverse rollback
- `MetalNegPreparedExecutable.java` — authentication, unwrapping, read-only enforcement, physical
  alias checks
- `MetalNegPreparedScheduleAssembler.java` — persistent binding creators and shared source lookup;
  remove per-run host fill/upload
- `MetalBackendRuntime.java` — read transfer/materialization support and write rejection

Focused tests:

- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutionTest.java`
  — deterministic counts, bits, repeated/concurrent identity, source-owner sharing, transfer/read-only,
  owner/result close, allocation/upload/factory/setup/release/trace failures, and leak/suppression rules
- `MetalNegRouteCandidateGeneratorTest.java` — plan-copy migration and unchanged version-12 identity

Documentation and planning:

- `docs/backend-guide/metal-backend.md`
- `docs/api/runtime-api.md`
- `docs/architecture/contracts/runtime-prepare-engine.md`
- `docs/design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md`
- this task, `../master-plan.md`, and `../../../roadmap.md`

No Engine, Prepare, Trace, tuning, benchmark, native, capability, configuration, ABI, workflow, or
Gradle production file may change.

## High-value permanent tests

1. Rewrite the existing custom splat repetition test to require one prepare allocation/upload,
   shared physical input across two runs, distinct run bindings/outputs, exact negation, and exact
   release boundaries.
2. Rewrite the existing six-splat MPSGraph repeated/concurrent fixture to require six preparation
   uploads, zero run uploads, stable exact raw bits/handles, and distinct outputs/workspaces.
3. Exercise a shared constant with two Metal consumers and a CPU-source case: exactly one source
   prepared splat in the former; no Metal persistent splat and ordinary transfer in the latter.
4. Close a prepared execution while two results remain open; require no new run, no early constant
   release, first-result non-release, and last-result exact-once release.
5. Cover second-splat allocation failure, upload failure, executable/finalizer-result failure,
   binding/setup failure followed by successful reuse, deferred release failure, and observer
   failure. Assert primary/suppressed order, reverse cleanup, zero RUN, and no live handle.
6. Cover Metal-to-CPU download/materialization from a binding and upload/output rejection.
7. Retain no-splat/caller-input tests to prove zero constant resources and unchanged ordinary
   behavior. Do not add source-text tests, timing assertions, shape/route tables, or duplicate
   parameter rows.

## Validation

Focused portable proof passed:

```bash
./gradlew :backends:metal:test \
  --tests '*MetalNegPreparedExecutionTest' \
  --tests '*MetalNegRouteCandidateGeneratorTest'
```

JUnit XML reported 49 execution tests with zero failures/errors and three expected environment-
gated native skips, plus nine candidate/identity tests with zero skips/failures/errors. The focused
fake-native tests assert exact native create/upload/release counts, raw bits, first-consumer source
ownership, repeated/concurrent binding identity, output/workspace isolation, read/write rejection,
open-result close deferral, later-run recovery, trace truth, reverse rollback, and failure
suppression.

The current native library was rebuilt, selected by its explicit absolute path, and exactly the two
authorized real-device tests were rerun:

```bash
./native/metal-macos-arm64/build.sh
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test \
    --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse' \
    --tests '*MetalNegPreparedExecutionTest.realDeviceTypedMpsGraphInitializesLogicalSplatsPerRun' \
    --rerun-tasks
```

The matching JUnit XML reports exactly two tests and zero skipped, failures, and errors. Metal and
Runtime Javadocs passed. Focused `RuntimeDependencyAndHotPathContractTest` passed. Changed-Markdown
local links/fences/final newlines, `git diff --check`, exact path scope, and clean committed status
are required before handoff. Per instruction, no full repository build was run; the one final full
build remains reserved for program verification.

No benchmark was run. The existing singleton-NEG benchmark has one caller-backed input and no
constant, so it cannot measure this optimization.

## Acceptance criteria

1. Exactly one immutable buffer is prepared per source-owned Metal splat and no non-source duplicate
   is prepared.
2. Every run gets a fresh valid read-only binding; repeated/concurrent runs share only exact physical
   constant storage and retain distinct mutable outputs/workspaces.
3. The deterministic custom and six-splat counts match the targets above with exact raw-bit and
   numerical output preservation.
4. Transfer/materialization read paths accept the binding; every write path rejects it before native
   mutation.
5. Prepared close, open results, concurrent runs, setup failure, finalization rollback, and deferred
   native cleanup preserve exact ownership, reverse order, attempt-all, and primary/suppressed
   behavior with no leak or double release.
6. PREPARE/RUN trace truth and version-12 tuning/candidate identity are unchanged; public API shape,
   capability, native ABI, and route policy are unchanged.
7. Output and address-workspace pooling are explicitly rejected under the current lifecycle rather
   than implemented partially or called complete by omission.
8. Focused fake-native tests, exactly two real-device no-skip tests, Javadocs, architecture/docs/diff
   validation, and an independent Class C review pass. No full repository build is run now.

## Result

Implemented from planning commit `08b9e7ea7782bc0165cb6b45c409afdb59673c2e`.
Each first-consumer/source-owned Metal splat is allocated, raw-bit-filled, and uploaded exactly once
per prepared execution. Every run receives a fresh authenticated read-only child binding; route
invocation, Metal-to-CPU transfer, and canonical materialization unwrap it only for reads. Prepared
owner close rejects new bindings and exact-once native release is deferred through any already-open
result. Finalization and all failure paths preserve acquisition order, reverse cleanup, and
primary/suppressed evidence.

The deterministic custom two-run path is reduced from four creates/two uploads to three creates/one
upload. The six-splat/eight-target two-run path is reduced from twenty-eight route-internal creates
and twelve uploads to twenty-two creates and six uploads. Exact raw bits, numerics, trace semantics,
version-12 identities, public API shape, capability, route policy, native ABI, and native exports
remain unchanged.

The optimization todo is closed without a misleading general pool: mutable outputs remain fresh
because `RunResult` owns them beyond synchronous return, and MPSGraph address workspaces remain
fresh because cold binding mutates their per-run pointer arrays. No pool, cache, matrix, benchmark,
retry, fallback, or cross-preparation reuse was added. Status is Review needed pending independent
Class C approval.
