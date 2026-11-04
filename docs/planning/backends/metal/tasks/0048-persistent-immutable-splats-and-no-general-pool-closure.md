# Task 0048: Persistent [immutable](../../../../glossary.md#term-immutable) [Metal](../../../../glossary.md#term-metal) splats and no-general-pool closure

## Status

Complete

Implemented at `89f9fbb9093db3fb86629189cb98ae521a631b27` from sole-[Ready](../../../../glossary.md#term-ready) [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit)
`08b9e7ea7782bc0165cb6b45c409afdb59673c2e`. Independent Class C review returned `APPROVE` with
zero findings. This documentation-only finalization corrects one stale [backend](../../../../glossary.md#backend)-guide limitation;
it changes no implementation, [API](../../../../glossary.md#term-application-programming-interface-api), [ABI](../../../../glossary.md#term-application-binary-interface-abi), [identity](../../../../glossary.md#term-identity-meanings), capability, route, [trace](../../../../glossary.md#trace), [cache](../../../../glossary.md#term-cache), or tuning
behavior.

## Change class

Class C — this task changes [backend](../../../../glossary.md#backend) prepared-resource [ownership](../../../../glossary.md#term-ownership), concurrent [run](../../../../glossary.md#term-run)/result [lifetime](../../../../glossary.md#term-lifetime),
failure timing, cleanup suppression, transfer/[materialization](../../../../glossary.md#materialization) compatibility, and [trace](../../../../glossary.md#trace) finalization.
Use one implementation context and require an independent [lifecycle](../../../../glossary.md#lifecycle)-focused review before marking
Complete.

## Goal

Remove the measured per-[run](../../../../glossary.md#term-run) allocation, [host](../../../../glossary.md#term-host) fill, upload, and [release](../../../../glossary.md#term-release) of [immutable](../../../../glossary.md#term-immutable) [Metal](../../../../glossary.md#term-metal) [scalar](../../../../glossary.md#term-scalar)
splat inputs. Allocate each source-owned splat once per [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), publish only read-only
fresh run [bindings](../../../../glossary.md#term-binding), and release the native [buffer](../../../../glossary.md#term-buffer) exactly once after the prepared [owner](../../../../glossary.md#term-owner) and every
already-issued binding have closed.

Close the buffer-[pooling](../../../../glossary.md#term-pooling) todo honestly by making a deliberate no-general-pool decision. [Mutable](../../../../glossary.md#term-mutable)
outputs can outlive the [synchronous](../../../../glossary.md#term-synchronous) call through [`RunResult`](../../../../glossary.md#run-result--runresult); concurrent runs require distinct
output [identity](../../../../glossary.md#term-identity-meanings). [MPSGraph](../../../../glossary.md#term-mpsgraph) address workspaces are mutable pointer arrays populated during cold bind.
Pooling either resource without a later async/result-[lifetime](../../../../glossary.md#term-lifetime) contract would require authenticated
exclusive [leases](../../../../glossary.md#term-lease), result-close return, reset/validity rules, capacity and eviction bounds, context
close deferral, and failure recovery. This task therefore keeps outputs and workspaces fresh per run
rather than silently [narrowing](../../../../glossary.md#term-narrowing) or pretending that persistent [constants](../../../../glossary.md#term-constant) constitute a general pool.

## Measured structural waste

`MetalNegPreparedScheduleAssembler.createSplatBuffer` currently allocates one native [buffer](../../../../glossary.md#term-buffer) and one
full-size confined [host](../../../../glossary.md#term-host) segment, performs an O([elements](../../../../glossary.md#term-element)) Java raw-bit fill, uploads the complete
[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), and later [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) the native buffer for every splat on every [run](../../../../glossary.md#term-run).

- The existing custom splat test observes four buffer creations and two uploads across two runs:
  one splat and one output per run.
- The existing positive-rank MPSGraph fixture has six splats and eight targets. Each run creates
  fourteen route-internal buffers and performs six uploads. Across two runs this is twenty-eight
  creations and twelve uploads, excluding the one caller buffer.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [targets glossary](../../../../glossary.md#term-target))</span>
- The same [fixture](../../../../glossary.md#term-fixture) already covers `+0.0f`, `-0.0f`, infinities, quiet NaN, and signaling NaN raw
  bits, so no optimization matrix is needed.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [quiet NaN glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signaling NaN glossary](../../../../glossary.md#term-signaling-nan); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

After this task, the custom fixture must observe one preparation-time splat create/upload and one
fresh output create per run: three creations and one upload across two runs. The six-splat fixture
must observe six preparation-time creates/uploads and only eight output creates with zero uploads
per run: twenty-two creations and six uploads across two runs.

These native-call counts are deterministic structural evidence. The existing
`MetalRouteBenchmark` has one caller-backed input and no splat, so running it cannot measure this
change; timing it would be noise from an unchanged path. Do not modify or run that benchmark for
this task.

## [Ownership](../../../../glossary.md#term-ownership-meanings) and [lifecycle](../../../../glossary.md#lifecycle) contract

### Source [ownership](../../../../glossary.md#term-ownership-meanings)

[Metal](../../../../glossary.md#term-metal) analysis retains one source-splat fact aligned with the existing feed order. A splat is
physically prepared by a [partition](../../../../glossary.md#partition) only when that partition is the first ordered consumer of the
[producer](../../../../glossary.md#term-producer)-free logical [constant](../../../../glossary.md#term-constant). A later Metal partition sharing that value reuses the one source
representation. If another [backend](../../../../glossary.md#backend) is the source [owner](../../../../glossary.md#term-owner), Metal retains the ordinary writable
transfer destination and receives the constant through the existing transfer path.

The fact changes only physical ownership. It does not change capability, route selection,
workload/candidate [identity](../../../../glossary.md#term-identity-meanings), logical memory requirements, [scalar](../../../../glossary.md#term-scalar) bits, or schedule order.

### Prepared splat resource and [run](../../../../glossary.md#term-run) [binding](../../../../glossary.md#term-binding)

Add one package-private `MetalPreparedSplatResource` per source-owned splat. It implements
[`PreparedResource`](../../../../glossary.md#prepared-resource--preparedresource), owns the exact [Metal](../../../../glossary.md#term-metal) context, byte [extent](../../../../glossary.md#term-extent), [scalar](../../../../glossary.md#term-scalar) type/[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), and one
uploaded `MetalBufferRepresentation`, and never mutates that [buffer](../../../../glossary.md#term-buffer) after [publication](../../../../glossary.md#term-publication).

Every creator call returns a fresh read-only `BufferRepresentation` binding object. The binding
owns one child [lease](../../../../glossary.md#term-lease), starts valid through the existing `InitializedBuffer` path, and closes only
its lease. The [resource owner](../../../../glossary.md#ownership-borrowing-and-lease) rejects new bindings after close. [Owner](../../../../glossary.md#term-owner) close [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) its [ownership](../../../../glossary.md#term-ownership-meanings)
reference; if an already-issued run binding remains in an open [`RunState`](../../../../glossary.md#run-state--runstate)/[`RunResult`](../../../../glossary.md#run-result--runresult), physical
[release](../../../../glossary.md#term-release) is deferred until the last binding closes. Exactly one thread claims the native release.
Repeated close is harmless.

Construct a binding object before incrementing its active count so an allocation `Error` cannot
leak a lease. All [mutable](../../../../glossary.md#term-mutable) state transitions are synchronized or otherwise linearizable. The
underlying buffer's existing context child lease keeps context cleanup correct.

This is not cross-execution caching: a fresh preparation gets fresh splat resources, and closing one
preparation cannot affect another.

### Finalization and [rollback](../../../../glossary.md#term-rollback)

The [partition](../../../../glossary.md#partition) finalizer validates all finalized assignments before native work. It acquires the
selected custom pipeline or [MPSGraph](../../../../glossary.md#term-mpsgraph) executable first, then source-owned splats in stable feed
order. Each splat is allocated, filled with exact [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), and uploaded before the final [PREPARE](../../../../glossary.md#prepare)
success [callback](../../../../glossary.md#term-callback).

Return resources in physical acquisition order:

```text
[route resource, source splat 0, source splat 1, ...]
```

On allocation, fill, upload, resource construction, executable construction, finalizer-result, or
PREPARE-success [observer](../../../../glossary.md#term-observer) failure, keep the original throwable primary and attempt cleanup of the
current [buffer](../../../../glossary.md#term-buffer) and all prior resources in exact reverse acquisition order. Add distinct cleanup
throwables as suppressed in encounter order; never self-suppress.

### Schedule and executable

Replace per-[run](../../../../glossary.md#term-run) splat creation with `InitializedBuffer(resource::newRunBinding)` for the exact
source representation. Composite schedules find exactly one source resource for a shared logical
[constant](../../../../glossary.md#term-constant); later [Metal](../../../../glossary.md#term-metal) consumers bind the same run representation. Non-source Metal representations
remain ordinary `CreatedBuffer` destinations.

Cold executable [binding](../../../../glossary.md#term-binding) unwraps a persistent splat binding only for a read selection, authenticates
context, byte [extent](../../../../glossary.md#term-extent), [scalar](../../../../glossary.md#term-scalar) type/[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), and compares unwrapped physical [identity](../../../../glossary.md#term-identity-meanings) for input/output
[alias](../../../../glossary.md#aliasing-and-arena) rejection. Outputs reject persistent bindings. Custom and [MPSGraph](../../../../glossary.md#term-mpsgraph) hot invocation retain only
the underlying direct [buffers](../../../../glossary.md#term-buffer)/handles, so the [hot path](../../../../glossary.md#term-hot-path) gains no wrapper cast or allocation.

MPSGraph error rescan continues to download exact input buffers. Outputs and address workspaces
remain fresh and isolated for repeated and concurrent runs.

### Transfer and [materialization](../../../../glossary.md#materialization)

[Metal](../../../../glossary.md#term-metal)-to-[CPU](../../../../glossary.md#term-cpu) transfer and [canonical](../../../../glossary.md#term-canonical-form) [publication](../../../../glossary.md#term-publication) materialization accept a live persistent splat
[binding](../../../../glossary.md#term-binding) as a read source by unwrapping its authenticated underlying [buffer](../../../../glossary.md#term-buffer). CPU-to-Metal upload and
all writable destinations reject it. Existing ordinary writable `MetalBufferRepresentation`
behavior is unchanged.

### Close and [concurrency](../../../../glossary.md#term-concurrency)

A [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) may close after [synchronous](../../../../glossary.md#term-synchronous) execution while returned results remain open.
[Resource owner](../../../../glossary.md#ownership-borrowing-and-lease) close must therefore prevent new [bindings](../../../../glossary.md#term-binding) yet retain physical storage under existing
child [leases](../../../../glossary.md#term-lease). With two open results, closing the first binding does not [release](../../../../glossary.md#term-release) the [buffer](../../../../glossary.md#term-buffer); closing
the last binding [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) it exactly once. New runs after prepared close are rejected before
creating a binding or output.

Concurrent runs share only [immutable](../../../../glossary.md#term-immutable) [constant](../../../../glossary.md#term-constant) bytes and receive distinct binding [identities](../../../../glossary.md#term-identity-meanings),
outputs, and address workspaces. The current custom and [MPSGraph](../../../../glossary.md#term-mpsgraph) route resources already serialize
native invocation, but correctness must not depend on that [serialization](../../../../glossary.md#term-serialization): shared splat reads are
immutable.

### Failure observation

A setup failure after [binding](../../../../glossary.md#term-binding) creation closes that [run](../../../../glossary.md#term-run) binding but not the still-open prepared
[owner](../../../../glossary.md#term-owner); a later run can succeed. If owner close has no outstanding binding, a native [release](../../../../glossary.md#term-release) failure
is observed by `PreparedExecution.close()` or the last [synchronous](../../../../glossary.md#term-synchronous) [lease](../../../../glossary.md#term-lease) release. If release is
deferred under a result-held binding, the last [`RunResult`](../../../../glossary.md#run-result--runresult) cleanup observes it and existing
attempt-all/suppression rules apply.

## Preserved boundaries

- Public Java [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape) is unchanged.
- Native [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, thirteen exports, [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) 11, local value types, and wires are unchanged.
- Capability, route policy, exact numerics, and numerical-profile behavior are unchanged.
- Metal candidate/workload/codec identities remain version 12. Scalar type/raw bits are already in
  the workload signature; source storage lifetime is not a route semantic.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [codec glossary](../../../../glossary.md#term-codec); [Scalar glossary](../../../../glossary.md#term-scalar); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [signature glossary](../../../../glossary.md#term-signature); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Trace adds no event or payload. Splat failure emits one failed PREPARE and no RUN; success is
  emitted only after all resources exist.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [PREPARE glossary](../../../../glossary.md#prepare); [RUN glossary](../../../../glossary.md#term-run))</span>
- Tuning trial and selected preparations each own and close their own splats. No cache, registry,
  intern table, context-global reuse, retry, fallback, or telemetry is added.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry))</span>

## Explicit no-general-pool decision

This task closes the requested [pooling](../../../../glossary.md#term-pooling)/persistent-[constant](../../../../glossary.md#term-constant) optimization with the following result:

- immutable source splats are persisted because exact bits, read-only access, preparation identity,
  and lifetime can be proven;<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [mutable](../../../../glossary.md#term-mutable) outputs are not pooled because [ownership](../../../../glossary.md#term-ownership-meanings) extends into [`RunResult`](../../../../glossary.md#run-result--runresult) and concurrent results
  require distinct writable resources;
- mutable [MPSGraph](../../../../glossary.md#term-mpsgraph) address workspaces are not pooled because cold [binding](../../../../glossary.md#term-binding) writes per-[run](../../../../glossary.md#term-run) addresses;
- a general pool remains unauthorized until a separate async/result-lifetime contract defines
  exclusive return, capacity/eviction, reset, close deferral, and failure semantics from evidence.

No follow-up pool placeholder, [matrix](../../../../glossary.md#term-matrix-meanings), or partial implementation belongs in this task.

## Dependencies and integration

- Depends on: Task 0046 Complete at independently approved implementation `4aad1ab6`; Runtime 0016
  prepared-resource lifecycle; Prepare 0006 finalizer ownership; Engine 0010 prepared ownership;
  current Metal finalization/schedule/trace/tuning implementation<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle); [Prepare glossary](../../../../glossary.md#prepare); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [trace glossary](../../../../glossary.md#trace))</span>
- Independent blocker retained: Task 0047 remains Blocked on a [GPU](../../../../glossary.md#term-gpu)-capable [hosted runner](../../../../glossary.md#continuous-integration-and-hosted-runners)
- Conflicts with: every concurrent Metal preparation/finalization/executable/schedule/buffer/transfer,
  trace/tuning identity, backend/runtime/API/decision documentation, Metal test, master, or roadmap
  edit<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Parallel group: None
- Common base revision: `a9adf4d707c9472283ad0621647bee22efc7ff82`
- Integration order: implemented from sole-[Ready](../../../../glossary.md#term-ready) [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit) `08b9e7ea`; independent Class C
  review approved implementation `89f9fbb9` with zero findings; Complete while 0047 remains
  independently Blocked
- Integration validation: focused fake-native proof, exactly two explicit real-device no-skip tests,
  Javadocs/architecture checks, documentation/diff checks, and independent Class C approval passed;
  no full repository build was run by instruction<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [device glossary](../../../../glossary.md#term-device); [run glossary](../../../../glossary.md#term-run))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task 0048 implementation owner

## Files and responsibilities

Production and contract:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedResource.java` —
  clarify owner close versus an already-issued backend child lease; no API-shape change<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [backend glossary](../../../../glossary.md#backend); [lease glossary](../../../../glossary.md#term-lease); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalPreparedSplatResource.java`
  — prepared immutable buffer owner, fresh read-only bindings, exact close/failure behavior<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [buffer glossary](../../../../glossary.md#term-buffer); [bindings glossary](../../../../glossary.md#term-binding))</span>
- `MetalDeviceContext.java` and `MetalBufferRepresentation.java` — correct [run](../../../../glossary.md#term-run)-only [ownership](../../../../glossary.md#term-ownership-meanings) wording
- `MetalNegPreparationPlan.java` and `MetalNegPartitionPreparer.java` — aligned first-consumer source
  fact
- `MetalNegPartitionFinalizer.java` — ordered acquisition, exact splat initialization, reverse [rollback](../../../../glossary.md#term-rollback)
- `MetalNegPreparedExecutable.java` — authentication, unwrapping, read-only enforcement, physical
  alias checks<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- `MetalNegPreparedScheduleAssembler.java` — persistent binding creators and shared source lookup;
  remove per-run host fill/upload<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [host glossary](../../../../glossary.md#term-host))</span>
- `MetalBackendRuntime.java` — read transfer/[materialization](../../../../glossary.md#materialization) support and write rejection

Focused tests:

- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutionTest.java`
  — deterministic counts, bits, repeated/concurrent identity, source-owner sharing, transfer/read-only,
  owner/result close, allocation/upload/factory/setup/release/trace failures, and leak/suppression rules<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>
- `MetalNegRouteCandidateGeneratorTest.java` — plan-copy migration and unchanged version-12 [identity](../../../../glossary.md#term-identity-meanings)

Documentation and [planning](../../../../glossary.md#term-planning):

- `docs/backend-guide/metal-backend.md`
- `docs/api/runtime-api.md`
- `docs/architecture/contracts/runtime-prepare-engine.md`
- `docs/design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md`
- this task, `../master-plan.md`, and `../../../roadmap.md`

No [Engine](../../../../glossary.md#term-engine), [Prepare](../../../../glossary.md#prepare), [Trace](../../../../glossary.md#trace), tuning, benchmark, native, capability, configuration, [ABI](../../../../glossary.md#term-application-binary-interface-abi), workflow, or
[Gradle](../../../../glossary.md#term-gradle) production file may change.

## High-value permanent tests

1. Rewrite the existing custom splat repetition test to require one prepare allocation/upload,
   shared physical input across two runs, distinct run bindings/outputs, exact negation, and exact
   release boundaries.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [release glossary](../../../../glossary.md#term-release))</span>
2. Rewrite the existing six-splat MPSGraph repeated/concurrent fixture to require six preparation
   uploads, zero run uploads, stable exact raw bits/handles, and distinct outputs/workspaces.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [fixture glossary](../../../../glossary.md#term-fixture); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
3. Exercise a shared constant with two Metal consumers and a CPU-source case: exactly one source
   prepared splat in the former; no Metal persistent splat and ordinary transfer in the latter.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
4. Close a prepared execution while two results remain open; require no new run, no early constant
   release, first-result non-release, and last-result exact-once release.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
5. Cover second-splat allocation failure, upload failure, executable/finalizer-result failure,
   binding/setup failure followed by successful reuse, deferred release failure, and observer
   failure. Assert primary/suppressed order, reverse cleanup, zero RUN, and no live handle.<span data-glossary-navigation="true"> ([observer glossary](../../../../glossary.md#term-observer))</span>
6. Cover [Metal](../../../../glossary.md#term-metal)-to-[CPU](../../../../glossary.md#term-cpu) download/[materialization](../../../../glossary.md#materialization) from a [binding](../../../../glossary.md#term-binding) and upload/output rejection.
7. Retain no-splat/caller-input tests to prove zero constant resources and unchanged ordinary
   behavior. Do not add source-text tests, timing assertions, shape/route tables, or duplicate
   parameter rows.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [parameter glossary](../../../../glossary.md#term-parameter))</span>

## [Validation](../../../../glossary.md#term-validation)

Focused portable proof passed:

```bash
./gradlew :backends:metal:test \
  --tests '*MetalNegPreparedExecutionTest' \
  --tests '*MetalNegRouteCandidateGeneratorTest'
```

[JUnit](../../../../glossary.md#term-junit) XML reported 49 execution tests with zero failures/errors and three expected environment-
gated native skips, plus nine candidate/[identity](../../../../glossary.md#term-identity-meanings) tests with zero skips/failures/errors. The focused
fake-native tests assert exact native create/upload/[release](../../../../glossary.md#term-release) counts, [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), first-consumer source
[ownership](../../../../glossary.md#term-ownership-meanings), repeated/concurrent [binding](../../../../glossary.md#term-binding) identity, output/workspace isolation, read/write rejection,
open-result close deferral, later-[run](../../../../glossary.md#term-run) recovery, [trace](../../../../glossary.md#trace) truth, reverse [rollback](../../../../glossary.md#term-rollback), and failure
suppression.

The current [native library](../../../../glossary.md#term-native-library) was rebuilt, selected by its explicit absolute path, and exactly the two
authorized real-[device](../../../../glossary.md#term-device) tests were rerun:

```bash
./native/metal-macos-arm64/build.sh
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test \
    --tests '*MetalNegPreparedExecutionTest.nativePreparedNegRoundTripAndReuse' \
    --tests '*MetalNegPreparedExecutionTest.realDeviceTypedMpsGraphInitializesLogicalSplatsPerRun' \
    --rerun-tasks
```

The matching JUnit XML reports exactly two tests and zero skipped, failures, and errors. [Metal](../../../../glossary.md#term-metal) and
[Runtime](../../../../glossary.md#term-runtime) Javadocs passed. Focused `RuntimeDependencyAndHotPathContractTest` passed. The final
documentation-only limitation correction passed changed-Markdown local-link/fence/final-newline,
`git diff --check`, exact path-[scope](../../../../glossary.md#term-scope), and clean-[commit](../../../../glossary.md#term-commit) validation. Per instruction, no full
repository build was run; the one final full build remains reserved for program verification.

No benchmark was run. The existing [singleton](../../../../glossary.md#term-singleton)-NEG benchmark has one caller-backed input and no
[constant](../../../../glossary.md#term-constant), so it cannot measure this optimization.

## Acceptance criteria

1. Exactly one immutable buffer is prepared per source-owned Metal splat and no non-source duplicate
   is prepared.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [buffer glossary](../../../../glossary.md#term-buffer); [Metal glossary](../../../../glossary.md#term-metal))</span>
2. Every run gets a fresh valid read-only binding; repeated/concurrent runs share only exact physical
   constant storage and retain distinct mutable outputs/workspaces.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding); [constant glossary](../../../../glossary.md#term-constant); [mutable glossary](../../../../glossary.md#term-mutable))</span>
3. The deterministic custom and six-splat counts match the targets above with exact raw-bit and
   numerical output preservation.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
4. Transfer/materialization read paths accept the binding; every write path rejects it before native
   mutation.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
5. Prepared close, open results, concurrent runs, setup failure, finalization rollback, and deferred
   native cleanup preserve exact ownership, reverse order, attempt-all, and primary/suppressed
   behavior with no leak or double release.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [release glossary](../../../../glossary.md#term-release))</span>
6. PREPARE/RUN trace truth and version-12 tuning/candidate identity are unchanged; public API shape,
   capability, native ABI, and route policy are unchanged.<span data-glossary-navigation="true"> ([PREPARE glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace); [identity glossary](../../../../glossary.md#term-identity-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
7. Output and address-workspace pooling are explicitly rejected under the current lifecycle rather
   than implemented partially or called complete by omission.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
8. Focused fake-native tests, exactly two real-device no-skip tests, Javadocs, architecture/docs/diff
   validation, and an independent Class C review pass. No full repository build is run now.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [validation glossary](../../../../glossary.md#term-validation))</span>

## Result

Implemented from [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit) `08b9e7ea7782bc0165cb6b45c409afdb59673c2e`.
Each first-consumer/source-owned [Metal](../../../../glossary.md#term-metal) splat is allocated, raw-bit-filled, and uploaded exactly once
per [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution). Every [run](../../../../glossary.md#term-run) receives a fresh authenticated read-only child [binding](../../../../glossary.md#term-binding); route
invocation, Metal-to-[CPU](../../../../glossary.md#term-cpu) transfer, and [canonical](../../../../glossary.md#term-canonical-form) [materialization](../../../../glossary.md#materialization) unwrap it only for reads. Prepared
[owner](../../../../glossary.md#term-owner) close rejects new bindings and exact-once native [release](../../../../glossary.md#term-release) is deferred through any already-open
result. Finalization and all failure paths preserve acquisition order, reverse cleanup, and
primary/suppressed evidence.

The deterministic custom two-run path is reduced from four creates/two uploads to three creates/one
upload. The six-splat/eight-[target](../../../../glossary.md#term-target) two-run path is reduced from twenty-eight route-internal creates
and twelve uploads to twenty-two creates and six uploads. Exact [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), numerics, [trace](../../../../glossary.md#trace) semantics,
version-12 [identities](../../../../glossary.md#term-identity-meanings), public [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape), capability, route policy, native [ABI](../../../../glossary.md#term-application-binary-interface-abi), and native exports
remain unchanged.

The optimization todo is closed without a misleading general pool: [mutable](../../../../glossary.md#term-mutable) outputs remain fresh
because [`RunResult`](../../../../glossary.md#run-result--runresult) owns them beyond [synchronous](../../../../glossary.md#term-synchronous) return, and [MPSGraph](../../../../glossary.md#term-mpsgraph) address workspaces remain
fresh because cold binding mutates their per-run pointer arrays. No pool, [cache](../../../../glossary.md#term-cache), [matrix](../../../../glossary.md#term-matrix-meanings), benchmark,
retry, fallback, or cross-preparation reuse was added. Independent Class C review approved
implementation `89f9fbb9` with zero findings; Task 0048 is Complete.
