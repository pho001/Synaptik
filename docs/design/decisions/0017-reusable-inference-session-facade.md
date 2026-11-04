# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0017: Reusable [Inference](../../glossary.md#term-inference) Session [Facade](../../glossary.md#term-facade)

## Status

Accepted — 2026-09-23. The thin session [facade](../../glossary.md#term-facade) is superseded on 2026-10-16 by
[ADR 0039's prepared-owner cutover](0039-portable-graph-model-persistence.md#engine-result-and-exclusion-lifecycle)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [owner glossary](../../glossary.md#term-owner))</span>.
The rationale below is historical; Engine0021 removes [InferenceSession](../../glossary.md#former-inference-session--inferencesession)/[Engine](../../glossary.md#term-engine).session while
preserving the prepared recipe, [Runtime](../../glossary.md#term-runtime) [lease](../../glossary.md#term-lease) and result lifetimes.

## Context

The public [Engine](../../glossary.md#term-engine) already exposes the complete [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), [run](../../glossary.md#term-run), result, and [materialization](../../glossary.md#materialization)
[lifecycle](../../glossary.md#lifecycle). Its [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) is [immutable](../../glossary.md#term-immutable) and reusable; [Runtime](../../glossary.md#term-runtime) creates one isolated
[`RunState`](../../glossary.md#run-state--runstate) for every sequential or concurrent run; preparation [captures](../../glossary.md#term-capture-meanings) direct input and
[publication](../../glossary.md#term-publication) adapters; and [`RunResult`](../../glossary.md#run-result--runresult) already owns ordered typed publication occurrences and
materialization [lifetime](../../glossary.md#term-lifetime). A consumer can therefore perform repeated [inference](../../glossary.md#term-inference) correctly, but must
manually retain and coordinate the compiled and prepared handles.

The missing public concept is one user-facing [owner](../../glossary.md#term-owner) that fixes a [graph](../../glossary.md#graph) and its preparation once,
then exposes repeated input [binding](../../glossary.md#term-binding) without duplicating [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), scheduling, execution, or
results. The
[Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#public-reusable-inference-session)
is authoritative. This record explains the selected [API](../../glossary.md#term-application-programming-interface-api) and does not replace that contract.

## Decision drivers

- Reuse the existing [Engine](../../glossary.md#term-engine) [compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare)/[run](../../glossary.md#term-run) [lifecycle](../../glossary.md#lifecycle) rather than create a parallel pipeline.
- Preserve `CompiledGraph.inputs()` and existing publication metadata as the only occurrence-order
  authorities.<span data-glossary-navigation="true"> ([publication glossary](../../glossary.md#term-publication); [metadata glossary](../../glossary.md#term-metadata); [occurrence glossary](../../glossary.md#term-occurrence))</span>
- Preserve [Runtime](../../glossary.md#term-runtime)'s unique prepared [lease](../../glossary.md#term-lease), per-run isolation, and non-waiting close protocol.
- Keep provider, availability, registry, backend selection, route selection, and schedule assembly
  out of repeated runs.<span data-glossary-navigation="true"> ([registry glossary](../../glossary.md#term-registry); [backend glossary](../../glossary.md#backend))</span>
- Preserve direct captured adapters for [CPU](../../glossary.md#term-cpu), [Metal](../../glossary.md#term-metal), and bounded mixed-[owner](../../glossary.md#term-owner) schedules.
- Make Engine, session, result, input-storage, and [detached](../../glossary.md#detached-model-results)-value [ownership](../../glossary.md#term-ownership-meanings) explicit.
- Add no avoidable hot-path allocation or copy beyond the existing run contract.

## Options considered

### Thin session over one existing prepared handle

`Engine.session(CompiledGraph)` prepares once and returns an [`InferenceSession`](../../glossary.md#former-inference-session--inferencesession) with
`compiledGraph()`, `run(List<Tensor>)`, `isClosed()`, and `close()`. The session owns exactly one
existing ordinary prepared handle and delegates every call to existing [Engine](../../glossary.md#term-engine) [run](../../glossary.md#term-run) orchestration.

### Session with raw storage maps or new input/output value types

A new map keyed by names or [tensor](../../glossary.md#tensor) IDs, a positional storage array, or another typed result could
hide Tensor [identity](../../glossary.md#term-identity-meanings) and existing [occurrence](../../glossary.md#term-occurrence) [metadata](../../glossary.md#term-metadata). It would also introduce another [binding](../../glossary.md#term-binding)
convention, more per-[run](../../glossary.md#term-run) allocation, or a parallel [materialization](../../glossary.md#materialization) contract without a current
capability need.

### Session that owns or constructs its [Engine](../../glossary.md#term-engine)

A static session factory could open [CPU](../../glossary.md#term-cpu) implicitly or absorb an explicitly composed Engine. That
would hide [backend composition](../../glossary.md#backend-composition) and make integration [ownership](../../glossary.md#term-ownership-meanings) and shutdown order ambiguous.

### [Cache](../../glossary.md#term-cache) compiled or prepared state inside one-shot [Engine](../../glossary.md#term-engine) methods

Implicit caching in `compute(...)` or `backward(...)` would make cache [identity](../../glossary.md#term-identity-meanings), invalidation,
[backend composition](../../glossary.md#backend-composition), resource [ownership](../../glossary.md#term-ownership), and cleanup invisible. It would also change the
established fresh-call semantics of those conveniences.

### Keep only the explicit prepared-handle [lifecycle](../../glossary.md#lifecycle)

The status quo is mechanically sufficient but does not provide the authorized user-facing fixed-
[graph](../../glossary.md#graph) [inference](../../glossary.md#term-inference) [owner](../../glossary.md#term-owner). Every consumer must manually [carry](../../glossary.md#term-carry) two handles and restate their lifecycle.

## Decision

Synaptik adds the thin session option. A caller explicitly compiles [graph](../../glossary.md#graph) outputs with an existing
[Engine](../../glossary.md#term-engine), then calls `engine.session(graph)`. Session construction validates exact Engine [ownership](../../glossary.md#term-ownership-meanings),
performs exactly one ordinary preparation under Engine [lifecycle](../../glossary.md#lifecycle) [admission](../../glossary.md#term-admission), and registers only the
session as the outward [owner](../../glossary.md#term-owner) of that preparation. The session borrows its Engine; ownership of the
Engine, [registry](../../glossary.md#term-registry), and [backend](../../glossary.md#backend) integrations never transfers.

`compiledGraph()` returns the exact graph. Its [immutable](../../glossary.md#term-immutable) `inputs()` list remains the stable logical
input membership and [Compiler](../../glossary.md#term-compiler) [occurrence](../../glossary.md#term-occurrence) order. `run(List<Tensor>)` keeps the existing arbitrary
caller list order, exact [Tensor](../../glossary.md#tensor)-[identity](../../glossary.md#term-identity-meanings) and complete-[descriptor](../../glossary.md#term-descriptor) [validation](../../glossary.md#term-validation), and compiled-order
[snapshot](../../glossary.md#term-snapshot) of current caller-owned storage. The graph's existing [publication](../../glossary.md#term-publication) specifications fix
forward-then-[gradient](../../glossary.md#term-gradient) result order and preserve [aliases](../../glossary.md#aliasing-and-arena) as distinct occurrences.

Every [run](../../glossary.md#term-run) delegates to the existing `Engine.run` path over the session's one prepared handle.
[Runtime](../../glossary.md#term-runtime) creates one isolated [`RunState`](../../glossary.md#run-state--runstate); results remain the existing independently closeable
[`RunResult`](../../glossary.md#run-result--runresult); [materialization](../../glossary.md#materialization) still authenticates the exact publication object and returns a fresh
[detached](../../glossary.md#detached-model-results) [`HostTensorValue`](../../glossary.md#host-tensor-value--hosttensorvalue). No [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), provider, availability, registry, backend, [kernel](../../glossary.md#kernel),
route, or scheduling decision occurs per run.

Session close delegates to the prepared handle and adds no [lease](../../glossary.md#term-lease). It rejects later runs, does not
wait for a run that already acquired Runtime's lease, and does not close an already returned
result. Engine closure remains the final boundary: it waits for admitted Engine operations, closes
results first, sessions and standalone preparations second, and integrations last.

The lower-level public `prepare(...)` and `engine.run(...)` methods remain supported for direct
prepared ownership and tuned-preparation [handoff](../../glossary.md#term-handoff). The new [facade](../../glossary.md#term-facade) does not supersede them.

## Rationale

The selected [API](../../glossary.md#term-application-programming-interface-api) names the [lifetime](../../glossary.md#term-lifetime) users already need while preserving every inward authority.
It creates no second [compiler](../../glossary.md#term-compiler), scheduler, runner, result, [cache](../../glossary.md#term-cache), or [backend](../../glossary.md#backend) abstraction. Exact
[Engine](../../glossary.md#term-engine) and [graph](../../glossary.md#graph) [identity](../../glossary.md#term-identity-meanings) keeps composition visible. Reusing the prepared handle delegates the
close [race](../../glossary.md#term-race) to [Runtime](../../glossary.md#term-runtime)'s established unique [lease](../../glossary.md#term-lease) authority, and delegating [run](../../glossary.md#term-run) preserves direct
captured adapters and current hot-path cost.

A raw-storage session would be superficially smaller at call sites but would create a second public
[binding](../../glossary.md#term-binding) contract beside [Tensor](../../glossary.md#tensor) identity. Keeping `List<Tensor>` also lets each call use the Tensor's
current compatible storage association without weakening [descriptor](../../glossary.md#term-descriptor) or lifetime [validation](../../glossary.md#term-validation).

## Consequences

### Positive

- Consumers can [compile](../../glossary.md#compile) and [prepare](../../glossary.md#prepare) a fixed [graph](../../glossary.md#graph) once and express its reuse with one public [owner](../../glossary.md#term-owner).
- [CPU](../../glossary.md#term-cpu), [Metal](../../glossary.md#term-metal), and supported mixed-owner sessions share identical [lifecycle](../../glossary.md#lifecycle) and result semantics.
- Concurrent calls reuse [immutable](../../glossary.md#term-immutable) prepared state while retaining isolated [mutable](../../glossary.md#term-mutable) [Runtime](../../glossary.md#term-runtime) state.
- Existing input and [publication occurrence](../../glossary.md#publication-occurrence) [metadata](../../glossary.md#term-metadata) remains authoritative.
- Existing direct-adapter and no-[registry](../../glossary.md#term-registry)-[lookup](../../glossary.md#term-lookup) guarantees [carry](../../glossary.md#term-carry) through unchanged.

### Negative and risks

- The public surface gains one type and one [Engine](../../glossary.md#term-engine) method in addition to the lower-level [lifecycle](../../glossary.md#lifecycle).
- Sessions borrow Engine [lifetime](../../glossary.md#term-lifetime), so callers must keep both scopes explicit.
- Per-call inputs remain [identity](../../glossary.md#term-identity-meanings)-bearing [Tensors](../../glossary.md#tensor) rather than raw positional storage values.
- Closing a session does not imply closing returned results; callers must continue to close each
  result independently.

### Migration, testing, and follow-up

No existing caller is forced to migrate. Ordinary repeated-[inference](../../glossary.md#term-inference) documentation now recommends
the session [facade](../../glossary.md#term-facade); direct prepared [ownership](../../glossary.md#term-ownership-meanings) and tuned preparation retain their current surface.
Tests cover one-time preparation, repeated and concurrent runs, current storage snapshots,
input [validation](../../glossary.md#term-validation), stable forward/[gradient](../../glossary.md#term-gradient)/[alias](../../glossary.md#aliasing-and-arena) [occurrence](../../glossary.md#term-occurrence) order, exact [materialization](../../glossary.md#materialization) [lifetime](../../glossary.md#term-lifetime),
session/[Engine](../../glossary.md#term-engine)/result close transitions, close during in-flight runs, direct-adapter [registry](../../glossary.md#term-registry)
poisoning, standard [CPU](../../glossary.md#term-cpu), real [Metal](../../glossary.md#term-metal), and bounded mixed-[owner](../../glossary.md#term-owner) execution. Any future raw-storage,
[asynchronous](../../glossary.md#term-asynchronous), dynamic-[shape](../../glossary.md#shape), [device](../../glossary.md#term-device)-result, or tuning-session [API](../../glossary.md#term-application-programming-interface-api) requires a separate decision.

## Related documentation

- [Authoritative Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#public-reusable-inference-session)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Engine glossary](../../glossary.md#term-engine))</span>
- [Lifecycle explanation](../../architecture/lifecycle.md#current-public-engine-lifecycle)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Runtime API](../../api/runtime-api.md#current-ordinary-engine-boundary)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [Run a reusable inference session](../../user-guide/running-models.md)<span data-glossary-navigation="true"> ([Run glossary](../../glossary.md#term-run); [inference glossary](../../glossary.md#term-inference))</span>
- [Engine task 0017](../../planning/modules/engine/tasks/0017-reusable-inference-session-api.md)
- [ADR 0016: CPU/Metal mixed-owner prepared schedule](0016-cpu-metal-mixed-owner-schedule.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../glossary.md#term-cpu); [Metal glossary](../../glossary.md#term-metal); [owner glossary](../../glossary.md#term-owner); [prepared schedule glossary](../../glossary.md#prepared-schedule--preparedschedule))</span>
