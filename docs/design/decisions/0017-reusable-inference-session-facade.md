# ADR 0017: Reusable Inference Session Facade

## Status

Accepted — 2026-09-23

## Context

The public Engine already exposes the complete compile, prepare, run, result, and materialization
lifecycle. Its `PreparedExecution` is immutable and reusable; Runtime creates one isolated
`RunState` for every sequential or concurrent run; preparation captures direct input and
publication adapters; and `RunResult` already owns ordered typed publication occurrences and
materialization lifetime. A consumer can therefore perform repeated inference correctly, but must
manually retain and coordinate the compiled and prepared handles.

The missing public concept is one user-facing owner that fixes a graph and its preparation once,
then exposes repeated input binding without duplicating compilation, scheduling, execution, or
results. The
[Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#public-reusable-inference-session)
is authoritative. This record explains the selected API and does not replace that contract.

## Decision drivers

- Reuse the existing Engine compile/prepare/run lifecycle rather than create a parallel pipeline.
- Preserve `CompiledGraph.inputs()` and existing publication metadata as the only occurrence-order
  authorities.
- Preserve Runtime's unique prepared lease, per-run isolation, and non-waiting close protocol.
- Keep provider, availability, registry, backend selection, route selection, and schedule assembly
  out of repeated runs.
- Preserve direct captured adapters for CPU, Metal, and bounded mixed-owner schedules.
- Make Engine, session, result, input-storage, and detached-value ownership explicit.
- Add no avoidable hot-path allocation or copy beyond the existing run contract.

## Options considered

### Thin session over one existing prepared handle

`Engine.session(CompiledGraph)` prepares once and returns an `InferenceSession` with
`compiledGraph()`, `run(List<Tensor>)`, `isClosed()`, and `close()`. The session owns exactly one
existing ordinary prepared handle and delegates every call to existing Engine run orchestration.

### Session with raw storage maps or new input/output value types

A new map keyed by names or tensor IDs, a positional storage array, or another typed result could
hide Tensor identity and existing occurrence metadata. It would also introduce another binding
convention, more per-run allocation, or a parallel materialization contract without a current
capability need.

### Session that owns or constructs its Engine

A static session factory could open CPU implicitly or absorb an explicitly composed Engine. That
would hide backend composition and make integration ownership and shutdown order ambiguous.

### Cache compiled or prepared state inside one-shot Engine methods

Implicit caching in `compute(...)` or `backward(...)` would make cache identity, invalidation,
backend composition, resource ownership, and cleanup invisible. It would also change the
established fresh-call semantics of those conveniences.

### Keep only the explicit prepared-handle lifecycle

The status quo is mechanically sufficient but does not provide the authorized user-facing fixed-
graph inference owner. Every consumer must manually carry two handles and restate their lifecycle.

## Decision

Synaptik adds the thin session option. A caller explicitly compiles graph outputs with an existing
Engine, then calls `engine.session(graph)`. Session construction validates exact Engine ownership,
performs exactly one ordinary preparation under Engine lifecycle admission, and registers only the
session as the outward owner of that preparation. The session borrows its Engine; ownership of the
Engine, registry, and backend integrations never transfers.

`compiledGraph()` returns the exact graph. Its immutable `inputs()` list remains the stable logical
input membership and Compiler occurrence order. `run(List<Tensor>)` keeps the existing arbitrary
caller list order, exact Tensor-identity and complete-descriptor validation, and compiled-order
snapshot of current caller-owned storage. The graph's existing publication specifications fix
forward-then-gradient result order and preserve aliases as distinct occurrences.

Every run delegates to the existing `Engine.run` path over the session's one prepared handle.
Runtime creates one isolated `RunState`; results remain the existing independently closeable
`RunResult`; materialization still authenticates the exact publication object and returns a fresh
detached `HostTensorValue`. No compile, prepare, provider, availability, registry, backend, kernel,
route, or scheduling decision occurs per run.

Session close delegates to the prepared handle and adds no lease. It rejects later runs, does not
wait for a run that already acquired Runtime's lease, and does not close an already returned
result. Engine closure remains the final boundary: it waits for admitted Engine operations, closes
results first, sessions and standalone preparations second, and integrations last.

The lower-level public `prepare(...)` and `engine.run(...)` methods remain supported for direct
prepared ownership and tuned-preparation handoff. The new facade does not supersede them.

## Rationale

The selected API names the lifetime users already need while preserving every inward authority.
It creates no second compiler, scheduler, runner, result, cache, or backend abstraction. Exact
Engine and graph identity keeps composition visible. Reusing the prepared handle delegates the
close race to Runtime's established unique lease authority, and delegating run preserves direct
captured adapters and current hot-path cost.

A raw-storage session would be superficially smaller at call sites but would create a second public
binding contract beside Tensor identity. Keeping `List<Tensor>` also lets each call use the Tensor's
current compatible storage association without weakening descriptor or lifetime validation.

## Consequences

### Positive

- Consumers can compile and prepare a fixed graph once and express its reuse with one public owner.
- CPU, Metal, and supported mixed-owner sessions share identical lifecycle and result semantics.
- Concurrent calls reuse immutable prepared state while retaining isolated mutable Runtime state.
- Existing input and publication occurrence metadata remains authoritative.
- Existing direct-adapter and no-registry-lookup guarantees carry through unchanged.

### Negative and risks

- The public surface gains one type and one Engine method in addition to the lower-level lifecycle.
- Sessions borrow Engine lifetime, so callers must keep both scopes explicit.
- Per-call inputs remain identity-bearing Tensors rather than raw positional storage values.
- Closing a session does not imply closing returned results; callers must continue to close each
  result independently.

### Migration, testing, and follow-up

No existing caller is forced to migrate. Ordinary repeated-inference documentation now recommends
the session facade; direct prepared ownership and tuned preparation retain their current surface.
Tests cover one-time preparation, repeated and concurrent runs, current storage snapshots,
input validation, stable forward/gradient/alias occurrence order, exact materialization lifetime,
session/Engine/result close transitions, close during in-flight runs, direct-adapter registry
poisoning, standard CPU, real Metal, and bounded mixed-owner execution. Any future raw-storage,
asynchronous, dynamic-shape, device-result, or tuning-session API requires a separate decision.

## Related documentation

- [Authoritative Runtime, Prepare, and Engine contract](../../architecture/contracts/runtime-prepare-engine.md#public-reusable-inference-session)
- [Lifecycle explanation](../../architecture/lifecycle.md#current-public-engine-lifecycle)
- [Runtime API](../../api/runtime-api.md#current-ordinary-engine-boundary)
- [Run a reusable inference session](../../user-guide/running-models.md)
- [Engine task 0017](../../planning/modules/engine/tasks/0017-reusable-inference-session-api.md)
- [ADR 0016: CPU/Metal mixed-owner prepared schedule](0016-cpu-metal-mixed-owner-schedule.md)
