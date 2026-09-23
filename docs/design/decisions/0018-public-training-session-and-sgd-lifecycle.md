# ADR 0018: Public Training Session and SGD Lifecycle

## Status

Accepted — 2026-09-24

## Context

Synaptik already has compiler-owned functional autograd, ordered gradient publication roles, one
public Engine compile/Prepare/Runtime path, and a reusable `InferenceSession`. The NN extension
owns deterministic recursive parameter paths and stable `Parameter` wrappers, but replacing a
wrapper's Tensor would not change a graph that was already compiled from the earlier Tensor
identity. The Training extension had no executable optimizer or session lifecycle.

A public training workflow therefore needs to reuse the prepared Engine path, keep graph input and
gradient occurrence authority intact, update parameter values without replacing compiled Tensor
identities, and make optimizer state, atomicity, concurrency, and closure explicit. The
[Extensions and training contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
is authoritative; this record explains the selected decision.

## Decision drivers

- Compile and prepare the combined forward/backward graph exactly once per training session.
- Reuse `CompiledGraph.inputs()`, gradient `targetIndex`, `InferenceSession`, and `RunResult`
  rather than create a second scheduler, runner, or publication model.
- Preserve stable NN parameter paths, wrappers, Tensor identities, and host-storage occurrences.
- Provide a real built-in optimizer with persistent per-parameter state and exact update order.
- Validate all fallible work before parameter or optimizer mutation.
- Keep Training independent of Runtime, Prepare, tuning, and concrete backend modules.
- Make standard runs, gradient accumulation, state snapshots, restore, and close races deterministic.

## Options considered

### Recompile after every `Parameter.replace`

This would make updated bindings visible but would repeat compiler and preparation work on every
step, discard optimizer/session lifecycle value, and violate the requested reusable execution
boundary.

### Add optimizer operations to the compiled graph

This could eventually enable fused backend routes, but current architecture explicitly keeps the
initial optimizer after gradient publication. It would require new Model operations, compiler
semantics, Prepare contracts, and backend implementations beyond the authorized lifecycle.

### Give Training a private Runtime or backend callback

A callback, reflection bridge, raw representation API, or private scheduler would hide the same
forbidden dependency behind another abstraction and duplicate Engine authority.

### Depend on the public Engine facade and update stable host parameter storage

Training can create one ordinary gradient `CompiledGraph`, own one `InferenceSession`, detach its
ordered publications, and perform backend-neutral optimizer arithmetic over supported Model host
storage. Stable parameter Tensors remain the graph inputs while their values change in place.

## Decision

The selected design is the public-Engine option. `extensions/training` intentionally adds one
Class C dependency on `modules/engine`. Because Training's public signatures also name NN
`Module` and Model Tensor/schema types, its NN, Model, and Engine project dependencies are all
published Gradle `api` edges. Training may use only the public Engine facade and remains forbidden
from depending on Runtime, Prepare, tuning, or concrete backends. `TrainingSession.open` borrows a
caller-owned Engine and Module, snapshots all recursive parameter paths and stable
wrapper/Tensor/storage identities, performs one scalar-objective first-order compile, and opens
one reusable `InferenceSession`. The session owns that inference session and all optimizer state;
it does not own the Engine or Module.

The initial update domain is fully static, non-empty, dense-contiguous, offset-zero, non-view
`FLOAT32` and `FLOAT64` parameters backed by exact-capacity writable, shareable native host
storage. Initial parameter values must be finite. Empty or non-finite state is rejected during
capture before private seed allocation or compilation. This bounded carrier requirement matches
the current CPU prepared-executable input contract while remaining a Model-owned `MemorySegment`,
not a CPU representation. Updating those bytes in place preserves compiled Tensor identity.
Callers grant exclusive use of the Module wrappers and parameter storage until close. Each run
supplies only the remaining compiled inputs; the session inserts captured parameters and its
scalar cotangent seed in Compiler occurrence order.

`Sgd` is the first built-in optimizer. Its immutable configuration includes learning rate,
momentum, dampening, coupled weight decay, and optional Nesterov momentum. The first momentum
step stores the adjusted gradient without dampening; later steps add
`(1 - dampening) * adjusted` to the prior scaled slot. Nesterov requires positive momentum and
zero dampening. Session-owned momentum slots start at zero and persist by stable parameter path.
Parameter candidates, momentum candidates, and accumulation candidates use reusable primitive
buffers. All Engine work, publication-role checks, decoding, finite-value checks, arithmetic, and
complete precommit validation finish before ordered parameter writes and state publication. No
optimizer-loop element allocation occurs.

`GradientMode` distinguishes reset-and-step, accumulate-only, and accumulate-and-step runs.
Optimizer step numbers count successful updates from one; execution numbers count every
successful forward/backward run. Current parameters, gradients, accumulation, momentum,
arithmetic intermediates, and candidates must remain finite; signed zero is finite, while NaN,
infinity, and represented overflow are rejected. A failed operation changes neither counters nor
parameter, momentum, or accumulation state. The detached `TrainingStep` and immutable
`TrainingState` remain usable after session and Engine close. Restore validates the complete
optimizer configuration, paths, schemas, counters, finite payloads, and lengths before
installation.

Only one training operation may be admitted at a time. Admission precedes argument validation and
uses Engine-closed, session-closed, then busy precedence. Close first rejects later admission,
waits for an admitted operation without holding an Engine/Runtime lock, attempts all owned cleanup
once, and retains any close failure for concurrent and repeated callers. An operation admitted
before session close begins finishes before cleanup. Engine close remains the final composition
boundary and wins operation-admission diagnostics whenever it has begun closure.

## Consequences

### Positive

- Repeated training steps pay one compile and one prepare.
- Compiler input order and gradient target positions remain the sole mapping authorities.
- SGD state, accumulation, step numbering, snapshots, restore, and lifecycle are public and
  deterministic.
- CPU, Metal, and supported mixed schedules use the same path; unsupported execution fails before
  optimizer mutation.
- Training imports no concrete backend and exposes no backend representation.

### Negative and limits

- The Training extension now publishes API dependencies on NN, Model, and Engine's public facade;
  the Engine edge remains an intentional exception to its earlier lower-only dependency set.
- Parameter bindings and host storage must remain exclusive and stable for the session lifetime.
- The first domain excludes empty/non-finite parameters, BFLOAT16, unresolved/strided/view layouts,
  JVM-heap, read-only, thread-confined or device-only parameter storage, parameter groups,
  Adam/AdamW, mixed precision, schedulers, and durable file checkpoints.
- Forward expressions that change structure or parameter Tensor identity require a new session.

### Follow-up

A later task may add parameter groups and Adam/AdamW within the same lifecycle. Durable checkpoint
encoding remains a separate Checkpoint-extension concern. Compiled or fused optimizer updates
require another explicit architecture decision.

## Related documentation

- [Authoritative extensions/training contract](../../architecture/contracts/extensions-training.md)
- [Training graph](../../architecture/training-graph.md)
- [Training API](../../api/training-api.md)
- [Training task 0001](../../planning/extensions/training/tasks/0001-public-training-session-and-sgd-lifecycle.md)
