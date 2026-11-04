# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0018: Public [Training Session](../../glossary.md#training-session--trainingsession) and [SGD](../../glossary.md#term-stochastic-gradient-descent-sgd) [Lifecycle](../../glossary.md#lifecycle)


## Editorial reading context

The application opens a Training session using an Engine and admitted parameter bindings. Training owns optimizer state and a prepared objective, while Compiler constructs gradients and Engine executes them. Parameters are updated in place so compiled identities stay valid. The reconciliations in Status distinguish the original NN-taking signatures from the later generic adapter boundary.

Use the [current session setup and storage lifetime](../../user-guide/training.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-09-24. [Engine](../../glossary.md#term-engine) [owner](../../glossary.md#term-owner) references reconciled on 2026-10-16 with
[ADR 0039](0039-portable-graph-model-persistence.md#engine-result-and-exclusion-lifecycle)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>:
Training owns Engine [PreparedExecution](../../glossary.md#prepared-execution--preparedexecution) through [prepare](../../glossary.md#prepare)/[run](../../glossary.md#term-run); [optimizer](../../glossary.md#term-optimizer) decisions are unchanged.

Training0007 reconciliation (2026-10-16): the
[generic live boundary](0039-portable-graph-model-persistence.md#necessary-boundary-types-not-a-new-model-hierarchy)
replaces [Module](../../glossary.md#term-module)-taking open [signatures](../../glossary.md#term-signature) with [`TrainingParameters`](../../glossary.md#live-parameter-source--trainingparameters). Physical `modules/training`
retains `:extensions:training` and its packages, with exactly public [Model](../../glossary.md#term-model)/Engine edges and no [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
[dependency](../../glossary.md#term-dependency). NN's live adapter owns wrapper/[Tensor](../../glossary.md#tensor) membership checks; Training owns storage checks.
The original NN edge/[capture](../../glossary.md#term-capture-meanings) wording below is historical. Prepared [ownership](../../glossary.md#term-ownership-meanings), native [admission](../../glossary.md#term-admission),
[SGD](../../glossary.md#term-stochastic-gradient-descent-sgd) arithmetic, error precedence, [rollback](../../glossary.md#term-rollback) and cleanup remain unchanged.

## Context

Synaptik already has [compiler](../../glossary.md#term-compiler)-owned functional [autograd](../../glossary.md#autograd), ordered [gradient](../../glossary.md#term-gradient) [publication](../../glossary.md#term-publication) roles, one
public [Engine](../../glossary.md#term-engine) [compile](../../glossary.md#compile)/[Prepare](../../glossary.md#prepare)/[Runtime](../../glossary.md#term-runtime) path, and a reusable Engine [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution). The [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) extension
owns deterministic recursive [parameter](../../glossary.md#term-parameter) paths and stable `Parameter` wrappers, but replacing a
wrapper's [Tensor](../../glossary.md#tensor) would not change a [graph](../../glossary.md#graph) that was already compiled from the earlier Tensor
[identity](../../glossary.md#term-identity-meanings). The Training extension had no executable [optimizer](../../glossary.md#term-optimizer) or session [lifecycle](../../glossary.md#lifecycle).

A public training workflow therefore needs to reuse the prepared Engine path, keep graph input and
gradient [occurrence](../../glossary.md#term-occurrence) authority intact, update parameter values without replacing compiled Tensor
identities, and make [optimizer state](../../glossary.md#term-optimizer-state), [atomicity](../../glossary.md#term-atomicity), [concurrency](../../glossary.md#term-concurrency), and closure explicit. The
[Extensions and training contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
is authoritative; this record explains the selected decision.

## Decision drivers

- [Compile](../../glossary.md#compile) and [prepare](../../glossary.md#prepare) the combined forward/[backward graph](../../glossary.md#backward-graph) exactly once per [training session](../../glossary.md#training-session--trainingsession).
- Reuse `CompiledGraph.inputs()`, [gradient](../../glossary.md#term-gradient) `targetIndex`, [Engine](../../glossary.md#term-engine) [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution), and [`RunResult`](../../glossary.md#run-result--runresult)
  rather than create a second scheduler, runner, or publication model.<span data-glossary-navigation="true"> ([publication glossary](../../glossary.md#term-publication); [model glossary](../../glossary.md#term-model))</span>
- Preserve stable [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [parameter](../../glossary.md#term-parameter) paths, wrappers, [Tensor](../../glossary.md#tensor) [identities](../../glossary.md#term-identity-meanings), and [host](../../glossary.md#term-host)-storage occurrences.
- Provide a real built-in [optimizer](../../glossary.md#term-optimizer) with persistent per-parameter state and exact update order.
- Validate all fallible work before parameter or optimizer mutation.
- Keep Training independent of [Runtime](../../glossary.md#term-runtime), Prepare, tuning, and [concrete backend](../../glossary.md#concrete-backend) [modules](../../glossary.md#term-module).
- Make standard runs, [gradient accumulation](../../glossary.md#term-gradient-accumulation), state snapshots, restore, and close races deterministic.

## Options considered

### Recompile after every `Parameter.replace`

This would make updated [bindings](../../glossary.md#term-binding) visible but would repeat [compiler](../../glossary.md#term-compiler) and preparation work on every
step, discard [optimizer](../../glossary.md#term-optimizer)/session [lifecycle](../../glossary.md#lifecycle) value, and violate the requested reusable execution
boundary.

### Add [optimizer](../../glossary.md#term-optimizer) operations to the [compiled graph](../../glossary.md#compiled-graph--compiledgraphmodel)

This could eventually enable [fused](../../glossary.md#specialization-and-fusion) [backend](../../glossary.md#backend) routes, but current architecture explicitly keeps the
initial optimizer after [gradient](../../glossary.md#term-gradient) [publication](../../glossary.md#term-publication). It would require new [Model](../../glossary.md#term-model) operations, [compiler](../../glossary.md#term-compiler)
semantics, [Prepare](../../glossary.md#prepare) contracts, and backend implementations beyond the authorized [lifecycle](../../glossary.md#lifecycle).

### Give Training a private [Runtime](../../glossary.md#term-runtime) or [backend](../../glossary.md#backend) [callback](../../glossary.md#term-callback)

A callback, reflection bridge, raw representation [API](../../glossary.md#term-application-programming-interface-api), or private scheduler would hide the same
forbidden [dependency](../../glossary.md#term-dependency) behind another abstraction and duplicate [Engine](../../glossary.md#term-engine) authority.

### Depend on the public [Engine](../../glossary.md#term-engine) [facade](../../glossary.md#term-facade) and update stable [host](../../glossary.md#term-host) [parameter](../../glossary.md#term-parameter) storage

Training can create one ordinary [gradient](../../glossary.md#term-gradient) `CompiledGraph`, own one Engine [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution), detach its
ordered publications, and perform [backend](../../glossary.md#backend)-neutral [optimizer](../../glossary.md#term-optimizer) arithmetic over supported [Model](../../glossary.md#term-model) [host
storage](../../glossary.md#host-storage). Stable parameter [Tensors](../../glossary.md#tensor) remain the [graph](../../glossary.md#graph) inputs while their values change in place.

## Decision

The selected design is the public-[Engine](../../glossary.md#term-engine) option. `extensions/training` intentionally adds one
Class C [dependency](../../glossary.md#term-dependency) on `modules/engine`. Because Training's public [signatures](../../glossary.md#term-signature) also name [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
[`Module`](../../glossary.md#term-module) and [Model](../../glossary.md#term-model) [Tensor](../../glossary.md#tensor)/[schema](../../glossary.md#term-schema) types, its NN, Model, and Engine project dependencies are all
published [Gradle](../../glossary.md#term-gradle) `api` edges. Training may use only the public Engine [facade](../../glossary.md#term-facade) and remains forbidden
from depending on [Runtime](../../glossary.md#term-runtime), [Prepare](../../glossary.md#prepare), tuning, or concrete [backends](../../glossary.md#backend). `TrainingSession.open` borrows a
caller-owned Engine and Module, snapshots all recursive [parameter](../../glossary.md#term-parameter) paths and stable
wrapper/Tensor/storage [identities](../../glossary.md#term-identity-meanings), performs one [scalar](../../glossary.md#term-scalar)-[objective](../../glossary.md#term-objective) first-order [compile](../../glossary.md#compile), and opens
one reusable Engine [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution). The session owns that [prepared execution](../../glossary.md#prepared-execution--preparedexecution) and all [optimizer state](../../glossary.md#term-optimizer-state);
it does not own the Engine or Module.

The initial update domain is fully static, non-empty, dense-[contiguous](../../glossary.md#term-contiguous), offset-zero, non-[view](../../glossary.md#term-view-meanings)
[`FLOAT32`](../../glossary.md#term-float32) and [`FLOAT64`](../../glossary.md#term-float64) parameters backed by exact-capacity writable, shareable native [host
storage](../../glossary.md#host-storage). Initial parameter values must be [finite](../../glossary.md#term-finite). Empty or non-finite state is rejected during
[capture](../../glossary.md#term-capture-meanings) before private [seed](../../glossary.md#term-seed) allocation or [compilation](../../glossary.md#compiler-engine-and-execution-stage-names). This bounded [carrier](../../glossary.md#term-storage-carrier) requirement matches
the current [CPU](../../glossary.md#term-cpu) prepared-executable input contract while remaining a Model-owned [`MemorySegment`](../../glossary.md#term-memorysegment),
not a CPU representation. Updating those bytes in place preserves compiled Tensor [identity](../../glossary.md#term-identity-meanings).
Callers grant exclusive use of the Module wrappers and parameter storage until close. Each [run](../../glossary.md#term-run)
supplies only the remaining compiled inputs; the session inserts captured parameters and its
scalar [cotangent seed](../../glossary.md#primal-adjoint-and-differentiation-seed) in [Compiler](../../glossary.md#term-compiler) [occurrence](../../glossary.md#term-occurrence) order.

`Sgd` is the first built-in [optimizer](../../glossary.md#term-optimizer). Its [immutable](../../glossary.md#term-immutable) configuration includes [learning rate](../../glossary.md#term-learning-rate),
[momentum](../../glossary.md#term-momentum), [dampening](../../glossary.md#term-dampening), coupled [weight decay](../../glossary.md#term-weight-decay), and optional [Nesterov](../../glossary.md#term-nesterov) momentum. The first momentum
step stores the adjusted [gradient](../../glossary.md#term-gradient) without dampening; later steps add
`(1 - dampening) * adjusted` to the prior scaled slot. Nesterov requires positive momentum and
zero dampening. Session-owned momentum slots start at zero and persist by stable parameter path.
Parameter candidates, momentum candidates, and accumulation candidates use reusable primitive
[buffers](../../glossary.md#term-buffer). All Engine work, [publication](../../glossary.md#term-publication)-role checks, decoding, finite-value checks, arithmetic, and
complete [precommit](../../glossary.md#term-precommit) [validation](../../glossary.md#term-validation) finish before ordered parameter writes and state publication. No
optimizer-loop [element](../../glossary.md#term-element) allocation occurs.

`GradientMode` distinguishes reset-and-step, accumulate-only, and accumulate-and-step runs.
[Optimizer step](../../glossary.md#optimizer-and-stochastic-gradient-descent) numbers count successful updates from one; execution numbers count every
successful forward/backward run. Current parameters, gradients, accumulation, momentum,
arithmetic intermediates, and candidates must remain finite; [signed zero](../../glossary.md#term-signed-zero) is finite, while [NaN](../../glossary.md#term-nan),
[infinity](../../glossary.md#term-infinity), and represented [overflow](../../glossary.md#term-overflow) are rejected. A failed [operation](../../glossary.md#operation) changes neither counters nor
parameter, momentum, or accumulation state. The [detached](../../glossary.md#detached-model-results) `TrainingStep` and immutable
`TrainingState` remain usable after session and Engine close. Restore validates the complete
optimizer configuration, paths, schemas, counters, finite payloads, and lengths before
installation.

Only one training operation may be admitted at a time. [Admission](../../glossary.md#term-admission) precedes [argument](../../glossary.md#term-argument) validation and
uses Engine-closed, session-closed, then busy precedence. Close first rejects later admission,
waits for an admitted operation without holding an Engine/Runtime lock, attempts all owned cleanup
once, and retains any close failure for concurrent and repeated callers. An operation admitted
before session close begins finishes before cleanup. Engine close remains the final composition
boundary and wins operation-admission diagnostics whenever it has begun closure.

## Consequences

### Positive

- Repeated training steps pay one [compile](../../glossary.md#compile) and one [prepare](../../glossary.md#prepare).
- [Compiler](../../glossary.md#term-compiler) input order and [gradient](../../glossary.md#term-gradient) [target](../../glossary.md#term-target) positions remain the sole mapping authorities.
- SGD state, accumulation, step numbering, snapshots, restore, and lifecycle are public and
  deterministic.<span data-glossary-navigation="true"> ([SGD glossary](../../glossary.md#term-stochastic-gradient-descent-sgd); [lifecycle glossary](../../glossary.md#lifecycle))</span>
- CPU, Metal, and supported mixed schedules use the same path; unsupported execution fails before
  optimizer mutation.<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu); [Metal glossary](../../glossary.md#term-metal); [optimizer glossary](../../glossary.md#term-optimizer))</span>
- Training imports no [concrete backend](../../glossary.md#concrete-backend) and exposes no [backend](../../glossary.md#backend) representation.

### Negative and limits

- The Training extension now publishes API dependencies on NN, Model, and Engine's public facade;
  the Engine edge remains an intentional exception to its earlier lower-only dependency set.<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api); [NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../glossary.md#term-model); [Engine glossary](../../glossary.md#term-engine); [facade glossary](../../glossary.md#term-facade); [dependency glossary](../../glossary.md#term-dependency))</span>
- [Parameter](../../glossary.md#term-parameter) [bindings](../../glossary.md#term-binding) and [host storage](../../glossary.md#host-storage) must remain exclusive and stable for the session [lifetime](../../glossary.md#term-lifetime).
- The first domain excludes empty/non-finite parameters, BFLOAT16, unresolved/strided/view layouts,
  JVM-heap, read-only, thread-confined or device-only parameter storage, parameter groups,
  Adam/AdamW, mixed precision, schedulers, and durable file checkpoints.<span data-glossary-navigation="true"> ([finite glossary](../../glossary.md#term-finite); [BFLOAT16 glossary](../../glossary.md#term-bfloat16); [view glossary](../../glossary.md#term-view-meanings); [layouts glossary](../../glossary.md#layout); [JVM glossary](../../glossary.md#term-java-virtual-machine-jvm); [heap glossary](../../glossary.md#term-heap); [device glossary](../../glossary.md#term-device); [Adam glossary](../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../glossary.md#adaptive-optimizer--adam-adamw); [mixed precision glossary](../../glossary.md#term-mixed-precision); [checkpoints glossary](../../glossary.md#term-checkpoint))</span>
- Forward expressions that change structure or parameter [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) require a new session.

### Follow-up

A later task may add [parameter](../../glossary.md#term-parameter) groups and [Adam](../../glossary.md#adaptive-optimizer--adam-adamw)/[AdamW](../../glossary.md#adaptive-optimizer--adam-adamw) within the same [lifecycle](../../glossary.md#lifecycle). Durable [checkpoint](../../glossary.md#term-checkpoint)
encoding remains a separate Checkpoint-extension concern. Compiled or [fused](../../glossary.md#specialization-and-fusion) [optimizer](../../glossary.md#term-optimizer) updates
require another explicit architecture decision.

## Related documentation

- [Authoritative extensions/training contract](../../architecture/contracts/extensions-training.md)
- [Training graph](../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [Training API](../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [Training task 0001](../../planning/extensions/training/tasks/0001-public-training-session-and-sgd-lifecycle.md)
