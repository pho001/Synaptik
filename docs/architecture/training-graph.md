# Training Graph

This document explains the training graph model established by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The root contract remains authoritative.

The compiler currently consumes all three compile modes. Backward-capable compilation accepts a
bounded functional-gradient request, constructs one or two reverse-mode stages, captures the
forward and derivative expressions together, and derives gradient publication bindings. The
public Training extension now consumes the ordinary Engine's one-stage first-gradient path for
its reusable session; it does not own or alter the combined compile-time graph described here.
This page is subordinate to the Compiler-owned contract.

## Compile modes

Synaptik distinguishes three compile modes:

- **Forward-only mode** (`FORWARD_ONLY`) compiles only the forward computation and its requested publications.
- **Forward-and-backward mode** (`FORWARD_AND_BACKWARD`) constructs gradient Tensor expressions
  from the original forward expression and captures both phases together.
- **Training-step mode** (`TRAINING_STEP`) initially uses that same combined forward/backward
  construction. Optimizer-update graph work remains future.

Both backward-capable modes require the same current `FunctionalGradientRequest`. The compile
mode changes the compile-time graph and publication needs. It does not allow a backend to own
global autograd.

## Combined forward and backward graph

When a compile mode requires gradients, the compiler builds backward expressions before the only
capture:

```text
original forward Tensor expression DAG
  -> fail-closed compiler autograd preflight
  -> gradient formulas through ordinary public Tensor operations
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware capture
  -> immutable combined forward + backward graph
  -> inference, validation, combined optimization, and final validation
  -> publication binding
  -> planning and partitioning
```

The compiler uses exact Tensor identity only for temporary reverse-accumulation bookkeeping and
ordinary `Tensor.add` for contribution accumulation. This is not a second graph representation.
Named compiler rules own dispatch; model operations remain derivative-agnostic. Seeds and other
derivative constants are storage-free leaves registered explicitly as compile-time splats.

Phase-aware capture receives the original forward-producer identity set, forward outputs,
all requested first- and optional second-stage gradient roots and target roles, and exact constant
facts. It assigns graph-local IDs once and retains `FORWARD` or `BACKWARD` on every node. The
separate derivative sidecar records order zero for original forward producers, order one for
first-stage producers, and order two for second-stage producers. Orders one and two are both
`BACKWARD`; derivative order does not replace graph phase.

Gradient publication bindings use orders one and two and are ordered by derivative order, then by
target position within that stage. Each binding retains its target identity and final gradient
value. Two targets may share one captured gradient value; their bindings remain distinct while
the graph output boundary starts with the forward-output prefix and appends that value only if it
was not already listed.

The combined graph is immutable compile-time graph state. Backends receive only their planned
regions during prepare and execute only those prepared regions at runtime.

Compile-time combination does not require a single runtime schedule. Prepare may expose separate forward and backward schedules or one training-step schedule, depending on prepare-time decisions.

## Why combine the graph

Seeing forward and backward work together allows compiler and planning passes to reason across the autograd boundary. This can enable:

- dead-code elimination and common subexpression elimination across both phases;
- cleanup of redundant transposes and no-op unbroadcast operations;
- simpler gradient accumulation;
- reuse of forward values required by backward computation;
- more accurate lifetime and logical memory planning; and
- ownership and partition decisions informed by the full training computation.

These are global graph concerns, so they belong to compiler and backend-neutral planning rather than concrete backends.

The first implementation reuses only already-proved exact rewrites and folding rules. Common
subexpression elimination stays phase-local, and every changed immutable candidate is revalidated.
No new algebraic identity follows merely from combining the phases.

## Failures and bounded derivative order

Preflight rejects any unsupported backward-reachable operation, exact attributes, or required
derivative policy before it constructs backward expressions. Full inference follows capture, so a
later failure can consume temporary Tensor IDs; callers must already treat those IDs as opaque.

The current functional request has exactly one or two stages. Each stage has non-empty ordered
outputs, output-aligned optional seeds, and non-empty identity-unique ordered targets from the
complete original forward inventory. Stage one selects exact requested forward-output Tensors
through `ForwardTensorReference`. Stage two may select only first-stage gradients through
`FirstStageGradientReference(targetIndex)`.

One stage requires `createGraph == false`. Two stages require stage-one
`createGraph == true` and stage-two `createGraph == false`; retention serves only the immediate
second stage in the same compile. An absent seed supplies an exact typed one only for an eligible
scalar output. A present seed must match the selected output's exact Shape and floating type and
must not request gradients. `ERROR` rejects a disconnected target, while `ZERO` produces an
ordinary exact typed zero expression that several target roles may share.

This boundary supports bounded second-stage construction, including a suitably seeded
Hessian-vector product when the selected formulas are covered. It does not support a third stage,
derivative orders above two, arbitrary nesting, a persistent or runtime tape, mutable Tensor
gradient state, or `Tensor.backward()`. Operation-specific derivative coverage remains
fail-closed; successful graph construction alone does not promise preparation or numerical
execution on a particular backend.

## Fixed recurrent scan and the initial BPTT boundary

The current Model fixed recurrent scan is an ordinary flat multi-output forward occurrence, not
a nested training graph. Model task 0025E owns only its fixed forward meanings, descriptor
construction, and canonical outputs. Generic capture preserves one producer as one node, and
current Compiler forward-only inference independently revalidates its fully static descriptors.
Both backward-capable modes reject any complete forward inventory containing recurrence before
any derivative Tensor is allocated.

Backpropagation through time (BPTT) therefore remains unsupported even though output gradient
eligibility is derived from differentiable floating inputs, states, and parameters. The runtime
`INT64[batch]` valid-length input is always non-differentiable and never contributes a gradient.
The fail-closed rule avoids a partially constructed backward expression and prevents a backend
from becoming the owner of global autograd.

This decision selects no saved-gate outputs, tape, checkpointing policy, recomputation policy,
backward operation, derivative formula, or physical saved-state representation. A later Compiler
architecture and implementation task must choose those contracts after the forward executable
path is stable. Current `FORWARD_ONLY` compilation is the only compile mode that adopts the family
until BPTT is designed. See
[ADR 0012](../design/decisions/0012-fixed-recurrent-scan-without-regions.md).

## Optimizer as a backend-agnostic step

`extensions/nn` owns the model-module side of a training step: modules declare their trainable
`Parameter` values, persistent `Buffer` values, and train/eval forward behavior.
`extensions/training` is downstream of that declaration and owns the implemented `Sgd`, `Adam`,
and `AdamW` mathematical updates and `TrainingSession` orchestration. This lets a module remain
usable for inference without importing an optimizer.

An initialized layer can begin with heap-backed parameters, but a session can update only
shareable native parameter storage. The caller may explicitly call
`TrainingParameterStorage.copyToSharedNative(module, arena)` before opening a session. Training
traverses NN's recursive parameter paths, validates the complete source set, copies exact
represented bytes into fresh aligned native segments, and rebinds only Model's borrowed
host-storage associations. The same Parameter and Tensor objects remain in previously constructed
expressions, so the normal Engine compile and prepare path can use them without a second graph
construction route. The caller owns the destination Arena through session close; Training owns
neither arena nor a backend transfer. The [owning Training contract](contracts/extensions-training.md#optimizertraining-lifecycle)
defines the validation and lifetime boundary, and [ADR 0028](../design/decisions/0028-explicit-native-parameter-copy.md)
records why the transition is explicit.

```text
NN initialization (possibly heap) -> Training explicit native copy -> Engine session open -> runs -> session close -> caller Arena close
```

Read this left to right: storage association changes only before open, while the parameter
identity used by the expression and compiled graph does not. For example, a `Linear` constructed
with heap-initialized weight and bias can build its loss expression, be copied, and then open a
CPU `TrainingSession` on that same expression. By contrast, opening without the copy still fails
at native-only parameter capture; attempting the copy while a session is open would invalidate
the session's fixed storage snapshot. Focused unit tests cover byte preservation and failure
before rebinding, and the CPU integration test covers the end-to-end path. This transition does
not authorize Training to select a backend or change Model or NN ownership.

The public initial lifecycle compiles and prepares the forward/backward graph once through Engine,
then applies the optimizer only after every objective and gradient publication is detached and
validated:

```text
one reusable prepared forward/backward execution
  -> target-indexed detached gradients
  -> stage assigned optimizer slots/accumulation candidates
  -> validate all stable parameter bindings again
  -> commit parameter bytes and optimizer/session state
```

Compiler target order and publication `targetIndex` are the mapping authorities; Training does not
infer a parameter association from Shape, storage, or publication order alone. At open, ordered
groups resolve one immutable algorithm per parameter; for example an Adam default, SGD weight
group, and AdamW bias group still use one graph and one global step. `Sgd` supports momentum,
dampening, coupled weight decay, and compatible Nesterov behavior. `Adam` and `AdamW` use
bias-corrected first and second moments; the former couples decay into gradients, while the latter
decays the master parameter separately. Their moments and low-precision master candidates are
staged before any parameter write. The session supports reset-and-step, accumulate-only,
accumulate-and-step, zeroing, immutable state snapshots,
strict restore, and fail-before-commit rejection of non-finite optimizer state or arithmetic.
Training owns none of `Parameter`, `Buffer`, layer behavior, train/eval mode, backend ownership,
or a fused update route.

## Continuing at the next training input

The [Training lifecycle contract](contracts/extensions-training.md#optimizertraining-lifecycle)
defines which owner must supply each part of a recovery point. Read the table from left to
right: capture is when the value is observed, while preflight names who can reject a bad restore
before changing its own state. A **detached** snapshot owns its values independently of live
Tensors and storage.

| Participant and owner | Capture point and detachedness | Restore or preflight authority | Unsupported boundary |
|---|---|---|---|
| Trainable parameter values and optimizer slots, Training | `TrainingSession.state()` after an admitted run; `TrainingState` owns canonical logical/master bytes, first/second slots, pending gradients, optimizer configuration, paths, and counters. | `TrainingSession.restore()` checks its complete fixed parameter view and payload before its own commit. | It cannot install a different parameter Tensor or storage binding. |
| Persistent buffers and bindings, NN | `Module.stateDictionary()` observes paths and exact Tensor references; it is shallow. A recovery owner needs a proven step-aligned detached value for every relevant buffer. `TrainingState` already detaches supported parameter values. | NN's `loadStateDictionary()` checks path, kind, type, Shape, and parameter eligibility before replacing bindings. | A shallow dictionary is not a detached value checkpoint; loading during an open Training session violates its fixed bindings. Training-mode `BatchNorm.forward` binds symbolic next-buffer expressions at graph construction; no current test or API proves a detached value corresponding to a completed session step, so this workflow is ineligible for exact continuation today. |
| Per-module forward mode, NN | Caller reads `mode()` on every relevant module when saving; no mode is in `StateDictionary`. | Caller restores the intended tree state before constructing the objective; recursive `train()`/`eval()` set a uniform tree or reconstruct mixed modes with parent-first child overrides under exclusive access. | A single root mode cannot reconstruct mixed local modes; local mode also does not prove which context a forward used. |
| Effective forward contexts, NN and caller | Caller records the immutable `ForwardContext` choice actually supplied to each mode-sensitive forward when constructing the fixed objective, with enough construction-site association to reproduce it. This is a separate caller-owned value, not a `StateDictionary` field. | Caller validates and rebuilds the same context choices and objective before `TrainingSession.open`. | `Dropout.forward` and `BatchNorm.forward` accept an explicit context that may differ from `Module.mode()`; if the effective choices cannot be reconstructed, refuse exact continuation. |
| Fixed objective and graph random state, Model/Engine and caller | Caller records graph construction and the key/counter supplied to `GraphRngState.initial`; the expression is storage-free and the session compiles it once. | Rebuild the same intended objective before `TrainingSession.open`; compile/prepare and Training capture check their respective domains. | No current per-run RNG input/output or automatic dropout stream advancement. |
| Next input and program progress, caller | At a quiescent successful-run boundary, record next batch identity/order, sampler cursor, epoch, and any external random generator state in the caller's own detached representation. | Caller validates and restores its own source before giving the next input to `run`. | Training counters do not identify a batch or restore a sampler; durable exact encoding requires a supported versioned schema, bounded codec, and owner validator/restorer. |
| Scheduler, scaler, or other program state, caller if present | Capture each concrete external participant at the same boundary. | Its owner validates and restores it. | No built-in scheduler/scaler contract or generic state registry exists; arbitrary caller objects cannot be encoded as exact state by Checkpoint. |

For the fixed-session route, no new Training public value is needed: the existing
`TrainingState` and a caller-owned, explicit progress/random handoff are sufficient for
the supported state-complete claim only when every affecting participant is demonstrably
capturable and restorable on the same admitted computation route. The current claim excludes
training-mode BatchNorm and any unreconstructible objective context. It is not a cross-device
bitwise replay promise. The [decision record](../design/decisions/0030-exact-training-resume-state-ownership.md)
explains the tradeoff and the future joint transition.

**Concrete save and restore scenario.** Suppose batch identifiers are `[A, B, C, D]`, the
caller has successfully run `A` and `B`, and the next batch is `C` at cursor `2` in epoch `0`.
The fixed objective was built with `GraphRngState.initial(17, 4)` and uses that same explicit
graph state at each session run. This example uses a tree with no persistent buffers or
training-mode BatchNorm. At this quiescent boundary the caller records cursor `2`, epoch `0`,
its external augmentation generator's next state `81`, graph construction inputs `(17, 4)`,
the effective `ForwardContext(TRAINING)` supplied to dropout when constructing the objective,
and every relevant module's local mode; it also obtains `TrainingState`. The explicit context
must be recorded even if dropout's local mode was `EVALUATION`. For recovery it builds a
compatible NN tree and parameter storage, restores local modes, supplies the recorded forward
context while reconstructing the same objective from the final NN bindings, then opens a new
session, restores `TrainingState`, restores caller progress and external generator state, and
supplies `C` as the next input. The key/counter is rebuilt as a graph expression;
`81` belongs to the separate caller-owned generator; its in-memory handoff is not a Checkpoint
codec. The current APIs do not make that multi-owner sequence atomic, so a recovery coordinator
must finish validation and installation
before publishing the recovered run as exact. The example does not claim a new dropout draw for
each batch: the fixed graph reuses `(17, 4)`.

**Refusal scenario.** If the augmentation generator is seeded again but its state after `B` is
unknown, or the saved batch cursor cannot identify `C`, restoring `TrainingState` still restores
optimizer state but cannot identify the next computation. Label this a partial restore or
restart. The same refusal applies if the forward context actually used for dropout cannot be
reconstructed, or if training-mode BatchNorm's symbolic running buffers have no proved
step-aligned detached publication. A future durable artifact also refuses an arbitrary caller
RNG/sampler/scheduler/scaler object without a supported versioned codec and owner validator.
Likewise, a program expecting dropout to consume graph state `(17, 4)` at `A` and
automatically use its returned next state at `B` cannot call the current fixed session an exact
advancing-stream continuation: `TrainingSession.run` exposes no per-run graph-RNG handoff.

Training 0004 is the gate for a future joint restore. Under exclusive caller coordination, it
must quiesce and close the old session, prepare detached NN values/modes and Training/caller state,
prevalidate every schema and value, including NN/Training parameter-byte agreement by stable
path, effective forward-context reconstruction, and fixed-objective compatibility before any
installation,
then install NN bindings and modes on an isolated unpublished replacement tree. Only after those
bindings are final may it construct the objective, open a session against that tree, and restore
Training state. It publishes the candidate and caller state together after success, or discards
the candidate on ordinary failure. An in-place alternative needs complete rollback. Today's
`loadStateDictionary` and `TrainingSession.restore` each validate only their own state; in
particular Training cannot prevalidate restore without opening a session. Training 0004 must
provide and test the missing joint preflight/commit mechanism before claiming atomic recovery.
A future Checkpoint adapter may only encode and recover participants with supported versioned,
bounded schemas/codecs and owner validators/restorers; it cannot create a missing sampler,
random-state handoff, or scheduler. The BatchNorm training path needs separate step-aligned
buffer publication evidence before inclusion.

## Future compiled optimizer graph

A future architecture version may generate optimizer update operations as part of a larger graph:

```text
forward + backward + optimizer update graph
  -> optimize and partition
  -> backend prepare
  -> training-step schedule
```

In that model, backend prepare may fuse or specialize optimizer updates while the training extension remains backend-agnostic. This direction requires the explicit architecture updates required by the contract; it is not permission to move backend implementation into training now.

## Why `MetalOptimizerBridge` does not belong in training

A `MetalOptimizerBridge` would make the training extension depend on a concrete backend and give training responsibility for backend execution. That reverses the intended dependency direction.

The correct ownership is:

```text
extensions/training  -> optimizer algorithm and mathematical intent
compiler/model       -> graph operations or backend-neutral step representation
backends/metal       -> MPSGraph or custom fused Metal implementation
runtime              -> execution of the prepared executable
```

The same rule excludes CPU- and CUDA-specific optimizer bridges from training. See [Module Boundaries](module-boundaries.md) and [Dependency Rules](dependency-rules.md).
