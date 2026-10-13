# ADR 0030: Own each participant in exact training continuation

## Status

Accepted — 2026-10-13, after independent Class C approval of Training 0003A. This record
explains the [authoritative Training lifecycle
contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle).

## Context

`TrainingSession.state()` produces detached optimizer and parameter bytes, slots, pending
gradients, and counters. NN owns buffers and a local mode per module, while its
`StateDictionary` retains Tensor references rather than detached values and omits mode. A
mode-sensitive forward may use a supplied immutable `ForwardContext` different from local mode.
Training-mode `BatchNorm.forward` replaces running buffers with symbolic next-value expressions
at graph construction; existing tests do not establish step-aligned materialization after a run.
A session captures fixed parameter Tensor/storage identities and one compiled objective. Model's
`GraphRngState.initial` is a storage-free expression; the current session has no per-run RNG
input/output handoff. Input order, epoch, sampler cursor, external randomness, and any scheduler
or scaler belong to the calling program. These owners currently offer no joint atomic restore.

## Decision drivers

- Say exactly what can continue from the next input on the same supported computation route.
- Preserve NN, Model, Engine, and Training ownership and the absence of an NN-to-Training edge.
- Reject an exact claim when a participant cannot be captured or restored.
- Reuse current Training state capture without a new progress API and give Training 0004 a
  testable joint gate.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Call `TrainingState` alone an exact checkpoint | Smallest handoff | Omits NN buffers/modes, next input, and external randomness. |
| Put opaque caller data and a generic state registry inside `TrainingState` | One container | Training cannot validate sampler, external RNG, scheduler, or scaler semantics; it invents ownership. |
| Keep participant states with their owners and coordinate an explicit handoff | Each owner validates its own domain | The caller must capture progress and future joint installation needs a separate protocol. |

The third option is selected. The table compares ownership and validation, not codec formats.

## Decision

An exact continuation means a complete recovery point at a successful-run boundary, a known next
input, and all state affecting that input restored for the same admitted computation route. The
[ownership table](../../architecture/training-graph.md#continuing-at-the-next-training-input)
records capture, detachedness, validation authority, and unsupported cases. `TrainingState`
alone is an exact restore of its documented session state, not an end-to-end recovery point.
No new public Training API is needed for that bounded claim; the caller keeps its own
typed next-input/progress, external RNG, and optional scheduler/scaler handoff. No built-in
scheduler, scaler, sampler, or generic state container is selected. The current claim excludes
training-mode BatchNorm because its step-aligned running-buffer capture boundary is unproved.

An open session cannot accept `Module.loadStateDictionary`: loading replaces bindings captured
by the session and leaves its compiled objective referring to earlier Tensors. Capture NN values
separately and detach them for recovery; capture each relevant module's local mode separately.
Recursive `train()`/`eval()` restore a uniform tree directly; a caller can reconstruct mixed
local modes with parent-first child overrides under exclusive access. Recording only the root
mode is insufficient for that case. The caller must also record and reproduce the effective
`ForwardContext` supplied at each mode-sensitive objective-construction site; `Module.mode()` is
not a proxy. If those contexts are unreconstructible, refuse exact continuation. Graph RNG
key/counter can be recorded and the fixed expression rebuilt, but the current session does not
advance a dropout stream across runs. Changing per-step graph RNG requires a
separate per-run API decision. No cross-device or cross-route bitwise promise follows.

For future joint recovery, Training 0004 must establish this ordered gate under exclusive caller
coordination:

1. Quiesce and close the old session before any NN binding change.
2. Stage proven step-aligned detached NN values, per-module modes, effective forward contexts,
   Training state, fixed-objective construction inputs, and caller
   progress/random/scheduler/scaler state; reject missing participants.
3. Prevalidate the complete NN schema and values, Training configuration/path/schema/payloads,
   matching NN parameter and Training logical bytes at each stable path,
   objective-construction inputs, effective contexts, route compatibility, and each caller-owned
   schema before installing any live participant.
4. On an isolated unpublished replacement tree, install NN bindings and modes, then construct the
   objective from those final Tensors, open the replacement session, and restore Training state.
   Stage caller-owned next-input and external state; publish the complete candidate only after
   success under external exclusion. Discard the candidate on ordinary failure. An in-place
   alternative must roll back every owner before it can claim joint atomicity.

Current per-owner APIs do not jointly prevalidate Training restore before session open or supply
the publication transition in step 4. Training 0004 must design and test the missing
preflight/commit mechanism before Checkpoint 0005–0006 can claim an
atomic end-to-end restore. A Checkpoint adapter may persist only participant types with supported
versioned, bounded schemas/codecs and owner validators/restorers. It cannot encode arbitrary
caller-owned RNG, sampler, scheduler, or scaler objects as exact state; unsupported participants
require refusal of an exact artifact or an explicitly labeled restart/partial mode. The adapter
does not own NN state semantics, create a Data module, or make NN depend on Training.

## Rationale

The state affecting a future run spans several lifecycles. Keeping each value with its owner
allows the owner to validate it and makes omissions visible. A single unvalidated Training blob
would hide missing progress and random state. Prevalidation before installation prevents a known
late schema failure from leaving NN and Training at different recovery points.

## Consequences

Existing `TrainingState` restore and its CPU continuation tests remain valid for their bounded
session claim. The caller must record the next batch and every external state source; `TrainingState`
counters cannot reconstruct them. Shallow `StateDictionary` exports require detached value capture
for recovery; the current training-mode BatchNorm path has no proved step-aligned value capture.
Mixed local modes and effective forward contexts require separate capture and reconstruction.
Advancing
graph-random streams cannot be represented by the current supported session workflow. Training
0004 needs a new coordinated validation/installation design and failure tests. Checkpoint work
must wait for that gate and supported caller-state schemas/codecs before promising joint recovery.
This decision adds no Java API, codec,
backend route, or persistence format.

## Related documentation

- [Root ownership route](../../../ARCHITECTURE.md#module-ownership-routing)
- [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
- [Training graph recovery scenario](../../architecture/training-graph.md#continuing-at-the-next-training-input)
- [Training 0003A task packet](../../planning/extensions/training/tasks/0003a-exact-resume-state-ownership-decision.md)
