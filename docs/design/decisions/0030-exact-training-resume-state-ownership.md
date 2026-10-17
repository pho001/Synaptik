# ADR 0030: Own each participant in exact training continuation

## Status

Accepted — 2026-10-13, after independent Class C approval of Training 0003A. This record
explains the [authoritative Training lifecycle
contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle).
Its participant ownership and exclusions remain accepted. The joint-restore ordering in its
original version was refined on 2026-10-14 by [ADR 0035](0035-private-candidate-training-restore.md).

2026-10-16 dependency reconciliation: Training0007/ [ADR0039](0039-portable-graph-model-persistence.md#necessary-boundary-types-not-a-new-model-hierarchy)
supersede only this record's blanket absence of an NN→Training edge. NN now exposes generic live
parameter admission through `Module.trainingParameters()`; it gains no recovery coordinator,
optimizer algorithm or state-installation authority. Participant ownership, private-candidate
recovery and exact-continuation exclusions below remain unchanged. Current dependency permission
comes from the [root](../../../ARCHITECTURE.md#dependency-rules), not this historical wording.

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

For future joint recovery, Training 0004 must coordinate the supported participants on an
isolated unpublished replacement and publish only after the candidate and caller-owned state
have passed validation under external exclusion. [ADR 0035](0035-private-candidate-training-restore.md)
sets the revised validation order: check available detached facts first, then allow
session-specific restore checks after opening a private candidate. It does not require a
general preflight/commit API or rollback of the old session. A Checkpoint adapter may persist
only participant types with supported versioned, bounded schemas/codecs and owner
validators/restorers. It cannot encode arbitrary
caller-owned RNG, sampler, scheduler, or scaler objects as exact state; unsupported participants
require refusal of an exact artifact or an explicitly labeled restart/partial mode. The adapter
does not own NN state semantics, create a Data module, or make NN depend on Training.

## Rationale

The state affecting a future run spans several lifecycles. Keeping each value with its owner
allows the owner to validate it and makes omissions visible. A single unvalidated Training blob
would hide missing progress and random state. Private-candidate validation prevents a late
schema failure from changing the published NN and Training recovery point.

## Consequences

Existing `TrainingState` restore and its CPU continuation tests remain valid for their bounded
session claim. The caller must record the next batch and every external state source; `TrainingState`
counters cannot reconstruct them. Shallow `StateDictionary` exports require detached value capture
for recovery; the current training-mode BatchNorm path has no proved step-aligned value capture.
Mixed local modes and effective forward contexts require separate capture and reconstruction.
Advancing
graph-random streams cannot be represented by the current supported session workflow. Training
0004 needs a coordinated private-candidate handoff and failure tests. Checkpoint work
must wait for that gate and supported caller-state schemas/codecs before promising joint recovery.
This decision adds no Java API, codec,
backend route, or persistence format.

## Related documentation

- [Root ownership route](../../../ARCHITECTURE.md#module-ownership-routing)
- [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
- [Training graph recovery scenario](../../architecture/training-graph.md#continuing-at-the-next-training-input)
- [Training 0003A task packet](../../planning/extensions/training/tasks/0003a-exact-resume-state-ownership-decision.md)
