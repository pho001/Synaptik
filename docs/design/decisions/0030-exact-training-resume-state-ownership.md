# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0030: Own each participant in [exact training continuation](../../glossary.md#exact-training-continuation)

## Status

Accepted — 2026-10-13, after independent Class C approval of Training 0003A. This record
explains the [authoritative Training lifecycle
contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>.
Its participant [ownership](../../glossary.md#term-ownership-meanings) and exclusions remain accepted. The joint-restore ordering in its
original version was refined on 2026-10-14 by [ADR 0035](0035-private-candidate-training-restore.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>.

2026-10-16 [dependency](../../glossary.md#term-dependency) reconciliation: Training0007/ [ADR0039](0039-portable-graph-model-persistence.md#necessary-boundary-types-not-a-new-model-hierarchy)
supersede only this record's blanket absence of an [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)→Training edge. NN now exposes generic live
[parameter](../../glossary.md#term-parameter) [admission](../../glossary.md#term-admission) through `Module.trainingParameters()`; it gains no recovery coordinator,
[optimizer](../../glossary.md#term-optimizer) algorithm or state-installation authority. Participant ownership, private-candidate
recovery and exact-continuation exclusions below remain unchanged. Current dependency permission
comes from the [root](../../../ARCHITECTURE.md#dependency-rules), not this historical wording.

## Context

`TrainingSession.state()` produces [detached](../../glossary.md#detached-model-results) [optimizer](../../glossary.md#term-optimizer) and [parameter](../../glossary.md#term-parameter) bytes, slots, pending
[gradients](../../glossary.md#term-gradient), and counters. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns [buffers](../../glossary.md#term-buffer) and a local mode per [module](../../glossary.md#term-module), while its
`StateDictionary` retains [Tensor](../../glossary.md#tensor) references rather than detached values and omits mode. A
mode-sensitive forward may use a supplied [immutable](../../glossary.md#term-immutable) `ForwardContext` different from local mode.
Training-mode `BatchNorm.forward` replaces running buffers with symbolic next-value expressions
at [graph](../../glossary.md#graph) construction; existing tests do not establish step-aligned [materialization](../../glossary.md#materialization) after a [run](../../glossary.md#term-run).
A session [captures](../../glossary.md#term-capture-meanings) fixed parameter Tensor/storage [identities](../../glossary.md#term-identity-meanings) and one compiled [objective](../../glossary.md#term-objective). [Model](../../glossary.md#term-model)'s
`GraphRngState.initial` is a storage-free expression; the current session has no per-run [RNG](../../glossary.md#graph-rng-state--graphrngstate)
input/output [handoff](../../glossary.md#term-handoff). Input order, [epoch](../../glossary.md#term-epoch-meanings), sampler cursor, external randomness, and any scheduler
or scaler belong to the calling program. These [owners](../../glossary.md#term-owner) currently offer no joint [atomic](../../glossary.md#resource-transaction-and-rollback) restore.

## Decision drivers

- Say exactly what can continue from the next input on the same supported computation route.
- Preserve [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [Model](../../glossary.md#term-model), [Engine](../../glossary.md#term-engine), and Training [ownership](../../glossary.md#term-ownership-meanings) and the absence of an NN-to-Training edge.
- Reject an exact claim when a participant cannot be captured or restored.
- Reuse current Training state capture without a new progress API and give Training 0004 a
  testable joint gate.<span data-glossary-navigation="true"> ([capture glossary](../../glossary.md#term-capture-meanings); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Call `TrainingState` alone an exact [checkpoint](../../glossary.md#term-checkpoint) | Smallest [handoff](../../glossary.md#term-handoff) | Omits [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [buffers](../../glossary.md#term-buffer)/modes, next input, and external randomness. |
| Put [opaque](../../glossary.md#term-opaque) caller data and a generic state [registry](../../glossary.md#term-registry) inside `TrainingState` | One container | Training cannot validate sampler, external [RNG](../../glossary.md#graph-rng-state--graphrngstate), scheduler, or scaler semantics; it invents [ownership](../../glossary.md#term-ownership-meanings). |
| Keep participant states with their [owners](../../glossary.md#term-owner) and coordinate an explicit handoff | Each [owner](../../glossary.md#term-owner) validates its own domain | The caller must [capture](../../glossary.md#term-capture-meanings) progress and future joint installation needs a separate protocol. |

The third option is selected. The table compares ownership and [validation](../../glossary.md#term-validation), not [codec](../../glossary.md#term-codec) formats.

## Decision

An exact continuation means a complete recovery point at a successful-[run](../../glossary.md#term-run) boundary, a known next
input, and all state affecting that input restored for the same admitted computation route. The
[ownership table](../../architecture/training-graph.md#continuing-at-the-next-training-input)<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings))</span>
records [capture](../../glossary.md#term-capture-meanings), detachedness, [validation](../../glossary.md#term-validation) authority, and unsupported cases. `TrainingState`
alone is an exact restore of its documented session state, not an end-to-end recovery point.
No new public Training [API](../../glossary.md#term-application-programming-interface-api) is needed for that bounded claim; the caller keeps its own
typed next-input/progress, external [RNG](../../glossary.md#graph-rng-state--graphrngstate), and optional scheduler/scaler [handoff](../../glossary.md#term-handoff). No built-in
scheduler, scaler, sampler, or generic state container is selected. The current claim excludes
training-mode BatchNorm because its step-aligned running-[buffer](../../glossary.md#term-buffer) capture boundary is unproved.

An open session cannot accept `Module.loadStateDictionary`: loading replaces [bindings](../../glossary.md#term-binding) captured
by the session and leaves its compiled [objective](../../glossary.md#term-objective) referring to earlier [Tensors](../../glossary.md#tensor). Capture [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) values
separately and detach them for recovery; capture each relevant [module](../../glossary.md#term-module)'s local mode separately.
Recursive `train()`/`eval()` restore a uniform tree directly; a caller can reconstruct mixed
local modes with parent-first child overrides under exclusive access. Recording only the root
mode is insufficient for that case. The caller must also record and reproduce the effective
`ForwardContext` supplied at each mode-sensitive objective-construction site; `Module.mode()` is
not a proxy. If those contexts are unreconstructible, refuse exact continuation. [Graph](../../glossary.md#graph) RNG
key/counter can be recorded and the fixed expression rebuilt, but the current session does not
advance a [dropout](../../glossary.md#dropout--inverted-dropout) stream across runs. Changing per-step graph RNG requires a
separate per-run API decision. No cross-[device](../../glossary.md#term-device) or cross-route bitwise promise follows.

For future joint recovery, Training 0004 must coordinate the supported participants on an
isolated unpublished replacement and publish only after the candidate and caller-owned state
have passed validation under external exclusion. [ADR 0035](0035-private-candidate-training-restore.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
sets the revised validation order: check available [detached](../../glossary.md#detached-model-results) facts first, then allow
session-specific restore checks after opening a private candidate. It does not require a
general [preflight](../../glossary.md#term-preflight)/[commit](../../glossary.md#term-commit) API or [rollback](../../glossary.md#term-rollback) of the old session. A [Checkpoint](../../glossary.md#term-checkpoint) adapter may persist
only participant types with supported versioned, bounded [schemas](../../glossary.md#term-schema)/codecs and [owner](../../glossary.md#term-owner)
validators/restorers. It cannot encode arbitrary
caller-owned RNG, sampler, scheduler, or scaler objects as exact state; unsupported participants
require refusal of an exact [artifact](../../glossary.md#term-artifact) or an explicitly labeled restart/partial mode. The adapter
does not own NN state semantics, create a Data module, or make NN depend on Training.

## Rationale

The state affecting a future [run](../../glossary.md#term-run) spans several lifecycles. Keeping each value with its [owner](../../glossary.md#term-owner)
allows the owner to validate it and makes omissions visible. A single unvalidated Training blob
would hide missing progress and random state. Private-candidate [validation](../../glossary.md#term-validation) prevents a late
[schema](../../glossary.md#term-schema) failure from changing the published [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and Training recovery point.

## Consequences

Existing `TrainingState` restore and its [CPU](../../glossary.md#term-cpu) continuation tests remain valid for their bounded
session claim. The caller must record the next [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) and every external state source; `TrainingState`
counters cannot reconstruct them. Shallow `StateDictionary` exports require [detached value](../../glossary.md#detached-model-results) [capture](../../glossary.md#term-capture-meanings)
for recovery; the current training-mode BatchNorm path has no proved step-aligned value capture.
Mixed local modes and effective forward contexts require separate capture and reconstruction.
Advancing
[graph](../../glossary.md#graph)-random streams cannot be represented by the current supported session workflow. Training
0004 needs a coordinated private-candidate [handoff](../../glossary.md#term-handoff) and failure tests. [Checkpoint](../../glossary.md#term-checkpoint) work
must wait for that gate and supported caller-state [schemas](../../glossary.md#term-schema)/codecs before promising joint recovery.
This decision adds no Java [API](../../glossary.md#term-application-programming-interface-api), [codec](../../glossary.md#term-codec),
[backend route](../../glossary.md#backend-route), or [persistence](../../glossary.md#term-persistence) format.

## Related documentation

- [Root ownership route](../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings))</span>
- [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Training graph recovery scenario](../../architecture/training-graph.md#continuing-at-the-next-training-input)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [Training 0003A task packet](../../planning/extensions/training/tasks/0003a-exact-resume-state-ownership-decision.md)
