# ADR 0037: Stored Model + TrainingState in an optional downstream adapter

## Status

Withdrawn Proposed direction — 2026-10-14; superseded as the primary product by
[Proposed ADR 0038](0038-declarative-model-and-training-resume.md).
Never accepted or implemented. The user revised direction to automatic usable-model load with
inference or Training-owned resume; values + TrainingState alone cannot deliver that workflow.
The original proposal below is retained as historical design evidence. Its linked 0006 brief is
now revised, not an unchanged wire specification. Raw optimizer/group/slot/counter encoding,
complete logical-byte comparison and bounded public-owner composition remain useful foundations.
The unpublished training-checkpoint.v1 wire is not claimed compatible with 0038's new kind/content.
Published model.v1/envelope APIs and current authoritative contracts remain unchanged.

## Context

Applications need one durable file for the model and detached optimizer/session state. ModelCheckpoint
already supplies supported envelope capture, inspection, canonical exact-path bytes and strict load;
TrainingState already supports defensive reconstruction. Neither owner provides a cross-owner
transaction. The selected product is stored-state restoration, not automatic exact continuation.
The optional adapter is currently reserved but absent and forbidden from build inclusion until its gate.

## Decision drivers

- Keep model-only consumers independent of Training and retain each owner's validation authority.
- Preserve current model/envelope formats and raw bytes without private model metadata parsing.
- Reject incoherent parameter snapshots and bound encoded allocations before copies.
- Keep application publication/exclusion explicit; avoid a general persistence framework.

## Options considered

1. Keep separate model file and application-specific Training serialization. This leaves applications
   implementing bounded codecs and coherence checks, and offers no single-file publication.
2. Put Training serialization in Checkpoint or file I/O in Training. This couples optional consumers
   or reverses current ownership; reject it.
3. Add generic participant/transaction/serializer APIs. This widens the settled product and would
   require unsupported external-state schemas; reject it.
4. Add an optional downstream Model + TrainingState adapter over existing public APIs. Propose this
   bounded option; no pure Training validator, shared transaction or core Artifact API extension.

## Proposed decision

Create optional `extensions/training-checkpoint`, package
`io.github.pho001.synaptik.training.checkpoint`, with exactly public Training and Checkpoint project
dependencies. Existing NN/Model/Engine public types are transitive; there are no reverse/lower/backend
edges. The original 0037 facade proposed TrainingCheckpoint.save/read and a privately constructed
immutable nested Snapshot with modelInspection/trainingState/loadModel/restoreTraining.
The [revised brief](../../planning/extensions/checkpoint/tasks/0006-training-checkpoint-adapter.md#proposed-types-and-high-level-api)
supersedes that primary facade; the original proposal added no other public production type.

Use outer kind `training-checkpoint.v1` over unchanged ArtifactFile framing/publication. Copy existing
model entry bytes into fixed `model/` names and reconstruct `model.v1` for public ModelCheckpoint
validation/load. Do not inspect private ordinal metadata. A separate `training/schema` and fixed
ordinal slot names encode explicit version/IDs, raw binary64 optimizer coefficients, ordered groups,
ordered path/type/static-shape descriptors, lengths, second-slot presence and counters. Raw logical,
master, momentum/moment and pending payload bytes are separate entries. The
[original wire proposal, now revised](../../planning/extensions/checkpoint/tasks/0006-training-checkpoint-adapter.md#proposed-canonical-wire-and-retained-foundation)
was not implemented; neither model.v1 nor envelope v1 changes. Existing framing SHA-256
protects all encoded bytes; preserve existing model fingerprints without another certificate layer.

Complete model parameter set/type/shape/logical-byte equality is required before save and before read
returns Snapshot. Buffers stay model-owned, support all seven model types, and are not Training entries.
Compare exact paths without Unicode normalization or resorting Training order. Decode structural
Training values through public constructors: zero-element shapes and numerically invalid slots may
be stored. Algorithm-specific slot validity, group membership/cross-group overlap, finite values,
coefficient narrowing and session compatibility remain Training.restore authority. Read is not proof
that any session can restore the stored state.

Preflight checked outer and reconstructed-model framing sizes and all existing ArtifactLimits before
adapter payload copies. Nested metadata counts/ranks/paths are bounded from remaining bytes before
allocation. Strict UTF-8 includes rejecting malformed surrogate input Strings. Public Entry and
ParameterState constructors clone; no internal adopt shortcut or peak-heap promise is available.
Format/limit/cross-component failures use ArtifactFormatException; I/O and owner/lifecycle failures
retain existing API taxonomy. Reuse ArtifactFile for actual checksums and atomic file publication.

## Rationale and recovery scenario

An application captures session.state() and ModelCheckpoint.capture under one application exclusion,
including model buffers, optimizer slots and counters; equality alone cannot prove quiescence. It
then saves detached values. It reads a file, strict-loads a distinct private tree before opening a
session, supplies supported native parameter storage and application contexts/objective, and opens
the private session. Snapshot.restoreTraining calls privateSession.state(), compares complete logical
parameter facts, then delegates restore(saved). These are separately admitted operations; the caller
excludes other candidate/session users across both, not through a new atomic admission or lock.
The caller must loadModel onto that SAME tree before session open and retain its loaded buffers
through restore/publication: restoreTraining checks only parameters, not tree identity or buffers.
Read and Snapshot accessors do not prove whole-model installation. No token/registry API is added.
The caller publishes only the fully validated replacement, discarding it on failure. A model load
while that tree's session is open remains forbidden. This preserves published state on candidate
failure without inventing rollback for an already retired session.

## Consequences

- One file has atomic live visibility under the existing temporary-file/force/atomic-move policy,
  not universal power-loss durability, authentication, or a non-atomic fallback.
- Detached values and public defensive copies may increase heap demand. Canonical encoded-byte
  limits do not constitute a memory budget or large-model support evidence.
- Logical low-precision values remain distinct from F32 masters; all raw signed zeros/slot bits
  survive encoding. Session validity is deliberately a later owner decision, not silently codec policy.
- Modes, effective ForwardContexts, objective construction, input position/cursor/epoch, RNG,
  scheduler and scaler remain application-owned. No automatic exact continuation, changing graph
  RNG stream, training-mode BatchNorm alignment, or cross-device bitwise identity is promised.
- Approval and independently Class C-reviewed authoritative root/owning-contract update precede Ready. Clean
  implementation delivers project/build/tests and required explanatory docs in the same overall
  change, followed by independent review and coordinator acceptance; no coordinator code work or
  new prerequisite task. Until approval, 0006 stays Draft, the project is absent, and its current
  absent-adapter test remains correct.
- Implementation requires clean execution and independent targeted Class C review/documentation,
  adapter/module raw-bit and bounds tests, durable private-candidate CPU next-input proof, bounded
  source-matched device smoke, and once-only coordinator repository gates. No broad performance
  campaign is proposed.

## Related documentation

- [Root dependency gate](../../../ARCHITECTURE.md#dependency-rules)
- [Owning Checkpoint contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
- [Optimizer/training lifecycle](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
- [Model-envelope boundary ADR 0036](0036-model-envelope-composition.md)
- [Private-candidate ADR 0035](0035-private-candidate-training-restore.md)
- [Current module-boundary explanation](../../architecture/module-boundaries.md#extensionscheckpoint)
- [Exact architecture-test cutover target](../../../testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java)
