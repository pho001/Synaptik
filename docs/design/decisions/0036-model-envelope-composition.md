# ADR 0036: Model-envelope composition through ModelCheckpoint

## Status

Accepted — 2026-10-14. Boundary approved; API implementation is planned in Checkpoint 0005A.
The optional Training Checkpoint adapter 0006 remains Draft and has no build authorization.

## Context

The current ModelCheckpoint file API validates and installs model state, but inspection exposes
descriptors without values. The selected future Model + TrainingState product needs complete
model parameter descriptors and raw logical bytes to compare against detached TrainingState.
Otherwise composition would require private implementation access or an unnecessary file round
trip. This record explains the [owning contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint);
it does not override the [architecture root](../../../ARCHITECTURE.md).

## Decision drivers

- Keep Checkpoint model-only and preserve owner validation and the private-candidate protocol.
- Reuse existing bounded envelope and model semantics without a new wire format or public wrapper.
- Preserve exact stored bits, detached lifetimes, file behavior, and all caller resource ceilings.
- Expose only the concrete model boundary needed by the future optional consumer.

## Options considered

1. Keep only file APIs. This preserves the status quo but forces composition through files and
   still leaves metadata-only inspection insufficient for the parameter-byte comparison.
2. Publish a model-image wrapper or expose the internal candidate/codec. This adds public
   representation and validation obligations when the existing envelope already carries the bytes.
3. Extend ModelCheckpoint with operations over ArtifactEnvelope. This preserves the semantic
   owner and existing public value types while permitting bounded, file-free composition.

## Decision

Choose option 3. Plan `capture`, envelope `inspect`, `canonicalStateBytes`, and envelope `load`
on the existing ModelCheckpoint. Reuse internal capture, decode/preflight, and restoration;
keep their production types package-private. The new operations declare checked
ArtifactFormatException for semantic decode rejection; capture retains existing unchecked
preflight/materialization failures. File methods retain IOException signatures and existing
failure types/order, reread files on inspect/load, and delegate the same semantic machinery.

Canonical-byte access returns an immutable map structure in model metadata order, keyed by exact
NN paths. Every call supplies independent read-only position-zero BIG_ENDIAN ByteBuffer views;
their bytes are immutable, not their cursors. Each operation fully validates the envelope and
every ArtifactLimits dimension without depending on prior file reads or inspection. No new
codec, schema, registry, temporary-file composition pipeline, or public model-image type is chosen.

The concrete future consumer compares the complete model parameter set, descriptors, and raw
logical bytes with TrainingState before owner restore and private-candidate publication.
The first durable product is Model + TrainingState only: saved-state restoration, not automatic
exact continuation. Modes, effective contexts, cursor, external RNG, scheduler, and scaler remain
application-owned. No Training validator, shared transaction, reverse dependency, or backend
access is introduced. Adapter wire design and build inclusion require their separate 0006 gate.

## Rationale

The existing decoder already checks file-equivalent, manifest, name, count, payload, and aggregate
bounds before schema copying, then validates model semantics. Reusing it avoids two acceptance
policies. A detached envelope and typed inspection provide sufficient composition facts without
making model-only users acquire Training or allowing inspection to act as a trusted credential.

## Consequences

- Model composition can avoid file I/O; current model files and v1 encoding remain compatible.
- The new supported facade increases documentation and review obligations. Overloaded methods
  require callers with an untyped null literal to disambiguate Path versus ArtifactEnvelope.
- Per-call validation costs remain; no cache or trusted-validation token is selected. Canonical
  aggregate limits still do not constitute a workspace, defensive-copy, or peak-heap budget.
- Load retains target-defined aliases and rejection of unbound targets. Existing dependency
  assertions remain correct because no project edge or optional-project inclusion changes.
- 0005A requires clean implementation and independent targeted review, raw-bit/lifetime/bounds
  and file-parity tests, then coordinator acceptance. This decision does not claim those passed.
- No earlier ADR is superseded; this adds a composition boundary to the existing model facade.

## Related documentation

- [Model-only boundary](0031-model-checkpoint-extension-boundary.md)
- [Current file facade](0034-public-model-checkpoint.md)
- [Private-candidate protocol](0035-private-candidate-training-restore.md)
- [Module-boundary explanation](../../architecture/module-boundaries.md#extensionscheckpoint)
- [0005A execution brief](../../planning/extensions/checkpoint/tasks/0005a-model-envelope-composition.md)
- [Existing dependency assertions](../../../testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java)
