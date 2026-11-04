# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0036: [Model](../../glossary.md#term-model)-envelope composition through ModelCheckpoint

## Status

Accepted — 2026-10-14. The approved [API](../../glossary.md#term-application-programming-interface-api) [scope](../../glossary.md#term-scope) is implemented in the [Checkpoint](../../glossary.md#term-checkpoint) 0005A change;
coordinator acceptance remains a separate [validation](../../glossary.md#term-validation) gate, not a new architecture decision.
The optional Training Checkpoint adapter 0006 remains Draft and has no build authorization.

## Context

At decision time, the ModelCheckpoint file [API](../../glossary.md#term-application-programming-interface-api) validated and installed [model](../../glossary.md#term-model) state, but inspection
exposed [descriptors](../../glossary.md#term-descriptor) without values. The selected future Model + TrainingState product needs complete
model [parameter](../../glossary.md#term-parameter) descriptors and raw logical bytes to compare against [detached](../../glossary.md#detached-model-results) TrainingState.
Otherwise composition would require private implementation access or an unnecessary file [round
trip](../../glossary.md#term-round-trip). This record explains the [owning contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint);
it does not override the [architecture root](../../../ARCHITECTURE.md).

## Decision drivers

- Keep [Checkpoint](../../glossary.md#term-checkpoint) [model](../../glossary.md#term-model)-only and preserve [owner](../../glossary.md#term-owner) [validation](../../glossary.md#term-validation) and the private-candidate protocol.
- Reuse existing bounded envelope and model semantics without a new [wire format](../../glossary.md#term-wire-format) or public wrapper.
- Preserve exact stored bits, [detached](../../glossary.md#detached-model-results) lifetimes, file behavior, and all caller resource ceilings.
- Expose only the concrete model boundary needed by the future optional consumer.

## Options considered

1. Keep only file APIs. This preserves the status quo but forces composition through files and
   still leaves metadata-only inspection insufficient for the parameter-byte comparison.<span data-glossary-navigation="true"> ([metadata glossary](../../glossary.md#term-metadata); [parameter glossary](../../glossary.md#term-parameter))</span>
2. Publish a model-image wrapper or expose the internal candidate/codec. This adds public
   representation and validation obligations when the existing envelope already carries the bytes.<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model); [codec glossary](../../glossary.md#term-codec); [validation glossary](../../glossary.md#term-validation))</span>
3. Extend ModelCheckpoint with operations over ArtifactEnvelope. This preserves the semantic
   owner and existing public value types while permitting bounded, file-free composition.<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>

## Decision

Choose option 3. Expose [`capture`](../../glossary.md#term-capture-meanings), envelope `inspect`, `canonicalStateBytes`, and envelope `load`
on the existing ModelCheckpoint. Reuse internal capture, decode/[preflight](../../glossary.md#term-preflight), and restoration;
keep their production types package-private. The new operations declare checked
ArtifactFormatException for semantic decode rejection; capture retains existing unchecked
preflight/[materialization](../../glossary.md#materialization) failures. File methods retain IOException [signatures](../../glossary.md#term-signature) and existing
failure types/order, reread files on inspect/load, and delegate the same semantic machinery.

[Canonical](../../glossary.md#term-canonical-form)-byte access returns an [immutable](../../glossary.md#term-immutable) map structure in [model](../../glossary.md#term-model) [metadata](../../glossary.md#term-metadata) order, keyed by exact
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) paths. Every call supplies independent read-only position-zero BIG_ENDIAN ByteBuffer [views](../../glossary.md#term-view-meanings);
their bytes are immutable, not their cursors. Each [operation](../../glossary.md#operation) fully validates the envelope and
every ArtifactLimits [dimension](../../glossary.md#dimension) without depending on prior file reads or inspection. No new
[codec](../../glossary.md#term-codec), [schema](../../glossary.md#term-schema), [registry](../../glossary.md#term-registry), temporary-file composition pipeline, or public model-image type is chosen.

The concrete future consumer compares the complete model [parameter](../../glossary.md#term-parameter) set, [descriptors](../../glossary.md#term-descriptor), and raw
logical bytes with TrainingState before [owner](../../glossary.md#term-owner) restore and private-candidate [publication](../../glossary.md#term-publication).
The first durable product is Model + TrainingState only: saved-state restoration, not automatic
exact continuation. Modes, effective contexts, cursor, external [RNG](../../glossary.md#graph-rng-state--graphrngstate), scheduler, and scaler remain
application-owned. No Training validator, shared [transaction](../../glossary.md#term-transaction), reverse [dependency](../../glossary.md#term-dependency), or [backend](../../glossary.md#backend)
access is introduced. Adapter wire design and build inclusion require their separate 0006 gate.

## Rationale

The existing decoder already checks file-equivalent, [manifest](../../glossary.md#term-manifest), name, count, [payload](../../glossary.md#artifact-envelope-and-atomic-publication), and aggregate
bounds before [schema](../../glossary.md#term-schema) copying, then validates [model](../../glossary.md#term-model) semantics. Reusing it avoids two acceptance
policies. A [detached](../../glossary.md#detached-model-results) envelope and typed inspection provide sufficient composition facts without
making model-only users acquire Training or allowing inspection to act as a trusted credential.

## Consequences

- [Model](../../glossary.md#term-model) composition can avoid file I/O; current model files and v1 encoding remain compatible.
- The new supported facade increases documentation and review obligations. Overloaded methods
  require callers with an untyped null literal to disambiguate Path versus ArtifactEnvelope.
- Per-call validation costs remain; no cache or trusted-validation token is selected. Canonical
  aggregate limits still do not constitute a workspace, defensive-copy, or peak-heap budget.<span data-glossary-navigation="true"> ([validation glossary](../../glossary.md#term-validation); [cache glossary](../../glossary.md#term-cache); [token glossary](../../glossary.md#term-token); [Canonical glossary](../../glossary.md#term-canonical-form); [heap glossary](../../glossary.md#term-heap))</span>
- Load retains target-defined aliases and rejection of unbound targets. Existing dependency
  assertions remain correct because no project edge or optional-project inclusion changes.<span data-glossary-navigation="true"> ([target glossary](../../glossary.md#term-target); [aliases glossary](../../glossary.md#aliasing-and-arena); [dependency glossary](../../glossary.md#term-dependency))</span>
- 0005A requires clean implementation and independent targeted review, raw-bit/lifetime/bounds
  and file-parity tests, then coordinator acceptance. This decision does not claim those passed.<span data-glossary-navigation="true"> ([lifetime glossary](../../glossary.md#term-lifetime))</span>
- No earlier [ADR](../../glossary.md#term-architecture-decision-record-adr) is superseded; this adds a composition boundary to the existing model [facade](../../glossary.md#term-facade).

## Related documentation

- [Model-only boundary](0031-model-checkpoint-extension-boundary.md)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model))</span>
- [Current file facade](0034-public-model-checkpoint.md)<span data-glossary-navigation="true"> ([facade glossary](../../glossary.md#term-facade))</span>
- [Private-candidate protocol](0035-private-candidate-training-restore.md)
- [Module-boundary explanation](../../architecture/module-boundaries.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>
- [File-free model API example](../../api/model-checkpoint-api.md#file-free-example)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [0005A execution brief](../../planning/extensions/checkpoint/tasks/0005a-model-envelope-composition.md)
- [Existing dependency assertions](../../../testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java)<span data-glossary-navigation="true"> ([dependency glossary](../../glossary.md#term-dependency))</span>
