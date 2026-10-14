# ADR 0035: Validate and publish a private training restore candidate

## Status

Accepted — 2026-10-14. This refines the future joint-restore sequence in
[ADR 0030](0030-exact-training-resume-state-ownership.md); its participant ownership and
exact-continuation exclusions remain in force. The [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
is authoritative. This decision does not claim that Training 0004 is implemented.

## Context

An exact continuation spans NN bindings and modes, Training parameters and optimizer state,
effective objective-construction contexts, and caller-owned next-input and external state.
`Module.loadStateDictionary` replaces Tensor bindings, which an open session has already
captured. `TrainingSession.restore` checks its fixed parameter view and payload only after a
session exists. `ModelCheckpoint.inspect` exposes descriptors, not model value bytes;
`ModelCheckpoint.load` rereads and validates a file, then installs into the supplied NN tree.
Training's logical parameter bytes are canonical big-endian, whereas loaded NN storage uses
native-order carriers. A new private session's `state()` can supply comparable canonical logical
bytes, at the cost of copying its complete parameter and optimizer snapshot.

ADR 0030's earlier joint gate required complete NN/Training prevalidation before *any* live
installation and a missing preflight/commit mechanism. That conflated mutation of a published
tree with mutation of an isolated replacement that nobody can observe. It could force an
otherwise unnecessary pure pre-open Training validator or cross-module value reader.

## Decision drivers

- Leave the published model and session untouched while ordinary candidate validation can fail.
- Detect missing participants and NN/Training parameter disagreement before publication.
- Reuse owner-specific restore validation and current public APIs where they suffice.
- Keep caller exclusion and external-state handoff explicit without a transaction framework.
- Preserve the exact-continuation exclusions and defer persistence to the later adapter.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Prevalidate all Training restore facts before opening a session | Every failure is found before any installation | The fixed session view does not yet exist; this requires a new validator and may require a new canonical NN value API. |
| Install into the published tree and roll back on failure | Reuses one model | NN binding replacement invalidates an open session; complete rollback across owners and external state is not available. |
| Prepare a distinct unpublished model and session, then publish under caller exclusion | Existing owner validation can run against the final candidate | Candidate preparation and a possible `state()` snapshot cost resources; callers must stage their external state and control the handoff. |

The third option is selected. The table compares failure isolation, not durable file formats.

## Decision

At a quiescent successful-run boundary, the caller establishes a complete, supported recovery
point: detached NN values and per-module modes, Training state, effective forward contexts,
fixed-objective construction inputs, and typed next-input, random, and other external state.
Every fact available from that detached evidence is checked before mutating a published object;
missing or unsupported participants prevent an exact claim. Comparable detached NN and Training
logical parameter bytes may be checked at this stage, but doing so before session open is not a
mandatory API contract.

The caller constructs a *distinct unpublished* NN tree. It loads NN bindings and modes there,
uses the existing native parameter copy when needed, builds the objective from the final
Tensors, and opens a private `TrainingSession`. It checks exact candidate NN-versus-Training
logical parameter bytes at every stable path before restore and publication. Where the source
already supplies trustworthy comparable detached bytes, that comparison can precede open;
otherwise the private session's `state()` supplies canonical candidate logical bytes. Raw
native-order NN storage is not compared directly with canonical Training bytes, and a prior
`ModelCheckpoint.inspect` is not trusted for a later `load`. The caller then invokes
`TrainingSession.restore`, which validates session-specific configuration, schema, counters,
payloads, and fixed bindings before its own installation.

Only a fully validated replacement and staged typed external state may be published under
caller-owned exclusion. Ordinary failure closes and discards the private candidate without
publishing it. The old session may remain quiescent during candidate preparation and is retired
by the caller at handoff; there is no promise to roll back an old session the caller already
closed. Readers outside the caller's exclusion have no simultaneous-visibility guarantee.
External state that cannot be staged for this handoff prevents an exact-continuation claim.
No general transaction API or mandatory pure pre-open Training validator is introduced.
The later Training Checkpoint adapter may persist only supported versioned, bounded participant
schemas with owner validation; it does not move state semantics into Checkpoint.

## Rationale

The publication boundary, not the private candidate's first mutation, is what protects the
existing live recovery point. Owner checks still run on the actual tree and session that will
be used. A canonical post-open comparison avoids inventing a public NN value reader merely to
bridge byte order. `TrainingSession.state()` copies parameter and optimizer payloads, so an
existing detached comparison can avoid that extra allocation and copy where it proves the same
candidate value; neither route relaxes the equality requirement.

## Consequences

This changes the architecture gate for future Training 0004, not current Java behavior or API.
Validation must cover missing and incompatible detached participants, NN/Training byte mismatch,
private open/restore failure, caller-state staging failure, and a successful next-input handoff;
failure must leave no candidate published. It does not make the old session rollback-capable or
support training-mode BatchNorm, an advancing graph-random stream, arbitrary external objects,
or cross-route bitwise replay. A later end-to-end Checkpoint integration test will need the
Checkpoint test dependency that the current integration-test module does not have; this ADR adds
no build edge. Persistence and its codec remain a separate adapter gate.

## Related documentation

- [Architecture root ownership route](../../../ARCHITECTURE.md#module-ownership-routing)
- [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
- [Training graph recovery scenario](../../architecture/training-graph.md#continuing-at-the-next-training-input)
- [Training restore implementation](../../../extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)
- [Model checkpoint load implementation](../../../extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/model/ModelCheckpoint.java)
