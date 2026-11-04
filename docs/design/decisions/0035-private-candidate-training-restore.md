# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0035: Validate and publish a private training restore candidate


## Editorial reading context

The recovery coordinator is the application. It builds a distinct unpublished model/session, loads and validates state there, then publishes the replacement under its own exclusion. Mutating that private candidate does not mutate the currently visible model. Failed candidates are discarded; this is not a promise to resurrect a session the application already closed.

Use the [private-candidate recovery scenario](../../architecture/training-graph.md#continuing-at-the-next-training-input) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-10-14. This refines the future joint-restore sequence in
[ADR 0030](0030-exact-training-resume-state-ownership.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>; its participant [ownership](../../glossary.md#term-ownership-meanings) and
exact-continuation exclusions remain in force. The [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
is authoritative. This decision does not claim that Training 0004 is implemented.

Training0007 reconciliation (2026-10-16): the existing private-candidate [fixture](../../glossary.md#term-fixture) now passes
`privateTree.trainingParameters()` to generic Training copy/open APIs. Its execution, native
storage [admission](../../glossary.md#term-admission), next-input [handoff](../../glossary.md#term-handoff) and [publication](../../glossary.md#term-publication) guarantees are unchanged. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s narrow live
adapter [dependency](../../glossary.md#term-dependency) grants no restore coordination or [optimizer](../../glossary.md#term-optimizer) ownership. Training's physical
source is now `modules/training`; the original decision and recovery exclusions below are retained.

## Context

An exact continuation spans [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [bindings](../../glossary.md#term-binding) and modes, Training [parameters](../../glossary.md#term-parameter) and [optimizer state](../../glossary.md#term-optimizer-state),
effective [objective](../../glossary.md#term-objective)-construction contexts, and caller-owned next-input and external state.
`Module.loadStateDictionary` replaces [Tensor](../../glossary.md#tensor) bindings, which an open session has already
captured. `TrainingSession.restore` checks its fixed parameter [view](../../glossary.md#term-view-meanings) and [payload](../../glossary.md#artifact-envelope-and-atomic-publication) only after a
session exists. `ModelCheckpoint.inspect` exposes [descriptors](../../glossary.md#term-descriptor), not [model](../../glossary.md#term-model) value bytes;
`ModelCheckpoint.load` rereads and validates a file, then installs into the supplied NN tree.
Training's logical parameter bytes are [canonical](../../glossary.md#term-canonical-form) [big-endian](../../glossary.md#text-encoding-and-file-boundaries), whereas loaded NN storage uses
native-order carriers. A new private session's `state()` can supply comparable canonical logical
bytes, at the cost of copying its complete parameter and [optimizer](../../glossary.md#term-optimizer) [snapshot](../../glossary.md#term-snapshot).

[ADR](../../glossary.md#term-architecture-decision-record-adr) 0030's earlier joint gate required complete NN/Training prevalidation before *any* live
installation and a missing [preflight](../../glossary.md#term-preflight)/[commit](../../glossary.md#term-commit) mechanism. That conflated mutation of a published
tree with mutation of an isolated replacement that nobody can observe. It could force an
otherwise unnecessary pure pre-open Training validator or cross-[module](../../glossary.md#term-module) value reader.

## Decision drivers

- Leave the published [model](../../glossary.md#term-model) and session untouched while ordinary candidate [validation](../../glossary.md#term-validation) can fail.
- Detect missing participants and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training [parameter](../../glossary.md#term-parameter) disagreement before [publication](../../glossary.md#term-publication).
- Reuse [owner](../../glossary.md#term-owner)-specific restore validation and current public APIs where they suffice.
- Keep caller exclusion and external-state [handoff](../../glossary.md#term-handoff) explicit without a [transaction](../../glossary.md#term-transaction) framework.
- Preserve the exact-continuation exclusions and defer [persistence](../../glossary.md#term-persistence) to the later adapter.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Prevalidate all Training restore facts before opening a session | Every failure is found before any installation | The fixed session [view](../../glossary.md#term-view-meanings) does not yet exist; this requires a new validator and may require a new [canonical](../../glossary.md#term-canonical-form) [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) value [API](../../glossary.md#term-application-programming-interface-api). |
| Install into the published tree and roll back on failure | Reuses one [model](../../glossary.md#term-model) | NN [binding](../../glossary.md#term-binding) replacement invalidates an open session; complete [rollback](../../glossary.md#term-rollback) across [owners](../../glossary.md#term-owner) and external state is not available. |
| [Prepare](../../glossary.md#prepare) a distinct unpublished model and session, then publish under caller exclusion | Existing [owner](../../glossary.md#term-owner) [validation](../../glossary.md#term-validation) can [run](../../glossary.md#term-run) against the final candidate | Candidate preparation and a possible `state()` [snapshot](../../glossary.md#term-snapshot) cost resources; callers must stage their external state and control the [handoff](../../glossary.md#term-handoff). |

The third option is selected. The table compares failure isolation, not durable file formats.

## Decision

At a [quiescent](../../glossary.md#term-quiescent) successful-[run](../../glossary.md#term-run) boundary, the caller establishes a complete, supported recovery
point: [detached](../../glossary.md#detached-model-results) [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) values and per-[module](../../glossary.md#term-module) modes, Training state, effective forward contexts,
fixed-[objective](../../glossary.md#term-objective) construction inputs, and typed next-input, random, and other external state.
Every fact available from that detached evidence is checked before mutating a published object;
missing or unsupported participants prevent an exact claim. Comparable detached NN and Training
logical [parameter](../../glossary.md#term-parameter) bytes may be checked at this stage, but doing so before session open is not a
mandatory [API](../../glossary.md#term-application-programming-interface-api) contract.

The caller constructs a *distinct unpublished* NN tree. It loads NN [bindings](../../glossary.md#term-binding) and modes there,
uses the existing [native parameter copy](../../glossary.md#native-parameter-copy--trainingparameterstorage) when needed, builds the objective from the final
[Tensors](../../glossary.md#tensor), and opens a private [`TrainingSession`](../../glossary.md#training-session--trainingsession). It checks exact candidate NN-versus-Training
logical parameter bytes at every stable path before restore and [publication](../../glossary.md#term-publication). Where the source
already supplies trustworthy comparable detached bytes, that comparison can precede open;
otherwise the private session's `state()` supplies [canonical](../../glossary.md#term-canonical-form) candidate logical bytes. Raw
native-order NN storage is not compared directly with canonical Training bytes, and a prior
`ModelCheckpoint.inspect` is not trusted for a later `load`. The caller then invokes
`TrainingSession.restore`, which validates session-specific configuration, [schema](../../glossary.md#term-schema), counters,
payloads, and fixed bindings before its own installation.

Only a fully validated replacement and staged typed external state may be published under
caller-owned exclusion. Ordinary failure closes and discards the private candidate without
publishing it. The old session may remain quiescent during candidate preparation and is retired
by the caller at [handoff](../../glossary.md#term-handoff); there is no promise to roll back an old session the caller already
closed. Readers outside the caller's exclusion have no simultaneous-visibility guarantee.
External state that cannot be staged for this handoff prevents an exact-continuation claim.
No general [transaction](../../glossary.md#term-transaction) API or mandatory pure pre-open Training validator is introduced.
The later Training [Checkpoint](../../glossary.md#term-checkpoint) adapter may persist only supported versioned, bounded participant
schemas with [owner](../../glossary.md#term-owner) [validation](../../glossary.md#term-validation); it does not move state semantics into Checkpoint.

## Rationale

The [publication](../../glossary.md#term-publication) boundary, not the private candidate's first mutation, is what protects the
existing live recovery point. [Owner](../../glossary.md#term-owner) checks still [run](../../glossary.md#term-run) on the actual tree and session that will
be used. A [canonical](../../glossary.md#term-canonical-form) post-open comparison avoids inventing a public [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) value reader merely to
bridge [byte order](../../glossary.md#text-encoding-and-file-boundaries). `TrainingSession.state()` copies [parameter](../../glossary.md#term-parameter) and [optimizer](../../glossary.md#term-optimizer) payloads, so an
existing [detached](../../glossary.md#detached-model-results) comparison can avoid that extra allocation and copy where it proves the same
candidate value; neither route relaxes the equality requirement.

## Consequences

This changes the architecture gate for future Training 0004, not current Java behavior or [API](../../glossary.md#term-application-programming-interface-api).
[Validation](../../glossary.md#term-validation) must cover missing and incompatible [detached](../../glossary.md#detached-model-results) participants, [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training byte mismatch,
private open/restore failure, caller-state staging failure, and a successful next-input [handoff](../../glossary.md#term-handoff);
failure must leave no candidate published. It does not make the old session [rollback](../../glossary.md#term-rollback)-capable or
support training-mode BatchNorm, an advancing [graph](../../glossary.md#graph)-[random stream](../../glossary.md#graph-rng-state--graphrngstate), arbitrary external objects,
or cross-route bitwise replay. A later end-to-end [Checkpoint](../../glossary.md#term-checkpoint) [integration test](../../glossary.md#term-integration-test) will need the
Checkpoint test [dependency](../../glossary.md#term-dependency) that the current integration-test [module](../../glossary.md#term-module) does not have; this [ADR](../../glossary.md#term-architecture-decision-record-adr) adds
no build edge. [Persistence](../../glossary.md#term-persistence) and its [codec](../../glossary.md#term-codec) remain a separate adapter gate.

## Related documentation

- [Architecture root ownership route](../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings))</span>
- [Training lifecycle contract](../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Training graph recovery scenario](../../architecture/training-graph.md#continuing-at-the-next-training-input)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [Training restore implementation](../../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)
- [Model checkpoint load implementation](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/checkpoint/ModelCheckpoint.java)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [checkpoint glossary](../../glossary.md#term-checkpoint))</span>
