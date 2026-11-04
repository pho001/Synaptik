# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0033: Internal [model](../../glossary.md#term-model)-candidate [schema](../../glossary.md#term-schema) and bounded [Engine](../../glossary.md#term-engine) [handoff](../../glossary.md#term-handoff)

## Status

Accepted — 2026-10-13, after independent Class C review. The
[architecture root](../../../ARCHITECTURE.md#module-ownership-routing) and
[owning extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
are authoritative; this record explains the decision.

## Context

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) exports a complete but shallow `StateDictionary`: paths and exact live [Tensor](../../glossary.md#tensor) references.
The [opaque](../../glossary.md#term-opaque) [artifact](../../glossary.md#term-artifact) file already carries bounded bytes, but it cannot infer [module](../../glossary.md#term-module)-state meaning
or make a live Tensor a durable value. [Engine](../../glossary.md#term-engine) alone owns [host](../../glossary.md#term-host) [materialization](../../glossary.md#materialization). The next step needs
a deterministic [schema](../../glossary.md#term-schema) and a safe [handoff](../../glossary.md#term-handoff) without publishing a [model](../../glossary.md#term-model) [API](../../glossary.md#term-application-programming-interface-api) or adding a Model edge.

## Decision drivers

- Bind ordered state meaning and caller-defined configuration/input [identities](../../glossary.md#term-identity-meanings) to stable bytes.
- Reject malformed paths, lengths, and resource demands before [Engine](../../glossary.md#term-engine) work or large copies.
- Respect [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state [ownership](../../glossary.md#term-ownership-meanings), Engine execution ownership, and the existing two project edges.
- Permit exact [Tensor](../../glossary.md#tensor) [aliases](../../glossary.md#aliasing-and-arena) without losing any state path or undercounting output bytes.

## Options considered

| Option | Benefit | Cost or rejection |
|---|---|---|
| Serialize [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) records or enum ordinals | Minimal [codec](../../glossary.md#term-codec) | Java [identity](../../glossary.md#term-identity-meanings)/order is not a stable wire contract; live [Tensors](../../glossary.md#tensor) are not [detached](../../glossary.md#detached-model-results) values. |
| Read Tensor [host storage](../../glossary.md#host-storage) in [Checkpoint](../../glossary.md#term-checkpoint) | Avoids [Engine](../../glossary.md#term-engine) [lifecycle](../../glossary.md#lifecycle) | Bypasses Engine [materialization](../../glossary.md#materialization) and cannot cover [backend](../../glossary.md#backend)-owned results. |
| Add a public [model](../../glossary.md#term-model) file [facade](../../glossary.md#term-facade) now | Immediately usable | Requires restore semantics and a public Model [dependency](../../glossary.md#term-dependency) decision reserved for 0004. |
| Internal fixed-ID [metadata](../../glossary.md#term-metadata) plus public bounded Engine compute | Stable [schema](../../glossary.md#term-schema) with current [ownership](../../glossary.md#term-ownership-meanings) | Extra copies and no public model [persistence](../../glossary.md#term-persistence) yet. |

The internal fixed-ID codec and Engine [handoff](../../glossary.md#term-handoff) are selected.

## Decision

[Checkpoint](../../glossary.md#term-checkpoint)'s package-private [model](../../glossary.md#term-model) [codec](../../glossary.md#term-codec) encodes `SYNMOD01` version-one [metadata](../../glossary.md#term-metadata) under [`schema`](../../glossary.md#term-schema).
It records exact 32-byte caller configuration [SHA-256](../../glossary.md#term-sha-256) and optional input-[artifact](../../glossary.md#term-artifact) SHA-256, then
ordered exact [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) paths, fixed kind/type IDs, static [Shape](../../glossary.md#shape) extents, and checked [canonical](../../glossary.md#term-canonical-form) byte
lengths. The owning contract defines the precise wire [layout](../../glossary.md#layout) and ID table. Its SHA-256 is the
schema [fingerprint](../../glossary.md#term-fingerprint). State payloads use eight-digit ordinal keys, not NN paths. The enclosing
artifact remains `model.v1` under the unchanged [opaque](../../glossary.md#term-opaque) v1 envelope.

[Capture](../../glossary.md#term-capture-meanings) snapshots `Module.stateDictionary()` once under caller [quiescence](../../glossary.md#term-quiescence). It preflights metadata,
entry, individual, aggregate, [manifest](../../glossary.md#term-manifest), and final-file limits before one public
`Engine.compute(List<Tensor>, limit)` call. Exact [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-reference-identity) is deduplicated for compute,
but every path receives its own [payload](../../glossary.md#artifact-envelope-and-atomic-publication) and counts toward the aggregate bound. Empty state skips
[Engine](../../glossary.md#term-engine). Returned [detached](../../glossary.md#detached-model-results) values are verified before candidate construction. Decode validates
strict [UTF-8](../../glossary.md#term-utf-8), fixed IDs, canonical [dimensions](../../glossary.md#dimension) and lengths, and exact ordinal correspondence.
No installation or public model file [API](../../glossary.md#term-application-programming-interface-api) is introduced.

## Rationale

An ordinal key isolates [artifact](../../glossary.md#term-artifact) filename grammar from arbitrary valid [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) names, including
[Unicode](../../glossary.md#term-unicode) or slash characters. Fixed IDs and versioned [framing](../../glossary.md#term-framing) allow later [schema](../../glossary.md#term-schema) changes to
reject old or unknown forms explicitly. A whole-artifact [preflight](../../glossary.md#term-preflight) prevents resource-limit
rejection after execution begins; the [Engine](../../glossary.md#term-engine) call keeps [backend](../../glossary.md#backend) access and [lifecycle](../../glossary.md#lifecycle) in its
[owner](../../glossary.md#term-owner). Caller digests bind external semantics without [Checkpoint](../../glossary.md#term-checkpoint) pretending to serialize Java
lambdas, tokenizers, or application configuration.

## Consequences

### Positive

Equal state and caller digests yield equal [schema](../../glossary.md#term-schema) and envelope bytes. A validated candidate is
[detached](../../glossary.md#detached-model-results) from [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and [Engine](../../glossary.md#term-engine) lifetimes. The two direct [Gradle](../../glossary.md#term-gradle) edges and [opaque](../../glossary.md#term-opaque) [artifact](../../glossary.md#term-artifact) format do
not change.

### Negative and risks

The aggregate limit covers [canonical](../../glossary.md#term-canonical-form) output bytes, including [metadata](../../glossary.md#term-metadata) and [aliases](../../glossary.md#aliasing-and-arena), not [Engine](../../glossary.md#term-engine)
workspace, source [`HostTensorValue`](../../glossary.md#host-tensor-value--hosttensorvalue)s, defensive copies, or peak [heap](../../glossary.md#term-heap). [Capture](../../glossary.md#term-capture-meanings) requires complete
already-bound and [quiescent](../../glossary.md#term-quiescent) [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state. Public save/load, [Tensor](../../glossary.md#tensor) reconstruction, and strict install
remain unavailable. [SHA-256](../../glossary.md#term-sha-256) detects accidental corruption and binds [identity](../../glossary.md#term-identity-meanings) but does not
authenticate an adversarial writer.

### Migration, testing, and follow-up

There is no prior [model](../../glossary.md#term-model) [schema](../../glossary.md#term-schema) to migrate. Focused tests exercise every fixed ID, raw [host](../../glossary.md#term-host) bytes
on supported [CPU](../../glossary.md#term-cpu) [materialization](../../glossary.md#materialization), [Unicode](../../glossary.md#term-unicode), [aliases](../../glossary.md#aliasing-and-arena), count/byte bounds, corruption, empty state,
and [detached](../../glossary.md#detached-model-results) [lifetime](../../glossary.md#term-lifetime). Independent Class C review identified and resolved one [preflight](../../glossary.md#term-preflight)-allocation
issue, with a virtual-million-state [regression](../../glossary.md#term-regression) and targeted sign-off. [Checkpoint](../../glossary.md#term-checkpoint) 0004 must decide
the public model [API](../../glossary.md#term-application-programming-interface-api) and strict restore without weakening this internal schema.

## Related documentation

- [Checkpoint 0003 task](../../planning/extensions/checkpoint/tasks/0003-model-schema-and-materialization-handoff.md)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint))</span>
- [Checkpoint module boundary](../../architecture/module-boundaries.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([module glossary](../../glossary.md#term-module))</span>
- [Artifact envelope decision](0032-checkpoint-artifact-envelope.md)<span data-glossary-navigation="true"> ([Artifact envelope glossary](../../glossary.md#term-artifact-envelope))</span>
