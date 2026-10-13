# ADR 0033: Internal model-candidate schema and bounded Engine handoff

## Status

Accepted — 2026-10-13, after independent Class C review. The
[architecture root](../../../ARCHITECTURE.md#module-ownership-routing) and
[owning extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
are authoritative; this record explains the decision.

## Context

NN exports a complete but shallow `StateDictionary`: paths and exact live Tensor references.
The opaque artifact file already carries bounded bytes, but it cannot infer module-state meaning
or make a live Tensor a durable value. Engine alone owns host materialization. The next step needs
a deterministic schema and a safe handoff without publishing a model API or adding a Model edge.

## Decision drivers

- Bind ordered state meaning and caller-defined configuration/input identities to stable bytes.
- Reject malformed paths, lengths, and resource demands before Engine work or large copies.
- Respect NN state ownership, Engine execution ownership, and the existing two project edges.
- Permit exact Tensor aliases without losing any state path or undercounting output bytes.

## Options considered

| Option | Benefit | Cost or rejection |
|---|---|---|
| Serialize NN records or enum ordinals | Minimal codec | Java identity/order is not a stable wire contract; live Tensors are not detached values. |
| Read Tensor host storage in Checkpoint | Avoids Engine lifecycle | Bypasses Engine materialization and cannot cover backend-owned results. |
| Add a public model file facade now | Immediately usable | Requires restore semantics and a public Model dependency decision reserved for 0004. |
| Internal fixed-ID metadata plus public bounded Engine compute | Stable schema with current ownership | Extra copies and no public model persistence yet. |

The internal fixed-ID codec and Engine handoff are selected.

## Decision

Checkpoint's package-private model codec encodes `SYNMOD01` version-one metadata under `schema`.
It records exact 32-byte caller configuration SHA-256 and optional input-artifact SHA-256, then
ordered exact NN paths, fixed kind/type IDs, static Shape extents, and checked canonical byte
lengths. The owning contract defines the precise wire layout and ID table. Its SHA-256 is the
schema fingerprint. State payloads use eight-digit ordinal keys, not NN paths. The enclosing
artifact remains `model.v1` under the unchanged opaque v1 envelope.

Capture snapshots `Module.stateDictionary()` once under caller quiescence. It preflights metadata,
entry, individual, aggregate, manifest, and final-file limits before one public
`Engine.compute(List<Tensor>, limit)` call. Exact Tensor identity is deduplicated for compute,
but every path receives its own payload and counts toward the aggregate bound. Empty state skips
Engine. Returned detached values are verified before candidate construction. Decode validates
strict UTF-8, fixed IDs, canonical dimensions and lengths, and exact ordinal correspondence.
No installation or public model file API is introduced.

## Rationale

An ordinal key isolates artifact filename grammar from arbitrary valid NN names, including
Unicode or slash characters. Fixed IDs and versioned framing allow later schema changes to
reject old or unknown forms explicitly. A whole-artifact preflight prevents resource-limit
rejection after execution begins; the Engine call keeps backend access and lifecycle in its
owner. Caller digests bind external semantics without Checkpoint pretending to serialize Java
lambdas, tokenizers, or application configuration.

## Consequences

### Positive

Equal state and caller digests yield equal schema and envelope bytes. A validated candidate is
detached from NN and Engine lifetimes. The two direct Gradle edges and opaque artifact format do
not change.

### Negative and risks

The aggregate limit covers canonical output bytes, including metadata and aliases, not Engine
workspace, source `HostTensorValue`s, defensive copies, or peak heap. Capture requires complete
already-bound and quiescent NN state. Public save/load, Tensor reconstruction, and strict install
remain unavailable. SHA-256 detects accidental corruption and binds identity but does not
authenticate an adversarial writer.

### Migration, testing, and follow-up

There is no prior model schema to migrate. Focused tests exercise every fixed ID, raw host bytes
on supported CPU materialization, Unicode, aliases, count/byte bounds, corruption, empty state,
and detached lifetime. Independent Class C review identified and resolved one preflight-allocation
issue, with a virtual-million-state regression and targeted sign-off. Checkpoint 0004 must decide
the public model API and strict restore without weakening this internal schema.

## Related documentation

- [Checkpoint 0003 task](../../planning/extensions/checkpoint/tasks/0003-model-schema-and-materialization-handoff.md)
- [Checkpoint module boundary](../../architecture/module-boundaries.md#extensionscheckpoint)
- [Artifact envelope decision](0032-checkpoint-artifact-envelope.md)
