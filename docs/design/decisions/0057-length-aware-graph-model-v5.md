# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0057: Length-aware [general Scan](../../glossary.md#general-differentiable-scan) [persistence](../../glossary.md#term-persistence)


## Editorial reading context

Checkpoint stores length-aware source metadata and eligible named values, but not external
input-batch values, including external valid lengths. A length Tensor explicitly registered
as eligible named state follows ordinary named-leaf persistence rules; it is not the same role
as an external input. A new version/opcode distinguishes the record from old all-valid bodies
so old decoders reject rather than misparse it. Compiler cold reconstruction and backend
execution still have separate admissions. The design acceptance below is not an inferred
implementation or release receipt.

Use the [current format selection and load boundary](../../api/graph-checkpoint-api.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status and date

Accepted design — 2026-11-01 under continuous production delivery. Independent exact-wire review
APPROVE, zero actionable blockers. Implementation and consumer acceptance belong to
[Checkpoint0014](../../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md),
after [Compiler0018](../../planning/modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md).
This [ADR](../../glossary.md#term-architecture-decision-record-adr) is explanatory; the root and incorporated scoped contracts remain authoritative.

## Context

[General Scan](../../glossary.md#general-differentiable-scan) v4 stores all-valid ordinary [tensor](../../glossary.md#tensor) bodies and raw named state in independent lexical
scopes. [Runtime](../../glossary.md#term-runtime)-length [Scan](../../glossary.md#term-cumulative-scan) adds explicit row roles and a final discrete lengths actual with no
body formal. Extending the existing [opcode](../../glossary.md#term-opcode) 23 header in place would let old decoders misinterpret
role bytes as body structure. [Model](../../glossary.md#term-model) source validity alone grants neither cold nor wire [admission](../../glossary.md#term-admission).

## Decision drivers

Preserve old file bytes, reject unsupported/corrupt inputs before state import and [Tensor](../../glossary.md#tensor) [producer](../../glossary.md#term-producer)
allocation, retain exact [aliases](../../glossary.md#aliasing-and-arena)/roles/[raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity), and reuse existing [graph](../../glossary.md#graph)/[codec](../../glossary.md#term-codec)/reconstruction
[owners](../../glossary.md#term-owner) without a second serializer or execution path. External [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) data must not become saved
[runtime](../../glossary.md#term-runtime) history.

## Options considered

- Mutate [opcode](../../glossary.md#term-opcode) 23 in v4: rejected because its old record [layout](../../glossary.md#layout) must remain unambiguous.
- Persist prepared/runtime state: rejected because it couples files to backend resources and
  does not represent ordinary portable model source.<span data-glossary-navigation="true"> ([runtime glossary](../../glossary.md#term-runtime); [backend glossary](../../glossary.md#backend); [model glossary](../../glossary.md#term-model))</span>
- Add a selective v5 outer opcode on the existing envelope and body [codec](../../glossary.md#term-codec): selected.

## Decision

Use [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model).v5 / SYNGRF05 / version 5 and outer [opcode](../../glossary.md#term-opcode) 33 only for length mode. The
[owning v5 contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
specifies exact field order, optional-[axis](../../glossary.md#term-axis) flags, count bounds, old-version [scope](../../glossary.md#term-scope) [admission](../../glossary.md#term-admission) and
failure ordering. Reuse the unchanged inline-body inventory, dry/write traversal, ArtifactLimits,
raw state [codec](../../glossary.md#term-codec), [Compiler](../../glossary.md#term-compiler) [cold reconstruction](../../glossary.md#term-cold-reconstruction) and [atomic](../../glossary.md#resource-transaction-and-rollback)-live-[publication](../../glossary.md#term-publication) behavior.

The final ordered actual reference supplies lengths exactly once; it is excluded from formal
count. External input bytes, including lengths, are not saved. Named values continue following
ordinary explicit [Checkpoint](../../glossary.md#term-checkpoint) state rules. Non-length sources select byte-identical smallest v1-v4;
v5 may contain both old all-valid and new length occurrences.

## Rationale

Explicit version and [opcode](../../glossary.md#term-opcode) boundaries keep parsing deterministic and make malformed-input tests
independent of the production encoder. This follows established [serialization](../../glossary.md#term-serialization) practice of
separating versioned [metadata](../../glossary.md#term-metadata) from raw [tensor](../../glossary.md#tensor) storage, while retaining Synaptik's stricter bounded
non-pickle format. See [PyTorch serialization semantics](https://docs.pytorch.org/docs/2.15/notes/serialization.html)
for that general metadata/storage distinction, not as a wire-format compatibility [target](../../glossary.md#term-target).

## Consequences

Readers/writers require coordinated source changes and independent wire/bounds review. Old readers
reject v5; new readers keep exact old records. Body/[node](../../glossary.md#node) reconstruction remains O(body), not O(T),
with no native/compiled state or [optimizer](../../glossary.md#term-optimizer)-[resume](../../glossary.md#schema-serialization-and-checkpoint) claim. Saved models still require a supported
[Engine](../../glossary.md#term-engine) at execution time. Source, cold, [AD](../../glossary.md#autograd), [backend](../../glossary.md#backend) and [persistence](../../glossary.md#term-persistence) qualifications stay separate.

## Related documentation

- [Root authority](../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
- [Length-aware Scan](../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../glossary.md#term-cumulative-scan))</span>
- [Cold constructor inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)
- [Existing general-body persistence decision](0047-bounded-general-scan-graph-persistence.md)<span data-glossary-navigation="true"> ([persistence glossary](../../glossary.md#term-persistence))</span>
