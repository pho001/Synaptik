# ADR 0057: Length-aware general Scan persistence

## Status and date

Accepted design — 2026-11-01 under continuous production delivery. Independent exact-wire review
APPROVE, zero actionable blockers. Implementation and consumer acceptance belong to
[Checkpoint0014](../../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md),
after [Compiler0018](../../planning/modules/compiler/tasks/0018-static-length-scan-cold-reconstruction.md).
This ADR is explanatory; the root and incorporated scoped contracts remain authoritative.

## Context

General Scan v4 stores all-valid ordinary tensor bodies and raw named state in independent lexical
scopes. Runtime-length Scan adds explicit row roles and a final discrete lengths actual with no
body formal. Extending the existing opcode 23 header in place would let old decoders misinterpret
role bytes as body structure. Model source validity alone grants neither cold nor wire admission.

## Decision drivers

Preserve old file bytes, reject unsupported/corrupt inputs before state import and Tensor producer
allocation, retain exact aliases/roles/raw bits, and reuse existing graph/codec/reconstruction
owners without a second serializer or execution path. External batch data must not become saved
runtime history.

## Options considered

- Mutate opcode 23 in v4: rejected because its old record layout must remain unambiguous.
- Persist prepared/runtime state: rejected because it couples files to backend resources and
  does not represent ordinary portable model source.
- Add a selective v5 outer opcode on the existing envelope and body codec: selected.

## Decision

Use graph-model.v5 / SYNGRF05 / version 5 and outer opcode 33 only for length mode. The
[owning v5 contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
specifies exact field order, optional-axis flags, count bounds, old-version scope admission and
failure ordering. Reuse the unchanged inline-body inventory, dry/write traversal, ArtifactLimits,
raw state codec, Compiler cold reconstruction and atomic-live-publication behavior.

The final ordered actual reference supplies lengths exactly once; it is excluded from formal
count. External input bytes, including lengths, are not saved. Named values continue following
ordinary explicit Checkpoint state rules. Non-length sources select byte-identical smallest v1-v4;
v5 may contain both old all-valid and new length occurrences.

## Rationale

Explicit version and opcode boundaries keep parsing deterministic and make malformed-input tests
independent of the production encoder. This follows established serialization practice of
separating versioned metadata from raw tensor storage, while retaining Synaptik's stricter bounded
non-pickle format. See [PyTorch serialization semantics](https://docs.pytorch.org/docs/2.15/notes/serialization.html)
for that general metadata/storage distinction, not as a wire-format compatibility target.

## Consequences

Readers/writers require coordinated source changes and independent wire/bounds review. Old readers
reject v5; new readers keep exact old records. Body/node reconstruction remains O(body), not O(T),
with no native/compiled state or optimizer-resume claim. Saved models still require a supported
Engine at execution time. Source, cold, AD, backend and persistence qualifications stay separate.

## Related documentation

- [Root authority](../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
- [Length-aware Scan](../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)
- [Cold constructor inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)
- [Existing general-body persistence decision](0047-bounded-general-scan-graph-persistence.md)
