# ADR 0047: Bounded general-scan graph persistence

## Status

Accepted decision — 2026-10-23. The user approved selective graph-model.v4 and the existing bounded
cold inventory with “ok, pust se do toho”. Main accepted CP0013 Ready and authorized the pre-code
authority/documentation cutover; independent pre-code approval and main acceptance then preceded
implementation. The codec and focused tests are implemented; main accepted final independent review,
module/Javadoc, integration, architecture and configured root gates. CP0013 is Complete.

This record explains the [architecture root](../../../ARCHITECTURE.md), not a separate authority.
It follows [ADR 0046](0046-general-scan-cold-source-and-owned-model-direction.md), whose cold-source
and owned-workflow decisions remain unchanged, and extends the graph persistence direction of
[ADR 0039](0039-portable-graph-model-persistence.md).

## Context

Compiler now reconstructs bounded general-scan bodies, Checkpoint copy/rebind preserves scoped
attachments and NN supports the all-valid owned compute/first-order training workflow. Disk
save/load rejected bodies independently of those capabilities when this decision was made.
Published graph-model v1/v2/v3
have no general-body field; silently changing v3 would break exact tuple/record compatibility.
Fluent `.rnn` migration cannot retain tested save/load merely because a general scan can execute.
That migration also has an independent backend/layout compatibility decision; this ADR does not
authorize it.

## Decision drivers

- Persist ordinary source and current raw values through existing APIs and module owners.
- Preserve exact old bytes/readers and fail closed before state import or publication.
- Match existing constructor-backed cold admission without importing CPU execution predicates.
- Retain bounded metadata, independent scope identity, aliases and all canonical ports.
- Add no new public API/type, reconstruction layer, execution owner or runtime history format.

## Options considered

| Option | Consequence |
|---|---|
| Keep body files rejected | Preserves the status quo but cannot deliver the selected persistence capability. |
| Encode only RNN-like bodies | Adds another admission subset despite existing bounded cold constructors; does not address the same general owned-source capability. |
| Mutate v3 or export unrolled graphs | Breaks old exact records or substitutes T-fold provenance for the accepted scoped source; rejected. |
| Encode arbitrary PURE bodies | Requires unsupported constructor coverage and broadens architecture beyond the selected inventory; rejected. |
| Selective v4 with the existing bounded cold inventory | Selected: explicit compatibility boundary with no second family reconstruction owner. |

The options differ in what can be saved and whether existing source/format boundaries remain intact.

## Decision and rationale

Use ArtifactFile v1 kind graph-model.v4 for sources containing general SCAN, with one inline scoped
body per occurrence. The [extensions-training owner](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
alone defines all wire fields, guards, bounds and failure order. Outer admission is v3 plus SCAN;
body admission is exactly [Compiler's existing cold inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
Model-valid PURE is not automatically reconstructible, differentiable or executable.

Retain the smallest sufficient byte-identical v1/v2/v3 write for body-free sources and their exact
readers. Old readers reject v4. No selector, registry, NN codec, implicit cast or exporter fallback
is added. Compiler preflights and reconstructs ordinary source once; Checkpoint owns wire/bounds/
host-value publication, and NN consumes the same existing facade. Residual bindings are metadata,
not activation-history payloads or saved derivative demand.

For example, a three-step `tanh(carry + step * weight)` source stores one inline body and its
named weight values, never three unrolled bodies or input batches. After load, fresh scoped formals
and ordinary provenance must use the restored weight. Compute/train still require actual body and
binder qualification; cold support for a low dtype does not qualify its execution. This is an
implemented cold roundtrip; focused loaded-consumer tests also compare qualified FLOAT32/FLOAT64
compute/training against independent unrolled oracles. Final acceptance still requires main's gates.

## Consequences

### Benefits

One cohesive CP0013 capability covers authority, codec, tests/docs and consumer proof without a
mechanical successor cascade. Existing ownership, APIs, limits, dependencies and stored raw bits
remain. Current NN acquisition documentation can reflect accepted NN0031 without inventing another
workflow. No Engine is created by save/load and no native/backend/generator change is needed.

### Costs and risks

The codec must validate a larger scoped schema, conservative residual metadata and aggregate
counts before allocation/import. Existing encoded limits are not peak-heap guarantees. Some valid
PURE bodies remain unsupported; exact cold inventory is intentionally narrower than Model validity.
Old readers cannot open v4. Exact optimizer/RNG/cursor continuation, padding and `.rnn` migration
remain outside this decision.

### Validation and migration

Independent pre-code contract review/main acceptance preceded clean codec implementation.
Focused evidence covers exact old-format bytes, scoped roles/ports, malformed late records,
checked limits and rejection before copying/import or body construction. Consumer tests save before
compute and after training, close the source, cold-load/compose, compute/train fresh FLOAT32/FLOAT64
batches against independent unrolled oracles and re-save. These tests followed implementation;
the authority-only cutover ran no executable validation. The [CP0013 brief](../../planning/extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md)
owns evidence and main's accepted final validation results. No fluent builder or backend migration follows
automatically; published old files remain supported without conversion.

## Related documentation

- [Graph checkpoint API and bounded v4 persistence](../../api/graph-checkpoint-api.md#bounded-general-scan-persistence)
- [Cold source and scoped persistence explanation](../../architecture/general-differentiable-scan.md#bounded-cold-persistence)
- [Owned general-scan workflow](../../user-guide/running-models.md#train-an-all-valid-general-scan)
