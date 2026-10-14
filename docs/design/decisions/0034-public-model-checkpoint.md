# ADR 0034: Public model checkpoint with target-defined v1 aliases

## Status

Accepted — 2026-10-14. Direct restored-heap Engine execution was subsequently validated after
CPU 0024, completing Checkpoint 0004's direct acceptance; the focused test now computes each
restored Tensor directly. The [architecture root](../../../ARCHITECTURE.md#dependency-rules) and
[owning extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
are authoritative; this record explains the decision and does not broaden backend scope.

## Context

The internal v1 model candidate and opaque file envelope can capture complete NN state, but
callers need a public save, inspection, and strict load workflow. Typed inspection names Model
`DataType` and `Shape`, while save/load name NN `Module` and public `Engine`. The file contains
one payload per path, not source Tensor identity. NN's loader can initialize reserved lazy state,
but the v1 file cannot establish an unbound target's intended aliases.

## Decision drivers

- Keep v1 bytes and opaque artifact publication unchanged.
- Let NN validate and install whole-tree state; use only public Engine materialization at save.
- Expose typed, detached inspection without leaking live owners.
- Fail closed on ambiguous target sharing or unbound lazy state.
- Preserve every represented bit and use compatible host carriers without an arena lifetime.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Opaque inspection only | No Model edge | Callers must parse state metadata themselves. |
| Typed inspection and direct Model `api` edge | Useful public descriptors | Expands the published dependency set and requires coordinated contract/test updates. |
| Infer source aliases from equal payloads | No format change | Equal values do not prove shared source Tensor identity. |
| Add alias IDs to v1 | Verifiable source topology | Breaks the frozen v1 wire and needs a separate format/NN decision. |
| Accept lazy target restore | Matches NN's in-memory loader | Its alias topology is not knowable through the complete bound snapshot. |

## Decision

Publish `ModelCheckpoint.save`, `inspect`, and `load` with NN, Engine, and Model as direct Gradle
`api` dependencies. Inspection validates the entire bounded file, model schema, payload mapping,
and BOOL semantics before returning immutable typed metadata. Load rereads and revalidates the
file and exact caller fingerprints, requires a complete bound target snapshot, checks every
target-defined alias group's stored bytes, reconstructs one native-order heap Tensor per group,
then calls NN's strict loader once. A group with any parameter is gradient-eligible. The caller's
versioned configuration fingerprint must cover intended sharing because v1 cannot verify source
aliases. No wire byte or artifact publication rule changes.

## Rationale

The direct Model edge expresses the actual public typed API rather than relying on transitive
implementation leakage. Target-defined sharing is the only complete alias policy possible with
unchanged v1 bytes; rejecting differing bytes prevents silently collapsing distinct values.
Refusing lazy targets avoids installing under unknown alias topology. Raw integer-width bit copy
avoids floating conversion and NaN canonicalization.

## Consequences

### Positive

Callers can inspect detached state before choosing a recreated target. Ordinary malformed,
digest, schema, alias, and reconstruction failures occur before any NN installation. Heap-backed
Tensors have automatic lifetime and no Checkpoint-owned close operation.

### Negative and risks

The API cannot authenticate adversarial files, reconstruct Java model code, prove source alias
identity, resume training, or promise rename survival after power loss. Limits do not bound peak
heap. Target and source must be quiescent during their respective calls.

### Migration, testing, and follow-up

The v1 wire remains unchanged, so there is no migration. Focused tests cover typed inspection,
seven data types, raw represented bits, target aliases, digest and schema failures, invalid BOOL,
unbound targets, and unchanged bindings on failure. After CPU 0024, the focused checkpoint test
also executes each restored heap Tensor directly through Engine and verifies its canonical bytes.
Model-only save, inspect, and load are accepted; a separate Training Checkpoint adapter and
cross-owner restore protocol remain future work for exact training continuation.

## Related documentation

- [Checkpoint 0004 task](../../planning/extensions/checkpoint/tasks/0004-public-model-checkpoint-save-inspect-load.md)
- [Model checkpoint API guide](../../api/model-checkpoint-api.md)
- [Focused model checkpoint test](../../../extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/model/ModelCheckpointTest.java)
- [Internal schema decision](0033-model-candidate-schema.md)
