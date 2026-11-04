# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0034: Public [model](../../glossary.md#term-model) [checkpoint](../../glossary.md#term-checkpoint) with [target](../../glossary.md#term-target)-defined v1 [aliases](../../glossary.md#aliasing-and-arena)


## Editorial reading context

The application supplies a fully bound target module and expected fingerprints. Checkpoint validates file values and target alias groups, reconstructs heap-backed Tensors, then asks NN to install them. Equal bytes do not independently establish intended parameter sharing; the target topology defines v1 aliases. Inspection metadata is not a promise that a later reread sees identical file bytes.

Use the [strict model load and aliases](../../api/model-checkpoint-api.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-10-14. Direct restored-[heap](../../glossary.md#term-heap) [Engine](../../glossary.md#term-engine) execution was subsequently validated after
[CPU](../../glossary.md#term-cpu) 0024, completing [Checkpoint](../../glossary.md#term-checkpoint) 0004's direct acceptance; the focused test now computes each
restored [Tensor](../../glossary.md#tensor) directly. The [architecture root](../../../ARCHITECTURE.md#dependency-rules) and
[owning extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
are authoritative; this record explains the decision and does not broaden [backend](../../glossary.md#backend) [scope](../../glossary.md#term-scope).

## Context

The internal v1 [model](../../glossary.md#term-model) candidate and [opaque](../../glossary.md#term-opaque) file envelope can [capture](../../glossary.md#term-capture-meanings) complete [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state, but
callers need a public save, inspection, and strict load workflow. Typed inspection names Model
[`DataType`](../../glossary.md#data-type--datatype) and [`Shape`](../../glossary.md#shape), while save/load name NN [`Module`](../../glossary.md#term-module) and public [`Engine`](../../glossary.md#term-engine). The file contains
one [payload](../../glossary.md#artifact-envelope-and-atomic-publication) per path, not source [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings). NN's loader can initialize reserved lazy state,
but the v1 file cannot establish an unbound [target](../../glossary.md#term-target)'s intended [aliases](../../glossary.md#aliasing-and-arena).

## Decision drivers

- Keep v1 bytes and [opaque](../../glossary.md#term-opaque) [artifact](../../glossary.md#term-artifact) [publication](../../glossary.md#term-publication) unchanged.
- Let [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) validate and install whole-tree state; use only public [Engine](../../glossary.md#term-engine) [materialization](../../glossary.md#materialization) at save.
- Expose typed, [detached](../../glossary.md#detached-model-results) inspection without leaking live [owners](../../glossary.md#term-owner).
- [Fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) on ambiguous [target](../../glossary.md#term-target) sharing or unbound lazy state.
- Preserve every represented bit and use compatible [host](../../glossary.md#term-host) carriers without an [arena](../../glossary.md#term-arena) [lifetime](../../glossary.md#term-lifetime).

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| [Opaque](../../glossary.md#term-opaque) inspection only | No [Model](../../glossary.md#term-model) edge | Callers must parse state [metadata](../../glossary.md#term-metadata) themselves. |
| Typed inspection and direct Model `api` edge | Useful public [descriptors](../../glossary.md#term-descriptor) | Expands the published [dependency](../../glossary.md#term-dependency) set and requires coordinated contract/test updates. |
| Infer source [aliases](../../glossary.md#aliasing-and-arena) from equal payloads | No format change | Equal values do not prove shared source [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings). |
| Add alias IDs to v1 | Verifiable source [topology](../../glossary.md#term-topology) | Breaks the frozen v1 wire and needs a separate format/[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) decision. |
| Accept lazy [target](../../glossary.md#term-target) restore | Matches NN's in-memory loader | Its alias topology is not knowable through the complete bound [snapshot](../../glossary.md#term-snapshot). |

## Decision

Publish `ModelCheckpoint.save`, `inspect`, and `load` with [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [Engine](../../glossary.md#term-engine), and [Model](../../glossary.md#term-model) as direct [Gradle](../../glossary.md#term-gradle)
`api` dependencies. Inspection validates the entire bounded file, model [schema](../../glossary.md#term-schema), [payload](../../glossary.md#artifact-envelope-and-atomic-publication) mapping,
and [BOOL](../../glossary.md#data-type--datatype) semantics before returning [immutable](../../glossary.md#term-immutable) typed [metadata](../../glossary.md#term-metadata). Load rereads and revalidates the
file and exact caller fingerprints, requires a complete bound [target](../../glossary.md#term-target) [snapshot](../../glossary.md#term-snapshot), checks every
target-defined [alias](../../glossary.md#aliasing-and-arena) group's stored bytes, reconstructs one native-order [heap](../../glossary.md#term-heap) [Tensor](../../glossary.md#tensor) per group,
then calls NN's strict loader once. A group with any [parameter](../../glossary.md#term-parameter) is [gradient](../../glossary.md#term-gradient)-eligible. The caller's
versioned configuration [fingerprint](../../glossary.md#term-fingerprint) must cover intended sharing because v1 cannot verify source
aliases. No wire byte or [artifact](../../glossary.md#term-artifact) [publication](../../glossary.md#term-publication) rule changes.

## Rationale

The direct [Model](../../glossary.md#term-model) edge expresses the actual public typed [API](../../glossary.md#term-application-programming-interface-api) rather than relying on transitive
implementation leakage. [Target](../../glossary.md#term-target)-defined sharing is the only complete [alias](../../glossary.md#aliasing-and-arena) policy possible with
unchanged v1 bytes; rejecting differing bytes prevents silently collapsing distinct values.
Refusing lazy targets avoids installing under unknown alias [topology](../../glossary.md#term-topology). Raw integer-width bit copy
avoids floating conversion and [NaN](../../glossary.md#term-nan) [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form).

## Consequences

### Positive

Callers can inspect [detached](../../glossary.md#detached-model-results) state before choosing a recreated [target](../../glossary.md#term-target). Ordinary malformed,
[digest](../../glossary.md#term-digest), [schema](../../glossary.md#term-schema), [alias](../../glossary.md#aliasing-and-arena), and reconstruction failures occur before any [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) installation. [Heap](../../glossary.md#term-heap)-backed
[Tensors](../../glossary.md#tensor) have automatic [lifetime](../../glossary.md#term-lifetime) and no [Checkpoint](../../glossary.md#term-checkpoint)-owned close [operation](../../glossary.md#operation).

### Negative and risks

The [API](../../glossary.md#term-application-programming-interface-api) cannot authenticate adversarial files, reconstruct Java [model](../../glossary.md#term-model) code, prove source [alias](../../glossary.md#aliasing-and-arena)
[identity](../../glossary.md#term-identity-meanings), [resume](../../glossary.md#schema-serialization-and-checkpoint) training, or promise rename survival after power loss. Limits do not bound peak
[heap](../../glossary.md#term-heap). [Target](../../glossary.md#term-target) and source must be [quiescent](../../glossary.md#term-quiescent) during their respective calls.

### Migration, testing, and follow-up

The v1 wire remains unchanged, so there is no migration. Focused tests cover typed inspection,
seven data types, raw [represented bits](../../glossary.md#term-represented-bits), [target](../../glossary.md#term-target) [aliases](../../glossary.md#aliasing-and-arena), [digest](../../glossary.md#term-digest) and [schema](../../glossary.md#term-schema) failures, invalid [BOOL](../../glossary.md#data-type--datatype),
unbound targets, and unchanged [bindings](../../glossary.md#term-binding) on failure. After [CPU](../../glossary.md#term-cpu) 0024, the focused [checkpoint](../../glossary.md#term-checkpoint) test
also executes each restored [heap](../../glossary.md#term-heap) [Tensor](../../glossary.md#tensor) directly through [Engine](../../glossary.md#term-engine) and verifies its [canonical](../../glossary.md#term-canonical-form) bytes.
[Model](../../glossary.md#term-model)-only save, inspect, and load are accepted; a separate Training Checkpoint adapter and
cross-[owner](../../glossary.md#term-owner) restore protocol remain future work for [exact training continuation](../../glossary.md#exact-training-continuation).

## Related documentation

- [Checkpoint 0004 task](../../planning/extensions/checkpoint/tasks/0004-public-model-checkpoint-save-inspect-load.md)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint))</span>
- [Model checkpoint API guide](../../api/model-checkpoint-api.md)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [Focused model checkpoint test](../../../extensions/nn/src/test/java/io/github/pho001/synaptik/nn/checkpoint/ModelCheckpointTest.java)
- [Internal schema decision](0033-model-candidate-schema.md)<span data-glossary-navigation="true"> ([schema glossary](../../glossary.md#term-schema))</span>
