# Checkpoint Extension Master Plan

## Goal

Plan durable, versioned model checkpoints without turning NN state references into backend storage
access or forcing Training and Text dependencies on model-only users. Plan exact training resume as
a separate optional downstream adapter over Training-owned snapshots.

Mental model:

```text
NN StateDictionary of exact Tensor references
  -> explicit Engine host materialization
  -> immutable model-checkpoint candidate
  -> complete validation and checksums
  -> durable atomic publication

Training-owned optimizer/session snapshot
  + validated model-checkpoint candidate
  -> optional Training Checkpoint adapter
  -> exact-resume artifact
```

This is future planning only. Neither `extensions/checkpoint` nor the optional Training Checkpoint
adapter exists in `ARCHITECTURE.md`, Gradle settings, or production code.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Lifecycle](../../../architecture/lifecycle.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [ADR 0007: Neural-network module and training boundary](../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [ADR 0011: Per-run Runtime resource ownership](../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../planning-guide.md)
- [Implementation roadmap](../../roadmap.md)
- [Engine master plan](../../modules/engine/master-plan.md)
- [NN state-dictionary task](../nn/tasks/0010-state-dictionary-and-checkpoint-contract.md)
- [NN master plan](../nn/master-plan.md)
- [Training master plan](../training/master-plan.md)
- [Text master plan](../text/master-plan.md)

## Scope

- one common bounded artifact envelope with explicit format version, manifest, payload checksums,
  deterministic names, and durable publication policy;
- model checkpoint schema over stable NN state paths/kinds, Tensor data type and structural Shape,
  materialized host payloads, and model schema/config fingerprints;
- an optional caller-supplied input-artifact fingerprint used by text applications to bind the
  exact tokenizer/vocabulary mapping without making Checkpoint parse tokenizer files;
- strict inspect/validate/load ordering and complete validation before any model binding changes;
- explicit Engine/publication materialization of current state values, never backend access from
  NN, Training, or Checkpoint;
- a separate optional Training Checkpoint adapter over Training-owned optimizer/session snapshots;
  and
- safe load of a topology recreated by application code or a later explicitly declarative built-in
  descriptor.

## Out of scope

- serializing arbitrary Java lambdas, closures, classes, random sources, or the executable body of
  `Model.define`;
- compiling, preparing, executing, or discovering a backend inside the checkpoint codec;
- direct physical representation, device-buffer, native-memory, or concrete-backend access;
- saving compiled graphs, compile artifacts, prepared executions, backend kernels, device
  residency, runtime `RunState`, or tuning caches as model/training state;
- tokenizer/vocabulary file parsing, text normalization, corpus readers, or automatic vocabulary
  growth;
- ONNX import/export or using ONNX as a substitute for optimizer/session resume state;
- Java object serialization, reflective class reconstruction, plugin discovery, a checkpoint
  repository/service locator, or a generic manager/facade;
- partial/best-effort state installation or silent version/config/type/Shape conversion; and
- implementation before coordinated architecture, dependency, build, and architecture-test work.

## Module and dependency decision

The preferred architecture uses two optional downstream artifacts rather than one dependency-heavy
module:

```text
modules/model -> extensions/nn
compiler/runtime/prepare/backends -> modules/engine

extensions/nn + modules/engine
  -> extensions/checkpoint

extensions/training + extensions/checkpoint
  -> optional extensions/training-checkpoint
```

`extensions/checkpoint` owns the common durable envelope and `ModelCheckpoint`. It depends on NN
for stable state paths and on Engine for a public host-materialization boundary. It does not depend
on Training or Text, so inference/model-only users do not acquire those extensions.

The optional `extensions/training-checkpoint` owns `TrainingCheckpoint` persistence and depends on
Training plus Checkpoint. Training owns the meaning and validation of optimizer slots/groups,
progress, scheduler, random-number-generator (RNG) state, sampler cursor, and scaler state; the
adapter owns their durable encoding and coordinated installation. Training does not gain file I/O,
and Checkpoint does not import optimizer concepts into its model-only core.

Text remains independent. It writes and loads its own complete tokenizer artifact. A text
application supplies that artifact's canonical fingerprint as the model checkpoint's input-artifact
fingerprint. Strict load compares the exact fingerprint, so two equal-size vocabularies with
different ID mappings fail compatibility without a Checkpoint-to-Text dependency.

These are proposed dependencies, not current architecture permission. The first task must update
the architecture contract and tests before either Gradle project is introduced.

## Proposed package structure

After architecture acceptance:

```text
io.github.pho001.synaptik.checkpoint/
  artifact/  bounded manifest, version, checksum, payload, and atomic-publication contracts
  model/     ModelCheckpoint schema, materialization handoff, strict save/inspect/load

io.github.pho001.synaptik.training.checkpoint/
  training checkpoint schema and exact-resume save/inspect/load adapter
```

Do not add `manager`, `service`, `repository`, `registry`, `plugin`, or backend-specific packages.
The optional Training Checkpoint package belongs to its separate downstream Gradle project, not to
the model-only project.

## Model checkpoint contract

A planned model checkpoint contains at least:

```text
artifact format and schema version
model schema/config fingerprint
optional input-artifact fingerprint
ordered state entries:
  stable path
  PARAMETER or BUFFER kind
  DataType
  structural Shape
  payload location, byte count, and checksum
materialized host payload bytes
manifest checksum/publication metadata
```

The model schema fingerprint covers the complete expected state-path/kind/type/Shape structure and
the caller-supplied configuration facts required to recreate the same topology. It must not be
derived only from a model class name or state count. The input-artifact fingerprint is optional for
numeric models; a text model supplies the tokenizer artifact fingerprint, which covers exact
normalization, special IDs, token-ID order, and algorithm data.

The current NN `StateDictionary` supplies exact in-memory Tensor references only. It performs no
evaluation, copy, backend transfer, or serialization. A checkpoint save therefore waits for the
future Engine host-materialization API, which must publish selected current values through prepared
Runtime/backend work into bounded caller-owned host payloads. Checkpoint never casts or reads a
backend representation.

## Training checkpoint contract

`TrainingCheckpoint` is a strict superset of a compatible model checkpoint. Exact resume includes
every state owner selected by the Training contract:

- optimizer type/configuration, parameter groups, stable parameter-path association, and every
  optimizer slot such as Adam moments;
- global step, epoch, scheduler state, and any accumulation counters;
- graph/dropout RNG state;
- data-shuffle RNG and sampler cursor when exact data-order resume is promised;
- mixed-precision scaler state when present; and
- the same model configuration and input-artifact fingerprints.

If a data source cannot expose a stable sampler cursor, the artifact may support a documented
restart mode but must not call it exact resume. Compiled/prepared/backend/device artifacts are
never included; they are reconstructed for the current Engine and target after restore.

## Validation, installation, and publication

Load is staged and fail-closed:

1. Bound file and aggregate sizes before allocating payloads.
2. Parse the manifest with duplicate/path-traversal rejection and supported version checks.
3. Verify every declared payload size and checksum before constructing a candidate.
4. Verify model schema/config and any supplied input-artifact fingerprint.
5. Validate the complete path/kind/type/Shape set against the recreated topology; a deferred model
   may bind from checkpoint Shapes only under the later NN lazy-binding contract.
6. For training resume, validate every model, optimizer, group, progress, RNG, sampler, scheduler,
   and scaler component across all owners before installing any one of them.
7. Install through NN and Training's public strict restore contracts only after complete
   validation. No backend storage bypass or partial permissive mode is implied.

Save publishes through a sibling temporary target on the same filesystem. It writes bounded
payloads and checksums, writes the final manifest only after payload completion, applies the
selected durable flush policy, reopens/validates the complete temporary artifact, and renames it to
the final target. The implementation task must define existing-target behavior and whether lack of
atomic rename fails or enters an explicitly named weaker mode. A normal save must never expose a
half-written final artifact.

## Model reconstruction decision

An arbitrary `Model.define(topology -> ...)` Java lambda is not a portable serialization format.
The primary workflow is:

```text
application code/config recreates the topology
  -> checkpoint inspect validates metadata
  -> strict load validates complete state
  -> exact materialized values are installed
```

A model checkpoint may persist a caller-defined, versioned configuration record/fingerprint, but
it does not reconstruct arbitrary application bytecode. A later built-in declarative model
descriptor is a separate capability with an explicit closed schema; it must not be inferred from a
lambda or added as a generic reflective registry.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Checkpoint and optional Training Checkpoint architecture boundary | Draft | Explicit architecture decision; current CPU frontier preserved | Update `ARCHITECTURE.md`, lifecycle/module/dependency explanations, a significant ADR, settings/build ownership, and architecture tests for model-only Checkpoint downstream of NN+Engine and optional Training Checkpoint downstream of Training+Checkpoint; add no persistence API in the architecture task. |
| 0002 | Bounded durable artifact envelope | Draft | 0001; selected codec/filesystem policy | Define deterministic versioned manifest/payload/checksum types, resource bounds, temporary-target publication, durable flush, reopen validation, rename semantics, and corruption rejection without model/training concepts. |
| 0003 | Model schema and Engine materialization handoff | Draft | 0002; Engine 0002–0003; NN 0019 lazy/checkpoint lifecycle | Define stable model state/config/input-artifact fingerprints and convert a complete NN state snapshot to bounded host payloads only through Engine publication/materialization. |
| 0004 | Model checkpoint save, inspect, and strict load | Draft | 0003 | Persist materialized parameters/buffers and validate the entire artifact plus recreated topology before one strict NN installation; never serialize arbitrary `Model.define` lambdas. |
| 0005 | Training-owned exact-resume snapshot prerequisite | Draft | Training 0002–0004; 0004 | Stabilize optimizer/session snapshot and cross-owner validate/install contracts before the optional adapter encodes them. |
| 0006 | Optional Training Checkpoint adapter | Draft | Architecture authorization for the optional project; 0002, 0004–0005 | Persist and restore the model plus optimizer/groups/progress/scheduler/RNG/sampler/scaler state without making model-only Checkpoint depend on Training. |
| 0007 | Checkpoint capability and recovery checkpoint | Draft | 0004; 0006 when exact training resume is selected; Text 0003 for text integration | Validate corruption/resource bounds, atomic publication and interruption recovery, strict no-partial-install behavior, text fingerprint mismatch, model-only load, exact training resume, documentation, and dependency enforcement. |

No detailed Checkpoint task exists. NN 0018 remains the sole new detailed `Ready` task in this
planning program. Promote Checkpoint 0001 only when the architecture change is the active and
explicitly authorized implementation frontier.

## Planned end-to-end flows

The following flows are conceptual and not current runnable Synaptik APIs.

Build and save a text model:

```text
train or load immutable Text TokenizerArtifact
  -> persist tokenizer artifact atomically
  -> define/bind Model using vocabulary size
  -> train or load NN state
  -> Engine materializes complete model state to host payloads
  -> save ModelCheckpoint with tokenizer fingerprint
```

Load for inference:

```text
load and validate TokenizerArtifact
  -> recreate Model topology from application code/config
  -> inspect ModelCheckpoint
  -> require exact tokenizer fingerprint
  -> validate all state before strict installation
  -> compile/prepare for the selected current Engine/backends
```

Resume training:

```text
load tokenizer artifact and verify fingerprint
  -> recreate Model, optimizer, scheduler, and TrainingSession
  -> parse/checksum/validate the complete TrainingCheckpoint
  -> validate model + optimizer/session candidate across owners
  -> install only after every validation succeeds
  -> rebuild compiled/prepared execution
  -> continue from recorded progress/RNG/sampler state
```

## Milestones

- Accepted checkpoint module/dependency architecture
- Bounded durable artifact and atomic-publication foundation
- Engine-owned host materialization used by model checkpointing
- Strict model save/inspect/load with configuration and input-artifact fingerprints
- Optional exact training-resume persistence without model-only Training dependency
- Corruption, recovery, and cross-extension integration checkpoint

## Current status

Draft planning only. Current Runtime publication leases representations privately and exposes no
public value access, so model bytes cannot yet be obtained through the architecture-approved
lifecycle. No planning text authorizes direct backend access as a workaround.

## Open questions

- Select the first bounded manifest and Tensor-payload encoding, byte order, alignment, and
  compression policy.
- Define Engine materialized-host-payload ownership and whether compatible already-host-backed
  values may take a proven direct path.
- Define the cross-owner validate/install transaction protocol required for exact training resume.
- Select overwrite/version-retention policy and crash-durability guarantees for supported
  filesystems.
- Decide when a closed built-in declarative model descriptor has a concrete consumer; arbitrary
  lambdas remain code-defined regardless.

## Decisions made

- `StateDictionary` remains in-memory exact Tensor references, not checkpoint bytes.
- Checkpoint materializes only through an explicit Engine/publication boundary and never imports a
  concrete backend or reads device storage directly.
- Model and Training checkpoints are distinct. TrainingCheckpoint is a semantic superset but lives
  in an optional downstream adapter so model-only use does not force Training.
- Text persists the tokenizer artifact. Checkpoint records and verifies its fingerprint through a
  neutral input-artifact boundary without parsing text formats or depending on Text.
- Arbitrary `Model.define` lambdas are recreated by code/config and strict-loaded; they are not
  serialized.
- Compiled, prepared, Runtime, backend, device, and tuning artifacts are rebuilt, not checkpointed.
- ONNX remains a separate model-interchange boundary; it neither replaces this strict NN state
  schema nor carries the complete Training exact-resume contract.
- Complete artifact and target validation precedes any installation; durable save uses a validated
  temporary target plus manifest/checksums and rename.

## Risks

- Reading Tensor host storage directly could save stale or absent data when the current value is
  backend-resident and would violate NN/Runtime/backend ownership.
- One checkpoint module depending on Training and Text would make optional consumers pay unrelated
  dependencies and encourage a god facade.
- Serializing a Java lambda/class name could be insecure, non-portable, and unable to reproduce
  captured configuration.
- Comparing only vocabulary size could accept a tokenizer that maps IDs to different tokens and
  silently reinterpret every Embedding row.
- Installing model state before optimizer/session validation could leave a partially resumed
  process after a later failure.
- Unbounded manifest fields, Shapes, token-linked metadata, or payload sizes could allocate before
  checksum/schema rejection.
- Rename and flush behavior varies by filesystem; claiming universal crash atomicity without an
  explicit support/failure contract would be misleading.

## Notes

This master plan is non-authoritative future coordination. The architecture contract wins.
