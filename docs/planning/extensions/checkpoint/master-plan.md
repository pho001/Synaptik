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

The artifact flows above remain future plans. [Checkpoint 0001](tasks/0001-checkpoint-module-boundary.md)
completed the model-only boundary and empty Gradle project from base `81d50901`, after focused
validation, independent Class C review, and the repository test/Javadoc gate.
The optional Training Checkpoint adapter is neither included nor implemented. No successor is
Ready: Checkpoint 0002–0006 remain Draft until their own decisions and briefs are prepared.

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

The adopted empty model-only boundary and planned optional downstream adapter use two artifacts
rather than one dependency-heavy module once persistence is implemented:

```text
modules/model -> extensions/nn
compiler/runtime/prepare/backends -> modules/engine

extensions/nn + modules/engine
  -> extensions/checkpoint

extensions/training + extensions/checkpoint
  -> optional extensions/training-checkpoint
```

`extensions/checkpoint` will own the common durable envelope and `ModelCheckpoint`. It depends on NN
for stable state paths and on Engine for a public host-materialization boundary. It does not depend
on Training or Text, so inference/model-only users do not acquire those extensions.

The optional `extensions/training-checkpoint` will own `TrainingCheckpoint` persistence and depend on
Training plus Checkpoint. Training owns the meaning and validation of optimizer/session state;
the caller owns next-input progress, external random-number-generator (RNG) state, sampler cursor,
and any scheduler/scaler state under its own explicit schema. Model owns graph RNG semantics, while
the current fixed Training session has no per-run graph-RNG handoff. The adapter may encode only a
participant set whose owners supply validation and coordinated installation. Training does not
gain file I/O, and Checkpoint does not import optimizer concepts into its model-only core.

Text remains independent. It writes and loads its own complete tokenizer artifact. A text
application supplies that artifact's canonical fingerprint as the model checkpoint's input-artifact
fingerprint. Strict load compares the exact fingerprint, so two equal-size vocabularies with
different ID mappings fail compatibility without a Checkpoint-to-Text dependency.

The model-only NN/Engine-to-Checkpoint edges are now in the root contract, Gradle build, and
architecture test, after independent review. The optional adapter edge is reserved in the
contract but its project remains a later, separately gated implementation. A later public
Checkpoint API that names Model types may justify a direct Model edge; no such edge exists now.

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
evaluation, copy, backend transfer, or serialization. Engine already exposes bounded detached
host materialization through `RunResult.materialize` and `Engine.compute`; Checkpoint must define
the complete state-selection and ownership handoff over those APIs rather than invent backend
access. Checkpoint never casts or reads a backend representation.

## Training checkpoint contract

The planned `TrainingCheckpoint` is a strict superset of a compatible model checkpoint. An exact
resume claim is conditional on every state owner selected by the Training contract:

- optimizer type/configuration, parameter groups, stable parameter-path association, and every
  optimizer slot such as Adam moments;
- Training execution/optimizer-step and accumulation counters, plus caller-owned epoch and next
  input/sampler cursor;
- graph/dropout RNG construction state only for a supported fixed graph; a changing per-step
  stream has no current Training handoff;
- relevant per-module modes and the effective forward contexts actually supplied while
  constructing the fixed objective, recorded separately; NN buffers require a proved step-aligned
  detached capture boundary (training-mode BatchNorm is not currently eligible);
- external data-shuffle/augmentation RNG state when used;
- caller-owned scheduler and mixed-precision scaler state when present; and
- the same model configuration and input-artifact fingerprints.

If a data source cannot expose a stable sampler cursor, the artifact may support a documented
restart mode but must not call it exact resume. Compiled/prepared/backend/device artifacts are
never included; they are reconstructed for the current Engine and target after restore.
The optional adapter cannot encode arbitrary caller-owned RNG, sampler, scheduler, or scaler
objects. Each included participant needs an explicitly supported versioned, bounded schema and
codec, owner validator/restorer, and mismatch/refusal tests. An exact artifact is refused if
any affecting participant lacks that support; a distinct restart/partial artifact may omit it.

## Validation, installation, and publication

Load is staged and fail-closed:

1. Bound file and aggregate sizes before allocating payloads.
2. Parse the manifest with duplicate/path-traversal rejection and supported version checks.
3. Verify every declared payload size and checksum before constructing a candidate.
4. Verify model schema/config and any supplied input-artifact fingerprint.
5. Validate the complete path/kind/type/Shape set against the recreated topology; a deferred model
   may bind from checkpoint Shapes only under the later NN lazy-binding contract.
6. For training resume, jointly prevalidate model, optimizer/session, effective forward contexts,
   fixed-objective construction, next-input, RNG, sampler, scheduler, and scaler candidates before
   any live installation; refuse unproved step-aligned buffer values and unsupported caller-state
   schemas/codecs;
   model parameter payloads must match Training logical bytes at each stable path.
7. Training 0004 must supply the missing cross-owner preflight/commit protocol. On an isolated
   unpublished replacement tree, install NN bindings and modes before constructing the objective
   and opening/restoring a replacement `TrainingSession`; publish the complete candidate only on
   success, or discard it on ordinary failure. An in-place alternative needs complete rollback.
   Today's per-owner APIs do not supply one atomic transaction. No backend storage bypass or
   partial permissive mode is implied.

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
| [0001](tasks/0001-checkpoint-module-boundary.md) | Model Checkpoint module boundary | Complete | None; architecture decision is the deliverable | Empty project, contract, ADR, explanations, and dependency test reviewed; focused and repository gates passed. No persistence API or optional adapter. |
| 0002 | Bounded durable artifact envelope | Draft | 0001; selected codec/filesystem policy | Define deterministic versioned manifest/payload/checksum types, resource bounds, temporary-target publication, durable flush, reopen validation, rename semantics, and corruption rejection without model/training concepts. |
| 0003 | Model schema and Engine materialization handoff | Draft | 0002; existing Engine host materialization; NN state/lazy-binding contracts | Define stable model state/config/input-artifact fingerprints and convert a complete NN state snapshot to bounded host payloads only through Engine publication/materialization. Existing Engine capability is available, but this adapter is not. |
| 0004 | Model checkpoint save, inspect, and strict load | Draft | 0003 | Persist materialized parameters/buffers and validate the entire artifact plus recreated topology before one strict NN installation; never serialize arbitrary `Model.define` lambdas. |
| 0005 | Training-owned exact-resume snapshot prerequisite | Draft | Training 0004; Checkpoint 0004 | Consume stabilized optimizer/session and caller-state handoff plus cross-owner preflight/commit contracts; identify which concrete caller-owned participant schemas have owner validators/restorers. Training 0004 does not depend on this adapter. |
| 0006 | Optional Training Checkpoint adapter | Draft | Architecture authorization for the optional project; 0002, 0004–0005; supported caller-state schemas/codecs | Encode and restore only the model, Training state, effective objective contexts, and caller-owned participants with versioned bounded codecs and owner validation; reject an exact artifact for any unsupported affecting state. Model-only Checkpoint stays independent of Training. |
| 0007 | Checkpoint capability and recovery checkpoint | Draft | 0004; 0006 when exact training resume is selected; Text 0003 for text integration | Validate corruption/resource bounds, atomic publication and interruption recovery, strict no-partial-install behavior, text fingerprint mismatch, model-only load, exact training resume, documentation, and dependency enforcement. |

Checkpoint 0001 has made the explicit architecture decision. There is no new authorized frontier:
0002 needs its codec/filesystem policy and a Ready brief; the optional adapter remains Draft.
Model 0032's numerical cutover and Training 0002B's
in-memory Adam/AdamW snapshot capability are Complete; neither creates durable checkpoint APIs.

The cross-area order is acyclic: accepted Training 0003A and Checkpoint 0001–0004 precede
Training 0004; then Checkpoint 0005 precedes adapter 0006. Training 0003 is Superseded. The
adapter consumes the Training handoff, never implements its prerequisite.

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
  -> parse/checksum/validate the complete TrainingCheckpoint
  -> recreate an isolated NN tree and only supported caller-owned progress/random/scheduler/scaler candidates
  -> jointly prevalidate all artifact, versioned owner-schema, effective forward-context, and fixed-objective inputs
  -> install NN bindings/modes on that unpublished tree
  -> construct the objective with the recorded contexts from final NN Tensors, then open and restore TrainingSession
  -> publish the complete candidate only after successful owner validation
  -> continue from recorded progress/RNG/sampler state
```

This is a future protocol, not a runnable current API or an atomicity guarantee from today's
`loadStateDictionary` and `TrainingSession.restore`.

## Milestones

- Accepted checkpoint module/dependency architecture
- Bounded durable artifact and atomic-publication foundation
- Engine-owned host materialization used by model checkpointing
- Strict model save/inspect/load with configuration and input-artifact fingerprints
- Optional exact training-resume persistence without model-only Training dependency
- Corruption, recovery, and cross-extension integration checkpoint

## Current status

Model-only module-boundary implementation is Complete after focused and repository validation.
Runtime publication still leases representations privately, while Engine already provides the
architecture-approved host-materialization boundary. Missing work is the bounded
codec, complete model-state handoff, and coordinated durable restore—not access to bytes through
Engine. Training snapshots are in-memory values, not a durable, cross-owner exact-resume format.

## Open questions

- Select the first bounded manifest and Tensor-payload encoding, byte order, alignment, and
  compression policy.
- Choose how the checkpoint adapter uses Engine's existing detached host values within aggregate
  resource bounds; do not change their ownership or introduce backend shortcuts implicitly.
- Implement and test Training 0004's cross-owner preflight/commit or rollback gate before an
  optional adapter claims exact, atomic recovery.
- Select overwrite/version-retention policy and crash-durability guarantees for supported
  filesystems.
- Decide when a closed built-in declarative model descriptor has a concrete consumer; arbitrary
  lambdas remain code-defined regardless.

## Decisions made

- `StateDictionary` remains in-memory exact Tensor references, not checkpoint bytes.
- [ADR 0030](../../../design/decisions/0030-exact-training-resume-state-ownership.md) assigns
  caller next-input/external state outside Training and refuses a changing graph-RNG stream with
  the current fixed session; no Training 0003 public API is needed.
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
