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
  -> file-flushed atomic live publication

Training-owned optimizer/session snapshot
  + validated model-checkpoint candidate
  -> optional Training Checkpoint adapter
  -> exact-resume artifact
```

The opaque artifact, internal detached model candidate, and public model save/inspect/load are
Complete, including direct Engine execution of restored heap Tensors; training flows remain
future plans. [Checkpoint 0001](tasks/0001-checkpoint-module-boundary.md)
completed the model-only boundary and initially empty Gradle project from base `81d50901`, after focused
validation, independent Class C review, and the repository test/Javadoc gate.
The optional Training Checkpoint adapter is neither included nor implemented.
[Checkpoint 0003](tasks/0003-model-schema-and-materialization-handoff.md) is Complete after its
22-test module check, Javadoc gate, and independent Class C review. Its scope is an internal bounded
model candidate and Engine handoff, not public save/load. [0004](tasks/0004-public-model-checkpoint-save-inspect-load.md)
is Complete after CPU 0024, direct restored-heap Engine acceptance, targeted documentation, and
the combined validation gate. Tasks 0005–0007 remain Draft and unauthorized.

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
  deterministic names, and file-flushed atomic live-publication policy;
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

The adopted model-only boundary and planned optional downstream adapter use two artifacts rather
than one dependency-heavy module once model-state persistence is implemented:

```text
modules/model -> extensions/nn
compiler/runtime/prepare/backends -> modules/engine

extensions/nn + modules/engine + modules/model
  -> extensions/checkpoint

extensions/training + extensions/checkpoint
  -> optional extensions/training-checkpoint
```

`extensions/checkpoint` owns the bounded opaque envelope, internal model-candidate schema, and
public `ModelCheckpoint` save/inspect/load facade. It depends on NN for stable state paths,
Engine for public host materialization, and Model for typed public inspection descriptors. It
does not depend on Training or Text, so
inference/model-only users do not acquire those extensions. The envelope forces file contents and
provides atomic live visibility on a successful move, not portable rename crash durability.

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

The model-only NN/Engine/Model-to-Checkpoint edges are in the root contract, Gradle build, and
architecture test after independent review. The optional adapter edge is reserved in the
contract but its project remains a later, separately gated implementation.

## Package structure

The model-only packages exist; the optional Training Checkpoint package remains planned:

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

The model checkpoint format contains:

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
evaluation, copy, backend transfer, or serialization. Engine exposes bounded detached host
materialization through `RunResult.materialize` and `Engine.compute`; Checkpoint uses the public
Engine boundary for state selection and materialization, never a backend representation.

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
5. Validate the complete path/kind/type/Shape set against a fully bound recreated topology;
   version-one load rejects an unbound/lazy target. A later NN lazy-binding contract would be
   required to admit one.
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

The current artifact save creates a sibling temporary file in a trusted local directory.
It writes the header and manifest before the contiguous bounded payloads, forces the temporary
file through its writing channel, closes and reopens it, and fully validates it before one
`ATOMIC_MOVE` to the final target. A pre-existing regular target may be replaced. A symlink or
directory target is refused before work; unsupported atomic replacement fails without a weaker
fallback. A failed move can leave the target in a provider-dependent state. Successful publication
gives atomic live visibility, not portable power-loss durability of the rename. A later training
save must retain this envelope policy.

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
| [0002](tasks/0002-bounded-artifact-envelope.md) | Bounded artifact envelope | Complete | 0001 Complete; corrected v1 policy in brief | Opaque file API, bounds, SHA-256, file flush and atomic live publication implemented; 15 checkpoint tests, Javadoc, architecture test, and independent Class C review passed. |
| [0003](tasks/0003-model-schema-and-materialization-handoff.md) | Model schema and Engine materialization handoff | Complete | 0002 Complete; existing NN export and Engine compute | Internal fixed-ID schema, detached candidate, and bounded public Engine handoff implemented; 22 module tests, Javadoc, and independent Class C review passed. No public save/load. |
| [0004](tasks/0004-public-model-checkpoint-save-inspect-load.md) | Model checkpoint save, inspect, and strict load | Complete | 0003; [CPU 0024](../../backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md) for direct restored-heap execution | Public save/inspect/load and direct restored-heap Engine regression passed, with independent review, documentation, and combined validation. |
| 0005 | Training-owned exact-resume snapshot prerequisite | Draft | Training 0004; Checkpoint 0004 | Consume stabilized optimizer/session and caller-state handoff plus cross-owner preflight/commit contracts; identify which concrete caller-owned participant schemas have owner validators/restorers. Training 0004 does not depend on this adapter. |
| 0006 | Optional Training Checkpoint adapter | Draft | Architecture authorization for the optional project; 0002, 0004–0005; supported caller-state schemas/codecs | Encode and restore only the model, Training state, effective objective contexts, and caller-owned participants with versioned bounded codecs and owner validation; reject an exact artifact for any unsupported affecting state. Model-only Checkpoint stays independent of Training. |
| 0007 | Checkpoint capability and recovery checkpoint | Draft | 0004; 0006 when exact training resume is selected; Text 0003 for text integration | Validate corruption/resource bounds, atomic publication and interruption recovery, strict no-partial-install behavior, text fingerprint mismatch, model-only load, exact training resume, documentation, and dependency enforcement. |

Checkpoint 0001 has made the explicit architecture decision. Checkpoint 0002 passed independent
review; its corrected policy and compact brief were committed at serial base `7261d11d`. It conflicts
with concurrent edits to the artifact format or Checkpoint contract, has no parallel group, and
precedes 0003. The 0002 implementation owner owns shared documents and runs
`./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc`; no second integration
suite is planned unless review changes executable behavior. The optional adapter remains Draft.
Model 0032's numerical cutover and Training 0002B's
in-memory Adam/AdamW snapshot capability are Complete; neither creates durable checkpoint APIs.

The cross-area order is acyclic: CPU 0022 → CPU 0024 → Checkpoint 0004 closure; accepted
Training 0003A and completed Checkpoint 0004 then precede Training 0004; Checkpoint 0005
precedes adapter 0006. CPU 0023 is an independent Draft, conflicts with CPU 0024, and is not a
prerequisite. Training 0003 is Superseded. The
adapter consumes the Training handoff, never implements its prerequisite.

## Planned end-to-end flows

The text and training flows below are conceptual. The model checkpoint API and direct
restored-heap Engine execution are implemented and validated.

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
- Bounded artifact and file-flushed atomic-publication foundation
- Engine-owned host materialization used by model checkpointing
- Strict model save/inspect/load with configuration and input-artifact fingerprints
- Optional exact training-resume persistence without model-only Training dependency
- Corruption, recovery, and cross-extension integration checkpoint

## Current status

Model-only boundary, bounded artifact, internal candidate and Engine handoff, and public
save/inspect/load are Complete after focused validation and independent Class C reviews. The
target-defined alias and fail-closed unbound-target policy is implemented in
[0004](tasks/0004-public-model-checkpoint-save-inspect-load.md). Direct restored-heap Engine
execution and the combined repository gate passed after
[CPU 0024](../../backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md).
Training snapshots remain in-memory values, not a durable cross-owner exact-resume format.

## Open questions

- Decide whether a later schema should record source alias identity or support lazy targets.
  Version one uses the fully bound recreated target's alias topology and rejects lazy targets;
  opaque artifact bytes alone cannot establish either missing contract.
- Implement and test Training 0004's cross-owner preflight/commit or rollback gate before an
  optional adapter claims exact, atomic recovery.
- Version retention beyond one explicitly replaceable target and crash-durable directory-entry
  publication remain later product decisions. Version 1 provides file flush and atomic live
  visibility, not a cross-platform power-loss guarantee or a non-atomic fallback.
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
- Complete artifact and target validation precedes any installation; artifact save uses a validated
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
