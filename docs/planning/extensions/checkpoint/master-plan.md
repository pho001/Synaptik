# Checkpoint Extension Master Plan

## Goal

Plan durable, versioned model checkpoints without turning NN state references into backend storage
access or forcing Training and Text dependencies on model-only users. Plan stored model and
TrainingState recovery as a separate optional downstream adapter over Training-owned snapshots;
external state remains application-owned, not automatic exact continuation.

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
  -> model + TrainingState artifact (external state remains application-owned)
```

The opaque artifact, internal detached model candidate, and public model save/inspect/load are
Complete, including direct Engine execution of restored heap Tensors. Training 0004's bounded
in-memory private-candidate handoff is proved; durable training flows remain future plans.
[Checkpoint 0001](tasks/0001-checkpoint-module-boundary.md)
completed the model-only boundary and initially empty Gradle project from base `81d50901`, after focused
validation, independent Class C review, and the repository test/Javadoc gate.
The optional Training Checkpoint adapter is neither included nor implemented.
[Checkpoint 0003](tasks/0003-model-schema-and-materialization-handoff.md) is Complete after its
22-test module check, Javadoc gate, and independent Class C review. Its scope is an internal bounded
model candidate and Engine handoff, not public save/load. [0004](tasks/0004-public-model-checkpoint-save-inspect-load.md)
is Complete after CPU 0024, direct restored-heap Engine acceptance, targeted documentation, and
the combined validation gate. [0004A](tasks/0004a-direct-leaf-model-save.md) is Complete with bounded
direct leaf-state capture and workaround removal. User-authorized
[0005](tasks/0005-training-state-reconstruction.md) is Complete: defensive reconstruction,
external-package tests, final Javadoc/example validation, and independent API review passed;
0006–0007 remain Draft and unauthorized.

## Dependency DAG and authorized frontier

Completed Checkpoint 0001 → 0002 → 0003 → 0004, with completed CPU 0024 → 0004.
Checkpoint 0004, CPU 0024, and Training 0004 → completed Checkpoint 0004A.
Training 0004 and Checkpoint 0004A → completed Training-owned 0005 → Checkpoint 0005A → Draft 0006;
0007 also awaits
the selected adapter/text gates. The user selected Model + TrainingState for 0006; external state
remains application-owned. The model-envelope composition boundary is approved in
[ADR 0036](../../../design/decisions/0036-model-envelope-composition.md); the adapter codec,
composition protocol, and build gate remain pending.
These edges do not authorize the Draft successors.

Authorized frontier: [0005A](tasks/0005a-model-envelope-composition.md), Ready after independent
contract/brief APPROVE and dependency/base/conflict verification. Coordinator acceptance/integration
and a recorded integrated launch revision remain required; no implementation is launched here.
[0005](tasks/0005-training-state-reconstruction.md) is Complete;
its implementation belongs to `extensions/training`, not Checkpoint.
[0004A](tasks/0004a-direct-leaf-model-save.md) closed after clean
implementation, independent targeted review, and the configured coordinator gate. Capture remains
Checkpoint-owned through existing public Engine compute; generic zero-node Engine/Prepare support
is excluded. Evidence is in the completed briefs; closure does not authorize 0006.

0005 closed serially (parallel group None) after independent review and affected-Training
validation. The Training/Checkpoint coordinator owns shared-document integration; the brief
records evidence. 0005A's approved boundary extends the existing ModelCheckpoint over
ArtifactEnvelope; the brief records serial integration metadata and preparation verification.
0006 stays Draft, absent from the build, and unimplemented.

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
the current fixed Training session has no per-run graph-RNG handoff. The selected first adapter
persists model + TrainingState only; it does not encode caller-owned participants. Training does not
gain file I/O, and Checkpoint does not import optimizer concepts into its model-only core.

Text remains independent. It writes and loads its own complete tokenizer artifact. A text
application supplies that artifact's canonical fingerprint as the model checkpoint's input-artifact
fingerprint. Strict load compares the exact fingerprint, so two equal-size vocabularies with
different ID mappings fail compatibility without a Checkpoint-to-Text dependency.

The model-only NN/Engine/Model-to-Checkpoint edges are in the root contract, Gradle build, and
architecture test after independent review. The optional adapter edge is reserved in the
contract but its project remains a later, separately gated implementation.

## Package structure

The model-only packages exist; completed 0005 exposes existing values in Training's own package. The
optional Training Checkpoint package remains planned:

```text
io.github.pho001.synaptik.checkpoint/
  artifact/  bounded manifest, version, checksum, payload, and atomic-publication contracts
  model/     ModelCheckpoint schema, materialization handoff, strict save/inspect/load

io.github.pho001.synaptik.training/
  TrainingState and nested ParameterState defensive public construction (0005)

io.github.pho001.synaptik.training.checkpoint/
  model + TrainingState schema and stored-state save/inspect/load adapter (design pending)
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

User decision (2026-10-14): the first durable `TrainingCheckpoint` persists Model + TrainingState.
The planned stored content is:

- compatible model state and its configuration/input-artifact fingerprints;
- exact Training optimizer configuration and ordered groups/parameter schema;
- logical/master parameter bytes and optimizer slots, including adaptive moments; and
- pending gradient accumulation and execution/optimizer-step/accumulation counters.

The guarantee is restoration of stored state, not complete automatic exact continuation. Data
position (including epoch/sampler cursor), external RNG, scheduler/scaler state, and any other
caller-owned state absent from the artifact remain application-owned. Module modes, effective
forward contexts, and objective construction inputs are not implicitly supplied by TrainingState;
the application must provide what its computation requires. Compiled/prepared/backend/device
artifacts are rebuilt, not stored. Concrete composition, bounded versioned codecs, validation,
and architecture design remain to be specified before 0006 becomes Ready.

Any application-level exact-continuation claim still requires all affecting state under the
existing Training lifecycle contract. The selected artifact alone does not establish that claim,
a changing per-step graph-RNG stream, or step-aligned training-mode BatchNorm buffer capture.
No caller-state codecs or automatic caller-state restoration are selected for this first adapter.

## Validation, installation, and publication

Load is staged and fail-closed:

1. Bound file and aggregate sizes before allocating payloads.
2. Parse the manifest with duplicate/path-traversal rejection and supported version checks.
3. Verify every declared payload size and checksum before constructing a candidate.
4. Verify model schema/config and any supplied input-artifact fingerprint.
5. Validate the complete path/kind/type/Shape set against a fully bound recreated topology;
   version-one load rejects an unbound/lazy target. A later NN lazy-binding contract would be
   required to admit one.
6. For the selected model + TrainingState adapter, validate all stored model and Training facts
   before changing published state. The application supplies objective/context inputs and owns
   external state; artifact load does not validate or restore absent caller-owned participants.
   Any wider exact-continuation claim remains subject to the Training lifecycle contract.
   Current file inspection does not expose parameter value bytes. Approved but unimplemented
   0005A provides independently validated canonical envelope bytes without trusting inspection.
7. Training 0004 has proved a bounded same-route in-memory instance of the caller-coordinated
   private-candidate protocol selected by
   [ADR 0035](../../../design/decisions/0035-private-candidate-training-restore.md). Load NN
   bindings and modes only on a distinct unpublished tree, copy parameters to native storage when
   needed, construct the objective from final Tensors, and open a private session. Compare the
   candidate's canonical logical parameter bytes with Training's at every stable path before
   restore/publication; `TrainingSession.state()` can provide those candidate bytes after open.
   Let `TrainingSession.restore` validate its session-specific state, then publish the complete
   candidate and staged caller state under exclusion. Ordinary failure discards the candidate.
   No pure pre-open validator, generic transaction, or backend-storage bypass is required.

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
| [0004A](tasks/0004a-direct-leaf-model-save.md) | Direct leaf-model save | Complete | Checkpoint 0004; CPU 0024; Training 0004, all Complete | Exact native/heap leaf capture, direct trained save/load and loaded re-save passed; workaround removed, independent review and configured coordinator gate passed. |
| [0005](tasks/0005-training-state-reconstruction.md) | Training-owned detached state reconstruction prerequisite | Complete | Training 0004; Checkpoint 0004A, both Complete | Three defensive constructors exposed; external-package tests, final Javadoc/example validation, and independent API review passed. No codec. |
| [0005A](tasks/0005a-model-envelope-composition.md) | Model-envelope composition | Ready | 0004A and 0005 Complete; independent contract/brief APPROVE | Verified serial frontier: existing ModelCheckpoint envelope operations. Not launched; integrate the approved packet and record the launch revision first. |
| 0006 | Optional Training Checkpoint adapter | Draft | Architecture authorization for the optional project; 0002, 0004–0005A; concrete adapter composition/codec design | Selected Model + TrainingState persistence restores stored state; data position/external RNG and other absent caller-owned state remain application-owned. No complete automatic exact-continuation claim or execution authorization. Model-only Checkpoint stays independent of Training. |
| 0007 | Checkpoint capability and recovery checkpoint | Draft | 0004A; 0006 for selected stored training-state recovery; Text 0003 for text integration | Validate corruption/resource bounds, atomic publication and interruption recovery, strict no-partial-install behavior, text fingerprint mismatch, model-only load, stored model + TrainingState recovery, documentation, and dependency enforcement. |

Checkpoint 0001 has made the explicit architecture decision. Checkpoint 0002 passed independent
review; its corrected policy and compact brief were committed at serial base `7261d11d`. It conflicts
with concurrent edits to the artifact format or Checkpoint contract, has no parallel group, and
precedes 0003. The 0002 implementation owner owns shared documents and runs
`./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc`; no second integration
suite is planned unless review changes executable behavior. The optional adapter remains Draft.
Model 0032's numerical cutover and Training 0002B's
in-memory Adam/AdamW snapshot capability are Complete; neither creates durable checkpoint APIs.

The cross-area order is acyclic: CPU 0022 → CPU 0024 → Checkpoint 0004 closure; accepted
Training 0003A and completed Checkpoint 0004 then precede Training 0004; Checkpoint 0004A repairs
direct save before Checkpoint 0005, which precedes 0005A and then adapter 0006.
CPU 0023 is an independent Draft,
conflicts with completed CPU 0024, and is not a
prerequisite. Training 0003 is Superseded. The
adapter consumes the Training handoff, never implements its prerequisite.

## Planned end-to-end flows

The text and durable-training flows below are conceptual. The model checkpoint API and direct
restored-heap Engine execution are implemented and validated; Training 0004 proves only a bounded
in-memory private-candidate handoff.

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

Restore stored model + TrainingState (conceptual; concrete design pending):

```text
load tokenizer artifact and verify fingerprint
  -> parse/checksum/validate the complete TrainingCheckpoint
  -> recreate an isolated NN tree; application supplies modes, contexts, and objective inputs
  -> validate stored model + TrainingState facts
  -> load NN bindings on that unpublished tree; copy parameters to native storage when needed
  -> construct the objective from final NN Tensors; open a private session
  -> compare canonical parameter bytes by path; restore TrainingSession
  -> publish the complete candidate only after successful owner validation
  -> application separately manages data position, external RNG, and other absent external state
```

This is a future protocol, not a runnable current API or an atomicity guarantee from today's
`loadStateDictionary` and `TrainingSession.restore`; publication remains caller-controlled.
It restores stored state, not complete automatic exact continuation.

## Milestones

- Accepted checkpoint module/dependency architecture
- Bounded artifact and file-flushed atomic-publication foundation
- Engine-owned host materialization used by model checkpointing
- Strict model save/inspect/load with configuration and input-artifact fingerprints
- Optional model + TrainingState persistence without model-only Training dependency
- Corruption, recovery, and cross-extension integration checkpoint

## Current status

Model-only boundary, bounded artifact, internal candidate and Engine handoff, and public
save/inspect/load are Complete after focused validation and independent Class C reviews. The
target-defined alias and fail-closed unbound-target policy is implemented in
[0004](tasks/0004-public-model-checkpoint-save-inspect-load.md). Direct restored-heap Engine
execution and the combined repository gate passed after
[CPU 0024](../../backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md).
Training 0004's same-route in-memory private-candidate handoff passed independent Class C review
and the integration-owner repository gate. Training snapshots remain in-memory values, not a
durable cross-owner exact-resume format. The user-authorized Training-owned
[0005](tasks/0005-training-state-reconstruction.md) reconstruction prerequisite is Complete;
0006–0007 remain Draft; 0006 product scope is Model + TrainingState with external state
application-owned. The model-envelope boundary and brief passed independent review; 0005A is Ready,
not launched, pending coordinator acceptance/integration. Adapter composition/codec/build
authorization remains pending. Complete
[0004A](tasks/0004a-direct-leaf-model-save.md) removes the private expression-view workaround:
ordinary save captures the actual quiescent trained tree and directly re-saves loaded heap state
on the validated CPU route. Checkpoint selects raw-preserving leaf requests internally without
changing NN bindings, original descriptors, or v1 encoding. Independent review and the configured
coordinator gate passed; no generic zero-node Engine/Prepare capability or new frontier is implied.

## Open questions

- Decide whether a later schema should record source alias identity or support lazy targets.
  Version one uses the fully bound recreated target's alias topology and rejects lazy targets;
  opaque artifact bytes alone cannot establish either missing contract.
- Design 0006's concrete model + TrainingState composition, bounded codecs, and architecture
  before making it Ready. Caller-state persistence is excluded from the selected first adapter;
  any future broader exact-recovery product would need a separate scope/design decision.
- Implement the approved existing-ArtifactEnvelope boundary in
  [0005A](tasks/0005a-model-envelope-composition.md) after coordinator acceptance/integration.
  It selects no training wire format and does not authorize adapter 0006 inclusion.
- Version retention beyond one explicitly replaceable target and crash-durable directory-entry
  publication remain later product decisions. Version 1 provides file flush and atomic live
  visibility, not a cross-platform power-loss guarantee or a non-atomic fallback.
- Decide when a closed built-in declarative model descriptor has a concrete consumer; arbitrary
  lambdas remain code-defined regardless.

## Decisions made

- [ADR 0036](../../../design/decisions/0036-model-envelope-composition.md) approves planned
  ModelCheckpoint envelope capture/inspection/canonical bytes/load, reusing current internal
  machinery without new production types, wire changes, or dependency edges.
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
