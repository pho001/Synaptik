# Task 0006: Whole-model TrainingCheckpoint save, inference load and resume

## Status

Draft — revised primary product, not implementation authorization.
Replaces the snapshot-only direction of withdrawn Proposed ADR 0037 with Proposed ADR 0038.
Support matrix, owner bridges, default limits and exact wire decisions must stabilize before Ready.

## Change class

Class C implementation — optional downstream module, persistent kind and cross-owner recovery.
This planning revision changes no normative contract, build, Java or test.

## Goal

Save one supported stored model + Training-owned recipe + TrainingState, then load a usable model
for inference or resume a Training-owned run automatically. No routine lambda, model reconstruction,
ArtifactEnvelope, config digest or Arena plumbing. Retain explicit advanced APIs.

## Scope and non-goals

Optional `extensions/training-checkpoint` has exactly two direct public project edges:
`api(project(":extensions:training"))` and `api(project(":extensions:checkpoint"))`.
Use published NN/Model/Engine transitively; no reverse or lower/backend edge.
Training owns recipe semantics, objective construction, batch binding and run lifecycle; Checkpoint
owns codecs/file publication; NN owns model definition and forward reconstruction.

No arbitrary Java serialization/callback capture/reflection/class-name factory/registry, general
transactions/participants, ModelExecutor, automatic backend discovery, saved device/compiled
formats or new pure Training validator. Exclude caller cursor/epoch/external RNG/scheduler/scaler.
Restore stored state, not complete automatic exact continuation, changing dropout stream,
BatchNorm training-buffer alignment, BPTT or Conv3d gradients.

## Contracts

- [Root authority](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning scoped headings: [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [extensions/training](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint),
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle).
- [Proposed ADR 0038](../../../../design/decisions/0038-declarative-model-and-training-resume.md).

The current contracts do not authorize the new product. Before Ready, coordinate root + owning
extensions contract + affected architecture explanations + ADR + architecture tests as applicable.
Stop on missing/ambiguous contracts; this proposal does not override them.

## Dependencies and integration

- Depends on: NN 0026; Checkpoint 0005B; Training 0006; completed Checkpoint 0002/0004/0004A/
  0005/0005A and Training 0004. New predecessors are Draft and NOT satisfied.
- Conflicts with: adapter wire/API/build, owning contracts and shared NN/Training/Checkpoint docs.
- Parallel group: None; separate clean implementation contexts, serialized contract cutover.
- Common base revision: planning base `52733fcfd32d278b651302e308487e0b6c6afa81`; refresh before Ready.
- Integration order: approve/cut over contracts → NN 0026 → Checkpoint 0005B + Training 0006
  → adapter 0006 → independent review → coordinator acceptance. No mechanical task split.
- Integration validation: once-only source-matched repository checkpoint specified below.
- Shared-document integration owner: NN/Training/Checkpoint coordinator.
- Authorized frontier: None. No implementation or build inclusion is authorized by this brief.

## Proposed types and high-level API

In `io.github.pho001.synaptik.training.checkpoint`: public final `TrainingCheckpoint` and nested
immutable advanced `Snapshot`; private `TrainingArtifactCodec`, `TrainingArtifactComposition`
and `TrainingParameterMatch`. Reuse TrainingRecipe/TrainingRun and ArtifactEnvelope, no live image
wrapper or duplicate validation credential. Final signature freeze is a Ready condition.

Conceptual, not existing APIs:

```text
TrainingCheckpoint.save(Path, TrainingRun): void throws IOException
TrainingCheckpoint.loadModel(Path): DeclarativeModel throws IOException
TrainingCheckpoint.resume(Path, Engine): TrainingRun throws IOException
TrainingCheckpoint.read(Path, ArtifactLimits): Snapshot throws IOException
```

Typed expected-input-identity and ArtifactLimits overloads serve advanced callers without generic
LoadOptions. Short overloads require absent input digest; present fingerprint rejects until supplied
expected identity is verified. No automatic tokenizer load (Text remains Draft).
Advanced detached save/read/model-component/TrainingState access and explicit target/session
restore remain available; do not turn successful read into a restorable-session certificate.
Existing published ArtifactFile and ModelCheckpoint explicit APIs are unchanged.

## Stored model and recipe

Initial proposal is NN 0026's Tensor-to-Tensor nested Sequential with Linear, LayerNorm, Embedding,
Conv1d/2d/3d and closed RELU/TANH/SIGMOID steps; exact matrix remains a decision before Ready.
Capture actual attrs and wiring, state descriptors and explicit sharing; never infer ModelForward
captures. NN input constraints need not freeze inference batch size. Training recipe separately
records STATIC dense/offset-zero input/target dtype/Shapes, roles/binding meaning, selected loss/
options/reduction and modes/effective contexts/gradient mode. Scalar objective is FLOAT32/FLOAT64;
low predictions cast explicitly as the approved loss requires. Saved optimizer alone is insufficient.
TrainingState alone stores optimizer/default/groups; do not duplicate them in recipe encoding.
Unknown custom/generic record models use existing explicit reconstruction, not routine resume.

Inference sharing must reconstruct source identity or reject it explicitly. Training rejects repeated
parameter Tensor identities across paths/overlapping regions; never silently duplicate shared
weights. A uniquely owned layer called multiple times is a different execution-DAG case.
Current Training capability, compiler/backend gradients and context limits remain fail-closed.

## Proposed canonical wire and retained foundation

Use new kind `training-model.v1` over UNCHANGED ArtifactFile v1 framing/publication. Prefix a
validated `whole-model.v1` component under `whole-model/`, plus `recipe/schema` and Training
schema/slot entries. Checkpoint 0005B defines whole-model canonical definition bytes and config
SHA automatically. Reconstruct components by namespace only; delegate public whole-model operations
and model.v1 inspect/canonicalStateBytes/load, NEVER parse private model metadata.
Published `model.v1` remains untouched. Unpublished 0037 `training-checkpoint.v1` is replaced,
not same-file-v1 compatible; no migration promise to an unimplemented wire.

Retain 0037's raw TrainingState codec foundation: BIG_ENDIAN, strict UTF-8 paths without
normalization, explicit SGD=1/Adam=2/AdamW=3 IDs; raw binary64 optimizer coefficients in accessor
order, ordered groups, execution/optimizer-step/pending counters, ordered static parameter schema.
Dtype IDs 1 FLOAT64, 2 FLOAT32, 3 BFLOAT16, 7 FLOAT16. For ordinal N (eight-digit decimal), keys
`training/parameter/N/{logical,master,first,pending}` plus `second` iff adaptive slot is present.
Logical payload uses its dtype width; masters/slots/pending use F64 for F64, F32 otherwise.
Preserve signed zeros, low logical/master distinctions and raw slot/pending bytes exactly.
Structural constructors permit zero-size values; session restore decides actual trainability.

Before Ready freeze magic/version/record order for revised Training schema and recipe loss IDs/
options/signatures/context records; no generic caller-object encoding. Do not invent a duplicate
certificate or pure validator. Public defensive TrainingState/ParameterState/ParameterGroup
constructors establish structure; algorithm slot validity/finiteness/group membership/order and
coefficient narrowing/session compatibility remain Training.restore authority.

## Validation, capture and recovery

Bound/checksum outer file first; preflight checked outer AND projected whole-model/model framing,
schema/payload/aggregate sizes and all ArtifactLimits before component or ParameterState clones.
Bound every nested count/rank/path/string by remaining bytes/minimum record size and array ceilings.
Defaults must be conservative and configurable; canonical encoded limits are not peak-heap budgets.
Reject missing/duplicate/extra keys, unknown versions/IDs, malformed/trailing data and truncation.
Retain existing checked ArtifactFormatException/I/O taxonomy; owner/lifecycle errors stay owner errors.

Before save/read returns any detached snapshot, compare COMPLETE model parameter paths, PARAMETER
kind, dtype/Shape and independent canonical logical bytes with TrainingState, including no extras.
Buffers stay NN/model-owned with all seven raw model types. Definition must predict the whole NN
descriptor/alias set; config SHA is not authentication, approval or a replacement for semantic checks.
High-level save holds TrainingRun's narrow checkpoint exclusion across model capture + session.state;
equality cannot prove quiescence. No old open-session binding replacement or capture race.

Recovery order is fixed:

1. Read/bound/validate all components, definition/recipe meaning, fingerprints and raw-value coherence.
2. NN reconstructs a private descriptor-bound model; public whole-model load installs heap state
   BEFORE any session opens, preserving sharing or rejecting unsupported aliases.
3. TrainingRun owns it exclusively, allocates/copies shared native parameter storage and applies
   saved modes/contexts. Build placeholders and supported objective from final parameter Tensors.
4. Open ONE ordinary TrainingSession with saved optimizer/groups; compare canonical parameters
   and call ordinary restore through Training's private-run admission/restore bridge.
5. Return only the complete run. Failure closes/discards everything acquired; session closes before
   parameter/placeholder Arenas, all cleanup attempted and failures retained/suppressed.

Engine is borrowed, never closed by load/resume/run. TrainingRun.trainBatch validates actual dense
dtype/Shape/layout/offset/view/storage and binds borrowed fresh batch storage to stable placeholder
IDs through replaceHostStorage; roles resolve once from compiled metadata, associations clear in
finally. No hot-path string dispatch/new input mapping authority, per-batch compile or input closure.
Shape changes fail for fixed prepared recipe; an advancing graph-RNG stream is not provided.
Run/native model never escapes as indefinitely usable state. loadModel instead returns independent
heap-backed eval model, with no arena or TrainingSession; inference uses forward-only Engine.compute.
High-level model-only load rejects legacy values-only files with a clear existing-target-load hint.

## Acceptance, documentation and validation

Tests cover high-level no-callback MLP inference and CPU next-input resume matching uninterrupted
admitted route; all four Training dtypes, optimizers/mixed groups/raw slots/counters/pending;
complete descriptor/raw-byte/alias/recipe/fingerprint mismatch; corruption/nested/outer/projected
bounds; immutable values after source close; failed candidate preserving published state; native
resource cleanup at every failure and borrowed Engine/input lifetime; reusable static batch binding.
Add bounded actual Metal smoke with source-matched library and report absence/skips honestly;
no cross-device exactness or broad performance claim.

Before Ready coordinate root + owning extensions contract + explanations + ADR0038 acceptance
and architecture tests; include optional project and exact two api edges only in implementation.
Update CheckpointDependencyContractTest's absent-adapter assertion/exact edges and forbid reverse
edges; retain model-only exact three-edge check. Add durable end-to-end integration fixture.
Clean implementation + independent targeted Class C API/docs review. Update user examples,
model/training API/Javadoc and targeted glossary terms; shared-doc coordinator integrates once.

Future worker: `./gradlew :extensions:training-checkpoint:test :extensions:training-checkpoint:javadoc`.
Coordinator once after integration:
`SYNAPTIK_METAL_TEST_LIBRARY=<source-matched-library> ./gradlew test javadoc --max-workers=2 --console=plain`.
Reuse owner evidence; independent reviewer does not rerun successful Java suites.
Planning now: local links/anchors, brief ≤200 lines/15KB, `git diff --check`; no Java/build edits/tests.

## Result

Empty until implementation. Record changes/files/validation/docs/limitations/follow-up and required
completion status; this planning revision marks no code implemented.
