# Task 0005B: Whole-model save and automatic inference load

## Status

Draft — immediate Checkpoint successor to NN 0026. This is a proposal, not implemented code.
Requires stable NN support matrix and whole-model wire/API contract before Ready.

## Change class

Class C implementation — new persistent kind and automatic semantic reconstruction boundary.

## Goal

Persist supported model definition + weights and return a usable NN model from high-level
ModelCheckpoint.load(Path), without callbacks, manual construction, digests or Arena plumbing.
Deliver whole-model file/envelope API, codec, tests and docs as one Checkpoint capability.

## Scope

Extend existing `io.github.pho001.synaptik.checkpoint.model.ModelCheckpoint` with distinctly named
whole-model envelope operations and high-level overloads using existing `ArtifactEnvelope`.
Proposed private `WholeModelDefinitionCodec` and `WholeModelComposition` in that package; no
public image wrapper, trusted validation token or private model-codec access by downstream adapters.
NN defines meaning/reconstruction; Checkpoint defines canonical persistent encoding/versions.

Conceptual signatures, not existing APIs:

```text
ModelCheckpoint.save(Path, DeclarativeModel, Engine): void
ModelCheckpoint.load(Path): DeclarativeModel
ModelCheckpoint.captureWholeModel(DeclarativeModel, Engine, ArtifactLimits): ArtifactEnvelope
ModelCheckpoint.loadWholeModel(ArtifactEnvelope, ArtifactLimits): DeclarativeModel
```

Freeze typed expected-input-fingerprint and configurable ArtifactLimits overloads before Ready.
Simple overloads require absent input fingerprint; present fingerprints reject with a diagnostic
requiring explicit expected identity. No generic LoadOptions, registry or automatic Text loading.
Preserve all existing explicit model.v1 file/envelope/target APIs exactly.

## Non-goals

No arbitrary callback/subclass capture inference, training recipe/state, backend discovery,
compiled/device formats, optimizer/session reconstruction, NN→Checkpoint edge or ModelExecutor.
Do not upconvert values-only files or guess topology from weight Shapes.

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

- Depends on: NN 0026; completed Checkpoint 0002/0004/0004A/0005A; approved 0038 whole-model contract.
- Conflicts with: shared NN/Training/Checkpoint API, lifecycle, wire and authoritative documents.
- Parallel group: None; owner tasks use separate clean contexts, serialized contract cutovers.
- Common base revision: planning base `52733fcfd32d278b651302e308487e0b6c6afa81`; reverify before Ready.
- Integration order: NN 0026 → Checkpoint 0005B; join Training 0006 before optional Checkpoint 0006.
- Integration validation: one combined source-matched test/Javadoc checkpoint after adapter integration.
- Shared-document integration owner: NN/Training/Checkpoint coordinator.
- Authorized frontier: None; dependencies/support matrix/contracts must be verified before Ready.

## Files and symbols

- `extensions/checkpoint/.../checkpoint/model/ModelCheckpoint.java`; new private whole-model codecs.
- Existing public `artifact/{ArtifactFile,ArtifactEnvelope,ArtifactLimits}.java` reused unchanged.
- NN specification/reconstruction API; Checkpoint API/module tests and integration fixtures.
- `docs/api/model-checkpoint-api.md`, running-models guide and affected architecture explanations.

## Acceptance criteria

- Proposed outer kind `whole-model.v1` has `definition/schema` and prefixed unchanged model.v1
  entries under `model/`. Checkpoint canonical encoding contains actual attrs, wiring, signatures
  and alias topology; automatically compute model config SHA-256 from exact definition bytes.
- Use public capture/inspect/canonicalStateBytes/load for model.v1 validation and installation.
  Definition version/kinds/attrs/input-output constraints and complete NN predicted state descriptors/
  aliases match independent validated raw data BEFORE any model is returned.
- Reconstruct a private descriptor-bound NN skeleton; load through public ModelCheckpoint, with
  no random weight initialization, private ordinal parsing or temporary component files.
- Return independent heap-backed eval model; gradient-eligible parameters remain ordinary NN state.
  Model load performs no autograd/compile; ordinary Engine.compute is forward-only and its
  HostTensorValue result detached. Model remains usable after file/source/native Arena closure.
- Values-only model.v1 high-level load fails clearly: missing definition, use existing target-load.
  Published model.v1 and ArtifactFile v1 remain byte/API compatible, not silently redefined.
- Selected NN matrix includes MLP activation/config round trips, Embedding mixed input/output dtype,
  all convolution attrs, supported leading-dimension constraints and explicit sharing decision.
  Reject arbitrary callback/custom kind; no promise about all standard models.
- Safe default existing ArtifactLimits, configurable advanced overloads; decide concrete defaults
  before Ready. Preflight outer AND projected model component, nested counts/ranks/strings,
  checked sizes and array ceilings before allocation/copy. Limits are not peak-heap budgets.
- Bound definition node/edge/depth/path counts; reject duplicate IDs, unknown/dangling references
  and ownership/execution cycles before reconstruction or model publication. Test malformed and
  over-limit definitions even when their values component is independently valid.
- Reject corruption/truncation/unknown/duplicate/extra keys, mismatched definitions/raw descriptors/
  alias bytes, config hash and expected input identity. Hashes are not authentication/approval.
- Empty/identity Sequential inference remains subject to existing Engine leaf/zero-node admission;
  decide eligibility explicitly before Ready without widening Engine or backend capability.
- Save uses existing file force/revalidation/atomic-live policy and caller quiescence; no weaker
  move fallback/power-loss claim. Failure returns no model and preserves published objects.

## Validation

Future worker: focused whole-model API/canonical/config/limits tests, then
`./gradlew :extensions:checkpoint:test :extensions:checkpoint:javadoc` once.
Coordinator: relevant architecture/end-to-end tests and one source-matched combined
`./gradlew test javadoc --max-workers=2 --console=plain` after program integration.
Independent review reuses successful Java evidence; no performance campaign.
Planning now: local links/anchors, ≤200 lines/15KB and `git diff --check`; no build.

## Documentation and review impact

Clean implementation and independent Class C targeted API/format/documentation review.
Explain new whole-model kind versus old values-only API, exact matrix, fingerprint expectations,
alias/input constraints, default limits and detached lifetime; final Javadoc/glossary targeted review.
Coordinator owns shared explanations and root/scoped-contract cutover before Ready.

## Result

Empty until implementation; record completed changes/files/validation/docs/limitations/follow-up
and required completion status. Planning does not mark code implemented.
