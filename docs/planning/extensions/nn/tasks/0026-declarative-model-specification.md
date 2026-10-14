# Task 0026: Declarative model specification and reconstruction

## Status

Draft — proposed immediate NN predecessor for whole-model persistence; no code is implemented.
Exact support matrix, alias policy and input constraints await decision under ADR 0038.

## Change class

Class C implementation — durable reconstruction meaning, ownership/sharing and downstream contract.
This brief is planning only; it changes no normative contract.

## Goal

Let NN describe and reconstruct supported models as usable Tensor-to-Tensor models without Java
callbacks. Deliver one cohesive NN capability: specification, model construction, focused tests,
Javadoc and user documentation, with Model as its only production dependency.

## Scope

Proposed public `ModelSpecification` and `DeclarativeModel extends Model<Tensor, Tensor>` in
`io.github.pho001.synaptik.nn.module`, including closed nested semantic layer/activation records;
private reconstruction/activation modules only when required. Final names/surface freeze before Ready.
Keep existing `ModelDefinition<I,O>` as the arbitrary Java callback, not a serializable DTO.
Propose nested Sequential; Linear, LayerNorm, Embedding, Conv1d/2d/3d and closed RELU/TANH/SIGMOID
steps. Include an ordinary Linear→RELU→Linear MLP, not merely a single Linear fixture.
Specifications retain ownership names/order, forward wiring, resolved state schema, dtype,
bias policy, typed epsilon and all convolution attrs. NN owns typed access to any configuration
facts not currently public; no reflection/private-field scraping or guessing attrs from weights.

NN distinguishes an owned child tree from execution wiring. Initial Sequential is closed linear
wiring; a branch/reused-node DAG is an explicit later support choice, not inferred from children.
Describe parameter alias groups; implement faithful inference aliases or reject explicitly
according to the approved matrix. Repeated forward invocation of one uniquely owned layer is not
a repeated parameter-path alias. Training's existing duplicate/overlap rejection is unchanged.
Reconstruction should create descriptor-only bound skeletons accepted by supplied-state layer
constructors and strict state load, if focused proof establishes this boundary; no random
initialization or full throwaway weight arrays. Installation remains NN-owned.

## Non-goals

No bytes, file I/O, persistent versions/codecs, Engine, Training or Checkpoint dependencies.
No arbitrary Java object/function serialization, class-name instantiation, registries or generic
operation interpreter. No loss/optimizer meanings, runner, backend discovery or capability widening.
No automatic export of existing Model.define captures or subclasses. Unknown custom models retain
the explicit reconstruction workflow. BatchNorm/Dropout and recurrent/custom record I/O are initially
excluded; declarative branches/contextual adapters are named successors, not detailed new tasks.

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

- Depends on: completed NN 0025/0025A/0025B and existing module/state/layer contracts; approved 0038 NN scope.
- Conflicts with: shared NN/Training/Checkpoint API, lifecycle, wire and authoritative documents.
- Parallel group: None; owner tasks use separate clean contexts, serialized contract cutovers.
- Common base revision: planning base `52733fcfd32d278b651302e308487e0b6c6afa81`; reverify before Ready.
- Integration order: NN 0026 → Checkpoint 0005B and Training 0006 → optional Checkpoint 0006.
- Integration validation: one combined source-matched test/Javadoc checkpoint after adapter integration.
- Shared-document integration owner: NN/Training/Checkpoint coordinator.
- Authorized frontier: None; dependencies/support matrix/contracts must be verified before Ready.

## Files and symbols

- `extensions/nn/.../nn/module/{Model,ModelDefinition,Sequential,ModuleFactory,Module,Parameter}.java`.
- Selected `nn/layers/{Linear,LayerNorm,Embedding,Conv1d,Conv2d,Conv3d}.java` and attrs.
- New specification/model in the existing module package; focused module/layer tests.
- `docs/user-guide/running-models.md`, targeted NN API/glossary entries and architecture explanations.

## Acceptance criteria

- Approved closed matrix rejects unknown kinds/attrs/wiring/signatures before publishing a model.
  Reject duplicate node IDs, unknown/dangling references and ownership/execution cycles; children
  and state paths do not substitute for validated forward wiring.
- Layer semantics and exact forward shape/provenance match ordinary direct NN construction:
  MLP, normalization, embedding→projection and dimension-specific convolution examples.
- State descriptors match names/order/kinds/types/Shapes; full data are loaded separately by NN.
  Tests prove descriptor skeleton load and no eager/random weight-sized allocations.
- Exact alias decisions are tested: supported aliases preserve identity; unsupported cases reject.
  No silent value-based sharing or splitting, including equal-valued independent parameters.
- Input/output contracts distinguish dtype and constraints: Embedding integral input/floating output;
  Linear compatible leading dimensions need not freeze batch size. Decide variable-batch constraints
  before Ready; static Training signatures remain a separate owner concern.
- Existing generic Model/ModelDefinition/Sequential/ModuleFactory behavior remains compatible.

## Validation

Future worker: focused new specification/reconstruction tests, then
`./gradlew :extensions:nn:test :extensions:nn:javadoc` once after final code/docs.
Coordinator adds architecture/integration coverage where the boundary changes; one combined
`./gradlew test javadoc --max-workers=2 --console=plain` with source-matched device library after
ordered program integration. No performance campaign or duplicate reviewer Java run.
Planning now: local links/anchors, brief ≤200 lines/15KB and `git diff --check` only.

## Documentation and review impact

Clean implementation + independent targeted Class C review/documentation are mandatory.
Document supported matrix, configuration/wiring versus callback distinction, signatures, sharing
and skeleton ownership; Javadoc every new input/result/failure. Targeted glossary search, not full
history. Coordinator owns shared documents and contract cutover before Ready.

## Result

Empty until implementation. Completion must record files, validation, documentation/Javadoc,
limits/follow-up and `Status: Complete` or `Status: Incomplete` with specific follow-up.
