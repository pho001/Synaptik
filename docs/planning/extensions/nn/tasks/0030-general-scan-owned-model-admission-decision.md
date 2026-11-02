# Task NN0030: General-scan owned Model admission decision

## Status

Complete — approved limited direction/planning decision only, not module implementation.
Explicit user approval “ok, pust se do toho” is recorded 2026-10-23 for the recommended existing-owner
direction and FIRST Compiler prerequisite. Prior independent approval was Draft planning only.
NN implementation frontier: None. [Compiler0012](../../../modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md)
is Complete after independent code/docs APPROVE and main's final gates (2026-10-23); its brief retains
historical pre-code APPROVE/Ready verification. Next is separately briefed Checkpoint copy/rebind,
then NN qualification; Compiler/NN/Checkpoint frontiers None, no successor launch.

## Change class

The approved cold-source/NN ownership cutover is Class C; this completed decision is not an
implementation grant. Clean implementation and independent targeted review remain mandatory.

## Goal and recommendation

Let an application use an ordinary Tensor `ScanBody` (a closed pure step expression) through
existing in-memory `Model.graph` / `Model.defineTensor`, then `compute` and
`configureTraining` / `train`. Initial tested compute/train scope: all-valid, canonical, static-T
homogeneous FLOAT32/FLOAT64 GENERAL (`ScanKind.SCAN`) scans in the qualified CPU domain; first-order automatic
differentiation (AD) remains Compiler-owned. Source/cold-copy admission is separate, as specified below.
Keep one private state set and the existing public Engine, TrainingSession and GraphCheckpoint owners.
No new public method/class or production type is presently justified by the source inventory.

## Scope and source-backed findings

Paths below are relative to the repository root; findings record the decision-time baseline, not
new contracts. Compiler0012's completed Result supersedes its former cold-reconstruction barrier;
Checkpoint attachment preservation remains the next separate gate, not implemented NN support.

- [NN Model](../../../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java):
  `graph` calls `GraphCheckpoint.copy`; `defineTensor` invokes one eval prototype callback and
  `OwnedModelSource.acquireDefinition`, then discards the callback. It is not safe to retain it
  to bypass copying: a closure can still refer to external weights after wrapper installation.
- [OwnedModelSource](../../../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/OwnedModelSource.java):
  `acquireDefinition` inventories state/constants, copies once and installs wrappers;
  `forward` delegates to `GraphCheckpoint.rebind` unless this is existing builder composition.
  `acquireComposition` uses `Source` only as a borrowed Tensor-list carrier, without codec/capture.
  Its inventory walks outer operands; body formals must never become external state/constants.
- [GraphCheckpoint](../../../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)
  `copy` / `rebind` use `GraphSource.capture`, NOT artifact encoding.
  [GraphSource](../../../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphSource.java)
  calls `GraphCompilationPort.captureSource` once, normalizes IDs and reconstructs through Compiler.
  Its `new CompiledNode(..., outputs)` currently omits `scanBody`; merely admitting GENERAL in
  Compiler would therefore be insufficient. Preserve the attachment and local IDs during outer remap.
- [Compiler cold owner](../../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/SourceGraphReconstruction.java)
  `preflight` admits a closed flat family inventory; `additional` admits fixed `RecurrentScanKind`,
  not `ScanKind.SCAN`. `reconstruct` has no GENERAL branch and lacks some declared body algebra
  (e.g. DIV, SUM and scalar forms). Existing
  [GraphCapture.captureSource](../../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java)
  already captures attachments, but `GraphCompilationPort.SourceCapture` calls this cold preflight.
  Low-level capture/AD success does not prove cold copy/reconstruction support.
- [ModelExecutionState](../../../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelExecutionState.java)
  `rejectScanTraining` uses `OwnedModelSource.containsScan`, which detects ONLY fixed recurrent kinds.
  Do not remove this guard to admit GENERAL. `configureOpen` already builds one objective, admits
  native parameter storage once and delegates to `TrainingSession.open`; compute/train cache
  prepared bindings and consume fresh values without per-batch reconstruction.
- [GraphArtifactCodec](../../../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphArtifactCodec.java)
  `encode` / `opcode` independently enforce opcodes 1..22 before storage admission and file publication.
  There is no GENERAL wire. After copy/rebind widening, save must still explicitly reject body
  attachments, including malformed attachments on otherwise encodable kinds, before value/file effects.

Compiler0012 and its inventory audit record focused source/test evidence; no new tests ran here.

## Approved decision and alternatives

Approved: **separate source semantics, cold reconstruction, execution and wire eligibility**.
Model0035 still admits existing known PURE operations at their applicable Model dtypes; never use
CPU0033 support as a source-validity/copy predicate. A valid FLOAT16 TANH or F32 RELU body remains
valid without an executable CPU route. Cold support must preserve all Model-valid dtypes of each
implemented reconstruction signature, not filter to F32/F64. The initial source capability should
reuse existing exact cold-family mappings and add missing DIV/scalar/SUM/layout mappings needed
for the declared algebra. Any narrower cold inventory/attribute guard must be justified by missing
constructor/reconstruction support (as above), explicitly listed in the first Ready brief and
reviewed independently of providers; unsupported reconstruction does not invalidate the Model body.
Do not promise all PURE-body copying merely because Model construction accepts it.
Alternative: leave owned GENERAL acquisition rejected and keep direct Engine use; smaller change,
but it does not deliver the requested ordinary owned Model workflow. NN-local cloning or a new
runner/session/source wrapper would duplicate Compiler reconstruction/ownership and conflicts with
the project's single ordinary Tensor graph and execution philosophy; do not select it.

Approved scope: **a Class C cutover extending Compiler cold source to body reconstruction and
Checkpoint copy/rebind based on exact reconstructible Model semantics, independent of CPU/dtype
execution support, then owned Model.graph/defineTensor with initially tested all-valid static-T
canonical F32/F64 compute/first-order training, keeping body save/load unsupported and fixed
runtime-length scans inference-only.**
Explicit approval is recorded above. Lovelace binding pre-code APPROVE and main verification are
complete and retained in Compiler0012, now Complete for that Compiler capability only. Scope is the
declared compute/AD algebra/closure; cold inventory needs its own constructor-backed review,
not an all-PURE or blanket outer-source widening.

## Smallest serial upstream frontier and packages

Approved serial order: reviewed authority cutover → **Compiler cold-source capability** → Checkpoint
attachment-preserving in-memory copy/rebind with independent wire rejection → NN acquisition/workflow
verification → main's integration join. Compiler0012 is Complete; the next separately briefed gate
is Checkpoint copy/rebind. No future per-module brief or Ready status is created here.

First module: existing `io.github.pho001.synaptik.compiler`, principally `SourceGraphReconstruction`,
`GraphCompilationPort`, `GraphCapture` and `ScanBodyInference`. Use existing `CompiledScanBody` /
`CompiledGraphModel` to preflight the complete outer graph AND every scoped body before constructing
any producer; reconstruct fresh distinct canonical formals with TensorFactory, ordinary body algebra,
one `ScanBody` and one `Scan.apply` per occurrence. Map every canonical output once, including latent
residuals and aliased roles; retain unused formals, repeated actual positions and independent scopes.
Rebind substitutes outer inputs only; keep produced/repeated substitutions' upstream provenance opaque.
Do not flatten body IDs into outer IDs or let private copies retain external actual leaves/templates.
The same immutable source template may be reused without merging scan occurrences or leaking formals.
Cold reconstruction is call-local and requested explicitly, never per train/compute invocation.

No added/moved production type selected. Existing records contain body graph, formal order and role
bindings, so no missing representation has been demonstrated. If implementation proves another type
necessary, stop with its exact name/package, missing responsibility, existing-owner alternatives and
the duplication/ownership cost; this decision does not authorize an abstraction or public API addition.

## Non-goals and code disposition

No padding, row skipping, dense masks, body migration, fixed-scan BPTT, `.rnn()`/`.gru()`/`.lstm()`
redirection, new builder recipes, source wire/version, low-type execution, vector/parallel/OpenBLAS widening,
new executor, DAG, AD layer, numerical policy or performance-parity claim. T=0 source/compute follows
existing scan semantics; it does not relax Training's empty/nonfinite/alias/storage admission.

No production code is redundant/deletable as a result of this planning task. After an approved
implementation, only blanket GENERAL cold/acquisition exclusions become obsolete within its exact
scope. Fixed `containsScan` / `rejectScanTraining`, fixed emitters, current cells/Sequences, static
builder recurrence, composition adapters, private copy/alias guards, Training native admission and
v1/v2/v3 readers/codec MUST remain. Do not cancel held NN0027 or unrelated Draft gates.

## Coordinated authority and documentation cutover

The root/owning contracts now distinguish the approved target from current implementation, with
bounded Compiler-only admission and unchanged consumer/wire rejection. Their cutover has binding
independent pre-code APPROVE/main acceptance; this plan is not authority. Exact headings:

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core lifecycle`,
  `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules` (preserve all edges).
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  `General differentiable scan`, especially `Qualification, performance and migration gates`;
  clarify independent all-valid NN admission versus later padding, without changing scan semantics.
- [Compiler owner](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
  `Compile lifecycle` — separate bounded cold body support from persistence, retaining full preflight.
- [Extension owner](../../../../architecture/contracts/extensions-training.md#owned-tensor-models):
  `Owned tensor Models`, `Generic graph source and values`, `Canonical graph-model version 3`;
  preserve `modules/training`, v1/v2/v3 closed wires and fixed early rejection.

The cutover includes [ADR0046](../../../../design/decisions/0046-general-scan-cold-source-and-owned-model-direction.md), targeted
`docs/architecture/module-boundaries.md` headings `Owned tensor Model composition` and
`modules/checkpoint` (Gradle `:extensions:checkpoint`), `docs/architecture/training-graph.md`
`Owned Model consumer`, and `docs/architecture/general-differentiable-scan.md`
`Source expressions and independent scopes`. Final API/Javadoc/user explanation review covers
`docs/api/graph-checkpoint-api.md`, `docs/user-guide/running-models.md#owned-static-model-workflow`
and glossary `Owned Model source`, `Generic graph checkpoint source`, `General differentiable scan`.
This task changes authority/planning/explanations/glossary only, no Java/Javadoc. General/planning,
architecture and ADR profiles apply. Class C implementation still needs independent code/docs review.

## Dependencies and integration

- Depends on: Model0034/0035, Planning0008, Compiler0011, Prepare0010, CPU0030–0033 and low-level
  root join; NN0028/0029, NN0022B, Compiler0008, Checkpoint0008–0011, Training0007/0008 — Complete.
- Conflicts with: cold source/reconstruction, NN admission/lifecycle, Checkpoint wire and shared
  authority/document edits; serialized under main. Curie is read-only and exclusively owns Gradle.
- Parallel group: None; one repository writer for this planning packet.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
  dirty baseline (257 paths, empty index); task-local archive records pre-edit content, not HEAD diff.
- Integration order: approved decision → independent authority/inventory review → serial module
  frontier above → independent review → main join; no executor or automatic successor launch.
- Integration validation: main's planning link/anchor/metadata/size/stale-fact checks and task-only
  diff review now; later approved capability checkpoint assigned in the first Ready brief.
- Shared-document integration owner: main; one writer, shared authority and explanations included.

## Acceptance criteria and validation

Planning acceptance: one source-backed approved decision, all seven fields, Compiler0012 Ready after
binding pre-code APPROVE/main verification, NN implementation frontier None, no API invention, and preserved
low-level completion/performance limits. Validate local Markdown links/anchors, targeted stale facts,
brief <=200 lines / <=15 KB and `git diff --check`; archive before/after SHA-256 and exact task-only diff.
No Java/Gradle/native/benchmark execution, source/test edits, commit or push in this planning pass.

Compiler0012 owns exact constructor/scope/consumer acceptance. Later NN qualification must prove
private aliases, one-shot definition, same owned weights/cached owners, F32/F64 compute/train against
ordinary unrolled/direct-session oracles, lifecycle/retry and unchanged mixed fixed-scan rejection.
Body save still rejects before storage/file effects; old wires stay compatible. Reuse accepted
CPU evidence; main runs one final native root test/Javadoc at the future NN join, no timing campaign.

## Result

Status: Complete — limited decision/planning only. User approval recorded; one compact Compiler0012
prerequisite prepared with constructor-backed inventory and test-only consumer rejection acceptance.
Root/three owning scopes, ADR0046, focused explanations/glossary and frontiers coordinated. Module
implementation was not launched by this decision; historical pre-code APPROVE/main Ready verification
remains recorded. Compiler0012 is now Complete; its Result owns review/gate evidence, not NN completion.
No new public API/type/dependency or Java edit. Planning link/size/whitespace and baseline-preservation
evidence is archived in the task-local handoff; main accepted the Compiler consumer checkpoint.
Size exception (>15 KB): preserve accepted decision/acceptance and historical Ready facts with current status.
