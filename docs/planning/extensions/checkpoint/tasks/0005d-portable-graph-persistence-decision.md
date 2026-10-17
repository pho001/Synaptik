# Task CP0005D: Model execution and persistence program decision

## Status

Complete — bounded planning/review only, frozen 2026-10-16 after independent Astra/xhigh review,
the sole P2 correction and coordinator acceptance. ADR0039 remains Proposed for later semantic/API
choices. Engine0020 is the sole Ready implementation frontier; no executor is launched and no
normative contract changes here. NN0026/CP0005B stay Complete; their earlier approvals are historical.

## Change class

Class C decision scope: cross-module ownership, session binding and cold persistence boundaries.
This execution is documentation-only in the supplied isolated worktree; no child agents.

## Goal

Replace the outputs(actualInputs)-first proposal with one coherent phased program for manual and
file-restored tensor models using the same proposed compute/train/save/load workflow, and prepare
one independently actionable backend/native-layout successor.

## Scope and non-goals

Revise [ADR0039](../../../../design/decisions/0039-portable-graph-model-persistence.md), this brief,
Checkpoint/NN/Training master plans and roadmap; add the
[cross-area program](../../../model-execution-refactor.md), update Engine master and prepare only
[Engine 0020](../../../modules/engine/tasks/0020-backend-native-layout-relocation.md).
Preserve completed history and holds. No Java/native/build/test-source/normative edits, moves,
stage/commit/push, worktree changes, implementation launch or speculative task cascade.

The latest user approval authorizes revised planning and the backend-only target layout, not
automatic acceptance of every former CP0005D choice. Generic API/ownership/source migration
decisions remain open. No broad audit, all-family codec proof, native run or performance campaign.

## Contracts

Current authority wins. The proposal identifies future conflicts without enacting them.

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`,
  `Repository layout`, `Core lifecycle`, `Scope-indexed normative contracts`,
  `Module-ownership routing`, `Dependency rules`, `Explicit non-goals`, `Testing requirements`.
- [Foundational Model](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  `modules/model` — Tensor identity/provenance, graph data and storage construction.
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler):
  `modules/compiler`, `Compile artifacts`, `Training graph model`,
  `Compiler-owned automatic differentiation`, `Compile lifecycle` — ordinary capture/autograd;
  proposed cold reconstruction is not current authority.
- [Engine/runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  `modules/engine`, `Public explicit composition`, `Construction failure and closure`,
  `Compile ownership and mixed-owner Prepare routing`, `Public reusable inference session`,
  `Prepare lifecycle`, `Run lifecycle`.
- [Extensions](../../../../architecture/contracts/extensions-training.md#extensionsnn):
  `extensions/nn`, `Closed declarative models`, `extensions/training`,
  `extensions/checkpoint`, `Whole-model definition and composition`,
  `Optimizer/training lifecycle` — current edges/v1 and native/precommit lifetime constraints.
- [Backends](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules):
  `Concrete backend modules`, `Metal backend`, `OpenBLAS provider` — preserve semantic owners
  and separate projects through relocation.

No missing boundary is inferred from a planning document. A future executor stops on unapproved
semantic scope; the user explicitly permits this bounded proposal to identify conflicts.

## Dependencies and integration

- Depends on: NN0026/CP0005B Complete; latest user-approved direction/layout and supplied source facts.
- Conflicts with: shared model/session/persistence/ownership contracts and all successor expansion
  of NN0027/CP0005C/Training0006/adapter0006.
- Parallel group: None.
- Common base revision: `9e7e8aa15e0833a1bc91263a7add99a80d88cd92`.
- Integration order: planning author → independent review → coordinator acceptance/Engine0020 Ready
  verification → separate launch; later API cutovers require their own accepted owner packets.
- Integration validation: supplied targeted Markdown validator, git diff --check, brief guardrails,
  exact diff scope and source-based proposed-example review; no executable suites.
- Shared-document integration owner: model-execution coordinator (NN/Training/Checkpoint coordinator
  with Engine/backend owners); no shared writer in this isolated worktree.

Launch verification for this planning pass: coordinator read planning guide/roadmap/active masters,
verified six existing drafts match main and no active writer; author verified branch
`work/graph-model-persistence-design`, exact HEAD above and the six supplied dirty paths.
Scope is bounded planning authorized by the latest user request, not implementation of the former
Draft proposal. Requested model/reasoning: gpt-6-astra / xhigh; author cannot independently verify
runtime selection and makes no model-selection claim.

Implementation frontier: Engine0020 Ready, none launched. Its reviewed mechanical layout packet
and coordinator verification are recorded in that brief. Model/Engine API answers remain pending
for later phases; coordinator acceptance of this planning task does not decide them. Integration
of the frozen packet and launch-state revalidation precede the separate implementation launch.

## Files and symbols

Focused evidence, not a repeat repository audit:

- Engine InferenceSession.run / EngineTypedLifecycleTest: exact IDs, one prep and storage snapshots.
- NN Model<I,O>, Module.train/stateDictionary, existing declarative/v1 behavior.
- TrainingSession.open/run/validateModuleAndParameters, TrainingParameter,
  TrainingParameterStorage and native-copy/failure tests.
- Checkpoint artifact six-file envelope versus nine NN-specific model classes.
- Compiler GraphCapture, CapturedGraphInference, GraphCompilationPort and Model Tensor constructors.
- EngineCompositionContractTest, CheckpointDependencyContractTest, NnTrainingDependencyContractTest;
  settings, relevant Gradle edges, architecture API-consumer/glob inputs, Metal verifier/evidence paths.

Concrete current→target production map and new-type costs live only in the
[program](../../../model-execution-refactor.md#current-to-target-map).

## Acceptance criteria

- Same proposed compute/train/save/load for manual/file tensors; no mandatory runner, special loaded
  graph hierarchy or per-batch cold reconstruction.
- Exact backend/native target paths, separate project identities, generic modules/training and
  modules/checkpoint targets, NN-owned glue and acyclic target edges are explicit.
- Named-parameter live stability, heap/native copy ownership, raw values/aliases, precommit checks,
  optimizer state/failure atomicity and fresh per-run borrowed inputs remain visible.
- Existing ordinary DAG/autograd/Prepare/Runtime reused; no new dynamic-shape, RNN recipe, backend/
  gradient or exact-resume promise. Old v1 and flexible declarative APIs stay supported.
- Unanswered Model/type, Engine/parameter ownership and migration choices have recommendations and
  exact gates; no manufactured approval of TensorModel or borrowed-state mutation.
- One narrow backend-only packet: approved layout, coordinated authority before/with moves,
  stable Gradle identities/packages/kernel bytes/ABI; no Training/Checkpoint move in that task.
- Synchronize current versus target in active masters/roadmap; preserve completed history and holds.
  Prior reviews are historical; independent review of this revision remains explicit.

## Validation

```bash
git diff --check
node /private/tmp/synaptik-model-plan-validation.vgqKbH/check-docs.mjs /private/tmp/synaptik-graph-persistence-design.m1nHof
wc -l -c docs/planning/extensions/checkpoint/tasks/0005d-portable-graph-persistence-decision.md docs/planning/modules/engine/tasks/0020-backend-native-layout-relocation.md
git status --short
```

The supplied checker validates all changed/new inline local links/anchors, whitespace, <=200-line/
15-KiB briefs and eight allowed documents plus at most one new brief. Also review the diff against
the initial six-draft inventory, proposed examples and exact production scope. No Java/Javadoc/
native tests for documentation-only work; later owner packets define their own validation.

## Documentation and review impact

Decision/planning/general/example profiles applied. Targeted glossary search covered Tensor,
provenance, inference session and native Training parameters; no accepted term or glossary edit.
Javadoc impact: none; Java and current APIs are unchanged. Independent targeted Astra/xhigh review
of the Class C proposal/coordination scope is complete; the sole P2 and coordinator acceptance are
recorded below. No glossary or normative change is implied.

## Result

- Frozen 2026-10-16: common execution program, exact owner/path/edge map and one backend-only packet.
- Nine allowed planning/design files changed; no code, build, test-source, normative or staged edits.
- NN0026/CP0005B and 70 existing completed task rows preserved; old v1/flexible models and holds retained.
- Independent Astra/xhigh reviewer Mendel01a143bd-81ec-7c32-b17a-0eed065338db: no P1; sole P2 was
  missing implicit :backends parent projectDir. Corrected brief/program/ADR to map parent plus four
  children and validate projects without the old directory masking failure under Gradle 9.6.1.
- Coordinator accepts this bounded planning/review scope after correction; Engine0020 Ready evidence
  is recorded in its brief. ADR0039 remains Proposed; Model/Engine questions remain unanswered.
- Validation: `git diff --check` and the exact supplied Node command above passed: nine files,
  252 local links/23 anchors, zero errors; both briefs below 200 lines/15 KiB. Scope/completed-row
  preservation checked. Proposed examples unchanged; no Java/Javadoc/native/performance runs.
- Documentation: decision/planning/general/example profiles; no Java/Javadoc or glossary change.
- Earlier proposal reviews/189-link evidence remain historical, not acceptance of later API defaults.
- Follow-up: coordinator integrates the frozen packet, reverifies launch revision/dirty inventory
  and inactive conflicts, then separately launches Engine0020. Later semantic choices retain gates.
- Status: Complete — planning/review only; no implementation or native/device success claimed.
