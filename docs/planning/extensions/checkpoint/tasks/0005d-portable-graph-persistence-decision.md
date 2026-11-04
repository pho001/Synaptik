# Task CP0005D: [Model](../../../../glossary.md#term-model) execution and [persistence](../../../../glossary.md#term-persistence) program decision

> **Editorial reading context — not a status update.** Completion here is completion of a
> planning/review decision, not evidence that the proposed execution program ran. The central
> distinction is between rebuilding output expressions for every actual batch and retaining
> one source whose prepared execution receives fresh values. Persistence stores supported source
> and state, not a prepared execution. Today's
> [owned Model execution](../../../../reference/nn/owned-model.md#owned-tensor-model-capability) explains
> the application-facing integration, and [graph persistence](../../../../reference/checkpoint/graph.md#purpose-and-mental-model)
> explains the separate cold reconstruction boundary. Those current explanations do not widen
> this packet's original authorization.

## Status

Complete — bounded [planning](../../../../glossary.md#term-planning)/review only, frozen 2026-10-16 after independent Astra/xhigh review,
the sole P2 correction and coordinator acceptance. ADR0039 remains Proposed for later semantic/[API](../../../../glossary.md#term-application-programming-interface-api)
choices. Engine0020 is the sole [Ready](../../../../glossary.md#term-ready) implementation frontier; no executor is launched and no
[normative](../../../../glossary.md#term-normative) contract changes here. NN0026/CP0005B stay Complete; their earlier approvals are historical.

## Change class

Class C decision [scope](../../../../glossary.md#term-scope): cross-[module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), session [binding](../../../../glossary.md#term-binding) and cold [persistence](../../../../glossary.md#term-persistence) boundaries.
This execution is documentation-only in the supplied isolated worktree; no child agents.

## Goal

Replace the outputs(actualInputs)-first proposal with one coherent phased program for manual and
file-restored [tensor](../../../../glossary.md#tensor) models using the same proposed compute/train/save/load workflow, and prepare
one independently actionable [backend](../../../../glossary.md#backend)/native-[layout](../../../../glossary.md#layout) successor.

## [Scope](../../../../glossary.md#term-scope) and non-goals

Revise [ADR0039](../../../../design/decisions/0039-portable-graph-model-persistence.md), this brief,
[Checkpoint](../../../../glossary.md#term-checkpoint)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training master plans and roadmap; add the
[cross-area program](../../../model-execution-refactor.md), update [Engine](../../../../glossary.md#term-engine) master and prepare only
[Engine 0020](../../../modules/engine/tasks/0020-backend-native-layout-relocation.md).
Preserve completed history and holds. No Java/native/build/test-source/[normative](../../../../glossary.md#term-normative) edits, moves,
stage/[commit](../../../../glossary.md#term-commit)/push, worktree changes, implementation launch or speculative task cascade.

The latest user approval authorizes revised [planning](../../../../glossary.md#term-planning) and the [backend](../../../../glossary.md#backend)-only [target](../../../../glossary.md#term-target) [layout](../../../../glossary.md#layout), not
automatic acceptance of every former CP0005D choice. Generic [API](../../../../glossary.md#term-application-programming-interface-api)/[ownership](../../../../glossary.md#term-ownership-meanings)/source migration
decisions remain open. No broad audit, all-family [codec](../../../../glossary.md#term-codec) proof, native [run](../../../../glossary.md#term-run) or performance campaign.

## Contracts

Current authority wins. The proposal identifies future conflicts without enacting them.

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`,
  `Repository layout`, `Core lifecycle`, `Scope-indexed normative contracts`,
  `Module-ownership routing`, `Dependency rules`, `Explicit non-goals`, `Testing requirements`.
- [Foundational Model](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>:
  `modules/model` — [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings)/[provenance](../../../../glossary.md#provenance), [graph](../../../../glossary.md#graph) data and storage construction.
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>:
  `modules/compiler`, [`Compile artifacts`](../../../../glossary.md#compile-artifacts), `Training graph model`,
  `Compiler-owned automatic differentiation`, `Compile lifecycle` — ordinary capture/autograd;
  proposed cold reconstruction is not current authority.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>
- [Engine/runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [runtime glossary](../../../../glossary.md#term-runtime))</span>:
  `modules/engine`, `Public explicit composition`, `Construction failure and closure`,
  `Compile ownership and mixed-owner Prepare routing`, `Public reusable inference session`,
  `Prepare lifecycle`, `Run lifecycle`.
- [Extensions](../../../../architecture/contracts/extensions-training.md#extensionsnn):
  `extensions/nn`, `Closed declarative models`, `extensions/training`,
  `extensions/checkpoint`, `Whole-model definition and composition`,
  `Optimizer/training lifecycle` — current edges/v1 and native/[precommit](../../../../glossary.md#term-precommit) [lifetime](../../../../glossary.md#term-lifetime) constraints.
- [Backends](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backends glossary](../../../../glossary.md#backend))</span>:
  `Concrete backend modules`, `Metal backend`, `OpenBLAS provider` — preserve semantic owners
  and separate projects through relocation.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>

No missing boundary is inferred from a [planning](../../../../glossary.md#term-planning) document. A future executor stops on unapproved
semantic [scope](../../../../glossary.md#term-scope); the user explicitly permits this bounded proposal to identify conflicts.

## Dependencies and integration

- Depends on: NN0026/CP0005B Complete; latest user-approved direction/[layout](../../../../glossary.md#layout) and supplied source facts.
- Conflicts with: shared model/session/persistence/ownership contracts and all successor expansion
  of NN0027/CP0005C/Training0006/adapter0006.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Parallel group: None.
- Common base revision: `9e7e8aa15e0833a1bc91263a7add99a80d88cd92`.
- Integration order: planning author → independent review → coordinator acceptance/Engine0020 Ready
  verification → separate launch; later API cutovers require their own accepted owner packets.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Integration validation: supplied targeted Markdown validator, git diff --check, brief guardrails,
  exact diff scope and source-based proposed-example review; no executable suites.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Shared-document integration owner: model-execution coordinator (NN/Training/Checkpoint coordinator
  with Engine/backend owners); no shared writer in this isolated worktree.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [backend glossary](../../../../glossary.md#backend))</span>

Launch verification for this [planning](../../../../glossary.md#term-planning) pass: coordinator read planning guide/roadmap/active masters,
verified six existing drafts match main and no active writer; author verified branch
`work/graph-model-persistence-design`, exact HEAD above and the six supplied dirty paths.
[Scope](../../../../glossary.md#term-scope) is bounded planning authorized by the latest user request, not implementation of the former
Draft proposal. Requested [model](../../../../glossary.md#term-model)/reasoning: gpt-6-astra / xhigh; author cannot independently verify
[runtime](../../../../glossary.md#term-runtime) selection and makes no model-selection claim.

Implementation frontier: Engine0020 [Ready](../../../../glossary.md#term-ready), none launched. Its reviewed mechanical layout packet
and coordinator verification are recorded in that brief. Model/[Engine](../../../../glossary.md#term-engine) [API](../../../../glossary.md#term-application-programming-interface-api) answers remain pending
for later phases; coordinator acceptance of this planning task does not decide them. Integration
of the frozen packet and launch-state revalidation precede the separate implementation launch.

## Files and symbols

Focused evidence, not a repeat repository audit:

- [Engine](../../../../glossary.md#term-engine) [InferenceSession](../../../../glossary.md#former-inference-session--inferencesession).[run](../../../../glossary.md#term-run) / EngineTypedLifecycleTest: exact IDs, one prep and storage snapshots.
- [NN Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions)<I,O>, [Module](../../../../glossary.md#term-module).train/stateDictionary, existing declarative/v1 behavior.
- TrainingSession.open/run/validateModuleAndParameters, TrainingParameter,
  TrainingParameterStorage and native-copy/failure tests.<span data-glossary-navigation="true"> ([TrainingSession glossary](../../../../glossary.md#training-session--trainingsession); [TrainingParameterStorage glossary](../../../../glossary.md#native-parameter-copy--trainingparameterstorage))</span>
- [Checkpoint](../../../../glossary.md#term-checkpoint) [artifact](../../../../glossary.md#term-artifact) six-file envelope versus nine [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-specific [model](../../../../glossary.md#term-model) classes.
- [Compiler](../../../../glossary.md#term-compiler) GraphCapture, CapturedGraphInference, GraphCompilationPort and Model [Tensor](../../../../glossary.md#tensor) constructors.
- EngineCompositionContractTest, CheckpointDependencyContractTest, NnTrainingDependencyContractTest;
  settings, relevant Gradle edges, architecture API-consumer/glob inputs, Metal verifier/evidence paths.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Metal glossary](../../../../glossary.md#term-metal))</span>

Concrete current→[target](../../../../glossary.md#term-target) production map and new-type costs live only in the
[program](../../../model-execution-refactor.md#current-to-target-map).

## Acceptance criteria

- Same proposed compute/train/save/load for manual/file tensors; no mandatory runner, special loaded
  graph hierarchy or per-batch cold reconstruction.<span data-glossary-navigation="true"> ([tensors glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>
- Exact backend/native target paths, separate project identities, generic modules/training and
  modules/checkpoint targets, NN-owned glue and acyclic target edges are explicit.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [target glossary](../../../../glossary.md#term-target); [modules glossary](../../../../glossary.md#term-module); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Named-parameter live stability, heap/native copy ownership, raw values/aliases, precommit checks,
  optimizer state/failure atomicity and fresh per-run borrowed inputs remain visible.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [heap glossary](../../../../glossary.md#term-heap); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [precommit glossary](../../../../glossary.md#term-precommit); [optimizer state glossary](../../../../glossary.md#term-optimizer-state); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity); [run glossary](../../../../glossary.md#term-run); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Existing ordinary DAG/autograd/Prepare/Runtime reused; no new dynamic-shape, RNN recipe, backend/
  gradient or exact-resume promise. Old v1 and flexible declarative APIs stay supported.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [autograd glossary](../../../../glossary.md#autograd); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [shape glossary](../../../../glossary.md#shape); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [gradient glossary](../../../../glossary.md#term-gradient); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>
- Unanswered Model/type, Engine/parameter ownership and migration choices have recommendations and
  exact gates; no manufactured approval of TensorModel or borrowed-state mutation.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine))</span>
- One narrow backend-only packet: approved layout, coordinated authority before/with moves,
  stable Gradle identities/packages/kernel bytes/ABI; no Training/Checkpoint move in that task.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [Gradle glossary](../../../../glossary.md#term-gradle); [kernel glossary](../../../../glossary.md#kernel); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Synchronize current versus target in active masters/roadmap; preserve completed history and holds.
  Prior reviews are historical; independent review of this revision remains explicit.

## [Validation](../../../../glossary.md#term-validation)

```bash
git diff --check
node /private/tmp/synaptik-model-plan-validation.vgqKbH/check-docs.mjs /private/tmp/synaptik-graph-persistence-design.m1nHof
wc -l -c docs/planning/extensions/checkpoint/tasks/0005d-portable-graph-persistence-decision.md docs/planning/modules/engine/tasks/0020-backend-native-layout-relocation.md
git status --short
```

The supplied checker validates all changed/new inline local links/anchors, whitespace, <=200-line/
15-KiB briefs and eight allowed documents plus at most one new brief. Also review the diff against
the initial six-draft inventory, proposed examples and exact production [scope](../../../../glossary.md#term-scope). No Java/[Javadoc](../../../../glossary.md#term-javadoc)/
native tests for documentation-only work; later [owner](../../../../glossary.md#term-owner) packets define their own validation.

## Documentation and review impact

Decision/[planning](../../../../glossary.md#term-planning)/general/example profiles applied. Targeted glossary search covered [Tensor](../../../../glossary.md#tensor),
[provenance](../../../../glossary.md#provenance), [inference](../../../../glossary.md#term-inference) session and native Training [parameters](../../../../glossary.md#term-parameter); no accepted term or glossary edit.
[Javadoc](../../../../glossary.md#term-javadoc) impact: none; Java and current APIs are unchanged. Independent targeted Astra/xhigh review
of the Class C proposal/coordination [scope](../../../../glossary.md#term-scope) is complete; the sole P2 and coordinator acceptance are
recorded below. No glossary or [normative](../../../../glossary.md#term-normative) change is implied.

## Result

- Frozen 2026-10-16: common execution program, exact [owner](../../../../glossary.md#term-owner)/path/edge map and one [backend](../../../../glossary.md#backend)-only packet.
- Nine allowed [planning](../../../../glossary.md#term-planning)/design files changed; no code, build, test-source, [normative](../../../../glossary.md#term-normative) or staged edits.
- NN0026/CP0005B and 70 existing completed task rows preserved; old v1/flexible models and holds retained.
- Independent Astra/xhigh reviewer Mendel01a143bd-81ec-7c32-b17a-0eed065338db: no P1; sole P2 was
  missing implicit :backends parent projectDir. Corrected brief/program/ADR to map parent plus four
  children and validate projects without the old directory masking failure under Gradle 9.6.1.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Coordinator accepts this bounded planning/review scope after correction; Engine0020 Ready evidence
  is recorded in its brief. ADR0039 remains Proposed; Model/Engine questions remain unanswered.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Ready glossary](../../../../glossary.md#term-ready); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [Validation](../../../../glossary.md#term-validation): `git diff --check` and the exact supplied Node command above passed: nine files,
  252 local links/23 anchors, zero errors; both briefs below 200 lines/15 KiB. Scope/completed-row
  preservation checked. Proposed examples unchanged; no Java/Javadoc/native/performance runs.<span data-glossary-navigation="true"> ([Node glossary](../../../../glossary.md#node))</span>
- Documentation: decision/planning/general/example profiles; no Java/[Javadoc](../../../../glossary.md#term-javadoc) or glossary change.
- Earlier proposal reviews/189-link evidence remain historical, not acceptance of later [API](../../../../glossary.md#term-application-programming-interface-api) defaults.
- Follow-up: coordinator integrates the frozen packet, reverifies launch revision/dirty inventory
  and inactive conflicts, then separately launches Engine0020. Later semantic choices retain gates.
- Status: Complete — planning/review only; no implementation or native/[device](../../../../glossary.md#term-device) success claimed.
