# Training 0003A — Decide exact-[resume](../../../../glossary.md#schema-serialization-and-checkpoint) state [ownership](../../../../glossary.md#term-ownership-meanings)

> **Editorial reading context — ownership decision, not a durable codec.** Exact next-step
> continuation requires agreement across model parameters/buffers, Training's optimizer and
> accumulation state, effective forward contexts, and the application's next-input cursor and
> affecting external state. A session run counter cannot reconstruct which dataset item comes
> next. An NN local mode also does not prove which explicit context built an objective.
> Today's [caller-coordinated recovery](../../../../reference/training/recovery.md#caller-coordinated-in-memory-recovery)
> names those participants, and [Training state limits](../../../../reference/training/optimizers.md#atomicity-state-concurrency-and-close)
> explain per-owner restore. Their integration responsibilities do not create a transaction across owners or cross-device replay guarantees.

## Status

Complete

## Change class

Class C — a claim of [exact training continuation](../../../../glossary.md#exact-training-continuation) crosses [NN model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) state, [Training session](../../../../glossary.md#training-session--trainingsession) state,
explicit [graph](../../../../glossary.md#graph) random state, caller-selected inputs, and future durable [Checkpoint](../../../../glossary.md#term-checkpoint) [artifacts](../../../../glossary.md#term-artifact).

## Goal

Decide precisely what a saved training [run](../../../../glossary.md#term-run) must contain to continue from the next step, who
owns each part, and when the framework may call that continuation exact. The selected [scope](../../../../glossary.md#term-scope) is
state-complete continuation of the currently supported [`TrainingSession`](../../../../glossary.md#training-session--trainingsession) workflow, not a promise
of bitwise-identical future results across devices or execution routes. This decision disposes
of Training 0003's proposed progress [API](../../../../glossary.md#term-application-programming-interface-api) and prepares Training 0004 and the optional Training
[Checkpoint](../../../../glossary.md#term-checkpoint) adapter without inventing a data [lifecycle](../../../../glossary.md#lifecycle).

## Current evidence and selected decision

`TrainingState` already owns [detached](../../../../glossary.md#detached-model-results) [optimizer](../../../../glossary.md#term-optimizer) configuration, ordered [parameter](../../../../glossary.md#term-parameter) paths,
logical and master bytes, first/second optimizer slots, pending [gradient](../../../../glossary.md#term-gradient) sums, and execution,
optimizer-step and accumulation counters. `TrainingSession.state()` and `restore()` validate and
install that state; [CPU](../../../../glossary.md#term-cpu) integration tests continue a session after restore. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns persistent
[model](../../../../glossary.md#term-model) [buffers](../../../../glossary.md#term-buffer). [Graph](../../../../glossary.md#graph) random state is an explicit Model/[Engine](../../../../glossary.md#term-engine) value. Input order, a sampler cursor,
[epoch](../../../../glossary.md#term-epoch-meanings), and external random sources are caller-controlled; no Data [module](../../../../glossary.md#term-module) or scheduler/scaler [API](../../../../glossary.md#term-application-programming-interface-api)
currently owns them.

Selected claim: a Training [snapshot](../../../../glossary.md#term-snapshot) restores exactly its documented optimizer/session state.
An end-to-end *exact [resume](../../../../glossary.md#schema-serialization-and-checkpoint)* claim additionally requires a matching [NN model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) state and complete
caller-provided next-input position plus every random/scheduler/scaler state used by that program.
The effective `ForwardContext` supplied to mode-sensitive forwards when constructing the fixed
[objective](../../../../glossary.md#term-objective) must also be reproducible; saved `Module.mode()` alone is insufficient. Training-mode
BatchNorm currently lacks proof of a step-aligned detached running-buffer value after a session
[run](../../../../glossary.md#term-run) and is outside the exact-continuation claim. Durable caller-owned participants need supported
versioned bounded [schemas](../../../../glossary.md#term-schema)/codecs and [owner](../../../../glossary.md#term-owner) validators/restorers; no arbitrary-object encoding is
selected.
If any participant cannot be represented and restored, documentation must say restart/partial
restore rather than exact continuation. Exactness applies to the same admitted computation route;
the current numerical contract does not promise cross-[backend](../../../../glossary.md#backend) bitwise replay.

## [Scope](../../../../glossary.md#term-scope) of the decision task

- Inventory every state participant for the supported [`TrainingSession`](../../../../glossary.md#training-session--trainingsession) workflow and name its
  current owner, capture/publication point, restore validation and failure behavior.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [capture glossary](../../../../glossary.md#term-capture-meanings); [publication glossary](../../../../glossary.md#term-publication); [validation glossary](../../../../glossary.md#term-validation))</span>
- Select a minimal handoff for caller-owned progress and explicit graph RNG state. Keep actual
  sampler algorithms and external random generators with their callers. Introduce no generic
  DataLoader, state registry, or opaque unvalidated blob inside <span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [registry glossary](../../../../glossary.md#term-registry); [opaque glossary](../../../../glossary.md#term-opaque))</span>`TrainingState`.
- Decide whether Training 0003 needs a new public value/API or whether existing counters plus an
  explicit caller handoff already suffice. Record the answer before implementing it.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Define how NN model state and Training state are jointly validated before any installation in
  Training 0004 / Checkpoint 0005–0006, without making NN depend on Training.<span data-glossary-navigation="true"> ([NN model glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Review the claimed scope for scheduler and scaler: no such built-in participant exists today;
  support is conditional on caller ownership unless a separate concrete capability is selected.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Non-goals

No [codec](../../../../glossary.md#term-codec), filesystem [publication](../../../../glossary.md#term-publication), [dataset](../../../../glossary.md#term-dataset)/sampler implementation, [device](../../../../glossary.md#term-device)-[buffer](../../../../glossary.md#term-buffer) [serialization](../../../../glossary.md#term-serialization),
[backend](../../../../glossary.md#backend) [optimizer](../../../../glossary.md#term-optimizer) route, or broad performance claim in this decision task.

## Contracts

- `ARCHITECTURE.md` headings `Core lifecycle`, `Core invariants`, `Scope-indexed normative
  contracts`, `Module-ownership routing`, and `Dependency rules`.
- `docs/architecture/contracts/extensions-training.md` headings `extensions/nn`,
  `extensions/training`, and `Optimizer/training lifecycle`.
- `docs/architecture/contracts/foundational-modules.md` heading `modules/model` for explicit
  graph RNG ownership; <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>`docs/architecture/contracts/runtime-prepare-engine.md` heading
  `modules/engine` for public [publication](../../../../glossary.md#term-publication)/[materialization](../../../../glossary.md#materialization) [ownership](../../../../glossary.md#term-ownership-meanings).

If these contracts leave the required [handoff](../../../../glossary.md#term-handoff) ambiguous, make a coordinated architecture
decision before changing code. Do not infer authority from this [planning](../../../../glossary.md#term-planning) brief.

## Dependencies and integration

- Depends on: Training 0002C Complete; project-review reconciliation Complete; owner-selected
  state-complete resume scope (2026-10-13).
- Conflicts with: Training state/session API, NN state installation, Checkpoint adapter contract,
  and shared root/scoped architecture text.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Parallel group: None; serialize contract [ownership](../../../../glossary.md#term-ownership-meanings) before dependent implementation.
- Common base revision: `34fb1c8d6a48f9a9352c25597a86045180477fbc`; serial task, no parallel group.
- Integration order: participant/claim decision → owning contracts and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) → 0004 gate; 0003 superseded.
- Integration validation: independent Class C contract/documentation review, targeted link and
  status checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`git diff --check`; executable tests only if behavior or code changes.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordinator.

Frontier verification (2026-10-13): Training 0002C and the repository-review reconciliation are
Complete in the active plans; no other Training task was [Ready](../../../../glossary.md#term-ready) or in progress; [Checkpoint](../../../../glossary.md#term-checkpoint)
0001–0006 were Draft and did not conflict with this serial decision. The owner accepted the
state-complete [resume](../../../../glossary.md#schema-serialization-and-checkpoint) [scope](../../../../glossary.md#term-scope) after the stable-[softmax](../../../../glossary.md#softmax--log-softmax) checkpoint. This was the sole authorized
Training frontier at the recorded base.

## Files and symbols

- `TrainingState.java`, `TrainingSession.java` and focused restore/integration tests — existing
  in-memory coverage and limitations.
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state dictionary and [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) — other current owners; Checkpoint master plan —
  future durable boundary.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Owning root/scoped contract, training explanation and significant ADR if the selected decision
  changes the durable cross-module claim.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- One ownership table covers every participant needed for the chosen replay claim, including
  program-specific external state, NN buffers/mode, and unsupported participants; distinguish
  state completeness from bitwise numerical replay.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- One concrete save/restore example identifies the next batch and random state after recovery,
  and a failure example shows when the exact-resume claim is refused.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>
- Training 0003's disposition and the Training 0004 / Checkpoint 0005–0006 acceptance gates are
  acyclic and testable, with no implied Data module or NN-to-Training dependency.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Independent Class C review approves the decision and affected explanatory documentation.

## [Validation](../../../../glossary.md#term-validation)

Worker: targeted source/contract/test audit and documentation link checks; `git diff --check`.
No executable code or behavior is planned, so no Java test rerun is required unless the decision
pass changes code or identifies a concrete [regression](../../../../glossary.md#term-regression) risk.

Integration: independent Class C review, then verify status/[dependency](../../../../glossary.md#term-dependency) consistency in this brief,
the Training master plan, [Checkpoint](../../../../glossary.md#term-checkpoint) master plan, and roadmap.

## Documentation and review impact

This is a [planning](../../../../glossary.md#term-planning) brief using the planning style profile. The eventual decision changes a
durable cross-[module](../../../../glossary.md#term-module) contract, so its architecture/[Javadoc](../../../../glossary.md#term-javadoc)/[API](../../../../glossary.md#term-application-programming-interface-api) explanation and independent
review belong to the same decision change.

## Result

The accepted decision keeps `TrainingState` scoped to [detached](../../../../glossary.md#detached-model-results) session state and requires an
explicit caller-owned next-input, external [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), and optional scheduler/scaler [handoff](../../../../glossary.md#term-handoff). [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [buffer](../../../../glossary.md#term-buffer)
values, per-[module](../../../../glossary.md#term-module) modes, and actual forward contexts are separate. The fixed compiled [objective](../../../../glossary.md#term-objective)
does not acquire a per-[run](../../../../glossary.md#term-run) [graph](../../../../glossary.md#graph)-RNG stream. Training-mode BatchNorm cannot be claimed exact until
step-aligned detached buffer [publication](../../../../glossary.md#term-publication) is demonstrated. A future durable exact [artifact](../../../../glossary.md#term-artifact) is
limited to concrete caller-owned participant [schemas](../../../../glossary.md#term-schema) with versioned bounded codecs and [owner](../../../../glossary.md#term-owner)
[validation](../../../../glossary.md#term-validation); arbitrary caller objects are unsupported. No new Training 0003 public [API](../../../../glossary.md#term-application-programming-interface-api) is needed,
so that task is Superseded. Training 0004 must
implement/test joint [preflight](../../../../glossary.md#term-preflight) and an unpublished-candidate [commit](../../../../glossary.md#term-commit) or complete [rollback](../../../../glossary.md#term-rollback) before
[Checkpoint](../../../../glossary.md#term-checkpoint) 0005–0006 may promise [atomic](../../../../glossary.md#resource-transaction-and-rollback) exact recovery.

Architecture/documentation: root [ownership](../../../../glossary.md#term-ownership-meanings) route, only the owning Training scoped contract,
training-graph explanation, accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0030, Training API clarification, `TrainingState`
opening [Javadoc](../../../../glossary.md#term-javadoc), and targeted glossary term. Shared Training/Checkpoint plans and roadmap are
synchronized. Independent Class C review initially blocked one P1 (effective context) and two P2
(caller-state [codec](../../../../glossary.md#term-codec) and BatchNorm buffer) findings. The targeted re-review found no P1/P2; its
remaining P3 [planning](../../../../glossary.md#term-planning)-status inconsistency was corrected, and final re-review returned APPROVE
with no P1/P2/P3 findings. Javadoc impact: the type opening names session-only state;
no executable contract or Java behavior changed. No Java tests were rerun. Neither the future
joint [transaction](../../../../glossary.md#term-transaction) nor any durable codec is implemented or authorized by this task.
Remediation validation: source and focused BatchNorm/Training test inspection; targeted local
link/heading check of 255 links with zero failures; status/[dependency](../../../../glossary.md#term-dependency) search and
`git diff --check` passed. The earlier Training Javadoc validation remains applicable because
the remediation changed documentation only.

Status: Complete
