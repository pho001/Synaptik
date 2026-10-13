# Training 0003A — Decide exact-resume state ownership

## Status

Complete

## Change class

Class C — a claim of exact training continuation crosses NN model state, Training session state,
explicit graph random state, caller-selected inputs, and future durable Checkpoint artifacts.

## Goal

Decide precisely what a saved training run must contain to continue from the next step, who
owns each part, and when the framework may call that continuation exact. The selected scope is
state-complete continuation of the currently supported `TrainingSession` workflow, not a promise
of bitwise-identical future results across devices or execution routes. This decision disposes
of Training 0003's proposed progress API and prepares Training 0004 and the optional Training
Checkpoint adapter without inventing a data lifecycle.

## Current evidence and selected decision

`TrainingState` already owns detached optimizer configuration, ordered parameter paths,
logical and master bytes, first/second optimizer slots, pending gradient sums, and execution,
optimizer-step and accumulation counters. `TrainingSession.state()` and `restore()` validate and
install that state; CPU integration tests continue a session after restore. NN owns persistent
model buffers. Graph random state is an explicit Model/Engine value. Input order, a sampler cursor,
epoch, and external random sources are caller-controlled; no Data module or scheduler/scaler API
currently owns them.

Selected claim: a Training snapshot restores exactly its documented optimizer/session state.
An end-to-end *exact resume* claim additionally requires a matching NN model state and complete
caller-provided next-input position plus every random/scheduler/scaler state used by that program.
The effective `ForwardContext` supplied to mode-sensitive forwards when constructing the fixed
objective must also be reproducible; saved `Module.mode()` alone is insufficient. Training-mode
BatchNorm currently lacks proof of a step-aligned detached running-buffer value after a session
run and is outside the exact-continuation claim. Durable caller-owned participants need supported
versioned bounded schemas/codecs and owner validators/restorers; no arbitrary-object encoding is
selected.
If any participant cannot be represented and restored, documentation must say restart/partial
restore rather than exact continuation. Exactness applies to the same admitted computation route;
the current numerical contract does not promise cross-backend bitwise replay.

## Scope of the decision task

- Inventory every state participant for the supported `TrainingSession` workflow and name its
  current owner, capture/publication point, restore validation and failure behavior.
- Select a minimal handoff for caller-owned progress and explicit graph RNG state. Keep actual
  sampler algorithms and external random generators with their callers. Introduce no generic
  DataLoader, state registry, or opaque unvalidated blob inside `TrainingState`.
- Decide whether Training 0003 needs a new public value/API or whether existing counters plus an
  explicit caller handoff already suffice. Record the answer before implementing it.
- Define how NN model state and Training state are jointly validated before any installation in
  Training 0004 / Checkpoint 0005–0006, without making NN depend on Training.
- Review the claimed scope for scheduler and scaler: no such built-in participant exists today;
  support is conditional on caller ownership unless a separate concrete capability is selected.

## Non-goals

No codec, filesystem publication, dataset/sampler implementation, device-buffer serialization,
backend optimizer route, or broad performance claim in this decision task.

## Contracts

- `ARCHITECTURE.md` headings `Core lifecycle`, `Core invariants`, `Scope-indexed normative
  contracts`, `Module-ownership routing`, and `Dependency rules`.
- `docs/architecture/contracts/extensions-training.md` headings `extensions/nn`,
  `extensions/training`, and `Optimizer/training lifecycle`.
- `docs/architecture/contracts/foundational-modules.md` heading `modules/model` for explicit
  graph RNG ownership; `docs/architecture/contracts/runtime-prepare-engine.md` heading
  `modules/engine` for public publication/materialization ownership.

If these contracts leave the required handoff ambiguous, make a coordinated architecture
decision before changing code. Do not infer authority from this planning brief.

## Dependencies and integration

- Depends on: Training 0002C Complete; project-review reconciliation Complete; owner-selected
  state-complete resume scope (2026-10-13).
- Conflicts with: Training state/session API, NN state installation, Checkpoint adapter contract,
  and shared root/scoped architecture text.
- Parallel group: None; serialize contract ownership before dependent implementation.
- Common base revision: `34fb1c8d6a48f9a9352c25597a86045180477fbc`; serial task, no parallel group.
- Integration order: participant/claim decision → owning contracts and ADR → 0004 gate; 0003 superseded.
- Integration validation: independent Class C contract/documentation review, targeted link and
  status checks, and `git diff --check`; executable tests only if behavior or code changes.
- Shared-document integration owner: main coordinator.

Frontier verification (2026-10-13): Training 0002C and the repository-review reconciliation are
Complete in the active plans; no other Training task was Ready or in progress; Checkpoint
0001–0006 were Draft and did not conflict with this serial decision. The owner accepted the
state-complete resume scope after the stable-softmax checkpoint. This was the sole authorized
Training frontier at the recorded base.

## Files and symbols

- `TrainingState.java`, `TrainingSession.java` and focused restore/integration tests — existing
  in-memory coverage and limitations.
- NN state dictionary and `GraphRngState` — other current owners; Checkpoint master plan —
  future durable boundary.
- Owning root/scoped contract, training explanation and significant ADR if the selected decision
  changes the durable cross-module claim.

## Acceptance criteria

- One ownership table covers every participant needed for the chosen replay claim, including
  program-specific external state, NN buffers/mode, and unsupported participants; distinguish
  state completeness from bitwise numerical replay.
- One concrete save/restore example identifies the next batch and random state after recovery,
  and a failure example shows when the exact-resume claim is refused.
- Training 0003's disposition and the Training 0004 / Checkpoint 0005–0006 acceptance gates are
  acyclic and testable, with no implied Data module or NN-to-Training dependency.
- Independent Class C review approves the decision and affected explanatory documentation.

## Validation

Worker: targeted source/contract/test audit and documentation link checks; `git diff --check`.
No executable code or behavior is planned, so no Java test rerun is required unless the decision
pass changes code or identifies a concrete regression risk.

Integration: independent Class C review, then verify status/dependency consistency in this brief,
the Training master plan, Checkpoint master plan, and roadmap.

## Documentation and review impact

This is a planning brief using the planning style profile. The eventual decision changes a
durable cross-module contract, so its architecture/Javadoc/API explanation and independent
review belong to the same decision change.

## Result

The accepted decision keeps `TrainingState` scoped to detached session state and requires an
explicit caller-owned next-input, external RNG, and optional scheduler/scaler handoff. NN buffer
values, per-module modes, and actual forward contexts are separate. The fixed compiled objective
does not acquire a per-run graph-RNG stream. Training-mode BatchNorm cannot be claimed exact until
step-aligned detached buffer publication is demonstrated. A future durable exact artifact is
limited to concrete caller-owned participant schemas with versioned bounded codecs and owner
validation; arbitrary caller objects are unsupported. No new Training 0003 public API is needed,
so that task is Superseded. Training 0004 must
implement/test joint preflight and an unpublished-candidate commit or complete rollback before
Checkpoint 0005–0006 may promise atomic exact recovery.

Architecture/documentation: root ownership route, only the owning Training scoped contract,
training-graph explanation, accepted ADR 0030, Training API clarification, `TrainingState`
opening Javadoc, and targeted glossary term. Shared Training/Checkpoint plans and roadmap are
synchronized. Independent Class C review initially blocked one P1 (effective context) and two P2
(caller-state codec and BatchNorm buffer) findings. The targeted re-review found no P1/P2; its
remaining P3 planning-status inconsistency was corrected, and final re-review returned APPROVE
with no P1/P2/P3 findings. Javadoc impact: the type opening names session-only state;
no executable contract or Java behavior changed. No Java tests were rerun. Neither the future
joint transaction nor any durable codec is implemented or authorized by this task.
Remediation validation: source and focused BatchNorm/Training test inspection; targeted local
link/heading check of 255 links with zero failures; status/dependency search and
`git diff --check` passed. The earlier Training Javadoc validation remains applicable because
the remediation changed documentation only.

Status: Complete
