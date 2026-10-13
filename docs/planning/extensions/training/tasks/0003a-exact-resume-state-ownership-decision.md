# Training 0003A — Decide exact-resume state ownership

## Status

Draft

## Change class

Class C — a claim of exact training continuation crosses NN model state, Training session state,
explicit graph random state, caller-selected inputs, and future durable Checkpoint artifacts.

## Goal

Decide precisely what a saved training run must contain to continue from the next step, who
owns each part, and when the framework may call that continuation exact. This decision prepares
Training 0003 and the optional Training Checkpoint adapter without inventing a data lifecycle.

## Current evidence and proposed decision

`TrainingState` already owns detached optimizer configuration, ordered parameter paths,
logical and master bytes, first/second optimizer slots, pending gradient sums, and execution,
optimizer-step and accumulation counters. `TrainingSession.state()` and `restore()` validate and
install that state; CPU integration tests continue a session after restore. NN owns persistent
model buffers. Graph random state is an explicit Model/Engine value. Input order, a sampler cursor,
epoch, and external random sources are caller-controlled; no Data module or scheduler/scaler API
currently owns them.

Recommended claim: a Training snapshot restores exactly its documented optimizer/session state.
An end-to-end *exact resume* claim additionally requires a matching NN model state and complete
caller-provided next-input position plus every random/scheduler/scaler state used by that program.
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

- Depends on: Training 0002C Complete; review reconciliation; selection of exact-resume scope.
- Conflicts with: Training state/session API, NN state installation, Checkpoint adapter contract,
  and shared root/scoped architecture text.
- Parallel group: None; serialize contract ownership before dependent implementation.
- Common base revision: select when this Draft becomes Ready.
- Integration order: participant/claim decision → owning contracts and ADR → 0003 packet.
- Integration validation: targeted architecture/documentation review and links; executable
  tests only if the decision changes behavior or code.
- Shared-document integration owner: future coordinator.

## Files and symbols

- `TrainingState.java`, `TrainingSession.java` and focused restore/integration tests — existing
  in-memory coverage and limitations.
- NN state dictionary and `GraphRngState` — other current owners; Checkpoint master plan —
  future durable boundary.
- Owning root/scoped contract, training explanation and significant ADR if the selected decision
  changes the durable cross-module claim.

## Acceptance criteria

- One ownership table covers every participant needed for the chosen replay claim, including
  program-specific external state and unsupported participants.
- One concrete save/restore example identifies the next batch and random state after recovery,
  and a failure example shows when the exact-resume claim is refused.
- Training 0003 and Checkpoint 0005–0006 have acyclic, testable acceptance gates with no implied
  Data module or NN-to-Training dependency.
- Independent Class C review approves the decision and affected explanatory documentation.

## Documentation and review impact

This is a planning brief using the planning style profile. The eventual decision changes a
durable cross-module contract, so its architecture/Javadoc/API explanation and independent
review belong to the same decision change.

## Result

Decision pending. No new Training API or exact-resume promise is established by this Draft.
