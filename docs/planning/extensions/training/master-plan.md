# Training Master Plan

## Goal

Define backend-agnostic optimizer algorithms, sessions, and training-step orchestration over
parameters declared by `extensions/nn`.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [NN master plan](../nn/master-plan.md)

## Scope

- public backend-neutral optimizer algorithms
- reusable Engine-backed training sessions and steps
- stable path-based optimizer state and gradient accumulation
- immutable optimizer/session snapshot and validate-before-install restore contracts needed by
  downstream exact-resume persistence
- path-based parameter groups, SGD, and planned backend-neutral Adam/AdamW optimizers

## Out of scope

- direct Runtime, Prepare, tuning, or concrete-backend integration
- kernel selection or backend-specific optimizer execution
- `Module`, `Parameter`, `Buffer`, layer behavior, and train/eval mode
- checkpoint file formats, codecs, filesystem publication, tokenizer artifacts, and backend
  materialization

## Module invariants

- Training owns algorithms and orchestration, not an execution pipeline.
- NN owns `Parameter` and `Buffer`; Training consumes declared parameters for optimization.
- Training intentionally depends on the public Engine facade for one compile/prepared session; it
  never imports Engine internals, Runtime, Prepare, tuning, or concrete backend modules.
- NN never depends on Training.

## Allowed dependencies

- modules/model
- extensions/nn
- the public modules/engine facade

## Forbidden dependencies

- backends/cpu
- backends/metal
- backends/cuda

## Dependency DAG and authorized frontier

`0001E -> 0002A -> 0002B -> 0003 -> 0004 -> 0005`; 0003 also awaits the Data lifecycle and
0004 awaits the Checkpoint model artifact/adapter boundary.
[0002A](tasks/0002a-sgd-parameter-groups.md) is Complete after focused validation, independent
Class B review, and the integration-owner `./gradlew test javadoc` checkpoint.
[0002B](tasks/0002b-adam-and-adamw-optimizers.md) is Complete after independent Class B approval
and the integration-owner repository test/Javadoc checkpoint. No Training task is currently Ready:
0003 also awaits the Data lifecycle. Governing
headings are `ARCHITECTURE.md` scope index, module ownership and dependency rules, plus
`docs/architecture/contracts/extensions-training.md` sections `extensions/nn`,
`extensions/training`, and `Optimizer/training lifecycle`.

The public `ParameterGroup` type in `io.github.pho001.synaptik.training` assigns stable Module
paths to immutable optimizer configuration; 0002B adds `Adam` and `AdamW` in that package and
extends only Training-owned state. NN keeps parameter declaration and traversal ownership.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-public-training-session-and-sgd-lifecycle.md) | Public Training Session and SGD Lifecycle | Complete | Stable NN parameters; published first gradients; Engine reusable sessions | Public reusable session and SGD lifecycle approved after Class C remediation. |
| [0001A](tasks/0001a-bound-parameter-snapshot-byte-sizes.md) | Bound parameter snapshot byte sizes | Complete | 0001 | Capture now rejects unsnapshotable logical or optimizer-width sizes before that parameter's buffers or graph compilation. |
| [0001B](tasks/0001b-remove-repeated-parameter-overlap-scan.md) | Remove repeated parameter overlap scan | Complete | 0001A | Fixed parameter pairs are checked at open; runs retain binding and dynamic-overlap validation. |
| [0001C](tasks/0001c-explicit-native-parameter-copy.md) | Explicit native parameter copy before training | Complete | 0001A–0001B | Caller-owned shared Arena and exact Tensor identities bridge initialized NN parameters to the existing Training storage domain. |
| [0001D](tasks/0001d-adopt-fresh-training-snapshot-payloads.md) | Adopt fresh Training snapshot payloads | Complete | 0001C | Removed four redundant full-size clones per `state()` parameter while preserving immutable detached snapshots. |
| [0001E](tasks/0001e-demand-driven-optimizer-candidates.md) | Demand-driven optimizer candidate buffers | Complete | 0001D | Inactive staging is demand-driven; active slots and rollback remain provisioned. |
| [0002A](tasks/0002a-sgd-parameter-groups.md) | SGD parameter groups | Complete | 0001E | Disjoint path-based SGD overrides have exact state/restore identity and preserve ungrouped behavior. |
| [0002B](tasks/0002b-adam-and-adamw-optimizers.md) | Adam and AdamW optimizer state | Complete | 0002A | Mixed SGD/Adam/AdamW groups have exact two-moment state/restore; independent review and repository checkpoint passed. |
| 0003 | Exact-resume progress state | Draft | 0002B; Data lifecycle | Define epoch/scheduler/graph RNG/data sampler cursor and mixed-precision scaler state only where an exact-resume claim is supportable. |
| 0004 | Training checkpoint integration boundary | Draft | 0003; Checkpoint model artifact and optional Training adapter | Expose one coordinated prevalidated state candidate; keep bytes, files, checksums, and atomic publication outside Training. |
| 0005 | Broader training capability checkpoint | Draft | 0002A–0004 | Validate broader optimizers, state-path stability, exact-resume claims, documentation, and architecture boundaries. |


## Milestones

- Public reusable TrainingSession and persistent SGD
- Broader optimizer and parameter-group contracts
- Exact-resume state and optional checkpoint integration

## Current status

Task [0001](tasks/0001-public-training-session-and-sgd-lifecycle.md) is Complete from exact base
`2e110c7688ead480faa4215ff67f9814fa81f4ad`. Mandatory external Class C review first returned
`BLOCK` with six P1 and one P2 findings. After remediation and full validation, external re-review
returned `APPROVE` with findings `0`. The task provides the first public TrainingSession, SGD,
accumulation, detached in-memory state, strict restore, qualified CPU parameter-carrier coverage,
and the existing Metal objective path recorded by ADR 0018. This does not claim BFLOAT16 or
FLOAT16 Metal training execution; low-precision parameters keep private FLOAT32 masters, momentum,
and accumulated gradients, with exact logical and optimizer-width snapshot payloads. Tasks
0001A–0001E are Complete after focused validation and independent review. Task 0001C also passed
a configured full-repository test/Javadoc run. Task 0001E removed idle inactive staging arrays
and was pushed as `cbd14f3b`. Task 0002A added path-based SGD groups after independent approval
and a successful repository test/Javadoc checkpoint, then was pushed as `19b7785c`. Task 0002B
added mixed SGD/Adam/AdamW groups and exact two-moment state at `10d507a3`, with independent
documentation review fix `23d1e39d` and a successful repository checkpoint. No Training task is
Ready; later exact-resume and checkpoint work remains Draft.

## Open questions

- Task 0001C established an explicit pre-session copy into caller-owned shared native storage for
  initialized NN parameters. Direct native initialization remains a possible later convenience,
  not an implicit `TrainingSession.open` behavior.
- Task 0001E makes only inactive staging candidates demand-driven. Rollback storage remains
  provisioned; any optional rollback policy needs a separate contract decision. Large snapshots
  remain bounded by one Java byte array per payload; chunking needs a separate state
  representation decision.
- Dynamic compiled-input overlap is still checked pairwise on each run. Replacing that check
  requires a separate proof for native, heap, and sliced segment identities and diagnostics.
- Define exact-resume scope for data samplers and external data sources; an unavailable sampler
  cursor must downgrade the claim rather than be silently omitted.
- Define a cross-owner validate/install protocol for model state plus optimizer/session state
  without making NN depend on Training or persistence.

## Decisions made

- [ADR 0018](../../../design/decisions/0018-public-training-session-and-sgd-lifecycle.md) selects
  one borrowed public Engine and one owned reusable inference session rather than a second
  execution pipeline.
- `Parameter` ownership and train/eval behavior belong to `extensions/nn`.
- Training owns optimizer/session snapshot semantics; Checkpoint owns durable encoding and file
  publication. A separate optional Training Checkpoint adapter preserves model-only checkpoint
  use without forcing a Training dependency.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Introducing concrete backend bridges into the training extension.
- Saving optimizer slots without stable parameter paths or restoring one owner before another has
  validated could produce a partially resumed training session.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
