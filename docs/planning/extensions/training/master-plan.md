# Training Master Plan

## Goal

Maintain [backend](../../../glossary.md#backend)-agnostic [optimizers](../../../glossary.md#term-optimizer)/session state and generalize [parameter](../../../glossary.md#term-parameter) [admission](../../../glossary.md#term-admission) independently
of [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context). Training0007 implements physical `modules/training` with a small live named-parameter
boundary consumed by NN; final independent approval and the coordinator gate passed.
The [cross-area program](../../model-execution-refactor.md) owns cutover phases and gates.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>
- [NN master plan](../nn/master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## [Scope](../../../glossary.md#term-scope)

- public [backend](../../../glossary.md#backend)-neutral [optimizer](../../../glossary.md#term-optimizer) algorithms
- reusable [Engine](../../../glossary.md#term-engine)-backed training sessions and steps
- stable path-based [optimizer state](../../../glossary.md#term-optimizer-state) and [gradient accumulation](../../../glossary.md#term-gradient-accumulation)
- immutable optimizer/session snapshot and validate-before-install restore contracts needed by
  downstream exact-resume persistence<span data-glossary-navigation="true"> ([immutable glossary](../../../glossary.md#term-immutable); [snapshot glossary](../../../glossary.md#term-snapshot); [resume glossary](../../../glossary.md#schema-serialization-and-checkpoint); [persistence glossary](../../../glossary.md#term-persistence))</span>
- path-based [parameter](../../../glossary.md#term-parameter) groups and backend-neutral [SGD](../../../glossary.md#term-stochastic-gradient-descent-sgd)/[Adam](../../../glossary.md#adaptive-optimizer--adam-adamw)/[AdamW](../../../glossary.md#adaptive-optimizer--adam-adamw) optimizers

## Out of [scope](../../../glossary.md#term-scope)

- direct [Runtime](../../../glossary.md#term-runtime), [Prepare](../../../glossary.md#prepare), tuning, or concrete-[backend](../../../glossary.md#backend) integration
- [kernel](../../../glossary.md#kernel) selection or backend-specific [optimizer](../../../glossary.md#term-optimizer) execution
- [`Module`](../../../glossary.md#term-module), [`Parameter`](../../../glossary.md#term-parameter), [`Buffer`](../../../glossary.md#term-buffer), layer behavior, and train/[eval mode](../../../glossary.md#inference-and-evaluation)
- checkpoint file formats, codecs, filesystem publication, tokenizer artifacts, and backend
  materialization<span data-glossary-navigation="true"> ([checkpoint glossary](../../../glossary.md#term-checkpoint); [publication glossary](../../../glossary.md#term-publication); [tokenizer glossary](../../../glossary.md#frozen-whitespace-tokenizer); [artifacts glossary](../../../glossary.md#term-artifact); [materialization glossary](../../../glossary.md#materialization))</span>

## [Module](../../../glossary.md#term-module) invariants

Current [ownership](../../../glossary.md#term-ownership-meanings) follows the root/scoped contract enacted by Training0007.

- Training owns algorithms and orchestration, not an execution pipeline.
- [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns [`Parameter`](../../../glossary.md#term-parameter) and [`Buffer`](../../../glossary.md#term-buffer); Training consumes generic live [`TrainingParameters`](../../../glossary.md#live-parameter-source--trainingparameters).
- Training intentionally depends on the public Engine facade for one compile/prepared session; it
  never imports Engine internals, Runtime, Prepare, tuning, or concrete backend modules.<span data-glossary-navigation="true"> ([Engine glossary](../../../glossary.md#term-engine); [facade glossary](../../../glossary.md#term-facade); [compile glossary](../../../glossary.md#compile); [Runtime glossary](../../../glossary.md#term-runtime); [Prepare glossary](../../../glossary.md#prepare); [concrete backend glossary](../../../glossary.md#concrete-backend))</span>
- NN depends on Training only for its live Module [admission](../../../glossary.md#term-admission) adapter, not [optimizer](../../../glossary.md#term-optimizer) ownership.

## Allowed dependencies

Current direct public project edges (later [targets](../../../glossary.md#term-target) remain in the program):

- [modules](../../../glossary.md#term-module)/[model](../../../glossary.md#term-model)
- the public modules/[engine](../../../glossary.md#term-engine) [facade](../../../glossary.md#term-facade)

## Forbidden dependencies

- [backends](../../../glossary.md#backend)/cpu
- backends/[metal](../../../glossary.md#term-metal)
- backends/cuda

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and [authorized frontier](../../../glossary.md#term-authorized-frontier)

Completed correctness/recovery chain:
`0001E → 0002A → 0002B → 0002C → 0003A → 0004`.
Training 0004 depends on completed [Checkpoint](../../../glossary.md#term-checkpoint) [model](../../../glossary.md#term-model) [persistence](../../../glossary.md#term-persistence) and [CPU](../../../glossary.md#term-cpu) 0024; it does not depend
on the future optional adapter. Existing broader Training capability checkpoint 0005 stays Draft.
Training-owned [Checkpoint 0005 reconstruction](../checkpoint/tasks/0005-training-state-reconstruction.md)
and [Checkpoint 0005A envelope composition](../checkpoint/tasks/0005a-model-envelope-composition.md)
are Complete; they do not authorize new execution.

Revised proposed product in [ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>:
automatic usable-model load and optional owned [resume](../../../glossary.md#schema-serialization-and-checkpoint), not [snapshot](../../../glossary.md#term-snapshot)-only Model + TrainingState.
[ADR 0037](../../../design/decisions/0037-stored-training-checkpoint-adapter.md) is withdrawn Proposed;
raw state codecs/comparison remain useful. Earlier proposed DAG (now held):
`NN0026 Complete → CP0005B → NN0027 semantics → CP0005C persistence → Training0006 → optional CP0006`;
completed Checkpoint 0005A also precedes the adapter. No cycle or unrelated 0005 prerequisite.

[ADR0039](../../../design/decisions/0039-portable-graph-model-persistence.md) now records the five
accepted product choices and approved P2 core/[API](../../../glossary.md#term-application-programming-interface-api) packet. [CP0005D](../checkpoint/tasks/0005d-portable-graph-persistence-decision.md)
and Engine0020 remain Complete. Engine0021 is now Complete after independent Class C approval
and the native-enabled integrated test/[Javadoc](../../../glossary.md#term-javadoc) gate; its brief owns evidence.
[Training0007](tasks/0007-generic-live-parameter-admission.md) is Complete after Pascal's final
independent APPROVE and the coordinator's native-enabled test/Javadoc gate. Compiler0008 and
Checkpoint0008 are also Complete after final independent reviews and coordinator gates.
Authorized Training implementation frontier: None. [Training0008](tasks/0008-fresh-value-training-session-runs.md)
is Complete after Euclid's definitive code/docs APPROVE and main's source-matched root gate
(2026-10-19); its brief owns evidence. NN0028 is Complete, accepted 2026-10-19.
The [program DAG](../../model-execution-refactor.md#phases-and-owner-dag)
replaces recurrent-first ordering for the new [graph](../../../glossary.md#graph) path, not the existing flexible declarative API.

CURRENT [TrainingSession](../../../glossary.md#training-session--trainingsession) owns [Engine](../../../glossary.md#term-engine) [PreparedExecution](../../../glossary.md#prepared-execution--preparedexecution) after Engine0021 mechanically changed its
field/constructor, open to engine.[prepare](../../../glossary.md#prepare)(graph), execution to
engine.[run](../../../glossary.md#term-run)(prepared, inputs), and closed/[admission](../../../glossary.md#term-admission)/cleanup checks to prepared.isClosed/close.
Keep public TrainingSession: [optimizer](../../../glossary.md#term-optimizer)/accumulation state, counters, busy admission and [atomic](../../../glossary.md#resource-transaction-and-rollback)
transactions justify it. Preserve live/native/[precommit](../../../glossary.md#term-precommit) checks, result-before-prepared-before-[seed](../../../glossary.md#term-seed)
cleanup and failed-open [rollback](../../../glossary.md#term-rollback). No new Training API or project edge in 0021; Training [fixture](../../../glossary.md#term-fixture)
session() accessors remain. Training0007/0008 below are not pulled into this migration.

Training0007 implementation: [modules](../../../glossary.md#term-module)/training, retaining :extensions:training via projectDir and
the training Java package, api(Model, Engine) only. This complete generic admission/native-copy
capability removes [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) imports in TrainingSession/TrainingParameter/[TrainingParameterStorage](../../../glossary.md#native-parameter-copy--trainingparameterstorage),
adds [TrainingParameters](../../../glossary.md#live-parameter-source--trainingparameters) and migrates callers. NN's Module.trainingParameters() adapter is included
in that atomic boundary cutover, removing Training→NN with NN→Training. No retained NN-taking
overloads inside generic Training and no compatibility project.

TrainingParameters has ordered Entry(path, [Tensor](../../../glossary.md#tensor)) values and live validateUnchanged().
The NN adapter rechecks current path/order/wrapper/Tensor membership, not a [detached](../../../glossary.md#detached-model-results) map.
Training retains native [descriptor](../../../glossary.md#term-descriptor)/storage/finiteness and precommit checks, optimizer algorithms,
slots/counters, rollback and [failure atomicity](../../../glossary.md#term-failure-atomicity). copyToSharedNative accepts this live source.
It is still an explicit pre-session copy into a caller-owned shared [Arena](../../../glossary.md#term-arena); the high-level Model
must first acquire its own [weight](../../../glossary.md#term-weight) leaves, never [rebind](../../../glossary.md#capture-graph-ports-and-reconstruction) external [borrowed](../../../glossary.md#ownership-borrowing-and-lease) Tensors.

Training0008 delivered fresh stored values in `inputs()` order, excluding [parameters](../../../glossary.md#term-parameter) and private seed,
using Engine0021. Existing run remains exact-[identity](../../../glossary.md#term-identity-meanings)/arbitrary-order. The bounded root/owning
Training [lifecycle](../../../glossary.md#lifecycle) cutover, implementation, independent review and coordinator gate are Complete.
No per-[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) graph build/copy/[compile](../../../glossary.md#compile)/prepare or [autotuning](../../../glossary.md#autotuning--model-autotuning) integration was added.
In completed NN0028, configured train(inputs,[targets](../../../glossary.md#term-target)) delegates RESET_AND_STEP over the same owned weights
used by compute/save. Module.train() remains only a mode toggle. [Heap](../../../glossary.md#term-heap)-to-native admission may
copy once and invalidate old [inference](../../../glossary.md#term-inference) preparations before creating the training [objective](../../../glossary.md#term-objective)/session.
Loaded/owned values need no [ownership](../../../glossary.md#term-ownership-meanings)-only duplicate; no automatic optimizer/[RNG](../../../glossary.md#graph-rng-state--graphrngstate)/cursor resume.

Training0007 depends on reviewed P2 and completed session/native-copy/state capabilities, not
Engine fresh [binding](../../../glossary.md#term-binding) or a [codec](../../../glossary.md#term-codec); it builds on 0021's migrated consumer in serial integration.
Training0008 depends on 0007 and Engine0021 and uses [Engine.runValues](../../../glossary.md#fresh-value-binding--enginerunvalues), not a removed session API. Serial integration:
Engine0021 → Training0007 → Compiler0008 → Checkpoint0008 → Training0008 → NN0028.
The shared-document order is not an invented code dependency. Parallel group None; conflicts are
NN0028 and shared root/scoped Training lifecycle and documentation writers. Current common base:
main `7386249479ccda991b96a7e27bd5ed0279ca1ae3`, including the completed Checkpoint0008 serial join.
The Training0008 brief records integration [metadata](../../../glossary.md#term-metadata), launch verification, Euclid's final APPROVE
and main's accepted source-matched root gate (2026-10-19). Model-execution coordinator owns shared
documents and combined [validation](../../../glossary.md#term-validation); implementation is complete, with no authorized Training frontier.

Hold NN0027/CP0005C/Training0006/adapter expansion as optional older proposals, not prerequisites.
TrainingRecipe/TrainingRun and nested [capture](../../../glossary.md#term-capture-meanings) borrow remain held, not accepted additions.
The first new configured graph training API uses a [scalar objective](../../../glossary.md#loss) builder once; arbitrary Java
callbacks are not persisted. Existing native/[gradient](../../../glossary.md#term-gradient)/[backend](../../../glossary.md#backend) admission remains [fail-closed](../../../glossary.md#term-fail-closed).
No RecurrentScan [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt), Data, changing [dropout](../../../glossary.md#dropout--inverted-dropout) stream or broad Training0005 gate is added.

The 0007 package map moves all existing production types unchanged in package to
`modules/training/src/main/java/io/github/pho001/synaptik/training`; adds public
`TrainingParameters` and nested `Entry` there; adds only `Module.trainingParameters()` and its
private live implementation in `io.github.pho001.synaptik.nn.module`. The brief enumerates types,
consumers, tests, coordinated authority and documentation [scope](../../../glossary.md#term-scope). No [Compiler](../../../glossary.md#term-compiler) consumer was found.

Worker validation covers affected modules once; independent targeted Class C review
reuses successful evidence. Coordinator runs one combined repository checkpoint per integrated
boundary-changing capability with the existing configured [Metal](../../../glossary.md#term-metal) [dylib](../../../glossary.md#term-dylib). The 0007 brief records
approved readiness, implementation and evidence; only its bounded root/scoped cutover is enacted.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-public-training-session-and-sgd-lifecycle.md) | Public [Training Session](../../../glossary.md#training-session--trainingsession) and [SGD](../../../glossary.md#term-stochastic-gradient-descent-sgd) [Lifecycle](../../../glossary.md#lifecycle) | Complete | Stable [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [parameters](../../../glossary.md#term-parameter); published first [gradients](../../../glossary.md#term-gradient); [Engine](../../../glossary.md#term-engine) reusable sessions | Public reusable session and SGD lifecycle approved after Class C remediation. |
| [0001A](tasks/0001a-bound-parameter-snapshot-byte-sizes.md) | Bound parameter [snapshot](../../../glossary.md#term-snapshot) byte sizes | Complete | 0001 | [Capture](../../../glossary.md#term-capture-meanings) now rejects unsnapshotable logical or [optimizer](../../../glossary.md#term-optimizer)-width sizes before that parameter's [buffers](../../../glossary.md#term-buffer) or [graph](../../../glossary.md#graph) [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names). |
| [0001B](tasks/0001b-remove-repeated-parameter-overlap-scan.md) | Remove repeated parameter overlap [scan](../../../glossary.md#term-cumulative-scan) | Complete | 0001A | Fixed parameter pairs are checked at open; runs retain [binding](../../../glossary.md#term-binding) and dynamic-overlap [validation](../../../glossary.md#term-validation). |
| [0001C](tasks/0001c-explicit-native-parameter-copy.md) | Explicit [native parameter copy](../../../glossary.md#native-parameter-copy--trainingparameterstorage) before training | Complete | 0001A–0001B | Caller-owned shared [Arena](../../../glossary.md#term-arena) and exact [Tensor](../../../glossary.md#tensor) [identities](../../../glossary.md#term-reference-identity) bridge initialized NN parameters to the existing Training storage domain. |
| [0001D](tasks/0001d-adopt-fresh-training-snapshot-payloads.md) | Adopt fresh Training snapshot payloads | Complete | 0001C | Removed four redundant full-size clones per `state()` parameter while preserving [immutable](../../../glossary.md#term-immutable) [detached](../../../glossary.md#detached-model-results) snapshots. |
| [0001E](tasks/0001e-demand-driven-optimizer-candidates.md) | Demand-driven optimizer candidate buffers | Complete | 0001D | Inactive staging is demand-driven; active slots and [rollback](../../../glossary.md#term-rollback) remain provisioned. |
| [0002A](tasks/0002a-sgd-parameter-groups.md) | SGD parameter groups | Complete | 0001E | Disjoint path-based SGD overrides have exact state/restore [identity](../../../glossary.md#term-identity-meanings) and preserve ungrouped behavior. |
| [0002B](tasks/0002b-adam-and-adamw-optimizers.md) | [Adam](../../../glossary.md#adaptive-optimizer--adam-adamw) and [AdamW](../../../glossary.md#adaptive-optimizer--adam-adamw) [optimizer state](../../../glossary.md#term-optimizer-state) | Complete | 0002A | Mixed SGD/Adam/AdamW groups have exact two-moment state/restore; independent review and repository [checkpoint](../../../glossary.md#term-checkpoint) passed. |
| [0002C](tasks/0002c-adaptive-update-operation-order.md) | Adaptive-update [operation](../../../glossary.md#operation) order | Complete | 0002B; project review | Preserve documented multiply-before-divide order and [finite](../../../glossary.md#term-finite)-update [admission](../../../glossary.md#term-admission) for Adam/AdamW. |
| [0003A](tasks/0003a-exact-resume-state-ownership-decision.md) | [Resume](../../../glossary.md#schema-serialization-and-checkpoint)-state [ownership](../../../glossary.md#term-ownership-meanings) and claim decision | Complete | 0002C; audit reconciliation | [ADR](../../../glossary.md#term-architecture-decision-record-adr) 0030 accepts bounded exact continuation and the future joint restore gate; independent Class C re-review approved. |
| 0003 | Exact-resume progress state | Superseded | 0003A decision | No new Training progress [API](../../../glossary.md#term-application-programming-interface-api): caller owns next input, sampler/[epoch](../../../glossary.md#term-epoch-meanings), external [RNG](../../../glossary.md#graph-rng-state--graphrngstate), scheduler, and scaler [handoff](../../../glossary.md#term-handoff). |
| [0004](tasks/0004-joint-restore-preflight-and-candidate-commit.md) | Joint restore on an unpublished candidate | Complete | 0003A; Checkpoint 0001 and 0004; [CPU](../../../glossary.md#term-cpu) 0024 | Bounded same-route in-memory private-candidate restore proved in integration tests; independent Class C review and repository gate passed. No production [facade](../../../glossary.md#term-facade) or durable [artifact](../../../glossary.md#term-artifact). |
| 0005 | Broader training capability checkpoint | Draft | 0002A–0004 | Validate broader optimizers, state-path stability, exact-resume claims, documentation, architecture boundaries, and a representative peak-memory measurement of parameter, slot, candidate, rollback, and snapshot storage before claiming large-[model](../../../glossary.md#term-model) support. |
| [0006](tasks/0006-supported-recipe-and-owned-run.md) | Supported recipe and owned reusable [run](../../../glossary.md#term-run) | Draft | NN0026/0027; CP0005B/0005C; completed Training0004/Checkpoint0005; future [owner](../../../glossary.md#term-owner) contract | Proposed Training [loss](../../../glossary.md#loss)/[signature](../../../glossary.md#term-signature)/[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/resource lifecycle; no file I/O. |
| [0007](tasks/0007-generic-live-parameter-admission.md) | Generic live parameter admission and native-copy cutover | Complete | Approved P2; completed 0001C/0002C/0004 and Training-owned CP0005; Engine0021 serial base | Generic admission/native copy, relocation and NN adapter delivered; final independent APPROVE and coordinator gate passed. |
| [0008](tasks/0008-fresh-value-training-session-runs.md) | Fresh-value [TrainingSession](../../../glossary.md#training-session--trainingsession) runs | Complete | 0007; Engine0021; Checkpoint0008 serial join | Fresh positional runs delivered; Euclid final APPROVE and main source-matched root gate passed; evidence in brief. |

## Milestones

- Public reusable [TrainingSession](../../../glossary.md#training-session--trainingsession) and persistent [SGD](../../../glossary.md#term-stochastic-gradient-descent-sgd)
- Broader [optimizer](../../../glossary.md#term-optimizer) and [parameter](../../../glossary.md#term-parameter)-group contracts
- Exact-[resume](../../../glossary.md#schema-serialization-and-checkpoint) state and optional [checkpoint](../../../glossary.md#term-checkpoint) integration

## Current status

Sessions, [SGD](../../../glossary.md#term-stochastic-gradient-descent-sgd)/[Adam](../../../glossary.md#adaptive-optimizer--adam-adamw)/[AdamW](../../../glossary.md#adaptive-optimizer--adam-adamw) groups, [detached](../../../glossary.md#detached-model-results) state, defensive reconstruction and bounded same-route
private-candidate restore are implemented. Completed evidence lives in linked briefs.
ModelCheckpoint direct [leaf](../../../glossary.md#term-leaf) save/re-save and public envelope composition are also Complete.
NN0026's whole-[model](../../../glossary.md#term-model) semantic specification and CP0005B whole-model [persistence](../../../glossary.md#term-persistence) are implemented;
no durable adapter or supported recipe/[run](../../../glossary.md#term-run) convenience is implemented.

The user revised [checkpoint](../../../glossary.md#term-checkpoint) direction (2026-10-14): normal load returns a usable supported model,
with [inference](../../../glossary.md#term-inference) or automatic Training-owned [resume](../../../glossary.md#schema-serialization-and-checkpoint). [ADR 0038](../../../design/decisions/0038-declarative-model-and-training-resume.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>
and the new Draft [owner](../../../glossary.md#term-owner) capabilities replace the [snapshot](../../../glossary.md#term-snapshot)-only primary UX. External cursor/[RNG](../../../glossary.md#graph-rng-state--graphrngstate)/
scheduler/scaler remain application-owned, not automatic exact continuation.
BatchNorm training-[buffer](../../../glossary.md#term-buffer) [alignment](../../../glossary.md#term-alignment), fixed RecurrentScan [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt), [Conv3d](../../../glossary.md#conv3d--ncdhw) [gradients](../../../glossary.md#term-gradient) and changing [dropout](../../../glossary.md#dropout--inverted-dropout) streams remain
[fail-closed](../../../glossary.md#term-fail-closed). Current native-copy/session APIs remain unchanged and available for advanced use.
Ordinary static [unrolling](../../../glossary.md#term-unrolling) depends on constituent-[operation](../../../glossary.md#operation) adjoints, not a blanket recurrent ban.

## Open questions

- Training0007/0008 are Complete; P2 and the five product choices are accepted. Authorized Training
  frontier: None. NN0028 and Model/withEngine are Complete, accepted 2026-10-19.
  Durable recipe/restore/capture remains a
  separate held proposal; broader Training0005 is not a prerequisite.<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model); [capture glossary](../../../glossary.md#term-capture-meanings))</span>
- Preserve existing native-only update domain and duplicate-path/overlap rejection; do not quietly
  split shared parameters or widen backend/gradient capability.<span data-glossary-navigation="true"> ([parameters glossary](../../../glossary.md#term-parameter); [backend glossary](../../../glossary.md#backend); [gradient glossary](../../../glossary.md#term-gradient))</span>
- Rollback storage remains provisioned; optional rollback policy requires a separate contract
  decision and must not be silently disabled as a memory optimization.<span data-glossary-navigation="true"> ([Rollback glossary](../../../glossary.md#term-rollback))</span>
- Snapshot payloads remain bounded by individual Java byte arrays; chunking requires a separate
  state-representation decision, not merely larger encoded limits.<span data-glossary-navigation="true"> ([Snapshot glossary](../../../glossary.md#term-snapshot))</span>
- Dynamic compiled-input overlap remains pairwise per run; replacing it requires proof for native,
  heap and sliced segment identities and preserved diagnostics.<span data-glossary-navigation="true"> ([identities glossary](../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([run glossary](../../../glossary.md#term-run); [heap glossary](../../../glossary.md#term-heap))</span>
- Caller progress/RNG and large-model aggregate memory policy remain outside this artifact.
  Stored encoded-byte limits do not establish a peak-memory budget.<span data-glossary-navigation="true"> ([RNG glossary](../../../glossary.md#graph-rng-state--graphrngstate); [artifact glossary](../../../glossary.md#term-artifact))</span>

## Decisions made

- [ADR 0018](../../../design/decisions/0018-public-training-session-and-sgd-lifecycle.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span> selects
  one borrowed public Engine and, after Engine0021's owner reconciliation, one owned prepared execution rather than a second
  execution pipeline.<span data-glossary-navigation="true"> ([borrowed glossary](../../../glossary.md#ownership-borrowing-and-lease); [Engine glossary](../../../glossary.md#term-engine); [owner glossary](../../../glossary.md#term-owner); [prepared execution glossary](../../../glossary.md#prepared-execution--preparedexecution))</span>
- [`Parameter`](../../../glossary.md#term-parameter) [ownership](../../../glossary.md#term-ownership-meanings) and train/eval behavior belong to `extensions/nn`.
- [ADR 0030](../../../design/decisions/0030-exact-training-resume-state-ownership.md) keeps
  caller next-input/progress and external state outside `TrainingState`; the fixed session does
  not advance graph RNG between runs, and no new Training 0003 public API is selected.<span data-glossary-navigation="true"> ([graph glossary](../../../glossary.md#graph); [RNG glossary](../../../glossary.md#graph-rng-state--graphrngstate); [API glossary](../../../glossary.md#term-application-programming-interface-api))</span>
- [ADR 0035](../../../design/decisions/0035-private-candidate-training-restore.md) selects the
  private-candidate validation/publication order for Training 0004.<span data-glossary-navigation="true"> ([validation glossary](../../../glossary.md#term-validation); [publication glossary](../../../glossary.md#term-publication))</span>
- Training owns optimizer/session snapshot semantics; Checkpoint owns durable encoding and file
  publication. A separate optional Training Checkpoint adapter preserves model-only checkpoint
  use without forcing a Training dependency.<span data-glossary-navigation="true"> ([optimizer glossary](../../../glossary.md#term-optimizer); [snapshot glossary](../../../glossary.md#term-snapshot); [Checkpoint glossary](../../../glossary.md#term-checkpoint); [model glossary](../../../glossary.md#term-model); [dependency glossary](../../../glossary.md#term-dependency))</span>
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Introducing [concrete backend](../../../glossary.md#concrete-backend) bridges into the training extension.
- Saving optimizer slots without stable parameter paths or restoring one owner before another has
  validated could produce a partially resumed training session.<span data-glossary-navigation="true"> ([optimizer glossary](../../../glossary.md#term-optimizer); [parameter glossary](../../../glossary.md#term-parameter); [owner glossary](../../../glossary.md#term-owner); [training session glossary](../../../glossary.md#training-session--trainingsession))</span>

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
