# Task NN0028: Owned Model compute training and persistence

## Status

Complete — coordinator accepted 2026-10-19 after definitive independent approval and source-matched native-enabled root gate.
Authorized NN frontier: None. No successor is launched.

## Change class

Class C — public Model lifecycle, owned storage, Engine/Training composition, exclusion and persistence.

## Goal

Give application-built and file-restored NN Models one owned working-weight set for compute/train/save.

## Scope

Implement [ADR0039 Selected API and type feasibility](../../../../design/decisions/0039-portable-graph-model-persistence.md#selected-api-and-type-feasibility),
including its single/list forms and borrowed-Engine overloads: Model stays abstract and generic,
gains AutoCloseable; defineTensor(signature, definition)/load return Model<Tensor,Tensor>;
graph/loadGraph return Model<List<Tensor>,List<Tensor>>. Add same-instance withEngine, terminal
compute, configureTraining, RESET_AND_STEP train, bounded list compute, save and close. Execution
returns HostTensorValue/TrainingStep, never a cast to generic O.

Source admission remains static ADD/MUL/NEG/MATMUL/PERMUTE[1,0]; seven raw dtypes persist independently
of execution/training support. Objectives use current Compiler support but are not persisted.
Retain generic/flexible APIs, mode-only Module.train() and old model.v1/whole-model.v1 readers.

## Non-goals

No GraphProgram, LoadedGraph, InferenceSession, ModelExecutor, public subtype/result/session hierarchy,
new IR, upstream API/build/dependency/native changes, backend additions, codec widening, optimizer/RNG/
data-cursor resume, Data infrastructure, backpropagation through time (BPTT), numerical/performance
campaign or Model/Training autotuning. Advanced Engine.prepareTuned remains separate.

## Contracts

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts): Core lifecycle;
  Core invariants (immutable Tensor/ownership); Module-ownership routing; Dependency rules;
  Documentation authority and architecture updates; Testing requirements.
- [extensions-training](../../../../architecture/contracts/extensions-training.md#extensionsnn):
  `extensions/nn`, Closed declarative models, `modules/training`, `extensions/checkpoint`,
  Generic graph source and values, Optimizer/training lifecycle.
- [runtime-prepare-engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  `modules/engine`, Public explicit composition, Construction failure and closure,
  Public reusable prepared execution.
- [compiler-autograd — Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
  cold source capture/reconstruction paragraphs only, consumed indirectly through GraphCheckpoint.

Before code relies on high-level ownership, update root Core lifecycle, NN Module-ownership routing,
NN Dependency rules role prose and Testing requirements. In the **owning extensions-training contract**,
add bounded owned-tensor Model rules under extensions/nn; reconcile its persistence-only Engine/
Checkpoint wording and modules/training's adapter-only NN-consumer wording. Delegate to existing
owners, not optimizer algorithms or another pipeline. Keep edges/upstream lifecycles unchanged.
Update explanations, ADR0039 and conformance together. This plan/ADR is not authority; stop on
additional boundary changes, missing capabilities or ambiguity.

## Dependencies and integration

- Depends on: Engine0021, Training0007, Compiler0008, Checkpoint0008, Training0008 — all Complete.
- Conflicts with: shared root/NN authority, Model API/lifecycle, NN persistence glue and three planning
  indexes; no active writer. NN0027 and older resume proposals stay held.
- Parallel group: None; serial implementation and independent review.
- Common base revision: main `97c3bbe7a0ee893a2ba76b5d8e90449c033cadd4`.
- Integration order: completed predecessors → NN authority update → capability/tests/docs →
  independent Class C review → coordinator checkpoint/status integration.
- Integration validation: accepted coordinator native-enabled root command and evidence in Result;
  Markdown checks and `git diff --check`.
- Shared-document integration owner: model-execution coordinator, with explicit serial delegation.

## Files and symbols

Java prefix: `io/github/pho001/synaptik/`; NN roots: `extensions/nn/src/{main,test}/java/`.

- `nn/module/Model.java`: factories and tensor methods; new private nested Model.SingleTensorModel
  and Model.TensorListModel provide concrete generic specializations. Keep FunctionalModel.
  Use Topology directly, not a temporary FunctionalModel followed by child reparenting.
- New package-private `nn/module/OwnedModelSource.java`: acquisition, state installation and live
  source guards; private nested StateNamespace extends Module supplies graph/load dotted paths.
- New package-private `nn/module/ModelExecutionState.java`: admission, Engine choice, cached owners,
  bindings and cleanup. These five types are the complete production additions, all in nn.module.
- Reuse `nn/module/{Module,Topology,StateDictionary,DeclarativeModel}.java`; Javadoc review only.
- Javadoc only: `nn/module/{ModelDefinition,ModelForward}.java` (authorized review correction).
- Read-only dependencies: `modules/checkpoint/src/main/java/` + prefix +
  `checkpoint/graph/GraphCheckpoint.java` (Source/Leaf/Role, copy/load/save/rebind);
  `modules/training/src/main/java/` + prefix + `training/{TrainingSession,TrainingParameterStorage}.java`
  (open/inputs/runValues/close/copyToSharedNative); `modules/engine/src/main/java/` + prefix +
  `engine/{Engine,CompiledGraph,RunResult,HostTensorValue}.java` (compile/prepare/runValues,
  inputs, publications/materialize/close).
- NN tests `nn/module/ModelTest.java`: update exact public surface/external javac consumer; add
  OwnedModelSourceTest, OwnedModelExecutionTest, OwnedModelTrainingTest and OwnedModelLifecycleTest.
  Retain DeclarativeModelTest, ModuleTrainingParametersTest and
  `nn/checkpoint/{ModelCheckpointTest,WholeModelCheckpointTest}.java`.
- `testing/integration-tests/src/test/java/` + prefix + `testing/integration/`: add
  OwnedModelWorkflowIntegrationTest; retain GraphCheckpointEngineIntegrationTest,
  NnDeclarativeModelEngineIntegrationTest and WholeModelCheckpointEngineIntegrationTest.
- `testing/architecture-tests/src/test/java/` + prefix + `testing/architecture/`:
  extend NnTrainingDependencyContractTest's role-aware NN boundary checks; preserve
  CheckpointDependencyContractTest and EngineCompositionContractTest. Existing build edges suffice.

Replace obsolete public-surface assertions and “NN never owns execution” prose, not implemented APIs/
readers/algorithms. The discarded per-batch proposal has no code to delete. No backend changes.

## Acceptance criteria

1. External-package compilation witnesses exact generics, overloads and try-with-resources. Generic/
   custom/DeclarativeModel forward stays unchanged; unsupported tensor methods throw
   UnsupportedOperationException before Engine/file work, default close is a no-op. Single-boundary
   calls check arity; load rejects multi-boundary files with a loadGraph hint.
2. defineTensor seals definition once, sets eval before one prototype forward, initializes deferred
   layers and discards callbacks. Inventory ordinary provenance; registered wrappers determine
   PARAMETER/BUFFER, never requiresGrad. Remaining reachable stored leaves are deterministic,
   collision-free CONSTANT registrations. Reject unreachable/unnamed state, conflicting roles,
   unsupported source and paths before publication. Graph/load require canonical Module traversal order.
3. GraphCheckpoint.copy acquires each external identity once and reconstructs against private leaves;
   strict installation preserves wrappers/paths/aliases. Original Tensor associations/bytes are untouched,
   including closure-captured weights. Load adopts final heap state without another copy; constants stay
   outside stateDictionary. Failed construction cleans owners without claiming callback-effect rollback.
4. Full static signatures include layout presence and requiresGrad; validate declared slots, even unused
   ones, on each call. Build positional associations once from final CompiledGraph.inputs and
   TrainingSession.inputs, never source encounter order. Map surviving batch slots plus owned state;
   tolerate eliminated slots, reject unmapped inputs. Fresh calls reuse preparations: no rebuilding,
   GraphCheckpoint.rebind, prototype mutation, state copying or provider lookup per batch.
5. Compute checks every publication size and aggregate byte bound before any materialization
   (default Long.MAX_VALUE), uses exact ordered RunResult occurrences and closes results on every path.
   No partial list escapes. Batch storage is borrowed for the whole call; detached values outlive it.
6. Apply [Model Engine binding](../../../../design/decisions/0039-portable-graph-model-persistence.md#target-model-engine-binding)
   exactly: same receiver, borrowed Engine never closed; selected Engine closed → Model closed/closing →
   busy → capability → freeze → argument checks for withEngine. Null/closed candidates preserve selection.
   Freeze after valid compute/configure arguments/source checks but before Engine creation, callback,
   native copy or preparation, even on failure. Same-Engine rebinding then rejects. Lazy owned
   Engine.standard is CPU-only, no discovery/global; construction/load/save/cold forward create no Engine.
7. configureTraining succeeds once, does not toggle mode, invokes its pure objective builder once per
   explicit attempt and retains no runtime callback. Require scalar FLOAT32/FLOAT64 objective; map batch,
   targets and owned BUFFER/CONSTANT leaves to session.inputs, excluding parameters/private seed.
   Reject otherwise unbindable objective leaves; do not retain arbitrary external callback state.
8. Close inference preparations before native replacement. Use Module.trainingParameters and
   TrainingParameterStorage.copyToSharedNative with model-owned shared Arena, retaining Tensor identity.
   Never recopy admitted stores. Failed copy preserves old associations/bytes; failed setup cleans
   unpublished owners and allows frozen-Engine retry. Inference may reprepare after this transition.
   Current unresolved-layout/empty/nonfinite/aliased-parameter rejection remains, without conversion.
9. train requires configuration, delegates RESET_AND_STEP to TrainingSession.runValues and preserves
   live precommit/overlap/atomicity checks. Compare fresh batches/targets with direct TrainingSession;
   verify updates appear in compute and trained save/reload, preparations/identities remain stable,
   results do not leak and rejected steps preserve bytes/counters.
10. Guard all module modes, paths/order/wrappers/Tensors/storage associations before compute/train/
    configure/cold forward/save. Changes invalidate capability and close cached owners, never recapture,
    adopt borrowed replacement state or reset optimizer history. Same-storage trained bytes remain valid.
11. Busy rejects competing high-level operations before arguments. Close bypasses ordinary admission,
    rejects new work, waits for admitted work, closes session/preparation before native owners then owned
    Engine, attempts all cleanup and retains/suppresses/replays failures. Deterministic private tests cover
    borrowed Engine closed first, failed setup, races and repeated/concurrent close failure. HostTensorValue/
    TrainingStep survive all closes; inherited Module mutators remain caller-coordinated.
12. Save-before-compute and trained re-save use GraphCheckpoint source plus current leaves only.
    Explicit ArtifactLimits remain; short overloads match NN's six existing ceilings: 256 MiB file/aggregate,
    4 MiB manifest, 16384 entries, 64-byte names, 64 MiB payload, not the old definition codec.
    Preserve raw bits/aliases/atomic publication/failures and separate old flexible v1 readers.
    Zero-node data may save/load; compute retains Engine empty-plan rejection without a fake operation.

## Validation

Completed worker validation covered cold composition, multiple outputs, reordered/unused inputs,
aliases and old flexible batches without duplicate upstream suites:

Worker `./gradlew :extensions:nn:test :extensions:nn:javadoc` and post-correction
`./gradlew :extensions:nn:javadoc` passed; their executable evidence is retained.

Accepted coordinator validation is the exact native-enabled root command in Result below.
Metadata closeout checks links/anchors and `git diff --check`, without Gradle execution.

## Documentation and review impact

Update root/owning contract first; explanations: `docs/architecture/{module-boundaries,dependency-rules,training-graph}.md`;
ADR0039's Model target/status; `docs/api/{public-api,training-api,graph-checkpoint-api}.md`;
`docs/user-guide/{running-models,training,declarative-models}.md`; targeted glossary entries for
functional Model, owned source, prepared execution and detached results. Explain both workflows,
static signatures, old readers and lifecycle. Review affected Javadoc with full input/result/failure
tags. General and matching architecture/API/user-guide/ADR profiles were applied. Independent
Class C code/docs/Javadoc/glossary review and coordinator acceptance are complete. Coordinator
synchronizes this brief, NN master, program and roadmap.

## Result

Status: Complete
Coordinator accepted 2026-10-19; Ramanujan definitive code/docs/Javadoc/glossary APPROVE, zero outstanding (3 doc findings fixed).
30 task files implemented/documented; exact accepted root gate:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain
```

EXIT 0, BUILD SUCCESSFUL in 5s; 107 tasks: 2 executed, 105 up-to-date.
Fresh architecture: 13; full integration: 134, 1 skipped CPU tuning fixture; actual Metal training passed.
19 current project reports: 4227 tests, 0 failures, 0 errors, 41 skips; mixed-age evidence, not all fresh.
NN 401 tests/53 suites and final Javadoc reused UP-TO-DATE; no failures/errors; prior worker selected integrations 7/4 passed.
Coordinator verified all 2279 source files unchanged through gate; both protected user files unchanged.
Source/authority manifest /private/tmp/nn0028-source-freeze.sha256 SHA256:
f46633c12787c053ecd287b163fdbb30ba5b79408a884244c263b2132e698d6f.
Docs/Javadocs/glossary complete; metadata links/diff checks passed. No unresolved issue or implementation follow-up.
Authorized NN frontier: None. Coordinator owns final commit/push; executor made no Git writes.
