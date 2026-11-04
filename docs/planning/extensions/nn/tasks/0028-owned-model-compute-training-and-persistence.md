# Task NN0028: [Owned Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) compute training and [persistence](../../../../glossary.md#term-persistence)

## Status

Complete — coordinator accepted 2026-10-19 after definitive independent approval and source-matched native-enabled root gate.
Authorized [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) frontier: None. No successor is launched.

## Change class

Class C — public [Model](../../../../glossary.md#term-model) [lifecycle](../../../../glossary.md#lifecycle), owned storage, [Engine](../../../../glossary.md#term-engine)/Training composition, exclusion and [persistence](../../../../glossary.md#term-persistence).

## Goal

Give application-built and file-restored [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) Models one owned working-[weight](../../../../glossary.md#term-weight) set for compute/train/save.

## [Scope](../../../../glossary.md#term-scope)

Implement [ADR0039 Selected API and type feasibility](../../../../design/decisions/0039-portable-graph-model-persistence.md#selected-api-and-type-feasibility)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>,
including its single/list forms and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease)-[Engine](../../../../glossary.md#term-engine) overloads: [Model](../../../../glossary.md#term-model) stays abstract and generic,
gains AutoCloseable; defineTensor([signature](../../../../glossary.md#term-signature), definition)/load return Model<[Tensor](../../../../glossary.md#tensor),Tensor>;
[graph](../../../../glossary.md#graph)/loadGraph return Model<List<Tensor>,List<Tensor>>. Add same-instance withEngine, terminal
compute, configureTraining, RESET_AND_STEP train, bounded list compute, save and close. Execution
returns [HostTensorValue](../../../../glossary.md#host-tensor-value--hosttensorvalue)/TrainingStep, never a cast to generic O.

Source [admission](../../../../glossary.md#term-admission) remains static ADD/MUL/NEG/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/PERMUTE[1,0]; seven raw [dtypes](../../../../glossary.md#data-type--datatype) persist independently
of execution/training support. Objectives use current [Compiler](../../../../glossary.md#term-compiler) support but are not persisted.
Retain generic/flexible APIs, mode-only [Module](../../../../glossary.md#term-module).train() and old model.v1/whole-model.v1 readers.

## Non-goals

No GraphProgram, LoadedGraph, [InferenceSession](../../../../glossary.md#former-inference-session--inferencesession), ModelExecutor, public subtype/result/session hierarchy,
new [IR](../../../../glossary.md#term-intermediate-representation-ir), upstream [API](../../../../glossary.md#term-application-programming-interface-api)/build/[dependency](../../../../glossary.md#term-dependency)/native changes, [backend](../../../../glossary.md#backend) additions, [codec](../../../../glossary.md#term-codec) [widening](../../../../glossary.md#term-widening), [optimizer](../../../../glossary.md#term-optimizer)/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/
data-cursor [resume](../../../../glossary.md#schema-serialization-and-checkpoint), Data infrastructure, [backpropagation through time](../../../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)), numerical/performance
campaign or [Model](../../../../glossary.md#term-model)/Training [autotuning](../../../../glossary.md#autotuning--model-autotuning). Advanced [Engine](../../../../glossary.md#term-engine).prepareTuned remains separate.

## Contracts

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts): Core lifecycle;
  Core invariants (immutable Tensor/ownership); Module-ownership routing; Dependency rules;
  Documentation authority and architecture updates; Testing requirements.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- [extensions-training](../../../../architecture/contracts/extensions-training.md#extensionsnn):
  `extensions/nn`, Closed declarative models, `modules/training`, `extensions/checkpoint`,
  Generic graph source and values, Optimizer/training lifecycle.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- [runtime-prepare-engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>:
  `modules/engine`, Public explicit composition, Construction failure and closure,
  Public reusable prepared execution.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- [compiler-autograd — Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [Compile glossary](../../../../glossary.md#compile))</span>:
  cold source capture/reconstruction paragraphs only, consumed indirectly through GraphCheckpoint.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-structural-capture))</span>

Before code relies on high-level [ownership](../../../../glossary.md#term-ownership-meanings), update root Core [lifecycle](../../../../glossary.md#lifecycle), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [Module](../../../../glossary.md#term-module)-ownership routing,
NN [Dependency](../../../../glossary.md#term-dependency) rules role prose and Testing requirements. In the **owning extensions-training contract**,
add bounded owned-[tensor](../../../../glossary.md#tensor) [Model](../../../../glossary.md#term-model) rules under extensions/nn; reconcile its [persistence](../../../../glossary.md#term-persistence)-only [Engine](../../../../glossary.md#term-engine)/
[Checkpoint](../../../../glossary.md#term-checkpoint) wording and modules/training's adapter-only NN-consumer wording. Delegate to existing
[owners](../../../../glossary.md#term-owner), not [optimizer](../../../../glossary.md#term-optimizer) algorithms or another pipeline. Keep edges/upstream lifecycles unchanged.
Update explanations, ADR0039 and [conformance](../../../../glossary.md#term-conformance) together. This plan/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) is not authority; stop on
additional boundary changes, missing capabilities or ambiguity.

## Dependencies and integration

- Depends on: Engine0021, Training0007, Compiler0008, Checkpoint0008, Training0008 — all Complete.
- Conflicts with: shared root/NN authority, Model API/lifecycle, NN persistence glue and three planning
  indexes; no active writer. NN0027 and older resume proposals stay held.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [persistence glossary](../../../../glossary.md#term-persistence); [planning glossary](../../../../glossary.md#term-planning); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>
- Parallel group: None; serial implementation and independent review.
- Common base revision: main `97c3bbe7a0ee893a2ba76b5d8e90449c033cadd4`.
- Integration order: completed predecessors → NN authority update → capability/tests/docs →
  independent Class C review → coordinator checkpoint/status integration.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration validation: accepted coordinator native-enabled root command and evidence in Result;
  Markdown checks and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`git diff --check`.
- Shared-document integration [owner](../../../../glossary.md#term-owner): [model](../../../../glossary.md#term-model)-execution coordinator, with explicit serial delegation.

## Files and symbols

Java prefix: `io/github/pho001/synaptik/`; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) roots: `extensions/nn/src/{main,test}/java/`.

- `nn/module/Model.java`: factories and tensor methods; new private nested Model.SingleTensorModel
  and Model.TensorListModel provide concrete generic specializations. Keep FunctionalModel.
  Use Topology directly, not a temporary FunctionalModel followed by child reparenting.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [Topology glossary](../../../../glossary.md#term-topology))</span>
- New package-private `nn/module/OwnedModelSource.java`: acquisition, state installation and live
  source guards; private nested StateNamespace extends Module supplies graph/load dotted paths.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [graph glossary](../../../../glossary.md#graph))</span>
- New package-private `nn/module/ModelExecutionState.java`: admission, Engine choice, cached owners,
  bindings and cleanup. These five types are the complete production additions, all in nn.module.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [Engine glossary](../../../../glossary.md#term-engine); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Reuse `nn/module/{Module,Topology,StateDictionary,DeclarativeModel}.java`; [Javadoc](../../../../glossary.md#term-javadoc) review only.
- Javadoc only: `nn/module/{ModelDefinition,ModelForward}.java` (authorized review correction).
- Read-only dependencies: `modules/checkpoint/src/main/java/` + prefix +
  `checkpoint/graph/GraphCheckpoint.java` (Source/[Leaf](../../../../glossary.md#term-leaf)/Role, copy/load/save/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction));
  `modules/training/src/main/java/` + prefix + `training/{TrainingSession,TrainingParameterStorage}.java`
  (open/inputs/runValues/close/copyToSharedNative); `modules/engine/src/main/java/` + prefix +
  `engine/{Engine,CompiledGraph,RunResult,HostTensorValue}.java` (compile/prepare/runValues,
  inputs, publications/materialize/close).<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare))</span>
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
readers/algorithms. The discarded per-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) proposal has no code to delete. No [backend](../../../../glossary.md#backend) changes.

## Acceptance criteria

1. External-package compilation witnesses exact generics, overloads and try-with-resources. Generic/
   custom/DeclarativeModel forward stays unchanged; unsupported tensor methods throw
   UnsupportedOperationException before Engine/file work, default close is a no-op. Single-boundary
   calls check arity; load rejects multi-boundary files with a loadGraph hint.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [tensor glossary](../../../../glossary.md#tensor); [Engine glossary](../../../../glossary.md#term-engine); [arity glossary](../../../../glossary.md#term-arity))</span>
2. defineTensor seals definition once, sets eval before one prototype forward, initializes deferred
   layers and discards callbacks. Inventory ordinary provenance; registered wrappers determine
   PARAMETER/BUFFER, never requiresGrad. Remaining reachable stored leaves are deterministic,
   collision-free CONSTANT registrations. Reject unreachable/unnamed state, conflicting roles,
   unsupported source and paths before publication. Graph/load require canonical Module traversal order.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([prototype glossary](../../../../glossary.md#term-prototype); [provenance glossary](../../../../glossary.md#provenance); [PARAMETER glossary](../../../../glossary.md#term-parameter); [BUFFER glossary](../../../../glossary.md#term-buffer); [CONSTANT glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication); [Graph glossary](../../../../glossary.md#graph); [canonical glossary](../../../../glossary.md#term-canonical-form); [Module glossary](../../../../glossary.md#term-module))</span>
3. GraphCheckpoint.copy acquires each external identity once and reconstructs against private leaves;
   strict installation preserves wrappers/paths/aliases. Original Tensor associations/bytes are untouched,
   including closure-captured weights. Load adopts final heap state without another copy; constants stay
   outside stateDictionary. Failed construction cleans owners without claiming callback-effect rollback.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [weights glossary](../../../../glossary.md#term-weight); [heap glossary](../../../../glossary.md#term-heap); [callback glossary](../../../../glossary.md#term-callback); [rollback glossary](../../../../glossary.md#term-rollback))</span>
4. Full static signatures include layout presence and requiresGrad; validate declared slots, even unused
   ones, on each call. Build positional associations once from final CompiledGraph.inputs and
   TrainingSession.inputs, never source encounter order. Map surviving batch slots plus owned state;
   tolerate eliminated slots, reject unmapped inputs. Fresh calls reuse preparations: no rebuilding,
   GraphCheckpoint.rebind, prototype mutation, state copying or provider lookup per batch.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [layout glossary](../../../../glossary.md#layout); [TrainingSession glossary](../../../../glossary.md#training-session--trainingsession); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [lookup glossary](../../../../glossary.md#term-lookup))</span>
5. Compute checks every publication size and aggregate byte bound before any materialization
   (default Long.MAX_VALUE), uses exact ordered RunResult occurrences and closes results on every path.
   No partial list escapes. Batch storage is borrowed for the whole call; detached values outlive it.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [RunResult glossary](../../../../glossary.md#run-result--runresult); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [detached glossary](../../../../glossary.md#detached-model-results))</span>
6. Apply [Model Engine binding](../../../../design/decisions/0039-portable-graph-model-persistence.md#target-model-engine-binding)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [binding glossary](../../../../glossary.md#term-binding))</span>
   exactly: same receiver, borrowed Engine never closed; selected Engine closed → Model closed/closing →
   busy → capability → freeze → argument checks for withEngine. Null/closed candidates preserve selection.
   Freeze after valid compute/configure arguments/source checks but before Engine creation, callback,
   native copy or preparation, even on failure. Same-Engine rebinding then rejects. Lazy owned
   Engine.standard is CPU-only, no discovery/global; construction/load/save/cold forward create no Engine.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [Lazy glossary](../../../../glossary.md#term-lazy-evaluation); [CPU glossary](../../../../glossary.md#term-cpu))</span>
7. configureTraining succeeds once, does not toggle mode, invokes its pure objective builder once per
   explicit attempt and retains no runtime callback. Require scalar FLOAT32/FLOAT64 objective; map batch,
   targets and owned BUFFER/CONSTANT leaves to session.inputs, excluding parameters/private seed.
   Reject otherwise unbindable objective leaves; do not retain arbitrary external callback state.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [runtime glossary](../../../../glossary.md#term-runtime); [scalar glossary](../../../../glossary.md#term-scalar); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [targets glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed))</span>
8. Close inference preparations before native replacement. Use Module.trainingParameters and
   TrainingParameterStorage.copyToSharedNative with model-owned shared Arena, retaining Tensor identity.
   Never recopy admitted stores. Failed copy preserves old associations/bytes; failed setup cleans
   unpublished owners and allows frozen-Engine retry. Inference may reprepare after this transition.
   Current unresolved-layout/empty/nonfinite/aliased-parameter rejection remains, without conversion.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [TrainingParameterStorage glossary](../../../../glossary.md#native-parameter-copy--trainingparameterstorage); [Arena glossary](../../../../glossary.md#term-arena))</span>
9. train requires configuration, delegates RESET_AND_STEP to TrainingSession.runValues and preserves
   live precommit/overlap/atomicity checks. Compare fresh batches/targets with direct TrainingSession;
   verify updates appear in compute and trained save/reload, preparations/identities remain stable,
   results do not leak and rejected steps preserve bytes/counters.<span data-glossary-navigation="true"> ([precommit glossary](../../../../glossary.md#term-precommit); [atomicity glossary](../../../../glossary.md#term-atomicity))</span>
10. Guard all module modes, paths/order/wrappers/Tensors/storage associations before compute/train/
    configure/cold forward/save. Changes invalidate capability and close cached owners, never recapture,
    adopt borrowed replacement state or reset optimizer history. Same-storage trained bytes remain valid.<span data-glossary-navigation="true"> ([Guard glossary](../../../../glossary.md#term-guard); [optimizer history glossary](../../../../glossary.md#term-optimizer-history))</span>
11. Busy rejects competing high-level operations before arguments. Close bypasses ordinary admission,
    rejects new work, waits for admitted work, closes session/preparation before native owners then owned
    Engine, attempts all cleanup and retains/suppresses/replays failures. Deterministic private tests cover
    borrowed Engine closed first, failed setup, races and repeated/concurrent close failure. HostTensorValue/
    TrainingStep survive all closes; inherited Module mutators remain caller-coordinated.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [HostTensorValue glossary](../../../../glossary.md#host-tensor-value--hosttensorvalue))</span>
12. Save-before-compute and trained re-save use GraphCheckpoint source plus current leaves only.
    Explicit ArtifactLimits remain; short overloads match NN's six existing ceilings: 256 MiB file/aggregate,
    4 MiB manifest, 16384 entries, 64-byte names, 64 MiB payload, not the old definition codec.
    Preserve raw bits/aliases/atomic publication/failures and separate old flexible v1 readers.
    Zero-node data may save/load; compute retains Engine empty-plan rejection without a fake operation.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [manifest glossary](../../../../glossary.md#term-manifest); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [codec glossary](../../../../glossary.md#term-codec); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [atomic publication glossary](../../../../glossary.md#term-atomic-publication); [node glossary](../../../../glossary.md#node); [operation glossary](../../../../glossary.md#operation))</span>

## [Validation](../../../../glossary.md#term-validation)

Completed worker validation covered cold composition, multiple outputs, reordered/unused inputs,
[aliases](../../../../glossary.md#aliasing-and-arena) and old flexible [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) without duplicate upstream suites:

Worker `./gradlew :extensions:nn:test :extensions:nn:javadoc` and post-correction
`./gradlew :extensions:nn:javadoc` passed; their executable evidence is retained.

Accepted coordinator validation is the exact native-enabled root command in Result below.
[Metadata](../../../../glossary.md#term-metadata) closeout checks links/anchors and `git diff --check`, without [Gradle](../../../../glossary.md#term-gradle) execution.

## Documentation and review impact

Update root/owning contract first; explanations: `docs/architecture/{module-boundaries,dependency-rules,training-graph}.md`;
ADR0039's [Model](../../../../glossary.md#term-model) [target](../../../../glossary.md#term-target)/status; `docs/api/{public-api,training-api,graph-checkpoint-api}.md`;
`docs/user-guide/{running-models,training,declarative-models}.md`; targeted glossary entries for
functional Model, owned source, [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) and [detached](../../../../glossary.md#detached-model-results) results. Explain both workflows,
static [signatures](../../../../glossary.md#term-signature), old readers and [lifecycle](../../../../glossary.md#lifecycle). Review affected [Javadoc](../../../../glossary.md#term-javadoc) with full input/result/failure
tags. General and matching architecture/[API](../../../../glossary.md#term-application-programming-interface-api)/user-guide/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) profiles were applied. Independent
Class C code/docs/Javadoc/glossary review and coordinator acceptance are complete. Coordinator
synchronizes this brief, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master, program and roadmap.

## Result

Status: Complete
Coordinator accepted 2026-10-19; Ramanujan definitive code/docs/[Javadoc](../../../../glossary.md#term-javadoc)/glossary APPROVE, zero outstanding (3 doc findings fixed).
30 task files implemented/documented; exact accepted root gate:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain
```

EXIT 0, BUILD SUCCESSFUL in 5s; 107 tasks: 2 executed, 105 up-to-date.
Fresh architecture: 13; full integration: 134, 1 skipped [CPU](../../../../glossary.md#term-cpu) tuning [fixture](../../../../glossary.md#term-fixture); actual [Metal](../../../../glossary.md#term-metal) training passed.
19 current project reports: 4227 tests, 0 failures, 0 errors, 41 skips; mixed-age evidence, not all fresh.
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 401 tests/53 suites and final Javadoc reused UP-TO-DATE; no failures/errors; prior worker selected integrations 7/4 passed.
Coordinator verified all 2279 source files unchanged through gate; both protected user files unchanged.
Source/authority [manifest](../../../../glossary.md#term-manifest) /private/tmp/nn0028-source-freeze.sha256 SHA256:
f46633c12787c053ecd287b163fdbb30ba5b79408a884244c263b2132e698d6f.
Docs/Javadocs/glossary complete; [metadata](../../../../glossary.md#term-metadata) links/diff checks passed. No unresolved issue or implementation follow-up.
Authorized NN frontier: None. Coordinator owns final [commit](../../../../glossary.md#term-commit)/push; executor made no Git writes.
