# Task 0029 Fluent Model builder and checkpoint independent execution

## Status

Complete — Planck final APPROVE, zero findings; main accepted after the shared root gate (2026-10-19).

## Change class and goal

Class C: construction API, private-state acquisition and cold-forward ownership. Return existing
`Model<Tensor,Tensor>` independently of eight-operation serialization; persistence remains fail-closed.

## Scope and exclusions

Reuse current layers, Sequential, Tensor operations, ModelExecutionState and public owners for
dense/convolutional/fixed-recurrent chains. Preserve define/defineTensor/graph/load/loadGraph and old APIs.
No arbitrary callback/module/graph builder, Layers registry, model/session hierarchy, second graph/
pipeline, NN reconstruction table, Compiler edge, codec/backend/build change, new adjoint, runtime
lengths/scan backpropagation through time (BPTT), streaming state, bidirectionality or optimizer/RNG resume.

## Contracts and required authority cutover

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then
root headings `Core lifecycle`, `Core invariants`, `Module-ownership routing`, `Dependency rules`, `Testing requirements`.
Read these exact headings in the incorporated files under `docs/architecture/contracts/`:

- `extensions-training.md`: `extensions/nn`, `Owned tensor Models`, `modules/training`,
  `Optimizer/training lifecycle`, `Generic graph source and values`, `Canonical graph-model version 2`.
- `foundational-modules.md`: `modules/model`.
- `runtime-prepare-engine.md`: `Public reusable prepared execution`.
- `compiler-autograd.md`: `Compile lifecycle`, `Compiler-owned automatic differentiation`.

Accepted before Java: root `Core lifecycle` and NN `Owned tensor Models` distinguish builder-owned
ordinary composition from unchanged callback/graph/load reconstruction and save-only eight-kind
admission. Signatures, guards, exclusion, native admission, bindings and cleanup remain unchanged.
Explanations and ADR0040 accompany the cutover; no Compiler/Checkpoint widening. Stop on broader decisions.

## Dependencies and integration

- Depends on: NN0028, Checkpoint0009, [CPU0025](../../../backends/cpu/tasks/0025-scatter-validation-coordinate-rank.md) and [Compiler0009](../../../modules/compiler/tasks/0009-static-select-and-slice-logical-layout-closure.md), all Complete with required independent reviews. Neither repair depends on NN0029.
- Conflicts with: NN API/lifecycle, root/NN contract, shared explanations/planning; none active.
- Parallel group: None.
- Common base revision: `7e880fb1044168043e45b10419a570d64c5283ae` plus accepted dirty CP0009/NN/CPU0025/Compiler0009, not HEAD alone.
- Integration order: accepted prerequisites/authority → NN → independent Class C review → main root gate.
- Integration validation: main's single native-enabled root command below passed; Result records the join.
- Shared-document integration owner: main; exclusive serial delegation.
- Historical resume (2026-10-19): main verified Ready/frontier/approval/authority/dependencies/base/conflicts;
  sole executor resumed with partial preserved.

## Public API and defaults

Add public final `io.github.pho001.synaptik.nn.module.ModelBuilder` via `public static ModelBuilder Model.builder()`;
no public constructor. Configuration returns the same builder. Signatures:

```java
ModelBuilder input(DataType type, long... shape);
ModelBuilder seed(long seed);
ModelBuilder linear(long outFeatures);
ModelBuilder linear(long outFeatures, boolean bias);
ModelBuilder linear(Tensor weight);
ModelBuilder linear(Tensor weight, Tensor bias);
ModelBuilder relu(); ModelBuilder tanh(); ModelBuilder sigmoid();
ModelBuilder conv1d(long outChannels, long kernelWidth);
ModelBuilder conv2d(long outChannels, long kernelHeight, long kernelWidth);
ModelBuilder conv3d(long outChannels, long kernelDepth, long kernelHeight, long kernelWidth);
ModelBuilder maxPool2d(long kernelHeight, long kernelWidth);
ModelBuilder flatten();
ModelBuilder rnn(long hiddenSize); ModelBuilder gru(long hiddenSize); ModelBuilder lstm(long hiddenSize);
ModelBuilder lastTimeStep();
Model<Tensor, Tensor> build();
```

Input is set once before steps: floating type, rank ≥1, positive static extents; snapshot shape.
Prototype is storage-free, canonical contiguous, offset-zero non-view, requiresGrad=false. Compute/
cold forward retain full-descriptor matching. Build requires input/nonempty chain. Caller-coordinated
builder is reusable: each build owns fresh children/state unaffected by later edits. Default seed 0;
the last seed selected before build applies to that whole build.
For zero-based appended step index i, derive `seed + 0x9E3779B97F4A7C15L * i` with Java long
wraparound; activations/transforms count. Use existing L64X128MixRandom algorithms, Glorot uniform,
exact zero bias enabled by default. Each layer uses a fresh transient generator; recurrent matrix
order/fans remain the cell's. No shared mutable/global RNG; same recipes/seed/algorithm reproduce bytes.
Linear infers the last feature extent. Supplied weights must satisfy current Linear guards
(positive static rank-two floating gradient-eligible weight, matching optional rank-one bias),
match input dtype, and be stored provenance-free leaves admitted by the existing raw copier.
No coercion, expression evaluation or mutation of supplied Tensor descriptors/storage/bytes.
Conv defaults: stride 1, padding 0, dilation 1, groups 1, bias true; layouts NCW/NCHW/NCDHW.
Pool defaults: stride equals kernel, padding 0, dilation 1, floor. Flatten requires rank ≥2 and
preserves the first axis: `[axis0, checked product of remaining extents]`. CNN axis 0 is batch;
TBF `[T,B,H]` becomes `[T,B*H]`. Use lastTimeStep first to retain recurrent batch rows.
Recurrent input is time/batch/features (TBF), positive fixed time and all-valid rows; outputs are
all hidden steps `[T,B,H]`, including LSTM. Stack recurrent layers directly; lastTimeStep requires
rank-three TBF and explicitly selects `select(0,T-1)` to `[B,H]`. No implicit final-step selection.
Use existing sequence.forward(input), assemble packedOutputs with reshape to `[1,B,H]` then
Tensor.concat(0,...). Each cold composition starts zero h (and zero c for LSTM); cached compute/train
reuses immutable zero constants, never final state from an earlier invocation. No runtime reset work.

## Concrete acquisition and files

Production changes are limited to existing `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/`
`Model.java`, `OwnedModelSource.java`, `ModelExecutionState.java`, and new `ModelBuilder.java`.
ModelBuilder privately owns `LayerRecipe` (internal factories, no public callback entry),
`MaxPoolStep`, `FlattenStep`, `LastTimeStepStep`, `RnnStep`, `GruStep`, `LstmStep`; these are its
only new nested types. Unary adapters use Tensor operations; recurrent adapters own child `sequence`.
No layer/cell/Sequential change is planned.

At build, snapshot recipes and deduplicate supplied leaves by Tensor identity. If nonempty, copy
them once using `GraphCheckpoint.copy(new Source(List.of(), uniqueLeaves, namedLeaves))` with
unique outputs/PARAMETER registrations: zero-node raw copy BEFORE expressions. A temporary leaf-only
identity association preserves aliases, not graph reconstruction. No copy of already-private state.
Create fresh layers using copied supplied state or automatic recipes. Add package-private
`Model.fromComposition(TensorDescriptor, Sequential)`, reusing existing SingleTensorModel, owning
the sequence as child `layers`, selecting eval and building its prototype forward once. Paths are
`layers.0.weight`, or `layers.0.sequence.cell.inputWeight`, etc.; preserve canonical traversal order.
Add `OwnedModelSource.acquireComposition(Module, Tensor, Tensor, Sequential)`: inventory registered
wrappers and reachable fresh constants using ordinary provenance, verify state reachability, retain
exact final leaves/output and chain, then capture existing guards. Use GraphCheckpoint.Source only
as the existing borrowed Tensor-list carrier; do NOT call copy/capture/rebind on this completed chain.
Keep acquireDefinition/adopt semantics unchanged. Add internal `OwnedModelSource.forward(List<Tensor>)`:
builder sources apply retained Sequential after exact signature validation; other sources use existing
GraphCheckpoint.rebind. ModelExecutionState.forward delegates under the same admission/live guards.
Compute/configure/train use fixed outputs/cached bindings, never recipes/forward/reconstruction per batch.
Built Models retain no recipes/caller leaves. Failed build publishes nothing and mutates no caller.
Save still delegates fixed outputs/current leaves to GraphCheckpoint.save. Unsupported convolution,
pool/reshape/concat/select/gather chains reject before ArtifactFile work, with no target mutation,
Engine creation, binding freeze or model invalidation; inference/training remains usable afterwards.

Read-only: Sequential/UnaryTensorModule; Linear/Activation/Conv1d/2d/3d and initialization;
Rnn/Gru/LstmSequence/results/cells; Tensor.linear/maxPool2d/reshape/concat/select, descriptors/attrs;
GraphCheckpoint.copy/save.

## Acceptance and focused tests

Add `ModelBuilderTest` and `ModelBuilderExecutionTest` in the NN module test package above;
extend OwnedModelSource/Execution/Training/LifecycleTest only where shared paths need regression.
Update `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModelTest.java:73`:
its exact public-method-set guard must add builder and preserve every old method.
Add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/FluentModelBuilderIntegrationTest.java`.

1. Planned API examples build without user lambdas: `Model.builder().input(DataType.FLOAT32,32,784)
   .seed(42).linear(128).relu().linear(10,false).build()`; both supplied-weight overloads and all activations.
2. Build and numerically execute the CNN: input `[32,1,28,28]`, conv2d(32,3,3), relu, maxPool2d(2,2),
   conv2d(64,3,3), relu, maxPool2d(2,2), flatten, linear(128), relu, linear(10). Spatial extents
   28→26→13→11→5; flatten `[32,1600]`, final `[32,10]`. Also real Conv1d/Conv3d forward cases.
3. All three recurrent builders compute against independent ordinary Tensor/cell equations plus
   Engine, not Model.forward/GraphCheckpoint as oracle. Check every time output, repeated same-input
   calls, changed batches, explicit final-step selection; `[20,32,8]→lstm(128)→lstm(64)` yields
   `[20,32,64]`, then lastTimeStep/linear(10) yields `[32,10]`. Use smaller nonzero numerical fixtures
   too; distinguish state carry, gate mistakes and implicit last-step behavior.
4. MLP train compares objective, every weight update and subsequent compute with independent Tensor/
   Engine backward plus explicit SGD arithmetic; include RELU/TANH/SIGMOID and nonzero gradients.
   At least one valid multi-step unrolled recurrent train must succeed with existing adjoints and
   match a direct oracle. Static unrolling is not scan BPTT; Conv3d backward still rejects.
5. Seed/default byte determinism, distinct step seeds, separate build ownership, frozen structure,
   checked invalid dimensions/ranks/dtypes/geometry, supplied-state alias preservation, and no caller
   mutation across build/forward/train/close. Unsupported supplied produced/unavailable state rejects.
6. Cold forward accepts exact-compatible produced inputs with arbitrary upstream provenance (e.g.
   abs().contiguous()), uses the same owned weights, creates no Engine and changes no cached graph.
   Reuse prepared/session identities across fresh batches; no per-compute graph builds/copies.
   Preserve freeze/retry, native-copy-once, invalidation, exclusion/close, borrowed Engine and detached
   result lifetimes; reuse existing lifecycle evidence.
7. MLP saves before compute and after train, five-kind v1/activation v2 reloads compute/train correctly.
   Unsupported builder save to absent and existing sentinel targets leaves both unchanged and no
   temporary file; subsequent compute/train succeeds. Existing generic/declarative/graph/load tests pass.

## Validation and documentation

Worker: focused command recorded below, then
`./gradlew :extensions:nn:test :extensions:nn:javadoc :testing:integration-tests:compileTestJava --max-workers=2 --console=plain` once.
Review reuses evidence; final Javadoc follows final edits. Main owns the source-matched gate:
`SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain`.
No worker native rebuild/root rerun/performance campaign. Architecture/backend-conformance changes:
none expected, edges/behavior unchanged; existing checks run at main's gate. NN/integration tests cover acquisition.

Class C documentation/Javadoc/glossary/authority review is complete; evidence below. Main syncs statuses.
Preserve THIRD_PARTY_NOTICES.md SHA-256 `8efe7e3c1f83bd5b30b913d447fa3ee95f9c0ee60485c30de7c0008638e7037f`
and CPU `CpuOrdinaryPointwiseCastCleanJavaOracle.java` SHA-256 `57ce4dc701a83d2ed01b96c725197a72f5aa752fe5e200f3a354f9d273f8f842`.
No commit/push.

## Result

Status: Complete

Delivered 17 NN code/test/doc files, including authority/ADR0040/API/examples/glossary/Javadocs.
Planck final APPROVE, zero findings; 17 hashes matched, no later code edits.
Focused `./gradlew :extensions:nn:test --tests '*ModelBuilder*Test' --max-workers=2 --console=plain`: 22 pass/0 skips; original 17 unchanged.
Final worker command above: 55 suites/424 NN tests, zero failures/errors/skips; warning-free Javadoc/integration compilation PASS.
`./gradlew :testing:integration-tests:test --tests '*FluentModelBuilderIntegrationTest' --max-workers=2 --console=plain`: 3 pass/0 skips; published examples compiled/executed.
CPU/Compiler final APPROVE, zero findings; local evidence retained.
Root command above: exit 0, BUILD SUCCESSFUL, 5m14s; 107 tasks, 18 executed/89 UP-TO-DATE.
2291-file reviewed snapshot/native hash unchanged. XML: 4279 tests,
0 failures/errors, 41 skips (CPU 30, Metal 10, integration 1); includes UP-TO-DATE evidence, not all newly run.
Architecture 13/0 skips; integration 145/1 skip. Configured mixed-Metal training and Metal-only
unsupported-operation cases executed, zero failures/skips.
Main links: 667 files/9490 links/1346 fragments, 7 historical errors/0 new; diff/protected hashes PASS.
Nonblocking follow-up: characterize larger stacked-LSTM CPU cold-prepare cost. Development 3-step
train was interrupted in active topological preparation; reduced 2-step train, required 3-step RNN
training/all-family stack numerics pass. Review found no acceptance gap, error or proved regression;
no new Ready task, benchmark campaign or gate.
