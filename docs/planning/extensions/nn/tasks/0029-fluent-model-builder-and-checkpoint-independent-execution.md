# Task 0029 [Fluent Model builder](../../../../glossary.md#fluent-model-builder) and [checkpoint](../../../../glossary.md#term-checkpoint) independent execution

> **Editorial reading context — not a status update.** The application collects builder
> recipes; build creates fresh private Model state and expressions once. Supplied parameter
> leaves are copied into that private state, unlike ordinary layer constructors that retain
> references. Compute later binds batches to cached execution, not to replayed recipes.
> Separating construction/execution from save admission means an unsupported file encoding
> need not disable a numerically supported Model. Today's [fluent builder](../../../../reference/nn/model-builder.md#fluent-model-builder)
> explains seed/Shape and copy rules, and [owned execution](../../../../reference/nn/owned-model.md#owned-tensor-model-capability)
> explains the resource lifetime behind compute, configured training and close.

## Status

Complete — Planck final APPROVE, zero findings; main accepted after the shared root gate (2026-10-19).

## Change class and goal

Class C: construction [API](../../../../glossary.md#term-application-programming-interface-api), private-state acquisition and cold-forward [ownership](../../../../glossary.md#term-ownership-meanings). Return existing
`Model<Tensor,Tensor>` independently of eight-[operation](../../../../glossary.md#operation) [serialization](../../../../glossary.md#term-serialization); [persistence](../../../../glossary.md#term-persistence) remains [fail-closed](../../../../glossary.md#term-fail-closed).

## [Scope](../../../../glossary.md#term-scope) and exclusions

Reuse current layers, Sequential, [Tensor](../../../../glossary.md#tensor) operations, ModelExecutionState and public [owners](../../../../glossary.md#term-owner) for
dense/convolutional/fixed-recurrent chains. Preserve define/defineTensor/[graph](../../../../glossary.md#graph)/load/loadGraph and old APIs.
No arbitrary [callback](../../../../glossary.md#term-callback)/[module](../../../../glossary.md#term-module)/graph builder, Layers [registry](../../../../glossary.md#term-registry), [model](../../../../glossary.md#term-model)/session hierarchy, second graph/
pipeline, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) reconstruction table, [Compiler](../../../../glossary.md#term-compiler) edge, [codec](../../../../glossary.md#term-codec)/[backend](../../../../glossary.md#backend)/build change, new [adjoint](../../../../glossary.md#term-adjoint), [runtime](../../../../glossary.md#term-runtime)
lengths/[scan](../../../../glossary.md#term-cumulative-scan) [backpropagation through time](../../../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)), streaming state, bidirectionality or [optimizer](../../../../glossary.md#term-optimizer)/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate) [resume](../../../../glossary.md#schema-serialization-and-checkpoint).

## Contracts and required authority cutover

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>, then
root headings `Core lifecycle`, `Core invariants`, `Module-ownership routing`, `Dependency rules`, `Testing requirements`.
Read these exact headings in the incorporated files under `docs/architecture/contracts/`:

- `extensions-training.md`: `extensions/nn`, `Owned tensor Models`, `modules/training`,
  `Optimizer/training lifecycle`, `Generic graph source and values`, `Canonical graph-model version 2`.
- `foundational-modules.md`: `modules/model`.
- `runtime-prepare-engine.md`: `Public reusable prepared execution`.
- `compiler-autograd.md`: `Compile lifecycle`, `Compiler-owned automatic differentiation`.

Accepted before Java: root `Core lifecycle` and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `Owned tensor Models` distinguish builder-owned
ordinary composition from unchanged [callback](../../../../glossary.md#term-callback)/[graph](../../../../glossary.md#graph)/load reconstruction and save-only eight-kind
[admission](../../../../glossary.md#term-admission). [Signatures](../../../../glossary.md#term-signature), guards, exclusion, native admission, [bindings](../../../../glossary.md#term-binding) and cleanup remain unchanged.
Explanations and ADR0040 accompany the cutover; no [Compiler](../../../../glossary.md#term-compiler)/[Checkpoint](../../../../glossary.md#term-checkpoint) [widening](../../../../glossary.md#term-widening). Stop on broader decisions.

## Dependencies and integration

- Depends on: NN0028, Checkpoint0009, [CPU0025](../../../backends/cpu/tasks/0025-scatter-validation-coordinate-rank.md) and [Compiler0009](../../../modules/compiler/tasks/0009-static-select-and-slice-logical-layout-closure.md), all Complete with required independent reviews. Neither repair depends on NN0029.
- Conflicts with: [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [API](../../../../glossary.md#term-application-programming-interface-api)/[lifecycle](../../../../glossary.md#lifecycle), root/NN contract, shared explanations/[planning](../../../../glossary.md#term-planning); none active.
- Parallel group: None.
- Common base revision: `7e880fb1044168043e45b10419a570d64c5283ae` plus accepted dirty CP0009/NN/CPU0025/Compiler0009, not HEAD alone.
- Integration order: accepted prerequisites/authority → NN → independent Class C review → main root gate.
- Integration [validation](../../../../glossary.md#term-validation): main's single native-enabled root command below passed; Result records the join.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; exclusive serial delegation.
- Historical resume (2026-10-19): main verified Ready/frontier/approval/authority/dependencies/base/conflicts;
  sole executor resumed with partial preserved.<span data-glossary-navigation="true"> ([resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [Ready glossary](../../../../glossary.md#term-ready))</span>

## Public [API](../../../../glossary.md#term-application-programming-interface-api) and defaults

Add public final `io.github.pho001.synaptik.nn.module.ModelBuilder` via `public static ModelBuilder Model.builder()`;
no public constructor. Configuration returns the same builder. [Signatures](../../../../glossary.md#term-signature):

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

Input is set once before steps: floating type, [rank](../../../../glossary.md#term-rank) ≥1, positive static extents; [snapshot](../../../../glossary.md#term-snapshot) [shape](../../../../glossary.md#shape).
[Prototype](../../../../glossary.md#term-prototype) is storage-free, [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous), offset-zero non-[view](../../../../glossary.md#term-view-meanings), requiresGrad=false. Compute/
cold forward retain full-[descriptor](../../../../glossary.md#term-descriptor) matching. Build requires input/nonempty chain. Caller-coordinated
builder is reusable: each build owns fresh children/state unaffected by later edits. Default [seed](../../../../glossary.md#term-seed) 0;
the last seed selected before build applies to that whole build.
For zero-based appended step index i, derive `seed + 0x9E3779B97F4A7C15L * i` with Java long
[wraparound](../../../../glossary.md#term-wraparound); [activations](../../../../glossary.md#term-activation)/transforms count. Use existing L64X128MixRandom algorithms, [Glorot](../../../../glossary.md#term-glorot-uniform) uniform,
exact zero [bias](../../../../glossary.md#term-bias) enabled by default. Each layer uses a fresh transient generator; recurrent [matrix](../../../../glossary.md#term-matrix-meanings)
order/fans remain the cell's. No shared [mutable](../../../../glossary.md#term-mutable)/global [RNG](../../../../glossary.md#graph-rng-state--graphrngstate); same recipes/seed/algorithm reproduce bytes.
[Linear](../../../../glossary.md#term-linear-meanings) infers the last [feature](../../../../glossary.md#term-feature) [extent](../../../../glossary.md#term-extent). Supplied [weights](../../../../glossary.md#term-weight) must satisfy current Linear guards
(positive static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) floating [gradient](../../../../glossary.md#term-gradient)-eligible weight, matching optional [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) bias),
match input [dtype](../../../../glossary.md#data-type--datatype), and be stored [provenance](../../../../glossary.md#provenance)-free [leaves](../../../../glossary.md#term-leaf) admitted by the existing raw copier.
No coercion, expression [evaluation](../../../../glossary.md#term-evaluation) or mutation of supplied [Tensor](../../../../glossary.md#tensor) descriptors/storage/bytes.
Conv defaults: [stride](../../../../glossary.md#term-stride) 1, [padding](../../../../glossary.md#padding) 0, [dilation](../../../../glossary.md#term-dilation) 1, groups 1, bias true; [layouts](../../../../glossary.md#layout) [NCW](../../../../glossary.md#term-ncw)/[NCHW](../../../../glossary.md#term-nchw)/[NCDHW](../../../../glossary.md#conv3d--ncdhw).
Pool defaults: stride equals [kernel](../../../../glossary.md#kernel), padding 0, dilation 1, floor. Flatten requires rank ≥2 and
preserves the first [axis](../../../../glossary.md#term-axis): `[axis0, checked product of remaining extents]`. [CNN](../../../../glossary.md#term-convolutional-neural-network-cnn) axis 0 is [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters);
[TBF](../../../../glossary.md#term-tbf) `[T,B,H]` becomes `[T,B*H]`. Use lastTimeStep first to retain recurrent batch rows.
Recurrent input is time/batch/features (TBF), positive fixed time and all-valid rows; outputs are
all hidden steps `[T,B,H]`, including [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm). Stack recurrent layers directly; lastTimeStep requires
rank-three TBF and explicitly selects `select(0,T-1)` to `[B,H]`. No implicit final-step selection.
Use existing sequence.forward(input), assemble packedOutputs with reshape to `[1,B,H]` then
Tensor.concat(0,...). Each cold composition starts zero h (and zero c for LSTM); cached compute/train
reuses [immutable](../../../../glossary.md#term-immutable) zero [constants](../../../../glossary.md#term-constant), never final state from an earlier invocation. No [runtime](../../../../glossary.md#term-runtime) reset work.

## Concrete acquisition and files

Production changes are limited to existing `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/`
`Model.java`, `OwnedModelSource.java`, `ModelExecutionState.java`, and new `ModelBuilder.java`.
ModelBuilder privately owns `LayerRecipe` (internal factories, no public [callback](../../../../glossary.md#term-callback) entry),
`MaxPoolStep`, `FlattenStep`, `LastTimeStepStep`, `RnnStep`, `GruStep`, `LstmStep`; these are its
only new nested types. [Unary](../../../../glossary.md#term-unary) adapters use [Tensor](../../../../glossary.md#tensor) operations; recurrent adapters own child `sequence`.
No layer/cell/Sequential change is planned.

At build, [snapshot](../../../../glossary.md#term-snapshot) recipes and deduplicate supplied leaves by Tensor [identity](../../../../glossary.md#term-identity-meanings). If nonempty, copy
them once using `GraphCheckpoint.copy(new Source(List.of(), uniqueLeaves, namedLeaves))` with
unique outputs/[PARAMETER](../../../../glossary.md#term-parameter) registrations: zero-[node](../../../../glossary.md#node) raw copy BEFORE expressions. A temporary [leaf](../../../../glossary.md#term-leaf)-only
identity association preserves [aliases](../../../../glossary.md#aliasing-and-arena), not [graph](../../../../glossary.md#graph) reconstruction. No copy of already-private state.
Create fresh layers using copied supplied state or automatic recipes. Add package-private
`Model.fromComposition(TensorDescriptor, Sequential)`, reusing existing SingleTensorModel, owning
the sequence as child `layers`, selecting eval and building its [prototype](../../../../glossary.md#term-prototype) forward once. Paths are
`layers.0.weight`, or `layers.0.sequence.cell.inputWeight`, etc.; preserve [canonical](../../../../glossary.md#term-canonical-form) traversal order.
Add `OwnedModelSource.acquireComposition(Module, Tensor, Tensor, Sequential)`: inventory registered
wrappers and reachable fresh [constants](../../../../glossary.md#term-constant) using ordinary [provenance](../../../../glossary.md#provenance), verify state reachability, retain
exact final leaves/output and chain, then [capture](../../../../glossary.md#term-capture-meanings) existing guards. Use GraphCheckpoint.Source only
as the existing [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) Tensor-list [carrier](../../../../glossary.md#term-storage-carrier); do NOT call copy/capture/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) on this completed chain.
Keep acquireDefinition/adopt semantics unchanged. Add internal `OwnedModelSource.forward(List<Tensor>)`:
builder sources apply retained Sequential after exact [signature](../../../../glossary.md#term-signature) [validation](../../../../glossary.md#term-validation); other sources use existing
GraphCheckpoint.rebind. ModelExecutionState.forward delegates under the same [admission](../../../../glossary.md#term-admission)/live guards.
Compute/configure/train use fixed outputs/cached [bindings](../../../../glossary.md#term-binding), never recipes/forward/reconstruction per [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters).
Built Models retain no recipes/caller leaves. Failed build publishes nothing and mutates no caller.
Save still delegates fixed outputs/current leaves to GraphCheckpoint.save. Unsupported [convolution](../../../../glossary.md#term-channels-first-convolution-layers),
pool/reshape/concat/select/[gather](../../../../glossary.md#gather) chains reject before ArtifactFile work, with no [target](../../../../glossary.md#term-target) mutation,
[Engine](../../../../glossary.md#term-engine) creation, binding freeze or [model](../../../../glossary.md#term-model) invalidation; [inference](../../../../glossary.md#term-inference)/training remains usable afterwards.

Read-only: Sequential/UnaryTensorModule; [Linear](../../../../glossary.md#term-linear-meanings)/[Activation](../../../../glossary.md#term-activation)/Conv1d/2d/3d and initialization;
Rnn/Gru/LstmSequence/results/cells; Tensor.linear/maxPool2d/reshape/concat/select, [descriptors](../../../../glossary.md#term-descriptor)/attrs;
GraphCheckpoint.copy/save.

## Acceptance and focused tests

Add `ModelBuilderTest` and `ModelBuilderExecutionTest` in the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module) test package above;
extend OwnedModelSource/Execution/Training/LifecycleTest only where shared paths need [regression](../../../../glossary.md#term-regression).
Update `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModelTest.java:73`:
its exact public-method-set [guard](../../../../glossary.md#term-guard) must add builder and preserve every old method.
Add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/FluentModelBuilderIntegrationTest.java`.

1. Planned [API](../../../../glossary.md#term-application-programming-interface-api) examples build without user lambdas: `Model.builder().input(DataType.FLOAT32,32,784)
   .seed(42).linear(128).relu().linear(10,false).build()`; both supplied-[weight](../../../../glossary.md#term-weight) overloads and all [activations](../../../../glossary.md#term-activation).
2. Build and numerically execute the [CNN](../../../../glossary.md#term-convolutional-neural-network-cnn): input `[32,1,28,28]`, conv2d(32,3,3), relu, maxPool2d(2,2),
   conv2d(64,3,3), relu, maxPool2d(2,2), flatten, linear(128), relu, linear(10). Spatial extents
   28→26→13→11→5; flatten <span data-glossary-navigation="true"> ([conv2d glossary](../../../../glossary.md#convolution--conv2d); [Spatial glossary](../../../../glossary.md#term-spatial-axes))</span>`[32,1600]`, final `[32,10]`. Also real Conv1d/[Conv3d](../../../../glossary.md#conv3d--ncdhw) forward cases.
3. All three recurrent builders compute against independent ordinary Tensor/cell equations plus
   Engine, not Model.forward/GraphCheckpoint as oracle. Check every time output, repeated same-input
   calls, changed batches, explicit final-step selection; <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Engine glossary](../../../../glossary.md#term-engine); [Model glossary](../../../../glossary.md#term-model); [oracle glossary](../../../../glossary.md#term-test-oracle); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`[20,32,8]→lstm(128)→lstm(64)` yields
   `[20,32,64]`, then lastTimeStep/[linear](../../../../glossary.md#term-linear-meanings)(10) yields `[32,10]`. Use smaller nonzero numerical fixtures
   too; distinguish state carry, gate mistakes and implicit last-step behavior.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>
4. MLP train compares objective, every weight update and subsequent compute with independent Tensor/
   Engine backward plus explicit SGD arithmetic; include RELU/TANH/SIGMOID and nonzero gradients.
   At least one valid multi-step unrolled recurrent train must succeed with existing adjoints and
   match a direct oracle. Static unrolling is not scan BPTT; Conv3d backward still rejects.<span data-glossary-navigation="true"> ([MLP glossary](../../../../glossary.md#term-multilayer-perceptron-mlp); [objective glossary](../../../../glossary.md#term-objective); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [gradients glossary](../../../../glossary.md#term-gradient); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [unrolling glossary](../../../../glossary.md#term-unrolling); [scan glossary](../../../../glossary.md#term-cumulative-scan); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
5. Seed/default byte determinism, distinct step seeds, separate build ownership, frozen structure,
   checked invalid dimensions/ranks/dtypes/geometry, supplied-state alias preservation, and no caller
   mutation across build/forward/train/close. Unsupported supplied produced/unavailable state rejects.<span data-glossary-navigation="true"> ([Seed glossary](../../../../glossary.md#term-seed); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dimensions glossary](../../../../glossary.md#dimension); [dtypes glossary](../../../../glossary.md#data-type--datatype); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
6. Cold forward accepts exact-compatible produced inputs with arbitrary upstream provenance (e.g.
   abs().contiguous()), uses the same owned weights, creates no Engine and changes no cached graph.
   Reuse prepared/session identities across fresh batches; no per-compute graph builds/copies.
   Preserve freeze/retry, native-copy-once, invalidation, exclusion/close, borrowed Engine and detached
   result lifetimes; reuse existing lifecycle evidence.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [contiguous glossary](../../../../glossary.md#term-contiguous); [graph glossary](../../../../glossary.md#graph); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [detached glossary](../../../../glossary.md#detached-model-results); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
7. MLP saves before compute and after train, five-kind v1/activation v2 reloads compute/train correctly.
   Unsupported builder save to absent and existing sentinel targets leaves both unchanged and no
   temporary file; subsequent compute/train succeeds. Existing generic/declarative/graph/load tests pass.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>

## [Validation](../../../../glossary.md#term-validation) and documentation

Worker: focused command recorded below, then
`./gradlew :extensions:nn:test :extensions:nn:javadoc :testing:integration-tests:compileTestJava --max-workers=2 --console=plain` once.
Review reuses evidence; final [Javadoc](../../../../glossary.md#term-javadoc) follows final edits. Main owns the source-matched gate:
`SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain`.
No worker native rebuild/root rerun/performance campaign. Architecture/[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) changes:
none expected, edges/behavior unchanged; existing checks [run](../../../../glossary.md#term-run) at main's gate. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/integration tests cover acquisition.

Class C documentation/Javadoc/glossary/authority review is complete; evidence below. Main syncs statuses.
Preserve THIRD_PARTY_NOTICES.md [SHA-256](../../../../glossary.md#term-sha-256) `8efe7e3c1f83bd5b30b913d447fa3ee95f9c0ee60485c30de7c0008638e7037f`
and [CPU](../../../../glossary.md#term-cpu) `CpuOrdinaryPointwiseCastCleanJavaOracle.java` SHA-256 `57ce4dc701a83d2ed01b96c725197a72f5aa752fe5e200f3a354f9d273f8f842`.
No [commit](../../../../glossary.md#term-commit)/push.

## Result

Status: Complete

Delivered 17 [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) code/test/doc files, including authority/ADR0040/[API](../../../../glossary.md#term-application-programming-interface-api)/examples/glossary/Javadocs.
Planck final APPROVE, zero findings; 17 [hashes](../../../../glossary.md#term-hash) matched, no later code edits.
Focused `./gradlew :extensions:nn:test --tests '*ModelBuilder*Test' --max-workers=2 --console=plain`: 22 pass/0 skips; original 17 unchanged.
Final worker command above: 55 suites/424 NN tests, zero failures/errors/skips; warning-free [Javadoc](../../../../glossary.md#term-javadoc)/integration [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) PASS.
`./gradlew :testing:integration-tests:test --tests '*FluentModelBuilderIntegrationTest' --max-workers=2 --console=plain`: 3 pass/0 skips; published examples compiled/executed.
[CPU](../../../../glossary.md#term-cpu)/[Compiler](../../../../glossary.md#term-compiler) final APPROVE, zero findings; local evidence retained.
Root command above: exit 0, BUILD SUCCESSFUL, 5m14s; 107 tasks, 18 executed/89 UP-TO-DATE.
2291-file reviewed [snapshot](../../../../glossary.md#term-snapshot)/native [hash](../../../../glossary.md#term-hash) unchanged. XML: 4279 tests,
0 failures/errors, 41 skips (CPU 30, [Metal](../../../../glossary.md#term-metal) 10, integration 1); includes UP-TO-DATE evidence, not all newly [run](../../../../glossary.md#term-run).
Architecture 13/0 skips; integration 145/1 skip. Configured mixed-Metal training and Metal-only
unsupported-[operation](../../../../glossary.md#operation) cases executed, zero failures/skips.
Main links: 667 files/9490 links/1346 fragments, 7 historical errors/0 new; diff/protected hashes PASS.
Nonblocking follow-up: characterize larger stacked-[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) CPU cold-[prepare](../../../../glossary.md#prepare) cost. Development 3-step
train was interrupted in active [topological](../../../../glossary.md#term-topological-order) preparation; reduced 2-step train, required 3-step [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)
training/all-family stack numerics pass. Review found no acceptance gap, error or proved [regression](../../../../glossary.md#term-regression);
no new [Ready](../../../../glossary.md#term-ready) task, benchmark campaign or gate.
