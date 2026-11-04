# Task 0026: Declarative [model](../../../../glossary.md#term-model) specification and reconstruction

> **Editorial reading context — not a status update.** A declarative specification records
> supported layer configuration and identity-sharing paths as data, not trained numbers or
> arbitrary Java forward code. NN reconstruction creates fully bound descriptor-only parameter
> placeholders. “Bound” here means wrappers hold real Tensor references; those leaves need
> not have stored values. Strict state load is the distinct value-installation bridge.
> Today's [declarative composition contract](../../../../reference/nn/declarative.md#current-nn-declarative-composition-contract)
> explains that skeleton, and [whole-model persistence](../../../../reference/checkpoint/models.md#whole-model-save-and-automatic-load)
> explains the later composition of definition plus values without retroactively adding a wire codec to this NN task.

## Status

Complete — implementation frozen; final code/documentation APPROVE; coordinator gates passed.
[Checkpoint](../../../../glossary.md#term-checkpoint) 0005B, Training 0006 and optional Checkpoint 0006 remain Draft; no wire/[lifecycle](../../../../glossary.md#lifecycle) acceptance.

## Change class

Class C — semantic reconstruction, [parameter](../../../../glossary.md#term-parameter) [identity](../../../../glossary.md#term-identity-meanings) and downstream state-load boundary.
One clean implementation context and independent targeted documentation/review are required.

## Goal and [scope](../../../../glossary.md#term-scope)

Describe supported Sequential trees and reconstruct their forward configuration/[parameter](../../../../glossary.md#term-parameter) [identity](../../../../glossary.md#term-identity-meanings)
as fully bound [descriptor](../../../../glossary.md#term-descriptor)-only models, with [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [API](../../../../glossary.md#term-application-programming-interface-api), unit/[CPU](../../../../glossary.md#term-cpu) integration tests, [Javadoc](../../../../glossary.md#term-javadoc) and guides.
NN retains [Model](../../../../glossary.md#term-model) as its sole production [dependency](../../../../glossary.md#term-dependency); existing supported APIs stay compatible.

## Contracts

- Root [authority](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [ownership](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [dependencies](../../../../../ARCHITECTURE.md#dependency-rules) and
  [testing](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  including [Closed declarative models](../../../../architecture/contracts/extensions-training.md#closed-declarative-models).
  Read [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  for unchanged fully bound target/identity-group loading; read<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  for unchanged duplicate/overlap rejection and no replacement in an open session.
- [ADR 0038 NN decision](../../../../design/decisions/0038-declarative-model-and-training-resume.md#nn-decision-selected-for-review)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>.

Stop on ambiguity; no architecture or downstream [scope](../../../../glossary.md#term-scope) may be invented during execution.

## Dependencies and integration

- Depends on: completed [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0025/0025A/0025B, existing [module](../../../../glossary.md#term-module)/state/layer capability and accepted NN-only contract packet.
- Conflicts with: NN configuration/state APIs and shared NN/Training/[Checkpoint](../../../../glossary.md#term-checkpoint) contracts.
- Parallel group: None; contract integration, then clean implementation/review.
- Common base revision: accepted NN contract [commit](../../../../glossary.md#term-commit) `4472320ceb5498c46a661af549d6de74aa27ab6a`.
- Integration order: NN 0026 → Checkpoint 0005B and Training 0006 → optional Checkpoint 0006.
- Integration validation: coordinator architecture suite and focused CPU integration after review;
  full source-matched repository test/Javadoc gate only at the four-capability join.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): NN/Training/Checkpoint coordinator.
- Ready verification (2026-10-14): coordinator verified completed dependencies, accepted NN-only
  contracts, inactive conflicts and user-approved NN0026 frontier; Model-only edge unchanged.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [Model glossary](../../../../glossary.md#term-model))</span>

## Exact public [API](../../../../glossary.md#term-application-programming-interface-api)

Implemented API. All records/interfaces below are public nested members of final
`io.github.pho001.synaptik.nn.module.ModelSpecification`; lists are defensive [immutable](../../../../glossary.md#term-immutable) copies.
[DataType](../../../../glossary.md#data-type--datatype)/[Shape](../../../../glossary.md#shape)/[ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue)/TensorDescriptor and Conv*dAttrs are existing public [Model](../../../../glossary.md#term-model) types.

```java
ModelSpecification(Sequence root, List<AliasGroup> aliases)
static ModelSpecification describe(Sequential source)
Sequence root()
List<AliasGroup> aliases()
List<StateDescriptor> stateDescriptors()
TensorDescriptor outputDescriptor(TensorDescriptor input)

sealed interface Step // permits precisely the eight records below
record Sequence(List<Step> steps) implements Step
record LinearStep(long inFeatures, long outFeatures, boolean bias,
                  DataType dataType, boolean automaticInputGuards) implements Step
record LayerNormStep(Shape normalizedShape, DataType dataType, ScalarValue epsilon) implements Step
record EmbeddingStep(long vocabularySize, long embeddingSize, DataType dataType) implements Step
record Conv1dStep(long inChannels, long outChannels, long kernelWidth,
                  boolean bias, DataType dataType, Conv1dAttrs attrs) implements Step
record Conv2dStep(long inChannels, long outChannels, long kernelHeight, long kernelWidth,
                  boolean bias, DataType dataType, Conv2dAttrs attrs) implements Step
record Conv3dStep(long inChannels, long outChannels, long kernelDepth, long kernelHeight,
                  long kernelWidth, boolean bias, DataType dataType, Conv3dAttrs attrs) implements Step
enum ActivationKind { RELU, TANH, SIGMOID }
record ActivationStep(ActivationKind kind) implements Step
record AliasGroup(List<String> paths)
record StateDescriptor(String path, StateKind kind, DataType dataType,
                       Shape shape, boolean requiresGrad)
```

Final `DeclarativeModel extends Model<Tensor, Tensor>` in nn.[module](../../../../glossary.md#term-module) has no public constructor;
adds only `static DeclarativeModel reconstruct(ModelSpecification specification)`,
`ModelSpecification describe()` and overridden `Tensor forward(Tensor input)` to inherited API.
It owns root indexed children directly: no `root`, `model` or other wrapper prefix. Ordinary
Sequential description borrows the source and changes neither [ownership](../../../../glossary.md#term-ownership-meanings) nor mode.
Each selected layer adds exactly one public typed `specification()` returning its corresponding
step [snapshot](../../../../glossary.md#term-snapshot), exposing retained attrs/[guard](../../../../glossary.md#term-guard) policy without reflection. New final
`io.github.pho001.synaptik.nn.layers.Activation extends UnaryTensorModule` has public
`Activation(ModelSpecification.ActivationKind kind)`, `ActivationStep specification()` and
`Tensor forward(Tensor input)`. No lambda/custom [activation](../../../../glossary.md#term-activation) extension point.

[Linear](../../../../glossary.md#term-linear-meanings) also adds `static Linear fromState(Tensor weight, Optional<Tensor> bias,
boolean automaticInputGuards)`: validates/binds the complete supplied group before return,
retains exact refs, stores the guard distinction explicitly. True preserves current automatic
exact-input-type, positive static final [feature](../../../../glossary.md#term-feature), bound-width and Java-array guards; false retains
supplied-state Model promotion/[contraction](../../../../glossary.md#contraction-dimension) semantics. Bound construction uses no initialization,
random source, factory or [weight](../../../../glossary.md#term-weight) array; existing constructors/first [binding](../../../../glossary.md#term-binding) stay unchanged.
An empty Optional means no [bias](../../../../glossary.md#term-bias); its container is non-null.

All other inputs/results are non-null. Null fails NPE; invalid [schemas](../../../../glossary.md#term-schema)/tree/[aliases](../../../../glossary.md#aliasing-and-arena) or incompatible
input fail IAE with path/step context; unbound description fails ISE. Checked arithmetic [overflow](../../../../glossary.md#term-overflow)
and [Tensor](../../../../glossary.md#tensor) ID exhaustion retain existing failures. Failure returns no partial model/specification,
mutates no source; no new checked exception, [resource owner](../../../../glossary.md#ownership-borrowing-and-lease) or lock.

## [Matrix](../../../../glossary.md#term-matrix-meanings), wiring and constraints

Support nested Sequential, six parameterized families and three [activations](../../../../glossary.md#term-activation) above, with all
[FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) [parameter](../../../../glossary.md#term-parameter) types. [Linear](../../../../glossary.md#term-linear-meanings) supplied state may promote; automatic
Linear and convolutions require exact configured input type; LayerNorm preserves [Model](../../../../glossary.md#term-model) promotion
and retained [epsilon](../../../../glossary.md#epsilon) compatibility (FLOAT16 parameter epsilon is FLOAT32, otherwise parameter type).
No [homogeneous](../../../../glossary.md#term-homogeneous) [dtype](../../../../glossary.md#data-type--datatype) restriction, implicit cast or epsilon conversion. [Embedding](../../../../glossary.md#embedding) accepts [INT32](../../../../glossary.md#data-type--datatype)/
[INT64](../../../../glossary.md#data-type--datatype) at any [rank](../../../../glossary.md#term-rank), appends embedding width and returns table dtype. Activation uses floating Model
semantics. Linear keeps compatible leading [Dimensions](../../../../glossary.md#dimension)/ranks; LayerNorm keeps compatible leading
Dimensions and normalized trailing [Shape](../../../../glossary.md#shape); [convolution](../../../../glossary.md#term-channels-first-convolution-layers) keeps rank 3/4/5 [NCW](../../../../glossary.md#term-ncw)/[NCHW](../../../../glossary.md#term-nchw)/[NCDHW](../../../../glossary.md#conv3d--ncdhw), fixed positive
static channels and compatible variable [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[spatial](../../../../glossary.md#term-spatial-axes) Dimensions. No saved example/fixed batch field.
`outputDescriptor(input)` may use existing storage-free Model [metadata](../../../../glossary.md#term-metadata) operations on a [cold path](../../../../glossary.md#term-cold-path) to
check composition and return output type/Shape/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient) metadata; empty root returns the
exact input [descriptor](../../../../glossary.md#term-descriptor). No data allocation/[evaluation](../../../../glossary.md#term-evaluation)/[backend](../../../../glossary.md#backend) [admission](../../../../glossary.md#term-admission) or generic shape solver.
Forward preserves ordinary [operation](../../../../glossary.md#operation)/[provenance](../../../../glossary.md#provenance) order; [runtime](../../../../glossary.md#term-runtime)-dependent checks retain their [owners](../../../../glossary.md#term-owner).

Sequence list order is [ownership](../../../../glossary.md#term-ownership-meanings)/forward order, each child once; names are zero-based decimal
indices. `Sequence([Linear, Sequence([LayerNorm])])` has `0.weight[,0.bias]`, then
`1.0.scale,1.0.bias`, identically in reconstructed Model. LayerNorm declares scale then [bias](../../../../glossary.md#term-bias);
others [weight](../../../../glossary.md#term-weight) then optional bias. State descriptors predict parameter-only paths/kinds/types/static
Shapes/true grad flags in [Module](../../../../glossary.md#term-module) traversal order, not lexical order. Values load separately.
[Kernel](../../../../glossary.md#kernel) order W/HW/DHW; complete attrs preserve [stride](../../../../glossary.md#term-stride)/[padding](../../../../glossary.md#padding)/[dilation](../../../../glossary.md#term-dilation)/groups; no layout option.
Description rejects custom/unknown modules and unbound reservations; never initializes by forward.

Validate closed typed records before reconstruction/copy: <=4096 step occurrences, depth <=64
(root 1). Stack-safe [validation](../../../../glossary.md#term-validation) protects cold tree processing, not [backend capability](../../../../glossary.md#backend-capability)/shape sizes.
Retain existing Shape/geometry/checked-count rules. No IDs/refs, so cycles/dangling wiring are
unrepresentable through safe constructors; reused record means independent module occurrences.
Future parser separately owns unknown wire IDs/fields, [canonical](../../../../glossary.md#term-canonical-form) encoding and string/rank/count/
aggregate/file-byte [preflight](../../../../glossary.md#term-preflight) before allocation; no wire work here.

## [Aliases](../../../../glossary.md#aliasing-and-arena) and skeleton proof

AliasGroup has >=2 distinct existing [parameter](../../../../glossary.md#term-parameter) paths, matching type/[Shape](../../../../glossary.md#shape)/grad, disjoint membership;
members follow state order, groups follow first-member order. Unlisted paths are singletons. Reject
malformed/repeated/unknown/mismatched/noncanonical groups. Both describe operations recompute exact
current [Tensor](../../../../glossary.md#tensor)-[identity](../../../../glossary.md#term-identity-meanings) groups under caller [quiescence](../../../../glossary.md#term-quiescence), including Parameter.replace/strict-load
joins/splits; no creation-time [cache](../../../../glossary.md#term-cache) or value-based sharing. Physical overlap, [views](../../../../glossary.md#term-view-meanings)/[provenance](../../../../glossary.md#provenance) are
not reconstructed. Training duplicate/overlap rejection and open-session replacement ban stay intact.

Reconstruct one TensorFactory.create [descriptor](../../../../glossary.md#term-descriptor) [leaf](../../../../glossary.md#term-leaf) (no [layout](../../../../glossary.md#layout)/storage/provenance) per group or
[singleton](../../../../glossary.md#term-singleton) and install that exact Tensor at all member paths. [Linear](../../../../glossary.md#term-linear-meanings).fromState and supplied [Embedding](../../../../glossary.md#embedding)/
LayerNorm constructors bind leaves directly; [convolution](../../../../glossary.md#term-channels-first-convolution-layers) reservations use harmless zeros policy/
[seed](../../../../glossary.md#term-seed) 0 and strict descriptor-state load before any forward. No initializer executes. Return an
[evaluation](../../../../glossary.md#term-evaluation)-mode fully bound skeleton with stateDictionary available but no materialized values.
This is the [target](../../../../glossary.md#term-target) bridge for existing public [Checkpoint](../../../../glossary.md#term-checkpoint) load, not a new value installer/runner.
Strict [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) load retains exact candidate refs; Checkpoint can derive target groups and load one fresh
value Tensor per identity group unchanged. Next describe observes current groups after value load.

## Files, acceptance and non-goals

- Add ModelSpecification/DeclarativeModel under `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/`;
  Activation in sibling <span data-glossary-navigation="true"> ([Activation glossary](../../../../glossary.md#term-activation))</span>`nn/layers/`; edit six selected layers and focused module/layer/API tests,
  including nested-record/public-surface guards. Helpers stay private within named owner types; no new top-level utility.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [owner glossary](../../../../glossary.md#term-owner))</span>
- Add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/NnDeclarativeModelEngineIntegrationTest.java`:
  CPU direct-vs-reconstruct+strict-load Linear→RELU→Linear MLP values, changed batch/leading ranks
  and alias identity. Unit fixtures include nested normalization, embedding→projection and all
  rank-specific convolution attrs/provenance; cover all four floating types and both index types.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Linear glossary](../../../../glossary.md#term-linear-meanings); [MLP glossary](../../../../glossary.md#term-multilayer-perceptron-mlp); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [alias glossary](../../../../glossary.md#aliasing-and-arena); [identity glossary](../../../../glossary.md#term-identity-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [embedding glossary](../../../../glossary.md#embedding); [rank glossary](../../../../glossary.md#term-rank); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [provenance glossary](../../../../glossary.md#provenance))</span>
- Test paths/order/schema, full attrs/epsilon bits, promotion success/failure, unsupported/unbound
  rejection, bounds and alias validation. Test joins/splits through both legal APIs, independent
  equal values and alias-preserving reload. Large descriptor-only fixtures prove no random/
  weight-sized/throwaway allocation, including automatic Linear. No native/performance work.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [epsilon glossary](../../../../glossary.md#epsilon); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [weight glossary](../../../../glossary.md#term-weight))</span>
- Empty/activation-only trees are semantically eligible; no generic zero-node Engine promise,
  backend widening, Conv3d gradient widening, recurrence, bytes/versions/files, production Engine/
  Training/Checkpoint imports, registry/runner, Model.define capture, branches, BatchNorm/Dropout,
  arbitrary records, loss or optimizer semantics. Existing published checkpoint APIs remain unchanged.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [widening glossary](../../../../glossary.md#term-widening); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [gradient glossary](../../../../glossary.md#term-gradient); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [registry glossary](../../../../glossary.md#term-registry); [Model glossary](../../../../glossary.md#term-model); [capture glossary](../../../../glossary.md#term-capture-meanings); [Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [loss glossary](../../../../glossary.md#loss); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>

## [Validation](../../../../glossary.md#term-validation) and documentation

Worker final once: `./gradlew :extensions:nn:test :extensions:nn:javadoc --console=plain`.
Coordinator validation commands/results are recorded below. Existing integration edges include [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[CPU](../../../../glossary.md#term-cpu);
NnTrainingDependencyContractTest enforces the unchanged exact [NN Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions)-only edge. Add NN [API](../../../../glossary.md#term-application-programming-interface-api) guards.
No [planning](../../../../glossary.md#term-planning)/native build; full repository test/[Javadoc](../../../../glossary.md#term-javadoc) belongs to the four-capability join.
Apply documentation profiles; update running-models user guide, targeted glossary/API/Javadoc
for describe→reconstruct→strict-load, promotion/sharing/limits. Independent Class C review reuses
worker evidence; coordinator owns shared docs. Planning checks local links/anchors, <=200 lines/15KB,
[scope](../../../../glossary.md#term-scope) and `git diff --check` only.

## Result

Implemented specification/reconstruction, [Activation](../../../../glossary.md#term-activation), six snapshots and [Linear](../../../../glossary.md#term-linear-meanings).fromState: 9 production,
6 [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files, 2-test [CPU](../../../../glossary.md#term-cpu) integration [fixture](../../../../glossary.md#term-fixture) and declarative-models guide.
Worker `./gradlew :extensions:nn:test :extensions:nn:javadoc --console=plain` passed:
326 tests/43 suites, including 20 new NN tests; no failures/errors/skips. Evidence reused, not rerun.
Coordinator `./gradlew :testing:integration-tests:test --tests '*NnDeclarativeModelEngineIntegrationTest'
--console=plain --max-workers=2` passed 2/2 with no failures/errors/skips.
Separate pre-existing architecture inventory repair `e26a3801c9466aa291c0fa915ce1434d6481e9b1`
is integrated without NN executable changes; actual execution passed 10/10 (8 suites, no failures/errors/skips).
Coordinator post-integration `:testing:architecture-tests:test` passed FROM-[CACHE](../../../../glossary.md#term-cache) with identical declared inputs.
`compileNnApiConsumer` was UP-TO-DATE from new-NN [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names); final code APPROVE, no P1/P2.
Documentation APPROVE: exact [API](../../../../glossary.md#term-application-programming-interface-api), guide example, glossary/status; 532 local links/anchors passed.
Final `./gradlew :extensions:nn:javadoc --console=plain --max-workers=2` passed after comment-only
edits; no Java test rerun. `git diff --check` passed.
No unresolved NN issues; full repository test/[Javadoc](../../../../glossary.md#term-javadoc) deferred to the four-capability join.
Status: Complete
