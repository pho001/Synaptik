# Task 0026: Declarative model specification and reconstruction

## Status

Complete — implementation frozen; final code/documentation APPROVE; coordinator gates passed.
Checkpoint 0005B, Training 0006 and optional Checkpoint 0006 remain Draft; no wire/lifecycle acceptance.

## Change class

Class C — semantic reconstruction, parameter identity and downstream state-load boundary.
One clean implementation context and independent targeted documentation/review are required.

## Goal and scope

Describe supported Sequential trees and reconstruct their forward configuration/parameter identity
as fully bound descriptor-only models, with NN API, unit/CPU integration tests, Javadoc and guides.
NN retains Model as its sole production dependency; existing supported APIs stay compatible.

## Contracts

- Root [authority](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [ownership](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [dependencies](../../../../../ARCHITECTURE.md#dependency-rules) and
  [testing](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  including [Closed declarative models](../../../../architecture/contracts/extensions-training.md#closed-declarative-models).
  Read [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)
  for unchanged fully bound target/identity-group loading; read
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  for unchanged duplicate/overlap rejection and no replacement in an open session.
- [ADR 0038 NN decision](../../../../design/decisions/0038-declarative-model-and-training-resume.md#nn-decision-selected-for-review).

Stop on ambiguity; no architecture or downstream scope may be invented during execution.

## Dependencies and integration

- Depends on: completed NN 0025/0025A/0025B, existing module/state/layer capability and accepted NN-only contract packet.
- Conflicts with: NN configuration/state APIs and shared NN/Training/Checkpoint contracts.
- Parallel group: None; contract integration, then clean implementation/review.
- Common base revision: accepted NN contract commit `4472320ceb5498c46a661af549d6de74aa27ab6a`.
- Integration order: NN 0026 → Checkpoint 0005B and Training 0006 → optional Checkpoint 0006.
- Integration validation: coordinator architecture suite and focused CPU integration after review;
  full source-matched repository test/Javadoc gate only at the four-capability join.
- Shared-document integration owner: NN/Training/Checkpoint coordinator.
- Ready verification (2026-10-14): coordinator verified completed dependencies, accepted NN-only
  contracts, inactive conflicts and user-approved NN0026 frontier; Model-only edge unchanged.

## Exact public API

Implemented API. All records/interfaces below are public nested members of final
`io.github.pho001.synaptik.nn.module.ModelSpecification`; lists are defensive immutable copies.
DataType/Shape/ScalarValue/TensorDescriptor and Conv*dAttrs are existing public Model types.

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

Final `DeclarativeModel extends Model<Tensor, Tensor>` in nn.module has no public constructor;
adds only `static DeclarativeModel reconstruct(ModelSpecification specification)`,
`ModelSpecification describe()` and overridden `Tensor forward(Tensor input)` to inherited API.
It owns root indexed children directly: no `root`, `model` or other wrapper prefix. Ordinary
Sequential description borrows the source and changes neither ownership nor mode.
Each selected layer adds exactly one public typed `specification()` returning its corresponding
step snapshot, exposing retained attrs/guard policy without reflection. New final
`io.github.pho001.synaptik.nn.layers.Activation extends UnaryTensorModule` has public
`Activation(ModelSpecification.ActivationKind kind)`, `ActivationStep specification()` and
`Tensor forward(Tensor input)`. No lambda/custom activation extension point.

Linear also adds `static Linear fromState(Tensor weight, Optional<Tensor> bias,
boolean automaticInputGuards)`: validates/binds the complete supplied group before return,
retains exact refs, stores the guard distinction explicitly. True preserves current automatic
exact-input-type, positive static final feature, bound-width and Java-array guards; false retains
supplied-state Model promotion/contraction semantics. Bound construction uses no initialization,
random source, factory or weight array; existing constructors/first binding stay unchanged.
An empty Optional means no bias; its container is non-null.

All other inputs/results are non-null. Null fails NPE; invalid schemas/tree/aliases or incompatible
input fail IAE with path/step context; unbound description fails ISE. Checked arithmetic overflow
and Tensor ID exhaustion retain existing failures. Failure returns no partial model/specification,
mutates no source; no new checked exception, resource owner or lock.

## Matrix, wiring and constraints

Support nested Sequential, six parameterized families and three activations above, with all
FLOAT64/FLOAT32/BFLOAT16/FLOAT16 parameter types. Linear supplied state may promote; automatic
Linear and convolutions require exact configured input type; LayerNorm preserves Model promotion
and retained epsilon compatibility (FLOAT16 parameter epsilon is FLOAT32, otherwise parameter type).
No homogeneous dtype restriction, implicit cast or epsilon conversion. Embedding accepts INT32/
INT64 at any rank, appends embedding width and returns table dtype. Activation uses floating Model
semantics. Linear keeps compatible leading Dimensions/ranks; LayerNorm keeps compatible leading
Dimensions and normalized trailing Shape; convolution keeps rank 3/4/5 NCW/NCHW/NCDHW, fixed positive
static channels and compatible variable batch/spatial Dimensions. No saved example/fixed batch field.
`outputDescriptor(input)` may use existing storage-free Model metadata operations on a cold path to
check composition and return output type/Shape/layout/gradient metadata; empty root returns the
exact input descriptor. No data allocation/evaluation/backend admission or generic shape solver.
Forward preserves ordinary operation/provenance order; runtime-dependent checks retain their owners.

Sequence list order is ownership/forward order, each child once; names are zero-based decimal
indices. `Sequence([Linear, Sequence([LayerNorm])])` has `0.weight[,0.bias]`, then
`1.0.scale,1.0.bias`, identically in reconstructed Model. LayerNorm declares scale then bias;
others weight then optional bias. State descriptors predict parameter-only paths/kinds/types/static
Shapes/true grad flags in Module traversal order, not lexical order. Values load separately.
Kernel order W/HW/DHW; complete attrs preserve stride/padding/dilation/groups; no layout option.
Description rejects custom/unknown modules and unbound reservations; never initializes by forward.

Validate closed typed records before reconstruction/copy: <=4096 step occurrences, depth <=64
(root 1). Stack-safe validation protects cold tree processing, not backend capability/shape sizes.
Retain existing Shape/geometry/checked-count rules. No IDs/refs, so cycles/dangling wiring are
unrepresentable through safe constructors; reused record means independent module occurrences.
Future parser separately owns unknown wire IDs/fields, canonical encoding and string/rank/count/
aggregate/file-byte preflight before allocation; no wire work here.

## Aliases and skeleton proof

AliasGroup has >=2 distinct existing parameter paths, matching type/Shape/grad, disjoint membership;
members follow state order, groups follow first-member order. Unlisted paths are singletons. Reject
malformed/repeated/unknown/mismatched/noncanonical groups. Both describe operations recompute exact
current Tensor-identity groups under caller quiescence, including Parameter.replace/strict-load
joins/splits; no creation-time cache or value-based sharing. Physical overlap, views/provenance are
not reconstructed. Training duplicate/overlap rejection and open-session replacement ban stay intact.

Reconstruct one TensorFactory.create descriptor leaf (no layout/storage/provenance) per group or
singleton and install that exact Tensor at all member paths. Linear.fromState and supplied Embedding/
LayerNorm constructors bind leaves directly; convolution reservations use harmless zeros policy/
seed 0 and strict descriptor-state load before any forward. No initializer executes. Return an
evaluation-mode fully bound skeleton with stateDictionary available but no materialized values.
This is the target bridge for existing public Checkpoint load, not a new value installer/runner.
Strict NN load retains exact candidate refs; Checkpoint can derive target groups and load one fresh
value Tensor per identity group unchanged. Next describe observes current groups after value load.

## Files, acceptance and non-goals

- Add ModelSpecification/DeclarativeModel under `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/`;
  Activation in sibling `nn/layers/`; edit six selected layers and focused module/layer/API tests,
  including nested-record/public-surface guards. Helpers stay private within named owner types; no new top-level utility.
- Add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/NnDeclarativeModelEngineIntegrationTest.java`:
  CPU direct-vs-reconstruct+strict-load Linear→RELU→Linear MLP values, changed batch/leading ranks
  and alias identity. Unit fixtures include nested normalization, embedding→projection and all
  rank-specific convolution attrs/provenance; cover all four floating types and both index types.
- Test paths/order/schema, full attrs/epsilon bits, promotion success/failure, unsupported/unbound
  rejection, bounds and alias validation. Test joins/splits through both legal APIs, independent
  equal values and alias-preserving reload. Large descriptor-only fixtures prove no random/
  weight-sized/throwaway allocation, including automatic Linear. No native/performance work.
- Empty/activation-only trees are semantically eligible; no generic zero-node Engine promise,
  backend widening, Conv3d gradient widening, recurrence, bytes/versions/files, production Engine/
  Training/Checkpoint imports, registry/runner, Model.define capture, branches, BatchNorm/Dropout,
  arbitrary records, loss or optimizer semantics. Existing published checkpoint APIs remain unchanged.

## Validation and documentation

Worker final once: `./gradlew :extensions:nn:test :extensions:nn:javadoc --console=plain`.
Coordinator validation commands/results are recorded below. Existing integration edges include NN/CPU;
NnTrainingDependencyContractTest enforces the unchanged exact NN Model-only edge. Add NN API guards.
No planning/native build; full repository test/Javadoc belongs to the four-capability join.
Apply documentation profiles; update running-models user guide, targeted glossary/API/Javadoc
for describe→reconstruct→strict-load, promotion/sharing/limits. Independent Class C review reuses
worker evidence; coordinator owns shared docs. Planning checks local links/anchors, <=200 lines/15KB,
scope and `git diff --check` only.

## Result

Implemented specification/reconstruction, Activation, six snapshots and Linear.fromState: 9 production,
6 NN test files, 2-test CPU integration fixture and declarative-models guide.
Worker `./gradlew :extensions:nn:test :extensions:nn:javadoc --console=plain` passed:
326 tests/43 suites, including 20 new NN tests; no failures/errors/skips. Evidence reused, not rerun.
Coordinator `./gradlew :testing:integration-tests:test --tests '*NnDeclarativeModelEngineIntegrationTest'
--console=plain --max-workers=2` passed 2/2 with no failures/errors/skips.
Separate pre-existing architecture inventory repair `e26a3801c9466aa291c0fa915ce1434d6481e9b1`
is integrated without NN executable changes; actual execution passed 10/10 (8 suites, no failures/errors/skips).
Coordinator post-integration `:testing:architecture-tests:test` passed FROM-CACHE with identical declared inputs.
`compileNnApiConsumer` was UP-TO-DATE from new-NN compilation; final code APPROVE, no P1/P2.
Documentation APPROVE: exact API, guide example, glossary/status; 532 local links/anchors passed.
Final `./gradlew :extensions:nn:javadoc --console=plain --max-workers=2` passed after comment-only
edits; no Java test rerun. `git diff --check` passed.
No unresolved NN issues; full repository test/Javadoc deferred to the four-capability join.
Status: Complete
