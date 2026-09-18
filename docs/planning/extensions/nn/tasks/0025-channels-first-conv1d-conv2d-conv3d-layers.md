# Task 0025: Channels-First Conv1d, Conv2d, and Conv3d Layers

## Status

Complete

## Goal

Add separate final public `Conv1d`, `Conv2d`, and `Conv3d` unary modules for channels-first NCW,
NCHW, and NCDHW input, plus matching stateless `ModuleFactory.standard()` recipes. Each layer
infers only the positive static input-channel extent from its first compatible forward call or a
strict state-dictionary load. Output channels, rank-specific kernel, stride, symmetric padding,
dilation, groups, bias presence, floating data type, `ParameterInitialization`, and seed remain
explicit construction-time facts.

Preserve the existing private reservation lifecycle: `weight` followed by optional `bias` is
published as one complete group; failed initialization is retryable; discovery/export fails while
reserved state is unbound; and a complete strict dictionary may bind the group without running an
initializer.

```text
construction
  -> validate explicit rank-specific geometry and output/group schema
  -> reserve weight, then optional bias
  -> create no Tensor, Parameter, generator, or Tensor ID

first forward(input)
  -> validate exact NCW/NCHW/NCDHW descriptor and checked geometry
  -> infer only C_in from static axis 1
  -> derive [C_out, C_in/groups, spatial kernel...]
  -> derive checked group-aware fan values privately
  -> initialize weight, then optional zero bias
  -> atomically publish the complete state group
  -> return the ordinary rank-specific Tensor expression
```

`Conv1d` delegates to current `Tensor.conv1d`, whose provenance is singleton-height input/weight
expansion, ordinary Conv2d, and height squeeze. `Conv2d` and `Conv3d` use their current first-class
Tensor operations. NN constructs expressions; it does not execute them. Current public execution
is forward-only at this checkpoint, and this task must not claim or implement Conv3d gradients.

## Motivation and mental model

Model, Compiler, CPU, and Engine already own convolution meaning and forward execution. NN adds
only stateful parameter ownership and construction:

```text
NN state and first-forward binding
  -> existing Model Tensor convolution expression
  -> existing Compiler/Planning/Prepare/CPU/Engine forward lifecycle
```

Three concrete types keep axis order, geometry, diagnostics, and parameter Shapes visible. A
public array-ranked `ConvNd` would obscure those contracts and is not justified.

## Scope

- Add final public `Conv1d`, `Conv2d`, and `Conv3d` under `nn.layers`, each extending
  `UnaryTensorModule` and owning only `weight` plus optional `bias`.
- Give each type exactly one automatic constructor, `weight()`, optional `bias()`, and
  `forward(Tensor)` as specified below. Strict state loading is the advanced exact-state ingress;
  add no supplied/eager constructor in this task.
- Use rank-specific scalar geometry components, never arrays, lists, nullable bias, default
  geometry, or a public configuration aggregate.
- Validate explicit facts before reservation. Construction creates no generator, Tensor, Tensor
  ID, or Parameter.
- Require the exact configured floating input type, exact rank three/four/five, and a positive
  static input-channel axis `1`. Batch and spatial Dimensions do not become state schema.
- Require positive output channels, kernels, strides, dilations, and groups; non-negative symmetric
  padding; and `outChannels % groups == 0`.
- Compute kernel volume and these group-aware fans with checked `long` multiplication:

  ```text
  channelsPerGroup = inChannels / groups
  fanIn            = channelsPerGroup * kernelVolume
  fanOut           = (outChannels / groups) * kernelVolume
  ```

- Derive fan values only inside one package-private layer helper. Add no public `Fan`, fan getter,
  convolution initializer type, callback, registry, or `ConvNd` helper.
- Recognize the four closed fan preset values by structural equality. Apply their existing
  Glorot/Kaiming distributions with the checked convolution fans over the actual rank-three,
  rank-four, or rank-five weight Shape. Delegate configured normal/uniform and zero/one once to
  the existing generic `ParameterInitializers.initialize` overload.
- For random policies create exactly one fresh
  `RandomGeneratorFactory.of("L64X128MixRandom").create(seed)` source per attempt. Retain neither
  factory nor source. Zero/one create no source. Optional bias is exact typed zero and is not
  governed by the weight policy or seed.
- Preflight parameter Shapes, checked counts, the Model Java-array limit, effective kernels,
  doubled padding, and every statically knowable spatial-fit/output-geometry fact before generator
  creation, sampling, Tensor allocation, ID consumption, or publication.
- Publish neither wrapper until weight and optional bias creation plus all reservation validation
  succeeds. Failure leaves the group unbound and retryable; completed draws, allocations, and
  opaque Tensor IDs are not rolled back.
- Serialize only the one-time binding critical section on the exact layer, following `Linear` and
  using Module's release/acquire gate. Compatible simultaneous first calls initialize once; an
  incompatible contender may lose after the winner fixes `inChannels`. Do not synchronize later
  expression construction or claim general Module thread safety.
- Add exact rank-specific `ModuleFactory` recipes that invoke the matching constructor once and
  return a fresh unowned module. The factory retains no per-call state.
- Add focused construction, provenance, validation/effect-order, fan/seed, concurrency/retry,
  strict dictionary, replacement, factory-parity, and public-surface tests.
- Finalize public/package Javadocs, `docs/api/training-api.md`, the glossary, and planning evidence
  through a separate clean documentation-focused pass.

## Out of scope

- Public or private semantic `ConvNd`, arbitrary/dynamic spatial rank, geometry arrays, a generic
  layer registry, plugin, reflection, or service lookup.
- Inferring output channels, geometry, groups, bias, data type, policy, algorithm, or seed.
- Asymmetric intrinsic padding, padding modes, causal-specialized, transposed, deformable,
  separable, depthwise-specific, sparse, quantized, or mixed-layout convolution.
- Default/supplied-state constructors, configuration getters, status queries, public
  bind/build/initialize, or relaxed/partial state load.
- New Model semantics or changes to convolution attributes/Tensor methods; Compiler gradients or
  layout inference; CPU lowering; Engine APIs; backend execution; or Conv3d gradients.
- A new Engine/integration fixture. Existing public Tensor-level Engine fixtures prove the three
  operation paths; layer-level execution, publication/cleanup, and the explicit Conv3d backward
  failure belong to NN 0025A.
- Architecture, ADR, dependency, Gradle, architecture-test, backend-conformance, integration-test,
  resource, legacy-source, Training, checkpoint transport, or other-module changes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)
- [ADR 0007](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [NN 0019 automatic Linear](0019-automatic-first-forward-linear-initialization.md)
- [NN 0020A initialized Embedding](0020a-initialized-embedding.md)
- [NN 0020B ModuleFactory](0020b-stateless-standard-module-factory.md)
- [Model 0020 Conv2d](../../../modules/model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md)
- [Model 0025G Conv1d](../../../modules/model/tasks/0025g-ncw-conv1d-composition.md)
- [Model 0025H Conv3d](../../../modules/model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md)
- [Compiler 0006B](../../../modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md)
- [Compiler 0006B6](../../../modules/compiler/tasks/0006b6-final-convolution-logical-layout-closure.md)
- [CPU 0008](../../../backends/cpu/tasks/0008-portable-grouped-nchw-conv2d-execution-foundation.md)
- [CPU 0008A](../../../backends/cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md)
- [Engine 0008](../../../modules/engine/tasks/0008-engine-lifecycle-capability-checkpoint.md)
- [Training API](../../../../api/training-api.md), [Tensor API](../../../../api/tensor-api.md), and
  [glossary](../../../../glossary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- NN owns module state, parameters, layer composition, and recipes and may depend only on Model.
- Model remains sole owner of convolution semantics, Shape, Tensor identity, and provenance. The
  layers create no operation kind, graph node, lowering, kernel, storage, or execution contract.
- `Parameter` remains a real bound wrapper. Reservations are private metadata, never placeholder
  Parameters or Tensors.
- One layer publishes `weight` and optional `bias` as one direct group. Discovery/export remains
  fail-closed while any owned reservation is unbound.
- Strict load remains complete-tree validate-before-install and may infer only `inChannels` from a
  valid candidate weight without initialization or RNG use.
- Conv1d must call `Tensor.conv1d`; it must not hide or bypass the visible
  `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition. Conv2d/Conv3d call their current receivers.
- Compiler owns autograd. Conv1d structurally inherits rank-edit/Conv2d gradients, Conv2d retains
  current gradients, and Conv3d remains rejected by backward-capable compilation.
- If implementation requires architecture, dependency, Module-lifecycle, public initializer, or
  execution changes, stop and report the conflict.

## Current-source audit

- `Module` already provides reservations, group publication, acquire/release observation,
  fail-closed discovery/export, and strict reserved-state load. It needs no change.
- `Linear` is the oracle for synchronized one-time binding, retry, access, variable leading
  Dimensions, and strict-load compatibility.
- `ParameterInitialization` is a closed eight-policy value whose kind/configuration is not public.
  `ParameterInitializers` exposes normal/uniform/zero/one, four rank-two fan methods, and exhaustive
  generic dispatch.
- Passing convolution weight Shapes to rank-two fan methods is invalid. The selected helper can
  recognize only the four public fan presets by value, calculate private convolution fans, call
  normal/uniform directly for them, and use generic dispatch for the other policies without
  widening public policy/initializer APIs.
- `ModuleFactory` is final, instance-field-free, and reflection-locked. It can add three methods
  without gaining fields or ownership.
- Model exposes `Tensor.conv1d`, `conv2d`, and `conv3d`, each with biased/unbiased overloads and
  rank-specific attributes.
- The implemented NN API explanation is `docs/api/training-api.md`; `docs/api/nn-api.md` does not
  exist. Update the established page rather than add an overlapping guide.
- NN's Gradle project depends only on Model. `NnTrainingDependencyContractTest` needs no change.
- `EngineConvolutionIntegrationTest` already executes Tensor-level public forward fixtures for all
  three ranks. NN 0025A remains the layer-level checkpoint owner.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — concrete layers and one package-private initializer.
- `io.github.pho001.synaptik.nn.module` — `UnaryTensorModule`, `ModuleFactory`, state ownership.
- `io.github.pho001.synaptik.nn.initialization` — unchanged policy and initializer primitives.
- Model convolution/datatype/Shape/Tensor packages — unchanged semantics.

No package is added or renamed.

Type placement:

- `...nn.layers.Conv1d`, `Conv2d`, `Conv3d` — public rank-specific stateful layers.
- `...nn.layers.ConvolutionParameterInitialization` — package-private final field-free owner only
  of checked kernel-volume/fan derivation and policy dispatch; not a layer/config/public ConvNd.
- `...nn.module.ModuleFactory` — existing stateless recipe namespace.

## Public API decision

Use these exact scalar signatures; add no overloads:

```java
public final class Conv1d extends UnaryTensorModule {
    public Conv1d(long outChannels, long kernelWidth, long stride, long padding,
            long dilation, long groups, boolean bias, DataType dataType,
            ParameterInitialization weightInitialization, long seed);
    public Parameter weight();
    public Optional<Parameter> bias();
    @Override public Tensor forward(Tensor input);
}

public final class Conv2d extends UnaryTensorModule {
    public Conv2d(long outChannels, long kernelHeight, long kernelWidth,
            long strideHeight, long strideWidth, long paddingHeight, long paddingWidth,
            long dilationHeight, long dilationWidth, long groups, boolean bias,
            DataType dataType, ParameterInitialization weightInitialization, long seed);
    public Parameter weight();
    public Optional<Parameter> bias();
    @Override public Tensor forward(Tensor input);
}

public final class Conv3d extends UnaryTensorModule {
    public Conv3d(long outChannels, long kernelDepth, long kernelHeight, long kernelWidth,
            long strideDepth, long strideHeight, long strideWidth, long paddingDepth,
            long paddingHeight, long paddingWidth, long dilationDepth, long dilationHeight,
            long dilationWidth, long groups, boolean bias, DataType dataType,
            ParameterInitialization weightInitialization, long seed);
    public Parameter weight();
    public Optional<Parameter> bias();
    @Override public Tensor forward(Tensor input);
}
```

`ModuleFactory` adds these exact matching signatures:

```java
public Conv1d conv1d(
        long outChannels, long kernelWidth, long stride, long padding, long dilation,
        long groups, boolean bias, DataType dataType,
        ParameterInitialization weightInitialization, long seed);

public Conv2d conv2d(
        long outChannels, long kernelHeight, long kernelWidth,
        long strideHeight, long strideWidth, long paddingHeight, long paddingWidth,
        long dilationHeight, long dilationWidth, long groups, boolean bias,
        DataType dataType, ParameterInitialization weightInitialization, long seed);

public Conv3d conv3d(
        long outChannels, long kernelDepth, long kernelHeight, long kernelWidth,
        long strideDepth, long strideHeight, long strideWidth,
        long paddingDepth, long paddingHeight, long paddingWidth,
        long dilationDepth, long dilationHeight, long dilationWidth, long groups,
        boolean bias, DataType dataType,
        ParameterInitialization weightInitialization, long seed);
```

Each delegates once without validation or exception translation.
No geometry/state getter is added; Parameter descriptors and expression provenance are the
truthful inspection surfaces.

## Tensor, weight, bias, and result Shapes

| Layer | Input | Weight | Bias | Result |
|---|---|---|---|---|
| Conv1d | `[N,C_in,W]` | `[C_out,C_in/groups,K_w]` | `[C_out]` optional | `[N,C_out,W_out]` |
| Conv2d | `[N,C_in,H,W]` | `[C_out,C_in/groups,K_h,K_w]` | `[C_out]` optional | `[N,C_out,H_out,W_out]` |
| Conv3d | `[N,C_in,D,H,W]` | `[C_out,C_in/groups,K_d,K_h,K_w]` | `[C_out]` optional | `[N,C_out,D_out,H_out,W_out]` |

For every spatial axis:

```text
effectiveKernel = dilation * (kernel - 1) + 1
numerator       = input + 2 * padding - effectiveKernel
output          = floor(numerator / stride) + 1
```

Literal/static arithmetic is checked. A static negative numerator fails before parameter effects.
Dynamic batch/spatial Dimensions remain Model obligations, but channel axis `1` must be static and
positive. Empty batch or output spatial extents follow current Model semantics where valid.

## Validation and error ownership

Construction validates: positive output; positive kernels, strides, dilations in rank order;
non-negative paddings in rank order; positive groups; output divisibility; non-null type then
policy; floating type; then checked kernel volume, construction-known fan-out, effective kernels,
and doubled padding. Only then are names reserved.

`forward` rejects null, then validates exact type, rank, static positive channel axis, input
divisibility, checked weight/bias Shapes/counts/Java-array limits, and rank-order spatial geometry.
Every deterministic descriptor failure precedes effects. After binding, the same checks require
the exact bound `inChannels` before Model delegation. Layer code owns configuration/inferred-schema
diagnostics; Tensor methods own delegated expression/ID effects. Factory failures are untranslated.

## Initialization, fan, and seed semantics

- Weight uses its actual convolution Shape and row-major initializer order.
- Glorot normal: `sqrt(2/(fanIn+fanOut))`; Glorot uniform: `sqrt(6/(fanIn+fanOut))`;
  Kaiming-ReLU normal: `sqrt(2/fanIn)`; Kaiming-ReLU uniform: `sqrt(6/fanIn)`.
  Convert checked positive fans separately to binary64 before addition/division.
- Configured normal/uniform and zero/one retain current exact meanings.
- A random attempt uses one fresh exact standard stream seeded with the layer seed. Retry repeats
  the deterministic stream. Bias consumes no draw and is always typed zero.
- Weight is created before bias. Policy/seed/groups/geometry are not state entries.

## First-forward concurrency and atomic publication

- Validate, acquire-check, synchronize only when unbound, revalidate inside, initialize, bind once
  in `weight[, bias]` order, verify publication, then construct the expression outside the lock.
- Racing compatible calls produce one group. A different-channel loser fails after the winner
  establishes schema. No supported accessor observes a partial group.
- Failure before publication leaves all reservations unbound. Later expression failure does not
  undo published state. IDs/draws are never rolled back.
- This is layer-local initialization safety, not a whole-Model transaction or general Module
  thread-safety promise.

## State paths and strict dictionary behavior

- State order is exactly `weight`, then optional `bias`.
- Before binding, parameter access/discovery/export fails under current reservation rules; no-bias
  `bias()` is always empty.
- Candidate weight must be gradient-eligible with exact type, rank, output/kernel extents, positive
  static group-local channels, checked count/array limit, and establish
  `inChannels = channelsPerGroup * groups` by checked multiplication. Bias must be gradient-eligible
  with exact type and `[C_out]`.
- Complete-tree validation/preparation precedes replacement/publication. Any missing/unexpected,
  kind/type/Shape/gradient/overflow/schema failure changes nothing.
- Load retains exact candidate references and uses no policy/seed/RNG/draw/new Tensor ID.
- After binding, `Parameter.replace` preserves exact type/Shape/gradient schema while allowing a
  different Tensor identity.

## Affected files

Production/Javadoc (6):

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Conv1d.java` (new)
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Conv2d.java` (new)
3. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Conv3d.java` (new)
4. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/ConvolutionParameterInitialization.java` (new, package-private)
5. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
6. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModuleFactory.java`

Focused tests (6):

7. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/Conv1dTest.java` (new)
8. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/Conv2dTest.java` (new)
9. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/Conv3dTest.java` (new)
10. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/ConvolutionInitializationTest.java` (new)
11. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/ConvolutionStateDictionaryTest.java` (new)
12. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleFactoryTest.java`

Documentation/planning (5):

13. `docs/api/training-api.md`
14. `docs/glossary.md`
15. this task
16. `docs/planning/extensions/nn/master-plan.md`
17. `docs/planning/roadmap.md`

Review unchanged unless an in-scope defect is proved: Module/Parameter/StateDictionary,
ParameterInitialization/ParameterInitializers, Linear/Embedding, Tensor/Compile APIs,
architecture/ADRs/tests, integration/conformance, Gradle, resources, and other modules.

## Maximum scope

At most 17 paths: six production/Javadoc, six tests, five documentation/planning. Stop for another
helper/test owner, public initializer API, Module lifecycle edit, Gradle/dependency/architecture
change, or cross-module executable edit. The family is cohesive because all ranks share the one
initialization/state/factory decision while keeping separate public geometry and diagnostics.

## Acceptance criteria

- Exactly three new final public unary layer types have the exact single constructors and methods
  above; no extra public/protected members, interfaces, nested types, arrays, or `ConvNd` exist.
- Construction validates explicit facts in order and reserves state with no Tensor/Parameter/RNG/
  ID effects. First forward infers only static positive axis-1 channels and publishes complete
  state before returning a usable expression.
- Conv1d provenance is the existing two expansions, Conv2d, squeeze; Conv2d/Conv3d use exactly one
  current first-class occurrence. Bias and attributes have exact Model order.
- All eight policies work for all ranks/types. Fan presets use exact checked group-aware formulas;
  configured/constants retain current meaning; deterministic seed/draw/zero-bias/retry/no-RNG
  behavior passes.
- Variable compatible batch/spatial Shapes work after binding. Wrong type/rank/channel/group,
  dynamic/zero channels, geometry, overflow, and array-limit cases fail with required effects.
- Compatible concurrency initializes once; incompatible races leave one complete winner and a
  clean loser failure; injected initialization/ID failure publishes no state and retry is exact.
- State paths/order, unbound discovery/export, complete strict-load binding/rollback, candidate
  identity, replacement, direct/factory parity, and no-RNG/no-ID load pass.
- `ModuleFactory` remains final, singleton, instance-field-free, and stateless, gaining exactly
  three public recipes returning fresh unowned concrete layers with constructor-parity failures.
- Reflection/`javap`, external-package compilation, Javadocs, imports, and forbidden-symbol scans
  lock the API. `Module`, `ParameterInitialization`, and `ParameterInitializers` surfaces do not
  change.
- No Engine fixture is added. Existing Tensor fixture stays unchanged; NN 0025A remains Draft.
- Documentation clearly states Shapes, cross-correlation delegation, initialization/state/
  concurrency, forward-only execution, and absent Conv3d gradients.
- A separate clean documentation-focused pass finalizes all Javadocs/docs/evidence.
- Focused and final NN tests, NN Javadoc, links/surface/scope/status/whitespace, and
  `git diff --check` pass. Repository-wide tests are deferred to NN 0025A/CI.

## Tests / validation

```bash
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.Conv1dTest \
  --tests io.github.pho001.synaptik.nn.layers.Conv2dTest \
  --tests io.github.pho001.synaptik.nn.layers.Conv3dTest \
  --tests io.github.pho001.synaptik.nn.layers.ConvolutionInitializationTest \
  --tests io.github.pho001.synaptik.nn.layers.ConvolutionStateDictionaryTest \
  --tests io.github.pho001.synaptik.nn.module.ModuleFactoryTest \
  --tests io.github.pho001.synaptik.nn.layers.LinearInitializationTest \
  --tests io.github.pho001.synaptik.nn.module.ModuleDeferredParameterTest \
  --tests io.github.pho001.synaptik.nn.module.StateDictionaryTest
```

After Java stabilizes, run once:

```bash
./gradlew :extensions:nn:test
```

Documentation pass:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

Also validate local links/anchors, fences, LF/final newlines/trailing whitespace, exact 17 paths,
empty staging, exact API with reflection/`javap` and external compilation, exact provenance, no
`ConvNd`/array/public fan/helper/downstream import/dependency/resource/build edit, and status
agreement (`0025` Complete after all gates, `0025A` Draft with no detailed file).

No Java/Javadoc run is required for this planning-only creation. The implementation documentation
pass reuses stable final NN test evidence unless it changes executable Java afterward.

## Documentation, Javadoc, and glossary impact

- Fully document every new type/member/helper, including all parameters, returns, nullability,
  schema, arithmetic, RNG, ID/allocation failures, state, and threading behavior.
- Update layers package Javadoc and Training API with the complete layer/factory mental model.
- Update glossary NN module/layer, parameter initialization, automatic initialization, factory,
  and convolution entries. Add no public fan or ConvNd term.
- Tensor/Compile APIs remain unchanged because no Model/Compiler surface changes. Architecture,
  integration/conformance, Gradle, and other modules remain unchanged because ownership,
  dependencies, and execution semantics do not change. Record reasoned no-change conclusions.

## Risks

- Dense fan formulas overcount grouped connectivity; cover groups and kernel volume greater than
  one for all fan presets.
- A generic helper could become de facto ConvNd; keep only initialization arithmetic shared.
- Late deterministic validation or partial publication could bind invalid state; preflight and one
  group publication are mandatory.
- Loose strict-load validation could accept a different kernel/group schema; lock all extents.
- Reusing generators makes state history-dependent; use a fresh exact seeded stream per attempt.
- Calling Conv2d directly from Conv1d would conceal required provenance; call `Tensor.conv1d`.
- Gradient-eligible Conv3d Parameters do not prove a Compiler gradient rule; document the
  fail-closed backward boundary.

## Dependencies

Verified `Complete`: NN 0001, 0003, 0004, 0004A, 0005, 0010, 0011, 0019, 0020, 0020A, 0020B;
Model 0020, 0025G, 0025H; Compiler 0006B and 0006B6; CPU 0008 and 0008A; Engine 0008.

NN 0021B-0024 are unrelated Draft rows. The global roadmap explicitly selects NN 0025 after
Engine 0008, so this specification records the authorized out-of-order exception to NN numeric
order. The skipped rows contribute no files/dependencies, and this task does not alter their
recurrent/Data contracts.

## Follow-up tasks

- NN 0025A remains a concise Draft row without a detailed specification. After 0025 it owns the
  layer-level Engine forward checkpoint, grouped/bias fixtures, materialization/publication/
  cleanup, documentation, and explicit Conv3d backward failure unless Compiler 0006C completes.
- Compiler 0006C remains the Draft owner of Conv3d adjoint proof/gradient closure and is not a
  forward dependency.
- Other padding/layout/convolution families need separate future tasks.

Do not create task 0025A during implementation.

## Architecture impact

Expected impact: None. The task stays on NN -> Model, reuses existing state lifecycle, and composes
existing Tensor semantics. Stop if implementation needs another boundary.

## Implementation prompt

```text
You are the clean-context implementation agent for Synaptik NN task 0025 in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit, stage, or push.

Read AGENTS.md, ARCHITECTURE.md, the planning guide, roadmap, NN master plan, this task, and every
directly referenced completed NN/Model/Compiler/CPU/Engine contract. Inspect current Module state
reservation/dictionary code, initialization, Linear, ModuleFactory, Tensor convolution APIs,
tests, Javadocs, API guides, glossary, Gradle, and architecture/integration tests.

Implement exactly this task within 17 paths. Preserve rank-specific scalar APIs, private checked
group-aware fans, exact L64X128MixRandom behavior, atomic publication, strict load, and visible
Tensor delegation. Add no ConvNd, public fan/helper, Model/Compiler/CPU/Engine behavior, Conv3d
gradient, integration fixture, Gradle/dependency/resource/architecture change. Stop on uncertainty
or scope overflow.

Run focused validation and one final NN suite after Java stabilizes. Hand the actual diff and
evidence to a distinct clean documentation-focused agent in the same change. That pass follows
the General, API/Javadoc, Planning, and Example profiles; finalizes Javadocs, Training API,
glossary and planning evidence/no-change conclusions; reuses stable Java evidence; and runs final
Javadoc/documentation/surface/scope/whitespace checks.

Keep 0025 Ready until every gate passes, then mark task/master/roadmap Complete while leaving
0025A Draft without a detailed file. Record context IDs, notes, evidence, completion summary,
unresolved issues, and exact final Status.
```

## Local decisions

- One automatic constructor per rank keeps the surface minimal; strict load is exact-state ingress.
- Scalar geometry, not attribute objects, keeps axis meaning and factory signatures explicit.
- Share only package-private initialization arithmetic; retain rank validation in each layer.
- Detect fan presets through structural equality, preserving the closed public policy API.
- Leave layer Engine execution to 0025A because underlying operation execution already has public
  fixtures and this task remains one-module cohesive.
- Update `training-api.md`; no `nn-api.md` currently exists.

## Known limitations

- Only channels-first ranks and symmetric intrinsic padding are supported.
- Input channels must be statically positive when binding; batch/spatial Dimensions may vary under
  current Model/Compiler rules.
- Eager parameter creation retains the Java-array element-count boundary.
- NN constructs expressions, not values. Conv3d forward works, but gradients remain fail-closed.

## Validation evidence

- Implementation context `01a0b4f8-4ad6-7ec2-a4cb-d6efba2b4cfc` passed the exact focused
  selection with 56 tests and zero failures, errors, or skips. Its one authoritative
  `./gradlew :extensions:nn:test` passed 292 tests in 41 suites with zero failures, errors, or
  skips. This remains the original implementation evidence; the later corrective run below is
  the final controlling test evidence.
- The implementation pass also passed `javap`, reflection/public-surface checks,
  external-package API compilation, forbidden-symbol/downstream-import/delegation/RNG/publication
  scans, exact 12-Java-path scope, LF/final-newline checks, and `git diff --check`.
- Documentation context `01a0b50a-a48a-7ed1-b48a-72b42b7b94d3` ran
  `./gradlew :extensions:nn:javadoc` exactly once after documentation stabilized; it completed
  successfully in two seconds with three actionable tasks, two executed and one up-to-date.
  Generated pages for `Conv1d`, `Conv2d`, `Conv3d`, `ModuleFactory`, and the layers package were
  inspected for the finalized contracts.
- A documented external-package `Conv2d` factory example compiled and ran against the built Model
  and NN public APIs, including its expected weight and output Shapes. Final `javap -public`
  inspection confirmed each final convolution type has one scalar constructor plus `weight()`,
  `bias()`, and `forward(Tensor)`, and confirmed the three factory recipes.
- Final documentation validation passed local Markdown links and anchors, code fences,
  terminology/status consistency, generated-page inspection, LF/final newlines, exact 17-path
  scope, empty staging, and `git diff --check`. Every documentation-pass Java patch hunk was
  confined to Javadoc or comments; a Java-aware lexical audit discarded comments and whitespace,
  fingerprinted the remaining executable token streams, and confirmed that this pass introduced
  no executable Java edit.
- Corrective test-only context `01a0b51a-34a3-7732-a476-dfd0ca7977f9` changed only `Conv2dTest`,
  `Conv3dTest`, `ConvolutionInitializationTest`, `ConvolutionStateDictionaryTest`, and
  `ModuleFactoryTest`. It added independent group-aware fan oracles, preflight/no-ID and
  constructor-order checks, compatible and incompatible first-binding races, identifier
  exhaustion between weight and bias with deterministic retry, stricter state-load rollback and
  boundary checks, all-rank variable-Shape checks, and direct/factory parity. Its focused
  selection passed 70 tests in the same nine suites with zero failures, errors, or skips. Its new
  single authoritative `./gradlew :extensions:nn:test` passed 306 tests in 41 suites with zero
  failures, errors, or skips. The coordinator independently confirmed those XML totals and
  reviewed the new tests against the acceptance criteria. These runs are the final controlling
  test evidence; the corrective context changed no production code, API, Javadoc, explanatory
  documentation, glossary semantics, architecture, or task scope and did not rerun Javadoc.

## Implementation notes

- Added separate final channels-first `Conv1d`, `Conv2d`, and `Conv3d` layers with deferred
  positive static input-channel binding, exact rank-specific state Shapes, atomic complete-state
  publication, retryable failure, strict complete-state load, and stateless factory recipes.
- Kept convolution fan calculation and policy dispatch package-private. Random policies use one
  fresh `L64X128MixRandom` per attempt; zero/one avoid generator creation; optional bias is typed
  zero and consumes no random draw.
- Preserved Tensor provenance and ownership: `Conv1d` delegates to `Tensor.conv1d`, `Conv2d` and
  `Conv3d` delegate to their matching Tensor methods, and every layer constructs expressions
  without executing values. Conv3d backward-capable compilation remains fail-closed.
- Documentation finalized all affected Javadocs, the layers package summary, the Training API
  explanation and example, glossary context, master plan, roadmap, and this completion evidence.
- No architecture contract or current-architecture-plan edit is needed because dependency
  direction, module ownership, and lifecycle boundaries are unchanged. No ADR or architecture
  test is needed because no architectural decision or dependency rule changed. No backend
  conformance or integration test is needed because the layers add NN expression composition,
  not backend execution; NN 0025A remains the Draft owner of Engine-facing layer fixtures.
- No Gradle, dependency, or resource edit is needed because NN continues to depend only on Model.
  Model, Compiler, CPU, Engine, and Training implementation APIs remain unchanged: existing Tensor
  convolution operations are composed, and no execution or gradient capability is added.
  `ParameterInitialization` and `ParameterInitializers` remain unchanged because group-aware fans
  are a private layer concern. No other module needs modification for this bounded NN capability.

## Completion summary

- Completed changes: Implemented the three rank-specific channels-first convolution layers,
  their private initialization support, stateless factory recipes, focused tests, complete
  Javadocs, and user/planning documentation.
- Files changed or created: Exactly the six production Java files, six focused test files, and
  five documentation/planning files listed by this task's bounded scope.
- Tests and validation: The final controlling corrective evidence is a passing 70-test focused
  selection across the same nine suites and a passing 306-test/41-suite authoritative NN run,
  both with zero failures, errors, or skips. The earlier 56-test and 292-test runs remain recorded
  as implementation history. The documentation context passed its single NN Javadoc run,
  generated-page review, external example compilation/run, public-surface inspection, and final
  documentation/scope/hygiene gates; Javadoc was not repeated after test-only correction.
- Documentation-agent review: Complete in clean context
  `01a0b50a-a48a-7ed1-b48a-72b42b7b94d3`; no executable Java behavior changed and no stable Java
  suite was repeated during that documentation pass. This resumed evidence synchronization found
  no documentation semantic mismatch in corrective test-only context
  `01a0b51a-34a3-7732-a476-dfd0ca7977f9`.
- Documentation impact: `training-api.md` now explains construction versus execution, deferred
  binding, state Shapes, groups, bias, initialization, state loading, factory use, and current
  execution/gradient boundaries with a runnable example. Planning status and evidence are aligned.
- Javadoc review: All affected public types, constructors, methods, the package summary, the
  package-private helper, and factory recipes document inputs, results, failures, state,
  ownership, retry, and threading where applicable.
- Glossary impact: Existing convolution, standard-module-factory, and unary-module definitions
  now include the precise channels-first/deferred-state context; no implementation-only term was
  added.
- Unresolved issues: None.
- Follow-up required: NN 0025A remains the next concise Draft checkpoint and has no detailed task
  file.

Status: Complete
