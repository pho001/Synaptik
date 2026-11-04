# Task 0025: Channels-First Conv1d, [Conv2d](../../../../glossary.md#convolution--conv2d), and [Conv3d](../../../../glossary.md#conv3d--ncdhw) Layers

> **Editorial reading context — not a status update.** NCW, NCHW and NCDHW name logical
> batch/channel/spatial axes for one, two and three spatial dimensions. The application fixes
> output channels and kernel geometry; the layer infers only input channels and owns its
> complete weight/bias publication. Forward delegates ordinary Tensor convolution composition,
> not a backend kernel call. Groups partition channel connections and therefore affect both
> weight Shape and initialization fan counts.
> Today's [NN convolution contract](../../../../reference/nn/convolution.md#current-nn-channels-first-convolution-contract)
> connects those roles to examples; [strict state loading](../../../../reference/nn/state.md#current-nn-state-dictionary-contract)
> explains the separate path that installs supplied exact references without running an initializer.

## Status

Complete

## Goal

Add separate final public `Conv1d`, [`Conv2d`](../../../../glossary.md#convolution--conv2d), and [`Conv3d`](../../../../glossary.md#conv3d--ncdhw) [unary](../../../../glossary.md#term-unary) [modules](../../../../glossary.md#term-module) for channels-first [NCW](../../../../glossary.md#term-ncw),
[NCHW](../../../../glossary.md#term-nchw), and [NCDHW](../../../../glossary.md#conv3d--ncdhw) input, plus matching stateless `ModuleFactory.standard()` recipes. Each layer
infers only the positive static input-[channel](../../../../glossary.md#term-channel) [extent](../../../../glossary.md#term-extent) from its first compatible forward call or a
strict state-dictionary load. Output channels, [rank](../../../../glossary.md#term-rank)-specific [kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), symmetric [padding](../../../../glossary.md#padding),
[dilation](../../../../glossary.md#term-dilation), groups, [bias](../../../../glossary.md#term-bias) presence, floating [data type](../../../../glossary.md#data-type--datatype), `ParameterInitialization`, and [seed](../../../../glossary.md#term-seed) remain
explicit construction-time facts.

Preserve the existing private reservation [lifecycle](../../../../glossary.md#lifecycle): [`weight`](../../../../glossary.md#term-weight) followed by optional `bias` is
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

`Conv1d` delegates to current `Tensor.conv1d`, whose [provenance](../../../../glossary.md#provenance) is [singleton](../../../../glossary.md#term-singleton)-height input/weight
expansion, ordinary Conv2d, and height squeeze. `Conv2d` and `Conv3d` use their current first-class
[Tensor](../../../../glossary.md#tensor) operations. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) constructs expressions; it does not execute them. Current public execution
is forward-only at this [checkpoint](../../../../glossary.md#term-checkpoint), and this task must not claim or implement Conv3d [gradients](../../../../glossary.md#term-gradient).

## Motivation and [mental model](../../../../glossary.md#term-mental-model)

[Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [CPU](../../../../glossary.md#term-cpu), and [Engine](../../../../glossary.md#term-engine) already own [convolution](../../../../glossary.md#term-channels-first-convolution-layers) meaning and forward execution. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) adds
only stateful [parameter](../../../../glossary.md#term-parameter) [ownership](../../../../glossary.md#term-ownership-meanings) and construction:

```text
NN state and first-forward binding
  -> existing Model Tensor convolution expression
  -> existing Compiler/Planning/Prepare/CPU/Engine forward lifecycle
```

Three concrete types keep [axis](../../../../glossary.md#term-axis) order, geometry, diagnostics, and parameter [Shapes](../../../../glossary.md#shape) visible. A
public array-ranked `ConvNd` would obscure those contracts and is not justified.

## [Scope](../../../../glossary.md#term-scope)

- Add final public `Conv1d`, [`Conv2d`](../../../../glossary.md#convolution--conv2d), and [`Conv3d`](../../../../glossary.md#conv3d--ncdhw) under `nn.layers`, each extending
  `UnaryTensorModule` and owning only [`weight`](../../../../glossary.md#term-weight) plus optional [`bias`](../../../../glossary.md#term-bias).
- Give each type exactly one automatic constructor, `weight()`, optional `bias()`, and
  `forward(Tensor)` as specified below. Strict state loading is the advanced exact-state ingress;
  add no supplied/eager constructor in this task.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Use rank-specific scalar geometry components, never arrays, lists, nullable bias, default
  geometry, or a public configuration aggregate.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Validate explicit facts before reservation. Construction creates no generator, Tensor, Tensor
  ID, or Parameter.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Parameter glossary](../../../../glossary.md#term-parameter))</span>
- Require the exact configured floating input type, exact rank three/four/five, and a positive
  static input-channel axis <span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis))</span>`1`. [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) and [spatial](../../../../glossary.md#term-spatial-axes) [Dimensions](../../../../glossary.md#dimension) do not become state [schema](../../../../glossary.md#term-schema).
- Require positive output channels, kernels, strides, dilations, and groups; non-negative symmetric
  padding; and <span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [strides glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding))</span>`outChannels % groups == 0`.
- Compute [kernel](../../../../glossary.md#kernel) volume and these group-aware fans with checked `long` multiplication:

  ```text
  channelsPerGroup = inChannels / groups
  fanIn            = channelsPerGroup * kernelVolume
  fanOut           = (outChannels / groups) * kernelVolume
  ```

- Derive fan values only inside one package-private layer helper. Add no public `Fan`, fan getter,
  convolution initializer type, callback, registry, or <span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [callback glossary](../../../../glossary.md#term-callback); [registry glossary](../../../../glossary.md#term-registry))</span>`ConvNd` helper.
- Recognize the four closed fan preset values by structural equality. Apply their existing
  Glorot/Kaiming distributions with the checked convolution fans over the actual rank-three,
  rank-four, or rank-five weight Shape. Delegate configured normal/uniform and zero/one once to
  the existing generic <span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality); [Glorot glossary](../../../../glossary.md#term-glorot-uniform); [Kaiming glossary](../../../../glossary.md#term-kaiming); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [Shape glossary](../../../../glossary.md#shape))</span>`ParameterInitializers.initialize` overload.
- For random policies create exactly one fresh
  `RandomGeneratorFactory.of("L64X128MixRandom").create(seed)` source per attempt. Retain neither
  factory nor source. Zero/one create no source. Optional bias is exact typed zero and is not
  governed by the weight policy or seed.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- Preflight parameter Shapes, checked counts, the Model Java-array limit, effective kernels,
  doubled padding, and every statically knowable spatial-fit/output-geometry fact before generator
  creation, sampling, Tensor allocation, ID consumption, or publication.<span data-glossary-navigation="true"> ([Preflight glossary](../../../../glossary.md#term-preflight); [Model glossary](../../../../glossary.md#term-model); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [publication glossary](../../../../glossary.md#term-publication))</span>
- Publish neither wrapper until weight and optional bias creation plus all reservation validation
  succeeds. Failure leaves the group unbound and retryable; completed draws, allocations, and
  opaque Tensor IDs are not rolled back.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- Serialize only the one-time [binding](../../../../glossary.md#term-binding) critical section on the exact layer, following [`Linear`](../../../../glossary.md#term-linear-meanings) and
  using Module's release/acquire gate. Compatible simultaneous first calls initialize once; an
  incompatible contender may lose after the winner fixes <span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [release glossary](../../../../glossary.md#term-release))</span>`inChannels`. Do not synchronize later
  expression construction or claim general Module thread safety.
- Add exact [rank](../../../../glossary.md#term-rank)-specific `ModuleFactory` recipes that invoke the matching constructor once and
  return a fresh unowned module. The factory retains no per-call state.
- Add focused construction, provenance, validation/effect-order, fan/seed, concurrency/retry,
  strict dictionary, replacement, factory-parity, and public-surface tests.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Finalize public/package Javadocs, `docs/api/training-api.md`, the glossary, and planning evidence
  through a separate clean documentation-focused pass.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Public or private semantic `ConvNd`, arbitrary/dynamic spatial rank, geometry arrays, a generic
  layer registry, plugin, reflection, or service lookup.<span data-glossary-navigation="true"> ([spatial glossary](../../../../glossary.md#term-spatial-axes); [rank glossary](../../../../glossary.md#term-rank); [registry glossary](../../../../glossary.md#term-registry); [plugin glossary](../../../../glossary.md#term-plugin); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Inferring output channels, geometry, groups, [bias](../../../../glossary.md#term-bias), [data type](../../../../glossary.md#data-type--datatype), policy, algorithm, or [seed](../../../../glossary.md#term-seed).
- Asymmetric intrinsic padding, padding modes, causal-specialized, transposed, deformable,
  separable, depthwise-specific, sparse, quantized, or mixed-layout convolution.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [layout glossary](../../../../glossary.md#layout); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Default/supplied-state constructors, configuration getters, status queries, public
  bind/build/initialize, or relaxed/partial state load.
- New Model semantics or changes to convolution attributes/Tensor methods; Compiler gradients or
  layout inference; CPU lowering; Engine APIs; backend execution; or Conv3d gradients.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient); [inference glossary](../../../../glossary.md#term-inference-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- A new Engine/integration fixture. Existing public Tensor-level Engine fixtures prove the three
  operation paths; layer-level execution, publication/cleanup, and the explicit Conv3d backward
  failure belong to NN 0025A.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [operation glossary](../../../../glossary.md#operation); [publication glossary](../../../../glossary.md#term-publication); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Architecture, ADR, dependency, Gradle, architecture-test, backend-conformance, integration-test,
  resource, legacy-source, Training, checkpoint transport, or other-module changes.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [ADR 0007](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [NN 0019 automatic Linear](0019-automatic-first-forward-linear-initialization.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- [NN 0020A initialized Embedding](0020a-initialized-embedding.md)<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding))</span>
- [NN 0020B ModuleFactory](0020b-stateless-standard-module-factory.md)
- [Model 0020 Conv2d](../../../modules/model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- [Model 0025G Conv1d](../../../modules/model/tasks/0025g-ncw-conv1d-composition.md)
- [Model 0025H Conv3d](../../../modules/model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- [Compiler 0006B](../../../modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Compiler 0006B6](../../../modules/compiler/tasks/0006b6-final-convolution-logical-layout-closure.md)
- [CPU 0008](../../../backends/cpu/tasks/0008-portable-grouped-nchw-conv2d-execution-foundation.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0008A](../../../backends/cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md)
- [Engine 0008](../../../modules/engine/tasks/0008-engine-lifecycle-capability-checkpoint.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>, [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>, and
  [glossary](../../../../glossary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns [module](../../../../glossary.md#term-module) state, [parameters](../../../../glossary.md#term-parameter), layer composition, and recipes and may depend only on [Model](../../../../glossary.md#term-model).
- Model remains sole owner of convolution semantics, Shape, Tensor identity, and provenance. The
  layers create no operation kind, graph node, lowering, kernel, storage, or execution contract.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>
- `Parameter` remains a real bound wrapper. Reservations are private metadata, never placeholder
  Parameters or Tensors.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- One layer publishes [`weight`](../../../../glossary.md#term-weight) and optional [`bias`](../../../../glossary.md#term-bias) as one direct group. Discovery/export remains
  fail-closed while any owned reservation is unbound.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Strict load remains complete-tree validate-before-install and may infer only `inChannels` from a
  valid candidate weight without initialization or RNG use.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Conv1d must call `Tensor.conv1d`; it must not hide or bypass the visible
  `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition. [Conv2d](../../../../glossary.md#convolution--conv2d)/[Conv3d](../../../../glossary.md#conv3d--ncdhw) call their current receivers.
- Compiler owns autograd. Conv1d structurally inherits rank-edit/Conv2d gradients, Conv2d retains
  current gradients, and Conv3d remains rejected by backward-capable compilation.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [rank glossary](../../../../glossary.md#term-rank); [gradients glossary](../../../../glossary.md#term-gradient); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- If implementation requires architecture, dependency, Module-lifecycle, public initializer, or
  execution changes, stop and report the conflict.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Current-source audit

- [`Module`](../../../../glossary.md#term-module) already provides reservations, group publication, acquire/release observation,
  fail-closed discovery/export, and strict reserved-state load. It needs no change.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [release glossary](../../../../glossary.md#term-release); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- [`Linear`](../../../../glossary.md#term-linear-meanings) is the oracle for synchronized one-time binding, retry, access, variable leading
  Dimensions, and strict-load compatibility.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [binding glossary](../../../../glossary.md#term-binding); [Dimensions glossary](../../../../glossary.md#dimension))</span>
- `ParameterInitialization` is a closed eight-policy value whose kind/configuration is not public.
  `ParameterInitializers` exposes normal/uniform/zero/one, four rank-two fan methods, and exhaustive
  generic dispatch.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Passing convolution weight Shapes to rank-two fan methods is invalid. The selected helper can
  recognize only the four public fan presets by value, calculate private convolution fans, call
  normal/uniform directly for them, and use generic dispatch for the other policies without
  widening public policy/initializer APIs.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [weight glossary](../../../../glossary.md#term-weight); [Shapes glossary](../../../../glossary.md#shape); [widening glossary](../../../../glossary.md#term-widening))</span>
- `ModuleFactory` is final, instance-field-free, and reflection-locked. It can add three methods
  without gaining fields or ownership.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Model](../../../../glossary.md#term-model) exposes `Tensor.conv1d`, [`conv2d`](../../../../glossary.md#convolution--conv2d), and [`conv3d`](../../../../glossary.md#conv3d--ncdhw), each with biased/unbiased overloads and
  rank-specific attributes.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- The implemented [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [API](../../../../glossary.md#term-application-programming-interface-api) explanation is `docs/api/training-api.md`; `docs/api/nn-api.md` does not
  exist. Update the established page rather than add an overlapping guide.
- NN's [Gradle](../../../../glossary.md#term-gradle) project depends only on Model. `NnTrainingDependencyContractTest` needs no change.
- `EngineConvolutionIntegrationTest` already executes Tensor-level public forward fixtures for all
  three ranks. NN 0025A remains the layer-level checkpoint owner.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [owner glossary](../../../../glossary.md#term-owner))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — concrete layers and one package-private initializer.
- `io.github.pho001.synaptik.nn.module` — `UnaryTensorModule`, `ModuleFactory`, state [ownership](../../../../glossary.md#term-ownership-meanings).
- `io.github.pho001.synaptik.nn.initialization` — unchanged policy and initializer primitives.
- [Model](../../../../glossary.md#term-model) [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/datatype/[Shape](../../../../glossary.md#shape)/[Tensor](../../../../glossary.md#tensor) packages — unchanged semantics.

No package is added or renamed.

Type placement:

- `...nn.layers.Conv1d`, [`Conv2d`](../../../../glossary.md#convolution--conv2d), [`Conv3d`](../../../../glossary.md#conv3d--ncdhw) — public [rank](../../../../glossary.md#term-rank)-specific stateful layers.
- `...nn.layers.ConvolutionParameterInitialization` — package-private final field-free owner only
  of checked kernel-volume/fan derivation and policy dispatch; not a layer/config/public ConvNd.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [kernel glossary](../../../../glossary.md#kernel); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `...nn.module.ModuleFactory` — existing stateless recipe namespace.

## Public [API](../../../../glossary.md#term-application-programming-interface-api) decision

Use these exact [scalar](../../../../glossary.md#term-scalar) [signatures](../../../../glossary.md#term-signature); add no overloads:

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

Each delegates once without [validation](../../../../glossary.md#term-validation) or exception translation.
No geometry/state getter is added; [Parameter](../../../../glossary.md#term-parameter) [descriptors](../../../../glossary.md#term-descriptor) and expression [provenance](../../../../glossary.md#provenance) are the
truthful inspection surfaces.

## [Tensor](../../../../glossary.md#tensor), [weight](../../../../glossary.md#term-weight), [bias](../../../../glossary.md#term-bias), and result [Shapes](../../../../glossary.md#shape)

| Layer | Input | Weight | Bias | Result |
|---|---|---|---|---|
| Conv1d | `[N,C_in,W]` | `[C_out,C_in/groups,K_w]` | `[C_out]` optional | `[N,C_out,W_out]` |
| [Conv2d](../../../../glossary.md#convolution--conv2d) | `[N,C_in,H,W]` | `[C_out,C_in/groups,K_h,K_w]` | `[C_out]` optional | `[N,C_out,H_out,W_out]` |
| [Conv3d](../../../../glossary.md#conv3d--ncdhw) | `[N,C_in,D,H,W]` | `[C_out,C_in/groups,K_d,K_h,K_w]` | `[C_out]` optional | `[N,C_out,D_out,H_out,W_out]` |

For every [spatial axis](../../../../glossary.md#spatial-axes-channels-and-window-sampling):

```text
effectiveKernel = dilation * (kernel - 1) + 1
numerator       = input + 2 * padding - effectiveKernel
output          = floor(numerator / stride) + 1
```

Literal/static arithmetic is checked. A static negative numerator fails before [parameter](../../../../glossary.md#term-parameter) effects.
Dynamic [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[spatial](../../../../glossary.md#term-spatial-axes) [Dimensions](../../../../glossary.md#dimension) remain [Model](../../../../glossary.md#term-model) obligations, but [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis) `1` must be static and
positive. Empty batch or output spatial extents follow current Model semantics where valid.

## [Validation](../../../../glossary.md#term-validation) and error [ownership](../../../../glossary.md#term-ownership-meanings)

Construction validates: positive output; positive [kernels](../../../../glossary.md#kernel), [strides](../../../../glossary.md#term-stride), dilations in [rank](../../../../glossary.md#term-rank) order;
non-negative paddings in rank order; positive groups; output divisibility; non-null type then
policy; floating type; then checked [kernel](../../../../glossary.md#kernel) volume, construction-known [fan-out](../../../../glossary.md#term-fan-out), effective kernels,
and doubled [padding](../../../../glossary.md#padding). Only then are names reserved.

`forward` rejects null, then validates exact type, rank, static positive [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis), input
divisibility, checked [weight](../../../../glossary.md#term-weight)/[bias](../../../../glossary.md#term-bias) [Shapes](../../../../glossary.md#shape)/counts/Java-array limits, and rank-order [spatial](../../../../glossary.md#term-spatial-axes) geometry.
Every deterministic [descriptor](../../../../glossary.md#term-descriptor) failure precedes effects. After [binding](../../../../glossary.md#term-binding), the same checks require
the exact bound `inChannels` before [Model](../../../../glossary.md#term-model) delegation. Layer code owns configuration/inferred-[schema](../../../../glossary.md#term-schema)
diagnostics; [Tensor](../../../../glossary.md#tensor) methods own delegated expression/ID effects. Factory failures are untranslated.

## Initialization, fan, and [seed](../../../../glossary.md#term-seed) semantics

- [Weight](../../../../glossary.md#term-weight) uses its actual [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [Shape](../../../../glossary.md#shape) and [row-major](../../../../glossary.md#term-row-major) initializer order.
- [Glorot](../../../../glossary.md#term-glorot-uniform) normal: `sqrt(2/(fanIn+fanOut))`; Glorot uniform: `sqrt(6/(fanIn+fanOut))`;
  Kaiming-ReLU normal: `sqrt(2/fanIn)`; [Kaiming](../../../../glossary.md#term-kaiming)-[ReLU](../../../../glossary.md#term-relu) uniform: `sqrt(6/fanIn)`.
  Convert checked positive fans separately to binary64 before addition/division.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Configured normal/uniform and zero/one retain current exact meanings.
- A random attempt uses one fresh exact standard stream seeded with the layer seed. Retry repeats
  the deterministic stream. Bias consumes no draw and is always typed zero.
- Weight is created before [bias](../../../../glossary.md#term-bias). Policy/seed/groups/geometry are not state entries.

## First-forward [concurrency](../../../../glossary.md#term-concurrency) and [atomic publication](../../../../glossary.md#term-atomic-publication)

- Validate, acquire-check, synchronize only when unbound, revalidate inside, initialize, bind once
  in `weight[, bias]` order, verify [publication](../../../../glossary.md#term-publication), then construct the expression outside the lock.
- Racing compatible calls produce one group. A different-channel loser fails after the winner
  establishes schema. No supported accessor observes a partial group.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [schema glossary](../../../../glossary.md#term-schema))</span>
- Failure before publication leaves all reservations unbound. Later expression failure does not
  undo published state. IDs/draws are never rolled back.
- This is layer-local initialization safety, not a whole-Model transaction or general Module
  thread-safety promise.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [transaction glossary](../../../../glossary.md#term-transaction); [Module glossary](../../../../glossary.md#term-module))</span>

## State paths and strict dictionary behavior

- State order is exactly [`weight`](../../../../glossary.md#term-weight), then optional [`bias`](../../../../glossary.md#term-bias).
- Before [binding](../../../../glossary.md#term-binding), [parameter](../../../../glossary.md#term-parameter) access/discovery/export fails under current reservation rules; no-bias
  `bias()` is always empty.
- Candidate weight must be gradient-eligible with exact type, rank, output/kernel extents, positive
  static group-local channels, checked count/array limit, and establish<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [rank glossary](../../../../glossary.md#term-rank); [kernel glossary](../../../../glossary.md#kernel))</span>
  `inChannels = channelsPerGroup * groups` by checked multiplication. Bias must be gradient-eligible
  with exact type and `[C_out]`.
- Complete-tree validation/preparation precedes replacement/publication. Any missing/unexpected,
  kind/type/Shape/gradient/overflow/schema failure changes nothing.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication); [Shape glossary](../../../../glossary.md#shape); [overflow glossary](../../../../glossary.md#term-overflow); [schema glossary](../../../../glossary.md#term-schema))</span>
- Load retains exact candidate references and uses no policy/[seed](../../../../glossary.md#term-seed)/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/draw/new [Tensor](../../../../glossary.md#tensor) ID.
- After binding, `Parameter.replace` preserves exact type/Shape/gradient schema while allowing a
  different Tensor identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Affected files

Production/[Javadoc](../../../../glossary.md#term-javadoc) (6):

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

Documentation/[planning](../../../../glossary.md#term-planning) (5):

13. `docs/api/training-api.md`
14. `docs/glossary.md`
15. this task
16. `docs/planning/extensions/nn/master-plan.md`
17. `docs/planning/roadmap.md`

Review unchanged unless an in-[scope](../../../../glossary.md#term-scope) defect is proved: [Module](../../../../glossary.md#term-module)/[Parameter](../../../../glossary.md#term-parameter)/StateDictionary,
ParameterInitialization/ParameterInitializers, [Linear](../../../../glossary.md#term-linear-meanings)/[Embedding](../../../../glossary.md#embedding), [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) APIs,
architecture/ADRs/tests, integration/[conformance](../../../../glossary.md#term-conformance), [Gradle](../../../../glossary.md#term-gradle), resources, and other modules.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 17 paths: six production/[Javadoc](../../../../glossary.md#term-javadoc), six tests, five documentation/[planning](../../../../glossary.md#term-planning). Stop for another
helper/test [owner](../../../../glossary.md#term-owner), public initializer [API](../../../../glossary.md#term-application-programming-interface-api), [Module](../../../../glossary.md#term-module) [lifecycle](../../../../glossary.md#lifecycle) edit, [Gradle](../../../../glossary.md#term-gradle)/[dependency](../../../../glossary.md#term-dependency)/architecture
change, or cross-module executable edit. The family is cohesive because all ranks share the one
initialization/state/factory decision while keeping separate public geometry and diagnostics.

## Acceptance criteria

- Exactly three new final public unary layer types have the exact single constructors and methods
  above; no extra public/protected members, interfaces, nested types, arrays, or <span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>`ConvNd` exist.
- Construction validates explicit facts in order and reserves state with no Tensor/Parameter/RNG/
  ID effects. First forward infers only static positive axis-1 channels and publishes complete
  state before returning a usable expression.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [axis glossary](../../../../glossary.md#term-axis))</span>
- Conv1d provenance is the existing two expansions, Conv2d, squeeze; Conv2d/Conv3d use exactly one
  current first-class occurrence. Bias and attributes have exact Model order.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [occurrence glossary](../../../../glossary.md#term-occurrence); [Bias glossary](../../../../glossary.md#term-bias); [Model glossary](../../../../glossary.md#term-model))</span>
- All eight policies work for all ranks/types. Fan presets use exact checked group-aware formulas;
  configured/constants retain current meaning; deterministic seed/draw/zero-bias/retry/no-RNG
  behavior passes.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [seed glossary](../../../../glossary.md#term-seed))</span>
- Variable compatible batch/spatial Shapes work after binding. Wrong type/rank/channel/group,
  dynamic/zero channels, geometry, overflow, and array-limit cases fail with required effects.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [spatial glossary](../../../../glossary.md#term-spatial-axes); [Shapes glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [rank glossary](../../../../glossary.md#term-rank); [channel glossary](../../../../glossary.md#term-channel); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Compatible concurrency initializes once; incompatible races leave one complete winner and a
  clean loser failure; injected initialization/ID failure publishes no state and retry is exact.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- State paths/order, unbound discovery/export, complete strict-load binding/rollback, candidate
  identity, replacement, direct/factory parity, and no-RNG/no-ID load pass.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `ModuleFactory` remains final, singleton, instance-field-free, and stateless, gaining exactly
  three public recipes returning fresh unowned concrete layers with constructor-parity failures.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- Reflection/`javap`, external-package compilation, Javadocs, imports, and forbidden-symbol scans
  lock the API. <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [symbol glossary](../../../../glossary.md#term-symbol); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`Module`](../../../../glossary.md#term-module), `ParameterInitialization`, and `ParameterInitializers` surfaces do not
  change.
- No [Engine](../../../../glossary.md#term-engine) [fixture](../../../../glossary.md#term-fixture) is added. Existing [Tensor](../../../../glossary.md#tensor) fixture stays unchanged; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0025A remains Draft.
- Documentation clearly states Shapes, cross-correlation delegation, initialization/state/
  concurrency, forward-only execution, and absent Conv3d gradients.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient))</span>
- A separate clean documentation-focused pass finalizes all Javadocs/docs/evidence.
- Focused and final NN tests, NN [Javadoc](../../../../glossary.md#term-javadoc), links/surface/[scope](../../../../glossary.md#term-scope)/status/whitespace, and
  `git diff --check` pass. Repository-wide tests are deferred to NN 0025A/[CI](../../../../glossary.md#term-continuous-integration-ci).

## Tests / [validation](../../../../glossary.md#term-validation)

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

After Java stabilizes, [run](../../../../glossary.md#term-run) once:

```bash
./gradlew :extensions:nn:test
```

Documentation pass:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

Also validate local links/anchors, fences, LF/final newlines/trailing whitespace, exact 17 paths,
empty staging, exact [API](../../../../glossary.md#term-application-programming-interface-api) with reflection/`javap` and external [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), exact [provenance](../../../../glossary.md#provenance), no
`ConvNd`/array/public fan/helper/downstream import/[dependency](../../../../glossary.md#term-dependency)/resource/build edit, and status
agreement (`0025` Complete after all gates, `0025A` Draft with no detailed file).

No Java/[Javadoc](../../../../glossary.md#term-javadoc) run is required for this [planning](../../../../glossary.md#term-planning)-only creation. The implementation documentation
pass reuses stable final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test evidence unless it changes executable Java afterward.

## Documentation, [Javadoc](../../../../glossary.md#term-javadoc), and glossary impact

- Fully document every new type/member/helper, including all parameters, returns, nullability,
  schema, arithmetic, RNG, ID/allocation failures, state, and threading behavior.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability); [schema glossary](../../../../glossary.md#term-schema); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Update layers package Javadoc and Training [API](../../../../glossary.md#term-application-programming-interface-api) with the complete layer/factory [mental model](../../../../glossary.md#term-mental-model).
- Update glossary NN module/layer, parameter initialization, automatic initialization, factory,
  and convolution entries. Add no public fan or ConvNd term.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Tensor/Compile APIs remain unchanged because no Model/Compiler surface changes. Architecture,
  integration/conformance, Gradle, and other modules remain unchanged because ownership,
  dependencies, and execution semantics do not change. Record reasoned no-change conclusions.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Risks

- Dense fan formulas overcount grouped connectivity; cover groups and kernel volume greater than
  one for all fan presets.
- A generic helper could become de facto ConvNd; keep only initialization arithmetic shared.
- Late deterministic validation or partial publication could bind invalid state; preflight and one
  group publication are mandatory.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Loose strict-load [validation](../../../../glossary.md#term-validation) could accept a different [kernel](../../../../glossary.md#kernel)/group [schema](../../../../glossary.md#term-schema); lock all extents.
- Reusing generators makes state history-dependent; use a fresh exact seeded stream per attempt.
- Calling [Conv2d](../../../../glossary.md#convolution--conv2d) directly from Conv1d would conceal required [provenance](../../../../glossary.md#provenance); call `Tensor.conv1d`.
- Gradient-eligible Conv3d Parameters do not prove a Compiler gradient rule; document the
  fail-closed backward boundary.<span data-glossary-navigation="true"> ([Gradient glossary](../../../../glossary.md#term-gradient); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Parameters glossary](../../../../glossary.md#term-parameter); [Compiler glossary](../../../../glossary.md#term-compiler); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Dependencies

Verified `Complete`: [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001, 0003, 0004, 0004A, 0005, 0010, 0011, 0019, 0020, 0020A, 0020B;
[Model](../../../../glossary.md#term-model) 0020, 0025G, 0025H; [Compiler](../../../../glossary.md#term-compiler) 0006B and 0006B6; [CPU](../../../../glossary.md#term-cpu) 0008 and 0008A; [Engine](../../../../glossary.md#term-engine) 0008.

NN 0021B-0024 are unrelated Draft rows. The global roadmap explicitly selects NN 0025 after
Engine 0008, so this specification records the authorized out-of-order exception to NN numeric
order. The skipped rows contribute no files/dependencies, and this task does not alter their
recurrent/Data contracts.

## Follow-up tasks

- NN 0025A remains a concise Draft row without a detailed specification. After 0025 it owns the
  layer-level Engine forward checkpoint, grouped/bias fixtures, materialization/publication/
  cleanup, documentation, and explicit Conv3d backward failure unless Compiler 0006C completes.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [bias glossary](../../../../glossary.md#term-bias); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compiler 0006C remains the Draft owner of Conv3d adjoint proof/gradient closure and is not a
  forward dependency.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [adjoint glossary](../../../../glossary.md#term-adjoint); [gradient glossary](../../../../glossary.md#term-gradient); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Other [padding](../../../../glossary.md#padding)/[layout](../../../../glossary.md#layout)/[convolution](../../../../glossary.md#term-channels-first-convolution-layers) families need separate future tasks.

Do not create task 0025A during implementation.

## Architecture impact

Expected impact: None. The task stays on [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) -> [Model](../../../../glossary.md#term-model), reuses existing state [lifecycle](../../../../glossary.md#lifecycle), and composes
existing [Tensor](../../../../glossary.md#tensor) semantics. Stop if implementation needs another boundary.

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

- One automatic constructor per [rank](../../../../glossary.md#term-rank) keeps the surface minimal; strict load is exact-state ingress.
- [Scalar](../../../../glossary.md#term-scalar) geometry, not attribute objects, keeps [axis](../../../../glossary.md#term-axis) meaning and factory [signatures](../../../../glossary.md#term-signature) explicit.
- Share only package-private initialization arithmetic; retain rank [validation](../../../../glossary.md#term-validation) in each layer.
- Detect fan presets through [structural equality](../../../../glossary.md#term-structural-equality), preserving the closed public policy [API](../../../../glossary.md#term-application-programming-interface-api).
- Leave layer Engine execution to 0025A because underlying operation execution already has public
  fixtures and this task remains one-module cohesive.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [operation glossary](../../../../glossary.md#operation); [module glossary](../../../../glossary.md#term-module))</span>
- Update `training-api.md`; no `nn-api.md` currently exists.

## Known limitations

- Only channels-first ranks and symmetric intrinsic [padding](../../../../glossary.md#padding) are supported.
- Input channels must be statically positive when binding; batch/spatial Dimensions may vary under
  current Model/Compiler rules.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [spatial glossary](../../../../glossary.md#term-spatial-axes); [Dimensions glossary](../../../../glossary.md#dimension); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Eager](../../../../glossary.md#term-eager-evaluation) [parameter](../../../../glossary.md#term-parameter) creation retains the Java-array [element](../../../../glossary.md#term-element)-count boundary.
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) constructs expressions, not values. [Conv3d](../../../../glossary.md#conv3d--ncdhw) forward works, but [gradients](../../../../glossary.md#term-gradient) remain [fail-closed](../../../../glossary.md#term-fail-closed).

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a0b4f8-4ad6-7ec2-a4cb-d6efba2b4cfc` passed the exact focused
  selection with 56 tests and zero failures, errors, or skips. Its one authoritative
  `./gradlew :extensions:nn:test` passed 292 tests in 41 suites with zero failures, errors, or
  skips. This remains the original implementation evidence; the later corrective run below is
  the final controlling test evidence.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The implementation pass also passed `javap`, reflection/public-surface checks,
  external-package API compilation, forbidden-symbol/downstream-import/delegation/RNG/publication
  scans, exact 12-Java-path scope, LF/final-newline checks, and <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [symbol glossary](../../../../glossary.md#term-symbol); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [publication glossary](../../../../glossary.md#term-publication); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation context `01a0b50a-a48a-7ed1-b48a-72b42b7b94d3` ran
  `./gradlew :extensions:nn:javadoc` exactly once after documentation stabilized; it completed
  successfully in two seconds with three actionable tasks, two executed and one up-to-date.
  Generated pages for `Conv1d`, [`Conv2d`](../../../../glossary.md#convolution--conv2d), [`Conv3d`](../../../../glossary.md#conv3d--ncdhw), `ModuleFactory`, and the layers package were
  inspected for the finalized contracts.
- A documented external-package `Conv2d` factory example compiled and ran against the built Model
  and NN public APIs, including its expected weight and output Shapes. Final <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [weight glossary](../../../../glossary.md#term-weight); [Shapes glossary](../../../../glossary.md#shape))</span>`javap -public`
  inspection confirmed each final convolution type has one scalar constructor plus <span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [scalar glossary](../../../../glossary.md#term-scalar))</span>`weight()`,
  `bias()`, and `forward(Tensor)`, and confirmed the three factory recipes.
- Final documentation validation passed local Markdown links and anchors, code fences,
  terminology/status consistency, generated-page inspection, LF/final newlines, exact 17-path
  scope, empty staging, and `git diff --check`. Every documentation-pass Java patch hunk was
  confined to Javadoc or comments; a Java-aware lexical audit discarded comments and whitespace,
  fingerprinted the remaining executable token streams, and confirmed that this pass introduced
  no executable Java edit.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [token glossary](../../../../glossary.md#term-token))</span>
- Corrective test-only context `01a0b51a-34a3-7732-a476-dfd0ca7977f9` changed only `Conv2dTest`,
  `Conv3dTest`, `ConvolutionInitializationTest`, `ConvolutionStateDictionaryTest`, and
  `ModuleFactoryTest`. It added independent group-aware fan oracles, preflight/no-ID and
  constructor-order checks, compatible and incompatible first-binding races, identifier
  exhaustion between weight and bias with deterministic retry, stricter state-load rollback and
  boundary checks, all-rank variable-Shape checks, and direct/factory parity. Its focused
  selection passed 70 tests in the same nine suites with zero failures, errors, or skips. Its new
  single authoritative <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [binding glossary](../../../../glossary.md#term-binding); [bias glossary](../../../../glossary.md#term-bias); [rollback glossary](../../../../glossary.md#term-rollback); [rank glossary](../../../../glossary.md#term-rank))</span>`./gradlew :extensions:nn:test` passed 306 tests in 41 suites with zero
  failures, errors, or skips. The coordinator independently confirmed those XML totals and
  reviewed the new tests against the acceptance criteria. These runs are the final controlling
  test evidence; the corrective context changed no production code, API, Javadoc, explanatory
  documentation, glossary semantics, architecture, or task scope and did not rerun Javadoc.

## Implementation notes

- Added separate final channels-first `Conv1d`, [`Conv2d`](../../../../glossary.md#convolution--conv2d), and [`Conv3d`](../../../../glossary.md#conv3d--ncdhw) layers with deferred
  positive static input-channel binding, exact rank-specific state Shapes, atomic complete-state
  publication, retryable failure, strict complete-state load, and stateless factory recipes.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [binding glossary](../../../../glossary.md#term-binding); [rank glossary](../../../../glossary.md#term-rank); [Shapes glossary](../../../../glossary.md#shape); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [publication glossary](../../../../glossary.md#term-publication))</span>
- Kept convolution fan calculation and policy dispatch package-private. Random policies use one
  fresh <span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`L64X128MixRandom` per attempt; zero/one avoid generator creation; optional bias is typed
  zero and consumes no random draw.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- Preserved [Tensor](../../../../glossary.md#tensor) [provenance](../../../../glossary.md#provenance) and [ownership](../../../../glossary.md#term-ownership-meanings): `Conv1d` delegates to `Tensor.conv1d`, `Conv2d` and
  `Conv3d` delegate to their matching Tensor methods, and every layer constructs expressions
  without executing values. Conv3d backward-capable compilation remains fail-closed.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Documentation finalized all affected Javadocs, the layers package summary, the Training API
  explanation and example, glossary context, master plan, roadmap, and this completion evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No architecture contract or current-architecture-plan edit is needed because dependency
  direction, module ownership, and lifecycle boundaries are unchanged. No ADR or architecture
  test is needed because no architectural decision or dependency rule changed. No backend
  conformance or integration test is needed because the layers add NN expression composition,
  not backend execution; NN 0025A remains the Draft owner of Engine-facing layer fixtures.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [owner glossary](../../../../glossary.md#term-owner); [Engine glossary](../../../../glossary.md#term-engine))</span>
- No Gradle, dependency, or resource edit is needed because NN continues to depend only on Model.
  Model, Compiler, CPU, Engine, and Training implementation APIs remain unchanged: existing Tensor
  convolution operations are composed, and no execution or gradient capability is added.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [gradient glossary](../../../../glossary.md#term-gradient))</span>
  `ParameterInitialization` and `ParameterInitializers` remain unchanged because group-aware fans
  are a private layer concern. No other module needs modification for this bounded NN capability.

## Completion summary

- Completed changes: Implemented the three rank-specific channels-first convolution layers,
  their private initialization support, stateless factory recipes, focused tests, complete
  Javadocs, and user/planning documentation.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: Exactly the six production Java files, six focused test files, and
  five documentation/planning files listed by this task's bounded scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Tests and validation: The final controlling corrective evidence is a passing 70-test focused
  selection across the same nine suites and a passing 306-test/41-suite authoritative NN run,
  both with zero failures, errors, or skips. The earlier 56-test and 292-test runs remain recorded
  as implementation history. The documentation context passed its single NN Javadoc run,
  generated-page review, external example compilation/run, public-surface inspection, and final
  documentation/scope/hygiene gates; Javadoc was not repeated after test-only correction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Documentation-agent review: Complete in clean context
  `01a0b50a-a48a-7ed1-b48a-72b42b7b94d3`; no executable Java behavior changed and no stable Java
  suite was repeated during that documentation pass. This resumed evidence synchronization found
  no documentation semantic mismatch in corrective test-only context<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
  `01a0b51a-34a3-7732-a476-dfd0ca7977f9`.
- Documentation impact: `training-api.md` now explains construction versus execution, deferred
  binding, state Shapes, groups, bias, initialization, state loading, factory use, and current
  execution/gradient boundaries with a runnable example. Planning status and evidence are aligned.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Shapes glossary](../../../../glossary.md#shape); [bias glossary](../../../../glossary.md#term-bias); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Javadoc review: All affected public types, constructors, methods, the package summary, the
  package-private helper, and factory recipes document inputs, results, failures, state,
  ownership, retry, and threading where applicable.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: Existing convolution, standard-module-factory, and unary-module definitions
  now include the precise channels-first/deferred-state context; no implementation-only term was
  added.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [unary glossary](../../../../glossary.md#term-unary))</span>
- Unresolved issues: None.
- Follow-up required: NN 0025A remains the next concise Draft checkpoint and has no detailed task
  file.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

Status: Complete
