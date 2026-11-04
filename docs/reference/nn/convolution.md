# Channels-first [convolution](../../glossary.md#term-channels-first-convolution-layers) layers and examples

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

A convolution layer learns local filters over one, two or three [spatial](../../glossary.md#term-spatial-axes) [axes](../../glossary.md#term-axis). A **[channel](../../glossary.md#term-channel)** is
one [feature](../../glossary.md#term-feature) plane at each position; **channels-first** places that axis before the spatial axes.
The **[kernel](../../glossary.md#kernel)** is the filter's spatial [extent](../../glossary.md#term-extent), **[stride](../../glossary.md#term-stride)** is its movement between positions,
**[padding](../../glossary.md#padding)** extends the boundary, and **[dilation](../../glossary.md#term-dilation)** spaces filter [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters) apart. Groups divide
channels into separate connections rather than mixing every input channel with every output.

## Follow the layer-to-Engine boundary

The application supplies an input Tensor to `layer.forward(input)`; the layer supplies its current
weight and optional bias Tensors to ordinary Tensor operations. This builds dependencies, not
output numbers. Initializing a layer creates its parameter values; strict state loading instead
retains the exact supplied parameter references, with no copy. In the second example those
references point to application arrays. They must remain usable and unmodified during
`engine.compute(output)`, which evaluates the expression and returns a detached copy. See
[state binding rules](state.md#current-nn-state-dictionary-contract) before replacing weights and
[Engine one-shot execution](../engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
before choosing between repeated preparation and a fresh compile on every call.

<a id="current-nn-channels-first-convolution-contract"></a>
## Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) channels-first [convolution](../../glossary.md#term-channels-first-convolution-layers) contract

`Conv1d`, [`Conv2d`](../../glossary.md#convolution--conv2d), and [`Conv3d`](../../glossary.md#conv3d--ncdhw) are separate final `UnaryTensorModule` layers for [NCW](../../glossary.md#term-ncw), [NCHW](../../glossary.md#term-nchw),
and [NCDHW](../../glossary.md#conv3d--ncdhw) input respectively. Here `N` is [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), `C` is [channel](../../glossary.md#term-channel), and `W`, `H`, and `D` are width,
height, and depth. The layers record grouped cross-correlation: stored [kernels](../../glossary.md#kernel) are used in their
existing order rather than reversed as in mathematical convolution. Each constructor keeps output
channels, [rank](../../glossary.md#term-rank)-specific [kernel](../../glossary.md#kernel)/[stride](../../glossary.md#term-stride)/symmetric-[padding](../../glossary.md#padding)/[dilation](../../glossary.md#term-dilation) values, groups, [bias](../../glossary.md#term-bias) presence,
floating [data type](../../glossary.md#data-type--datatype), `ParameterInitialization`, and [seed](../../glossary.md#term-seed) explicit. There is no public `ConvNd`,
array-valued geometry, configuration object, or geometry getter.

The first compatible `forward(input)` or a complete strict state-dictionary load binds the only
deferred fact: the positive static input-channel [extent](../../glossary.md#term-extent) at [axis](../../glossary.md#term-axis) 1. [Parameter](../../glossary.md#term-parameter) [Shapes](../../glossary.md#shape) are:

| Layer | Input Shape | [Weight](../../glossary.md#term-weight) Shape | Optional bias | Result Shape |
|---|---|---|---|---|
| `Conv1d` | `[N,C_in,W]` | `[C_out,C_in/groups,K_w]` | `[C_out]` | `[N,C_out,W_out]` |
| `Conv2d` | `[N,C_in,H,W]` | `[C_out,C_in/groups,K_h,K_w]` | `[C_out]` | `[N,C_out,H_out,W_out]` |
| `Conv3d` | `[N,C_in,D,H,W]` | `[C_out,C_in/groups,K_d,K_h,K_w]` | `[C_out]` | `[N,C_out,D_out,H_out,W_out]` |

Both input and output channels must be divisible by the positive integer group count $G$.
Let $C_{\mathrm{in}}$ and $C_{\mathrm{out}}$ be positive channel counts, and let $K$ be the product
of the positive kernel extents (the number of spatial samples per channel). Fan-based
initialization counts connected inputs and outputs as

$$
\mathrm{fanIn}=\frac{C_{\mathrm{in}}}{G}K,\qquad
\mathrm{fanOut}=\frac{C_{\mathrm{out}}}{G}K.
$$

For the grouped example below, $C_{\mathrm{in}}=4$, $C_{\mathrm{out}}=6$, $G=2$ and
$K=3\cdot3=9$, giving fan-in $18$ and fan-out $27$. These exact integer counts select the
initializer's distribution; they are not output values. A random policy creates one fresh
deterministic `L64X128MixRandom` source from the layer seed per binding attempt. Zero/one
policies create none; optional bias is fresh typed zero and draws no random values.

Weight followed by optional bias is published as one complete layer-local group. Deterministic
input and Shape [validation](../../glossary.md#term-validation) precedes initializer work. If initialization, allocation, identifier
creation, or [publication](../../glossary.md#term-publication) fails before completion, no parameter wrapper becomes visible and a
later compatible forward may retry from the same seed. Compatible simultaneous first calls bind
once; this narrow publication guarantee does not make general [module](../../glossary.md#term-module) traversal, replacement,
loading, or forward construction [thread-safe](../../glossary.md#term-thread-safe). Before binding, parameter discovery and state
export [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) rather than expose partial state.

A complete strict dictionary can instead bind exact compatible weight and bias [Tensor](../../glossary.md#tensor) references.
The load validates the whole module tree before installation and performs no initialization,
random draw, Tensor copy, or new Tensor-identifier allocation. After binding, the usual
`Parameter.replace` contract accepts another exact-type, exact-Shape, [gradient](../../glossary.md#term-gradient)-eligible Tensor.
Batch and [spatial](../../glossary.md#term-spatial-axes) [Dimensions](../../glossary.md#dimension) may vary across later calls when their geometry remains valid, but
the exact data type and bound input-channel count cannot change.

<a id="example-construct-a-grouped-conv2d-expression"></a>
### Example: construct a grouped [Conv2d](../../glossary.md#convolution--conv2d) expression

The goal is to construct an [NCHW](../../glossary.md#term-nchw) [convolution](../../glossary.md#term-channels-first-convolution-layers) expression and inspect the state [schema](../../glossary.md#term-schema) established
by its first compatible input. The example uses two groups, four input channels, six output
channels, a `3 x 3` [kernel](../../glossary.md#kernel), unit [stride](../../glossary.md#term-stride) and [dilation](../../glossary.md#term-dilation), one position of symmetric [padding](../../glossary.md#padding), and [bias](../../glossary.md#term-bias).

```java
// The first compatible forward binds channel-dependent parameters, then describes output work.
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Conv2d;
import io.github.pho001.synaptik.nn.module.ModuleFactory;
import java.util.Optional;

ModuleFactory modules = ModuleFactory.standard();
Conv2d convolution = modules.conv2d(
        6,                 // output channels
        3, 3,              // kernel height and width
        1, 1,              // stride height and width
        1, 1,              // symmetric padding per side
        1, 1,              // dilation height and width
        2,                 // groups
        true,              // bias
        DataType.FLOAT32,
        ParameterInitialization.glorotUniform(),
        41L);

Tensor input = TensorFactory.zeros(
        Shape.of(2, 4, 8, 8), DataType.FLOAT32, Optional.empty(), false);
Tensor output = convolution.forward(input);

Shape weightShape = convolution.weight().value().descriptor().shape();
Shape outputShape = output.descriptor().shape();
```

The input establishes `C_in = 4`, so each of the two groups receives two input channels.
`weightShape` is `[6,2,3,3]`, the optional bias is `[6]`, and `outputShape` is `[2,6,8,8]`.
The call has created [parameter](../../glossary.md#term-parameter) [leaves](../../glossary.md#term-leaf) and a fresh [Tensor](../../glossary.md#tensor) expression; it has not calculated the
output values. `Conv1d` visibly delegates through input/[weight](../../glossary.md#term-weight) expansion, ordinary Conv2d, and
squeeze, while `Conv2d` and [`Conv3d`](../../glossary.md#conv3d--ncdhw) each delegate to their matching Tensor method.

<a id="example-strict-load-and-execute-a-conv2d-layer"></a>
### Example: strict-load and execute a [Conv2d](../../glossary.md#convolution--conv2d) layer

The goal is to bind caller-owned [host](../../glossary.md#term-host) values as the complete state of a factory-created layer,
construct its forward expression, and execute that expression through the standard [Engine](../../glossary.md#term-engine). This
small helper creates a resolved [contiguous](../../glossary.md#term-contiguous) [`FLOAT32`](../../glossary.md#term-float32) [leaf](../../glossary.md#term-leaf) backed by the caller's array:

```java
// Strict load installs exact supplied references; Engine reads their arrays during the later compute.
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Conv2d;
import io.github.pho001.synaptik.nn.module.ModuleFactory;
import io.github.pho001.synaptik.nn.module.StateDictionary;
import io.github.pho001.synaptik.nn.module.StateEntry;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;

static Tensor floatTensor(Shape shape, boolean requiresGrad, float... values) {
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.contiguous(shape)),
            requiresGrad);
    var storage = new MemorySegmentStorage(
            DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
    return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
}

Conv2d layer = ModuleFactory.standard().conv2d(
        1,                  // output channels
        2, 2,               // kernel height and width
        1, 1,               // stride height and width
        0, 0,               // symmetric padding per side
        1, 1,               // dilation height and width
        1,                  // groups
        true,               // bias
        DataType.FLOAT32,
        ParameterInitialization.zeros(),
        41L);

Tensor input = floatTensor(
        Shape.of(1, 1, 3, 3), false,
        1, 2, 3,
        4, 5, 6,
        7, 8, 9);
Tensor weight = floatTensor(Shape.of(1, 1, 2, 2), true, 1, 0, 0, -1);
Tensor bias = floatTensor(Shape.of(1), true, 0.5f);

// Install supplied Tensors without running the zero initializer or copying their arrays.
layer.loadStateDictionary(new StateDictionary(List.of(
        new StateEntry("weight", StateKind.PARAMETER, weight),
        new StateEntry("bias", StateKind.PARAMETER, bias))));

Tensor output = layer.forward(input);
HostTensorValue value;
try (Engine engine = Engine.standard()) {
    // Engine discovers stored leaves and returns detached bytes before its temporary owners close.
    value = engine.compute(output);
}

float[] actual = new float[Math.toIntExact(value.elementCount())];
value.bytes().asFloatBuffer().get(actual);
```

The dictionary is complete and strict: it binds the exact [`weight`](../../glossary.md#term-weight) and [`bias`](../../glossary.md#term-bias) [Tensor](../../glossary.md#tensor) references
without running the configured zero initializer. Those two leaves and `input` retain their
caller-owned [host storage](../../glossary.md#host-storage). `forward` creates the storage-free `[1,1,2,2]` expression; it does not
execute it or load state automatically. `Engine.compute` discovers the reachable leaves, then
compiles, prepares, runs, materializes, and cleans up its temporary result. The [detached](../../glossary.md#detached-model-results)
[`HostTensorValue`](../../glossary.md#host-tensor-value--hosttensorvalue) owns no closeable resource, so `actual` can be read after the Engine closes; its
four values are all `-3.5f`. For the top-left window, the real-number calculation is
$1\cdot1+2\cdot0+4\cdot0+5\cdot(-1)+0.5=-3.5$: the four window values multiply the
four stored filter values in order, then add the bias. Each shifted window has the same difference;
these particular values are exactly representable, while general floating convolutions round. Caller storage is never transferred to the Engine.

For repeated execution, selective [publication](../../glossary.md#term-publication) copies, or explicit input ordering, use the
[reusable compile, session, run, and materialize workflow](../runtime/engine-boundary.md#current-ordinary-engine-boundary).

These [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layers construct forward expressions; they do not [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), select a [backend](../../glossary.md#backend)
or execute values. [Compiler](../../glossary.md#term-compiler) derives selected input, weight and bias [gradients](../../glossary.md#term-gradient) through the shared
grouped Conv2d/[Conv3d](../../glossary.md#conv3d--ncdhw) rule and the visible [Conv1d composition](../../glossary.md#conv1d-composition). Static [CPU](../../glossary.md#term-cpu) FLOAT32/[FLOAT64](../../glossary.md#term-float64)
and [Metal](../../glossary.md#term-metal) FLOAT32 Conv3d training, including overlap and save/load/new training for the existing
default biased cold source, are qualified in the
[production support matrix](../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary).
CPU low arithmetic and [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) use the existing shared [derivative](../../glossary.md#term-derivative) [owners](../../glossary.md#term-owner) under their geometry guards;
gradient-eligible [parameter](../../glossary.md#term-parameter) flags alone never prove a backend can execute a complete [graph](../../glossary.md#graph).
`ModuleFactory.standard()` supplies matching stateless `conv1d`, `conv2d`, and `conv3d` recipes,
but each direct or factory-created layer has the same state, [validation](../../glossary.md#term-validation), and forwarding contract.


---

[Previous topic](composition.md) · [Next topic](declarative.md)
