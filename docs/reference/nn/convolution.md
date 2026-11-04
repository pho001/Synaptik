# Channels-first convolution layers and examples

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

<a id="current-nn-channels-first-convolution-contract"></a>
## Current NN channels-first convolution contract

`Conv1d`, `Conv2d`, and `Conv3d` are separate final `UnaryTensorModule` layers for NCW, NCHW,
and NCDHW input respectively. Here `N` is batch, `C` is channel, and `W`, `H`, and `D` are width,
height, and depth. The layers record grouped cross-correlation: stored kernels are used in their
existing order rather than reversed as in mathematical convolution. Each constructor keeps output
channels, rank-specific kernel/stride/symmetric-padding/dilation values, groups, bias presence,
floating data type, `ParameterInitialization`, and seed explicit. There is no public `ConvNd`,
array-valued geometry, configuration object, or geometry getter.

The first compatible `forward(input)` or a complete strict state-dictionary load binds the only
deferred fact: the positive static input-channel extent at axis 1. Parameter Shapes are:

| Layer | Input Shape | Weight Shape | Optional bias | Result Shape |
|---|---|---|---|---|
| `Conv1d` | `[N,C_in,W]` | `[C_out,C_in/groups,K_w]` | `[C_out]` | `[N,C_out,W_out]` |
| `Conv2d` | `[N,C_in,H,W]` | `[C_out,C_in/groups,K_h,K_w]` | `[C_out]` | `[N,C_out,H_out,W_out]` |
| `Conv3d` | `[N,C_in,D,H,W]` | `[C_out,C_in/groups,K_d,K_h,K_w]` | `[C_out]` | `[N,C_out,D_out,H_out,W_out]` |

Both input and output channels must be divisible by `groups`. Fan-based initialization uses the
actual grouped connectivity: `fanIn = (C_in/groups) * kernelVolume` and
`fanOut = (C_out/groups) * kernelVolume`. A random policy creates one fresh deterministic
`L64X128MixRandom` source from the layer seed for each binding attempt. Zero and one policies
create no source. Optional bias is always a fresh typed-zero parameter and consumes no random
draw, regardless of the weight policy.

Weight followed by optional bias is published as one complete layer-local group. Deterministic
input and Shape validation precedes initializer work. If initialization, allocation, identifier
creation, or publication fails before completion, no parameter wrapper becomes visible and a
later compatible forward may retry from the same seed. Compatible simultaneous first calls bind
once; this narrow publication guarantee does not make general module traversal, replacement,
loading, or forward construction thread-safe. Before binding, parameter discovery and state
export fail closed rather than expose partial state.

A complete strict dictionary can instead bind exact compatible weight and bias Tensor references.
The load validates the whole module tree before installation and performs no initialization,
random draw, Tensor copy, or new Tensor-identifier allocation. After binding, the usual
`Parameter.replace` contract accepts another exact-type, exact-Shape, gradient-eligible Tensor.
Batch and spatial Dimensions may vary across later calls when their geometry remains valid, but
the exact data type and bound input-channel count cannot change.

<a id="example-construct-a-grouped-conv2d-expression"></a>
### Example: construct a grouped Conv2d expression

The goal is to construct an NCHW convolution expression and inspect the state schema established
by its first compatible input. The example uses two groups, four input channels, six output
channels, a `3 x 3` kernel, unit stride and dilation, one position of symmetric padding, and bias.

```java
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
The call has created parameter leaves and a fresh Tensor expression; it has not calculated the
output values. `Conv1d` visibly delegates through input/weight expansion, ordinary Conv2d, and
squeeze, while `Conv2d` and `Conv3d` each delegate to their matching Tensor method.

<a id="example-strict-load-and-execute-a-conv2d-layer"></a>
### Example: strict-load and execute a Conv2d layer

The goal is to bind caller-owned host values as the complete state of a factory-created layer,
construct its forward expression, and execute that expression through the standard Engine. This
small helper creates a resolved contiguous `FLOAT32` leaf backed by the caller's array:

```java
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

layer.loadStateDictionary(new StateDictionary(List.of(
        new StateEntry("weight", StateKind.PARAMETER, weight),
        new StateEntry("bias", StateKind.PARAMETER, bias))));

Tensor output = layer.forward(input);
HostTensorValue value;
try (Engine engine = Engine.standard()) {
    value = engine.compute(output);
}

float[] actual = new float[Math.toIntExact(value.elementCount())];
value.bytes().asFloatBuffer().get(actual);
```

The dictionary is complete and strict: it binds the exact `weight` and `bias` Tensor references
without running the configured zero initializer. Those two leaves and `input` retain their
caller-owned host storage. `forward` creates the storage-free `[1,1,2,2]` expression; it does not
execute it or load state automatically. `Engine.compute` discovers the reachable leaves, then
compiles, prepares, runs, materializes, and cleans up its temporary result. The detached
`HostTensorValue` owns no closeable resource, so `actual` can be read after the Engine closes; its
four values are all `-3.5f`. Caller storage is never transferred to the Engine.

For repeated execution, selective publication copies, or explicit input ordering, use the
[reusable compile, session, run, and materialize workflow](../runtime/engine-boundary.md#current-ordinary-engine-boundary).

These NN layers construct forward expressions; they do not compile, prepare, select a backend
or execute values. Compiler derives selected input, weight and bias gradients through the shared
grouped Conv2d/Conv3d rule and the visible Conv1d composition. Static CPU FLOAT32/FLOAT64
and Metal FLOAT32 Conv3d training, including overlap and save/load/new training for the existing
default biased cold source, are qualified in the
[production support matrix](../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary).
CPU low arithmetic and Pool3d use the existing shared derivative owners under their geometry guards;
gradient-eligible parameter flags alone never prove a backend can execute a complete graph.
`ModuleFactory.standard()` supplies matching stateless `conv1d`, `conv2d`, and `conv3d` recipes,
but each direct or factory-created layer has the same state, validation, and forwarding contract.


---

[Previous topic](composition.md) · [Next topic](declarative.md)
