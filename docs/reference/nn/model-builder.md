# Fluent [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) builder

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

A **fluent builder** collects choices through chained method calls before creating the [Model](../../glossary.md#term-model).
The **input [signature](../../glossary.md#term-signature)** specifies the required [element](../../glossary.md#term-element) type, [shape](../../glossary.md#shape), [layout](../../glossary.md#layout) and [gradient](../../glossary.md#term-gradient) flag;
it is not an example [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)'s contents. Building acquires private state and constructs expressions,
while compute and configured training later use those expressions through [Engine](../../glossary.md#term-engine).

## What does the application hand to the builder?

`input(...)` supplies a description of future batches, while `linear(weight, bias)` supplies
current stored parameter values. The builder is the callee of these application calls; at build,
it copies supported supplied state into the returned Model instead of borrowing the application's
weight storage for future execution. A leaf here means a Tensor with no producing expression,
not a promise that an arbitrary Tensor contains readable values. Automatic recipes create fresh
state instead. The examples below return Models, so their callers must eventually close them;
the builder itself does not execute a batch. Compare [ordinary layer composition](composition.md)
when exact supplied references, rather than private acquired copies, are needed.

<a id="fluent-model-builder"></a>
### [Fluent Model builder](../../glossary.md#fluent-model-builder)

`Model.builder()` returns a fresh public final `ModelBuilder`, with no public constructor or
[callback](../../glossary.md#term-callback)/[module](../../glossary.md#term-module) extension point. Set `input(DataType,long...)` once before steps. Configuration
returns the same caller-coordinated builder; `build()` returns a new closeable `Model<Tensor,Tensor>`
owning a frozen Sequential under `layers`. Every build owns fresh automatic state and privately
copies supplied leaves once by [identity](../../glossary.md#term-identity-meanings) before expressions, retaining [aliases](../../glossary.md#aliasing-and-arena) but no caller storage
or recipes. Build requires a floating positive-static [rank](../../glossary.md#term-rank)-at-least-one input and a nonempty chain.
The input [prototype](../../glossary.md#term-prototype) is storage-free, [canonical](../../glossary.md#term-canonical-form) [contiguous](../../glossary.md#term-contiguous), offset-zero, non-[view](../../glossary.md#term-view-meanings), requiresGrad=false;
compute/cold forward match its complete [descriptor](../../glossary.md#term-descriptor), with no implicit [dtype](../../glossary.md#data-type--datatype) or [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[shape](../../glossary.md#shape) conversion.

Supported steps are `linear(outFeatures[,bias])`, `linear(weight[,bias])`, `relu/tanh/sigmoid`,
`conv1d/conv2d/conv3d`, `maxPool2d`, `flatten`, static `rnn/gru/lstm`, `lastTimeStep`,
[runtime](../../glossary.md#term-runtime)-length `rnnScan/gruScan/lstmScan(long hiddenSize)` and `finalHidden()`.
Automatic layers infer input widths, use [Glorot](../../glossary.md#term-glorot-uniform) uniform, and enable exact zero [bias](../../glossary.md#term-bias) by default.
Default [seed](../../glossary.md#term-seed) is 0; the last `seed(long)` applies to the whole build. Appended step i, including
stateless steps, derives `seed + 0x9E3779B97F4A7C15L * i` with Java long [wraparound](../../glossary.md#term-wraparound) and a fresh
L64X128MixRandom generator per initialized layer. Supplied [Linear](../../glossary.md#term-linear-meanings) [leaves](../../glossary.md#term-leaf) must be stored,
[provenance](../../glossary.md#provenance)-free, floating, [gradient](../../glossary.md#term-gradient)-eligible and match input dtype and existing Linear shape guards.
Build reads but never mutates caller state; unavailable or produced supplied state rejects.

Convolutions use [NCW](../../glossary.md#term-ncw)/[NCHW](../../glossary.md#term-nchw)/[NCDHW](../../glossary.md#conv3d--ncdhw), [stride](../../glossary.md#term-stride)/[dilation](../../glossary.md#term-dilation)/groups 1, [padding](../../glossary.md#padding) 0 and bias true. Pool stride
equals [kernel](../../glossary.md#kernel), padding 0, dilation 1, floor geometry. Flatten preserves [axis](../../glossary.md#term-axis) zero and checks the
remaining product: [CNN](../../glossary.md#term-convolutional-neural-network-cnn) `[B,C,H,W]` becomes `[B,C*H*W]`, [TBF](../../glossary.md#term-tbf) `[T,B,H]` becomes `[T,B*H]`.
Existing static `rnn/gru/lstm` steps use fixed all-valid TBF, start from zero h (and c for [LSTM](../../glossary.md#term-long-short-term-memory-lstm)),
and return every hidden time step. Static stacks preserve TBF. Explicit `lastTimeStep()` selects
`[B,H]`; use it before flatten when batch rows must remain separate. Cached runs [carry](../../glossary.md#term-carry) no previous
final state.

This compilable example constructs an [MLP](../../glossary.md#term-multilayer-perceptron-mlp), CNN and recurrent stack; inputs/results are noted inline:

```java
// Each build acquires a new private Model; the caller of these factory methods must close it.
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.Model;

public final class FluentBuilderShapesExample {
    public static Model<Tensor, Tensor> mlp() {
        return Model.builder().input(DataType.FLOAT32, 32, 784).seed(42)
                .linear(128).relu().linear(10, false).build(); // [32,10]
    }
    public static Model<Tensor, Tensor> cnn() {
        return Model.builder().input(DataType.FLOAT32, 32, 1, 28, 28).seed(42)
                .conv2d(32, 3, 3).relu().maxPool2d(2, 2)
                .conv2d(64, 3, 3).relu().maxPool2d(2, 2)
                .flatten().linear(128).relu().linear(10).build(); // [32,10]
    }
    public static Model<Tensor, Tensor> recurrent() {
        return Model.builder().input(DataType.FLOAT32, 20, 32, 8).seed(42)
                .lstm(128).lstm(64).lastTimeStep().linear(10).build(); // [32,10]
    }
}
```

The application closes each returned Model. In the CNN, each unpadded unit-stride $3\times3$
convolution reduces both spatial extents by two, while each $2\times2$ pool uses stride two
and floor geometry. Thus each axis follows $28\to26\to13\to11\to5$ positions. Flatten combines
$C=64$ channels, height $H=5$ and width $W=5$ into $CHW=64\cdot5\cdot5=1600$ features per
batch row. These are exact non-negative integer shape counts, not rounded tensor arithmetic.
The LSTMs instead produce `[20,32,128]`, then `[20,32,64]` before explicit last-time selection.
See the complete [compute/train/save example](../../user-guide/running-models.md#build-a-fluent-model).

Construction and cold forward create no [Engine](../../glossary.md#term-engine). Compute/configured training consume fixed outputs
and cached [bindings](../../glossary.md#term-binding), never recipes per batch. They are independent of [serialization](../../glossary.md#term-serialization) [admission](../../glossary.md#term-admission).
[Graph](../../glossary.md#graph)-model v3 enables saves for the selected default CNN/static recurrent DAGs;
other unsupported saves still fail before file work without freezing Engine selection or
invalidating execution. Only sources wholly within the original five/eight-kind slices retain
byte-identical graph-model.v1/v2 saves; any selected added body-free kind chooses v3, including CNN+RELU.
See the [single owning wire contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3).
Execution still requires actual Engine capability;
fixed scans retain their separately qualified first-order training support; runtime-length recipes are described below, and no recipe implies streaming state. [Conv3d](../../glossary.md#conv3d--ncdhw) separately uses
the shared grouped [convolution](../../glossary.md#term-channels-first-convolution-layers) adjoints within its qualified [backend](../../glossary.md#backend) domain.

#### Build a separate values-and-lengths boundary

For runtime lengths, append `rnnScan`, `gruScan` or `lstmScan` with a positive hidden width and end
with `buildWithLengths()`, returning `Model<List<Tensor>,List<Tensor>>`. The two inputs are
`[values, validLengths]`, with one chain result. Each recipe uses a biased FORWARD scan, existing
initialization/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), independent [parameters](../../glossary.md#term-parameter) and zero hidden state (also zero cell for LSTM).
There is no hidden carry across batches. The first scan requires positive static TBF `[T,B,F]`
and establishes shared canonical, non-gradient `INT64[B]` lengths; later scans preserve exact T/B
and infer F. An ordinary prefix establishes this interpretation, not necessarily the original input
axes. Runtime length values must lie in `[0,T]`, including zero. Advanced graph/list workflows retain
explicit states, finalCell, reverse/bias-free variants and zero T/B; the builder remains positive-only.

`finalHidden()` must immediately follow a new scan recipe, with no intervening transform, static
recurrence or repeated selection. It selects the existing `[B,H]` port after each row's actual last
transition; length zero returns initial hidden state (zero here), not `select(length-1)`. It counts
as an appended seed step. Append an ordinary classifier head afterwards.
`build()` rejects scans and `buildWithLengths()` rejects no-scan chains before acquisition effects;
both [snapshot](../../glossary.md#term-snapshot) recipes and create fresh private state.

[FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) `rnnScan` uses the general composition above, supporting qualified first-order
training and v5 [persistence](../../glossary.md#term-persistence) with canonical source outputs. Retained low [RNN](../../glossary.md#term-recurrent-neural-network-rnn), [GRU](../../glossary.md#term-gated-recurrent-unit-gru)/LSTM and direct
fixed expressions remain fixed recipes with their own working/[publication](../../glossary.md#term-publication) semantics. Qualified
first-order configured training now uses their existing [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd) without rewriting them as
[general Scan](../../glossary.md#general-differentiable-scan). Old v3 fixed files remain fixed after load; fresh training does not restore [optimizer
history](../../glossary.md#term-optimizer-history). Low execution still requires a supporting backend and explicit FLOAT32/FLOAT64 [objective](../../glossary.md#term-objective).


#### Do not infer a global padding mask

There is no model-wide [mask](../../glossary.md#term-boolean-mask): bias and [sigmoid](../../glossary.md#term-sigmoid) may change padded zeros; convolution/[pooling](../../glossary.md#term-pooling) keep
dense channels-first semantics, flatten preserves axis zero, static recurrence ignores lengths, and
`lastTimeStep()` still selects `T-1`, not final hidden. Each scan ignores its own suffix, not arbitrary
ordinary results. Shape-valid construction or save does not guarantee backend execution. General
[F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) RNN uses its narrower dense zero-offset [CPU](../../glossary.md#term-cpu) read [qualification](../../glossary.md#term-qualification); fixed recipes retain
their separately declared read domain. No descriptor is erased and no [layout](../../glossary.md#layout) copy is inserted.
See the [runtime-length inference example](../../user-guide/running-models.md#run-a-runtime-length-scan-model)
and [training journey](../../user-guide/running-models.md#train-a-runtime-length-rnn).


---

[Previous topic](recurrent.md) · [Next topic](../engine/advanced.md)
