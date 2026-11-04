# Fluent owned Model builder

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

A **fluent builder** collects choices through chained method calls before creating the Model.
The **input signature** specifies the required element type, shape, layout and gradient flag;
it is not an example batch's contents. Building acquires private state and constructs expressions,
while compute and configured training later use those expressions through Engine.

<a id="fluent-model-builder"></a>
### Fluent Model builder

`Model.builder()` returns a fresh public final `ModelBuilder`, with no public constructor or
callback/module extension point. Set `input(DataType,long...)` once before steps. Configuration
returns the same caller-coordinated builder; `build()` returns a new closeable `Model<Tensor,Tensor>`
owning a frozen Sequential under `layers`. Every build owns fresh automatic state and privately
copies supplied leaves once by identity before expressions, retaining aliases but no caller storage
or recipes. Build requires a floating positive-static rank-at-least-one input and a nonempty chain.
The input prototype is storage-free, canonical contiguous, offset-zero, non-view, requiresGrad=false;
compute/cold forward match its complete descriptor, with no implicit dtype or batch-shape conversion.

Supported steps are `linear(outFeatures[,bias])`, `linear(weight[,bias])`, `relu/tanh/sigmoid`,
`conv1d/conv2d/conv3d`, `maxPool2d`, `flatten`, static `rnn/gru/lstm`, `lastTimeStep`,
runtime-length `rnnScan/gruScan/lstmScan(long hiddenSize)` and `finalHidden()`.
Automatic layers infer input widths, use Glorot uniform, and enable exact zero bias by default.
Default seed is 0; the last `seed(long)` applies to the whole build. Appended step i, including
stateless steps, derives `seed + 0x9E3779B97F4A7C15L * i` with Java long wraparound and a fresh
L64X128MixRandom generator per initialized layer. Supplied Linear leaves must be stored,
provenance-free, floating, gradient-eligible and match input dtype and existing Linear shape guards.
Build reads but never mutates caller state; unavailable or produced supplied state rejects.

Convolutions use NCW/NCHW/NCDHW, stride/dilation/groups 1, padding 0 and bias true. Pool stride
equals kernel, padding 0, dilation 1, floor geometry. Flatten preserves axis zero and checks the
remaining product: CNN `[B,C,H,W]` becomes `[B,C*H*W]`, TBF `[T,B,H]` becomes `[T,B*H]`.
Existing static `rnn/gru/lstm` steps use fixed all-valid TBF, start from zero h (and c for LSTM),
and return every hidden time step. Static stacks preserve TBF. Explicit `lastTimeStep()` selects
`[B,H]`; use it before flatten when batch rows must remain separate. Cached runs carry no previous
final state.

This compilable example constructs an MLP, CNN and recurrent stack; inputs/results are noted inline:

```java
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

Callers close each returned Model. CNN spatial extents are 28→26→13→11→5, so flatten width is
`64*5*5=1600`; the LSTMs produce `[20,32,128]`, then `[20,32,64]` before explicit selection.
See the complete [compute/train/save example](../../user-guide/running-models.md#build-a-fluent-model).

Construction and cold forward create no Engine. Compute/configured training consume fixed outputs
and cached bindings, never recipes per batch. They are independent of serialization admission.
Graph-model v3 enables saves for the selected default CNN/static recurrent DAGs;
other unsupported saves still fail before file work without freezing Engine selection or
invalidating execution. Only sources wholly within the original five/eight-kind slices retain
byte-identical graph-model.v1/v2 saves; any selected added body-free kind chooses v3, including CNN+RELU.
See the [single owning wire contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3).
Execution still requires actual Engine capability;
fixed scans retain their separately qualified first-order training support; runtime-length recipes are described below, and no recipe implies streaming state. Conv3d separately uses
the shared grouped convolution adjoints within its qualified backend domain.

For runtime lengths, append `rnnScan`, `gruScan` or `lstmScan` with a positive hidden width and end
with `buildWithLengths()`, returning `Model<List<Tensor>,List<Tensor>>`. The two inputs are
`[values, validLengths]`, with one chain result. Each recipe uses a biased FORWARD scan, existing
initialization/seeds, independent parameters and zero hidden state (also zero cell for LSTM).
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
both snapshot recipes and create fresh private state.

FLOAT32/FLOAT64 `rnnScan` uses the general composition above, supporting qualified first-order
training and v5 persistence with canonical source outputs. Retained low RNN, GRU/LSTM and direct
fixed expressions remain fixed recipes with their own working/publication semantics. Qualified
first-order configured training now uses their existing Compiler AD without rewriting them as
general Scan. Old v3 fixed files remain fixed after load; fresh training does not restore optimizer
history. Low execution still requires a supporting backend and explicit FLOAT32/FLOAT64 objective.


There is no model-wide mask: bias and sigmoid may change padded zeros; convolution/pooling keep
dense channels-first semantics, flatten preserves axis zero, static recurrence ignores lengths, and
`lastTimeStep()` still selects `T-1`, not final hidden. Each scan ignores its own suffix, not arbitrary
ordinary results. Shape-valid construction or save does not guarantee backend execution. General
F32/F64 RNN uses its narrower dense zero-offset CPU read qualification; fixed recipes retain
their separately declared read domain. No descriptor is erased and no layout copy is inserted.
See the [runtime-length inference example](../../user-guide/running-models.md#run-a-runtime-length-scan-model)
and [training journey](../../user-guide/running-models.md#train-a-runtime-length-rnn).


---

[Previous topic](recurrent.md) · [Next topic](../engine/advanced.md)
