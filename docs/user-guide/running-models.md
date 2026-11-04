<a id="run-a-reusable-inference-session"></a>

# Run a prepared graph

## Outcome

This guide compiles one fixed inference graph, prepares one closeable `PreparedExecution`, and runs the same
prepared execution repeatedly with current caller-owned input storage. Each run returns typed,
ordered publication metadata and supports explicit copying into detached immutable
`HostTensorValue` objects.

## JVM prerequisites

Use JDK 26 and the runtime dependencies of the selected API. The default scalar CPU route does
not require `jdk.incubator.vector`. Explicit Vector API CPU computation requires
`--add-modules jdk.incubator.vector`.

Enable FFM native access with `--enable-native-access=ALL-UNNAMED` for `Engine.standard()` as well
as explicit Metal/OpenBLAS use. The standard CPU composition performs bounded optional OpenBLAS
library discovery when opened, even when the selected numerical work is portable Java. JDK 26
otherwise reports restricted-access warnings from that probe. The flag grants the required FFM
access; it does not choose a backend, require an installed BLAS library or change the prepared route.

## Build a fluent Model

Use `Model.builder()` when fixed-shape inputs should flow through dense, convolutional or
recurrent layers without a definition callback. Static `rnn/gru/lstm` and explicit runtime-length
`rnnScan/gruScan/lstmScan` are separate recipes. A build creates a private evaluation-mode Model
with one working weight set for compute and supported saves; sources without fixed scans may also train.
It creates no Engine. Use Java 26 and `:extensions:nn`. Close the Model when finished; compute
results are detached values.

For example, a multilayer perceptron (MLP) maps FLOAT32 `[32,784]` to `[32,10]`:

```java
try (var model = Model.builder().input(DataType.FLOAT32, 32, 784)
        .seed(42).linear(128).relu().linear(10, false).build()) {
    // compute requires stored FLOAT32 [32,784] batches with the exact input descriptor.
}
```

Declare input once before steps: floating dtype, at least one axis, every extent positive and
static. Its prototype is storage-free, canonical contiguous, offset-zero, non-view and
`requiresGrad=false`. Both compute and cold `forward` require that full descriptor. The builder
is caller-coordinated and reusable: every build owns fresh children/state, and later builder edits
do not alter earlier Models. Build requires at least one step.

Automatic layers infer input feature/channel widths. They use existing Glorot-uniform weights,
zero bias enabled by default, and a fresh L64X128MixRandom generator per layer. Default seed is 0;
the last seed set before build applies to the whole recipe. Appended step `i`, starting at zero,
uses `seed + 0x9E3779B97F4A7C15L * i` with Java long wraparound. Activations and transforms count
as steps. Identical recipes, seeds and algorithms reproduce initialization bytes; this is not
optimizer or random-stream checkpoint continuation.

### Train and save private supplied weights

This complete example starts with supplied weight 2, predicts 6 on input 3, and trains toward 1.
`linear(weight)` disables bias; `linear(weight,bias)` copies both. Supplied state must satisfy
Linear's floating, positive-static, gradient-eligible schema and match input dtype. It must be
stored provenance-free state, not an expression to evaluate. Build copies each Tensor identity
once before constructing expressions, preserving aliases; it never mutates caller state. Training
retains its separate rejection of parameter aliases. Keep supplied storage live and exclude mutation
during build; a successful Model does not retain that storage.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import java.nio.file.Path;
import java.util.Optional;

public final class FluentModelExample {
    static Tensor matrix(float value, boolean gradient) {
        Shape shape = Shape.of(1, 1);
        return TensorFactory.fromFlatArray(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient), Optional.empty(),
                new float[]{value});
    }

    public static void main(String[] args) throws Exception {
        Path file = Path.of(args.length == 0 ? "fluent.graph" : args[0]);
        Tensor externalWeight = matrix(2, true);
        try (var model = Model.builder().input(DataType.FLOAT32, 1, 1)
                .linear(externalWeight).build()) {
            model.save(file); // supported graph-model.v1, before any Engine exists
            System.out.println(model.compute(matrix(3, false)).bytes().getFloat()); // 6
            model.configureTraining(matrix(0, false).descriptor(), (prediction, target) -> {
                Tensor error = prediction.sub(target);
                return error.mul(error).sum();
            }, new Sgd(0.1));
            System.out.println(model.train(matrix(3, false), matrix(1, false))
                    .objective().bytes().getFloat()); // 25
            model.save(file); // private weight -1; externalWeight remains 2
        }
        try (var restored = Model.load(file)) {
            System.out.println(restored.compute(matrix(2, false)).bytes().getFloat()); // -2
        }
    }
}
```

Run with the NN runtime classpath and the JVM prerequisites above. The destination's parent
must exist. Output is `6.0`, `25.0`, `-2.0`: squared error `(2*3-1)^2` has weight gradient 30,
so SGD changes only the private weight to `2-0.1*30=-1`. Configure training again after load;
the file contains neither the objective nor optimizer history.

### Convolution and recurrent axes

These chains use the same imports and Model lifecycle. A convolutional neural network (CNN) takes
channels-first input; convolution defaults are stride/dilation/groups 1, padding 0 and bias enabled.
Maximum pooling uses stride equal to kernel, padding 0, dilation 1 and floor geometry:

```java
try (var cnn = Model.builder().input(DataType.FLOAT32, 32, 1, 28, 28).seed(42)
        .conv2d(32, 3, 3).relu().maxPool2d(2, 2)
        .conv2d(64, 3, 3).relu().maxPool2d(2, 2)
        .flatten().linear(128).relu().linear(10).build()) {
    // Spatial extents: 28 -> 26 -> 13 -> 11 -> 5.
    // flatten: [32,64,5,5] -> [32,1600]; final output: [32,10].
}
try (var recurrent = Model.builder().input(DataType.FLOAT32, 20, 32, 8).seed(42)
        .lstm(128).lstm(64).lastTimeStep().linear(10).build()) {
    // Each LSTM returns all hidden time steps: [20,32,128], then [20,32,64].
    // lastTimeStep explicitly selects [32,64]; final output: [32,10].
}
```

`conv1d` uses NCW (batch/channels/width), `conv2d` NCHW and `conv3d` NCDHW.
Static `rnn/gru/lstm` steps use TBF (time/batch/features), positive fixed time, and all-valid rows.
They return **every hidden step**, including LSTM; there is no implicit final-step selection.
Each composition starts with zero hidden state (and zero cell state for LSTM). Cached compute/train
reuse fixed zero constants, never carry state across batches, and do not rebuild the graph per call.
Static unrolling uses existing constituent-operation adjoints; it is not scan backpropagation through
time (BPTT), streaming state, or runtime-length support. Conv3d uses shared grouped convolution
adjoints in the [qualified static CPU F32/F64 and Metal F32 domains](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary).

`flatten()` requires rank at least two and preserves axis 0: `[axis0, product of other extents]`.
For CNNs axis 0 is batch; for TBF `[T,B,H]` it is time, so flatten gives `[T,B*H]`.
Use `lastTimeStep()` first when recurrent batch rows must remain separate.

Default Conv1d/2d/3d, MaxPool2d/flatten and static RNN/GRU/LSTM compositions save as their ordinary
Tensor graphs, not layer recipes. Save before compute, close the original Model, cold-load through
`Model.load`, then compute fresh batches with the exact original descriptor. For example, the CNN
above retains `[32,1,28,28]` input and `[32,10]` output, while the stacked LSTM retains `[20,32,8]`
input and `[32,10]` output.
Its zero initial-state and row-index constants are saved as reachable CONSTANT leaves, not layer
recipes. Configure a fresh objective/optimizer after load to train Conv1d/2d/3d or static recurrence
where current adjoints and backend predicates permit. Default biased Conv3d save/load/new-train
is qualified; arbitrary convolution checkpoint geometry is not. Saving trained values still does not
resume optimizer history, RNG or data progress. Any selected added operation chooses v3 even if
the graph also contains RELU/TANH/SIGMOID.

Sources containing only ADD/MUL/NEG/MATMUL, rank-two PERMUTE `[1,0]` and floating
RELU/TANH/SIGMOID retain byte-identical v1/v2 saves. Construction, cold forward and execution
remain independent of serialization admission. Other source variants still reject: convolution
without bias or with nondefault geometry, unsupported indexing axes and arbitrary recurrent bodies
are outside the selected slice. Source
rejection occurs before file work, leaving absent targets absent and existing targets unchanged.
A failed unsupported save does not freeze Engine selection, invalidate the Model, or disable
compute/training. Persistence does not make
every builder occurrence executable: the selected Engine's dtype/layout/gradient capabilities apply.
See [builder API and lifecycle](../api/public-api.md#fluent-model-builder).

### Run a runtime-length scan Model

Use `buildWithLengths()` to build a Model with two ordered inputs: values and valid lengths.
F32/F64 `rnnScan` and qualified fixed recipes support first-order configured training.
The list-typed `Model<List<Tensor>,List<Tensor>>` keeps inputs explicit, while `compute(values,
validLengths)` returns the sole detached result. No prior forward is required, before or after load;
this workflow has no Data dependency. The example below retains fixed GRU/LSTM inference.

The example below uses the same Java 26/NN/CPU prerequisites as the fluent examples
above. It supplies canonical FLOAT32 `[4,3,2]` values and non-gradient INT64 `[3]` lengths
`[4,2,0]`: four valid time steps for row 0, two for row 1, none for row 2. Use an existing parent
directory; save atomically replaces an existing destination. It saves before the first compute,
closes the original Model, then loads and calls the same compute overload:

```java
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import java.nio.file.Path;
import java.util.Optional;

public final class RuntimeLengthScanModelExample {
    static TensorDescriptor descriptor(DataType type, long... axes) {
        Shape shape = Shape.of(axes);
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    public static void main(String[] args) throws Exception {
        Path file = Path.of(args.length == 0 ? "scan.graph" : args[0]);
        Tensor values = TensorFactory.fromFlatArray(descriptor(DataType.FLOAT32, 4, 3, 2),
                Optional.empty(), new float[]{
                    .1f, .2f, .3f, .4f, .5f, .6f,
                    .7f, .8f, .9f, 1f, 1.1f, 1.2f,
                    1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
                    1.9f, 2f, 2.1f, 2.2f, 2.3f, 2.4f});
        Tensor validLengths = TensorFactory.fromFlatArray(descriptor(DataType.INT64, 3),
                Optional.empty(), new long[]{4, 2, 0});
        HostTensorValue scores;
        try (var model = Model.builder().input(DataType.FLOAT32, 4, 3, 2).seed(42)
                .gruScan(8).lstmScan(6).finalHidden().linear(5).buildWithLengths()) {
            model.save(file); // No compute or forward has occurred.
            scores = model.compute(values, validLengths);
            System.out.println("shape [3,5]: " + scores.shape().equals(Shape.of(3, 5)));
        }
        try (var loaded = Model.loadGraph(file)) {
            HostTensorValue restored = loaded.compute(values, validLengths);
            System.out.println("same scores: " + scores.bytes().equals(restored.bytes()));
        }
    }
}
```

Expected output is `shape [3,5]: true` followed by `same scores: true`. The GRU produces
`[4,3,8]`, the LSTM `[4,3,6]`, and `finalHidden()` selects the existing `[3,6]` final hidden port
before the ordinary Linear head produces `[3,5]`. It selects states after four, two and zero
transitions, not dense time index 3. The zero-length hidden row is zero. This example's fresh head
also has zero bias, but an ordinary supplied nonzero head bias can make that row's scores nonzero.
The detached original scores remain readable after the original Model closes. The file contains
the graph and private state, not input batches or a prepared executable; input positions survive
without inferred length-role metadata or a new format. Use `loadGraph`, not single-boundary `load`.

Each scan starts from independent zero hidden state (and cell state for LSTM), uses biased FORWARD
transitions and existing step seeds, and shares one canonical INT64 `[B]` lengths input. No state
carries between calls. The first scan requires positive static time/batch/features (TBF); an ordinary
prefix establishes that interpretation. Later scans require the same T/B and infer only features.
`finalHidden()` must immediately follow a scan, not static recurrence, a transform or another
selection, and counts as an appended seed step. Append a classifier after it. Advanced graph/list
boundaries remain available for explicit states, finalCell, reverse/bias-free scans and zero T/B.

Only scans skip their own padded transitions. Read these ordinary steps with their existing dense
meaning; there is no model-wide mask or original-padding-independence guarantee:

| Step | Meaning alongside scans |
|---|---|
| Linear, RELU/TANH/SIGMOID | Operate at every coordinate; bias can change padded zeros and sigmoid(0) is 0.5. |
| Convolution/pooling | Keep channels-first geometry and may mix axes interpreted elsewhere as time/batch; they are not sequence-aware. |
| Flatten | Preserves axis zero: `[T,B,H]` becomes `[T,B*H]`, not `[B,T*H]`. A subsequent scan requires rank three again. |
| Static rnn/gru/lstm | Ignore runtime lengths and unroll every time step, unchanged. |
| lastTimeStep | Still selects `T-1`, not each row's final hidden state; ordinary intervening steps may change padded zeros. |
| Another scan | Shares lengths and ignores its own input suffix, including suffix values changed by ordinary intervening steps. |

F32/F64 RNN now uses general Scan with canonical source outputs and qualified first-order training.
Its CPU actual reads require dense zero-offset geometry (including dense views) or existing
proved scalar-broadcast exceptions. Fixed GRU/LSTM and low RNN retain their separately declared
resolved nonnegative-stride/offset read domain. Each occurrence still needs backend qualification;
there is no implicit conversion, layout copy or CPU fallback for unsupported composition.

| Symptom | Cause and correction |
|---|---|
| `build()` rejects a scan chain | Finish with `buildWithLengths()`; that terminal rejects chains with no scan before state acquisition. |
| `finalHidden()` rejects | Place it immediately after a new scan recipe, before any classifier or transform. |
| Later scan rejects geometry | Preserve exact time/batch axes; do not reshape lengths or guess a different batch axis. |
| Compute rejects length input | Match canonical non-gradient INT64 `[B]` and all declared descriptor fields. Values must lie in `[0,T]`; repair out-of-range values and reuse the Model. |
| Configure/train rejects | Check the selected backend's actual forward/AD closure and use a scalar gradient-eligible F32/F64 objective. Low objectives need an explicit cast; fixed versus general recipe identity is preserved rather than bypassed. |

See [builder API details](../api/public-api.md#fluent-model-builder) and
[training admission](../api/training-api.md#owned-model-training-convenience).

### Train a runtime-length RNN

F32/F64 `rnnScan` composes the existing RNN cell equation through general Scan. Configure once,
then supply fresh values and INT64 lengths at the same static signature. Save before compute or
after training; close and `loadGraph`, then configure a new objective and optimizer on restored
weights. This is not exact optimizer continuation. Old v3 fixed-RNN files keep their fixed recipe;
fresh first-order configured training is available when that recipe's backend/AD closure qualifies.

The example below uses the same JVM/NN prerequisites and an existing destination directory.
Inactive sequence suffixes contain NaN deliberately: recurrence skips them rather than masking
an already-computed value. Padding does not automatically mask the classifier loss.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

public final class RuntimeLengthRnnTrainingExample {
    private static Tensor values(long[] axes, float... lanes) {
        Shape shape = Shape.of(axes);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), lanes);
    }

    private static Tensor lengths(long... lanes) {
        Shape shape = Shape.of(lanes.length);
        var descriptor = new TensorDescriptor(DataType.INT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), lanes);
    }

    private static Tensor loss(List<Tensor> prediction, List<Tensor> target) {
        Tensor error = prediction.getFirst().sub(target.getFirst());
        return error.mul(error).sum();
    }

    public static void main(String[] args) throws Exception {
        Path file = Path.of(args[0]);
        Tensor target = values(new long[]{2, 1}, .2f, -.1f);
        Tensor first = values(new long[]{3, 2, 2},
                .1f, .2f, .3f, .4f,
                .2f, .3f, Float.NaN, Float.NaN,
                .3f, .4f, Float.NaN, Float.NaN);
        Tensor fresh = values(new long[]{3, 2, 2},
                .2f, .1f, Float.NaN, Float.NaN,
                .3f, .2f, Float.NaN, Float.NaN,
                Float.NaN, Float.NaN, Float.NaN, Float.NaN);
        Tensor firstLengths = lengths(3, 1), freshLengths = lengths(2, 0);

        try (var model = Model.builder().input(DataType.FLOAT32, 3, 2, 2).seed(37)
                .rnnScan(3).finalHidden().linear(1).buildWithLengths()) {
            model.save(file); // Cold: no compute or configured training yet.
            model.configureTraining(List.of(target), RuntimeLengthRnnTrainingExample::loss, new Sgd(.02));
            model.train(List.of(first, firstLengths), List.of(target));
            model.train(List.of(fresh, freshLengths), List.of(target));
            model.save(file);
        }
        try (var loaded = Model.loadGraph(file)) {
            System.out.println("shape [2,1]: " + loaded.compute(fresh, freshLengths).shape().equals(Shape.of(2, 1)));
            loaded.configureTraining(List.of(target), RuntimeLengthRnnTrainingExample::loss, new Sgd(.02));
            var step = loaded.train(List.of(fresh, freshLengths), List.of(target));
            System.out.println("new optimizer step: " + step.optimizerStep().orElseThrow());
        }
    }
}
```

Output is `shape [2,1]: true` and `new optimizer step: 1`. All-zero lengths preserve the recurrent
initial state and produce zero recurrent-parameter gradients; a downstream head can still train.
Invalid lengths reject before parameter/optimizer mutation; repair values and reuse the session.
Direct `RnnSequence.forwardGeneralScan` additionally exposes reverse traversal, bias-free cells,
explicit initial state and zero T/B. General final hidden is carry port 0 and dense output emission
port 1; retained fixed result ordering is unchanged. See the [NN0033 Result](../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result)
for independent forward/BPTT and owned lifecycle evidence in both CPU dtypes.


### Compose explicit-state scans

Fixed scans remain separate from static sequences. For advanced cold persistence, use the
existing list `Model.graph`/`Model.loadGraph` boundary with INT64 runtime lengths
as an input, alongside sequence and any external initial states. The cold roundtrip
recreates storage-free prototypes and rebinds fresh exact-descriptor inputs without reading length
values. It retains all canonical ports of one scan even when only finalCell is returned.
The three one-directional NN Sequences also offer additive `forwardScan` methods using the same
owned cell parameters. For example, this complete composition fragment creates a reverse LSTM
scan with explicit lengths and states (imports shown); it does not execute values:

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.LstmSequence;
import java.util.Optional;

var lstm = new LstmSequence(4, true, DataType.FLOAT32,
        ParameterInitialization.glorotUniform(), 42L);
var x = TensorFactory.zeros(Shape.of(3, 2, 2), DataType.FLOAT32, Optional.empty(), false);
var lengthShape = Shape.of(2);
var lengths = TensorFactory.fromFlatArray(new TensorDescriptor(DataType.INT64, lengthShape,
        Optional.of(LayoutDescriptor.contiguous(lengthShape)), false),
        Optional.empty(), new long[] {3, 1});
var h0 = TensorFactory.zeros(Shape.of(2, 4), DataType.FLOAT32, Optional.empty(), false);
var c0 = TensorFactory.zeros(Shape.of(2, 4), DataType.FLOAT32, Optional.empty(), false);
var scan = lstm.forwardScan(x, lengths, h0, c0, RecurrentDirection.REVERSE);
```

`scan.outputs()` is dense `[3,2,4]`, and both final states are `[2,4]`. Reverse traversal visits
row 0 at times 2, 1, 0 and row 1 only at time 0, retaining original output coordinates. With these
zero inputs/states and zero bias, every computed output is zero; padding at row 1 times 1 and 2
is specifically positive zero. Static `forward(..., long[])` instead returns a compact output
list and keeps its existing provenance/training behavior. `forwardScan` creates one producer and
binds an automatic cell even for empty time/batch or all-zero runtime lengths; static unrolling
with no represented step leaves it unbound. All floating descriptors must match the configured
type exactly; a mismatch fails before initialization. Neither path infers lengths from padding.

Advanced callers separately use ordinary Engine compile/prepare/run or runValues for fully static
homogeneous FLOAT32/FLOAT64 CPU scans, with resolved nonnegative read layouts,
INT64 lengths, explicit states and packed parameters. Fresh input storage must contain its full
referenced span, including any offset prefix and strided gaps, not only its logical element count.
Prepare reads no length values;
each run checks all lengths in `[0,T]` before writing scan results. Repair invalid lengths and
reuse the preparation with the same descriptors. Backpropagation through time and serialized
prepared loops remain unsupported. Scan gradient flags remain intact even in eval. Owned Model
compute uses the same Engine path, while fixed-scan training and BPTT remain unsupported. Compile selected `outputs()`,
`finalHidden()` and `finalCell()` Tensors and bind every compiled input slot, including weights
and states. Follow [fresh-value binding](#run-fresh-values-in-compiled-order), replacing lengths
with `[0,2]` under the same descriptors to reuse the preparation; row 0 then preserves its initial
states semantically, not by Java Tensor identity. Lengths `[4,1]` fail for `T=3`; repair and reuse.
For symbolic continuation, pass both final-state Tensors to another LSTM `forwardScan`; across
separate runs, explicitly supply materialized states matching the next graph's descriptors.
Existing builder `rnn/gru/lstm` and bidirectional containers remain static; explicit builder scan
recipes opt into runtime lengths. Data integration is not part of this API.
See the [cold scan scenario](../api/graph-checkpoint-api.md#selected-cnn-and-recurrent-persistence)
and [CPU execution scope](../backend-guide/cpu-backend.md#fixed-recurrent-forward-execution).
Low fixed-scan source constructors, storage and cold files remain supported in their declared
source domain. CPU BFLOAT16, FLOAT16 and both FP8 types use the existing fixed-scan owner,
FLOAT32 working sites and declared state narrowing; geometry and actual derivative guards remain.

## Ordinary CPU execution with changing batch size

Use the ordinary Engine API when one prepared FLOAT32 matrix-loss graph must accept different
batch sizes. This does not change NN Model or Training session signatures, which remain static.
With JDK 26 and the Engine module available, this complete example compiles a scalar loss and
actual first-order gradients once, then supplies fresh canonical values in compiled input order:

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;

class ChangingBatch {
    static TensorDescriptor descriptor(Shape shape, boolean grad) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }
    static Tensor ones(Shape shape, boolean grad) {
        float[] data = new float[Math.toIntExact(shape.knownElementCount().orElseThrow())];
        Arrays.fill(data, 1);
        return TensorFactory.fromFlatArray(descriptor(shape, grad), Optional.empty(), data);
    }
    public static void main(String[] args) {
        Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(4)),
                Optional.empty(), true));
        Tensor w = TensorFactory.create(descriptor(Shape.of(4, 3), true));
        Tensor seed = ones(Shape.scalar(), false);
        try (Engine engine = Engine.standard()) {
            var compiled = engine.compile(List.of(x.matmul(w).mean()), List.of(seed), List.of(x, w));
            try (var prepared = engine.prepare(compiled)) {
                for (int b : new int[] {64, 40, 1, 64, 0}) {
                    Tensor actualX = ones(Shape.of(b, 4), true);
                    Tensor actualW = ones(Shape.of(4, 3), true);
                    var values = compiled.inputs().stream().map(i -> i.tensorId().equals(x.id())
                            ? actualX : i.tensorId().equals(w.id()) ? actualW : seed).toList();
                    try (var result = engine.runValues(prepared, values)) {
                        var loss = result.materialize(result.publications().get(0), 1_000_000);
                        System.out.println(b + ": " + loss.bytes().getFloat());
                    }
                }
            }
        }
    }
}
```

The symbolic X has no storage; actual X values declare concrete B and keep F=4 fixed. The explicit
scalar-one seed differentiates only the published scalar loss. The result order is loss, dX, dW:
for positive B, loss is 4, each dX is approximately `1/B`, and each dW is approximately `1/3` under
FLOAT32 rounding. Their shapes are scalar, `[B,4]`, `[4,3]`; the parameter gradient is not a bank
of B parameter copies. Printed losses are `4.0, 4.0, 4.0, 4.0, NaN`. At B=0, dX is empty and dW
contains positive zeros. Slots, route and generated code are reused; results own independent
writable storage and remain materializable after prepared close while Engine remains open.

A wrong fixed feature count, gradient flag, missing/noncanonical/offset dynamic input layout,
inconsistent B, overflow or unusable storage rejects before borrowing. Dynamic inputs require
present canonical zero-offset layouts; dense read views qualify. Static slots still require exact
descriptors. For qualified Scan execution, use the example below. Neither example enables
NN/Training reuse, symbolic cold/wire loading, native/mixed-owner/vector/parallel execution or
tuning. See the [CPU slice](../backend-guide/cpu-backend.md#ordinary-named-batch-cpu-execution).

## Symbolic Scan with changing batch size

Use the same public Engine lifecycle for an all-valid Scan: every time step participates, time
and feature widths stay fixed, and one named batch B changes between calls. JDK 26 and Engine/CPU
modules are prerequisites. Current ordinary CPU general Scan qualifies FLOAT32/FLOAT64 forward
and actual selected VJPs under these ordinary single-B, fixed-T/features guards. The FLOAT32
example below retains the historical generated scalar, single-range Prepare0014 layer; its
[task](../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md) retains that evidence,
not a global FLOAT32-only restriction or broader ordinary symbolic F64 MATMUL admission.

This complete example uses three steps of `next = h + step`, width three, zero initial carry,
one-valued sequence and joint final/emission one seeds. It compiles the forward outputs and
actual Compiler-generated gradients once, then binds fresh values in final compiled input order.

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;

class ChangingScanBatch {
    static TensorDescriptor descriptor(Shape shape, boolean grad) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape))
                        : Optional.empty(), grad);
    }
    static Tensor filled(Shape shape, boolean grad, float value) {
        float[] data = new float[Math.toIntExact(shape.knownElementCount().orElseThrow())];
        Arrays.fill(data, value);
        return TensorFactory.fromFlatArray(descriptor(shape, grad), Optional.empty(), data);
    }
    public static void main(String[] args) {
        var b = new DynamicDimension("B");
        Shape row = Shape.ofDimensions(b, new StaticDimension(3));
        Shape sequence = Shape.ofDimensions(new StaticDimension(3), b, new StaticDimension(3));
        Tensor h = TensorFactory.create(descriptor(row, true));
        Tensor step = TensorFactory.create(descriptor(row, true));
        Tensor initial = TensorFactory.create(descriptor(row, true));
        Tensor x = TensorFactory.create(descriptor(sequence, true));
        Tensor next = h.add(step);
        var scan = Scan.apply(List.of(initial), List.of(x), List.of(),
                new ScanBody(List.of(h), List.of(step), List.of(), List.of(next), List.of(next)), false);
        Tensor fs = TensorFactory.create(descriptor(row, false));
        Tensor es = TensorFactory.create(descriptor(sequence, false));
        try (Engine engine = Engine.standard()) {
            var graph = engine.compile(List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()),
                    List.of(fs, es), List.of(initial, x));
            try (var prepared = engine.prepare(graph)) {
                for (int batch : new int[] {64, 40, 1, 64}) {
                    Shape actualRow = Shape.of(batch, 3), actualSequence = Shape.of(3, batch, 3);
                    var supplied = Map.of(initial.id(), filled(actualRow, true, 0),
                            x.id(), filled(actualSequence, true, 1),
                            fs.id(), filled(actualRow, false, 1), es.id(), filled(actualSequence, false, 1));
                    var values = graph.inputs().stream().map(i -> supplied.get(i.tensorId())).toList();
                    try (var result = engine.runValues(prepared, values)) {
                        var carry = result.materialize(result.publications().get(0), 1_000_000);
                        var dInitial = result.materialize(result.publications().get(2), 1_000_000);
                        System.out.println(batch + ": " + carry.bytes().getFloat()
                                + " / " + dInitial.bytes().getFloat());
                    }
                }
            }
        }
    }
}
```

Each line prints `B: 3.0 / 4.0`, with B equal to `64, 40, 1, 64`. Forward carry progresses
through 1, 2 and 3; emissions retain those original-time frames. Publications are final carry,
emissions, dInitial and dX. The final seed and three emission seeds each contribute once, giving
dInitial=4 and dX time frames 4, 3 and 2. This arithmetic walkthrough is not a performance or
nonuniform conformance oracle; focused tests also compare actual reverse with ordinary unrolling.

Symbolic templates have no storage or resolved numeric layout. Actual inputs/seeds have canonical
static descriptors and unchanged gradient flags; caller storage stays borrowed until result close.
Code/routes/slots are reused, but writable outputs, working arrays and results are fresh. B=0 is
valid for this algebra and gives empty row outputs; the example omits it because it prints a cell.
Other bodies can still require nonempty shared work at B=0. Changing time requires a separate
compile/prepare, including T=0. Inconsistent B, changed fixed widths, unusable storage or unsupported
layouts reject. Current all-valid symbolic general Scan composes ordinary CPU-supported floating
types, including all four low types, under the same guards; adding lengths selects the separate
length-aware rules below. Neither route grants arbitrary symbolic geometry, NN/Training reuse,
symbolic cold/copy/wire, native/mixed-owner execution or a geometry memo. See the
[qualified CPU domain](../backend-guide/cpu-backend.md#symbolic-scan-cpu-execution).

## Symbolic-length Scan with changing batch size

For variable active prefixes, pass non-gradient INT64 `[B]` lengths to the existing `Scan.apply`
overload. Keep time/features fixed and use a qualified floating body under its per-role dtype guards.
The same Engine/CPU and JDK 26 prerequisites apply. Prepare0015 implements this path; owning
tests and independent code/structural review passed, with root validation and the sole bounded probe completed
(evidence and limits:
[Prepare0015 Result](../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. No new Engine API or per-batch preparation is needed.

This complete FLOAT64 example uses three steps of `next = h + step`, width two, and joint final/
emission one seeds. The original carry and every original step are ROW(0): their leading coordinate
is the same named B. Each call supplies new lengths and static storage in compiled input order.

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;

class ChangingLengthScanBatch {
    static TensorDescriptor descriptor(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape,
                shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape))
                        : Optional.empty(), grad);
    }
    static Tensor filled(Shape shape, boolean grad, double value) {
        double[] data = new double[Math.toIntExact(shape.knownElementCount().orElseThrow())];
        Arrays.fill(data, value);
        return TensorFactory.fromFlatArray(descriptor(DataType.FLOAT64, shape, grad),
                Optional.empty(), data);
    }
    public static void main(String[] args) {
        var b = new DynamicDimension("B");
        Shape row = Shape.ofDimensions(b, new StaticDimension(2));
        Shape stack = Shape.ofDimensions(new StaticDimension(3), b, new StaticDimension(2));
        Tensor h = TensorFactory.create(descriptor(DataType.FLOAT64, row, true));
        Tensor step = TensorFactory.create(descriptor(DataType.FLOAT64, row, true));
        Tensor initial = TensorFactory.create(descriptor(DataType.FLOAT64, row, true));
        Tensor x = TensorFactory.create(descriptor(DataType.FLOAT64, stack, true));
        Tensor lengths = TensorFactory.create(descriptor(DataType.INT64, Shape.ofDimensions(b), false));
        Tensor next = h.add(step);
        var scan = Scan.apply(List.of(initial), List.of(x), List.of(), lengths, 1,
                List.of(OptionalInt.of(0)),
                new ScanBody(List.of(h), List.of(step), List.of(), List.of(next), List.of(next)), false);
        Tensor fs = TensorFactory.create(descriptor(DataType.FLOAT64, row, false));
        Tensor es = TensorFactory.create(descriptor(DataType.FLOAT64, stack, false));
        try (Engine engine = Engine.standard()) {
            var graph = engine.compile(List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()),
                    List.of(fs, es), List.of(initial, x));
            try (var prepared = engine.prepare(graph)) {
                for (int batch : new int[] {3, 1, 0, 3}) {
                    long[] ls = new long[batch];
                    for (int r = 0; r < batch; r++) ls[r] = r % 4;
                    Tensor lv = TensorFactory.fromFlatArray(
                            descriptor(DataType.INT64, Shape.of(batch), false), Optional.empty(), ls);
                    var values = Map.of(initial.id(), filled(Shape.of(batch, 2), true, 0),
                            x.id(), filled(Shape.of(3, batch, 2), true, 1), lengths.id(), lv,
                            fs.id(), filled(Shape.of(batch, 2), false, 1),
                            es.id(), filled(Shape.of(3, batch, 2), false, 1));
                    var ordered = graph.inputs().stream().map(i -> values.get(i.tensorId())).toList();
                    try (var result = engine.runValues(prepared, ordered)) {
                        var carry = result.materialize(result.publications().get(0), 1_000_000);
                        var dInitial = result.materialize(result.publications().get(2), 1_000_000);
                        System.out.println(batch + ": " + carry.shape());
                        if (batch > 0)
                            System.out.println(carry.bytes().getDouble() + " / "
                                    + dInitial.bytes().getDouble());
                    }
                }
            }
        }
    }
}
```

At B=3, lengths are `[0,1,2]`: each final row is its length, inactive emission frames are +0,
and each dInitial row is `1 + length` from the final seed and active emission seeds. The first row
therefore prints `0.0 / 1.0`, including B=1. B=0 publishes empty row outputs safely; do not read a
cell from them. The carry descriptor is `[B,2]` resolved to each actual batch. Forward and reverse
reuse the same code/routes/slots while each run owns fresh outputs, resources and mutable arrays.

Use FLOAT32 descriptors and `float[]`/`getFloat()` for the corresponding FLOAT32 example; do not
change just the descriptor while leaving double storage. For produced lengths, construct a
non-gradient floating `[B]` template, pass its `.cast(DataType.INT64)` result to `Scan.apply`, and
bind the floating producer input instead of the derived INT64 value. Casting follows ordinary
cast semantics; the resulting lengths must still lie in `[0,3]`.

Lengths are checked after producers and before any Scan-local write, even when time or features
are zero. An invalid length can be repaired for a later run on the same preparation; producer writes
are not rolled back. Padding is not read in primal or adjoint work; an inactive row preserves raw
initial carry/final-seed bits. Lengths do not make `mean(finalCarry)` divide by the active count:
its denominator remains dense B×width, and an empty mean is NaN. Changing T or feature widths needs
a different compilation/preparation. CPU low types reuse the ordinary floating/actual derivative
owners here; this does not grant arbitrary geometry, higher-order AD, native/mixed-owner execution,
symbolic cold/copy/wire, NN/Training reuse or a geometry memo. See the
[exact supported occurrences](../backend-guide/cpu-backend.md#symbolic-length-scan-cpu-execution).

## Owned static Model workflow

For an application-built and file-restored model with the same compute/train/save API, opt into
an **owned source**: ordinary static Tensor provenance over private weight values. The builder above
retains layer composition; the definition/graph/load paths below reconstruct the bounded source.
Use Java 26 and an explicit `:extensions:nn` dependency with the JVM prerequisites above.
No caller Arena is required for this workflow.
Definition/graph acquisition uses the bounded
[cold source slice](../architecture/contracts/compiler-autograd.md#compile-lifecycle), including its
general-body exception, not all NN layers or arbitrary PURE bodies. Save/load uses the
[body-free persistence slice](../api/graph-checkpoint-api.md#selected-cnn-and-recurrent-persistence)
or [bounded v4 body persistence](../api/graph-checkpoint-api.md#bounded-general-scan-persistence),
through the same Model API.
Execution and training support are independent of raw graph persistence's nine data types.
FP8 graph sources select v7; NN values/definitions select their separate v2 formats only when
FP8 is present. Old-only sources retain their original smallest sufficient formats and bytes.
Files retain logical low storage, not FLOAT32 optimizer masters. Loading and opening a new
Training session starts fresh optimizer state; this is not optimizer continuation.

This complete example learns one FLOAT32 weight, starting at 2. Both batch and target are [1,1].
It saves before compute, trains on input 3 and target 1, then restores the trained source.

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Optional;

public final class OwnedModelExample {
    static Tensor matrix(float value, boolean gradient) {
        Shape shape = Shape.of(1, 1);
        Tensor tensor = TensorFactory.allocate(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient));
        tensor.hostStorage().orElseThrow().segment()
                .set(ValueLayout.JAVA_FLOAT_UNALIGNED, 0, value);
        return tensor;
    }

    public static void main(String[] args) throws Exception {
        Tensor externalWeight = matrix(2, true);
        TensorDescriptor signature = matrix(0, false).descriptor();
        Path file = Path.of("owned.graph");
        HostTensorValue retained;
        try (Model<Tensor, Tensor> model = Model.defineTensor(signature, topology -> {
            Linear linear = topology.addModule("linear", new Linear(externalWeight));
            return linear::forward;
        })) {
            model.save(file); // no Engine yet; source and signature are already fixed
            retained = model.compute(matrix(3, false)); // prediction 6, lazy CPU Engine
            model.configureTraining(signature, (prediction, target) -> {
                Tensor error = prediction.sub(target);
                return error.mul(error).sum();
            }, new Sgd(0.1));
            var step = model.train(matrix(3, false), matrix(1, false));
            System.out.println(step.objective().bytes().getFloat()); // 25
            model.save(file); // working weight is now -1; externalWeight remains 2
        }
        try (Engine borrowed = Engine.standard();
                Model<Tensor, Tensor> restored = Model.load(file).withEngine(borrowed)) {
            System.out.println(restored.compute(matrix(2, false)).bytes().getFloat()); // -2
        }
        System.out.println(retained.bytes().getFloat()); // 6 after all owners close
    }
}
```

The squared error is `(2*3-1)^2 = 25`; SGD applies gradient 30, so weight becomes `2-0.1*30 = -1`.
Acquisition copied external values once and installed final private leaves into the adopted wrappers.
Definition and prototype forward each ran once in eval mode; their callbacks are discarded. Explicit
`forward` is cold symbolic composition, not a batch execution path. Construction failure cannot
undo arbitrary callback effects. Do not mutate adopted child modules after construction.

The signature includes dtype, static Shape, layout presence/geometry and requiresGrad. Even unused
declared slots must match and have usable storage. A [2,1] batch rejects rather than recompiling.
`configureTraining` preserves mode, creates one native parameter working set when needed, and
succeeds once; compute can prepare again after this explicit transition, not after each update.
Configure training anew after load: objectives, optimizer history, RNG and data cursors are not saved.

For multiple Tensor inputs/outputs, use `Model.graph(inputs, outputs, namedLeaves)` and list
compute/train methods; name every reachable state leaf with its explicit PARAMETER, BUFFER or
CONSTANT role in canonical Module traversal order. `loadGraph` restores that list boundary;
single `load` rejects it with a hint. List compute accepts an aggregate `maximumTotalBytes`, checks
all output sizes first, closes its result and returns only a complete detached list.

Select a borrowed Engine before the first valid compute/configure setup; `withEngine` returns the
same model. That selection freezes even if setup fails, and a closed selected Engine cannot be
replaced. Without a selection, only execution opens a fresh model-owned CPU Engine. Close never
closes a borrowed Engine; it waits for active work and closes session/preparation before native
storage and an owned Engine, attempting all cleanup and replaying retained failures.
Fixed and mixed recurrent sources use this same setup ordering. Unsupported backend or derivative
closure is not replaced by a separate inference-only NN guard or an implicit recipe conversion.

| Symptom | Cause and correction |
|---|---|
| Unsupported tensor capability | Generic `Model.define`, custom subclasses and DeclarativeModel retain old forward behavior. Explicitly acquire a supported graph boundary. |
| Busy or closed/closing | Another high-level operation or close owns admission. Serialize calls; keep borrowed Engine open through Model use. |
| Source changed | Mode, paths/wrappers, Tensor or storage associations changed. Construct a replacement Model; restoring a binding does not revive an invalid capability. |
| Native training admission rejects | Current Training rejects unresolved/empty/nonfinite/aliased parameters. Supply supported state before acquisition; there is no implicit conversion or alias splitting. |
| Unsupported source operation | Graph/definition acquisition and general-body persistence require the bounded cold inventory, not arbitrary PURE operations or attributes. Body-only signatures do not widen outer admission. Builder execution is independent of save admission. |

Short save/load forms use 256 MiB file/aggregate, 4 MiB manifest, 16384 entries, 64-byte keys and
64 MiB payload ceilings; explicit `ArtifactLimits` overloads remain. Old `model.v1` and
`whole-model.v1` files still use `nn.checkpoint.ModelCheckpoint`, not `Model.load`.
Zero-node source may save/load but retains Engine's empty-plan rejection. Model/Training autotuning
is not integrated; advanced `Engine.prepareTuned` remains separate. See the
[API contracts](../api/public-api.md#owned-tensor-model-capability).

### Train an all-valid general scan

Use `Model.graph` or `defineTensor` for a fixed-count general scan, not a new ModelBuilder recipe.
**All-valid** means every static time step participates; there is no runtime length or padding mask.
Use the Java 26/NN/CPU setup above. Current general Scan composes every ordinary CPU-supported
PURE unit under ordinary dtype/layout/value and derivative guards, with selected actual
floating first-order AD, including the four low types; forward and actual reverse qualify independently. All-valid
bodies use WholeTensor. Families without backward retain that limit. Cold acquisition is
separate: numerical CPU parity for BFLOAT16, FLOAT16 and both OFP8 types does not itself widen
source acquisition. It requires the exact
[cold constructor inventory](../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction),
not merely PURE operations. CPU low execution uses FLOAT32 working values and final logical
narrowing, with existing family, geometry and actual derivative guards. It does not widen NN
recipes or Metal Scan admission; Metal has no FP8 support.
Length-aware body values may be proved ROW(axis) values or arbitrary-shape SHARED tensors.
Lengths is a separate final non-gradient INT64 `[B]` control with no body formal, not a scalar
alternative. Original AD carries/sequences and emission boundaries remain ROW(0); intermediate
and history ROW axes/ranks follow their proofs. Supported row-mixing materialization requires
exactly the same length values, not Tensor identity. Structural absence differs from a present
all-zero seed; cross-Scan materialization is explicit.

This complete FLOAT32 example acquires weight `0.5` privately once. The body computes
`next = carry * weight + step` and emits the pre-step carry. Input `[1,2,3]`, initial carry `0`
and forward traversal produce final carry `4.25` and original-time emissions `[0,1,2.5]`.
It saves before compute and after training, then closes and reloads the trained source. Run in a
trusted writable working directory: the example atomically replaces `general-scan-before.graph`
and `general-scan-trained.graph` if they already exist.

```java
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

public final class OwnedGeneralScanExample {
    static Tensor value(Shape shape, boolean grad, float... lanes) {
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), lanes);
    }

    static Tensor squared(Tensor prediction, Tensor target) {
        Tensor error = prediction.sub(target);
        return error.mul(error).sum();
    }

    public static void main(String[] args) throws IOException {
        Path before = Path.of("general-scan-before.graph");
        Path trained = Path.of("general-scan-trained.graph");
        Tensor initial = value(Shape.of(1), false, 0);
        Tensor weight = value(Shape.of(1), true, .5f);
        Tensor sequence = TensorFactory.create(value(Shape.of(3, 1), false, 0, 0, 0).descriptor());
        Tensor h = TensorFactory.create(initial.descriptor());
        Tensor x = TensorFactory.create(initial.descriptor());
        Tensor w = TensorFactory.create(weight.descriptor());
        var body = new ScanBody(List.of(h), List.of(x), List.of(w),
                List.of(h.mul(w).add(x)), List.of(h));
        var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, false);
        var leaves = List.of(
                new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight),
                new GraphCheckpoint.Leaf("initial", GraphCheckpoint.Role.CONSTANT, initial));
        try (var model = Model.graph(List.of(sequence),
                List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()), leaves)) {
            model.save(before); // Cold save creates no Engine and saves no input batch.
            var batch = value(Shape.of(3, 1), false, 1, 2, 3);
            System.out.println(model.compute(List.of(batch)).getFirst().bytes().getFloat()); // 4.25
            Tensor finalTarget = TensorFactory.create(initial.descriptor());
            Tensor emissionTarget = TensorFactory.create(sequence.descriptor());
            model.configureTraining(List.of(finalTarget, emissionTarget), (p, t) ->
                    squared(p.get(0), t.get(0)).add(squared(p.get(1), t.get(1))), new Sgd(.01));
            var step = model.train(List.of(batch), List.of(value(Shape.of(1), false, 0),
                    value(Shape.of(3, 1), false, 0, 0, 0)));
            System.out.println(step.objective().bytes().getFloat()); // 25.3125, before update
            model.save(trained);
        }
        try (var restored = Model.loadGraph(trained)) { // Heap state; no Engine on load.
            var freshBatch = value(Shape.of(3, 1), false, 1, 2, 3);
            System.out.println(restored.compute(List.of(freshBatch))
                    .getFirst().bytes().getFloat()); // approximately 3.428025
            // Objectives and optimizer state are not saved: configure a new session.
            restored.configureTraining(List.of(TensorFactory.create(initial.descriptor()),
                    TensorFactory.create(sequence.descriptor())), (p, t) ->
                    squared(p.get(0), t.get(0)).add(squared(p.get(1), t.get(1))), new Sgd(.01));
            restored.train(List.of(freshBatch), List.of(value(Shape.of(1), false, 0),
                    value(Shape.of(3, 1), false, 0, 0, 0)));
        }
    }
}
```

The invariant formal `w` makes the shared weight explicit; a captured weight inside the body would
be an undeclared free leaf and reject. The CONSTANT initial carry stays outside `stateDictionary`.
The scalar objective consumes both output roles: `4.25² + 0² + 1² + 2.5² = 25.3125`.
Its weight gradient is `30.5`, so SGD changes the owned weight to approximately `0.195`; the caller's
weight remains `0.5`. Later compute sees that same updated native store. Pass fresh exact-descriptor
batches/targets to further compute/train calls; neither cold `forward` nor reconstruction is needed.
There is one inference preparation per setup and one configured training session, not one per batch.

At T=0 the body does not execute; Training still requires connected, nonempty parameters. Residual
ports may be cold roots, but residual-root differentiation rejects. Qualified fixed and mixed
fixed/general sources use the same configured session and actual Compiler AD.
This all-valid source saves as v4; the separate static length intersection uses v5.
Low execution still requires a supporting backend and an explicit F32/F64 training objective.

The two saved sources retain the initial and trained raw weights respectively, with one scoped
body rather than three unrolled steps. `Model.loadGraph` adopts final heap values and reconstructs
ordinary provenance without creating an Engine. The restored compute above uses the trained weight
of approximately `0.195`; its new training session starts without the previous optimizer history.
No input batch or residual history is saved.
Single-boundary sources use `Model.load` instead. Execution remains limited to qualified occurrences.
Focused tests cover this save/load workflow; main accepted final module/Javadoc, integration,
architecture and configured root gates in CP0013. This capability does not migrate `.rnn` or
reinterpret an old file. See [bounded v4 persistence](../api/graph-checkpoint-api.md#bounded-general-scan-persistence).

### Two Linear layers with an activation

A multilayer perceptron (MLP) uses two Linear layers with an intervening activation. Add these
methods to `OwnedModelExample` above and call `activationMlp()` from `main`. This FLOAT32 example
uses the same weights, shapes, batches and squared-error objective as the parameterized
`OwnedModelWorkflowIntegrationTest`. It saves before compute, trains private weights and loads
the trained model; the external leaves remain unchanged.

```java
static Tensor values(Shape shape, boolean gradient, float... lanes) {
    Tensor tensor = TensorFactory.allocate(new TensorDescriptor(DataType.FLOAT32, shape,
            Optional.of(LayoutDescriptor.contiguous(shape)), gradient));
    for (int i = 0; i < lanes.length; i++) {
        tensor.hostStorage().orElseThrow().segment()
                .set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, lanes[i]);
    }
    return tensor;
}

static void activationMlp() throws Exception {
    Tensor firstWeight = values(Shape.of(2, 2), true, 0.5f, -0.25f, 0.25f, 0.5f);
    Tensor firstBias = values(Shape.of(2), true, 0.2f, -0.1f);
    Tensor lastWeight = values(Shape.of(1, 2), true, 0.75f, -0.4f);
    Tensor lastBias = values(Shape.of(1), true, 0.1f);
    Tensor firstBatch = values(Shape.of(1, 2), false, 1, 2);
    Tensor secondBatch = values(Shape.of(1, 2), false, -0.5f, 1);
    Path file = Path.of("activation.graph");
    try (var model = Model.defineTensor(firstBatch.descriptor(), topology -> {
        var first = topology.addModule("first", new Linear(firstWeight, firstBias));
        var last = topology.addModule("last", new Linear(lastWeight, lastBias));
        return input -> last.forward(first.forward(input).relu());
    })) {
        model.save(file); // graph-model.v2, no saved batch or Engine
        System.out.println(model.compute(firstBatch).bytes().getFloat());  // approximately -0.21
        System.out.println(model.compute(secondBatch).bytes().getFloat()); // approximately -0.01
        model.configureTraining(matrix(0, false).descriptor(), (prediction, target) -> {
            Tensor error = prediction.sub(target);
            return error.mul(error).sum();
        }, new Sgd(0.01));
        model.train(firstBatch, matrix(0.3f, false)); // positive loss and nonzero parameter updates
        model.train(secondBatch, matrix(0.3f, false));
        model.save(file); // current private working weights, not optimizer state
    }
    try (var restored = Model.load(file)) {
        System.out.println(restored.compute(firstBatch).bytes().getFloat()); // trained prediction
        // Supply configureTraining again to train the restored model.
    }
}
```

Before training, the hidden values are `[0.2,1.15]` for the first batch and `[-0.3,0.275]` for
the second. RELU removes the second batch's negative contribution; it is not an identity or
transpose. Replace `.relu()` with `.tanh()` or `.sigmoid()` for the other supported activations.
Each activation variant in this eight-kind-only example selects graph-model.v2 without changing
its ordinary Tensor semantics. Graphs with only the original
five kinds still save byte-identical v1; old graph readers reject v2. The regression compares all
three activations' forward values, objectives and nonzero updates against an independent ordinary
Tensor/Engine oracle, including training after load and reloading trained state.

## Prerequisites

- One open `Engine.standard()`, or one explicitly composed CPU/Metal Engine.
- A non-empty graph supported by the Engine's registered backends.
- Every Tensor reported by `CompiledGraph.inputs()`, with compatible live caller-owned host
  storage when each run begins.

The Engine remains the composition owner. A prepared handle borrows it and must not outlive it.
For a supported NN Sequential, [declarative reconstruction](declarative-models.md) can
first build a descriptor-only model with stable state paths. Load its parameter values
before constructing the forward expression to compile; reconstruction itself neither
executes the model nor reconstructs it from a stored-model file.

## Load a saved whole model

To recover a supported bound `Sequential` without rebuilding its layers, use
`ModelCheckpoint.save(path, source, engine)` and `ModelCheckpoint.load(path)`. Load returns a
value-loaded `DeclarativeModel`, not a descriptor-only skeleton. It needs no Engine or Arena;
ordinary forward construction and execution still use the caller's Engine. No backend or prepared
execution is stored in the file.

Use an explicit `:extensions:nn` dependency and import `nn.checkpoint.ModelCheckpoint`;
the generic Checkpoint project no longer exports NN. The old flexible readers/defaults are
unchanged, distinct from [static generic graph persistence](../api/graph-checkpoint-api.md).
This complete Java 26 example needs NN and its public dependencies on the classpath.
Its trusted local directory is created before save. The source has
FLOAT32 diagonal weights `[1, 0; 0, 2]` followed by RELU. Both source owners close before load;
the saved file is then deleted before fresh-CPU inference. The input `[-3, 4]` should produce
`[0, 8]`.

```java
import io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Activation;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.DeclarativeModel;
import io.github.pho001.synaptik.nn.module.ModelSpecification.ActivationKind;
import io.github.pho001.synaptik.nn.module.Sequential;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

public final class WholeModelExample {
    static Tensor tensor(MemorySegment storage, Shape shape, boolean gradientEligible) {
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradientEligible);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32,
                        shape.knownElementCount().orElseThrow(), storage)));
    }

    public static void main(String[] args) throws Exception {
        Path directory = Files.createTempDirectory("whole-model-example");
        Path file = directory.resolve("model.chk");
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            MemorySegment initial = MemorySegment.ofArray(new float[] {1, 0, 0, 2});
            MemorySegment storage = arena.allocate(initial.byteSize(), 4);
            storage.copyFrom(initial);
            Sequential source = new Sequential(List.of(
                    new Linear(tensor(storage, Shape.of(2, 2), true)),
                    new Activation(ActivationKind.RELU)));
            source.train();
            ModelCheckpoint.save(file, source, engine);
        }
        DeclarativeModel loaded = ModelCheckpoint.load(file);
        Files.delete(file);
        Tensor input = tensor(MemorySegment.ofArray(new float[] {-3, 4}), Shape.of(2), false);
        try (Engine fresh = Engine.standard()) {
            var result = fresh.compute(loaded.forward(input)).bytes().asFloatBuffer();
            float first = result.get();
            float second = result.get();
            if (first != 0 || second != 8) throw new AssertionError("restored inference");
            ModelCheckpoint.save(directory.resolve("resaved.chk"), loaded, fresh);
            System.out.println(first + ", " + second);
        }
    }
}
```

Expected output is `0.0, 8.0`. Linear first produces `[-3, 8]`; RELU produces `[0, 8]`.
The source's training mode is not persisted: the loaded tree starts in evaluation mode, while
parameters remain gradient-eligible. Its heap values survive source Arena/Engine closure and file
deletion; re-saving needs no caller expression wrappers or native parameter copy. This does not
prove a saved execution plan or training continuation. Exact repeated parameter identities are
preserved, and compatible batch dimensions can change when a new forward expression is built.

The short calls use absent input-artifact identity. If the file stores one, default load raises
`ArtifactFormatException`; pass an application-verified `InputArtifactFingerprint` and explicit
`ArtifactLimits` instead. A values-only `model.v1` file needs the existing fully bound target-load
API, not automatic topology inference. Custom modules, unbound state and recurrent models are not
supported by this whole-model workflow. See [whole-model limits and failures](../api/model-checkpoint-api.md#bounds-lifetime-and-failures)
before selecting advanced encoded bounds; they are not peak-heap budgets.

## Compile and prepare once

Compile the requested output boundary once, then retain its prepared owner:

```java
CompiledGraph graph = engine.compile(List.of(output));
try (PreparedExecution prepared = engine.prepare(graph)) {
    assert prepared.compiledGraph() == graph;
    // Reuse engine.run(prepared, ...) while both prepared handle and Engine remain open.
}
```

`engine.prepare(graph)` performs exactly one preparation before returning. It does not execute the
graph or bind input storage. The graph fixes both boundaries:

- `graph.inputs()` is the stable input membership and Compiler occurrence order; and
- every result uses the graph's stable forward-then-gradient publication occurrence order,
  including distinct occurrences that alias one inward representation.

## Run and materialize

The input list may use any order because Engine binds by exact Tensor identity:

```java
HostTensorValue retained;
try (PreparedExecution prepared = engine.prepare(graph)) {
    try (RunResult result = engine.run(prepared, List.of(rightInput, leftInput))) {
        RunResult.Publication first = result.publications().getFirst();
        retained = result.materialize(first, 2L * Float.BYTES);
    }

    // Replace a required Tensor's host-storage association here when the next call
    // should use another compatible caller-owned storage.
    try (RunResult result = engine.run(prepared, List.of(leftInput, rightInput))) {
        assert result.publications().getFirst().index() == 0;
    }
}

assert retained.byteSize() == 2L * Float.BYTES;
```

Each call verifies exact Tensor identity and the complete compiled descriptor, then snapshots the
current host-storage associations in `graph.inputs()` order. It validates storage data type,
resolved-layout capacity, liveness, and current-thread accessibility before borrowing. The caller
continues to own the storage and must keep it usable and free from conflicting mutation until that
run's result closes.

`RunResult` is the existing publication lease. Each `Publication` records its dense index,
`TensorId`, final descriptor, forward or gradient role, and applicable derivative metadata.
`materialize(...)` accepts only the exact occurrence object from that result and performs a fresh
bounded copy through the publication adapter captured during preparation. A detached
`HostTensorValue` remains readable after the result, prepared handle, Engine, and input-storage arena close.

## Run fresh values in compiled order

**Goal and inputs.** Prepare negation of a storage-free FLOAT32 [2] prototype once, then supply
fresh batches [1,-2] and [3,-4]. With Java 26, Engine and Model on the classpath, this complete
block uses their ordinary scalar CPU heap-input support:

```java
var shape = Shape.of(2);
var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
        Optional.of(LayoutDescriptor.contiguous(shape)), false);
Tensor prototype = TensorFactory.create(descriptor);
Tensor firstBatch = TensorFactory.fromFlatArray(
        descriptor, Optional.empty(), new float[] {1, -2});
Tensor secondBatch = TensorFactory.fromFlatArray(
        descriptor, Optional.empty(), new float[] {3, -4});
try (Engine engine = Engine.standard()) {
    var graph = engine.compile(List.of(prototype.neg()));
    try (var prepared = engine.prepare(graph)) {
        for (Tensor batch : List.of(firstBatch, secondBatch)) {
            try (var result = engine.runValues(prepared, List.of(batch))) {
                var values = result.materialize(result.publications().getFirst(), 8).bytes();
                System.out.println(values.getFloat() + ", " + values.getFloat());
            }
        }
    }
}
```

Use the Model/Engine imports from the whole-model example above. Compilation fixes the single
input position; preparation is outside the loop. Each run binds that batch's current storage
without changing the prototype. Output is `-1.0, 2.0`, then `-3.0, 4.0`; detached values may be
retained after every owner closes. With multiple inputs, follow `graph.inputs()` order, not
expression discovery order. A value may fill multiple matching slots.

runValues requires exact count and full static descriptors (dtype, shape, layout including absence
versus presence, and requiresGrad). A compiled [32,784] input rejects [64,784]. All storage is
validated before any borrow, including checked referenced spans for resolved layouts or checked
canonical element/byte geometry for absent layouts. Unknown/overflow geometry rejects. This extra
absent-layout check does not change identity-based run; `engine.run(prepared, List.of(firstBatch))`
rejects here because the fresh ID is not the prototype ID.

## Reuse and concurrency

Sequential and concurrent calls share the immutable prepared recipe and direct captured
input/publication adapters. Every call receives a distinct mutable Runtime `RunState`, borrowed
input wrappers, internal buffers, and workspaces. A run performs no compilation, preparation,
provider or availability interrogation, Engine-registry lookup, backend selection, kernel
selection, or schedule assembly.

The prepared owner adds no cache or result type. `Engine.compute(...)` and
`Engine.backward(...)` make a different lifetime choice: every call performs fresh compilation,
preparation, execution, materialization, and cleanup.

## Close and in-flight runs

Close the prepared handle when its reuse window ends. Close is idempotent, closes the one prepared
execution exactly once, and immediately prevents later run admission. It does not close a
`RunResult` that was already returned.

If close races a run, Runtime's existing prepared lease decides the race:

- a run that acquired the lease first may finish; prepared close does not wait for it; and
- a close-first run is rejected.

Engine admission is checked first. Under an open Engine, both run methods check prepared nullness/exact ownership and the exact inward delegate's close
gate before reading the input list, Tensor metadata, storage, or adapters. A run attempted after
prepared close therefore reports `prepared execution is closed` without borrowing storage even when
the supplied inputs are invalid. If Engine closure has begun, `advanced engine is closed` wins
before any argument inspection.

Engine closure waits for admitted Engine operations, then closes retained results in reverse
successful-run order, ordinary and advanced prepared handles in reverse
preparation-publication order, and finally backend integrations. Engine closure is the final
cleanup boundary for a prepared handle the caller leaves open.

Both methods now check the exact prepared delegate before inputs. This intentionally changes
former direct-run diagnostic precedence: closed prepared wins null/malformed/dead inputs without
borrowing; null or foreign prepared handles also precede list errors.

`InferenceSession` and `Engine.session` are removed. Migrate to prepare/run explicitly and
recompile; no replacement wrapper is introduced.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Preparation rejects a graph | The graph belongs to another Engine, Engine closure began, or cold preparation rejected its plan. | Compile and prepare with the same open Engine; inspect the prepare-time capability failure. |
| A run reports a missing, duplicate, or unexpected input | The supplied identities do not exactly match `graph.inputs()`. | Supply every required Tensor exactly once; list order is arbitrary. |
| A run rejects a descriptor or storage | The Tensor descriptor changed identity expectations, or storage type, capacity, lifetime, or thread accessibility is incompatible. | Use the exact compiled Tensor and attach compatible accessible live storage before the run. |
| A run reports `prepared execution is closed` | Prepared close won admission. | Prepare a new handle from the still-valid graph and Engine if more runs are needed. |
| Materialization rejects a publication | The occurrence belongs to another result, the result or Engine is closed, or the byte limit is too small. | Use an occurrence from the same open result and a sufficient checked bound. |

## Limitations

Prepared runs are synchronous and fixed to one compiled graph. Fresh identities are supported only
by runValues; neither method provides dynamic descriptors, asynchronous submission, implicit
materialization, persistence, generic device-result handles, or tuning controls. Explicit
CPU/Metal composition supports current single-owner execution and the bounded all-seven-carrier
rank-0..16 static canonical or positive-stride non-overlapping transfer domain; unsupported
capability or transfer combinations fail during compile or preparation rather than
falling back during a run.

## Related documentation

- [Whole-model and values-only checkpoint API](../api/model-checkpoint-api.md)
- [Describe, reconstruct and load a closed NN model](declarative-models.md)
- [Compile a graph](compiling-graphs.md)
- [Prepare execution](preparing-execution.md)
- [Runtime API](../api/runtime-api.md#current-ordinary-engine-boundary)
- [Public API status](../api/public-api.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
- [Lifecycle architecture](../architecture/lifecycle.md#current-public-engine-lifecycle)
