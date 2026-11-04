<a id="run-a-reusable-inference-session"></a>

# [Run](../glossary.md#term-run) a prepared [graph](../glossary.md#graph)

## Outcome

This guide compiles one fixed [inference](../glossary.md#term-inference) [graph](../glossary.md#graph), prepares one closeable [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution), and runs the same
[prepared execution](../glossary.md#prepared-execution--preparedexecution) repeatedly with current caller-owned input storage. Each [run](../glossary.md#term-run) returns typed,
ordered [publication](../glossary.md#term-publication) [metadata](../glossary.md#term-metadata) and supports explicit copying into [detached](../glossary.md#detached-model-results) [immutable](../glossary.md#term-immutable)
[`HostTensorValue`](../glossary.md#host-tensor-value--hosttensorvalue) objects.

**Inference** computes [predictions](../glossary.md#term-prediction) without an [optimizer](../glossary.md#term-optimizer) update. **[Eval mode](../glossary.md#inference-and-evaluation)** is a separate
[Module](../glossary.md#term-module) setting controlling mode-dependent forward behavior; it does not disable every [gradient](../glossary.md#term-gradient)
or [mean](../glossary.md#term-mean) prediction quality was measured. **[Capture](../glossary.md#term-capture-meanings)** records expression structure, **[compile](../glossary.md#compile)**
validates meaning and chooses [owners](../glossary.md#term-owner), **[prepare](../glossary.md#prepare)** builds reusable executable resources, and
**[binding](../glossary.md#term-binding)** supplies each run's actual input storage. A [cache](../glossary.md#term-cache) can retain setup, not the answer
for every future [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters).

Start with the [one-weight tutorial](../tutorials/index.md#the-experiment-in-concrete-terms) if these
distinctions are new. Its actual example remains [FLOAT32](../glossary.md#term-float32) `[2,1]`, input `[1,3]`, [target](../glossary.md#term-target) `[2,6]`,
one `[1,1]` [weight](../glossary.md#term-weight) initially zero, no [bias](../glossary.md#term-bias) and summed squared error. One [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) update at [learning
rate](../glossary.md#term-learning-rate) `0.05` changes weight `0` to `2`. The independent examples below retain their own existing
inputs and expected outputs; they are reference workflows, not replacements for that journey.

## [JVM](../glossary.md#term-java-virtual-machine-jvm) prerequisites

Use [JDK](../glossary.md#term-java-development-kit-jdk) 26 and the [runtime](../glossary.md#term-runtime) dependencies of the selected [API](../glossary.md#term-application-programming-interface-api). The default [scalar](../glossary.md#term-scalar) [CPU](../glossary.md#term-cpu) route does
not require `jdk.incubator.vector`. Explicit [Vector API](../glossary.md#term-vector-api) CPU computation requires
`--add-modules jdk.incubator.vector`.

Enable [FFM](../glossary.md#term-ffm) native access with `--enable-native-access=ALL-UNNAMED` for `Engine.standard()` as well
as explicit [Metal](../glossary.md#term-metal)/[OpenBLAS](../glossary.md#term-openblas) use. The standard CPU composition performs bounded optional OpenBLAS
library discovery when opened, even when the selected numerical work is portable Java. JDK 26
otherwise reports restricted-access warnings from that probe. The flag grants the required FFM
access; it does not choose a [backend](../glossary.md#backend), require an installed [BLAS](../glossary.md#term-blas) library or change the prepared route.

## Build a fluent [Model](../glossary.md#term-model)

Use `Model.builder()` when fixed-[shape](../glossary.md#shape) inputs should flow through dense, convolutional or
recurrent layers without a definition [callback](../glossary.md#term-callback). Static `rnn/gru/lstm` and explicit [runtime](../glossary.md#term-runtime)-length
`rnnScan/gruScan/lstmScan` are separate recipes. A build creates a private [evaluation](../glossary.md#term-evaluation)-mode Model
with one working [weight](../glossary.md#term-weight) set for compute and supported saves. Ordinary, fixed-[scan](../glossary.md#term-cumulative-scan) and general-scan
sources may configure first-order training when their actual source, [derivatives](../glossary.md#term-derivative) and [backend](../glossary.md#backend) qualify.
It creates no [Engine](../glossary.md#term-engine). Use Java 26 and `:extensions:nn`. Close the Model when finished; compute
results are [detached](../glossary.md#detached-model-results) values.

For example, a [multilayer perceptron](../glossary.md#term-multilayer-perceptron-mlp) ([MLP](../glossary.md#neural-network-families)) maps [FLOAT32](../glossary.md#term-float32) `[32,784]` to `[32,10]`:

```java
try (var model = Model.builder().input(DataType.FLOAT32, 32, 784)
        .seed(42).linear(128).relu().linear(10, false).build()) {
    // compute requires stored FLOAT32 [32,784] batches with the exact input descriptor.
}
```

Declare input once before steps: floating [dtype](../glossary.md#data-type--datatype), at least one [axis](../glossary.md#term-axis), every [extent](../glossary.md#term-extent) positive and
static. Its [prototype](../glossary.md#term-prototype) is storage-free, [canonical](../glossary.md#term-canonical-form) [contiguous](../glossary.md#term-contiguous), offset-zero, non-[view](../glossary.md#term-view-meanings) and
`requiresGrad=false`. Both compute and cold `forward` require that full [descriptor](../glossary.md#term-descriptor). The builder
is caller-coordinated and reusable: every build owns fresh children/state, and later builder edits
do not alter earlier Models. Build requires at least one step.

Automatic layers infer input [feature](../glossary.md#term-feature)/[channel](../glossary.md#term-channel) widths. They use existing [Glorot-uniform](../glossary.md#term-glorot-uniform) weights,
zero [bias](../glossary.md#term-bias) enabled by default, and a fresh L64X128MixRandom generator per layer. Default [seed](../glossary.md#term-seed) is 0;
the last seed set before build applies to the whole recipe. Appended step `i`, starting at zero,
uses `seed + 0x9E3779B97F4A7C15L * i` with Java long [wraparound](../glossary.md#term-wraparound). [Activations](../glossary.md#term-activation) and transforms count
as steps. Identical recipes, [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) and algorithms reproduce initialization bytes; this is not
[optimizer](../glossary.md#term-optimizer) or random-stream [checkpoint](../glossary.md#term-checkpoint) continuation.

A **dense** layer combines every input feature with its output weights. A **feature** is one
input measurement; a **[parameter](../glossary.md#term-parameter)** is stored state training may update. A **bias** is an
adjustable [constant](../glossary.md#term-constant) added to the weighted sum. An **activation**, such as RELU, transforms
the result; RELU keeps positive values and maps nonpositive values to zero.
**Initialization** chooses starting parameter values before learning. Glorot-uniform draws
weights from a bounded [uniform distribution](../glossary.md#term-uniform-distribution) with bounds based on the input and output widths.
A **seed** controls the random generator's starting sequence; reproducing that sequence is
different from resuming [optimizer history](../glossary.md#term-optimizer-history).

### Train and save private supplied [weights](../glossary.md#term-weight)

This complete example starts with supplied weight 2, predicts 6 on input 3, and trains toward 1.
`linear(weight)` disables [bias](../glossary.md#term-bias); `linear(weight,bias)` copies both. Supplied state must satisfy
[Linear](../glossary.md#term-linear-meanings)'s floating, positive-static, [gradient](../glossary.md#term-gradient)-eligible [schema](../glossary.md#term-schema) and match input [dtype](../glossary.md#data-type--datatype). It must be
stored [provenance](../glossary.md#provenance)-free state, not an expression to evaluate. Build copies each [Tensor](../glossary.md#tensor) [identity](../glossary.md#term-identity-meanings)
once before constructing expressions, preserving [aliases](../glossary.md#aliasing-and-arena); it never mutates caller state. Training
retains its separate rejection of [parameter](../glossary.md#term-parameter) aliases. Keep supplied storage live and exclude mutation
during build; a successful [Model](../glossary.md#term-model) does not retain that storage.

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

[Run](../glossary.md#term-run) with the [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [runtime](../glossary.md#term-runtime) [classpath](../glossary.md#term-classpath) and the [JVM](../glossary.md#term-java-virtual-machine-jvm) prerequisites above. The destination's parent
must exist. Output is `6.0`, `25.0`, `-2.0`: squared error `(2*3-1)^2` has weight gradient 30,
so [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) changes only the private weight to `2-0.1*30=-1`. Configure training again after load;
the file contains neither the [objective](../glossary.md#term-objective) nor [optimizer history](../glossary.md#term-optimizer-history).

### [Convolution](../glossary.md#term-channels-first-convolution-layers) and recurrent [axes](../glossary.md#term-axis)

These chains use the same imports and [Model](../glossary.md#term-model) [lifecycle](../glossary.md#lifecycle). A [convolutional neural network](../glossary.md#term-convolutional-neural-network-cnn) ([CNN](../glossary.md#neural-network-families)) takes
channels-first input; convolution defaults are [stride](../glossary.md#term-stride)/[dilation](../glossary.md#term-dilation)/groups 1, [padding](../glossary.md#padding) 0 and [bias](../glossary.md#term-bias) enabled.
Maximum [pooling](../glossary.md#term-pooling) uses stride equal to [kernel](../glossary.md#kernel), padding 0, dilation 1 and floor geometry:

**Convolution** applies a shared [weight](../glossary.md#term-weight) pattern, called a **kernel**, at multiple positions.
**Channels** identify separate [feature](../glossary.md#term-feature) maps, such as image color components. **Stride** is the
step between kernel positions, **padding** adds boundary positions, and **dilation** spaces out
positions within the kernel. **Groups** split the [channel](../glossary.md#term-channel) connections into independent sets.
**Pooling** summarizes local windows, for example by their maximum. **Floor geometry** rounds
the computed output [extent](../glossary.md#term-extent) downward. These terms describe axes and calculations, not a promise
that every geometry is supported.

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

`conv1d` uses [NCW](../glossary.md#term-ncw) ([batch](../glossary.md#term-batch-machine-learning-and-hyperparameters)/channels/width), [`conv2d`](../glossary.md#convolution--conv2d) [NCHW](../glossary.md#term-nchw) and [`conv3d`](../glossary.md#conv3d--ncdhw) [NCDHW](../glossary.md#conv3d--ncdhw).
Static `rnn/gru/lstm` steps use [TBF](../glossary.md#term-tbf) (time/batch/features), positive fixed time, and all-valid rows.
They return **every hidden step**, including [LSTM](../glossary.md#term-long-short-term-memory-lstm); there is no implicit final-step selection.
Each composition starts with zero hidden state (and zero cell state for LSTM). Cached compute/train
reuse fixed zero [constants](../glossary.md#term-constant), never [carry](../glossary.md#term-carry) state across batches, and do not rebuild the [graph](../glossary.md#graph) per call.
Static [unrolling](../glossary.md#term-unrolling) uses existing constituent-[operation](../glossary.md#operation) adjoints; it is not [scan](../glossary.md#term-cumulative-scan) [backpropagation through
time](../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../glossary.md#backpropagation-through-time-and-recurrent-state)), streaming state, or [runtime](../glossary.md#term-runtime)-length support. Conv3d uses shared grouped convolution
adjoints in the [qualified static CPU F32/F64 and Metal F32 domains](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu); [F32 glossary](../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../glossary.md#floating-point-precision-and-rounding); [Metal glossary](../glossary.md#term-metal))</span>.

`flatten()` requires [rank](../glossary.md#term-rank) at least two and preserves axis 0: `[axis0, product of other extents]`.
For CNNs axis 0 is batch; for TBF `[T,B,H]` it is time, so flatten gives `[T,B*H]`.
Use `lastTimeStep()` first when recurrent batch rows must remain separate.

A **recurrent** layer carries state from one time step to the next. [RNN](../glossary.md#term-recurrent-neural-network-rnn) expands to [recurrent
neural network](../glossary.md#neural-network-families); [GRU](../glossary.md#term-gated-recurrent-unit-gru) to [gated recurrent unit](../glossary.md#neural-network-families); LSTM to [long short-term memory](../glossary.md#neural-network-families). Their **hidden state**
is the working representation carried across steps; LSTM also carries **cell state**.
**Unrolling** describes a separate operation sequence for each fixed step.
**BPTT**, backpropagation through time, propagates sensitivities backward through those time
dependencies. An **[adjoint](../glossary.md#term-adjoint)** here is the backward [sensitivity](../glossary.md#term-sensitivity) calculation for an operation.
Keeping states within one sequence does not imply keeping them across separate Model calls.

Default Conv1d/2d/3d, MaxPool2d/flatten and static RNN/GRU/LSTM compositions save as their ordinary
[Tensor](../glossary.md#tensor) graphs, not layer recipes. Save before compute, close the original Model, cold-load through
`Model.load`, then compute fresh batches with the exact original [descriptor](../glossary.md#term-descriptor). For example, the CNN
above retains `[32,1,28,28]` input and `[32,10]` output, while the stacked LSTM retains `[20,32,8]`
input and `[32,10]` output.
Its zero initial-state and row-index constants are saved as reachable CONSTANT [leaves](../glossary.md#term-leaf), not layer
recipes. Configure a fresh [objective](../glossary.md#term-objective)/[optimizer](../glossary.md#term-optimizer) after load to train Conv1d/2d/3d or static recurrence
where current adjoints and [backend](../glossary.md#backend) predicates permit. Default biased Conv3d save/load/new-train
is qualified; arbitrary convolution [checkpoint](../glossary.md#term-checkpoint) geometry is not. Saving trained values still does not
[resume](../glossary.md#schema-serialization-and-checkpoint) [optimizer history](../glossary.md#term-optimizer-history), [RNG](../glossary.md#graph-rng-state--graphrngstate) or data progress. Any selected added operation chooses v3 even if
the graph also contains RELU/[TANH](../glossary.md#term-tanh)/[SIGMOID](../glossary.md#term-sigmoid).

Sources containing only ADD/MUL/NEG/[MATMUL](../glossary.md#matrix-multiplication--matmul), [rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) PERMUTE `[1,0]` and floating
RELU/TANH/SIGMOID retain byte-identical v1/v2 saves. Construction, cold forward and execution
remain independent of [serialization](../glossary.md#term-serialization) [admission](../glossary.md#term-admission). Other source variants still reject: convolution
without bias or with nondefault geometry, unsupported indexing axes and arbitrary recurrent bodies
are outside the selected [slice](../glossary.md#slice). Source
rejection occurs before file work, leaving absent [targets](../glossary.md#term-target) absent and existing targets unchanged.
A failed unsupported save does not freeze [Engine](../glossary.md#term-engine) selection, invalidate the Model, or disable
compute/training. [Persistence](../glossary.md#term-persistence) does not make
every builder [occurrence](../glossary.md#term-occurrence) executable: the selected Engine's [dtype](../glossary.md#data-type--datatype)/[layout](../glossary.md#layout)/[gradient](../glossary.md#term-gradient) capabilities apply.
See [builder API and lifecycle](../reference/nn/model-builder.md#fluent-model-builder)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>.

### [Run](../glossary.md#term-run) a [runtime](../glossary.md#term-runtime)-length [scan](../glossary.md#term-cumulative-scan) [Model](../glossary.md#term-model)

Use `buildWithLengths()` to build a Model with two ordered inputs: values and valid lengths.
[F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) `rnnScan` and qualified fixed recipes support first-order configured training.
The list-typed `Model<List<Tensor>,List<Tensor>>` keeps inputs explicit, while `compute(values,
validLengths)` returns the sole [detached](../glossary.md#detached-model-results) result. No prior forward is required, before or after load;
this workflow has no Data [dependency](../glossary.md#term-dependency). The example below retains fixed [GRU](../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../glossary.md#term-long-short-term-memory-lstm) [inference](../glossary.md#term-inference).

A **scan** repeatedly applies a body calculation while passing state, called **[carry](../glossary.md#term-carry)**, between
iterations. **Runtime lengths** say how many leading time steps of each [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) row are valid.
**[Padding](../glossary.md#padding)** fills the remaining positions so rows can share one rectangular [Tensor](../glossary.md#tensor);
**masking** excludes positions according to a stated rule. A scan that skips its own padding
does not automatically make later layers or their [loss](../glossary.md#loss) ignore padding.

The example below uses the same Java 26/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[CPU](../glossary.md#term-cpu) prerequisites as the fluent examples
above. It supplies [canonical](../glossary.md#term-canonical-form) [FLOAT32](../glossary.md#term-float32) `[4,3,2]` values and non-[gradient](../glossary.md#term-gradient) [INT64](../glossary.md#data-type--datatype) `[3]` lengths
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
before the ordinary [Linear](../glossary.md#term-linear-meanings) head produces `[3,5]`. It selects states after four, two and zero
transitions, not dense time index 3. The zero-length hidden row is zero. This example's fresh head
also has zero [bias](../glossary.md#term-bias), but an ordinary supplied nonzero head bias can make that row's scores nonzero.
The detached original scores remain readable after the original Model closes. The file contains
the [graph](../glossary.md#graph) and private state, not input batches or a [prepared executable](../glossary.md#prepared-executable--preparedexecutable); input positions survive
without inferred length-role [metadata](../glossary.md#term-metadata) or a new format. Use `loadGraph`, not single-boundary `load`.

Each scan starts from independent zero hidden state (and cell state for LSTM), uses biased FORWARD
transitions and existing step [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings), and shares one canonical INT64 `[B]` lengths input. No state
carries between calls. The first scan requires positive static time/batch/[features](../glossary.md#term-feature) ([TBF](../glossary.md#term-tbf)); an ordinary
prefix establishes that interpretation. Later scans require the same T/B and infer only features.
`finalHidden()` must immediately follow a scan, not static recurrence, a transform or another
selection, and counts as an appended [seed](../glossary.md#term-seed) step. Append a classifier after it. Advanced graph/list
boundaries remain available for explicit states, finalCell, reverse/bias-free scans and zero T/B.

Only scans skip their own padded transitions. Read these ordinary steps with their existing dense
meaning; there is no model-wide [mask](../glossary.md#term-boolean-mask) or original-padding-independence guarantee:

| Step | Meaning alongside scans |
|---|---|
| Linear, RELU/[TANH](../glossary.md#term-tanh)/[SIGMOID](../glossary.md#term-sigmoid) | Operate at every [coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix); bias can change padded zeros and sigmoid(0) is 0.5. |
| [Convolution](../glossary.md#term-channels-first-convolution-layers)/[pooling](../glossary.md#term-pooling) | Keep channels-first geometry and may mix [axes](../glossary.md#term-axis) interpreted elsewhere as time/batch; they are not sequence-aware. |
| Flatten | Preserves axis zero: `[T,B,H]` becomes `[T,B*H]`, not `[B,T*H]`. A subsequent scan requires [rank](../glossary.md#term-rank) three again. |
| Static rnn/gru/lstm | Ignore runtime lengths and unroll every time step, unchanged. |
| lastTimeStep | Still selects `T-1`, not each row's final hidden state; ordinary intervening steps may change padded zeros. |
| Another scan | Shares lengths and ignores its own input suffix, including suffix values changed by ordinary intervening steps. |

F32/F64 [RNN](../glossary.md#term-recurrent-neural-network-rnn) now uses [general Scan](../glossary.md#general-differentiable-scan) with canonical source outputs and qualified first-order training.
Its CPU actual reads require dense zero-offset geometry (including dense [views](../glossary.md#term-view-meanings)) or existing
proved [scalar](../glossary.md#term-scalar)-broadcast exceptions. Fixed GRU/LSTM and low RNN retain their separately declared
resolved nonnegative-[stride](../glossary.md#term-stride)/offset read domain. Each [occurrence](../glossary.md#term-occurrence) still needs [backend](../glossary.md#backend) [qualification](../glossary.md#term-qualification);
there is no implicit conversion, [layout](../glossary.md#layout) copy or CPU fallback for unsupported composition.

| Symptom | Cause and correction |
|---|---|
| `build()` rejects a scan chain | Finish with `buildWithLengths()`; that terminal rejects chains with no scan before state acquisition. |
| `finalHidden()` rejects | Place it immediately after a new scan recipe, before any classifier or transform. |
| Later scan rejects geometry | Preserve exact time/batch axes; do not reshape lengths or guess a different batch axis. |
| Compute rejects length input | Match canonical non-gradient INT64 `[B]` and all declared [descriptor](../glossary.md#term-descriptor) fields. Values must lie in `[0,T]`; repair out-of-range values and reuse the Model. |
| Configure/train rejects | Check the selected backend's actual forward/[AD](../glossary.md#autograd) closure and use a scalar gradient-eligible F32/F64 [objective](../glossary.md#term-objective). Low objectives need an explicit cast; fixed versus general recipe [identity](../glossary.md#term-identity-meanings) is preserved rather than bypassed. |

See [builder API details](../reference/nn/model-builder.md#fluent-model-builder)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span> and
[training admission](../reference/training/sessions.md#owned-model-training-convenience)<span data-glossary-navigation="true"> ([admission glossary](../glossary.md#term-admission))</span>.

### Train a [runtime](../glossary.md#term-runtime)-length [RNN](../glossary.md#term-recurrent-neural-network-rnn)

[F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) `rnnScan` composes the existing RNN cell equation through [general Scan](../glossary.md#general-differentiable-scan). Configure once,
then supply fresh values and [INT64](../glossary.md#data-type--datatype) lengths at the same static [signature](../glossary.md#term-signature). Save before compute or
after training; close and `loadGraph`, then configure a new [objective](../glossary.md#term-objective) and [optimizer](../glossary.md#term-optimizer) on restored
[weights](../glossary.md#term-weight). This is not exact optimizer continuation. Old v3 fixed-RNN files keep their fixed recipe;
fresh first-order configured training is available when that recipe's [backend](../glossary.md#backend)/[AD](../glossary.md#autograd) closure qualifies.

The example below uses the same [JVM](../glossary.md#term-java-virtual-machine-jvm)/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) prerequisites and an existing destination directory.
Inactive sequence suffixes contain [NaN](../glossary.md#term-nan) deliberately: recurrence skips them rather than masking
an already-computed value. [Padding](../glossary.md#padding) does not automatically [mask](../glossary.md#term-boolean-mask) the classifier [loss](../glossary.md#loss).

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
initial state and produce zero recurrent-[parameter](../glossary.md#term-parameter) [gradients](../glossary.md#term-gradient); a downstream head can still train.
Invalid lengths reject before parameter/optimizer mutation; repair values and reuse the session.
Direct `RnnSequence.forwardGeneralScan` additionally exposes reverse traversal, [bias](../glossary.md#term-bias)-free cells,
explicit initial state and zero T/B. General final hidden is [carry](../glossary.md#term-carry) port 0 and dense output emission
port 1; retained fixed result ordering is unchanged. See the [NN0033 Result](../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result)
for independent forward/[BPTT](../glossary.md#term-backpropagation-through-time-bptt) and owned [lifecycle](../glossary.md#lifecycle) evidence in both [CPU](../glossary.md#term-cpu) [dtypes](../glossary.md#data-type--datatype).


### Compose explicit-state scans

Fixed scans remain separate from static sequences. For advanced cold [persistence](../glossary.md#term-persistence), use the
existing list `Model.graph`/`Model.loadGraph` boundary with [INT64](../glossary.md#data-type--datatype) [runtime](../glossary.md#term-runtime) lengths
as an input, alongside sequence and any external initial states. The cold roundtrip
recreates storage-free [prototypes](../glossary.md#term-prototype) and rebinds fresh exact-[descriptor](../glossary.md#term-descriptor) inputs without reading length
values. It retains all [canonical](../glossary.md#term-canonical-form) ports of one [scan](../glossary.md#term-cumulative-scan) even when only finalCell is returned.
The three one-directional [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) Sequences also offer additive `forwardScan` methods using the same
owned cell [parameters](../glossary.md#term-parameter). For example, this complete composition fragment creates a reverse [LSTM](../glossary.md#term-long-short-term-memory-lstm)
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
row 0 at times 2, 1, 0 and row 1 only at time 0, retaining original output [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix). With these
zero inputs/states and zero [bias](../glossary.md#term-bias), every computed output is zero; [padding](../glossary.md#padding) at row 1 times 1 and 2
is specifically [positive zero](../glossary.md#nan-infinity-and-signed-zero). Static `forward(..., long[])` instead returns a compact output
list and keeps its existing [provenance](../glossary.md#provenance)/training behavior. `forwardScan` creates one [producer](../glossary.md#term-producer) and
binds an automatic cell even for empty time/[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) or all-zero runtime lengths; static [unrolling](../glossary.md#term-unrolling)
with no represented step leaves it unbound. All floating descriptors must match the configured
type exactly; a mismatch fails before initialization. Neither path infers lengths from padding.

Advanced callers separately use ordinary [Engine](../glossary.md#term-engine) [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare)/[run](../glossary.md#term-run) or runValues for fully static
[homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) [CPU](../glossary.md#term-cpu) scans, with resolved nonnegative read [layouts](../glossary.md#layout),
INT64 lengths, explicit states and packed parameters. Fresh input storage must contain its full
[referenced span](../glossary.md#referenced-element-span), including any offset prefix and strided gaps, not only its logical [element](../glossary.md#term-element) count.
Prepare reads no length values;
each run checks all lengths in `[0,T]` before writing scan results. Repair invalid lengths and
reuse the preparation with the same descriptors. Qualified first-order [backpropagation through time](../glossary.md#term-backpropagation-through-time-bptt)
uses [Compiler](../glossary.md#term-compiler)'s existing fixed-cell/Scan [derivative](../glossary.md#term-derivative) machinery and the same Engine/Training [owners](../glossary.md#term-owner);
higher-order recurrence and serialized prepared loops remain unsupported. Scan [gradient](../glossary.md#term-gradient) flags remain
intact in eval. [Owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) compute and configured fixed/general training use these same qualified
paths; loading still requires a fresh [objective](../glossary.md#term-objective) and [optimizer](../glossary.md#term-optimizer), not restored history. Compile selected `outputs()`,
`finalHidden()` and `finalCell()` [Tensors](../glossary.md#tensor) and bind every compiled input slot, including [weights](../glossary.md#term-weight)
and states. Follow [fresh-value binding](#run-fresh-values-in-compiled-order)<span data-glossary-navigation="true"> ([fresh-value binding glossary](../glossary.md#fresh-value-binding--enginerunvalues))</span>, replacing lengths
with `[0,2]` under the same descriptors to reuse the preparation; row 0 then preserves its initial
states semantically, not by Java Tensor [identity](../glossary.md#term-identity-meanings). Lengths `[4,1]` fail for `T=3`; repair and reuse.
For symbolic continuation, pass both final-state Tensors to another LSTM `forwardScan`; across
separate runs, explicitly supply materialized states matching the next [graph](../glossary.md#graph)'s descriptors.
Existing builder `rnn/gru/lstm` and bidirectional containers remain static; explicit builder scan
recipes opt into runtime lengths. Data integration is not part of this [API](../glossary.md#term-application-programming-interface-api).
See the [cold scan scenario](../reference/checkpoint/graph-formats.md#selected-cnn-and-recurrent-persistence)
and [CPU execution scope](../backend-guide/cpu-backend.md#fixed-recurrent-forward-execution)<span data-glossary-navigation="true"> ([scope glossary](../glossary.md#term-scope))</span>.
Low fixed-scan source constructors, storage and cold files remain supported in their declared
source domain. CPU [BFLOAT16](../glossary.md#term-bfloat16), [FLOAT16](../glossary.md#term-float16) and both [FP8](../glossary.md#term-fp8) types use the existing fixed-scan [owner](../glossary.md#term-owner),
FLOAT32 working sites and declared state [narrowing](../glossary.md#term-narrowing); geometry and actual derivative guards remain.

## Ordinary [CPU](../glossary.md#term-cpu) execution with changing [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) size

Use the ordinary [Engine](../glossary.md#term-engine) [API](../glossary.md#term-application-programming-interface-api) when one prepared [FLOAT32](../glossary.md#term-float32) [matrix](../glossary.md#term-matrix-meanings)-[loss](../glossary.md#loss) [graph](../glossary.md#graph) must accept different
batch sizes. This does not change [NN Model](../glossary.md#model-module-buffer-and-scan-distinctions) or [Training session](../glossary.md#training-session--trainingsession) [signatures](../glossary.md#term-signature), which remain static.
With [JDK](../glossary.md#term-java-development-kit-jdk) 26 and the Engine [module](../glossary.md#term-module) available, this complete example compiles a [scalar](../glossary.md#term-scalar) loss and
actual first-order [gradients](../glossary.md#term-gradient) once, then supplies fresh [canonical](../glossary.md#term-canonical-form) values in compiled input order:

**Symbolic** batch size means the [descriptor](../glossary.md#term-descriptor) names a [dimension](../glossary.md#dimension) `B` without fixing its numeric
value until a [run](../glossary.md#term-run) supplies concrete data. Time and [feature](../glossary.md#term-feature) widths can still be fixed.
This is different from [broadcasting](../glossary.md#broadcasting), which matches compatible operand [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix); use the
existing [broadcast visual](../explanation/tensors-and-gradients.md). It also differs from
rebuilding the owned tutorial [Model](../glossary.md#term-model), whose complete `[2,1]` signature stays fixed.

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
scalar-one [seed](../glossary.md#term-seed) differentiates only the published scalar loss. The result order is loss, dX, dW:
for positive B, loss is 4, each dX is approximately `1/B`, and each dW is approximately `1/3` under
FLOAT32 [rounding](../glossary.md#term-rounding). Their [shapes](../glossary.md#shape) are scalar, `[B,4]`, `[4,3]`; the [parameter](../glossary.md#term-parameter) gradient is not a bank
of B parameter copies. Printed losses are `4.0, 4.0, 4.0, 4.0, NaN`. At B=0, dX is empty and dW
contains positive zeros. Slots, route and generated code are reused; results own independent
writable storage and remain materializable after prepared close while Engine remains open.

A wrong fixed feature count, gradient flag, missing/noncanonical/offset dynamic input [layout](../glossary.md#layout),
inconsistent B, [overflow](../glossary.md#term-overflow) or unusable storage rejects before [borrowing](../glossary.md#term-borrowing). Dynamic inputs require
present canonical zero-offset layouts; dense read [views](../glossary.md#term-view-meanings) qualify. Static slots still require exact
descriptors. For qualified [Scan](../glossary.md#term-cumulative-scan) execution, use the example below. Neither example enables
[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training reuse, symbolic cold/wire loading, native/mixed-[owner](../glossary.md#term-owner)/[vector](../glossary.md#term-vector)/parallel execution or
tuning. See the [CPU slice](../backend-guide/cpu-backend.md#ordinary-named-batch-cpu-execution)<span data-glossary-navigation="true"> ([slice glossary](../glossary.md#slice))</span>.

## [Symbolic Scan](../glossary.md#symbolic-scan) with changing [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) size

Use the same public [Engine](../glossary.md#term-engine) [lifecycle](../glossary.md#lifecycle) for an all-valid [Scan](../glossary.md#term-cumulative-scan): every time step participates, time
and [feature](../glossary.md#term-software-feature) widths stay fixed, and one named batch B changes between calls. [JDK](../glossary.md#term-java-development-kit-jdk) 26 and Engine/[CPU](../glossary.md#term-cpu)
[modules](../glossary.md#term-module) are prerequisites. Current ordinary CPU [general Scan](../glossary.md#general-differentiable-scan) qualifies [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) forward
and actual selected VJPs under these ordinary single-B, fixed-T/features guards. The FLOAT32
example below retains the historical generated [scalar](../glossary.md#term-scalar), single-range Prepare0014 layer; its
[task](../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md) retains that evidence,
not a global FLOAT32-only restriction or broader ordinary symbolic [F64](../glossary.md#floating-point-precision-and-rounding) [MATMUL](../glossary.md#matrix-multiplication--matmul) [admission](../glossary.md#term-admission).

This complete example uses three steps of `next = h + step`, width three, zero initial [carry](../glossary.md#term-carry),
one-valued sequence and joint final/emission one [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings). It compiles the forward outputs and
actual [Compiler](../glossary.md#term-compiler)-generated [gradients](../glossary.md#term-gradient) once, then binds fresh values in final compiled input order.

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
emissions, dInitial and dX. The final [seed](../glossary.md#term-seed) and three emission seeds each contribute once, giving
dInitial=4 and dX time frames 4, 3 and 2. This arithmetic walkthrough is not a performance or
nonuniform [conformance](../glossary.md#term-conformance) [oracle](../glossary.md#term-test-oracle); focused tests also compare actual reverse with ordinary [unrolling](../glossary.md#term-unrolling).

Symbolic templates have no storage or resolved numeric [layout](../glossary.md#layout). Actual inputs/seeds have [canonical](../glossary.md#term-canonical-form)
static [descriptors](../glossary.md#term-descriptor) and unchanged gradient flags; caller storage stays [borrowed](../glossary.md#ownership-borrowing-and-lease) until result close.
Code/routes/slots are reused, but writable outputs, working arrays and results are fresh. B=0 is
valid for this algebra and gives empty row outputs; the example omits it because it prints a cell.
Other bodies can still require nonempty shared work at B=0. Changing time requires a separate
[compile](../glossary.md#compile)/[prepare](../glossary.md#prepare), including T=0. Inconsistent B, changed fixed widths, unusable storage or unsupported
layouts reject. Current all-valid symbolic general Scan composes ordinary CPU-supported floating
types, including all four low types, under the same guards; adding lengths selects the separate
length-aware rules below. Neither route grants arbitrary symbolic geometry, [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training reuse,
symbolic cold/copy/wire, native/mixed-[owner](../glossary.md#term-owner) execution or a geometry [memo](../glossary.md#cache-and-memoization). See the
[qualified CPU domain](../backend-guide/cpu-backend.md#symbolic-scan-cpu-execution).

## Symbolic-length [Scan](../glossary.md#term-cumulative-scan) with changing [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) size

For variable active prefixes, pass non-[gradient](../glossary.md#term-gradient) [INT64](../glossary.md#data-type--datatype) `[B]` lengths to the existing `Scan.apply`
overload. Keep time/[features](../glossary.md#term-software-feature) fixed and use a qualified floating body under its per-role [dtype](../glossary.md#data-type--datatype) guards.
The same [Engine](../glossary.md#term-engine)/[CPU](../glossary.md#term-cpu) and [JDK](../glossary.md#term-java-development-kit-jdk) 26 prerequisites apply. Prepare0015 implements this path; owning
tests and independent code/structural review passed, with root [validation](../glossary.md#term-validation) and the sole bounded probe completed
(evidence and limits:
[Prepare0015 Result](../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. No new Engine [API](../glossary.md#term-application-programming-interface-api) or per-batch preparation is needed.

This complete [FLOAT64](../glossary.md#term-float64) example uses three steps of `next = h + step`, width two, and joint final/
emission one [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings). The original [carry](../glossary.md#term-carry) and every original step are ROW(0): their leading [coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix)
is the same named B. Each call supplies new lengths and static storage in compiled input order.

A **prefix** is the leading valid portion of each row's time sequence.
**ROW([axis](../glossary.md#term-axis))** records a value whose named axis follows batch rows; **SHARED** denotes a value
not split by that row axis. **[Primal](../glossary.md#term-primal)** means the original forward calculation, while
**[adjoint](../glossary.md#term-adjoint)** refers to its reverse [sensitivity](../glossary.md#term-sensitivity) calculation. A **[seed](../glossary.md#term-seed)** in this context supplies
output sensitivities for differentiation, not a random-generator starting value.

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
cell from them. The carry [descriptor](../glossary.md#term-descriptor) is `[B,2]` resolved to each actual batch. Forward and reverse
reuse the same code/routes/slots while each [run](../glossary.md#term-run) owns fresh outputs, resources and [mutable](../glossary.md#term-mutable) arrays.

Use [FLOAT32](../glossary.md#term-float32) descriptors and `float[]`/`getFloat()` for the corresponding FLOAT32 example; do not
change just the descriptor while leaving double storage. For produced lengths, construct a
non-gradient floating `[B]` template, pass its `.cast(DataType.INT64)` result to `Scan.apply`, and
bind the floating [producer](../glossary.md#term-producer) input instead of the derived INT64 value. Casting follows ordinary
cast semantics; the resulting lengths must still lie in `[0,3]`.

Lengths are checked after producers and before any Scan-local write, even when time or [features](../glossary.md#term-feature)
are zero. An invalid length can be repaired for a later run on the same preparation; producer writes
are not rolled back. [Padding](../glossary.md#padding) is not read in primal or adjoint work; an inactive row preserves raw
initial carry/final-seed bits. Lengths do not make `mean(finalCarry)` divide by the active count:
its denominator remains dense B×width, and an empty [mean](../glossary.md#term-mean) is [NaN](../glossary.md#term-nan). Changing T or feature widths needs
a different [compilation](../glossary.md#compiler-engine-and-execution-stage-names)/preparation. CPU low types reuse the ordinary floating/actual [derivative](../glossary.md#term-derivative)
[owners](../glossary.md#term-owner) here; this does not grant arbitrary geometry, higher-order [AD](../glossary.md#autograd), native/mixed-[owner](../glossary.md#term-owner) execution,
symbolic cold/copy/wire, [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training reuse or a geometry [memo](../glossary.md#cache-and-memoization). See the
[exact supported occurrences](../backend-guide/cpu-backend.md#symbolic-length-scan-cpu-execution).

## Owned static [Model](../glossary.md#term-model) workflow

For an application-built and file-restored model with the same compute/train/save [API](../glossary.md#term-application-programming-interface-api), opt into
an **owned source**: ordinary static [Tensor](../glossary.md#tensor) [provenance](../glossary.md#provenance) over private [weight](../glossary.md#term-weight) values. The builder above
retains layer composition; the definition/[graph](../glossary.md#graph)/load paths below reconstruct the bounded source.
Use Java 26 and an explicit `:extensions:nn` [dependency](../glossary.md#term-dependency) with the [JVM](../glossary.md#term-java-virtual-machine-jvm) prerequisites above.
No caller [Arena](../glossary.md#term-arena) is required for this workflow.
Definition/graph acquisition uses the bounded
[cold source slice](../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([slice glossary](../glossary.md#slice))</span>, including its
general-body exception, not all [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layers or arbitrary [PURE](../glossary.md#term-pure) bodies. Save/load uses the
[body-free persistence slice](../reference/checkpoint/graph-formats.md#selected-cnn-and-recurrent-persistence)
or [bounded v4 body persistence](../reference/checkpoint/graph-formats.md#bounded-general-scan-persistence),
through the same Model API.
Execution and training support are independent of raw graph [persistence](../glossary.md#term-persistence)'s nine data types.
[FP8](../glossary.md#term-fp8) graph sources select v7; NN values/definitions select their separate v2 formats only when
FP8 is present. Old-only sources retain their original smallest sufficient formats and bytes.
Files retain logical low storage, not [FLOAT32](../glossary.md#term-float32) [optimizer](../glossary.md#term-optimizer) masters. Loading and opening a new
[Training session](../glossary.md#training-session--trainingsession) starts fresh [optimizer state](../glossary.md#term-optimizer-state); this is not optimizer continuation.

This complete example learns one FLOAT32 weight, starting at 2. Both [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) and [target](../glossary.md#term-target) are [1,1].
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

The squared error is `(2*3-1)^2 = 25`; [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) applies [gradient](../glossary.md#term-gradient) 30, so weight becomes `2-0.1*30 = -1`.
Acquisition copied external values once and installed final private leaves into the adopted wrappers.
Definition and [prototype](../glossary.md#term-prototype) forward each ran once in [eval mode](../glossary.md#inference-and-evaluation); their callbacks are discarded. Explicit
`forward` is cold symbolic composition, not a batch execution path. Construction failure cannot
undo arbitrary [callback](../glossary.md#term-callback) effects. Do not mutate adopted child [modules](../glossary.md#term-module) after construction.

The [signature](../glossary.md#term-signature) includes [dtype](../glossary.md#data-type--datatype), static [Shape](../glossary.md#shape), [layout](../glossary.md#layout) presence/geometry and requiresGrad. Even unused
declared slots must match and have usable storage. A [2,1] batch rejects rather than recompiling.
`configureTraining` preserves mode, creates one native [parameter](../glossary.md#term-parameter) working set when needed, and
succeeds once; compute can [prepare](../glossary.md#prepare) again after this explicit transition, not after each update.
Configure training anew after load: objectives, [optimizer history](../glossary.md#term-optimizer-history), [RNG](../glossary.md#graph-rng-state--graphrngstate) and data cursors are not saved.

For multiple Tensor inputs/outputs, use `Model.graph(inputs, outputs, namedLeaves)` and list
compute/train methods; name every reachable state [leaf](../glossary.md#term-leaf) with its explicit PARAMETER, [BUFFER](../glossary.md#term-buffer) or
[CONSTANT](../glossary.md#term-constant) role in [canonical](../glossary.md#term-canonical-form) Module traversal order. `loadGraph` restores that list boundary;
single `load` rejects it with a hint. List compute accepts an aggregate `maximumTotalBytes`, checks
all output sizes first, closes its result and returns only a complete [detached](../glossary.md#detached-model-results) list.

Select a [borrowed](../glossary.md#ownership-borrowing-and-lease) [Engine](../glossary.md#term-engine) before the first valid compute/configure setup; `withEngine` returns the
same model. That selection freezes even if setup fails, and a closed selected Engine cannot be
replaced. Without a selection, only execution opens a fresh model-owned [CPU](../glossary.md#term-cpu) Engine. Close never
closes a borrowed Engine; it waits for active work and closes session/preparation before native
storage and an owned Engine, attempting all cleanup and replaying retained failures.

**Owned** state has one responsible [lifecycle](../glossary.md#lifecycle) [owner](../glossary.md#term-owner); **borrowed** resources must remain usable
under another owner's [lifetime](../glossary.md#term-lifetime). **Detached** results contain independent copied bytes, so
retaining them does not retain a right to [run](../glossary.md#term-run) a closed Model. Read the
[ownership visual](../explanation/ownership.md#a-result-is-not-always-an-independent-value)<span data-glossary-navigation="true"> ([ownership glossary](../glossary.md#term-ownership-meanings))</span>
alongside the closure sequence above. A Java reference to a Tensor alone cannot keep a closed
native Arena's memory usable.

Fixed and mixed recurrent sources use this same setup ordering. Unsupported [backend](../glossary.md#backend) or [derivative](../glossary.md#term-derivative)
closure is not replaced by a separate [inference](../glossary.md#term-inference)-only NN [guard](../glossary.md#term-guard) or an implicit recipe conversion.

| Symptom | Cause and correction |
|---|---|
| Unsupported tensor capability | Generic `Model.define`, custom subclasses and DeclarativeModel retain old forward behavior. Explicitly acquire a supported graph boundary. |
| Busy or closed/closing | Another high-level [operation](../glossary.md#operation) or close owns [admission](../glossary.md#term-admission). Serialize calls; keep borrowed Engine open through Model use. |
| Source changed | Mode, paths/wrappers, Tensor or storage associations changed. Construct a replacement Model; restoring a [binding](../glossary.md#term-binding) does not revive an invalid capability. |
| Native training admission rejects | Current Training rejects unresolved/empty/nonfinite/aliased parameters. Supply supported state before acquisition; there is no implicit conversion or [alias](../glossary.md#aliasing-and-arena) splitting. |
| Unsupported source operation | Graph/definition acquisition and general-body persistence require the bounded cold inventory, not arbitrary PURE operations or attributes. Body-only signatures do not widen outer admission. Builder execution is independent of save admission. |

Short save/load forms use 256 MiB file/aggregate, 4 MiB [manifest](../glossary.md#term-manifest), 16384 entries, 64-byte keys and
64 MiB [payload](../glossary.md#artifact-envelope-and-atomic-publication) ceilings; explicit `ArtifactLimits` overloads remain. Old `model.v1` and
`whole-model.v1` files still use `nn.checkpoint.ModelCheckpoint`, not `Model.load`.
Zero-[node](../glossary.md#node) source may save/load but retains Engine's empty-plan rejection. Model/Training [autotuning](../glossary.md#autotuning--model-autotuning)
is not integrated; advanced `Engine.prepareTuned` remains separate. See the
[API contracts](../reference/nn/owned-model.md#owned-tensor-model-capability).

### Train an all-valid [general scan](../glossary.md#general-differentiable-scan)

Use `Model.graph` or `defineTensor` for a fixed-count general scan, not a new ModelBuilder recipe.
**All-valid** means every static time step participates; there is no [runtime](../glossary.md#term-runtime) length or [padding mask](../glossary.md#masks-padding-and-eager-evaluation).
Use the Java 26/[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[CPU](../glossary.md#term-cpu) setup above. Current general Scan composes every ordinary CPU-supported
[PURE](../glossary.md#term-pure) unit under ordinary [dtype](../glossary.md#data-type--datatype)/[layout](../glossary.md#layout)/value and [derivative](../glossary.md#term-derivative) guards, with selected actual
floating first-order [AD](../glossary.md#autograd), including the four low types; forward and actual reverse qualify independently. All-valid
bodies use WholeTensor. Families without backward retain that limit. Cold acquisition is
separate: numerical CPU parity for [BFLOAT16](../glossary.md#term-bfloat16), [FLOAT16](../glossary.md#term-float16) and both [OFP8](../glossary.md#term-ofp8) types does not itself widen
source acquisition. It requires the exact
[cold constructor inventory](../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction),
not merely PURE operations. CPU low execution uses [FLOAT32](../glossary.md#term-float32) working values and final logical
[narrowing](../glossary.md#term-narrowing), with existing family, geometry and actual derivative guards. It does not widen NN
recipes or [Metal](../glossary.md#term-metal) [Scan](../glossary.md#term-cumulative-scan) [admission](../glossary.md#term-admission); Metal has no [FP8](../glossary.md#term-fp8) support.
Length-aware body values may be proved ROW([axis](../glossary.md#term-axis)) values or arbitrary-[shape](../glossary.md#shape) SHARED [tensors](../glossary.md#tensor).
Lengths is a separate final non-[gradient](../glossary.md#term-gradient) [INT64](../glossary.md#data-type--datatype) `[B]` control with no body formal, not a [scalar](../glossary.md#term-scalar)
alternative. Original AD carries/sequences and emission boundaries remain ROW(0); intermediate
and history ROW axes/ranks follow their proofs. Supported row-mixing [materialization](../glossary.md#materialization) requires
exactly the same length values, not Tensor [identity](../glossary.md#term-identity-meanings). Structural absence differs from a present
all-zero [seed](../glossary.md#term-seed); cross-Scan materialization is explicit.

A body **formal** is a placeholder local to the scan; an **actual** is the outer Tensor bound
to it. An **[invariant](../glossary.md#term-invariant)** is an explicit input available unchanged across steps.
An **emission** is a per-step output, distinct from the final carried state.
A **[residual](../glossary.md#scan-residual-port-and-history-demand)** is retained intermediate information needed by later work such as differentiation.
An undeclared **free [leaf](../glossary.md#term-leaf)** is an outside [dependency](../glossary.md#term-dependency) used without an explicit body [binding](../glossary.md#term-binding);
making invariant bindings explicit prevents that hidden dependency.

This complete FLOAT32 example acquires [weight](../glossary.md#term-weight) `0.5` privately once. The body computes
`next = carry * weight + step` and emits the pre-step [carry](../glossary.md#term-carry). Input `[1,2,3]`, initial carry `0`
and forward traversal produce final carry `4.25` and original-time emissions `[0,1,2.5]`.
It saves before compute and after training, then closes and reloads the trained source. [Run](../glossary.md#term-run) in a
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
be an undeclared free leaf and reject. The [CONSTANT](../glossary.md#term-constant) initial carry stays outside `stateDictionary`.
The [scalar objective](../glossary.md#loss) consumes both output roles: `4.25² + 0² + 1² + 2.5² = 25.3125`.
Its weight gradient is `30.5`, so [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) changes the owned weight to approximately `0.195`; the caller's
weight remains `0.5`. Later compute sees that same updated native store. Pass fresh exact-[descriptor](../glossary.md#term-descriptor)
[batches](../glossary.md#term-batch-machine-learning-and-hyperparameters)/[targets](../glossary.md#term-target) to further compute/train calls; neither cold `forward` nor reconstruction is needed.
There is one [inference](../glossary.md#term-inference) preparation per setup and one configured [training session](../glossary.md#training-session--trainingsession), not one per batch.

At T=0 the body does not execute; Training still requires connected, nonempty [parameters](../glossary.md#term-parameter). Residual
ports may be cold roots, but residual-root differentiation rejects. Qualified fixed and mixed
fixed/general sources use the same configured session and actual [Compiler](../glossary.md#term-compiler) AD.
This all-valid source saves as v4; the separate static length intersection uses v5.
Low execution still requires a supporting [backend](../glossary.md#backend) and an explicit [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) training [objective](../glossary.md#term-objective).

The two saved sources retain the initial and trained raw weights respectively, with one scoped
body rather than three [unrolled](../glossary.md#backpropagation-through-time-and-recurrent-state) steps. `Model.loadGraph` adopts final [heap](../glossary.md#term-heap) values and reconstructs
ordinary [provenance](../glossary.md#provenance) without creating an [Engine](../glossary.md#term-engine). The restored compute above uses the trained weight
of approximately `0.195`; its new training session starts without the previous [optimizer history](../glossary.md#term-optimizer-history).
No input batch or residual history is saved.
Single-boundary sources use `Model.load` instead. Execution remains limited to qualified occurrences.
Focused tests cover this save/load workflow; main accepted final [module](../glossary.md#term-module)/[Javadoc](../glossary.md#term-javadoc), integration,
architecture and configured root gates in CP0013. This capability does not migrate `.rnn` or
reinterpret an old file. See [bounded v4 persistence](../reference/checkpoint/graph-formats.md#bounded-general-scan-persistence)<span data-glossary-navigation="true"> ([persistence glossary](../glossary.md#term-persistence))</span>.

### Two [Linear](../glossary.md#term-linear-meanings) layers with an [activation](../glossary.md#term-activation)

A [multilayer perceptron](../glossary.md#term-multilayer-perceptron-mlp) ([MLP](../glossary.md#neural-network-families)) uses two Linear layers with an intervening activation. Add these
methods to `OwnedModelExample` above and call `activationMlp()` from `main`. This [FLOAT32](../glossary.md#term-float32) example
uses the same [weights](../glossary.md#term-weight), [shapes](../glossary.md#shape), [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters) and squared-error [objective](../glossary.md#term-objective) as the parameterized
`OwnedModelWorkflowIntegrationTest`. It saves before compute, trains private weights and loads
the trained [model](../glossary.md#term-model); the external leaves remain unchanged.

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
the second. RELU removes the second batch's negative contribution; it is not an [identity](../glossary.md#term-identity-meanings) or
[transpose](../glossary.md#term-transpose). Replace `.relu()` with `.tanh()` or `.sigmoid()` for the other supported activations.
Each activation variant in this eight-kind-only example selects [graph](../glossary.md#graph)-model.v2 without changing
its ordinary [Tensor](../glossary.md#tensor) semantics. Graphs with only the original
five kinds still save byte-identical v1; old graph readers reject v2. The [regression](../glossary.md#term-regression) compares all
three activations' forward values, objectives and nonzero updates against an independent ordinary
Tensor/[Engine](../glossary.md#term-engine) [oracle](../glossary.md#term-test-oracle), including training after load and reloading trained state.

## Prerequisites

- One open `Engine.standard()`, or one explicitly composed [CPU](../glossary.md#term-cpu)/[Metal](../glossary.md#term-metal) [Engine](../glossary.md#term-engine).
- A non-empty [graph](../glossary.md#graph) supported by the Engine's registered [backends](../glossary.md#backend).
- Every [Tensor](../glossary.md#tensor) reported by `CompiledGraph.inputs()`, with compatible live caller-owned host
  storage when each run begins.<span data-glossary-navigation="true"> ([host storage glossary](../glossary.md#host-storage); [run glossary](../glossary.md#term-run))</span>

The Engine remains the composition [owner](../glossary.md#term-owner). A prepared handle borrows it and must not outlive it.
For a supported [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) Sequential, [declarative reconstruction](declarative-models.md) can
first build a [descriptor](../glossary.md#term-descriptor)-only [model](../glossary.md#term-model) with stable state paths. Load its [parameter](../glossary.md#term-parameter) values
before constructing the forward expression to [compile](../glossary.md#compile); reconstruction itself neither
executes the model nor reconstructs it from a stored-model file.

## Load a saved whole [model](../glossary.md#term-model)

To recover a supported bound `Sequential` without rebuilding its layers, use
`ModelCheckpoint.save(path, source, engine)` and `ModelCheckpoint.load(path)`. Load returns a
value-loaded `DeclarativeModel`, not a [descriptor](../glossary.md#term-descriptor)-only skeleton. It needs no [Engine](../glossary.md#term-engine) or [Arena](../glossary.md#term-arena);
ordinary forward construction and execution still use the caller's Engine. No [backend](../glossary.md#backend) or [prepared
execution](../glossary.md#prepared-execution--preparedexecution) is stored in the file.

Use an explicit `:extensions:nn` [dependency](../glossary.md#term-dependency) and import `nn.checkpoint.ModelCheckpoint`;
the generic [Checkpoint](../glossary.md#term-checkpoint) project no longer exports [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context). The old flexible readers/defaults are
unchanged, distinct from [static generic graph persistence](../reference/checkpoint/graph.md)<span data-glossary-navigation="true"> ([graph glossary](../glossary.md#graph); [persistence glossary](../glossary.md#term-persistence))</span>.
This complete Java 26 example needs NN and its public dependencies on the [classpath](../glossary.md#term-classpath).
Its trusted local directory is created before save. The source has
[FLOAT32](../glossary.md#term-float32) diagonal [weights](../glossary.md#term-weight) `[1, 0; 0, 2]` followed by RELU. Both source [owners](../glossary.md#term-owner) close before load;
the saved file is then deleted before fresh-[CPU](../glossary.md#term-cpu) [inference](../glossary.md#term-inference). The input `[-3, 4]` should produce
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

Expected output is `0.0, 8.0`. [Linear](../glossary.md#term-linear-meanings) first produces `[-3, 8]`; RELU produces `[0, 8]`.
The source's training mode is not persisted: the loaded tree starts in [evaluation](../glossary.md#term-evaluation) mode, while
[parameters](../glossary.md#term-parameter) remain [gradient](../glossary.md#term-gradient)-eligible. Its [heap](../glossary.md#term-heap) values survive source Arena/Engine closure and file
deletion; re-saving needs no caller expression wrappers or [native parameter copy](../glossary.md#native-parameter-copy--trainingparameterstorage). This does not
prove a saved execution plan or training continuation. Exact repeated parameter [identities](../glossary.md#term-identity-meanings) are
preserved, and compatible [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) [dimensions](../glossary.md#dimension) can change when a new forward expression is built.

The short calls use absent input-[artifact](../glossary.md#term-artifact) [identity](../glossary.md#term-identity-meanings). If the file stores one, default load raises
`ArtifactFormatException`; pass an application-verified `InputArtifactFingerprint` and explicit
`ArtifactLimits` instead. A values-only `model.v1` file needs the existing fully bound [target](../glossary.md#term-target)-load
[API](../glossary.md#term-application-programming-interface-api), not automatic [topology](../glossary.md#term-topology) inference. Custom [modules](../glossary.md#term-module), unbound state and recurrent models are not
supported by this whole-model workflow. See [whole-model limits and failures](../reference/checkpoint/models.md#bounds-lifetime-and-failures)
before selecting advanced encoded bounds; they are not peak-heap budgets.

## [Compile](../glossary.md#compile) and [prepare](../glossary.md#prepare) once

Compile the requested output boundary once, then retain its prepared [owner](../glossary.md#term-owner):

```java
CompiledGraph graph = engine.compile(List.of(output));
try (PreparedExecution prepared = engine.prepare(graph)) {
    assert prepared.compiledGraph() == graph;
    // Reuse engine.run(prepared, ...) while both prepared handle and Engine remain open.
}
```

`engine.prepare(graph)` performs exactly one preparation before returning. It does not execute the
[graph](../glossary.md#graph) or bind input storage. The graph fixes both boundaries:

- `graph.inputs()` is the stable input membership and [Compiler](../glossary.md#term-compiler) [occurrence](../glossary.md#term-occurrence) order; and
- every result uses the graph's stable forward-then-gradient publication occurrence order,
  including distinct occurrences that alias one inward representation.<span data-glossary-navigation="true"> ([gradient glossary](../glossary.md#term-gradient); [publication occurrence glossary](../glossary.md#publication-occurrence); [alias glossary](../glossary.md#aliasing-and-arena))</span>

## [Run](../glossary.md#term-run) and materialize

The input list may use any order because [Engine](../glossary.md#term-engine) binds by exact [Tensor](../glossary.md#tensor) [identity](../glossary.md#term-reference-identity):

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

Each call verifies exact Tensor identity and the complete compiled [descriptor](../glossary.md#term-descriptor), then snapshots the
current [host](../glossary.md#term-host)-storage associations in `graph.inputs()` order. It validates storage [data type](../glossary.md#data-type--datatype),
resolved-[layout](../glossary.md#layout) capacity, [liveness](../glossary.md#term-liveness), and current-thread accessibility before [borrowing](../glossary.md#term-borrowing). The caller
continues to own the storage and must keep it usable and free from conflicting mutation until that
run's result closes.

[`RunResult`](../glossary.md#run-result--runresult) is the existing [publication](../glossary.md#term-publication) [lease](../glossary.md#term-lease). Each `Publication` records its dense index,
[`TensorId`](../glossary.md#tensorid), final descriptor, forward or [gradient](../glossary.md#term-gradient) role, and applicable [derivative](../glossary.md#term-derivative) [metadata](../glossary.md#term-metadata).
`materialize(...)` accepts only the exact [occurrence](../glossary.md#term-occurrence) object from that result and performs a fresh
bounded copy through the publication adapter captured during preparation. A [detached](../glossary.md#detached-model-results)
[`HostTensorValue`](../glossary.md#host-tensor-value--hosttensorvalue) remains readable after the result, prepared handle, Engine, and input-storage [arena](../glossary.md#term-arena) close.

A **lease** gives access while the responsible result remains open. **[Materialization](../glossary.md#materialization)** makes
an explicit numerical copy, not merely another Java reference to the same storage.
**[Canonical](../glossary.md#term-canonical-form) [row-major](../glossary.md#term-row-major)** output puts one complete row before the next in the standard detached
representation. The byte ceiling limits that copy, not all temporary execution memory.

## [Run](../glossary.md#term-run) fresh values in compiled order

**Goal and inputs.** [Prepare](../glossary.md#prepare) negation of a storage-free [FLOAT32](../glossary.md#term-float32) [2] [prototype](../glossary.md#term-prototype) once, then supply
fresh [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters) [1,-2] and [3,-4]. With Java 26, [Engine](../glossary.md#term-engine) and [Model](../glossary.md#term-model) on the [classpath](../glossary.md#term-classpath), this complete
block uses their ordinary [scalar](../glossary.md#term-scalar) [CPU](../glossary.md#term-cpu) [heap](../glossary.md#term-heap)-input support:

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

Use the Model/Engine imports from the whole-model example above. [Compilation](../glossary.md#compiler-engine-and-execution-stage-names) fixes the single
input position; preparation is outside the loop. Each run binds that batch's current storage
without changing the prototype. Output is `-1.0, 2.0`, then `-3.0, 4.0`; [detached](../glossary.md#detached-model-results) values may be
retained after every [owner](../glossary.md#term-owner) closes. With multiple inputs, follow `graph.inputs()` order, not
expression discovery order. A value may fill multiple matching slots.

runValues requires exact count and full static [descriptors](../glossary.md#term-descriptor) ([dtype](../glossary.md#data-type--datatype), [shape](../glossary.md#shape), [layout](../glossary.md#layout) including absence
versus presence, and requiresGrad). A compiled [32,784] input rejects [64,784]. All storage is
validated before any borrow, including checked referenced spans for resolved layouts or checked
[canonical](../glossary.md#term-canonical-form) [element](../glossary.md#term-element)/byte geometry for absent layouts. Unknown/[overflow](../glossary.md#term-overflow) geometry rejects. This extra
absent-layout check does not change [identity](../glossary.md#term-identity-meanings)-based run; `engine.run(prepared, List.of(firstBatch))`
rejects here because the fresh ID is not the prototype ID.

## Reuse and [concurrency](../glossary.md#term-concurrency)

Sequential and concurrent calls share the [immutable](../glossary.md#term-immutable) prepared recipe and direct captured
input/[publication](../glossary.md#term-publication) adapters. Every call receives a distinct [mutable](../glossary.md#term-mutable) [Runtime](../glossary.md#term-runtime) [`RunState`](../glossary.md#run-state--runstate), [borrowed](../glossary.md#ownership-borrowing-and-lease)
input wrappers, internal [buffers](../glossary.md#term-buffer), and workspaces. A [run](../glossary.md#term-run) performs no [compilation](../glossary.md#compiler-engine-and-execution-stage-names), preparation,
provider or availability interrogation, [Engine](../glossary.md#term-engine)-[registry](../glossary.md#term-registry) [lookup](../glossary.md#term-lookup), [backend](../glossary.md#backend) selection, [kernel](../glossary.md#kernel)
selection, or schedule assembly.

The prepared [owner](../glossary.md#term-owner) adds no [cache](../glossary.md#term-cache) or result type. `Engine.compute(...)` and
`Engine.backward(...)` make a different [lifetime](../glossary.md#term-lifetime) choice: every call performs fresh compilation,
preparation, execution, [materialization](../glossary.md#materialization), and cleanup.

## Close and in-flight runs

Close the prepared handle when its reuse window ends. Close is [idempotent](../glossary.md#term-idempotent), closes the one [prepared
execution](../glossary.md#prepared-execution--preparedexecution) exactly once, and immediately prevents later [run](../glossary.md#term-run) [admission](../glossary.md#term-admission). It does not close a
[`RunResult`](../glossary.md#run-result--runresult) that was already returned.

**Idempotent close** means repeating close does not [release](../glossary.md#term-release) the same resource again.
An **in-flight** run has already begun admission; **admission** is permission to start after
[lifecycle](../glossary.md#lifecycle) checks. A **[race](../glossary.md#term-race)** means close and run overlap in time, so which obtains permission
first determines whether that run can proceed. These rules do not grant permission for
conflicting mutations of shared input or [Model](../glossary.md#term-model) storage.

If close races a run, [Runtime](../glossary.md#term-runtime)'s existing prepared [lease](../glossary.md#term-lease) decides the race:

- a run that acquired the lease first may finish; prepared close does not wait for it; and
- a close-first run is rejected.

[Engine](../glossary.md#term-engine) admission is checked first. Under an open Engine, both run methods check prepared nullness/exact [ownership](../glossary.md#term-ownership-meanings) and the exact inward delegate's close
gate before reading the input list, [Tensor](../glossary.md#tensor) [metadata](../glossary.md#term-metadata), storage, or adapters. A run attempted after
prepared close therefore reports `prepared execution is closed` without [borrowing](../glossary.md#term-borrowing) storage even when
the supplied inputs are invalid. If Engine closure has begun, `advanced engine is closed` wins
before any [argument](../glossary.md#term-argument) inspection.

Engine closure waits for admitted Engine operations, then closes retained results in reverse
successful-run order, ordinary and advanced prepared handles in reverse
preparation-[publication](../glossary.md#term-publication) order, and finally [backend](../glossary.md#backend) integrations. Engine closure is the final
cleanup boundary for a prepared handle the caller leaves open.

Both methods now check the exact prepared delegate before inputs. This intentionally changes
former direct-run diagnostic precedence: closed prepared wins null/malformed/dead inputs without
borrowing; null or foreign prepared handles also precede list errors.

[`InferenceSession`](../glossary.md#former-inference-session--inferencesession) and `Engine.session` are removed. Migrate to [prepare](../glossary.md#prepare)/run explicitly and
recompile; no replacement wrapper is introduced.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Preparation rejects a [graph](../glossary.md#graph) | The graph belongs to another [Engine](../glossary.md#term-engine), Engine closure began, or cold preparation rejected its plan. | [Compile](../glossary.md#compile) and [prepare](../glossary.md#prepare) with the same open Engine; inspect the prepare-time capability failure. |
| A [run](../glossary.md#term-run) reports a missing, duplicate, or unexpected input | The supplied [identities](../glossary.md#term-identity-meanings) do not exactly match `graph.inputs()`. | Supply every required [Tensor](../glossary.md#tensor) exactly once; list order is arbitrary. |
| A run rejects a [descriptor](../glossary.md#term-descriptor) or storage | The [Tensor descriptor](../glossary.md#tensor-descriptor) changed [identity](../glossary.md#term-identity-meanings) expectations, or storage type, capacity, [lifetime](../glossary.md#term-lifetime), or thread accessibility is incompatible. | Use the exact compiled Tensor and attach compatible accessible live storage before the run. |
| A run reports `prepared execution is closed` | Prepared close won [admission](../glossary.md#term-admission). | Prepare a new handle from the still-valid graph and Engine if more runs are needed. |
| [Materialization](../glossary.md#materialization) rejects a [publication](../glossary.md#term-publication) | The [occurrence](../glossary.md#term-occurrence) belongs to another result, the result or Engine is closed, or the byte limit is too small. | Use an occurrence from the same open result and a sufficient checked bound. |

## Limitations

Prepared runs are [synchronous](../glossary.md#term-synchronous) and fixed to one [compiled graph](../glossary.md#compiled-graph--compiledgraphmodel). Fresh [identities](../glossary.md#term-identity-meanings) are supported only
by runValues; neither method provides dynamic [descriptors](../glossary.md#term-descriptor), [asynchronous](../glossary.md#term-asynchronous) submission, implicit
[materialization](../glossary.md#materialization), [persistence](../glossary.md#term-persistence), generic [device](../glossary.md#term-device)-result handles, or tuning controls. Explicit
[CPU](../glossary.md#term-cpu)/[Metal](../glossary.md#term-metal) composition supports current single-[owner](../glossary.md#term-owner) execution and the bounded all-seven-[carrier](../glossary.md#term-storage-carrier)
[rank](../glossary.md#term-rank)-0..16 static [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping transfer domain; unsupported
capability or transfer combinations fail during [compile](../glossary.md#compile) or preparation rather than
falling back during a [run](../glossary.md#term-run).

## Related documentation

- [Whole-model and values-only checkpoint API](../reference/checkpoint/models.md)<span data-glossary-navigation="true"> ([model glossary](../glossary.md#term-model); [checkpoint glossary](../glossary.md#term-checkpoint); [API glossary](../glossary.md#term-application-programming-interface-api))</span>
- [Describe, reconstruct and load a closed NN model](declarative-models.md)<span data-glossary-navigation="true"> ([NN model glossary](../glossary.md#model-module-buffer-and-scan-distinctions))</span>
- [Compile a graph](compiling-graphs.md)<span data-glossary-navigation="true"> ([Compile glossary](../glossary.md#compile); [graph glossary](../glossary.md#graph))</span>
- [Prepare execution](preparing-execution.md)<span data-glossary-navigation="true"> ([Prepare glossary](../glossary.md#prepare))</span>
- [Runtime API](../reference/runtime/engine-boundary.md#current-ordinary-engine-boundary)<span data-glossary-navigation="true"> ([Runtime glossary](../glossary.md#term-runtime))</span>
- [Public API status](../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
- [Lifecycle architecture](../architecture/lifecycle.md#current-public-engine-lifecycle)<span data-glossary-navigation="true"> ([Lifecycle glossary](../glossary.md#lifecycle))</span>
