<a id="data-sequence-batching-api"></a>
# Sequence batching and valid lengths

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/data/src/main/java/io/github/pho001/synaptik/data/PaddedSequenceBatch.java)

A sequence [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) puts several ordered sequences into one rectangular [Tensor](../../glossary.md#tensor). **[Padding](../../glossary.md#padding)** fills
positions needed to make their lengths equal; a **valid count** records how many positions belong
to each original sequence. Logical **[axis](../../glossary.md#term-axis) order** says whether batch or time comes first. These
[metadata](../../glossary.md#term-metadata) declarations do not themselves rearrange storage or make a downstream [operation](../../glossary.md#operation) ignore padding.

## Which participant copies values, and which only describes them?

The application calls `PaddedSequenceBatch.builder(values)` with an already-created Tensor.
Data borrows that exact Tensor and copies only the host length array supplied to the setter;
it does not read the Tensor's storage or make its lifetime safe. `LongSequenceBatcher.batch`
is a different boundary: Data copies application long-array sequences into a fresh allocated
INT64 Tensor and retains no input arrays afterward. Text calls that batcher internally for
encoded token IDs. Neither object executes a network.

For recurrence, the application explicitly exports a new length Tensor, then supplies both
values and lengths to [NN's runtime-length API](../nn/recurrent.md) or an Engine run. Export copies
counts only; execution borrows the stored inputs under its own lifetime rules. Read
[Text batching](../text/tokenization.md#explicit-batch-configuration) for the integer-ID producer
and [Engine binding](../runtime/engine-boundary.md) for the later numerical consumer.

<a id="purpose-and-mental-model"></a>
## Purpose and [mental model](../../glossary.md#term-mental-model)

Data can pad ordered long arrays into a [Tensor](../../glossary.md#tensor) or pair an existing Tensor with declared valid counts.
Counts are position independent: RIGHT selects a valid prefix (the first `length` time steps),
and LEFT selects a valid suffix (the last `length` time steps). [Padding](../../glossary.md#padding) occupies the remaining
positions. Counts and side, not numeric sentinels, distinguish valid data from padding. Zero is
ordinary data inside either valid interval.

`PaddedSequenceBatch` borrows the Tensor, `ValidSequenceLengths` owns an [immutable](../../glossary.md#term-immutable) copy of
the [host](../../glossary.md#term-host) counts, `PaddingSide` selects the valid interval, and `SequenceAxisOrder` identifies
the logical leading [axes](../../glossary.md#term-axis). Its fluent builder
checks declarations without reading or changing Tensor
contents; callers supply values that already have the declared order and padding.

The unpublished `RightPaddedSequenceBatch` has been replaced, without a compatibility [alias](../../glossary.md#aliasing-and-arena).
Migrate to `PaddedSequenceBatch`, add `PaddingSide.RIGHT` as the fourth constructor [argument](../../glossary.md#term-argument),
or add `.rightPadded()` to the values builder to retain the previous prefix semantics.
Use `LongSequenceBatcher` when Data should allocate and pad values rather than only declare [metadata](../../glossary.md#term-metadata).
For RIGHT/TIME_MAJOR [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters), explicitly call `toScanLengthsTensor()` to copy counts into an
ordinary [runtime](../../glossary.md#term-runtime) input. The application, not Data, passes values and lengths to [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) or [Engine](../../glossary.md#term-engine).

These types are in `io.github.pho001.synaptik.data`, [Gradle](../../glossary.md#term-gradle) project `:extensions:data`. Data exports
the public [Model](../../glossary.md#term-model) [dependency](../../glossary.md#term-dependency) used by its Tensor [signatures](../../glossary.md#term-signature). The
[owning Data contract](../../architecture/contracts/extensions-training.md#extensionsdata) defines
the boundary; this reference explains the implemented [API](../../glossary.md#term-application-programming-interface-api).

<a id="complete-fluent-example"></a>
## Complete fluent example

The goal is to describe three [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-major [FLOAT32](../../glossary.md#term-float32) sequences with four time positions each.
This Java example uses public [Model](../../glossary.md#term-model) factories to create values first, then wraps that existing
[Tensor](../../glossary.md#tensor) with Data [metadata](../../glossary.md#term-metadata). It needs Data and its exported Model [API](../../glossary.md#term-application-programming-interface-api), not [Engine](../../glossary.md#term-engine) or a session.

```java
// Data borrows values and copies host counts; it never treats numeric zero as the padding rule.
import io.github.pho001.synaptik.data.PaddedSequenceBatch;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

Shape shape = Shape.of(3, 4);
var descriptor = new TensorDescriptor(
        DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
Tensor values = TensorFactory.fromFlatArray(descriptor, Optional.empty(), new float[] {
    0, 4, 0, 99,
    7, 0, 99, 99,
    0, 99, 99, 99
});

long[] lengths = {3, 2, 1};
var builder = PaddedSequenceBatch.builder(values).batchMajor().rightPadded().validLengths(lengths);
lengths[0] = 0; // The setter has already copied the lengths.
var batch = builder.build();

System.out.println(batch.values() == values);
System.out.println(batch.axisOrder());
System.out.println(Arrays.toString(batch.validLengths().toLongArray()));
```

The output is:

```text
true
BATCH_MAJOR
[3, 2, 1]
```

`TensorFactory.fromFlatArray` copies the flat data into a Model Tensor with the supplied resolved
[layout](../../glossary.md#layout). That copy belongs to Model construction, not Data. The builder retains the exact
`values` reference and snapshots only the lengths at `validLengths(...)`; `build()` does not
copy either the values or the [immutable](../../glossary.md#term-immutable) lengths again.

The declared prefixes and derived [padding](../../glossary.md#padding) lengths are:

| Original row | Valid prefix | Padding length at T = 4 |
|---|---|---|
| 0 | `[0, 4, 0]` | $4-3=1$ |
| 1 | `[7, 0]` | $4-2=2$ |
| 2 | `[0]` | $4-1=3$ |

Here `99` is caller-chosen padding, not a sentinel checked by Data. Replacing any suffix value
with zero, [NaN](../../glossary.md#term-nan) or another value does not change metadata [validation](../../glossary.md#term-validation). The wrapper declares valid
positions; it does not prove the supplied contents are padded correctly.

<a id="axis-order-and-geometry"></a>
## [Axis](../../glossary.md#term-axis) order and geometry

`.batchMajor()` declares `[B, T, ...]`; `.timeMajor()` declares `[T, B, ...]`. B is the original
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-row count and T is the [Tensor](../../glossary.md#tensor)'s declared time [extent](../../glossary.md#term-extent) in logical steps. For the example's lengths,
[time-major](../../glossary.md#term-time-major) values would need [Shape](../../glossary.md#shape) `[4,3]` rather than `[3,4]`. Changing the builder's order does
not [transpose](../../glossary.md#term-transpose) data or prove [contiguity](../../glossary.md#storage-mapping-contiguity-and-injectivity). Callers must supply the matching Tensor themselves.

[Rank](../../glossary.md#term-rank) must be at least two, and both leading axes must be [Model](../../glossary.md#term-model) `StaticDimension` values. B must
equal the number of lengths, and each length must lie in `[0, T]`. Trailing [dimensions](../../glossary.md#dimension) are [opaque](../../glossary.md#term-opaque):
they may be absent, zero, dynamic or expressions. A zero [feature](../../glossary.md#term-feature) extent does not force zero
counts. Every Model-valid [data type](../../glossary.md#data-type--datatype), including [BOOL](../../glossary.md#data-type--datatype), [gradient](../../glossary.md#term-gradient) flag, [provenance](../../glossary.md#provenance) and physical
[layout](../../glossary.md#layout) is accepted; unresolved layout and absent storage are also accepted.

B = 0 accepts explicitly empty lengths at any T, including zero. T = 0 accepts only zero lengths.
For example, Shape `[2,3]` with batch-major lengths `[0,2]` declares an empty first sequence and
two valid steps in the second. Their numeric values may both be zero.

Extents and lengths remain `long`: T and lengths may reach `Long.MAX_VALUE`. B above
`Integer.MAX_VALUE` cannot match the [host](../../glossary.md#term-host)-array count and is rejected without allocating a giant
array. [Validation](../../glossary.md#term-validation) does not multiply extents or calculate storage spans. Length creation and batch
validation are [O(B)](../../glossary.md#memory-management-atomic-counters-and-complexity), independent of T and Tensor [element](../../glossary.md#term-element) count.

<a id="builder-state-and-reuse"></a>
## Builder state and reuse

[Axis](../../glossary.md#term-axis) order, [padding](../../glossary.md#padding) side and lengths must be selected explicitly, in any order.
Use `.leftPadded()` for a valid suffix or `.rightPadded()` for a valid prefix. There are no defaults.
`validLengths()` with no [arguments](../../glossary.md#term-argument) selects empty lengths; omitting the call leaves lengths
missing. `build()` throws `IllegalStateException` for any missing selection before checking
geometry. `builder(null)` instead fails immediately with `NullPointerException("values")`.

Every setter returns the same builder; the last successful selection wins. `validLengths(...)`
immediately copies and checks non-negative entries, even before an order is selected. A null array
or negative entry fails without replacing the previous selection. Callers must coordinate source
array mutation during the setter call, not merely during `build()`.

Each successful build creates a fresh record sharing the exact selected [immutable](../../glossary.md#term-immutable) lengths and
[borrowed](../../glossary.md#ownership-borrowing-and-lease) [Tensor](../../glossary.md#tensor). Repeated builds are equal but not identical. Later setters cannot change earlier
[batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters). Continuing the example:

```java
// The new batch shares immutable prior counts; later builder choices do not rewrite earlier batches.
var repeated = builder.build(); // Fresh batch, same immutable lengths as batch.
builder.validLengths(4, 0, 1);
var changed = builder.build();  // batch still declares [3, 2, 1].
```

A setter does not check T or B. For example, `builder.validLengths(5, 0, 1)` succeeds because all
entries are non-negative, but the following `build()` fails because T is 4. A failed build leaves
that selection in place; replace it with valid geometry before retrying. The reusable builder is
caller-coordinated and not [thread-safe](../../glossary.md#term-thread-safe). Immutable lengths may be shared independently.

<a id="direct-construction-for-advanced-callers"></a>
## Direct construction for advanced callers

Callers that already manage [immutable](../../glossary.md#term-immutable) [metadata](../../glossary.md#term-metadata) can use the public record constructor:

```java
// Direct construction declares logical order and side; it does not transpose or pad values.
import io.github.pho001.synaptik.data.SequenceAxisOrder;
import io.github.pho001.synaptik.data.PaddingSide;
import io.github.pho001.synaptik.data.ValidSequenceLengths;

var prefixes = ValidSequenceLengths.of(3, 2, 1);
var direct = new PaddedSequenceBatch(values, prefixes, SequenceAxisOrder.BATCH_MAJOR, PaddingSide.RIGHT);
```

This continues the example with the same `values`. Both construction paths use the [canonical](../../glossary.md#term-canonical-form)
record constructor's geometry checks and retain the exact four component references.
`ValidSequenceLengths.of(long...)` copies once and rejects negative entries in row order; it
stores no T. `batchSize()` returns the `int` count, `length(row)` reads an original zero-based row,
and every `toLongArray()` returns a fresh [mutable](../../glossary.md#term-mutable) copy, including for empty lengths.

Length equality and hashing compare ordered lengths structurally. [Batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) record equality uses
[Tensor](../../glossary.md#tensor) object [identity](../../glossary.md#term-reference-identity), structural counts and both enum [identities](../../glossary.md#term-identity-meanings), not Tensor contents. Diagnostics
describe metadata, not a serialized format or storage state.

<a id="validation-failures-and-borrowing"></a>
## [Validation](../../glossary.md#term-validation) failures and [borrowing](../../glossary.md#term-borrowing)

The [canonical](../../glossary.md#term-canonical-form) constructor null-checks `values`, `validLengths`, then `axisOrder` and `paddingSide`, with a
`NullPointerException` naming the [parameter](../../glossary.md#term-parameter). It then checks [rank](../../glossary.md#term-rank), static time/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [axes](../../glossary.md#term-axis), B versus
length count, and upper bounds in row order. Geometry failures throw `IllegalArgumentException`
identifying the mismatch or first offending row and T. Negative [host](../../glossary.md#term-host) lengths also throw
`IllegalArgumentException` identifying the first row; a null length array throws
`NullPointerException("lengths")`. Invalid `length(row)` indices throw `IndexOutOfBoundsException`;
negative indices never wrap.

[Metadata](../../glossary.md#term-metadata) wrapping reads only [immutable](../../glossary.md#term-immutable) [descriptor](../../glossary.md#term-descriptor)/[Shape](../../glossary.md#shape) metadata and host lengths. It does not call
`hostStorage()`, check [liveness](../../glossary.md#term-liveness), obtain a segment or [lease](../../glossary.md#term-lease), attach or replace storage, close
resources, scan values, create expressions, call `contiguous()` or execute. A [borrowed](../../glossary.md#ownership-borrowing-and-lease) storage
[scope](../../glossary.md#term-scope) may already be dead after its earlier [Model](../../glossary.md#term-model) attachment; wrapping adds no liveness check.
Later replacement or clearing of the [Tensor](../../glossary.md#tensor)'s storage association remains governed by Model.
The batch fixes references, not Tensor contents, and adds no readiness, [lifetime](../../glossary.md#term-lifetime) or thread-safety
guarantee for Tensor access. See [host storage](../../glossary.md#host-storage).

Metadata wrapping does not pad or allocate values; LongSequenceBatcher below performs that work.
Neither path truncates input. Host lengths are not themselves a [runtime](../../glossary.md#term-runtime) [graph](../../glossary.md#graph) input.
The opt-in export below creates an ordinary [INT64](../../glossary.md#data-type--datatype) length Tensor but does not bind it or add an
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) batch-compute overload. [Padding](../../glossary.md#padding) lengths and dense [masks](../../glossary.md#term-boolean-mask) are derived concepts, not stored fields.

<a id="explicit-runtime-scan-lengths"></a>
## Explicit [runtime](../../glossary.md#term-runtime) scan lengths

`public Tensor PaddedSequenceBatch.toScanLengthsTensor()` exports valid counts in original row
order. It checks RIGHT [padding](../../glossary.md#padding) first, then TIME_MAJOR logical [axes](../../glossary.md#term-axis). LEFT or BATCH_MAJOR throws
`IllegalArgumentException` before [Tensor](../../glossary.md#tensor) creation, allocation or storage access; neither order nor
side is converted. Logical TIME_MAJOR order alone does not prove [canonical](../../glossary.md#term-canonical-form) physical [layout](../../glossary.md#layout).

Every successful call allocates a fresh, non-null ordinary [heap](../../glossary.md#term-heap)-backed [INT64](../../glossary.md#data-type--datatype) `[B]` [leaf](../../glossary.md#term-leaf) through
public `TensorFactory.allocate`, then fills its writable [Model](../../glossary.md#term-model) [host storage](../../glossary.md#host-storage) directly from counts.
The result is [provenance](../../glossary.md#provenance)-free, canonical [contiguous](../../glossary.md#term-contiguous), offset-zero, non-[view](../../glossary.md#term-view-meanings) and `requiresGrad=false`.
B=0 returns a fresh Tensor with fresh empty heap storage. Allocation and count copying cost [O(B)](../../glossary.md#memory-management-atomic-counters-and-complexity);
there is no cached [mutable](../../glossary.md#term-mutable) export. Mutating an export cannot change [immutable](../../glossary.md#term-immutable) counts or another
export. Model allocation failures, including identifier exhaustion and heap exhaustion, propagate.

Only counts are copied. Export never reads values storage or checks its [liveness](../../glossary.md#term-liveness), copies or repacks
sequence contents, changes [descriptors](../../glossary.md#term-descriptor) or [gradient](../../glossary.md#term-gradient) flags, or attaches/replaces storage. It works
even for storage-free values or values whose former storage [scope](../../glossary.md#term-scope) has closed. The constructor's
broader geometry contract remains unchanged, including every Model-valid [dtype](../../glossary.md#data-type--datatype)/layout/provenance,
[opaque](../../glossary.md#term-opaque) trailing [dimensions](../../glossary.md#dimension), zero extents and `long` time/count values up to `Long.MAX_VALUE`.
Export success is not execution [admission](../../glossary.md#term-admission).

### Pass exported counts to a numerical consumer

The caller passes ordinary Tensors to existing `model.compute(values, lengths)`. Both must match
the Model's exact input descriptors and have usable storage free from conflicting mutation for the
whole [synchronous](../../glossary.md#term-synchronous) call. Heap lengths have ordinary garbage-collected [lifetime](../../glossary.md#term-lifetime): no [Arena](../../glossary.md#term-arena), close
protocol, native allocation, [device](../../glossary.md#term-device) transfer or new [resource owner](../../glossary.md#ownership-borrowing-and-lease) is introduced. Compute returns
[detached](../../glossary.md#detached-model-results) results; Model and [Engine](../../glossary.md#term-engine) retain their existing close and preparation responsibilities.

<a id="complete-numeric-inference-example"></a>
## Complete numeric [inference](../../glossary.md#term-inference) example

This runnable Java program processes sequences `[1,2,3]` and `[10,20]` in original row order.
Use the repository [JDK](../../glossary.md#term-java-development-kit-jdk) 26 and an application [classpath](../../glossary.md#term-classpath) containing Data, [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and [Engine](../../glossary.md#term-engine) with its
standard [backend](../../glossary.md#backend)/[runtime](../../glossary.md#term-runtime) dependencies (including tools used by the standard composition).
[Compile](../../glossary.md#compile) and [run](../../glossary.md#term-run) with `--add-modules jdk.incubator.vector`; also pass
`--enable-native-access=ALL-UNNAMED` to `java`. Those application dependencies add no Data-to-NN edge.

Save as `ScanExample.java`:

```java
// Export counts explicitly; only the later Model call consumes actual values and runtime lengths.
import io.github.pho001.synaptik.data.PaddedSequenceBatch;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import java.util.Optional;

public class ScanExample {
    public static void main(String[] args) {
        var shape = Shape.of(3, 2, 1);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        Tensor values = TensorFactory.fromFlatArray(descriptor, Optional.empty(),
                new float[] {1, 10, 2, 20, 3, Float.NaN});
        var batch = PaddedSequenceBatch.builder(values).timeMajor().rightPadded()
                .validLengths(3, 2).build();
        try (var model = Model.builder().input(DataType.FLOAT32, 3, 2, 1).seed(42)
                .rnnScan(4).finalHidden().linear(1).buildWithLengths()) {
            // The NaN suffix is excluded by counts, not by testing whether values equal a sentinel.
            var result = model.compute(batch.values(), batch.toScanLengthsTensor());
            if (!result.shape().equals(Shape.of(2, 1))) throw new AssertionError(result.shape());
            float first = result.bytes().getFloat(0);
            float second = result.bytes().getFloat(Float.BYTES);
            if (!Float.isFinite(first) || !Float.isFinite(second)) throw new AssertionError("Padding leaked");
            System.out.println(result.shape());
            System.out.println(first + ", " + second);
        }
    }
}
```

The [canonical](../../glossary.md#term-canonical-form) [FLOAT32](../../glossary.md#term-float32) `[T,B,F] = [3,2,1]` [Tensor](../../glossary.md#tensor) stores time slices `[1,10]`, `[2,20]`,
`[3,NaN]`. `fromFlatArray` copies these values during [Model](../../glossary.md#term-model) Tensor construction; wrapping and
exporting counts never copy them again. Counts `[3,2]` exclude the [NaN](../../glossary.md#term-nan) padded suffix from recurrence.
`rnnScan(4)` produces dense hidden values `[3,2,4]`; `finalHidden()` selects each row's semantic
final state `[2,4]`, and `linear(1)` produces one score per row, `[2,1]`.
With [seed](../../glossary.md#term-seed) 42, the qualified [CPU](../../glossary.md#term-cpu) route gives [finite](../../glossary.md#term-finite) scores approximately
`[-0.47897735, -0.6437197]`. These are untrained model outputs, not calibrated [predictions](../../glossary.md#term-prediction).

`gruScan(4)` and `lstmScan(4)` use the same values/lengths workflow. Without `finalHidden()` the
scan returns dense `[T,B,H]`, not compact outputs; `lastTimeStep()` selects `T-1`, not each row's
last valid transition. A zero-length row returns its zero initial hidden state at the scan's
final-state port, although an ordinary classifier [bias](../../glossary.md#term-bias) can subsequently make its score nonzero.

Keep the same Model open to reuse its preparation for fresh values and counts at the same static
[signature](../../glossary.md#term-signature). Every replacement must match [shape](../../glossary.md#shape), [dtype](../../glossary.md#data-type--datatype), [layout](../../glossary.md#layout) presence/geometry and [gradient](../../glossary.md#term-gradient)
eligibility exactly; changing counts does not authorize arbitrary T, B or F. For direct Engine
reuse, [prepare](../../glossary.md#prepare) once and pass fresh values in `compiled.inputs()` order to `runValues`; preserve
caller storage until its [`RunResult`](../../glossary.md#run-result--runresult) closes. Data performs neither [binding](../../glossary.md#term-binding) nor preparation.

This Data consumer slice uses canonical [contiguous](../../glossary.md#term-contiguous) offset-zero non-[view](../../glossary.md#term-view-meanings) FLOAT32 `[T,B,F]` with
`requiresGrad=false`, RIGHT/TIME_MAJOR [metadata](../../glossary.md#term-metadata) and positive static T/B/F for the fluent builder.
Empty count export does not widen that builder contract; direct zero-T/B scans remain separate.
[F32](../../glossary.md#floating-point-precision-and-rounding) `rnnScan` now composes NN0033's general [RNN](../../glossary.md#term-recurrent-neural-network-rnn), whose source outputs are canonical; retained
fixed families keep their old source-layout semantics. Gradient flags are not cleared.
Data only exports values/counts: NN independently qualifies general RNN compute/train/v5
[persistence](../../glossary.md#term-persistence), and lengths never receive gradients. Export itself does not qualify fixed GRU/LSTM BPTT; their actual first-order
training uses the independently admitted [Compiler/Training route](../training/sessions.md#owned-model-training-convenience).

Export does not enable a fluent Text IDs → [Embedding](../../glossary.md#embedding) → scan chain. ModelBuilder input is currently
floating-only, has no Embedding recipe and instantiates recipes with the original input dtype.
That integer-to-floating transition needs a separately selected NN capability.

<a id="integral-sequence-batching"></a>
## Integral sequence batching

Use `LongSequenceBatcher` to copy long-array sequences into a fresh [INT64](../../glossary.md#data-type--datatype) [Tensor](../../glossary.md#tensor).
[Padding](../../glossary.md#padding) value, side and logical [axis](../../glossary.md#term-axis) order have no defaults. The configuration is [immutable](../../glossary.md#term-immutable);
its reusable builder is not [thread-safe](../../glossary.md#term-thread-safe). Later builder changes leave existing configurations
unchanged. This example requires only Data and its exported [Model](../../glossary.md#term-model) [API](../../glossary.md#term-application-programming-interface-api):

```java
// This batcher allocates and pads fresh INT64 values; unlike metadata wrapping, it copies row contents.
import io.github.pho001.synaptik.data.LongSequenceBatcher;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.List;

var batcher = LongSequenceBatcher.builder()
        .paddingValue(0).leftPadded().batchMajor().build();
var padded = batcher.batch(List.of(new long[] {2, 3}, new long[] {1}));
System.out.println(Arrays.toString(padded.validLengths().toLongArray()));
System.out.println(Arrays.toString(padded.values().hostStorage().orElseThrow()
        .segment().toArray(ValueLayout.JAVA_LONG)));
```

The output is `[2, 1]` followed by `[2, 3, 0, 1]`: [Shape](../../glossary.md#shape) is `[2,2]` and the
second row has one left padding position. With `.rightPadded()`, the flat values become
`[2, 3, 1, 0]` without changing counts. A real zero inside an input row remains valid data.
For a non-negative integer valid count $L$ and time extent $T$ with $0\le L\le T$,
LEFT declares integer time coordinates $T-L\le t<T$, while RIGHT declares $0\le t<L$.
At $T=4$, $L=2$, LEFT uses positions $2,3$ and RIGHT uses $0,1$; each leaves two padding
positions. For $L=0$ the valid set is empty. The batcher never truncates and preserves both
row order and element order.

B is list size and T is the maximum row length. Empty input yields B=0,T=0; all-empty
rows yield T=0 in either order and side. The result has [contiguous](../../glossary.md#term-contiguous) offset-zero [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix)
[layout](../../glossary.md#layout), no [gradients](../../glossary.md#term-gradient) or [provenance](../../glossary.md#provenance), and writable garbage-collected [heap](../../glossary.md#term-heap) storage.
One Model allocation is filled directly; no full flat staging/import copy is made.
There is no [Arena](../../glossary.md#term-arena), close [operation](../../glossary.md#operation), or additional storage-[lifetime](../../glossary.md#term-lifetime) [owner](../../glossary.md#term-owner).

Callers coordinate list and row mutation throughout the [synchronous](../../glossary.md#term-synchronous) call; it is not a
transactional [snapshot](../../glossary.md#term-snapshot). Once the call returns, no caller list or arrays are retained, and their
mutation cannot change the output. Repeated calls allocate independent values.
Both list passes use sequential traversal, avoiding quadratic indexed access on linked lists.

Null input throws `NullPointerException("sequences")`; a null row throws
`NullPointerException` identifying its index. Geometry is checked incrementally before
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-sized [metadata](../../glossary.md#term-metadata) or output allocation, so a first row proving oversized geometry rejects
without scanning later rows. The checked element count $BT$, with $B$ rows and $T$ padded time steps, must fit
`Integer.MAX_VALUE` elements, and the byte
span uses checked long arithmetic, not an int byte cap. Invalid geometry throws
`IllegalArgumentException`; Model allocation failures, including heap exhaustion, propagate.

LEFT metadata does not extend recurrent scans: they still require valid prefixes with right padding.
The explicit count export above does not add [runtime](../../glossary.md#term-runtime) [binding](../../glossary.md#term-binding), [tokenization](../../glossary.md#frozen-whitespace-tokenizer), execution or [persistence](../../glossary.md#term-persistence).

<a id="related-contracts"></a>
## Related contracts

- [Public API status](../extensions/overview.md#current-public-contracts)
- [Tensor construction and storage](../tensor/index.md)
- [Host sequence metadata terminology](../../glossary.md#host-sequence-metadata)
- [Data module boundary](../../architecture/module-boundaries.md#extensionsdata)
- [Authoritative architecture root](../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
