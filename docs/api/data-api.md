# Data sequence batching API

## Purpose and mental model

Data can pad ordered long arrays into a Tensor or pair an existing Tensor with declared valid counts.
Counts are position independent: RIGHT selects a valid prefix (the first `length` time steps),
and LEFT selects a valid suffix (the last `length` time steps). Padding occupies the remaining
positions. Counts and side, not numeric sentinels, distinguish valid data from padding. Zero is
ordinary data inside either valid interval.

`PaddedSequenceBatch` borrows the Tensor, `ValidSequenceLengths` owns an immutable copy of
the host counts, `PaddingSide` selects the valid interval, and `SequenceAxisOrder` identifies
the logical leading axes. Its fluent builder
checks declarations without reading or changing Tensor
contents; callers supply values that already have the declared order and padding.

The unpublished `RightPaddedSequenceBatch` has been replaced, without a compatibility alias.
Migrate to `PaddedSequenceBatch`, add `PaddingSide.RIGHT` as the fourth constructor argument,
or add `.rightPadded()` to the values builder to retain the previous prefix semantics.
Use `LongSequenceBatcher` when Data should allocate and pad values rather than only declare metadata.
For RIGHT/TIME_MAJOR batches, explicitly call `toScanLengthsTensor()` to copy counts into an
ordinary runtime input. The application, not Data, passes values and lengths to NN or Engine.

These types are in `io.github.pho001.synaptik.data`, Gradle project `:extensions:data`. Data exports
the public Model dependency used by its Tensor signatures. The
[owning Data contract](../architecture/contracts/extensions-training.md#extensionsdata) defines
the boundary; this reference explains the implemented API.

## Complete fluent example

The goal is to describe three batch-major FLOAT32 sequences with four time positions each.
This Java example uses public Model factories to create values first, then wraps that existing
Tensor with Data metadata. It needs Data and its exported Model API, not Engine or a session.

```java
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
layout. That copy belongs to Model construction, not Data. The builder retains the exact
`values` reference and snapshots only the lengths at `validLengths(...)`; `build()` does not
copy either the values or the immutable lengths again.

The declared prefixes and derived padding lengths are:

| Original row | Valid prefix | Padding length at T = 4 |
|---|---|---|
| 0 | `[0, 4, 0]` | `4 - 3 = 1` |
| 1 | `[7, 0]` | `4 - 2 = 2` |
| 2 | `[0]` | `4 - 1 = 3` |

Here `99` is caller-chosen padding, not a sentinel checked by Data. Replacing any suffix value
with zero, NaN or another value does not change metadata validation. The wrapper declares valid
positions; it does not prove the supplied contents are padded correctly.

## Axis order and geometry

`.batchMajor()` declares `[B, T, ...]`; `.timeMajor()` declares `[T, B, ...]`. B is the original
batch-row count and T is the Tensor's declared time extent in logical steps. For the example's lengths,
time-major values would need Shape `[4,3]` rather than `[3,4]`. Changing the builder's order does
not transpose data or prove contiguity. Callers must supply the matching Tensor themselves.

Rank must be at least two, and both leading axes must be Model `StaticDimension` values. B must
equal the number of lengths, and each length must lie in `[0, T]`. Trailing dimensions are opaque:
they may be absent, zero, dynamic or expressions. A zero feature extent does not force zero
counts. Every Model-valid data type, including BOOL, gradient flag, provenance and physical
layout is accepted; unresolved layout and absent storage are also accepted.

B = 0 accepts explicitly empty lengths at any T, including zero. T = 0 accepts only zero lengths.
For example, Shape `[2,3]` with batch-major lengths `[0,2]` declares an empty first sequence and
two valid steps in the second. Their numeric values may both be zero.

Extents and lengths remain `long`: T and lengths may reach `Long.MAX_VALUE`. B above
`Integer.MAX_VALUE` cannot match the host-array count and is rejected without allocating a giant
array. Validation does not multiply extents or calculate storage spans. Length creation and batch
validation are O(B), independent of T and Tensor element count.

## Builder state and reuse

Axis order, padding side and lengths must be selected explicitly, in any order.
Use `.leftPadded()` for a valid suffix or `.rightPadded()` for a valid prefix. There are no defaults.
`validLengths()` with no arguments selects empty lengths; omitting the call leaves lengths
missing. `build()` throws `IllegalStateException` for any missing selection before checking
geometry. `builder(null)` instead fails immediately with `NullPointerException("values")`.

Every setter returns the same builder; the last successful selection wins. `validLengths(...)`
immediately copies and checks non-negative entries, even before an order is selected. A null array
or negative entry fails without replacing the previous selection. Callers must coordinate source
array mutation during the setter call, not merely during `build()`.

Each successful build creates a fresh record sharing the exact selected immutable lengths and
borrowed Tensor. Repeated builds are equal but not identical. Later setters cannot change earlier
batches. Continuing the example:

```java
var repeated = builder.build(); // Fresh batch, same immutable lengths as batch.
builder.validLengths(4, 0, 1);
var changed = builder.build();  // batch still declares [3, 2, 1].
```

A setter does not check T or B. For example, `builder.validLengths(5, 0, 1)` succeeds because all
entries are non-negative, but the following `build()` fails because T is 4. A failed build leaves
that selection in place; replace it with valid geometry before retrying. The reusable builder is
caller-coordinated and not thread-safe. Immutable lengths may be shared independently.

## Direct construction for advanced callers

Callers that already manage immutable metadata can use the public record constructor:

```java
import io.github.pho001.synaptik.data.SequenceAxisOrder;
import io.github.pho001.synaptik.data.PaddingSide;
import io.github.pho001.synaptik.data.ValidSequenceLengths;

var prefixes = ValidSequenceLengths.of(3, 2, 1);
var direct = new PaddedSequenceBatch(values, prefixes, SequenceAxisOrder.BATCH_MAJOR, PaddingSide.RIGHT);
```

This continues the example with the same `values`. Both construction paths use the canonical
record constructor's geometry checks and retain the exact four component references.
`ValidSequenceLengths.of(long...)` copies once and rejects negative entries in row order; it
stores no T. `batchSize()` returns the `int` count, `length(row)` reads an original zero-based row,
and every `toLongArray()` returns a fresh mutable copy, including for empty lengths.

Length equality and hashing compare ordered lengths structurally. Batch record equality uses
Tensor object identity, structural counts and both enum identities, not Tensor contents. Diagnostics
describe metadata, not a serialized format or storage state.

## Validation failures and borrowing

The canonical constructor null-checks `values`, `validLengths`, then `axisOrder` and `paddingSide`, with a
`NullPointerException` naming the parameter. It then checks rank, static time/batch axes, B versus
length count, and upper bounds in row order. Geometry failures throw `IllegalArgumentException`
identifying the mismatch or first offending row and T. Negative host lengths also throw
`IllegalArgumentException` identifying the first row; a null length array throws
`NullPointerException("lengths")`. Invalid `length(row)` indices throw `IndexOutOfBoundsException`;
negative indices never wrap.

Metadata wrapping reads only immutable descriptor/Shape metadata and host lengths. It does not call
`hostStorage()`, check liveness, obtain a segment or lease, attach or replace storage, close
resources, scan values, create expressions, call `contiguous()` or execute. A borrowed storage
scope may already be dead after its earlier Model attachment; wrapping adds no liveness check.
Later replacement or clearing of the Tensor's storage association remains governed by Model.
The batch fixes references, not Tensor contents, and adds no readiness, lifetime or thread-safety
guarantee for Tensor access. See [host storage](../glossary.md#host-storage).

Metadata wrapping does not pad or allocate values; LongSequenceBatcher below performs that work.
Neither path truncates input. Host lengths are not themselves a runtime graph input.
The opt-in export below creates an ordinary INT64 length Tensor but does not bind it or add an
NN batch-compute overload. Padding lengths and dense masks are derived concepts, not stored fields.

## Explicit runtime scan lengths

`public Tensor PaddedSequenceBatch.toScanLengthsTensor()` exports valid counts in original row
order. It checks RIGHT padding first, then TIME_MAJOR logical axes. LEFT or BATCH_MAJOR throws
`IllegalArgumentException` before Tensor creation, allocation or storage access; neither order nor
side is converted. Logical TIME_MAJOR order alone does not prove canonical physical layout.

Every successful call allocates a fresh, non-null ordinary heap-backed INT64 `[B]` leaf through
public `TensorFactory.allocate`, then fills its writable Model host storage directly from counts.
The result is provenance-free, canonical contiguous, offset-zero, non-view and `requiresGrad=false`.
B=0 returns a fresh Tensor with fresh empty heap storage. Allocation and count copying cost O(B);
there is no cached mutable export. Mutating an export cannot change immutable counts or another
export. Model allocation failures, including identifier exhaustion and heap exhaustion, propagate.

Only counts are copied. Export never reads values storage or checks its liveness, copies or repacks
sequence contents, changes descriptors or gradient flags, or attaches/replaces storage. It works
even for storage-free values or values whose former storage scope has closed. The constructor's
broader geometry contract remains unchanged, including every Model-valid dtype/layout/provenance,
opaque trailing dimensions, zero extents and `long` time/count values up to `Long.MAX_VALUE`.
Export success is not execution admission.

The caller passes ordinary Tensors to existing `model.compute(values, lengths)`. Both must match
the Model's exact input descriptors and have usable storage free from conflicting mutation for the
whole synchronous call. Heap lengths have ordinary garbage-collected lifetime: no Arena, close
protocol, native allocation, device transfer or new resource owner is introduced. Compute returns
detached results; Model and Engine retain their existing close and preparation responsibilities.

## Complete numeric inference example

This runnable Java program processes sequences `[1,2,3]` and `[10,20]` in original row order.
Use the repository JDK 26 and an application classpath containing Data, NN and Engine with its
standard backend/runtime dependencies (including tools used by the standard composition).
Compile and run with `--add-modules jdk.incubator.vector`; also pass
`--enable-native-access=ALL-UNNAMED` to `java`. Those application dependencies add no Data-to-NN edge.

Save as `ScanExample.java`:

```java
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

The canonical FLOAT32 `[T,B,F] = [3,2,1]` Tensor stores time slices `[1,10]`, `[2,20]`,
`[3,NaN]`. `fromFlatArray` copies these values during Model Tensor construction; wrapping and
exporting counts never copy them again. Counts `[3,2]` exclude the NaN padded suffix from recurrence.
`rnnScan(4)` produces dense hidden values `[3,2,4]`; `finalHidden()` selects each row's semantic
final state `[2,4]`, and `linear(1)` produces one score per row, `[2,1]`.
With seed 42, the qualified CPU route gives finite scores approximately
`[-0.47897735, -0.6437197]`. These are untrained model outputs, not calibrated predictions.

`gruScan(4)` and `lstmScan(4)` use the same values/lengths workflow. Without `finalHidden()` the
scan returns dense `[T,B,H]`, not compact outputs; `lastTimeStep()` selects `T-1`, not each row's
last valid transition. A zero-length row returns its zero initial hidden state at the scan's
final-state port, although an ordinary classifier bias can subsequently make its score nonzero.

Keep the same Model open to reuse its preparation for fresh values and counts at the same static
signature. Every replacement must match shape, dtype, layout presence/geometry and gradient
eligibility exactly; changing counts does not authorize arbitrary T, B or F. For direct Engine
reuse, prepare once and pass fresh values in `compiled.inputs()` order to `runValues`; preserve
caller storage until its `RunResult` closes. Data performs neither binding nor preparation.

This first execution slice is inference-only, canonical contiguous offset-zero non-view FLOAT32
`[T,B,F]` with `requiresGrad=false`, RIGHT/TIME_MAJOR metadata and positive static T/B/F for the
fluent builder, using existing scalar CPU scan kernels. Empty count export does not widen that
builder contract; advanced zero-T/B scans remain separate. Expression scan outputs retain unresolved
Model layout and gradient eligibility from their floating roles (including trainable parameters).
Lengths never contribute gradients; detached compute results do not imply backpropagation through
time (BPTT) support. No flags are cleared to bypass admission.

Export does not enable a fluent Text IDs → Embedding → scan chain. ModelBuilder input is currently
floating-only, has no Embedding recipe and instantiates recipes with the original input dtype.
That integer-to-floating transition needs a separately selected NN capability.

## Integral sequence batching

Use `LongSequenceBatcher` to copy long-array sequences into a fresh INT64 Tensor.
Padding value, side and logical axis order have no defaults. The configuration is immutable;
its reusable builder is not thread-safe. Later builder changes leave existing configurations
unchanged. This example requires only Data and its exported Model API:

```java
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

The output is `[2, 1]` followed by `[2, 3, 0, 1]`: Shape is `[2,2]` and the
second row has one left padding position. With `.rightPadded()`, the flat values become
`[2, 3, 1, 0]` without changing counts. A real zero inside an input row remains valid data.
For count L at time extent T, LEFT declares `[T-L,T)` and RIGHT declares `[0,L)`.
The batcher never truncates and preserves both row order and element order.

B is list size and T is the maximum row length. Empty input yields B=0,T=0; all-empty
rows yield T=0 in either order and side. The result has contiguous offset-zero rank-two
layout, no gradients or provenance, and writable garbage-collected heap storage.
One Model allocation is filled directly; no full flat staging/import copy is made.
There is no Arena, close operation, or additional storage-lifetime owner.

Callers coordinate list and row mutation throughout the synchronous call; it is not a
transactional snapshot. Once the call returns, no caller list or arrays are retained, and their
mutation cannot change the output. Repeated calls allocate independent values.
Both list passes use sequential traversal, avoiding quadratic indexed access on linked lists.

Null input throws `NullPointerException("sequences")`; a null row throws
`NullPointerException` identifying its index. Geometry is checked incrementally before
batch-sized metadata or output allocation, so a first row proving oversized geometry rejects
without scanning later rows. Checked B*T must fit `Integer.MAX_VALUE` elements, and the byte
span uses checked long arithmetic, not an int byte cap. Invalid geometry throws
`IllegalArgumentException`; Model allocation failures, including heap exhaustion, propagate.

LEFT metadata does not extend recurrent scans: they still require valid prefixes with right padding.
The explicit count export above does not add runtime binding, tokenization, execution or persistence.

## Related contracts

- [Public API status](public-api.md#current-public-contracts)
- [Tensor construction and storage](tensor-api.md)
- [Host sequence metadata terminology](../glossary.md#host-sequence-metadata)
- [Data module boundary](../architecture/module-boundaries.md#extensionsdata)
- [Authoritative architecture root](../../ARCHITECTURE.md#scope-indexed-normative-contracts)
