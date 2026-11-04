# Build a dense [Tensor](../glossary.md#tensor) expression

## Outcome

You will create an independent dense [`FLOAT32`](../glossary.md#term-float32) [Tensor](../glossary.md#tensor) with copied [host storage](../glossary.md#host-storage), build one
[elementwise](../glossary.md#term-elementwise) negation expression, and inspect both objects' [metadata](../glossary.md#term-metadata) and expression [provenance](../glossary.md#provenance).
The example constructs [Model](../glossary.md#term-model) state only; it does not evaluate the expression.

The workflow is:

```text
float[] values --copy--> host-backed leaf Tensor --neg()--> storage-free expression Tensor
                         identity + descriptor          descriptor + provenance
```

Here the Java application calls `TensorFactory.fromFlatArray` with its array and descriptor.
The factory copies the array into new heap storage, then associates that storage with a new leaf
Tensor. The original array remains the application's and can change independently.
The Tensor does not manage the storage object's lifetime or become closeable because it has
storage. Calling `input.neg()` records a producing operation and its input reference, but
attaches no computed values to the expression.

A **tensor** is an ordered numerical value with one or more [coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix) directions, or no
directions for a [scalar](../glossary.md#term-scalar). Its **[shape](../glossary.md#shape)** lists the size of each [axis](../glossary.md#term-axis); **[rank](../glossary.md#term-rank)** counts those axes.
A **[descriptor](../glossary.md#term-descriptor)** combines those sizes with the [element](../glossary.md#term-element) type, storage geometry when known, and
[gradient](../glossary.md#term-gradient) eligibility. A **leaf** has no producing [operation](../glossary.md#operation); **provenance** records the [producer](../glossary.md#term-producer)
and inputs of a derived expression. Negation changes a number's sign, but constructing `neg()`
only records that rule. See the [expression and shape explanation](../explanation/tensors-and-gradients.md)
for the existing coordinate and broadcast visual.

## Prerequisites

- [JDK](../glossary.md#term-java-development-kit-jdk) 26.
- The `modules:model` [artifact](../glossary.md#term-artifact) on the [compile](../glossary.md#compile) and [runtime](../glossary.md#term-runtime) classpaths. Inside this repository, [run](../glossary.md#term-run)
  `./gradlew :modules:model:classes` before compiling the example directly.
- In another [Gradle](../glossary.md#term-gradle) [module](../glossary.md#term-module), add `implementation(project(":modules:model"))`. See
  [Getting started](../getting-started.md) for the repository setup.

## Inputs and setup

The input has [data type](../glossary.md#data-type--datatype) [`FLOAT32`](../glossary.md#term-float32), [Shape](../glossary.md#shape) `[2, 3]`, [canonical](../glossary.md#term-canonical-form) dense [row-major](../glossary.md#term-row-major) [layout](../glossary.md#layout), no [gradient](../glossary.md#term-gradient)
request, label `input`, and the six row-major values `1.0, -2.0, 3.5, 4.0, 0.0, -6.0`.

**Dense row-major** means values occupy consecutive storage positions, with one whole row before
the next. FLOAT32 uses four bytes per [element](../glossary.md#term-element), like Java `float`; the six elements therefore need
24 bytes. **[Host](../glossary.md#term-host)** storage is accessible from the Java process. [Borrowing](../glossary.md#term-borrowing) that storage does not
transfer responsibility for its [lifetime](../glossary.md#term-lifetime); the
[ownership visual](../explanation/ownership.md#a-result-is-not-always-an-independent-value)<span data-glossary-navigation="true"> ([ownership glossary](../glossary.md#term-ownership-meanings))</span>
separates stored inputs from independent [detached](../glossary.md#detached-model-results) outputs.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class TensorGuideExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        float[] source = {1.0f, -2.0f, 3.5f, 4.0f, 0.0f, -6.0f};

        // Copy the application array so later source-array writes cannot change input values.
        Tensor input = TensorFactory.fromFlatArray(
                descriptor, Optional.of("input"), source);
        HostTensorStorage storage = input.hostStorage().orElseThrow();

        Tensor expression = input.neg();
        // Describe negation only: no numerical evaluation or computed result storage yet.
        TensorProvenance origin = expression.provenance().orElseThrow();

        System.out.println("inputType=" + input.descriptor().dataType());
        System.out.println("inputLabel=" + input.label().orElseThrow());
        System.out.println("inputLayoutResolved=" + input.descriptor().layout().isPresent());
        System.out.println("inputCapacity=" + storage.elementCapacity());
        System.out.println("inputBytes=" + storage.byteSize());
        System.out.println("inputUsesSourceArray="
                + (storage.segment().heapBase().orElseThrow() == source));
        System.out.println("inputHasProvenance=" + input.provenance().isPresent());
        System.out.println("expressionKind=" + origin.operation().kind());
        System.out.println("expressionUsesInput=" + (origin.inputs().getFirst() == input));
        System.out.println("expressionShapeMatches="
                + expression.descriptor().shape().equals(shape));
        System.out.println("expressionLayoutResolved="
                + expression.descriptor().layout().isPresent());
        System.out.println("expressionHasStorage=" + expression.hostStorage().isPresent());
        System.out.println("expressionCanonicalOutput="
                + (origin.producer().output(origin.outputIndex()) == expression));
    }
}
```

## Steps

1. `Shape.of(2, 3)` and `LayoutDescriptor.contiguous(shape)` establish fully resolved logical and
   dense row-major geometry. <span data-glossary-navigation="true"> ([row-major glossary](../glossary.md#term-row-major))</span>`TensorDescriptor` combines that geometry with the element type and
   gradient-eligibility flag.<span data-glossary-navigation="true"> ([element glossary](../glossary.md#term-element); [gradient glossary](../glossary.md#term-gradient))</span>
2. `TensorFactory.fromFlatArray` validates the descriptor and six-element source, allocates a new
   matching heap array, copies the values, and returns a provenance-free leaf Tensor. The source
   array remains caller-owned and is not retained.<span data-glossary-navigation="true"> ([descriptor glossary](../glossary.md#term-descriptor); [heap glossary](../glossary.md#term-heap); [provenance glossary](../glossary.md#provenance); [leaf glossary](../glossary.md#term-leaf))</span>
3. `input.neg()` creates a fresh [Tensor](../glossary.md#tensor) for one `NEG` operation occurrence. It preserves the input
   type and Shape, leaves result layout unresolved, and attaches neither storage nor a result
   value.<span data-glossary-navigation="true"> ([operation glossary](../glossary.md#operation); [occurrence glossary](../glossary.md#term-occurrence); [Shape glossary](../glossary.md#shape); [layout glossary](../glossary.md#layout))</span>
4. `TensorProvenance` identifies output zero of that exact producer occurrence. Its ordered input
   list retains the exact <span data-glossary-navigation="true"> ([producer glossary](../glossary.md#term-producer))</span>`input` object, and the producer returns the exact canonical expression
   wrapper for the selected output position.<span data-glossary-navigation="true"> ([canonical glossary](../glossary.md#term-canonical-form))</span>

## Expected result

Running the class prints:

```text
inputType=FLOAT32
inputLabel=input
inputLayoutResolved=true
inputCapacity=6
inputBytes=24
inputUsesSourceArray=false
inputHasProvenance=false
expressionKind=NEG
expressionUsesInput=true
expressionShapeMatches=true
expressionLayoutResolved=false
expressionHasStorage=false
expressionCanonicalOutput=true
```

The capacity is six elements. At four bytes per FLOAT32 element, $6\cdot4=24$ bytes.
`inputUsesSourceArray=false` confirms that the destination storage uses a different [heap](../glossary.md#term-heap) backing
array; the flat-import contract additionally guarantees that it does not retain the source array.
The [leaf](../glossary.md#term-leaf) has no [provenance](../glossary.md#provenance) because it is an input, while the derived [Tensor](../glossary.md#tensor) records one exact
`NEG` [occurrence](../glossary.md#term-occurrence) and its exact input. The expression observations prove construction and
provenance only; they do not prove [evaluation](../glossary.md#term-evaluation) or [backend](../glossary.md#backend) support.

Tensor [identity](../glossary.md#term-identity-meanings), [descriptor](../glossary.md#term-descriptor), label, and provenance are [immutable](../glossary.md#term-immutable). The optional [host](../glossary.md#term-host)-storage
association is the sole [mutable](../glossary.md#term-mutable) Tensor state: its synchronized methods serialize [snapshot](../glossary.md#term-snapshot),
replacement, and clearing with respect to one another. That synchronization does not protect
raw-memory access, sharing through another Tensor, Arena closure, or thread accessibility, and it
does not extend the storage [lifetime](../glossary.md#term-lifetime). Factory-created heap storage has an automatic [JVM](../glossary.md#term-java-virtual-machine-jvm) lifetime;
when callers instead attach an arena-backed `HostTensorStorage`, they continue to own that arena,
must keep its segment alive, and may access it only from threads permitted by the segment [scope](../glossary.md#term-scope).

## Low floating types and raw bytes

[`BFLOAT16`](../glossary.md#term-bfloat16), [`FLOAT16`](../glossary.md#term-float16), `FLOAT8_E4M3` and `FLOAT8_E5M2` are distinct floating,
[gradient](../glossary.md#term-gradient)-eligible storage types. [OFP8](../glossary.md#term-ofp8) means the unscaled eight-bit floating formats here, not
integer quantization: no scale, calibration, clipping or automatic [loss](../glossary.md#loss) scaling is attached.
Both [FP8](../glossary.md#term-fp8) types use one raw `byte` per [element](../glossary.md#term-element) and a `byte[]` [heap](../glossary.md#term-heap) [carrier](../glossary.md#term-storage-carrier), not `short[]`.
Interpret a diagnostic raw byte as unsigned with `bits & 0xff`.

[Floating-point](../glossary.md#term-floating-point) types encode a sign, a size scale (**[exponent](../glossary.md#term-exponent)**) and significant digits
(**fraction**). Fewer bits generally permit fewer distinct values and a smaller range.
**[Rounding](../glossary.md#term-rounding)** chooses a representable value when the exact number does not fit; [ties-to-even](../glossary.md#term-ties-to-even)
chooses the candidate with an even final retained [binary](../glossary.md#term-binary-descriptor-and-metadata) digit when exactly halfway between two.
**[Narrowing](../glossary.md#term-narrowing)** converts to a smaller representation; **[widening](../glossary.md#term-widening)** converts to a larger one.
A **[subnormal](../glossary.md#term-subnormal)** uses a special representation for tiny values near zero. **[NaN](../glossary.md#term-nan)** means “[not a
number](../glossary.md#nan-infinity-and-signed-zero)”; [infinity](../glossary.md#term-infinity) is a separate special value where the format supports it. **[Overflow](../glossary.md#term-overflow)** means
the result's magnitude is beyond the format's [finite](../glossary.md#term-finite) range.

The public `io.github.pho001.synaptik.model.datatype.Float8E4M3Bits` and
`Float8E5M2Bits` helpers each expose `toFloat(byte)`, `fromFloat(float)` and
`fromDouble(double)`. Decoding accepts all 256 byte patterns; numeric encoding uses
non-[saturating](../glossary.md#saturation-pole-and-extremum) [round-to-nearest, ties-to-even](../glossary.md#floating-point-precision-and-rounding):

| Format | Representation and numeric conversion |
|---|---|
| [E4M3](../glossary.md#term-e4m3) | Sign/exponent/fraction 1/4/3, [bias](../glossary.md#term-bias) 7, signed zeros and subnormals, finite maximum 448, no infinity. NaNs are `0x7f`/`0xff`; numeric NaN, infinity input and overflow become [canonical](../glossary.md#term-canonical-form) `0x7f`. Magnitude 464 is the inclusive tie that rounds to 448; larger magnitudes become NaN. |
| [E5M2](../glossary.md#term-e5m2) | Sign/exponent/fraction 1/5/2, bias 15, signed zeros and subnormals, finite maximum 57344. [Infinities](../glossary.md#nan-infinity-and-signed-zero) are `0x7c`/`0xfc`; exponent 31 with nonzero fraction is NaN. Numeric NaN becomes `0x7e`; magnitude at or above the 61440 overflow tie becomes signed infinity. |

In this table, **bias** is the fixed offset used to encode an exponent, not an adjustable
neural-network [bias parameter](../glossary.md#bias-and-activation). A NaN **[payload](../glossary.md#term-nan-payload)** is extra
information in its fraction bits, not a file payload or an ordinary numeric value.
**Canonical** NaN means the chosen standard bit pattern used by numeric conversion;
raw import instead preserves the exact supplied pattern.
**Non-saturating** conversion does not clamp overflow to the largest finite value.

`ScalarValue.float8E4m3(float)` and `float8E5m2(float)` encode numeric values.
`ScalarValue.float8E4m3Bits(byte)` / `float8E5m2Bits(byte)` and the corresponding
instance bit accessors preserve exact raw patterns, including NaN sign/payload.
`TensorFactory.scalarFloat8E4m3(float, label, requiresGrad)` and
`scalarFloat8E5m2(float, label, requiresGrad)` create [rank-zero](../glossary.md#axis-rank-scalar-vector-and-matrix) leaves;
`fullFloat8E4m3(shape, float, label, requiresGrad)` and
`fullFloat8E5m2(shape, float, label, requiresGrad)` fill fully static [Shapes](../glossary.md#shape).
These numeric factories round once and allocate independent storage.

For exact import, use `TensorFactory.fromFlatArray(descriptor, label, byte[])` with the
FP8 [descriptor](../glossary.md#term-descriptor) and a resolved dense-[contiguous](../glossary.md#term-contiguous) [layout](../glossary.md#layout). It copies exactly one byte per logical
element into independent storage without retaining the source array or normalizing any bit.
The same overload normalizes nonzero bytes only when the descriptor is [BOOL](../glossary.md#data-type--datatype).
[Engine](../glossary.md#term-engine) `HostTensorValue.bytes()` publishes FP8 as exact raw bytes in its [detached](../glossary.md#detached-model-results) read-only
canonical [row-major](../glossary.md#term-row-major) payload; multi-byte types use [big-endian](../glossary.md#text-encoding-and-file-boundaries) elements.

Same-type casts and raw movement/[persistence](../glossary.md#term-persistence) preserve every bit. Direct [FLOAT64](../glossary.md#term-float64) and integral
casts round to the [target](../glossary.md#term-target) without a lossy [FLOAT32](../glossary.md#term-float32) intermediate; `fromDouble` likewise rounds
directly. Widening preserves represented NaN sign/payload bits where the destination permits.
Numeric narrowing may canonicalize NaNs and is not a substitute for a raw import.

[Homogeneous](../glossary.md#term-homogeneous) low arithmetic retains the low result type, computes with FLOAT32 working values
and accumulators, and narrows once at each declared ordinary [operation](../glossary.md#operation), saved or public boundary.
Unlike low operands promote to FLOAT32; FLOAT32 dominates low operands and FLOAT64 dominates
all other floating types. Exact sign operations, comparisons, selection, [masks](../glossary.md#term-boolean-mask) and indices
remain exact. An E4M3 infinity-class formula result, [pole](../glossary.md#term-pole) or empty floating [extremum](../glossary.md#term-extremum) narrows to
NaN; a nonempty extremum still selects an actual represented input winner.
[CPU](../glossary.md#term-cpu) execution reuses existing families, shared [Compiler](../glossary.md#term-compiler) [AD](../glossary.md#autograd), [Scan](../glossary.md#term-cumulative-scan) and recurrent [owners](../glossary.md#term-owner) under
their geometry and [derivative](../glossary.md#term-derivative) constraints. Native [oneDNN](../glossary.md#term-onednn) availability is separate; no conversion
of FP8 inputs to FLOAT32 constitutes native FP8 execution. [Metal](../glossary.md#term-metal) retains its seven-carrier
[scope](../glossary.md#term-scope) without FP8 and its independently qualified BFLOAT16/FLOAT16 restrictions.

A **working value** is the representation used while calculating; an **[accumulator](../glossary.md#accumulator-type)** holds a
running total. **Promotion** chooses a common calculation type for unlike operands. These rules
explain why a low-storage result may be calculated using FLOAT32 without becoming a stored
FLOAT32 [Tensor](../glossary.md#tensor). A **pole** is an input at which a formula becomes unbounded, and an **extremum**
is a minimum or maximum. Those numerical words describe the result rules, not extra execution
support on a [backend](../glossary.md#backend).

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `tensor allocation requires a resolved layout` | `TensorFactory.allocate` received a [descriptor](../glossary.md#term-descriptor) whose [layout](../glossary.md#layout) is unresolved. | Supply a descriptor with compatible resolved geometry, such as `LayoutDescriptor.contiguous` for a fully static [Shape](../glossary.md#shape). |
| Flat import rejects the descriptor or source length | The layout is not dense-[contiguous](../glossary.md#term-contiguous), the primitive [carrier](../glossary.md#term-storage-carrier) does not match the [data type](../glossary.md#data-type--datatype), or the source count differs from the logical [element](../glossary.md#term-element) count. | Use a matching primitive array with exactly one value per logical position and a resolved dense-contiguous descriptor. |
| A [binary operation](../glossary.md#term-binary-operation) (two operands), such as `left.add(right)`, throws `IllegalArgumentException` | The operand types cross numeric categories or their Shapes cannot broadcast. | Use compatible floating or signed-integral types and right-aligned broadcast-compatible Shapes. |
| [Engine](../glossary.md#term-engine) preparation or execution rejects a constructed expression | [Model](../glossary.md#term-model) expression construction does not promise a [CPU](../glossary.md#term-cpu) route for every [operation](../glossary.md#operation), type, Shape, or layout. | Check current [backend capability](../glossary.md#backend-capability) and use only an explicitly supported combination. |

## Limitations

- [`Tensor`](../glossary.md#tensor) and `HostTensorStorage` expose raw host storage facts, not typed Tensor element getters
  or setters. Raw <span data-glossary-navigation="true"> ([host storage glossary](../glossary.md#host-storage); [element glossary](../glossary.md#term-element))</span>[`MemorySegment`](../glossary.md#term-memorysegment) access remains subject to [JDK](../glossary.md#term-java-development-kit-jdk) [scope](../glossary.md#term-scope) and thread-access rules.
- Expression construction does not capture, compile, prepare, execute, or materialize a graph.
  The separate Engine lifecycle owns those stages and currently composes one CPU integration.<span data-glossary-navigation="true"> ([capture glossary](../glossary.md#term-capture-meanings); [compile glossary](../glossary.md#compile); [prepare glossary](../glossary.md#prepare); [graph glossary](../glossary.md#graph); [Engine glossary](../glossary.md#term-engine); [lifecycle glossary](../glossary.md#lifecycle); [CPU glossary](../glossary.md#term-cpu))</span>
- Current CPU execution is bounded and fail-closed. The presence of a public Tensor operation does
  not mean that every data type, Shape, layout, or operation combination can execute.<span data-glossary-navigation="true"> ([fail-closed glossary](../glossary.md#term-fail-closed); [operation glossary](../glossary.md#operation); [data type glossary](../glossary.md#data-type--datatype); [Shape glossary](../glossary.md#shape); [layout glossary](../glossary.md#layout))</span>
- [Device](../glossary.md#term-device) [residency](../glossary.md#residency), [runtime](../glossary.md#term-runtime) [buffers](../glossary.md#term-buffer), and prepared resources do not belong to Tensor state.

## Related tasks and [API](../glossary.md#term-application-programming-interface-api) reference

- [Public Tensor state](../reference/tensor/identity.md#public-tensor-state)<span data-glossary-navigation="true"> ([Tensor glossary](../glossary.md#tensor))</span> defines identity, descriptor,
  provenance, and the mutable borrowed-storage association.<span data-glossary-navigation="true"> ([identity glossary](../glossary.md#term-identity-meanings); [descriptor glossary](../glossary.md#term-descriptor); [provenance glossary](../glossary.md#provenance); [mutable glossary](../glossary.md#term-mutable); [borrowed glossary](../glossary.md#ownership-borrowing-and-lease))</span>
- [Complete flat-import example](../reference/tensor/import.md#complete-flat-import-example) covers copied
  primitive-array import and its validation details.<span data-glossary-navigation="true"> ([validation glossary](../glossary.md#term-validation))</span>
- [Unary numeric transforms](../reference/tensor/elementwise.md#unary-numeric-transforms-and-floating-classifications)<span data-glossary-navigation="true"> ([Unary glossary](../glossary.md#term-unary))</span>
  describes `neg()` and the other storage-free expression constructors.
- [Current public Engine lifecycle](../architecture/lifecycle.md#current-public-engine-lifecycle)<span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine); [lifecycle glossary](../glossary.md#lifecycle))</span>
  explains separate compile, prepare, run and materialization. To actually evaluate this guide's
  expression, follow [compile and prepare once](running-models.md#compile-and-prepare-once);
  to supply different stored inputs, use [runValues](running-models.md#run-fresh-values-in-compiled-order).
