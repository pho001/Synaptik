# Build a dense Tensor expression

## Outcome

You will create an independent dense `FLOAT32` Tensor with copied host storage, build one
elementwise negation expression, and inspect both objects' metadata and expression provenance.
The example constructs Model state only; it does not evaluate the expression.

The workflow is:

```text
float[] values --copy--> host-backed leaf Tensor --neg()--> storage-free expression Tensor
                         identity + descriptor          descriptor + provenance
```

The leaf owns stable logical metadata and borrows its associated host-storage object. The derived
Tensor records what to compute and from which exact input, but it has no computed values or host
storage merely because the expression exists.

## Prerequisites

- JDK 26.
- The `modules:model` artifact on the compile and runtime classpaths. Inside this repository, run
  `./gradlew :modules:model:classes` before compiling the example directly.
- In another Gradle module, add `implementation(project(":modules:model"))`. See
  [Getting started](../getting-started.md) for the repository setup.

## Inputs and setup

The input has data type `FLOAT32`, Shape `[2, 3]`, canonical dense row-major layout, no gradient
request, label `input`, and the six row-major values `1.0, -2.0, 3.5, 4.0, 0.0, -6.0`.

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

        Tensor input = TensorFactory.fromFlatArray(
                descriptor, Optional.of("input"), source);
        HostTensorStorage storage = input.hostStorage().orElseThrow();

        Tensor expression = input.neg();
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
   dense row-major geometry. `TensorDescriptor` combines that geometry with the element type and
   gradient-eligibility flag.
2. `TensorFactory.fromFlatArray` validates the descriptor and six-element source, allocates a new
   matching heap array, copies the values, and returns a provenance-free leaf Tensor. The source
   array remains caller-owned and is not retained.
3. `input.neg()` creates a fresh Tensor for one `NEG` operation occurrence. It preserves the input
   type and Shape, leaves result layout unresolved, and attaches neither storage nor a result
   value.
4. `TensorProvenance` identifies output zero of that exact producer occurrence. Its ordered input
   list retains the exact `input` object, and the producer returns the exact canonical expression
   wrapper for the selected output position.

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

The capacity is six complete elements and the byte size is `6 x 4 = 24`.
`inputUsesSourceArray=false` confirms that the destination storage uses a different heap backing
array; the flat-import contract additionally guarantees that it does not retain the source array.
The leaf has no provenance because it is an input, while the derived Tensor records one exact
`NEG` occurrence and its exact input. The expression observations prove construction and
provenance only; they do not prove evaluation or backend support.

Tensor identity, descriptor, label, and provenance are immutable. The optional host-storage
association is the sole mutable Tensor state: its synchronized methods serialize snapshot,
replacement, and clearing with respect to one another. That synchronization does not coordinate
raw-memory access, sharing through another Tensor, arena closure, or thread accessibility, and it
does not extend the storage lifetime. Factory-created heap storage has an automatic JVM lifetime;
when callers instead attach an arena-backed `HostTensorStorage`, they continue to own that arena,
must keep its segment alive, and may access it only from threads permitted by the segment scope.

## Low floating types and raw bytes

`BFLOAT16`, `FLOAT16`, `FLOAT8_E4M3` and `FLOAT8_E5M2` are distinct floating,
gradient-eligible storage types. OFP8 means the unscaled eight-bit floating formats here, not
integer quantization: no scale, calibration, clipping or automatic loss scaling is attached.
Both FP8 types use one raw `byte` per element and a `byte[]` heap carrier, not `short[]`.
Interpret a diagnostic raw byte as unsigned with `bits & 0xff`.

The public `io.github.pho001.synaptik.model.datatype.Float8E4M3Bits` and
`Float8E5M2Bits` helpers each expose `toFloat(byte)`, `fromFloat(float)` and
`fromDouble(double)`. Decoding accepts all 256 byte patterns; numeric encoding uses
non-saturating round-to-nearest, ties-to-even:

| Format | Representation and numeric conversion |
|---|---|
| E4M3 | Sign/exponent/fraction 1/4/3, bias 7, signed zeros and subnormals, finite maximum 448, no infinity. NaNs are `0x7f`/`0xff`; numeric NaN, infinity input and overflow become canonical `0x7f`. Magnitude 464 is the inclusive tie that rounds to 448; larger magnitudes become NaN. |
| E5M2 | Sign/exponent/fraction 1/5/2, bias 15, signed zeros and subnormals, finite maximum 57344. Infinities are `0x7c`/`0xfc`; exponent 31 with nonzero fraction is NaN. Numeric NaN becomes `0x7e`; magnitude at or above the 61440 overflow tie becomes signed infinity. |

`ScalarValue.float8E4m3(float)` and `float8E5m2(float)` encode numeric values.
`ScalarValue.float8E4m3Bits(byte)` / `float8E5m2Bits(byte)` and the corresponding
instance bit accessors preserve exact raw patterns, including NaN sign/payload.
`TensorFactory.scalarFloat8E4m3(float, label, requiresGrad)` and
`scalarFloat8E5m2(float, label, requiresGrad)` create rank-zero leaves;
`fullFloat8E4m3(shape, float, label, requiresGrad)` and
`fullFloat8E5m2(shape, float, label, requiresGrad)` fill fully static Shapes.
These numeric factories round once and allocate independent storage.

For exact import, use `TensorFactory.fromFlatArray(descriptor, label, byte[])` with the
FP8 descriptor and a resolved dense-contiguous layout. It copies exactly one byte per logical
element into independent storage without retaining the source array or normalizing any bit.
The same overload normalizes nonzero bytes only when the descriptor is BOOL.
Engine `HostTensorValue.bytes()` publishes FP8 as exact raw bytes in its detached read-only
canonical row-major payload; multi-byte types use big-endian elements.

Same-type casts and raw movement/persistence preserve every bit. Direct FLOAT64 and integral
casts round to the target without a lossy FLOAT32 intermediate; `fromDouble` likewise rounds
directly. Widening preserves represented NaN sign/payload bits where the destination permits.
Numeric narrowing may canonicalize NaNs and is not a substitute for a raw import.

Homogeneous low arithmetic retains the low result type, computes with FLOAT32 working values
and accumulators, and narrows once at each declared ordinary operation, saved or public boundary.
Unlike low operands promote to FLOAT32; FLOAT32 dominates low operands and FLOAT64 dominates
all other floating types. Exact sign operations, comparisons, selection, masks and indices
remain exact. An E4M3 infinity-class formula result, pole or empty floating extremum narrows to
NaN; a nonempty extremum still selects an actual represented input winner.
CPU execution reuses existing families, shared Compiler AD, Scan and recurrent owners under
their geometry and derivative constraints. Native oneDNN availability is separate; no conversion
of FP8 inputs to FLOAT32 constitutes native FP8 execution. Metal retains its seven-carrier
scope without FP8 and its independently qualified BFLOAT16/FLOAT16 restrictions.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `tensor allocation requires a resolved layout` | `TensorFactory.allocate` received a descriptor whose layout is unresolved. | Supply a descriptor with compatible resolved geometry, such as `LayoutDescriptor.contiguous` for a fully static Shape. |
| Flat import rejects the descriptor or source length | The layout is not dense-contiguous, the primitive carrier does not match the data type, or the source count differs from the logical element count. | Use a matching primitive array with exactly one value per logical position and a resolved dense-contiguous descriptor. |
| A binary expression such as `left.add(right)` throws `IllegalArgumentException` | The operand types cross numeric categories or their Shapes cannot broadcast. | Use compatible floating or signed-integral types and right-aligned broadcast-compatible Shapes. |
| Engine preparation or execution rejects a constructed expression | Model expression construction does not promise a CPU route for every operation, type, Shape, or layout. | Check current backend capability and use only an explicitly supported combination. |

## Limitations

- `Tensor` and `HostTensorStorage` expose raw host storage facts, not typed Tensor element getters
  or setters. Raw `MemorySegment` access remains subject to JDK scope and thread-access rules.
- Expression construction does not capture, compile, prepare, execute, or materialize a graph.
  The separate Engine lifecycle owns those stages and currently composes one CPU integration.
- Current CPU execution is bounded and fail-closed. The presence of a public Tensor operation does
  not mean that every data type, Shape, layout, or operation combination can execute.
- Device residency, runtime buffers, and prepared resources do not belong to Tensor state.

## Related tasks and API reference

- [Public Tensor state](../reference/tensor/identity.md#public-tensor-state) defines identity, descriptor,
  provenance, and the mutable borrowed-storage association.
- [Complete flat-import example](../reference/tensor/import.md#complete-flat-import-example) covers copied
  primitive-array import and its validation details.
- [Unary numeric transforms](../reference/tensor/elementwise.md#unary-numeric-transforms-and-floating-classifications)
  describes `neg()` and the other storage-free expression constructors.
- [Current public Engine lifecycle](../architecture/lifecycle.md#current-public-engine-lifecycle)
  explains the separate compile, session, run, and materialization boundary.
