package io.github.pho001.synaptik.model.tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;




import java.util.Optional;

import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorScalarElementwiseTest {
    private static final List<ScalarCall> SCALAR_CALLS = List.of(
            new ScalarCall("add", ScalarElementwiseKind.ADD, Tensor::add),
            new ScalarCall("sub", ScalarElementwiseKind.SUB, Tensor::sub),
            new ScalarCall("mul", ScalarElementwiseKind.MUL, Tensor::mul),
            new ScalarCall("div", ScalarElementwiseKind.DIV, Tensor::div),
            new ScalarCall("minimum", ScalarElementwiseKind.MIN, Tensor::minimum),
            new ScalarCall("maximum", ScalarElementwiseKind.MAX, Tensor::maximum),
            new ScalarCall("pow", ScalarElementwiseKind.POW, Tensor::pow));
    private static final List<ScalarCall> INTEGRAL_CALLS = List.of(
            new ScalarCall("add", ScalarElementwiseKind.ADD, Tensor::add),
            new ScalarCall("sub", ScalarElementwiseKind.SUB, Tensor::sub),
            new ScalarCall("mul", ScalarElementwiseKind.MUL, Tensor::mul),
            new ScalarCall("minimum", ScalarElementwiseKind.MIN, Tensor::minimum),
            new ScalarCall("maximum", ScalarElementwiseKind.MAX, Tensor::maximum));

    

    @Test
    void mapsTypedOverloadsToExactKindsAttributesAndOneInputProvenance() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        ScalarValue value = ScalarValue.float32(Float.intBitsToFloat(0xFFC0_1234));

        for (ScalarCall call : SCALAR_CALLS) {
            Tensor result = call.apply(input, value);
            TensorProvenance provenance = result.provenance().orElseThrow();
            ScalarValueAttrs attrs = (ScalarValueAttrs) provenance.operation().attrs();
            assertAll(
                    () -> assertSame(call.kind(), provenance.operation().kind()),
                    () -> assertSame(value, attrs.value()),
                    () -> assertEquals(List.of(input), provenance.inputs()),
                    () -> assertSame(input, provenance.inputs().getFirst()),
                    () -> assertEquals(0, provenance.outputIndex()),
                    () -> assertEquals(1, provenance.producer().outputCount()),
                    () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
        }

        ScalarValue min = ScalarValue.float32(-0.0f);
        ScalarValue max = ScalarValue.float32(Float.intBitsToFloat(0x7FC0_5678));
        Tensor clamped = input.clamp(min, max);
        TensorProvenance provenance = clamped.provenance().orElseThrow();
        ClampRangeAttrs attrs = (ClampRangeAttrs) provenance.operation().attrs();
        assertAll(
                () -> assertSame(ScalarElementwiseKind.CLAMP, provenance.operation().kind()),
                () -> assertSame(min, attrs.minValue()),
                () -> assertSame(max, attrs.maxValue()),
                () -> assertEquals(List.of(input), provenance.inputs()));

        Tensor right = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        assertAll(
                () -> assertSame(BinaryArithmeticKind.ADD,
                        input.add(right).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.SUB,
                        input.sub(right).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.MUL,
                        input.mul(right).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.DIV,
                        input.div(right).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.MIN,
                        input.minimum(right).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.MAX,
                        input.maximum(right).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.POW,
                        input.pow(right).provenance().orElseThrow().operation().kind()));

        Tensor minimumClamp = input.clampMin(value);
        Tensor maximumClamp = input.clampMax(value);
        TensorProvenance minimumClampProvenance = minimumClamp.provenance().orElseThrow();
        TensorProvenance maximumClampProvenance = maximumClamp.provenance().orElseThrow();
        assertAll(
                () -> assertSame(
                        ScalarElementwiseKind.MAX, minimumClampProvenance.operation().kind()),
                () -> assertSame(
                        ScalarElementwiseKind.MIN, maximumClampProvenance.operation().kind()),
                () -> assertSame(value,
                        ((ScalarValueAttrs) minimumClampProvenance.operation().attrs()).value()),
                () -> assertSame(value,
                        ((ScalarValueAttrs) maximumClampProvenance.operation().attrs()).value()),
                () -> assertEquals(List.of(input), minimumClampProvenance.inputs()),
                () -> assertEquals(List.of(input), maximumClampProvenance.inputs()),
                () -> assertEquals(0, minimumClampProvenance.outputIndex()),
                () -> assertEquals(0, maximumClampProvenance.outputIndex()));
    }

    @Test
    void preservesMetadataFreshnessAndExactTypeAcrossFloatingTypes() {
        for (DataType dataType : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            Shape shape = Shape.of(2, 0, 3);
            Tensor input = tensor(dataType, shape, true);
            ScalarValue scalar = scalar(dataType, -0.0);
            ScalarValue min = scalar(dataType, -1.0);
            ScalarValue max = scalar(dataType, 1.0);
            for (ScalarCall call : SCALAR_CALLS) {
                Tensor first = call.apply(input, scalar);
                Tensor second = call.apply(input, scalar);
                assertAll(
                        () -> assertResultMetadata(first, dataType, shape, true),
                        () -> assertNotSame(input, first),
                        () -> assertNotSame(first, second),
                        () -> assertNotEquals(first.id(), second.id()));
            }
            assertResultMetadata(input.clamp(min, max), dataType, shape, true);
        }
    }

    @Test
    void acceptsSelectedExactIntegralScalarsAndOneBoundClampConveniences() {
        Shape shape = Shape.of(2, 0, 3);

        for (DataType dataType : List.of(DataType.INT32, DataType.INT64)) {
            Tensor input = tensor(dataType, shape, false);
            ScalarValue value = dataType == DataType.INT32
                    ? ScalarValue.int32(Integer.MAX_VALUE)
                    : ScalarValue.int64(Long.MIN_VALUE);

            for (ScalarCall call : INTEGRAL_CALLS) {
                Tensor result = call.apply(input, value);
                TensorProvenance provenance = result.provenance().orElseThrow();
                ScalarValueAttrs attrs = (ScalarValueAttrs) provenance.operation().attrs();

                assertAll(
                        () -> assertResultMetadata(result, dataType, shape, false),
                        () -> assertSame(call.kind(), provenance.operation().kind()),
                        () -> assertSame(value, attrs.value()),
                        () -> assertEquals(List.of(input), provenance.inputs()),
                        () -> assertEquals(0, provenance.outputIndex()),
                        () -> assertEquals(1, provenance.producer().outputCount()));
            }

            Tensor lowerBound = input.clampMin(value);
            Tensor upperBound = input.clampMax(value);
            assertAll(
                    () -> assertSame(ScalarElementwiseKind.MAX,
                            lowerBound.provenance().orElseThrow().operation().kind()),
                    () -> assertSame(ScalarElementwiseKind.MIN,
                            upperBound.provenance().orElseThrow().operation().kind()),
                    () -> assertSame(value, ((ScalarValueAttrs) lowerBound.provenance()
                            .orElseThrow().operation().attrs()).value()),
                    () -> assertSame(value, ((ScalarValueAttrs) upperBound.provenance()
                            .orElseThrow().operation().attrs()).value()));
        }
    }

    @Test
    void integralAddSubMulRecordFixedWidthModularRequestsWithoutEvaluation() {
        Tensor input = tensor(DataType.INT32, Shape.scalar(), false);
        List<Tensor> requests = List.of(
                input.add(ScalarValue.int32(Integer.MAX_VALUE)),
                input.sub(ScalarValue.int32(Integer.MIN_VALUE)),
                input.mul(ScalarValue.int32(-1)));

        assertAll(
                () -> assertSame(ScalarElementwiseKind.ADD,
                        requests.get(0).provenance().orElseThrow().operation().kind()),
                () -> assertSame(ScalarElementwiseKind.SUB,
                        requests.get(1).provenance().orElseThrow().operation().kind()),
                () -> assertSame(ScalarElementwiseKind.MUL,
                        requests.get(2).provenance().orElseThrow().operation().kind()),
                () -> assertTrue(requests.stream()
                        .allMatch(result -> result.hostStorage().isEmpty())));
    }

    @Test
    void doubleOverloadsAreExactFloat64Adapters() {
        Tensor float64 = tensor(DataType.FLOAT64, Shape.scalar(), false);
        double rawNaN = Double.longBitsToDouble(0xFFF8_0000_0000_1234L);

        List<Tensor> scalarResults = List.of(
                float64.add(rawNaN),
                float64.sub(rawNaN),
                float64.mul(rawNaN),
                float64.div(rawNaN),
                float64.minimum(rawNaN),
                float64.maximum(rawNaN),
                float64.pow(rawNaN));
        ClampRangeAttrs range = (ClampRangeAttrs) float64.clamp(-0.0d, rawNaN)
                .provenance().orElseThrow().operation().attrs();

        for (Tensor result : scalarResults) {
            ScalarValueAttrs attrs = (ScalarValueAttrs) result.provenance()
                    .orElseThrow().operation().attrs();
            assertEquals(ScalarValue.float64(rawNaN), attrs.value());
        }
        assertAll(
                () -> assertEquals(ScalarValue.float64(-0.0d), range.minValue()),
                () -> assertEquals(ScalarValue.float64(rawNaN), range.maxValue()),
                () -> assertSame(ScalarElementwiseKind.MAX,
                        float64.clampMin(rawNaN).provenance().orElseThrow().operation().kind()),
                () -> assertSame(ScalarElementwiseKind.MIN,
                        float64.clampMax(rawNaN).provenance().orElseThrow().operation().kind()));
    }

    @Test
    void validatesInExactOrderAndAllocatesNoIdentityOnFailure() throws Exception {
        Tensor floating = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor integral = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor bool = tensor(DataType.BOOL, Shape.scalar(), false);
        AtomicLong nextId = nextTensorIdState();
        long beforeFailures = nextId.get();

        assertFailure(NullPointerException.class, () -> TensorScalarExpressions.applyScalar(null, null, null));
        assertFailure(NullPointerException.class, () -> TensorScalarExpressions.applyScalar(floating, null, null));
        assertFailure(NullPointerException.class, () -> TensorScalarExpressions.applyScalar(
                floating, ScalarElementwiseKind.MUL, null));
        assertFailure(IllegalArgumentException.class, () -> TensorScalarExpressions.applyScalar(
                integral, ScalarElementwiseKind.CLAMP, ScalarValue.int32(1)));
        assertFailure(IllegalArgumentException.class, () -> TensorScalarExpressions.applyScalar(
                integral, ScalarElementwiseKind.ADD, ScalarValue.float64(1.0)));
        assertFailure(IllegalArgumentException.class, () -> TensorScalarExpressions.applyScalar(
                bool, ScalarElementwiseKind.ADD, ScalarValue.bool(true)));
        assertFailure(IllegalArgumentException.class, () -> TensorScalarExpressions.applyScalar(
                integral, ScalarElementwiseKind.DIV, ScalarValue.float64(1.0)));
        assertFailure(IllegalArgumentException.class, () -> TensorScalarExpressions.applyScalar(
                integral, ScalarElementwiseKind.POW, ScalarValue.float64(1.0)));
        assertFailure(IllegalArgumentException.class, () -> TensorScalarExpressions.applyClamp(
                integral, ScalarValue.int32(2), ScalarValue.int32(1)));
        assertFailure(IllegalArgumentException.class, () -> floating.add(1.0d));
        assertFailure(IllegalArgumentException.class, () -> floating.clamp(ScalarValue.float64(0.0), ScalarValue.float64(1.0)));
        assertFailure(IllegalArgumentException.class, () -> floating.clamp(ScalarValue.float32(0.0f), ScalarValue.float64(1.0)));
        assertFailure(IllegalArgumentException.class, () -> floating.clamp(ScalarValue.float32(2.0f), ScalarValue.float32(1.0f)));
        assertEquals(beforeFailures, nextId.get());
    }

    @Test
    void preservesInputMetadataStorageAndContents() {
        float[] values = {-1.0f, 0.0f, 2.0f};
        Shape shape = Shape.of(3);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), true);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
        Tensor leaf = TensorFactory.create(descriptor, Optional.of("leaf"), Optional.of(storage));
        Operation inputOperation = new Operation(
                ScalarElementwiseKind.MUL,
                new ScalarValueAttrs(ScalarValue.float32(2.0f)));
        Tensor input = TensorFactory.createDerived(
                descriptor, Optional.of("derived"), inputOperation, List.of(leaf));
        TensorProvenance provenance = input.provenance().orElseThrow();
        input.replaceHostStorage(storage);

        Tensor result = input.clamp(
                ScalarValue.float32(-1.0f), ScalarValue.float32(1.0f));

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(provenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(input, result.provenance().orElseThrow().inputs().getFirst()),
                () -> assertArrayEquals(new float[] {-1.0f, 0.0f, 2.0f}, values));
    }

    private static ScalarValue scalar(DataType dataType, double value) {
        return switch (dataType) {
            case FLOAT64 -> ScalarValue.float64(value);
            case FLOAT32 -> ScalarValue.float32((float) value);
            case BFLOAT16 -> ScalarValue.bfloat16((float) value);
            case FLOAT16 -> ScalarValue.float16((float) value);
            case FLOAT8_E4M3 -> ScalarValue.float8E4m3((float) value);
            case FLOAT8_E5M2 -> ScalarValue.float8E5m2((float) value);
            default -> throw new IllegalArgumentException("not floating: " + dataType);
        };
    }

    

    

    private static void assertResultMetadata(
            Tensor result, DataType dataType, Shape shape, boolean requiresGrad) {
        assertAll(
                () -> assertSame(dataType, result.descriptor().dataType()),
                () -> assertSame(shape, result.descriptor().shape()),
                () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    private static <T extends Throwable> void assertFailure(
                Class<T> type, org.junit.jupiter.api.function.Executable executable) {
            assertThrows(type, executable);
            
        }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    @FunctionalInterface
    private interface ScalarFunction {
        Tensor apply(Tensor input, ScalarValue value);
    }

    private record ScalarCall(
            String methodName, ScalarElementwiseKind kind, ScalarFunction function) {
        private Tensor apply(Tensor input, ScalarValue value) {
            return function.apply(input, value);
        }
    }
}
