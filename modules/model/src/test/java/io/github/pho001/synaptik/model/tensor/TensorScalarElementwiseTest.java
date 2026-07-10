package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
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

    @Test
    void helperAndTensorOverloadsHaveExactlyTheRequiredShape() throws Exception {
        var constructors = TensorScalarExpressions.class.getDeclaredConstructors();
        var methods = TensorScalarExpressions.class.getDeclaredMethods();
        Method applyScalar = TensorScalarExpressions.class.getDeclaredMethod(
                "applyScalar", Tensor.class, ScalarElementwiseKind.class, ScalarValue.class);
        Method applyClamp = TensorScalarExpressions.class.getDeclaredMethod(
                "applyClamp", Tensor.class, ScalarValue.class, ScalarValue.class);
        Method create = TensorScalarExpressions.class.getDeclaredMethod(
                "create", Tensor.class, DataType.class, Operation.class);

        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorScalarExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorScalarExpressions.class.getModifiers())),
                () -> assertEquals(Set.of(), Set.of(TensorScalarExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorScalarExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorScalarExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(3, methods.length));
        assertPackageStaticTensorMethod(applyScalar);
        assertPackageStaticTensorMethod(applyClamp);
        assertAll(
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())),
                () -> assertTrue(Modifier.isStatic(create.getModifiers())),
                () -> assertEquals(Set.of(applyScalar, applyClamp, create),
                        Set.copyOf(Arrays.asList(methods))));

        for (ScalarCall call : SCALAR_CALLS) {
            assertPublicTensorMethod(
                    Tensor.class.getDeclaredMethod(call.methodName(), ScalarValue.class),
                    ScalarValue.class);
            assertPublicTensorMethod(
                    Tensor.class.getDeclaredMethod(call.methodName(), double.class), double.class);
        }
        assertPublicTensorMethod(
                Tensor.class.getDeclaredMethod("clamp", ScalarValue.class, ScalarValue.class),
                ScalarValue.class, ScalarValue.class);
        assertPublicTensorMethod(
                Tensor.class.getDeclaredMethod("clamp", double.class, double.class),
                double.class, double.class);
        for (String methodName : List.of(
                "add", "sub", "mul", "div", "minimum", "maximum", "pow")) {
            assertPublicTensorMethod(
                    Tensor.class.getDeclaredMethod(methodName, Tensor.class), Tensor.class);
        }
        assertAll(
                () -> assertThrows(NoSuchMethodException.class,
                        () -> Tensor.class.getDeclaredMethod("min", Tensor.class)),
                () -> assertThrows(NoSuchMethodException.class,
                        () -> Tensor.class.getDeclaredMethod("max", Tensor.class)),
                () -> assertThrows(NoSuchMethodException.class,
                        () -> Tensor.class.getDeclaredMethod("min", ScalarValue.class)),
                () -> assertThrows(NoSuchMethodException.class,
                        () -> Tensor.class.getDeclaredMethod("max", ScalarValue.class)),
                () -> assertThrows(NoSuchMethodException.class,
                        () -> Tensor.class.getDeclaredMethod("min", double.class)),
                () -> assertThrows(NoSuchMethodException.class,
                        () -> Tensor.class.getDeclaredMethod("max", double.class)));
    }

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
        AtomicLong nextId = nextTensorIdState();
        long beforeFailures = nextId.get();

        assertFailure(NullPointerException.class, "input",
                () -> TensorScalarExpressions.applyScalar(null, null, null));
        assertFailure(NullPointerException.class, "kind",
                () -> TensorScalarExpressions.applyScalar(floating, null, null));
        assertFailure(NullPointerException.class, "value",
                () -> TensorScalarExpressions.applyScalar(
                        floating, ScalarElementwiseKind.MUL, null));
        assertFailure(IllegalArgumentException.class, "CLAMP requires ClampRangeAttrs",
                () -> TensorScalarExpressions.applyScalar(
                        integral, ScalarElementwiseKind.CLAMP, ScalarValue.int32(1)));
        assertFailure(IllegalArgumentException.class,
                "input must be a floating data type, but was INT32",
                () -> TensorScalarExpressions.applyScalar(
                        integral, ScalarElementwiseKind.ADD, ScalarValue.float64(1.0)));
        assertFailure(IllegalArgumentException.class,
                "input must be a floating data type, but was INT32",
                () -> TensorScalarExpressions.applyClamp(
                        integral, ScalarValue.int32(2), ScalarValue.int32(1)));
        assertFailure(IllegalArgumentException.class,
                "scalar data type FLOAT64 must match input data type FLOAT32",
                () -> floating.add(1.0d));
        assertFailure(IllegalArgumentException.class,
                "clamp data type FLOAT64 must match input data type FLOAT32",
                () -> floating.clamp(ScalarValue.float64(0.0), ScalarValue.float64(1.0)));
        assertFailure(IllegalArgumentException.class,
                "minValue and maxValue must have the same data type: FLOAT32 != FLOAT64",
                () -> floating.clamp(ScalarValue.float32(0.0f), ScalarValue.float64(1.0)));
        assertFailure(IllegalArgumentException.class,
                "minValue must be less than or equal to maxValue",
                () -> floating.clamp(ScalarValue.float32(2.0f), ScalarValue.float32(1.0f)));
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
            default -> throw new IllegalArgumentException("not floating: " + dataType);
        };
    }

    private static void assertPackageStaticTensorMethod(Method method) {
        assertAll(
                () -> assertEquals(Tensor.class, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(method.getModifiers())));
    }

    private static void assertPublicTensorMethod(Method method, Class<?>... parameters) {
        assertAll(
                () -> assertEquals(Tensor.class, method.getReturnType()),
                () -> assertEquals(List.of(parameters), Arrays.asList(method.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isStatic(method.getModifiers())));
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
            Class<T> type, String message, org.junit.jupiter.api.function.Executable executable) {
        T failure = assertThrows(type, executable);
        assertEquals(message, failure.getMessage());
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
