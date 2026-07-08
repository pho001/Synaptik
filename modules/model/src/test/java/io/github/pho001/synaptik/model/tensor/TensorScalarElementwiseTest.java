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
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
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
            new ScalarCall("mul", ScalarElementwiseKind.MUL, Tensor::mul),
            new ScalarCall("pow", ScalarElementwiseKind.POW, Tensor::pow),
            new ScalarCall("clampMin", ScalarElementwiseKind.CLAMP_MIN, Tensor::clampMin),
            new ScalarCall("clampMax", ScalarElementwiseKind.CLAMP_MAX, Tensor::clampMax));

    @Test
    void helperAndTensorOverloadsHaveExactlyTheRequiredShape()
            throws ReflectiveOperationException {
        int classModifiers = TensorScalarExpressions.class.getModifiers();
        var constructors = TensorScalarExpressions.class.getDeclaredConstructors();
        var methods = TensorScalarExpressions.class.getDeclaredMethods();

        assertAll(
                () -> assertTrue(Modifier.isFinal(classModifiers)),
                () -> assertFalse(Modifier.isPublic(classModifiers)),
                () -> assertFalse(Modifier.isProtected(classModifiers)),
                () -> assertFalse(TensorScalarExpressions.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(TensorScalarExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorScalarExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorScalarExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(3, methods.length));

        Method applyScalar = TensorScalarExpressions.class.getDeclaredMethod(
                "applyScalar", Tensor.class, ScalarElementwiseKind.class, double.class);
        Method applyClamp = TensorScalarExpressions.class.getDeclaredMethod(
                "applyClamp", Tensor.class, double.class, double.class);
        Method create = TensorScalarExpressions.class.getDeclaredMethod(
                "create", Tensor.class, DataType.class, Operation.class);

        assertPackageStaticTensorMethod(applyScalar);
        assertPackageStaticTensorMethod(applyClamp);
        assertAll(
                () -> assertEquals(Tensor.class, create.getReturnType()),
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())),
                () -> assertTrue(Modifier.isStatic(create.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(create.getModifiers())),
                () -> assertEquals(
                        Set.of(applyScalar, applyClamp, create), Set.copyOf(Arrays.asList(methods))));

        for (ScalarCall call : SCALAR_CALLS) {
            Method method = Tensor.class.getDeclaredMethod(call.methodName(), double.class);
            assertPublicTensorMethod(method, double.class);
        }
        Method clamp = Tensor.class.getDeclaredMethod("clamp", double.class, double.class);
        assertPublicTensorMethod(clamp, double.class, double.class);

        assertPublicTensorMethod(Tensor.class.getDeclaredMethod("mul", Tensor.class), Tensor.class);
        assertPublicTensorMethod(Tensor.class.getDeclaredMethod("pow", Tensor.class), Tensor.class);
    }

    @Test
    void mapsEveryOverloadToExactKindAttributesAndOneInputProvenance() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        double value = Double.longBitsToDouble(0xfff8_0000_0000_1234L);

        for (ScalarCall call : SCALAR_CALLS) {
            Tensor result = call.apply(input, value);
            TensorProvenance provenance = result.provenance().orElseThrow();
            Operation operation = provenance.operation();
            ScalarValueAttrs attrs = (ScalarValueAttrs) operation.attrs();
            assertAll(
                    () -> assertSame(call.kind(), operation.kind()),
                    () -> assertEquals(Double.doubleToRawLongBits(value),
                            Double.doubleToRawLongBits(attrs.value())),
                    () -> assertEquals(1, provenance.inputs().size()),
                    () -> assertSame(input, provenance.inputs().getFirst()));
        }

        double min = Double.longBitsToDouble(0x8000_0000_0000_0000L);
        double max = Double.longBitsToDouble(0x7ff8_0000_0000_5678L);
        Tensor clamped = input.clamp(min, max);
        TensorProvenance provenance = clamped.provenance().orElseThrow();
        ClampRangeAttrs attrs = (ClampRangeAttrs) provenance.operation().attrs();
        assertAll(
                () -> assertSame(ScalarElementwiseKind.CLAMP, provenance.operation().kind()),
                () -> assertEquals(Double.doubleToRawLongBits(min),
                        Double.doubleToRawLongBits(attrs.minValue())),
                () -> assertEquals(Double.doubleToRawLongBits(max),
                        Double.doubleToRawLongBits(attrs.maxValue())),
                () -> assertEquals(1, provenance.inputs().size()),
                () -> assertSame(input, provenance.inputs().getFirst()));

        Tensor right = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        assertAll(
                () -> assertSame(BinaryArithmeticKind.MUL,
                        input.mul(right).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.POW,
                        input.pow(right).provenance().orElseThrow().operation().kind()),
                () -> assertEquals(2,
                        input.mul(right).provenance().orElseThrow().inputs().size()),
                () -> assertEquals(2,
                        input.pow(right).provenance().orElseThrow().inputs().size()));
    }

    @Test
    void retainsExactTypeShapeAndGradientAcrossFloatingTypesAndShapeStates() {
        Shape scalar = Shape.scalar();
        Shape empty = Shape.of(2, 0, 4);
        Shape ordinary = Shape.of(2, 3);
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));

        for (DataType dataType : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            for (Shape shape : List.of(scalar, empty, ordinary, dynamic)) {
                for (boolean requiresGrad : List.of(false, true)) {
                    Tensor input = tensor(dataType, shape, requiresGrad);
                    for (ScalarCall call : SCALAR_CALLS) {
                        assertResultMetadata(call.apply(input, -0.0), dataType, shape, requiresGrad);
                    }
                    assertResultMetadata(
                            input.clamp(-1.0, 1.0), dataType, shape, requiresGrad);
                }
            }
        }

        TensorDescriptor resolvedDescriptor = new TensorDescriptor(
                DataType.FLOAT32,
                ordinary,
                Optional.of(LayoutDescriptor.contiguous(ordinary)),
                true);
        Tensor resolvedInput = TensorFactory.create(resolvedDescriptor);
        assertResultMetadata(resolvedInput.clamp(0.0, 1.0), DataType.FLOAT32, ordinary, true);
    }

    @Test
    void retainsAllScalarAndRangeBitsWithoutInputDependentConversion() {
        long[] bits = {
            0x0000_0000_0000_0000L,
            0x8000_0000_0000_0000L,
            0x3ff0_0000_0000_0000L,
            0x7ff0_0000_0000_0000L,
            0xfff0_0000_0000_0000L,
            0x7ff8_0000_0000_1234L,
            0xfff8_0000_0000_5678L
        };

        for (DataType dataType : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor input = tensor(dataType, Shape.scalar(), false);
            for (long rawBits : bits) {
                double value = Double.longBitsToDouble(rawBits);
                for (ScalarCall call : SCALAR_CALLS) {
                    ScalarValueAttrs attrs = (ScalarValueAttrs) call.apply(input, value)
                            .provenance().orElseThrow().operation().attrs();
                    assertEquals(rawBits, Double.doubleToRawLongBits(attrs.value()));
                }
            }

            for (long[] range : List.of(
                    new long[] {0x8000_0000_0000_0000L, 0x0000_0000_0000_0000L},
                    new long[] {0xfff0_0000_0000_0000L, 0x7ff0_0000_0000_0000L},
                    new long[] {0x7ff8_0000_0000_1234L, 0xfff8_0000_0000_5678L})) {
                double min = Double.longBitsToDouble(range[0]);
                double max = Double.longBitsToDouble(range[1]);
                ClampRangeAttrs attrs = (ClampRangeAttrs) input.clamp(min, max)
                        .provenance().orElseThrow().operation().attrs();
                assertAll(
                        () -> assertEquals(range[0],
                                Double.doubleToRawLongBits(attrs.minValue())),
                        () -> assertEquals(range[1],
                                Double.doubleToRawLongBits(attrs.maxValue())));
            }
        }
    }

    @Test
    void everyValidCallIsFreshUnlabeledStorageFreeAndNeverCanonicalized() {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2), true);

        for (double value : List.of(0.0, 1.0, -1.0, Double.POSITIVE_INFINITY, Double.NaN)) {
            for (ScalarCall call : SCALAR_CALLS) {
                Tensor first = call.apply(input, value);
                Tensor second = call.apply(input, value);
                assertAll(
                        () -> assertNotSame(input, first),
                        () -> assertNotSame(first, second),
                        () -> assertNotEquals(first.id(), second.id()),
                        () -> assertTrue(first.label().isEmpty()),
                        () -> assertTrue(first.hostStorage().isEmpty()),
                        () -> assertTrue(first.descriptor().layout().isEmpty()));
            }
        }

        for (double[] range : List.of(
                new double[] {0.0, 0.0},
                new double[] {-0.0, 0.0},
                new double[] {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY},
                new double[] {Double.NaN, 1.0},
                new double[] {0.0, Double.NaN})) {
            Tensor first = input.clamp(range[0], range[1]);
            Tensor second = input.clamp(range[0], range[1]);
            Tensor nested = first.clamp(range[0], range[1]);
            assertAll(
                    () -> assertNotSame(first, second),
                    () -> assertNotSame(first, nested),
                    () -> assertSame(first,
                            nested.provenance().orElseThrow().inputs().getFirst()),
                    () -> assertSame(ScalarElementwiseKind.CLAMP,
                            nested.provenance().orElseThrow().operation().kind()));
        }
    }

    @Test
    void validatesInExactOrderAndAllocatesNoIdentityOnFailure()
            throws ReflectiveOperationException {
        Tensor floating = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor integral = tensor(DataType.INT32, Shape.scalar(), false);
        AtomicLong nextId = nextTensorIdState();
        long beforeFailures = nextId.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorScalarExpressions.applyScalar(null, null, 1.0));
        NullPointerException nullKind = assertThrows(
                NullPointerException.class,
                () -> TensorScalarExpressions.applyScalar(floating, null, 1.0));
        IllegalArgumentException clampKind = assertThrows(
                IllegalArgumentException.class,
                () -> TensorScalarExpressions.applyScalar(
                        integral, ScalarElementwiseKind.CLAMP, 1.0));
        NullPointerException nullClampInput = assertThrows(
                NullPointerException.class,
                () -> TensorScalarExpressions.applyClamp(null, 2.0, 1.0));
        IllegalArgumentException typeBeforeRange = assertThrows(
                IllegalArgumentException.class,
                () -> TensorScalarExpressions.applyClamp(integral, 2.0, 1.0));
        IllegalArgumentException inverted = assertThrows(
                IllegalArgumentException.class, () -> floating.clamp(2.0, 1.0));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("kind", nullKind.getMessage()),
                () -> assertEquals("CLAMP requires ClampRangeAttrs", clampKind.getMessage()),
                () -> assertEquals("input", nullClampInput.getMessage()),
                () -> assertEquals(
                        "input must be a floating data type, but was INT32",
                        typeBeforeRange.getMessage()),
                () -> assertEquals(
                        "minValue must be less than or equal to maxValue",
                        inverted.getMessage()),
                () -> assertEquals(beforeFailures, nextId.get()));

        for (DataType dataType : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            Tensor invalid = tensor(dataType, Shape.scalar(), false);
            long beforeTypeFailures = nextId.get();
            for (ScalarCall call : SCALAR_CALLS) {
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class, () -> call.apply(invalid, 1.0));
                assertEquals(
                        "input must be a floating data type, but was " + dataType,
                        failure.getMessage());
            }
            IllegalArgumentException clampFailure = assertThrows(
                    IllegalArgumentException.class, () -> invalid.clamp(0.0, 1.0));
            assertEquals(
                    "input must be a floating data type, but was " + dataType,
                    clampFailure.getMessage());
            assertEquals(beforeTypeFailures, nextId.get());
        }
    }

    @Test
    void preservesInputMetadataProvenanceStorageAssociationAndContents() {
        float[] values = {-1.0f, 0.0f, 2.0f};
        Shape shape = Shape.of(3);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
        Tensor leaf = TensorFactory.create(
                descriptor, Optional.of("leaf"), Optional.of(storage));
        Operation inputOperation =
                new Operation(ScalarElementwiseKind.MUL, new ScalarValueAttrs(2.0));
        Tensor input = TensorFactory.createDerived(
                descriptor, Optional.of("derived"), inputOperation, List.of(leaf));
        TensorProvenance inputProvenance = input.provenance().orElseThrow();
        input.replaceHostStorage(storage);

        Tensor result = input.clamp(-1.0, 1.0);

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertEquals(Optional.of("derived"), input.label()),
                () -> assertSame(inputProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(input, result.provenance().orElseThrow().inputs().getFirst()),
                () -> assertArrayEquals(new float[] {-1.0f, 0.0f, 2.0f}, values));
    }

    private static void assertPackageStaticTensorMethod(Method method) {
        assertAll(
                () -> assertEquals(Tensor.class, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isProtected(method.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(method.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
    }

    private static void assertPublicTensorMethod(Method method, Class<?>... parameters) {
        assertAll(
                () -> assertEquals(Tensor.class, method.getReturnType()),
                () -> assertEquals(List.of(parameters), Arrays.asList(method.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
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

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    @FunctionalInterface
    private interface ScalarFunction {
        Tensor apply(Tensor input, double value);
    }

    private record ScalarCall(
            String methodName,
            ScalarElementwiseKind kind,
            ScalarFunction function) {
        private Tensor apply(Tensor input, double value) {
            return function.apply(input, value);
        }
    }
}
