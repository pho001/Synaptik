package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
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
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorSoftmaxExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(30_000);

    @Test
    void exposesExactlyTheTwoPublicMethodsAndThreeMethodHelperBoundary() throws Exception {
        for (String methodName : List.of("softmax", "logSoftmax")) {
            Method method = Tensor.class.getDeclaredMethod(methodName, int.class);
            assertAll(
                    () -> assertSame(Tensor.class, method.getReturnType()),
                    () -> assertEquals(List.of(int.class),
                            Arrays.asList(method.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        Method apply = TensorSoftmaxExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, SoftmaxKind.class, int.class);
        Method validate = TensorSoftmaxExpressions.class.getDeclaredMethod(
                "validateFloatingInput", Tensor.class);
        Method create = TensorSoftmaxExpressions.class.getDeclaredMethod(
                "create", Tensor.class, SoftmaxKind.class, Shape.class, SoftmaxAttrs.class);
        var constructor = TensorSoftmaxExpressions.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorSoftmaxExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorSoftmaxExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorSoftmaxExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorSoftmaxExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorSoftmaxExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(
                        List.of("apply", "create", "validateFloatingInput"),
                        Arrays.stream(TensorSoftmaxExpressions.class.getDeclaredMethods())
                                .map(Method::getName)
                                .sorted()
                                .toList()),
                () -> assertPackagePrivateStatic(apply, Tensor.class),
                () -> assertPrivateStatic(validate, void.class),
                () -> assertPrivateStatic(create, Tensor.class));
    }

    @Test
    void delegatesEachPublicMethodToItsExactKindAndNormalizesPositiveAndNegativeAxes() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);

        Tensor softmax = input.softmax(-1);
        Tensor logSoftmax = input.logSoftmax(0);
        Operation softmaxOperation = softmax.provenance().orElseThrow().operation();
        Operation logSoftmaxOperation = logSoftmax.provenance().orElseThrow().operation();
        SoftmaxAttrs softmaxAttrs = (SoftmaxAttrs) softmaxOperation.attrs();
        SoftmaxAttrs logSoftmaxAttrs = (SoftmaxAttrs) logSoftmaxOperation.attrs();

        assertAll(
                () -> assertSame(SoftmaxKind.SOFTMAX, softmaxOperation.kind()),
                () -> assertSame(softmaxAttrs, softmaxOperation.attrs()),
                () -> assertEquals(1, softmaxAttrs.axis()),
                () -> assertSame(SoftmaxKind.LOG_SOFTMAX, logSoftmaxOperation.kind()),
                () -> assertSame(logSoftmaxAttrs, logSoftmaxOperation.attrs()),
                () -> assertEquals(0, logSoftmaxAttrs.axis()),
                () -> assertNotEquals(softmaxOperation, logSoftmaxOperation));
    }

    @Test
    void acceptsExactlyEveryFloatingTypeAndRetainsDescriptorAndProvenanceFacts() {
        Shape shape = Shape.of(2, 3);
        for (DataType dataType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (boolean requiresGrad : List.of(false, true)) {
                Tensor input = tensor(dataType, shape, requiresGrad);
                Tensor result = input.softmax(1);
                TensorProvenance provenance = result.provenance().orElseThrow();

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertSame(shape, result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertEquals(1, provenance.inputs().size()),
                        () -> assertSame(input, provenance.inputs().getFirst()),
                        () -> assertSame(SoftmaxKind.SOFTMAX,
                                provenance.operation().kind()),
                        () -> assertEquals(new SoftmaxAttrs(1),
                                provenance.operation().attrs()));
            }
        }
    }

    @Test
    void acceptsDynamicAndZeroExtentsAndRetainsTheExactShapeReference() {
        DynamicDimension batch = new DynamicDimension("batch");
        Shape dynamic = Shape.ofDimensions(batch, new StaticDimension(4));
        Shape zeroExtent = Shape.of(2, 0, 3);
        Tensor dynamicInput = tensor(DataType.BFLOAT16, dynamic, true);
        Tensor zeroInput = tensor(DataType.FLOAT64, zeroExtent, false);

        assertAll(
                () -> assertSame(dynamic,
                        dynamicInput.logSoftmax(-1).descriptor().shape()),
                () -> assertSame(batch,
                        dynamicInput.softmax(0).descriptor().shape().dimensions().getFirst()),
                () -> assertSame(zeroExtent,
                        zeroInput.softmax(1).descriptor().shape()));
    }

    @Test
    void validatesNullKindTypeAndAxisInDeterministicOrderWithoutConsumingIdentity()
            throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2), false);
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorSoftmaxExpressions.apply(null, null, 9));
        NullPointerException nullKind = assertThrows(
                NullPointerException.class,
                () -> TensorSoftmaxExpressions.apply(floating, null, 9));

        for (DataType dataType : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            Tensor invalid = tensor(dataType, Shape.of(2), false);
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> invalid.softmax(9));
            assertEquals(
                    "input must have a floating data type, but was " + dataType,
                    failure.getMessage());
        }

        IndexOutOfBoundsException positive = assertThrows(
                IndexOutOfBoundsException.class, () -> floating.softmax(1));
        IndexOutOfBoundsException negative = assertThrows(
                IndexOutOfBoundsException.class, () -> floating.logSoftmax(-2));
        IndexOutOfBoundsException scalar = assertThrows(
                IndexOutOfBoundsException.class,
                () -> tensor(DataType.FLOAT64, Shape.scalar(), true).softmax(0));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("kind", nullKind.getMessage()),
                () -> assertEquals("Axis 1 is outside shape rank 1", positive.getMessage()),
                () -> assertEquals("Axis -2 is outside shape rank 1", negative.getMessage()),
                () -> assertEquals("Axis 0 is outside shape rank 0", scalar.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void discardsResolvedLayoutWithoutInspectingOrMutatingInputStorageAndMetadata() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(layout), true);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.FLOAT32, shape, false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(SoftmaxKind.SOFTMAX, new SoftmaxAttrs(0)),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));

        Tensor result = input.logSoftmax(1);

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(originalProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertEquals(List.of(1.0f, 2.0f, 3.0f, 4.0f),
                        List.of(values[0], values[1], values[2], values[3])),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(input,
                        result.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void repeatedAndNestedCallsAreFreshAndNeverCanonicalizedOrDecomposed() {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(3), true);
        Tensor first = input.softmax(0);
        Tensor second = input.softmax(0);
        Tensor nested = first.logSoftmax(0);

        assertAll(
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(second, nested),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotEquals(second.id(), nested.id()),
                () -> assertSame(SoftmaxKind.LOG_SOFTMAX,
                        nested.provenance().orElseThrow().operation().kind()),
                () -> assertSame(first,
                        nested.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void propagatesIdentifierExhaustionAfterValidLocalConstruction() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.logSoftmax(0));
            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static void assertPackagePrivateStatic(Method method, Class<?> returnType) {
        assertAll(
                () -> assertSame(returnType, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isProtected(method.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(method.getModifiers())));
    }

    private static void assertPrivateStatic(Method method, Class<?> returnType) {
        assertAll(
                () -> assertSame(returnType, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(method.getModifiers())));
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimedState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }
}
