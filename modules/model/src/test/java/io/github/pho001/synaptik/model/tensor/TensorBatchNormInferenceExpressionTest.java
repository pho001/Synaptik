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
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorBatchNormInferenceExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(95_000);

    @Test
    void exposesExactlyOneCanonicalReceiverAndOnePackageEntry() throws Exception {
        Method method = Tensor.class.getDeclaredMethod(
                "batchNormInference", int.class, Tensor.class, Tensor.class, Tensor.class,
                Tensor.class, ScalarValue.class);
        List<Method> entries = Arrays.stream(
                        TensorBatchNormInferenceExpressions.class.getDeclaredMethods())
                .filter(candidate -> !Modifier.isPrivate(candidate.getModifiers()))
                .toList();
        assertAll(
                () -> assertEquals(Tensor.class, method.getReturnType()),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                () -> assertTrue(Modifier.isFinal(
                        TensorBatchNormInferenceExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorBatchNormInferenceExpressions.class.getModifiers())),
                () -> assertEquals(0,
                        TensorBatchNormInferenceExpressions.class.getDeclaredFields().length),
                () -> assertEquals(1, entries.size()),
                () -> assertEquals("apply", entries.getFirst().getName()),
                () -> assertTrue(Modifier.isStatic(entries.getFirst().getModifiers())),
                () -> assertEquals(206, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(candidate -> Modifier.isPublic(candidate.getModifiers())).count()));
    }

    @Test
    void retainsExactMetadataNormalizedAxisAndFiveInputProvenance() {
        Shape inputShape = Shape.of(2, 3, 4);
        Tensor input = tensor(DataType.BFLOAT16, inputShape, false);
        Tensor scale = tensor(DataType.FLOAT32, Shape.of(4), false);
        Tensor bias = tensor(DataType.BFLOAT16, Shape.of(4), false);
        Tensor mean = tensor(DataType.FLOAT32, Shape.of(4), false);
        Tensor variance = tensor(DataType.FLOAT64, Shape.of(4), true);
        ScalarValue epsilon = ScalarValue.float64(1.0e-5);

        Tensor result = input.batchNormInference(-1, scale, bias, mean, variance, epsilon);
        TensorProvenance provenance = result.provenance().orElseThrow();
        BatchNormInferenceAttrs attrs = (BatchNormInferenceAttrs) provenance.operation().attrs();
        assertAll(
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertSame(inputShape, result.descriptor().shape()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(BatchNormKind.BATCH_NORM_INFERENCE,
                        provenance.operation().kind()),
                () -> assertEquals(2, attrs.channelAxis()),
                () -> assertSame(epsilon, attrs.epsilon()),
                () -> assertEquals(List.of(input, scale, bias, mean, variance),
                        provenance.inputs()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
    }

    @Test
    void acceptsEveryAxisSymbolicDeferralAndEmptyChannelOrOtherAxis() {
        DynamicDimension c = new DynamicDimension("C");
        DynamicDimension deferred = new DynamicDimension("D");
        Shape shape = Shape.ofDimensions(new StaticDimension(2), c, new StaticDimension(0));
        Tensor input = tensor(DataType.FLOAT32, shape, false);
        Tensor exact = input.batchNormInference(1,
                vector(DataType.FLOAT32, c), vector(DataType.FLOAT32, c),
                vector(DataType.FLOAT32, c), vector(DataType.FLOAT32, c),
                ScalarValue.float32(1.0e-5f));
        Tensor deferredResult = input.batchNormInference(-2,
                vector(DataType.FLOAT32, deferred), vector(DataType.FLOAT32, deferred),
                vector(DataType.FLOAT32, deferred), vector(DataType.FLOAT32, deferred),
                ScalarValue.float32(1.0e-5f));
        Tensor channelZero = tensor(DataType.FLOAT32, Shape.of(2, 0), false)
                .batchNormInference(1,
                        tensor(DataType.FLOAT32, Shape.of(0), false),
                        tensor(DataType.FLOAT32, Shape.of(0), false),
                        tensor(DataType.FLOAT32, Shape.of(0), false),
                        tensor(DataType.FLOAT32, Shape.of(0), false),
                        ScalarValue.float32(1.0e-5f));
        assertAll(
                () -> assertSame(shape, exact.descriptor().shape()),
                () -> assertSame(shape, deferredResult.descriptor().shape()),
                () -> assertEquals(0L,
                        channelZero.descriptor().shape().knownElementCount().orElseThrow()));
    }

    @Test
    void rejectsRankAxisVectorAndStaticDimensionFailuresWithExactMessages() {
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        Tensor vectorInput = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor v3 = tensor(DataType.FLOAT32, Shape.of(3), false);
        assertEquals("batchNormInference input rank must be at least 2, but was 1", assertThrows(
                IllegalArgumentException.class,
                () -> vectorInput.batchNormInference(0, v3, v3, v3, v3, epsilon)).getMessage());

        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        assertEquals("Axis 2 is outside shape rank 2", assertThrows(
                IndexOutOfBoundsException.class,
                () -> input.batchNormInference(2, v3, v3, v3, v3, epsilon)).getMessage());
        Tensor rankTwo = tensor(DataType.FLOAT32, Shape.of(1, 3), false);
        assertEquals("batchNormInference scale rank must be one, but was 2", assertThrows(
                IllegalArgumentException.class,
                () -> input.batchNormInference(1, rankTwo, v3, v3, v3, epsilon)).getMessage());
        Tensor v4 = tensor(DataType.FLOAT32, Shape.of(4), false);
        assertEquals(
                "batchNormInference scale channel dimension mismatch: input=StaticDimension[size=3], scale=StaticDimension[size=4]",
                assertThrows(IllegalArgumentException.class,
                        () -> input.batchNormInference(1, v4, v3, v3, v3, epsilon)).getMessage());
        assertEquals(
                "batchNormInference runningVariance channel dimension mismatch: input=StaticDimension[size=3], runningVariance=StaticDimension[size=4]",
                assertThrows(IllegalArgumentException.class,
                        () -> input.batchNormInference(1, v3, v3, v3, v4, epsilon)).getMessage());
    }

    @Test
    void validatesNullTypeShapeAndEpsilonOrderWithoutConsumingFactoryIdentity() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor integral = tensor(DataType.INT32, Shape.of(3), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertEquals("input", assertThrows(NullPointerException.class,
                () -> TensorBatchNormInferenceExpressions.apply(
                        null, 1, null, null, null, null, null)).getMessage());
        assertEquals("scale", assertThrows(NullPointerException.class,
                () -> TensorBatchNormInferenceExpressions.apply(
                        input, 1, null, null, null, null, null)).getMessage());
        assertEquals("bias", assertThrows(NullPointerException.class,
                () -> TensorBatchNormInferenceExpressions.apply(
                        input, 1, floating, null, null, null, null)).getMessage());
        assertEquals("runningMean", assertThrows(NullPointerException.class,
                () -> TensorBatchNormInferenceExpressions.apply(
                        input, 1, floating, floating, null, null, null)).getMessage());
        assertEquals("runningVariance", assertThrows(NullPointerException.class,
                () -> TensorBatchNormInferenceExpressions.apply(
                        input, 1, floating, floating, floating, null, null)).getMessage());
        assertEquals("epsilon", assertThrows(NullPointerException.class,
                () -> TensorBatchNormInferenceExpressions.apply(
                        input, 1, floating, floating, floating, floating, null)).getMessage());
        assertEquals("batchNormInference scale must have a floating data type, but was INT32",
                assertThrows(IllegalArgumentException.class,
                        () -> input.batchNormInference(1, integral, floating, floating, floating,
                                ScalarValue.float32(1.0e-5f))).getMessage());
        assertEquals(
                "batchNormInference epsilon data type must match result data type: epsilon=FLOAT64, result=FLOAT32",
                assertThrows(IllegalArgumentException.class,
                        () -> input.batchNormInference(1, floating, floating, floating, floating,
                                ScalarValue.float64(1.0e-5))).getMessage());
        assertEquals(before, next.get());
    }

    @Test
    void equalRequestsStayFreshAndConsumeExactlyOneIdentityEach() throws Exception {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), false);
        Tensor vector = tensor(DataType.FLOAT64, Shape.of(3), false);
        ScalarValue epsilon = ScalarValue.float64(1.0e-5);
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor first = input.batchNormInference(1, vector, vector, vector, vector, epsilon);
        Tensor second = input.batchNormInference(1, vector, vector, vector, vector, epsilon);
        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()));
    }

    private static Tensor vector(DataType type, io.github.pho001.synaptik.model.shape.Dimension d) {
        return tensor(type, Shape.ofDimensions(d), false);
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }
}
