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
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
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

class TensorRmsNormExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(90_000);

    @Test
    void exposesExactlyTwoCanonicalReceiverMethodsAndTwoPackageEntries() throws Exception {
        Method plain = Tensor.class.getDeclaredMethod(
                "rmsNorm", Shape.class, ScalarValue.class);
        Method scaled = Tensor.class.getDeclaredMethod(
                "rmsNorm", Shape.class, Tensor.class, ScalarValue.class);
        List<Method> entries = Arrays.stream(TensorRmsNormExpressions.class.getDeclaredMethods())
                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                .toList();

        assertAll(
                () -> assertEquals(Tensor.class, plain.getReturnType()),
                () -> assertEquals(Tensor.class, scaled.getReturnType()),
                () -> assertTrue(Modifier.isPublic(plain.getModifiers())),
                () -> assertTrue(Modifier.isPublic(scaled.getModifiers())),
                () -> assertFalse(Modifier.isStatic(plain.getModifiers())),
                () -> assertFalse(Modifier.isStatic(scaled.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorRmsNormExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorRmsNormExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorRmsNormExpressions.class.getDeclaredFields().length),
                () -> assertEquals(2, entries.size()),
                () -> assertTrue(entries.stream().allMatch(method ->
                        method.getName().equals("apply")
                                && Modifier.isStatic(method.getModifiers()))),
                () -> assertEquals(183, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()));
    }

    @Test
    void retainsNoScaleMetadataAndExactOneOutputProvenanceForEveryFloatingType() {
        Shape inputShape = Shape.of(2, 3, 4);
        Shape normalizedShape = Shape.of(3, 4);
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor input = tensor(type, inputShape, true);
            ScalarValue epsilon = epsilon(type);
            Tensor result = input.rmsNorm(normalizedShape, epsilon);
            TensorProvenance provenance = result.provenance().orElseThrow();
            RmsNormAttrs attrs = (RmsNormAttrs) provenance.operation().attrs();

            assertAll(
                    () -> assertSame(type, result.descriptor().dataType()),
                    () -> assertSame(inputShape, result.descriptor().shape()),
                    () -> assertTrue(result.descriptor().requiresGrad()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.label().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()),
                    () -> assertSame(RmsNormKind.RMS_NORM, provenance.operation().kind()),
                    () -> assertSame(normalizedShape, attrs.normalizedShape()),
                    () -> assertSame(epsilon, attrs.epsilon()),
                    () -> assertEquals(List.of(input), provenance.inputs()),
                    () -> assertEquals(1, provenance.producer().outputCount()),
                    () -> assertEquals(0, provenance.outputIndex()),
                    () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
        }
    }

    @Test
    void promotesScaleInOccurrenceOrderAndPropagatesEitherGradientRequest() {
        Shape inputShape = Shape.of(2, 3);
        Shape normalized = Shape.of(3);
        Tensor input = tensor(DataType.BFLOAT16, inputShape, false);
        Tensor scale = tensor(DataType.FLOAT64, normalized, true);
        ScalarValue epsilon = ScalarValue.float64(1.0e-5);

        Tensor result = input.rmsNorm(normalized, scale, epsilon);
        TensorProvenance provenance = result.provenance().orElseThrow();
        RmsNormAttrs attrs = (RmsNormAttrs) provenance.operation().attrs();

        assertAll(
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertSame(inputShape, result.descriptor().shape()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertEquals(List.of(input, scale), provenance.inputs()),
                () -> assertSame(normalized, attrs.normalizedShape()),
                () -> assertSame(epsilon, attrs.epsilon()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(1, provenance.producer().outputCount()));
    }

    @Test
    void acceptsExactDynamicEqualityDefersOtherUnresolvedEqualityAndAcceptsEmptyResults() {
        DynamicDimension n = new DynamicDimension("N");
        DynamicDimension m = new DynamicDimension("M");
        Shape inputShape = Shape.ofDimensions(new StaticDimension(0), n, new StaticDimension(4));
        Tensor input = tensor(DataType.FLOAT32, inputShape, false);

        Tensor exact = input.rmsNorm(
                Shape.ofDimensions(n, new StaticDimension(4)), ScalarValue.float32(1.0e-5f));
        Tensor deferred = input.rmsNorm(
                Shape.ofDimensions(m, new StaticDimension(4)), ScalarValue.float32(1.0e-5f));
        Tensor zeroNormalized = tensor(DataType.FLOAT32, Shape.of(2, 0), false)
                .rmsNorm(Shape.of(0), ScalarValue.float32(1.0e-5f));

        assertAll(
                () -> assertSame(inputShape, exact.descriptor().shape()),
                () -> assertSame(inputShape, deferred.descriptor().shape()),
                () -> assertEquals(0L,
                        zeroNormalized.descriptor().shape().knownElementCount().orElseThrow()));
    }

    @Test
    void rejectsRankStaticAndScaleShapeFailuresWithExactMessages() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        assertEquals("normalizedShape rank must be positive", assertThrows(
                IllegalArgumentException.class,
                () -> input.rmsNorm(Shape.scalar(), epsilon)).getMessage());
        assertEquals(
                "rmsNorm normalized rank must not exceed input rank: normalizedRank=4, inputRank=3",
                assertThrows(IllegalArgumentException.class,
                        () -> input.rmsNorm(Shape.of(1, 2, 3, 4), epsilon)).getMessage());
        assertEquals(
                "rmsNorm normalized dimension mismatch at normalized axis 0: input=StaticDimension[size=3], normalized=StaticDimension[size=5]",
                assertThrows(IllegalArgumentException.class,
                        () -> input.rmsNorm(Shape.of(5, 4), epsilon)).getMessage());

        Shape normalized = Shape.of(3, 4);
        Tensor wrongScale = tensor(DataType.FLOAT32, Shape.of(1, 3, 4), false);
        assertEquals(
                "rmsNorm scale Shape must equal normalizedShape: scale=Shape[1, 3, 4], normalizedShape=Shape[3, 4]",
                assertThrows(IllegalArgumentException.class,
                        () -> input.rmsNorm(normalized, wrongScale, epsilon)).getMessage());
    }

    @Test
    void validatesNullTypeShapeAndEpsilonOrderWithoutConsumingFactoryIdentity() throws Exception {
        Shape normalized = Shape.of(3);
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor integral = tensor(DataType.INT32, Shape.of(2, 3), false);
        Tensor integralScale = tensor(DataType.INT64, normalized, false);
        Tensor floatingScale = tensor(DataType.FLOAT32, normalized, false);
        Tensor wrongScale = tensor(DataType.FLOAT32, Shape.of(1, 3), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertEquals("input", assertThrows(NullPointerException.class,
                () -> TensorRmsNormExpressions.apply(null, null, null)).getMessage());
        assertEquals("normalizedShape", assertThrows(NullPointerException.class,
                () -> TensorRmsNormExpressions.apply(floating, null, null)).getMessage());
        assertEquals("epsilon", assertThrows(NullPointerException.class,
                () -> TensorRmsNormExpressions.apply(floating, normalized, null)).getMessage());
        assertEquals("scale", assertThrows(NullPointerException.class,
                () -> TensorRmsNormExpressions.apply(
                        floating, normalized, null, null)).getMessage());
        assertEquals("epsilon", assertThrows(NullPointerException.class,
                () -> TensorRmsNormExpressions.apply(
                        floating, normalized, floatingScale, null)).getMessage());
        assertEquals("rmsNorm input must have a floating data type, but was INT32",
                assertThrows(IllegalArgumentException.class,
                        () -> integral.rmsNorm(normalized, epsilon(DataType.INT32))).getMessage());
        assertEquals("rmsNorm scale must have a floating data type, but was INT64",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.rmsNorm(normalized, integralScale,
                                ScalarValue.float32(1.0e-5f))).getMessage());
        assertEquals(
                "rmsNorm scale Shape must equal normalizedShape: scale=Shape[1, 3], normalizedShape=Shape[3]",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.rmsNorm(normalized, wrongScale,
                                ScalarValue.float32(1.0e-5f))).getMessage());
        assertEquals(
                "rmsNorm epsilon data type must match result data type: epsilon=FLOAT64, result=FLOAT32",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.rmsNorm(normalized,
                                ScalarValue.float64(1.0e-5))).getMessage());
        assertEquals(before, next.get());
    }

    @Test
    void repeatedEqualRequestsRemainFreshAndConsumeExactlyOneIdentityEach() throws Exception {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), false);
        Shape normalized = Shape.of(3);
        ScalarValue epsilon = ScalarValue.float64(1.0e-5);
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor first = input.rmsNorm(normalized, epsilon);
        Tensor second = input.rmsNorm(normalized, epsilon);

        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()));
    }

    private static ScalarValue epsilon(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16(1.0e-2f);
            case FLOAT32 -> ScalarValue.float32(1.0e-5f);
            case FLOAT64 -> ScalarValue.float64(1.0e-5);
            case INT32 -> ScalarValue.int32(1);
            case INT64 -> ScalarValue.int64(1);
            case BOOL -> ScalarValue.bool(true);
        };
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
