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
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(140_000);

    @Test
    void exposesDenseReceiverWithinTheDispatchedLossHelper() throws Exception {
        Method receiver = Tensor.class.getDeclaredMethod(
                "categoricalCrossEntropyWithLogits",
                Tensor.class,
                int.class,
                LossReduction.class);
        Method entry = TensorLossExpressions.class.getDeclaredMethod(
                "categoricalCrossEntropyWithLogits",
                Tensor.class,
                Tensor.class,
                int.class,
                LossReduction.class);
        Set<String> names = Arrays.stream(TensorLossExpressions.class.getDeclaredMethods())
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(receiver.getModifiers())),
                () -> assertFalse(Modifier.isStatic(receiver.getModifiers())),
                () -> assertEquals(Tensor.class, receiver.getReturnType()),
                () -> assertTrue(Modifier.isStatic(entry.getModifiers())),
                () -> assertFalse(Modifier.isPublic(entry.getModifiers())),
                () -> assertFalse(Modifier.isProtected(entry.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(entry.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorLossExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorLossExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorLossExpressions.class.getDeclaredFields().length),
                () -> assertEquals(Set.of(
                                "meanSquaredError",
                                "categoricalCrossEntropyWithLogits",
                                "denseCategoricalCrossEntropyWithLogits",
                                "indexCategoricalCrossEntropyWithLogits",
                                "requireFloating",
                                "validateExactShape",
                                "validateIndexTargetShape",
                                "validateClassExtent",
                                "removeAxis"),
                        names),
                () -> assertEquals(213, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()),
                () -> assertEquals(2, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .filter(method -> method.getName().equals(
                                "categoricalCrossEntropyWithLogits"))
                        .count()));
    }

    @Test
    void promotesEveryFloatingPairAndRecordsOrderedFirstClassMetadata() {
        Shape logitsShape = Shape.of(2, 3, 4);
        for (DataType logitsType : floatingTypes()) {
            for (DataType targetType : floatingTypes()) {
                Tensor logits = tensor(logitsType, logitsShape, false);
                Tensor target = tensor(targetType, Shape.of(2, 3, 4), true);
                Tensor result = logits.categoricalCrossEntropyWithLogits(
                        target, -2, LossReduction.NONE);
                TensorProvenance provenance = result.provenance().orElseThrow();
                DenseCategoricalCrossEntropyWithLogitsAttrs attrs =
                        (DenseCategoricalCrossEntropyWithLogitsAttrs)
                                provenance.operation().attrs();

                assertAll(
                        () -> assertSame(widest(logitsType, targetType),
                                result.descriptor().dataType()),
                        () -> assertEquals(Shape.of(2, 4), result.descriptor().shape()),
                        () -> assertSame(logitsShape.dimension(0),
                                result.descriptor().shape().dimension(0)),
                        () -> assertSame(logitsShape.dimension(2),
                                result.descriptor().shape().dimension(1)),
                        () -> assertTrue(result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertSame(
                                LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                                provenance.operation().kind()),
                        () -> assertEquals(1, attrs.axis()),
                        () -> assertSame(LossReduction.NONE, attrs.reduction()),
                        () -> assertEquals(List.of(logits, target), provenance.inputs()),
                        () -> assertEquals(0, provenance.outputIndex()),
                        () -> assertEquals(1, provenance.producer().outputCount()),
                        () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
            }
        }
    }

    @Test
    void derivesNoneAndReducedShapesFromLogitsIncludingRankOneAndEmptySamples() {
        Shape logitsShape = Shape.of(2, 3, 4);
        Tensor logits = tensor(DataType.FLOAT32, logitsShape, false);
        Tensor target = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false);
        Tensor none = logits.categoricalCrossEntropyWithLogits(
                target, 1, LossReduction.NONE);
        Tensor sum = logits.categoricalCrossEntropyWithLogits(target, 1, LossReduction.SUM);
        Tensor mean = logits.categoricalCrossEntropyWithLogits(target, 1, LossReduction.MEAN);
        Tensor rankOne = tensor(DataType.FLOAT64, Shape.of(3), false)
                .categoricalCrossEntropyWithLogits(
                        tensor(DataType.FLOAT64, Shape.of(3), false),
                        0,
                        LossReduction.NONE);
        Tensor emptySamples = tensor(DataType.FLOAT32, Shape.of(0, 0), false)
                .categoricalCrossEntropyWithLogits(
                        tensor(DataType.FLOAT32, Shape.of(0, 0), false),
                        1,
                        LossReduction.NONE);

        assertAll(
                () -> assertEquals(Shape.of(2, 4), none.descriptor().shape()),
                () -> assertSame(logitsShape.dimension(0),
                        none.descriptor().shape().dimension(0)),
                () -> assertSame(logitsShape.dimension(2),
                        none.descriptor().shape().dimension(1)),
                () -> assertSame(Shape.scalar(), sum.descriptor().shape()),
                () -> assertSame(Shape.scalar(), mean.descriptor().shape()),
                () -> assertSame(Shape.scalar(), rankOne.descriptor().shape()),
                () -> assertEquals(0L,
                        emptySamples.descriptor().shape().knownElementCount().orElseThrow()));
    }

    @Test
    void acceptsExactAndDeferredShapeEqualityWithoutBroadcasting() {
        DynamicDimension n = new DynamicDimension("N");
        DynamicDimension m = new DynamicDimension("M");
        Shape logitsShape = Shape.ofDimensions(n, new StaticDimension(3));
        Tensor logits = tensor(DataType.FLOAT32, logitsShape, false);

        Tensor exact = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(n, new StaticDimension(3)), false),
                1,
                LossReduction.NONE);
        Tensor unresolved = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(m, new StaticDimension(3)), false),
                1,
                LossReduction.NONE);
        Tensor unresolvedStatic = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.FLOAT32, Shape.of(7, 3), false),
                1,
                LossReduction.NONE);

        assertAll(
                () -> assertSame(n, exact.descriptor().shape().dimension(0)),
                () -> assertSame(n, unresolved.descriptor().shape().dimension(0)),
                () -> assertSame(n, unresolvedStatic.descriptor().shape().dimension(0)));
    }

    @Test
    void enforcesExactRankDimensionsAndClassExtentRules() {
        Tensor logits = tensor(DataType.FLOAT32, Shape.of(2, 3), false);

        assertEquals(
                "categoricalCrossEntropyWithLogits target rank must equal logits rank: "
                        + "logits=2, target=1",
                assertThrows(IllegalArgumentException.class,
                        () -> logits.categoricalCrossEntropyWithLogits(
                                tensor(DataType.FLOAT32, Shape.of(3), false),
                                1,
                                LossReduction.NONE)).getMessage());
        assertEquals(
                "categoricalCrossEntropyWithLogits target dimension mismatch at axis 0: "
                        + "logits=StaticDimension[size=2], target=StaticDimension[size=1]",
                assertThrows(IllegalArgumentException.class,
                        () -> logits.categoricalCrossEntropyWithLogits(
                                tensor(DataType.FLOAT32, Shape.of(1, 3), false),
                                1,
                                LossReduction.NONE)).getMessage());
        assertEquals(
                "categoricalCrossEntropyWithLogits class dimension must be positive when "
                        + "sample domain is non-empty: axis=1, dimension=StaticDimension[size=0]",
                assertThrows(IllegalArgumentException.class,
                        () -> tensor(DataType.FLOAT32, Shape.of(2, 0), false)
                                .categoricalCrossEntropyWithLogits(
                                        tensor(DataType.FLOAT32, Shape.of(2, 0), false),
                                        1,
                                        LossReduction.NONE)).getMessage());

        Tensor dynamicSamples = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(0)),
                false);
        Tensor deferred = dynamicSamples.categoricalCrossEntropyWithLogits(
                tensor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(
                                new DynamicDimension("M"), new StaticDimension(0)),
                        false),
                1,
                LossReduction.NONE);
        assertTrue(deferred.descriptor().shape().dimension(0).isDynamic());
    }

    @Test
    void validatesNullTypeAxisShapeAndClassExtentBeforeIdentityAllocation() throws Exception {
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertEquals("logits", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.categoricalCrossEntropyWithLogits(
                        null, null, 0, null)).getMessage());
        assertEquals("target", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.categoricalCrossEntropyWithLogits(
                        floating, null, 0, null)).getMessage());
        assertEquals("reduction", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.categoricalCrossEntropyWithLogits(
                        floating, floating, 0, null)).getMessage());
        for (DataType invalid : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            Tensor invalidLogits = tensor(invalid, Shape.of(2, 3), false);
            assertEquals(
                    "categoricalCrossEntropyWithLogits logits must have a floating data type, "
                            + "but was " + invalid,
                    assertThrows(IllegalArgumentException.class,
                            () -> invalidLogits.categoricalCrossEntropyWithLogits(
                                    floating, 0, LossReduction.NONE)).getMessage());
        }
        assertEquals("categoricalCrossEntropyWithLogits target must have a floating data type, "
                        + "but was BOOL",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.categoricalCrossEntropyWithLogits(
                                tensor(DataType.BOOL, Shape.of(1), false),
                                7,
                                LossReduction.NONE)).getMessage());
        assertEquals("Axis 2 is outside shape rank 2",
                assertThrows(IndexOutOfBoundsException.class,
                        () -> floating.categoricalCrossEntropyWithLogits(
                                tensor(DataType.FLOAT32, Shape.of(3), false),
                                2,
                                LossReduction.NONE)).getMessage());
        assertEquals("Axis 0 is outside shape rank 0",
                assertThrows(IndexOutOfBoundsException.class,
                        () -> tensor(DataType.FLOAT32, Shape.scalar(), false)
                                .categoricalCrossEntropyWithLogits(
                                        tensor(DataType.FLOAT32, Shape.scalar(), false),
                                        0,
                                        LossReduction.NONE)).getMessage());
        assertEquals(before, next.get());
    }

    @Test
    void combinesGradientEligibilityKeepsInputsUntouchedAndFreshensEqualRequests()
            throws Exception {
        Shape shape = Shape.of(2, 3);
        Tensor logits = tensor(DataType.FLOAT64, shape, true);
        Tensor target = tensor(DataType.FLOAT32, shape, false);
        TensorDescriptor logitsDescriptor = logits.descriptor();
        TensorDescriptor targetDescriptor = target.descriptor();
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = logits.categoricalCrossEntropyWithLogits(
                target, 1, LossReduction.MEAN);
        Tensor second = logits.categoricalCrossEntropyWithLogits(
                target, 1, LossReduction.MEAN);

        assertAll(
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertSame(logitsDescriptor, logits.descriptor()),
                () -> assertSame(targetDescriptor, target.descriptor()),
                () -> assertSame(shape, logits.descriptor().shape()),
                () -> assertTrue(logits.hostStorage().isEmpty()),
                () -> assertTrue(target.hostStorage().isEmpty()));
    }

    @Test
    void locksStableFormulaRawWeightingAndEmptySpecialPoliciesWithoutReadingStorage() {
        double maximum = 3.0;
        double logSumExp = maximum + Math.log(
                Math.exp(1.0 - maximum)
                        + Math.exp(2.0 - maximum)
                        + Math.exp(3.0 - maximum));
        double oneHot = logSumExp - 3.0;
        double dense = 0.2 * (logSumExp - 1.0)
                + 0.3 * (logSumExp - 2.0)
                + 0.5 * (logSumExp - 3.0);

        assertAll(
                () -> assertEquals(0.407605964, oneHot, 1.0e-9),
                () -> assertEquals(1.107605964, dense, 1.0e-9),
                () -> assertEquals(0L, Double.doubleToRawLongBits(0.0d)),
                () -> assertTrue(Double.isNaN(
                        Double.POSITIVE_INFINITY - Double.POSITIVE_INFINITY)),
                () -> assertTrue(Double.isNaN(
                        Double.NEGATIVE_INFINITY - Double.NEGATIVE_INFINITY)),
                () -> assertEquals(Double.POSITIVE_INFINITY,
                        1.0d * Double.POSITIVE_INFINITY),
                () -> assertTrue(Double.isNaN(0.0d / 0.0d)));
    }

    private static List<DataType> floatingTypes() {
        return List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64);
    }

    private static DataType widest(DataType left, DataType right) {
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }
}
