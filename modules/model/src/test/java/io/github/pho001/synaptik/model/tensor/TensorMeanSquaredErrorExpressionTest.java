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
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
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

class TensorMeanSquaredErrorExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(130_000);

    @Test
    void exposesExactlyOneReceiverAndOnePackageEntry() throws ReflectiveOperationException {
        Method receiver = Tensor.class.getDeclaredMethod(
                "meanSquaredError", Tensor.class, LossReduction.class);
        Method entry = TensorLossExpressions.class.getDeclaredMethod(
                "meanSquaredError", Tensor.class, Tensor.class, LossReduction.class);
        Set<String> methodNames = Arrays.stream(TensorLossExpressions.class.getDeclaredMethods())
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
                () -> assertEquals(0, TensorLossExpressions.class.getDeclaredClasses().length),
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
                        methodNames),
                () -> assertEquals(208, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()),
                () -> assertEquals(1, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .filter(method -> method.getName().equals("meanSquaredError")).count()));
    }

    @Test
    void promotesAllFloatingPairsAndRecordsExactOrderedMetadata() {
        Shape shape = Shape.of(2, 3);
        for (DataType predictionType : floatingTypes()) {
            for (DataType targetType : floatingTypes()) {
                Tensor prediction = tensor(predictionType, shape, false);
                Tensor target = tensor(targetType, Shape.of(2, 3), true);
                Tensor result = prediction.meanSquaredError(target, LossReduction.NONE);
                TensorProvenance provenance = result.provenance().orElseThrow();
                MeanSquaredErrorAttrs attrs = (MeanSquaredErrorAttrs) provenance.operation().attrs();

                assertAll(
                        () -> assertSame(widest(predictionType, targetType),
                                result.descriptor().dataType()),
                        () -> assertSame(shape, result.descriptor().shape()),
                        () -> assertTrue(result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertSame(LossKind.MEAN_SQUARED_ERROR,
                                provenance.operation().kind()),
                        () -> assertSame(LossReduction.NONE, attrs.reduction()),
                        () -> assertEquals(List.of(prediction, target), provenance.inputs()),
                        () -> assertEquals(0, provenance.outputIndex()),
                        () -> assertEquals(1, provenance.producer().outputCount()),
                        () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
            }
        }
    }

    @Test
    void selectsExactNoneShapeAndSharedScalarReducedShapesIncludingEmptyAndScalar() {
        Shape empty = Shape.of(2, 0, 3);
        Tensor prediction = tensor(DataType.FLOAT32, empty, false);
        Tensor target = tensor(DataType.FLOAT32, Shape.of(2, 0, 3), false);

        Tensor none = prediction.meanSquaredError(target, LossReduction.NONE);
        Tensor sum = prediction.meanSquaredError(target, LossReduction.SUM);
        Tensor mean = prediction.meanSquaredError(target, LossReduction.MEAN);
        Tensor scalar = tensor(DataType.FLOAT64, Shape.scalar(), false)
                .meanSquaredError(tensor(DataType.FLOAT64, Shape.scalar(), false),
                        LossReduction.NONE);

        assertAll(
                () -> assertSame(empty, none.descriptor().shape()),
                () -> assertSame(Shape.scalar(), sum.descriptor().shape()),
                () -> assertSame(Shape.scalar(), mean.descriptor().shape()),
                () -> assertSame(Shape.scalar(), scalar.descriptor().shape()),
                () -> assertFalse(none.descriptor().requiresGrad()),
                () -> assertEquals(0L, none.descriptor().shape().knownElementCount().orElseThrow()));
    }

    @Test
    void acceptsStructuralEqualityAndDefersEveryUnequalUnresolvedPair() {
        DynamicDimension n = new DynamicDimension("N");
        DynamicDimension m = new DynamicDimension("M");
        Shape predictionShape = Shape.ofDimensions(n, new StaticDimension(4));
        Tensor prediction = tensor(DataType.FLOAT32, predictionShape, false);

        Tensor exact = prediction.meanSquaredError(
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(n, new StaticDimension(4)), false),
                LossReduction.NONE);
        Tensor unresolved = prediction.meanSquaredError(
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(m, new StaticDimension(4)), false),
                LossReduction.NONE);
        Tensor unresolvedStatic = prediction.meanSquaredError(
                tensor(DataType.FLOAT32, Shape.of(7, 4), false), LossReduction.NONE);

        assertAll(
                () -> assertSame(predictionShape, exact.descriptor().shape()),
                () -> assertSame(predictionShape, unresolved.descriptor().shape()),
                () -> assertSame(predictionShape, unresolvedStatic.descriptor().shape()));
    }

    @Test
    void rejectsRankAndStaticDimensionMismatchWithoutBroadcasting() {
        Tensor prediction = tensor(DataType.FLOAT32, Shape.of(2, 3), false);

        assertEquals(
                "meanSquaredError target rank must equal prediction rank: prediction=2, target=1",
                assertThrows(IllegalArgumentException.class,
                        () -> prediction.meanSquaredError(
                                tensor(DataType.FLOAT32, Shape.of(3), false),
                                LossReduction.NONE)).getMessage());
        assertEquals(
                "meanSquaredError target dimension mismatch at axis 0: prediction=StaticDimension[size=2], target=StaticDimension[size=1]",
                assertThrows(IllegalArgumentException.class,
                        () -> prediction.meanSquaredError(
                                tensor(DataType.FLOAT32, Shape.of(1, 3), false),
                                LossReduction.NONE)).getMessage());
        assertEquals(
                "meanSquaredError target dimension mismatch at axis 1: prediction=StaticDimension[size=3], target=StaticDimension[size=1]",
                assertThrows(IllegalArgumentException.class,
                        () -> prediction.meanSquaredError(
                                tensor(DataType.FLOAT32, Shape.of(2, 1), false),
                                LossReduction.NONE)).getMessage());
    }

    @Test
    void validatesNullTypeRankAndDimensionOrderBeforeIdentityAllocation() throws Exception {
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor integralTarget = tensor(DataType.INT64, Shape.of(4), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertEquals("prediction", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.meanSquaredError(null, null, null)).getMessage());
        assertEquals("target", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.meanSquaredError(floating, null, null)).getMessage());
        assertEquals("reduction", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.meanSquaredError(
                        floating, floating, null)).getMessage());
        assertEquals("target", assertThrows(NullPointerException.class,
                () -> floating.meanSquaredError(null, LossReduction.NONE)).getMessage());
        for (DataType invalid : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            Tensor invalidPrediction = tensor(invalid, Shape.of(2, 3), false);
            Tensor invalidTarget = tensor(invalid, Shape.of(2, 3), false);
            assertEquals(
                    "meanSquaredError prediction must have a floating data type, but was "
                            + invalid,
                    assertThrows(IllegalArgumentException.class,
                            () -> invalidPrediction.meanSquaredError(
                                    floating, LossReduction.NONE)).getMessage());
            assertEquals(
                    "meanSquaredError target must have a floating data type, but was " + invalid,
                    assertThrows(IllegalArgumentException.class,
                            () -> floating.meanSquaredError(
                                    invalidTarget, LossReduction.NONE)).getMessage());
        }
        assertEquals("meanSquaredError target must have a floating data type, but was INT64",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.meanSquaredError(
                                integralTarget, LossReduction.NONE)).getMessage());
        assertEquals(before, next.get());
    }

    @Test
    void combinesEitherGradientRequestAndKeepsEqualRequestsDistinct() throws Exception {
        Shape shape = Shape.of(3);
        Tensor prediction = tensor(DataType.FLOAT64, shape, true);
        Tensor target = tensor(DataType.FLOAT64, shape, false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = prediction.meanSquaredError(target, LossReduction.MEAN);
        Tensor second = prediction.meanSquaredError(target, LossReduction.MEAN);

        assertAll(
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertSame(LossReduction.MEAN,
                        ((MeanSquaredErrorAttrs) first.provenance().orElseThrow()
                                .operation().attrs()).reduction()));
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
