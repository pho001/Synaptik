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
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

final class TensorBatchNormTrainingExpressionTest {
    @Test
    void exposesOnlyOneReceiverThreeComponentResultAndFieldFreePackageHelper()
            throws ReflectiveOperationException {
        var apply = TensorBatchNormTrainingExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, int.class, Tensor.class, Tensor.class, Tensor.class,
                Tensor.class, ScalarValue.class, ScalarValue.class);
        var receiver = Tensor.class.getDeclaredMethod(
                "batchNormTraining", int.class, Tensor.class, Tensor.class, Tensor.class,
                Tensor.class, ScalarValue.class, ScalarValue.class);
        var components = BatchNormTrainingResult.class.getRecordComponents();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        TensorBatchNormTrainingExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorBatchNormTrainingExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorBatchNormTrainingExpressions.class
                        .getDeclaredFields().length),
                () -> assertEquals(1, TensorBatchNormTrainingExpressions.class
                        .getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(TensorBatchNormTrainingExpressions.class
                        .getDeclaredConstructors()[0].getModifiers())),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertEquals(BatchNormTrainingResult.class, apply.getReturnType()),
                () -> assertTrue(Modifier.isPublic(receiver.getModifiers())),
                () -> assertFalse(Modifier.isStatic(receiver.getModifiers())),
                () -> assertEquals(BatchNormTrainingResult.class, receiver.getReturnType()),
                () -> assertTrue(BatchNormTrainingResult.class.isRecord()),
                () -> assertEquals(List.of("output", "nextRunningMean", "nextRunningVariance"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(Tensor.class, Tensor.class, Tensor.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(Set.of("output", "nextRunningMean", "nextRunningVariance",
                                "equals", "hashCode", "toString"),
                        Arrays.stream(BatchNormTrainingResult.class.getDeclaredMethods())
                                .map(method -> method.getName()).collect(java.util.stream.Collectors.toSet())),
                () -> assertThrows(NoSuchMethodException.class,
                        () -> Tensor.class.getDeclaredMethod("batchNormTraining",
                                BatchNormTrainingAttrs.class)));
    }

    @Test
    void createsExactlyFiveOutputsWithOrderedInputsSharedProducerShapesAndGradientFlags()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        Shape inputShape = Shape.of(2, 3, 4);
        Tensor input = tensor(DataType.BFLOAT16, inputShape, true);
        Tensor scale = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor bias = tensor(DataType.BFLOAT16, Shape.of(3), true);
        Tensor oldMean = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor oldVariance = tensor(DataType.FLOAT32, Shape.of(3), true);
        ScalarValue momentum = ScalarValue.float32(0.25f);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        long before = next.get();

        BatchNormTrainingResult result = input.batchNormTraining(
                1, scale, bias, oldMean, oldVariance, momentum, epsilon);
        Tensor output = result.output();
        Tensor nextMean = result.nextRunningMean();
        Tensor nextVariance = result.nextRunningVariance();
        TensorProducer producer = output.provenance().orElseThrow().producer();
        List<TensorDescriptor> descriptors = producer.outputDescriptors();
        Tensor savedMean = producer.output(3);
        Tensor savedInverseStandardDeviation = producer.output(4);

        assertAll(
                () -> assertEquals(before, output.id().value()),
                () -> assertEquals(before + 1, nextMean.id().value()),
                () -> assertEquals(before + 2, nextVariance.id().value()),
                () -> assertEquals(before + 5, next.get()),
                () -> assertEquals(5, producer.outputCount()),
                () -> assertSame(output, producer.output(0)),
                () -> assertSame(nextMean, producer.output(1)),
                () -> assertSame(nextVariance, producer.output(2)),
                () -> assertSame(producer, nextMean.provenance().orElseThrow().producer()),
                () -> assertSame(producer, nextVariance.provenance().orElseThrow().producer()),
                () -> assertSame(
                        producer, savedMean.provenance().orElseThrow().producer()),
                () -> assertSame(
                        producer,
                        savedInverseStandardDeviation
                                .provenance().orElseThrow().producer()),
                () -> assertEquals(0, output.provenance().orElseThrow().outputIndex()),
                () -> assertEquals(1, nextMean.provenance().orElseThrow().outputIndex()),
                () -> assertEquals(2, nextVariance.provenance().orElseThrow().outputIndex()),
                () -> assertEquals(
                        3, savedMean.provenance().orElseThrow().outputIndex()),
                () -> assertEquals(
                        4,
                        savedInverseStandardDeviation
                                .provenance().orElseThrow().outputIndex()),
                () -> assertSame(BatchNormKind.BATCH_NORM_TRAINING,
                        producer.operation().kind()),
                () -> assertEquals(new BatchNormTrainingAttrs(1, momentum, epsilon),
                        producer.operation().attrs()),
                () -> assertEquals(5, producer.inputs().size()),
                () -> assertSame(input, producer.inputs().get(0)),
                () -> assertSame(scale, producer.inputs().get(1)),
                () -> assertSame(bias, producer.inputs().get(2)),
                () -> assertSame(oldMean, producer.inputs().get(3)),
                () -> assertSame(oldVariance, producer.inputs().get(4)),
                () -> assertSame(inputShape, descriptors.get(0).shape()),
                () -> assertSame(descriptors.get(1).shape(), descriptors.get(2).shape()),
                () -> assertSame(descriptors.get(1).shape(), descriptors.get(3).shape()),
                () -> assertSame(descriptors.get(1).shape(), descriptors.get(4).shape()),
                () -> assertSame(inputShape.dimension(1),
                        descriptors.get(1).shape().dimension(0)),
                () -> assertTrue(descriptors.stream()
                        .allMatch(descriptor -> descriptor.dataType() == DataType.FLOAT32)),
                () -> assertTrue(descriptors.stream()
                        .allMatch(descriptor -> descriptor.layout().isEmpty())),
                () -> assertTrue(descriptors.get(0).requiresGrad()),
                () -> assertTrue(descriptors.get(1).requiresGrad()),
                () -> assertTrue(descriptors.get(2).requiresGrad()),
                () -> assertTrue(descriptors.get(3).requiresGrad()),
                () -> assertTrue(descriptors.get(4).requiresGrad()),
                () -> assertSame(descriptors.get(3), savedMean.descriptor()),
                () -> assertSame(
                        descriptors.get(4), savedInverseStandardDeviation.descriptor()),
                () -> assertTrue(savedMean.label().isEmpty()),
                () -> assertTrue(savedMean.hostStorage().isEmpty()),
                () -> assertTrue(savedInverseStandardDeviation.label().isEmpty()),
                () -> assertTrue(savedInverseStandardDeviation.hostStorage().isEmpty()),
                () -> assertTrue(List.of(output, nextMean, nextVariance).stream()
                        .allMatch(tensor -> tensor.label().isEmpty()
                                && tensor.hostStorage().isEmpty())));
    }

    @Test
    void derivesEachGradientFlagOnlyFromItsDirectDependencies() {
        Shape inputShape = Shape.of(2, 3);
        Tensor input = tensor(DataType.FLOAT64, inputShape, false);
        Tensor scale = tensor(DataType.FLOAT64, Shape.of(3), true);
        Tensor bias = tensor(DataType.FLOAT64, Shape.of(3), false);
        Tensor oldMean = tensor(DataType.FLOAT64, Shape.of(3), true);
        Tensor oldVariance = tensor(DataType.FLOAT64, Shape.of(3), false);

        TensorProducer producer = input.batchNormTraining(
                -1, scale, bias, oldMean, oldVariance,
                ScalarValue.float64(0.5), ScalarValue.float64(1.0e-5))
                .output().provenance().orElseThrow().producer();

        assertEquals(List.of(true, true, false, false, false),
                producer.outputDescriptors().stream()
                        .map(TensorDescriptor::requiresGrad).toList());
    }

    @Test
    void acceptsEmptyChannelsDeferredEqualityDynamicCountsAndOverflowingPositiveCounts() {
        Tensor empty = tensor(DataType.FLOAT32, Shape.of(0, 1), false);
        Tensor emptyVector = tensor(DataType.FLOAT32, Shape.of(0), false);
        assertEquals(Shape.of(0), empty.batchNormTraining(
                0, emptyVector, emptyVector, emptyVector, emptyVector,
                ScalarValue.float32(0.0f), ScalarValue.float32(1.0e-5f))
                .nextRunningMean().descriptor().shape());

        DynamicDimension channel = new DynamicDimension("C");
        Tensor dynamic = tensor(DataType.FLOAT64,
                Shape.ofDimensions(new DynamicDimension("N"), channel), false);
        Tensor unrelatedVector = tensor(DataType.FLOAT64,
                Shape.ofDimensions(new DynamicDimension("other")), false);
        BatchNormTrainingResult deferred = dynamic.batchNormTraining(
                -1, unrelatedVector, unrelatedVector, unrelatedVector, unrelatedVector,
                ScalarValue.float64(1.0), ScalarValue.float64(1.0e-5));
        assertSame(channel, deferred.nextRunningMean().descriptor().shape().dimension(0));

        Tensor overflow = tensor(DataType.FLOAT64,
                Shape.of(Long.MAX_VALUE, 2, 3), false);
        Tensor vector = tensor(DataType.FLOAT64, Shape.of(3), false);
        assertSame(overflow.descriptor().shape(), overflow.batchNormTraining(
                2, vector, vector, vector, vector,
                ScalarValue.float64(0.5), ScalarValue.float64(1.0e-5))
                .output().descriptor().shape());
    }

    @Test
    void rejectsStaticReductionCountsZeroAndOneOnlyForPositiveChannels() {
        Tensor vector = tensor(DataType.FLOAT32, Shape.of(3), false);
        for (Shape shape : List.of(Shape.of(0, 3), Shape.of(1, 3))) {
            IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                    () -> tensor(DataType.FLOAT32, shape, false).batchNormTraining(
                            1, vector, vector, vector, vector,
                            ScalarValue.float32(0.5f), ScalarValue.float32(1.0e-5f)));
            long count = shape.dimension(0).staticSize().orElseThrow();
            assertEquals("batchNormTraining reduction domain count " + count
                            + " must be at least 2 when channel extent is non-zero",
                    failure.getMessage());
        }
    }

    @Test
    void validatesNullTypeRankAxisVectorChannelCountPromotionAndScalarsInExactOrder()
            throws ReflectiveOperationException {
        Tensor validInput = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor validVector = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor integral = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor rankOne = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor wrongVector = tensor(DataType.FLOAT32, Shape.of(4), false);
        Tensor wideBias = tensor(DataType.FLOAT64, Shape.of(3), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Throwable nullInput = assertThrows(NullPointerException.class,
                () -> TensorBatchNormTrainingExpressions.apply(
                        null, 99, null, null, null, null, null, null));
        Throwable type = assertThrows(IllegalArgumentException.class,
                () -> integral.batchNormTraining(
                        99, validVector, validVector, validVector, validVector,
                        ScalarValue.float32(0.5f), ScalarValue.float32(1.0e-5f)));
        Throwable rank = assertThrows(IllegalArgumentException.class,
                () -> rankOne.batchNormTraining(
                        99, validVector, validVector, validVector, validVector,
                        ScalarValue.float32(0.5f), ScalarValue.float32(1.0e-5f)));
        Throwable axis = assertThrows(IndexOutOfBoundsException.class,
                () -> validInput.batchNormTraining(
                        2, validVector, validVector, validVector, validVector,
                        ScalarValue.float32(0.5f), ScalarValue.float32(1.0e-5f)));
        Throwable vectorRank = assertThrows(IllegalArgumentException.class,
                () -> validInput.batchNormTraining(
                        1, validInput, validVector, validVector, validVector,
                        ScalarValue.float32(0.5f), ScalarValue.float32(1.0e-5f)));
        Throwable channel = assertThrows(IllegalArgumentException.class,
                () -> validInput.batchNormTraining(
                        1, wrongVector, validVector, validVector, validVector,
                        ScalarValue.float32(0.5f), ScalarValue.float32(1.0e-5f)));
        Throwable momentumType = assertThrows(IllegalArgumentException.class,
                () -> validInput.batchNormTraining(
                        1, validVector, wideBias, validVector, validVector,
                        ScalarValue.float32(0.5f), ScalarValue.float64(1.0e-5)));
        Throwable epsilonType = assertThrows(IllegalArgumentException.class,
                () -> validInput.batchNormTraining(
                        1, validVector, validVector, validVector, validVector,
                        ScalarValue.float32(0.5f), ScalarValue.float64(1.0e-5)));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals(
                        "batchNormTraining input must have a floating data type, but was INT32",
                        type.getMessage()),
                () -> assertEquals(
                        "batchNormTraining input rank must be at least 2, but was 1",
                        rank.getMessage()),
                () -> assertEquals("Axis 2 is outside shape rank 2", axis.getMessage()),
                () -> assertEquals(
                        "batchNormTraining scale rank must be one, but was 2",
                        vectorRank.getMessage()),
                () -> assertEquals(
                        "batchNormTraining scale channel dimension mismatch: input=StaticDimension[size=3], scale=StaticDimension[size=4]",
                        channel.getMessage()),
                () -> assertEquals(
                        "batchNormTraining momentum data type must match result data type: momentum=FLOAT32, result=FLOAT64",
                        momentumType.getMessage()),
                () -> assertEquals(
                        "batchNormTraining epsilon data type must match result data type: epsilon=FLOAT64, result=FLOAT32",
                        epsilonType.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void repeatedCallsAreFreshResultRejectsNullsAndPartialExhaustionConsumesEarlierIds()
            throws ReflectiveOperationException {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), false);
        Tensor vector = tensor(DataType.FLOAT64, Shape.of(3), false);
        ScalarValue momentum = ScalarValue.float64(0.5);
        ScalarValue epsilon = ScalarValue.float64(1.0e-5);
        BatchNormTrainingResult first = input.batchNormTraining(
                1, vector, vector, vector, vector, momentum, epsilon);
        BatchNormTrainingResult second = input.batchNormTraining(
                1, vector, vector, vector, vector, momentum, epsilon);

        assertAll(
                () -> assertNotSame(first.output(), second.output()),
                () -> assertNotEquals(first.output().id(), second.output().id()),
                () -> assertEquals("output", assertThrows(NullPointerException.class,
                        () -> new BatchNormTrainingResult(null, null, null)).getMessage()),
                () -> assertEquals("nextRunningMean", assertThrows(NullPointerException.class,
                        () -> new BatchNormTrainingResult(first.output(), null, null)).getMessage()),
                () -> assertEquals("nextRunningVariance", assertThrows(NullPointerException.class,
                        () -> new BatchNormTrainingResult(
                                first.output(), first.nextRunningMean(), null)).getMessage()));

        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE - 1);
            claimed.set(false);
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> input.batchNormTraining(
                            1, vector, vector, vector, vector, momentum, epsilon));
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
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

    private static AtomicBoolean maximumClaimedState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
