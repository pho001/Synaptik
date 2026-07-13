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
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
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

class TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(150_000);

    @Test
    void exposesExactlyTwoReceiverFormsAndNoPublicOptional() throws Exception {
        Method noIgnore = Tensor.class.getDeclaredMethod(
                "categoricalCrossEntropyWithLogits",
                Tensor.class,
                int.class,
                LossReduction.class);
        Method ignored = Tensor.class.getDeclaredMethod(
                "categoricalCrossEntropyWithLogits",
                Tensor.class,
                int.class,
                LossReduction.class,
                ScalarValue.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(noIgnore.getModifiers())),
                () -> assertTrue(Modifier.isPublic(ignored.getModifiers())),
                () -> assertFalse(Modifier.isStatic(ignored.getModifiers())),
                () -> assertEquals(Tensor.class, ignored.getReturnType()),
                () -> assertEquals(2, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .filter(method -> method.getName().equals(
                                "categoricalCrossEntropyWithLogits"))
                        .count()),
                () -> assertEquals(192, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count()),
                () -> assertTrue(Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .flatMap(method -> Arrays.stream(method.getParameterTypes()))
                        .noneMatch(Optional.class::equals)));
    }

    @Test
    void dispatchesEveryIntegralTargetWithoutIgnoreAndRecordsExactMetadata() {
        for (DataType logitsType : floatingTypes()) {
            for (DataType targetType : integralTypes()) {
                Shape targetShape = Shape.of(2, 4);
                Tensor logits = tensor(logitsType, Shape.of(2, 3, 4), true);
                Tensor target = tensor(targetType, targetShape, false);
                Tensor result = logits.categoricalCrossEntropyWithLogits(
                        target, -2, LossReduction.NONE);
                TensorProvenance provenance = result.provenance().orElseThrow();
                IndexCategoricalCrossEntropyWithLogitsAttrs attrs =
                        (IndexCategoricalCrossEntropyWithLogitsAttrs)
                                provenance.operation().attrs();

                assertAll(
                        () -> assertSame(logitsType, result.descriptor().dataType()),
                        () -> assertSame(targetShape, result.descriptor().shape()),
                        () -> assertTrue(result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertSame(
                                LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                                provenance.operation().kind()),
                        () -> assertEquals(1, attrs.axis()),
                        () -> assertSame(LossReduction.NONE, attrs.reduction()),
                        () -> assertTrue(attrs.ignoreIndex().isEmpty()),
                        () -> assertEquals(List.of(logits, target), provenance.inputs()),
                        () -> assertEquals(0, provenance.outputIndex()),
                        () -> assertEquals(1, provenance.producer().outputCount()),
                        () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
            }
        }
    }

    @Test
    void storesExactTypedIgnoreAsAttributesRatherThanAProducerInput() {
        ScalarValue ignore = ScalarValue.int64(-1);
        Tensor logits = tensor(DataType.FLOAT64, Shape.of(2, 3), false);
        Tensor target = tensor(DataType.INT64, Shape.of(2), false);
        Tensor result = logits.categoricalCrossEntropyWithLogits(
                target, 1, LossReduction.MEAN, ignore);
        TensorProvenance provenance = result.provenance().orElseThrow();
        IndexCategoricalCrossEntropyWithLogitsAttrs attrs =
                (IndexCategoricalCrossEntropyWithLogitsAttrs) provenance.operation().attrs();

        assertAll(
                () -> assertSame(ignore, attrs.ignoreIndex().orElseThrow()),
                () -> assertEquals(List.of(logits, target), provenance.inputs()),
                () -> assertEquals(2, provenance.inputs().size()),
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertSame(Shape.scalar(), result.descriptor().shape()),
                () -> assertFalse(result.descriptor().requiresGrad()));
    }

    @Test
    void mapsTargetShapeAcrossEveryClassAxisAndPreservesNoneShapeIdentity() {
        Shape targetAxisZero = Shape.of(3, 4);
        Shape targetAxisOne = Shape.of(2, 4);
        Shape targetAxisTwo = Shape.of(2, 3);
        Tensor logits = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false);

        Tensor axisZero = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32, targetAxisZero, false), 0, LossReduction.NONE);
        Tensor axisOne = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32, targetAxisOne, false), 1, LossReduction.NONE);
        Tensor axisTwo = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32, targetAxisTwo, false), 2, LossReduction.NONE);
        Tensor rankOne = tensor(DataType.FLOAT32, Shape.of(3), false)
                .categoricalCrossEntropyWithLogits(
                        tensor(DataType.INT64, Shape.scalar(), false),
                        0,
                        LossReduction.NONE);
        Tensor sum = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32, Shape.of(2, 4), false), 1, LossReduction.SUM);

        assertAll(
                () -> assertSame(targetAxisZero, axisZero.descriptor().shape()),
                () -> assertSame(targetAxisOne, axisOne.descriptor().shape()),
                () -> assertSame(targetAxisTwo, axisTwo.descriptor().shape()),
                () -> assertSame(Shape.scalar(), rankOne.descriptor().shape()),
                () -> assertSame(Shape.scalar(), sum.descriptor().shape()));
    }

    @Test
    void acceptsStructuralAndDeferredMappingButRejectsStaticMismatchWithoutBroadcasting() {
        DynamicDimension n = new DynamicDimension("N");
        DynamicDimension m = new DynamicDimension("M");
        Tensor logits = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(n, new StaticDimension(3), new StaticDimension(4)),
                false);

        Tensor exact = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32,
                        Shape.ofDimensions(n, new StaticDimension(4)), false),
                1,
                LossReduction.NONE);
        Tensor deferred = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32,
                        Shape.ofDimensions(m, new StaticDimension(4)), false),
                1,
                LossReduction.NONE);
        Tensor dynamicStatic = logits.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32, Shape.of(7, 4), false),
                1,
                LossReduction.NONE);

        assertAll(
                () -> assertSame(n, exact.descriptor().shape().dimension(0)),
                () -> assertSame(m, deferred.descriptor().shape().dimension(0)),
                () -> assertEquals(new StaticDimension(7),
                        dynamicStatic.descriptor().shape().dimension(0)),
                () -> assertEquals(
                        "categoricalCrossEntropyWithLogits index target dimension mismatch at "
                                + "target axis 1 (logits axis 2): logits=StaticDimension[size=4], "
                                + "target=StaticDimension[size=1]",
                        assertThrows(IllegalArgumentException.class,
                                () -> logits.categoricalCrossEntropyWithLogits(
                                        tensor(DataType.INT32, Shape.of(7, 1), false),
                                        1,
                                        LossReduction.NONE)).getMessage()));
    }

    @Test
    void distinguishesZeroClassNoIgnoreIgnorePresentAndEmptySampleDomains() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor nonEmpty = tensor(DataType.FLOAT32, Shape.of(2, 0), false);
        Tensor nonEmptyTarget = tensor(DataType.INT64, Shape.of(2), false);
        long beforeFailure = next.get();

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> nonEmpty.categoricalCrossEntropyWithLogits(
                        nonEmptyTarget, 1, LossReduction.NONE));
        Tensor ignored = nonEmpty.categoricalCrossEntropyWithLogits(
                nonEmptyTarget, 1, LossReduction.NONE, ScalarValue.int64(-1));
        Tensor emptyNoIgnore = tensor(DataType.FLOAT32, Shape.of(0, 0), false)
                .categoricalCrossEntropyWithLogits(
                        tensor(DataType.INT32, Shape.of(0), false),
                        1,
                        LossReduction.NONE);
        Tensor emptyIgnored = tensor(DataType.FLOAT32, Shape.of(0, 0), false)
                .categoricalCrossEntropyWithLogits(
                        tensor(DataType.INT32, Shape.of(0), false),
                        1,
                        LossReduction.MEAN,
                        ScalarValue.int32(-1));
        Tensor oneClass = tensor(DataType.FLOAT64, Shape.of(2, 1), false)
                .categoricalCrossEntropyWithLogits(
                        tensor(DataType.INT64, Shape.of(2), false),
                        1,
                        LossReduction.NONE);

        assertAll(
                () -> assertEquals(
                        "categoricalCrossEntropyWithLogits class dimension must be positive when "
                                + "sample domain is non-empty: axis=1, "
                                + "dimension=StaticDimension[size=0]",
                        failure.getMessage()),
                () -> assertEquals(beforeFailure + 4, next.get()),
                () -> assertSame(nonEmptyTarget.descriptor().shape(),
                        ignored.descriptor().shape()),
                () -> assertEquals(0L,
                        emptyNoIgnore.descriptor().shape().knownElementCount().orElseThrow()),
                () -> assertSame(Shape.scalar(), emptyIgnored.descriptor().shape()),
                () -> assertEquals(Shape.of(2), oneClass.descriptor().shape()));
    }

    @Test
    void retainsDynamicClassAndSampleObligationsWithoutReadingValues() {
        DynamicDimension classes = new DynamicDimension("C");
        DynamicDimension samples = new DynamicDimension("S");
        Tensor dynamicClass = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(2), classes),
                false);
        Tensor dynamicSamplesZeroClass = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(samples, new StaticDimension(0)),
                false);

        Tensor classDeferred = dynamicClass.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32, Shape.of(2), false),
                1,
                LossReduction.NONE);
        Tensor sampleDeferred = dynamicSamplesZeroClass.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32,
                        Shape.ofDimensions(new DynamicDimension("T")), false),
                1,
                LossReduction.NONE);
        Tensor ignoredAlternative = dynamicSamplesZeroClass.categoricalCrossEntropyWithLogits(
                tensor(DataType.INT32,
                        Shape.ofDimensions(new DynamicDimension("U")), false),
                1,
                LossReduction.NONE,
                ScalarValue.int32(-1));

        assertAll(
                () -> assertEquals(Shape.of(2), classDeferred.descriptor().shape()),
                () -> assertTrue(sampleDeferred.descriptor().shape().dimension(0).isDynamic()),
                () -> assertTrue(ignoredAlternative.descriptor().shape().dimension(0).isDynamic()));
    }

    @Test
    void validatesNullTypesIgnoreAxisAndShapeInExactOrderWithoutConsumingIds()
            throws Exception {
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor index = tensor(DataType.INT32, Shape.of(2), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertEquals("logits", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.categoricalCrossEntropyWithLogits(
                        null, null, 0, null, null)).getMessage());
        assertEquals("target", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.categoricalCrossEntropyWithLogits(
                        floating, null, 0, null, null)).getMessage());
        assertEquals("reduction", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.categoricalCrossEntropyWithLogits(
                        floating, index, 0, null, null)).getMessage());
        assertEquals("ignoreIndex", assertThrows(NullPointerException.class,
                () -> TensorLossExpressions.categoricalCrossEntropyWithLogits(
                        floating, index, 0, LossReduction.NONE, null)).getMessage());
        assertEquals(
                "categoricalCrossEntropyWithLogits logits must have a floating data type, but was "
                        + "INT32",
                assertThrows(IllegalArgumentException.class,
                        () -> tensor(DataType.INT32, Shape.of(2, 3), false)
                                .categoricalCrossEntropyWithLogits(
                                        index, 1, LossReduction.NONE)).getMessage());
        assertEquals(
                "categoricalCrossEntropyWithLogits target must have data type INT32 or INT64 when "
                        + "ignoreIndex is present, but was FLOAT64",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.categoricalCrossEntropyWithLogits(
                                tensor(DataType.FLOAT64, Shape.of(2), false),
                                9,
                                LossReduction.NONE,
                                ScalarValue.int32(-1))).getMessage());
        assertEquals(
                "ignoreIndex must have data type INT32 or INT64, but was FLOAT32",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.categoricalCrossEntropyWithLogits(
                                index, 9, LossReduction.NONE, ScalarValue.float32(-1))).getMessage());
        assertEquals(
                "categoricalCrossEntropyWithLogits ignoreIndex data type must equal target data "
                        + "type: target=INT32, ignoreIndex=INT64",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.categoricalCrossEntropyWithLogits(
                                index, 9, LossReduction.NONE, ScalarValue.int64(-1))).getMessage());
        assertEquals("Axis 2 is outside shape rank 2",
                assertThrows(IndexOutOfBoundsException.class,
                        () -> floating.categoricalCrossEntropyWithLogits(
                                index, 2, LossReduction.NONE)).getMessage());
        assertEquals("Axis 0 is outside shape rank 0",
                assertThrows(IndexOutOfBoundsException.class,
                        () -> tensor(DataType.FLOAT32, Shape.scalar(), false)
                                .categoricalCrossEntropyWithLogits(
                                        tensor(DataType.INT32, Shape.scalar(), false),
                                        0,
                                        LossReduction.NONE)).getMessage());
        assertEquals(
                "categoricalCrossEntropyWithLogits index target rank must equal logits rank minus "
                        + "one: logits=2, target=2",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.categoricalCrossEntropyWithLogits(
                                tensor(DataType.INT32, Shape.of(2, 1), false),
                                1,
                                LossReduction.NONE)).getMessage());
        assertEquals(before, next.get());
    }

    @Test
    void usesLogitsOnlyEligibilityLeavesInputsUntouchedAndFreshensRequests() throws Exception {
        Shape logitsShape = Shape.of(2, 3);
        Shape targetShape = Shape.of(2);
        Tensor logits = tensor(DataType.FLOAT32, logitsShape, false);
        Tensor target = tensor(DataType.INT64, targetShape, false);
        TensorDescriptor logitsDescriptor = logits.descriptor();
        TensorDescriptor targetDescriptor = target.descriptor();
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = logits.categoricalCrossEntropyWithLogits(
                target, 1, LossReduction.MEAN);
        Tensor second = logits.categoricalCrossEntropyWithLogits(
                target, 1, LossReduction.MEAN);

        assertAll(
                () -> assertFalse(first.descriptor().requiresGrad()),
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertSame(logitsDescriptor, logits.descriptor()),
                () -> assertSame(targetDescriptor, target.descriptor()),
                () -> assertSame(logitsShape, logits.descriptor().shape()),
                () -> assertSame(targetShape, target.descriptor().shape()),
                () -> assertTrue(logits.hostStorage().isEmpty()),
                () -> assertTrue(target.hostStorage().isEmpty()));
    }

    @Test
    void locksIgnoreBeforeBoundsReductionAndSpecialValueMeaningWithoutExecution() {
        double maximum = 3.0;
        double logSumExp = maximum + Math.log(
                Math.exp(1.0 - maximum)
                        + Math.exp(2.0 - maximum)
                        + Math.exp(3.0 - maximum));
        double targetTwo = logSumExp - 3.0;
        double targetZero = logSumExp - 1.0;
        double ignoredOutOfBounds = 0.0d;

        assertAll(
                () -> assertEquals(0.407605964, targetTwo, 1.0e-9),
                () -> assertEquals(2.407605964, targetZero, 1.0e-9),
                () -> assertEquals(0L, Double.doubleToRawLongBits(ignoredOutOfBounds)),
                () -> assertEquals((targetTwo + targetZero) / 2.0,
                        (targetTwo + ignoredOutOfBounds + targetZero) / 2.0, 0.0),
                () -> assertTrue(Double.isNaN(0.0d / 0.0d)),
                () -> assertTrue(Double.isNaN(Double.NaN + targetTwo)),
                () -> assertTrue(Double.isNaN(
                        Double.POSITIVE_INFINITY - Double.POSITIVE_INFINITY)),
                () -> assertTrue(Double.isNaN(
                        Double.NEGATIVE_INFINITY - Double.NEGATIVE_INFINITY)),
                () -> assertEquals(Double.POSITIVE_INFINITY,
                        logSumExp - Double.NEGATIVE_INFINITY));
    }

    private static List<DataType> floatingTypes() {
        return List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64);
    }

    private static List<DataType> integralTypes() {
        return List.of(DataType.INT32, DataType.INT64);
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
