package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorSumToShapeExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(30_000_000);

    

    @Test
    void acceptsEveryNumericTypeAndPreservesExactDescriptorMetadata() {
        for (DataType dataType : List.of(
                DataType.BFLOAT16,
                DataType.FLOAT32,
                DataType.FLOAT64,
                DataType.INT32,
                DataType.INT64)) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Shape inputShape = Shape.of(2, 3, 4);
                Shape targetShape = Shape.of(3, 1);
                Tensor input = tensor(
                        dataType,
                        inputShape,
                        Optional.of(LayoutDescriptor.contiguous(inputShape)),
                        requiresGrad,
                        Optional.of("source"));

                Tensor result = input.sumToShape(targetShape);

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertSame(targetShape, result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertNotSame(input, result));
            }
        }
    }

    @Test
    void recordsExactOperationProducerAndOutputIndexZeroProvenance() {
        Shape inputShape = Shape.of(2, 3, 4);
        Shape targetShape = Shape.of(3, 1);
        Tensor input = tensor(DataType.FLOAT64, inputShape, Optional.empty(), true, Optional.empty());

        Tensor result = input.sumToShape(targetShape);
        TensorProvenance provenance = result.provenance().orElseThrow();
        TensorProducer producer = provenance.producer();
        SumToShapeAttrs attrs = (SumToShapeAttrs) provenance.operation().attrs();

        assertAll(
                () -> assertSame(AggregateReductionKind.SUM, provenance.operation().kind()),
                () -> assertSame(targetShape, attrs.targetShape()),
                () -> assertEquals(
                        OperationSignature.fixed(SumToShapeAttrs.class, 1, 1),
                        provenance.operation().signature()),
                () -> assertEquals(1, producer.inputs().size()),
                () -> assertSame(input, producer.inputs().get(0)),
                () -> assertEquals(1, producer.outputCount()),
                () -> assertSame(result.descriptor(), producer.outputDescriptors().get(0)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
    }

    @Test
    void appliesLeadingAlignedEqualSingletonAndScalarTargetRules() {
        Tensor rankThree = tensor(
                DataType.FLOAT32, Shape.of(2, 3, 4), Optional.empty(), true, Optional.empty());
        Tensor scalar = tensor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), true, Optional.empty());

        Tensor lowerRank = rankThree.sumToShape(Shape.of(3, 1));
        Tensor equal = rankThree.sumToShape(rankThree.descriptor().shape());
        Tensor scalarTarget = rankThree.sumToShape(Shape.scalar());
        Tensor scalarResult = scalar.sumToShape(Shape.scalar());
        Tensor equalAgain = rankThree.sumToShape(rankThree.descriptor().shape());

        assertAll(
                () -> assertEquals(Shape.of(3, 1), lowerRank.descriptor().shape()),
                () -> assertSame(rankThree.descriptor().shape(), equal.descriptor().shape()),
                () -> assertSame(Shape.scalar(), scalarTarget.descriptor().shape()),
                () -> assertSame(Shape.scalar(), scalarResult.descriptor().shape()),
                () -> assertNotSame(rankThree, equal),
                () -> assertNotEquals(equal.id(), equalAgain.id()),
                () -> assertNotSame(
                        equal.provenance().orElseThrow().producer(),
                        equalAgain.provenance().orElseThrow().producer()));
    }

    @Test
    void acceptsExactZeroExtentCasesAndRejectsTargetZeroAgainstNonzeroSource() {
        Tensor zero = tensor(
                DataType.INT64, Shape.of(0), Optional.empty(), false, Optional.empty());
        Tensor nonzero = tensor(
                DataType.INT64, Shape.of(2), Optional.empty(), false, Optional.empty());

        Tensor emptySum = zero.sumToShape(Shape.of(1));
        Tensor emptyResult = zero.sumToShape(Shape.of(0));
        assertThrows(IllegalArgumentException.class,
        () -> nonzero.sumToShape(Shape.of(0)));

        assertAll(
                () -> assertEquals(Shape.of(1), emptySum.descriptor().shape()),
                () -> assertEquals(Shape.of(0), emptyResult.descriptor().shape()));
    }

    @Test
    void acceptsEveryUnresolvedAlignedPairCategoryAndRetainsExactTargetReferences() {
        Dimension named = new DynamicDimension("N");
        Dimension equalNamed = new DynamicDimension("N");
        Dimension otherNamed = new DynamicDimension("M");
        Dimension expression = DimensionExpressions.addConstant(new DynamicDimension("K"), 2);
        Shape equalTarget = Shape.ofDimensions(equalNamed, new StaticDimension(4));
        Shape unresolvedToStatic = Shape.of(7, 4);
        Shape staticToUnresolved = Shape.ofDimensions(otherNamed, new StaticDimension(4));
        Shape unresolvedToUnresolved = Shape.ofDimensions(expression, new StaticDimension(4));

        Tensor unresolvedInput = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(named, new StaticDimension(4)),
                Optional.empty(),
                true,
                Optional.empty());
        Tensor staticInput = tensor(
                DataType.FLOAT32, Shape.of(8, 4), Optional.empty(), true, Optional.empty());

        Tensor equal = unresolvedInput.sumToShape(equalTarget);
        Tensor dynamicAgainstStatic = unresolvedInput.sumToShape(unresolvedToStatic);
        Tensor staticAgainstDynamic = staticInput.sumToShape(staticToUnresolved);
        Tensor unequalUnresolved = unresolvedInput.sumToShape(unresolvedToUnresolved);

        assertAll(
                () -> assertSame(equalTarget, equal.descriptor().shape()),
                () -> assertSame(unresolvedToStatic, dynamicAgainstStatic.descriptor().shape()),
                () -> assertSame(staticToUnresolved, staticAgainstDynamic.descriptor().shape()),
                () -> assertSame(unresolvedToUnresolved, unequalUnresolved.descriptor().shape()),
                () -> assertSame(equalTarget,
                        ((SumToShapeAttrs) equal.provenance().orElseThrow().operation().attrs())
                                .targetShape()),
                () -> assertEquals(named, equalNamed),
                () -> assertNotEquals(named, otherNamed));
    }

    @Test
    void validatesInExactOrderReportsFirstStaticMismatchAndConsumesNoId() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor numeric = tensor(
                DataType.FLOAT32, Shape.of(2, 3, 4), Optional.empty(), true, Optional.empty());
        Tensor boolScalar = tensor(
                DataType.BOOL, Shape.scalar(), Optional.empty(), false, Optional.empty());
        Tensor scalar = tensor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), true, Optional.empty());
        long before = next.get();

        assertThrows(NullPointerException.class,
        () -> TensorSumToShapeExpressions.apply(null, null));
        assertThrows(NullPointerException.class,
        () -> TensorSumToShapeExpressions.apply(numeric, null));
        assertThrows(IllegalArgumentException.class,
        () -> boolScalar.sumToShape(Shape.of(1)));
        assertThrows(IllegalArgumentException.class,
        () -> scalar.sumToShape(Shape.of(1)));
        assertThrows(IllegalArgumentException.class,
        () -> numeric.sumToShape(Shape.of(5, 6)));

        assertAll(
                () -> assertEquals(before, next.get()));
    }

    @Test
    void consumesExactlyOneFreshIdAndLeavesInputStateUnchanged() throws Exception {
        Shape inputShape = Shape.of(2, 3);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(inputShape);
        Tensor input = tensor(
                DataType.FLOAT64,
                inputShape,
                Optional.of(layout),
                true,
                Optional.of(" input "));
        TensorDescriptor descriptor = input.descriptor();
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor result = input.sumToShape(Shape.of(1, 3));

        assertAll(
                () -> assertEquals(before, result.id().value()),
                () -> assertEquals(before + 1, next.get()),
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(inputShape, input.descriptor().shape()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertTrue(input.provenance().isEmpty()),
                () -> assertTrue(input.hostStorage().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void propagatesCurrentIdentifierExhaustionAtFactoryDelegation() throws Exception {
        Tensor input = tensor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), true, Optional.empty());
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumTensorIdClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            assertThrows(IllegalStateException.class,
            () -> input.sumToShape(Shape.of(1)));

            assertAll(
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static List<Boolean> validGradientChoices(DataType dataType) {
        return dataType.isDifferentiable() ? List.of(false, true) : List.of(false);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumTensorIdClaimedState()
            throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad,
            Optional<String> label) {
        return new Tensor(
                new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, layout, requiresGrad),
                label,
                Optional.empty(),
                Optional.empty());
    }

}
