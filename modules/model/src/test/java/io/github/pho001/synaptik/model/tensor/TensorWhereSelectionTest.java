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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorWhereSelectionTest {
    private static final AtomicLong NEXT_INPUT_ID = new AtomicLong(10_000);

    @Test
    void exposesExactPublicEntryAndPackagePrivateHelperShape()
            throws ReflectiveOperationException {
        var constructors = TensorWhereExpressions.class.getDeclaredConstructors();
        var methods = Arrays.stream(TensorWhereExpressions.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic())
                .toList();
        var where = Tensor.class.getDeclaredMethod(
                "where", Tensor.class, Tensor.class, Tensor.class);

        assertAll(
                () -> assertEquals(Tensor.class, where.getReturnType()),
                () -> assertEquals(
                        List.of(Tensor.class, Tensor.class, Tensor.class),
                        Arrays.asList(where.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(where.getModifiers())),
                () -> assertTrue(Modifier.isStatic(where.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(where.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorWhereExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorWhereExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isProtected(TensorWhereExpressions.class.getModifiers())),
                () -> assertFalse(TensorWhereExpressions.class.isRecord()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(0, TensorWhereExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorWhereExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, methods.size()),
                () -> assertEquals("apply", methods.getFirst().getName()),
                () -> assertEquals(Tensor.class, methods.getFirst().getReturnType()),
                () -> assertEquals(
                        List.of(Tensor.class, Tensor.class, Tensor.class),
                        Arrays.asList(methods.getFirst().getParameterTypes())),
                () -> assertTrue(Modifier.isStatic(methods.getFirst().getModifiers())),
                () -> assertFalse(Modifier.isPublic(methods.getFirst().getModifiers())),
                () -> assertFalse(Modifier.isProtected(methods.getFirst().getModifiers())),
                () -> assertFalse(Modifier.isPrivate(methods.getFirst().getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(methods.getFirst().getModifiers())));
    }

    @Test
    void constructsExactOperationDescriptorAndOrderedProvenance() {
        Tensor condition = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor ifTrue = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor ifFalse = tensor(DataType.FLOAT32, Shape.of(2, 3), false);

        Tensor result = Tensor.where(condition, ifTrue, ifFalse);
        TensorProvenance provenance = result.provenance().orElseThrow();

        assertAll(
                () -> assertSame(WhereSelectionKind.WHERE, provenance.operation().kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE, provenance.operation().attrs()),
                () -> assertEquals(3, provenance.inputs().size()),
                () -> assertSame(condition, provenance.inputs().get(0)),
                () -> assertSame(ifTrue, provenance.inputs().get(1)),
                () -> assertSame(ifFalse, provenance.inputs().get(2)),
                () -> assertEquals(DataType.FLOAT32, result.descriptor().dataType()),
                () -> assertEquals(ifTrue.descriptor().shape(), result.descriptor().shape()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void acceptsEveryFloatingBranchPairWithExactPromotion() {
        List<DataType> floating = List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64);
        Tensor condition = tensor(DataType.BOOL, Shape.scalar(), false);

        for (DataType ifTrueType : floating) {
            for (DataType ifFalseType : floating) {
                Tensor result = Tensor.where(
                        condition,
                        tensor(ifTrueType, Shape.scalar(), false),
                        tensor(ifFalseType, Shape.scalar(), false));
                DataType expected =
                        ifTrueType == DataType.FLOAT64 || ifFalseType == DataType.FLOAT64
                                ? DataType.FLOAT64
                                : ifTrueType == DataType.FLOAT32 || ifFalseType == DataType.FLOAT32
                                        ? DataType.FLOAT32
                                        : DataType.BFLOAT16;
                assertEquals(expected, result.descriptor().dataType());
            }
        }
    }

    @Test
    void composesBranchFirstThenConditionBroadcasting() {
        record BroadcastCase(Shape condition, Shape ifTrue, Shape ifFalse, Shape expected) {}

        List<BroadcastCase> cases = List.of(
                new BroadcastCase(Shape.scalar(), Shape.scalar(), Shape.scalar(), Shape.scalar()),
                new BroadcastCase(Shape.of(1, 3), Shape.of(0, 1), Shape.of(1, 3), Shape.of(0, 3)),
                new BroadcastCase(
                        Shape.of(1, 3, 1),
                        Shape.of(2, 1, 4),
                        Shape.of(3, 4),
                        Shape.of(2, 3, 4)),
                new BroadcastCase(
                        Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(4)),
                        Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(1)),
                        Shape.of(1, 4),
                        Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(4))),
                new BroadcastCase(
                        Shape.of(1, 1),
                        Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(1)),
                        Shape.of(1, 5),
                        Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(5))));

        for (BroadcastCase broadcastCase : cases) {
            Tensor result = Tensor.where(
                    tensor(DataType.BOOL, broadcastCase.condition(), false),
                    tensor(DataType.FLOAT32, broadcastCase.ifTrue(), false),
                    tensor(DataType.FLOAT32, broadcastCase.ifFalse(), false));
            assertEquals(broadcastCase.expected(), result.descriptor().shape());
        }
    }

    @Test
    void propagatesGradientEligibilityFromBranchesOnly() {
        Tensor condition = tensor(DataType.BOOL, Shape.scalar(), false);

        for (boolean ifTrueGrad : List.of(false, true)) {
            for (boolean ifFalseGrad : List.of(false, true)) {
                Tensor result = Tensor.where(
                        condition,
                        tensor(DataType.FLOAT32, Shape.scalar(), ifTrueGrad),
                        tensor(DataType.FLOAT32, Shape.scalar(), ifFalseGrad));
                assertEquals(ifTrueGrad || ifFalseGrad, result.descriptor().requiresGrad());
            }
        }
    }

    @Test
    void acceptsDerivedBoolConditionsWithoutSpecialHandling() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2, 1), false);
        Tensor right = tensor(DataType.FLOAT32, Shape.of(1, 3), false);
        Tensor comparison = left.greaterThan(right);
        Tensor logical = comparison.logicalNot();
        TensorProvenance comparisonProvenance = comparison.provenance().orElseThrow();
        TensorProvenance logicalProvenance = logical.provenance().orElseThrow();
        Tensor ifTrue = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor ifFalse = tensor(DataType.FLOAT32, Shape.scalar(), false);

        Tensor fromComparison = Tensor.where(comparison, ifTrue, ifFalse);
        Tensor fromLogical = Tensor.where(logical, ifTrue, ifFalse);

        assertAll(
                () -> assertSame(
                        comparison,
                        fromComparison.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(
                        logical,
                        fromLogical.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(
                        comparisonProvenance, comparison.provenance().orElseThrow()),
                () -> assertSame(logicalProvenance, logical.provenance().orElseThrow()),
                () -> assertEquals(Shape.of(2, 3), fromComparison.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 3), fromLogical.descriptor().shape()));
    }

    @Test
    void preservesRepeatedBranchesAndAlwaysCreatesFreshIdentity() {
        Tensor condition = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor branch = tensor(DataType.FLOAT64, Shape.scalar(), true);

        Tensor first = Tensor.where(condition, branch, branch);
        Tensor second = Tensor.where(condition, branch, branch);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertSame(
                        branch, first.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(
                        branch, first.provenance().orElseThrow().inputs().get(2)));
    }

    @Test
    void validatesNullsConditionAndBranchesInExactOrder() throws ReflectiveOperationException {
        Tensor bool = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor floating = tensor(DataType.FLOAT32, Shape.scalar(), false);
        long before = nextTensorIdState().get();

        NullPointerException conditionNull = assertThrows(
                NullPointerException.class, () -> Tensor.where(null, null, null));
        NullPointerException trueNull = assertThrows(
                NullPointerException.class, () -> Tensor.where(bool, null, null));
        NullPointerException falseNull = assertThrows(
                NullPointerException.class, () -> Tensor.where(bool, floating, null));
        NullPointerException helperConditionNull = assertThrows(
                NullPointerException.class, () -> TensorWhereExpressions.apply(null, null, null));
        NullPointerException helperTrueNull = assertThrows(
                NullPointerException.class, () -> TensorWhereExpressions.apply(bool, null, null));
        NullPointerException helperFalseNull = assertThrows(
                NullPointerException.class,
                () -> TensorWhereExpressions.apply(bool, floating, null));

        assertAll(
                () -> assertEquals("condition", conditionNull.getMessage()),
                () -> assertEquals("ifTrue", trueNull.getMessage()),
                () -> assertEquals("ifFalse", falseNull.getMessage()),
                () -> assertEquals("condition", helperConditionNull.getMessage()),
                () -> assertEquals("ifTrue", helperTrueNull.getMessage()),
                () -> assertEquals("ifFalse", helperFalseNull.getMessage()));

        for (DataType invalid : List.of(
                DataType.BFLOAT16,
                DataType.FLOAT32,
                DataType.FLOAT64,
                DataType.INT32,
                DataType.INT64)) {
            IllegalArgumentException invalidCondition = assertThrows(
                    IllegalArgumentException.class,
                    () -> Tensor.where(
                            tensor(invalid, Shape.scalar(), invalid.isFloating()),
                            floating,
                            floating));
            assertEquals(
                    "condition must have BOOL data type, but was " + invalid,
                    invalidCondition.getMessage());
        }
        assertEquals(before, nextTensorIdState().get());
    }

    @Test
    void rejectsEveryNonFloatingBranchAtItsPromotionPosition()
            throws ReflectiveOperationException {
        Tensor condition = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor floating = tensor(DataType.FLOAT32, Shape.scalar(), false);
        long before = nextTensorIdState().get();

        for (DataType invalid : List.of(DataType.BOOL, DataType.INT32, DataType.INT64)) {
            IllegalArgumentException invalidTrue = assertThrows(
                    IllegalArgumentException.class,
                    () -> Tensor.where(
                            condition, tensor(invalid, Shape.scalar(), false), floating));
            IllegalArgumentException invalidFalse = assertThrows(
                    IllegalArgumentException.class,
                    () -> Tensor.where(
                            condition, floating, tensor(invalid, Shape.scalar(), false)));
            assertAll(
                    () -> assertEquals(
                            "left must be a floating data type, but was " + invalid,
                            invalidTrue.getMessage()),
                    () -> assertEquals(
                            "right must be a floating data type, but was " + invalid,
                            invalidFalse.getMessage()));
        }
        assertEquals(before, nextTensorIdState().get());
    }

    @Test
    void reportsBranchFailureBeforeConditionFailureAndRejectsUnprovableDynamics()
            throws ReflectiveOperationException {
        Tensor condition = tensor(DataType.BOOL, Shape.of(7), false);
        Tensor incompatibleTrue = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor incompatibleFalse = tensor(DataType.FLOAT32, Shape.of(3), false);
        long before = nextTensorIdState().get();

        IllegalArgumentException branchFailure = assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.where(condition, incompatibleTrue, incompatibleFalse));
        IllegalArgumentException conditionFailure = assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.where(
                        condition,
                        tensor(DataType.FLOAT32, Shape.of(2), false),
                        tensor(DataType.FLOAT32, Shape.of(1), false)));
        IllegalArgumentException differentSymbols = assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.where(
                        tensor(
                                DataType.BOOL,
                                Shape.ofDimensions(new DynamicDimension("M")),
                                false),
                        tensor(
                                DataType.FLOAT32,
                                Shape.ofDimensions(new DynamicDimension("N")),
                                false),
                        tensor(DataType.FLOAT32, Shape.of(1), false)));
        IllegalArgumentException symbolicStatic = assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.where(
                        tensor(DataType.BOOL, Shape.of(2), false),
                        tensor(
                                DataType.FLOAT32,
                                Shape.ofDimensions(new DynamicDimension("N")),
                                false),
                        tensor(DataType.FLOAT32, Shape.of(1), false)));

        assertAll(
                () -> assertTrue(branchFailure.getMessage().contains("Shape[2] and Shape[3]")),
                () -> assertTrue(conditionFailure.getMessage().contains("Shape[7] and Shape[2]")),
                () -> assertTrue(differentSymbols.getMessage().contains("Shape[M] and Shape[N]")),
                () -> assertTrue(symbolicStatic.getMessage().contains("Shape[2] and Shape[N]")),
                () -> assertEquals(before, nextTensorIdState().get()));
    }

    @Test
    void leavesEveryInputAndItsStorageMetadataUnchanged() {
        Shape shape = Shape.of(2);
        HostTensorStorage conditionStorage = storage(DataType.BOOL, 2);
        HostTensorStorage trueStorage = storage(DataType.FLOAT32, 2);
        HostTensorStorage falseStorage = storage(DataType.FLOAT32, 2);
        conditionStorage.segment().set(ValueLayout.JAVA_BYTE, 0, (byte) 1);
        trueStorage.segment().set(ValueLayout.JAVA_FLOAT, 0, 2.5f);
        falseStorage.segment().set(ValueLayout.JAVA_FLOAT, 0, -3.25f);
        Tensor condition = tensor(DataType.BOOL, shape, false, "condition", conditionStorage);
        Tensor ifTrue = tensor(DataType.FLOAT32, shape, true, "true", trueStorage);
        Tensor ifFalse = tensor(DataType.FLOAT32, shape, false, "false", falseStorage);
        TensorDescriptor conditionDescriptor = condition.descriptor();
        TensorDescriptor trueDescriptor = ifTrue.descriptor();
        TensorDescriptor falseDescriptor = ifFalse.descriptor();
        LayoutDescriptor conditionLayout = conditionDescriptor.layout().orElseThrow();
        LayoutDescriptor trueLayout = trueDescriptor.layout().orElseThrow();
        LayoutDescriptor falseLayout = falseDescriptor.layout().orElseThrow();

        Tensor result = Tensor.where(condition, ifTrue, ifFalse);

        assertAll(
                () -> assertSame(conditionDescriptor, condition.descriptor()),
                () -> assertSame(trueDescriptor, ifTrue.descriptor()),
                () -> assertSame(falseDescriptor, ifFalse.descriptor()),
                () -> assertEquals(Optional.of("condition"), condition.label()),
                () -> assertEquals(Optional.of("true"), ifTrue.label()),
                () -> assertEquals(Optional.of("false"), ifFalse.label()),
                () -> assertSame(conditionStorage, condition.hostStorage().orElseThrow()),
                () -> assertSame(trueStorage, ifTrue.hostStorage().orElseThrow()),
                () -> assertSame(falseStorage, ifFalse.hostStorage().orElseThrow()),
                () -> assertEquals(
                        (byte) 1,
                        conditionStorage.segment().get(ValueLayout.JAVA_BYTE, 0)),
                () -> assertEquals(
                        2.5f,
                        trueStorage.segment().get(ValueLayout.JAVA_FLOAT, 0)),
                () -> assertEquals(
                        -3.25f,
                        falseStorage.segment().get(ValueLayout.JAVA_FLOAT, 0)),
                () -> assertSame(
                        conditionLayout, condition.descriptor().layout().orElseThrow()),
                () -> assertSame(trueLayout, ifTrue.descriptor().layout().orElseThrow()),
                () -> assertSame(falseLayout, ifFalse.descriptor().layout().orElseThrow()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            boolean requiresGrad,
            String label,
            HostTensorStorage storage) {
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad),
                Optional.of(label),
                Optional.empty(),
                Optional.of(storage));
    }

    private static HostTensorStorage storage(DataType dataType, int elementCount) {
        return switch (dataType) {
            case BOOL -> new MemorySegmentStorage(
                    dataType, elementCount, MemorySegment.ofArray(new byte[elementCount]));
            case FLOAT32 -> new MemorySegmentStorage(
                    dataType, elementCount, MemorySegment.ofArray(new float[elementCount]));
            default -> throw new IllegalArgumentException(
                    "unsupported test storage type " + dataType);
        };
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
