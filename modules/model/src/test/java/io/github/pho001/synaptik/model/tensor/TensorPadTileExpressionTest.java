package io.github.pho001.synaptik.model.tensor;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
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
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorPadTileExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(90_000);

    @Test
    void exposesExactlyTwoPublicMethodsAndFiveMethodStatelessHelper() throws Exception {
        Method pad = Tensor.class.getDeclaredMethod(
                "pad", long[].class, long[].class, double.class);
        Method tile = Tensor.class.getDeclaredMethod("tile", long[].class);
        var constructor = TensorPadTileExpressions.class.getDeclaredConstructor();
        Method helperPad = TensorPadTileExpressions.class.getDeclaredMethod(
                "pad", Tensor.class, long[].class, long[].class, double.class);
        Method helperTile = TensorPadTileExpressions.class.getDeclaredMethod(
                "tile", Tensor.class, long[].class);
        Method paddedShape = TensorPadTileExpressions.class.getDeclaredMethod(
                "paddedShape", Shape.class, PadAttrs.class);
        Method tiledShape = TensorPadTileExpressions.class.getDeclaredMethod(
                "tiledShape", Shape.class, TileAttrs.class);
        Method create = TensorPadTileExpressions.class.getDeclaredMethod(
                "create", Tensor.class, TensorDescriptor.class, Shape.class, Operation.class);
        Set<String> methodNames = Arrays.stream(
                        TensorPadTileExpressions.class.getDeclaredMethods())
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertEquals(Tensor.class, pad.getReturnType()),
                () -> assertEquals(
                        List.of(long[].class, long[].class, double.class),
                        Arrays.asList(pad.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(pad.getModifiers())),
                () -> assertFalse(Modifier.isStatic(pad.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(pad.getModifiers())),
                () -> assertFalse(pad.isVarArgs()),
                () -> assertEquals(Tensor.class, tile.getReturnType()),
                () -> assertEquals(List.of(long[].class),
                        Arrays.asList(tile.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(tile.getModifiers())),
                () -> assertFalse(Modifier.isStatic(tile.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(tile.getModifiers())),
                () -> assertTrue(tile.isVarArgs()),
                () -> assertTrue(Modifier.isFinal(TensorPadTileExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorPadTileExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorPadTileExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorPadTileExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorPadTileExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(5,
                        TensorPadTileExpressions.class.getDeclaredMethods().length),
                () -> assertEquals(
                        Set.of("pad", "tile", "paddedShape", "tiledShape", "create"),
                        methodNames));
        for (Method method : List.of(helperPad, helperTile)) {
            assertAll(
                    () -> assertSame(Tensor.class, method.getReturnType()),
                    () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isProtected(method.getModifiers())),
                    () -> assertFalse(Modifier.isPrivate(method.getModifiers())));
        }
        for (Method method : List.of(paddedShape, tiledShape, create)) {
            assertAll(
                    () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                    () -> assertTrue(Modifier.isPrivate(method.getModifiers())));
        }
    }

    @Test
    void acceptsAllDataTypesAndEligibilityStatesWithExactOperationsAndUnresolvedLayout() {
        Shape shape = Shape.of(2, 3);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                for (Optional<LayoutDescriptor> layout : List.of(
                        Optional.<LayoutDescriptor>empty(),
                        Optional.of(LayoutDescriptor.contiguous(shape)),
                        Optional.of(LayoutDescriptor.of(
                                shape, new long[] {3, 1}, 2, true)),
                        Optional.of(LayoutDescriptor.of(
                                shape, new long[] {4, 1}, 0, true)),
                        Optional.of(LayoutDescriptor.of(
                                shape, new long[] {0, 1}, 0, true)))) {
                    Tensor input = tensor(dataType, shape, layout, requiresGrad);

                    Tensor padded = input.pad(new long[] {1, 2}, new long[] {3, 4}, -7.25);
                    Tensor tiled = input.tile(2, 5);
                    TensorProvenance padProvenance = padded.provenance().orElseThrow();
                    TensorProvenance tileProvenance = tiled.provenance().orElseThrow();

                    assertAll(
                            () -> assertSame(dataType, padded.descriptor().dataType()),
                            () -> assertEquals(Shape.of(6, 9), padded.descriptor().shape()),
                            () -> assertEquals(
                                    requiresGrad, padded.descriptor().requiresGrad()),
                            () -> assertTrue(padded.descriptor().layout().isEmpty()),
                            () -> assertTrue(padded.label().isEmpty()),
                            () -> assertTrue(padded.hostStorage().isEmpty()),
                            () -> assertSame(PadKind.PAD, padProvenance.operation().kind()),
                            () -> assertEquals(
                                    new PadAttrs(List.of(1L, 2L), List.of(3L, 4L), -7.25),
                                    padProvenance.operation().attrs()),
                            () -> assertEquals(List.of(input), padProvenance.inputs()),
                            () -> assertSame(input, padProvenance.inputs().getFirst()),
                            () -> assertSame(dataType, tiled.descriptor().dataType()),
                            () -> assertEquals(Shape.of(4, 15), tiled.descriptor().shape()),
                            () -> assertEquals(requiresGrad, tiled.descriptor().requiresGrad()),
                            () -> assertTrue(tiled.descriptor().layout().isEmpty()),
                            () -> assertTrue(tiled.label().isEmpty()),
                            () -> assertTrue(tiled.hostStorage().isEmpty()),
                            () -> assertSame(TileKind.TILE, tileProvenance.operation().kind()),
                            () -> assertEquals(
                                    new TileAttrs(List.of(2L, 5L)),
                                    tileProvenance.operation().attrs()),
                            () -> assertEquals(List.of(input), tileProvenance.inputs()),
                            () -> assertSame(input, tileProvenance.inputs().getFirst()));
                }
            }
        }
    }

    @Test
    void retainsEveryRawPaddingConstantWithoutConversion() {
        Tensor input = tensor(DataType.BOOL, Shape.of(1), Optional.empty(), false);
        double[] values = {
                0.0,
                -0.0,
                1.25,
                Double.NaN,
                Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                Double.longBitsToDouble(0x7ff8_0000_0000_1234L)
        };

        for (double value : values) {
            Tensor result = input.pad(new long[] {0}, new long[] {0}, value);
            PadAttrs attrs = (PadAttrs) result.provenance().orElseThrow().operation().attrs();
            assertEquals(
                    Double.doubleToRawLongBits(value),
                    Double.doubleToRawLongBits(attrs.constantValue()));
        }
    }

    @Test
    void clonesCallerArraysBeforeAttributesAndShapeDerivation() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);
        long[] before = {1, 2};
        long[] after = {3, 4};
        long[] repeats = {5, 6};

        Tensor padded = input.pad(before, after, 9.0);
        Tensor tiled = input.tile(repeats);
        before[0] = Long.MAX_VALUE;
        after[1] = Long.MAX_VALUE;
        repeats[0] = Long.MAX_VALUE;
        PadAttrs padAttrs = (PadAttrs) padded.provenance().orElseThrow().operation().attrs();
        TileAttrs tileAttrs = (TileAttrs) tiled.provenance().orElseThrow().operation().attrs();

        assertAll(
                () -> assertEquals(List.of(1L, 2L), padAttrs.before()),
                () -> assertEquals(List.of(3L, 4L), padAttrs.after()),
                () -> assertEquals(Shape.of(6, 9), padded.descriptor().shape()),
                () -> assertEquals(List.of(5L, 6L), tileAttrs.repeats()),
                () -> assertEquals(Shape.of(10, 18), tiled.descriptor().shape()));
    }

    @Test
    void validatesNullRankAndAttributeRequestsInExactOrderWithoutConsumingIdentity()
            throws Exception {
        Tensor rankTwo = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long beforeId = next.get();

        NullPointerException nullPadInput = assertThrows(
                NullPointerException.class,
                () -> TensorPadTileExpressions.pad(null, null, null, 0.0));
        NullPointerException nullBefore = assertThrows(
                NullPointerException.class,
                () -> TensorPadTileExpressions.pad(rankTwo, null, null, 0.0));
        NullPointerException nullAfter = assertThrows(
                NullPointerException.class,
                () -> TensorPadTileExpressions.pad(rankTwo, new long[] {0, 0}, null, 0.0));
        IllegalArgumentException beforeRank = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(new long[] {0}, new long[0], 0.0));
        IllegalArgumentException afterRank = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(new long[] {0, 0}, new long[] {0}, 0.0));
        IllegalArgumentException negativeBefore = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(new long[] {-1, 0}, new long[] {-2, 0}, 0.0));
        IllegalArgumentException negativeAfter = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(new long[] {0, 0}, new long[] {-2, 0}, 0.0));
        NullPointerException nullTileInput = assertThrows(
                NullPointerException.class,
                () -> TensorPadTileExpressions.tile(null, null));
        NullPointerException nullRepeats = assertThrows(
                NullPointerException.class,
                () -> TensorPadTileExpressions.tile(rankTwo, null));
        IllegalArgumentException repeatRank = assertThrows(
                IllegalArgumentException.class, () -> rankTwo.tile(1));
        IllegalArgumentException zeroRepeat = assertThrows(
                IllegalArgumentException.class, () -> rankTwo.tile(0, -1));
        IllegalArgumentException negativeRepeat = assertThrows(
                IllegalArgumentException.class, () -> rankTwo.tile(1, -1));

        assertAll(
                () -> assertEquals("input", nullPadInput.getMessage()),
                () -> assertEquals("before", nullBefore.getMessage()),
                () -> assertEquals("after", nullAfter.getMessage()),
                () -> assertEquals(
                        "padding before length 1 must equal input rank 2",
                        beforeRank.getMessage()),
                () -> assertEquals(
                        "padding after length 1 must equal input rank 2",
                        afterRank.getMessage()),
                () -> assertEquals(
                        "before[0] must be non-negative: -1", negativeBefore.getMessage()),
                () -> assertEquals(
                        "after[0] must be non-negative: -2", negativeAfter.getMessage()),
                () -> assertEquals("input", nullTileInput.getMessage()),
                () -> assertEquals("repeats", nullRepeats.getMessage()),
                () -> assertEquals(
                        "tile repeats length 1 must equal input rank 2",
                        repeatRank.getMessage()),
                () -> assertEquals(
                        "repeats[0] must be positive: 0", zeroRepeat.getMessage()),
                () -> assertEquals(
                        "repeats[1] must be positive: -1", negativeRepeat.getMessage()),
                () -> assertEquals(beforeId, next.get()));
    }

    @Test
    void supportsScalarStaticZeroAndIdentityDynamicShapes() {
        Tensor scalarInput = tensor(DataType.INT64, Shape.scalar(), Optional.empty(), false);
        Tensor scalarPad = scalarInput.pad(new long[0], new long[0], Double.NaN);
        Tensor scalarTile = scalarInput.tile(new long[0]);
        StaticDimension zero = new StaticDimension(0);
        DynamicDimension batch = new DynamicDimension("batch");
        Shape mixed = Shape.ofDimensions(batch, zero, new StaticDimension(3));
        Tensor mixedInput = tensor(DataType.FLOAT64, mixed, Optional.empty(), true);

        Tensor padded = mixedInput.pad(
                new long[] {0, 2, 1}, new long[] {0, 3, 4}, -0.0);
        Tensor tiled = mixedInput.tile(1, Long.MAX_VALUE, 2);
        Tensor identityPadded = mixedInput.pad(
                new long[] {0, 0, 0}, new long[] {0, 0, 0}, 0.0);
        Tensor identityTiled = mixedInput.tile(1, 1, 1);
        Tensor extremePadded = tensor(
                        DataType.BOOL, Shape.of(0), Optional.empty(), false)
                .pad(new long[] {Long.MAX_VALUE}, new long[] {0}, 0.0);

        assertAll(
                () -> assertSame(Shape.scalar(), scalarPad.descriptor().shape()),
                () -> assertSame(Shape.scalar(), scalarTile.descriptor().shape()),
                () -> assertSame(batch, padded.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(new StaticDimension(5),
                        padded.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(new StaticDimension(8),
                        padded.descriptor().shape().dimensions().get(2)),
                () -> assertNotSame(zero, padded.descriptor().shape().dimensions().get(1)),
                () -> assertSame(batch, tiled.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(new StaticDimension(0),
                        tiled.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(new StaticDimension(6),
                        tiled.descriptor().shape().dimensions().get(2)),
                () -> assertNotSame(zero, tiled.descriptor().shape().dimensions().get(1)),
                () -> assertTrue(padded.descriptor().layout().isEmpty()),
                () -> assertTrue(tiled.descriptor().layout().isEmpty()),
                () -> assertSame(batch,
                        identityPadded.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(zero,
                        identityPadded.descriptor().shape().dimensions().get(1)),
                () -> assertNotSame(zero,
                        identityPadded.descriptor().shape().dimensions().get(1)),
                () -> assertSame(batch,
                        identityTiled.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(zero,
                        identityTiled.descriptor().shape().dimensions().get(1)),
                () -> assertNotSame(zero,
                        identityTiled.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(
                        Shape.of(Long.MAX_VALUE), extremePadded.descriptor().shape()));
    }

    @Test
    void rejectsNonIdentityDynamicTransformsWithExactMessagesAndNoIdentityConsumption()
            throws Exception {
        DynamicDimension first = new DynamicDimension("first");
        DynamicDimension second = new DynamicDimension("second");
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(first, second),
                Optional.empty(),
                true);
        AtomicLong next = nextTensorIdState();
        long beforeId = next.get();

        IllegalArgumentException pad = assertThrows(
                IllegalArgumentException.class,
                () -> input.pad(new long[] {0, 2}, new long[] {0, 3}, 0.0));
        IllegalArgumentException tile = assertThrows(
                IllegalArgumentException.class, () -> input.tile(1, 4));

        assertAll(
                () -> assertEquals(
                        "cannot pad dynamic axis 1 with before=2 and after=3",
                        pad.getMessage()),
                () -> assertEquals(
                        "cannot tile dynamic axis 1 with repeat=4", tile.getMessage()),
                () -> assertEquals(beforeId, next.get()));
    }

    @Test
    void checkedShapeOverflowOccursBeforeIdentityAllocation() throws Exception {
        Tensor addFirst = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        Tensor addSecond = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE - 1), Optional.empty(), false);
        Tensor multiply = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long beforeId = next.get();

        ArithmeticException firstAdd = assertThrows(
                ArithmeticException.class,
                () -> addFirst.pad(new long[] {1}, new long[] {0}, 0.0));
        ArithmeticException secondAdd = assertThrows(
                ArithmeticException.class,
                () -> addSecond.pad(new long[] {1}, new long[] {1}, 0.0));
        ArithmeticException product = assertThrows(
                ArithmeticException.class, () -> multiply.tile(2));

        assertAll(
                () -> assertEquals("long overflow", firstAdd.getMessage()),
                () -> assertEquals("long overflow", secondAdd.getMessage()),
                () -> assertEquals("long overflow", product.getMessage()),
                () -> assertEquals(beforeId, next.get()));
    }

    @Test
    void identityRepeatedAndNestedRequestsRemainExplicitAndFresh() {
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 3))),
                true);

        Tensor firstPad = input.pad(new long[] {0, 0}, new long[] {0, 0}, 0.0);
        Tensor secondPad = input.pad(new long[] {0, 0}, new long[] {0, 0}, 0.0);
        Tensor nestedPad = firstPad.pad(new long[] {0, 0}, new long[] {0, 0}, 0.0);
        Tensor firstTile = input.tile(1, 1);
        Tensor secondTile = input.tile(1, 1);
        Tensor nestedTile = firstTile.tile(1, 1);

        assertAll(
                () -> assertNotSame(input, firstPad),
                () -> assertNotSame(firstPad, secondPad),
                () -> assertNotSame(firstPad, nestedPad),
                () -> assertNotEquals(input.id(), firstPad.id()),
                () -> assertNotEquals(firstPad.id(), secondPad.id()),
                () -> assertSame(firstPad,
                        nestedPad.provenance().orElseThrow().inputs().getFirst()),
                () -> assertTrue(firstPad.descriptor().layout().isEmpty()),
                () -> assertNotSame(input, firstTile),
                () -> assertNotSame(firstTile, secondTile),
                () -> assertNotSame(firstTile, nestedTile),
                () -> assertNotEquals(input.id(), firstTile.id()),
                () -> assertNotEquals(firstTile.id(), secondTile.id()),
                () -> assertSame(firstTile,
                        nestedTile.provenance().orElseThrow().inputs().getFirst()),
                () -> assertTrue(firstTile.descriptor().layout().isEmpty()));
    }

    @Test
    void leavesInputMetadataLiveOrDeadStorageAndValuesUntouched() {
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                shape, new long[] {2, 1}, 2, true);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true);
        Arena arena = Arena.ofConfined();
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 6, arena.allocate(24, 1));
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        storage.segment().asSlice(8, 16).copyFrom(MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.FLOAT32, shape, Optional.empty(), false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new Operation(TileKind.TILE, new TileAttrs(List.of(1L, 1L))),
                List.of(leaf));
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));
        float[] before = storage.segment().asSlice(8, 16).toArray(JAVA_FLOAT);

        Tensor padded = input.pad(new long[] {1, 0}, new long[] {0, 1}, 5.0);
        Tensor tiled = input.tile(2, 3);
        float[] after = storage.segment().asSlice(8, 16).toArray(JAVA_FLOAT);
        arena.close();

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(inputLayout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(originalProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertFalse(storage.isAlive()),
                () -> assertArrayEquals(before, after),
                () -> assertTrue(padded.label().isEmpty()),
                () -> assertTrue(padded.hostStorage().isEmpty()),
                () -> assertTrue(tiled.label().isEmpty()),
                () -> assertTrue(tiled.hostStorage().isEmpty()));
    }

    @Test
    void successfulCallsConsumeOneIdentityAndExhaustionOccursOnlyAtFinalDelegation()
            throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long beforeId = next.get();

        input.pad(new long[] {1}, new long[] {2}, 3.0);
        assertEquals(beforeId + 1, next.get());
        input.tile(4);
        assertEquals(beforeId + 2, next.get());

        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException padFailure = assertThrows(
                    IllegalStateException.class,
                    () -> input.pad(new long[] {0}, new long[] {0}, 0.0));
            IllegalStateException tileFailure = assertThrows(
                    IllegalStateException.class, () -> input.tile(1));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", padFailure.getMessage()),
                    () -> assertEquals(
                            "tensor identifier space exhausted", tileFailure.getMessage()),
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

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, layout, requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }
}
