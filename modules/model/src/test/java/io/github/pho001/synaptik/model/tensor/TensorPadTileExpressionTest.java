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
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
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
    void exposesTypedAndDoublePadMethodsAndFiveMethodStatelessHelper() throws Exception {
        Method pad = Tensor.class.getDeclaredMethod(
                "pad", long[].class, long[].class, double.class);
        Method typedPad = Tensor.class.getDeclaredMethod(
                "pad", long[].class, long[].class, ScalarValue.class);
        Method tile = Tensor.class.getDeclaredMethod("tile", long[].class);
        var constructor = TensorPadTileExpressions.class.getDeclaredConstructor();
        Method helperPad = TensorPadTileExpressions.class.getDeclaredMethod(
                "pad", Tensor.class, long[].class, long[].class, ScalarValue.class);
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
                () -> assertEquals(Tensor.class, typedPad.getReturnType()),
                () -> assertEquals(
                        List.of(long[].class, long[].class, ScalarValue.class),
                        Arrays.asList(typedPad.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(typedPad.getModifiers())),
                () -> assertFalse(Modifier.isStatic(typedPad.getModifiers())),
                () -> assertFalse(typedPad.isVarArgs()),
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

                    ScalarValue constant = scalar(dataType, -7.25);
                    Tensor padded = input.pad(new long[] {1, 2}, new long[] {3, 4}, constant);
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
                                    new PadAttrs(List.of(1L, 2L), List.of(3L, 4L), constant),
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
    void retainsExactTypedPaddingConstantsWithoutConversion() {
        Tensor input = tensor(DataType.INT64, Shape.of(1), Optional.empty(), false);
        long[] values = {Long.MIN_VALUE, -1L, 0L, 9_007_199_254_740_993L, Long.MAX_VALUE};

        for (long value : values) {
            ScalarValue constant = ScalarValue.int64(value);
            Tensor result = input.pad(new long[] {0}, new long[] {0}, constant);
            PadAttrs attrs = (PadAttrs) result.provenance().orElseThrow().operation().attrs();
            assertSame(constant, attrs.constantValue());
            assertEquals(value, attrs.constantValue().int64Value());
        }
    }

    @Test
    void clonesCallerArraysBeforeAttributesAndShapeDerivation() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);
        long[] before = {1, 2};
        long[] after = {3, 4};
        long[] repeats = {5, 6};

        Tensor padded = input.pad(before, after, ScalarValue.float32(9.0f));
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
                () -> TensorPadTileExpressions.pad(null, null, null, null));
        NullPointerException nullBefore = assertThrows(
                NullPointerException.class,
                () -> TensorPadTileExpressions.pad(rankTwo, null, null, null));
        NullPointerException nullAfter = assertThrows(
                NullPointerException.class,
                () -> TensorPadTileExpressions.pad(rankTwo, new long[] {0, 0}, null, null));
        NullPointerException nullConstant = assertThrows(
                NullPointerException.class,
                () -> TensorPadTileExpressions.pad(
                        rankTwo, new long[] {0, 0}, new long[] {0, 0}, null));
        IllegalArgumentException beforeRank = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(new long[] {0}, new long[0], ScalarValue.float32(0.0f)));
        IllegalArgumentException afterRank = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(
                        new long[] {0, 0}, new long[] {0}, ScalarValue.float32(0.0f)));
        IllegalArgumentException negativeBefore = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(new long[] {-1, 0}, new long[] {-2, 0},
                        ScalarValue.float32(0.0f)));
        IllegalArgumentException negativeAfter = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(new long[] {0, 0}, new long[] {-2, 0},
                        ScalarValue.float32(0.0f)));
        IllegalArgumentException typeMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(
                        new long[] {0, 0}, new long[] {0, 0}, ScalarValue.float64(0.0)));
        IllegalArgumentException doubleMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> rankTwo.pad(new long[] {0, 0}, new long[] {0, 0}, 0.0));
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
                () -> assertEquals("constantValue", nullConstant.getMessage()),
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
                () -> assertEquals(
                        "padding constant data type FLOAT64 must match input data type FLOAT32",
                        typeMismatch.getMessage()),
                () -> assertEquals(
                        "padding constant data type FLOAT64 must match input data type FLOAT32",
                        doubleMismatch.getMessage()),
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
    void supportsScalarStaticZeroAndCanonicalIdentityReferences() {
        Tensor scalarInput = tensor(DataType.INT64, Shape.scalar(), Optional.empty(), false);
        Tensor scalarPad = scalarInput.pad(
                new long[0], new long[0], ScalarValue.int64(Long.MIN_VALUE));
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
                .pad(new long[] {Long.MAX_VALUE}, new long[] {0}, ScalarValue.bool(false));

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
                () -> assertSame(zero,
                        identityPadded.descriptor().shape().dimensions().get(1)),
                () -> assertSame(batch,
                        identityTiled.descriptor().shape().dimensions().get(0)),
                () -> assertSame(zero,
                        identityTiled.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(
                        Shape.of(Long.MAX_VALUE), extremePadded.descriptor().shape()));
    }

    @Test
    void derivesCanonicalDynamicAndPreExistingExpressionExtents() {
        DynamicDimension first = new DynamicDimension("N");
        Dimension firstPlusFour = DimensionExpressions.addConstant(first, 4);
        Dimension unknown = DimensionExpressions.unknown(2, Optional.empty());
        Dimension divided = DimensionExpressions.ceilingDivide(first, 2);
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(first, firstPlusFour, unknown, divided),
                Optional.empty(),
                true);

        Tensor padded = input.pad(
                new long[] {2, 1, 3, 1}, new long[] {3, 4, 0, 2},
                ScalarValue.float32(0.0f));
        Tensor tiled = input.tile(4, 3, 2, 5);
        Tensor identityPadded = input.pad(
                new long[4], new long[4], ScalarValue.float32(0.0f));
        Tensor identityTiled = input.tile(1, 1, 1, 1);
        Shape independentlyPadded = Shape.ofDimensions(
                DimensionExpressions.addConstant(new DynamicDimension("N"), 5),
                DimensionExpressions.addConstant(new DynamicDimension("N"), 9),
                DimensionExpressions.addConstant(unknown, 3),
                DimensionExpressions.addConstant(divided, 3));
        Shape independentlyTiled = Shape.ofDimensions(
                DimensionExpressions.multiply(new DynamicDimension("N"), 4),
                DimensionExpressions.addConstant(
                        DimensionExpressions.multiply(new DynamicDimension("N"), 3), 12),
                DimensionExpressions.multiply(unknown, 2),
                DimensionExpressions.multiply(divided, 5));

        assertAll(
                () -> assertEquals(independentlyPadded, padded.descriptor().shape()),
                () -> assertEquals(independentlyTiled, tiled.descriptor().shape()),
                () -> assertSame(first,
                        identityPadded.descriptor().shape().dimensions().get(0)),
                () -> assertSame(firstPlusFour,
                        identityPadded.descriptor().shape().dimensions().get(1)),
                () -> assertSame(unknown,
                        identityPadded.descriptor().shape().dimensions().get(2)),
                () -> assertSame(divided,
                        identityPadded.descriptor().shape().dimensions().get(3)),
                () -> assertSame(first,
                        identityTiled.descriptor().shape().dimensions().get(0)),
                () -> assertSame(firstPlusFour,
                        identityTiled.descriptor().shape().dimensions().get(1)),
                () -> assertSame(unknown,
                        identityTiled.descriptor().shape().dimensions().get(2)),
                () -> assertSame(divided,
                        identityTiled.descriptor().shape().dimensions().get(3)),
                () -> assertTrue(padded.descriptor().layout().isEmpty()),
                () -> assertTrue(tiled.descriptor().layout().isEmpty()));
    }

    @Test
    void checkedShapeOverflowOccursBeforeIdentityAllocation() throws Exception {
        Tensor addFirst = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        Tensor addSecond = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE - 1), Optional.empty(), false);
        Tensor multiply = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        DynamicDimension dynamic = new DynamicDimension("N");
        Tensor offsetOverflow = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(DimensionExpressions.addConstant(dynamic, Long.MAX_VALUE)),
                Optional.empty(),
                false);
        Tensor coefficientOverflow = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(DimensionExpressions.multiply(dynamic, Long.MAX_VALUE)),
                Optional.empty(),
                false);
        AtomicLong next = nextTensorIdState();
        long beforeId = next.get();

        ArithmeticException firstAdd = assertThrows(
                ArithmeticException.class,
                () -> addFirst.pad(
                        new long[] {1}, new long[] {0}, ScalarValue.int64(0L)));
        ArithmeticException secondAdd = assertThrows(
                ArithmeticException.class,
                () -> addSecond.pad(
                        new long[] {1}, new long[] {1}, ScalarValue.int64(0L)));
        ArithmeticException product = assertThrows(
                ArithmeticException.class, () -> multiply.tile(2));
        ArithmeticException offset = assertThrows(
                ArithmeticException.class,
                () -> offsetOverflow.pad(
                        new long[] {1}, new long[] {0}, ScalarValue.float32(0.0f)));
        ArithmeticException coefficient = assertThrows(
                ArithmeticException.class, () -> coefficientOverflow.tile(2));

        assertAll(
                () -> assertEquals("long overflow", firstAdd.getMessage()),
                () -> assertEquals("long overflow", secondAdd.getMessage()),
                () -> assertEquals("long overflow", product.getMessage()),
                () -> assertEquals("long overflow", offset.getMessage()),
                () -> assertEquals("long overflow", coefficient.getMessage()),
                () -> assertEquals(beforeId, next.get()));
    }

    @Test
    void identityRepeatedAndNestedRequestsRemainExplicitAndFresh() {
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 3))),
                true);

        Tensor firstPad = input.pad(
                new long[] {0, 0}, new long[] {0, 0}, ScalarValue.float32(0.0f));
        Tensor secondPad = input.pad(
                new long[] {0, 0}, new long[] {0, 0}, ScalarValue.float32(0.0f));
        Tensor nestedPad = firstPad.pad(
                new long[] {0, 0}, new long[] {0, 0}, ScalarValue.float32(0.0f));
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
                new TensorProducer(
                        new Operation(TileKind.TILE, new TileAttrs(List.of(1L, 1L))),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));
        float[] before = storage.segment().asSlice(8, 16).toArray(JAVA_FLOAT);

        Tensor padded = input.pad(
                new long[] {1, 0}, new long[] {0, 1}, ScalarValue.float32(5.0f));
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

        input.pad(new long[] {1}, new long[] {2}, ScalarValue.float32(3.0f));
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
                    () -> input.pad(
                            new long[] {0}, new long[] {0}, ScalarValue.float32(0.0f)));
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

    private static ScalarValue scalar(DataType dataType, double value) {
        return switch (dataType) {
            case FLOAT64 -> ScalarValue.float64(value);
            case FLOAT32 -> ScalarValue.float32((float) value);
            case BFLOAT16 -> ScalarValue.bfloat16((float) value);
            case INT32 -> ScalarValue.int32((int) value);
            case INT64 -> ScalarValue.int64((long) value);
            case BOOL -> ScalarValue.bool(value != 0.0d);
        };
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
