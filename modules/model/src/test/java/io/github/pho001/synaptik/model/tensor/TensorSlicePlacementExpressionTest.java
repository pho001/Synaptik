package io.github.pho001.synaptik.model.tensor;

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
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
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

class TensorSlicePlacementExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(40_000_000);

    @Test
    void helperAndExactlyTwoPublicMethodsHaveTheRequiredSurface()
            throws ReflectiveOperationException {
        var constructors = TensorSlicePlacementExpressions.class.getDeclaredConstructors();
        Set<String> methodNames = Arrays.stream(
                        TensorSlicePlacementExpressions.class.getDeclaredMethods())
                .map(Method::getName)
                .collect(Collectors.toSet());
        Method update = TensorSlicePlacementExpressions.class.getDeclaredMethod(
                "update", Tensor.class, Tensor.class, long[].class, int[].class, long[].class);
        Method crop = TensorSlicePlacementExpressions.class.getDeclaredMethod(
                "cropToShape", Tensor.class, Shape.class, Shape.class);
        Method publicUpdate = Tensor.class.getDeclaredMethod(
                "sliceUpdate", Tensor.class, long[].class, int[].class, long[].class);
        Method publicCrop = Tensor.class.getDeclaredMethod(
                "cropToShape", Shape.class, Shape.class);
        long publicTensorMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .count();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        TensorSlicePlacementExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorSlicePlacementExpressions.class.getModifiers())),
                () -> assertFalse(TensorSlicePlacementExpressions.class.isRecord()),
                () -> assertEquals(0,
                        TensorSlicePlacementExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorSlicePlacementExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(
                        Set.of(
                                "update",
                                "cropToShape",
                                "normalizeUpdate",
                                "normalizeUpdateStart",
                                "expectedUpdateShape",
                                "validateStaticCropBounds",
                                "createUpdate",
                                "createCrop"),
                        methodNames),
                () -> assertTrue(Modifier.isStatic(update.getModifiers())),
                () -> assertFalse(Modifier.isPublic(update.getModifiers())),
                () -> assertFalse(Modifier.isProtected(update.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(update.getModifiers())),
                () -> assertSame(Tensor.class, update.getReturnType()),
                () -> assertTrue(Modifier.isStatic(crop.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(crop.getModifiers())),
                () -> assertSame(Tensor.class, crop.getReturnType()),
                () -> assertTrue(Modifier.isPublic(publicUpdate.getModifiers())),
                () -> assertFalse(Modifier.isStatic(publicUpdate.getModifiers())),
                () -> assertFalse(publicUpdate.isVarArgs()),
                () -> assertSame(Tensor.class, publicUpdate.getReturnType()),
                () -> assertTrue(Modifier.isPublic(publicCrop.getModifiers())),
                () -> assertFalse(Modifier.isStatic(publicCrop.getModifiers())),
                () -> assertFalse(publicCrop.isVarArgs()),
                () -> assertSame(Tensor.class, publicCrop.getReturnType()),
                () -> assertEquals(196, publicTensorMethods),
                () -> assertEquals(1, publicMethodsNamed("sliceUpdate")),
                () -> assertEquals(1, publicMethodsNamed("cropToShape")));
    }

    @Test
    void sliceUpdateAcceptsEveryTypeAndValidGradientCombination() {
        for (DataType dataType : DataType.values()) {
            for (boolean baseGrad : validGradientChoices(dataType)) {
                for (boolean updateGrad : validGradientChoices(dataType)) {
                    Shape baseShape = Shape.of(3, 6);
                    Shape updateShape = Shape.of(3, 2);
                    Tensor base = tensor(dataType, baseShape, trueLayout(baseShape), baseGrad);
                    Tensor update = tensor(dataType, updateShape, Optional.empty(), updateGrad);

                    Tensor result = base.sliceUpdate(
                            update, new long[] {1}, new int[] {1}, new long[] {2});

                    assertAll(
                            () -> assertSame(dataType, result.descriptor().dataType()),
                            () -> assertSame(baseShape, result.descriptor().shape()),
                            () -> assertEquals(
                                    baseGrad || updateGrad,
                                    result.descriptor().requiresGrad()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertTrue(result.label().isEmpty()),
                            () -> assertTrue(result.hostStorage().isEmpty()));
                }
            }
        }
    }

    @Test
    void sliceUpdateNormalizesSignedMultiAxisCoordinatesAndSnapshotsArrays() {
        Tensor base = tensor(DataType.FLOAT32, Shape.of(5, 6), Optional.empty(), true);
        Tensor update = tensor(DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), false);
        long[] starts = {-1, 1};
        int[] axes = {-2, -1};
        long[] steps = {-2, 2};

        Tensor result = base.sliceUpdate(update, starts, axes, steps);
        SliceAttrs attrs = (SliceAttrs) result.provenance().orElseThrow().operation().attrs();
        starts[0] = 0;
        axes[0] = 1;
        steps[0] = 1;

        assertAll(
                () -> assertEquals(List.of(4L, 1L), attrs.starts()),
                () -> assertEquals(List.of(3L, 2L), attrs.lengths()),
                () -> assertEquals(List.of(0, 1), attrs.axes()),
                () -> assertEquals(List.of(-2L, 2L), attrs.steps()),
                () -> assertArrayEquals(new long[] {0, 1}, starts),
                () -> assertArrayEquals(new int[] {1, -1}, axes),
                () -> assertArrayEquals(new long[] {1, 2}, steps));
    }

    @Test
    void sliceUpdateRecordsExactOperationProducerAndProvenance() {
        Shape baseShape = Shape.of(3, 6);
        Tensor base = tensor(DataType.FLOAT64, baseShape, Optional.empty(), false);
        Tensor update = tensor(DataType.FLOAT64, Shape.of(3, 2), Optional.empty(), true);

        Tensor result = base.sliceUpdate(
                update, new long[] {1}, new int[] {1}, new long[] {2});
        TensorProvenance provenance = result.provenance().orElseThrow();
        TensorProducer producer = provenance.producer();

        assertAll(
                () -> assertSame(SliceKind.SLICE_UPDATE, provenance.operation().kind()),
                () -> assertEquals(
                        OperationSignature.fixed(SliceAttrs.class, 2, 1),
                        provenance.operation().signature()),
                () -> assertEquals(2, producer.inputs().size()),
                () -> assertSame(base, producer.inputs().get(0)),
                () -> assertSame(update, producer.inputs().get(1)),
                () -> assertEquals(1, producer.outputCount()),
                () -> assertSame(result.descriptor(), producer.outputDescriptors().get(0)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
    }

    @Test
    void sliceUpdateCreatesExplicitFullReplacementAndCanonicalEmptyRegions() {
        Tensor base = tensor(DataType.INT64, Shape.of(3, 6), Optional.empty(), false);
        Tensor full = tensor(DataType.INT64, Shape.of(3, 6), Optional.empty(), false);
        Tensor empty = tensor(DataType.INT64, Shape.of(3, 0), Optional.empty(), false);

        Tensor fullResult = base.sliceUpdate(
                full, new long[0], new int[0], new long[0]);
        Tensor emptyResult = base.sliceUpdate(
                empty,
                new long[] {Long.MIN_VALUE},
                new int[] {1},
                new long[] {Long.MIN_VALUE});
        SliceAttrs fullAttrs = (SliceAttrs) fullResult.provenance().orElseThrow().operation().attrs();
        SliceAttrs emptyAttrs =
                (SliceAttrs) emptyResult.provenance().orElseThrow().operation().attrs();

        assertAll(
                () -> assertEquals(List.of(), fullAttrs.starts()),
                () -> assertSame(base.descriptor().shape(), fullResult.descriptor().shape()),
                () -> assertNotSame(base, fullResult),
                () -> assertEquals(List.of(0L), emptyAttrs.starts()),
                () -> assertEquals(List.of(0L), emptyAttrs.lengths()),
                () -> assertEquals(List.of(Long.MIN_VALUE), emptyAttrs.steps()),
                () -> assertNotSame(base, emptyResult));
    }

    @Test
    void sliceUpdateDefersOnlyDynamicUpperBoundsAndRequiresStaticSelectedUpdateExtents() {
        Dimension dynamic = new DynamicDimension("N");
        Shape baseShape = Shape.ofDimensions(dynamic, new StaticDimension(4));
        Tensor base = tensor(DataType.FLOAT32, baseShape, Optional.empty(), true);
        Tensor valid = tensor(DataType.FLOAT32, Shape.of(2, 4), Optional.empty(), false);
        Tensor dynamicSelected = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("U"), new StaticDimension(4)),
                Optional.empty(),
                false);

        Tensor deferred = base.sliceUpdate(
                valid, new long[] {10}, new int[] {0}, new long[] {1});
        IllegalArgumentException negative = assertThrows(
                IllegalArgumentException.class,
                () -> base.sliceUpdate(
                        valid, new long[] {-1}, new int[] {0}, new long[] {1}));
        IllegalArgumentException negativeLast = assertThrows(
                IllegalArgumentException.class,
                () -> base.sliceUpdate(
                        valid, new long[] {0}, new int[] {0}, new long[] {-1}));
        IllegalArgumentException selectedDynamic = assertThrows(
                IllegalArgumentException.class,
                () -> base.sliceUpdate(
                        dynamicSelected, new long[] {0}, new int[] {0}, new long[] {1}));

        assertAll(
                () -> assertSame(baseShape, deferred.descriptor().shape()),
                () -> assertEquals(
                        "slice update start -1 at index 0 cannot be negative for dynamic base axis 0",
                        negative.getMessage()),
                () -> assertEquals(
                        "slice update coordinates at index 0 do not fit base extent "
                                + "DynamicDimension[symbol=N]: "
                                + "start=0, length=2, step=-1",
                        negativeLast.getMessage()),
                () -> assertEquals(
                        "slice update axis 0 at index 0 must have a statically known update dimension",
                        selectedDynamic.getMessage()));
    }

    @Test
    void sliceUpdateRejectsExactTypeRankEntryCoordinateAndShapeFailuresInOrder()
            throws Exception {
        Tensor base = tensor(DataType.FLOAT32, Shape.of(3, 6), Optional.empty(), true);
        Tensor wrongType = tensor(DataType.FLOAT64, Shape.of(3, 2), Optional.empty(), true);
        Tensor wrongRank = tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), true);
        Tensor update = tensor(DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), true);
        Tensor wrongShape = tensor(DataType.FLOAT32, Shape.of(4, 2), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertFailure(NullPointerException.class, "base",
                () -> TensorSlicePlacementExpressions.update(null, null, null, null, null));
        assertFailure(NullPointerException.class, "update",
                () -> TensorSlicePlacementExpressions.update(base, null, null, null, null));
        assertFailure(NullPointerException.class, "starts",
                () -> TensorSlicePlacementExpressions.update(base, update, null, null, null));
        assertFailure(NullPointerException.class, "axes",
                () -> TensorSlicePlacementExpressions.update(base, update, new long[0], null, null));
        assertFailure(NullPointerException.class, "steps",
                () -> TensorSlicePlacementExpressions.update(
                        base, update, new long[0], new int[0], null));
        assertFailure(IllegalArgumentException.class,
                "starts, axes, and steps must have matching lengths",
                () -> base.sliceUpdate(update, new long[] {0}, new int[0], new long[0]));
        assertFailure(IllegalArgumentException.class,
                "slice update data types must match: base=FLOAT32, update=FLOAT64",
                () -> base.sliceUpdate(wrongType, new long[0], new int[0], new long[0]));
        assertFailure(IllegalArgumentException.class,
                "slice update rank must match base rank: base=2, update=1",
                () -> base.sliceUpdate(wrongRank, new long[0], new int[0], new long[0]));
        assertFailure(IllegalArgumentException.class,
                "slice update axis -3 at index 0 is outside rank 2",
                () -> base.sliceUpdate(update, new long[] {0}, new int[] {-3}, new long[] {1}));
        assertFailure(IllegalArgumentException.class,
                "slice update contains duplicate normalized axis 0 at index 1",
                () -> base.sliceUpdate(
                        tensor(DataType.FLOAT32, Shape.of(3, 6), Optional.empty(), true),
                        new long[] {0, 0}, new int[] {0, -2}, new long[] {1, 1}));
        assertFailure(IllegalArgumentException.class, "steps[0] must be non-zero: 0",
                () -> base.sliceUpdate(update, new long[] {0}, new int[] {1}, new long[] {0}));
        assertFailure(IllegalArgumentException.class,
                "slice update coordinates at index 0 do not fit base extent 6: "
                        + "start=5, length=2, step=1",
                () -> base.sliceUpdate(update, new long[] {5}, new int[] {1}, new long[] {1}));
        assertFailure(IllegalArgumentException.class,
                "slice update shape must match base Shape with selected axes replaced: "
                        + "expected=Shape[3, 2], actual=Shape[4, 2]",
                () -> base.sliceUpdate(wrongShape, new long[] {0}, new int[] {1}, new long[] {1}));

        assertEquals(before, next.get());
    }

    @Test
    void sliceUpdatePropagatesCheckedCoordinateOverflowBeforeIdentityAllocation()
            throws Exception {
        Tensor base = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        Tensor update = tensor(DataType.INT64, Shape.of(3), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                ArithmeticException.class,
                () -> base.sliceUpdate(
                        update,
                        new long[] {Long.MAX_VALUE - 1},
                        new int[] {0},
                        new long[] {Long.MAX_VALUE}));

        assertEquals(before, next.get());
    }

    @Test
    void cropRetainsExactStaticAndSymbolicShapesAndMetadata() {
        Dimension n = new DynamicDimension("N");
        Shape inputShape = Shape.ofDimensions(DimensionExpressions.addConstant(n, 3));
        Shape targetShape = Shape.ofDimensions(n);
        Shape prefixShape = Shape.of(1);
        Tensor input = tensor(DataType.FLOAT64, inputShape, Optional.empty(), true);

        Tensor result = input.cropToShape(targetShape, prefixShape);
        TensorProvenance provenance = result.provenance().orElseThrow();
        CropToShapeAttrs attrs = (CropToShapeAttrs) provenance.operation().attrs();

        assertAll(
                () -> assertSame(targetShape, result.descriptor().shape()),
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertSame(SliceKind.SLICE, provenance.operation().kind()),
                () -> assertEquals(
                        OperationSignature.fixed(CropToShapeAttrs.class, 1, 1),
                        provenance.operation().signature()),
                () -> assertSame(targetShape, attrs.targetShape()),
                () -> assertSame(prefixShape, attrs.prefixShape()),
                () -> assertEquals(List.of(input), provenance.inputs()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void cropAcceptsScalarZeroAndAllUnresolvedBoundCategories() {
        Tensor scalar = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        Tensor zero = tensor(DataType.INT32, Shape.of(3), Optional.empty(), false);
        Dimension n = new DynamicDimension("N");
        Tensor dynamicInput = tensor(
                DataType.FLOAT32, Shape.ofDimensions(n), Optional.empty(), true);
        Tensor staticInput = tensor(DataType.FLOAT32, Shape.of(4), Optional.empty(), true);

        Tensor scalarResult = scalar.cropToShape(Shape.scalar(), Shape.scalar());
        Tensor zeroResult = zero.cropToShape(Shape.of(0), Shape.of(3));
        Tensor dynamicSource = dynamicInput.cropToShape(Shape.of(9), Shape.of(8));
        Tensor dynamicTarget = staticInput.cropToShape(
                Shape.ofDimensions(new DynamicDimension("T")), Shape.of(8));
        Tensor dynamicPrefix = staticInput.cropToShape(
                Shape.of(9), Shape.ofDimensions(new DynamicDimension("P")));

        assertAll(
                () -> assertSame(Shape.scalar(), scalarResult.descriptor().shape()),
                () -> assertEquals(Shape.of(0), zeroResult.descriptor().shape()),
                () -> assertEquals(Shape.of(9), dynamicSource.descriptor().shape()),
                () -> assertEquals("Shape[T]", dynamicTarget.descriptor().shape().toString()),
                () -> assertEquals(Shape.of(9), dynamicPrefix.descriptor().shape()));
    }

    @Test
    void cropValidatesNullRankAndFirstStaticBoundInExactOrderWithoutIds()
            throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3, 6), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertFailure(NullPointerException.class, "input",
                () -> TensorSlicePlacementExpressions.cropToShape(null, null, null));
        assertFailure(NullPointerException.class, "targetShape",
                () -> TensorSlicePlacementExpressions.cropToShape(input, null, null));
        assertFailure(NullPointerException.class, "prefixShape",
                () -> TensorSlicePlacementExpressions.cropToShape(
                        input, Shape.of(1, 1), null));
        assertFailure(IllegalArgumentException.class,
                "crop target rank must match input rank: input=2, target=1",
                () -> input.cropToShape(Shape.of(1), Shape.of(1)));
        assertFailure(IllegalArgumentException.class,
                "crop prefix rank must match input rank: input=2, prefix=1",
                () -> input.cropToShape(Shape.of(1, 1), Shape.of(1)));
        assertFailure(IllegalArgumentException.class,
                "crop region exceeds input extent at axis 0: input=3, prefix=2, target=2",
                () -> input.cropToShape(Shape.of(2, 7), Shape.of(2, 0)));

        assertEquals(before, next.get());
    }

    @Test
    void cropPropagatesCheckedStaticBoundOverflowBeforeIdentityAllocation()
            throws Exception {
        Tensor input = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                ArithmeticException.class,
                () -> input.cropToShape(Shape.of(1), Shape.of(Long.MAX_VALUE)));

        assertEquals(before, next.get());
    }

    @Test
    void successfulCallsConsumeOneFreshIdEachAndLeaveInputsUnchanged() throws Exception {
        Shape baseShape = Shape.of(3, 6);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(baseShape);
        Tensor base = tensor(DataType.FLOAT32, baseShape, Optional.of(layout), true);
        Tensor update = tensor(DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), false);
        TensorDescriptor baseDescriptor = base.descriptor();
        TensorDescriptor updateDescriptor = update.descriptor();
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor updated = base.sliceUpdate(
                update, new long[] {1}, new int[] {1}, new long[] {2});
        Tensor cropped = base.cropToShape(Shape.of(2, 3), Shape.of(1, 2));

        assertAll(
                () -> assertEquals(before, updated.id().value()),
                () -> assertEquals(before + 1, cropped.id().value()),
                () -> assertEquals(before + 2, next.get()),
                () -> assertSame(baseDescriptor, base.descriptor()),
                () -> assertSame(updateDescriptor, update.descriptor()),
                () -> assertSame(layout, base.descriptor().layout().orElseThrow()),
                () -> assertTrue(base.provenance().isEmpty()),
                () -> assertTrue(update.provenance().isEmpty()),
                () -> assertNotEquals(updated.id(), cropped.id()),
                () -> assertNotSame(
                        updated.provenance().orElseThrow().producer(),
                        cropped.provenance().orElseThrow().producer()));
    }

    private static long publicMethodsNamed(String name) {
        return Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .filter(method -> method.getName().equals(name))
                .count();
    }

    private static Optional<LayoutDescriptor> trueLayout(Shape shape) {
        return Optional.of(LayoutDescriptor.contiguous(shape));
    }

    private static List<Boolean> validGradientChoices(DataType dataType) {
        return dataType.isDifferentiable() ? List.of(false, true) : List.of(false);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad) {
        return new Tensor(
                new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, layout, requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type,
            String message,
            org.junit.jupiter.api.function.Executable executable) {
        assertEquals(message, assertThrows(type, executable).getMessage());
    }
}
