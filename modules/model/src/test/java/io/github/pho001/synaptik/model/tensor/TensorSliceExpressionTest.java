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
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
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

class TensorSliceExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(80_000);

    @Test
    void exposesExactlyFivePublicMethodsAndElevenMethodStatelessHelper() throws Exception {
        Method slice = Tensor.class.getDeclaredMethod(
                "slice", long[].class, long[].class, int[].class, long[].class);
        Method sliceByLength = Tensor.class.getDeclaredMethod(
                "sliceByLength", long[].class, long[].class, int[].class, long[].class);
        Method sliceAxis = Tensor.class.getDeclaredMethod(
                "sliceAxis", int.class, long.class, long.class);
        Method steppedSliceAxis = Tensor.class.getDeclaredMethod(
                "sliceAxis", int.class, long.class, long.class, long.class);
        Method flip = Tensor.class.getDeclaredMethod("flip", int[].class);
        var constructor = TensorSliceExpressions.class.getDeclaredConstructor();
        List<Method> methods = Arrays.asList(TensorSliceExpressions.class.getDeclaredMethods());
        Method apply = TensorSliceExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, long[].class, long[].class, int[].class, long[].class);
        Method applyByLength = TensorSliceExpressions.class.getDeclaredMethod(
                "applyByLength",
                Tensor.class, long[].class, long[].class, int[].class, long[].class);
        Method applyAxis = TensorSliceExpressions.class.getDeclaredMethod(
                "applyAxis", Tensor.class, int.class, long.class, long.class, long.class);
        Method flipHelper = TensorSliceExpressions.class.getDeclaredMethod(
                "flip", Tensor.class, int[].class);
        Method normalize = TensorSliceExpressions.class.getDeclaredMethod(
                "normalize", Shape.class, long[].class, long[].class, int[].class, long[].class);
        Method normalizeByLength = TensorSliceExpressions.class.getDeclaredMethod(
                "normalizeByLength",
                Shape.class, long[].class, long[].class, int[].class, long[].class);
        Method normalizeBound = TensorSliceExpressions.class.getDeclaredMethod(
                "normalizeBound", long.class, long.class, long.class, boolean.class);
        Method sliceLength = TensorSliceExpressions.class.getDeclaredMethod(
                "sliceLength", long.class, long.class, long.class);
        Method deriveShape = TensorSliceExpressions.class.getDeclaredMethod(
                "deriveShape", Shape.class, SliceAttrs.class);
        Method resolve = TensorSliceExpressions.class.getDeclaredMethod(
                "resolveViewLayout", TensorDescriptor.class, Shape.class, SliceAttrs.class);
        Method create = TensorSliceExpressions.class.getDeclaredMethod(
                "create", Tensor.class, TensorDescriptor.class, Shape.class, Optional.class,
                SliceAttrs.class);

        assertAll(
                () -> assertSame(Tensor.class, slice.getReturnType()),
                () -> assertEquals(
                        List.of(long[].class, long[].class, int[].class, long[].class),
                        Arrays.asList(slice.getParameterTypes())),
                () -> assertFalse(slice.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(slice.getModifiers())),
                () -> assertFalse(Modifier.isStatic(slice.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(slice.getModifiers())),
                () -> assertSame(Tensor.class, sliceByLength.getReturnType()),
                () -> assertEquals(
                        List.of(long[].class, long[].class, int[].class, long[].class),
                        Arrays.asList(sliceByLength.getParameterTypes())),
                () -> assertFalse(sliceByLength.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(sliceByLength.getModifiers())),
                () -> assertFalse(Modifier.isStatic(sliceByLength.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(sliceByLength.getModifiers())),
                () -> assertSame(Tensor.class, sliceAxis.getReturnType()),
                () -> assertEquals(List.of(int.class, long.class, long.class),
                        Arrays.asList(sliceAxis.getParameterTypes())),
                () -> assertFalse(sliceAxis.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(sliceAxis.getModifiers())),
                () -> assertFalse(Modifier.isStatic(sliceAxis.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(sliceAxis.getModifiers())),
                () -> assertSame(Tensor.class, steppedSliceAxis.getReturnType()),
                () -> assertEquals(List.of(int.class, long.class, long.class, long.class),
                        Arrays.asList(steppedSliceAxis.getParameterTypes())),
                () -> assertFalse(steppedSliceAxis.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(steppedSliceAxis.getModifiers())),
                () -> assertSame(Tensor.class, flip.getReturnType()),
                () -> assertEquals(List.of(int[].class), Arrays.asList(flip.getParameterTypes())),
                () -> assertTrue(flip.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(flip.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorSliceExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorSliceExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorSliceExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorSliceExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorSliceExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(11, methods.size()),
                () -> assertEquals(
                        Set.of("apply", "applyByLength", "applyAxis", "flip", "normalize",
                                "normalizeByLength", "normalizeBound", "sliceLength",
                                "deriveShape", "resolveViewLayout", "create"),
                        methods.stream().map(Method::getName).collect(Collectors.toSet())),
                () -> assertEquals(4, methods.stream()
                        .filter(method -> !Modifier.isPrivate(method.getModifiers())).count()),
                () -> assertTrue(methods.stream().allMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertSame(Tensor.class, apply.getReturnType()),
                () -> assertSame(Tensor.class, applyByLength.getReturnType()),
                () -> assertSame(Tensor.class, applyAxis.getReturnType()),
                () -> assertSame(Tensor.class, flipHelper.getReturnType()),
                () -> assertSame(SliceAttrs.class, normalize.getReturnType()),
                () -> assertSame(SliceAttrs.class, normalizeByLength.getReturnType()),
                () -> assertSame(long.class, normalizeBound.getReturnType()),
                () -> assertSame(long.class, sliceLength.getReturnType()),
                () -> assertSame(Shape.class, deriveShape.getReturnType()),
                () -> assertSame(Optional.class, resolve.getReturnType()),
                () -> assertSame(Tensor.class, create.getReturnType()),
                () -> assertTrue(Modifier.isPrivate(normalize.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(normalizeByLength.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(normalizeBound.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(sliceLength.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(deriveShape.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(resolve.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())),
                () -> assertEquals(1, publicMethodsNamed("sliceByLength")),
                () -> assertEquals(1, publicMethodsNamed("slice")),
                () -> assertEquals(2, publicMethodsNamed("sliceAxis")),
                () -> assertEquals(1, publicMethodsNamed("flip")),
                () -> assertEquals(0, publicMethodsNamed("dynamicSlice")),
                () -> assertEquals(0, publicMethodsNamed("sliceLengths")));
    }

    @Test
    void sliceByLengthPreservesEveryDataTypeEligibilityAndExactMetadata() {
        Shape shape = Shape.of(4, 8);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Tensor input = tensor(dataType, shape, Optional.empty(), requiresGrad);

                Tensor result = input.sliceByLength(
                        new long[] {1, 6},
                        new long[] {2, 3},
                        new int[] {0, -1},
                        new long[] {2, -2});
                TensorProvenance provenance = result.provenance().orElseThrow();
                SliceAttrs attrs = (SliceAttrs) provenance.operation().attrs();

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertEquals(Shape.of(2, 3), result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertSame(SliceKind.SLICE, provenance.operation().kind()),
                        () -> assertEquals(List.of(1L, 6L), attrs.starts()),
                        () -> assertEquals(List.of(2L, 3L), attrs.lengths()),
                        () -> assertEquals(List.of(0, 1), attrs.axes()),
                        () -> assertEquals(List.of(2L, -2L), attrs.steps()),
                        () -> assertEquals(List.of(input), provenance.inputs()),
                        () -> assertSame(input, provenance.inputs().getFirst()),
                        () -> assertEquals(0, provenance.outputIndex()),
                        () -> assertEquals(1, provenance.producer().outputCount()),
                        () -> assertSame(result, provenance.producer().output(0)),
                        () -> assertSame(result.descriptor(), provenance.outputDescriptor()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertNotSame(input, result),
                        () -> assertNotEquals(input.id(), result.id()));
            }
        }
    }

    @Test
    void sliceByLengthSnapshotsArraysAndPreservesOnlyUnaffectedDimensionReferences() {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension selected = new DynamicDimension("selected");
        StaticDimension width = new StaticDimension(8);
        Shape shape = Shape.ofDimensions(batch, selected, width);
        Tensor input = tensor(DataType.FLOAT32, shape, Optional.empty(), true);
        long[] starts = {2, 6};
        long[] lengths = {3, 2};
        int[] axes = {1, -1};
        long[] steps = {2, -3};

        Tensor result = input.sliceByLength(starts, lengths, axes, steps);
        starts[0] = -1;
        lengths[0] = 0;
        axes[0] = 0;
        steps[0] = 0;
        SliceAttrs attrs = (SliceAttrs) result.provenance().orElseThrow().operation().attrs();
        Shape resultShape = result.descriptor().shape();

        assertAll(
                () -> assertEquals(List.of(2L, 6L), attrs.starts()),
                () -> assertEquals(List.of(3L, 2L), attrs.lengths()),
                () -> assertEquals(List.of(1, 2), attrs.axes()),
                () -> assertEquals(List.of(2L, -3L), attrs.steps()),
                () -> assertSame(batch, resultShape.dimension(0)),
                () -> assertEquals(new StaticDimension(3), resultShape.dimension(1)),
                () -> assertEquals(new StaticDimension(2), resultShape.dimension(2)),
                () -> assertNotSame(selected, resultShape.dimension(1)),
                () -> assertNotSame(width, resultShape.dimension(2)));
    }

    @Test
    void sliceByLengthAcceptsEveryUnresolvedSelectedCategoryAndCanonicalEmptyState() {
        DynamicDimension named = new DynamicDimension("N");
        var expression = DimensionExpressions.addConstant(new DynamicDimension("M"), 2);
        var unknown = DimensionExpressions.unknown(0, Optional.empty());
        Tensor namedInput = tensor(
                DataType.FLOAT32, Shape.ofDimensions(named), Optional.empty(), true);
        Tensor expressionInput = tensor(
                DataType.FLOAT32, Shape.ofDimensions(expression), Optional.empty(), true);
        Tensor unknownInput = tensor(
                DataType.FLOAT32, Shape.ofDimensions(unknown), Optional.empty(), true);
        Tensor zeroInput = tensor(DataType.INT32, Shape.of(0), Optional.empty(), false);

        Tensor namedResult = namedInput.sliceByLength(
                new long[] {4}, new long[] {3}, new int[] {0}, new long[] {-2});
        Tensor expressionResult = expressionInput.sliceByLength(
                new long[] {2}, new long[] {4}, new int[] {0}, new long[] {3});
        Tensor unknownResult = unknownInput.sliceByLength(
                new long[] {0}, new long[] {1}, new int[] {0}, new long[] {Long.MIN_VALUE});
        Tensor emptyResult = zeroInput.sliceByLength(
                new long[] {Long.MAX_VALUE},
                new long[] {0},
                new int[] {0},
                new long[] {Long.MIN_VALUE});
        SliceAttrs emptyAttrs =
                (SliceAttrs) emptyResult.provenance().orElseThrow().operation().attrs();

        assertAll(
                () -> assertEquals(Shape.of(3), namedResult.descriptor().shape()),
                () -> assertEquals(Shape.of(4), expressionResult.descriptor().shape()),
                () -> assertEquals(Shape.of(1), unknownResult.descriptor().shape()),
                () -> assertEquals(Shape.of(0), emptyResult.descriptor().shape()),
                () -> assertEquals(List.of(0L), emptyAttrs.starts()),
                () -> assertEquals(List.of(0L), emptyAttrs.lengths()),
                () -> assertEquals(List.of(Long.MIN_VALUE), emptyAttrs.steps()),
                () -> assertTrue(emptyResult.descriptor().layout().isEmpty()));
    }

    @Test
    void sliceByLengthValidatesExactPrecedenceMessagesAndConsumesNoIds() throws Exception {
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(3), new DynamicDimension("N")),
                Optional.empty(),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertFailure(NullPointerException.class, "input",
                () -> TensorSliceExpressions.applyByLength(null, null, null, null, null));
        assertFailure(NullPointerException.class, "starts",
                () -> TensorSliceExpressions.applyByLength(input, null, null, null, null));
        assertFailure(NullPointerException.class, "lengths",
                () -> TensorSliceExpressions.applyByLength(
                        input, new long[0], null, null, null));
        assertFailure(NullPointerException.class, "axes",
                () -> TensorSliceExpressions.applyByLength(
                        input, new long[0], new long[0], null, null));
        assertFailure(NullPointerException.class, "steps",
                () -> TensorSliceExpressions.applyByLength(
                        input, new long[0], new long[0], new int[0], null));
        assertFailure(IllegalArgumentException.class,
                "starts, lengths, axes, and steps must have matching lengths",
                () -> input.sliceByLength(
                        new long[] {0}, new long[0], new int[] {0}, new long[] {1}));
        assertFailure(IllegalArgumentException.class,
                "slice by length axis -3 at index 0 is outside rank 2",
                () -> input.sliceByLength(
                        new long[] {-1}, new long[] {-1}, new int[] {-3}, new long[] {0}));
        assertFailure(IllegalArgumentException.class,
                "slice by length axis -2147483648 at index 0 is outside rank 2",
                () -> input.sliceByLength(
                        new long[] {0}, new long[] {1},
                        new int[] {Integer.MIN_VALUE}, new long[] {1}));
        assertFailure(IllegalArgumentException.class,
                "slice by length contains duplicate normalized axis 0 at index 1",
                () -> input.sliceByLength(
                        new long[] {0, -1},
                        new long[] {1, -1},
                        new int[] {0, -2},
                        new long[] {1, 0}));
        assertFailure(IllegalArgumentException.class, "starts[0] must be non-negative: -1",
                () -> input.sliceByLength(
                        new long[] {-1}, new long[] {-1}, new int[] {0}, new long[] {0}));
        assertFailure(IllegalArgumentException.class, "lengths[0] must be non-negative: -1",
                () -> input.sliceByLength(
                        new long[] {0}, new long[] {-1}, new int[] {0}, new long[] {0}));
        assertFailure(IllegalArgumentException.class, "steps[0] must be non-zero: 0",
                () -> input.sliceByLength(
                        new long[] {0}, new long[] {1}, new int[] {0}, new long[] {0}));
        assertFailure(IllegalArgumentException.class,
                "slice by length coordinates at index 0 do not fit input extent 3: "
                        + "start=3, length=1, step=1",
                () -> input.sliceByLength(
                        new long[] {3}, new long[] {1}, new int[] {0}, new long[] {1}));
        assertFailure(IllegalArgumentException.class,
                "slice by length coordinates at index 0 do not fit input extent 3: "
                        + "start=1, length=2, step=2",
                () -> input.sliceByLength(
                        new long[] {1}, new long[] {2}, new int[] {0}, new long[] {2}));
        assertFailure(IllegalArgumentException.class,
                "slice by length coordinates at index 0 do not fit input extent "
                        + "DynamicDimension[symbol=N]: start=1, length=2, step=-2",
                () -> input.sliceByLength(
                        new long[] {1}, new long[] {2}, new int[] {1}, new long[] {-2}));

        assertEquals(before, next.get());
    }

    @Test
    void sliceByLengthPropagatesCoordinateAndLayoutOverflowBeforeIdentityAllocation()
            throws Exception {
        Tensor staticInput = tensor(
                DataType.INT64, Shape.of(Long.MAX_VALUE), Optional.empty(), false);
        Shape singleton = Shape.of(1);
        Tensor resolvedInput = tensor(
                DataType.FLOAT32,
                singleton,
                Optional.of(LayoutDescriptor.of(singleton, new long[] {2}, 0, true)),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                ArithmeticException.class,
                () -> staticInput.sliceByLength(
                        new long[] {Long.MAX_VALUE - 1},
                        new long[] {3},
                        new int[] {0},
                        new long[] {Long.MAX_VALUE}));
        assertThrows(
                ArithmeticException.class,
                () -> resolvedInput.sliceByLength(
                        new long[] {0},
                        new long[] {1},
                        new int[] {0},
                        new long[] {Long.MAX_VALUE}));

        assertEquals(before, next.get());
    }

    @Test
    void sliceByLengthDerivesOnlyProvenPositiveStepNonemptyResolvedViews() {
        Shape shape = Shape.of(4, 8);
        LayoutDescriptor dense = LayoutDescriptor.contiguous(shape);
        LayoutDescriptor offset = LayoutDescriptor.of(shape, new long[] {8, 1}, 5, true);
        LayoutDescriptor strided = LayoutDescriptor.of(shape, new long[] {10, 1}, 3, true);
        LayoutDescriptor broadcast = LayoutDescriptor.of(shape, new long[] {0, 1}, 2, true);

        LayoutDescriptor denseResult = slicedByLengthLayout(shape, dense);
        LayoutDescriptor offsetResult = slicedByLengthLayout(shape, offset);
        LayoutDescriptor stridedResult = slicedByLengthLayout(shape, strided);
        LayoutDescriptor broadcastResult = slicedByLengthLayout(shape, broadcast);
        Tensor negative = tensor(
                        DataType.FLOAT32, shape, Optional.of(dense), true)
                .sliceByLength(
                        new long[] {3}, new long[] {2}, new int[] {0}, new long[] {-2});
        Tensor empty = tensor(
                        DataType.FLOAT32, shape, Optional.of(dense), true)
                .sliceByLength(
                        new long[] {7}, new long[] {0}, new int[] {1}, new long[] {2});
        Tensor unresolved = tensor(DataType.FLOAT32, shape, Optional.empty(), true)
                .sliceByLength(
                        new long[] {1}, new long[] {2}, new int[] {1}, new long[] {2});

        assertAll(
                () -> assertArrayEquals(new long[] {16, 2}, denseResult.strides()),
                () -> assertEquals(9, denseResult.storageOffset()),
                () -> assertSame(LayoutKind.STRIDED, denseResult.kind()),
                () -> assertTrue(denseResult.isView()),
                () -> assertArrayEquals(new long[] {16, 2}, offsetResult.strides()),
                () -> assertEquals(14, offsetResult.storageOffset()),
                () -> assertArrayEquals(new long[] {20, 2}, stridedResult.strides()),
                () -> assertEquals(14, stridedResult.storageOffset()),
                () -> assertArrayEquals(new long[] {0, 2}, broadcastResult.strides()),
                () -> assertEquals(3, broadcastResult.storageOffset()),
                () -> assertSame(
                        LayoutKind.BROADCAST_ZERO_STRIDE, broadcastResult.kind()),
                () -> assertTrue(negative.descriptor().layout().isEmpty()),
                () -> assertTrue(empty.descriptor().layout().isEmpty()),
                () -> assertTrue(unresolved.descriptor().layout().isEmpty()));
    }

    @Test
    void sliceByLengthScalarAndIdentityCallsAreFreshCanonicalOutputs() {
        Tensor scalarInput = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        Tensor scalar = scalarInput.sliceByLength(
                new long[0], new long[0], new int[0], new long[0]);
        DynamicDimension n = new DynamicDimension("N");
        Shape dynamicShape = Shape.ofDimensions(n);
        Tensor input = tensor(DataType.FLOAT32, dynamicShape, Optional.empty(), true);
        Tensor first = input.sliceByLength(
                new long[0], new long[0], new int[0], new long[0]);
        Tensor second = input.sliceByLength(
                new long[0], new long[0], new int[0], new long[0]);

        assertAll(
                () -> assertSame(Shape.scalar(), scalar.descriptor().shape()),
                () -> assertSame(n, first.descriptor().shape().dimension(0)),
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(
                        first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertSame(first, first.provenance().orElseThrow().producer().output(0)));
    }

    @Test
    void preservesEveryDataTypeEligibilityAndExactSliceSemantics() {
        Shape shape = Shape.of(3, 6);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Tensor input = tensor(dataType, shape, Optional.empty(), requiresGrad);

                Tensor result = input.slice(
                        new long[] {0, 1},
                        new long[] {3, 6},
                        new int[] {0, 1},
                        new long[] {1, 2});
                TensorProvenance provenance = result.provenance().orElseThrow();
                SliceAttrs attrs = (SliceAttrs) provenance.operation().attrs();

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertEquals(Shape.of(3, 3), result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertSame(SliceKind.SLICE, provenance.operation().kind()),
                        () -> assertEquals(List.of(0L, 1L), attrs.starts()),
                        () -> assertEquals(List.of(3L, 3L), attrs.lengths()),
                        () -> assertEquals(List.of(0, 1), attrs.axes()),
                        () -> assertEquals(List.of(1L, 2L), attrs.steps()),
                        () -> assertEquals(List.of(input), provenance.inputs()),
                        () -> assertSame(input, provenance.inputs().getFirst()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertNotSame(input, result),
                        () -> assertNotEquals(input.id(), result.id()));
            }
        }
    }

    @Test
    void clonesAllArraysNormalizesOnceClampsAndPreservesUnaffectedDimensions() {
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension height = new StaticDimension(3);
        StaticDimension width = new StaticDimension(6);
        Shape shape = Shape.ofDimensions(batch, height, width);
        Tensor input = tensor(DataType.FLOAT32, shape, Optional.empty(), true);
        long[] starts = {Long.MIN_VALUE, -2};
        long[] ends = {Long.MAX_VALUE, -1};
        int[] axes = {1, -1};
        long[] steps = {2, 1};

        Tensor result = input.slice(starts, ends, axes, steps);
        starts[0] = 2;
        ends[0] = 0;
        axes[0] = 2;
        steps[0] = Long.MAX_VALUE;
        SliceAttrs attrs = (SliceAttrs) result.provenance().orElseThrow().operation().attrs();
        Shape resultShape = result.descriptor().shape();

        assertAll(
                () -> assertEquals(List.of(0L, 4L), attrs.starts()),
                () -> assertEquals(List.of(2L, 1L), attrs.lengths()),
                () -> assertEquals(List.of(1, 2), attrs.axes()),
                () -> assertEquals(List.of(2L, 1L), attrs.steps()),
                () -> assertSame(batch, resultShape.dimensions().get(0)),
                () -> assertEquals(new StaticDimension(2), resultShape.dimensions().get(1)),
                () -> assertEquals(new StaticDimension(1), resultShape.dimensions().get(2)),
                () -> assertNotSame(height, resultShape.dimensions().get(1)),
                () -> assertNotSame(width, resultShape.dimensions().get(2)));
    }

    @Test
    void normalizesDirectionalHalfOpenBoundsAndHandlesExtremeSignedSteps() {
        Shape shape = Shape.of(5, 6);
        Tensor input = tensor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);

        Tensor mixed = input.slice(
                new long[] {1, 4},
                new long[] {5, -7},
                new int[] {0, 1},
                new long[] {2, -1});
        SliceAttrs mixedAttrs =
                (SliceAttrs) mixed.provenance().orElseThrow().operation().attrs();
        Tensor explicitMinusOne = input.sliceAxis(0, 4, -1, -1);
        SliceAttrs emptyAttrs =
                (SliceAttrs) explicitMinusOne.provenance().orElseThrow().operation().attrs();
        Tensor throughZero = input.sliceAxis(0, 4, -6, -1);
        SliceAttrs reverseAttrs =
                (SliceAttrs) throughZero.provenance().orElseThrow().operation().attrs();
        Tensor huge = tensor(
                        DataType.INT64,
                        Shape.of(Long.MAX_VALUE),
                        Optional.empty(),
                        false)
                .sliceAxis(0, Long.MAX_VALUE, Long.MIN_VALUE, Long.MIN_VALUE);
        SliceAttrs hugeAttrs =
                (SliceAttrs) huge.provenance().orElseThrow().operation().attrs();

        assertAll(
                () -> assertEquals(List.of(1L, 4L), mixedAttrs.starts()),
                () -> assertEquals(List.of(2L, 5L), mixedAttrs.lengths()),
                () -> assertEquals(List.of(0, 1), mixedAttrs.axes()),
                () -> assertEquals(List.of(2L, -1L), mixedAttrs.steps()),
                () -> assertEquals(Shape.of(2, 5), mixed.descriptor().shape()),
                () -> assertTrue(mixed.descriptor().layout().isEmpty()),
                () -> assertEquals(List.of(0L), emptyAttrs.starts()),
                () -> assertEquals(List.of(0L), emptyAttrs.lengths()),
                () -> assertEquals(Shape.of(0, 6), explicitMinusOne.descriptor().shape()),
                () -> assertEquals(List.of(4L), reverseAttrs.starts()),
                () -> assertEquals(List.of(5L), reverseAttrs.lengths()),
                () -> assertEquals(Shape.of(5, 6), throughZero.descriptor().shape()),
                () -> assertEquals(List.of(Long.MAX_VALUE - 1L), hugeAttrs.starts()),
                () -> assertEquals(List.of(1L), hugeAttrs.lengths()),
                () -> assertEquals(List.of(Long.MIN_VALUE), hugeAttrs.steps()),
                () -> assertEquals(Shape.of(1), huge.descriptor().shape()));
    }

    @Test
    void flipBuildsOneOrderedNegativeStepSliceAndSupportsIdentityAndZeroExtent() {
        StaticDimension empty = new StaticDimension(0);
        Shape shape = Shape.ofDimensions(
                new StaticDimension(2), empty, new StaticDimension(4));
        Tensor input = tensor(
                DataType.INT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        int[] axes = {-1, 1};

        Tensor result = input.flip(axes);
        axes[0] = 0;
        TensorProvenance provenance = result.provenance().orElseThrow();
        SliceAttrs attrs = (SliceAttrs) provenance.operation().attrs();
        Tensor identity = input.flip();
        SliceAttrs identityAttrs =
                (SliceAttrs) identity.provenance().orElseThrow().operation().attrs();
        Tensor scalar = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false).flip();

        assertAll(
                () -> assertEquals(List.of(3L, 0L), attrs.starts()),
                () -> assertEquals(List.of(4L, 0L), attrs.lengths()),
                () -> assertEquals(List.of(2, 1), attrs.axes()),
                () -> assertEquals(List.of(-1L, -1L), attrs.steps()),
                () -> assertEquals(shape, result.descriptor().shape()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertSame(SliceKind.SLICE, provenance.operation().kind()),
                () -> assertEquals(List.of(input), provenance.inputs()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()),
                () -> assertEquals(List.of(), identityAttrs.axes()),
                () -> assertNotSame(input, identity),
                () -> assertNotSame(
                        provenance.producer(),
                        identity.provenance().orElseThrow().producer()),
                () -> assertSame(Shape.scalar(), scalar.descriptor().shape()));
    }

    @Test
    void flipRejectsInvalidDuplicateAndDynamicAxesBeforeIdentityAllocation() throws Exception {
        DynamicDimension dynamic = new DynamicDimension("items");
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(2), dynamic),
                Optional.empty(),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class, () -> TensorSliceExpressions.flip(null, null));
        NullPointerException nullAxes = assertThrows(
                NullPointerException.class, () -> TensorSliceExpressions.flip(input, null));
        IllegalArgumentException invalid = assertThrows(
                IllegalArgumentException.class, () -> input.flip(Integer.MIN_VALUE));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class, () -> input.flip(0, -2));
        IllegalArgumentException dynamicSelected = assertThrows(
                IllegalArgumentException.class, () -> input.flip(1));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("axes", nullAxes.getMessage()),
                () -> assertEquals(
                        "flip axis -2147483648 at index 0 is outside rank 2",
                        invalid.getMessage()),
                () -> assertEquals(
                        "flip contains duplicate normalized axis 0 at index 1",
                        duplicate.getMessage()),
                () -> assertEquals(
                        "flip axis 1 at index 0 must have a statically known dimension",
                        dynamicSelected.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsInvalidRequestsWithExactPrecedenceMessagesAndNoIdentityConsumption()
            throws Exception {
        DynamicDimension dynamic = new DynamicDimension("items");
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(2), dynamic),
                Optional.empty(),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorSliceExpressions.apply(null, null, null, null, null));
        NullPointerException nullStarts = assertThrows(
                NullPointerException.class,
                () -> TensorSliceExpressions.apply(input, null, null, null, null));
        NullPointerException nullEnds = assertThrows(
                NullPointerException.class,
                () -> TensorSliceExpressions.apply(input, new long[0], null, null, null));
        NullPointerException nullAxes = assertThrows(
                NullPointerException.class,
                () -> TensorSliceExpressions.apply(
                        input, new long[0], new long[0], null, null));
        NullPointerException nullSteps = assertThrows(
                NullPointerException.class,
                () -> TensorSliceExpressions.apply(
                        input, new long[0], new long[0], new int[0], null));
        IllegalArgumentException lengths = assertThrows(
                IllegalArgumentException.class,
                () -> input.slice(
                        new long[] {0}, new long[0], new int[] {0}, new long[] {1}));
        IllegalArgumentException low = assertThrows(
                IllegalArgumentException.class,
                () -> input.slice(
                        new long[] {0}, new long[] {1}, new int[] {-3}, new long[] {1}));
        IllegalArgumentException minimum = assertThrows(
                IllegalArgumentException.class,
                () -> input.slice(new long[] {0}, new long[] {1},
                        new int[] {Integer.MIN_VALUE}, new long[] {1}));
        IllegalArgumentException high = assertThrows(
                IllegalArgumentException.class,
                () -> input.slice(new long[] {0}, new long[] {1},
                        new int[] {Integer.MAX_VALUE}, new long[] {1}));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> input.slice(new long[] {0, 0}, new long[] {1, 1},
                        new int[] {0, -2}, new long[] {1, 1}));
        IllegalArgumentException zeroStep = assertThrows(
                IllegalArgumentException.class,
                () -> input.slice(
                        new long[] {0}, new long[] {1}, new int[] {0}, new long[] {0}));
        IllegalArgumentException dynamicSelected = assertThrows(
                IllegalArgumentException.class,
                () -> input.slice(
                        new long[] {0}, new long[] {1}, new int[] {1}, new long[] {1}));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("starts", nullStarts.getMessage()),
                () -> assertEquals("ends", nullEnds.getMessage()),
                () -> assertEquals("axes", nullAxes.getMessage()),
                () -> assertEquals("steps", nullSteps.getMessage()),
                () -> assertEquals(
                        "starts, ends, axes, and steps must have matching lengths",
                        lengths.getMessage()),
                () -> assertEquals(
                        "slice axis -3 at index 0 is outside rank 2", low.getMessage()),
                () -> assertEquals(
                        "slice axis -2147483648 at index 0 is outside rank 2",
                        minimum.getMessage()),
                () -> assertEquals(
                        "slice axis 2147483647 at index 0 is outside rank 2",
                        high.getMessage()),
                () -> assertEquals(
                        "slice contains duplicate normalized axis 0 at index 1",
                        duplicate.getMessage()),
                () -> assertEquals("steps[0] must be non-zero: 0", zeroStep.getMessage()),
                () -> assertEquals(
                        "slice axis 1 at index 0 must have a statically known dimension",
                        dynamicSelected.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void supportsScalarIdentityEmptyZeroDynamicAndOverflowSafeLargeExtents() {
        Tensor scalarInput = tensor(DataType.INT64, Shape.scalar(), Optional.empty(), false);
        Tensor scalar = scalarInput.slice(new long[0], new long[0], new int[0], new long[0]);
        StaticDimension empty = new StaticDimension(0);
        DynamicDimension batch = new DynamicDimension("batch");
        Shape mixed = Shape.ofDimensions(batch, new StaticDimension(6), empty);
        Tensor mixedInput = tensor(DataType.FLOAT64, mixed, Optional.empty(), true);
        Tensor reversed = mixedInput.sliceAxis(1, 5, 2);
        Tensor zeroSelected = mixedInput.sliceAxis(2, Long.MIN_VALUE, Long.MAX_VALUE);
        Tensor huge = tensor(
                        DataType.INT64,
                        Shape.of(Long.MAX_VALUE),
                        Optional.empty(),
                        false)
                .sliceAxis(0, 0, Long.MAX_VALUE);
        Tensor hugeStep = tensor(
                        DataType.INT64,
                        Shape.of(Long.MAX_VALUE),
                        Optional.empty(),
                        false)
                .slice(
                        new long[] {0},
                        new long[] {Long.MAX_VALUE},
                        new int[] {0},
                        new long[] {Long.MAX_VALUE});

        assertAll(
                () -> assertSame(Shape.scalar(), scalar.descriptor().shape()),
                () -> assertEquals(List.of(),
                        ((SliceAttrs) scalar.provenance().orElseThrow()
                                .operation().attrs()).axes()),
                () -> assertSame(batch, reversed.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(new StaticDimension(0),
                        reversed.descriptor().shape().dimensions().get(1)),
                () -> assertSame(empty, reversed.descriptor().shape().dimensions().get(2)),
                () -> assertSame(batch, zeroSelected.descriptor().shape().dimensions().get(0)),
                () -> assertEquals(new StaticDimension(0),
                        zeroSelected.descriptor().shape().dimensions().get(2)),
                () -> assertEquals(Shape.of(Long.MAX_VALUE), huge.descriptor().shape()),
                () -> assertEquals(Shape.of(1), hugeStep.descriptor().shape()));
    }

    @Test
    void derivesCheckedViewGeometryFromEveryResolvedInputLayoutKind() {
        Shape shape = Shape.of(4, 6);
        LayoutDescriptor dense = LayoutDescriptor.contiguous(shape);
        LayoutDescriptor offset = LayoutDescriptor.of(shape, new long[] {6, 1}, 5, true);
        LayoutDescriptor strided = LayoutDescriptor.of(shape, new long[] {8, 1}, 3, true);
        LayoutDescriptor broadcast = LayoutDescriptor.of(shape, new long[] {0, 1}, 2, true);

        LayoutDescriptor denseResult = slicedLayout(shape, dense);
        LayoutDescriptor offsetResult = slicedLayout(shape, offset);
        LayoutDescriptor stridedResult = slicedLayout(shape, strided);
        LayoutDescriptor broadcastResult = slicedLayout(shape, broadcast);

        assertAll(
                () -> assertArrayEquals(new long[] {12, 2}, denseResult.strides()),
                () -> assertEquals(7, denseResult.storageOffset()),
                () -> assertSame(LayoutKind.STRIDED, denseResult.kind()),
                () -> assertEquals(24, denseResult.referencedElementSpan()),
                () -> assertTrue(denseResult.isView()),
                () -> assertNotSame(dense, denseResult),
                () -> assertArrayEquals(new long[] {12, 2}, offsetResult.strides()),
                () -> assertEquals(12, offsetResult.storageOffset()),
                () -> assertEquals(29, offsetResult.referencedElementSpan()),
                () -> assertNotSame(offset, offsetResult),
                () -> assertArrayEquals(new long[] {16, 2}, stridedResult.strides()),
                () -> assertEquals(12, stridedResult.storageOffset()),
                () -> assertEquals(33, stridedResult.referencedElementSpan()),
                () -> assertNotSame(strided, stridedResult),
                () -> assertArrayEquals(new long[] {0, 2}, broadcastResult.strides()),
                () -> assertEquals(3, broadcastResult.storageOffset()),
                () -> assertSame(LayoutKind.BROADCAST_ZERO_STRIDE, broadcastResult.kind()),
                () -> assertEquals(8, broadcastResult.referencedElementSpan()),
                () -> assertNotSame(broadcast, broadcastResult));
    }

    @Test
    void keepsUnresolvedAndEmptyResultsUnresolvedButResolvesNonEmptyScalarIdentity() {
        Shape staticShape = Shape.of(2, 3);
        Tensor unresolved = tensor(DataType.INT32, staticShape, Optional.empty(), false);
        Shape emptyShape = Shape.of(2, 0, 3);
        LayoutDescriptor emptyLayout = LayoutDescriptor.contiguous(emptyShape);
        Tensor empty = tensor(
                DataType.FLOAT32, emptyShape, Optional.of(emptyLayout), true);
        Shape scalarShape = Shape.scalar();
        LayoutDescriptor scalarLayout = LayoutDescriptor.contiguous(scalarShape);
        Tensor scalar = tensor(
                DataType.BOOL, scalarShape, Optional.of(scalarLayout), false);

        Tensor unresolvedResult = unresolved.sliceAxis(0, 0, 2);
        Tensor emptyIdentity = empty.slice(new long[0], new long[0], new int[0], new long[0]);
        Tensor emptySelection = tensor(
                        DataType.FLOAT64,
                        staticShape,
                        Optional.of(LayoutDescriptor.contiguous(staticShape)),
                        true)
                .sliceAxis(1, 2, 2);
        LayoutDescriptor scalarResult = scalar
                .slice(new long[0], new long[0], new int[0], new long[0])
                .descriptor()
                .layout()
                .orElseThrow();

        assertAll(
                () -> assertTrue(unresolvedResult.descriptor().layout().isEmpty()),
                () -> assertTrue(emptyIdentity.descriptor().layout().isEmpty()),
                () -> assertTrue(emptySelection.descriptor().layout().isEmpty()),
                () -> assertArrayEquals(new long[0], scalarResult.strides()),
                () -> assertEquals(0, scalarResult.storageOffset()),
                () -> assertEquals(1, scalarResult.referencedElementSpan()),
                () -> assertTrue(scalarResult.isView()),
                () -> assertNotSame(scalarLayout, scalarResult));
    }

    @Test
    void checkedStrideOverflowFailsBeforeIdentityAllocation() throws Exception {
        Shape singleton = Shape.of(1);
        LayoutDescriptor largeStride = LayoutDescriptor.of(
                singleton, new long[] {2}, 0, true);
        Tensor input = tensor(
                DataType.FLOAT32, singleton, Optional.of(largeStride), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        ArithmeticException failure = assertThrows(
                ArithmeticException.class,
                () -> input.slice(
                        new long[] {0},
                        new long[] {1},
                        new int[] {0},
                        new long[] {Long.MAX_VALUE}));

        assertAll(
                () -> assertEquals("long overflow", failure.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void singleAxisUsesOneStepOneEntryAndIdentityRepeatedNestedCallsRemainFresh() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3, 6), Optional.empty(), true);

        Tensor axis = input.sliceAxis(-1, -2, -1);
        SliceAttrs axisAttrs = (SliceAttrs) axis.provenance().orElseThrow().operation().attrs();
        Tensor first = input.slice(new long[0], new long[0], new int[0], new long[0]);
        Tensor second = input.slice(new long[0], new long[0], new int[0], new long[0]);
        Tensor nested = first.sliceAxis(0, 0, 3);

        assertAll(
                () -> assertEquals(List.of(4L), axisAttrs.starts()),
                () -> assertEquals(List.of(1L), axisAttrs.lengths()),
                () -> assertEquals(List.of(1), axisAttrs.axes()),
                () -> assertEquals(List.of(1L), axisAttrs.steps()),
                () -> assertEquals(Shape.of(3, 1), axis.descriptor().shape()),
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(first, nested),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotEquals(first.id(), nested.id()),
                () -> assertSame(input,
                        first.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(first,
                        nested.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void leavesInputMetadataStorageLivenessAndValuesUntouched() {
        Shape shape = Shape.of(3, 6);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                shape, new long[] {6, 1}, 2, true);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true);
        Arena arena = Arena.ofConfined();
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 20, arena.allocate(80, 1));
        float[] values = new float[18];
        for (int index = 0; index < values.length; index++) {
            values[index] = index + 1;
        }
        storage.segment().asSlice(8, 72).copyFrom(MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.FLOAT32, shape, Optional.empty(), false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(
                                SliceKind.SLICE,
                                new SliceAttrs(List.of(), List.of(), List.of(), List.of())),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));
        float[] before = storage.segment().asSlice(8, 72).toArray(JAVA_FLOAT);

        Tensor result = input.sliceAxis(1, 1, 6);
        float[] after = storage.segment().asSlice(8, 72).toArray(JAVA_FLOAT);
        arena.close();

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(inputLayout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(originalProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertFalse(storage.isAlive()),
                () -> assertArrayEquals(before, after),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void successfulCallConsumesOneIdentityAndExhaustionOccursAtFinalDelegation()
            throws Exception {
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.of(3, 6),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(3, 6))),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        input.sliceAxis(1, 1, 5);
        input.sliceByLength(
                new long[] {1}, new long[] {2}, new int[] {1}, new long[] {2});
        assertEquals(before + 2, next.get());

        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class,
                    () -> input.sliceByLength(
                            new long[] {1},
                            new long[] {2},
                            new int[] {1},
                            new long[] {2}));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static LayoutDescriptor slicedLayout(
            Shape inputShape, LayoutDescriptor inputLayout) {
        return tensor(DataType.FLOAT32, inputShape, Optional.of(inputLayout), false)
                .slice(
                        new long[] {1, 1},
                        new long[] {4, 6},
                        new int[] {0, 1},
                        new long[] {2, 2})
                .descriptor()
                .layout()
                .orElseThrow();
    }

    private static LayoutDescriptor slicedByLengthLayout(
            Shape inputShape, LayoutDescriptor inputLayout) {
        return tensor(DataType.FLOAT32, inputShape, Optional.of(inputLayout), false)
                .sliceByLength(
                        new long[] {1, 1},
                        new long[] {2, 3},
                        new int[] {0, 1},
                        new long[] {2, 2})
                .descriptor()
                .layout()
                .orElseThrow();
    }

    private static long publicMethodsNamed(String name) {
        return Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .filter(method -> method.getName().equals(name))
                .count();
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

    private static <T extends Throwable> void assertFailure(
            Class<T> type,
            String message,
            org.junit.jupiter.api.function.Executable executable) {
        assertEquals(message, assertThrows(type, executable).getMessage());
    }
}
