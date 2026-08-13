package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.*;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuCapabilityProviderTest {
    @Test void reportsExactStaticStableOrderingAndTwoOutputTopK() {
        var provider = new CpuCapabilityProvider();
        var input = descriptor(DataType.BFLOAT16, Shape.of(2, 5));
        var values = descriptor(DataType.BFLOAT16, Shape.of(2, 3));
        var indices = descriptor(DataType.INT64, Shape.of(2, 3));
        assertAll(() -> assertTrue(provider.supports(query(OrderingKind.SORT,
                        new SortAttrs(1, true), List.of(input), input))),
                () -> assertTrue(provider.supports(query(OrderingKind.ARGSORT,
                        new SortAttrs(1, false), List.of(input),
                        descriptor(DataType.INT64, Shape.of(2, 5))))),
                () -> assertTrue(provider.supports(new OperationCapabilityQuery(
                        new Operation(TopKKind.TOP_K, new TopKAttrs(1, 3, true, false)),
                        List.of(input), List.of(values, indices)))),
                () -> assertFalse(provider.supports(new OperationCapabilityQuery(
                        new Operation(TopKKind.TOP_K, new TopKAttrs(1, 3, true, false)),
                        List.of(input), List.of(indices, values)))));
    }
    @Test void reportsOnlyExactStaticSliceUpdateOccurrences() {
        var provider = new CpuCapabilityProvider();
        var base = descriptor(DataType.BFLOAT16, Shape.of(5));
        var update = descriptor(DataType.BFLOAT16, Shape.of(2));
        var output = descriptor(DataType.BFLOAT16, Shape.of(5));
        var signed = new SliceAttrs(List.of(4L), List.of(2L), List.of(0), List.of(-2L));
        var crop = new CropToShapeAttrs(Shape.of(2), Shape.of(3));
        assertAll(
                () -> assertTrue(provider.supports(query(SliceKind.SLICE_UPDATE, signed,
                        List.of(base, update), output))),
                () -> assertTrue(provider.supports(query(SliceKind.SLICE_UPDATE, crop,
                        List.of(base, update), output))),
                () -> assertFalse(provider.supports(query(SliceKind.SLICE_UPDATE,
                        new CropToShapeAttrs(Shape.of(1), Shape.of(3)),
                        List.of(base, update), output))),
                () -> assertFalse(provider.supports(query(SliceKind.SLICE_UPDATE, signed,
                        List.of(base, descriptor(DataType.BFLOAT16, Shape.of(3))), output))),
                () -> assertFalse(provider.supports(query(SliceKind.SLICE_UPDATE, signed,
                        List.of(base, update), descriptor(DataType.BFLOAT16, Shape.of(4))))));
    }

    @Test void sliceUpdateAcceptsExactEndpointsAndRejectsEveryOutOfBoundsEndpoint() {
        var provider = new CpuCapabilityProvider();
        var base = descriptor(DataType.INT32, Shape.of(5));
        var update = descriptor(DataType.INT32, Shape.of(2));
        var output = descriptor(DataType.INT32, Shape.of(5));
        var positiveBoundary = new SliceAttrs(List.of(0L), List.of(2L), List.of(0), List.of(4L));
        var negativeBoundary = new SliceAttrs(List.of(4L), List.of(2L), List.of(0), List.of(-4L));
        assertAll(
                () -> assertTrue(provider.supports(query(SliceKind.SLICE_UPDATE,
                        positiveBoundary, List.of(base, update), output))),
                () -> assertTrue(provider.supports(query(SliceKind.SLICE_UPDATE,
                        negativeBoundary, List.of(base, update), output))),
                () -> assertFalse(provider.supports(query(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(1L), List.of(2L), List.of(0), List.of(4L)),
                        List.of(base, update), output))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new SliceAttrs(List.of(3L), List.of(2L), List.of(0),
                                List.of(-4L))),
                () -> assertTrue(provider.supports(query(SliceKind.SLICE_UPDATE,
                        new CropToShapeAttrs(Shape.of(2), Shape.of(3)),
                        List.of(base, update), output))),
                () -> assertFalse(provider.supports(query(SliceKind.SLICE_UPDATE,
                        new CropToShapeAttrs(Shape.of(2), Shape.of(4)),
                        List.of(base, update), output))));
    }

    @Test void reportsExactResolvedPointwiseOccurrences() {
        var provider = new CpuCapabilityProvider();
        Shape shape = Shape.of(2, 0, 3);
        var dense = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        assertAll(
                () -> assertSame(CpuCapabilityProvider.CPU_BACKEND_ID, provider.backendId()),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(dense, dense), dense))),
                () -> assertTrue(provider.supports(query(UnaryElementwiseKind.GELU,
                        List.of(dense), dense))),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.MUL,
                        List.of(dense, dense), dense))),
                () -> assertTrue(provider.supports(query(UnaryElementwiseKind.GELU_TANH_APPROXIMATION,
                        List.of(dense), dense))));
        var unresolved = new TensorDescriptor(DataType.FLOAT64, shape, Optional.empty(), false);
        var float32 = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        assertAll(
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(unresolved, unresolved), unresolved))),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(float32, float32), float32))),
                () -> assertEquals("query", assertThrows(NullPointerException.class,
                        () -> provider.supports(null)).getMessage()));
    }

    @Test void reportsStaticRightBroadcastAndResolvedStridedOccurrencesTruthfully() {
        var provider = new CpuCapabilityProvider();
        Shape leftShape = Shape.of(2, 1, 3), rightShape = Shape.of(4, 3);
        Shape outputShape = Shape.of(2, 4, 3);
        var left = descriptor(leftShape, LayoutDescriptor.of(leftShape,
                new long[]{7, 0, 2}, 3, true));
        var right = descriptor(rightShape, LayoutDescriptor.contiguous(rightShape));
        var output = descriptor(outputShape, LayoutDescriptor.contiguous(outputShape));
        var wrong = descriptor(Shape.of(2, 3), LayoutDescriptor.contiguous(Shape.of(2, 3)));
        assertAll(
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(left, right), output))),
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.ADD,
                        List.of(left, right), wrong))),
                () -> assertFalse(provider.supports(query(UnaryElementwiseKind.GELU,
                        List.of(left), output))));
    }

    @Test void reportsTheExactTypeAttributeAndFamilyMatrix() {
        var provider = new CpuCapabilityProvider();
        Shape shape = Shape.of(3);
        var f64 = descriptor(DataType.FLOAT64, shape);
        var f32 = descriptor(DataType.FLOAT32, shape);
        var i32 = descriptor(DataType.INT32, shape);
        var i64 = descriptor(DataType.INT64, shape);
        var bool = descriptor(DataType.BOOL, shape);
        var bf16 = descriptor(DataType.BFLOAT16, shape);
        for (var kind : UnaryElementwiseKind.values()) {
            assertTrue(provider.supports(query(kind, List.of(f64), f64)), kind + " FLOAT64");
            assertTrue(provider.supports(query(kind, List.of(f32), f32)), kind + " FLOAT32");
            assertFalse(provider.supports(query(kind, List.of(bf16), bf16)), kind + " BFLOAT16");
        }
        for (var type : List.of(f64, f32, i32, i64)) {
            for (var kind : List.of(BinaryArithmeticKind.ADD, BinaryArithmeticKind.SUB,
                    BinaryArithmeticKind.MUL)) assertTrue(provider.supports(query(kind,
                    NoOperationAttrs.INSTANCE, List.of(type, type), type)));
            for (var kind : BinaryComparisonKind.values()) assertTrue(provider.supports(query(kind,
                    NoOperationAttrs.INSTANCE, List.of(type, type), bool)));
        }
        assertAll(
                () -> assertTrue(provider.supports(query(ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.int32(7)), List.of(i32), i32))),
                () -> assertFalse(provider.supports(query(ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.int64(7)), List.of(i32), i32))),
                () -> assertTrue(provider.supports(query(FloatingClassificationKind.IS_NAN,
                        NoOperationAttrs.INSTANCE, List.of(f32), bool))),
                () -> assertTrue(provider.supports(query(WhereSelectionKind.WHERE,
                        NoOperationAttrs.INSTANCE, List.of(bool, f64, f64), f64))),
                () -> assertFalse(provider.supports(query(WhereSelectionKind.WHERE,
                        NoOperationAttrs.INSTANCE, List.of(bool, i32, i32), i32))),
                () -> assertTrue(provider.supports(query(CastKind.CAST,
                        new CastAttrs(DataType.BOOL), List.of(bool), bool))),
                () -> assertFalse(provider.supports(query(CastKind.CAST,
                        new CastAttrs(DataType.FLOAT64), List.of(f32), f64))),
                () -> assertFalse(provider.supports(query(CastKind.CAST,
                        new CastAttrs(DataType.BFLOAT16), List.of(bf16), bf16))),
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.ADD,
                        NoOperationAttrs.INSTANCE, List.of(f32, f64), f64))),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.DIV,
                        NoOperationAttrs.INSTANCE, List.of(f64, f64), f64))),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.DIV,
                        NoOperationAttrs.INSTANCE, List.of(f32, f32), f32))),
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.DIV,
                        NoOperationAttrs.INSTANCE, List.of(i32, i32), i32))),
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.DIV,
                        NoOperationAttrs.INSTANCE, List.of(bf16, bf16), bf16))),
                () -> assertTrue(provider.supports(query(ScalarElementwiseKind.DIV,
                        new ScalarValueAttrs(ScalarValue.float32(-0.0f)), List.of(f32), f32))),
                () -> assertTrue(provider.supports(query(ScalarElementwiseKind.POW,
                        new ScalarValueAttrs(ScalarValue.float64(0.5d)), List.of(f64), f64))),
                () -> assertFalse(provider.supports(query(ScalarElementwiseKind.DIV,
                        new ScalarValueAttrs(ScalarValue.float64(2.0d)), List.of(f32), f32))),
                () -> assertFalse(provider.supports(query(ScalarElementwiseKind.POW,
                        new ScalarValueAttrs(ScalarValue.int32(2)), List.of(i32), i32))),
                () -> assertFalse(provider.supports(query(ScalarElementwiseKind.POW,
                        new ScalarValueAttrs(ScalarValue.bfloat16(2.0f)), List.of(bf16), bf16))),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.MIN,
                        NoOperationAttrs.INSTANCE, List.of(i64, i64), i64))),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.MAX,
                        NoOperationAttrs.INSTANCE, List.of(f32, f32), f32))),
                () -> assertTrue(provider.supports(query(BinaryArithmeticKind.POW,
                        NoOperationAttrs.INSTANCE, List.of(f64, f64), f64))),
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.POW,
                        NoOperationAttrs.INSTANCE, List.of(i32, i32), i32))),
                () -> assertTrue(provider.supports(query(ScalarElementwiseKind.MIN,
                        new ScalarValueAttrs(ScalarValue.int32(-1)), List.of(i32), i32))),
                () -> assertTrue(provider.supports(query(ScalarElementwiseKind.CLAMP,
                        new ClampRangeAttrs(ScalarValue.float32(-0.0f), ScalarValue.float32(+0.0f)),
                        List.of(f32), f32))),
                () -> assertFalse(provider.supports(query(ScalarElementwiseKind.CLAMP,
                        new ClampRangeAttrs(ScalarValue.int32(-1), ScalarValue.int32(1)),
                        List.of(i32), i32))),
                () -> assertTrue(provider.supports(query(BooleanLogicalKind.AND,
                        NoOperationAttrs.INSTANCE, List.of(bool, bool), bool))),
                () -> assertTrue(provider.supports(query(BooleanLogicalKind.NOT,
                        NoOperationAttrs.INSTANCE, List.of(bool), bool))),
                () -> assertFalse(provider.supports(query(BooleanLogicalKind.OR,
                        NoOperationAttrs.INSTANCE, List.of(i32, i32), bool))));
    }

    @Test void reportsOnlyExactStaticResolvedAffineOccurrencesForEveryDataType() {
        var provider = new CpuCapabilityProvider();
        Shape inputShape = Shape.of(2,3), selectedShape = Shape.of(2);
        for (DataType type : DataType.values()) {
            var input = descriptor(type, inputShape);
            var selected = new TensorDescriptor(type, selectedShape, Optional.of(
                    LayoutDescriptor.of(selectedShape, new long[]{3}, 1, true)), false);
            assertTrue(provider.supports(query(SelectKind.SELECT, new SelectAttrs(1,1),
                    List.of(input), selected)), type.toString());
        }
        var input = descriptor(DataType.FLOAT64, inputShape);
        var selected = new TensorDescriptor(DataType.FLOAT64, selectedShape, Optional.of(
                LayoutDescriptor.of(selectedShape, new long[]{3}, 1, true)), false);
        var wrongLayout = new TensorDescriptor(DataType.FLOAT64, selectedShape, Optional.of(
                LayoutDescriptor.contiguous(selectedShape)), false);
        var wrongType = new TensorDescriptor(DataType.FLOAT32, selectedShape, Optional.of(
                LayoutDescriptor.of(selectedShape, new long[]{3}, 1, true)), false);
        var unresolved = new TensorDescriptor(DataType.FLOAT64, selectedShape, Optional.empty(), false);
        assertAll(
                () -> assertTrue(provider.supports(query(ContiguousKind.CONTIGUOUS,
                        NoOperationAttrs.INSTANCE, List.of(input), input))),
                () -> assertTrue(provider.supports(query(ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of(3,2)), List.of(input),
                        new TensorDescriptor(DataType.FLOAT64, Shape.of(3,2), Optional.of(
                                LayoutDescriptor.of(Shape.of(3,2), new long[]{2,1}, 0, true)), false)))),
                () -> assertFalse(provider.supports(query(SelectKind.SELECT, new SelectAttrs(1,1),
                        List.of(input), wrongLayout))),
                () -> assertFalse(provider.supports(query(SelectKind.SELECT, new SelectAttrs(1,1),
                        List.of(input), wrongType))),
                () -> assertFalse(provider.supports(query(SelectKind.SELECT, new SelectAttrs(1,1),
                        List.of(input), unresolved))),
                () -> assertFalse(provider.supports(query(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(0L), List.of(1L), List.of(0), List.of(1L)),
                        List.of(input, input), input))));
    }

    @Test void reportsOnlyTheExactStaticResolvedMovementMatrix() {
        var provider = new CpuCapabilityProvider();
        var two = descriptor(DataType.INT32, Shape.of(2));
        var one = descriptor(DataType.INT32, Shape.of(1));
        var five = descriptor(DataType.INT32, Shape.of(5));
        var six = descriptor(DataType.INT32, Shape.of(6));
        var stacked = descriptor(DataType.INT32, Shape.of(2, 2));
        var nonInjective = new TensorDescriptor(DataType.INT32, Shape.of(4), Optional.of(
                LayoutDescriptor.of(Shape.of(4), new long[]{0}, 0, true)), false);
        assertAll(
                () -> assertTrue(provider.supports(query(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(2L), ScalarValue.int32(-1)),
                        List.of(two), five))),
                () -> assertTrue(provider.supports(query(TileKind.TILE,
                        new TileAttrs(List.of(3L)), List.of(two), six))),
                () -> assertTrue(provider.supports(query(TensorCompositionKind.CONCAT,
                        new CompositionAxisAttrs(0), List.of(two, one, two), five))),
                () -> assertTrue(provider.supports(query(TensorCompositionKind.STACK,
                        new CompositionAxisAttrs(1), List.of(two, two), stacked))),
                () -> assertFalse(provider.supports(query(TileKind.TILE,
                        new TileAttrs(List.of(2L)), List.of(two), nonInjective))),
                () -> assertFalse(provider.supports(query(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(2L), ScalarValue.int64(-1)),
                        List.of(two), five))));
    }

    @Test void reportsExactWindowExtractionTypeAndGeometryMatrix() {
        var provider = new CpuCapabilityProvider();
        var boolInput = descriptor(DataType.BOOL, Shape.of(2, 3));
        var boolOutput = descriptor(DataType.BOOL, Shape.of(2, 2, 2));
        var imageF32 = descriptor(DataType.FLOAT32, Shape.of(1, 1, 3, 3));
        var columnsF32 = descriptor(DataType.FLOAT32, Shape.of(1, 4, 16));
        var imageI32 = descriptor(DataType.INT32, Shape.of(1, 1, 3, 3));
        var columnsI32 = descriptor(DataType.INT32, Shape.of(1, 4, 16));
        var window = new Window2dAttrs(2, 2, 1, 1, 1, 1, 1, 1, false);
        assertAll(
                () -> assertTrue(provider.supports(query(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(1, 2, 1), List.of(boolInput), boolOutput))),
                () -> assertTrue(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        window, List.of(imageF32), columnsF32))),
                () -> assertTrue(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        new Unfold2dAttrs(window, ScalarValue.float32(-0.0f)),
                        List.of(imageF32), columnsF32))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        window, List.of(imageI32), columnsI32))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        new Unfold2dAttrs(window, ScalarValue.float64(0.0)),
                        List.of(imageF32), columnsF32))),
                () -> assertTrue(provider.supports(query(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(Shape.of(1, 1, 3, 3), window),
                        List.of(columnsF32), imageF32))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.FOLD2D,
                        new Fold2dAttrs(Shape.of(1, 1, 3, 3), window),
                        List.of(columnsI32), imageI32))),
                () -> assertTrue(provider.supports(query(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1),
                        List.of(descriptor(DataType.INT64, Shape.of(3, 3))),
                        descriptor(DataType.INT64, Shape.of(5))))));
    }

    @Test void windowExtractionFailsClosedForEveryExcludedStructuralBoundary() {
        var provider = new CpuCapabilityProvider();
        var input = descriptor(DataType.FLOAT32, Shape.of(1, 1, 3, 3));
        var output = descriptor(DataType.FLOAT32, Shape.of(1, 4, 4));
        var window = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        var unresolved = new TensorDescriptor(DataType.FLOAT32, Shape.of(1, 4, 4),
                Optional.empty(), false);
        var dynamicShape = Shape.ofDimensions(new StaticDimension(1),
                new DynamicDimension("columns"), new StaticDimension(4));
        var dynamic = new TensorDescriptor(DataType.FLOAT32, dynamicShape,
                Optional.empty(), false);
        var nonInjective = new TensorDescriptor(DataType.FLOAT32, Shape.of(1, 4, 4),
                Optional.of(LayoutDescriptor.of(Shape.of(1, 4, 4),
                        new long[]{0, 0, 1}, 0, true)), false);
        var overflowWindow = new Window2dAttrs(Long.MAX_VALUE, 1, 1, 1,
                0, 0, Long.MAX_VALUE, 1, false);
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> query(WindowTransformKind.UNFOLD2D,
                                window, List.of(input, input), output)),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        window, List.of(descriptor(DataType.FLOAT32, Shape.of(1, 3, 3))), output))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        window, List.of(input), descriptor(DataType.FLOAT32, Shape.of(1, 4, 5))))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        window, List.of(input), unresolved))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        window, List.of(input), dynamic))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        window, List.of(input), nonInjective))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD2D,
                        overflowWindow, List.of(input), output))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> query(WindowTransformKind.UNFOLD_AXIS,
                                window, List.of(input), output)),
                () -> assertFalse(provider.supports(query(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(1, 3, 1), List.of(input), output))),
                () -> assertFalse(provider.supports(query(WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(3, 1, 1), List.of(input), output))));
    }

    @Test void reportsEveryExactStaticResolvedIndexingTypeAndRankRow() {
        var provider = new CpuCapabilityProvider();
        for (DataType dataType : DataType.values()) {
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                var data = descriptor(dataType, Shape.of(2, 3));
                var axisIndices = descriptor(indexType, Shape.of(4));
                var elementIndices = descriptor(indexType, Shape.of(2, 4));
                var ndIndices = descriptor(indexType, Shape.of(4, 1));
                assertAll(dataType + "/" + indexType,
                        () -> assertTrue(provider.supports(query(AxisGatherKind.GATHER,
                                new IndexAxisAttrs(1), List.of(data, axisIndices),
                                descriptor(dataType, Shape.of(2, 4))))),
                        () -> assertTrue(provider.supports(query(
                                AxisGatherKind.GATHER_ELEMENTS, new IndexAxisAttrs(1),
                                List.of(data, elementIndices),
                                descriptor(dataType, Shape.of(2, 4))))),
                        () -> assertTrue(provider.supports(query(GatherNdKind.GATHER_ND,
                                new GatherNdAttrs(0), List.of(data, ndIndices),
                                descriptor(dataType, Shape.of(4, 3))))));
            }
        }
        for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
            assertAll(indexType.toString(),
                    () -> assertTrue(provider.supports(query(AxisGatherKind.GATHER,
                            new IndexAxisAttrs(0),
                            List.of(descriptor(DataType.INT64, Shape.of(3)),
                                    descriptor(indexType, Shape.scalar())),
                            descriptor(DataType.INT64, Shape.scalar())))),
                    () -> assertTrue(provider.supports(query(OneHotKind.ONE_HOT,
                            new OneHotAttrs(5), List.of(descriptor(indexType, Shape.of(2, 3))),
                            descriptor(DataType.BOOL, Shape.of(2, 3, 5))))));
        }
        assertTrue(provider.supports(query(GatherNdKind.GATHER_ND, new GatherNdAttrs(1),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3, 4)),
                        descriptor(DataType.INT64, Shape.of(2, 5, 2))),
                descriptor(DataType.FLOAT32, Shape.of(2, 5)))));
    }

    @Test void indexingCapabilityFailsClosedAtEveryStructuralBoundary() {
        var provider = new CpuCapabilityProvider();
        var data = descriptor(DataType.FLOAT32, Shape.of(2, 3));
        var indices = descriptor(DataType.INT32, Shape.of(2));
        var validOutput = descriptor(DataType.FLOAT32, Shape.of(2, 2));
        var unresolved = new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 2),
                Optional.empty(), false);
        var nonInjective = new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 2), Optional.of(
                LayoutDescriptor.of(Shape.of(2, 2), new long[]{0, 1}, 0, true)), false);
        assertAll(
                () -> assertFalse(provider.supports(query(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(2), List.of(data, indices), validOutput))),
                () -> assertFalse(provider.supports(query(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(1), List.of(data, descriptor(DataType.FLOAT32,
                                Shape.of(2))), validOutput))),
                () -> assertFalse(provider.supports(query(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(1), List.of(data, indices),
                        descriptor(DataType.INT32, Shape.of(2, 2))))),
                () -> assertFalse(provider.supports(query(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(1), List.of(data, indices),
                        descriptor(DataType.FLOAT32, Shape.of(2, 3))))),
                () -> assertFalse(provider.supports(query(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(1), List.of(data, indices), unresolved))),
                () -> assertFalse(provider.supports(query(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(1), List.of(data, indices), nonInjective))),
                () -> assertFalse(provider.supports(query(AxisGatherKind.GATHER_ELEMENTS,
                        new IndexAxisAttrs(1), List.of(data,
                                descriptor(DataType.INT32, Shape.of(3, 2))), validOutput))),
                () -> assertFalse(provider.supports(query(GatherNdKind.GATHER_ND,
                        new GatherNdAttrs(1), List.of(data,
                                descriptor(DataType.INT32, Shape.of(3, 1))),
                        descriptor(DataType.FLOAT32, Shape.of(3))))),
                () -> assertFalse(provider.supports(query(GatherNdKind.GATHER_ND,
                        new GatherNdAttrs(0), List.of(data,
                                descriptor(DataType.INT32, Shape.of(2, 3))),
                        descriptor(DataType.FLOAT32, Shape.of(2))))),
                () -> assertFalse(provider.supports(query(OneHotKind.ONE_HOT,
                        new OneHotAttrs(3), List.of(descriptor(DataType.INT32, Shape.of(2))),
                        descriptor(DataType.INT32, Shape.of(2, 3))))),
                () -> assertFalse(provider.supports(query(OneHotKind.ONE_HOT,
                        new OneHotAttrs(3), List.of(descriptor(DataType.INT32, Shape.of(2))),
                        descriptor(DataType.BOOL, Shape.of(3, 2))))));
    }

    @Test void scatterCapabilityAdmitsOnlyCurrentExactFormsTypesAndShapes() {
        var provider=new CpuCapabilityProvider();
        var data=descriptor(DataType.FLOAT32,Shape.of(2,3));
        var indices=descriptor(DataType.INT64,Shape.of(2,4));
        var updates=descriptor(DataType.FLOAT32,Shape.of(2,4));
        var output=descriptor(DataType.FLOAT32,Shape.of(2,3));
        assertAll(() -> assertTrue(provider.supports(query(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1,ScatterReduction.MUL),List.of(data,indices,updates),output))),
                () -> assertTrue(provider.supports(query(AxisScatterKind.SCATTER_ADD,
                        new IndexAxisAttrs(1),List.of(data,descriptor(DataType.INT32,Shape.of(4)),
                                descriptor(DataType.FLOAT32,Shape.of(2,4))),output))),
                () -> assertTrue(provider.supports(query(ScatterNdKind.SCATTER_ND,
                        new ScatterNdAttrs(0,ScatterReduction.NONE),List.of(data,
                                descriptor(DataType.INT32,Shape.of(5,1)),
                                descriptor(DataType.FLOAT32,Shape.of(5,3))),output))),
                () -> assertFalse(provider.supports(query(AxisScatterKind.SCATTER_ADD,
                        new IndexAxisAttrs(1),List.of(data,descriptor(DataType.INT32,Shape.of(4)),
                                descriptor(DataType.FLOAT32,Shape.of(4))),output))),
                () -> assertFalse(provider.supports(query(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1,ScatterReduction.ADD),List.of(
                                descriptor(DataType.BOOL,Shape.of(2,3)),
                                descriptor(DataType.INT32,Shape.of(2,4)),
                                descriptor(DataType.BOOL,Shape.of(2,4))),
                        descriptor(DataType.BOOL,Shape.of(2,3))))),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new OperationCapabilityQuery(
                        new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                        List.of(data, indices), List.of(output))),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new OperationCapabilityQuery(
                        new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                        List.of(data, indices, updates), List.of())),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new OperationCapabilityQuery(
                        new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                                new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                        List.of(data, indices, updates), List.of(output, output))),
                () -> assertFalse(provider.supports(query(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.NONE), List.of(data,
                                descriptor(DataType.FLOAT32, Shape.of(2, 4)), updates), output))),
                () -> assertFalse(provider.supports(query(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.NONE), List.of(data, indices,
                                descriptor(DataType.INT32, Shape.of(2, 4))), output))),
                () -> assertFalse(provider.supports(query(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.NONE), List.of(data, indices,
                                updates), new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 3),
                                Optional.empty(), false)))),
                () -> assertFalse(provider.supports(query(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.NONE), List.of(data, indices,
                                updates), new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 3),
                                Optional.of(LayoutDescriptor.of(Shape.of(2, 3),
                                        new long[]{0, 1}, 0, true)), false)))));
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static OperationCapabilityQuery query(Object kind, List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        return query(kind, NoOperationAttrs.INSTANCE, inputs, output);
    }

    private static OperationCapabilityQuery query(Object kind,
            io.github.pho001.synaptik.model.operation.OperationAttrs attrs,
            List<TensorDescriptor> inputs, TensorDescriptor output) {
        return new OperationCapabilityQuery(new Operation(
                (io.github.pho001.synaptik.model.operation.OperationKind) kind, attrs),
                inputs, List.of(output));
    }
}
