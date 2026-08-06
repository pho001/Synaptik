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
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuCapabilityProviderTest {
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
                () -> assertFalse(provider.supports(query(UnaryElementwiseKind.GELU_TANH_APPROXIMATION,
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
                () -> assertFalse(provider.supports(query(BinaryArithmeticKind.DIV,
                        NoOperationAttrs.INSTANCE, List.of(f64, f64), f64))));
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
