package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Aligns preflight with independently queried scalar occurrences, without widening capability. */
final class MetalScalarAdmissionTest {
    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "BFLOAT16", "FLOAT16"})
    void advertisedScalarReluMultiplyAndClampPassPreflight(DataType type) {
        for (boolean grad : List.of(false, true)) {
            assertAll(type + " grad=" + grad,
                    () -> check(type, grad, MetalMpsGraphProgram.NodeKind.RELU, Shape.scalar()),
                    () -> check(type, grad, MetalMpsGraphProgram.NodeKind.SCALAR_MUL, Shape.scalar()),
                    () -> check(type, grad, MetalMpsGraphProgram.NodeKind.CLAMP, Shape.scalar()));
        }
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "BFLOAT16", "FLOAT16"})
    void positiveRanksAndExistingShapeTypeAndOperationRejectionsRemain(DataType type) {
        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.RELU,
                MetalMpsGraphProgram.NodeKind.SCALAR_MUL, MetalMpsGraphProgram.NodeKind.CLAMP)) {
            check(type, false, kind, Shape.of(2, 3));
            check(type, true, kind, Shape.of(1));
            var scalar = value(type, Shape.scalar(), false);
            var program = new MetalMpsGraphProgram(List.of(node(type, kind)));
            assertThrows(IllegalArgumentException.class, () -> validate(
                    List.of(scalar, value(type, Shape.of(1), false)), program, route(type, kind)));
            assertThrows(IllegalArgumentException.class, () -> validate(
                    List.of(scalar, value(type == DataType.FLOAT16 ? DataType.BFLOAT16
                            : DataType.FLOAT16, Shape.scalar(), false)), program, route(type, kind)));
            if (type != DataType.FLOAT32) {
                assertThrows(IllegalArgumentException.class, () -> validate(
                        List.of(scalar, scalar), program, MetalPreparedRoute.MPSGRAPH));
            }
        }
        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.FLOOR,
                MetalMpsGraphProgram.NodeKind.CEIL, MetalMpsGraphProgram.NodeKind.SIGN,
                MetalMpsGraphProgram.NodeKind.EXP, MetalMpsGraphProgram.NodeKind.SIGMOID,
                MetalMpsGraphProgram.NodeKind.SCALAR_ADD, MetalMpsGraphProgram.NodeKind.SCALAR_SUB,
                MetalMpsGraphProgram.NodeKind.SCALAR_DIV, MetalMpsGraphProgram.NodeKind.RECIPROCAL)) {
            var scalar = value(type, Shape.scalar(), false);
            var program = new MetalMpsGraphProgram(List.of(node(type, kind)));
            assertThrows(IllegalArgumentException.class, () -> validate(
                    List.of(scalar, scalar), program, MetalPreparedRoute.CUSTOM_PROGRAM), kind.name());
        }
    }

    private static void check(DataType type, boolean grad,
            MetalMpsGraphProgram.NodeKind kind, Shape shape) {
        ScalarValue two = scalar(type, 2.0f);
        Operation operation = switch (kind) {
            case RELU -> new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE);
            case SCALAR_MUL -> new Operation(ScalarElementwiseKind.MUL, new ScalarValueAttrs(two));
            case CLAMP -> new Operation(ScalarElementwiseKind.CLAMP,
                    new ClampRangeAttrs(scalar(type, -2.0f), two));
            default -> throw new AssertionError(kind);
        };
        var descriptor = new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
        assertTrue(new MetalCapabilityProvider().supports(new OperationCapabilityQuery(
                operation, List.of(descriptor), List.of(descriptor))), type + "/" + kind);
        var value = value(type, shape, grad);
        assertDoesNotThrow(() -> validate(List.of(value, value),
                new MetalMpsGraphProgram(List.of(node(type, kind))), route(type, kind)), kind.name());
    }

    private static MetalMpsGraphProgram.Node node(DataType type, MetalMpsGraphProgram.NodeKind kind) {
        if (kind == MetalMpsGraphProgram.NodeKind.CLAMP) {
            return MetalMpsGraphProgram.Node.clamp(0, 1, type,
                    bits(type, -2.0f), bits(type, 2.0f));
        }
        if (kind == MetalMpsGraphProgram.NodeKind.SCALAR_MUL
                || kind == MetalMpsGraphProgram.NodeKind.SCALAR_ADD
                || kind == MetalMpsGraphProgram.NodeKind.SCALAR_SUB
                || kind == MetalMpsGraphProgram.NodeKind.SCALAR_DIV) {
            return MetalMpsGraphProgram.Node.scalarValue(kind, 0, 1, type,
                    bits(type, 2.0f));
        }
        return MetalMpsGraphProgram.Node.generic(kind, new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static ScalarValue scalar(DataType type, float number) {
        return switch (type) {
            case FLOAT32 -> ScalarValue.float32(number);
            case BFLOAT16 -> ScalarValue.bfloat16(number);
            case FLOAT16 -> ScalarValue.float16(number);
            default -> throw new AssertionError(type);
        };
    }

    private static long bits(DataType type, float number) {
        ScalarValue value = scalar(type, number);
        return switch (type) {
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(value.float32Value()));
            case BFLOAT16 -> Short.toUnsignedLong(value.bfloat16Bits());
            case FLOAT16 -> Short.toUnsignedLong(value.float16Bits());
            default -> throw new AssertionError(type);
        };
    }

    private static MetalPreparedRoute route(DataType type, MetalMpsGraphProgram.NodeKind kind) {
        return type == DataType.FLOAT32 && kind == MetalMpsGraphProgram.NodeKind.SCALAR_MUL
                ? MetalPreparedRoute.MPSGRAPH : MetalPreparedRoute.CUSTOM_PROGRAM;
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(DataType type, Shape shape, boolean grad) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape.toLongArray(), grad);
    }

    private static void validate(List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalMpsGraphProgram program, MetalPreparedRoute route) {
        MetalNativeApi.ProgramExecutableAbi.validateCreate(values, program,
                new int[] {0}, new int[] {1}, route);
    }
}
