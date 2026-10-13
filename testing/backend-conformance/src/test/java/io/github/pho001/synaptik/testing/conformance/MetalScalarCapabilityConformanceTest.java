package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
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
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Independent dtype queries for the existing scalar domain; the repair adds no provider rows. */
final class MetalScalarCapabilityConformanceTest {
    private final MetalCapabilityProvider metal = new MetalCapabilityProvider();

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "BFLOAT16", "FLOAT16"})
    void scalarReluMultiplyAndClampRetainExactDescriptorAndGradientDomain(DataType type) {
        List<Operation> operations = List.of(
                new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE),
                new Operation(ScalarElementwiseKind.MUL, new ScalarValueAttrs(scalar(type, -2.0f))),
                new Operation(ScalarElementwiseKind.CLAMP,
                        new ClampRangeAttrs(scalar(type, -0.0f), scalar(type, 0.0f))));
        for (Operation operation : operations) {
            for (boolean grad : List.of(false, true)) {
                TensorDescriptor value = value(type, Shape.scalar(), grad);
                assertTrue(supports(operation, value, value), type + "/" + operation.kind());
                assertFalse(supports(operation, value, value(type, Shape.scalar(), !grad)));
                assertFalse(supports(operation, value, value(type, Shape.of(1), grad)));
                for (DataType other : DataType.values()) {
                    if (other != type) {
                        assertFalse(supports(operation, value, value(other, Shape.scalar(), false)));
                    }
                }
                TensorDescriptor noLayout = new TensorDescriptor(type, Shape.scalar(), Optional.empty(), grad);
                assertFalse(supports(operation, noLayout, value));
                assertFalse(supports(operation, value, noLayout));
            }
            for (Shape shape : List.of(Shape.of(1), Shape.of(2, 3))) {
                TensorDescriptor value = value(type, shape, false);
                assertTrue(supports(operation, value, value));
            }
            long[] rank17 = new long[17];
            Arrays.fill(rank17, 1L);
            for (Shape shape : List.of(Shape.of(0), Shape.of(rank17))) {
                TensorDescriptor invalid = value(type, shape, false);
                assertFalse(supports(operation, invalid, invalid));
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "BFLOAT16", "FLOAT16"})
    void scalarRepairDoesNotAdmitAdjacentPositiveRankOnlyOperations(DataType type) {
        TensorDescriptor value = value(type, Shape.scalar(), false);
        for (var kind : List.of(UnaryElementwiseKind.NEG, UnaryElementwiseKind.ABS,
                UnaryElementwiseKind.FLOOR, UnaryElementwiseKind.CEIL, UnaryElementwiseKind.SIGN,
                UnaryElementwiseKind.EXP, UnaryElementwiseKind.SIGMOID, UnaryElementwiseKind.RECIPROCAL)) {
            assertFalse(supports(new Operation(kind, NoOperationAttrs.INSTANCE), value, value), kind.name());
        }
        for (var kind : List.of(ScalarElementwiseKind.ADD, ScalarElementwiseKind.SUB,
                ScalarElementwiseKind.DIV, ScalarElementwiseKind.MIN, ScalarElementwiseKind.MAX,
                ScalarElementwiseKind.POW)) {
            assertFalse(supports(new Operation(kind, new ScalarValueAttrs(scalar(type, 2.0f))),
                    value, value), kind.name());
        }
        TensorDescriptor float64 = value(DataType.FLOAT64, Shape.scalar(), false);
        assertFalse(supports(new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE),
                float64, float64));
        assertFalse(supports(new Operation(ScalarElementwiseKind.MUL,
                new ScalarValueAttrs(ScalarValue.float64(2.0))), float64, float64));
        assertFalse(supports(new Operation(ScalarElementwiseKind.CLAMP,
                new ClampRangeAttrs(ScalarValue.float64(-1.0), ScalarValue.float64(1.0))),
                float64, float64));
    }

    private boolean supports(Operation operation, TensorDescriptor input, TensorDescriptor output) {
        return metal.supports(new OperationCapabilityQuery(operation, List.of(input), List.of(output)));
    }

    private static TensorDescriptor value(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    private static ScalarValue scalar(DataType type, float value) {
        return switch (type) {
            case FLOAT32 -> ScalarValue.float32(value);
            case BFLOAT16 -> ScalarValue.bfloat16(value);
            case FLOAT16 -> ScalarValue.float16(value);
            default -> throw new AssertionError(type);
        };
    }
}
