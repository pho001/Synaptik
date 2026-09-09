package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Proves independently constructed unsupported occurrences remain outside CPU ownership. */
public class CpuGeneratedCoverageUnsupportedBoundaryTest {
    @Test void providerRejectsMixedPointwiseOperands() {
        var fixture = rejectedFixtures().getFirst();
        assertFalse(new CpuCapabilityProvider().supports(fixture.query()), fixture.id());
    }

    @Test void providerRejectsBooleanArithmeticAndMismatchedResultShape() {
        CpuCapabilityProvider provider = new CpuCapabilityProvider();
        for (var fixture : rejectedFixtures().subList(1, 3)) assertFalse(provider.supports(fixture.query()), fixture.id());
        Shape shape = Shape.of(2, 3); TensorDescriptor f32 = descriptor(DataType.FLOAT32, shape);
        assertTrue(provider.supports(new OperationCapabilityQuery(
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(f32, f32), List.of(f32))), "control occurrence must remain supported");
    }

    /**
     * Keeps CPU 0009C's four adjacent provider rejections distinct from generated coverage.
     *
     * <p>These are the exact ordinary inventory boundaries: a non-integral GATHER index role,
     * a non-BOOL ONE_HOT result role, BOOL SCATTER_ADD, and BFLOAT16 DROPOUT. They deliberately
     * remain outside {@link #rejectedFixtures()}, whose independent generic rejections are used
     * by the cross-category checkpoint accounting.</p>
     */
    @Test void providerRejectsTheFourExactCpu0009cAdjacentBoundaries() {
        CpuCapabilityProvider provider = new CpuCapabilityProvider();
        TensorDescriptor f32_2x3 = descriptor(DataType.FLOAT32, Shape.of(2, 3));
        TensorDescriptor int32_2 = descriptor(DataType.INT32, Shape.of(2));
        TensorDescriptor bool_2 = descriptor(DataType.BOOL, Shape.of(2));
        TensorDescriptor state = descriptor(DataType.INT64, Shape.of(2));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                List.of(f32_2x3, descriptor(DataType.FLOAT32, Shape.of(2))),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 2))))), "INVALID_INDEX_ROLE");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)), List.of(int32_2),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3))))), "INVALID_OUTPUT_ROLE");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0)),
                List.of(bool_2, int32_2, bool_2), List.of(bool_2))), "BOOL_INAPPLICABLE");
        TensorDescriptor bf16_2x3 = descriptor(DataType.BFLOAT16, Shape.of(2, 3));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                new Operation(DropoutKind.DROPOUT, new DropoutAttrs(.2d)),
                List.of(bf16_2x3, state), List.of(bf16_2x3,
                        descriptor(DataType.BOOL, Shape.of(2, 3)), state))),
                "BFLOAT16_DROPOUT_INAPPLICABLE");
    }

    /** Exact independently-owned provider rejections, exposed to the inventory checkpoint. */
    public static List<RejectedFixture> rejectedFixtures() {
        Shape shape = Shape.of(2, 3);
        TensorDescriptor bool = descriptor(DataType.BOOL, shape);
        TensorDescriptor f32 = descriptor(DataType.FLOAT32, shape);
        TensorDescriptor f64 = descriptor(DataType.FLOAT64, shape);
        TensorDescriptor wrongShape = descriptor(DataType.FLOAT32, Shape.of(3, 2));
        Operation operation = new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        return List.of(new RejectedFixture("mixed-pointwise-operands", operation, List.of(f32, f64), List.of(f64),
                        "MIXED_NUMERIC_INPUTS"),
                new RejectedFixture("boolean-arithmetic", operation, List.of(bool, bool), List.of(bool),
                        "INVALID_BOOLEAN_NUMERIC_ROLE"),
                new RejectedFixture("mismatched-result-shape", operation, List.of(f32, f32), List.of(wrongShape),
                        "SHAPE_INCOMPATIBLE"));
    }

    public record RejectedFixture(String id, Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs, String reason) {
        public OperationCapabilityQuery query() { return new OperationCapabilityQuery(operation, inputs, outputs); }
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}
