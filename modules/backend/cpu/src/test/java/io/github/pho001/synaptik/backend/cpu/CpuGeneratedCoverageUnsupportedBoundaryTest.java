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
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Proves independently constructed unsupported occurrences remain outside CPU ownership. */
public class CpuGeneratedCoverageUnsupportedBoundaryTest {
    /**
     * Independent expected lock matrix for formerly positive source fixtures.
     * No production policy helper determines these expected answers.
     *
     * @param query exact source fixture
     * @return whether a low operand occupies a temporarily unavailable numerical role
     */
    public static boolean lockedLowComputation(OperationCapabilityQuery query) {
        if (query.inputs().stream().noneMatch(d -> d.dataType() == DataType.BFLOAT16
                || d.dataType() == DataType.FLOAT16)) return false;
        var attrs = query.operation().attrs();
        if (attrs instanceof io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs scatter)
            return scatter.reduction() == ScatterReduction.ADD || scatter.reduction() == ScatterReduction.MUL;
        if (attrs instanceof io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs scatter)
            return scatter.reduction() == ScatterReduction.ADD || scatter.reduction() == ScatterReduction.MUL;
        return java.util.Set.of("ADD", "SUB", "MUL", "DIV", "POW", "RECIPROCAL", "LOG",
                "LOG1P", "EXP", "EXPM1", "ERF", "SQRT", "RSQRT", "SIGMOID", "TANH",
                "GELU", "GELU_TANH_APPROXIMATION", "SILU", "SUM", "MEAN", "PROD",
                "LOG_SUM_EXP", "VARIANCE", "STANDARD_DEVIATION", "L1_NORM", "L2_NORM",
                "CUM_SUM", "CUM_PROD", "CUMSUM", "CUMPROD", "MATMUL", "CONV2D", "CONV3D",
                "AVERAGE_POOL2D", "AVERAGE_POOL3D", "SOFTMAX", "LOG_SOFTMAX",
                "LAYER_NORM", "AFFINE_LAYER_NORM", "RMS_NORM", "AFFINE_RMS_NORM",
                "BATCH_NORM_INFERENCE", "BATCH_NORM_TRAINING", "MEAN_SQUARED_ERROR",
                "DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS", "INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS",
                "SCALED_DOT_PRODUCT_ATTENTION", "DROPOUT", "SCATTER_ADD", "FOLD_AXIS",
                "FOLD2D", "RNN_TANH", "GRU_RESET_AFTER", "LSTM").contains(query.operation().kind().toString());
    }

    /**
     * Asserts the current outcome of a historically supported source fixture, including explicit
     * unsupported low numerical rows rather than removing them from its coverage domain.
     * @param provider actual CPU provider
     * @param query exact historically positive occurrence
     * @param detail optional fixture identity
     */
    public static void assertCurrentSupport(CpuCapabilityProvider provider,
            OperationCapabilityQuery query, String... detail) {
        org.junit.jupiter.api.Assertions.assertEquals(!lockedLowComputation(query),
                provider.supports(query), detail.length == 0 ? query.operation().toString() : detail[0]);
    }

    /**
     * Runs a numerical fixture under its current expected CPU outcome. Low rows prove an actual
     * unsupported exception; F32/F64/integral rows still execute all original numerical assertions.
     * @param type numerical operand type, not a storage-only role
     * @param computation actual source/lowering/generation/execution path under test
     */
    public static void assertNumericalOutcome(DataType type,
            org.junit.jupiter.api.function.Executable computation) {
        if (type == DataType.BFLOAT16 || type == DataType.FLOAT16)
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class, computation, type.toString());
        else org.junit.jupiter.api.Assertions.assertDoesNotThrow(computation, type.toString());
    }

    /**
     * Executes negative admission witnesses for a locked row of a mixed fixture inventory.
     * @param context exact CPU fixture context
     * @return true only after proving its explicit unsupported outcome; false for retained rows
     */
    public static boolean assertLowUnsupported(
            io.github.pho001.synaptik.prepare.analysis.PrepareContext<
                    io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs> context) {
        var values = new java.util.HashMap<io.github.pho001.synaptik.model.graph.ValueId, TensorDescriptor>();
        context.values().forEach(value -> values.put(value.id(), value.descriptor()));
        boolean locked = false;
        for (var node : context.nodes()) {
            var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(values::get).toList(),
                    node.outputs().stream().map(values::get).toList(), node.scanBody(),
                    java.util.Optional.ofNullable(context.scanActivityPlans().get(node.id())));
            if (lockedLowComputation(query)) {
                assertFalse(new CpuCapabilityProvider().supports(query), node.operation().toString());
                locked = true;
            }
        }
        if (!locked) return false;
        var preparer = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer();
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class, () -> preparer.analyze(context));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class, () -> preparer.completePlanCandidates(context));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> preparer.analyzeSelected(context, SelectedControl.PLAN));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering().lower(context));
        return true;
    }

    /** A real independently admitted control identity; selection cannot grant low legality. */
    private static final class SelectedControl {
        private static final io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer.SelectedCompletePlan PLAN =
                new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer.SelectedCompletePlan(
                        1, new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer()
                                .completePlanCandidates(io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest.context(Shape.of(2)))
                                .identities().getFirst(), List.of(1));
    }


    @Test void providerRejectsBooleanArithmeticAndMismatchedResultShape() {
        CpuCapabilityProvider provider = new CpuCapabilityProvider();
        for (var fixture : rejectedFixtures()) assertFalse(provider.supports(fixture.query()), fixture.id());
        Shape shape = Shape.of(2, 3); TensorDescriptor f32 = descriptor(DataType.FLOAT32, shape);
        io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(provider, new OperationCapabilityQuery(new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE), List.of(f32, f32), List.of(f32)), "control occurrence must remain supported");
    }

    /**
     * Keeps CPU 0009C's three adjacent provider rejections distinct from generated coverage.
     *
     * <p>These are the exact remaining ordinary inventory boundaries: a non-integral GATHER
     * index role, a non-BOOL ONE_HOT result role, and BOOL SCATTER_ADD. They deliberately remain
     * outside {@link #rejectedFixtures()}, whose independent generic rejections are used by the
     * cross-category checkpoint accounting.</p>
     */
    @Test void providerRejectsTheThreeExactCpu0009cAdjacentBoundaries() {
        CpuCapabilityProvider provider = new CpuCapabilityProvider();
        TensorDescriptor f32_2x3 = descriptor(DataType.FLOAT32, Shape.of(2, 3));
        TensorDescriptor int32_2 = descriptor(DataType.INT32, Shape.of(2));
        TensorDescriptor bool_2 = descriptor(DataType.BOOL, Shape.of(2));
        assertFalse(provider.supports(new OperationCapabilityQuery(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)), List.of(f32_2x3, descriptor(DataType.FLOAT32, Shape.of(2))), List.of(descriptor(DataType.FLOAT32, Shape.of(2, 2))))), "INVALID_INDEX_ROLE");
        assertFalse(provider.supports(new OperationCapabilityQuery(new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)), List.of(int32_2), List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3))))), "INVALID_OUTPUT_ROLE");
        assertFalse(provider.supports(new OperationCapabilityQuery(new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(0)), List.of(bool_2, int32_2, bool_2), List.of(bool_2))), "BOOL_INAPPLICABLE");
    }

    /** Exact independently-owned provider rejections, exposed to the inventory checkpoint. */
    public static List<RejectedFixture> rejectedFixtures() {
        Shape shape = Shape.of(2, 3);
        TensorDescriptor bool = descriptor(DataType.BOOL, shape);
        TensorDescriptor f32 = descriptor(DataType.FLOAT32, shape);
        TensorDescriptor wrongShape = descriptor(DataType.FLOAT32, Shape.of(3, 2));
        Operation operation = new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        return List.of(new RejectedFixture("boolean-arithmetic", operation, List.of(bool, bool), List.of(bool),
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
