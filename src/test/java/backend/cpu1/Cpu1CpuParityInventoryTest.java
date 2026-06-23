package backend.cpu1;

import backend.cpu.kernels.CpuKernelRegistry;
import operations.Operation.OpType;
import org.junit.jupiter.api.Test;

import java.util.Collection;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static java.util.Map.entry;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class Cpu1CpuParityInventoryTest {
    private static final Map<OpType, String> OLD_CPU_DIRECT_KERNELS = map(
            entry(OpType.ADD, "elementwise.binary.CpuAddKernel"),
            entry(OpType.SUB, "elementwise.binary.CpuSubKernel"),
            entry(OpType.MUL, "elementwise.binary.CpuMulKernel"),
            entry(OpType.DIV, "elementwise.binary.CpuDivKernel"),
            entry(OpType.MIN, "elementwise.binary.CpuMinKernel"),
            entry(OpType.MAX, "elementwise.binary.CpuMaxKernel"),
            entry(OpType.GT, "elementwise.compare.CpuGreaterThanKernel"),
            entry(OpType.GE, "elementwise.compare.CpuGreaterOrEqualKernel"),
            entry(OpType.LT, "elementwise.compare.CpuLessThanKernel"),
            entry(OpType.LE, "elementwise.compare.CpuLessOrEqualKernel"),
            entry(OpType.EQ, "elementwise.compare.CpuEqualToKernel"),
            entry(OpType.NE, "elementwise.compare.CpuNotEqualToKernel"),
            entry(OpType.LOGICAL_AND, "elementwise.logical.CpuLogicalAndKernel"),
            entry(OpType.LOGICAL_OR, "elementwise.logical.CpuLogicalOrKernel"),
            entry(OpType.LOGICAL_NOT, "elementwise.logical.CpuLogicalNotKernel"),
            entry(OpType.GATHER, "index.CpuGatherKernel"),
            entry(OpType.GATHER_AXIS, "index.CpuGatherAxisKernel"),
            entry(OpType.GATHER_ND, "index.CpuGatherNdKernel"),
            entry(OpType.TAKE_ALONG_AXIS, "index.CpuTakeAlongAxisKernel"),
            entry(OpType.SCATTER_ADD, "index.CpuScatterAddKernel"),
            entry(OpType.SCATTER_AXIS_ADD, "index.CpuScatterAxisAddKernel"),
            entry(OpType.SCATTER_ELEMENTS, "index.CpuScatterElementsKernel"),
            entry(OpType.SCATTER_ND, "index.CpuScatterNdKernel"),
            entry(OpType.SCALED_DOT_PRODUCT_ATTENTION, "linalg.CpuScaledDotProductAttentionKernel"),
            entry(OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS, "linalg.CpuScaledDotProductAttentionWeightsKernel"),
            entry(OpType.LINEAR, "linalg.CpuLinearKernel"),
            entry(OpType.CONV2D, "nn.CpuConv2dKernel"),
            entry(OpType.MAX_POOL2D, "nn.CpuMaxPool2dKernel"),
            entry(OpType.AVG_POOL2D, "nn.CpuAvgPool2dKernel"),
            entry(OpType.LAYER_NORM, "nn.CpuLayerNormKernel"),
            entry(OpType.RMS_NORM, "nn.CpuRmsNormKernel"),
            entry(OpType.REDUCE_MIN, "reduction.CpuReduceMinKernel"),
            entry(OpType.REDUCE_MAX, "reduction.CpuReduceMaxKernel"),
            entry(OpType.REDUCE_PROD, "reduction.CpuReduceProdKernel"),
            entry(OpType.CUMSUM, "reduction.CpuCumSumKernel"),
            entry(OpType.ARGMAX, "reduction.CpuArgMaxKernel"),
            entry(OpType.REDUCE_ALL, "reduction.CpuReduceAllKernel"),
            entry(OpType.REDUCE_ANY, "reduction.CpuReduceAnyKernel"),
            entry(OpType.SOFTMAX, "reduction.CpuSoftmaxKernel"),
            entry(OpType.LOG_SOFTMAX, "reduction.CpuLogSoftmaxKernel"),
            entry(OpType.NLL_LOSS, "reduction.CpuNllLossKernel"),
            entry(OpType.CROSS_ENTROPY_LOSS, "reduction.CpuCrossEntropyLossKernel"),
            entry(OpType.CROSS_ENTROPY_LOSS_INDICES, "reduction.CpuCrossEntropyLossIndicesKernel"),
            entry(OpType.MATMUL, "linalg.CpuMatMulKernel"),
            entry(OpType.NEG, "elementwise.unary.CpuNegKernel"),
            entry(OpType.INV, "elementwise.unary.CpuInvKernel"),
            entry(OpType.LOG, "elementwise.unary.CpuLogKernel"),
            entry(OpType.EXP, "elementwise.unary.CpuExpKernel"),
            entry(OpType.FAST_EXP, "elementwise.unary.CpuFastExpKernel"),
            entry(OpType.ERF, "elementwise.unary.CpuErfKernel"),
            entry(OpType.TANH, "elementwise.unary.CpuTanhKernel"),
            entry(OpType.FAST_TANH, "elementwise.unary.CpuFastTanhKernel"),
            entry(OpType.POW, "elementwise.unary.CpuPowKernel"),
            entry(OpType.POW_TENSOR, "elementwise.binary.CpuPowTensorKernel"),
            entry(OpType.SQRT, "elementwise.unary.CpuSqrtKernel"),
            entry(OpType.ABS, "elementwise.unary.CpuAbsKernel"),
            entry(OpType.FLOOR, "elementwise.unary.CpuFloorKernel"),
            entry(OpType.CEIL, "elementwise.unary.CpuCeilKernel"),
            entry(OpType.SIGN, "elementwise.unary.CpuSignKernel"),
            entry(OpType.MUL_SCALAR, "elementwise.unary.CpuMulScalarKernel"),
            entry(OpType.SUM, "reduction.CpuSumKernel"),
            entry(OpType.MEAN, "reduction.CpuMeanKernel"),
            entry(OpType.RELU, "elementwise.unary.CpuReluKernel"),
            entry(OpType.CLAMP_MIN, "elementwise.unary.CpuClampMinKernel"),
            entry(OpType.CLAMP_MAX, "elementwise.unary.CpuClampMaxKernel"),
            entry(OpType.SIGMOID, "elementwise.unary.CpuSigmoidKernel"),
            entry(OpType.WHERE, "elementwise.where.CpuWhereKernel"),
            entry(OpType.CONTIGUOUS, "layout.CpuContiguousKernel"),
            entry(OpType.RESHAPE, "layout.CpuReshapeLikeKernel"),
            entry(OpType.EXPAND, "layout.CpuExpandKernel"),
            entry(OpType.SELECT, "layout.CpuAliasViewKernel"),
            entry(OpType.SLICE, "layout.CpuAliasViewKernel"),
            entry(OpType.EXPAND_DIMS, "layout.CpuAliasViewKernel"),
            entry(OpType.SQUEEZE, "layout.CpuAliasViewKernel"),
            entry(OpType.SLICE_BACKWARD, "layout.CpuSliceBackwardKernel"),
            entry(OpType.CONCAT, "layout.CpuConcatKernel"),
            entry(OpType.PAD, "layout.CpuPadKernel"),
            entry(OpType.TILE, "layout.CpuTileKernel"),
            entry(OpType.UNFOLD_AXIS, "layout.CpuUnfoldAxisKernel"),
            entry(OpType.UNFOLD2D, "layout.CpuUnfold2dKernel"),
            entry(OpType.FOLD2D, "layout.CpuFold2dKernel"),
            entry(OpType.CAST, "layout.CpuCastKernel"),
            entry(OpType.PERMUTE, "layout.CpuPermuteKernel"),
            entry(OpType.NOOP, "layout.CpuNoopKernel"),
            entry(OpType.FUSED, "fused.CpuFusedKernel")
    );

    private static final Map<OpType, String> CPU1_KNOWN_PREPARED_FAMILY_ROUTES = map(
            entry(OpType.ADD, "elementwise"),
            entry(OpType.SUB, "elementwise"),
            entry(OpType.MUL, "elementwise"),
            entry(OpType.DIV, "elementwise"),
            entry(OpType.MIN, "elementwise"),
            entry(OpType.MAX, "elementwise"),
            entry(OpType.POW_TENSOR, "elementwise"),
            entry(OpType.GT, "elementwise"),
            entry(OpType.GE, "elementwise"),
            entry(OpType.LT, "elementwise"),
            entry(OpType.LE, "elementwise"),
            entry(OpType.EQ, "elementwise"),
            entry(OpType.NE, "elementwise"),
            entry(OpType.LOGICAL_AND, "elementwise"),
            entry(OpType.LOGICAL_OR, "elementwise"),
            entry(OpType.LOGICAL_NOT, "elementwise"),
            entry(OpType.WHERE, "elementwise"),
            entry(OpType.GATHER_AXIS, "index"),
            entry(OpType.GATHER_ND, "index"),
            entry(OpType.TAKE_ALONG_AXIS, "index"),
            entry(OpType.SCATTER_ADD, "index"),
            entry(OpType.SCATTER_AXIS_ADD, "index"),
            entry(OpType.SCATTER_ELEMENTS, "index"),
            entry(OpType.SCATTER_ND, "index"),
            entry(OpType.NEG, "elementwise"),
            entry(OpType.INV, "elementwise"),
            entry(OpType.LOG, "elementwise"),
            entry(OpType.EXP, "elementwise"),
            entry(OpType.FAST_EXP, "elementwise"),
            entry(OpType.ERF, "elementwise"),
            entry(OpType.TANH, "elementwise"),
            entry(OpType.FAST_TANH, "elementwise"),
            entry(OpType.POW, "elementwise"),
            entry(OpType.SQRT, "elementwise"),
            entry(OpType.ABS, "elementwise"),
            entry(OpType.FLOOR, "elementwise"),
            entry(OpType.CEIL, "elementwise"),
            entry(OpType.SIGN, "elementwise"),
            entry(OpType.MUL_SCALAR, "elementwise"),
            entry(OpType.RELU, "elementwise"),
            entry(OpType.CLAMP_MIN, "elementwise"),
            entry(OpType.CLAMP_MAX, "elementwise"),
            entry(OpType.SIGMOID, "elementwise"),
            entry(OpType.SUM, "reduction"),
            entry(OpType.MEAN, "reduction"),
            entry(OpType.REDUCE_MIN, "reduction"),
            entry(OpType.REDUCE_MAX, "reduction"),
            entry(OpType.REDUCE_PROD, "reduction"),
            entry(OpType.REDUCE_ALL, "reduction"),
            entry(OpType.REDUCE_ANY, "reduction"),
            entry(OpType.ARGMAX, "reduction"),
            entry(OpType.CUMSUM, "reduction"),
            entry(OpType.SOFTMAX, "reduction"),
            entry(OpType.LOG_SOFTMAX, "reduction"),
            entry(OpType.GATHER, "index"),
            entry(OpType.MATMUL, "matmul"),
            entry(OpType.LINEAR, "matmul"),
            entry(OpType.SCALED_DOT_PRODUCT_ATTENTION, "attention"),
            entry(OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS, "attention"),
            entry(OpType.NOOP, "layout"),
            entry(OpType.RESHAPE, "layout"),
            entry(OpType.EXPAND, "layout"),
            entry(OpType.SELECT, "layout"),
            entry(OpType.SLICE, "layout"),
            entry(OpType.PERMUTE, "layout"),
            entry(OpType.EXPAND_DIMS, "layout"),
            entry(OpType.SQUEEZE, "layout"),
            entry(OpType.CONTIGUOUS, "layout"),
            entry(OpType.CONCAT, "layout"),
            entry(OpType.PAD, "layout"),
            entry(OpType.TILE, "layout"),
            entry(OpType.UNFOLD_AXIS, "layout"),
            entry(OpType.UNFOLD2D, "layout"),
            entry(OpType.FOLD2D, "layout"),
            entry(OpType.SLICE_BACKWARD, "layout"),
            entry(OpType.CAST, "dtype"),
            entry(OpType.NLL_LOSS, "loss"),
            entry(OpType.CROSS_ENTROPY_LOSS, "loss"),
            entry(OpType.CROSS_ENTROPY_LOSS_INDICES, "loss"),
            entry(OpType.LAYER_NORM, "normalization"),
            entry(OpType.RMS_NORM, "normalization"),
            entry(OpType.MAX_POOL2D, "pool2d"),
            entry(OpType.AVG_POOL2D, "pool2d"),
            entry(OpType.CONV2D, "conv2d")
    );

    private static final Map<OpType, String> LEGACY_BACKWARD_OR_SPECIAL_WITHOUT_OLD_CPU_DIRECT_KERNEL = map(
            entry(OpType.MIN_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.MAX_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.REDUCE_MIN_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.REDUCE_MAX_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.SOFTMAX_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.LOG_SOFTMAX_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.GATHER_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.GATHER_AXIS_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.GATHER_ND_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.TAKE_ALONG_AXIS_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, "legacy backward op; old CPU registry intentionally has no direct kernel"),
            entry(OpType.CONST_SCALAR, "internal fused-plan scalar, not a standalone old CPU kernel"),
            entry(OpType.UNKNOWN, "invalid sentinel, no parity expectation")
    );

    private static final Map<OpType, String> INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT = map(
            entry(OpType.FUSED, "old CPU fused op is represented in cpu1 by lowered fused elementwise prepared units")
    );

    private static final Map<OpType, String> KNOWN_MISSING_CPU1_PARITY_FROM_OLD_CPU_DIRECT = map();

    @Test
    void oldCpuDirectKernelInventoryMatchesRegistry() {
        for (OpType opType : OpType.values()) {
            boolean registryHasDirectKernel = oldCpuRegistryResolves(opType);
            if (registryHasDirectKernel) {
                assertTrue(
                        OLD_CPU_DIRECT_KERNELS.containsKey(opType),
                        () -> "Old CPU registry resolves " + opType
                                + " but OLD_CPU_DIRECT_KERNELS omits it. Add it with the concrete kernel."
                );
            } else {
                assertTrue(
                        LEGACY_BACKWARD_OR_SPECIAL_WITHOUT_OLD_CPU_DIRECT_KERNEL.containsKey(opType),
                        () -> opType + " has no old CPU direct kernel and is not explicitly classified."
                );
            }
        }
    }

    @Test
    void oldCpuDirectOpsHaveExplicitCpu1ParityStatus() {
        EnumSet<OpType> classified = EnumSet.noneOf(OpType.class);
        classified.addAll(CPU1_KNOWN_PREPARED_FAMILY_ROUTES.keySet());
        classified.addAll(KNOWN_MISSING_CPU1_PARITY_FROM_OLD_CPU_DIRECT.keySet());
        classified.addAll(INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT.keySet());

        EnumSet<OpType> unclassifiedOldCpuDirect = copyOf(OLD_CPU_DIRECT_KERNELS.keySet());
        unclassifiedOldCpuDirect.removeAll(classified);

        assertTrue(
                unclassifiedOldCpuDirect.isEmpty(),
                () -> "Old CPU direct ops missing explicit cpu1 parity status: "
                        + format(unclassifiedOldCpuDirect)
                        + "\nKnown cpu1 routes: " + format(CPU1_KNOWN_PREPARED_FAMILY_ROUTES.keySet())
                        + "\nKnown missing cpu1 parity: " + format(KNOWN_MISSING_CPU1_PARITY_FROM_OLD_CPU_DIRECT.keySet())
                        + "\nIntentionally graph-lowered/not-direct: " + format(INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT.keySet())
        );
    }

    @Test
    void parityBucketsAreDisjointAndScoped() {
        assertNoOverlap(
                "old CPU direct kernels",
                OLD_CPU_DIRECT_KERNELS.keySet(),
                "legacy/special ops without old CPU direct kernels",
                LEGACY_BACKWARD_OR_SPECIAL_WITHOUT_OLD_CPU_DIRECT_KERNEL.keySet()
        );
        assertNoOverlap(
                "cpu1 known prepared family routes",
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.keySet(),
                "known missing cpu1 parity",
                KNOWN_MISSING_CPU1_PARITY_FROM_OLD_CPU_DIRECT.keySet()
        );
        assertNoOverlap(
                "cpu1 known prepared family routes",
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.keySet(),
                "intentionally graph-lowered/not-direct",
                INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT.keySet()
        );
        assertNoOverlap(
                "known missing cpu1 parity",
                KNOWN_MISSING_CPU1_PARITY_FROM_OLD_CPU_DIRECT.keySet(),
                "intentionally graph-lowered/not-direct",
                INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT.keySet()
        );
        assertSubset(
                "cpu1 known prepared family routes",
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.keySet(),
                "old CPU direct kernels",
                OLD_CPU_DIRECT_KERNELS.keySet()
        );
        assertSubset(
                "known missing cpu1 parity",
                KNOWN_MISSING_CPU1_PARITY_FROM_OLD_CPU_DIRECT.keySet(),
                "old CPU direct kernels",
                OLD_CPU_DIRECT_KERNELS.keySet()
        );
        assertSubset(
                "intentionally graph-lowered/not-direct",
                INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT.keySet(),
                "old CPU direct kernels",
                OLD_CPU_DIRECT_KERNELS.keySet()
        );
    }

    @Test
    void requiredSentinelOpsArePresent() {
        assertTrue(
                OLD_CPU_DIRECT_KERNELS.containsKey(OpType.CROSS_ENTROPY_LOSS_INDICES),
                "old CPU direct inventory must include CROSS_ENTROPY_LOSS_INDICES"
        );
        assertTrue(
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.MATMUL),
                "cpu1 inventory must include the MATMUL prepared family route"
        );
        assertTrue(
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.LINEAR),
                "cpu1 inventory must include the LINEAR prepared matmul route"
        );
        assertTrue(
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.SUM)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.REDUCE_PROD)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.ARGMAX)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.LOG_SOFTMAX),
                "cpu1 inventory must include representative reduction routes"
        );
        assertTrue(
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.GATHER),
                "cpu1 inventory must include the first index route"
        );
        assertTrue(
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.CONTIGUOUS)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.SLICE_BACKWARD)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.CAST),
                "cpu1 inventory must include layout and dtype routes, including SLICE_BACKWARD and CAST"
        );
        assertTrue(
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.LAYER_NORM)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.RMS_NORM),
                "cpu1 inventory must include normalization routes"
        );
        assertTrue(
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.MAX_POOL2D)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.AVG_POOL2D),
                "cpu1 inventory must include pool2d routes"
        );
        assertTrue(
                CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.ADD)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.WHERE)
                        && CPU1_KNOWN_PREPARED_FAMILY_ROUTES.containsKey(OpType.LOGICAL_AND),
                "cpu1 inventory must include representative elementwise routes"
        );
        assertTrue(
                INTENTIONALLY_GRAPH_LOWERED_OR_NOT_DIRECT.containsKey(OpType.FUSED),
                "FUSED must remain explicitly classified as intentionally graph-lowered/not-direct"
        );
    }

    private static boolean oldCpuRegistryResolves(OpType opType) {
        try {
            CpuKernelRegistry.resolve(opType);
            return true;
        } catch (IllegalStateException ex) {
            return false;
        }
    }

    private static void assertNoOverlap(String leftName, Set<OpType> left, String rightName, Set<OpType> right) {
        EnumSet<OpType> overlap = copyOf(left);
        overlap.retainAll(right);
        assertTrue(
                overlap.isEmpty(),
                () -> leftName + " overlaps " + rightName + ": " + format(overlap)
        );
    }

    private static void assertSubset(String subsetName, Set<OpType> subset, String supersetName, Set<OpType> superset) {
        EnumSet<OpType> missing = copyOf(subset);
        missing.removeAll(superset);
        assertTrue(
                missing.isEmpty(),
                () -> subsetName + " contains ops outside " + supersetName + ": " + format(missing)
        );
    }

    private static EnumSet<OpType> copyOf(Collection<OpType> opTypes) {
        EnumSet<OpType> copy = EnumSet.noneOf(OpType.class);
        copy.addAll(opTypes);
        return copy;
    }

    private static String format(Collection<OpType> opTypes) {
        List<OpType> sorted = opTypes.stream()
                .sorted(Comparator.comparingInt(Enum::ordinal))
                .toList();
        return sorted.toString();
    }

    @SafeVarargs
    private static Map<OpType, String> map(Map.Entry<OpType, String>... entries) {
        EnumMap<OpType, String> map = new EnumMap<>(OpType.class);
        for (Map.Entry<OpType, String> entry : entries) {
            String previous = map.put(entry.getKey(), entry.getValue());
            if (previous != null) {
                throw new IllegalArgumentException("Duplicate inventory entry for " + entry.getKey());
            }
        }
        return Map.copyOf(map);
    }
}
