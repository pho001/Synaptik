package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringCoverageEntry;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.accelerator.lowering.GpuLoweringCoverageStatus;
import backend.metal.MetalMpsCapabilities;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.linalg.scaledDotProductAttention;

/**
 * Shared Metal partition planner predicates.
 */
public final class MetalPartitionSupport {
    private MetalPartitionSupport() {
    }

    /**
     * Returns whether a compiled node is currently supported by Metal graph lowering.
     *
     * @param node compiled node to test
     * @param context planning context; reserved for capability checks that need graph context
     * @return true when the node operation and output dtype can be represented by the Metal bridge
     */
    public static boolean isPlannerSupported(CompiledNode node, PartitionPlanningContext context) {
        return plannerUnsupportedReason(node, context).isBlank();
    }

    /**
     * Returns a stable diagnostic reason when a node is not currently legal for Metal planning.
     *
     * <p>This method is intentionally capability-oriented. It explains tested Metal planner coverage; it does not
     * estimate profitability and it does not inspect runtime tensor storage layout.</p>
     *
     * @param node compiled node to test
     * @param context planning context; reserved for capability checks that need graph context
     * @return empty string when supported, otherwise a readable rejection reason
     */
    public static String plannerUnsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        if (node == null) {
            return "node is null";
        }
        if (node.operation() == null) {
            return "node has no operation";
        }
        if (node.inputIds().isEmpty()) {
            return "leaf nodes are external inputs, not Metal compute nodes";
        }
        if (!MetalMpsCapabilities.supportsComputeDType(node.dataType())
                || !MetalMpsCapabilities.supportsOutputDType(node.dataType())) {
            return MetalMpsCapabilities.unsupportedDTypeMessage(node.dataType());
        }
        Operation.OpType opType = node.operation().opType();
        if (!node.backwardNode() && opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            if (node.operation() instanceof scaledDotProductAttention attention && attention.hasMask()) {
                return "direct masked SDPA disabled until bool-mask semantics are verified against MPSGraph floating masks";
            }
            return "direct forward SDPA disabled until native MPSGraph scale contract matches CPU semantics";
        }
        if (node.backwardNode() && opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            return "forward SDPA nodes are not legal inside Metal backward regions";
        }
        GpuLoweringCoverageEntry entry = GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, opType);
        if (entry.status() != GpuLoweringCoverageStatus.SUPPORTED) {
            return compoundPatternPrefix(opType) + GpuLoweringCoverageMatrix.plannerUnsupportedDetail(ComputeBackend.GPU_METAL, opType);
        }
        return "";
    }

    /**
     * Returns whether an external producer can feed a Metal consumer at a specific input index.
     *
     * @param producer producer outside the selected candidate
     * @param consumer consumer inside the selected candidate
     * @param inputIndex input position on the consumer
     * @return true when the producer dtype is legal for that role
     */
    public static boolean isExternalInputSupported(CompiledNode producer, CompiledNode consumer, int inputIndex) {
        return MetalMpsCapabilities.supportsExternalInputRole(producer, consumer, inputIndex);
    }

    /**
     * Returns whether a node belongs to the matmul or linear operation family.
     */
    public static boolean containsMatMulFamily(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return false;
        }
        Operation.OpType opType = node.operation().opType();
        return opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR;
    }

    private static String compoundPatternPrefix(Operation.OpType opType) {
        return switch (opType) {
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, LAYER_NORM, RMS_NORM -> "REDUCTION_ADJACENT: ";
            default -> "";
        };
    }
}
