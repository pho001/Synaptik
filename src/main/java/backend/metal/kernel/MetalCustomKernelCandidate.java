package backend.metal.kernel;

import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.metal.exec.MetalRouteReasonCode;
import backend.metal.lowering.MetalPartitionPlan;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Prepare-time eligibility contract for the scoped custom Metal kernel route.
 *
 * <p>The first custom route is intentionally narrow: a single dense-buffer-bound
 * FLOAT32 RELU DAG. Runtime layout/currentness validation still belongs to the
 * buffer transport plan; this class only prevents route selection for unsupported
 * lowered DAG families before native execution exists.</p>
 */
public record MetalCustomKernelCandidate(
        boolean supported,
        String kernelId,
        List<String> primitiveIds,
        MetalRouteReasonCode reasonCode,
        String reason
) {
    public static final String RELU_F32_KERNEL_ID = "relu_f32";

    public MetalCustomKernelCandidate {
        kernelId = kernelId == null ? "" : kernelId;
        primitiveIds = List.copyOf(primitiveIds == null ? List.of() : primitiveIds);
        reasonCode = Objects.requireNonNullElse(reasonCode, MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE);
        reason = reason == null ? "" : reason;
    }

    /**
     * Classifies the current custom-kernel candidate subset.
     */
    public static MetalCustomKernelCandidate evaluate(MetalPartitionPlan plan) {
        List<String> primitiveIds = loweredPrimitiveIds(plan);
        if (plan == null || plan.lowering() == null || plan.lowering().dagSpec() == null) {
            return unsupported(
                    primitiveIds,
                    MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE,
                    "custom Metal kernel candidate unavailable: missing lowered DAG"
            );
        }
        var dag = plan.lowering().dagSpec();
        if (dag.nodes().size() != 1) {
            return unsupported(
                    primitiveIds,
                    MetalRouteReasonCode.UNSUPPORTED_OPERATION_FAMILY,
                    "custom Metal kernel candidate requires a single lowered DAG node"
            );
        }
        if (dag.externalInputs().size() != 1) {
            return unsupported(
                    primitiveIds,
                    MetalRouteReasonCode.UNSUPPORTED_OPERATION_FAMILY,
                    "custom Metal kernel candidate requires exactly one external input"
            );
        }
        if (dag.outputNodeIndices().size() != 1 || dag.outputNodeIds().size() != 1) {
            return unsupported(
                    primitiveIds,
                    MetalRouteReasonCode.UNSUPPORTED_OPERATION_FAMILY,
                    "custom Metal kernel candidate requires exactly one output"
            );
        }
        int outputIndex = dag.outputNodeIndices().getFirst();
        if (outputIndex != 0) {
            return unsupported(
                    primitiveIds,
                    MetalRouteReasonCode.UNSUPPORTED_OPERATION_FAMILY,
                    "custom Metal kernel candidate output must be the single lowered DAG node"
            );
        }
        AcceleratorDagNode node = dag.nodes().getFirst();
        if (node.type() != AcceleratorDagNodeType.RELU) {
            return unsupported(
                    primitiveIds,
                    MetalRouteReasonCode.UNSUPPORTED_OPERATION_FAMILY,
                    "custom Metal kernel candidate supports RELU only; got " + node.type()
            );
        }
        AcceleratorDagInput input = dag.externalInputs().getFirst();
        if (input.dataType() != DataType.FLOAT32 || node.outputDataType() != DataType.FLOAT32) {
            return unsupported(
                    primitiveIds,
                    MetalRouteReasonCode.UNSUPPORTED_DTYPE,
                    "custom Metal kernel candidate supports FLOAT32 input/output only"
            );
        }
        return new MetalCustomKernelCandidate(
                true,
                RELU_F32_KERNEL_ID,
                primitiveIds,
                MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED,
                "custom Metal kernel candidate supported: dense FLOAT32 RELU"
        );
    }

    private static MetalCustomKernelCandidate unsupported(
            List<String> primitiveIds,
            MetalRouteReasonCode reasonCode,
            String reason
    ) {
        return new MetalCustomKernelCandidate(false, "", primitiveIds, reasonCode, reason);
    }

    private static List<String> loweredPrimitiveIds(MetalPartitionPlan plan) {
        if (plan == null || plan.manifest() == null) {
            return List.of();
        }
        return plan.manifest().loweredPrimitives().stream()
                .map(primitive -> primitive.primitiveId())
                .filter(id -> id != null && !id.isBlank())
                .toList();
    }
}
