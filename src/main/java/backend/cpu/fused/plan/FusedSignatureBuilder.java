package backend.cpu.fused.plan;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedNumericContract;

/**
 * Internal signature builder for fused scheduler and generated-kernel cache keys.
 */
public final class FusedSignatureBuilder {
    private FusedSignatureBuilder() {
    }

    /**
     * Builds a stable signature from the lowered fused expression plan.
     */
    public static String buildFromPlan(FusedExpressionPlan plan, FusedNumericContract numericContract) {
        return buildFromPlan(plan, numericContract, FusedApproximationContract.STRICT);
    }

    /**
     * Builds a stable signature from the lowered fused expression plan and prepared execution contracts.
     */
    public static String buildFromPlan(
            FusedExpressionPlan plan,
            FusedNumericContract numericContract,
            FusedApproximationContract approximationContract
    ) {
        StringBuilder sb = new StringBuilder(128);
        sb.append("fused:numeric=").append(numericContract.signatureToken()).append('|');
        sb.append("approx=").append(approximationContract.signatureToken()).append('|');
        sb.append("inputs=");
        for (backend.cpu.fused.ir.FusedExternalInputPlan input : plan.inputs()) {
            sb.append(input.dataType()).append('@')
                    .append(input.accessKind()).append('@')
                    .append(input.storageOffset()).append('@')
                    .append(java.util.Arrays.toString(input.effectiveStrides()))
                    .append(';');
        }
        sb.append('|');
        for (FusedNodePlan node : plan.nodes()) {
            sb.append(node.opType()).append(':').append(node.outputType()).append(',');
            if (!(node.attributes() instanceof backend.cpu.fused.ir.NoAttributes)) {
                sb.append('(').append(node.attributes()).append(')');
            }
            sb.append(';');
        }
        sb.append("out=").append(plan.outputRef());
        return sb.toString();
    }
}
