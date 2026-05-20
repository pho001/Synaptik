package backend.cpu.fused.optimize;

import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedNodePlan;

/**
 * Internal signature builder for fused scheduler and generated-kernel cache keys.
 */
public class FusedSignatureBuilder {
    /**
     * Builds a stable signature from the lowered fused expression plan.
     */
    public static String buildFromPlan(FusedExpressionPlan plan, int precisionMode) {
        StringBuilder sb = new StringBuilder(128);
        sb.append("fused:pm=").append(precisionMode).append('|');
        sb.append("inputs=");
        for (backend.cpu.fused.codegen.FusedExternalInputPlan input : plan.inputs()) {
            sb.append(input.dataType()).append('@')
                    .append(input.accessKind()).append('@')
                    .append(input.storageOffset()).append('@')
                    .append(java.util.Arrays.toString(input.effectiveStrides()))
                    .append(';');
        }
        sb.append('|');
        for (FusedNodePlan node : plan.nodes()) {
            sb.append(node.opType()).append(':').append(node.outputType()).append(',');
            if (!(node.attributes() instanceof backend.cpu.fused.codegen.NoAttributes)) {
                sb.append('(').append(node.attributes()).append(')');
            }
            sb.append(';');
        }
        sb.append("out=").append(plan.outputRef());
        return sb.toString();
    }
}
