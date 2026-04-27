package backend.cpu.fused.optimize;

import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedNodePlan;
import tensor.Tensor;

import java.util.List;

public class FusedSignatureBuilder {
    public static String buildSchedulerSignature(List<Tensor> cluster, int precisionMode) {
        StringBuilder sb = new StringBuilder(128);
        sb.append("fused:pm=").append(precisionMode).append('|');
        if (cluster != null) {
            for (Tensor t : cluster) {
                if (t == null || t.getOperation() == null) {
                    continue;
                }
                sb.append(t.getOperation().opType()).append(',');
            }
        }
        return sb.toString();
    }

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
