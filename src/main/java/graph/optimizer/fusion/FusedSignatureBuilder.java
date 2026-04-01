package graph.optimizer.fusion;

import graph.codegen.FusedExpressionPlan;
import graph.codegen.FusedNodePlan;
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
        for (FusedNodePlan node : plan.nodes()) {
            sb.append(node.opType()).append(',');
            if (node.parameter() != null) {
                sb.append('(').append(node.parameter()).append(')');
            }
            sb.append(';');
        }
        sb.append("out=").append(plan.outputRef());
        return sb.toString();
    }
}
