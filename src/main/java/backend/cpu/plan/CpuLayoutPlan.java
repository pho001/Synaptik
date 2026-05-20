package backend.cpu.plan;

import backend.runtime.ExecutionContext;
import backend.cpu.kernels.elementwise.strided.StridedLayoutDecision;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.cpu.kernels.layout.plan.ResolvedWhereBroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.layout.TensorRemap;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public record CpuLayoutPlan(
        StridedLayoutDecision layoutDecision,
        DataType targetType,
        int materializeThreshold,
        ResolvedBroadcastPlan broadcastPlan,
        ResolvedWhereBroadcastPlan whereBroadcastPlan,
        List<CpuPreparedInput> preparedInputs
) {
    public CpuLayoutPlan {
        layoutDecision = layoutDecision == null ? StridedLayoutDecision.NONE : layoutDecision;
        Objects.requireNonNull(targetType, "targetType cannot be null");
        materializeThreshold = Math.max(0, materializeThreshold);
        preparedInputs = List.copyOf(preparedInputs == null ? List.of() : preparedInputs);
    }

    public boolean stridedPath() {
        return layoutDecision.useStridedPath();
    }

    public List<Tensor> apply(int nodeId, List<Tensor> originalInputs, ExecutionContext executionContext) {
        Objects.requireNonNull(executionContext, "executionContext cannot be null");
        if (originalInputs == null || originalInputs.isEmpty()) {
            return List.of();
        }
        if (preparedInputs.isEmpty()) {
            return originalInputs;
        }
        List<Tensor> resolvedInputs = new ArrayList<>(originalInputs);
        for (CpuPreparedInput preparedInput : preparedInputs) {
            Tensor source = originalInputs.get(preparedInput.inputIndex());
            Tensor runtimePrepared = executionContext.preparedInputTensorFor(nodeId, preparedInput.inputIndex());
            TensorRemap.applyTrusted(
                    source,
                    runtimePrepared,
                    preparedInput.remapPlan(),
                    materializeThreshold
            );
            executionContext.mirrorRuntimeState(source, runtimePrepared);
            resolvedInputs.set(preparedInput.inputIndex(), runtimePrepared);
        }
        return List.copyOf(resolvedInputs);
    }
}
