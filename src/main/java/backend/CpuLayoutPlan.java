package backend;

import backend.runtime.ExecutionContext;
import backend.kernels.cpu.layout.plan.ResolvedBroadcastPlan;
import backend.kernels.cpu.layout.plan.ResolvedWhereBroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorRemap;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public record CpuLayoutPlan(
        boolean stridedPath,
        DataType targetType,
        int materializeThreshold,
        ResolvedBroadcastPlan broadcastPlan,
        ResolvedWhereBroadcastPlan whereBroadcastPlan,
        List<CpuPreparedInput> preparedInputs,
        List<Tensor> runtimeInputs
) {
    public CpuLayoutPlan {
        Objects.requireNonNull(targetType, "targetType cannot be null");
        materializeThreshold = Math.max(0, materializeThreshold);
        preparedInputs = List.copyOf(preparedInputs == null ? List.of() : preparedInputs);
        runtimeInputs = List.copyOf(runtimeInputs == null ? List.of() : runtimeInputs);
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
