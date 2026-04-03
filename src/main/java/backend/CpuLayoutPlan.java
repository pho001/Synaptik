package backend;

import backend.kernels.cpu.ResolvedBroadcastPlan;
import backend.kernels.cpu.ResolvedWhereBroadcastPlan;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorRemap;

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

    public List<Tensor> apply(List<Tensor> originalInputs) {
        if (preparedInputs.isEmpty()) {
            return runtimeInputs.isEmpty() ? originalInputs : runtimeInputs;
        }
        if (originalInputs == null) {
            throw new IllegalArgumentException("originalInputs cannot be null when prepared inputs are present");
        }
        for (CpuPreparedInput preparedInput : preparedInputs) {
            Tensor source = originalInputs.get(preparedInput.inputIndex());
            TensorRemap.applyTrusted(
                    source,
                    preparedInput.runtimeTensor(),
                    preparedInput.remapPlan(),
                    materializeThreshold
            );
        }
        return runtimeInputs;
    }
}
