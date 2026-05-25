package backend.cpu.kernels;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.execution.CpuNodeWorkspace;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.storage.CpuStorageBindings;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public record CpuKernelCall(
        Operation operation,
        List<Tensor> inputTensors,
        Tensor outputTensor,
        List<CpuStorageView> inputs,
        CpuStorageView output,
        CpuNodeExecutionPlan plan,
        CpuKernelContext context,
        CpuNodeWorkspace workspace
) {
    public CpuKernelCall {
        Objects.requireNonNull(operation, "operation cannot be null");
        inputTensors = List.copyOf(inputTensors == null ? List.of() : inputTensors);
        Objects.requireNonNull(outputTensor, "outputTensor cannot be null");
        inputs = List.copyOf(inputs == null ? List.of() : inputs);
        for (CpuStorageView input : inputs) {
            Objects.requireNonNull(input, "inputs cannot contain null views");
        }
        Objects.requireNonNull(output, "output cannot be null");
        Objects.requireNonNull(plan, "plan cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
    }

    public CpuKernelCall(
            Operation operation,
            List<Tensor> inputTensors,
            Tensor outputTensor,
            CpuStorageBindings storage,
            CpuNodeExecutionPlan plan,
            CpuKernelContext context,
            CpuNodeWorkspace workspace
    ) {
        this(
                operation,
                inputTensors,
                outputTensor,
                Objects.requireNonNull(storage, "storage cannot be null").inputs(),
                storage.output(),
                plan,
                context,
                workspace
        );
    }
}
