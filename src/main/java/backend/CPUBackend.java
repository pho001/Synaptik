package backend;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.elementwise.strided.CpuStridedElementWise;
import backend.kernels.cpu.plan.CpuExecutionPlanner;
import backend.kernels.cpu.plan.CpuPlanAssembler;
import backend.runtime.ExecutionContext;
import config.runtime.BlasConfig;
import config.runtime.Conv2dConfig;
import graph.execution.CompiledNodeExecutionMetadata;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class CPUBackend {
    public void execute(
            Tensor node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext executionContext
    ) {
        Operation op = node.getOperation();
        if (op == null) {
            return;
        }

        CpuKernel kernel = metadata.cpuKernel();
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CPU kernel for opType=" + op.opType() +
                            " (operation class: " + op.getClass().getName() + ")"
            );
        }
        if (metadata.cpuWorkspace() != null) {
            metadata.cpuWorkspace().clearFloatContinuation();
        }

        CpuNodeExecutionPlan executionPlan = metadata.cpuPlan();
        if (executionPlan == null) {
            throw new IllegalStateException("Missing CpuNodeExecutionPlan for node " + node.getLabel());
        }

        List<Tensor> originalInputs = node.getPrevTensors();
        List<Tensor> inputs = executionPlan.apply(originalInputs);
        List<CompiledNodeExecutionMetadata> inputMetadatas = resolveInputMetadatas(originalInputs, inputs, executionContext);
        if (executionPlan.stridedPath()) {
            CpuStridedElementWise.forward(op, inputs, node, new CpuKernelContext(executionPlan, executionContext, metadata, inputMetadatas));
            return;
        }

        CpuKernelContext kernelContext = new CpuKernelContext(executionPlan, executionContext, metadata, inputMetadatas);

        switch (node.getDataType()) {
            case FLOAT64 -> kernel.forwardF64(op, inputs, node, kernelContext);
            case FLOAT32 -> kernel.forwardF32(op, inputs, node, kernelContext);
            case BFLOAT16 -> kernel.forwardBF16(op, inputs, node, kernelContext);
            case INT32 -> kernel.forwardI32(op, inputs, node, kernelContext);
            case BOOL -> kernel.forwardBOOL(op, inputs, node, kernelContext);
        }

        if (node.getDataType() != DataType.FLOAT64) {
            node.markDataViewStale();
        }
    }

    public static CpuNodeExecutionPlan buildExecutionPlan(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            CpuExecutionPlanner planner,
            BlasConfig blasConfig,
            Conv2dConfig conv2dConfig,
            boolean publishFloatContinuation,
            ResolvedDispatchHints dispatchHintsOverride
    ) {
        return CpuPlanAssembler.buildExecutionPlan(
                op,
                inputs,
                node,
                planner,
                blasConfig,
                conv2dConfig,
                publishFloatContinuation,
                dispatchHintsOverride
        );
    }

    private static List<CompiledNodeExecutionMetadata> resolveInputMetadatas(
            List<Tensor> originalInputs,
            List<Tensor> runtimeInputs,
            ExecutionContext executionContext
    ) {
        if (runtimeInputs == null || runtimeInputs.isEmpty()) {
            return List.of();
        }
        List<CompiledNodeExecutionMetadata> out = new ArrayList<>(runtimeInputs.size());
        for (int i = 0; i < runtimeInputs.size(); i++) {
            Tensor runtime = runtimeInputs.get(i);
            Tensor original = (originalInputs != null && i < originalInputs.size()) ? originalInputs.get(i) : null;
            if (runtime != original || original == null) {
                out.add(null);
                continue;
            }
            out.add(executionContext.metadataFor(original));
        }
        return out;
    }
}
