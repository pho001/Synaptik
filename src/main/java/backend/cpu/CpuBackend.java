package backend.cpu;

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
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class CpuBackend {
    public void execute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext executionContext
    ) {
        Operation op = metadata.executionOperation() != null ? metadata.executionOperation() : node.operation();
        if (op == null) {
            return;
        }
        Tensor runtimeTensor = executionContext.runtimeTensorForNodeId(node.id());

        CpuKernel kernel = metadata.cpuKernel();
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CPU kernel for opType=" + op.opType() +
                    " (operation class: " + op.getClass().getName() + ")"
            );
        }
        var cpuWorkspace = executionContext.cpuWorkspaceForNodeId(node.id());
        if (cpuWorkspace != null) {
            cpuWorkspace.clearFloatContinuation();
        }

        CpuNodeExecutionPlan executionPlan = metadata.cpuPlan();
        if (executionPlan == null) {
            throw new IllegalStateException("Missing CpuNodeExecutionPlan for node " + node.label());
        }

        List<Integer> effectiveInputNodeIds = metadata.executionInputNodeIds().isEmpty()
                ? node.inputIds()
                : metadata.executionInputNodeIds();
        List<Tensor> originalInputs = resolveRuntimeInputs(effectiveInputNodeIds, executionContext);
        List<Tensor> inputs = executionPlan.apply(node.id(), originalInputs, executionContext);
        List<CompiledNodeExecutionMetadata> inputMetadatas = resolveInputMetadatas(effectiveInputNodeIds, originalInputs, inputs, executionContext);
        if (executionPlan.stridedPath()) {
            CpuStridedElementWise.forward(op, inputs, runtimeTensor, new CpuKernelContext(node.id(), effectiveInputNodeIds, executionPlan, executionContext, metadata, inputMetadatas));
            return;
        }

        CpuKernelContext kernelContext = new CpuKernelContext(node.id(), effectiveInputNodeIds, executionPlan, executionContext, metadata, inputMetadatas);

        switch (node.dataType()) {
            case FLOAT64 -> kernel.forwardF64(op, inputs, runtimeTensor, kernelContext);
            case FLOAT32 -> kernel.forwardF32(op, inputs, runtimeTensor, kernelContext);
            case BFLOAT16 -> kernel.forwardBF16(op, inputs, runtimeTensor, kernelContext);
            case INT32 -> kernel.forwardI32(op, inputs, runtimeTensor, kernelContext);
            case BOOL -> kernel.forwardBOOL(op, inputs, runtimeTensor, kernelContext);
        }

        if (node.dataType() != DataType.FLOAT64) {
            runtimeTensor.markDataViewStale();
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
            List<Integer> inputNodeIds,
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
            if (runtime != original || original == null || i >= inputNodeIds.size()) {
                out.add(null);
                continue;
            }
            out.add(executionContext.metadataForNodeId(inputNodeIds.get(i)));
        }
        return out;
    }

    private static List<Tensor> resolveRuntimeInputs(
            List<Integer> inputNodeIds,
            ExecutionContext executionContext
    ) {
        if (inputNodeIds.isEmpty()) {
            return List.of();
        }
        List<Tensor> out = new ArrayList<>(inputNodeIds.size());
        for (int inputNodeId : inputNodeIds) {
            out.add(executionContext.runtimeTensorForNodeId(inputNodeId));
        }
        return out;
    }
}
