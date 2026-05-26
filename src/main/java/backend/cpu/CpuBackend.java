package backend.cpu;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.execution.CpuKernelExecutor;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.prepare.CpuExecutionPlanner;
import backend.cpu.prepare.CpuPlanAssembler;
import backend.runtime.ExecutionContext;
import config.runtime.BlasConfig;
import config.runtime.Conv2dConfig;
import config.runtime.CpuStorageProfile;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class CpuBackend {
    private final CpuKernelExecutor kernelExecutor = new CpuKernelExecutor();

    public void execute(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext executionContext
    ) {
        if (metadata.artifact() instanceof CpuRegionExecutionArtifact regionArtifact) {
            regionArtifact.executable().execute(executionContext);
            return;
        }
        CpuKernel kernel = cpuKernel(metadata);
        CpuNodeExecutionPlan executionPlan = cpuPlan(metadata);
        Operation op = metadata.executionOperation() != null ? metadata.executionOperation() : node.operation();
        if (op == null) {
            return;
        }
        Tensor runtimeTensor = executionContext.runtimeTensorForNodeId(node.id());

        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CPU kernel for opType=" + op.opType() +
                    " (operation class: " + op.getClass().getName() + ")"
            );
        }
        if (executionPlan == null) {
            throw new IllegalStateException("Missing CpuNodeExecutionPlan for node " + node.label());
        }

        List<Integer> effectiveInputNodeIds = metadata.executionInputNodeIds().isEmpty()
                ? node.inputIds()
                : metadata.executionInputNodeIds();
        List<Tensor> originalInputs = resolveRuntimeInputs(effectiveInputNodeIds, executionContext);
        List<Tensor> inputs = executionPlan.apply(node.id(), originalInputs, executionContext);
        List<CompiledNodeExecutionMetadata> inputMetadatas = resolveInputMetadatas(effectiveInputNodeIds, originalInputs, inputs, executionContext);
        kernelExecutor.execute(
                kernel,
                op,
                inputs,
                runtimeTensor,
                node.id(),
                effectiveInputNodeIds,
                executionPlan,
                executionContext,
                metadata,
                inputMetadatas
        );

        if (node.dataType() != DataType.FLOAT64) {
        }
    }

    public static CpuNodeExecutionPlan buildExecutionPlan(
            Operation op,
            List<CompiledTensorDescriptor> inputDescriptors,
            CompiledTensorDescriptor nodeDescriptor,
            CompiledTensorDescriptorIndex descriptorIndex,
            CpuExecutionPlanner planner,
            BlasConfig blasConfig,
            Conv2dConfig conv2dConfig,
            CpuStorageProfile cpuStorageProfile,
            boolean publishFloatContinuation,
            ResolvedDispatchHints dispatchHintsOverride
    ) {
        return CpuPlanAssembler.buildExecutionPlan(
                op,
                inputDescriptors,
                nodeDescriptor,
                descriptorIndex,
                planner,
                blasConfig,
                cpuStorageProfile,
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

    private static CpuKernel cpuKernel(CompiledNodeExecutionMetadata metadata) {
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        return null;
    }

    private static CpuNodeExecutionPlan cpuPlan(CompiledNodeExecutionMetadata metadata) {
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        return null;
    }
}
