package graph.execution.state;

import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.plan.CpuPreparedInput;
import graph.execution.PreparedNodeExecution;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import tensor.Tensor;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Run-scoped CPU workspaces and prepared input buffers.
 */
final class RuntimeWorkspaceStore {
    private final Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId;
    private final Map<Long, Tensor> preparedInputTensorByKey;

    static RuntimeWorkspaceStore create(Map<Integer, CompiledNodeExecutionMetadata> metadataIndex) {
        Map<Integer, CpuNodeWorkspace> workspaces = new HashMap<>();
        Map<CpuNodeWorkspace, CpuNodeWorkspace> runtimeWorkspaceByTemplate = new IdentityHashMap<>();
        Map<Long, Tensor> preparedInputs = new HashMap<>();
        for (Map.Entry<Integer, CompiledNodeExecutionMetadata> entry : metadataIndex.entrySet()) {
            CpuNodeWorkspace workspace = entry.getValue().cpuWorkspace();
            if (workspace != null) {
                CpuNodeWorkspace runtimeWorkspace = runtimeWorkspaceByTemplate.computeIfAbsent(workspace, ignored -> workspace.fork());
                workspaces.put(entry.getKey(), runtimeWorkspace);
            }
            allocatePreparedInputs(entry.getKey(), entry.getValue().cpuPlan(), preparedInputs);
            if (entry.getValue().acceleratorExecutable() != null) {
                for (var fallbackStep : entry.getValue().acceleratorExecutable().cpuFallbackSteps()) {
                    allocatePreparedInputs(fallbackStep.node().id(), fallbackStep.metadata().cpuPlan(), preparedInputs);
                }
            }
            if (entry.getValue().cpuRegionExecutable() != null) {
                for (PreparedNodeExecution regionStep : entry.getValue().cpuRegionExecutable().nativeSteps()) {
                    allocateCpuRegionStepRuntimeState(regionStep, runtimeWorkspaceByTemplate, workspaces, preparedInputs);
                }
                for (PreparedNodeExecution regionStep : entry.getValue().cpuRegionExecutable().fallbackSteps()) {
                    allocateCpuRegionStepRuntimeState(regionStep, runtimeWorkspaceByTemplate, workspaces, preparedInputs);
                }
            }
        }
        return new RuntimeWorkspaceStore(workspaces, preparedInputs);
    }

    RuntimeWorkspaceStore(
            Map<Integer, CpuNodeWorkspace> cpuWorkspaceByNodeId,
            Map<Long, Tensor> preparedInputTensorByKey
    ) {
        this.cpuWorkspaceByNodeId = Map.copyOf(cpuWorkspaceByNodeId);
        this.preparedInputTensorByKey = Map.copyOf(preparedInputTensorByKey);
    }

    private static void allocateCpuRegionStepRuntimeState(
            PreparedNodeExecution step,
            Map<CpuNodeWorkspace, CpuNodeWorkspace> runtimeWorkspaceByTemplate,
            Map<Integer, CpuNodeWorkspace> workspaces,
            Map<Long, Tensor> preparedInputs
    ) {
        if (step == null || step.metadata() == null) {
            return;
        }
        CpuNodeWorkspace workspace = step.metadata().cpuWorkspace();
        if (workspace != null) {
            CpuNodeWorkspace runtimeWorkspace = runtimeWorkspaceByTemplate.computeIfAbsent(workspace, ignored -> workspace.fork());
            workspaces.put(step.compiledNode().id(), runtimeWorkspace);
        }
        allocatePreparedInputs(step.compiledNode().id(), step.metadata().cpuPlan(), preparedInputs);
    }

    private static void allocatePreparedInputs(
            int nodeId,
            CpuNodeExecutionPlan cpuPlan,
            Map<Long, Tensor> preparedInputs
    ) {
        if (cpuPlan == null || cpuPlan.layoutPlan().preparedInputs().isEmpty()) {
            return;
        }
        for (CpuPreparedInput preparedInput : cpuPlan.layoutPlan().preparedInputs()) {
            Tensor template = preparedInput.runtimeTensor();
            Tensor runtimePrepared = new Tensor(
                    template.getShapeUnsafe().clone(),
                    template.getStridesUnsafe().clone(),
                    template.getStorageOffsetUnsafe(),
                    null,
                    null,
                    template.getLabel(),
                    template.getDataType()
            );
            runtimePrepared.setRequiresGrad(template.getRequiresGrad());
            preparedInputs.put(preparedInputKey(nodeId, preparedInput.inputIndex()), runtimePrepared);
        }
    }

    static long preparedInputKey(int nodeId, int inputIndex) {
        return ((long) nodeId << Integer.SIZE) ^ (inputIndex & 0xffffffffL);
    }

    CpuNodeWorkspace cpuWorkspaceForNodeId(int nodeId) {
        return cpuWorkspaceByNodeId.get(nodeId);
    }

    Tensor preparedInputTensorFor(int nodeId, int inputIndex) {
        Tensor tensor = preparedInputTensorByKey.get(preparedInputKey(nodeId, inputIndex));
        if (tensor == null) {
            throw new IllegalStateException("Missing prepared runtime tensor for nodeId=" + nodeId + ", inputIndex=" + inputIndex);
        }
        return tensor;
    }
}
