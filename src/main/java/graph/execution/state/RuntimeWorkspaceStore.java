package graph.execution.state;

import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu.CpuNodeExecutionArtifact;
import backend.cpu.CpuRegionExecutionArtifact;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.CpuNodeWorkspace;
import backend.cpu.plan.CpuPreparedInput;
import graph.execution.PreparedExecutionStep;
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
            CpuNodeWorkspace workspace = cpuWorkspace(entry.getValue());
            if (workspace != null) {
                CpuNodeWorkspace runtimeWorkspace = runtimeWorkspaceByTemplate.computeIfAbsent(workspace, ignored -> workspace.fork());
                workspaces.put(entry.getKey(), runtimeWorkspace);
            }
            allocatePreparedInputs(entry.getKey(), cpuPlan(entry.getValue()), preparedInputs);
            if (entry.getValue().artifact() instanceof AcceleratorExecutionArtifact artifact && artifact.executable() != null) {
                for (var fallbackStep : artifact.executable().cpuFallbackSteps()) {
                    allocatePreparedInputs(fallbackStep.node().id(), cpuPlan(fallbackStep.metadata()), preparedInputs);
                }
            }
            if (entry.getValue().artifact() instanceof CpuRegionExecutionArtifact artifact && artifact.executable() != null) {
                for (PreparedExecutionStep regionStep : artifact.executable().nativeSteps()) {
                    allocateCpuRegionStepRuntimeState(regionStep, runtimeWorkspaceByTemplate, workspaces, preparedInputs);
                }
                for (PreparedExecutionStep regionStep : artifact.executable().fallbackSteps()) {
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
            PreparedExecutionStep step,
            Map<CpuNodeWorkspace, CpuNodeWorkspace> runtimeWorkspaceByTemplate,
            Map<Integer, CpuNodeWorkspace> workspaces,
            Map<Long, Tensor> preparedInputs
    ) {
        if (step == null || step.metadata() == null) {
            return;
        }
        CpuNodeWorkspace workspace = cpuWorkspace(step.metadata());
        if (workspace != null) {
            CpuNodeWorkspace runtimeWorkspace = runtimeWorkspaceByTemplate.computeIfAbsent(workspace, ignored -> workspace.fork());
            workspaces.put(step.compiledNode().id(), runtimeWorkspace);
        }
        allocatePreparedInputs(step.compiledNode().id(), cpuPlan(step.metadata()), preparedInputs);
    }

    private static CpuNodeWorkspace cpuWorkspace(CompiledNodeExecutionMetadata metadata) {
        if (metadata == null) {
            return null;
        }
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuWorkspace();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuWorkspace();
        }
        return null;
    }

    private static CpuNodeExecutionPlan cpuPlan(CompiledNodeExecutionMetadata metadata) {
        if (metadata == null) {
            return null;
        }
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        return null;
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
