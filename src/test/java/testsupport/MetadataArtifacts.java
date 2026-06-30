package testsupport;

import backend.accelerator.exec.AcceleratorExecutionArtifact;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.contract.ComputeBackend;
import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu.CpuNodeExecutionArtifact;
import backend.cpu.fused.exec.PreparedFusedExecutable;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.execution.CpuNodeWorkspace;
import graph.execution.plan.CompiledNodeExecutionMetadata;

import java.util.List;

public final class MetadataArtifacts {
    private MetadataArtifacts() {
    }

    public static CpuKernel cpuKernel(CompiledNodeExecutionMetadata metadata) {
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        return null;
    }

    public static CpuNodeExecutionPlan cpuPlan(CompiledNodeExecutionMetadata metadata) {
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        return null;
    }

    public static PreparedFusedExecutable fusedExecutable(CompiledNodeExecutionMetadata metadata) {
        return metadata.artifact() instanceof CpuFusedExecutionArtifact artifact
                ? artifact.fusedExecutable()
                : null;
    }

    public static CpuNodeWorkspace cpuWorkspace(CompiledNodeExecutionMetadata metadata) {
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuWorkspace();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuWorkspace();
        }
        return null;
    }

    public static PreparedAcceleratorExecutable acceleratorExecutable(CompiledNodeExecutionMetadata metadata) {
        return metadata.artifact() instanceof AcceleratorExecutionArtifact artifact
                ? artifact.executable()
                : null;
    }

    public static CompiledNodeExecutionMetadata cpuMetadata(CpuNodeExecutionPlan cpuPlan) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                List.of(),
                new CpuNodeExecutionArtifact(null, cpuPlan, null)
        );
    }

    public static CompiledNodeExecutionMetadata acceleratorMetadata(
            ComputeBackend backend,
            PreparedAcceleratorExecutable executable
    ) {
        return new CompiledNodeExecutionMetadata(
                backend,
                null,
                List.of(),
                new AcceleratorExecutionArtifact(executable)
        );
    }

    public static CompiledNodeExecutionMetadata metadata(ComputeBackend backend) {
        return new CompiledNodeExecutionMetadata(backend, null, List.of(), null);
    }
}
