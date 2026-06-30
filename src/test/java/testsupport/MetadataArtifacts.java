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
import runtime.execution.PreparedStepMetadata;
import runtime.execution.PreparedStepExecutable;
import runtime.execution.InputResidencyRequirement;
import runtime.execution.OutputResidencyEffect;

import java.util.List;

public final class MetadataArtifacts {
    private MetadataArtifacts() {
    }

    public static CpuKernel cpuKernel(PreparedStepMetadata metadata) {
        if (metadata.executable() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        if (metadata.executable() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuKernel();
        }
        return null;
    }

    public static CpuNodeExecutionPlan cpuPlan(PreparedStepMetadata metadata) {
        if (metadata.executable() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        if (metadata.executable() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        return null;
    }

    public static PreparedFusedExecutable fusedExecutable(PreparedStepMetadata metadata) {
        return metadata.executable() instanceof CpuFusedExecutionArtifact artifact
                ? artifact.fusedExecutable()
                : null;
    }

    public static CpuNodeWorkspace cpuWorkspace(PreparedStepMetadata metadata) {
        if (metadata.executable() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuWorkspace();
        }
        if (metadata.executable() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuWorkspace();
        }
        return null;
    }

    public static PreparedAcceleratorExecutable acceleratorExecutable(PreparedStepMetadata metadata) {
        return metadata.executable() instanceof AcceleratorExecutionArtifact artifact
                ? artifact.executable()
                : null;
    }

    public static PreparedStepMetadata cpuMetadata(CpuNodeExecutionPlan cpuPlan) {
        return new PreparedStepMetadata(
                ComputeBackend.CPU,
                null,
                List.of(),
                new CpuNodeExecutionArtifact(null, cpuPlan, null),
                InputResidencyRequirement.cpuReadableAll(),
                OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }

    public static PreparedStepMetadata acceleratorMetadata(
            ComputeBackend backend,
            PreparedAcceleratorExecutable executable
    ) {
        return new PreparedStepMetadata(
                backend,
                null,
                List.of(),
                new AcceleratorExecutionArtifact(executable),
                InputResidencyRequirement.none(),
                OutputResidencyEffect.cpuCurrentIfUnset("accelerator test output")
        );
    }

    public static PreparedStepMetadata metadata(ComputeBackend backend) {
        return new PreparedStepMetadata(
                backend,
                null,
                List.of(),
                noopExecutable(),
                backend == ComputeBackend.CPU
                        ? InputResidencyRequirement.cpuReadableAll()
                        : InputResidencyRequirement.none(),
                backend == ComputeBackend.CPU
                        ? OutputResidencyEffect.cpuCurrentPreserveNative()
                        : OutputResidencyEffect.cpuCurrentIfUnset("test backend output")
        );
    }

    public static PreparedStepExecutable noopExecutable() {
        return (node, metadata, context) -> {
        };
    }
}
