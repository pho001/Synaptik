package backend.cpu;

import backend.cpu.fused.exec.PreparedFusedExecutable;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.CpuNodeWorkspace;
import graph.execution.plan.PreparedExecutionArtifact;

public record CpuFusedExecutionArtifact(
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        PreparedFusedExecutable fusedExecutable,
        CpuNodeWorkspace cpuWorkspace
) implements PreparedExecutionArtifact {
}
