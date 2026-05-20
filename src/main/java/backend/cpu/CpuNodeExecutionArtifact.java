package backend.cpu;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.CpuNodeWorkspace;
import graph.execution.plan.PreparedExecutionArtifact;

public record CpuNodeExecutionArtifact(
        CpuKernel cpuKernel,
        CpuNodeExecutionPlan cpuPlan,
        CpuNodeWorkspace cpuWorkspace
) implements PreparedExecutionArtifact {
}
