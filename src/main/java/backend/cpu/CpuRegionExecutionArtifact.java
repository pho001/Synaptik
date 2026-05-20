package backend.cpu;

import backend.cpu.region.PreparedCpuRegionExecutable;
import graph.execution.plan.PreparedExecutionArtifact;

public record CpuRegionExecutionArtifact(
        PreparedCpuRegionExecutable executable
) implements PreparedExecutionArtifact {
}
