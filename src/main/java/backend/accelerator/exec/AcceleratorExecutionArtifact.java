package backend.accelerator.exec;

import graph.execution.plan.PreparedExecutionArtifact;

public record AcceleratorExecutionArtifact(
        PreparedAcceleratorExecutable executable
) implements PreparedExecutionArtifact {
}
