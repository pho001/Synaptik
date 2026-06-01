package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ExecutableUnit;
import backend.runtime.ExecutionContext;
import graph.execution.plan.PreparedExecutionArtifact;

import java.util.Objects;

/**
 * Prepared execution artifact attached to cpu1 node metadata.
 */
public final class Cpu1PreparedArtifact implements PreparedExecutionArtifact {
    private final Cpu1PreparedUnit preparedUnit;
    private final Cpu1ExecutableUnit executableUnit;

    public Cpu1PreparedArtifact(Cpu1PreparedUnit preparedUnit) {
        this.preparedUnit = Objects.requireNonNull(preparedUnit, "preparedUnit cannot be null");
        this.executableUnit = new Cpu1ExecutableUnit(preparedUnit);
    }

    public Cpu1PreparedUnit preparedUnit() {
        return preparedUnit;
    }

    public Cpu1ExecutableUnit executableUnit() {
        return executableUnit;
    }

    public void execute(ExecutionContext context) {
        executableUnit.run(context);
    }
}
