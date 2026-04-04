package benchmark.scenario;

import backend.runtime.ExecutionMode;
import graph.execution.PreparedExecution;
import tensor.Tensor;

import java.util.Objects;

public record PreparedHotPathScenario(
        String name,
        Tensor root,
        PreparedExecution execution,
        ExecutionMode mode
) {
    public PreparedHotPathScenario {
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(root, "root cannot be null");
        Objects.requireNonNull(execution, "execution cannot be null");
        Objects.requireNonNull(mode, "mode cannot be null");
    }

    public void run() {
        execution.execute(mode);
    }

    public double sink() {
        return root.scalarAsDouble();
    }
}
