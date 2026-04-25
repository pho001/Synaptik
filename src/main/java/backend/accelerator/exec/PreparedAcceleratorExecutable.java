package backend.accelerator.exec;

import backend.ComputeBackend;
import backend.runtime.ExecutionContext;

public interface PreparedAcceleratorExecutable {
    ComputeBackend backend();

    void execute(ExecutionContext context);
}
