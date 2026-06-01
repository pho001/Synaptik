package backend.cpu1.kernels.matmul;

import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.runtime.ExecutionContext;

/**
 * Prepared cpu1 matmul runner.
 */
public interface Cpu1MatmulKernel {
    void run(Cpu1PreparedMatmulUnit unit, ExecutionContext context);
}
