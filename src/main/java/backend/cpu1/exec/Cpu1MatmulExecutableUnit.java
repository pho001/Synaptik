package backend.cpu1.exec;

import backend.blas.OpenBlasRuntime;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernel;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import runtime.execution.ExecutionContext;

import java.util.OptionalInt;

/**
 * Runtime wrapper for a prepared cpu1 matmul node.
 */
public final class Cpu1MatmulExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedMatmulUnit preparedUnit;
    private final Cpu1MatmulKernel kernel;

    public Cpu1MatmulExecutableUnit(Cpu1PreparedMatmulUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
        this.kernel = preparedUnit.kernel();
    }

    public Cpu1PreparedMatmulUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return preparedUnit.scratchBufferSpec();
    }

    @Override
    public void run(ExecutionContext context) {
        int requestedThreads = preparedUnit.openBlasThreads();
        if (requestedThreads <= 0) {
            kernel.run(preparedUnit, context);
            return;
        }

        OptionalInt previousThreads = OpenBlasRuntime.getNumThreads();
        if (previousThreads.isEmpty()) {
            throw new IllegalStateException("cpu1 OpenBLAS thread override requires openblas_get_num_threads.");
        }
        int previous = previousThreads.getAsInt();
        if (previous == requestedThreads) {
            kernel.run(preparedUnit, context);
            return;
        }
        if (!OpenBlasRuntime.setNumThreads(requestedThreads)) {
            throw new IllegalStateException("cpu1 OpenBLAS thread override requires openblas_set_num_threads.");
        }
        try {
            kernel.run(preparedUnit, context);
        } finally {
            OpenBlasRuntime.setNumThreads(previous);
        }
    }
}
