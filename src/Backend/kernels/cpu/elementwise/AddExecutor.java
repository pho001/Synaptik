package Backend.kernels.cpu.elementwise;

import Backend.kernels.cpu.CpuExecutionMode;
import Backend.kernels.cpu.CpuExecutionConfig;

public final class AddExecutor {
    public void execute(double[] a, double[] b, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        if (a.length != out.length || b.length != out.length) {
            throw new IllegalArgumentException("Input and output lengths must match for add");
        }

        switch (mode) {
            case VECTOR -> AddLoops.vector(a, b, out);
            case PARALLEL -> AddLoops.parallel(a, b, out, config);
            case PARALLEL_VECTOR -> AddLoops.parallelVector(a, b, out, config);
            case SCALAR -> AddLoops.scalar(a, b, out);
        }
    }
}
