package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuExecutionConfig;

public final class AddExecutor {
    public void execute(double[] a, double[] b, double[] out, CpuExecutionMode mode, CpuExecutionConfig config, int precisionMode) {
        if (a.length != out.length || b.length != out.length) {
            throw new IllegalArgumentException("Input and output lengths must match for add");
        }

        switch (mode) {
            case VECTOR -> AddLoops.vector(a, b, out, precisionMode);
            case PARALLEL -> AddLoops.parallel(a, b, out, config, precisionMode);
            case PARALLEL_VECTOR -> AddLoops.parallelVector(a, b, out, config, precisionMode);
            case SCALAR -> AddLoops.scalar(a, b, out, precisionMode);
        }
    }
}
