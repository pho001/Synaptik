package backend.cpu.kernels.elementwise;

import backend.cpu.plan.CpuExecutionMode;
import backend.cpu.execution.CpuThreadPool;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;

public final class ElementwiseRangeLoop {
    private ElementwiseRangeLoop() {}

    public static void runScalar(int length, ResolvedDispatchHints hints, RangeBody body) {
        if (hints == null) {
            hints = new ResolvedDispatchHints(length, CpuExecutionMode.SCALAR, length, length, 1, 1, false);
        }
        if (hints.parallel()) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, length);
                body.run(start, end);
            });
            return;
        }
        body.run(0, length);
    }

    public static void run(int length, ResolvedDispatchHints hints, boolean vectorized, RangeBody scalar, RangeBody vector) {
        if (hints == null) {
            hints = new ResolvedDispatchHints(length, CpuExecutionMode.SCALAR, length, length, 1, 1, false);
        }
        boolean useVector = vectorized && hints.vectorized();
        if (hints.parallel()) {
            int chunkSize = useVector ? hints.vectorChunkSize() : hints.scalarChunkSize();
            int chunks = (length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, length);
                if (useVector) {
                    vector.run(start, end);
                } else {
                    scalar.run(start, end);
                }
            });
            return;
        }
        if (useVector) {
            vector.run(0, length);
        } else {
            scalar.run(0, length);
        }
    }

    @FunctionalInterface
    public interface RangeBody {
        void run(int start, int end);
    }
}
