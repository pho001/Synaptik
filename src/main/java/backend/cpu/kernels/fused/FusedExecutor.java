package backend.cpu.kernels.fused;

import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCostClass;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.fused.exec.PreparedFusedExecutable;
import backend.cpu.fused.plan.FusedOperation;
import tensor.Tensor;

import java.util.List;

final class FusedExecutor {
    private FusedExecutor() {
    }

    static CpuKernelCostClass costClass(FusedOperation fused) {
        return fused != null && fused.isLowCostHint() && fused.getDispatchScale() == 1
                ? CpuKernelCostClass.LOW
                : CpuKernelCostClass.MEDIUM;
    }

    static void execute(FusedOperation fused, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (fused == null) {
            throw new IllegalStateException("FusedExecutor requires FusedOperation descriptor");
        }

        PreparedFusedExecutable executable = context.fusedExecutable();
        if (executable == null) {
            throw new IllegalStateException("Missing prepared fused executable in prepared metadata");
        }

        ResolvedDispatchHints hints = requireDispatchHints(context);
        int length = node.getFlatDataSize();
        CpuExecutionMode mode = hints.mode();
        CpuKernelCostClass costClass = costClass(fused);
        boolean recommendVector = hints.vectorWidth() > 1;
        long t0 = FusedExecutionProfiler.enabled() ? System.nanoTime() : 0L;
        switch (mode) {
            case SCALAR -> {
                executable.applyRangeScalar(inputs, node, context, 0, length);
                recordProfile(fused, mode, length, 1, false, false, t0);
            }
            case VECTOR -> {
                if (recommendVector) {
                    executable.applyRangeVector(inputs, node, context, 0, length);
                    recordProfile(fused, mode, length, 1, false, true, t0);
                } else {
                    executable.applyRangeScalar(inputs, node, context, 0, length);
                    recordProfile(fused, mode, length, 1, false, false, t0);
                }
            }
            case PARALLEL -> runParallel(executable, inputs, node, context, hints, false, fused, mode, costClass);
            case PARALLEL_VECTOR -> runParallel(executable, inputs, node, context, hints, recommendVector, fused, mode, costClass);
        }
    }

    private static void runParallel(
            PreparedFusedExecutable executable,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            ResolvedDispatchHints hints,
            boolean preferVector,
            FusedOperation fused,
            CpuExecutionMode mode,
            CpuKernelCostClass costClass
    ) {
        int length = node.getFlatDataSize();
        int chunkSize = preferVector ? hints.vectorChunkSize() : hints.scalarChunkSize();
        int chunks = (length + chunkSize - 1) / chunkSize;
        boolean useCommonPool = hints.useCommonPool() && costClass == CpuKernelCostClass.LOW;
        long t0 = System.nanoTime();
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, length);
            if (preferVector) {
                executable.applyRangeVector(inputs, node, context, start, end);
            } else {
                executable.applyRangeScalar(inputs, node, context, start, end);
            }
        }, useCommonPool);
        if (FusedExecutionProfiler.enabled()) {
            FusedExecutionProfiler.recordRun(
                    fused.getSchedulerSignature(),
                    mode,
                    length,
                    chunks,
                    useCommonPool,
                    preferVector,
                    System.nanoTime() - t0
            );
        }
    }

    private static void recordProfile(
            FusedOperation fused,
            CpuExecutionMode mode,
            int length,
            int chunks,
            boolean useCommonPool,
            boolean preferVector,
            long startedNs
    ) {
        if (!FusedExecutionProfiler.enabled()) {
            return;
        }
        FusedExecutionProfiler.recordRun(
                fused.getSchedulerSignature(),
                mode,
                length,
                chunks,
                useCommonPool,
                preferVector,
                System.nanoTime() - startedNs
        );
    }

    private static ResolvedDispatchHints requireDispatchHints(CpuKernelContext context) {
        ResolvedDispatchHints hints = context.dispatchHints();
        if (hints == null) {
            throw new IllegalStateException("Missing ResolvedDispatchHints for fused execution");
        }
        return hints;
    }
}
