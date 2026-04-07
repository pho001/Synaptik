package backend.kernels.cpu;

import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.fused.PreparedFusedExecutable;
import operations.FusedOperation;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuFusedKernel implements CpuKernel {
    @Override
    public CpuKernelCostClass costClass(Operation op) {
        if (op instanceof FusedOperation fused) {
            return fused.isLowCostHint() && fused.getDispatchScale() == 1
                    ? CpuKernelCostClass.LOW
                    : CpuKernelCostClass.MEDIUM;
        }
        return CpuKernel.super.costClass(op);
    }

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!(op instanceof FusedOperation fused)) {
            throw new IllegalStateException("CpuFusedKernel requires FusedOperation descriptor");
        }

        PreparedFusedExecutable executable = context.fusedExecutable();
        if (executable == null) {
            throw new IllegalStateException("Missing prepared fused executable in prepared metadata");
        }

        ResolvedDispatchHints hints = requireDispatchHints(context);
        FusedExecutionOptions options = new FusedExecutionOptions(
                context.useFastExpApprox(),
                context.useFastTanhApprox()
        );
        int length = node.getFlatDataSize();
        CpuExecutionMode mode = hints.mode();
        CpuKernelCostClass costClass = fused.isLowCostHint() && fused.getDispatchScale() == 1
                ? CpuKernelCostClass.LOW
                : CpuKernelCostClass.MEDIUM;
        String schedulerKey = fused.getSchedulerSignature();
        boolean recommendVector = hints.vectorWidth() > 1;
        long t0 = FusedExecutionProfiler.enabled() ? System.nanoTime() : 0L;
        switch (mode) {
            case SCALAR -> {
                executable.applyRangeScalar(inputs, node, context, 0, length, options);
                recordProfile(fused, mode, length, 1, false, false, t0);
            }
            case VECTOR -> {
                if (recommendVector) {
                    executable.applyRangeVector(inputs, node, context, 0, length, options);
                    recordProfile(fused, mode, length, 1, false, true, t0);
                } else {
                    executable.applyRangeScalar(inputs, node, context, 0, length, options);
                    recordProfile(fused, mode, length, 1, false, false, t0);
                }
            }
            case PARALLEL -> runParallel(
                    executable,
                    inputs,
                    node,
                    context,
                    hints,
                    options,
                    false,
                    fused,
                    mode,
                    costClass
            );
            case PARALLEL_VECTOR -> runParallel(
                    executable,
                    inputs,
                    node,
                    context,
                    hints,
                    options,
                    recommendVector,
                    fused,
                    mode,
                    costClass
            );
        }
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) { forwardF64(op, inputs, node, context); }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) { forwardF64(op, inputs, node, context); }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardF64(op, inputs, node, context);
    }

    private static void runParallel(
            PreparedFusedExecutable executable,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            ResolvedDispatchHints hints,
            FusedExecutionOptions options,
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
                executable.applyRangeVector(inputs, node, context, start, end, options);
            } else {
                executable.applyRangeScalar(inputs, node, context, start, end, options);
            }
        }, useCommonPool);
        long elapsed = System.nanoTime() - t0;
        if (FusedExecutionProfiler.enabled()) {
            FusedExecutionProfiler.recordRun(
                    fused.getSchedulerSignature(),
                    mode,
                    length,
                    chunks,
                    useCommonPool,
                    preferVector,
                    elapsed
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
