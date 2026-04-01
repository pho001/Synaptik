package backend.kernels.cpu;

import graph.codegen.FusedVectorOps;
import backend.kernels.cpu.fused.CompiledFusedKernel;
import backend.kernels.cpu.fused.FusedExecutionOptions;
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

        CompiledFusedKernel ranged = context.fusedKernel();
        if (ranged == null) {
            throw new IllegalStateException("Missing compiled fused kernel in prepared metadata");
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
        boolean recommendVector = FusedVectorOps.isRecommended(fused.getPrecisionMode());
        long t0 = FusedExecutionProfiler.enabled() ? System.nanoTime() : 0L;
        switch (mode) {
            case SCALAR -> {
                ranged.applyRangeScalar(inputs, node, 0, length, options);
                recordProfile(fused, mode, length, 1, false, false, t0);
            }
            case VECTOR -> {
                if (recommendVector) {
                    ranged.applyRangeVector(inputs, node, 0, length, options);
                    recordProfile(fused, mode, length, 1, false, true, t0);
                } else {
                    ranged.applyRangeScalar(inputs, node, 0, length, options);
                    recordProfile(fused, mode, length, 1, false, false, t0);
                }
            }
            case PARALLEL -> runParallel(
                    ranged,
                    inputs,
                    node,
                    hints,
                    context.planner().lowCostNsPerElementThreshold(),
                    options,
                    false,
                    fused,
                    mode,
                    costClass,
                    schedulerKey
            );
            case PARALLEL_VECTOR -> runParallel(
                    ranged,
                    inputs,
                    node,
                    hints,
                    context.planner().lowCostNsPerElementThreshold(),
                    options,
                    recommendVector,
                    fused,
                    mode,
                    costClass,
                    schedulerKey
            );
        }
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardF64(op, inputs, node, context);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardF64(op, inputs, node, context);
    }

    private static void runParallel(
            CompiledFusedKernel ranged,
            List<Tensor> inputs,
            Tensor node,
            ResolvedDispatchHints hints,
            double lowCostNsPerElementThreshold,
            FusedExecutionOptions options,
            boolean preferVector,
            FusedOperation fused,
            CpuExecutionMode mode,
            CpuKernelCostClass costClass,
            String schedulerKey
    ) {
        int length = node.getFlatDataSize();
        int chunkSize = preferVector ? hints.vectorChunkSize() : hints.scalarChunkSize();
        int chunks = (length + chunkSize - 1) / chunkSize;
        boolean useCommonPool = CpuSchedulerAdvisor.shouldUseCommonPool(
                costClass,
                schedulerKey,
                length,
                lowCostNsPerElementThreshold
        );
        long t0 = System.nanoTime();
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, length);
            if (preferVector) {
                ranged.applyRangeVector(inputs, node, start, end, options);
            } else {
                ranged.applyRangeScalar(inputs, node, start, end, options);
            }
        }, useCommonPool);
        long elapsed = System.nanoTime() - t0;
        CpuSchedulerAdvisor.recordSample(schedulerKey, length, elapsed);
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
