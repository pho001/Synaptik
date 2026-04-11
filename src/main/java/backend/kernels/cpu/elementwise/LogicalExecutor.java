package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedBroadcastPlan;
import backend.kernels.cpu.ResolvedDispatchHints;
import tensor.Tensor;

import java.util.List;

public final class LogicalExecutor {
    private LogicalExecutor() {}

    public static void executeBinary(LogicalBinaryOp op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        byte[] left = inputs.get(0).getBoolData();
        byte[] right = inputs.get(1).getBoolData();
        byte[] out = node.getBoolData();
        ResolvedBroadcastPlan plan = context.broadcastPlan();
        ResolvedDispatchHints hints = context.dispatchHints();
        if (plan != null && !plan.isNoBroadcast()) {
            broadcastBinary(op, left, right, out, plan, hints);
            return;
        }
        directBinary(op, left, right, out, hints);
    }

    public static void executeUnary(LogicalUnaryOp op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        byte[] in = inputs.get(0).getBoolData();
        byte[] out = node.getBoolData();
        ResolvedDispatchHints hints = context.dispatchHints();
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarUnary(op, in, out, start, end);
            });
            return;
        }
        scalarUnary(op, in, out, 0, out.length);
    }

    private static void directBinary(LogicalBinaryOp op, byte[] left, byte[] right, byte[] out, ResolvedDispatchHints hints) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarBinary(op, left, right, out, start, end);
            });
            return;
        }
        scalarBinary(op, left, right, out, 0, out.length);
    }

    private static void broadcastBinary(LogicalBinaryOp op, byte[] left, byte[] right, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarBroadcastBinary(op, left, right, out, plan, start, end);
            });
            return;
        }
        scalarBroadcastBinary(op, left, right, out, plan, 0, out.length);
    }

    private static void scalarBinary(LogicalBinaryOp op, byte[] left, byte[] right, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.apply(left[i], right[i]);
        }
    }

    private static void scalarUnary(LogicalUnaryOp op, byte[] in, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.apply(in[i]);
        }
    }

    private static void scalarBroadcastBinary(LogicalBinaryOp op, byte[] left, byte[] right, byte[] out, ResolvedBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] leftEff = plan.aEffStrides();
        int[] rightEff = plan.bEffStrides();
        int[] leftResets = plan.aResets();
        int[] rightResets = plan.bResets();
        int rank = outStrides.length;
        int[] coords = ElementwiseBinaryExecutor.initCoords(start, outStrides, rank);
        int leftIdx = ElementwiseBinaryExecutor.initialIndex(coords, leftEff);
        int rightIdx = ElementwiseBinaryExecutor.initialIndex(coords, rightEff);
        for (int i = start; i < end; i++) {
            out[i] = op.apply(left[leftIdx], right[rightIdx]);
            if (i + 1 < end) {
                int[] next = ElementwiseBinaryExecutor.nextIndices(coords, outShape, leftEff, rightEff, leftResets, rightResets, rank, leftIdx, rightIdx);
                leftIdx = next[0];
                rightIdx = next[1];
            }
        }
    }
}
