package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedBroadcastPlan;
import backend.kernels.cpu.ResolvedDispatchHints;
import tensor.Tensor;

import java.util.List;

public final class CompareExecutor {
    private CompareExecutor() {}

    public static void execute(CompareOp op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        switch (inputs.get(0).getDataType()) {
            case FLOAT64 -> runF64(op, inputs.get(0).getFloat64Data(), inputs.get(1).getFloat64Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case FLOAT32 -> runF32(op, inputs.get(0).getFloat32Data(), inputs.get(1).getFloat32Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case BFLOAT16 -> runBF16(op, inputs.get(0).getBFloat16Data(), inputs.get(1).getBFloat16Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case INT32, BOOL -> throw new IllegalArgumentException("Compare ops do not support INT32/BOOL inputs.");
        }
    }

    private static void runF64(CompareOp op, double[] left, double[] right, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan != null && !plan.isNoBroadcast()) {
            broadcastF64(op, left, right, out, plan, hints);
            return;
        }
        directF64(op, left, right, out, hints);
    }

    private static void runF32(CompareOp op, float[] left, float[] right, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan != null && !plan.isNoBroadcast()) {
            broadcastF32(op, left, right, out, plan, hints);
            return;
        }
        directF32(op, left, right, out, hints);
    }

    private static void runBF16(CompareOp op, short[] left, short[] right, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan != null && !plan.isNoBroadcast()) {
            broadcastBF16(op, left, right, out, plan, hints);
            return;
        }
        directBF16(op, left, right, out, hints);
    }

    private static void directF64(CompareOp op, double[] left, double[] right, byte[] out, ResolvedDispatchHints hints) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarDirectF64(op, left, right, out, start, end);
            });
            return;
        }
        scalarDirectF64(op, left, right, out, 0, out.length);
    }

    private static void directF32(CompareOp op, float[] left, float[] right, byte[] out, ResolvedDispatchHints hints) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarDirectF32(op, left, right, out, start, end);
            });
            return;
        }
        scalarDirectF32(op, left, right, out, 0, out.length);
    }

    private static void directBF16(CompareOp op, short[] left, short[] right, byte[] out, ResolvedDispatchHints hints) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarDirectBF16(op, left, right, out, start, end);
            });
            return;
        }
        scalarDirectBF16(op, left, right, out, 0, out.length);
    }

    private static void scalarDirectF64(CompareOp op, double[] left, double[] right, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.test(left[i], right[i]) ? (byte) 1 : (byte) 0;
        }
    }

    private static void scalarDirectF32(CompareOp op, float[] left, float[] right, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.test(left[i], right[i]) ? (byte) 1 : (byte) 0;
        }
    }

    private static void scalarDirectBF16(CompareOp op, short[] left, short[] right, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.test(CpuDTypeOps.fromBFloat16Bits(left[i]), CpuDTypeOps.fromBFloat16Bits(right[i])) ? (byte) 1 : (byte) 0;
        }
    }

    private static void broadcastF64(CompareOp op, double[] left, double[] right, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        parallelOrScalarBroadcast(hints, out.length, (start, end) -> scalarBroadcastF64(op, left, right, out, plan, start, end));
    }

    private static void broadcastF32(CompareOp op, float[] left, float[] right, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        parallelOrScalarBroadcast(hints, out.length, (start, end) -> scalarBroadcastF32(op, left, right, out, plan, start, end));
    }

    private static void broadcastBF16(CompareOp op, short[] left, short[] right, byte[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        parallelOrScalarBroadcast(hints, out.length, (start, end) -> scalarBroadcastBF16(op, left, right, out, plan, start, end));
    }

    private static void parallelOrScalarBroadcast(ResolvedDispatchHints hints, int length, RangeConsumer consumer) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, length);
                consumer.accept(start, end);
            });
            return;
        }
        consumer.accept(0, length);
    }

    private static void scalarBroadcastF64(CompareOp op, double[] left, double[] right, byte[] out, ResolvedBroadcastPlan plan, int start, int end) {
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
            out[i] = op.test(left[leftIdx], right[rightIdx]) ? (byte) 1 : (byte) 0;
            if (i + 1 < end) {
                int[] next = ElementwiseBinaryExecutor.nextIndices(coords, outShape, leftEff, rightEff, leftResets, rightResets, rank, leftIdx, rightIdx);
                leftIdx = next[0];
                rightIdx = next[1];
            }
        }
    }

    private static void scalarBroadcastF32(CompareOp op, float[] left, float[] right, byte[] out, ResolvedBroadcastPlan plan, int start, int end) {
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
            out[i] = op.test(left[leftIdx], right[rightIdx]) ? (byte) 1 : (byte) 0;
            if (i + 1 < end) {
                int[] next = ElementwiseBinaryExecutor.nextIndices(coords, outShape, leftEff, rightEff, leftResets, rightResets, rank, leftIdx, rightIdx);
                leftIdx = next[0];
                rightIdx = next[1];
            }
        }
    }

    private static void scalarBroadcastBF16(CompareOp op, short[] left, short[] right, byte[] out, ResolvedBroadcastPlan plan, int start, int end) {
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
            out[i] = op.test(CpuDTypeOps.fromBFloat16Bits(left[leftIdx]), CpuDTypeOps.fromBFloat16Bits(right[rightIdx])) ? (byte) 1 : (byte) 0;
            if (i + 1 < end) {
                int[] next = ElementwiseBinaryExecutor.nextIndices(coords, outShape, leftEff, rightEff, leftResets, rightResets, rank, leftIdx, rightIdx);
                leftIdx = next[0];
                rightIdx = next[1];
            }
        }
    }

    @FunctionalInterface
    private interface RangeConsumer {
        void accept(int start, int end);
    }
}
