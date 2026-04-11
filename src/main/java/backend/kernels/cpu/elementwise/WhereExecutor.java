package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.ResolvedDispatchHints;
import backend.kernels.cpu.ResolvedWhereBroadcastPlan;
import tensor.Tensor;

import java.util.List;

public final class WhereExecutor {
    private WhereExecutor() {}

    public static void execute(List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        switch (node.getDataType()) {
            case FLOAT64 -> runF64(inputs.get(0).getBoolData(), inputs.get(1).getFloat64Data(), inputs.get(2).getFloat64Data(), node.getFloat64Data(), context.whereBroadcastPlan(), context.dispatchHints());
            case FLOAT32 -> runF32(inputs.get(0).getBoolData(), inputs.get(1).getFloat32Data(), inputs.get(2).getFloat32Data(), node.getFloat32Data(), context.whereBroadcastPlan(), context.dispatchHints());
            case BFLOAT16 -> runBF16(inputs.get(0).getBoolData(), inputs.get(1).getBFloat16Data(), inputs.get(2).getBFloat16Data(), node.getBFloat16Data(), context.whereBroadcastPlan(), context.dispatchHints());
            default -> throw new UnsupportedOperationException("WHERE only supports floating output tensors");
        }
    }

    private static void runF64(byte[] cond, double[] ifTrue, double[] ifFalse, double[] out, ResolvedWhereBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan == null || plan.isNoBroadcast()) {
            directF64(cond, ifTrue, ifFalse, out, hints);
            return;
        }
        broadcastF64(cond, ifTrue, ifFalse, out, plan, hints);
    }

    private static void runF32(byte[] cond, float[] ifTrue, float[] ifFalse, float[] out, ResolvedWhereBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan == null || plan.isNoBroadcast()) {
            directF32(cond, ifTrue, ifFalse, out, hints);
            return;
        }
        broadcastF32(cond, ifTrue, ifFalse, out, plan, hints);
    }

    private static void runBF16(byte[] cond, short[] ifTrue, short[] ifFalse, short[] out, ResolvedWhereBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan == null || plan.isNoBroadcast()) {
            directBF16(cond, ifTrue, ifFalse, out, hints);
            return;
        }
        broadcastBF16(cond, ifTrue, ifFalse, out, plan, hints);
    }

    private static void directF64(byte[] cond, double[] ifTrue, double[] ifFalse, double[] out, ResolvedDispatchHints hints) {
        runParallelIfNeeded(hints, out.length, (start, end) -> {
            for (int i = start; i < end; i++) {
                out[i] = cond[i] != 0 ? ifTrue[i] : ifFalse[i];
            }
        });
    }

    private static void directF32(byte[] cond, float[] ifTrue, float[] ifFalse, float[] out, ResolvedDispatchHints hints) {
        runParallelIfNeeded(hints, out.length, (start, end) -> {
            for (int i = start; i < end; i++) {
                out[i] = cond[i] != 0 ? ifTrue[i] : ifFalse[i];
            }
        });
    }

    private static void directBF16(byte[] cond, short[] ifTrue, short[] ifFalse, short[] out, ResolvedDispatchHints hints) {
        runParallelIfNeeded(hints, out.length, (start, end) -> {
            for (int i = start; i < end; i++) {
                out[i] = cond[i] != 0 ? ifTrue[i] : ifFalse[i];
            }
        });
    }

    private static void broadcastF64(byte[] cond, double[] ifTrue, double[] ifFalse, double[] out, ResolvedWhereBroadcastPlan plan, ResolvedDispatchHints hints) {
        runParallelIfNeeded(hints, out.length, (start, end) -> scalarBroadcastF64(cond, ifTrue, ifFalse, out, plan, start, end));
    }

    private static void broadcastF32(byte[] cond, float[] ifTrue, float[] ifFalse, float[] out, ResolvedWhereBroadcastPlan plan, ResolvedDispatchHints hints) {
        runParallelIfNeeded(hints, out.length, (start, end) -> scalarBroadcastF32(cond, ifTrue, ifFalse, out, plan, start, end));
    }

    private static void broadcastBF16(byte[] cond, short[] ifTrue, short[] ifFalse, short[] out, ResolvedWhereBroadcastPlan plan, ResolvedDispatchHints hints) {
        runParallelIfNeeded(hints, out.length, (start, end) -> scalarBroadcastBF16(cond, ifTrue, ifFalse, out, plan, start, end));
    }

    private static void scalarBroadcastF64(byte[] cond, double[] ifTrue, double[] ifFalse, double[] out, ResolvedWhereBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] condEff = plan.condEffStrides();
        int[] trueEff = plan.trueEffStrides();
        int[] falseEff = plan.falseEffStrides();
        int[] condResets = plan.condResets();
        int[] trueResets = plan.trueResets();
        int[] falseResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = ElementwiseBinaryExecutor.initCoords(start, outStrides, rank);
        int condIdx = ElementwiseBinaryExecutor.initialIndex(coords, condEff);
        int trueIdx = ElementwiseBinaryExecutor.initialIndex(coords, trueEff);
        int falseIdx = ElementwiseBinaryExecutor.initialIndex(coords, falseEff);
        for (int i = start; i < end; i++) {
            out[i] = cond[condIdx] != 0 ? ifTrue[trueIdx] : ifFalse[falseIdx];
            if (i + 1 < end) {
                int[] next = nextTernaryIndices(coords, outShape, condEff, trueEff, falseEff, condResets, trueResets, falseResets, rank, condIdx, trueIdx, falseIdx);
                condIdx = next[0];
                trueIdx = next[1];
                falseIdx = next[2];
            }
        }
    }

    private static void scalarBroadcastF32(byte[] cond, float[] ifTrue, float[] ifFalse, float[] out, ResolvedWhereBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] condEff = plan.condEffStrides();
        int[] trueEff = plan.trueEffStrides();
        int[] falseEff = plan.falseEffStrides();
        int[] condResets = plan.condResets();
        int[] trueResets = plan.trueResets();
        int[] falseResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = ElementwiseBinaryExecutor.initCoords(start, outStrides, rank);
        int condIdx = ElementwiseBinaryExecutor.initialIndex(coords, condEff);
        int trueIdx = ElementwiseBinaryExecutor.initialIndex(coords, trueEff);
        int falseIdx = ElementwiseBinaryExecutor.initialIndex(coords, falseEff);
        for (int i = start; i < end; i++) {
            out[i] = cond[condIdx] != 0 ? ifTrue[trueIdx] : ifFalse[falseIdx];
            if (i + 1 < end) {
                int[] next = nextTernaryIndices(coords, outShape, condEff, trueEff, falseEff, condResets, trueResets, falseResets, rank, condIdx, trueIdx, falseIdx);
                condIdx = next[0];
                trueIdx = next[1];
                falseIdx = next[2];
            }
        }
    }

    private static void scalarBroadcastBF16(byte[] cond, short[] ifTrue, short[] ifFalse, short[] out, ResolvedWhereBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] condEff = plan.condEffStrides();
        int[] trueEff = plan.trueEffStrides();
        int[] falseEff = plan.falseEffStrides();
        int[] condResets = plan.condResets();
        int[] trueResets = plan.trueResets();
        int[] falseResets = plan.falseResets();
        int rank = outStrides.length;
        int[] coords = ElementwiseBinaryExecutor.initCoords(start, outStrides, rank);
        int condIdx = ElementwiseBinaryExecutor.initialIndex(coords, condEff);
        int trueIdx = ElementwiseBinaryExecutor.initialIndex(coords, trueEff);
        int falseIdx = ElementwiseBinaryExecutor.initialIndex(coords, falseEff);
        for (int i = start; i < end; i++) {
            out[i] = cond[condIdx] != 0 ? ifTrue[trueIdx] : ifFalse[falseIdx];
            if (i + 1 < end) {
                int[] next = nextTernaryIndices(coords, outShape, condEff, trueEff, falseEff, condResets, trueResets, falseResets, rank, condIdx, trueIdx, falseIdx);
                condIdx = next[0];
                trueIdx = next[1];
                falseIdx = next[2];
            }
        }
    }

    private static void runParallelIfNeeded(ResolvedDispatchHints hints, int length, RangeConsumer consumer) {
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

    private static int[] nextTernaryIndices(
            int[] coords,
            int[] outShape,
            int[] condEff,
            int[] trueEff,
            int[] falseEff,
            int[] condResets,
            int[] trueResets,
            int[] falseResets,
            int rank,
            int condIndex,
            int trueIndex,
            int falseIndex
    ) {
        int nextCond = condIndex;
        int nextTrue = trueIndex;
        int nextFalse = falseIndex;
        for (int d = rank - 1; d >= 0; d--) {
            coords[d]++;
            nextCond += condEff[d];
            nextTrue += trueEff[d];
            nextFalse += falseEff[d];
            if (coords[d] < outShape[d]) {
                break;
            }
            coords[d] = 0;
            nextCond -= condResets[d];
            nextTrue -= trueResets[d];
            nextFalse -= falseResets[d];
        }
        return new int[]{nextCond, nextTrue, nextFalse};
    }

    @FunctionalInterface
    private interface RangeConsumer {
        void accept(int start, int end);
    }
}
