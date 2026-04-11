package backend.kernels.cpu.elementwise;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedBroadcastPlan;
import backend.kernels.cpu.ResolvedDispatchHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import tensor.Tensor;

import java.util.List;

public final class ElementwiseBinaryExecutor {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private ElementwiseBinaryExecutor() {}

    public static void execute(NumericBinaryOp op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        switch (node.getDataType()) {
            case FLOAT64 -> runF64(op, inputs.get(0).getFloat64Data(), inputs.get(1).getFloat64Data(), node.getFloat64Data(),
                    context.broadcastPlan(), context.dispatchHints());
            case FLOAT32 -> runF32(op, inputs.get(0).getFloat32Data(), inputs.get(1).getFloat32Data(), node.getFloat32Data(),
                    context.broadcastPlan(), context.dispatchHints());
            case BFLOAT16 -> runBF16(
                    op,
                    inputs.get(0).getBFloat16Data(),
                    inputs.get(1).getBFloat16Data(),
                    context.inputFloatContinuation(0, node.getFlatDataSize()),
                    context.inputFloatContinuation(1, node.getFlatDataSize()),
                    node.getBFloat16Data(),
                    context.broadcastPlan(),
                    context.dispatchHints()
            );
            default -> throw new UnsupportedOperationException("Unsupported dtype for numeric binary elementwise op: " + node.getDataType());
        }
    }

    private static void runF64(NumericBinaryOp op, double[] left, double[] right, double[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcastF64(op, left, right, out, plan, hints);
            return;
        }
        runDirectF64(op, left, right, out, hints);
    }

    private static void runF32(NumericBinaryOp op, float[] left, float[] right, float[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcastF32(op, left, right, out, plan, hints);
            return;
        }
        runDirectF32(op, left, right, out, hints);
    }

    private static void runBF16(
            NumericBinaryOp op,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (plan != null && !plan.isNoBroadcast()) {
            runBroadcastBF16(op, leftStorage, rightStorage, leftContinuation, rightContinuation, out, plan, hints);
            return;
        }
        runDirectBF16(op, leftStorage, rightStorage, leftContinuation, rightContinuation, out, hints);
    }

    private static void runDirectF64(NumericBinaryOp op, double[] left, double[] right, double[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case SCALAR -> scalarDirectF64(op, left, right, out, 0, out.length);
            case VECTOR -> vectorDirectF64(op, left, right, out, 0, out.length);
            case PARALLEL -> parallelDirectF64(op, left, right, out, hints, false);
            case PARALLEL_VECTOR -> parallelDirectF64(op, left, right, out, hints, true);
        }
    }

    private static void runDirectF32(NumericBinaryOp op, float[] left, float[] right, float[] out, ResolvedDispatchHints hints) {
        switch (hints.mode()) {
            case SCALAR -> scalarDirectF32(op, left, right, out, 0, out.length);
            case VECTOR -> vectorDirectF32(op, left, right, out, 0, out.length);
            case PARALLEL -> parallelDirectF32(op, left, right, out, hints, false);
            case PARALLEL_VECTOR -> parallelDirectF32(op, left, right, out, hints, true);
        }
    }

    private static void runDirectBF16(
            NumericBinaryOp op,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedDispatchHints hints
    ) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarDirectBF16(op, leftStorage, rightStorage, leftContinuation, rightContinuation, out, start, end);
            });
            return;
        }
        scalarDirectBF16(op, leftStorage, rightStorage, leftContinuation, rightContinuation, out, 0, out.length);
    }

    private static void parallelDirectF64(NumericBinaryOp op, double[] left, double[] right, double[] out, ResolvedDispatchHints hints, boolean preferVector) {
        int chunkSize = preferVector ? hints.vectorChunkSize() : hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            if (preferVector) {
                vectorDirectF64(op, left, right, out, start, end);
            } else {
                scalarDirectF64(op, left, right, out, start, end);
            }
        });
    }

    private static void parallelDirectF32(NumericBinaryOp op, float[] left, float[] right, float[] out, ResolvedDispatchHints hints, boolean preferVector) {
        int chunkSize = preferVector ? hints.vectorChunkSize() : hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            if (preferVector) {
                vectorDirectF32(op, left, right, out, start, end);
            } else {
                scalarDirectF32(op, left, right, out, start, end);
            }
        });
    }

    private static void scalarDirectF64(NumericBinaryOp op, double[] left, double[] right, double[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.applyF64(left[i], right[i]);
        }
    }

    private static void scalarDirectF32(NumericBinaryOp op, float[] left, float[] right, float[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = op.applyF32(left[i], right[i]);
        }
    }

    private static void scalarDirectBF16(
            NumericBinaryOp op,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            float left = leftContinuation != null ? leftContinuation[i] : CpuDTypeOps.fromBFloat16Bits(leftStorage[i]);
            float right = rightContinuation != null ? rightContinuation[i] : CpuDTypeOps.fromBFloat16Bits(rightStorage[i]);
            out[i] = CpuDTypeOps.toBFloat16Bits(op.applyBF16(left, right));
        }
    }

    private static void vectorDirectF64(NumericBinaryOp op, double[] left, double[] right, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            DoubleVector l = DoubleVector.fromArray(F64, left, i);
            DoubleVector r = DoubleVector.fromArray(F64, right, i);
            op.applyVectorF64(l, r).intoArray(out, i);
        }
        scalarDirectF64(op, left, right, out, i, end);
    }

    private static void vectorDirectF32(NumericBinaryOp op, float[] left, float[] right, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            FloatVector l = FloatVector.fromArray(F32, left, i);
            FloatVector r = FloatVector.fromArray(F32, right, i);
            op.applyVectorF32(l, r).intoArray(out, i);
        }
        scalarDirectF32(op, left, right, out, i, end);
    }

    private static void runBroadcastF64(NumericBinaryOp op, double[] left, double[] right, double[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarBroadcastF64(op, left, right, out, plan, start, end);
            });
            return;
        }
        scalarBroadcastF64(op, left, right, out, plan, 0, out.length);
    }

    private static void runBroadcastF32(NumericBinaryOp op, float[] left, float[] right, float[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarBroadcastF32(op, left, right, out, plan, start, end);
            });
            return;
        }
        scalarBroadcastF32(op, left, right, out, plan, 0, out.length);
    }

    private static void runBroadcastBF16(
            NumericBinaryOp op,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        if (hints.mode() == CpuExecutionMode.PARALLEL || hints.mode() == CpuExecutionMode.PARALLEL_VECTOR) {
            int chunkSize = hints.scalarChunkSize();
            int chunks = (out.length + chunkSize - 1) / chunkSize;
            CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                int start = chunk * chunkSize;
                int end = Math.min(start + chunkSize, out.length);
                scalarBroadcastBF16(op, leftStorage, rightStorage, leftContinuation, rightContinuation, out, plan, start, end);
            });
            return;
        }
        scalarBroadcastBF16(op, leftStorage, rightStorage, leftContinuation, rightContinuation, out, plan, 0, out.length);
    }

    private static void scalarBroadcastF64(NumericBinaryOp op, double[] left, double[] right, double[] out, ResolvedBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] leftEff = plan.aEffStrides();
        int[] rightEff = plan.bEffStrides();
        int[] leftResets = plan.aResets();
        int[] rightResets = plan.bResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int leftIdx = initialIndex(coords, leftEff);
        int rightIdx = initialIndex(coords, rightEff);
        for (int i = start; i < end; i++) {
            out[i] = op.applyF64(left[leftIdx], right[rightIdx]);
            if (i + 1 < end) {
                int[] next = nextIndices(coords, outShape, leftEff, rightEff, leftResets, rightResets, rank, leftIdx, rightIdx);
                leftIdx = next[0];
                rightIdx = next[1];
            }
        }
    }

    private static void scalarBroadcastF32(NumericBinaryOp op, float[] left, float[] right, float[] out, ResolvedBroadcastPlan plan, int start, int end) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] leftEff = plan.aEffStrides();
        int[] rightEff = plan.bEffStrides();
        int[] leftResets = plan.aResets();
        int[] rightResets = plan.bResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int leftIdx = initialIndex(coords, leftEff);
        int rightIdx = initialIndex(coords, rightEff);
        for (int i = start; i < end; i++) {
            out[i] = op.applyF32(left[leftIdx], right[rightIdx]);
            if (i + 1 < end) {
                int[] next = nextIndices(coords, outShape, leftEff, rightEff, leftResets, rightResets, rank, leftIdx, rightIdx);
                leftIdx = next[0];
                rightIdx = next[1];
            }
        }
    }

    private static void scalarBroadcastBF16(
            NumericBinaryOp op,
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedBroadcastPlan plan,
            int start,
            int end
    ) {
        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] leftEff = plan.aEffStrides();
        int[] rightEff = plan.bEffStrides();
        int[] leftResets = plan.aResets();
        int[] rightResets = plan.bResets();
        int rank = outStrides.length;
        int[] coords = initCoords(start, outStrides, rank);
        int leftIdx = initialIndex(coords, leftEff);
        int rightIdx = initialIndex(coords, rightEff);
        for (int i = start; i < end; i++) {
            float left = leftContinuation != null ? leftContinuation[leftIdx] : CpuDTypeOps.fromBFloat16Bits(leftStorage[leftIdx]);
            float right = rightContinuation != null ? rightContinuation[rightIdx] : CpuDTypeOps.fromBFloat16Bits(rightStorage[rightIdx]);
            out[i] = CpuDTypeOps.toBFloat16Bits(op.applyBF16(left, right));
            if (i + 1 < end) {
                int[] next = nextIndices(coords, outShape, leftEff, rightEff, leftResets, rightResets, rank, leftIdx, rightIdx);
                leftIdx = next[0];
                rightIdx = next[1];
            }
        }
    }

    static int[] initCoords(int start, int[] outStrides, int rank) {
        int[] coords = new int[rank];
        int temp = start;
        for (int d = 0; d < rank; d++) {
            coords[d] = temp / outStrides[d];
            temp %= outStrides[d];
        }
        return coords;
    }

    static int initialIndex(int[] coords, int[] effectiveStrides) {
        int index = 0;
        for (int d = 0; d < coords.length; d++) {
            index += coords[d] * effectiveStrides[d];
        }
        return index;
    }

    static int[] nextIndices(
            int[] coords,
            int[] outShape,
            int[] leftEff,
            int[] rightEff,
            int[] leftResets,
            int[] rightResets,
            int rank,
            int leftIndex,
            int rightIndex
    ) {
        int nextLeft = leftIndex;
        int nextRight = rightIndex;
        for (int d = rank - 1; d >= 0; d--) {
            coords[d]++;
            nextLeft += leftEff[d];
            nextRight += rightEff[d];
            if (coords[d] < outShape[d]) {
                break;
            }
            coords[d] = 0;
            nextLeft -= leftResets[d];
            nextRight -= rightResets[d];
        }
        return new int[]{nextLeft, nextRight};
    }
}
