package backend.kernels.cpu;

import operations.Operation;

public final class BroadcastBinaryKernel {
    private BroadcastBinaryKernel() {}

    public static void runF64(
            Operation.OpType type,
            double[] a,
            double[] b,
            double[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        CpuExecutionMode mode = hints.mode();
        if (rank1F64(type, a, b, out, plan)) {
            return;
        }
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            parallelF64(type, a, b, out, plan, hints);
            return;
        }
        scalarF64(type, a, b, out, plan, 0, out.length);
    }

    public static void runF32(
            Operation.OpType type,
            float[] a,
            float[] b,
            float[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        CpuExecutionMode mode = hints.mode();
        if (rank1F32(type, a, b, out, plan)) {
            return;
        }
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            parallelF32(type, a, b, out, plan, hints);
            return;
        }
        scalarF32(type, a, b, out, plan, 0, out.length);
    }

    public static void runBF16(
            Operation.OpType type,
            short[] a,
            short[] b,
            short[] out,
            ResolvedBroadcastPlan plan,
            ResolvedDispatchHints hints
    ) {
        CpuExecutionMode mode = hints.mode();
        if (rank1F16(type, a, b, out, plan)) {
            return;
        }
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            parallelF16(type, a, b, out, plan, hints);
            return;
        }
        scalarF16(type, a, b, out, plan, 0, out.length);
    }

    private static boolean rank1F64(Operation.OpType type, double[] a, double[] b, double[] out, ResolvedBroadcastPlan plan) {
        int[] outShape = plan.outShape();
        if (outShape.length != 1) {
            return false;
        }
        int strideA = plan.aEffStrides()[0];
        int strideB = plan.bEffStrides()[0];
        switch (type) {
            case ADD -> {
                for (int i = 0; i < out.length; i++) out[i] = a[i * strideA] + b[i * strideB];
            }
            case SUB -> {
                for (int i = 0; i < out.length; i++) out[i] = a[i * strideA] - b[i * strideB];
            }
            case MUL -> {
                for (int i = 0; i < out.length; i++) out[i] = a[i * strideA] * b[i * strideB];
            }
            case DIV -> {
                for (int i = 0; i < out.length; i++) out[i] = a[i * strideA] / b[i * strideB];
            }
            case MIN -> {
                for (int i = 0; i < out.length; i++) out[i] = Math.min(a[i * strideA], b[i * strideB]);
            }
            case MAX -> {
                for (int i = 0; i < out.length; i++) out[i] = Math.max(a[i * strideA], b[i * strideB]);
            }
            default -> {
                return false;
            }
        }
        return true;
    }

    private static boolean rank1F32(Operation.OpType type, float[] a, float[] b, float[] out, ResolvedBroadcastPlan plan) {
        int[] outShape = plan.outShape();
        if (outShape.length != 1) {
            return false;
        }
        int strideA = plan.aEffStrides()[0];
        int strideB = plan.bEffStrides()[0];
        switch (type) {
            case ADD -> {
                for (int i = 0; i < out.length; i++) out[i] = a[i * strideA] + b[i * strideB];
            }
            case SUB -> {
                for (int i = 0; i < out.length; i++) out[i] = a[i * strideA] - b[i * strideB];
            }
            case MUL -> {
                for (int i = 0; i < out.length; i++) out[i] = a[i * strideA] * b[i * strideB];
            }
            case DIV -> {
                for (int i = 0; i < out.length; i++) out[i] = a[i * strideA] / b[i * strideB];
            }
            case MIN -> {
                for (int i = 0; i < out.length; i++) out[i] = Math.min(a[i * strideA], b[i * strideB]);
            }
            case MAX -> {
                for (int i = 0; i < out.length; i++) out[i] = Math.max(a[i * strideA], b[i * strideB]);
            }
            default -> {
                return false;
            }
        }
        return true;
    }

    private static boolean rank1F16(Operation.OpType type, short[] a, short[] b, short[] out, ResolvedBroadcastPlan plan) {
        int[] outShape = plan.outShape();
        if (outShape.length != 1) {
            return false;
        }
        int strideA = plan.aEffStrides()[0];
        int strideB = plan.bEffStrides()[0];
        switch (type) {
            case ADD -> {
                for (int i = 0; i < out.length; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[i * strideB]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(av + bv);
                }
            }
            case SUB -> {
                for (int i = 0; i < out.length; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[i * strideB]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(av - bv);
                }
            }
            case MUL -> {
                for (int i = 0; i < out.length; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[i * strideB]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(av * bv);
                }
            }
            case DIV -> {
                for (int i = 0; i < out.length; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[i * strideB]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(av / bv);
                }
            }
            case MIN -> {
                for (int i = 0; i < out.length; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[i * strideB]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(Math.min(av, bv));
                }
            }
            case MAX -> {
                for (int i = 0; i < out.length; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[i * strideA]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[i * strideB]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(Math.max(av, bv));
                }
            }
            default -> {
                return false;
            }
        }
        return true;
    }

    private static void parallelF64(Operation.OpType type, double[] a, double[] b, double[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalarF64(type, a, b, out, plan, start, end);
        });
    }

    private static void parallelF32(Operation.OpType type, float[] a, float[] b, float[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalarF32(type, a, b, out, plan, start, end);
        });
    }

    private static void parallelF16(Operation.OpType type, short[] a, short[] b, short[] out, ResolvedBroadcastPlan plan, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalarF16(type, a, b, out, plan, start, end);
        });
    }

    private static void scalarF64(Operation.OpType type, double[] a, double[] b, double[] out, ResolvedBroadcastPlan plan, int start, int end) {
        if (start >= end) {
            return;
        }

        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] aEffStrides = plan.aEffStrides();
        int[] bEffStrides = plan.bEffStrides();
        int[] aResets = plan.aResets();
        int[] bResets = plan.bResets();
        int rank = outStrides.length;

        int[] coords = new int[rank];
        int temp = start;
        for (int d = 0; d < rank; d++) {
            coords[d] = temp / outStrides[d];
            temp %= outStrides[d];
        }

        int aIdx = 0;
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            aIdx += coords[d] * aEffStrides[d];
            bIdx += coords[d] * bEffStrides[d];
        }

        switch (type) {
            case ADD -> {
                for (int i = start; i < end; i++) {
                    out[i] = a[aIdx] + b[bIdx];
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case SUB -> {
                for (int i = start; i < end; i++) {
                    out[i] = a[aIdx] - b[bIdx];
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MUL -> {
                for (int i = start; i < end; i++) {
                    out[i] = a[aIdx] * b[bIdx];
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case DIV -> {
                for (int i = start; i < end; i++) {
                    out[i] = a[aIdx] / b[bIdx];
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MIN -> {
                for (int i = start; i < end; i++) {
                    out[i] = Math.min(a[aIdx], b[bIdx]);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MAX -> {
                for (int i = start; i < end; i++) {
                    out[i] = Math.max(a[aIdx], b[bIdx]);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported broadcast op type: " + type);
        }
    }

    private static void scalarF32(Operation.OpType type, float[] a, float[] b, float[] out, ResolvedBroadcastPlan plan, int start, int end) {
        if (start >= end) {
            return;
        }

        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] aEffStrides = plan.aEffStrides();
        int[] bEffStrides = plan.bEffStrides();
        int[] aResets = plan.aResets();
        int[] bResets = plan.bResets();
        int rank = outStrides.length;

        // Decode start index once per chunk.
        int[] coords = new int[rank];
        int temp = start;
        for (int d = 0; d < rank; d++) {
            coords[d] = temp / outStrides[d];
            temp %= outStrides[d];
        }

        int aIdx = 0;
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            aIdx += coords[d] * aEffStrides[d];
            bIdx += coords[d] * bEffStrides[d];
        }

        switch (type) {
            case ADD -> {
                for (int i = start; i < end; i++) {
                    out[i] = a[aIdx] + b[bIdx];
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case SUB -> {
                for (int i = start; i < end; i++) {
                    out[i] = a[aIdx] - b[bIdx];
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MUL -> {
                for (int i = start; i < end; i++) {
                    out[i] = a[aIdx] * b[bIdx];
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case DIV -> {
                for (int i = start; i < end; i++) {
                    out[i] = a[aIdx] / b[bIdx];
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MIN -> {
                for (int i = start; i < end; i++) {
                    out[i] = Math.min(a[aIdx], b[bIdx]);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MAX -> {
                for (int i = start; i < end; i++) {
                    out[i] = Math.max(a[aIdx], b[bIdx]);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported broadcast op type: " + type);
        }
    }

    private static void scalarF16(Operation.OpType type, short[] a, short[] b, short[] out, ResolvedBroadcastPlan plan, int start, int end) {
        if (start >= end) {
            return;
        }

        int[] outStrides = plan.outStrides();
        int[] outShape = plan.outShape();
        int[] aEffStrides = plan.aEffStrides();
        int[] bEffStrides = plan.bEffStrides();
        int[] aResets = plan.aResets();
        int[] bResets = plan.bResets();
        int rank = outStrides.length;

        int[] coords = new int[rank];
        int temp = start;
        for (int d = 0; d < rank; d++) {
            coords[d] = temp / outStrides[d];
            temp %= outStrides[d];
        }

        int aIdx = 0;
        int bIdx = 0;
        for (int d = 0; d < rank; d++) {
            aIdx += coords[d] * aEffStrides[d];
            bIdx += coords[d] * bEffStrides[d];
        }

        switch (type) {
            case ADD -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aIdx]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bIdx]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(av + bv);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case SUB -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aIdx]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bIdx]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(av - bv);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MUL -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aIdx]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bIdx]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(av * bv);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case DIV -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aIdx]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bIdx]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(av / bv);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MIN -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aIdx]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bIdx]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(Math.min(av, bv));
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            case MAX -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromBFloat16Bits(a[aIdx]);
                    float bv = CpuDTypeOps.fromBFloat16Bits(b[bIdx]);
                    out[i] = CpuDTypeOps.toBFloat16Bits(Math.max(av, bv));
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= aResets[d];
                        bIdx -= bResets[d];
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported broadcast op type: " + type);
        }
    }

}
