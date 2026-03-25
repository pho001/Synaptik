package Backend.kernels.cpu;

import Operations.Operation;

public final class BroadcastBinaryKernel {
    private BroadcastBinaryKernel() {}

    public static void runF64(
            Operation.OpType type,
            double[] a,
            double[] b,
            double[] out,
            ResolvedBroadcastPlan plan,
            CpuExecutionMode mode,
            CpuExecutionConfig config
    ) {
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            parallelF64(type, a, b, out, plan, config);
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
            CpuExecutionMode mode,
            CpuExecutionConfig config
    ) {
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            parallelF32(type, a, b, out, plan, config);
            return;
        }
        scalarF32(type, a, b, out, plan, 0, out.length);
    }

    public static void runF16(
            Operation.OpType type,
            short[] a,
            short[] b,
            short[] out,
            ResolvedBroadcastPlan plan,
            CpuExecutionMode mode,
            CpuExecutionConfig config
    ) {
        if (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR) {
            parallelF16(type, a, b, out, plan, config);
            return;
        }
        scalarF16(type, a, b, out, plan, 0, out.length);
    }

    private static void parallelF64(Operation.OpType type, double[] a, double[] b, double[] out, ResolvedBroadcastPlan plan, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalarF64(type, a, b, out, plan, start, end);
        });
    }

    private static void parallelF32(Operation.OpType type, float[] a, float[] b, float[] out, ResolvedBroadcastPlan plan, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalarF32(type, a, b, out, plan, start, end);
        });
    }

    private static void parallelF16(Operation.OpType type, short[] a, short[] b, short[] out, ResolvedBroadcastPlan plan, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
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
                    float av = CpuDTypeOps.fromHalfBits(a[aIdx]);
                    float bv = CpuDTypeOps.fromHalfBits(b[bIdx]);
                    out[i] = CpuDTypeOps.toHalfBits(av + bv);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
                    }
                }
            }
            case SUB -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromHalfBits(a[aIdx]);
                    float bv = CpuDTypeOps.fromHalfBits(b[bIdx]);
                    out[i] = CpuDTypeOps.toHalfBits(av - bv);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
                    }
                }
            }
            case MUL -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromHalfBits(a[aIdx]);
                    float bv = CpuDTypeOps.fromHalfBits(b[bIdx]);
                    out[i] = CpuDTypeOps.toHalfBits(av * bv);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
                    }
                }
            }
            case DIV -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromHalfBits(a[aIdx]);
                    float bv = CpuDTypeOps.fromHalfBits(b[bIdx]);
                    out[i] = CpuDTypeOps.toHalfBits(av / bv);
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
                    }
                }
            }
            case MIN -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromHalfBits(a[aIdx]);
                    float bv = CpuDTypeOps.fromHalfBits(b[bIdx]);
                    out[i] = CpuDTypeOps.toHalfBits(Math.min(av, bv));
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
                    }
                }
            }
            case MAX -> {
                for (int i = start; i < end; i++) {
                    float av = CpuDTypeOps.fromHalfBits(a[aIdx]);
                    float bv = CpuDTypeOps.fromHalfBits(b[bIdx]);
                    out[i] = CpuDTypeOps.toHalfBits(Math.max(av, bv));
                    for (int d = rank - 1; d >= 0; d--) {
                        coords[d]++;
                        aIdx += aEffStrides[d];
                        bIdx += bEffStrides[d];
                        if (coords[d] < outShape[d]) {
                            break;
                        }
                        coords[d] = 0;
                        aIdx -= outShape[d] * aEffStrides[d];
                        bIdx -= outShape[d] * bEffStrides[d];
                    }
                }
            }
            default -> throw new UnsupportedOperationException("Unsupported broadcast op type: " + type);
        }
    }

}
