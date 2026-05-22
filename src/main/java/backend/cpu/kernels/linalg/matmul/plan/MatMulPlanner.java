package backend.cpu.kernels.linalg.matmul.plan;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasRuntime;
import backend.cpu.kernels.linalg.matmul.plan.ResolvedMatMulHints;
import backend.cpu.kernels.plan.CpuPlanningPolicy;
import config.backend.CpuMatMulMicroKernel;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import tensor.DataType;
import tensor.Tensor;

import java.util.Objects;

public final class MatMulPlanner {
    private static final double WIDE_N_OVER_K_TRIGGER = 4.0d;
    private final CpuPlanningPolicy policy;

    public MatMulPlanner(CpuPlanningPolicy policy) {
        this.policy = policy;
    }

    public ResolvedMatMulHints resolve(Tensor a, Tensor b, Tensor out, BlasConfig blasConfig) {
        return resolve(a, b, out, blasConfig, CpuStorageProfile.AUTO, false);
    }

    public ResolvedMatMulHints resolve(Tensor a, Tensor b, Tensor out, BlasConfig blasConfig, boolean publishFloatContinuation) {
        return resolve(a, b, out, blasConfig, CpuStorageProfile.AUTO, publishFloatContinuation);
    }

    public ResolvedMatMulHints resolve(
            Tensor a,
            Tensor b,
            Tensor out,
            BlasConfig blasConfig,
            CpuStorageProfile cpuStorageProfile,
            boolean publishFloatContinuation
    ) {
        Objects.requireNonNull(a, "a cannot be null");
        Objects.requireNonNull(b, "b cannot be null");
        Objects.requireNonNull(out, "out cannot be null");
        Objects.requireNonNull(blasConfig, "blasConfig cannot be null");

        return resolve(
                a.getShapeUnsafe(),
                a.isContiguous(),
                b.getShapeUnsafe(),
                b.isContiguous(),
                out.getShapeUnsafe(),
                out.getDataType(),
                out.isContiguous(),
                blasConfig,
                cpuStorageProfile,
                publishFloatContinuation
        );
    }

    public ResolvedMatMulHints resolve(
            int[] aShape,
            boolean aContiguous,
            int[] bShape,
            boolean bContiguous,
            int[] outShape,
            DataType outDataType,
            boolean outContiguous,
            BlasConfig blasConfig
    ) {
        return resolve(
                aShape,
                aContiguous,
                bShape,
                bContiguous,
                outShape,
                outDataType,
                outContiguous,
                blasConfig,
                CpuStorageProfile.AUTO,
                false
        );
    }

    public ResolvedMatMulHints resolve(
            int[] aShape,
            boolean aContiguous,
            int[] bShape,
            boolean bContiguous,
            int[] outShape,
            DataType outDataType,
            boolean outContiguous,
            BlasConfig blasConfig,
            boolean publishFloatContinuation
    ) {
        return resolve(
                aShape,
                aContiguous,
                bShape,
                bContiguous,
                outShape,
                outDataType,
                outContiguous,
                blasConfig,
                CpuStorageProfile.AUTO,
                publishFloatContinuation
        );
    }

    public ResolvedMatMulHints resolve(
            int[] aShape,
            boolean aContiguous,
            int[] bShape,
            boolean bContiguous,
            int[] outShape,
            DataType outDataType,
            boolean outContiguous,
            BlasConfig blasConfig,
            CpuStorageProfile cpuStorageProfile,
            boolean publishFloatContinuation
    ) {
        Objects.requireNonNull(aShape, "aShape cannot be null");
        Objects.requireNonNull(bShape, "bShape cannot be null");
        Objects.requireNonNull(outShape, "outShape cannot be null");
        Objects.requireNonNull(outDataType, "outDataType cannot be null");
        Objects.requireNonNull(blasConfig, "blasConfig cannot be null");

        if (aShape.length < 2 || bShape.length < 2) {
            throw new IllegalArgumentException("MatMul expects rank >= 2 tensors.");
        }

        int m = aShape[aShape.length - 2];
        int k = aShape[aShape.length - 1];
        int n = bShape[bShape.length - 1];
        long batchCount = 1L;
        int outRank = outShape.length;
        for (int i = 0; i < outRank - 2; i++) {
            batchCount *= outShape[i];
        }
        long work = batchCount * m * n * k;

        boolean parallel = work >= policy.matMulParallelMinSize() && policy.plannedWorkers() > 1;
        boolean useBlas = aShape.length == 2 && bShape.length == 2
                && shouldUseBlas(outDataType, aContiguous, bContiguous, outContiguous, m, n, k, blasConfig, publishFloatContinuation);
        boolean useBatchedBlas = aShape.length > 2
                && shouldUseBatchedBlas(aShape, aContiguous, bShape, bContiguous, outShape, outDataType, outContiguous, m, n, k, work, blasConfig, publishFloatContinuation);
        MatMulExecutionRoute route = resolveRoute(
                aShape,
                aContiguous,
                bShape,
                bContiguous,
                outDataType,
                outContiguous,
                m,
                n,
                k,
                work,
                blasConfig,
                cpuStorageProfile,
                publishFloatContinuation,
                useBlas,
                useBatchedBlas
        );
        if (route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT) {
            useBlas = true;
            useBatchedBlas = false;
        }

        return new ResolvedMatMulHints(
                useBlas,
                useBatchedBlas,
                route,
                parallel,
                policy.matMulTileM(),
                policy.matMulTileN(),
                policy.matMulTileK(),
                policy.plannedWorkers(),
                work,
                policy.matMulMicroKernel().resolve(outDataType)
        );
    }

    public ResolvedMatMulHints resolveJava(int[] aShape, int[] bShape, int[] outShape, DataType outDataType) {
        Objects.requireNonNull(aShape, "aShape cannot be null");
        Objects.requireNonNull(bShape, "bShape cannot be null");
        Objects.requireNonNull(outShape, "outShape cannot be null");
        Objects.requireNonNull(outDataType, "outDataType cannot be null");
        if (aShape.length < 2 || bShape.length < 2 || outShape.length < 2) {
            throw new IllegalArgumentException("MatMul expects rank >= 2 shapes.");
        }

        int m = aShape[aShape.length - 2];
        int k = aShape[aShape.length - 1];
        int n = bShape[bShape.length - 1];
        long batchCount = 1L;
        for (int i = 0; i < outShape.length - 2; i++) {
            batchCount *= outShape[i];
        }
        long work = batchCount * m * n * k;
        boolean parallel = work >= policy.matMulParallelMinSize() && policy.plannedWorkers() > 1;
        return new ResolvedMatMulHints(
                false,
                false,
                parallel,
                policy.matMulTileM(),
                policy.matMulTileN(),
                policy.matMulTileK(),
                policy.plannedWorkers(),
                work,
                policy.matMulMicroKernel().resolve(outDataType)
        );
    }

    public ResolvedMatMulHints resolveAttention(Tensor a, Tensor b, Tensor out, BlasConfig blasConfig) {
        ResolvedMatMulHints generic = resolve(a, b, out, blasConfig);
        return withMatMulConfig(
                generic,
                policy.attentionMatMulTileM(),
                policy.attentionMatMulTileN(),
                policy.attentionMatMulTileK(),
                policy.attentionMatMulMicroKernel().resolve(out.getDataType())
        );
    }

    public ResolvedMatMulHints resolveAttention(
            int[] aShape,
            boolean aContiguous,
            int[] bShape,
            boolean bContiguous,
            int[] outShape,
            DataType outDataType,
            boolean outContiguous,
            BlasConfig blasConfig
    ) {
        ResolvedMatMulHints generic = resolve(
                aShape,
                aContiguous,
                bShape,
                bContiguous,
                outShape,
                outDataType,
                outContiguous,
                blasConfig
        );
        return withMatMulConfig(
                generic,
                policy.attentionMatMulTileM(),
                policy.attentionMatMulTileN(),
                policy.attentionMatMulTileK(),
                policy.attentionMatMulMicroKernel().resolve(outDataType)
        );
    }

    public ResolvedMatMulHints resolveAttentionJava(int[] aShape, int[] bShape, int[] outShape, DataType outDataType) {
        ResolvedMatMulHints generic = resolveJava(aShape, bShape, outShape, outDataType);
        return withMatMulConfig(
                generic,
                policy.attentionMatMulTileM(),
                policy.attentionMatMulTileN(),
                policy.attentionMatMulTileK(),
                policy.attentionMatMulMicroKernel().resolve(outDataType)
        );
    }

    private boolean shouldUseBlas(
            DataType outDataType,
            boolean aContiguous,
            boolean bContiguous,
            boolean outContiguous,
            int m,
            int n,
            int k,
            BlasConfig blasConfig,
            boolean publishFloatContinuation
    ) {
        if (outDataType != DataType.FLOAT32 && outDataType != DataType.FLOAT64 && outDataType != DataType.BFLOAT16) {
            return false;
        }
        if (blasConfig.provider() != BlasProvider.OPENBLAS_FFM) {
            return false;
        }
        if (outDataType == DataType.BFLOAT16 && !bf16BlasSymbolAvailable(publishFloatContinuation)) {
            return false;
        }
        long work = (long) m * n * k;
        if (work < blasConfig.matmulMinWork()) {
            return false;
        }
        if (!aContiguous || !bContiguous || !outContiguous) {
            return false;
        }
        if (outDataType == DataType.FLOAT32 || outDataType == DataType.BFLOAT16) {
            MatMulBlasShapeHeuristics heuristics = selectBlasShapeHeuristics(m, n, k, blasConfig);
            if (heuristics.requireMgeK() && m < k) {
                return false;
            }
            if (((double) n / Math.max(1, k)) > heuristics.maxNOverK()) {
                return false;
            }
        }
        return true;
    }

    private MatMulExecutionRoute resolveRoute(
            int[] aShape,
            boolean aContiguous,
            int[] bShape,
            boolean bContiguous,
            DataType outDataType,
            boolean outContiguous,
            int m,
            int n,
            int k,
            long work,
            BlasConfig blasConfig,
            CpuStorageProfile cpuStorageProfile,
            boolean publishFloatContinuation,
            boolean useBlas,
            boolean useBatchedBlas
    ) {
        if (useBatchedBlas) {
            return MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING;
        }
        if (blasConfig.provider() != BlasProvider.OPENBLAS_FFM) {
            return MatMulExecutionRoute.JAVA_DIRECT;
        }
        BlasStorageMode storageMode = effectiveStorageMode(blasConfig, cpuStorageProfile);
        if (storageMode == BlasStorageMode.CPU_ARRAY) {
            return useBlas ? MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING : MatMulExecutionRoute.JAVA_DIRECT;
        }
        boolean nativeEligible = isNativeSegmentEligible(
                aShape,
                aContiguous,
                bShape,
                bContiguous,
                outDataType,
                outContiguous,
                publishFloatContinuation
        );
        if (!nativeEligible) {
            return useBlas ? MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING : MatMulExecutionRoute.JAVA_DIRECT;
        }
        if (storageMode == BlasStorageMode.CPU_NATIVE) {
            return MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
        }
        long shapeWork = (long) m * n * k;
        return work >= blasConfig.matmulMinWork() && shapeWork >= blasConfig.matmulMinWork()
                ? MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT
                : (useBlas ? MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING : MatMulExecutionRoute.JAVA_DIRECT);
    }

    private static BlasStorageMode effectiveStorageMode(BlasConfig blasConfig, CpuStorageProfile cpuStorageProfile) {
        CpuStorageProfile profile = cpuStorageProfile == null ? CpuStorageProfile.AUTO : cpuStorageProfile;
        return switch (profile) {
            case CPU_ARRAY -> BlasStorageMode.CPU_ARRAY;
            case CPU_NATIVE -> BlasStorageMode.CPU_NATIVE;
            case AUTO -> blasConfig.storageMode();
        };
    }

    private static boolean isNativeSegmentEligible(
            int[] aShape,
            boolean aContiguous,
            int[] bShape,
            boolean bContiguous,
            DataType outDataType,
            boolean outContiguous,
            boolean publishFloatContinuation
    ) {
        if (aShape.length != 2 || bShape.length != 2) {
            return false;
        }
        if (!aContiguous || !bContiguous || !outContiguous) {
            return false;
        }
        return switch (outDataType) {
            case FLOAT32 -> OpenBlasRuntime.isFloat32GemmAvailable();
            case FLOAT64 -> OpenBlasRuntime.isFloat64GemmAvailable();
            case BFLOAT16 -> !publishFloatContinuation && OpenBlasRuntime.isBFloat16OutputGemmAvailable();
            default -> false;
        };
    }

    private boolean shouldUseBatchedBlas(
            int[] aShape,
            boolean aContiguous,
            int[] bShape,
            boolean bContiguous,
            int[] outShape,
            DataType outDataType,
            boolean outContiguous,
            int m,
            int n,
            int k,
            long work,
            BlasConfig blasConfig,
            boolean publishFloatContinuation
    ) {
        if (outDataType != DataType.FLOAT32 && outDataType != DataType.FLOAT64 && outDataType != DataType.BFLOAT16) {
            return false;
        }
        if (!isAttentionLikeBatchedMatMul(aShape, bShape, outShape)) {
            return false;
        }
        if (!aContiguous || !bContiguous || !outContiguous) {
            return false;
        }
        if (blasConfig.provider() != BlasProvider.OPENBLAS_FFM) {
            return false;
        }
        if (outDataType == DataType.BFLOAT16 && !bf16BlasSymbolAvailable(publishFloatContinuation)) {
            return false;
        }
        if (work < blasConfig.matmulMinWork()) {
            return false;
        }
        if (outDataType == DataType.FLOAT32 || outDataType == DataType.BFLOAT16) {
            MatMulBlasShapeHeuristics heuristics = selectBlasShapeHeuristics(m, n, k, blasConfig);
            if (heuristics.requireMgeK() && m < k) {
                return false;
            }
            if (((double) n / Math.max(1, k)) > heuristics.maxNOverK()) {
                return false;
            }
        }
        return switch (policy.attentionMatMulPolicy()) {
            case FORCE_OFF -> false;
            case FORCE_ON -> true;
            case AUTO -> true;
        };
    }

    private static boolean bf16BlasSymbolAvailable(boolean publishFloatContinuation) {
        return publishFloatContinuation
                ? OpenBlasRuntime.isBFloat16ToFloatGemmAvailable()
                : OpenBlasRuntime.isBFloat16OutputGemmAvailable();
    }

    private MatMulBlasShapeHeuristics selectBlasShapeHeuristics(int m, int n, int k, BlasConfig blasConfig) {
        if (isWideShape(n, k)) {
            return new MatMulBlasShapeHeuristics(
                    blasConfig.f32WideRequireMgeK(),
                    blasConfig.f32WideMaxNOverK()
            );
        }
        return new MatMulBlasShapeHeuristics(
                blasConfig.f32RequireMgeK(),
                blasConfig.f32MaxNOverK()
        );
    }

    private static boolean isWideShape(int n, int k) {
        return ((double) n / Math.max(1, k)) > WIDE_N_OVER_K_TRIGGER;
    }

    private record MatMulBlasShapeHeuristics(boolean requireMgeK, double maxNOverK) {}

    private boolean isAttentionLikeBatchedMatMul(int[] as, int[] bs, int[] os) {
        if (as.length < 3 || bs.length < 3 || os.length < 3) {
            return false;
        }
        int aBatchRank = as.length - 2;
        int bBatchRank = bs.length - 2;
        int oBatchRank = os.length - 2;
        if (aBatchRank != bBatchRank || aBatchRank != oBatchRank) {
            return false;
        }
        if (aBatchRank < 1) {
            return false;
        }
        if (oBatchRank == 1) {
            return false;
        }
        return os[oBatchRank - 1] == as[aBatchRank - 1];
    }

    private static ResolvedMatMulHints withMatMulConfig(
            ResolvedMatMulHints hints,
            int tileM,
            int tileN,
            int tileK,
            CpuMatMulMicroKernel microKernel
    ) {
        if (hints == null) {
            return null;
        }
        return new ResolvedMatMulHints(
                hints.useBlas(),
                hints.useBatchedBlas(),
                hints.route(),
                hints.parallel(),
                tileM,
                tileN,
                tileK,
                hints.plannedWorkers(),
                hints.work(),
                microKernel
        );
    }
}
