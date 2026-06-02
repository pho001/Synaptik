package backend.cpu1.prepare;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasRuntime;
import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.provider.matmul.Cpu1MatmulProvider;
import backend.cpu1.provider.matmul.Cpu1MatmulProviders;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares the initial dense Java matmul subset for cpu1.
 */
public final class Cpu1MatmulPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        if (operation.opType() != Operation.OpType.MATMUL) {
            throw new UnsupportedOperationException("cpu1 matmul preparer does not support " + operation.opType());
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 MATMUL requires descriptors.");
        }
        if (node.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 MATMUL expects 2 inputs, got " + node.inputIds().size());
        }
        if (config.storageKind() != Cpu1StorageKind.JAVA_ARRAY) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL supports only JAVA_ARRAY storage.");
        }
        CompiledTensorDescriptor left = descriptorIndex.byNodeId(node.inputIds().get(0));
        CompiledTensorDescriptor right = descriptorIndex.byNodeId(node.inputIds().get(1));
        requireDenseArrayContract(node, left, right);

        int[] leftShape = left.shape();
        int[] rightShape = right.shape();
        int[] outputShape = node.shape();
        int[] leftStrides = left.strides();
        int[] rightStrides = right.strides();
        int[] outputStrides = node.strides();
        validateShape(leftShape, rightShape, outputShape);

        int batchCount = batchCount(outputShape);
        int m = outputShape[outputShape.length - 2];
        int n = outputShape[outputShape.length - 1];
        int k = leftShape[leftShape.length - 1];
        Cpu1MatmulRoute route = resolveMatmulRoute(config, node.dataType(), batchCount, m, n, k);
        Cpu1MatmulProvider provider = Cpu1MatmulProviders.forRoute(route);
        Cpu1LaunchConfig launchConfig = resolveMatmulLaunch(provider.route(), batchCount, m, n, k, config);
        Cpu1VectorizationKind vectorizationKind = resolveMatmulVectorization(
                provider.route(),
                node.dataType(),
                batchCount,
                m,
                n,
                k,
                config
        );
        Cpu1MatmulKernelId kernelId = resolveMatmulKernelId(provider, node.dataType(), vectorizationKind);
        Cpu1PreparedMatmulUnit unit = new Cpu1PreparedMatmulUnit(
                node.id(),
                node.inputIds().get(0),
                node.inputIds().get(1),
                node.dataType(),
                config.storageKind(),
                route,
                vectorizationKind,
                kernelId,
                batchCount,
                m,
                n,
                k,
                leftStrides[leftStrides.length - 2],
                leftStrides[leftStrides.length - 1],
                rightStrides[rightStrides.length - 2],
                rightStrides[rightStrides.length - 1],
                outputStrides[outputStrides.length - 2],
                outputStrides[outputStrides.length - 1],
                batchOffsets(leftShape, leftStrides, outputShape, true),
                batchOffsets(rightShape, rightStrides, outputShape, true),
                batchOffsets(outputShape, outputStrides, outputShape, false),
                launchConfig,
                workspaceSpec(kernelId, batchCount, n, k)
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isMatmulOp(Operation.OpType opType) {
        return opType == Operation.OpType.MATMUL;
    }

    private static void requireDenseArrayContract(
            CompiledNode node,
            CompiledTensorDescriptor left,
            CompiledTensorDescriptor right
    ) {
        DataType dataType = node.dataType();
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16) {
            throw new UnsupportedOperationException("cpu1 MATMUL does not support output dtype " + dataType);
        }
        if (left.dataType() != dataType || right.dataType() != dataType) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL requires matching input/output dtype. left="
                    + left.dataType() + ", right=" + right.dataType() + ", output=" + dataType);
        }
        if (!left.denseContiguousWithoutOffset() || !right.denseContiguousWithoutOffset()) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL supports only dense contiguous inputs without storage offset.");
        }
        if (!node.contiguous() || node.hasStorageOffset()) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL supports only dense contiguous output without storage offset.");
        }
    }

    private static void validateShape(int[] leftShape, int[] rightShape, int[] outputShape) {
        if (leftShape.length < 2 || rightShape.length < 2 || outputShape.length < 2) {
            throw new UnsupportedOperationException("cpu1 MATMUL requires rank >= 2. left="
                    + Arrays.toString(leftShape) + ", right=" + Arrays.toString(rightShape)
                    + ", output=" + Arrays.toString(outputShape));
        }
        int leftBatchRank = leftShape.length - 2;
        int rightBatchRank = rightShape.length - 2;
        int outputBatchRank = outputShape.length - 2;
        int expectedOutputRank = Math.max(leftBatchRank, rightBatchRank) + 2;
        if (outputShape.length != expectedOutputRank) {
            throw new UnsupportedOperationException("cpu1 MATMUL output rank mismatch. expected="
                    + expectedOutputRank + ", actual=" + outputShape.length);
        }
        int m = leftShape[leftShape.length - 2];
        int k = leftShape[leftShape.length - 1];
        int rightK = rightShape[rightShape.length - 2];
        int n = rightShape[rightShape.length - 1];
        if (rightK != k || outputShape[outputShape.length - 2] != m || outputShape[outputShape.length - 1] != n) {
            throw new UnsupportedOperationException("cpu1 MATMUL core dimensions mismatch. left="
                    + Arrays.toString(leftShape) + ", right=" + Arrays.toString(rightShape)
                    + ", output=" + Arrays.toString(outputShape));
        }
        validateBroadcastBatch(leftShape, outputShape, outputBatchRank);
        validateBroadcastBatch(rightShape, outputShape, outputBatchRank);
    }

    private static void validateBroadcastBatch(int[] inputShape, int[] outputShape, int outputBatchRank) {
        int inputBatchRank = inputShape.length - 2;
        int shift = outputBatchRank - inputBatchRank;
        if (shift < 0) {
            throw new UnsupportedOperationException("cpu1 MATMUL input batch rank exceeds output batch rank.");
        }
        for (int dim = 0; dim < outputBatchRank; dim++) {
            int inputDim = dim < shift ? 1 : inputShape[dim - shift];
            int outputDim = outputShape[dim];
            if (inputDim != 1 && inputDim != outputDim) {
                throw new UnsupportedOperationException("cpu1 MATMUL batch dimensions are not broadcast-compatible. input="
                        + Arrays.toString(inputShape) + ", output=" + Arrays.toString(outputShape));
            }
        }
    }

    private static int[] batchOffsets(
            int[] shape,
            int[] strides,
            int[] outputShape,
            boolean allowBroadcast
    ) {
        int outputBatchRank = outputShape.length - 2;
        int inputBatchRank = shape.length - 2;
        int shift = outputBatchRank - inputBatchRank;
        int batchCount = batchCount(outputShape);
        int[] offsets = new int[batchCount];
        if (outputBatchRank == 0) {
            return offsets;
        }
        int[] outputBatchShape = Arrays.copyOf(outputShape, outputBatchRank);
        int[] outputBatchDenseStrides = denseStrides(outputBatchShape);
        for (int batch = 0; batch < batchCount; batch++) {
            int remaining = batch;
            int offset = 0;
            for (int dim = 0; dim < outputBatchRank; dim++) {
                int coordinate = remaining / outputBatchDenseStrides[dim];
                remaining %= outputBatchDenseStrides[dim];
                int inputDim = dim - shift;
                if (inputDim < 0) {
                    continue;
                }
                if (allowBroadcast && shape[inputDim] == 1) {
                    continue;
                }
                offset += coordinate * strides[inputDim];
            }
            offsets[batch] = offset;
        }
        return offsets;
    }

    private static int batchCount(int[] outputShape) {
        int count = 1;
        for (int dim = 0; dim < outputShape.length - 2; dim++) {
            count = Math.multiplyExact(count, outputShape[dim]);
        }
        return count;
    }

    private static Cpu1LaunchConfig resolveMatmulLaunch(
            Cpu1MatmulRoute route,
            int batchCount,
            int m,
            int n,
            int k,
            Cpu1PrepareConfig config
    ) {
        if (route != Cpu1MatmulRoute.JAVA_SCALAR) {
            return Cpu1LaunchConfig.singleThread();
        }
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        CpuKernelConfig cpuKernelConfig = requireCpuKernelConfig(config);
        int maxWorkers = config.launchConfig().workerCount();
        long outputRows = Math.multiplyExact((long) batchCount, m);
        long work = matmulWork(batchCount, m, n, k);
        if (maxWorkers <= 1 || outputRows <= 1 || work < cpuKernelConfig.matMulParallelMinSize()) {
            return Cpu1LaunchConfig.singleThread();
        }

        int plannedWorkers = (int) Math.min(maxWorkers, outputRows);
        long targetChunks = Math.multiplyExact(
                (long) plannedWorkers,
                cpuKernelConfig.highCostTargetChunksPerWorker()
        );
        long chunk = Math.max(1L, ((outputRows - 1) / targetChunks) + 1);
        return Cpu1LaunchConfig.parallel(plannedWorkers, Math.toIntExact(chunk));
    }

    private static Cpu1MatmulRoute resolveMatmulRoute(
            Cpu1PrepareConfig config,
            DataType dataType,
            int batchCount,
            int m,
            int n,
            int k
    ) {
        Cpu1MatmulRoute requested = config.matmulRoute();
        if (requested != Cpu1MatmulRoute.AUTO) {
            return requested;
        }
        return openBlasArrayCopyingEligible(config, dataType, batchCount, m, n, k)
                ? Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING
                : Cpu1MatmulRoute.JAVA_SCALAR;
    }

    private static boolean openBlasArrayCopyingEligible(
            Cpu1PrepareConfig config,
            DataType dataType,
            int batchCount,
            int m,
            int n,
            int k
    ) {
        if (config.storageKind() != Cpu1StorageKind.JAVA_ARRAY) {
            return false;
        }
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64) {
            return false;
        }
        BlasConfig blasConfig = config.blasConfig();
        if (blasConfig.provider() != BlasProvider.OPENBLAS_FFM) {
            return false;
        }
        BlasStorageMode storageMode = blasConfig.storageMode();
        if (storageMode != BlasStorageMode.CPU_ARRAY && storageMode != BlasStorageMode.AUTO) {
            return false;
        }
        long work = matmulWork(batchCount, m, n, k);
        if (work < blasConfig.matmulMinWork()) {
            return false;
        }
        return switch (dataType) {
            case FLOAT32 -> OpenBlasRuntime.isFloat32GemmAvailable();
            case FLOAT64 -> OpenBlasRuntime.isFloat64GemmAvailable();
            default -> false;
        };
    }

    private static Cpu1VectorizationKind resolveMatmulVectorization(
            Cpu1MatmulRoute route,
            DataType dataType,
            int batchCount,
            int m,
            int n,
            int k,
            Cpu1PrepareConfig config
    ) {
        if (route != Cpu1MatmulRoute.JAVA_SCALAR) {
            return Cpu1VectorizationKind.SCALAR;
        }
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64) {
            return Cpu1VectorizationKind.SCALAR;
        }
        if (!config.automaticVectorization()) {
            return config.vectorizationKind();
        }
        if (preferredVectorLength(dataType) <= 1) {
            return Cpu1VectorizationKind.SCALAR;
        }
        CpuKernelConfig cpuKernelConfig = requireCpuKernelConfig(config);
        long work = matmulWork(batchCount, m, n, k);
        return work >= cpuKernelConfig.cheapVectorMinSize()
                ? Cpu1VectorizationKind.VECTOR
                : Cpu1VectorizationKind.SCALAR;
    }

    private static Cpu1MatmulKernelId resolveMatmulKernelId(
            Cpu1MatmulProvider provider,
            DataType dataType,
            Cpu1VectorizationKind vectorizationKind
    ) {
        Cpu1MatmulKernelId scalarKernelId = provider.kernelId(dataType);
        if (provider.route() == Cpu1MatmulRoute.JAVA_SCALAR
                && vectorizationKind == Cpu1VectorizationKind.VECTOR
                && dataType == DataType.FLOAT32) {
            return Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR;
        }
        if (provider.route() == Cpu1MatmulRoute.JAVA_SCALAR
                && vectorizationKind == Cpu1VectorizationKind.VECTOR
                && dataType == DataType.FLOAT64) {
            return Cpu1MatmulKernelId.MATMUL_F64_DENSE_PACKED_B_VECTOR;
        }
        return scalarKernelId;
    }

    private static long matmulWork(int batchCount, int m, int n, int k) {
        return Math.multiplyExact(Math.multiplyExact(Math.multiplyExact((long) batchCount, m), n), k);
    }

    private static Cpu1WorkspaceSpec workspaceSpec(
            Cpu1MatmulKernelId kernelId,
            int batchCount,
            int n,
            int k
    ) {
        return switch (kernelId) {
            case MATMUL_F32_DENSE_PACKED_B_VECTOR -> Cpu1WorkspaceSpec.arrays(packedBElements(batchCount, n, k), 0, 0);
            case MATMUL_F64_DENSE_PACKED_B_VECTOR -> Cpu1WorkspaceSpec.arrays(0, packedBElements(batchCount, n, k), 0);
            default -> Cpu1WorkspaceSpec.none();
        };
    }

    private static int preferredVectorLength(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            default -> 1;
        };
    }

    private static int packedBElements(int batchCount, int n, int k) {
        return Math.toIntExact(Math.multiplyExact(Math.multiplyExact((long) batchCount, n), k));
    }

    private static CpuKernelConfig requireCpuKernelConfig(Cpu1PrepareConfig config) {
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 matmul dispatch requires CpuKernelConfig-backed prepare config.");
        }
        return cpuKernelConfig;
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            strides[dim] = stride;
            stride = Math.multiplyExact(stride, shape[dim]);
        }
        return strides;
    }
}
