package backend.cpu1.prepare;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasRuntime;
import backend.cpu1.exec.Cpu1ScratchBufferSpec;
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
        return prepare(node, descriptorIndex, config, Cpu1MatmulPostOp.NONE);
    }

    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Cpu1MatmulPostOp postOp
    ) {
        return prepare(node, node, descriptorIndex, config, postOp);
    }

    public Cpu1PreparedArtifact prepare(
            CompiledNode matmulNode,
            CompiledNode outputNode,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Cpu1MatmulPostOp postOp
    ) {
        return prepare(matmulNode, outputNode, descriptorIndex, config, postOp, -1, null);
    }

    public Cpu1PreparedArtifact prepareMatmulBiasRelu(
            CompiledNode matmulNode,
            CompiledNode addNode,
            CompiledNode outputNode,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(matmulNode, "matmulNode cannot be null");
        Objects.requireNonNull(addNode, "addNode cannot be null");
        if (addNode.operation() == null || addNode.operation().opType() != Operation.OpType.ADD) {
            throw new UnsupportedOperationException("cpu1 MATMUL ADD_BIAS_RELU requires an ADD node.");
        }
        if (addNode.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 MATMUL ADD_BIAS_RELU expects ADD with 2 inputs, got "
                    + addNode.inputIds().size());
        }
        int biasNodeId = biasNodeId(addNode, matmulNode.id());
        return prepare(
                matmulNode,
                outputNode,
                descriptorIndex,
                config,
                Cpu1MatmulPostOp.ADD_BIAS_RELU,
                biasNodeId,
                addNode
        );
    }

    public Cpu1PreparedArtifact prepareLinearBiasRelu(
            CompiledNode linearNode,
            CompiledNode outputNode,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        return prepare(
                linearNode,
                outputNode,
                descriptorIndex,
                config,
                Cpu1MatmulPostOp.ADD_BIAS_RELU,
                -1,
                null
        );
    }

    private Cpu1PreparedArtifact prepare(
            CompiledNode matmulNode,
            CompiledNode outputNode,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Cpu1MatmulPostOp postOp,
            int biasNodeId,
            CompiledNode addNode
    ) {
        Objects.requireNonNull(matmulNode, "matmulNode cannot be null");
        Objects.requireNonNull(outputNode, "outputNode cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Objects.requireNonNull(postOp, "postOp cannot be null");
        Operation operation = Objects.requireNonNull(matmulNode.operation(), "matmulNode operation cannot be null");
        boolean linearWithBias = operation.opType() == Operation.OpType.LINEAR;
        if (operation.opType() != Operation.OpType.MATMUL && !linearWithBias) {
            throw new UnsupportedOperationException("cpu1 matmul preparer does not support " + operation.opType());
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 MATMUL requires descriptors.");
        }
        if (linearWithBias && postOp != Cpu1MatmulPostOp.ADD_BIAS_RELU) {
            throw new UnsupportedOperationException("cpu1 LINEAR specialization requires ADD_BIAS_RELU post-op.");
        }
        if (linearWithBias && matmulNode.inputIds().size() != 3) {
            throw new UnsupportedOperationException("cpu1 LINEAR expects 3 inputs, got " + matmulNode.inputIds().size());
        }
        if (!linearWithBias && matmulNode.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 MATMUL expects 2 inputs, got " + matmulNode.inputIds().size());
        }
        CompiledTensorDescriptor left = descriptorIndex.byNodeId(matmulNode.inputIds().get(0));
        CompiledTensorDescriptor right = descriptorIndex.byNodeId(matmulNode.inputIds().get(1));
        requireDenseMatmulContract(matmulNode, outputNode, left, right);
        int effectiveBiasNodeId = linearWithBias ? matmulNode.inputIds().get(2) : biasNodeId;
        CompiledTensorDescriptor bias = postOp.requiresBias() ? descriptorIndex.byNodeId(effectiveBiasNodeId) : null;
        if (postOp.requiresBias()) {
            requireBiasContract(linearWithBias ? matmulNode : addNode, outputNode, bias);
        }

        int[] leftShape = left.shape();
        int[] rightShape = right.shape();
        int[] outputShape = outputNode.shape();
        int[] leftStrides = left.strides();
        int[] rightStrides = right.strides();
        int[] outputStrides = outputNode.strides();
        validateShape(leftShape, rightShape, outputShape);

        int batchCount = batchCount(outputShape);
        int m = outputShape[outputShape.length - 2];
        int n = outputShape[outputShape.length - 1];
        int k = leftShape[leftShape.length - 1];
        int[] biasBroadcastStrides = bias == null
                ? null
                : broadcastStrides(bias.shape(), bias.strides(), outputShape, "bias");
        Cpu1MatmulRoute route = resolveMatmulRoute(config, postOp, outputNode.dataType(), batchCount, m, n, k);
        requireStorageContract(route, config.storageKind());
        requirePostOpContract(route, postOp);
        Cpu1MatmulProvider provider = Cpu1MatmulProviders.forRoute(route);
        Cpu1LaunchConfig launchConfig = resolveMatmulLaunch(provider.route(), batchCount, m, n, k, config);
        Cpu1VectorizationKind vectorizationKind = resolveMatmulVectorization(
                provider.route(),
                outputNode.dataType(),
                batchCount,
                m,
                n,
                k,
                config
        );
        Cpu1MatmulKernelId kernelId = resolveMatmulKernelId(provider, outputNode.dataType(), vectorizationKind);
        Cpu1PreparedMatmulUnit unit = new Cpu1PreparedMatmulUnit(
                outputNode.id(),
                matmulNode.inputIds().get(0),
                matmulNode.inputIds().get(1),
                outputNode.dataType(),
                config.storageKind(),
                route,
                postOp,
                bias == null ? -1 : bias.nodeId(),
                bias == null ? 0 : biasBroadcastStrides[biasBroadcastStrides.length - 2],
                bias == null ? 0 : biasBroadcastStrides[biasBroadcastStrides.length - 1],
                bias == null ? null : batchOffsetsFromBroadcastStrides(outputShape, biasBroadcastStrides),
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
                scratchBufferSpec(kernelId, batchCount, n, k),
                openBlasThreads(route, config.blasConfig())
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isMatmulOp(Operation.OpType opType) {
        return opType == Operation.OpType.MATMUL;
    }

    private static void requireDenseMatmulContract(
            CompiledNode matmulNode,
            CompiledNode outputNode,
            CompiledTensorDescriptor left,
            CompiledTensorDescriptor right
    ) {
        DataType dataType = outputNode.dataType();
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16) {
            throw new UnsupportedOperationException("cpu1 MATMUL does not support output dtype " + dataType);
        }
        if (matmulNode.dataType() != dataType || left.dataType() != dataType || right.dataType() != dataType) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL requires matching input/output dtype. left="
                    + left.dataType() + ", right=" + right.dataType() + ", matmul=" + matmulNode.dataType()
                    + ", output=" + dataType);
        }
        if (!Arrays.equals(matmulNode.shape(), outputNode.shape())) {
            throw new UnsupportedOperationException("cpu1 MATMUL post-op output shape must match matmul shape. matmul="
                    + Arrays.toString(matmulNode.shape()) + ", output=" + Arrays.toString(outputNode.shape()));
        }
        if (!left.denseContiguousWithoutOffset() || !right.denseContiguousWithoutOffset()) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL supports only dense contiguous inputs without storage offset.");
        }
        if (!outputNode.contiguous() || outputNode.hasStorageOffset()) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL supports only dense contiguous output without storage offset.");
        }
    }

    private static void requireBiasContract(
            CompiledNode addNode,
            CompiledNode outputNode,
            CompiledTensorDescriptor bias
    ) {
        Objects.requireNonNull(addNode, "addNode cannot be null");
        Objects.requireNonNull(bias, "bias cannot be null");
        if (!Arrays.equals(addNode.shape(), outputNode.shape())) {
            throw new UnsupportedOperationException("cpu1 MATMUL ADD_BIAS_RELU ADD output shape must match RELU output. add="
                    + Arrays.toString(addNode.shape()) + ", output=" + Arrays.toString(outputNode.shape()));
        }
        if (addNode.dataType() != outputNode.dataType() || bias.dataType() != outputNode.dataType()) {
            throw new UnsupportedOperationException("cpu1 MATMUL ADD_BIAS_RELU requires matching add/bias/output dtype. add="
                    + addNode.dataType() + ", bias=" + bias.dataType() + ", output=" + outputNode.dataType());
        }
        if (!bias.denseContiguousWithoutOffset()) {
            throw new UnsupportedOperationException("cpu1 MATMUL ADD_BIAS_RELU supports only dense contiguous bias without storage offset.");
        }
    }

    private static void requireStorageContract(Cpu1MatmulRoute route, Cpu1StorageKind storageKind) {
        if (route == Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT) {
            if (storageKind != Cpu1StorageKind.MEMORY_SEGMENT) {
                throw new UnsupportedOperationException("cpu1 OPENBLAS_NATIVE_SEGMENT MATMUL requires MEMORY_SEGMENT storage, got "
                        + storageKind);
            }
            return;
        }
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY) {
            throw new UnsupportedOperationException("cpu1 " + route + " MATMUL supports only JAVA_ARRAY storage, got "
                    + storageKind);
        }
    }

    private static void requirePostOpContract(Cpu1MatmulRoute route, Cpu1MatmulPostOp postOp) {
        if (!postOp.supportedBy(route)) {
            throw new UnsupportedOperationException("cpu1 " + route + " MATMUL does not support post-op " + postOp);
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

    private static int[] broadcastStrides(
            int[] inputShape,
            int[] inputStrides,
            int[] outputShape,
            String label
    ) {
        if (inputShape.length > outputShape.length) {
            throw new UnsupportedOperationException("cpu1 MATMUL cannot broadcast " + label + " rank "
                    + inputShape.length + " to output rank " + outputShape.length);
        }
        int[] out = new int[outputShape.length];
        int shift = outputShape.length - inputShape.length;
        for (int dim = 0; dim < outputShape.length; dim++) {
            int inputDim = dim - shift;
            int sourceSize = inputDim < 0 ? 1 : inputShape[inputDim];
            int sourceStride = inputDim < 0 ? 0 : inputStrides[inputDim];
            int targetSize = outputShape[dim];
            if (sourceSize == targetSize) {
                out[dim] = sourceStride;
            } else if (sourceSize == 1) {
                out[dim] = 0;
            } else {
                throw new UnsupportedOperationException("cpu1 MATMUL cannot broadcast " + label + " shape "
                        + Arrays.toString(inputShape) + " to output shape " + Arrays.toString(outputShape));
            }
        }
        return out;
    }

    private static int[] batchOffsetsFromBroadcastStrides(int[] outputShape, int[] broadcastStrides) {
        int outputBatchRank = outputShape.length - 2;
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
                offset += coordinate * broadcastStrides[dim];
            }
            offsets[batch] = offset;
        }
        return offsets;
    }

    private static int biasNodeId(CompiledNode addNode, int matmulNodeId) {
        int first = addNode.inputIds().get(0);
        int second = addNode.inputIds().get(1);
        if (first == matmulNodeId && second != matmulNodeId) {
            return second;
        }
        if (second == matmulNodeId && first != matmulNodeId) {
            return first;
        }
        throw new UnsupportedOperationException("cpu1 MATMUL ADD_BIAS_RELU ADD must consume the MATMUL output once.");
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
            Cpu1MatmulPostOp postOp,
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
        if (postOp != Cpu1MatmulPostOp.NONE) {
            return Cpu1MatmulRoute.JAVA_SCALAR;
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

    private static int openBlasThreads(Cpu1MatmulRoute route, BlasConfig blasConfig) {
        return switch (route) {
            case OPENBLAS_ARRAY_COPYING -> blasConfig.openBlasArrayCopyEffectiveThreads();
            case OPENBLAS_NATIVE_SEGMENT -> blasConfig.openBlasNativeSegmentEffectiveThreads();
            case JAVA_SCALAR, AUTO -> 0;
        };
    }

    private static long matmulWork(int batchCount, int m, int n, int k) {
        return Math.multiplyExact(Math.multiplyExact(Math.multiplyExact((long) batchCount, m), n), k);
    }

    private static Cpu1ScratchBufferSpec scratchBufferSpec(
            Cpu1MatmulKernelId kernelId,
            int batchCount,
            int n,
            int k
    ) {
        return switch (kernelId) {
            case MATMUL_F32_DENSE_PACKED_B_VECTOR -> Cpu1ScratchBufferSpec.arrays(packedBElements(batchCount, n, k), 0, 0);
            case MATMUL_F64_DENSE_PACKED_B_VECTOR -> Cpu1ScratchBufferSpec.arrays(0, packedBElements(batchCount, n, k), 0);
            default -> Cpu1ScratchBufferSpec.none();
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
