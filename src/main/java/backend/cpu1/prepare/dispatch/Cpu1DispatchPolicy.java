package backend.cpu1.prepare.dispatch;

import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import tensor.DataType;

import java.util.List;

/**
 * Prepare-time policy for selecting cpu1 execution variants from config and node metadata.
 */
public final class Cpu1DispatchPolicy {
    public Cpu1DispatchDecision decideElementwise(
            Operation operation,
            DataType computeType,
            long elementCount,
            Cpu1PrepareConfig config
    ) {
        if (operation == null) {
            throw new IllegalArgumentException("operation cannot be null");
        }
        Operation.OpType opType = operation.opType();
        if (opType == null) {
            throw new IllegalArgumentException("operation opType cannot be null");
        }
        if (computeType == null) {
            throw new IllegalArgumentException("computeType cannot be null");
        }
        if (config == null) {
            throw new IllegalArgumentException("config cannot be null");
        }
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must be >= 0");
        }
        int totalLength = saturatingElementCount(elementCount);
        Operation.OpType kernelOpType = kernelOpType(opType, config);
        Cpu1CostClass costClass = classifyElementwise(operation);
        int vectorWidth = preferredVectorWidth(computeType);
        CpuKernelConfig cpuKernelConfig = cpuKernelConfig(config);
        Cpu1VectorizationKind vectorizationKind = resolveVectorizationKind(
                config,
                cpuKernelConfig,
                costClass,
                config.storageKind(),
                computeType,
                totalLength,
                vectorWidth
        );
        int plannedWorkers = resolvePlannedWorkers(config, cpuKernelConfig, costClass, totalLength);
        int scalarChunkSize = scalarChunkSize(config, cpuKernelConfig, costClass, totalLength, plannedWorkers);
        int vectorChunkSize = vectorChunkSize(config, cpuKernelConfig, costClass, totalLength, vectorWidth, plannedWorkers);
        Cpu1LaunchConfig launchConfig = launchConfig(config, plannedWorkers, vectorizationKind, scalarChunkSize, vectorChunkSize);
        return new Cpu1DispatchDecision(
                kernelOpType,
                costClass,
                vectorizationKind,
                launchConfig,
                config.storageKind(),
                scalarChunkSize,
                vectorChunkSize,
                plannedWorkers
        );
    }

    public Cpu1FusedDispatchDecision decideFusedElementwise(
            Cpu1FusedExpressionPlan plan,
            List<Operation> sourceOperations,
            DataType computeType,
            long elementCount,
            Cpu1PrepareConfig config
    ) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        if (sourceOperations == null || sourceOperations.isEmpty()) {
            throw new IllegalArgumentException("sourceOperations cannot be null or empty");
        }
        if (computeType == null) {
            throw new IllegalArgumentException("computeType cannot be null");
        }
        if (config == null) {
            throw new IllegalArgumentException("config cannot be null");
        }
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must be >= 0");
        }
        Cpu1CostClass costClass = fusedCostClass(sourceOperations);
        int totalLength = saturatingElementCount(elementCount);
        int vectorWidth = preferredVectorWidth(computeType);
        CpuKernelConfig cpuKernelConfig = cpuKernelConfig(config);
        Cpu1VectorizationKind vectorizationKind = resolveFusedVectorizationKind(
                config,
                cpuKernelConfig,
                costClass,
                totalLength,
                vectorWidth
        );
        int plannedWorkers = resolveFusedPlannedWorkers(config, cpuKernelConfig, costClass, totalLength);
        int scalarChunkSize = scalarChunkSize(config, cpuKernelConfig, costClass, totalLength, plannedWorkers);
        int vectorChunkSize = vectorChunkSize(config, cpuKernelConfig, costClass, totalLength, vectorWidth, plannedWorkers);
        Cpu1LaunchConfig launchConfig = launchConfig(config, plannedWorkers, vectorizationKind, scalarChunkSize, vectorChunkSize);
        return new Cpu1FusedDispatchDecision(
                costClass,
                vectorizationKind,
                launchConfig,
                config.storageKind(),
                scalarChunkSize,
                vectorChunkSize,
                plannedWorkers
        );
    }

    public Cpu1VectorizationKind kernelVectorizationKind(
            Cpu1DispatchDecision decision,
            Cpu1LayoutKind layoutKind
    ) {
        if (decision == null) {
            throw new IllegalArgumentException("decision cannot be null");
        }
        if (layoutKind == null) {
            throw new IllegalArgumentException("layoutKind cannot be null");
        }
        if (layoutKind != Cpu1LayoutKind.CONTIGUOUS && layoutKind != Cpu1LayoutKind.BROADCAST_INNER) {
            return Cpu1VectorizationKind.SCALAR;
        }
        if (!supportsVectorKernel(decision.kernelOpType())) {
            return Cpu1VectorizationKind.SCALAR;
        }
        return decision.requestedVectorizationKind();
    }

    public boolean canUseBroadcastInnerVectorLayout(Cpu1DispatchDecision decision) {
        if (decision == null) {
            throw new IllegalArgumentException("decision cannot be null");
        }
        return decision.requestedVectorizationKind() == Cpu1VectorizationKind.VECTOR
                && supportsVectorKernel(decision.kernelOpType());
    }

    private static Operation.OpType kernelOpType(Operation.OpType opType, Cpu1PrepareConfig config) {
        return switch (opType) {
            case EXP -> config.useFastExpApprox() ? Operation.OpType.FAST_EXP : Operation.OpType.EXP;
            case TANH -> config.useFastTanhApprox() ? Operation.OpType.FAST_TANH : Operation.OpType.TANH;
            default -> opType;
        };
    }

    private static Cpu1CostClass classifyElementwise(Operation operation) {
        return switch (operation.computationalCost()) {
            case MEDIUM, EXPENSIVE -> Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
            case TRIVIAL, CHEAP, UNKNOWN -> Cpu1CostClass.CHEAP_ELEMENTWISE;
        };
    }

    private static Cpu1CostClass fusedCostClass(List<Operation> sourceOperations) {
        for (Operation operation : sourceOperations) {
            if (operationCostClass(operation) == Cpu1CostClass.EXPENSIVE_ELEMENTWISE) {
                return Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
            }
        }
        return Cpu1CostClass.CHEAP_ELEMENTWISE;
    }

    private static Cpu1CostClass operationCostClass(Operation operation) {
        if (operation == null) {
            return Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
        }
        return switch (operation.computationalCost()) {
            case MEDIUM, EXPENSIVE -> Cpu1CostClass.EXPENSIVE_ELEMENTWISE;
            case TRIVIAL, CHEAP, UNKNOWN -> Cpu1CostClass.CHEAP_ELEMENTWISE;
        };
    }

    private static Cpu1VectorizationKind resolveVectorizationKind(
            Cpu1PrepareConfig config,
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass,
            Cpu1StorageKind storageKind,
            DataType computeType,
            int totalLength,
            int vectorWidth
    ) {
        if (!config.automaticVectorization()) {
            return config.vectorizationKind();
        }
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("cpuKernelConfig cannot be null");
        }
        if (vectorWidth <= 1) {
            return Cpu1VectorizationKind.SCALAR;
        }
        int minSize = vectorMinSize(cpuKernelConfig, costClass, storageKind, computeType);
        return totalLength >= minSize ? Cpu1VectorizationKind.VECTOR : Cpu1VectorizationKind.SCALAR;
    }

    private static int vectorMinSize(
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass,
            Cpu1StorageKind storageKind,
            DataType computeType
    ) {
        if (costClass == Cpu1CostClass.EXPENSIVE_ELEMENTWISE) {
            return cpuKernelConfig.transcendentalVectorMinSize();
        }
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (computeType) {
                case FLOAT32 -> cpuKernelConfig.nativeF32CheapVectorMinSize();
                case FLOAT64 -> cpuKernelConfig.nativeF64CheapVectorMinSize();
                default -> cpuKernelConfig.cheapVectorMinSize();
            };
        }
        return cpuKernelConfig.cheapVectorMinSize();
    }

    private static Cpu1VectorizationKind resolveFusedVectorizationKind(
            Cpu1PrepareConfig config,
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass,
            int totalLength,
            int vectorWidth
    ) {
        if (!config.automaticVectorization()) {
            return config.vectorizationKind();
        }
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("cpuKernelConfig cannot be null");
        }
        if (vectorWidth <= 1) {
            return Cpu1VectorizationKind.SCALAR;
        }
        int minSize = fusedVectorMinSize(cpuKernelConfig, costClass);
        return totalLength >= minSize ? Cpu1VectorizationKind.VECTOR : Cpu1VectorizationKind.SCALAR;
    }

    private static int fusedVectorMinSize(
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass
    ) {
        return costClass == Cpu1CostClass.EXPENSIVE_ELEMENTWISE
                ? cpuKernelConfig.fusedTranscendentalVectorMinSize()
                : cpuKernelConfig.fusedCheapVectorMinSize();
    }

    private static int resolvePlannedWorkers(
            Cpu1PrepareConfig config,
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass,
            int totalLength
    ) {
        if (!config.automaticLaunch()) {
            return config.launchConfig().workerCount();
        }
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("cpuKernelConfig cannot be null");
        }
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1) {
            return 1;
        }
        int minSize = switch (costClass) {
            case EXPENSIVE_ELEMENTWISE -> cpuKernelConfig.transcendentalParallelMinSize();
            default -> cpuKernelConfig.cheapParallelMinSize();
        };
        if (totalLength < minSize) {
            return 1;
        }
        return Math.min(maxWorkers, Math.max(1, totalLength));
    }

    private static int resolveFusedPlannedWorkers(
            Cpu1PrepareConfig config,
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass,
            int totalLength
    ) {
        if (!config.automaticLaunch()) {
            return config.launchConfig().workerCount();
        }
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("cpuKernelConfig cannot be null");
        }
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1) {
            return 1;
        }
        int minSize = fusedParallelMinSize(cpuKernelConfig, costClass);
        if (totalLength < minSize) {
            return 1;
        }
        return Math.min(maxWorkers, Math.max(1, totalLength));
    }

    private static int fusedParallelMinSize(
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass
    ) {
        return costClass == Cpu1CostClass.EXPENSIVE_ELEMENTWISE
                ? cpuKernelConfig.fusedTranscendentalParallelMinSize()
                : cpuKernelConfig.fusedCheapParallelMinSize();
    }

    private static int scalarChunkSize(
            Cpu1PrepareConfig config,
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass,
            int totalLength,
            int plannedWorkers
    ) {
        if (cpuKernelConfig == null) {
            return explicitChunkSize(config.launchConfig());
        }
        return computeChunkSize(
                totalLength,
                1,
                targetChunksPerWorker(cpuKernelConfig, costClass),
                cpuKernelConfig.minScalarChunkSize(),
                plannedWorkers
        );
    }

    private static int vectorChunkSize(
            Cpu1PrepareConfig config,
            CpuKernelConfig cpuKernelConfig,
            Cpu1CostClass costClass,
            int totalLength,
            int vectorWidth,
            int plannedWorkers
    ) {
        if (cpuKernelConfig == null) {
            return explicitChunkSize(config.launchConfig());
        }
        return computeChunkSize(
                totalLength,
                vectorWidth,
                targetChunksPerWorker(cpuKernelConfig, costClass),
                cpuKernelConfig.minVectorChunkSize(),
                plannedWorkers
        );
    }

    private static Cpu1LaunchConfig launchConfig(
            Cpu1PrepareConfig config,
            int plannedWorkers,
            Cpu1VectorizationKind vectorizationKind,
            int scalarChunkSize,
            int vectorChunkSize
    ) {
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        if (plannedWorkers == 1) {
            return Cpu1LaunchConfig.singleThread();
        }
        return Cpu1LaunchConfig.parallel(
                plannedWorkers,
                vectorizationKind == Cpu1VectorizationKind.VECTOR ? vectorChunkSize : scalarChunkSize
        );
    }

    private static int explicitChunkSize(Cpu1LaunchConfig launchConfig) {
        return launchConfig.hasResolvedChunkSize() ? launchConfig.chunkSize() : 1;
    }

    private static int computeChunkSize(
            int totalLength,
            int alignment,
            int targetChunksPerWorker,
            int minChunkSize,
            int plannedWorkers
    ) {
        int length = Math.max(1, totalLength);
        int workers = Math.max(1, plannedWorkers);
        int targets = Math.max(workers, workers * Math.max(1, targetChunksPerWorker));
        int candidate = (length + targets - 1) / targets;
        int chunk = Math.max(Math.max(1, minChunkSize), candidate);

        int align = Math.max(1, alignment);
        if (align > 1) {
            int remainder = chunk % align;
            if (remainder != 0) {
                chunk += align - remainder;
            }
        }
        return chunk;
    }

    private static int targetChunksPerWorker(CpuKernelConfig cpuKernelConfig, Cpu1CostClass costClass) {
        return switch (costClass) {
            case CHEAP_ELEMENTWISE -> cpuKernelConfig.lowCostTargetChunksPerWorker();
            case EXPENSIVE_ELEMENTWISE -> cpuKernelConfig.mediumCostTargetChunksPerWorker();
            case REDUCTION, MATMUL -> cpuKernelConfig.highCostTargetChunksPerWorker();
        };
    }

    private static CpuKernelConfig cpuKernelConfig(Cpu1PrepareConfig config) {
        if (!config.automaticVectorization() && !config.automaticLaunch()) {
            return null;
        }
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 dispatch requires CpuKernelConfig-backed prepare config.");
        }
        return cpuKernelConfig;
    }

    private static int preferredVectorWidth(DataType computeType) {
        return switch (computeType) {
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            case FLOAT32, BFLOAT16 -> FloatVector.SPECIES_PREFERRED.length();
            case BOOL, INT32, INT64 -> 1;
        };
    }

    private static int saturatingElementCount(long elementCount) {
        return (int) Math.min(Integer.MAX_VALUE, Math.max(0L, elementCount));
    }

    private static boolean supportsVectorKernel(Operation.OpType kernelOpType) {
        return switch (kernelOpType) {
            case FAST_EXP, FAST_TANH, POW_TENSOR, POW, FLOOR, CEIL, SIGN,
                    ERF, WHERE, GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> false;
            default -> true;
        };
    }
}
