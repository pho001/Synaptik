package backend.cpu1.prepare;

import backend.runtime.ExecutionMode;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import config.runtime.RuntimeConfig;
import tensor.DataType;

import java.util.Objects;

/**
 * Prepare-time variant selection for experimental cpu1 units.
 */
public record Cpu1PrepareConfig(
        Cpu1VectorizationKind vectorizationKind,
        Cpu1LaunchConfig launchConfig,
        Cpu1StorageKind storageKind,
        boolean useFastExpApprox,
        boolean useFastTanhApprox,
        boolean automaticVectorization,
        boolean automaticLaunch,
        CpuKernelConfig cpuKernelConfig,
        Cpu1MatmulRoute matmulRoute
) {
    public Cpu1PrepareConfig(Cpu1VectorizationKind vectorizationKind, Cpu1LaunchConfig launchConfig) {
        this(vectorizationKind, launchConfig, Cpu1StorageKind.JAVA_ARRAY);
    }

    public Cpu1PrepareConfig(
            Cpu1VectorizationKind vectorizationKind,
            Cpu1LaunchConfig launchConfig,
            Cpu1StorageKind storageKind
    ) {
        this(vectorizationKind, launchConfig, storageKind, false, false, false, false, null);
    }

    public Cpu1PrepareConfig(
            Cpu1VectorizationKind vectorizationKind,
            Cpu1LaunchConfig launchConfig,
            Cpu1StorageKind storageKind,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            boolean automaticVectorization,
            boolean automaticLaunch,
            CpuKernelConfig cpuKernelConfig
    ) {
        this(
                vectorizationKind,
                launchConfig,
                storageKind,
                useFastExpApprox,
                useFastTanhApprox,
                automaticVectorization,
                automaticLaunch,
                cpuKernelConfig,
                Cpu1MatmulRoute.JAVA_SCALAR
        );
    }

    public Cpu1PrepareConfig {
        Objects.requireNonNull(vectorizationKind, "vectorizationKind cannot be null");
        Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        Objects.requireNonNull(storageKind, "storageKind cannot be null");
        Objects.requireNonNull(matmulRoute, "matmulRoute cannot be null");
    }

    public static Cpu1PrepareConfig scalarSingleThread() {
        return new Cpu1PrepareConfig(Cpu1VectorizationKind.SCALAR, Cpu1LaunchConfig.singleThread(), Cpu1StorageKind.JAVA_ARRAY);
    }

    public static Cpu1PrepareConfig vectorSingleThread() {
        return new Cpu1PrepareConfig(Cpu1VectorizationKind.VECTOR, Cpu1LaunchConfig.singleThread(), Cpu1StorageKind.JAVA_ARRAY);
    }

    public static Cpu1PrepareConfig vectorParallel(int workerCount) {
        return new Cpu1PrepareConfig(Cpu1VectorizationKind.VECTOR, Cpu1LaunchConfig.parallel(workerCount), Cpu1StorageKind.JAVA_ARRAY);
    }

    public static Cpu1PrepareConfig scalarMemorySegmentSingleThread() {
        return new Cpu1PrepareConfig(Cpu1VectorizationKind.SCALAR, Cpu1LaunchConfig.singleThread(), Cpu1StorageKind.MEMORY_SEGMENT);
    }

    public static Cpu1PrepareConfig vectorMemorySegmentSingleThread() {
        return new Cpu1PrepareConfig(Cpu1VectorizationKind.VECTOR, Cpu1LaunchConfig.singleThread(), Cpu1StorageKind.MEMORY_SEGMENT);
    }

    public static Cpu1PrepareConfig automatic() {
        return automatic(DataType.FLOAT32, ExecutionMode.FORWARD);
    }

    public static Cpu1PrepareConfig automatic(DataType dataType, ExecutionMode executionMode) {
        return automatic(dataType, executionMode, Runtime.getRuntime().availableProcessors());
    }

    public static Cpu1PrepareConfig automatic(int maxWorkerCount) {
        return automatic(DataType.FLOAT32, ExecutionMode.FORWARD, maxWorkerCount);
    }

    public static Cpu1PrepareConfig automatic(DataType dataType, ExecutionMode executionMode, int maxWorkerCount) {
        RuntimeConfig runtimeConfig = executionMode == ExecutionMode.FORWARD_BACKWARD
                ? RuntimeConfig.trainingDefaults(dataType)
                : RuntimeConfig.inferenceDefaults(dataType);
        return automatic(runtimeConfig.cpuKernelConfig(), maxWorkerCount);
    }

    public static Cpu1PrepareConfig automatic(CpuKernelConfig cpuKernelConfig, int maxWorkerCount) {
        return automatic(cpuKernelConfig, maxWorkerCount, Cpu1StorageKind.JAVA_ARRAY);
    }

    public static Cpu1PrepareConfig automatic(CpuKernelConfig cpuKernelConfig, int maxWorkerCount, Cpu1StorageKind storageKind) {
        return new Cpu1PrepareConfig(
                Cpu1VectorizationKind.VECTOR,
                Cpu1LaunchConfig.parallel(maxWorkerCount),
                storageKind,
                false,
                false,
                true,
                true,
                Objects.requireNonNull(cpuKernelConfig, "cpuKernelConfig cannot be null"),
                Cpu1MatmulRoute.JAVA_SCALAR
        );
    }

    public Cpu1PrepareConfig withApproximation(boolean useFastExpApprox, boolean useFastTanhApprox) {
        return new Cpu1PrepareConfig(
                vectorizationKind,
                launchConfig,
                storageKind,
                useFastExpApprox,
                useFastTanhApprox,
                automaticVectorization,
                automaticLaunch,
                cpuKernelConfig,
                matmulRoute
        );
    }

    public Cpu1PrepareConfig withMatmulRoute(Cpu1MatmulRoute route) {
        return new Cpu1PrepareConfig(
                vectorizationKind,
                launchConfig,
                storageKind,
                useFastExpApprox,
                useFastTanhApprox,
                automaticVectorization,
                automaticLaunch,
                cpuKernelConfig,
                Objects.requireNonNull(route, "route cannot be null")
        );
    }
}
