package backend.cpu1.prepare;

import runtime.contract.ExecutionMode;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import tensor.DataType;

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
        Cpu1MatmulRoute matmulRoute,
        BlasConfig blasConfig
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
                Cpu1MatmulRoute.JAVA_SCALAR,
                BlasConfig.disabled()
        );
    }

    public Cpu1PrepareConfig {
        if (vectorizationKind == null) {
            throw new IllegalArgumentException("vectorizationKind cannot be null");
        }
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (matmulRoute == null) {
            throw new IllegalArgumentException("matmulRoute cannot be null");
        }
        blasConfig = blasConfig == null ? BlasConfig.disabled() : blasConfig;
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
        return automatic(runtimeConfig, maxWorkerCount);
    }

    public static Cpu1PrepareConfig automatic(RuntimeConfig runtimeConfig, int maxWorkerCount) {
        return automatic(runtimeConfig, maxWorkerCount, Cpu1StorageKind.JAVA_ARRAY);
    }

    public static Cpu1PrepareConfig automaticForRuntimeStorage(RuntimeConfig runtimeConfig, int maxWorkerCount) {
        return automatic(runtimeConfig, maxWorkerCount, storageKindFor(runtimeConfig));
    }

    public static Cpu1StorageKind storageKindFor(RuntimeConfig runtimeConfig) {
        if (runtimeConfig == null) {
            throw new IllegalArgumentException("runtimeConfig cannot be null");
        }
        return switch (runtimeConfig.cpuStorageProfile()) {
            case CPU_NATIVE -> Cpu1StorageKind.MEMORY_SEGMENT;
            case CPU_ARRAY, AUTO -> Cpu1StorageKind.JAVA_ARRAY;
        };
    }

    public static Cpu1PrepareConfig automatic(
            RuntimeConfig runtimeConfig,
            int maxWorkerCount,
            Cpu1StorageKind storageKind
    ) {
        if (runtimeConfig == null) {
            throw new IllegalArgumentException("runtimeConfig cannot be null");
        }
        return automatic(runtimeConfig.cpuKernelConfig(), maxWorkerCount, storageKind)
                .withBlasConfig(runtimeConfig.blas());
    }

    public static Cpu1PrepareConfig automatic(CpuKernelConfig cpuKernelConfig, int maxWorkerCount) {
        return automatic(cpuKernelConfig, maxWorkerCount, Cpu1StorageKind.JAVA_ARRAY);
    }

    public static Cpu1PrepareConfig automatic(CpuKernelConfig cpuKernelConfig, int maxWorkerCount, Cpu1StorageKind storageKind) {
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("cpuKernelConfig cannot be null");
        }
        return new Cpu1PrepareConfig(
                Cpu1VectorizationKind.VECTOR,
                Cpu1LaunchConfig.parallel(maxWorkerCount),
                storageKind,
                false,
                false,
                true,
                true,
                cpuKernelConfig,
                Cpu1MatmulRoute.JAVA_SCALAR,
                BlasConfig.disabled()
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
                matmulRoute,
                blasConfig
        );
    }

    public Cpu1PrepareConfig withMatmulRoute(Cpu1MatmulRoute route) {
        if (route == null) {
            throw new IllegalArgumentException("route cannot be null");
        }
        return new Cpu1PrepareConfig(
                vectorizationKind,
                launchConfig,
                storageKind,
                useFastExpApprox,
                useFastTanhApprox,
                automaticVectorization,
                automaticLaunch,
                cpuKernelConfig,
                route,
                blasConfig
        );
    }

    public Cpu1PrepareConfig withBlasConfig(BlasConfig blasConfig) {
        return new Cpu1PrepareConfig(
                vectorizationKind,
                launchConfig,
                storageKind,
                useFastExpApprox,
                useFastTanhApprox,
                automaticVectorization,
                automaticLaunch,
                cpuKernelConfig,
                matmulRoute,
                blasConfig
        );
    }
}
