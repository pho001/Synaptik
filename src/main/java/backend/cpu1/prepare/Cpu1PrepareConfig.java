package backend.cpu1.prepare;

import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.storage.Cpu1StorageKind;

import java.util.Objects;

/**
 * Prepare-time variant selection for experimental cpu1 units.
 */
public record Cpu1PrepareConfig(
        Cpu1VectorizationKind vectorizationKind,
        Cpu1LaunchConfig launchConfig,
        Cpu1StorageKind storageKind,
        boolean useFastExpApprox,
        boolean useFastTanhApprox
) {
    public Cpu1PrepareConfig(Cpu1VectorizationKind vectorizationKind, Cpu1LaunchConfig launchConfig) {
        this(vectorizationKind, launchConfig, Cpu1StorageKind.JAVA_ARRAY);
    }

    public Cpu1PrepareConfig(
            Cpu1VectorizationKind vectorizationKind,
            Cpu1LaunchConfig launchConfig,
            Cpu1StorageKind storageKind
    ) {
        this(vectorizationKind, launchConfig, storageKind, false, false);
    }

    public Cpu1PrepareConfig {
        Objects.requireNonNull(vectorizationKind, "vectorizationKind cannot be null");
        Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        Objects.requireNonNull(storageKind, "storageKind cannot be null");
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

    public Cpu1PrepareConfig withApproximation(boolean useFastExpApprox, boolean useFastTanhApprox) {
        return new Cpu1PrepareConfig(
                vectorizationKind,
                launchConfig,
                storageKind,
                useFastExpApprox,
                useFastTanhApprox
        );
    }
}
