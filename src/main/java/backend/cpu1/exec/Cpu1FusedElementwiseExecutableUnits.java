package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;

/**
 * Factory for storage-specific prepared generated fused elementwise executable units.
 */
public final class Cpu1FusedElementwiseExecutableUnits {
    private Cpu1FusedElementwiseExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedFusedElementwiseUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1FusedJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1FusedMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 fused storage kind " + preparedUnit.storageKind());
        };
    }
}
