package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedElementwiseUnit;

/**
 * Factory for storage-specific prepared elementwise executable units.
 */
public final class Cpu1ElementwiseExecutableUnits {
    private Cpu1ElementwiseExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedElementwiseUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1ElementwiseJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1ElementwiseMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 elementwise storage kind " + preparedUnit.storageKind());
        };
    }
}
