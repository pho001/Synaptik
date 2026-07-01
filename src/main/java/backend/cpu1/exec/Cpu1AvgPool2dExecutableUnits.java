package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAvgPool2dUnit;

/**
 * Factory for storage-specific prepared AVG_POOL2D executable units.
 */
public final class Cpu1AvgPool2dExecutableUnits {
    private Cpu1AvgPool2dExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedAvgPool2dUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1AvgPool2dJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1AvgPool2dMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 AVG_POOL2D storage kind " + preparedUnit.storageKind());
        };
    }
}
