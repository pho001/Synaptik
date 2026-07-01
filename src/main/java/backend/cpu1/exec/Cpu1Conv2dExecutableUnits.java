package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedConv2dUnit;

/**
 * Factory for storage-specific prepared CONV2D executable units.
 */
public final class Cpu1Conv2dExecutableUnits {
    private Cpu1Conv2dExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedConv2dUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1Conv2dJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1Conv2dMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 CONV2D storage kind " + preparedUnit.storageKind());
        };
    }
}
