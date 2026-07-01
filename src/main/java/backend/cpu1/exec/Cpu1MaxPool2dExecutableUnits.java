package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedMaxPool2dUnit;

/**
 * Factory for storage-specific prepared MAX_POOL2D executable units.
 */
public final class Cpu1MaxPool2dExecutableUnits {
    private Cpu1MaxPool2dExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedMaxPool2dUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1MaxPool2dJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1MaxPool2dMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 MAX_POOL2D storage kind " + preparedUnit.storageKind());
        };
    }
}
