package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;

/**
 * Factory for storage-specific prepared dtype executable units.
 */
public final class Cpu1DTypeExecutableUnits {
    private Cpu1DTypeExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedDTypeUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1DTypeJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1DTypeMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 dtype storage kind " + preparedUnit.storageKind());
        };
    }
}
