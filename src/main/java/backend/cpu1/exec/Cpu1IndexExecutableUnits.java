package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedIndexUnit;

/**
 * Factory for storage-specific prepared index executable units.
 */
public final class Cpu1IndexExecutableUnits {
    private Cpu1IndexExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedIndexUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1IndexJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1IndexMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 index storage kind " + preparedUnit.storageKind());
        };
    }
}
