package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedRmsNormUnit;

/**
 * Factory for storage-specific prepared RMSNorm executable units.
 */
public final class Cpu1RmsNormExecutableUnits {
    private Cpu1RmsNormExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedRmsNormUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1RmsNormJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1RmsNormMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 RMSNorm storage kind " + preparedUnit.storageKind());
        };
    }
}
