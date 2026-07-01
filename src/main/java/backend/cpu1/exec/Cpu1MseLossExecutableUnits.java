package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedMseLossUnit;

/**
 * Factory for storage-specific prepared MSE loss executable units.
 */
public final class Cpu1MseLossExecutableUnits {
    private Cpu1MseLossExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedMseLossUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1MseLossJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1MseLossMemorySegmentExecutableUnit(preparedUnit);
        };
    }
}
