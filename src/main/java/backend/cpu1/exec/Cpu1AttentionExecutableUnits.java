package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionUnit;

/**
 * Factory for storage-specific prepared attention executable units.
 */
public final class Cpu1AttentionExecutableUnits {
    private Cpu1AttentionExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedAttentionUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1AttentionJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1AttentionMemorySegmentExecutableUnit(preparedUnit);
        };
    }
}
