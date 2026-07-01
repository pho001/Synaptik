package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;

/**
 * Factory for storage-specific prepared attention backward executable units.
 */
public final class Cpu1AttentionBackwardExecutableUnits {
    private Cpu1AttentionBackwardExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedAttentionBackwardUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1AttentionBackwardJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1AttentionBackwardMemorySegmentExecutableUnit(preparedUnit);
        };
    }
}
