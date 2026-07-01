package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedLayerNormUnit;

/**
 * Factory for storage-specific prepared LayerNorm executable units.
 */
public final class Cpu1LayerNormExecutableUnits {
    private Cpu1LayerNormExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedLayerNormUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1LayerNormJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1LayerNormMemorySegmentExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 LayerNorm storage kind " + preparedUnit.storageKind());
        };
    }
}
