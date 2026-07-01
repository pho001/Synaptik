package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedReductionUnit;

/**
 * Factory for storage-specific prepared reduction executable units.
 */
public final class Cpu1ReductionExecutableUnits {
    private Cpu1ReductionExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedReductionUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> new Cpu1ReductionJavaArrayExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> new Cpu1ReductionMemorySegmentExecutableUnit(preparedUnit);
        };
    }
}
