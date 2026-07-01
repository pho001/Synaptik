package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import backend.cpu1.storage.Cpu1StorageKind;

/**
 * Base runtime wrapper for a prepared cpu1 index node.
 */
public abstract class Cpu1IndexExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedIndexUnit preparedUnit;

    protected Cpu1IndexExecutableUnit(Cpu1PreparedIndexUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedIndexUnit preparedUnit() {
        return preparedUnit;
    }

    protected static void requireArrayView(String role, Cpu1TensorView view) {
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 index dense slice supports only JAVA_ARRAY runtime "
                + role + " storage, got " + view.storageKind());
    }

    protected static void requireSegmentView(String role, Cpu1TensorView view) {
        if (view.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 index dense slice supports only MEMORY_SEGMENT runtime "
                + role + " storage, got " + view.storageKind());
    }
}
