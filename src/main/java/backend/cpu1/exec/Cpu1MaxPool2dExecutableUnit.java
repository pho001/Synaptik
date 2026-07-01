package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedMaxPool2dUnit;
import backend.cpu1.storage.Cpu1StorageKind;

import java.util.Arrays;

/**
 * Base runtime wrapper for a prepared cpu1 MAX_POOL2D node.
 */
public abstract class Cpu1MaxPool2dExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedMaxPool2dUnit preparedUnit;

    protected Cpu1MaxPool2dExecutableUnit(Cpu1PreparedMaxPool2dUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedMaxPool2dUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return preparedUnit.scratchBufferSpec();
    }

    protected void validateViews(Cpu1TensorView input, Cpu1TensorView output) {
        requireShape("input", input.shape(), preparedUnit.inputShape());
        requireShape("output", output.shape(), preparedUnit.outputShape());
        if (output.elementCount() != preparedUnit.outputElementCount()) {
            throw new UnsupportedOperationException("cpu1 MAX_POOL2D output element count mismatch. expected="
                    + preparedUnit.outputElementCount() + ", actual=" + output.elementCount());
        }
    }

    protected static void requireDenseNoOffset(
            String role,
            Cpu1TensorView view,
            Cpu1StorageKind expectedStorageKind
    ) {
        if (view.storageKind() != expectedStorageKind) {
            throw new UnsupportedOperationException("cpu1 MAX_POOL2D dense direct route supports only "
                    + expectedStorageKind + " " + role + " runtime storage, got " + view.storageKind());
        }
        if (!view.contiguous() || view.storageOffset() != 0) {
            throw new UnsupportedOperationException("cpu1 MAX_POOL2D dense direct route supports only dense "
                    + "contiguous no-offset " + role + " runtime view; contiguous=" + view.contiguous()
                    + ", storageOffset=" + view.storageOffset());
        }
    }

    private static void requireShape(String role, int[] actual, int[] expected) {
        if (Arrays.equals(actual, expected)) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 MAX_POOL2D " + role + " shape mismatch. expected="
                + Arrays.toString(expected) + ", actual=" + Arrays.toString(actual));
    }
}
