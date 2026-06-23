package backend.cpu1.exec;

import backend.cpu1.kernels.layout.Cpu1LayoutKernelId;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;

/**
 * Factory for alias/materializing prepared layout executable units.
 */
public final class Cpu1LayoutExecutableUnits {
    private Cpu1LayoutExecutableUnits() {
    }

    public static Cpu1ExecutableUnit create(Cpu1PreparedLayoutUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        boolean aliasKernel = isAliasKernel(preparedUnit.kernelId());
        return switch (preparedUnit.storageKind()) {
            case JAVA_ARRAY -> aliasKernel
                    ? new Cpu1LayoutJavaArrayAliasExecutableUnit(preparedUnit)
                    : new Cpu1LayoutJavaArrayMaterializingExecutableUnit(preparedUnit);
            case MEMORY_SEGMENT -> aliasKernel
                    ? new Cpu1LayoutMemorySegmentAliasExecutableUnit(preparedUnit)
                    : new Cpu1LayoutMemorySegmentMaterializingExecutableUnit(preparedUnit);
            default -> throw new UnsupportedOperationException(
                    "Unsupported cpu1 layout storage kind " + preparedUnit.storageKind());
        };
    }

    private static boolean isAliasKernel(Cpu1LayoutKernelId kernelId) {
        return switch (kernelId) {
            case NOOP_ALIAS,
                 RESHAPE_ALIAS,
                 EXPAND_ALIAS,
                 SELECT_ALIAS,
                 SLICE_ALIAS,
                 PERMUTE_ALIAS,
                 EXPAND_DIMS_ALIAS,
                 SQUEEZE_ALIAS -> true;
            default -> false;
        };
    }
}
