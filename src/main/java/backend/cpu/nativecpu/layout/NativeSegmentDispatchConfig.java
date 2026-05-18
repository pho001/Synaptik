package backend.cpu.nativecpu.layout;

/**
 * Dispatch knobs for optional parallel MemorySegment kernels.
 */
public record NativeSegmentDispatchConfig(
        int plannedWorkers,
        int chunkSize,
        int parallelMinElements
) {
    public NativeSegmentDispatchConfig {
        plannedWorkers = Math.max(1, plannedWorkers);
        chunkSize = Math.max(1, chunkSize);
        parallelMinElements = Math.max(1, parallelMinElements);
    }

    public static NativeSegmentDispatchConfig scalar() {
        return new NativeSegmentDispatchConfig(1, Integer.MAX_VALUE, Integer.MAX_VALUE);
    }

    public static NativeSegmentDispatchConfig parallel(int plannedWorkers, int chunkSize, int parallelMinElements) {
        return new NativeSegmentDispatchConfig(plannedWorkers, chunkSize, parallelMinElements);
    }

    boolean parallelEligible(int logicalSize) {
        return plannedWorkers > 1 && logicalSize >= parallelMinElements;
    }

    int chunks(int logicalSize) {
        if (logicalSize <= 0) {
            return 0;
        }
        return (logicalSize + chunkSize - 1) / chunkSize;
    }
}
