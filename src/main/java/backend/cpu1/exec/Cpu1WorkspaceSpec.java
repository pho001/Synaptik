package backend.cpu1.exec;

/**
 * Exact prepare-time scratch requirements for one cpu1 executable unit.
 */
public record Cpu1WorkspaceSpec(
        int f32ArrayElements,
        int f64ArrayElements,
        int i32ArrayElements,
        long segmentBytes,
        boolean needsProviderCache
) {
    private static final Cpu1WorkspaceSpec NONE = new Cpu1WorkspaceSpec(0, 0, 0, 0L, false);

    public Cpu1WorkspaceSpec {
        requireNonNegative(f32ArrayElements, "f32ArrayElements");
        requireNonNegative(f64ArrayElements, "f64ArrayElements");
        requireNonNegative(i32ArrayElements, "i32ArrayElements");
        requireNonNegative(segmentBytes, "segmentBytes");
        if (segmentBytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("segmentBytes currently must fit a heap-backed MemorySegment: " + segmentBytes);
        }
    }

    public static Cpu1WorkspaceSpec none() {
        return NONE;
    }

    public static Cpu1WorkspaceSpec arrays(int f32ArrayElements, int f64ArrayElements, int i32ArrayElements) {
        return new Cpu1WorkspaceSpec(f32ArrayElements, f64ArrayElements, i32ArrayElements, 0L, false);
    }

    public static Cpu1WorkspaceSpec segment(long segmentBytes) {
        return new Cpu1WorkspaceSpec(0, 0, 0, segmentBytes, false);
    }

    public static Cpu1WorkspaceSpec providerCache() {
        return new Cpu1WorkspaceSpec(0, 0, 0, 0L, true);
    }

    public Cpu1WorkspaceSpec withSegmentBytes(long segmentBytes) {
        return new Cpu1WorkspaceSpec(f32ArrayElements, f64ArrayElements, i32ArrayElements, segmentBytes, needsProviderCache);
    }

    public Cpu1WorkspaceSpec withProviderCache() {
        return new Cpu1WorkspaceSpec(f32ArrayElements, f64ArrayElements, i32ArrayElements, segmentBytes, true);
    }

    public boolean isEmpty() {
        return f32ArrayElements == 0
                && f64ArrayElements == 0
                && i32ArrayElements == 0
                && segmentBytes == 0L
                && !needsProviderCache;
    }

    private static void requireNonNegative(long value, String name) {
        if (value < 0L) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
    }
}
