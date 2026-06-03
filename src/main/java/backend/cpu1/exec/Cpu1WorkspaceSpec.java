package backend.cpu1.exec;

import tensor.DataType;

/**
 * Exact prepare-time scratch requirements for one cpu1 executable unit.
 */
public record Cpu1WorkspaceSpec(
        int f32ArrayElements,
        int f64ArrayElements,
        int i32ArrayElements,
        long segmentBytes,
        boolean needsProviderCache,
        DataType nativeOutputDataType,
        int nativeOutputElements
) {
    private static final Cpu1WorkspaceSpec NONE = new Cpu1WorkspaceSpec(0, 0, 0, 0L, false, null, 0);

    public Cpu1WorkspaceSpec(
            int f32ArrayElements,
            int f64ArrayElements,
            int i32ArrayElements,
            long segmentBytes,
            boolean needsProviderCache
    ) {
        this(f32ArrayElements, f64ArrayElements, i32ArrayElements, segmentBytes, needsProviderCache, null, 0);
    }

    public Cpu1WorkspaceSpec {
        requireNonNegative(f32ArrayElements, "f32ArrayElements");
        requireNonNegative(f64ArrayElements, "f64ArrayElements");
        requireNonNegative(i32ArrayElements, "i32ArrayElements");
        requireNonNegative(segmentBytes, "segmentBytes");
        requireNonNegative(nativeOutputElements, "nativeOutputElements");
        if (segmentBytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("segmentBytes currently must fit a heap-backed MemorySegment: " + segmentBytes);
        }
        if (nativeOutputDataType == null && nativeOutputElements != 0) {
            throw new IllegalArgumentException("nativeOutputElements requires nativeOutputDataType.");
        }
    }

    public static Cpu1WorkspaceSpec none() {
        return NONE;
    }

    public static Cpu1WorkspaceSpec arrays(int f32ArrayElements, int f64ArrayElements, int i32ArrayElements) {
        return new Cpu1WorkspaceSpec(f32ArrayElements, f64ArrayElements, i32ArrayElements, 0L, false, null, 0);
    }

    public static Cpu1WorkspaceSpec segment(long segmentBytes) {
        return new Cpu1WorkspaceSpec(0, 0, 0, segmentBytes, false, null, 0);
    }

    public static Cpu1WorkspaceSpec providerCache() {
        return new Cpu1WorkspaceSpec(0, 0, 0, 0L, true, null, 0);
    }

    public static Cpu1WorkspaceSpec nativeOutput(DataType dataType, int elements) {
        return new Cpu1WorkspaceSpec(0, 0, 0, 0L, false, dataType, elements);
    }

    public Cpu1WorkspaceSpec withSegmentBytes(long segmentBytes) {
        return new Cpu1WorkspaceSpec(
                f32ArrayElements,
                f64ArrayElements,
                i32ArrayElements,
                segmentBytes,
                needsProviderCache,
                nativeOutputDataType,
                nativeOutputElements
        );
    }

    public Cpu1WorkspaceSpec withProviderCache() {
        return new Cpu1WorkspaceSpec(
                f32ArrayElements,
                f64ArrayElements,
                i32ArrayElements,
                segmentBytes,
                true,
                nativeOutputDataType,
                nativeOutputElements
        );
    }

    public Cpu1WorkspaceSpec withNativeOutput(DataType dataType, int elements) {
        return new Cpu1WorkspaceSpec(
                f32ArrayElements,
                f64ArrayElements,
                i32ArrayElements,
                segmentBytes,
                needsProviderCache,
                dataType,
                elements
        );
    }

    public boolean isEmpty() {
        return f32ArrayElements == 0
                && f64ArrayElements == 0
                && i32ArrayElements == 0
                && segmentBytes == 0L
                && !needsProviderCache
                && nativeOutputDataType == null;
    }

    private static void requireNonNegative(long value, String name) {
        if (value < 0L) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
    }
}
