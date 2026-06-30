package backend.cpu1.exec;

import java.lang.foreign.MemorySegment;

/**
 * Run-scoped scratch memory for a prepared cpu1 executable unit.
 */
public final class Cpu1ScratchBuffer {
    private final Cpu1ScratchBufferSpec spec;
    private final float[] f32Array;
    private final double[] f64Array;
    private final int[] i32Array;
    private final MemorySegment segment;
    private final Cpu1ProviderCache providerCache;

    private Cpu1ScratchBuffer(
            Cpu1ScratchBufferSpec spec,
            float[] f32Array,
            double[] f64Array,
            int[] i32Array,
            MemorySegment segment,
            Cpu1ProviderCache providerCache
    ) {
        if (spec == null) {
            throw new IllegalArgumentException("spec cannot be null");
        }
        this.spec = spec;
        this.f32Array = f32Array;
        this.f64Array = f64Array;
        this.i32Array = i32Array;
        this.segment = segment;
        this.providerCache = providerCache;
    }

    public static Cpu1ScratchBuffer allocate(Cpu1ScratchBufferSpec spec) {
        if (spec == null) {
            throw new IllegalArgumentException("spec cannot be null");
        }
        return new Cpu1ScratchBuffer(
                spec,
                spec.f32ArrayElements() == 0 ? null : new float[spec.f32ArrayElements()],
                spec.f64ArrayElements() == 0 ? null : new double[spec.f64ArrayElements()],
                spec.i32ArrayElements() == 0 ? null : new int[spec.i32ArrayElements()],
                spec.segmentBytes() == 0L ? null : MemorySegment.ofArray(new byte[Math.toIntExact(spec.segmentBytes())]),
                spec.needsProviderCache() ? new Cpu1ProviderCache() : null
        );
    }

    public Cpu1ScratchBufferSpec spec() {
        return spec;
    }

    public float[] requireF32Array() {
        return requireF32Array(0);
    }

    public float[] requireF32Array(int requiredElements) {
        requireNonNegative(requiredElements, "requiredElements");
        if (f32Array == null || f32Array.length < requiredElements) {
            throw new IllegalStateException("cpu1 scratch buffer does not provide enough F32 array scratch. required="
                    + requiredElements + ", actual=" + (f32Array == null ? 0 : f32Array.length));
        }
        return f32Array;
    }

    public double[] requireF64Array() {
        return requireF64Array(0);
    }

    public double[] requireF64Array(int requiredElements) {
        requireNonNegative(requiredElements, "requiredElements");
        if (f64Array == null || f64Array.length < requiredElements) {
            throw new IllegalStateException("cpu1 scratch buffer does not provide enough F64 array scratch. required="
                    + requiredElements + ", actual=" + (f64Array == null ? 0 : f64Array.length));
        }
        return f64Array;
    }

    public int[] requireI32Array() {
        return requireI32Array(0);
    }

    public int[] requireI32Array(int requiredElements) {
        requireNonNegative(requiredElements, "requiredElements");
        if (i32Array == null || i32Array.length < requiredElements) {
            throw new IllegalStateException("cpu1 scratch buffer does not provide enough I32 array scratch. required="
                    + requiredElements + ", actual=" + (i32Array == null ? 0 : i32Array.length));
        }
        return i32Array;
    }

    public MemorySegment requireSegment() {
        return requireSegment(0L);
    }

    public MemorySegment requireSegment(long requiredBytes) {
        requireNonNegative(requiredBytes, "requiredBytes");
        if (segment == null || segment.byteSize() < requiredBytes) {
            throw new IllegalStateException("cpu1 scratch buffer does not provide enough MemorySegment scratch. requiredBytes="
                    + requiredBytes + ", actualBytes=" + (segment == null ? 0L : segment.byteSize()));
        }
        return segment;
    }

    public Cpu1ProviderCache providerCache() {
        if (providerCache == null) {
            throw new IllegalStateException("cpu1 scratch buffer does not provide provider cache.");
        }
        return providerCache;
    }

    private static void requireNonNegative(long value, String name) {
        if (value < 0L) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
    }
}
