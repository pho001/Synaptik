package backend.cpu1.exec;

import backend.runtime.ExecutionContext;
import tensor.DataType;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Run-scoped scratch memory for a prepared cpu1 executable unit.
 */
public final class Cpu1Workspace {
    private final Cpu1WorkspaceSpec spec;
    private final float[] f32Array;
    private final double[] f64Array;
    private final int[] i32Array;
    private final MemorySegment segment;
    private final Cpu1ProviderCache providerCache;
    private NativeTensorStorage nativeOutputStorage;

    private Cpu1Workspace(
            Cpu1WorkspaceSpec spec,
            float[] f32Array,
            double[] f64Array,
            int[] i32Array,
            MemorySegment segment,
            Cpu1ProviderCache providerCache
    ) {
        this.spec = Objects.requireNonNull(spec, "spec cannot be null");
        this.f32Array = f32Array;
        this.f64Array = f64Array;
        this.i32Array = i32Array;
        this.segment = segment;
        this.providerCache = providerCache;
    }

    public static Cpu1Workspace allocate(Cpu1WorkspaceSpec spec) {
        Objects.requireNonNull(spec, "spec cannot be null");
        if (spec.isEmpty()) {
            throw new IllegalArgumentException("Cannot allocate empty cpu1 workspace.");
        }
        return new Cpu1Workspace(
                spec,
                spec.f32ArrayElements() == 0 ? null : new float[spec.f32ArrayElements()],
                spec.f64ArrayElements() == 0 ? null : new double[spec.f64ArrayElements()],
                spec.i32ArrayElements() == 0 ? null : new int[spec.i32ArrayElements()],
                spec.segmentBytes() == 0L ? null : MemorySegment.ofArray(new byte[Math.toIntExact(spec.segmentBytes())]),
                spec.needsProviderCache() ? new Cpu1ProviderCache() : null
        );
    }

    public Cpu1WorkspaceSpec spec() {
        return spec;
    }

    public float[] requireF32Array() {
        return requireF32Array(0);
    }

    public float[] requireF32Array(int requiredElements) {
        requireNonNegative(requiredElements, "requiredElements");
        if (f32Array == null || f32Array.length < requiredElements) {
            throw new IllegalStateException("cpu1 workspace does not provide enough F32 array scratch. required="
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
            throw new IllegalStateException("cpu1 workspace does not provide enough F64 array scratch. required="
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
            throw new IllegalStateException("cpu1 workspace does not provide enough I32 array scratch. required="
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
            throw new IllegalStateException("cpu1 workspace does not provide enough MemorySegment scratch. requiredBytes="
                    + requiredBytes + ", actualBytes=" + (segment == null ? 0L : segment.byteSize()));
        }
        return segment;
    }

    public NativeTensorStorage requireNativeOutputStorage(
            DataType dataType,
            int requiredElements,
            int nodeId,
            ExecutionContext context,
            String label
    ) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        requireNonNegative(requiredElements, "requiredElements");
        if (spec.nativeOutputDataType() == null) {
            throw new IllegalStateException("cpu1 workspace does not provide native output storage.");
        }
        if (spec.nativeOutputDataType() != dataType || spec.nativeOutputElements() != requiredElements) {
            throw new IllegalStateException("cpu1 workspace native output storage mismatch. expected="
                    + spec.nativeOutputDataType() + "[" + spec.nativeOutputElements() + "], requested="
                    + dataType + "[" + requiredElements + "]");
        }
        if (nativeOutputStorage == null) {
            nativeOutputStorage = context.allocateNativeStorage(dataType, requiredElements, label);
        }
        if (nativeOutputStorage.getType() != dataType || nativeOutputStorage.getSize() != requiredElements) {
            throw new IllegalStateException("cpu1 workspace native output storage is invalid for nodeId=" + nodeId
                    + ". expected=" + dataType + "[" + requiredElements + "], actual="
                    + nativeOutputStorage.getType() + "[" + nativeOutputStorage.getSize() + "]");
        }
        nativeOutputStorage.ensureOpen();
        context.reserveNativeOutputStorage(nodeId, nativeOutputStorage);
        return nativeOutputStorage;
    }

    public Cpu1ProviderCache providerCache() {
        if (providerCache == null) {
            throw new IllegalStateException("cpu1 workspace does not provide provider cache.");
        }
        return providerCache;
    }

    private static void requireNonNegative(long value, String name) {
        if (value < 0L) {
            throw new IllegalArgumentException(name + " cannot be negative: " + value);
        }
    }
}
