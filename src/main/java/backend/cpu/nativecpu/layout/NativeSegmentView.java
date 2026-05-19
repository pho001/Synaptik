package backend.cpu.nativecpu.layout;

import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Lowered CPU-native view over a {@link NativeTensorStorage} and its {@link MemorySegment}.
 */
public record NativeSegmentView(
        TensorPhysicalView physicalView,
        NativeTensorStorage storage,
        MemorySegment segment,
        long baseByteOffset,
        long[] byteStrides,
        long elementSizeBytes,
        long physicalByteSpan
) {
    public NativeSegmentView {
        Objects.requireNonNull(physicalView, "physicalView cannot be null");
        Objects.requireNonNull(storage, "storage cannot be null");
        Objects.requireNonNull(segment, "segment cannot be null");
        Objects.requireNonNull(byteStrides, "byteStrides cannot be null");
        if (physicalView.storageFamily() != NativeCpuStorageFamily.CPU_NATIVE) {
            throw new IllegalArgumentException("physicalView storageFamily must be CPU_NATIVE");
        }
        if (storage.getType() != physicalView.dataType()) {
            throw new IllegalArgumentException("native storage dtype " + storage.getType()
                    + " does not match physical view dtype " + physicalView.dataType());
        }
        if (byteStrides.length != physicalView.elementStrides().length) {
            throw new IllegalArgumentException("byteStrides length must match physical view rank");
        }
        if (elementSizeBytes != storage.elementSizeBytes()
                || elementSizeBytes != physicalView.elementSizeBytes()) {
            throw new IllegalArgumentException("elementSizeBytes mismatch");
        }
        if (baseByteOffset != physicalView.baseByteOffset()) {
            throw new IllegalArgumentException("baseByteOffset must match physical view storage offset");
        }
        if (physicalByteSpan != physicalView.physicalByteSpan()) {
            throw new IllegalArgumentException("physicalByteSpan must match physical view span");
        }
        if (physicalByteSpan > storage.byteSize()) {
            throw new IllegalArgumentException("native segment view exceeds storage byte size. required="
                    + physicalByteSpan + ", storage=" + storage.byteSize());
        }
        byteStrides = byteStrides.clone();
    }

    public static NativeSegmentView from(TensorPhysicalView physicalView, NativeTensorStorage storage) {
        Objects.requireNonNull(physicalView, "physicalView cannot be null");
        Objects.requireNonNull(storage, "storage cannot be null");
        storage.ensureOpen();
        return new NativeSegmentView(
                physicalView,
                storage,
                storage.segment(),
                physicalView.baseByteOffset(),
                physicalView.byteStrides(),
                storage.elementSizeBytes(),
                physicalView.physicalByteSpan()
        );
    }

    public long byteOffsetForLogicalIndex(int logicalIndex) {
        int[] shape = physicalView.shape();
        long[] strides = byteStrides;
        if (logicalIndex < 0 || logicalIndex >= physicalView.logicalElementCount()) {
            throw new IndexOutOfBoundsException("logicalIndex out of bounds. index="
                    + logicalIndex + ", elements=" + physicalView.logicalElementCount());
        }
        long offset = baseByteOffset;
        int rem = logicalIndex;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            int dimension = shape[dim];
            int coord = dimension == 0 ? 0 : rem % dimension;
            rem = dimension == 0 ? 0 : rem / dimension;
            offset = Math.addExact(offset, Math.multiplyExact((long) coord, strides[dim]));
        }
        return offset;
    }

    @Override
    public long[] byteStrides() {
        return byteStrides.clone();
    }
}
