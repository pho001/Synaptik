package backend.blas;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Row-major CBLAS range math shared by OpenBLAS array and segment GEMM callers.
 */
final class OpenBlasGemmLayout {
    private OpenBlasGemmLayout() {
    }

    static int requiredElements(int rows, int leadingDim) {
        return Math.max(0, rows) * Math.max(0, leadingDim);
    }

    static MemorySegment segmentSlice(
            MemorySegment segment,
            long byteOffset,
            int elements,
            int elementBytes,
            String name
    ) {
        Objects.requireNonNull(segment, name + " segment cannot be null");
        long byteLength = Math.multiplyExact(Math.max(0L, elements), Math.max(1L, elementBytes));
        if (byteOffset < 0L || byteLength < 0L || byteOffset > segment.byteSize() || segment.byteSize() - byteOffset < byteLength) {
            throw new IllegalArgumentException("OpenBLAS segment slice is outside backing storage for " + name
                    + ". byteOffset=" + byteOffset
                    + ", byteLength=" + byteLength
                    + ", segmentBytes=" + segment.byteSize());
        }
        return segment.asSlice(byteOffset, byteLength);
    }
}
