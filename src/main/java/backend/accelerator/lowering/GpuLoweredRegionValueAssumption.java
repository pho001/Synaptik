package backend.accelerator.lowering;

import tensor.DataType;

import java.util.List;

/**
 * Value-level dtype/layout/storage assumptions captured for a lowered GPU region.
 *
 * @param nodeId graph node id for the value
 * @param role manifest role such as input, primitive, or output
 * @param dataType value dtype
 * @param rank value rank
 * @param shape value logical shape
 * @param layout layout category such as CONTIGUOUS, STRIDED_VIEW, or STORAGE_OFFSET_VIEW
 * @param contiguous whether the value is logically contiguous
 * @param hasStorageOffset whether the value has a non-zero storage offset
 * @param storageOffset storage offset in logical elements
 */
public record GpuLoweredRegionValueAssumption(
        int nodeId,
        String role,
        DataType dataType,
        int rank,
        List<Integer> shape,
        String layout,
        boolean contiguous,
        boolean hasStorageOffset,
        long storageOffset
) {
    public GpuLoweredRegionValueAssumption {
        role = role == null ? "UNKNOWN" : role;
        dataType = dataType == null ? DataType.FLOAT32 : dataType;
        rank = Math.max(0, rank);
        shape = List.copyOf(shape == null ? List.of() : shape);
        layout = layout == null ? "UNKNOWN" : layout;
        storageOffset = Math.max(0L, storageOffset);
    }
}
