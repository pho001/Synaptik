package backend.cpu1.fused.ir;

import tensor.DataType;

import java.util.Arrays;

public record Cpu1FusedInputPlan(
        int ref,
        int nodeId,
        DataType dataType,
        int[] shape,
        int[] strides,
        int[] logicalOutputShape,
        int[] logicalOutputDenseStrides,
        int storageOffset,
        int[] effectiveStrides,
        Cpu1FusedAccessKind accessKind
) {
    public Cpu1FusedInputPlan {
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        shape = shape == null ? new int[0] : shape.clone();
        strides = strides == null ? new int[0] : strides.clone();
        logicalOutputShape = logicalOutputShape == null ? new int[0] : logicalOutputShape.clone();
        logicalOutputDenseStrides = logicalOutputDenseStrides == null ? new int[0] : logicalOutputDenseStrides.clone();
        effectiveStrides = effectiveStrides == null ? new int[0] : effectiveStrides.clone();
        if (accessKind == null) {
            throw new IllegalArgumentException("accessKind cannot be null");
        }
    }

    public boolean isLinearAccess() {
        return accessKind == Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                || accessKind == Cpu1FusedAccessKind.OFFSET_CONTIGUOUS;
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    @Override
    public int[] logicalOutputShape() {
        return logicalOutputShape.clone();
    }

    @Override
    public int[] logicalOutputDenseStrides() {
        return logicalOutputDenseStrides.clone();
    }

    @Override
    public int[] effectiveStrides() {
        return effectiveStrides.clone();
    }

    @Override
    public String toString() {
        return "Cpu1FusedInputPlan{"
                + "ref=" + ref
                + ", nodeId=" + nodeId
                + ", dataType=" + dataType
                + ", shape=" + Arrays.toString(shape)
                + ", effectiveStrides=" + Arrays.toString(effectiveStrides)
                + ", accessKind=" + accessKind
                + '}';
    }
}
