package backend.cpu.kernels.storage;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;
import java.util.Objects;

public final class CpuStorageResolver {
    public CpuStorageBindings bindArrayOnly(List<Tensor> inputs, Tensor output) {
        Objects.requireNonNull(inputs, "inputs cannot be null");
        return new CpuStorageBindings(
                inputs.stream().map(this::bindArrayOnly).toList(),
                bindArrayOnly(output)
        );
    }

    public CpuStorageView bindArrayOnly(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return CpuStorageView.array(
                tensor.getDataType(),
                arrayFor(tensor),
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize()
        );
    }

    private Object arrayFor(Tensor tensor) {
        DataType dtype = tensor.getDataType();
        return switch (dtype) {
            case FLOAT64 -> TensorInternalAccess.float64Data(tensor);
            case FLOAT32 -> TensorInternalAccess.float32Data(tensor);
            case BFLOAT16 -> TensorInternalAccess.bfloat16Data(tensor);
            case INT32 -> TensorInternalAccess.int32Data(tensor);
            case INT64 -> TensorInternalAccess.int64Data(tensor);
            case BOOL -> TensorInternalAccess.boolData(tensor);
        };
    }
}
