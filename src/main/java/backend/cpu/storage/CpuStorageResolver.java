package backend.cpu.storage;

import runtime.residency.TensorResidencyState;
import runtime.execution.ExecutionContext;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.storage.NativeTensorStorage;

import java.util.List;
import java.util.Objects;

public final class CpuStorageResolver {
    public CpuStorageBindings bindRuntime(
            List<Integer> inputNodeIds,
            List<Tensor> inputs,
            int outputNodeId,
            Tensor output,
            ExecutionContext context
    ) {
        Objects.requireNonNull(inputNodeIds, "inputNodeIds cannot be null");
        Objects.requireNonNull(inputs, "inputs cannot be null");
        if (inputNodeIds.size() != inputs.size()) {
            throw new IllegalArgumentException("inputNodeIds size must match inputs size. inputNodeIds="
                    + inputNodeIds.size() + ", inputs=" + inputs.size());
        }
        return new CpuStorageBindings(
                java.util.stream.IntStream.range(0, inputs.size())
                        .mapToObj(i -> bindRuntime(inputNodeIds.get(i), inputs.get(i), context))
                        .toList(),
                bindRuntime(outputNodeId, output, context)
        );
    }

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

    private CpuStorageView bindRuntime(int nodeId, Tensor tensor, ExecutionContext context) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        NativeTensorStorage nativeStorage = nativeCurrentStorage(nodeId, context);
        if (nativeStorage == null) {
            return bindArrayOnly(tensor);
        }
        if (nativeStorage.getType() != tensor.getDataType()) {
            throw new IllegalStateException("Native storage dtype does not match tensor dtype for node "
                    + nodeId + ". native=" + nativeStorage.getType() + ", tensor=" + tensor.getDataType());
        }
        return CpuStorageView.segment(
                tensor.getDataType(),
                nativeStorage.segment(),
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize()
        );
    }

    private NativeTensorStorage nativeCurrentStorage(int nodeId, ExecutionContext context) {
        if (context == null) {
            return null;
        }
        TensorResidencyState residency = context.residencyForNodeId(nodeId);
        if (residency == null || !residency.nativeCurrent()) {
            return null;
        }
        NativeTensorStorage storage = context.nativeStorageForNodeId(nodeId);
        if (storage == null || storage.closed()) {
            return null;
        }
        return storage;
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
