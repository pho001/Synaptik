package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

/**
 * Runtime wrapper for a prepared cpu1 index node.
 */
public final class Cpu1IndexExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedIndexUnit preparedUnit;

    public Cpu1IndexExecutableUnit(Cpu1PreparedIndexUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedIndexUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor indexTensor = context.runtimeTensorForNodeId(preparedUnit.indexNodeId());
        Tensor updateTensor = preparedUnit.hasUpdateInput()
                ? context.runtimeTensorForNodeId(preparedUnit.updateNodeId())
                : null;
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());
        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView input;
        Cpu1TensorView indices;
        Cpu1TensorView updates = null;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            NativeTensorStorage nativeInput = context.requireNativeReadable(
                    preparedUnit.inputNodeId(),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            NativeTensorStorage nativeIndices = context.requireNativeReadable(
                    preparedUnit.indexNodeId(),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            NativeTensorStorage nativeUpdates = preparedUnit.hasUpdateInput()
                    ? context.requireNativeReadable(
                            preparedUnit.updateNodeId(),
                            CpuMaterializationReason.CPU_CONSUMER
                    )
                    : null;
            nativeOutput = context.requireNativeOutputStorage(
                    preparedUnit.nodeId(),
                    preparedUnit.valueDataType(),
                    preparedUnit.outputElementCount(),
                    "cpu1-index-node-" + preparedUnit.nodeId()
            );
            input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
            indices = Cpu1TensorView.fromNativeStorage(indexTensor, nativeIndices);
            if (nativeUpdates != null) {
                updates = Cpu1TensorView.fromNativeStorage(updateTensor, nativeUpdates);
            }
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
            requireSegmentView("input", input);
            requireSegmentView("indices", indices);
            if (updates != null) {
                requireSegmentView("updates", updates);
            }
            requireSegmentView("output", output);
        } else if (preparedUnit.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            context.requireCpuReadable(preparedUnit.indexNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            if (preparedUnit.hasUpdateInput()) {
                context.requireCpuReadable(preparedUnit.updateNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            }
            input = Cpu1TensorView.fromTensor(inputTensor);
            indices = Cpu1TensorView.fromTensor(indexTensor);
            if (updateTensor != null) {
                updates = Cpu1TensorView.fromTensor(updateTensor);
            }
            output = Cpu1TensorView.fromTensor(outputTensor);
            requireArrayView("input", input);
            requireArrayView("indices", indices);
            if (updates != null) {
                requireArrayView("updates", updates);
            }
            requireArrayView("output", output);
        } else {
            throw new UnsupportedOperationException("cpu1 " + preparedUnit.opType()
                    + " executable does not support storage " + preparedUnit.storageKind());
        }

        preparedUnit.kernel().run(preparedUnit, input, indices, updates, output);

        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 " + preparedUnit.opType() + " wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(
                    preparedUnit.nodeId(),
                    nativeOutput,
                    "cpu1 " + preparedUnit.opType() + " wrote native CPU segment"
            );
        }
    }

    private static void requireArrayView(String role, Cpu1TensorView view) {
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 index dense slice supports only JAVA_ARRAY runtime "
                + role + " storage, got " + view.storageKind());
    }

    private static void requireSegmentView(String role, Cpu1TensorView view) {
        if (view.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 index dense slice supports only MEMORY_SEGMENT runtime "
                + role + " storage, got " + view.storageKind());
    }
}
