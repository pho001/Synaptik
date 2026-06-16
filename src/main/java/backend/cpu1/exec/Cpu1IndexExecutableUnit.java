package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;

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
        if (preparedUnit.storageKind() != Cpu1StorageKind.JAVA_ARRAY) {
            throw new UnsupportedOperationException("cpu1 GATHER executable supports only JAVA_ARRAY storage, got "
                    + preparedUnit.storageKind());
        }
        context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        context.requireCpuReadable(preparedUnit.indexNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor indexTensor = context.runtimeTensorForNodeId(preparedUnit.indexNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());
        Cpu1TensorView input = Cpu1TensorView.fromTensor(inputTensor);
        Cpu1TensorView indices = Cpu1TensorView.fromTensor(indexTensor);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireArrayView("input", input);
        requireArrayView("indices", indices);
        requireArrayView("output", output);

        preparedUnit.kernel().run(preparedUnit, input, indices, output);

        output.markStorageModified();
        context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 GATHER wrote CPU array");
    }

    private static void requireArrayView(String role, Cpu1TensorView view) {
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 GATHER first slice supports only JAVA_ARRAY runtime "
                + role + " storage, got " + view.storageKind());
    }
}
