package backend.cpu1.exec;

import backend.memory.CpuMaterializationReason;
import backend.cpu1.prepare.Cpu1PreparedUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Runtime executable wrapper for one prepared cpu1 unit.
 */
public final class Cpu1ExecutableUnit {
    private final Cpu1PreparedUnit preparedUnit;

    public Cpu1ExecutableUnit(Cpu1PreparedUnit preparedUnit) {
        this.preparedUnit = Objects.requireNonNull(preparedUnit, "preparedUnit cannot be null");
    }

    public Cpu1PreparedUnit preparedUnit() {
        return preparedUnit;
    }

    /**
     * Binds run-owned tensors from the execution context and runs the prepared kernel.
     *
     * @param context run-scoped execution context
     */
    public void run(ExecutionContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.outputNodeId());
        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            nativeOutput = context.allocateNativeStorage(
                    preparedUnit.dataType(),
                    preparedUnit.elementCount(),
                    "cpu1-node-" + preparedUnit.outputNodeId()
            );
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        } else {
            output = Cpu1TensorView.fromTensor(outputTensor);
        }
        List<Cpu1TensorView> inputs = new ArrayList<>(preparedUnit.inputNodeIds().size());
        for (int inputNodeId : preparedUnit.inputNodeIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
                NativeTensorStorage nativeInput = context.requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
                inputs.add(Cpu1TensorView.fromNativeStorage(tensor, nativeInput)
                        .broadcastToShape(preparedUnit.iterationPlan().shape()));
            } else {
                inputs.add(Cpu1TensorView.fromTensor(tensor)
                        .broadcastToShape(preparedUnit.iterationPlan().shape()));
            }
        }
        Cpu1KernelArgs args = new Cpu1KernelArgs(preparedUnit, inputs, output);
        preparedUnit.launchPolicy().launch(preparedUnit.kernelRunner(), args);
        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.outputNodeId(), "cpu1 wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(preparedUnit.outputNodeId(), nativeOutput, "cpu1 wrote native CPU segment");
        }
    }
}
