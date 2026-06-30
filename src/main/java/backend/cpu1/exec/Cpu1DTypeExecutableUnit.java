package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

/**
 * Executable unit for prepared cpu1 dtype kernels.
 */
public final class Cpu1DTypeExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedDTypeUnit preparedUnit;

    public Cpu1DTypeExecutableUnit(Cpu1PreparedDTypeUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedDTypeUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());
        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView input;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            NativeTensorStorage nativeInput = context.requireNativeReadable(
                    preparedUnit.inputNodeId(),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            nativeOutput = context.requireNativeOutputStorage(
                    preparedUnit.nodeId(),
                    preparedUnit.outputDataType(),
                    preparedUnit.elementCount(),
                    "cpu1-dtype-node-" + preparedUnit.nodeId()
            );
            input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        } else {
            context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
            input = Cpu1TensorView.fromTensor(inputTensor);
            output = Cpu1TensorView.fromTensor(outputTensor);
        }

        preparedUnit.kernel().run(preparedUnit, input, output);

        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 CAST wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(preparedUnit.nodeId(), nativeOutput, "cpu1 CAST wrote native CPU segment");
        }
    }
}
