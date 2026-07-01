package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedRmsNormUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;

/**
 * Runtime wrapper for a prepared cpu1 RMSNorm node using Java array storage.
 */
public final class Cpu1RmsNormJavaArrayExecutableUnit extends Cpu1RmsNormExecutableUnit {
    public Cpu1RmsNormJavaArrayExecutableUnit(Cpu1PreparedRmsNormUnit preparedUnit) {
        super(preparedUnit);
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Cpu1PreparedRmsNormUnit preparedUnit = preparedUnit();
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor gammaTensor = context.runtimeTensorForNodeId(preparedUnit.gammaNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());

        context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        context.requireCpuReadable(preparedUnit.gammaNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Cpu1TensorView input = Cpu1TensorView.fromTensor(inputTensor);
        Cpu1TensorView gamma = Cpu1TensorView.fromTensor(gammaTensor);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("input", input, Cpu1StorageKind.JAVA_ARRAY);
        requireDenseNoOffset("gamma", gamma, Cpu1StorageKind.JAVA_ARRAY);
        requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        validateViews(input, output);

        preparedUnit.kernel().run(preparedUnit, input, gamma, output);

        output.markStorageModified();
        context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 RMS_NORM wrote CPU array");
    }
}
