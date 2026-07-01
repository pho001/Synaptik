package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedLayerNormUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;

/**
 * Runtime wrapper for a prepared cpu1 LayerNorm node using Java array storage.
 */
public final class Cpu1LayerNormJavaArrayExecutableUnit extends Cpu1LayerNormExecutableUnit {
    public Cpu1LayerNormJavaArrayExecutableUnit(Cpu1PreparedLayerNormUnit preparedUnit) {
        super(preparedUnit);
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Cpu1PreparedLayerNormUnit preparedUnit = preparedUnit();
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor gammaTensor = context.runtimeTensorForNodeId(preparedUnit.gammaNodeId());
        Tensor betaTensor = context.runtimeTensorForNodeId(preparedUnit.betaNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());

        context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        context.requireCpuReadable(preparedUnit.gammaNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        context.requireCpuReadable(preparedUnit.betaNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Cpu1TensorView input = Cpu1TensorView.fromTensor(inputTensor);
        Cpu1TensorView gamma = Cpu1TensorView.fromTensor(gammaTensor);
        Cpu1TensorView beta = Cpu1TensorView.fromTensor(betaTensor);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("input", input, Cpu1StorageKind.JAVA_ARRAY);
        requireDenseNoOffset("gamma", gamma, Cpu1StorageKind.JAVA_ARRAY);
        requireDenseNoOffset("beta", beta, Cpu1StorageKind.JAVA_ARRAY);
        requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        validateViews(input, output);

        preparedUnit.kernel().run(preparedUnit, input, gamma, beta, output);

        output.markStorageModified();
        context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 LAYER_NORM wrote CPU array");
    }
}
