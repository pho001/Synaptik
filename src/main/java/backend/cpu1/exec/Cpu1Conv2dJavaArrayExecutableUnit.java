package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedConv2dUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;

/**
 * Runtime wrapper for a prepared cpu1 CONV2D node using Java array storage.
 */
public final class Cpu1Conv2dJavaArrayExecutableUnit extends Cpu1Conv2dExecutableUnit {
    public Cpu1Conv2dJavaArrayExecutableUnit(Cpu1PreparedConv2dUnit preparedUnit) {
        super(preparedUnit);
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Cpu1PreparedConv2dUnit preparedUnit = preparedUnit();
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor weightTensor = context.runtimeTensorForNodeId(preparedUnit.weightNodeId());
        Tensor biasTensor = preparedUnit.hasBias()
                ? context.runtimeTensorForNodeId(preparedUnit.biasNodeId())
                : null;
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());

        context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        context.requireCpuReadable(preparedUnit.weightNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        if (preparedUnit.hasBias()) {
            context.requireCpuReadable(preparedUnit.biasNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        }
        Cpu1TensorView input = Cpu1TensorView.fromTensor(inputTensor);
        Cpu1TensorView weight = Cpu1TensorView.fromTensor(weightTensor);
        Cpu1TensorView bias = biasTensor == null ? null : Cpu1TensorView.fromTensor(biasTensor);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("input", input, Cpu1StorageKind.JAVA_ARRAY);
        requireDenseNoOffset("weight", weight, Cpu1StorageKind.JAVA_ARRAY);
        if (bias != null) {
            requireDenseNoOffset("bias", bias, Cpu1StorageKind.JAVA_ARRAY);
        }
        requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        validateViews(input, weight, bias, output);

        preparedUnit.kernel().run(preparedUnit, input, weight, bias, output);

        output.markStorageModified();
        context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 CONV2D wrote CPU array");
    }
}
