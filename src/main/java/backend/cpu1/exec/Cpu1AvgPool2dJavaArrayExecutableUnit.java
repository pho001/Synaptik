package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAvgPool2dUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;

/**
 * Runtime wrapper for a prepared cpu1 AVG_POOL2D node using Java array storage.
 */
public final class Cpu1AvgPool2dJavaArrayExecutableUnit extends Cpu1AvgPool2dExecutableUnit {
    public Cpu1AvgPool2dJavaArrayExecutableUnit(Cpu1PreparedAvgPool2dUnit preparedUnit) {
        super(preparedUnit);
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Cpu1PreparedAvgPool2dUnit preparedUnit = preparedUnit();
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());

        context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Cpu1TensorView input = Cpu1TensorView.fromTensor(inputTensor);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("input", input, Cpu1StorageKind.JAVA_ARRAY);
        requireDenseNoOffset("output", output, Cpu1StorageKind.JAVA_ARRAY);
        validateViews(input, output);

        preparedUnit.kernel().run(preparedUnit, input, output);

        output.markStorageModified();
        context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 AVG_POOL2D wrote CPU array");
    }
}
