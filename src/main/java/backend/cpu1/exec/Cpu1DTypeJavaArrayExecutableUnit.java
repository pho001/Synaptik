package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;

/**
 * Executable unit for prepared dtype kernels using Java array storage.
 */
public final class Cpu1DTypeJavaArrayExecutableUnit extends Cpu1DTypeExecutableUnit {
    public Cpu1DTypeJavaArrayExecutableUnit(Cpu1PreparedDTypeUnit preparedUnit) {
        super(preparedUnit);
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Cpu1PreparedDTypeUnit preparedUnit = preparedUnit();
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());
        context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Cpu1TensorView input = Cpu1TensorView.fromTensor(inputTensor);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);

        preparedUnit.kernel().run(preparedUnit, input, output);

        output.markStorageModified();
        context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 CAST wrote CPU array");
    }
}
