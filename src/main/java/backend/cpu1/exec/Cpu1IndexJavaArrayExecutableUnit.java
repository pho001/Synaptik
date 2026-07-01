package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;

/**
 * Runtime wrapper for a prepared cpu1 index node using Java array storage.
 */
public final class Cpu1IndexJavaArrayExecutableUnit extends Cpu1IndexExecutableUnit {
    public Cpu1IndexJavaArrayExecutableUnit(Cpu1PreparedIndexUnit preparedUnit) {
        super(preparedUnit);
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Cpu1PreparedIndexUnit preparedUnit = preparedUnit();
        Tensor inputTensor = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor indexTensor = context.runtimeTensorForNodeId(preparedUnit.indexNodeId());
        Tensor updateTensor = preparedUnit.hasUpdateInput()
                ? context.runtimeTensorForNodeId(preparedUnit.updateNodeId())
                : null;
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.nodeId());

        context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        context.requireCpuReadable(preparedUnit.indexNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        if (preparedUnit.hasUpdateInput()) {
            context.requireCpuReadable(preparedUnit.updateNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        }
        Cpu1TensorView input = Cpu1TensorView.fromTensor(inputTensor);
        Cpu1TensorView indices = Cpu1TensorView.fromTensor(indexTensor);
        Cpu1TensorView updates = updateTensor == null ? null : Cpu1TensorView.fromTensor(updateTensor);
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);
        requireArrayView("input", input);
        requireArrayView("indices", indices);
        if (updates != null) {
            requireArrayView("updates", updates);
        }
        requireArrayView("output", output);

        preparedUnit.kernel().run(preparedUnit, input, indices, updates, output);

        output.markStorageModified();
        context.markCpuCurrent(preparedUnit.nodeId(), "cpu1 " + preparedUnit.opType() + " wrote CPU array");
    }
}
