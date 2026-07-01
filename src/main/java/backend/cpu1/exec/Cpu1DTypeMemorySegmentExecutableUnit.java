package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

/**
 * Executable unit for prepared dtype kernels using native CPU segment storage.
 */
public final class Cpu1DTypeMemorySegmentExecutableUnit extends Cpu1DTypeExecutableUnit {
    public Cpu1DTypeMemorySegmentExecutableUnit(Cpu1PreparedDTypeUnit preparedUnit) {
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
        NativeTensorStorage nativeInput = context.requireNativeReadable(
                preparedUnit.inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                preparedUnit.nodeId(),
                preparedUnit.outputDataType(),
                preparedUnit.elementCount(),
                "cpu1-dtype-node-" + preparedUnit.nodeId()
        );
        Cpu1TensorView input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);

        preparedUnit.kernel().run(preparedUnit, input, output);

        nativeOutput.markModified();
        context.attachNativeStorage(preparedUnit.nodeId(), nativeOutput, "cpu1 CAST wrote native CPU segment");
    }
}
