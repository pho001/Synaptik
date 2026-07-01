package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAvgPool2dUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

/**
 * Runtime wrapper for a prepared cpu1 AVG_POOL2D node using native CPU segment storage.
 */
public final class Cpu1AvgPool2dMemorySegmentExecutableUnit extends Cpu1AvgPool2dExecutableUnit {
    public Cpu1AvgPool2dMemorySegmentExecutableUnit(Cpu1PreparedAvgPool2dUnit preparedUnit) {
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

        NativeTensorStorage nativeInput = context.requireNativeReadable(
                preparedUnit.inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                preparedUnit.nodeId(),
                preparedUnit.dataType(),
                preparedUnit.outputElementCount(),
                "cpu1-avgpool2d-node-" + preparedUnit.nodeId()
        );
        Cpu1TensorView input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        requireDenseNoOffset("input", input, Cpu1StorageKind.MEMORY_SEGMENT);
        requireDenseNoOffset("output", output, Cpu1StorageKind.MEMORY_SEGMENT);
        validateViews(input, output);

        preparedUnit.kernel().run(preparedUnit, input, output);

        nativeOutput.markModified();
        context.attachNativeStorage(
                preparedUnit.nodeId(),
                nativeOutput,
                "cpu1 AVG_POOL2D wrote native CPU segment"
        );
    }
}
