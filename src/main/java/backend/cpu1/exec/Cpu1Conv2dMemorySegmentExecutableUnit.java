package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedConv2dUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

/**
 * Runtime wrapper for a prepared cpu1 CONV2D node using native CPU segment storage.
 */
public final class Cpu1Conv2dMemorySegmentExecutableUnit extends Cpu1Conv2dExecutableUnit {
    public Cpu1Conv2dMemorySegmentExecutableUnit(Cpu1PreparedConv2dUnit preparedUnit) {
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

        NativeTensorStorage nativeInput = context.requireNativeReadable(
                preparedUnit.inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeWeight = context.requireNativeReadable(
                preparedUnit.weightNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeBias = preparedUnit.hasBias()
                ? context.requireNativeReadable(preparedUnit.biasNodeId(), CpuMaterializationReason.CPU_CONSUMER)
                : null;
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                preparedUnit.nodeId(),
                preparedUnit.dataType(),
                preparedUnit.outputElementCount(),
                "cpu1-conv2d-node-" + preparedUnit.nodeId()
        );
        Cpu1TensorView input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
        Cpu1TensorView weight = Cpu1TensorView.fromNativeStorage(weightTensor, nativeWeight);
        Cpu1TensorView bias = nativeBias == null ? null : Cpu1TensorView.fromNativeStorage(biasTensor, nativeBias);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        requireDenseNoOffset("input", input, Cpu1StorageKind.MEMORY_SEGMENT);
        requireDenseNoOffset("weight", weight, Cpu1StorageKind.MEMORY_SEGMENT);
        if (bias != null) {
            requireDenseNoOffset("bias", bias, Cpu1StorageKind.MEMORY_SEGMENT);
        }
        requireDenseNoOffset("output", output, Cpu1StorageKind.MEMORY_SEGMENT);
        validateViews(input, weight, bias, output);

        preparedUnit.kernel().run(preparedUnit, input, weight, bias, output);

        nativeOutput.markModified();
        context.attachNativeStorage(
                preparedUnit.nodeId(),
                nativeOutput,
                "cpu1 CONV2D wrote native CPU segment"
        );
    }
}
