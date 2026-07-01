package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedLayerNormUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

/**
 * Runtime wrapper for a prepared cpu1 LayerNorm node using native CPU segment storage.
 */
public final class Cpu1LayerNormMemorySegmentExecutableUnit extends Cpu1LayerNormExecutableUnit {
    public Cpu1LayerNormMemorySegmentExecutableUnit(Cpu1PreparedLayerNormUnit preparedUnit) {
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

        NativeTensorStorage nativeInput = context.requireNativeReadable(
                preparedUnit.inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeGamma = context.requireNativeReadable(
                preparedUnit.gammaNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeBeta = context.requireNativeReadable(
                preparedUnit.betaNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                preparedUnit.nodeId(),
                preparedUnit.dataType(),
                preparedUnit.outputElementCount(),
                "cpu1-layernorm-node-" + preparedUnit.nodeId()
        );
        Cpu1TensorView input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
        Cpu1TensorView gamma = Cpu1TensorView.fromNativeStorage(gammaTensor, nativeGamma);
        Cpu1TensorView beta = Cpu1TensorView.fromNativeStorage(betaTensor, nativeBeta);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        requireDenseNoOffset("input", input, Cpu1StorageKind.MEMORY_SEGMENT);
        requireDenseNoOffset("gamma", gamma, Cpu1StorageKind.MEMORY_SEGMENT);
        requireDenseNoOffset("beta", beta, Cpu1StorageKind.MEMORY_SEGMENT);
        requireDenseNoOffset("output", output, Cpu1StorageKind.MEMORY_SEGMENT);
        validateViews(input, output);

        preparedUnit.kernel().run(preparedUnit, input, gamma, beta, output);

        nativeOutput.markModified();
        context.attachNativeStorage(
                preparedUnit.nodeId(),
                nativeOutput,
                "cpu1 LAYER_NORM wrote native CPU segment"
        );
    }
}
