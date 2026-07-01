package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

/**
 * Runtime wrapper for a prepared cpu1 index node using native CPU segment storage.
 */
public final class Cpu1IndexMemorySegmentExecutableUnit extends Cpu1IndexExecutableUnit {
    public Cpu1IndexMemorySegmentExecutableUnit(Cpu1PreparedIndexUnit preparedUnit) {
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

        NativeTensorStorage nativeInput = context.requireNativeReadable(
                preparedUnit.inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeIndices = context.requireNativeReadable(
                preparedUnit.indexNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage nativeUpdates = preparedUnit.hasUpdateInput()
                ? context.requireNativeReadable(preparedUnit.updateNodeId(), CpuMaterializationReason.CPU_CONSUMER)
                : null;
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                preparedUnit.nodeId(),
                preparedUnit.valueDataType(),
                preparedUnit.outputElementCount(),
                "cpu1-index-node-" + preparedUnit.nodeId()
        );
        Cpu1TensorView input = Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput);
        Cpu1TensorView indices = Cpu1TensorView.fromNativeStorage(indexTensor, nativeIndices);
        Cpu1TensorView updates = nativeUpdates == null
                ? null
                : Cpu1TensorView.fromNativeStorage(updateTensor, nativeUpdates);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        requireSegmentView("input", input);
        requireSegmentView("indices", indices);
        if (updates != null) {
            requireSegmentView("updates", updates);
        }
        requireSegmentView("output", output);

        preparedUnit.kernel().run(preparedUnit, input, indices, updates, output);

        nativeOutput.markModified();
        context.attachNativeStorage(
                preparedUnit.nodeId(),
                nativeOutput,
                "cpu1 " + preparedUnit.opType() + " wrote native CPU segment"
        );
    }
}
