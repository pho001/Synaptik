package backend.cpu1.exec;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.ArrayList;
import java.util.List;

/**
 * Executable unit for prepared generated fused elementwise kernels.
 */
public final class Cpu1FusedElementwiseExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedFusedElementwiseUnit preparedUnit;

    public Cpu1FusedElementwiseExecutableUnit(Cpu1PreparedFusedElementwiseUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedFusedElementwiseUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return Cpu1ScratchBufferSpec.none();
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.outputNodeId());
        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            nativeOutput = context.requireNativeOutputStorage(
                    preparedUnit.outputNodeId(),
                    preparedUnit.outputDataType(),
                    preparedUnit.elementCount(),
                    "cpu1-fused-" + preparedUnit.outputNodeId()
            );
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        } else {
            output = Cpu1TensorView.fromTensor(outputTensor);
        }

        List<Cpu1TensorView> inputs = new ArrayList<>(preparedUnit.inputNodeIds().size());
        for (int inputNodeId : preparedUnit.inputNodeIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
                NativeTensorStorage nativeInput = context.requireNativeReadable(
                        inputNodeId,
                        CpuMaterializationReason.CPU_CONSUMER
                );
                inputs.add(Cpu1TensorView.fromNativeStorage(tensor, nativeInput)
                        .broadcastToShape(preparedUnit.outputShape()));
            } else {
                inputs.add(Cpu1TensorView.fromTensor(tensor)
                        .broadcastToShape(preparedUnit.outputShape()));
            }
        }

        Cpu1FusedKernelArgs args = new Cpu1FusedKernelArgs(preparedUnit, inputs, output);
        preparedUnit.launchPolicy().launch(
                args.elementCount(),
                (startInclusive, endExclusive) ->
                        preparedUnit.generatedKernel().computeRange(args, startInclusive, endExclusive)
        );

        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.outputNodeId(), "cpu1 fused wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(preparedUnit.outputNodeId(), nativeOutput, "cpu1 fused wrote native CPU segment");
        }
    }
}
