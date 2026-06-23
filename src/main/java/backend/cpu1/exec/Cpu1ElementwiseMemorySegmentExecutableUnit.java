package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedElementwiseUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.ArrayList;
import java.util.List;

/**
 * Executable unit for prepared elementwise kernels using native CPU segment storage.
 */
public final class Cpu1ElementwiseMemorySegmentExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedElementwiseUnit preparedUnit;

    public Cpu1ElementwiseMemorySegmentExecutableUnit(Cpu1PreparedElementwiseUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedElementwiseUnit preparedUnit() {
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
        NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                preparedUnit.outputNodeId(),
                preparedUnit.dataType(),
                preparedUnit.elementCount(),
                "cpu1-node-" + preparedUnit.outputNodeId()
        );
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);

        List<Cpu1TensorView> inputs = new ArrayList<>(preparedUnit.inputNodeIds().size());
        for (int inputNodeId : preparedUnit.inputNodeIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            NativeTensorStorage nativeInput = context.requireNativeReadable(
                    inputNodeId,
                    CpuMaterializationReason.CPU_CONSUMER
            );
            inputs.add(Cpu1TensorView.fromNativeStorage(tensor, nativeInput)
                    .broadcastToShape(preparedUnit.iterationPlan().shape()));
        }

        Cpu1KernelArgs args = new Cpu1KernelArgs(
                preparedUnit,
                inputs,
                output,
                context.cpu1ScratchBufferForNodeId(preparedUnit.nodeId())
        );
        preparedUnit.launchPolicy().launch(
                args.elementCount(),
                (startInclusive, endExclusive) ->
                        preparedUnit.kernelRunner().computeRange(args, startInclusive, endExclusive)
        );

        nativeOutput.markModified();
        context.attachNativeStorage(preparedUnit.outputNodeId(), nativeOutput, "cpu1 wrote native CPU segment");
    }
}
