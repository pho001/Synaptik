package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedElementwiseUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Executable unit for prepared elementwise kernels.
 */
public final class Cpu1ElementwiseExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedElementwiseUnit preparedUnit;

    public Cpu1ElementwiseExecutableUnit(Cpu1PreparedElementwiseUnit preparedUnit) {
        this.preparedUnit = Objects.requireNonNull(preparedUnit, "preparedUnit cannot be null");
    }

    public Cpu1PreparedElementwiseUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return Cpu1ScratchBufferSpec.none();
    }

    /**
     * Binds run-owned tensors from the execution context and runs the prepared kernel.
     *
     * @param context run-scoped execution context
     */
    @Override
    public void run(ExecutionContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        Tensor outputTensor = context.runtimeTensorForNodeId(preparedUnit.outputNodeId());
        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView output;
        if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            nativeOutput = context.requireNativeOutputStorage(
                    preparedUnit.outputNodeId(),
                    preparedUnit.dataType(),
                    preparedUnit.elementCount(),
                    "cpu1-node-" + preparedUnit.outputNodeId()
            );
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        } else {
            output = Cpu1TensorView.fromTensor(outputTensor);
        }
        List<Cpu1TensorView> inputs = new ArrayList<>(preparedUnit.inputNodeIds().size());
        for (int inputNodeId : preparedUnit.inputNodeIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            if (preparedUnit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
                NativeTensorStorage nativeInput = context.requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
                inputs.add(Cpu1TensorView.fromNativeStorage(tensor, nativeInput)
                        .broadcastToShape(preparedUnit.iterationPlan().shape()));
            } else {
                inputs.add(Cpu1TensorView.fromTensor(tensor)
                        .broadcastToShape(preparedUnit.iterationPlan().shape()));
            }
        }
        Cpu1KernelArgs args = new Cpu1KernelArgs(
                preparedUnit,
                inputs,
                output,
                context.cpu1ScratchBufferForNodeId(preparedUnit.nodeId())
        );
        preparedUnit.launchPolicy().launch(preparedUnit.kernelRunner(), args);
        if (nativeOutput == null) {
            output.markStorageModified();
            context.markCpuCurrent(preparedUnit.outputNodeId(), "cpu1 wrote CPU array");
        } else {
            nativeOutput.markModified();
            context.attachNativeStorage(preparedUnit.outputNodeId(), nativeOutput, "cpu1 wrote native CPU segment");
        }
    }
}
