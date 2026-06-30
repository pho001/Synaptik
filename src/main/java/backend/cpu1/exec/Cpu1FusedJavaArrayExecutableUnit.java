package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Executable unit for prepared generated fused elementwise kernels using Java array storage.
 */
public final class Cpu1FusedJavaArrayExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedFusedElementwiseUnit preparedUnit;

    public Cpu1FusedJavaArrayExecutableUnit(Cpu1PreparedFusedElementwiseUnit preparedUnit) {
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
        Cpu1TensorView output = Cpu1TensorView.fromTensor(outputTensor);

        List<Cpu1TensorView> inputs = new ArrayList<>(preparedUnit.inputNodeIds().size());
        for (int inputNodeId : preparedUnit.inputNodeIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            context.requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
            inputs.add(Cpu1TensorView.fromTensor(tensor)
                    .broadcastToShape(preparedUnit.outputShape()));
        }

        Cpu1FusedKernelArgs args = new Cpu1FusedKernelArgs(preparedUnit, inputs, output);
        preparedUnit.launchPolicy().launch(
                args.elementCount(),
                (startInclusive, endExclusive) ->
                        preparedUnit.generatedKernel().computeRange(args, startInclusive, endExclusive)
        );

        output.markStorageModified();
        context.markCpuCurrent(preparedUnit.outputNodeId(), "cpu1 fused wrote CPU array");
    }
}
