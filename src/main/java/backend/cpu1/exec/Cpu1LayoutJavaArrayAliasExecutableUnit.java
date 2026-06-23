package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.TensorInternalAccess;

/**
 * Executable unit for prepared layout view aliases using Java array storage.
 */
public final class Cpu1LayoutJavaArrayAliasExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedLayoutUnit preparedUnit;

    public Cpu1LayoutJavaArrayAliasExecutableUnit(Cpu1PreparedLayoutUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedLayoutUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return preparedUnit.scratchBufferSpec();
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        Tensor input = context.runtimeTensorForNodeId(preparedUnit.inputNodeId());
        Tensor output = context.runtimeTensorForNodeId(preparedUnit.nodeId());
        TensorInternalAccess.aliasRuntimeFrom(output, input);
        context.requireCpuReadable(preparedUnit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        context.markCpuCurrent(
                preparedUnit.nodeId(),
                "cpu1 " + preparedUnit.opType() + " view aliases CPU array"
        );
    }
}
