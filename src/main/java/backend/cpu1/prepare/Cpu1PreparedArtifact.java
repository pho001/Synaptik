package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ExecutableUnit;
import backend.cpu1.exec.Cpu1DTypeExecutableUnit;
import backend.cpu1.exec.Cpu1ElementwiseExecutableUnit;
import backend.cpu1.exec.Cpu1FusedElementwiseExecutableUnit;
import backend.cpu1.exec.Cpu1IndexExecutableUnit;
import backend.cpu1.exec.Cpu1LayoutExecutableUnit;
import backend.cpu1.exec.Cpu1LossExecutableUnit;
import backend.cpu1.exec.Cpu1MatmulExecutableUnit;
import backend.cpu1.exec.Cpu1MseLossExecutableUnit;
import backend.cpu1.exec.Cpu1ReductionExecutableUnit;
import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.trace.Cpu1TraceContributor;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.PreparedExecutionArtifact;
import graph.execution.plan.PreparedRuntimeStateAllocator;
import graph.execution.trace.StepTraceContribution;

/**
 * Prepared execution artifact attached to cpu1 node metadata.
 */
public final class Cpu1PreparedArtifact implements PreparedExecutionArtifact {
    private final Cpu1PreparedElementwiseUnit preparedUnit;
    private final Cpu1PreparedLayoutUnit preparedLayoutUnit;
    private final Cpu1PreparedDTypeUnit preparedDTypeUnit;
    private final Cpu1PreparedReductionUnit preparedReductionUnit;
    private final Cpu1PreparedMatmulUnit preparedMatmulUnit;
    private final Cpu1PreparedMseLossUnit preparedMseLossUnit;
    private final Cpu1PreparedCrossEntropyLossUnit preparedCrossEntropyLossUnit;
    private final Cpu1PreparedDenseCrossEntropyLossUnit preparedDenseCrossEntropyLossUnit;
    private final Cpu1PreparedNllLossUnit preparedNllLossUnit;
    private final Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit;
    private final Cpu1PreparedIndexUnit preparedIndexUnit;
    private final Cpu1ExecutableUnit executableUnit;

    public Cpu1PreparedArtifact(Cpu1PreparedElementwiseUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1ElementwiseExecutableUnit(preparedUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedLayoutUnit preparedLayoutUnit) {
        if (preparedLayoutUnit == null) {
            throw new IllegalArgumentException("preparedLayoutUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = preparedLayoutUnit;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1LayoutExecutableUnit(preparedLayoutUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedDTypeUnit preparedDTypeUnit) {
        if (preparedDTypeUnit == null) {
            throw new IllegalArgumentException("preparedDTypeUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = preparedDTypeUnit;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1DTypeExecutableUnit(preparedDTypeUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedReductionUnit preparedReductionUnit) {
        if (preparedReductionUnit == null) {
            throw new IllegalArgumentException("preparedReductionUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = preparedReductionUnit;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1ReductionExecutableUnit(preparedReductionUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedMatmulUnit preparedMatmulUnit) {
        if (preparedMatmulUnit == null) {
            throw new IllegalArgumentException("preparedMatmulUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = preparedMatmulUnit;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1MatmulExecutableUnit(preparedMatmulUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedMseLossUnit preparedMseLossUnit) {
        if (preparedMseLossUnit == null) {
            throw new IllegalArgumentException("preparedMseLossUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = preparedMseLossUnit;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1MseLossExecutableUnit(preparedMseLossUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedCrossEntropyLossUnit preparedCrossEntropyLossUnit) {
        if (preparedCrossEntropyLossUnit == null) {
            throw new IllegalArgumentException("preparedCrossEntropyLossUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = preparedCrossEntropyLossUnit;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1LossExecutableUnit(preparedCrossEntropyLossUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedDenseCrossEntropyLossUnit preparedDenseCrossEntropyLossUnit) {
        if (preparedDenseCrossEntropyLossUnit == null) {
            throw new IllegalArgumentException("preparedDenseCrossEntropyLossUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = preparedDenseCrossEntropyLossUnit;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1LossExecutableUnit(preparedDenseCrossEntropyLossUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedNllLossUnit preparedNllLossUnit) {
        if (preparedNllLossUnit == null) {
            throw new IllegalArgumentException("preparedNllLossUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = preparedNllLossUnit;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1LossExecutableUnit(preparedNllLossUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit) {
        if (preparedFusedElementwiseUnit == null) {
            throw new IllegalArgumentException("preparedFusedElementwiseUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = preparedFusedElementwiseUnit;
        this.preparedIndexUnit = null;
        this.executableUnit = new Cpu1FusedElementwiseExecutableUnit(preparedFusedElementwiseUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedIndexUnit preparedIndexUnit) {
        if (preparedIndexUnit == null) {
            throw new IllegalArgumentException("preparedIndexUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = preparedIndexUnit;
        this.executableUnit = new Cpu1IndexExecutableUnit(preparedIndexUnit);
    }

    public Cpu1PreparedArtifact(Cpu1ExecutableUnit executableUnit) {
        if (executableUnit == null) {
            throw new IllegalArgumentException("executableUnit cannot be null");
        }
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedDTypeUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.preparedMseLossUnit = null;
        this.preparedCrossEntropyLossUnit = null;
        this.preparedDenseCrossEntropyLossUnit = null;
        this.preparedNllLossUnit = null;
        this.preparedFusedElementwiseUnit = null;
        this.preparedIndexUnit = null;
        this.executableUnit = executableUnit;
    }

    public Cpu1PreparedElementwiseUnit preparedUnit() {
        if (preparedUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared elementwise unit");
        }
        return preparedUnit;
    }

    public Cpu1PreparedLayoutUnit preparedLayoutUnit() {
        if (preparedLayoutUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared layout unit");
        }
        return preparedLayoutUnit;
    }

    public Cpu1PreparedDTypeUnit preparedDTypeUnit() {
        if (preparedDTypeUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared dtype unit");
        }
        return preparedDTypeUnit;
    }

    public Cpu1PreparedReductionUnit preparedReductionUnit() {
        if (preparedReductionUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared reduction unit");
        }
        return preparedReductionUnit;
    }

    public Cpu1PreparedMatmulUnit preparedMatmulUnit() {
        if (preparedMatmulUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared matmul unit");
        }
        return preparedMatmulUnit;
    }

    public Cpu1PreparedMseLossUnit preparedMseLossUnit() {
        if (preparedMseLossUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared MSE loss unit");
        }
        return preparedMseLossUnit;
    }

    public Cpu1PreparedCrossEntropyLossUnit preparedCrossEntropyLossUnit() {
        if (preparedCrossEntropyLossUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared cross entropy loss unit");
        }
        return preparedCrossEntropyLossUnit;
    }

    public Cpu1PreparedDenseCrossEntropyLossUnit preparedDenseCrossEntropyLossUnit() {
        if (preparedDenseCrossEntropyLossUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared dense cross entropy loss unit");
        }
        return preparedDenseCrossEntropyLossUnit;
    }

    public Cpu1PreparedNllLossUnit preparedNllLossUnit() {
        if (preparedNllLossUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared NLL loss unit");
        }
        return preparedNllLossUnit;
    }

    public Cpu1PreparedFusedElementwiseUnit preparedFusedElementwiseUnit() {
        if (preparedFusedElementwiseUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared fused elementwise unit");
        }
        return preparedFusedElementwiseUnit;
    }

    public Cpu1PreparedIndexUnit preparedIndexUnit() {
        if (preparedIndexUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared index unit");
        }
        return preparedIndexUnit;
    }

    public Cpu1ExecutableUnit executableUnit() {
        return executableUnit;
    }

    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return executableUnit.scratchBufferSpec();
    }

    @Override
    public void allocateRuntimeState(int nodeId, PreparedRuntimeStateAllocator allocator) {
        if (allocator == null) {
            return;
        }
        Cpu1ScratchBufferSpec spec = scratchBufferSpec();
        if (spec.isEmpty()) {
            return;
        }
        allocator.putWorkspace(nodeId, Cpu1ScratchBuffer.allocate(spec));
    }

    public void execute(ExecutionContext context) {
        executableUnit.run(context);
    }

    @Override
    public StepTraceContribution traceContribution(
            CompiledNode node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        return Cpu1TraceContributor.traceContribution(
                node,
                preparedLayoutUnit,
                preparedDTypeUnit,
                preparedReductionUnit,
                preparedMatmulUnit,
                preparedMseLossUnit,
                preparedCrossEntropyLossUnit,
                preparedDenseCrossEntropyLossUnit,
                preparedNllLossUnit,
                preparedFusedElementwiseUnit,
                preparedIndexUnit
        );
    }
}
