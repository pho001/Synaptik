package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ExecutableUnit;
import backend.cpu1.exec.Cpu1ElementwiseExecutableUnit;
import backend.cpu1.exec.Cpu1LayoutExecutableUnit;
import backend.cpu1.exec.Cpu1MatmulExecutableUnit;
import backend.cpu1.exec.Cpu1ReductionExecutableUnit;
import backend.cpu1.exec.Cpu1Workspace;
import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.trace.Cpu1TraceContributor;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.PreparedExecutionArtifact;
import graph.execution.plan.PreparedRuntimeStateAllocator;
import graph.execution.trace.StepTraceContribution;

import java.util.Objects;

/**
 * Prepared execution artifact attached to cpu1 node metadata.
 */
public final class Cpu1PreparedArtifact implements PreparedExecutionArtifact {
    private final Cpu1PreparedElementwiseUnit preparedUnit;
    private final Cpu1PreparedLayoutUnit preparedLayoutUnit;
    private final Cpu1PreparedReductionUnit preparedReductionUnit;
    private final Cpu1PreparedMatmulUnit preparedMatmulUnit;
    private final Cpu1ExecutableUnit executableUnit;

    public Cpu1PreparedArtifact(Cpu1PreparedElementwiseUnit preparedUnit) {
        this.preparedUnit = Objects.requireNonNull(preparedUnit, "preparedUnit cannot be null");
        this.preparedLayoutUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.executableUnit = new Cpu1ElementwiseExecutableUnit(preparedUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedLayoutUnit preparedLayoutUnit) {
        this.preparedUnit = null;
        this.preparedLayoutUnit = Objects.requireNonNull(preparedLayoutUnit, "preparedLayoutUnit cannot be null");
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.executableUnit = new Cpu1LayoutExecutableUnit(preparedLayoutUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedReductionUnit preparedReductionUnit) {
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedReductionUnit = Objects.requireNonNull(preparedReductionUnit, "preparedReductionUnit cannot be null");
        this.preparedMatmulUnit = null;
        this.executableUnit = new Cpu1ReductionExecutableUnit(preparedReductionUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedMatmulUnit preparedMatmulUnit) {
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = Objects.requireNonNull(preparedMatmulUnit, "preparedMatmulUnit cannot be null");
        this.executableUnit = new Cpu1MatmulExecutableUnit(preparedMatmulUnit);
    }

    public Cpu1PreparedArtifact(Cpu1ExecutableUnit executableUnit) {
        this.preparedUnit = null;
        this.preparedLayoutUnit = null;
        this.preparedReductionUnit = null;
        this.preparedMatmulUnit = null;
        this.executableUnit = Objects.requireNonNull(executableUnit, "executableUnit cannot be null");
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

    public Cpu1ExecutableUnit executableUnit() {
        return executableUnit;
    }

    public Cpu1WorkspaceSpec workspaceSpec() {
        return executableUnit.workspaceSpec();
    }

    @Override
    public void allocateRuntimeState(int nodeId, PreparedRuntimeStateAllocator allocator) {
        if (allocator == null) {
            return;
        }
        Cpu1WorkspaceSpec spec = workspaceSpec();
        if (spec.isEmpty()) {
            return;
        }
        allocator.putWorkspace(nodeId, Cpu1Workspace.allocate(spec));
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
                preparedReductionUnit,
                preparedMatmulUnit
        );
    }
}
