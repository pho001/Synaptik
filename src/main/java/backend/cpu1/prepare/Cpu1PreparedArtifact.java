package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ExecutableUnit;
import backend.cpu1.exec.Cpu1ElementwiseExecutableUnit;
import backend.cpu1.exec.Cpu1LayoutExecutableUnit;
import backend.cpu1.exec.Cpu1MatmulExecutableUnit;
import backend.cpu1.exec.Cpu1ReductionExecutableUnit;
import backend.cpu1.exec.Cpu1Workspace;
import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.plan.PreparedExecutionArtifact;
import graph.execution.plan.PreparedRuntimeStateAllocator;
import graph.execution.trace.DispatchTraceMetadata;
import graph.execution.trace.LayoutTraceMetadata;
import graph.execution.trace.MatMulTraceMetadata;
import graph.execution.trace.ReductionTraceMetadata;
import graph.execution.trace.StepTraceContribution;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;

import java.util.LinkedHashMap;
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
        if (preparedLayoutUnit != null) {
            return layoutTrace(node, preparedLayoutUnit);
        }
        if (preparedReductionUnit != null) {
            return reductionTrace(preparedReductionUnit);
        }
        if (preparedMatmulUnit != null) {
            return matmulTrace(preparedMatmulUnit);
        }
        return StepTraceContribution.empty();
    }

    private static StepTraceContribution layoutTrace(CompiledNode node, Cpu1PreparedLayoutUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1LayoutKernelId", unit.kernelId().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1VectorizationKind", unit.vectorizationKind().name());
        attrs.put("cpu1LaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1LaunchChunkSize", unit.launchConfig().chunkSize());
        attrs.put("cpu1MaterializeThreshold", unit.materializeThreshold());
        LayoutTraceMetadata layout = new LayoutTraceMetadata(
                node.storageOffset(),
                node.contiguous(),
                false,
                unit.kernelId().name()
        );
        DispatchTraceMetadata dispatch = new DispatchTraceMetadata(
                unit.vectorizationKind().name(),
                layoutVectorWidth(unit),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                unit.launchConfig().chunkSize()
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                layout,
                dispatch,
                null,
                null,
                null,
                null
        );
    }

    private static int layoutVectorWidth(Cpu1PreparedLayoutUnit unit) {
        return switch (unit.vectorizationKind()) {
            case SCALAR -> 1;
            case VECTOR -> switch (unit.dataType()) {
                case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
                case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
                case BFLOAT16 -> ShortVector.SPECIES_PREFERRED.length();
                case BOOL -> ByteVector.SPECIES_PREFERRED.length();
                case INT32, INT64 -> 1;
            };
        };
    }

    private static StepTraceContribution reductionTrace(Cpu1PreparedReductionUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1ReductionKernelId", unit.kernelId().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1ReductionAxis", unit.axis());
        attrs.put("cpu1ReductionAxisSize", unit.axisSize());
        attrs.put("cpu1ReductionInnerSize", unit.innerSize());
        attrs.put("cpu1ReductionOuterSize", unit.outerSize());
        attrs.put("cpu1ReductionOutputElements", unit.outputElementCount());
        attrs.put("cpu1ReductionKeepDims", unit.keepDims());
        ReductionTraceMetadata reduction = new ReductionTraceMetadata(
                unit.opType().name(),
                1,
                unit.outputElementCount(),
                1,
                unit.dataType() == tensor.DataType.BFLOAT16 ? "F32_ACCUMULATE" : unit.dataType().name()
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                null,
                null,
                reduction,
                null,
                null,
                null
        );
    }

    private static StepTraceContribution matmulTrace(Cpu1PreparedMatmulUnit unit) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("cpu1KernelId", unit.kernelId().name());
        attrs.put("cpu1MatmulKernelId", unit.kernelId().name());
        attrs.put("cpu1MatmulRoute", unit.route().name());
        attrs.put("cpu1StorageKind", unit.storageKind().name());
        attrs.put("cpu1MatmulVectorizationKind", unit.vectorizationKind().name());
        attrs.put("cpu1MatmulVectorWidth", matmulVectorWidth(unit));
        attrs.put("cpu1MatmulBatchCount", unit.batchCount());
        attrs.put("cpu1MatmulM", unit.m());
        attrs.put("cpu1MatmulN", unit.n());
        attrs.put("cpu1MatmulK", unit.k());
        attrs.put("cpu1MatmulWork", unit.work());
        attrs.put("cpu1MatmulLaunchWorkers", unit.launchConfig().workerCount());
        attrs.put("cpu1MatmulLaunchChunkSize", unit.launchConfig().chunkSize());
        MatMulTraceMetadata matMul = new MatMulTraceMetadata(
                false,
                false,
                "",
                "",
                "",
                unit.route().name(),
                unit.storageKind().name(),
                "",
                unit.storageKind().name(),
                unit.storageKind().name(),
                "",
                false,
                false,
                false,
                false,
                unit.dataType() == tensor.DataType.BFLOAT16 ? "JAVA" : "",
                unit.dataType() == tensor.DataType.BFLOAT16 ? "JAVA" : "",
                unit.dataType() == tensor.DataType.BFLOAT16 ? "F32_PROMOTED" : unit.dataType().name(),
                unit.dataType().name(),
                0L,
                0L,
                0L,
                "SINGLE_THREAD",
                "",
                false,
                unit.m(),
                unit.n(),
                unit.k(),
                1,
                unit.work(),
                unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR
                        ? "JAVA_VECTOR_PACKED_B"
                        : "JAVA_SCALAR"
        );
        return new StepTraceContribution(
                unit.kernelId().name(),
                attrs,
                null,
                null,
                null,
                null,
                matMul,
                null,
                null
        );
    }

    private static int matmulVectorWidth(Cpu1PreparedMatmulUnit unit) {
        if (unit.vectorizationKind() == Cpu1VectorizationKind.SCALAR) {
            return 1;
        }
        return switch (unit.dataType()) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            case BFLOAT16 -> ShortVector.SPECIES_PREFERRED.length();
            case BOOL -> ByteVector.SPECIES_PREFERRED.length();
            case INT32, INT64 -> 1;
        };
    }
}
