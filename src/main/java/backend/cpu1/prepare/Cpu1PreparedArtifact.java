package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ExecutableUnit;
import backend.cpu1.exec.Cpu1AttentionBackwardExecutableUnits;
import backend.cpu1.exec.Cpu1AttentionBackwardExecutableUnit;
import backend.cpu1.exec.Cpu1AttentionExecutableUnits;
import backend.cpu1.exec.Cpu1AvgPool2dExecutableUnits;
import backend.cpu1.exec.Cpu1Conv2dExecutableUnits;
import backend.cpu1.exec.Cpu1DTypeExecutableUnits;
import backend.cpu1.exec.Cpu1ElementwiseExecutableUnits;
import backend.cpu1.exec.Cpu1FusedElementwiseExecutableUnits;
import backend.cpu1.exec.Cpu1IndexExecutableUnits;
import backend.cpu1.exec.Cpu1LayoutExecutableUnits;
import backend.cpu1.exec.Cpu1LayerNormExecutableUnits;
import backend.cpu1.exec.Cpu1LossExecutableUnit;
import backend.cpu1.exec.Cpu1MatmulExecutableUnit;
import backend.cpu1.exec.Cpu1MaxPool2dExecutableUnits;
import backend.cpu1.exec.Cpu1MseLossExecutableUnits;
import backend.cpu1.exec.Cpu1ReductionExecutableUnits;
import backend.cpu1.exec.Cpu1RmsNormExecutableUnits;
import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.cpu1.trace.Cpu1TraceContributor;
import runtime.execution.ExecutionContext;
import graph.model.CompiledNode;
import runtime.execution.PreparedStepMetadata;
import runtime.execution.PreparedStepExecutable;
import runtime.execution.PreparedRuntimeStateAllocator;
import trace.backend.StepTraceContribution;

/**
 * Prepared execution artifact attached to cpu1 node metadata.
 */
public final class Cpu1PreparedArtifact implements PreparedStepExecutable {
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
    private final Cpu1PreparedLayerNormUnit preparedLayerNormUnit;
    private final Cpu1PreparedRmsNormUnit preparedRmsNormUnit;
    private final Cpu1PreparedMaxPool2dUnit preparedMaxPool2dUnit;
    private final Cpu1PreparedAvgPool2dUnit preparedAvgPool2dUnit;
    private final Cpu1PreparedConv2dUnit preparedConv2dUnit;
    private final Cpu1PreparedAttentionUnit preparedAttentionUnit;
    private final Cpu1PreparedAttentionBackwardUnit preparedAttentionBackwardUnit;
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1ElementwiseExecutableUnits.create(preparedUnit);
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1LayoutExecutableUnits.create(preparedLayoutUnit);
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1DTypeExecutableUnits.create(preparedDTypeUnit);
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1ReductionExecutableUnits.create(preparedReductionUnit);
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1MseLossExecutableUnits.create(preparedMseLossUnit);
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1FusedElementwiseExecutableUnits.create(preparedFusedElementwiseUnit);
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1IndexExecutableUnits.create(preparedIndexUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedLayerNormUnit preparedLayerNormUnit) {
        if (preparedLayerNormUnit == null) {
            throw new IllegalArgumentException("preparedLayerNormUnit cannot be null");
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
        this.preparedLayerNormUnit = preparedLayerNormUnit;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1LayerNormExecutableUnits.create(preparedLayerNormUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedRmsNormUnit preparedRmsNormUnit) {
        if (preparedRmsNormUnit == null) {
            throw new IllegalArgumentException("preparedRmsNormUnit cannot be null");
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = preparedRmsNormUnit;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1RmsNormExecutableUnits.create(preparedRmsNormUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedMaxPool2dUnit preparedMaxPool2dUnit) {
        if (preparedMaxPool2dUnit == null) {
            throw new IllegalArgumentException("preparedMaxPool2dUnit cannot be null");
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = preparedMaxPool2dUnit;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1MaxPool2dExecutableUnits.create(preparedMaxPool2dUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedAvgPool2dUnit preparedAvgPool2dUnit) {
        if (preparedAvgPool2dUnit == null) {
            throw new IllegalArgumentException("preparedAvgPool2dUnit cannot be null");
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = preparedAvgPool2dUnit;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1AvgPool2dExecutableUnits.create(preparedAvgPool2dUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedConv2dUnit preparedConv2dUnit) {
        if (preparedConv2dUnit == null) {
            throw new IllegalArgumentException("preparedConv2dUnit cannot be null");
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = preparedConv2dUnit;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1Conv2dExecutableUnits.create(preparedConv2dUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedAttentionUnit preparedAttentionUnit) {
        if (preparedAttentionUnit == null) {
            throw new IllegalArgumentException("preparedAttentionUnit cannot be null");
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = preparedAttentionUnit;
        this.preparedAttentionBackwardUnit = null;
        this.executableUnit = Cpu1AttentionExecutableUnits.create(preparedAttentionUnit);
    }

    public Cpu1PreparedArtifact(Cpu1PreparedAttentionBackwardUnit preparedAttentionBackwardUnit) {
        if (preparedAttentionBackwardUnit == null) {
            throw new IllegalArgumentException("preparedAttentionBackwardUnit cannot be null");
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = preparedAttentionBackwardUnit;
        this.executableUnit = Cpu1AttentionBackwardExecutableUnits.create(preparedAttentionBackwardUnit);
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
        this.preparedLayerNormUnit = null;
        this.preparedRmsNormUnit = null;
        this.preparedMaxPool2dUnit = null;
        this.preparedAvgPool2dUnit = null;
        this.preparedConv2dUnit = null;
        this.preparedAttentionUnit = null;
        this.preparedAttentionBackwardUnit = executableUnit instanceof Cpu1AttentionBackwardExecutableUnit attentionBackward
                ? attentionBackward.preparedUnit()
                : null;
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

    public Cpu1PreparedLayerNormUnit preparedLayerNormUnit() {
        if (preparedLayerNormUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared LayerNorm unit");
        }
        return preparedLayerNormUnit;
    }

    public Cpu1PreparedRmsNormUnit preparedRmsNormUnit() {
        if (preparedRmsNormUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared RMSNorm unit");
        }
        return preparedRmsNormUnit;
    }

    public Cpu1PreparedMaxPool2dUnit preparedMaxPool2dUnit() {
        if (preparedMaxPool2dUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared MAX_POOL2D unit");
        }
        return preparedMaxPool2dUnit;
    }

    public Cpu1PreparedAvgPool2dUnit preparedAvgPool2dUnit() {
        if (preparedAvgPool2dUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared AVG_POOL2D unit");
        }
        return preparedAvgPool2dUnit;
    }

    public Cpu1PreparedConv2dUnit preparedConv2dUnit() {
        if (preparedConv2dUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared CONV2D unit");
        }
        return preparedConv2dUnit;
    }

    public Cpu1PreparedAttentionUnit preparedAttentionUnit() {
        if (preparedAttentionUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared attention unit");
        }
        return preparedAttentionUnit;
    }

    public Cpu1PreparedAttentionBackwardUnit preparedAttentionBackwardUnit() {
        if (preparedAttentionBackwardUnit == null) {
            throw new IllegalStateException("This cpu1 artifact does not expose a prepared attention backward unit");
        }
        return preparedAttentionBackwardUnit;
    }

    public Cpu1ExecutableUnit executableUnit() {
        return executableUnit;
    }

    public Cpu1StorageKind storageKind() {
        if (preparedUnit != null) {
            return preparedUnit.storageKind();
        }
        if (preparedLayoutUnit != null) {
            return preparedLayoutUnit.storageKind();
        }
        if (preparedDTypeUnit != null) {
            return preparedDTypeUnit.storageKind();
        }
        if (preparedReductionUnit != null) {
            return preparedReductionUnit.storageKind();
        }
        if (preparedMatmulUnit != null) {
            return preparedMatmulUnit.storageKind();
        }
        if (preparedMseLossUnit != null) {
            return preparedMseLossUnit.storageKind();
        }
        if (preparedCrossEntropyLossUnit != null) {
            return preparedCrossEntropyLossUnit.storageKind();
        }
        if (preparedDenseCrossEntropyLossUnit != null) {
            return preparedDenseCrossEntropyLossUnit.storageKind();
        }
        if (preparedNllLossUnit != null) {
            return preparedNllLossUnit.storageKind();
        }
        if (preparedFusedElementwiseUnit != null) {
            return preparedFusedElementwiseUnit.storageKind();
        }
        if (preparedIndexUnit != null) {
            return preparedIndexUnit.storageKind();
        }
        if (preparedLayerNormUnit != null) {
            return preparedLayerNormUnit.storageKind();
        }
        if (preparedRmsNormUnit != null) {
            return preparedRmsNormUnit.storageKind();
        }
        if (preparedMaxPool2dUnit != null) {
            return preparedMaxPool2dUnit.storageKind();
        }
        if (preparedAvgPool2dUnit != null) {
            return preparedAvgPool2dUnit.storageKind();
        }
        if (preparedConv2dUnit != null) {
            return preparedConv2dUnit.storageKind();
        }
        if (preparedAttentionUnit != null) {
            return preparedAttentionUnit.storageKind();
        }
        if (preparedAttentionBackwardUnit != null) {
            return preparedAttentionBackwardUnit.storageKind();
        }
        throw new IllegalStateException("This cpu1 artifact does not expose a prepared storage kind");
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
        allocator.putWorkspace(nodeId, Cpu1ScratchBuffer.allocate(spec));
    }

    @Override
    public void execute(CompiledNode node, PreparedStepMetadata metadata, ExecutionContext context) {
        executableUnit.run(context);
    }

    @Override
    public StepTraceContribution traceContribution(
            CompiledNode node,
            PreparedStepMetadata metadata,
            ExecutionContext context
    ) {
        return Cpu1TraceContributor.traceContribution(
                node,
                preparedUnit,
                preparedLayoutUnit,
                preparedDTypeUnit,
                preparedReductionUnit,
                preparedMatmulUnit,
                preparedMseLossUnit,
                preparedCrossEntropyLossUnit,
                preparedDenseCrossEntropyLossUnit,
                preparedNllLossUnit,
                preparedFusedElementwiseUnit,
                preparedIndexUnit,
                preparedLayerNormUnit,
                preparedRmsNormUnit,
                preparedMaxPool2dUnit,
                preparedAvgPool2dUnit,
                preparedConv2dUnit,
                preparedAttentionUnit,
                preparedAttentionBackwardUnit
        );
    }
}
