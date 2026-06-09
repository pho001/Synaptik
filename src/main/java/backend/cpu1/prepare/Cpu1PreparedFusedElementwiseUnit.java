package backend.cpu1.prepare;

import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernel;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenRejectionReason;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
import tensor.DataType;

import java.util.List;

/**
 * Immutable prepare-time cpu1 fused elementwise unit.
 */
public final class Cpu1PreparedFusedElementwiseUnit {
    private final String unitId;
    private final List<Integer> orderedNodeIds;
    private final List<Integer> inputNodeIds;
    private final int outputNodeId;
    private final DataType outputDataType;
    private final int elementCount;
    private final int[] outputShape;
    private final Cpu1FusedExpressionPlan plan;
    private final Cpu1LayoutKind layoutKind;
    private final Cpu1StorageKind storageKind;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1FusedDispatchDecision dispatchDecision;
    private final Cpu1FusedCodegenRejectionReason codegenRejectionReason;
    private final Cpu1FusedCodegenKernel generatedKernel;
    private final boolean approximateExp;
    private final boolean approximateTanh;

    public Cpu1PreparedFusedElementwiseUnit(
            String unitId,
            List<Integer> orderedNodeIds,
            List<Integer> inputNodeIds,
            int outputNodeId,
            DataType outputDataType,
            int elementCount,
            int[] outputShape,
            Cpu1FusedExpressionPlan plan,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1LaunchPolicy launchPolicy,
            Cpu1LaunchConfig launchConfig,
            Cpu1FusedDispatchDecision dispatchDecision,
            Cpu1FusedCodegenRejectionReason codegenRejectionReason,
            Cpu1FusedCodegenKernel generatedKernel,
            boolean approximateExp,
            boolean approximateTanh
    ) {
        if (unitId == null) {
            throw new IllegalArgumentException("unitId cannot be null");
        }
        if (orderedNodeIds == null) {
            throw new IllegalArgumentException("orderedNodeIds cannot be null");
        }
        if (inputNodeIds == null) {
            throw new IllegalArgumentException("inputNodeIds cannot be null");
        }
        if (outputDataType == null) {
            throw new IllegalArgumentException("outputDataType cannot be null");
        }
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must be >= 0");
        }
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        if (layoutKind == null) {
            throw new IllegalArgumentException("layoutKind cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (launchPolicy == null) {
            throw new IllegalArgumentException("launchPolicy cannot be null");
        }
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (dispatchDecision == null) {
            throw new IllegalArgumentException("dispatchDecision cannot be null");
        }
        if (codegenRejectionReason == null) {
            throw new IllegalArgumentException("codegenRejectionReason cannot be null");
        }
        if (codegenRejectionReason == Cpu1FusedCodegenRejectionReason.NONE && generatedKernel == null) {
            throw new IllegalArgumentException("generatedKernel cannot be null when codegenRejectionReason is NONE");
        }
        this.unitId = unitId;
        this.orderedNodeIds = List.copyOf(orderedNodeIds);
        this.inputNodeIds = List.copyOf(inputNodeIds);
        this.outputNodeId = outputNodeId;
        this.outputDataType = outputDataType;
        this.elementCount = elementCount;
        this.outputShape = outputShape == null ? new int[0] : outputShape.clone();
        this.plan = plan;
        this.layoutKind = layoutKind;
        this.storageKind = storageKind;
        this.launchPolicy = launchPolicy;
        this.launchConfig = launchConfig;
        this.dispatchDecision = dispatchDecision;
        this.codegenRejectionReason = codegenRejectionReason;
        this.generatedKernel = generatedKernel;
        this.approximateExp = approximateExp;
        this.approximateTanh = approximateTanh;
    }

    public String unitId() {
        return unitId;
    }

    public List<Integer> orderedNodeIds() {
        return orderedNodeIds;
    }

    public List<Integer> inputNodeIds() {
        return inputNodeIds;
    }

    public int outputNodeId() {
        return outputNodeId;
    }

    public DataType outputDataType() {
        return outputDataType;
    }

    public int elementCount() {
        return elementCount;
    }

    public int[] outputShape() {
        return outputShape.clone();
    }

    public Cpu1FusedExpressionPlan plan() {
        return plan;
    }

    public Cpu1LayoutKind layoutKind() {
        return layoutKind;
    }

    public Cpu1StorageKind storageKind() {
        return storageKind;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1FusedDispatchDecision dispatchDecision() {
        return dispatchDecision;
    }

    public Cpu1FusedCodegenRejectionReason codegenRejectionReason() {
        return codegenRejectionReason;
    }

    public Cpu1FusedCodegenKernel generatedKernel() {
        return generatedKernel;
    }

    public boolean approximateExp() {
        return approximateExp;
    }

    public boolean approximateTanh() {
        return approximateTanh;
    }
}
