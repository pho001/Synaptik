package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.fused.ir.Cpu1FusedAccessKind;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.ArrayList;
import java.util.List;

/**
 * Prepare-time input for cpu1 fused ASM code generation.
 */
public record Cpu1FusedCodegenPlan(
        Cpu1FusedExpressionPlan expressionPlan,
        DataType computeType,
        Cpu1LayoutKind layoutKind,
        Cpu1StorageKind storageKind,
        Cpu1FusedCodegenLoopKind loopKind,
        boolean useFastExpApprox,
        boolean useFastTanhApprox,
        Cpu1FusedCodegenClassSignature classSignature
) {
    public Cpu1FusedCodegenPlan {
        if (expressionPlan == null) {
            throw new IllegalArgumentException("expressionPlan cannot be null");
        }
        if (computeType == null) {
            throw new IllegalArgumentException("computeType cannot be null");
        }
        if (layoutKind == null) {
            throw new IllegalArgumentException("layoutKind cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (loopKind == null) {
            throw new IllegalArgumentException("loopKind cannot be null");
        }
        if (classSignature == null) {
            throw new IllegalArgumentException("classSignature cannot be null");
        }
    }

    public static Cpu1FusedCodegenPlan from(
            Cpu1FusedExpressionPlan expressionPlan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1FusedCodegenLoopKind loopKind,
            Cpu1PrepareConfig config
    ) {
        if (config == null) {
            throw new IllegalArgumentException("config cannot be null");
        }
        boolean useFastExpApprox = config.useFastExpApprox();
        boolean useFastTanhApprox = config.useFastTanhApprox();
        return new Cpu1FusedCodegenPlan(
                expressionPlan,
                computeType,
                layoutKind,
                storageKind,
                loopKind,
                useFastExpApprox,
                useFastTanhApprox,
                buildClassSignature(
                        expressionPlan,
                        computeType,
                        layoutKind,
                        storageKind,
                        loopKind,
                        useFastExpApprox,
                        useFastTanhApprox
                )
        );
    }

    public Cpu1FusedCodegenRejectionReason rejectionReason() {
        if (!isSupportedDType(computeType)) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
        }
        for (Cpu1FusedInputPlan input : expressionPlan.inputs()) {
            if (!isSupportedDType(input.dataType())) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
            }
            if (!isSupportedAccess(input.accessKind())) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LAYOUT_OR_ACCESS;
            }
        }
        for (Cpu1FusedNodePlan node : expressionPlan.nodes()) {
            if (!isSupportedDType(node.outputType())) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
            }
            Cpu1FusedCodegenRejectionReason operationReason = operationRejectionReason(node);
            if (operationReason != Cpu1FusedCodegenRejectionReason.NONE) {
                return operationReason;
            }
        }
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY && storageKind != Cpu1StorageKind.MEMORY_SEGMENT) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_STORAGE_KIND;
        }
        if (layoutKind != Cpu1LayoutKind.CONTIGUOUS) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LAYOUT_OR_ACCESS;
        }
        if (loopKind != Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR
                && loopKind != Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR
                && loopKind != Cpu1FusedCodegenLoopKind.STRIDED_SCALAR) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LOOP_KIND;
        }
        return Cpu1FusedCodegenRejectionReason.NONE;
    }

    private static Cpu1FusedCodegenClassSignature buildClassSignature(
            Cpu1FusedExpressionPlan expressionPlan,
            DataType computeType,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1FusedCodegenLoopKind loopKind,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (expressionPlan == null) {
            throw new IllegalArgumentException("expressionPlan cannot be null");
        }
        StringBuilder signature = new StringBuilder(256);
        signature.append("cpu1-fused:v1");
        signature.append("|supportAbi=0");
        signature.append("|storage=").append(storageKind);
        signature.append("|layout=").append(layoutKind);
        signature.append("|loop=").append(loopKind);
        signature.append("|compute=").append(computeType);
        signature.append("|fastExp=").append(useFastExpApprox);
        signature.append("|fastTanh=").append(useFastTanhApprox);
        signature.append("|inputs=");
        for (Cpu1FusedInputPlan input : expressionPlan.inputs()) {
            signature.append(input.ref())
                    .append(':')
                    .append(input.dataType())
                    .append(':')
                    .append(input.accessKind())
                    .append(":rank")
                    .append(input.logicalOutputShape().length)
                    .append(';');
        }
        signature.append("|nodes=");
        for (Cpu1FusedNodePlan node : expressionPlan.nodes()) {
            signature.append(node.index())
                    .append(':')
                    .append(node.opType())
                    .append(':')
                    .append(node.inputRefs())
                    .append("->")
                    .append(node.outputRef())
                    .append(':')
                    .append(node.outputType())
                    .append(":scalar=")
                    .append(node.scalarParameter().present())
                    .append(';');
        }
        signature.append("|outputRef=").append(expressionPlan.outputRef());
        return new Cpu1FusedCodegenClassSignature(signature.toString(), 0, List.of());
    }

    private static Cpu1FusedCodegenRejectionReason operationRejectionReason(Cpu1FusedNodePlan node) {
        if (node.scalarParameter().present()) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_SCALAR_BINDING;
        }
        return switch (node.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, ABS, RELU, CLAMP_MIN, CLAMP_MAX, NOOP ->
                    Cpu1FusedCodegenRejectionReason.NONE;
            case EXP, FAST_EXP, LOG, TANH, FAST_TANH, ERF, POW, POW_TENSOR, SQRT, SIGMOID ->
                    Cpu1FusedCodegenRejectionReason.UNSUPPORTED_INTRINSIC;
            default -> Cpu1FusedCodegenRejectionReason.UNSUPPORTED_OPERATION;
        };
    }

    private static boolean isSupportedAccess(Cpu1FusedAccessKind accessKind) {
        return accessKind == Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                || accessKind == Cpu1FusedAccessKind.OFFSET_CONTIGUOUS;
    }

    private static boolean isSupportedDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    public List<Cpu1FusedNodePlan> scalarBindingNodes() {
        List<Cpu1FusedNodePlan> nodes = new ArrayList<>();
        for (Cpu1FusedNodePlan node : expressionPlan.nodes()) {
            if (node.scalarParameter().present()) {
                nodes.add(node);
            }
        }
        return List.copyOf(nodes);
    }
}
