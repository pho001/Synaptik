package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.fused.ir.Cpu1FusedAccessKind;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.fused.codegen.asm.Cpu1FusedAsmCallEmitter;
import backend.cpu1.kernels.fused.codegen.asm.Cpu1FusedAsmIntrinsicRegistry;
import backend.cpu1.kernels.fused.codegen.support.Cpu1FusedGeneratedSupport;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

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
        if (!isSupportedOutputDType(computeType)) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
        }
        for (Cpu1FusedInputPlan input : expressionPlan.inputs()) {
            if (!isSupportedInputDType(input.dataType())) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
            }
            if (input.dataType() != DataType.BOOL && input.dataType() != computeType) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
            }
            if (!isSupportedAccess(input.accessKind(), loopKind)) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LAYOUT_OR_ACCESS;
            }
        }
        for (Cpu1FusedNodePlan node : expressionPlan.nodes()) {
            if (!isSupportedOutputDType(node.outputType())) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
            }
            if (node.outputType() != computeType) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
            }
            Cpu1FusedCodegenRejectionReason operationReason = operationRejectionReason(node, loopKind, computeType);
            if (operationReason != Cpu1FusedCodegenRejectionReason.NONE) {
                return operationReason;
            }
            if (!hasSupportedInputRefs(node)) {
                return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_DTYPE;
            }
        }
        if (loopKind != Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR
                && loopKind != Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR
                && loopKind != Cpu1FusedCodegenLoopKind.STRIDED_SCALAR) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LOOP_KIND;
        }
        if (storageKind != Cpu1StorageKind.JAVA_ARRAY
                && storageKind != Cpu1StorageKind.MEMORY_SEGMENT) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_STORAGE_KIND;
        }
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                && loopKind == Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_SEGMENT_VECTOR;
        }
        if (computeType == DataType.BFLOAT16 && loopKind == Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_BF16_VECTOR;
        }
        if (computeType == DataType.BFLOAT16 && storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_BF16_SEGMENT;
        }
        if (!isSupportedLayout(layoutKind, loopKind)) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_LAYOUT_OR_ACCESS;
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
        signature.append("|supportAbi=").append(Cpu1FusedGeneratedSupport.ABI_VERSION);
        signature.append("|helperTargets=").append(helperTargets(
                expressionPlan,
                computeType,
                loopKind,
                useFastExpApprox,
                useFastTanhApprox
        ));
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
        return new Cpu1FusedCodegenClassSignature(signature.toString());
    }

    private static List<String> helperTargets(
            Cpu1FusedExpressionPlan expressionPlan,
            DataType computeType,
            Cpu1FusedCodegenLoopKind loopKind,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (computeType != DataType.FLOAT32 && computeType != DataType.FLOAT64 && computeType != DataType.BFLOAT16
                || loopKind == Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR) {
            return List.of();
        }
        TreeSet<String> targets = new TreeSet<>();
        for (Cpu1FusedNodePlan node : expressionPlan.nodes()) {
            Operation.OpType opType = node.opType();
            switch (opType) {
                case RELU -> targets.add(Cpu1FusedAsmCallEmitter.reluTarget(computeType));
                case ABS -> targets.add(Cpu1FusedAsmCallEmitter.absTarget(computeType));
                case MIN, CLAMP_MAX -> targets.add(Cpu1FusedAsmCallEmitter.minTarget(computeType));
                case MAX, CLAMP_MIN -> targets.add(Cpu1FusedAsmCallEmitter.maxTarget(computeType));
                case EXP, FAST_EXP, LOG, TANH, FAST_TANH, ERF, SQRT, SIGMOID, FLOOR, CEIL, SIGN ->
                        targets.add(Cpu1FusedAsmCallEmitter.unaryIntrinsicTarget(
                                effectiveUnaryIntrinsic(opType, useFastExpApprox, useFastTanhApprox),
                                computeType
                        ));
                case POW, POW_TENSOR -> targets.add(Cpu1FusedAsmCallEmitter.powTarget(computeType));
                default -> {
                }
            }
        }
        return List.copyOf(targets);
    }

    private static Operation.OpType effectiveUnaryIntrinsic(
            Operation.OpType opType,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        return switch (opType) {
            case EXP -> useFastExpApprox ? Operation.OpType.FAST_EXP : Operation.OpType.EXP;
            case TANH -> useFastTanhApprox ? Operation.OpType.FAST_TANH : Operation.OpType.TANH;
            default -> opType;
        };
    }

    private static Cpu1FusedCodegenRejectionReason operationRejectionReason(
            Cpu1FusedNodePlan node,
            Cpu1FusedCodegenLoopKind loopKind,
            DataType computeType
    ) {
        Cpu1FusedCodegenRejectionReason scalarReason = Cpu1FusedAsmIntrinsicRegistry.rejectionReason(node);
        if (scalarReason != Cpu1FusedCodegenRejectionReason.NONE) {
            return scalarReason;
        }
        if (computeType == DataType.BFLOAT16 && !isSupportedBFloat16Operation(node.opType())) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_BF16_OPERATION;
        }
        if (loopKind == Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR && !isSupportedVectorOperation(node.opType())) {
            return Cpu1FusedCodegenRejectionReason.UNSUPPORTED_VECTOR_OPERATION;
        }
        return Cpu1FusedCodegenRejectionReason.NONE;
    }

    private static boolean isSupportedBFloat16Operation(Operation.OpType opType) {
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, ABS, RELU,
                 CLAMP_MIN, CLAMP_MAX, MUL_SCALAR, CONST_SCALAR, NOOP,
                 EXP, FAST_EXP, LOG, TANH, FAST_TANH, ERF, POW, POW_TENSOR,
                 SQRT, SIGMOID, FLOOR, CEIL, SIGN -> true;
            default -> false;
        };
    }

    private static boolean isSupportedVectorOperation(Operation.OpType opType) {
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, ABS, RELU,
                 CLAMP_MIN, CLAMP_MAX, MUL_SCALAR, CONST_SCALAR, NOOP -> true;
            default -> false;
        };
    }

    private boolean hasSupportedInputRefs(Cpu1FusedNodePlan node) {
        if (node.opType() == operations.Operation.OpType.CONST_SCALAR) {
            return node.inputRefs().isEmpty();
        }
        if (node.opType() == operations.Operation.OpType.WHERE) {
            if (node.inputRefs().size() != 3) {
                return false;
            }
            return refType(node.inputRefs().get(0)) == DataType.BOOL
                    && refType(node.inputRefs().get(1)) == computeType
                    && refType(node.inputRefs().get(2)) == computeType;
        }
        for (int ref : node.inputRefs()) {
            if (refType(ref) != computeType) {
                return false;
            }
        }
        return true;
    }

    private DataType refType(int ref) {
        if (ref < 0) {
            throw new IllegalArgumentException("ref must be >= 0");
        }
        if (ref < expressionPlan.inputCount()) {
            return expressionPlan.inputs().get(ref).dataType();
        }
        int nodeIndex = ref - expressionPlan.inputCount();
        if (nodeIndex < 0 || nodeIndex >= expressionPlan.nodeCount()) {
            throw new IllegalArgumentException("Invalid fused ref " + ref);
        }
        return expressionPlan.nodes().get(nodeIndex).outputType();
    }

    private static boolean isSupportedAccess(Cpu1FusedAccessKind accessKind, Cpu1FusedCodegenLoopKind loopKind) {
        if (loopKind == Cpu1FusedCodegenLoopKind.STRIDED_SCALAR) {
            return accessKind == Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                    || accessKind == Cpu1FusedAccessKind.OFFSET_CONTIGUOUS
                    || accessKind == Cpu1FusedAccessKind.DIRECT_STRIDED
                    || accessKind == Cpu1FusedAccessKind.OFFSET_STRIDED
                    || accessKind == Cpu1FusedAccessKind.BROADCAST_STRIDED;
        }
        return accessKind == Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                || accessKind == Cpu1FusedAccessKind.OFFSET_CONTIGUOUS;
    }

    private static boolean isSupportedInputDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16
                || dataType == DataType.BOOL;
    }

    private static boolean isSupportedOutputDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean isSupportedLayout(Cpu1LayoutKind layoutKind, Cpu1FusedCodegenLoopKind loopKind) {
        if (loopKind == Cpu1FusedCodegenLoopKind.CONTIGUOUS_VECTOR
                || loopKind == Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR) {
            return layoutKind == Cpu1LayoutKind.CONTIGUOUS;
        }
        return loopKind == Cpu1FusedCodegenLoopKind.STRIDED_SCALAR
                && layoutKind != Cpu1LayoutKind.CONTIGUOUS
                && layoutKind != Cpu1LayoutKind.BROADCAST_INNER;
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
