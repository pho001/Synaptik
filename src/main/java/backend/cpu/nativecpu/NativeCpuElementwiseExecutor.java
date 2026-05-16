package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.binary.BinaryElementwiseKernel;
import backend.cpu.kernels.elementwise.unary.UnaryElementwiseKernel;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.NativeFloat32Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * First native CPU non-BLAS elementwise slice for dense contiguous F32 tensors.
 */
public final class NativeCpuElementwiseExecutor {
    private static final EnumSet<Operation.OpType> POLICY_HANDLED_OPS = EnumSet.of(
            Operation.OpType.ADD,
            Operation.OpType.SUB,
            Operation.OpType.MUL,
            Operation.OpType.DIV,
            Operation.OpType.MIN,
            Operation.OpType.MAX,
            Operation.OpType.POW_TENSOR,
            Operation.OpType.NEG,
            Operation.OpType.INV,
            Operation.OpType.LOG,
            Operation.OpType.EXP,
            Operation.OpType.FAST_EXP,
            Operation.OpType.ERF,
            Operation.OpType.TANH,
            Operation.OpType.FAST_TANH,
            Operation.OpType.SQRT,
            Operation.OpType.ABS,
            Operation.OpType.FLOOR,
            Operation.OpType.CEIL,
            Operation.OpType.SIGN,
            Operation.OpType.RELU,
            Operation.OpType.SIGMOID
    );

    private NativeCpuElementwiseExecutor() {
    }

    public static boolean acceptsNativeInputs(Operation op, DataType dataType, backend.cpu.kernels.CpuNodeExecutionPlan plan, config.runtime.RuntimeConfig runtimeConfig) {
        if (!nativeRequested(runtimeConfig) || op == null || dataType != DataType.FLOAT32 || plan == null || plan.stridedPath()) {
            return false;
        }
        Operation.OpType opType = op.opType();
        if (isNativeUnaryOp(opType)) {
            return true;
        }
        if (opType == Operation.OpType.ADD) {
            ResolvedBroadcastPlan broadcastPlan = plan.broadcastPlan();
            return broadcastPlan == null || broadcastPlan.isNoBroadcast() || isLastDimBiasBroadcast(broadcastPlan);
        }
        if (isNativeSameShapeBinaryOp(opType)) {
            ResolvedBroadcastPlan broadcastPlan = plan.broadcastPlan();
            return broadcastPlan == null || broadcastPlan.isNoBroadcast();
        }
        return POLICY_HANDLED_OPS.contains(opType);
    }

    public static boolean tryRunUnary(UnaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (node.getDataType() != DataType.FLOAT32 || op == null || !isNativeUnaryOp(op.opType()) || context.nodePlan().stridedPath()) {
            fallbackUnary(kernel, inputs, node, context, fact, ineligibleReason(op, node, context));
            return true;
        }
        if (inputs == null || inputs.size() != 1) {
            fallbackUnary(kernel, inputs, node, context, fact, "native-kernel-ineligible:" + opLabel(op) + "-input-count");
            return true;
        }
        try {
            String label = opLabel(op);
            NativeFloat32Storage input = requireF32NativeInput(context, 0, label.toUpperCase());
            NativeFloat32Storage out = allocateF32(node, context, label);
            runDenseUnary(op.opType(), input, out, node.getFlatDataSize());
            out.markModified();
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU " + label.toUpperCase() + " wrote FLOAT32 output");
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackUnary(kernel, inputs, node, context, fact, "native-kernel-failed:" + opLabel(op) + ":" + safeMessage(t));
        }
        return true;
    }

    public static boolean tryRunBinary(BinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (node.getDataType() != DataType.FLOAT32 || op == null || !isNativeBinaryOp(op.opType()) || context.nodePlan().stridedPath()) {
            fallbackBinary(kernel, inputs, node, context, fact, ineligibleReason(op, node, context));
            return true;
        }
        ResolvedBroadcastPlan broadcastPlan = context.broadcastPlan();
        if (broadcastPlan != null && !broadcastPlan.isNoBroadcast() && !supportsBroadcast(op.opType(), broadcastPlan)) {
            fallbackBinary(kernel, inputs, node, context, fact, "native-kernel-ineligible:" + opLabel(op) + "-broadcast");
            return true;
        }
        if (inputs == null || inputs.size() != 2) {
            fallbackBinary(kernel, inputs, node, context, fact, "native-kernel-ineligible:" + opLabel(op) + "-shape");
            return true;
        }
        BiasBroadcastSpec biasSpec = null;
        if (broadcastPlan != null && !broadcastPlan.isNoBroadcast()) {
            biasSpec = biasBroadcastSpec(
                    broadcastPlan,
                    inputs.get(0).getFlatDataSize(),
                    inputs.get(1).getFlatDataSize(),
                    node.getFlatDataSize()
            );
            if (biasSpec == null) {
                fallbackBinary(kernel, inputs, node, context, fact, "native-kernel-ineligible:" + opLabel(op) + "-broadcast");
                return true;
            }
        } else if (inputs.get(0).getFlatDataSize() != node.getFlatDataSize()
                || inputs.get(1).getFlatDataSize() != node.getFlatDataSize()) {
            fallbackBinary(kernel, inputs, node, context, fact, "native-kernel-ineligible:" + opLabel(op) + "-shape");
            return true;
        }
        try {
            String label = opLabel(op);
            NativeFloat32Storage left = requireF32NativeInput(context, 0, label.toUpperCase());
            NativeFloat32Storage right = requireF32NativeInput(context, 1, label.toUpperCase());
            NativeFloat32Storage out = allocateF32(node, context, label);
            if (biasSpec == null) {
                runDenseBinary(op.opType(), left, right, out, node.getFlatDataSize());
            } else {
                runLastDimBiasAdd(left, right, out, node.getFlatDataSize(), biasSpec);
            }
            out.markModified();
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU " + label.toUpperCase() + " wrote FLOAT32 output");
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackBinary(kernel, inputs, node, context, fact, "native-kernel-failed:" + opLabel(op) + ":" + safeMessage(t));
        }
        return true;
    }

    private static void runDenseBinary(
            Operation.OpType opType,
            NativeFloat32Storage left,
            NativeFloat32Storage right,
            NativeFloat32Storage out,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            float leftValue = left.segment().get(JAVA_FLOAT, offset);
            float rightValue = right.segment().get(JAVA_FLOAT, offset);
            out.segment().set(JAVA_FLOAT, offset, applyBinary(opType, leftValue, rightValue));
        }
    }

    private static void runDenseUnary(Operation.OpType opType, NativeFloat32Storage input, NativeFloat32Storage out, int size) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            float value = input.segment().get(JAVA_FLOAT, offset);
            out.segment().set(JAVA_FLOAT, offset, applyUnary(opType, value));
        }
    }

    private static float applyBinary(Operation.OpType opType, float leftValue, float rightValue) {
        return switch (opType) {
            case ADD -> leftValue + rightValue;
            case SUB -> leftValue - rightValue;
            case MUL -> leftValue * rightValue;
            case DIV -> leftValue / rightValue;
            default -> throw new IllegalArgumentException("Unsupported native binary op: " + opType);
        };
    }

    private static float applyUnary(Operation.OpType opType, float value) {
        return switch (opType) {
            case NEG -> -value;
            case RELU -> Math.max(0.0f, value);
            default -> throw new IllegalArgumentException("Unsupported native unary op: " + opType);
        };
    }

    private static boolean isNativeUnaryOp(Operation.OpType opType) {
        return opType == Operation.OpType.NEG || opType == Operation.OpType.RELU;
    }

    private static boolean isNativeBinaryOp(Operation.OpType opType) {
        return opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV;
    }

    private static boolean isNativeSameShapeBinaryOp(Operation.OpType opType) {
        return opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV;
    }

    private static boolean supportsBroadcast(Operation.OpType opType, ResolvedBroadcastPlan broadcastPlan) {
        return opType == Operation.OpType.ADD && isLastDimBiasBroadcast(broadcastPlan);
    }

    private static void runLastDimBiasAdd(
            NativeFloat32Storage left,
            NativeFloat32Storage right,
            NativeFloat32Storage out,
            int size,
            BiasBroadcastSpec spec
    ) {
        for (int i = 0; i < size; i++) {
            long outOffset = (long) i * Float.BYTES;
            long biasOffset = (long) (i % spec.lastDim()) * Float.BYTES;
            float leftValue = left.segment().get(JAVA_FLOAT, spec.leftBias() ? biasOffset : outOffset);
            float rightValue = right.segment().get(JAVA_FLOAT, spec.leftBias() ? outOffset : biasOffset);
            out.segment().set(JAVA_FLOAT, outOffset, leftValue + rightValue);
        }
    }

    private static boolean isLastDimBiasBroadcast(ResolvedBroadcastPlan plan) {
        if (plan == null || plan.isNoBroadcast()) {
            return false;
        }
        return isFullOutputSide(plan.aEffStrides(), plan.outStrides()) && isLastDimBiasSide(plan.bEffStrides(), plan.outShape())
                || isLastDimBiasSide(plan.aEffStrides(), plan.outShape()) && isFullOutputSide(plan.bEffStrides(), plan.outStrides());
    }

    private static BiasBroadcastSpec biasBroadcastSpec(ResolvedBroadcastPlan plan, int leftSize, int rightSize, int outSize) {
        if (plan == null || plan.isNoBroadcast() || product(plan.outShape()) != outSize) {
            return null;
        }
        int[] shape = plan.outShape();
        int lastDim = shape[shape.length - 1];
        boolean leftFull = isFullOutputSide(plan.aEffStrides(), plan.outStrides());
        boolean rightFull = isFullOutputSide(plan.bEffStrides(), plan.outStrides());
        boolean leftBias = isLastDimBiasSide(plan.aEffStrides(), shape) && leftSize == lastDim;
        boolean rightBias = isLastDimBiasSide(plan.bEffStrides(), shape) && rightSize == lastDim;
        if (leftFull && leftSize == outSize && rightBias) {
            return new BiasBroadcastSpec(false, lastDim);
        }
        if (leftBias && rightFull && rightSize == outSize) {
            return new BiasBroadcastSpec(true, lastDim);
        }
        return null;
    }

    private static boolean isFullOutputSide(int[] effectiveStrides, int[] outStrides) {
        return Arrays.equals(effectiveStrides, outStrides);
    }

    private static boolean isLastDimBiasSide(int[] effectiveStrides, int[] outShape) {
        if (effectiveStrides == null || outShape == null || effectiveStrides.length != outShape.length || outShape.length < 2) {
            return false;
        }
        int last = effectiveStrides.length - 1;
        if (outShape[last] <= 0 || effectiveStrides[last] != 1) {
            return false;
        }
        for (int dim = 0; dim < last; dim++) {
            if (effectiveStrides[dim] != 0) {
                return false;
            }
        }
        return true;
    }

    private static int product(int[] values) {
        int product = 1;
        for (int value : values) {
            product *= value;
        }
        return product;
    }

    private static void fallbackUnary(UnaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context, NativeCpuKernelFact fact, String reason) {
        handleRequireNative(context, "unary elementwise", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        ElementwiseLoops.runUnary(kernel, inputs.get(0), node, context);
    }

    private static void fallbackBinary(BinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context, NativeCpuKernelFact fact, String reason) {
        handleRequireNative(context, "binary elementwise", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
    }

    private static void handleRequireNative(CpuKernelContext context, String family, String reason) {
        if (context.executionContext().runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native CPU execution required but " + family + " fell back to Java: " + reason);
        }
    }

    private static void requireCpuReadableInputs(CpuKernelContext context) {
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static NativeFloat32Storage requireF32NativeInput(CpuKernelContext context, int inputIndex, String op) {
        int inputNodeId = context.inputNodeIds().get(inputIndex);
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat32Storage f32) {
            return f32;
        }
        throw new IllegalStateException("native " + op + " requires FLOAT32 native input storage");
    }

    private static NativeFloat32Storage allocateF32(Tensor node, CpuKernelContext context, String label) {
        return (NativeFloat32Storage) new NativeCpuStorageFactory().allocate(
                DataType.FLOAT32,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-f32-" + label
        );
    }

    private static void publishTrace(CpuKernelContext context, NativeCpuKernelFact fact, String actualCpuStorage, String fallbackReason) {
        var runtime = context.executionContext().runtimeConfig();
        context.putRuntimeState(
                context.executionContext().runtimeTensorForNodeId(context.nodeId()),
                new NativeCpuTraceState(
                        runtime.cpuStorageProfile().name(),
                        runtime.nativeCpuFailurePolicy().name(),
                        "CPU_NATIVE",
                        actualCpuStorage,
                        fact.status().name(),
                        fact.family().name(),
                        fallbackReason
                )
        );
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return context != null && nativeRequested(context.executionContext().runtimeConfig());
    }

    private static boolean nativeRequested(config.runtime.RuntimeConfig runtimeConfig) {
        return runtimeConfig != null && runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String opLabel(Operation op) {
        return opType(op).name().toLowerCase();
    }

    private static String ineligibleReason(Operation op, Tensor node, CpuKernelContext context) {
        Operation.OpType opType = opType(op);
        if (node.getDataType() != DataType.FLOAT32) {
            return "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase();
        }
        if (context.nodePlan().stridedPath()) {
            return "native-kernel-ineligible:" + opType.name().toLowerCase() + "-strided";
        }
        return "native-kernel-unsupported:" + opType.name().toLowerCase();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }

    private record BiasBroadcastSpec(boolean leftBias, int lastDim) {
    }
}
