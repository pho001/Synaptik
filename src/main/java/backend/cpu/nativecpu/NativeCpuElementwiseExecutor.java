package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.binary.BinaryElementwiseKernel;
import backend.cpu.kernels.elementwise.unary.ScalarUnaryElementwiseKernel;
import backend.cpu.kernels.elementwise.unary.UnaryElementwiseKernel;
import backend.cpu.kernels.elementwise.where.WhereElementwiseKernel;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.cpu.kernels.layout.plan.ResolvedWhereBroadcastPlan;
import backend.memory.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import operations.elementwise.unary.mulScalar;
import tensor.DataType;
import tensor.NativeBFloat16Storage;
import tensor.NativeFloat32Storage;
import tensor.NativeFloat64Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;
import utils.FastTranscendentals;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * First native CPU non-BLAS elementwise slice for dense contiguous F32/F64 tensors.
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
            Operation.OpType.SIGMOID,
            Operation.OpType.WHERE
    );

    private NativeCpuElementwiseExecutor() {
    }

    public static boolean acceptsNativeInputs(Operation op, DataType dataType, backend.cpu.kernels.CpuNodeExecutionPlan plan, config.runtime.RuntimeConfig runtimeConfig) {
        if (!nativeRequested(runtimeConfig) || op == null || !supportsNativeElementwiseDType(dataType) || plan == null || plan.stridedPath()) {
            return false;
        }
        Operation.OpType opType = op.opType();
        if (isNativeUnaryOp(opType, dataType)) {
            return true;
        }
        if (dataType == DataType.FLOAT32 && opType == Operation.OpType.ADD) {
            ResolvedBroadcastPlan broadcastPlan = plan.broadcastPlan();
            return broadcastPlan == null || broadcastPlan.isNoBroadcast() || isLastDimBiasBroadcast(broadcastPlan);
        }
        if (isNativeBinaryOp(opType, dataType)) {
            ResolvedBroadcastPlan broadcastPlan = plan.broadcastPlan();
            return broadcastPlan == null || broadcastPlan.isNoBroadcast();
        }
        if (dataType == DataType.FLOAT32 && opType == Operation.OpType.WHERE) {
            ResolvedWhereBroadcastPlan whereBroadcastPlan = plan.whereBroadcastPlan();
            return whereBroadcastPlan == null || whereBroadcastPlan.isNoBroadcast();
        }
        return dataType == DataType.FLOAT32 && POLICY_HANDLED_OPS.contains(opType);
    }

    public static boolean requiresCpuReadableConditionOnly(Operation op, DataType dataType, backend.cpu.kernels.CpuNodeExecutionPlan plan, config.runtime.RuntimeConfig runtimeConfig) {
        return acceptsNativeInputs(op, dataType, plan, runtimeConfig)
                && op != null
                && op.opType() == Operation.OpType.WHERE;
    }

    public static boolean tryRunUnary(UnaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (!supportsNativeElementwiseDType(node.getDataType()) || op == null || !isNativeUnaryOp(op.opType(), node.getDataType()) || context.nodePlan().stridedPath()) {
            fallbackUnary(kernel, inputs, node, context, fact, ineligibleReason(op, node, context));
            return true;
        }
        if (inputs == null || inputs.size() != 1) {
            fallbackUnary(kernel, inputs, node, context, fact, "native-kernel-ineligible:" + opLabel(op) + "-input-count");
            return true;
        }
        try {
            String label = opLabel(op);
            NativeTensorStorage out;
            if (node.getDataType() == DataType.FLOAT64) {
                NativeFloat64Storage input = requireF64NativeInput(context, 0, label.toUpperCase());
                NativeFloat64Storage f64Out = allocateF64(node, context, label);
                runDenseUnaryF64(op, input, f64Out, node.getFlatDataSize());
                f64Out.markModified();
                out = f64Out;
            } else if (node.getDataType() == DataType.BFLOAT16) {
                NativeBFloat16Storage input = requireBF16NativeInput(context, 0, label.toUpperCase());
                NativeBFloat16Storage bf16Out = allocateBF16(node, context, label);
                runDenseUnaryBF16(op, input, bf16Out, node.getFlatDataSize());
                bf16Out.markModified();
                out = bf16Out;
            } else {
                NativeFloat32Storage input = requireF32NativeInput(context, 0, label.toUpperCase());
                NativeFloat32Storage f32Out = allocateF32(node, context, label);
                runDenseUnaryF32(op, input, f32Out, node.getFlatDataSize(), context.useFastExpApprox(), context.useFastTanhApprox());
                f32Out.markModified();
                out = f32Out;
            }
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU " + label.toUpperCase() + " wrote " + node.getDataType() + " output");
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackUnary(kernel, inputs, node, context, fact, "native-kernel-failed:" + opLabel(op) + ":" + safeMessage(t));
        }
        return true;
    }

    public static boolean tryRunScalarUnary(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (!supportsNativeElementwiseDType(node.getDataType()) || op == null || op.opType() != Operation.OpType.MUL_SCALAR || context.nodePlan().stridedPath()) {
            fallbackScalarUnary(kernel, parameterF64, parameterF32, inputs, node, context, fact, ineligibleReason(op, node, context));
            return true;
        }
        if (inputs == null || inputs.size() != 1) {
            fallbackScalarUnary(kernel, parameterF64, parameterF32, inputs, node, context, fact, "native-kernel-ineligible:" + opLabel(op) + "-input-count");
            return true;
        }
        try {
            String label = opLabel(op);
            NativeTensorStorage out;
            if (node.getDataType() == DataType.FLOAT64) {
                NativeFloat64Storage input = requireF64NativeInput(context, 0, label.toUpperCase());
                NativeFloat64Storage f64Out = allocateF64(node, context, label);
                runDenseUnaryF64(op, input, f64Out, node.getFlatDataSize());
                f64Out.markModified();
                out = f64Out;
            } else if (node.getDataType() == DataType.BFLOAT16) {
                NativeBFloat16Storage input = requireBF16NativeInput(context, 0, label.toUpperCase());
                NativeBFloat16Storage bf16Out = allocateBF16(node, context, label);
                runDenseUnaryBF16(op, input, bf16Out, node.getFlatDataSize());
                bf16Out.markModified();
                out = bf16Out;
            } else {
                NativeFloat32Storage input = requireF32NativeInput(context, 0, label.toUpperCase());
                NativeFloat32Storage f32Out = allocateF32(node, context, label);
                runDenseUnaryF32(op, input, f32Out, node.getFlatDataSize(), context.useFastExpApprox(), context.useFastTanhApprox());
                f32Out.markModified();
                out = f32Out;
            }
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU " + label.toUpperCase() + " wrote " + node.getDataType() + " output");
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackScalarUnary(kernel, parameterF64, parameterF32, inputs, node, context, fact, "native-kernel-failed:" + opLabel(op) + ":" + safeMessage(t));
        }
        return true;
    }

    public static boolean tryRunBinary(BinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (!supportsNativeElementwiseDType(node.getDataType()) || op == null || !isNativeBinaryOp(op.opType(), node.getDataType()) || context.nodePlan().stridedPath()) {
            fallbackBinary(kernel, inputs, node, context, fact, ineligibleReason(op, node, context));
            return true;
        }
        ResolvedBroadcastPlan broadcastPlan = context.broadcastPlan();
        if (broadcastPlan != null && !broadcastPlan.isNoBroadcast() && !supportsBroadcast(op.opType(), node.getDataType(), broadcastPlan)) {
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
            NativeTensorStorage out;
            if (node.getDataType() == DataType.FLOAT64) {
                NativeFloat64Storage left = requireF64NativeInput(context, 0, label.toUpperCase());
                NativeFloat64Storage right = requireF64NativeInput(context, 1, label.toUpperCase());
                NativeFloat64Storage f64Out = allocateF64(node, context, label);
                runDenseBinaryF64(op.opType(), left, right, f64Out, node.getFlatDataSize());
                f64Out.markModified();
                out = f64Out;
            } else if (node.getDataType() == DataType.BFLOAT16) {
                NativeBFloat16Storage left = requireBF16NativeInput(context, 0, label.toUpperCase());
                NativeBFloat16Storage right = requireBF16NativeInput(context, 1, label.toUpperCase());
                NativeBFloat16Storage bf16Out = allocateBF16(node, context, label);
                runDenseBinaryBF16(op.opType(), left, right, bf16Out, node.getFlatDataSize());
                bf16Out.markModified();
                out = bf16Out;
            } else {
                NativeFloat32Storage left = requireF32NativeInput(context, 0, label.toUpperCase());
                NativeFloat32Storage right = requireF32NativeInput(context, 1, label.toUpperCase());
                NativeFloat32Storage f32Out = allocateF32(node, context, label);
                if (biasSpec == null) {
                    runDenseBinaryF32(op.opType(), left, right, f32Out, node.getFlatDataSize());
                } else {
                    runLastDimBiasAdd(left, right, f32Out, node.getFlatDataSize(), biasSpec);
                }
                f32Out.markModified();
                out = f32Out;
            }
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU " + label.toUpperCase() + " wrote " + node.getDataType() + " output");
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackBinary(kernel, inputs, node, context, fact, "native-kernel-failed:" + opLabel(op) + ":" + safeMessage(t));
        }
        return true;
    }

    public static boolean tryRunWhere(WhereElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), node.getDataType());
        if (op == null || op.opType() != Operation.OpType.WHERE || context.nodePlan().stridedPath()) {
            fallbackWhere(kernel, inputs, node, context, fact, ineligibleReason(op, node, context));
            return true;
        }
        if (node.getDataType() != DataType.FLOAT32) {
            fallbackWhere(kernel, inputs, node, context, fact, "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase());
            return true;
        }
        if (inputs == null || inputs.size() != 3) {
            fallbackWhere(kernel, inputs, node, context, fact, "native-kernel-ineligible:where-input-count");
            return true;
        }
        if (inputs.get(0).getDataType() != DataType.BOOL
                || inputs.get(1).getDataType() != DataType.FLOAT32
                || inputs.get(2).getDataType() != DataType.FLOAT32) {
            fallbackWhere(kernel, inputs, node, context, fact, "native-kernel-ineligible:where-dtype");
            return true;
        }
        ResolvedWhereBroadcastPlan whereBroadcastPlan = context.whereBroadcastPlan();
        if (whereBroadcastPlan != null && !whereBroadcastPlan.isNoBroadcast()) {
            fallbackWhere(kernel, inputs, node, context, fact, "native-kernel-ineligible:where-broadcast");
            return true;
        }
        int size = node.getFlatDataSize();
        if (inputs.get(0).getFlatDataSize() != size
                || inputs.get(1).getFlatDataSize() != size
                || inputs.get(2).getFlatDataSize() != size) {
            fallbackWhere(kernel, inputs, node, context, fact, "native-kernel-ineligible:where-shape");
            return true;
        }
        try {
            byte[] condition = inputs.get(0).getBoolData();
            NativeFloat32Storage ifTrue = requireF32NativeInput(context, 1, "WHERE");
            NativeFloat32Storage ifFalse = requireF32NativeInput(context, 2, "WHERE");
            NativeFloat32Storage out = allocateF32(node, context, "where");
            runDenseWhere(kernel, condition, ifTrue, ifFalse, out, size);
            out.markModified();
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU WHERE wrote FLOAT32 output");
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackWhere(kernel, inputs, node, context, fact, "native-kernel-failed:where:" + safeMessage(t));
        }
        return true;
    }

    private static void runDenseBinaryF32(
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

    private static void runDenseBinaryF64(
            Operation.OpType opType,
            NativeFloat64Storage left,
            NativeFloat64Storage right,
            NativeFloat64Storage out,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            double leftValue = left.segment().get(JAVA_DOUBLE, offset);
            double rightValue = right.segment().get(JAVA_DOUBLE, offset);
            out.segment().set(JAVA_DOUBLE, offset, applyBinary(opType, leftValue, rightValue));
        }
    }

    private static void runDenseBinaryBF16(
            Operation.OpType opType,
            NativeBFloat16Storage left,
            NativeBFloat16Storage right,
            NativeBFloat16Storage out,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            float leftValue = CpuDTypeOps.fromBFloat16Bits(left.getBFloat16BitsAt(i));
            float rightValue = CpuDTypeOps.fromBFloat16Bits(right.getBFloat16BitsAt(i));
            out.setBFloat16BitsAt(i, CpuDTypeOps.toBFloat16Bits(applyBinary(opType, leftValue, rightValue)));
        }
    }

    private static void runDenseUnaryF32(
            Operation op,
            NativeFloat32Storage input,
            NativeFloat32Storage out,
            int size,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        float scalar = scalarParameter(op);
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            float value = input.segment().get(JAVA_FLOAT, offset);
            out.segment().set(JAVA_FLOAT, offset, applyUnary(op.opType(), value, scalar, useFastExpApprox, useFastTanhApprox));
        }
    }

    private static void runDenseUnaryF64(Operation op, NativeFloat64Storage input, NativeFloat64Storage out, int size) {
        double scalar = scalarParameterF64(op);
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            double value = input.segment().get(JAVA_DOUBLE, offset);
            out.segment().set(JAVA_DOUBLE, offset, applyUnary(op.opType(), value, scalar));
        }
    }

    private static void runDenseUnaryBF16(Operation op, NativeBFloat16Storage input, NativeBFloat16Storage out, int size) {
        float scalar = scalarParameter(op);
        for (int i = 0; i < size; i++) {
            float value = CpuDTypeOps.fromBFloat16Bits(input.getBFloat16BitsAt(i));
            out.setBFloat16BitsAt(i, CpuDTypeOps.toBFloat16Bits(applyUnaryBF16(op.opType(), value, scalar)));
        }
    }

    private static void runDenseWhere(
            WhereElementwiseKernel kernel,
            byte[] condition,
            NativeFloat32Storage ifTrue,
            NativeFloat32Storage ifFalse,
            NativeFloat32Storage out,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            float trueValue = ifTrue.segment().get(JAVA_FLOAT, offset);
            float falseValue = ifFalse.segment().get(JAVA_FLOAT, offset);
            out.segment().set(JAVA_FLOAT, offset, kernel.applyF32(condition[i], trueValue, falseValue));
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

    private static double applyBinary(Operation.OpType opType, double leftValue, double rightValue) {
        return switch (opType) {
            case ADD -> leftValue + rightValue;
            case SUB -> leftValue - rightValue;
            case MUL -> leftValue * rightValue;
            case DIV -> leftValue / rightValue;
            default -> throw new IllegalArgumentException("Unsupported native binary op: " + opType);
        };
    }

    private static float applyUnary(
            Operation.OpType opType,
            float value,
            float scalar,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        return switch (opType) {
            case MUL_SCALAR -> value * scalar;
            case NEG -> -value;
            case RELU -> Math.max(0.0f, value);
            case LOG -> (float) Math.log(value);
            case EXP -> useFastExpApprox ? FastTranscendentals.fastExpF32(value) : (float) Math.exp(value);
            case FAST_EXP -> FastTranscendentals.fastExpF32(value);
            case SQRT -> (float) Math.sqrt(value);
            case ABS -> Math.abs(value);
            case TANH -> useFastTanhApprox ? FastTranscendentals.fastTanhF32(value) : (float) Math.tanh(value);
            case FAST_TANH -> FastTranscendentals.fastTanhF32(value);
            case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-value));
            default -> throw new IllegalArgumentException("Unsupported native unary op: " + opType);
        };
    }

    private static double applyUnary(Operation.OpType opType, double value, double scalar) {
        return switch (opType) {
            case MUL_SCALAR -> value * scalar;
            case NEG -> -value;
            default -> throw new IllegalArgumentException("Unsupported native unary op: " + opType);
        };
    }

    private static float applyUnaryBF16(Operation.OpType opType, float value, float scalar) {
        return switch (opType) {
            case MUL_SCALAR -> value * scalar;
            case NEG -> -value;
            case RELU -> Math.max(0.0f, value);
            case ABS -> Math.abs(value);
            default -> throw new IllegalArgumentException("Unsupported native BF16 unary op: " + opType);
        };
    }

    private static boolean isNativeUnaryOp(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.FLOAT64) {
            return opType == Operation.OpType.MUL_SCALAR
                    || opType == Operation.OpType.NEG;
        }
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.MUL_SCALAR
                    || opType == Operation.OpType.NEG
                    || opType == Operation.OpType.RELU
                    || opType == Operation.OpType.ABS;
        }
        return dataType == DataType.FLOAT32
                && (opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.NEG
                || opType == Operation.OpType.RELU
                || opType == Operation.OpType.LOG
                || opType == Operation.OpType.EXP
                || opType == Operation.OpType.FAST_EXP
                || opType == Operation.OpType.SQRT
                || opType == Operation.OpType.ABS
                || opType == Operation.OpType.TANH
                || opType == Operation.OpType.FAST_TANH
                || opType == Operation.OpType.SIGMOID);
    }

    private static float scalarParameter(Operation op) {
        if (op instanceof mulScalar mul) {
            return mul.getScalarF32();
        }
        return 0.0f;
    }

    private static double scalarParameterF64(Operation op) {
        if (op instanceof mulScalar mul) {
            return mul.getScalar();
        }
        return 0.0d;
    }

    private static boolean isNativeBinaryOp(Operation.OpType opType, DataType dataType) {
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16)
                && (opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV);
    }

    private static boolean supportsBroadcast(Operation.OpType opType, DataType dataType, ResolvedBroadcastPlan broadcastPlan) {
        return dataType == DataType.FLOAT32 && opType == Operation.OpType.ADD && isLastDimBiasBroadcast(broadcastPlan);
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

    private static void fallbackScalarUnary(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            NativeCpuKernelFact fact,
            String reason
    ) {
        handleRequireNative(context, "scalar unary elementwise", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        ElementwiseLoops.runScalarUnary(kernel, parameterF64, parameterF32, inputs.get(0), node, context);
    }

    private static void fallbackBinary(BinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context, NativeCpuKernelFact fact, String reason) {
        handleRequireNative(context, "binary elementwise", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
    }

    private static void fallbackWhere(WhereElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context, NativeCpuKernelFact fact, String reason) {
        handleRequireNative(context, "where elementwise", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        ElementwiseLoops.runWhere(kernel, inputs.get(0), inputs.get(1), inputs.get(2), node, context);
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

    private static NativeFloat64Storage requireF64NativeInput(CpuKernelContext context, int inputIndex, String op) {
        int inputNodeId = context.inputNodeIds().get(inputIndex);
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeFloat64Storage f64) {
            return f64;
        }
        throw new IllegalStateException("native " + op + " requires FLOAT64 native input storage");
    }

    private static NativeBFloat16Storage requireBF16NativeInput(CpuKernelContext context, int inputIndex, String op) {
        int inputNodeId = context.inputNodeIds().get(inputIndex);
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeBFloat16Storage bf16) {
            return bf16;
        }
        throw new IllegalStateException("native " + op + " requires BFLOAT16 native input storage");
    }

    private static NativeFloat32Storage allocateF32(Tensor node, CpuKernelContext context, String label) {
        return (NativeFloat32Storage) new NativeCpuStorageFactory().allocate(
                DataType.FLOAT32,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-f32-" + label
        );
    }

    private static NativeFloat64Storage allocateF64(Tensor node, CpuKernelContext context, String label) {
        return (NativeFloat64Storage) new NativeCpuStorageFactory().allocate(
                DataType.FLOAT64,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-f64-" + label
        );
    }

    private static NativeBFloat16Storage allocateBF16(Tensor node, CpuKernelContext context, String label) {
        return (NativeBFloat16Storage) new NativeCpuStorageFactory().allocate(
                DataType.BFLOAT16,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-bf16-" + label
        );
    }

    private static void publishTrace(CpuKernelContext context, NativeCpuKernelFact fact, String actualCpuStorage, String fallbackReason) {
        var runtime = context.executionContext().runtimeConfig();
        Tensor runtimeTensor = context.executionContext().runtimeTensorForNodeId(context.nodeId());
        boolean bf16Promoted = runtimeTensor.getDataType() == DataType.BFLOAT16
                && "CPU_NATIVE".equals(actualCpuStorage)
                && (fallbackReason == null || fallbackReason.isBlank());
        context.putRuntimeState(
                runtimeTensor,
                new NativeCpuTraceState(
                        runtime.cpuStorageProfile().name(),
                        runtime.nativeCpuFailurePolicy().name(),
                        "CPU_NATIVE",
                        actualCpuStorage,
                        fact.status().name(),
                        fact.family().name(),
                        fallbackReason,
                        bf16Promoted ? "BF16" : "",
                        bf16Promoted ? "F32_PROMOTED" : ""
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
        if (!supportsNativeElementwiseDType(node.getDataType())) {
            return "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase();
        }
        if (context.nodePlan().stridedPath()) {
            return "native-kernel-ineligible:" + opType.name().toLowerCase() + "-strided";
        }
        return "native-kernel-unsupported:" + opType.name().toLowerCase();
    }

    private static boolean supportsNativeElementwiseDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }

    private record BiasBroadcastSpec(boolean leftBias, int lastDim) {
    }
}
