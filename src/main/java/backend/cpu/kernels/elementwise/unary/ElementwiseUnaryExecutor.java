package backend.cpu.kernels.elementwise.unary;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.nativecpu.CpuNativeTraceSupport;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.ElementwiseNativeSupport;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

public final class ElementwiseUnaryExecutor {
    private ElementwiseUnaryExecutor() {}

    public static void execute(UnaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Unary elementwise executor requires exactly 1 input.");
        }
        if (!ElementwiseNativeSupport.nativeRequested(context)) {
            ElementwiseLoops.runUnary(kernel, inputs.get(0), node, context);
            return;
        }
        Operation.OpType opType = opType(context.executionOperation());
        String ineligibleReason = nativeIneligibleReason(opType, inputs, node, context);
        if (!ineligibleReason.isBlank()) {
            fallbackUnary(kernel, inputs, node, context, ineligibleReason);
            return;
        }
        try {
            String label = opLabel(opType);
            NativeTensorStorage inputStorage = ElementwiseNativeSupport.requireNativeInput(context, 0, node.getDataType(), label.toUpperCase());
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(node, context, "unary-segment-" + label);
            runDenseSegment(kernel, node.getDataType(), inputStorage.segment(), outputStorage.segment(), node.getFlatDataSize(), context.dispatchHints());
            attachNativeOutput(node, context, label, outputStorage);
        } catch (Throwable t) {
            fallbackUnary(kernel, inputs, node, context,
                    "native-kernel-failed:" + opLabel(opType) + ":" + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    public static void execute(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Scalar unary elementwise executor requires exactly 1 input.");
        }
        if (!ElementwiseNativeSupport.nativeRequested(context)) {
            ElementwiseLoops.runScalarUnary(kernel, parameterF64, parameterF32, inputs.get(0), node, context);
            return;
        }
        Operation.OpType opType = opType(context.executionOperation());
        String ineligibleReason = nativeScalarIneligibleReason(opType, inputs, node, context);
        if (!ineligibleReason.isBlank()) {
            fallbackScalarUnary(kernel, parameterF64, parameterF32, inputs, node, context, ineligibleReason);
            return;
        }
        try {
            String label = opLabel(opType);
            NativeTensorStorage inputStorage = ElementwiseNativeSupport.requireNativeInput(context, 0, node.getDataType(), label.toUpperCase());
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(node, context, "scalar-unary-segment-" + label);
            runDenseSegment(kernel, parameterF64, parameterF32, node.getDataType(), inputStorage.segment(), outputStorage.segment(), node.getFlatDataSize(), context.dispatchHints());
            attachNativeOutput(node, context, label, outputStorage);
        } catch (Throwable t) {
            fallbackScalarUnary(kernel, parameterF64, parameterF32, inputs, node, context,
                    "native-kernel-failed:" + opLabel(opType) + ":" + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    static void runDenseSegment(
            UnaryElementwiseKernel kernel,
            DataType dtype,
            MemorySegment input,
            MemorySegment out,
            int length,
            ResolvedDispatchHints hints
    ) {
        ElementwiseRangeLoop.runScalar(length, hints, (start, end) -> {
            switch (dtype) {
                case FLOAT64 -> kernel.runSegmentF64(input, out, start, end);
                case FLOAT32 -> kernel.runSegmentF32(input, out, start, end);
                case BFLOAT16 -> kernel.runSegmentBF16(input, out, start, end);
                case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Unary segment execution does not support dtype: " + dtype);
            }
        });
    }

    static void runDenseSegment(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            DataType dtype,
            MemorySegment input,
            MemorySegment out,
            int length,
            ResolvedDispatchHints hints
    ) {
        ElementwiseRangeLoop.runScalar(length, hints, (start, end) -> {
            switch (dtype) {
                case FLOAT64 -> kernel.runSegmentF64(input, parameterF64, out, start, end);
                case FLOAT32 -> kernel.runSegmentF32(input, parameterF32, out, start, end);
                case BFLOAT16 -> kernel.runSegmentBF16(input, parameterF32, out, start, end);
                case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Scalar unary segment execution does not support dtype: " + dtype);
            }
        });
    }

    private static void fallbackUnary(
            UnaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            String reason
    ) {
        ElementwiseNativeSupport.requireFallbackAllowed(context, "unary elementwise", reason);
        ElementwiseNativeSupport.requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        ElementwiseLoops.runUnary(kernel, inputs.get(0), node, context);
    }

    private static void fallbackScalarUnary(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            String reason
    ) {
        ElementwiseNativeSupport.requireFallbackAllowed(context, "scalar unary elementwise", reason);
        ElementwiseNativeSupport.requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        ElementwiseLoops.runScalarUnary(kernel, parameterF64, parameterF32, inputs.get(0), node, context);
    }

    private static String nativeIneligibleReason(
            Operation.OpType opType,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        String label = opLabel(opType);
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:" + label + "-plan";
        }
        if (!supportsNativeElementwiseDType(node.getDataType())) {
            return "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase();
        }
        if (!isNativeUnaryOp(opType, node.getDataType())) {
            return "native-kernel-unsupported:" + label;
        }
        if (context.nodePlan().stridedPath()) {
            return "native-kernel-ineligible:" + label + "-strided";
        }
        if (inputs.get(0).getDataType() != node.getDataType()) {
            return "native-kernel-ineligible:" + label + "-dtype";
        }
        if (inputs.get(0).getFlatDataSize() != node.getFlatDataSize()) {
            return "native-kernel-ineligible:" + label + "-shape";
        }
        if (!ElementwiseNativeSupport.isDenseView(inputs.get(0)) || !ElementwiseNativeSupport.isDenseView(node)) {
            return "native-kernel-ineligible:" + label + "-layout";
        }
        return "";
    }

    private static String nativeScalarIneligibleReason(
            Operation.OpType opType,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (!isNativeScalarUnaryOp(opType)) {
            return "native-kernel-unsupported:" + opLabel(opType);
        }
        return nativeIneligibleReason(opType, inputs, node, context);
    }

    private static void attachNativeOutput(
            Tensor node,
            CpuKernelContext context,
            String label,
            NativeTensorStorage outputStorage
    ) {
        outputStorage.markModified();
        context.executionContext().attachNativeStorage(
                context.nodeId(),
                outputStorage,
                "unary executor " + label.toUpperCase() + " wrote " + node.getDataType() + " native output"
        );
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
    }

    private static boolean supportsNativeElementwiseDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean isNativeUnaryOp(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.FLOAT64) {
            return opType == Operation.OpType.MUL_SCALAR
                    || opType == Operation.OpType.NEG
                    || opType == Operation.OpType.RELU
                    || opType == Operation.OpType.CLAMP_MIN
                    || opType == Operation.OpType.CLAMP_MAX
                    || opType == Operation.OpType.LOG
                    || opType == Operation.OpType.EXP
                    || opType == Operation.OpType.FAST_EXP
                    || opType == Operation.OpType.SQRT
                    || opType == Operation.OpType.ABS
                    || opType == Operation.OpType.FLOOR
                    || opType == Operation.OpType.CEIL
                    || opType == Operation.OpType.SIGN
                    || opType == Operation.OpType.POW
                    || opType == Operation.OpType.TANH
                    || opType == Operation.OpType.FAST_TANH
                    || opType == Operation.OpType.SIGMOID
                    || opType == Operation.OpType.INV;
        }
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.MUL_SCALAR
                    || opType == Operation.OpType.NEG
                    || opType == Operation.OpType.RELU
                    || opType == Operation.OpType.CLAMP_MIN
                    || opType == Operation.OpType.CLAMP_MAX
                    || opType == Operation.OpType.ABS;
        }
        return dataType == DataType.FLOAT32
                && (opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.NEG
                || opType == Operation.OpType.RELU
                || opType == Operation.OpType.CLAMP_MIN
                || opType == Operation.OpType.CLAMP_MAX
                || opType == Operation.OpType.LOG
                || opType == Operation.OpType.EXP
                || opType == Operation.OpType.FAST_EXP
                || opType == Operation.OpType.SQRT
                || opType == Operation.OpType.ABS
                || opType == Operation.OpType.FLOOR
                || opType == Operation.OpType.CEIL
                || opType == Operation.OpType.SIGN
                || opType == Operation.OpType.POW
                || opType == Operation.OpType.TANH
                || opType == Operation.OpType.FAST_TANH
                || opType == Operation.OpType.SIGMOID);
    }

    private static boolean isNativeScalarUnaryOp(Operation.OpType opType) {
        return opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.CLAMP_MIN
                || opType == Operation.OpType.CLAMP_MAX
                || opType == Operation.OpType.POW;
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String opLabel(Operation.OpType opType) {
        return opType.name().toLowerCase();
    }
}
