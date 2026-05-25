package backend.cpu.kernels.elementwise.binary;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.nativecpu.CpuNativeTraceSupport;
import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.elementwise.ElementwiseLayoutPlan;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.ElementwiseNativeSupport;
import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class ElementwiseBinaryExecutor {
    private ElementwiseBinaryExecutor() {}

    public static void execute(BinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Binary elementwise executor requires exactly 2 inputs.");
        }
        if (!ElementwiseNativeSupport.nativeRequested(context)) {
            ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
            return;
        }

        Operation.OpType opType = opType(context.executionOperation());
        String ineligibleReason = nativeIneligibleReason(opType, inputs, node, context);
        if (!ineligibleReason.isBlank()) {
            fallbackToArray(kernel, inputs, node, context, ineligibleReason);
            return;
        }

        try {
            String label = opLabel(opType);
            NativeTensorStorage leftStorage = ElementwiseNativeSupport.requireNativeInput(context, 0, node.getDataType(), label.toUpperCase());
            NativeTensorStorage rightStorage = ElementwiseNativeSupport.requireNativeInput(context, 1, node.getDataType(), label.toUpperCase());
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(node, context, "binary-segment-" + label);
            if (context.broadcastPlan() == null || context.broadcastPlan().isNoBroadcast()) {
                runDenseSegment(kernel, node.getDataType(), leftStorage.segment(), rightStorage.segment(), outputStorage.segment(), node.getFlatDataSize(), context.dispatchHints());
            } else {
                runIndexedSegment(
                        kernel,
                        node.getDataType(),
                        leftStorage.segment(),
                        rightStorage.segment(),
                        outputStorage.segment(),
                        ElementwiseLayoutPlan.binary(inputs.get(0), inputs.get(1), node),
                        context.dispatchHints()
                );
            }
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "binary executor " + label.toUpperCase() + " wrote " + node.getDataType() + " native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
        } catch (Throwable t) {
            fallbackToArray(kernel, inputs, node, context,
                    "native-kernel-failed:" + opLabel(opType) + ":" + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    static void runDenseSegment(
            BinaryElementwiseKernel kernel,
            DataType dtype,
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            int length,
            ResolvedDispatchHints hints
    ) {
        ElementwiseRangeLoop.runScalar(length, hints, (start, end) -> {
            switch (dtype) {
                case FLOAT64 -> kernel.runSegmentF64(left, right, out, start, end);
                case FLOAT32 -> kernel.runSegmentF32(left, right, out, start, end);
                case BFLOAT16 -> kernel.runSegmentBF16(left, right, out, start, end);
                case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Binary segment execution does not support dtype: " + dtype);
            }
        });
    }

    static void runIndexedSegment(
            BinaryElementwiseKernel kernel,
            DataType dtype,
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            ElementwiseLayoutPlan plan,
            ResolvedDispatchHints hints
    ) {
        ElementwiseRangeLoop.runScalar(plan.length(), hints, (start, end) -> {
            ElementwiseOffsetCursor cursor = new ElementwiseOffsetCursor(
                    plan.shape(),
                    plan.cursorStrides(),
                    plan.cursorBaseOffsets(),
                    start
            );
            for (int outIndex = start; outIndex < end; outIndex++) {
                switch (dtype) {
                    case FLOAT64 -> {
                        double leftValue = left.get(JAVA_DOUBLE, (long) cursor.offset(1) * Double.BYTES);
                        double rightValue = right.get(JAVA_DOUBLE, (long) cursor.offset(2) * Double.BYTES);
                        out.set(JAVA_DOUBLE, (long) cursor.offset(0) * Double.BYTES, kernel.applyF64(leftValue, rightValue));
                    }
                    case FLOAT32 -> {
                        float leftValue = left.get(JAVA_FLOAT, (long) cursor.offset(1) * Float.BYTES);
                        float rightValue = right.get(JAVA_FLOAT, (long) cursor.offset(2) * Float.BYTES);
                        out.set(JAVA_FLOAT, (long) cursor.offset(0) * Float.BYTES, kernel.applyF32(leftValue, rightValue));
                    }
                    case BFLOAT16 -> {
                        float leftValue = TensorDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, (long) cursor.offset(1) * Short.BYTES));
                        float rightValue = TensorDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, (long) cursor.offset(2) * Short.BYTES));
                        out.set(JAVA_SHORT, (long) cursor.offset(0) * Short.BYTES, TensorDTypeOps.toBFloat16Bits(kernel.applyBF16(leftValue, rightValue)));
                    }
                    case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Binary indexed segment execution does not support dtype: " + dtype);
                }
                if (outIndex + 1 < end) {
                    cursor.step();
                }
            }
        });
    }

    private static void fallbackToArray(
            BinaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            String reason
    ) {
        ElementwiseNativeSupport.requireFallbackAllowed(context, "binary elementwise", reason);
        ElementwiseNativeSupport.requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
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
        if (!isNativeBinaryOp(opType, node.getDataType())) {
            return "native-kernel-unsupported:" + label;
        }
        if (context.nodePlan().stridedPath()) {
            return "native-kernel-ineligible:" + label + "-strided";
        }
        if (inputs.get(0).getDataType() != node.getDataType()
                || inputs.get(1).getDataType() != node.getDataType()) {
            return "native-kernel-ineligible:" + label + "-dtype";
        }
        if (!ElementwiseNativeSupport.isDenseView(inputs.get(0))
                || !ElementwiseNativeSupport.isDenseView(inputs.get(1))
                || !ElementwiseNativeSupport.isDenseView(node)) {
            return "native-kernel-ineligible:" + label + "-layout";
        }
        int size = node.getFlatDataSize();
        if ((context.broadcastPlan() == null || context.broadcastPlan().isNoBroadcast())
                && (inputs.get(0).getFlatDataSize() != size || inputs.get(1).getFlatDataSize() != size)) {
            return "native-kernel-ineligible:" + label + "-shape";
        }
        return "";
    }

    private static boolean supportsNativeElementwiseDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean isNativeBinaryOp(Operation.OpType opType, DataType dataType) {
        if (opType == Operation.OpType.POW_TENSOR) {
            return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
        }
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16)
                && (opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.MIN
                || opType == Operation.OpType.MAX);
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String opLabel(Operation.OpType opType) {
        return opType.name().toLowerCase();
    }
}
