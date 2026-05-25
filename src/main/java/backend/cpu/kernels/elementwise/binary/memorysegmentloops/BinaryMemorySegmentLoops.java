package backend.cpu.kernels.elementwise.binary.memorysegmentloops;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNativeTraceSupport;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.ElementwiseNativeSupport;
import backend.cpu.kernels.elementwise.binary.BinaryElementwiseKernel;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class BinaryMemorySegmentLoops {
    private BinaryMemorySegmentLoops() {
    }

    public static void execute(BinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!ElementwiseNativeSupport.nativeRequested(context)) {
            ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
            return;
        }
        Operation op = context.executionOperation();
        Operation.OpType opType = opType(op);
        BiasBroadcastSpec biasSpec = biasBroadcastSpec(opType, context, inputs, node);
        String ineligibleReason = nativeIneligibleReason(opType, inputs, node, context, biasSpec);
        if (!ineligibleReason.isBlank()) {
            fallbackToArray(kernel, inputs, node, context, ineligibleReason);
            return;
        }
        try {
            String label = opLabel(opType);
            NativeTensorStorage leftStorage = ElementwiseNativeSupport.requireNativeInput(context, 0, node.getDataType(), label.toUpperCase());
            NativeTensorStorage rightStorage = ElementwiseNativeSupport.requireNativeInput(context, 1, node.getDataType(), label.toUpperCase());
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(node, context, "binary-memory-segment-loop-" + label);
            CpuStorageBindings bindings = new CpuStorageBindings(
                    List.of(
                            ElementwiseNativeSupport.segmentView(inputs.get(0), leftStorage),
                            ElementwiseNativeSupport.segmentView(inputs.get(1), rightStorage)
                    ),
                    ElementwiseNativeSupport.segmentView(node, outputStorage)
            );
            if (biasSpec != null) {
                runLastDimBiasAddF32(leftStorage.segment(), rightStorage.segment(), outputStorage.segment(), node.getFlatDataSize(), biasSpec);
            } else if (opType == Operation.OpType.ADD) {
                runAddSegmentDense(bindings);
            } else {
                runSegmentDense(kernel, bindings);
            }
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "binary MemorySegment loop " + label.toUpperCase() + " wrote " + node.getDataType() + " native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
        } catch (Throwable t) {
            fallbackToArray(kernel, inputs, node, context,
                    "native-kernel-failed:" + opLabel(opType) + ":" + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    static void runAddSegmentDense(CpuStorageBindings bindings) {
        validateDenseBinary(bindings);
        DataType dtype = bindings.output().dtype();
        switch (dtype) {
            case FLOAT64 -> runAddSegmentDenseF64(
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case FLOAT32 -> runAddSegmentDenseF32(
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case BFLOAT16 -> runAddSegmentDenseBF16(
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Binary MemorySegment loop does not support ADD dtype: " + dtype);
        }
    }

    static void runSegmentDense(BinaryElementwiseKernel kernel, CpuStorageBindings bindings) {
        validateDenseBinary(bindings);
        DataType dtype = bindings.output().dtype();
        switch (dtype) {
            case FLOAT64 -> runSegmentDenseF64(
                    kernel,
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case FLOAT32 -> runSegmentDenseF32(
                    kernel,
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case BFLOAT16 -> runSegmentDenseBF16(
                    kernel,
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Binary MemorySegment loop does not support dtype: " + dtype);
        }
    }

    private static void runAddSegmentDenseF32(
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            output.set(JAVA_FLOAT, offset, left.get(JAVA_FLOAT, offset) + right.get(JAVA_FLOAT, offset));
        }
    }

    private static void runAddSegmentDenseF64(
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            output.set(JAVA_DOUBLE, offset, left.get(JAVA_DOUBLE, offset) + right.get(JAVA_DOUBLE, offset));
        }
    }

    private static void runAddSegmentDenseBF16(
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Short.BYTES;
            float leftValue = CpuDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, offset));
            float rightValue = CpuDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, offset));
            output.set(JAVA_SHORT, offset, CpuDTypeOps.toBFloat16Bits(leftValue + rightValue));
        }
    }

    private static void runLastDimBiasAddF32(
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size,
            BiasBroadcastSpec spec
    ) {
        for (int i = 0; i < size; i++) {
            long outputOffset = (long) i * Float.BYTES;
            long biasOffset = (long) (i % spec.lastDim()) * Float.BYTES;
            float leftValue = left.get(JAVA_FLOAT, spec.leftBias() ? biasOffset : outputOffset);
            float rightValue = right.get(JAVA_FLOAT, spec.leftBias() ? outputOffset : biasOffset);
            output.set(JAVA_FLOAT, outputOffset, leftValue + rightValue);
        }
    }

    private static void runSegmentDenseF32(
            BinaryElementwiseKernel kernel,
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            output.set(JAVA_FLOAT, offset, kernel.applyF32(left.get(JAVA_FLOAT, offset), right.get(JAVA_FLOAT, offset)));
        }
    }

    private static void runSegmentDenseF64(
            BinaryElementwiseKernel kernel,
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            output.set(JAVA_DOUBLE, offset, kernel.applyF64(left.get(JAVA_DOUBLE, offset), right.get(JAVA_DOUBLE, offset)));
        }
    }

    private static void runSegmentDenseBF16(
            BinaryElementwiseKernel kernel,
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Short.BYTES;
            float leftValue = CpuDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, offset));
            float rightValue = CpuDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, offset));
            output.set(JAVA_SHORT, offset, CpuDTypeOps.toBFloat16Bits(kernel.applyBF16(leftValue, rightValue)));
        }
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
            CpuKernelContext context,
            BiasBroadcastSpec biasSpec
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
        if (inputs == null || inputs.size() != 2
                || inputs.get(0).getDataType() != node.getDataType()
                || inputs.get(1).getDataType() != node.getDataType()) {
            return "native-kernel-ineligible:" + label + "-dtype";
        }
        if (!ElementwiseNativeSupport.isDenseView(inputs.get(0))
                || !ElementwiseNativeSupport.isDenseView(inputs.get(1))
                || !ElementwiseNativeSupport.isDenseView(node)) {
            return "native-kernel-ineligible:" + label + "-layout";
        }
        if (context.broadcastPlan() != null && !context.broadcastPlan().isNoBroadcast()) {
            if (biasSpec != null) {
                return "";
            }
            return "native-kernel-ineligible:" + label + "-broadcast";
        }
        int size = node.getFlatDataSize();
        if (inputs.get(0).getFlatDataSize() != size || inputs.get(1).getFlatDataSize() != size) {
            return "native-kernel-ineligible:" + label + "-shape";
        }
        return "";
    }

    private static BiasBroadcastSpec biasBroadcastSpec(
            Operation.OpType opType,
            CpuKernelContext context,
            List<Tensor> inputs,
            Tensor node
    ) {
        if (opType != Operation.OpType.ADD || context == null || inputs == null || inputs.size() != 2) {
            return null;
        }
        var plan = context.broadcastPlan();
        if (node.getDataType() != DataType.FLOAT32
                || plan == null
                || plan.isNoBroadcast()
                || product(plan.outShape()) != node.getFlatDataSize()) {
            return null;
        }
        int[] shape = plan.outShape();
        int lastDim = shape[shape.length - 1];
        boolean leftFull = Arrays.equals(plan.aEffStrides(), plan.outStrides());
        boolean rightFull = Arrays.equals(plan.bEffStrides(), plan.outStrides());
        boolean leftBias = isLastDimBiasSide(plan.aEffStrides(), shape)
                && inputs.get(0).getFlatDataSize() == lastDim;
        boolean rightBias = isLastDimBiasSide(plan.bEffStrides(), shape)
                && inputs.get(1).getFlatDataSize() == lastDim;
        if (leftFull && inputs.get(0).getFlatDataSize() == node.getFlatDataSize() && rightBias) {
            return new BiasBroadcastSpec(false, lastDim);
        }
        if (leftBias && rightFull && inputs.get(1).getFlatDataSize() == node.getFlatDataSize()) {
            return new BiasBroadcastSpec(true, lastDim);
        }
        return null;
    }

    private static boolean isLastDimBiasSide(int[] effectiveStrides, int[] outputShape) {
        if (effectiveStrides == null || outputShape == null || effectiveStrides.length != outputShape.length || outputShape.length < 2) {
            return false;
        }
        int last = effectiveStrides.length - 1;
        if (outputShape[last] <= 0 || effectiveStrides[last] != 1) {
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

    private static void validateDenseBinary(CpuStorageBindings bindings) {
        if (bindings.inputs().size() != 2) {
            throw new IllegalArgumentException("Binary MemorySegment loop requires exactly 2 inputs.");
        }
        CpuStorageView left = bindings.input(0);
        CpuStorageView right = bindings.input(1);
        CpuStorageView output = bindings.output();
        if (left.dtype() != output.dtype() || right.dtype() != output.dtype()) {
            throw new IllegalArgumentException("Binary MemorySegment loop dtype mismatch.");
        }
        if (left.kind() != output.kind() || right.kind() != output.kind()) {
            throw new IllegalArgumentException("Binary MemorySegment loop requires matching storage kinds.");
        }
        if (left.logicalSize() != output.logicalSize() || right.logicalSize() != output.logicalSize()) {
            throw new IllegalArgumentException("Binary MemorySegment loop requires same-shape dense inputs.");
        }
        if (!ElementwiseNativeSupport.isDenseView(left)
                || !ElementwiseNativeSupport.isDenseView(right)
                || !ElementwiseNativeSupport.isDenseView(output)) {
            throw new IllegalArgumentException("Binary MemorySegment loop requires dense zero-offset views.");
        }
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

    private record BiasBroadcastSpec(boolean leftBias, int lastDim) {
    }
}
