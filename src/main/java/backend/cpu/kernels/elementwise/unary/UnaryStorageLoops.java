package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuNativeTraceSupport;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.ElementwiseNativeSupport;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class UnaryStorageLoops {
    private UnaryStorageLoops() {
    }

    static void execute(UnaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!ElementwiseNativeSupport.nativeRequested(context)) {
            ElementwiseLoops.runUnary(kernel, inputs.get(0), node, context);
            return;
        }
        Operation op = context.executionOperation();
        Operation.OpType opType = opType(op);
        String ineligibleReason = nativeIneligibleReason(opType, inputs, node, context);
        if (!ineligibleReason.isBlank()) {
            fallbackUnary(kernel, inputs, node, context, ineligibleReason);
            return;
        }
        try {
            String label = opLabel(opType);
            NativeTensorStorage inputStorage = ElementwiseNativeSupport.requireNativeInput(context, 0, node.getDataType(), label.toUpperCase());
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(node, context, "unary-storage-loop-" + label);
            CpuStorageBindings bindings = new CpuStorageBindings(
                    List.of(ElementwiseNativeSupport.segmentView(inputs.get(0), inputStorage)),
                    ElementwiseNativeSupport.segmentView(node, outputStorage)
            );
            runSegmentDense(kernel, bindings);
            attachNativeOutput(node, context, label, outputStorage);
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
        } catch (Throwable t) {
            fallbackUnary(kernel, inputs, node, context,
                    "native-kernel-failed:" + opLabel(opType) + ":" + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    static void execute(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (!ElementwiseNativeSupport.nativeRequested(context)) {
            ElementwiseLoops.runScalarUnary(kernel, parameterF64, parameterF32, inputs.get(0), node, context);
            return;
        }
        Operation op = context.executionOperation();
        Operation.OpType opType = opType(op);
        String ineligibleReason = nativeScalarIneligibleReason(opType, inputs, node, context);
        if (!ineligibleReason.isBlank()) {
            fallbackScalarUnary(kernel, parameterF64, parameterF32, inputs, node, context, ineligibleReason);
            return;
        }
        try {
            String label = opLabel(opType);
            NativeTensorStorage inputStorage = ElementwiseNativeSupport.requireNativeInput(context, 0, node.getDataType(), label.toUpperCase());
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(node, context, "scalar-unary-storage-loop-" + label);
            CpuStorageBindings bindings = new CpuStorageBindings(
                    List.of(ElementwiseNativeSupport.segmentView(inputs.get(0), inputStorage)),
                    ElementwiseNativeSupport.segmentView(node, outputStorage)
            );
            runSegmentDense(kernel, parameterF64, parameterF32, bindings);
            attachNativeOutput(node, context, label, outputStorage);
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
        } catch (Throwable t) {
            fallbackScalarUnary(kernel, parameterF64, parameterF32, inputs, node, context,
                    "native-kernel-failed:" + opLabel(opType) + ":" + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    static void runSegmentDense(UnaryElementwiseKernel kernel, CpuStorageBindings bindings) {
        validateDenseUnary(bindings);
        DataType dtype = bindings.output().dtype();
        switch (dtype) {
            case FLOAT64 -> runSegmentDenseF64(
                    kernel,
                    bindings.input(0).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case FLOAT32 -> runSegmentDenseF32(
                    kernel,
                    bindings.input(0).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case BFLOAT16 -> runSegmentDenseBF16(
                    kernel,
                    bindings.input(0).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Unary storage loop does not support dtype: " + dtype);
        }
    }

    static void runSegmentDense(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            CpuStorageBindings bindings
    ) {
        validateDenseUnary(bindings);
        DataType dtype = bindings.output().dtype();
        switch (dtype) {
            case FLOAT64 -> runSegmentDenseF64(
                    kernel,
                    parameterF64,
                    bindings.input(0).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case FLOAT32 -> runSegmentDenseF32(
                    kernel,
                    parameterF32,
                    bindings.input(0).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case BFLOAT16 -> runSegmentDenseBF16(
                    kernel,
                    parameterF32,
                    bindings.input(0).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Scalar unary storage loop does not support dtype: " + dtype);
        }
    }

    private static void runSegmentDenseF32(UnaryElementwiseKernel kernel, MemorySegment input, MemorySegment output, int size) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            output.set(JAVA_FLOAT, offset, kernel.applyF32(input.get(JAVA_FLOAT, offset)));
        }
    }

    private static void runSegmentDenseF64(UnaryElementwiseKernel kernel, MemorySegment input, MemorySegment output, int size) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            output.set(JAVA_DOUBLE, offset, kernel.applyF64(input.get(JAVA_DOUBLE, offset)));
        }
    }

    private static void runSegmentDenseBF16(UnaryElementwiseKernel kernel, MemorySegment input, MemorySegment output, int size) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Short.BYTES;
            float value = CpuDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, offset));
            output.set(JAVA_SHORT, offset, CpuDTypeOps.toBFloat16Bits(kernel.applyBF16(value)));
        }
    }

    private static void runSegmentDenseF32(
            ScalarUnaryElementwiseKernel kernel,
            float parameter,
            MemorySegment input,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            output.set(JAVA_FLOAT, offset, kernel.applyF32(input.get(JAVA_FLOAT, offset), parameter));
        }
    }

    private static void runSegmentDenseF64(
            ScalarUnaryElementwiseKernel kernel,
            double parameter,
            MemorySegment input,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            output.set(JAVA_DOUBLE, offset, kernel.applyF64(input.get(JAVA_DOUBLE, offset), parameter));
        }
    }

    private static void runSegmentDenseBF16(
            ScalarUnaryElementwiseKernel kernel,
            float parameter,
            MemorySegment input,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Short.BYTES;
            float value = CpuDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, offset));
            output.set(JAVA_SHORT, offset, CpuDTypeOps.toBFloat16Bits(kernel.applyBF16(value, parameter)));
        }
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
        if (inputs == null || inputs.size() != 1 || inputs.get(0).getDataType() != node.getDataType()) {
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

    private static void validateDenseUnary(CpuStorageBindings bindings) {
        if (bindings.inputs().size() != 1) {
            throw new IllegalArgumentException("Unary storage loop requires exactly 1 input.");
        }
        CpuStorageView input = bindings.input(0);
        CpuStorageView output = bindings.output();
        if (input.dtype() != output.dtype()) {
            throw new IllegalArgumentException("Unary storage loop dtype mismatch.");
        }
        if (input.kind() != output.kind()) {
            throw new IllegalArgumentException("Unary storage loop requires matching storage kinds.");
        }
        if (input.logicalSize() != output.logicalSize()) {
            throw new IllegalArgumentException("Unary storage loop requires same-shape dense input.");
        }
        if (!ElementwiseNativeSupport.isDenseView(input) || !ElementwiseNativeSupport.isDenseView(output)) {
            throw new IllegalArgumentException("Unary storage loop requires dense zero-offset views.");
        }
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
                "unary storage loop " + label.toUpperCase() + " wrote " + node.getDataType() + " native output"
        );
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
