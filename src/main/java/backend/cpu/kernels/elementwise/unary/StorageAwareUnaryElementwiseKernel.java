package backend.cpu.kernels.elementwise.unary;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.kernels.elementwise.ElementwiseLayoutPlan;
import backend.cpu.kernels.elementwise.ElementwiseNativeSupport;
import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.nativecpu.CpuNativeTraceSupport;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.storage.CpuStorageBindings;
import backend.cpu.storage.CpuStorageResolver;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

abstract class StorageAwareUnaryElementwiseKernel implements CpuStorageAwareKernel {
    @Override
    public final CpuKernelResult execute(CpuKernelCall call) {
        requireUnaryCall(call);
        StorageAwareUnaryElementwiseKernel executionKernel = selectExecutionKernel(call.context());
        if (ElementwiseNativeSupport.nativeRequested(call.context())) {
            return executeNative(call, executionKernel);
        }
        executionKernel.executeStorage(call.inputs().getFirst(), call.output(), call.context());
        return CpuKernelResult.completed();
    }

    protected StorageAwareUnaryElementwiseKernel selectExecutionKernel(CpuKernelContext context) {
        return this;
    }

    protected abstract Operation.OpType opType();

    protected abstract String opLabel();

    protected boolean supportsNativeOpDType(DataType dtype) {
        return supportsNativeElementwiseDType(dtype);
    }

    protected abstract void runDirectF64(double[] in, double[] out, ResolvedDispatchHints hints);

    protected abstract void runDirectF32(float[] in, float[] out, ResolvedDispatchHints hints);

    protected abstract void runDirectBF16(
            short[] in,
            float[] continuation,
            short[] out,
            ResolvedDispatchHints hints
    );

    protected abstract void runDirectBF16ToFloat(
            short[] in,
            float[] continuation,
            float[] out,
            ResolvedDispatchHints hints
    );

    protected abstract void runSegmentF64(MemorySegment in, MemorySegment out, int start, int end);

    protected abstract void runSegmentF32(MemorySegment in, MemorySegment out, int start, int end);

    protected abstract void runSegmentBF16(MemorySegment in, MemorySegment out, int start, int end);

    protected abstract void runIndexedArrayF64(
            double[] in,
            double[] out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedArrayF32(
            float[] in,
            float[] out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedArrayBF16(
            short[] in,
            float[] continuation,
            short[] out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedArrayBF16ToFloat(
            short[] in,
            float[] continuation,
            float[] out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentF64(
            MemorySegment in,
            MemorySegment out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentF32(
            MemorySegment in,
            MemorySegment out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentBF16(
            MemorySegment in,
            MemorySegment out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedF64(
            CpuStorageView in,
            CpuStorageView out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedF32(
            CpuStorageView in,
            CpuStorageView out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedBF16(
            CpuStorageView in,
            CpuStorageView out,
            UnaryStorageLayout layout,
            int start,
            int end
    );

    private CpuKernelResult executeNative(CpuKernelCall call, StorageAwareUnaryElementwiseKernel executionKernel) {
        String ineligibleReason = nativeIneligibleReason(call);
        if (!ineligibleReason.isBlank()) {
            return fallbackToArray(call, ineligibleReason, executionKernel);
        }

        CpuKernelContext context = call.context();
        try {
            String op = opLabel().toUpperCase();
            NativeTensorStorage inputStorage = ElementwiseNativeSupport.requireNativeInput(
                    context,
                    0,
                    call.output().dtype(),
                    op
            );
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(
                    call.outputTensor(),
                    context,
                    "unary-segment-" + opLabel()
            );
            CpuStorageView input = ElementwiseNativeSupport.segmentView(call.inputTensors().getFirst(), inputStorage);
            CpuStorageView out = ElementwiseNativeSupport.segmentView(call.outputTensor(), outputStorage);

            executionKernel.executeStorage(input, out, context);
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    op + " wrote " + call.output().dtype() + " native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
            return CpuKernelResult.route(CpuNativeTraceSupport.CPU_NATIVE);
        } catch (Throwable t) {
            return fallbackToArray(
                    call,
                    "native-kernel-failed:" + opLabel() + ":" + ElementwiseNativeSupport.safeMessage(t),
                    executionKernel
            );
        }
    }

    private CpuKernelResult fallbackToArray(
            CpuKernelCall call,
            String reason,
            StorageAwareUnaryElementwiseKernel executionKernel
    ) {
        CpuKernelContext context = call.context();
        ElementwiseNativeSupport.requireFallbackAllowed(context, opLabel().toUpperCase(), reason);
        ElementwiseNativeSupport.requireCpuReadableInputs(context);
        CpuStorageBindings storage = new CpuStorageResolver().bindArrayOnly(call.inputTensors(), call.outputTensor());
        executionKernel.executeStorage(storage.input(0), storage.output(), context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        return CpuKernelResult.fallback(CpuNativeTraceSupport.CPU_ARRAY, reason);
    }

    private void executeStorage(CpuStorageView input, CpuStorageView out, CpuKernelContext context) {
        if (input.dtype() != out.dtype()) {
            throw new IllegalStateException(opLabel().toUpperCase() + " storage dtype mismatch. input="
                    + input.dtype() + ", out=" + out.dtype());
        }
        switch (out.dtype()) {
            case FLOAT64 -> executeF64(input, out, context.dispatchHints());
            case FLOAT32 -> executeF32(input, out, context.dispatchHints());
            case BFLOAT16 -> executeBF16(input, out, context);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    opLabel().toUpperCase() + " does not support dtype: " + out.dtype()
            );
        }
    }

    private void executeF64(CpuStorageView input, CpuStorageView out, ResolvedDispatchHints hints) {
        int length = out.logicalSize();
        if (canUseDenseDirectArray(input, out)) {
            runDirectF64(input.requireF64Array(), out.requireF64Array(), hints);
            return;
        }
        if (canUseDenseDirectSegment(input, out)) {
            MemorySegment inputSegment = input.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentF64(inputSegment, outSegment, start, end));
            return;
        }

        UnaryStorageLayout layout = UnaryStorageLayout.from(input, out);
        if (allArrays(input, out)) {
            double[] inputArray = input.requireF64Array();
            double[] outArray = out.requireF64Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayF64(inputArray, outArray, layout, start, end));
            return;
        }
        if (allSegments(input, out)) {
            MemorySegment inputSegment = input.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentF64(inputSegment, outSegment, layout, start, end));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedF64(input, out, layout, start, end));
    }

    private void executeF32(CpuStorageView input, CpuStorageView out, ResolvedDispatchHints hints) {
        int length = out.logicalSize();
        if (canUseDenseDirectArray(input, out)) {
            runDirectF32(input.requireF32Array(), out.requireF32Array(), hints);
            return;
        }
        if (canUseDenseDirectSegment(input, out)) {
            MemorySegment inputSegment = input.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentF32(inputSegment, outSegment, start, end));
            return;
        }

        UnaryStorageLayout layout = UnaryStorageLayout.from(input, out);
        if (allArrays(input, out)) {
            float[] inputArray = input.requireF32Array();
            float[] outArray = out.requireF32Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayF32(inputArray, outArray, layout, start, end));
            return;
        }
        if (allSegments(input, out)) {
            MemorySegment inputSegment = input.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentF32(inputSegment, outSegment, layout, start, end));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedF32(input, out, layout, start, end));
    }

    private void executeBF16(CpuStorageView input, CpuStorageView out, CpuKernelContext context) {
        ResolvedDispatchHints hints = context.dispatchHints();
        int length = out.logicalSize();
        if (canUseDenseDirectArray(input, out)) {
            short[] inputArray = input.requireBF16Array();
            float[] continuation = context.inputFloatContinuation(0, input.logicalSize());
            if (canPublishFloatContinuation(context)) {
                float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
                runDirectBF16ToFloat(inputArray, continuation, outFloat, hints);
                context.cpuWorkspace().publishFloatContinuation(length);
                return;
            }
            runDirectBF16(inputArray, continuation, out.requireBF16Array(), hints);
            return;
        }
        if (canUseDenseDirectSegment(input, out)) {
            MemorySegment inputSegment = input.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentBF16(inputSegment, outSegment, start, end));
            return;
        }

        UnaryStorageLayout layout = UnaryStorageLayout.from(input, out);
        if (allArrays(input, out)) {
            short[] inputArray = input.requireBF16Array();
            float[] continuation = context.inputFloatContinuation(0, input.logicalSize());
            if (canPublishFloatContinuation(context)) {
                float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
                ElementwiseRangeLoop.runScalar(length, hints,
                        (start, end) -> runIndexedArrayBF16ToFloat(
                                inputArray,
                                continuation,
                                outFloat,
                                layout,
                                start,
                                end
                        ));
                context.cpuWorkspace().publishFloatContinuation(length);
                return;
            }
            short[] outArray = out.requireBF16Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayBF16(inputArray, continuation, outArray, layout, start, end));
            return;
        }
        if (allSegments(input, out)) {
            MemorySegment inputSegment = input.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentBF16(inputSegment, outSegment, layout, start, end));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedBF16(input, out, layout, start, end));
    }

    protected static double readF64(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireF64Array()[offset]
                : view.requireSegment().get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    protected static void writeF64(CpuStorageView view, int offset, double value) {
        if (view.isArray()) {
            view.requireF64Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    protected static float readF32(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireF32Array()[offset]
                : view.requireSegment().get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    protected static void writeF32(CpuStorageView view, int offset, float value) {
        if (view.isArray()) {
            view.requireF32Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    protected static float readBF16(CpuStorageView view, int offset) {
        short bits = view.isArray()
                ? view.requireBF16Array()[offset]
                : view.requireSegment().get(JAVA_SHORT, (long) offset * Short.BYTES);
        return TensorDTypeOps.fromBFloat16Bits(bits);
    }

    protected static void writeBF16(CpuStorageView view, int offset, float value) {
        short bits = TensorDTypeOps.toBFloat16Bits(value);
        if (view.isArray()) {
            view.requireBF16Array()[offset] = bits;
        } else {
            view.requireSegment().set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
        }
    }

    protected static float loadBF16(float[] continuation, short[] storage, int index) {
        return continuation != null ? continuation[index] : TensorDTypeOps.fromBFloat16Bits(storage[index]);
    }

    private static boolean allArrays(CpuStorageView input, CpuStorageView out) {
        return input.isArray() && out.isArray();
    }

    private static boolean allSegments(CpuStorageView input, CpuStorageView out) {
        return input.isMemorySegment() && out.isMemorySegment();
    }

    private static boolean canUseDenseDirectArray(CpuStorageView input, CpuStorageView out) {
        return isDenseZeroArray(input)
                && isDenseZeroArray(out)
                && input.logicalSize() == out.logicalSize();
    }

    private static boolean canUseDenseDirectSegment(CpuStorageView input, CpuStorageView out) {
        return isDenseZeroSegment(input)
                && isDenseZeroSegment(out)
                && input.logicalSize() == out.logicalSize();
    }

    private static boolean isDenseZeroArray(CpuStorageView view) {
        return view.isArray() && isDenseZero(view);
    }

    private static boolean isDenseZeroSegment(CpuStorageView view) {
        return view.isMemorySegment() && isDenseZero(view);
    }

    private static boolean isDenseZero(CpuStorageView view) {
        if (view.storageOffset() != 0) {
            return false;
        }
        int[] shape = view.shape();
        int[] strides = view.strides();
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape[i]);
        }
        return true;
    }

    private static boolean canPublishFloatContinuation(CpuKernelContext context) {
        return context != null
                && context.publishFloatContinuation()
                && context.cpuWorkspace() != null;
    }

    private void requireUnaryCall(CpuKernelCall call) {
        if (call.inputs().size() != 1 || call.inputTensors().size() != 1) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires exactly 1 input.");
        }
    }

    private String nativeIneligibleReason(CpuKernelCall call) {
        if (call.context() == null || call.context().nodePlan() == null) {
            return "native-kernel-ineligible:" + opLabel() + "-plan";
        }
        if (call.operation() == null || call.operation().opType() != opType()) {
            return "native-kernel-unsupported:" + opLabel();
        }
        DataType dtype = call.output().dtype();
        if (!supportsNativeElementwiseDType(dtype)) {
            return "native-storage-dtype-unsupported:" + dtype.name().toLowerCase();
        }
        if (!supportsNativeOpDType(dtype)) {
            return "native-kernel-unsupported:" + opLabel();
        }
        if (call.inputTensors().getFirst().getDataType() != dtype) {
            return "native-kernel-ineligible:" + opLabel() + "-dtype";
        }
        if (!ElementwiseLayoutPlan.canBroadcastTo(call.inputs().getFirst().shape(), call.output().shape())) {
            return "native-kernel-ineligible:" + opLabel() + "-shape";
        }
        return "";
    }

    private static boolean supportsNativeElementwiseDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.FLOAT64 || dtype == DataType.BFLOAT16;
    }

    protected record UnaryStorageLayout(int[] shape, int[][] cursorStrides, int[] cursorBaseOffsets) {
        static UnaryStorageLayout from(CpuStorageView input, CpuStorageView out) {
            int[] outputShape = out.shape();
            return new UnaryStorageLayout(
                    outputShape,
                    new int[][]{
                            out.strides(),
                            ElementwiseLayoutPlan.broadcastStrides(input.shape(), input.strides(), outputShape)
                    },
                    new int[]{out.storageOffset(), input.storageOffset()}
            );
        }

        protected UnaryStorageLayout {
            shape = shape.clone();
            cursorStrides = cursorStrides.clone();
            cursorBaseOffsets = cursorBaseOffsets.clone();
        }

        ElementwiseOffsetCursor cursor(int start) {
            return new ElementwiseOffsetCursor(shape, cursorStrides, cursorBaseOffsets, start);
        }
    }
}
