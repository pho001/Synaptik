package backend.cpu.kernels.elementwise.binary;

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

abstract class StorageAwareBinaryElementwiseKernel implements CpuStorageAwareKernel {
    @Override
    public final CpuKernelResult execute(CpuKernelCall call) {
        requireBinaryCall(call);
        if (ElementwiseNativeSupport.nativeRequested(call.context())) {
            return executeNative(call);
        }
        executeStorage(call.inputs().get(0), call.inputs().get(1), call.output(), call.context());
        return CpuKernelResult.completed();
    }

    protected abstract Operation.OpType opType();

    protected abstract String opLabel();

    protected boolean supportsNativeOpDType(DataType dtype) {
        return supportsNativeElementwiseDType(dtype);
    }

    protected abstract void runDirectF64(double[] left, double[] right, double[] out, ResolvedDispatchHints hints);

    protected abstract void runDirectF32(float[] left, float[] right, float[] out, ResolvedDispatchHints hints);

    protected abstract void runDirectBF16(
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedDispatchHints hints
    );

    protected abstract void runDirectBF16ToFloat(
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            float[] out,
            ResolvedDispatchHints hints
    );

    protected abstract void runSegmentF64(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end);

    protected abstract void runSegmentF32(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end);

    protected abstract void runSegmentBF16(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end);

    protected abstract void runIndexedArrayF64(
            double[] left,
            double[] right,
            double[] out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedArrayF32(
            float[] left,
            float[] right,
            float[] out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedArrayBF16(
            short[] left,
            short[] right,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedArrayBF16ToFloat(
            short[] left,
            short[] right,
            float[] leftContinuation,
            float[] rightContinuation,
            float[] out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentF64(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentF32(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentBF16(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedF64(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedF32(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedBF16(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            BinaryStorageLayout layout,
            int start,
            int end
    );

    private CpuKernelResult executeNative(CpuKernelCall call) {
        String ineligibleReason = nativeIneligibleReason(call);
        if (!ineligibleReason.isBlank()) {
            return fallbackToArray(call, ineligibleReason);
        }

        CpuKernelContext context = call.context();
        try {
            String op = opLabel().toUpperCase();
            NativeTensorStorage leftStorage = ElementwiseNativeSupport.requireNativeInput(
                    context,
                    0,
                    call.output().dtype(),
                    op
            );
            NativeTensorStorage rightStorage = ElementwiseNativeSupport.requireNativeInput(
                    context,
                    1,
                    call.output().dtype(),
                    op
            );
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(
                    call.outputTensor(),
                    context,
                    "binary-segment-" + opLabel()
            );
            CpuStorageView left = ElementwiseNativeSupport.segmentView(call.inputTensors().get(0), leftStorage);
            CpuStorageView right = ElementwiseNativeSupport.segmentView(call.inputTensors().get(1), rightStorage);
            CpuStorageView out = ElementwiseNativeSupport.segmentView(call.outputTensor(), outputStorage);

            executeStorage(left, right, out, context);
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    op + " wrote " + call.output().dtype() + " native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
            return CpuKernelResult.route(CpuNativeTraceSupport.CPU_NATIVE);
        } catch (Throwable t) {
            return fallbackToArray(call, "native-kernel-failed:" + opLabel() + ":"
                    + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    private CpuKernelResult fallbackToArray(CpuKernelCall call, String reason) {
        CpuKernelContext context = call.context();
        ElementwiseNativeSupport.requireFallbackAllowed(context, opLabel().toUpperCase(), reason);
        ElementwiseNativeSupport.requireCpuReadableInputs(context);
        CpuStorageBindings storage = new CpuStorageResolver().bindArrayOnly(call.inputTensors(), call.outputTensor());
        executeStorage(storage.input(0), storage.input(1), storage.output(), context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        return CpuKernelResult.fallback(CpuNativeTraceSupport.CPU_ARRAY, reason);
    }

    private void executeStorage(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            CpuKernelContext context
    ) {
        if (left.dtype() != out.dtype() || right.dtype() != out.dtype()) {
            throw new IllegalStateException(opLabel().toUpperCase() + " storage dtype mismatch. left=" + left.dtype()
                    + ", right=" + right.dtype() + ", out=" + out.dtype());
        }
        switch (out.dtype()) {
            case FLOAT64 -> executeF64(left, right, out, context.dispatchHints());
            case FLOAT32 -> executeF32(left, right, out, context.dispatchHints());
            case BFLOAT16 -> executeBF16(left, right, out, context);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    opLabel().toUpperCase() + " does not support dtype: " + out.dtype()
            );
        }
    }

    private void executeF64(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            ResolvedDispatchHints hints
    ) {
        int length = out.logicalSize();
        if (canUseDenseDirectArray(left, right, out)) {
            runDirectF64(left.requireF64Array(), right.requireF64Array(), out.requireF64Array(), hints);
            return;
        }
        if (canUseDenseDirectSegment(left, right, out)) {
            MemorySegment leftSegment = left.requireSegment();
            MemorySegment rightSegment = right.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentF64(leftSegment, rightSegment, outSegment, start, end));
            return;
        }

        BinaryStorageLayout layout = BinaryStorageLayout.from(left, right, out);
        if (allArrays(left, right, out)) {
            double[] leftArray = left.requireF64Array();
            double[] rightArray = right.requireF64Array();
            double[] outArray = out.requireF64Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayF64(leftArray, rightArray, outArray, layout, start, end));
            return;
        }
        if (allSegments(left, right, out)) {
            MemorySegment leftSegment = left.requireSegment();
            MemorySegment rightSegment = right.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentF64(leftSegment, rightSegment, outSegment, layout, start, end));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedF64(left, right, out, layout, start, end));
    }

    private void executeF32(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            ResolvedDispatchHints hints
    ) {
        int length = out.logicalSize();
        if (canUseDenseDirectArray(left, right, out)) {
            runDirectF32(left.requireF32Array(), right.requireF32Array(), out.requireF32Array(), hints);
            return;
        }
        if (canUseDenseDirectSegment(left, right, out)) {
            MemorySegment leftSegment = left.requireSegment();
            MemorySegment rightSegment = right.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentF32(leftSegment, rightSegment, outSegment, start, end));
            return;
        }

        BinaryStorageLayout layout = BinaryStorageLayout.from(left, right, out);
        if (allArrays(left, right, out)) {
            float[] leftArray = left.requireF32Array();
            float[] rightArray = right.requireF32Array();
            float[] outArray = out.requireF32Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayF32(leftArray, rightArray, outArray, layout, start, end));
            return;
        }
        if (allSegments(left, right, out)) {
            MemorySegment leftSegment = left.requireSegment();
            MemorySegment rightSegment = right.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentF32(leftSegment, rightSegment, outSegment, layout, start, end));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedF32(left, right, out, layout, start, end));
    }

    private void executeBF16(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            CpuKernelContext context
    ) {
        ResolvedDispatchHints hints = context.dispatchHints();
        int length = out.logicalSize();
        if (canUseDenseDirectArray(left, right, out)) {
            short[] leftArray = left.requireBF16Array();
            short[] rightArray = right.requireBF16Array();
            float[] leftContinuation = context.inputFloatContinuation(0, left.logicalSize());
            float[] rightContinuation = context.inputFloatContinuation(1, right.logicalSize());
            if (canPublishFloatContinuation(context)) {
                float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
                runDirectBF16ToFloat(leftArray, rightArray, leftContinuation, rightContinuation, outFloat, hints);
                context.cpuWorkspace().publishFloatContinuation(length);
                return;
            }
            runDirectBF16(leftArray, rightArray, leftContinuation, rightContinuation, out.requireBF16Array(), hints);
            return;
        }
        if (canUseDenseDirectSegment(left, right, out)) {
            MemorySegment leftSegment = left.requireSegment();
            MemorySegment rightSegment = right.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentBF16(leftSegment, rightSegment, outSegment, start, end));
            return;
        }

        BinaryStorageLayout layout = BinaryStorageLayout.from(left, right, out);
        if (allArrays(left, right, out)) {
            short[] leftArray = left.requireBF16Array();
            short[] rightArray = right.requireBF16Array();
            float[] leftContinuation = context.inputFloatContinuation(0, left.logicalSize());
            float[] rightContinuation = context.inputFloatContinuation(1, right.logicalSize());
            if (canPublishFloatContinuation(context)) {
                float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
                ElementwiseRangeLoop.runScalar(length, hints,
                        (start, end) -> runIndexedArrayBF16ToFloat(
                                leftArray,
                                rightArray,
                                leftContinuation,
                                rightContinuation,
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
                    (start, end) -> runIndexedArrayBF16(
                            leftArray,
                            rightArray,
                            leftContinuation,
                            rightContinuation,
                            outArray,
                            layout,
                            start,
                            end
                    ));
            return;
        }
        if (allSegments(left, right, out)) {
            MemorySegment leftSegment = left.requireSegment();
            MemorySegment rightSegment = right.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentBF16(leftSegment, rightSegment, outSegment, layout, start, end));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedBF16(left, right, out, layout, start, end));
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

    private static boolean allArrays(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return left.isArray() && right.isArray() && out.isArray();
    }

    private static boolean allSegments(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return left.isMemorySegment() && right.isMemorySegment() && out.isMemorySegment();
    }

    private static boolean isDenseZeroArray(CpuStorageView view) {
        return view.isArray() && isDenseZero(view);
    }

    private static boolean isDenseZeroSegment(CpuStorageView view) {
        return view.isMemorySegment() && isDenseZero(view);
    }

    private static boolean canUseDenseDirectArray(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return isDenseZeroArray(left)
                && isDenseZeroArray(right)
                && isDenseZeroArray(out)
                && left.logicalSize() == out.logicalSize()
                && right.logicalSize() == out.logicalSize();
    }

    private static boolean canUseDenseDirectSegment(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return isDenseZeroSegment(left)
                && isDenseZeroSegment(right)
                && isDenseZeroSegment(out)
                && left.logicalSize() == out.logicalSize()
                && right.logicalSize() == out.logicalSize();
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

    private void requireBinaryCall(CpuKernelCall call) {
        if (call.inputs().size() != 2 || call.inputTensors().size() != 2) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires exactly 2 inputs.");
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
        if (call.inputTensors().get(0).getDataType() != dtype || call.inputTensors().get(1).getDataType() != dtype) {
            return "native-kernel-ineligible:" + opLabel() + "-dtype";
        }
        if (!ElementwiseLayoutPlan.canBroadcastTo(call.inputs().get(0).shape(), call.output().shape())
                || !ElementwiseLayoutPlan.canBroadcastTo(call.inputs().get(1).shape(), call.output().shape())) {
            return "native-kernel-ineligible:" + opLabel() + "-shape";
        }
        return "";
    }

    private static boolean supportsNativeElementwiseDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.FLOAT64 || dtype == DataType.BFLOAT16;
    }

    protected record BinaryStorageLayout(int[] shape, int[][] cursorStrides, int[] cursorBaseOffsets) {
        static BinaryStorageLayout from(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
            int[] outputShape = out.shape();
            return new BinaryStorageLayout(
                    outputShape,
                    new int[][]{
                            out.strides(),
                            ElementwiseLayoutPlan.broadcastStrides(left.shape(), left.strides(), outputShape),
                            ElementwiseLayoutPlan.broadcastStrides(right.shape(), right.strides(), outputShape)
                    },
                    new int[]{out.storageOffset(), left.storageOffset(), right.storageOffset()}
            );
        }

        protected BinaryStorageLayout {
            shape = shape.clone();
            cursorStrides = cursorStrides.clone();
            cursorBaseOffsets = cursorBaseOffsets.clone();
        }

        ElementwiseOffsetCursor cursor(int start) {
            return new ElementwiseOffsetCursor(shape, cursorStrides, cursorBaseOffsets, start);
        }
    }
}
