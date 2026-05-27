package backend.cpu.kernels.elementwise.compare;

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
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

abstract class StorageAwareCompareElementwiseKernel implements CpuStorageAwareKernel {
    @Override
    public final CpuKernelResult execute(CpuKernelCall call) {
        requireCompareCall(call);
        if (ElementwiseNativeSupport.nativeRequested(call.context())) {
            return executeNative(call);
        }
        executeStorage(call.inputs().get(0), call.inputs().get(1), call.output(), call.context());
        return CpuKernelResult.completed();
    }

    protected abstract Operation.OpType opType();

    protected abstract String opLabel();

    protected abstract void runArrayF64(double[] left, double[] right, byte[] out, int start, int end);

    protected abstract void runArrayF32(float[] left, float[] right, byte[] out, int start, int end);

    protected abstract void runArrayBF16(short[] left, short[] right, byte[] out, int start, int end);

    protected abstract void runSegmentF64(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end);

    protected abstract void runSegmentF32(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end);

    protected abstract void runSegmentBF16(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end);

    protected abstract void runIndexedArrayF64(
            double[] left,
            double[] right,
            byte[] out,
            CompareStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedArrayF32(
            float[] left,
            float[] right,
            byte[] out,
            CompareStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedArrayBF16(
            short[] left,
            short[] right,
            byte[] out,
            CompareStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentF64(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            CompareStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentF32(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            CompareStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegmentBF16(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            CompareStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedF64(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            CompareStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedF32(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            CompareStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixedBF16(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            CompareStorageLayout layout,
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
            DataType inputDType = call.inputTensors().get(0).getDataType();
            NativeTensorStorage leftStorage = ElementwiseNativeSupport.requireNativeInput(context, 0, inputDType, op);
            NativeTensorStorage rightStorage = ElementwiseNativeSupport.requireNativeInput(context, 1, inputDType, op);
            CpuStorageView left = ElementwiseNativeSupport.segmentView(call.inputTensors().get(0), leftStorage);
            CpuStorageView right = ElementwiseNativeSupport.segmentView(call.inputTensors().get(1), rightStorage);
            CpuStorageView out = new CpuStorageResolver().bindArrayOnly(call.outputTensor());

            executeStorage(left, right, out, context);
            TensorInternalAccess.markStorageModified(call.outputTensor());
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, "");
            return CpuKernelResult.route(CpuNativeTraceSupport.CPU_ARRAY);
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
        if (out.dtype() != DataType.BOOL) {
            throw new IllegalStateException(opLabel().toUpperCase() + " requires BOOL output storage, actual="
                    + out.dtype());
        }
        if (left.dtype() != right.dtype()) {
            throw new IllegalStateException(opLabel().toUpperCase() + " input dtype mismatch. left=" + left.dtype()
                    + ", right=" + right.dtype());
        }
        switch (left.dtype()) {
            case FLOAT64 -> executeF64(left, right, out, hints(context));
            case FLOAT32 -> executeF32(left, right, out, hints(context));
            case BFLOAT16 -> executeBF16(left, right, out, hints(context));
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    opLabel().toUpperCase() + " does not support input dtype: " + left.dtype()
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
            double[] leftArray = left.requireF64Array();
            double[] rightArray = right.requireF64Array();
            byte[] outArray = out.requireBoolArray();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayF64(leftArray, rightArray, outArray, start, end));
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

        CompareStorageLayout layout = CompareStorageLayout.from(left, right, out);
        if (allArrays(left, right, out)) {
            double[] leftArray = left.requireF64Array();
            double[] rightArray = right.requireF64Array();
            byte[] outArray = out.requireBoolArray();
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
            float[] leftArray = left.requireF32Array();
            float[] rightArray = right.requireF32Array();
            byte[] outArray = out.requireBoolArray();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayF32(leftArray, rightArray, outArray, start, end));
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

        CompareStorageLayout layout = CompareStorageLayout.from(left, right, out);
        if (allArrays(left, right, out)) {
            float[] leftArray = left.requireF32Array();
            float[] rightArray = right.requireF32Array();
            byte[] outArray = out.requireBoolArray();
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
            ResolvedDispatchHints hints
    ) {
        int length = out.logicalSize();
        if (canUseDenseDirectArray(left, right, out)) {
            short[] leftArray = left.requireBF16Array();
            short[] rightArray = right.requireBF16Array();
            byte[] outArray = out.requireBoolArray();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayBF16(leftArray, rightArray, outArray, start, end));
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

        CompareStorageLayout layout = CompareStorageLayout.from(left, right, out);
        if (allArrays(left, right, out)) {
            short[] leftArray = left.requireBF16Array();
            short[] rightArray = right.requireBF16Array();
            byte[] outArray = out.requireBoolArray();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayBF16(leftArray, rightArray, outArray, layout, start, end));
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

    protected static byte bool(boolean value) {
        return value ? (byte) 1 : (byte) 0;
    }

    protected static double readF64(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireF64Array()[offset]
                : view.requireSegment().get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    protected static float readF32(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireF32Array()[offset]
                : view.requireSegment().get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    protected static float readBF16(CpuStorageView view, int offset) {
        short bits = view.isArray()
                ? view.requireBF16Array()[offset]
                : view.requireSegment().get(JAVA_SHORT, (long) offset * Short.BYTES);
        return TensorDTypeOps.fromBFloat16Bits(bits);
    }

    protected static void writeBool(CpuStorageView view, int offset, boolean value) {
        byte stored = bool(value);
        if (view.isArray()) {
            view.requireBoolArray()[offset] = stored;
        } else {
            view.requireSegment().set(JAVA_BYTE, offset, stored);
        }
    }

    private static boolean allArrays(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return left.isArray() && right.isArray() && out.isArray();
    }

    private static boolean allSegments(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return left.isMemorySegment() && right.isMemorySegment() && out.isMemorySegment();
    }

    private static boolean canUseDenseDirectArray(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return left.isArray()
                && right.isArray()
                && out.isArray()
                && isDenseZero(left)
                && isDenseZero(right)
                && isDenseZero(out)
                && left.logicalSize() == out.logicalSize()
                && right.logicalSize() == out.logicalSize();
    }

    private static boolean canUseDenseDirectSegment(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return left.isMemorySegment()
                && right.isMemorySegment()
                && out.isMemorySegment()
                && isDenseZero(left)
                && isDenseZero(right)
                && isDenseZero(out)
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

    private void requireCompareCall(CpuKernelCall call) {
        if (call.inputs().size() != 2 || call.inputTensors().size() != 2) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires exactly 2 inputs.");
        }
        if (call.output().dtype() != DataType.BOOL || call.outputTensor().getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires BOOL output.");
        }
    }

    private String nativeIneligibleReason(CpuKernelCall call) {
        if (call.context() == null || call.context().nodePlan() == null) {
            return "native-kernel-ineligible:" + opLabel() + "-plan";
        }
        if (call.operation() == null || call.operation().opType() != opType()) {
            return "native-kernel-unsupported:" + opLabel();
        }
        if (call.outputTensor().getDataType() != DataType.BOOL) {
            return "native-kernel-ineligible:" + opLabel() + "-output-dtype";
        }
        if (call.inputTensors().size() != 2 || call.context().inputNodeIds().size() != 2) {
            return "native-kernel-ineligible:" + opLabel() + "-input-count";
        }
        DataType inputDType = call.inputTensors().get(0).getDataType();
        if (!supportsInputDType(inputDType) || call.inputTensors().get(1).getDataType() != inputDType) {
            return "native-storage-dtype-unsupported:" + inputDType.name().toLowerCase();
        }
        if (call.context().broadcastPlan() != null && !call.context().broadcastPlan().isNoBroadcast()) {
            return "native-kernel-ineligible:" + opLabel() + "-broadcast";
        }
        if (call.context().nodePlan().stridedPath()
                || !ElementwiseNativeSupport.isDenseView(call.inputTensors().get(0))
                || !ElementwiseNativeSupport.isDenseView(call.inputTensors().get(1))
                || !ElementwiseNativeSupport.isDenseView(call.outputTensor())) {
            return "native-kernel-ineligible:" + opLabel() + "-strided";
        }
        int size = call.outputTensor().getFlatDataSize();
        if (call.inputTensors().get(0).getFlatDataSize() != size
                || call.inputTensors().get(1).getFlatDataSize() != size) {
            return "native-kernel-ineligible:" + opLabel() + "-shape";
        }
        return "";
    }

    private static ResolvedDispatchHints hints(CpuKernelContext context) {
        return context == null ? null : context.dispatchHints();
    }

    private static boolean supportsInputDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    protected record CompareStorageLayout(int[] shape, int[][] cursorStrides, int[] cursorBaseOffsets) {
        static CompareStorageLayout from(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
            int[] outputShape = out.shape();
            return new CompareStorageLayout(
                    outputShape,
                    new int[][]{
                            out.strides(),
                            ElementwiseLayoutPlan.broadcastStrides(left.shape(), left.strides(), outputShape),
                            ElementwiseLayoutPlan.broadcastStrides(right.shape(), right.strides(), outputShape)
                    },
                    new int[]{out.storageOffset(), left.storageOffset(), right.storageOffset()}
            );
        }

        protected CompareStorageLayout {
            shape = shape.clone();
            cursorStrides = cursorStrides.clone();
            cursorBaseOffsets = cursorBaseOffsets.clone();
        }

        ElementwiseOffsetCursor cursor(int start) {
            return new ElementwiseOffsetCursor(shape, cursorStrides, cursorBaseOffsets, start);
        }
    }
}
