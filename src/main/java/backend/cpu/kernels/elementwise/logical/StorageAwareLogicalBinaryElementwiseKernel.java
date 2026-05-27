package backend.cpu.kernels.elementwise.logical;

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
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

abstract class StorageAwareLogicalBinaryElementwiseKernel implements CpuStorageAwareKernel {
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

    protected abstract void runArray(byte[] left, byte[] right, byte[] out, int start, int end);

    protected abstract void runSegment(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end);

    protected abstract void runIndexedArray(
            byte[] left,
            byte[] right,
            byte[] out,
            LogicalBinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegment(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            LogicalBinaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixed(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            LogicalBinaryStorageLayout layout,
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
                    DataType.BOOL,
                    op
            );
            NativeTensorStorage rightStorage = ElementwiseNativeSupport.requireNativeInput(
                    context,
                    1,
                    DataType.BOOL,
                    op
            );
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(
                    call.outputTensor(),
                    context,
                    "logical-binary-segment-" + opLabel()
            );
            CpuStorageView left = ElementwiseNativeSupport.segmentView(call.inputTensors().get(0), leftStorage);
            CpuStorageView right = ElementwiseNativeSupport.segmentView(call.inputTensors().get(1), rightStorage);
            CpuStorageView out = ElementwiseNativeSupport.segmentView(call.outputTensor(), outputStorage);

            executeStorage(left, right, out, context);
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    op + " wrote BOOL native output"
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
        if (left.dtype() != DataType.BOOL || right.dtype() != DataType.BOOL || out.dtype() != DataType.BOOL) {
            throw new IllegalStateException(opLabel().toUpperCase()
                    + " requires BOOL input and output storage. left=" + left.dtype()
                    + ", right=" + right.dtype() + ", out=" + out.dtype());
        }

        int length = out.logicalSize();
        ResolvedDispatchHints hints = context.dispatchHints();
        if (canUseDenseDirectArray(left, right, out)) {
            byte[] leftArray = left.requireBoolArray();
            byte[] rightArray = right.requireBoolArray();
            byte[] outArray = out.requireBoolArray();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArray(leftArray, rightArray, outArray, start, end));
            return;
        }
        if (canUseDenseDirectSegment(left, right, out)) {
            MemorySegment leftSegment = left.requireSegment();
            MemorySegment rightSegment = right.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegment(leftSegment, rightSegment, outSegment, start, end));
            return;
        }

        LogicalBinaryStorageLayout layout = LogicalBinaryStorageLayout.from(left, right, out);
        if (allArrays(left, right, out)) {
            byte[] leftArray = left.requireBoolArray();
            byte[] rightArray = right.requireBoolArray();
            byte[] outArray = out.requireBoolArray();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArray(leftArray, rightArray, outArray, layout, start, end));
            return;
        }
        if (allSegments(left, right, out)) {
            MemorySegment leftSegment = left.requireSegment();
            MemorySegment rightSegment = right.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegment(leftSegment, rightSegment, outSegment, layout, start, end));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixed(left, right, out, layout, start, end));
    }

    protected static byte bool(boolean value) {
        return value ? (byte) 1 : (byte) 0;
    }

    protected static byte readBool(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireBoolArray()[offset]
                : view.requireSegment().get(JAVA_BYTE, offset);
    }

    protected static void writeBool(CpuStorageView view, int offset, byte value) {
        if (view.isArray()) {
            view.requireBoolArray()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_BYTE, offset, value);
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
                && ElementwiseNativeSupport.isDenseView(left)
                && ElementwiseNativeSupport.isDenseView(right)
                && ElementwiseNativeSupport.isDenseView(out)
                && left.logicalSize() == out.logicalSize()
                && right.logicalSize() == out.logicalSize();
    }

    private static boolean canUseDenseDirectSegment(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
        return left.isMemorySegment()
                && right.isMemorySegment()
                && out.isMemorySegment()
                && ElementwiseNativeSupport.isDenseView(left)
                && ElementwiseNativeSupport.isDenseView(right)
                && ElementwiseNativeSupport.isDenseView(out)
                && left.logicalSize() == out.logicalSize()
                && right.logicalSize() == out.logicalSize();
    }

    private void requireBinaryCall(CpuKernelCall call) {
        if (call.inputs().size() != 2 || call.inputTensors().size() != 2) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires exactly 2 inputs.");
        }
        if (call.inputTensors().get(0).getDataType() != DataType.BOOL
                || call.inputTensors().get(1).getDataType() != DataType.BOOL
                || call.outputTensor().getDataType() != DataType.BOOL
                || call.output().dtype() != DataType.BOOL) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires BOOL inputs and BOOL output.");
        }
    }

    private String nativeIneligibleReason(CpuKernelCall call) {
        if (call.context() == null || call.context().nodePlan() == null) {
            return "native-kernel-ineligible:" + opLabel() + "-plan";
        }
        if (call.operation() == null || call.operation().opType() != opType()) {
            return "native-kernel-unsupported:" + opLabel();
        }
        if (call.inputTensors().size() != 2 || call.context().inputNodeIds().size() != 2) {
            return "native-kernel-ineligible:" + opLabel() + "-input-count";
        }
        if (call.inputTensors().get(0).getDataType() != DataType.BOOL
                || call.inputTensors().get(1).getDataType() != DataType.BOOL
                || call.outputTensor().getDataType() != DataType.BOOL) {
            return "native-storage-dtype-unsupported:bool";
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

    protected record LogicalBinaryStorageLayout(int[] shape, int[][] cursorStrides, int[] cursorBaseOffsets) {
        static LogicalBinaryStorageLayout from(CpuStorageView left, CpuStorageView right, CpuStorageView out) {
            int[] outputShape = out.shape();
            return new LogicalBinaryStorageLayout(
                    outputShape,
                    new int[][]{
                            out.strides(),
                            ElementwiseLayoutPlan.broadcastStrides(left.shape(), left.strides(), outputShape),
                            ElementwiseLayoutPlan.broadcastStrides(right.shape(), right.strides(), outputShape)
                    },
                    new int[]{out.storageOffset(), left.storageOffset(), right.storageOffset()}
            );
        }

        protected LogicalBinaryStorageLayout {
            shape = shape.clone();
            cursorStrides = cursorStrides.clone();
            cursorBaseOffsets = cursorBaseOffsets.clone();
        }

        ElementwiseOffsetCursor cursor(int start) {
            return new ElementwiseOffsetCursor(shape, cursorStrides, cursorBaseOffsets, start);
        }
    }
}
