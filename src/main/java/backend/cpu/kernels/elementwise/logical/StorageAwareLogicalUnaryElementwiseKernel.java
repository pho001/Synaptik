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

abstract class StorageAwareLogicalUnaryElementwiseKernel implements CpuStorageAwareKernel {
    @Override
    public final CpuKernelResult execute(CpuKernelCall call) {
        requireUnaryCall(call);
        if (ElementwiseNativeSupport.nativeRequested(call.context())) {
            return executeNative(call);
        }
        executeStorage(call.inputs().getFirst(), call.output(), call.context());
        return CpuKernelResult.completed();
    }

    protected abstract Operation.OpType opType();

    protected abstract String opLabel();

    protected abstract void runArray(byte[] input, byte[] out, int start, int end);

    protected abstract void runSegment(MemorySegment input, MemorySegment out, int start, int end);

    protected abstract void runIndexedArray(
            byte[] input,
            byte[] out,
            LogicalUnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedSegment(
            MemorySegment input,
            MemorySegment out,
            LogicalUnaryStorageLayout layout,
            int start,
            int end
    );

    protected abstract void runIndexedMixed(
            CpuStorageView input,
            CpuStorageView out,
            LogicalUnaryStorageLayout layout,
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
            NativeTensorStorage inputStorage = ElementwiseNativeSupport.requireNativeInput(
                    context,
                    0,
                    DataType.BOOL,
                    op
            );
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(
                    call.outputTensor(),
                    context,
                    "logical-unary-segment-" + opLabel()
            );
            CpuStorageView input = ElementwiseNativeSupport.segmentView(call.inputTensors().getFirst(), inputStorage);
            CpuStorageView out = ElementwiseNativeSupport.segmentView(call.outputTensor(), outputStorage);

            executeStorage(input, out, context);
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
        executeStorage(storage.input(0), storage.output(), context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        return CpuKernelResult.fallback(CpuNativeTraceSupport.CPU_ARRAY, reason);
    }

    private void executeStorage(CpuStorageView input, CpuStorageView out, CpuKernelContext context) {
        if (input.dtype() != DataType.BOOL || out.dtype() != DataType.BOOL) {
            throw new IllegalStateException(opLabel().toUpperCase()
                    + " requires BOOL input and output storage. input=" + input.dtype() + ", out=" + out.dtype());
        }

        int length = out.logicalSize();
        ResolvedDispatchHints hints = context.dispatchHints();
        if (canUseDenseDirectArray(input, out)) {
            byte[] inputArray = input.requireBoolArray();
            byte[] outArray = out.requireBoolArray();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArray(inputArray, outArray, start, end));
            return;
        }
        if (canUseDenseDirectSegment(input, out)) {
            MemorySegment inputSegment = input.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegment(inputSegment, outSegment, start, end));
            return;
        }

        LogicalUnaryStorageLayout layout = LogicalUnaryStorageLayout.from(input, out);
        if (allArrays(input, out)) {
            byte[] inputArray = input.requireBoolArray();
            byte[] outArray = out.requireBoolArray();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArray(inputArray, outArray, layout, start, end));
            return;
        }
        if (allSegments(input, out)) {
            MemorySegment inputSegment = input.requireSegment();
            MemorySegment outSegment = out.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegment(inputSegment, outSegment, layout, start, end));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixed(input, out, layout, start, end));
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

    private static boolean allArrays(CpuStorageView input, CpuStorageView out) {
        return input.isArray() && out.isArray();
    }

    private static boolean allSegments(CpuStorageView input, CpuStorageView out) {
        return input.isMemorySegment() && out.isMemorySegment();
    }

    private static boolean canUseDenseDirectArray(CpuStorageView input, CpuStorageView out) {
        return input.isArray()
                && out.isArray()
                && ElementwiseNativeSupport.isDenseView(input)
                && ElementwiseNativeSupport.isDenseView(out)
                && input.logicalSize() == out.logicalSize();
    }

    private static boolean canUseDenseDirectSegment(CpuStorageView input, CpuStorageView out) {
        return input.isMemorySegment()
                && out.isMemorySegment()
                && ElementwiseNativeSupport.isDenseView(input)
                && ElementwiseNativeSupport.isDenseView(out)
                && input.logicalSize() == out.logicalSize();
    }

    private void requireUnaryCall(CpuKernelCall call) {
        if (call.inputs().size() != 1 || call.inputTensors().size() != 1) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires exactly 1 input.");
        }
        if (call.inputTensors().getFirst().getDataType() != DataType.BOOL
                || call.outputTensor().getDataType() != DataType.BOOL
                || call.output().dtype() != DataType.BOOL) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires BOOL input and BOOL output.");
        }
    }

    private String nativeIneligibleReason(CpuKernelCall call) {
        if (call.context() == null || call.context().nodePlan() == null) {
            return "native-kernel-ineligible:" + opLabel() + "-plan";
        }
        if (call.operation() == null || call.operation().opType() != opType()) {
            return "native-kernel-unsupported:" + opLabel();
        }
        if (call.inputTensors().size() != 1 || call.context().inputNodeIds().size() != 1) {
            return "native-kernel-ineligible:" + opLabel() + "-input-count";
        }
        if (call.inputTensors().getFirst().getDataType() != DataType.BOOL
                || call.outputTensor().getDataType() != DataType.BOOL) {
            return "native-storage-dtype-unsupported:bool";
        }
        if (call.context().nodePlan().stridedPath()
                || !ElementwiseNativeSupport.isDenseView(call.inputTensors().getFirst())
                || !ElementwiseNativeSupport.isDenseView(call.outputTensor())) {
            return "native-kernel-ineligible:" + opLabel() + "-strided";
        }
        if (call.inputTensors().getFirst().getFlatDataSize() != call.outputTensor().getFlatDataSize()) {
            return "native-kernel-ineligible:" + opLabel() + "-shape";
        }
        return "";
    }

    protected record LogicalUnaryStorageLayout(int[] shape, int[][] cursorStrides, int[] cursorBaseOffsets) {
        static LogicalUnaryStorageLayout from(CpuStorageView input, CpuStorageView out) {
            int[] outputShape = out.shape();
            return new LogicalUnaryStorageLayout(
                    outputShape,
                    new int[][]{
                            out.strides(),
                            ElementwiseLayoutPlan.broadcastStrides(input.shape(), input.strides(), outputShape)
                    },
                    new int[]{out.storageOffset(), input.storageOffset()}
            );
        }

        protected LogicalUnaryStorageLayout {
            shape = shape.clone();
            cursorStrides = cursorStrides.clone();
            cursorBaseOffsets = cursorBaseOffsets.clone();
        }

        ElementwiseOffsetCursor cursor(int start) {
            return new ElementwiseOffsetCursor(shape, cursorStrides, cursorBaseOffsets, start);
        }
    }
}
