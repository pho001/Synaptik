package backend.cpu.kernels.elementwise.where;

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

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuWhereKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        requireWhereCall(call);
        if (ElementwiseNativeSupport.nativeRequested(call.context())) {
            return executeNative(call);
        }
        executeStorage(call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(), call.context());
        return CpuKernelResult.completed();
    }

    public double applyF64(byte condition, double ifTrue, double ifFalse) {
        return condition != 0 ? ifTrue : ifFalse;
    }

    public float applyF32(byte condition, float ifTrue, float ifFalse) {
        return condition != 0 ? ifTrue : ifFalse;
    }

    public float applyBF16(byte condition, float ifTrue, float ifFalse) {
        return condition != 0 ? ifTrue : ifFalse;
    }

    private CpuKernelResult executeNative(CpuKernelCall call) {
        String ineligibleReason = nativeIneligibleReason(call);
        if (!ineligibleReason.isBlank()) {
            return fallbackToArray(call, ineligibleReason);
        }

        CpuKernelContext context = call.context();
        try {
            NativeTensorStorage trueStorage = ElementwiseNativeSupport.requireNativeInput(
                    context,
                    1,
                    call.output().dtype(),
                    "WHERE"
            );
            NativeTensorStorage falseStorage = ElementwiseNativeSupport.requireNativeInput(
                    context,
                    2,
                    call.output().dtype(),
                    "WHERE"
            );
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(
                    call.outputTensor(),
                    context,
                    "where-storage-loop"
            );
            CpuStorageView condition = new CpuStorageResolver().bindArrayOnly(call.inputTensors().get(0));
            CpuStorageView ifTrue = ElementwiseNativeSupport.segmentView(call.inputTensors().get(1), trueStorage);
            CpuStorageView ifFalse = ElementwiseNativeSupport.segmentView(call.inputTensors().get(2), falseStorage);
            CpuStorageView output = ElementwiseNativeSupport.segmentView(call.outputTensor(), outputStorage);

            executeStorage(condition, ifTrue, ifFalse, output, context);
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "WHERE storage loop wrote " + call.output().dtype() + " native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
            return CpuKernelResult.route(CpuNativeTraceSupport.CPU_NATIVE);
        } catch (Throwable t) {
            return fallbackToArray(call, "native-kernel-failed:where:" + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    private CpuKernelResult fallbackToArray(CpuKernelCall call, String reason) {
        CpuKernelContext context = call.context();
        ElementwiseNativeSupport.requireFallbackAllowed(context, "where elementwise", reason);
        ElementwiseNativeSupport.requireCpuReadableInputs(context);
        CpuStorageBindings storage = new CpuStorageResolver().bindArrayOnly(call.inputTensors(), call.outputTensor());
        executeStorage(storage.input(0), storage.input(1), storage.input(2), storage.output(), context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        return CpuKernelResult.fallback(CpuNativeTraceSupport.CPU_ARRAY, reason);
    }

    private void executeStorage(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        if (condition.dtype() != DataType.BOOL) {
            throw new IllegalStateException("WHERE requires BOOL condition storage, actual=" + condition.dtype());
        }
        if (ifTrue.dtype() != output.dtype() || ifFalse.dtype() != output.dtype()) {
            throw new IllegalStateException("WHERE branch/output dtype mismatch. true=" + ifTrue.dtype()
                    + ", false=" + ifFalse.dtype() + ", out=" + output.dtype());
        }
        switch (output.dtype()) {
            case FLOAT64 -> executeF64(condition, ifTrue, ifFalse, output, hints(context));
            case FLOAT32 -> executeF32(condition, ifTrue, ifFalse, output, hints(context));
            case BFLOAT16 -> executeBF16(condition, ifTrue, ifFalse, output, context);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "WHERE only supports floating output tensors"
            );
        }
    }

    private void executeF64(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output,
            ResolvedDispatchHints hints
    ) {
        int length = output.logicalSize();
        if (canUseDenseDirectArray(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            double[] trueArray = ifTrue.requireF64Array();
            double[] falseArray = ifFalse.requireF64Array();
            double[] outArray = output.requireF64Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayF64(conditionArray, trueArray, falseArray, outArray, start, end));
            return;
        }
        if (canUseDenseDirectSegment(condition, ifTrue, ifFalse, output)) {
            MemorySegment conditionSegment = condition.requireSegment();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentF64(conditionSegment, trueSegment, falseSegment, outSegment, start, end));
            return;
        }
        if (canUseDenseArrayConditionSegmentBranches(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayConditionSegmentBranchesF64(
                            conditionArray,
                            trueSegment,
                            falseSegment,
                            outSegment,
                            start,
                            end
                    ));
            return;
        }

        WhereStorageLayout layout = WhereStorageLayout.from(condition, ifTrue, ifFalse, output);
        if (allArrays(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            double[] trueArray = ifTrue.requireF64Array();
            double[] falseArray = ifFalse.requireF64Array();
            double[] outArray = output.requireF64Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayF64(
                            conditionArray,
                            trueArray,
                            falseArray,
                            outArray,
                            layout,
                            start,
                            end
                    ));
            return;
        }
        if (allSegments(condition, ifTrue, ifFalse, output)) {
            MemorySegment conditionSegment = condition.requireSegment();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentF64(
                            conditionSegment,
                            trueSegment,
                            falseSegment,
                            outSegment,
                            layout,
                            start,
                            end
                    ));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedF64(condition, ifTrue, ifFalse, output, layout, start, end));
    }

    private void executeF32(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output,
            ResolvedDispatchHints hints
    ) {
        int length = output.logicalSize();
        if (canUseDenseDirectArray(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            float[] trueArray = ifTrue.requireF32Array();
            float[] falseArray = ifFalse.requireF32Array();
            float[] outArray = output.requireF32Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayF32(conditionArray, trueArray, falseArray, outArray, start, end));
            return;
        }
        if (canUseDenseDirectSegment(condition, ifTrue, ifFalse, output)) {
            MemorySegment conditionSegment = condition.requireSegment();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentF32(conditionSegment, trueSegment, falseSegment, outSegment, start, end));
            return;
        }
        if (canUseDenseArrayConditionSegmentBranches(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayConditionSegmentBranchesF32(
                            conditionArray,
                            trueSegment,
                            falseSegment,
                            outSegment,
                            start,
                            end
                    ));
            return;
        }

        WhereStorageLayout layout = WhereStorageLayout.from(condition, ifTrue, ifFalse, output);
        if (allArrays(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            float[] trueArray = ifTrue.requireF32Array();
            float[] falseArray = ifFalse.requireF32Array();
            float[] outArray = output.requireF32Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayF32(
                            conditionArray,
                            trueArray,
                            falseArray,
                            outArray,
                            layout,
                            start,
                            end
                    ));
            return;
        }
        if (allSegments(condition, ifTrue, ifFalse, output)) {
            MemorySegment conditionSegment = condition.requireSegment();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentF32(
                            conditionSegment,
                            trueSegment,
                            falseSegment,
                            outSegment,
                            layout,
                            start,
                            end
                    ));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedF32(condition, ifTrue, ifFalse, output, layout, start, end));
    }

    private void executeBF16(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        ResolvedDispatchHints hints = hints(context);
        int length = output.logicalSize();
        if (canUseDenseDirectArray(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            short[] trueArray = ifTrue.requireBF16Array();
            short[] falseArray = ifFalse.requireBF16Array();
            float[] trueContinuation = context.inputFloatContinuation(1, ifTrue.logicalSize());
            float[] falseContinuation = context.inputFloatContinuation(2, ifFalse.logicalSize());
            if (canPublishFloatContinuation(context)) {
                float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
                ElementwiseRangeLoop.runScalar(length, hints,
                        (start, end) -> runArrayBF16ToFloat(
                                conditionArray,
                                trueArray,
                                falseArray,
                                trueContinuation,
                                falseContinuation,
                                outFloat,
                                start,
                                end
                        ));
                context.cpuWorkspace().publishFloatContinuation(length);
                return;
            }
            short[] outArray = output.requireBF16Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayBF16(
                            conditionArray,
                            trueArray,
                            falseArray,
                            trueContinuation,
                            falseContinuation,
                            outArray,
                            start,
                            end
                    ));
            return;
        }
        if (canUseDenseDirectSegment(condition, ifTrue, ifFalse, output)) {
            MemorySegment conditionSegment = condition.requireSegment();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runSegmentBF16(
                            conditionSegment,
                            trueSegment,
                            falseSegment,
                            outSegment,
                            start,
                            end
                    ));
            return;
        }
        if (canUseDenseArrayConditionSegmentBranches(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runArrayConditionSegmentBranchesBF16(
                            conditionArray,
                            trueSegment,
                            falseSegment,
                            outSegment,
                            start,
                            end
                    ));
            return;
        }

        WhereStorageLayout layout = WhereStorageLayout.from(condition, ifTrue, ifFalse, output);
        if (allArrays(condition, ifTrue, ifFalse, output)) {
            byte[] conditionArray = condition.requireBoolArray();
            short[] trueArray = ifTrue.requireBF16Array();
            short[] falseArray = ifFalse.requireBF16Array();
            float[] trueContinuation = context.inputFloatContinuation(1, ifTrue.logicalSize());
            float[] falseContinuation = context.inputFloatContinuation(2, ifFalse.logicalSize());
            if (canPublishFloatContinuation(context)) {
                float[] outFloat = context.cpuWorkspace().requireFloatWorkspace();
                ElementwiseRangeLoop.runScalar(length, hints,
                        (start, end) -> runIndexedArrayBF16ToFloat(
                                conditionArray,
                                trueArray,
                                falseArray,
                                trueContinuation,
                                falseContinuation,
                                outFloat,
                                layout,
                                start,
                                end
                        ));
                context.cpuWorkspace().publishFloatContinuation(length);
                return;
            }
            short[] outArray = output.requireBF16Array();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedArrayBF16(
                            conditionArray,
                            trueArray,
                            falseArray,
                            trueContinuation,
                            falseContinuation,
                            outArray,
                            layout,
                            start,
                            end
                    ));
            return;
        }
        if (allSegments(condition, ifTrue, ifFalse, output)) {
            MemorySegment conditionSegment = condition.requireSegment();
            MemorySegment trueSegment = ifTrue.requireSegment();
            MemorySegment falseSegment = ifFalse.requireSegment();
            MemorySegment outSegment = output.requireSegment();
            ElementwiseRangeLoop.runScalar(length, hints,
                    (start, end) -> runIndexedSegmentBF16(
                            conditionSegment,
                            trueSegment,
                            falseSegment,
                            outSegment,
                            layout,
                            start,
                            end
                    ));
            return;
        }
        ElementwiseRangeLoop.runScalar(length, hints,
                (start, end) -> runIndexedMixedBF16(condition, ifTrue, ifFalse, output, layout, start, end));
    }

    private void runArrayF64(byte[] condition, double[] ifTrue, double[] ifFalse, double[] output, int start, int end) {
        for (int i = start; i < end; i++) {
            output[i] = applyF64(condition[i], ifTrue[i], ifFalse[i]);
        }
    }

    private void runArrayF32(byte[] condition, float[] ifTrue, float[] ifFalse, float[] output, int start, int end) {
        for (int i = start; i < end; i++) {
            output[i] = applyF32(condition[i], ifTrue[i], ifFalse[i]);
        }
    }

    private void runArrayBF16(
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] trueContinuation,
            float[] falseContinuation,
            short[] output,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            output[i] = TensorDTypeOps.toBFloat16Bits(applyBF16(
                    condition[i],
                    loadBF16(trueContinuation, ifTrue, i),
                    loadBF16(falseContinuation, ifFalse, i)
            ));
        }
    }

    private void runArrayBF16ToFloat(
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] trueContinuation,
            float[] falseContinuation,
            float[] output,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            output[i] = applyBF16(
                    condition[i],
                    loadBF16(trueContinuation, ifTrue, i),
                    loadBF16(falseContinuation, ifFalse, i)
            );
        }
    }

    private void runSegmentF64(
            MemorySegment condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            output.set(JAVA_DOUBLE, offset, applyF64(
                    condition.get(JAVA_BYTE, i),
                    ifTrue.get(JAVA_DOUBLE, offset),
                    ifFalse.get(JAVA_DOUBLE, offset)
            ));
        }
    }

    private void runSegmentF32(
            MemorySegment condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            output.set(JAVA_FLOAT, offset, applyF32(
                    condition.get(JAVA_BYTE, i),
                    ifTrue.get(JAVA_FLOAT, offset),
                    ifFalse.get(JAVA_FLOAT, offset)
            ));
        }
    }

    private void runSegmentBF16(
            MemorySegment condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float trueValue = TensorDTypeOps.fromBFloat16Bits(ifTrue.get(JAVA_SHORT, offset));
            float falseValue = TensorDTypeOps.fromBFloat16Bits(ifFalse.get(JAVA_SHORT, offset));
            output.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(
                    applyBF16(condition.get(JAVA_BYTE, i), trueValue, falseValue)
            ));
        }
    }

    private void runArrayConditionSegmentBranchesF64(
            byte[] condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            output.set(JAVA_DOUBLE, offset, applyF64(
                    condition[i],
                    ifTrue.get(JAVA_DOUBLE, offset),
                    ifFalse.get(JAVA_DOUBLE, offset)
            ));
        }
    }

    private void runArrayConditionSegmentBranchesF32(
            byte[] condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            output.set(JAVA_FLOAT, offset, applyF32(
                    condition[i],
                    ifTrue.get(JAVA_FLOAT, offset),
                    ifFalse.get(JAVA_FLOAT, offset)
            ));
        }
    }

    private void runArrayConditionSegmentBranchesBF16(
            byte[] condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float trueValue = TensorDTypeOps.fromBFloat16Bits(ifTrue.get(JAVA_SHORT, offset));
            float falseValue = TensorDTypeOps.fromBFloat16Bits(ifFalse.get(JAVA_SHORT, offset));
            output.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(
                    applyBF16(condition[i], trueValue, falseValue)
            ));
        }
    }

    private void runIndexedArrayF64(
            byte[] condition,
            double[] ifTrue,
            double[] ifFalse,
            double[] output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            output[cursor.offset(0)] = applyF64(
                    condition[cursor.offset(1)],
                    ifTrue[cursor.offset(2)],
                    ifFalse[cursor.offset(3)]
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedArrayF32(
            byte[] condition,
            float[] ifTrue,
            float[] ifFalse,
            float[] output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            output[cursor.offset(0)] = applyF32(
                    condition[cursor.offset(1)],
                    ifTrue[cursor.offset(2)],
                    ifFalse[cursor.offset(3)]
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedArrayBF16(
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] trueContinuation,
            float[] falseContinuation,
            short[] output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            output[cursor.offset(0)] = TensorDTypeOps.toBFloat16Bits(applyBF16(
                    condition[cursor.offset(1)],
                    loadBF16(trueContinuation, ifTrue, cursor.offset(2)),
                    loadBF16(falseContinuation, ifFalse, cursor.offset(3))
            ));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedArrayBF16ToFloat(
            byte[] condition,
            short[] ifTrue,
            short[] ifFalse,
            float[] trueContinuation,
            float[] falseContinuation,
            float[] output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            output[outIndex] = applyBF16(
                    condition[cursor.offset(1)],
                    loadBF16(trueContinuation, ifTrue, cursor.offset(2)),
                    loadBF16(falseContinuation, ifFalse, cursor.offset(3))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedSegmentF64(
            MemorySegment condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            output.set(JAVA_DOUBLE, (long) cursor.offset(0) * Double.BYTES, applyF64(
                    condition.get(JAVA_BYTE, cursor.offset(1)),
                    ifTrue.get(JAVA_DOUBLE, (long) cursor.offset(2) * Double.BYTES),
                    ifFalse.get(JAVA_DOUBLE, (long) cursor.offset(3) * Double.BYTES)
            ));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedSegmentF32(
            MemorySegment condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            output.set(JAVA_FLOAT, (long) cursor.offset(0) * Float.BYTES, applyF32(
                    condition.get(JAVA_BYTE, cursor.offset(1)),
                    ifTrue.get(JAVA_FLOAT, (long) cursor.offset(2) * Float.BYTES),
                    ifFalse.get(JAVA_FLOAT, (long) cursor.offset(3) * Float.BYTES)
            ));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedSegmentBF16(
            MemorySegment condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            float trueValue = TensorDTypeOps.fromBFloat16Bits(
                    ifTrue.get(JAVA_SHORT, (long) cursor.offset(2) * Short.BYTES)
            );
            float falseValue = TensorDTypeOps.fromBFloat16Bits(
                    ifFalse.get(JAVA_SHORT, (long) cursor.offset(3) * Short.BYTES)
            );
            output.set(
                    JAVA_SHORT,
                    (long) cursor.offset(0) * Short.BYTES,
                    TensorDTypeOps.toBFloat16Bits(applyBF16(
                            condition.get(JAVA_BYTE, cursor.offset(1)),
                            trueValue,
                            falseValue
                    ))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedMixedF64(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeF64(output, cursor.offset(0), applyF64(
                    readBool(condition, cursor.offset(1)),
                    readF64(ifTrue, cursor.offset(2)),
                    readF64(ifFalse, cursor.offset(3))
            ));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedMixedF32(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeF32(output, cursor.offset(0), applyF32(
                    readBool(condition, cursor.offset(1)),
                    readF32(ifTrue, cursor.offset(2)),
                    readF32(ifFalse, cursor.offset(3))
            ));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private void runIndexedMixedBF16(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output,
            WhereStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBF16(output, cursor.offset(0), applyBF16(
                    readBool(condition, cursor.offset(1)),
                    readBF16(ifTrue, cursor.offset(2)),
                    readBF16(ifFalse, cursor.offset(3))
            ));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    private static byte readBool(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireBoolArray()[offset]
                : view.requireSegment().get(JAVA_BYTE, offset);
    }

    private static double readF64(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireF64Array()[offset]
                : view.requireSegment().get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static void writeF64(CpuStorageView view, int offset, double value) {
        if (view.isArray()) {
            view.requireF64Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    private static float readF32(CpuStorageView view, int offset) {
        return view.isArray()
                ? view.requireF32Array()[offset]
                : view.requireSegment().get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static void writeF32(CpuStorageView view, int offset, float value) {
        if (view.isArray()) {
            view.requireF32Array()[offset] = value;
        } else {
            view.requireSegment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    private static float readBF16(CpuStorageView view, int offset) {
        short bits = view.isArray()
                ? view.requireBF16Array()[offset]
                : view.requireSegment().get(JAVA_SHORT, (long) offset * Short.BYTES);
        return TensorDTypeOps.fromBFloat16Bits(bits);
    }

    private static void writeBF16(CpuStorageView view, int offset, float value) {
        short bits = TensorDTypeOps.toBFloat16Bits(value);
        if (view.isArray()) {
            view.requireBF16Array()[offset] = bits;
        } else {
            view.requireSegment().set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
        }
    }

    private static float loadBF16(float[] continuation, short[] storage, int index) {
        return continuation != null ? continuation[index] : TensorDTypeOps.fromBFloat16Bits(storage[index]);
    }

    private static boolean allArrays(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output
    ) {
        return condition.isArray() && ifTrue.isArray() && ifFalse.isArray() && output.isArray();
    }

    private static boolean allSegments(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output
    ) {
        return condition.isMemorySegment()
                && ifTrue.isMemorySegment()
                && ifFalse.isMemorySegment()
                && output.isMemorySegment();
    }

    private static boolean canUseDenseDirectArray(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output
    ) {
        return condition.isArray()
                && ifTrue.isArray()
                && ifFalse.isArray()
                && output.isArray()
                && sameDenseZeroSize(condition, ifTrue, ifFalse, output);
    }

    private static boolean canUseDenseDirectSegment(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output
    ) {
        return condition.isMemorySegment()
                && ifTrue.isMemorySegment()
                && ifFalse.isMemorySegment()
                && output.isMemorySegment()
                && sameDenseZeroSize(condition, ifTrue, ifFalse, output);
    }

    private static boolean canUseDenseArrayConditionSegmentBranches(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output
    ) {
        return condition.isArray()
                && ifTrue.isMemorySegment()
                && ifFalse.isMemorySegment()
                && output.isMemorySegment()
                && sameDenseZeroSize(condition, ifTrue, ifFalse, output);
    }

    private static boolean sameDenseZeroSize(
            CpuStorageView condition,
            CpuStorageView ifTrue,
            CpuStorageView ifFalse,
            CpuStorageView output
    ) {
        return isDenseZero(condition)
                && isDenseZero(ifTrue)
                && isDenseZero(ifFalse)
                && isDenseZero(output)
                && condition.logicalSize() == output.logicalSize()
                && ifTrue.logicalSize() == output.logicalSize()
                && ifFalse.logicalSize() == output.logicalSize();
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

    private static ResolvedDispatchHints hints(CpuKernelContext context) {
        return context == null ? null : context.dispatchHints();
    }

    private void requireWhereCall(CpuKernelCall call) {
        if (call.inputs().size() != 3 || call.inputTensors().size() != 3) {
            throw new IllegalArgumentException("WHERE requires exactly 3 inputs.");
        }
        if (call.inputs().get(0).dtype() != DataType.BOOL || call.inputTensors().get(0).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("WHERE requires BOOL condition input.");
        }
    }

    private String nativeIneligibleReason(CpuKernelCall call) {
        if (call.context() == null || call.context().nodePlan() == null) {
            return "native-kernel-ineligible:where-plan";
        }
        Operation op = call.operation();
        Operation.OpType opType = op == null ? Operation.OpType.UNKNOWN : op.opType();
        if (opType != Operation.OpType.WHERE) {
            return "native-kernel-unsupported:" + opType.name().toLowerCase();
        }
        if (call.context().nodePlan().stridedPath()) {
            return "native-kernel-ineligible:where-strided";
        }
        if (!supportsNativeWhereDType(call.output().dtype())) {
            return "native-storage-dtype-unsupported:" + call.output().dtype().name().toLowerCase();
        }
        if (call.inputTensors().size() != 3 || call.context().inputNodeIds().size() != 3) {
            return "native-kernel-ineligible:where-input-count";
        }
        DataType branchDataType = call.output().dtype();
        if (call.inputTensors().get(0).getDataType() != DataType.BOOL
                || call.inputTensors().get(1).getDataType() != branchDataType
                || call.inputTensors().get(2).getDataType() != branchDataType) {
            return "native-kernel-ineligible:where-dtype";
        }
        if (call.context().whereBroadcastPlan() != null && !call.context().whereBroadcastPlan().isNoBroadcast()) {
            return "native-kernel-ineligible:where-broadcast";
        }
        int size = call.outputTensor().getFlatDataSize();
        if (call.inputTensors().get(0).getFlatDataSize() != size
                || call.inputTensors().get(1).getFlatDataSize() != size
                || call.inputTensors().get(2).getFlatDataSize() != size) {
            return "native-kernel-ineligible:where-shape";
        }
        if (!ElementwiseNativeSupport.isDenseView(call.inputTensors().get(0))
                || !ElementwiseNativeSupport.isDenseView(call.inputTensors().get(1))
                || !ElementwiseNativeSupport.isDenseView(call.inputTensors().get(2))
                || !ElementwiseNativeSupport.isDenseView(call.outputTensor())) {
            return "native-kernel-ineligible:where-layout";
        }
        return "";
    }

    private static boolean supportsNativeWhereDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private record WhereStorageLayout(int[] shape, int[][] cursorStrides, int[] cursorBaseOffsets) {
        static WhereStorageLayout from(
                CpuStorageView condition,
                CpuStorageView ifTrue,
                CpuStorageView ifFalse,
                CpuStorageView output
        ) {
            int[] outputShape = output.shape();
            return new WhereStorageLayout(
                    outputShape,
                    new int[][]{
                            output.strides(),
                            ElementwiseLayoutPlan.broadcastStrides(condition.shape(), condition.strides(), outputShape),
                            ElementwiseLayoutPlan.broadcastStrides(ifTrue.shape(), ifTrue.strides(), outputShape),
                            ElementwiseLayoutPlan.broadcastStrides(ifFalse.shape(), ifFalse.strides(), outputShape)
                    },
                    new int[]{
                            output.storageOffset(),
                            condition.storageOffset(),
                            ifTrue.storageOffset(),
                            ifFalse.storageOffset()
                    }
            );
        }

        private WhereStorageLayout {
            shape = shape.clone();
            cursorStrides = cursorStrides.clone();
            cursorBaseOffsets = cursorBaseOffsets.clone();
        }

        ElementwiseOffsetCursor cursor(int start) {
            return new ElementwiseOffsetCursor(shape, cursorStrides, cursorBaseOffsets, start);
        }
    }
}
