package backend.cpu1.kernels.loss.crossentropy;

import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedCrossEntropyLossUnit;
import runtime.contract.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.loss.LossReduction;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class Cpu1CrossEntropyLossIndicesLoops {
    private Cpu1CrossEntropyLossIndicesLoops() {
    }

    public static void runF32I32DenseArray(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        runF32(unit, context, logits.float32Array(), targets.int32Array(), null, logits.storageOffset(),
                targets.storageOffset(), output);
    }

    public static void runF32I64DenseArray(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        runF32(unit, context, logits.float32Array(), null, targets.int64Array(), logits.storageOffset(),
                targets.storageOffset(), output);
    }

    public static void runF64I32DenseArray(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        runF64(unit, context, logits.float64Array(), targets.int32Array(), null, logits.storageOffset(),
                targets.storageOffset(), output);
    }

    public static void runF64I64DenseArray(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        runF64(unit, context, logits.float64Array(), null, targets.int64Array(), logits.storageOffset(),
                targets.storageOffset(), output);
    }

    public static void runBf16I32DenseArray(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        runBf16(unit, context, logits.bfloat16Array(), targets.int32Array(), null, logits.storageOffset(),
                targets.storageOffset(), output);
    }

    public static void runBf16I64DenseArray(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputArrayView(unit.targetsNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        runBf16(unit, context, logits.bfloat16Array(), null, targets.int64Array(), logits.storageOffset(),
                targets.storageOffset(), output);
    }

    public static void runF32I32DenseSegment(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        runF32Segment(unit, context, logits.segment(), targets.segment(), false, logits.storageOffset(),
                targets.storageOffset(), output, nativeOutput);
    }

    public static void runF32I64DenseSegment(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        runF32Segment(unit, context, logits.segment(), targets.segment(), true, logits.storageOffset(),
                targets.storageOffset(), output, nativeOutput);
    }

    public static void runF64I32DenseSegment(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        runF64Segment(unit, context, logits.segment(), targets.segment(), false, logits.storageOffset(),
                targets.storageOffset(), output, nativeOutput);
    }

    public static void runF64I64DenseSegment(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        runF64Segment(unit, context, logits.segment(), targets.segment(), true, logits.storageOffset(),
                targets.storageOffset(), output, nativeOutput);
    }

    public static void runBf16I32DenseSegment(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        runBf16Segment(unit, context, logits.segment(), targets.segment(), false, logits.storageOffset(),
                targets.storageOffset(), output, nativeOutput);
    }

    public static void runBf16I64DenseSegment(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Cpu1TensorView logits = inputSegmentView(unit.logitsNodeId(), context);
        Cpu1TensorView targets = inputSegmentView(unit.targetsNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        runBf16Segment(unit, context, logits.segment(), targets.segment(), true, logits.storageOffset(),
                targets.storageOffset(), output, nativeOutput);
    }

    private static void runF32(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            float[] logits,
            int[] targetsI32,
            long[] targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset,
            Cpu1TensorView output
    ) {
        if (unit.reduction() == LossReduction.NONE) {
            float[] out = output.float32Array();
            int outBase = output.storageOffset();
            unit.launchPolicy().launch(unit.groupCount(), (start, end) -> {
                for (int group = start; group < end; group++) {
                    SampleLoss sample = sampleLossF32(
                            unit,
                            logits,
                            logitsBaseOffset,
                            targetIndex(targetsI32, targetsI64, targetsBaseOffset, group),
                            group
                    );
                    out[outBase + group] = sample.valid() ? (float) sample.loss() : 0.0f;
                }
            });
            markOutputWritten(unit, output, context);
            return;
        }
        ReductionResult result = reduceF32(unit, context, logits, targetsI32, targetsI64, logitsBaseOffset,
                targetsBaseOffset);
        output.float32Array()[output.storageOffset()] = (float) finalizeReduction(unit, result);
        markOutputWritten(unit, output, context);
    }

    private static void runF64(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            double[] logits,
            int[] targetsI32,
            long[] targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset,
            Cpu1TensorView output
    ) {
        if (unit.reduction() == LossReduction.NONE) {
            double[] out = output.float64Array();
            int outBase = output.storageOffset();
            unit.launchPolicy().launch(unit.groupCount(), (start, end) -> {
                for (int group = start; group < end; group++) {
                    SampleLoss sample = sampleLossF64(
                            unit,
                            logits,
                            logitsBaseOffset,
                            targetIndex(targetsI32, targetsI64, targetsBaseOffset, group),
                            group
                    );
                    out[outBase + group] = sample.valid() ? sample.loss() : 0.0d;
                }
            });
            markOutputWritten(unit, output, context);
            return;
        }
        ReductionResult result = reduceF64(unit, context, logits, targetsI32, targetsI64, logitsBaseOffset,
                targetsBaseOffset);
        output.float64Array()[output.storageOffset()] = finalizeReduction(unit, result);
        markOutputWritten(unit, output, context);
    }

    private static void runBf16(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            short[] logits,
            int[] targetsI32,
            long[] targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset,
            Cpu1TensorView output
    ) {
        if (unit.reduction() == LossReduction.NONE) {
            short[] out = output.bfloat16Array();
            int outBase = output.storageOffset();
            unit.launchPolicy().launch(unit.groupCount(), (start, end) -> {
                for (int group = start; group < end; group++) {
                    SampleLoss sample = sampleLossBf16(
                            unit,
                            logits,
                            logitsBaseOffset,
                            targetIndex(targetsI32, targetsI64, targetsBaseOffset, group),
                            group
                    );
                    out[outBase + group] = TensorDTypeOps.toBFloat16Bits(sample.valid()
                            ? (float) sample.loss()
                            : 0.0f);
                }
            });
            markOutputWritten(unit, output, context);
            return;
        }
        ReductionResult result = reduceBf16(unit, context, logits, targetsI32, targetsI64, logitsBaseOffset,
                targetsBaseOffset);
        output.bfloat16Array()[output.storageOffset()] = TensorDTypeOps.toBFloat16Bits(
                (float) finalizeReduction(unit, result)
        );
        markOutputWritten(unit, output, context);
    }

    private static void runF32Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            MemorySegment logits,
            MemorySegment targets,
            boolean targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset,
            Cpu1TensorView output,
            NativeTensorStorage nativeOutput
    ) {
        MemorySegment out = output.segment();
        if (unit.reduction() == LossReduction.NONE) {
            int outBase = output.storageOffset();
            unit.launchPolicy().launch(unit.groupCount(), (start, end) -> {
                for (int group = start; group < end; group++) {
                    SampleLoss sample = sampleLossF32Segment(
                            unit,
                            logits,
                            logitsBaseOffset,
                            targetIndexSegment(targets, targetsI64, targetsBaseOffset, group),
                            group
                    );
                    out.set(JAVA_FLOAT, (long) (outBase + group) * Float.BYTES,
                            sample.valid() ? (float) sample.loss() : 0.0f);
                }
            });
            markNativeOutputWritten(unit, nativeOutput, context);
            return;
        }
        ReductionResult result = reduceF32Segment(unit, context, logits, targets, targetsI64, logitsBaseOffset,
                targetsBaseOffset);
        out.set(JAVA_FLOAT, (long) output.storageOffset() * Float.BYTES, (float) finalizeReduction(unit, result));
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void runF64Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            MemorySegment logits,
            MemorySegment targets,
            boolean targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset,
            Cpu1TensorView output,
            NativeTensorStorage nativeOutput
    ) {
        MemorySegment out = output.segment();
        if (unit.reduction() == LossReduction.NONE) {
            int outBase = output.storageOffset();
            unit.launchPolicy().launch(unit.groupCount(), (start, end) -> {
                for (int group = start; group < end; group++) {
                    SampleLoss sample = sampleLossF64Segment(
                            unit,
                            logits,
                            logitsBaseOffset,
                            targetIndexSegment(targets, targetsI64, targetsBaseOffset, group),
                            group
                    );
                    out.set(JAVA_DOUBLE, (long) (outBase + group) * Double.BYTES,
                            sample.valid() ? sample.loss() : 0.0d);
                }
            });
            markNativeOutputWritten(unit, nativeOutput, context);
            return;
        }
        ReductionResult result = reduceF64Segment(unit, context, logits, targets, targetsI64, logitsBaseOffset,
                targetsBaseOffset);
        out.set(JAVA_DOUBLE, (long) output.storageOffset() * Double.BYTES, finalizeReduction(unit, result));
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void runBf16Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            MemorySegment logits,
            MemorySegment targets,
            boolean targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset,
            Cpu1TensorView output,
            NativeTensorStorage nativeOutput
    ) {
        MemorySegment out = output.segment();
        if (unit.reduction() == LossReduction.NONE) {
            int outBase = output.storageOffset();
            unit.launchPolicy().launch(unit.groupCount(), (start, end) -> {
                for (int group = start; group < end; group++) {
                    SampleLoss sample = sampleLossBf16Segment(
                            unit,
                            logits,
                            logitsBaseOffset,
                            targetIndexSegment(targets, targetsI64, targetsBaseOffset, group),
                            group
                    );
                    short value = TensorDTypeOps.toBFloat16Bits(sample.valid() ? (float) sample.loss() : 0.0f);
                    out.set(JAVA_SHORT, (long) (outBase + group) * Short.BYTES, value);
                }
            });
            markNativeOutputWritten(unit, nativeOutput, context);
            return;
        }
        ReductionResult result = reduceBf16Segment(unit, context, logits, targets, targetsI64, logitsBaseOffset,
                targetsBaseOffset);
        out.set(
                JAVA_SHORT,
                (long) output.storageOffset() * Short.BYTES,
                TensorDTypeOps.toBFloat16Bits((float) finalizeReduction(unit, result))
        );
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static ReductionResult reduceF32(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            float[] logits,
            int[] targetsI32,
            long[] targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            double total = 0.0d;
            int valid = 0;
            for (int group = 0; group < unit.groupCount(); group++) {
                SampleLoss sample = sampleLossF32(unit, logits, logitsBaseOffset,
                        targetIndex(targetsI32, targetsI64, targetsBaseOffset, group), group);
                if (sample.valid()) {
                    total += sample.loss();
                    valid++;
                }
            }
            return new ReductionResult(total, valid);
        }
        return reduceParallel(unit, context, (group) -> sampleLossF32(unit, logits, logitsBaseOffset,
                targetIndex(targetsI32, targetsI64, targetsBaseOffset, group), group));
    }

    private static ReductionResult reduceF64(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            double[] logits,
            int[] targetsI32,
            long[] targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            double total = 0.0d;
            int valid = 0;
            for (int group = 0; group < unit.groupCount(); group++) {
                SampleLoss sample = sampleLossF64(unit, logits, logitsBaseOffset,
                        targetIndex(targetsI32, targetsI64, targetsBaseOffset, group), group);
                if (sample.valid()) {
                    total += sample.loss();
                    valid++;
                }
            }
            return new ReductionResult(total, valid);
        }
        return reduceParallel(unit, context, (group) -> sampleLossF64(unit, logits, logitsBaseOffset,
                targetIndex(targetsI32, targetsI64, targetsBaseOffset, group), group));
    }

    private static ReductionResult reduceBf16(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            short[] logits,
            int[] targetsI32,
            long[] targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            double total = 0.0d;
            int valid = 0;
            for (int group = 0; group < unit.groupCount(); group++) {
                SampleLoss sample = sampleLossBf16(unit, logits, logitsBaseOffset,
                        targetIndex(targetsI32, targetsI64, targetsBaseOffset, group), group);
                if (sample.valid()) {
                    total += sample.loss();
                    valid++;
                }
            }
            return new ReductionResult(total, valid);
        }
        return reduceParallel(unit, context, (group) -> sampleLossBf16(unit, logits, logitsBaseOffset,
                targetIndex(targetsI32, targetsI64, targetsBaseOffset, group), group));
    }

    private static ReductionResult reduceF32Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            MemorySegment logits,
            MemorySegment targets,
            boolean targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            double total = 0.0d;
            int valid = 0;
            for (int group = 0; group < unit.groupCount(); group++) {
                SampleLoss sample = sampleLossF32Segment(unit, logits, logitsBaseOffset,
                        targetIndexSegment(targets, targetsI64, targetsBaseOffset, group), group);
                if (sample.valid()) {
                    total += sample.loss();
                    valid++;
                }
            }
            return new ReductionResult(total, valid);
        }
        return reduceParallel(unit, context, (group) -> sampleLossF32Segment(unit, logits, logitsBaseOffset,
                targetIndexSegment(targets, targetsI64, targetsBaseOffset, group), group));
    }

    private static ReductionResult reduceF64Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            MemorySegment logits,
            MemorySegment targets,
            boolean targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            double total = 0.0d;
            int valid = 0;
            for (int group = 0; group < unit.groupCount(); group++) {
                SampleLoss sample = sampleLossF64Segment(unit, logits, logitsBaseOffset,
                        targetIndexSegment(targets, targetsI64, targetsBaseOffset, group), group);
                if (sample.valid()) {
                    total += sample.loss();
                    valid++;
                }
            }
            return new ReductionResult(total, valid);
        }
        return reduceParallel(unit, context, (group) -> sampleLossF64Segment(unit, logits, logitsBaseOffset,
                targetIndexSegment(targets, targetsI64, targetsBaseOffset, group), group));
    }

    private static ReductionResult reduceBf16Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            MemorySegment logits,
            MemorySegment targets,
            boolean targetsI64,
            int logitsBaseOffset,
            int targetsBaseOffset
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            double total = 0.0d;
            int valid = 0;
            for (int group = 0; group < unit.groupCount(); group++) {
                SampleLoss sample = sampleLossBf16Segment(unit, logits, logitsBaseOffset,
                        targetIndexSegment(targets, targetsI64, targetsBaseOffset, group), group);
                if (sample.valid()) {
                    total += sample.loss();
                    valid++;
                }
            }
            return new ReductionResult(total, valid);
        }
        return reduceParallel(unit, context, (group) -> sampleLossBf16Segment(unit, logits, logitsBaseOffset,
                targetIndexSegment(targets, targetsI64, targetsBaseOffset, group), group));
    }

    private static ReductionResult reduceParallel(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context,
            SampleComputer sampleComputer
    ) {
        int slotCount = Cpu1RangeLauncher.slotCount(unit.groupCount(), unit.launchConfig());
        Cpu1ScratchBuffer scratchBuffer = context.cpu1ScratchBufferForNodeId(unit.nodeId());
        if (scratchBuffer == null) {
            throw new IllegalStateException("cpu1 CROSS_ENTROPY_LOSS_INDICES parallel nodeId=" + unit.nodeId()
                    + " requires prepared F64/I32 scratch buffers.");
        }
        double[] partialLosses = scratchBuffer.requireF64Array(slotCount);
        int[] partialValid = scratchBuffer.requireI32Array(slotCount);
        Cpu1RangeLauncher.launchIndexed(unit.groupCount(), unit.launchConfig(), (slot, start, end) -> {
            double total = 0.0d;
            int valid = 0;
            for (int group = start; group < end; group++) {
                SampleLoss sample = sampleComputer.compute(group);
                if (sample.valid()) {
                    total += sample.loss();
                    valid++;
                }
            }
            partialLosses[slot] = total;
            partialValid[slot] = valid;
        });
        double total = 0.0d;
        int valid = 0;
        for (int slot = 0; slot < slotCount; slot++) {
            total += partialLosses[slot];
            valid += partialValid[slot];
        }
        return new ReductionResult(total, valid);
    }

    private static SampleLoss sampleLossF32(
            Cpu1PreparedCrossEntropyLossUnit unit,
            float[] logits,
            int logitsBaseOffset,
            int targetIndex,
            int group
    ) {
        if (unit.hasIgnoreIndex() && targetIndex == unit.ignoreIndex()) {
            return SampleLoss.ignored();
        }
        validateTargetIndex(targetIndex, unit.axisSize());
        int base = logitsBaseOffset + baseLogitsOffset(unit, group);
        float targetLogit = logits[base + targetIndex * unit.axisStride()];
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, logits[offset]);
        }
        double sumExp = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            sumExp += Math.exp(logits[offset] - max);
        }
        return new SampleLoss(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleLoss sampleLossF64(
            Cpu1PreparedCrossEntropyLossUnit unit,
            double[] logits,
            int logitsBaseOffset,
            int targetIndex,
            int group
    ) {
        if (unit.hasIgnoreIndex() && targetIndex == unit.ignoreIndex()) {
            return SampleLoss.ignored();
        }
        validateTargetIndex(targetIndex, unit.axisSize());
        int base = logitsBaseOffset + baseLogitsOffset(unit, group);
        double targetLogit = logits[base + targetIndex * unit.axisStride()];
        double max = Double.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, logits[offset]);
        }
        double sumExp = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            sumExp += Math.exp(logits[offset] - max);
        }
        return new SampleLoss(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleLoss sampleLossBf16(
            Cpu1PreparedCrossEntropyLossUnit unit,
            short[] logits,
            int logitsBaseOffset,
            int targetIndex,
            int group
    ) {
        if (unit.hasIgnoreIndex() && targetIndex == unit.ignoreIndex()) {
            return SampleLoss.ignored();
        }
        validateTargetIndex(targetIndex, unit.axisSize());
        int base = logitsBaseOffset + baseLogitsOffset(unit, group);
        float targetLogit = TensorDTypeOps.fromBFloat16Bits(logits[base + targetIndex * unit.axisStride()]);
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(logits[offset]));
        }
        double sumExp = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            sumExp += Math.exp(TensorDTypeOps.fromBFloat16Bits(logits[offset]) - max);
        }
        return new SampleLoss(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleLoss sampleLossF32Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            MemorySegment logits,
            int logitsBaseOffset,
            int targetIndex,
            int group
    ) {
        if (unit.hasIgnoreIndex() && targetIndex == unit.ignoreIndex()) {
            return SampleLoss.ignored();
        }
        validateTargetIndex(targetIndex, unit.axisSize());
        int base = logitsBaseOffset + baseLogitsOffset(unit, group);
        int targetOffset = base + targetIndex * unit.axisStride();
        float targetLogit = logits.get(JAVA_FLOAT, (long) targetOffset * Float.BYTES);
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, logits.get(JAVA_FLOAT, (long) offset * Float.BYTES));
        }
        double sumExp = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            sumExp += Math.exp(logits.get(JAVA_FLOAT, (long) offset * Float.BYTES) - max);
        }
        return new SampleLoss(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleLoss sampleLossF64Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            MemorySegment logits,
            int logitsBaseOffset,
            int targetIndex,
            int group
    ) {
        if (unit.hasIgnoreIndex() && targetIndex == unit.ignoreIndex()) {
            return SampleLoss.ignored();
        }
        validateTargetIndex(targetIndex, unit.axisSize());
        int base = logitsBaseOffset + baseLogitsOffset(unit, group);
        int targetOffset = base + targetIndex * unit.axisStride();
        double targetLogit = logits.get(JAVA_DOUBLE, (long) targetOffset * Double.BYTES);
        double max = Double.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, logits.get(JAVA_DOUBLE, (long) offset * Double.BYTES));
        }
        double sumExp = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            sumExp += Math.exp(logits.get(JAVA_DOUBLE, (long) offset * Double.BYTES) - max);
        }
        return new SampleLoss(max + Math.log(sumExp) - targetLogit, true);
    }

    private static SampleLoss sampleLossBf16Segment(
            Cpu1PreparedCrossEntropyLossUnit unit,
            MemorySegment logits,
            int logitsBaseOffset,
            int targetIndex,
            int group
    ) {
        if (unit.hasIgnoreIndex() && targetIndex == unit.ignoreIndex()) {
            return SampleLoss.ignored();
        }
        validateTargetIndex(targetIndex, unit.axisSize());
        int base = logitsBaseOffset + baseLogitsOffset(unit, group);
        int targetOffset = base + targetIndex * unit.axisStride();
        float targetLogit = TensorDTypeOps.fromBFloat16Bits(
                logits.get(JAVA_SHORT, (long) targetOffset * Short.BYTES)
        );
        float max = Float.NEGATIVE_INFINITY;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            max = Math.max(max, TensorDTypeOps.fromBFloat16Bits(
                    logits.get(JAVA_SHORT, (long) offset * Short.BYTES)
            ));
        }
        double sumExp = 0.0d;
        for (int index = 0, offset = base; index < unit.axisSize(); index++, offset += unit.axisStride()) {
            sumExp += Math.exp(TensorDTypeOps.fromBFloat16Bits(
                    logits.get(JAVA_SHORT, (long) offset * Short.BYTES)
            ) - max);
        }
        return new SampleLoss(max + Math.log(sumExp) - targetLogit, true);
    }

    private static int baseLogitsOffset(Cpu1PreparedCrossEntropyLossUnit unit, int group) {
        int[] logitsShape = unit.logitsShape();
        int[] targetShape = unit.targetShape();
        int remaining = group;
        int offset = 0;
        for (int dim = 0, targetDim = 0; dim < logitsShape.length; dim++) {
            if (dim == unit.classAxis()) {
                continue;
            }
            int stride = denseStride(logitsShape, dim);
            int coord = targetShape.length == 0 ? 0 : remaining / denseStride(targetShape, targetDim);
            if (targetShape.length > 0) {
                remaining %= denseStride(targetShape, targetDim);
            }
            offset += coord * stride;
            targetDim++;
        }
        return offset;
    }

    private static int denseStride(int[] shape, int dim) {
        int stride = 1;
        for (int index = dim + 1; index < shape.length; index++) {
            stride = Math.multiplyExact(stride, shape[index]);
        }
        return stride;
    }

    private static int targetIndex(int[] targetsI32, long[] targetsI64, int baseOffset, int group) {
        if (targetsI32 != null) {
            return targetsI32[baseOffset + group];
        }
        return Math.toIntExact(targetsI64[baseOffset + group]);
    }

    private static int targetIndexSegment(MemorySegment targets, boolean targetsI64, int baseOffset, int group) {
        int offset = baseOffset + group;
        if (targetsI64) {
            return Math.toIntExact(targets.get(JAVA_LONG, (long) offset * Long.BYTES));
        }
        return targets.get(JAVA_INT, (long) offset * Integer.BYTES);
    }

    private static double finalizeReduction(Cpu1PreparedCrossEntropyLossUnit unit, ReductionResult result) {
        return switch (unit.reduction()) {
            case NONE -> throw new IllegalStateException("NONE reduction does not produce scalar output.");
            case SUM -> result.totalLoss();
            case MEAN -> result.validCount() == 0 ? 0.0d : result.totalLoss() / result.validCount();
        };
    }

    private static void validateTargetIndex(int targetIndex, int axisSize) {
        if (targetIndex < 0 || targetIndex >= axisSize) {
            throw new IllegalArgumentException("Target index out of range: " + targetIndex + " for classes=" + axisSize);
        }
    }

    private static Cpu1TensorView inputArrayView(int inputNodeId, ExecutionContext context) {
        context.requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(inputNodeId);
        return Cpu1TensorView.fromTensor(input);
    }

    private static Cpu1TensorView inputSegmentView(int inputNodeId, ExecutionContext context) {
        NativeTensorStorage nativeInput = context.requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(inputNodeId);
        return Cpu1TensorView.fromNativeStorage(input, nativeInput);
    }

    private static Cpu1TensorView outputArrayView(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static NativeTensorStorage outputSegmentStorage(
            Cpu1PreparedCrossEntropyLossUnit unit,
            ExecutionContext context
    ) {
        int outputElements = unit.reduction() == LossReduction.NONE ? unit.groupCount() : 1;
        return context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.logitsDataType(),
                outputElements,
                "cpu1-node-" + unit.nodeId() + ":cross-entropy-indices-native-segment"
        );
    }

    private static void markOutputWritten(
            Cpu1PreparedCrossEntropyLossUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 CROSS_ENTROPY_LOSS_INDICES CPU array");
    }

    private static void markNativeOutputWritten(
            Cpu1PreparedCrossEntropyLossUnit unit,
            NativeTensorStorage nativeOutput,
            ExecutionContext context
    ) {
        nativeOutput.markModified();
        context.attachNativeStorage(
                unit.nodeId(),
                nativeOutput,
                "cpu1 CROSS_ENTROPY_LOSS_INDICES native CPU segment"
        );
    }

    @FunctionalInterface
    private interface SampleComputer {
        SampleLoss compute(int group);
    }

    private record SampleLoss(double loss, boolean valid) {
        private static SampleLoss ignored() {
            return new SampleLoss(0.0d, false);
        }
    }

    private record ReductionResult(double totalLoss, int validCount) {
    }
}
