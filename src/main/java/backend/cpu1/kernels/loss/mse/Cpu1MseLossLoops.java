package backend.cpu1.kernels.loss.mse;

import backend.cpu1.exec.Cpu1Workspace;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedMseLossUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

public final class Cpu1MseLossLoops {
    private Cpu1MseLossLoops() {
    }

    public static void sumF32DenseScalar(Cpu1PreparedMseLossUnit unit, ExecutionContext context) {
        runF32(unit, context, false);
    }

    public static void meanF32DenseScalar(Cpu1PreparedMseLossUnit unit, ExecutionContext context) {
        runF32(unit, context, true);
    }

    public static void sumF64DenseScalar(Cpu1PreparedMseLossUnit unit, ExecutionContext context) {
        runF64(unit, context, false);
    }

    public static void meanF64DenseScalar(Cpu1PreparedMseLossUnit unit, ExecutionContext context) {
        runF64(unit, context, true);
    }

    public static void sumBf16DenseScalar(Cpu1PreparedMseLossUnit unit, ExecutionContext context) {
        runBf16(unit, context, false);
    }

    public static void meanBf16DenseScalar(Cpu1PreparedMseLossUnit unit, ExecutionContext context) {
        runBf16(unit, context, true);
    }

    private static void runF32(Cpu1PreparedMseLossUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            runF32Segment(unit, context, mean);
            return;
        }
        Cpu1TensorView prediction = inputArrayView(unit.predictionNodeId(), context);
        Cpu1TensorView target = inputArrayView(unit.targetNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        float[] predictionArray = prediction.float32Array();
        float[] targetArray = target.float32Array();
        float[] outputArray = output.float32Array();
        double sum = sumF32Array(unit, context, predictionArray, targetArray, prediction.storageOffset(), target.storageOffset());
        double value = mean ? sum / unit.elementCount() : sum;
        outputArray[output.storageOffset()] = (float) value;
        markOutputWritten(unit, output, context);
    }

    private static void runF64(Cpu1PreparedMseLossUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            runF64Segment(unit, context, mean);
            return;
        }
        Cpu1TensorView prediction = inputArrayView(unit.predictionNodeId(), context);
        Cpu1TensorView target = inputArrayView(unit.targetNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double[] predictionArray = prediction.float64Array();
        double[] targetArray = target.float64Array();
        double[] outputArray = output.float64Array();
        double sum = sumF64Array(unit, context, predictionArray, targetArray, prediction.storageOffset(), target.storageOffset());
        outputArray[output.storageOffset()] = mean ? sum / unit.elementCount() : sum;
        markOutputWritten(unit, output, context);
    }

    private static void runBf16(Cpu1PreparedMseLossUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() != Cpu1StorageKind.JAVA_ARRAY) {
            throw new UnsupportedOperationException("cpu1 MSE_LOSS BFLOAT16 supports JAVA_ARRAY storage only.");
        }
        Cpu1TensorView prediction = inputArrayView(unit.predictionNodeId(), context);
        Cpu1TensorView target = inputArrayView(unit.targetNodeId(), context);
        Cpu1TensorView output = outputArrayView(unit, context);
        short[] predictionArray = prediction.bfloat16Array();
        short[] targetArray = target.bfloat16Array();
        short[] outputArray = output.bfloat16Array();
        double sum = 0.0d;
        int predictionBase = prediction.storageOffset();
        int targetBase = target.storageOffset();
        for (int i = 0; i < unit.elementCount(); i++) {
            double left = TensorDTypeOps.fromBFloat16Bits(predictionArray[predictionBase + i]);
            double right = TensorDTypeOps.fromBFloat16Bits(targetArray[targetBase + i]);
            double diff = left - right;
            sum += diff * diff;
        }
        double value = mean ? sum / unit.elementCount() : sum;
        outputArray[output.storageOffset()] = TensorDTypeOps.toBFloat16Bits((float) value);
        markOutputWritten(unit, output, context);
    }

    private static void runF32Segment(Cpu1PreparedMseLossUnit unit, ExecutionContext context, boolean mean) {
        Cpu1TensorView prediction = inputSegmentView(unit.predictionNodeId(), context);
        Cpu1TensorView target = inputSegmentView(unit.targetNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.outputNodeId()), nativeOutput);
        double sum = sumF32Segment(
                unit,
                context,
                prediction.segment(),
                target.segment(),
                prediction.storageOffset(),
                target.storageOffset()
        );
        double value = mean ? sum / unit.elementCount() : sum;
        output.segment().set(JAVA_FLOAT, (long) output.storageOffset() * Float.BYTES, (float) value);
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void runF64Segment(Cpu1PreparedMseLossUnit unit, ExecutionContext context, boolean mean) {
        Cpu1TensorView prediction = inputSegmentView(unit.predictionNodeId(), context);
        Cpu1TensorView target = inputSegmentView(unit.targetNodeId(), context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.outputNodeId()), nativeOutput);
        double sum = sumF64Segment(
                unit,
                context,
                prediction.segment(),
                target.segment(),
                prediction.storageOffset(),
                target.storageOffset()
        );
        output.segment().set(JAVA_DOUBLE, (long) output.storageOffset() * Double.BYTES, mean ? sum / unit.elementCount() : sum);
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static double sumF32Array(
            Cpu1PreparedMseLossUnit unit,
            ExecutionContext context,
            float[] prediction,
            float[] target,
            int predictionBase,
            int targetBase
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            return sumF32ArrayRange(prediction, target, predictionBase, targetBase, 0, unit.elementCount());
        }
        int slotCount = Cpu1RangeLauncher.slotCount(unit.elementCount(), unit.launchConfig());
        double[] partialSums = partialSums(unit, context, slotCount);
        Cpu1RangeLauncher.launchIndexed(unit.elementCount(), unit.launchConfig(), (slotIndex, start, end) ->
                partialSums[slotIndex] = sumF32ArrayRange(prediction, target, predictionBase, targetBase, start, end));
        return sumPartialSums(partialSums, slotCount);
    }

    private static double sumF64Array(
            Cpu1PreparedMseLossUnit unit,
            ExecutionContext context,
            double[] prediction,
            double[] target,
            int predictionBase,
            int targetBase
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            return sumF64ArrayRange(prediction, target, predictionBase, targetBase, 0, unit.elementCount());
        }
        int slotCount = Cpu1RangeLauncher.slotCount(unit.elementCount(), unit.launchConfig());
        double[] partialSums = partialSums(unit, context, slotCount);
        Cpu1RangeLauncher.launchIndexed(unit.elementCount(), unit.launchConfig(), (slotIndex, start, end) ->
                partialSums[slotIndex] = sumF64ArrayRange(prediction, target, predictionBase, targetBase, start, end));
        return sumPartialSums(partialSums, slotCount);
    }

    private static double sumF32Segment(
            Cpu1PreparedMseLossUnit unit,
            ExecutionContext context,
            MemorySegment prediction,
            MemorySegment target,
            int predictionBase,
            int targetBase
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            return sumF32SegmentRange(prediction, target, predictionBase, targetBase, 0, unit.elementCount());
        }
        int slotCount = Cpu1RangeLauncher.slotCount(unit.elementCount(), unit.launchConfig());
        double[] partialSums = partialSums(unit, context, slotCount);
        Cpu1RangeLauncher.launchIndexed(unit.elementCount(), unit.launchConfig(), (slotIndex, start, end) ->
                partialSums[slotIndex] = sumF32SegmentRange(prediction, target, predictionBase, targetBase, start, end));
        return sumPartialSums(partialSums, slotCount);
    }

    private static double sumF64Segment(
            Cpu1PreparedMseLossUnit unit,
            ExecutionContext context,
            MemorySegment prediction,
            MemorySegment target,
            int predictionBase,
            int targetBase
    ) {
        if (unit.launchConfig().workerCount() == 1) {
            return sumF64SegmentRange(prediction, target, predictionBase, targetBase, 0, unit.elementCount());
        }
        int slotCount = Cpu1RangeLauncher.slotCount(unit.elementCount(), unit.launchConfig());
        double[] partialSums = partialSums(unit, context, slotCount);
        Cpu1RangeLauncher.launchIndexed(unit.elementCount(), unit.launchConfig(), (slotIndex, start, end) ->
                partialSums[slotIndex] = sumF64SegmentRange(prediction, target, predictionBase, targetBase, start, end));
        return sumPartialSums(partialSums, slotCount);
    }

    private static double sumF32ArrayRange(
            float[] prediction,
            float[] target,
            int predictionBase,
            int targetBase,
            int start,
            int end
    ) {
        double sum = 0.0d;
        for (int i = start; i < end; i++) {
            double diff = (double) prediction[predictionBase + i] - (double) target[targetBase + i];
            sum += diff * diff;
        }
        return sum;
    }

    private static double sumF64ArrayRange(
            double[] prediction,
            double[] target,
            int predictionBase,
            int targetBase,
            int start,
            int end
    ) {
        double sum = 0.0d;
        for (int i = start; i < end; i++) {
            double diff = prediction[predictionBase + i] - target[targetBase + i];
            sum += diff * diff;
        }
        return sum;
    }

    private static double sumF32SegmentRange(
            MemorySegment prediction,
            MemorySegment target,
            int predictionBase,
            int targetBase,
            int start,
            int end
    ) {
        double sum = 0.0d;
        for (int i = start; i < end; i++) {
            double left = prediction.get(JAVA_FLOAT, (long) (predictionBase + i) * Float.BYTES);
            double right = target.get(JAVA_FLOAT, (long) (targetBase + i) * Float.BYTES);
            double diff = left - right;
            sum += diff * diff;
        }
        return sum;
    }

    private static double sumF64SegmentRange(
            MemorySegment prediction,
            MemorySegment target,
            int predictionBase,
            int targetBase,
            int start,
            int end
    ) {
        double sum = 0.0d;
        for (int i = start; i < end; i++) {
            double diff = prediction.get(JAVA_DOUBLE, (long) (predictionBase + i) * Double.BYTES)
                    - target.get(JAVA_DOUBLE, (long) (targetBase + i) * Double.BYTES);
            sum += diff * diff;
        }
        return sum;
    }

    private static double sumPartialSums(double[] partialSums, int slotCount) {
        double sum = 0.0d;
        for (int slot = 0; slot < slotCount; slot++) {
            sum += partialSums[slot];
        }
        return sum;
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

    private static Cpu1TensorView outputArrayView(Cpu1PreparedMseLossUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.outputNodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static NativeTensorStorage outputSegmentStorage(Cpu1PreparedMseLossUnit unit, ExecutionContext context) {
        return context.requireNativeOutputStorage(
                unit.outputNodeId(),
                unit.dataType(),
                1,
                "cpu1-node-" + unit.outputNodeId() + ":mse-loss-native-segment"
        );
    }

    private static double[] partialSums(Cpu1PreparedMseLossUnit unit, ExecutionContext context, int slotCount) {
        Cpu1Workspace workspace = context.cpu1WorkspaceForNodeId(unit.outputNodeId());
        if (workspace == null) {
            throw new IllegalStateException("cpu1 MSE_LOSS parallel nodeId=" + unit.outputNodeId()
                    + " requires prepared F64 partial-sum workspace.");
        }
        return workspace.requireF64Array(slotCount);
    }

    private static void markOutputWritten(
            Cpu1PreparedMseLossUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.outputNodeId(), "cpu1 MSE_LOSS specialized CPU array");
    }

    private static void markNativeOutputWritten(
            Cpu1PreparedMseLossUnit unit,
            NativeTensorStorage nativeOutput,
            ExecutionContext context
    ) {
        nativeOutput.markModified();
        context.attachNativeStorage(unit.outputNodeId(), nativeOutput, "cpu1 MSE_LOSS specialized native CPU segment");
    }
}
