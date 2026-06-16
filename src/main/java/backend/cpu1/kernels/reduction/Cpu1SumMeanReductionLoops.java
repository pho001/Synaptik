package backend.cpu1.kernels.reduction;

import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Dense scalar SUM/MEAN reductions for the first cpu1 reduction phase.
 */
public final class Cpu1SumMeanReductionLoops {
    private Cpu1SumMeanReductionLoops() {
    }

    public static void sumF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32(unit, context, false);
    }

    public static void sumF32StridedScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32Strided(unit, context, false);
    }

    public static void meanF32DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32(unit, context, true);
    }

    public static void meanF32StridedScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF32Strided(unit, context, true);
    }

    public static void sumF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64(unit, context, false);
    }

    public static void sumF64StridedScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64Strided(unit, context, false);
    }

    public static void meanF64DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64(unit, context, true);
    }

    public static void meanF64StridedScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceF64Strided(unit, context, true);
    }

    public static void sumBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBf16(unit, context, false);
    }

    public static void meanBf16DenseScalar(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        reduceBf16(unit, context, true);
    }

    private static void reduceF32(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF32Segment(unit, context, mean);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        float[] inputArray = input.float32Array();
        float[] outputArray = output.float32Array();
        int reduction = unit.axisSize();
        int outputWorkItems = outputWorkItems(unit);
        if (shouldUseScalarPartialReduction(unit, outputWorkItems)) {
            double sum = partialSumF32(unit, context, inputArray, input.storageOffset(), reduction);
            outputArray[output.storageOffset()] = (float) (mean ? sum / reduction : sum);
            markOutputWritten(unit, output, context);
            return;
        }
        if (shouldUseOutputParallel(unit, outputWorkItems)) {
            Cpu1RangeLauncher.launch(outputWorkItems, unit.launchConfig(), (start, end) -> {
                for (int work = start; work < end; work++) {
                    int outer = work / unit.innerSize();
                    int inner = work - outer * unit.innerSize();
                    int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
                    int outputOffset = output.storageOffset() + outer * unit.innerSize() + inner;
                    float sum = 0.0f;
                    for (int index = 0; index < reduction; index++) {
                        sum += inputArray[inputOuterBase + index * unit.innerSize() + inner];
                    }
                    outputArray[outputOffset] = mean ? sum / reduction : sum;
                }
            });
            markOutputWritten(unit, output, context);
            return;
        }
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    sum += inputArray[inputOuterBase + index * unit.innerSize() + inner];
                }
                outputArray[outputOuterBase + inner] = mean ? sum / reduction : sum;
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceF64(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF64Segment(unit, context, mean);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double[] inputArray = input.float64Array();
        double[] outputArray = output.float64Array();
        int reduction = unit.axisSize();
        int outputWorkItems = outputWorkItems(unit);
        if (shouldUseScalarPartialReduction(unit, outputWorkItems)) {
            double sum = partialSumF64(unit, context, inputArray, input.storageOffset(), reduction);
            outputArray[output.storageOffset()] = mean ? sum / reduction : sum;
            markOutputWritten(unit, output, context);
            return;
        }
        if (shouldUseOutputParallel(unit, outputWorkItems)) {
            Cpu1RangeLauncher.launch(outputWorkItems, unit.launchConfig(), (start, end) -> {
                for (int work = start; work < end; work++) {
                    int outer = work / unit.innerSize();
                    int inner = work - outer * unit.innerSize();
                    int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
                    int outputOffset = output.storageOffset() + outer * unit.innerSize() + inner;
                    double sum = 0.0d;
                    for (int index = 0; index < reduction; index++) {
                        sum += inputArray[inputOuterBase + index * unit.innerSize() + inner];
                    }
                    outputArray[outputOffset] = mean ? sum / reduction : sum;
                }
            });
            markOutputWritten(unit, output, context);
            return;
        }
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                double sum = 0.0d;
                for (int index = 0; index < reduction; index++) {
                    sum += inputArray[inputOuterBase + index * unit.innerSize() + inner];
                }
                outputArray[outputOuterBase + inner] = mean ? sum / reduction : sum;
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceBf16(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceBf16Segment(unit, context, mean);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        short[] inputArray = input.bfloat16Array();
        short[] outputArray = output.bfloat16Array();
        int reduction = unit.axisSize();
        int outputWorkItems = outputWorkItems(unit);
        if (shouldUseScalarPartialReduction(unit, outputWorkItems)) {
            double sum = partialSumBf16(unit, context, inputArray, input.storageOffset(), reduction);
            double value = mean ? sum / reduction : sum;
            outputArray[output.storageOffset()] = TensorDTypeOps.toBFloat16Bits((float) value);
            markOutputWritten(unit, output, context);
            return;
        }
        if (shouldUseOutputParallel(unit, outputWorkItems)) {
            Cpu1RangeLauncher.launch(outputWorkItems, unit.launchConfig(), (start, end) -> {
                for (int work = start; work < end; work++) {
                    int outer = work / unit.innerSize();
                    int inner = work - outer * unit.innerSize();
                    int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
                    int outputOffset = output.storageOffset() + outer * unit.innerSize() + inner;
                    double sum = 0.0d;
                    for (int index = 0; index < reduction; index++) {
                        sum += TensorDTypeOps.fromBFloat16Bits(
                                inputArray[inputOuterBase + index * unit.innerSize() + inner]
                        );
                    }
                    double value = mean ? sum / reduction : sum;
                    outputArray[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                }
            });
            markOutputWritten(unit, output, context);
            return;
        }
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    sum += TensorDTypeOps.fromBFloat16Bits(inputArray[inputOuterBase + index * unit.innerSize() + inner]);
                }
                float value = mean ? sum / reduction : sum;
                outputArray[outputOuterBase + inner] = TensorDTypeOps.toBFloat16Bits(value);
            }
        }
        markOutputWritten(unit, output, context);
    }

    private static void reduceF32Strided(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF32StridedSegment(unit, context, mean);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        float[] inputArray = input.float32Array();
        float[] outputArray = output.float32Array();
        int[] shape = unit.inputAccessPlan().shape();
        int[] strides = unit.inputAccessPlan().strides();
        int reduction = unit.axisSize();
        int axisStride = strides[unit.axis()];
        int outputWorkItems = outputWorkItems(unit);
        Cpu1RangeLauncher.launch(outputWorkItems, unit.launchConfig(), (start, end) -> {
            for (int work = start; work < end; work++) {
                int outer = work / unit.innerSize();
                int inner = work - outer * unit.innerSize();
                int inputBase = stridedInputBaseOffset(unit, shape, strides, input.storageOffset(), outer, inner);
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    sum += inputArray[inputBase + index * axisStride];
                }
                outputArray[output.storageOffset() + work] = mean ? sum / reduction : sum;
            }
        });
        markOutputWritten(unit, output, context);
    }

    private static void reduceF64Strided(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            reduceF64StridedSegment(unit, context, mean);
            return;
        }
        Cpu1TensorView input = inputArrayView(unit, context);
        Cpu1TensorView output = outputArrayView(unit, context);
        double[] inputArray = input.float64Array();
        double[] outputArray = output.float64Array();
        int[] shape = unit.inputAccessPlan().shape();
        int[] strides = unit.inputAccessPlan().strides();
        int reduction = unit.axisSize();
        int axisStride = strides[unit.axis()];
        int outputWorkItems = outputWorkItems(unit);
        Cpu1RangeLauncher.launch(outputWorkItems, unit.launchConfig(), (start, end) -> {
            for (int work = start; work < end; work++) {
                int outer = work / unit.innerSize();
                int inner = work - outer * unit.innerSize();
                int inputBase = stridedInputBaseOffset(unit, shape, strides, input.storageOffset(), outer, inner);
                double sum = 0.0d;
                for (int index = 0; index < reduction; index++) {
                    sum += inputArray[inputBase + index * axisStride];
                }
                outputArray[output.storageOffset() + work] = mean ? sum / reduction : sum;
            }
        });
        markOutputWritten(unit, output, context);
    }

    private static void reduceF32Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    sum += inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                }
                int outputOffset = outputOuterBase + inner;
                outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, mean ? sum / reduction : sum);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceF64Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                double sum = 0.0d;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    sum += inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                }
                int outputOffset = outputOuterBase + inner;
                outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, mean ? sum / reduction : sum);
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceBf16Segment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(
                context.runtimeTensorForNodeId(unit.nodeId()),
                nativeOutput
        );
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int reduction = unit.axisSize();
        for (int outer = 0; outer < unit.outerSize(); outer++) {
            int inputOuterBase = input.storageOffset() + outer * reduction * unit.innerSize();
            int outputOuterBase = output.storageOffset() + outer * unit.innerSize();
            for (int inner = 0; inner < unit.innerSize(); inner++) {
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                    sum += TensorDTypeOps.fromBFloat16Bits(
                            inputSegment.get(JAVA_SHORT, (long) inputOffset * Short.BYTES)
                    );
                }
                int outputOffset = outputOuterBase + inner;
                float value = mean ? sum / reduction : sum;
                outputSegment.set(
                        JAVA_SHORT,
                        (long) outputOffset * Short.BYTES,
                        TensorDTypeOps.toBFloat16Bits(value)
                );
            }
        }
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceF32StridedSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(
                context.runtimeTensorForNodeId(unit.nodeId()),
                nativeOutput
        );
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int[] shape = unit.inputAccessPlan().shape();
        int[] strides = unit.inputAccessPlan().strides();
        int reduction = unit.axisSize();
        int axisStride = strides[unit.axis()];
        int outputWorkItems = outputWorkItems(unit);
        Cpu1RangeLauncher.launch(outputWorkItems, unit.launchConfig(), (start, end) -> {
            for (int work = start; work < end; work++) {
                int outer = work / unit.innerSize();
                int inner = work - outer * unit.innerSize();
                int inputBase = stridedInputBaseOffset(unit, shape, strides, input.storageOffset(), outer, inner);
                float sum = 0.0f;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputBase + index * axisStride;
                    sum += inputSegment.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                }
                int outputOffset = output.storageOffset() + work;
                outputSegment.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, mean ? sum / reduction : sum);
            }
        });
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static void reduceF64StridedSegment(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
        Cpu1TensorView input = inputSegmentView(unit, context);
        NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
        Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(
                context.runtimeTensorForNodeId(unit.nodeId()),
                nativeOutput
        );
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        int[] shape = unit.inputAccessPlan().shape();
        int[] strides = unit.inputAccessPlan().strides();
        int reduction = unit.axisSize();
        int axisStride = strides[unit.axis()];
        int outputWorkItems = outputWorkItems(unit);
        Cpu1RangeLauncher.launch(outputWorkItems, unit.launchConfig(), (start, end) -> {
            for (int work = start; work < end; work++) {
                int outer = work / unit.innerSize();
                int inner = work - outer * unit.innerSize();
                int inputBase = stridedInputBaseOffset(unit, shape, strides, input.storageOffset(), outer, inner);
                double sum = 0.0d;
                for (int index = 0; index < reduction; index++) {
                    int inputOffset = inputBase + index * axisStride;
                    sum += inputSegment.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                }
                int outputOffset = output.storageOffset() + work;
                outputSegment.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, mean ? sum / reduction : sum);
            }
        });
        markNativeOutputWritten(unit, nativeOutput, context);
    }

    private static int stridedInputBaseOffset(
            Cpu1PreparedReductionUnit unit,
            int[] shape,
            int[] strides,
            int storageOffset,
            int outer,
            int inner
    ) {
        int offset = storageOffset;
        int outerRemainder = outer;
        for (int dim = unit.axis() - 1; dim >= 0; dim--) {
            int coordinate = outerRemainder % shape[dim];
            outerRemainder /= shape[dim];
            offset += coordinate * strides[dim];
        }
        int innerRemainder = inner;
        for (int dim = shape.length - 1; dim > unit.axis(); dim--) {
            int coordinate = innerRemainder % shape[dim];
            innerRemainder /= shape[dim];
            offset += coordinate * strides[dim];
        }
        return offset;
    }

    private static int outputWorkItems(Cpu1PreparedReductionUnit unit) {
        return Math.multiplyExact(unit.outerSize(), unit.innerSize());
    }

    private static boolean shouldUseOutputParallel(Cpu1PreparedReductionUnit unit, int outputWorkItems) {
        return unit.launchConfig().workerCount() > 1 && outputWorkItems > 1;
    }

    private static boolean shouldUseScalarPartialReduction(Cpu1PreparedReductionUnit unit, int outputWorkItems) {
        return unit.launchConfig().workerCount() > 1
                && outputWorkItems <= 1
                && unit.axisSize() >= unit.launchConfig().workerCount();
    }

    private static double partialSumF32(
            Cpu1PreparedReductionUnit unit,
            ExecutionContext context,
            float[] inputArray,
            int inputBase,
            int reduction
    ) {
        int slotCount = Cpu1RangeLauncher.slotCount(reduction, unit.launchConfig());
        double[] partialSums = partialSums(unit, context, slotCount);
        Cpu1RangeLauncher.launchIndexed(reduction, unit.launchConfig(), (slotIndex, start, end) -> {
            double sum = 0.0d;
            for (int index = start; index < end; index++) {
                sum += inputArray[inputBase + index];
            }
            partialSums[slotIndex] = sum;
        });
        return sumPartialSums(partialSums, slotCount);
    }

    private static double partialSumF64(
            Cpu1PreparedReductionUnit unit,
            ExecutionContext context,
            double[] inputArray,
            int inputBase,
            int reduction
    ) {
        int slotCount = Cpu1RangeLauncher.slotCount(reduction, unit.launchConfig());
        double[] partialSums = partialSums(unit, context, slotCount);
        Cpu1RangeLauncher.launchIndexed(reduction, unit.launchConfig(), (slotIndex, start, end) -> {
            double sum = 0.0d;
            for (int index = start; index < end; index++) {
                sum += inputArray[inputBase + index];
            }
            partialSums[slotIndex] = sum;
        });
        return sumPartialSums(partialSums, slotCount);
    }

    private static double partialSumBf16(
            Cpu1PreparedReductionUnit unit,
            ExecutionContext context,
            short[] inputArray,
            int inputBase,
            int reduction
    ) {
        int slotCount = Cpu1RangeLauncher.slotCount(reduction, unit.launchConfig());
        double[] partialSums = partialSums(unit, context, slotCount);
        Cpu1RangeLauncher.launchIndexed(reduction, unit.launchConfig(), (slotIndex, start, end) -> {
            double sum = 0.0d;
            for (int index = start; index < end; index++) {
                sum += TensorDTypeOps.fromBFloat16Bits(inputArray[inputBase + index]);
            }
            partialSums[slotIndex] = sum;
        });
        return sumPartialSums(partialSums, slotCount);
    }

    private static double[] partialSums(Cpu1PreparedReductionUnit unit, ExecutionContext context, int slotCount) {
        Cpu1ScratchBuffer scratchBuffer = context.cpu1ScratchBufferForNodeId(unit.nodeId());
        if (scratchBuffer == null) {
            throw new IllegalStateException("cpu1 " + unit.opType() + " parallel scalar reduction nodeId="
                    + unit.nodeId() + " requires prepared F64 partial-sum scratch buffer.");
        }
        return scratchBuffer.requireF64Array(slotCount);
    }

    private static double sumPartialSums(double[] partialSums, int slotCount) {
        double sum = 0.0d;
        for (int slot = 0; slot < slotCount; slot++) {
            sum += partialSums[slot];
        }
        return sum;
    }

    private static Cpu1TensorView inputArrayView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        context.requireCpuReadable(unit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        return Cpu1TensorView.fromTensor(input);
    }

    private static Cpu1TensorView inputSegmentView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        NativeTensorStorage nativeInput = context.requireNativeReadable(
                unit.inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        return Cpu1TensorView.fromNativeStorage(input, nativeInput);
    }

    private static Cpu1TensorView outputArrayView(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static NativeTensorStorage outputSegmentStorage(Cpu1PreparedReductionUnit unit, ExecutionContext context) {
        return context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                unit.outputElementCount(),
                "cpu1-node-" + unit.nodeId() + ":reduction-native-segment"
        );
    }

    private static void markOutputWritten(
            Cpu1PreparedReductionUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " reduced CPU array");
    }

    private static void markNativeOutputWritten(
            Cpu1PreparedReductionUnit unit,
            NativeTensorStorage nativeOutput,
            ExecutionContext context
    ) {
        nativeOutput.markModified();
        context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 " + unit.opType() + " reduced native CPU segment");
    }
}
