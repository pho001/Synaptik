package backend.cpu1.kernels.linalg.attention.backward;

import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.contract.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import planning.region.specialization.SdpaBackwardOutputKind;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Dense SDPA backward loops for cpu1 specialized regions.
 */
public final class Cpu1AttentionBackwardLoops {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private Cpu1AttentionBackwardLoops() {
    }

    public static void run(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        if (unit.dataType() == DataType.FLOAT64) {
            runF64(unit, context);
        } else if (unit.dataType() == DataType.FLOAT32) {
            runF32(unit, context);
        } else {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD kernel supports only FLOAT32/FLOAT64, got "
                    + unit.dataType());
        }
    }

    private static void runF32(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            F32SegmentInputs inputs = f32SegmentInputs(unit, context);
            Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
            NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                    unit.nodeId(),
                    unit.dataType(),
                    unit.outputElementCount(),
                    "cpu1-sdpa-backward-node-" + unit.nodeId()
            );
            Cpu1TensorView outputView = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
            requireDenseNoOffset("output", outputView, Cpu1StorageKind.MEMORY_SEGMENT);
            requireShape("output", outputView.shape(), unit.outputShape());
            MemorySegment output = outputView.segment();
            boolean vector = unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR;
            switch (unit.outputKind()) {
                case QUERY -> {
                    float[] scratch = requireScratch(unit, context).requireF32Array();
                    if (vector) {
                        runF32QuerySegmentVector(unit, inputs, output, scratch);
                    } else {
                        runF32QuerySegment(unit, inputs, output, scratch);
                    }
                }
                case KEY -> {
                    float[] scratch = requireScratch(unit, context).requireF32Array();
                    if (vector) {
                        runF32KeySegmentVector(unit, inputs, output, scratch);
                    } else {
                        runF32KeySegment(unit, inputs, output, scratch);
                    }
                }
                case VALUE -> {
                    if (vector) {
                        runF32ValueSegmentVector(unit, inputs, output);
                    } else {
                        runF32ValueSegment(unit, inputs, output);
                    }
                }
            }
            nativeOutput.markModified();
            context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 SDPA_BACKWARD wrote native CPU segment");
            return;
        }
        F32Inputs inputs = f32Inputs(unit, context);
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        Cpu1TensorView outputView = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("output", outputView, Cpu1StorageKind.JAVA_ARRAY);
        requireShape("output", outputView.shape(), unit.outputShape());
        float[] output = outputView.float32Array();
        boolean vector = unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR;
        switch (unit.outputKind()) {
            case QUERY -> {
                float[] scratch = requireScratch(unit, context).requireF32Array();
                if (vector) {
                    runF32QueryVector(unit, inputs, output, scratch);
                } else {
                    runF32Query(unit, inputs, output, scratch);
                }
            }
            case KEY -> {
                float[] scratch = requireScratch(unit, context).requireF32Array();
                if (vector) {
                    runF32KeyVector(unit, inputs, output, scratch);
                } else {
                    runF32Key(unit, inputs, output, scratch);
                }
            }
            case VALUE -> {
                if (vector) {
                    runF32ValueVector(unit, inputs, output);
                } else {
                    runF32Value(unit, inputs, output);
                }
            }
        }
        outputView.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 SDPA_BACKWARD wrote CPU array");
    }

    private static void runF64(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        if (unit.storageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            F64SegmentInputs inputs = f64SegmentInputs(unit, context);
            Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
            NativeTensorStorage nativeOutput = context.requireNativeOutputStorage(
                    unit.nodeId(),
                    unit.dataType(),
                    unit.outputElementCount(),
                    "cpu1-sdpa-backward-node-" + unit.nodeId()
            );
            Cpu1TensorView outputView = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
            requireDenseNoOffset("output", outputView, Cpu1StorageKind.MEMORY_SEGMENT);
            requireShape("output", outputView.shape(), unit.outputShape());
            MemorySegment output = outputView.segment();
            boolean vector = unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR;
            switch (unit.outputKind()) {
                case QUERY -> {
                    double[] scratch = requireScratch(unit, context).requireF64Array();
                    if (vector) {
                        runF64QuerySegmentVector(unit, inputs, output, scratch);
                    } else {
                        runF64QuerySegment(unit, inputs, output, scratch);
                    }
                }
                case KEY -> {
                    double[] scratch = requireScratch(unit, context).requireF64Array();
                    if (vector) {
                        runF64KeySegmentVector(unit, inputs, output, scratch);
                    } else {
                        runF64KeySegment(unit, inputs, output, scratch);
                    }
                }
                case VALUE -> {
                    if (vector) {
                        runF64ValueSegmentVector(unit, inputs, output);
                    } else {
                        runF64ValueSegment(unit, inputs, output);
                    }
                }
            }
            nativeOutput.markModified();
            context.attachNativeStorage(unit.nodeId(), nativeOutput, "cpu1 SDPA_BACKWARD wrote native CPU segment");
            return;
        }
        F64Inputs inputs = f64Inputs(unit, context);
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        Cpu1TensorView outputView = Cpu1TensorView.fromTensor(outputTensor);
        requireDenseNoOffset("output", outputView, Cpu1StorageKind.JAVA_ARRAY);
        requireShape("output", outputView.shape(), unit.outputShape());
        double[] output = outputView.float64Array();
        boolean vector = unit.vectorizationKind() == Cpu1VectorizationKind.VECTOR;
        switch (unit.outputKind()) {
            case QUERY -> {
                double[] scratch = requireScratch(unit, context).requireF64Array();
                if (vector) {
                    runF64QueryVector(unit, inputs, output, scratch);
                } else {
                    runF64Query(unit, inputs, output, scratch);
                }
            }
            case KEY -> {
                double[] scratch = requireScratch(unit, context).requireF64Array();
                if (vector) {
                    runF64KeyVector(unit, inputs, output, scratch);
                } else {
                    runF64Key(unit, inputs, output, scratch);
                }
            }
            case VALUE -> {
                if (vector) {
                    runF64ValueVector(unit, inputs, output);
                } else {
                    runF64Value(unit, inputs, output);
                }
            }
        }
        outputView.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 SDPA_BACKWARD wrote CPU array");
    }

    private static void runF32Query(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            float[] output,
            float[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            for (int row = start; row < end; row++) {
                int batch = row / unit.queryLen();
                int queryIndex = row - batch * unit.queryLen();
                computeF32ScoreRow(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int outputBase = (batch * unit.queryLen() + queryIndex) * unit.depth();
                int keyBase = batch * unit.keyLen() * unit.depth();
                for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                    float sum = 0.0f;
                    for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                        sum += scratch[dScoresBase + keyIndex]
                                * inputs.key[keyBase + keyIndex * unit.depth() + depthIndex];
                    }
                    output[outputBase + depthIndex] = sum;
                }
            }
        });
    }

    private static void runF32Key(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            float[] output,
            float[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
            for (int row = start; row < end; ) {
                int batch = row / unit.keyLen();
                int keyStart = row - batch * unit.keyLen();
                int nextBatchRow = Math.min(end, (batch + 1) * unit.keyLen());
                int keyEnd = nextBatchRow - batch * unit.keyLen();
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    computeF32ScoreRow(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                    int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                    System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
                }

                int outputBatchBase = batch * unit.keyLen() * unit.depth();
                int queryBatchBase = batch * unit.queryLen() * unit.depth();
                for (int keyIndex = keyStart; keyIndex < keyEnd; keyIndex++) {
                    int outputBase = outputBatchBase + keyIndex * unit.depth();
                    for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                        float sum = 0.0f;
                        for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                            float dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                            sum += dScore * inputs.query[queryBatchBase + queryIndex * unit.depth() + depthIndex];
                        }
                        output[outputBase + depthIndex] = sum;
                    }
                }
                row = nextBatchRow;
            }
        });
    }

    private static void runF32Value(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            float[] output
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            for (int row = start; row < end; row++) {
                int batch = row / unit.keyLen();
                int keyIndex = row - batch * unit.keyLen();
                int outputBase = (batch * unit.keyLen() + keyIndex) * unit.valueDim();
                int weightsBase = batch * unit.queryLen() * unit.keyLen();
                for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                    float sum = 0.0f;
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        sum += inputs.weights[weightsBase + queryIndex * unit.keyLen() + keyIndex]
                                * outGradF32(unit, inputs, batch, queryIndex, valueIndex);
                    }
                    output[outputBase + valueIndex] = sum;
                }
            }
        });
    }

    private static void computeF32ScoreRow(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            int batch,
            int queryIndex,
            float[] scratch,
            int dWeightsBase,
            int dScoresBase
    ) {
        int valueBase = batch * unit.keyLen() * unit.valueDim();
        int weightsBase = batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen();
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float sum = 0.0f;
            int valueRowBase = valueBase + keyIndex * unit.valueDim();
            for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                sum += outGradF32(unit, inputs, batch, queryIndex, valueIndex) * inputs.value[valueRowBase + valueIndex];
            }
            scratch[dWeightsBase + keyIndex] = sum;
        }
        float dot = 0.0f;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            dot += scratch[dWeightsBase + keyIndex] * inputs.weights[weightsBase + keyIndex];
        }
        int maskBase = unit.hasMask()
                ? batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen()
                : -1;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float dScore = inputs.weights[weightsBase + keyIndex] * (scratch[dWeightsBase + keyIndex] - dot);
            if (inputs.mask != null && inputs.mask[maskBase + keyIndex] == 0) {
                dScore = 0.0f;
            }
            scratch[dScoresBase + keyIndex] = dScore * unit.scaleF32();
        }
    }

    private static void runF32QueryVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            float[] output,
            float[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            for (int row = start; row < end; row++) {
                int batch = row / unit.queryLen();
                int queryIndex = row - batch * unit.queryLen();
                computeF32ScoreRowVector(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int outputBase = (batch * unit.queryLen() + queryIndex) * unit.depth();
                int keyBase = batch * unit.keyLen() * unit.depth();
                Arrays.fill(output, outputBase, outputBase + unit.depth(), 0.0f);
                for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                    float dScore = scratch[dScoresBase + keyIndex];
                    if (dScore != 0.0f) {
                        accumulateF32Vector(
                                output,
                                outputBase,
                                inputs.key,
                                keyBase + keyIndex * unit.depth(),
                                dScore,
                                unit.depth()
                        );
                    }
                }
            }
        });
    }

    private static void runF32KeyVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            float[] output,
            float[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
            for (int row = start; row < end; ) {
                int batch = row / unit.keyLen();
                int keyStart = row - batch * unit.keyLen();
                int nextBatchRow = Math.min(end, (batch + 1) * unit.keyLen());
                int keyEnd = nextBatchRow - batch * unit.keyLen();
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    computeF32ScoreRowVector(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                    int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                    System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
                }

                int outputBatchBase = batch * unit.keyLen() * unit.depth();
                int queryBatchBase = batch * unit.queryLen() * unit.depth();
                for (int keyIndex = keyStart; keyIndex < keyEnd; keyIndex++) {
                    int outputBase = outputBatchBase + keyIndex * unit.depth();
                    Arrays.fill(output, outputBase, outputBase + unit.depth(), 0.0f);
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        float dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                        if (dScore != 0.0f) {
                            accumulateF32Vector(
                                    output,
                                    outputBase,
                                    inputs.query,
                                    queryBatchBase + queryIndex * unit.depth(),
                                    dScore,
                                    unit.depth()
                            );
                        }
                    }
                }
                row = nextBatchRow;
            }
        });
    }

    private static void runF32ValueVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            float[] output
    ) {
        if (!inputs.outGradDense) {
            runF32Value(unit, inputs, output);
            return;
        }
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            for (int row = start; row < end; row++) {
                int batch = row / unit.keyLen();
                int keyIndex = row - batch * unit.keyLen();
                int outputBase = (batch * unit.keyLen() + keyIndex) * unit.valueDim();
                int weightsBase = batch * unit.queryLen() * unit.keyLen();
                int outGradBatchBase = batch * unit.queryLen() * unit.valueDim();
                Arrays.fill(output, outputBase, outputBase + unit.valueDim(), 0.0f);
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    float weight = inputs.weights[weightsBase + queryIndex * unit.keyLen() + keyIndex];
                    if (weight != 0.0f) {
                        accumulateF32Vector(
                                output,
                                outputBase,
                                inputs.outGrad,
                                outGradBatchBase + queryIndex * unit.valueDim(),
                                weight,
                                unit.valueDim()
                        );
                    }
                }
            }
        });
    }

    private static void computeF32ScoreRowVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            int batch,
            int queryIndex,
            float[] scratch,
            int dWeightsBase,
            int dScoresBase
    ) {
        int valueBase = batch * unit.keyLen() * unit.valueDim();
        int weightsBase = batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen();
        if (inputs.outGradDense) {
            int outGradBase = batch * unit.queryLen() * unit.valueDim() + queryIndex * unit.valueDim();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                int valueRowBase = valueBase + keyIndex * unit.valueDim();
                scratch[dWeightsBase + keyIndex] = dotF32Vector(
                        inputs.outGrad,
                        outGradBase,
                        inputs.value,
                        valueRowBase,
                        unit.valueDim()
                );
            }
        } else {
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                float sum = 0.0f;
                int valueRowBase = valueBase + keyIndex * unit.valueDim();
                for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                    sum += outGradF32(unit, inputs, batch, queryIndex, valueIndex)
                            * inputs.value[valueRowBase + valueIndex];
                }
                scratch[dWeightsBase + keyIndex] = sum;
            }
        }
        float dot = dotF32Vector(scratch, dWeightsBase, inputs.weights, weightsBase, unit.keyLen());
        int maskBase = unit.hasMask()
                ? batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen()
                : -1;
        if (inputs.mask == null) {
            computeF32DScoresNoMaskVector(unit, inputs, scratch, dWeightsBase, dScoresBase, weightsBase, dot);
            return;
        }
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float dScore = inputs.weights[weightsBase + keyIndex] * (scratch[dWeightsBase + keyIndex] - dot);
            if (inputs.mask[maskBase + keyIndex] == 0) {
                dScore = 0.0f;
            }
            scratch[dScoresBase + keyIndex] = dScore * unit.scaleF32();
        }
    }

    private static void computeF32DScoresNoMaskVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            float[] scratch,
            int dWeightsBase,
            int dScoresBase,
            int weightsBase,
            float dot
    ) {
        FloatVector dotVector = FloatVector.broadcast(F32, dot);
        FloatVector scaleVector = FloatVector.broadcast(F32, unit.scaleF32());
        int i = 0;
        int upper = F32.loopBound(unit.keyLen());
        for (; i < upper; i += F32.length()) {
            FloatVector weights = FloatVector.fromArray(F32, inputs.weights, weightsBase + i);
            FloatVector dWeights = FloatVector.fromArray(F32, scratch, dWeightsBase + i);
            weights.mul(dWeights.sub(dotVector)).mul(scaleVector).intoArray(scratch, dScoresBase + i);
        }
        for (; i < unit.keyLen(); i++) {
            scratch[dScoresBase + i] = inputs.weights[weightsBase + i]
                    * (scratch[dWeightsBase + i] - dot)
                    * unit.scaleF32();
        }
    }

    private static void runF32QuerySegment(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            float[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            for (int row = start; row < end; row++) {
                int batch = row / unit.queryLen();
                int queryIndex = row - batch * unit.queryLen();
                computeF32ScoreRowSegment(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int outputBase = (batch * unit.queryLen() + queryIndex) * unit.depth();
                int keyBase = batch * unit.keyLen() * unit.depth();
                for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                    float sum = 0.0f;
                    for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                        sum += scratch[dScoresBase + keyIndex]
                                * f32(inputs.key, keyBase + keyIndex * unit.depth() + depthIndex);
                    }
                    setF32(output, outputBase + depthIndex, sum);
                }
            }
        });
    }

    private static void runF32KeySegment(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            float[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
            for (int row = start; row < end; ) {
                int batch = row / unit.keyLen();
                int keyStart = row - batch * unit.keyLen();
                int nextBatchRow = Math.min(end, (batch + 1) * unit.keyLen());
                int keyEnd = nextBatchRow - batch * unit.keyLen();
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    computeF32ScoreRowSegment(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                    int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                    System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
                }

                int outputBatchBase = batch * unit.keyLen() * unit.depth();
                int queryBatchBase = batch * unit.queryLen() * unit.depth();
                for (int keyIndex = keyStart; keyIndex < keyEnd; keyIndex++) {
                    int outputBase = outputBatchBase + keyIndex * unit.depth();
                    for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                        float sum = 0.0f;
                        for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                            float dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                            sum += dScore * f32(inputs.query, queryBatchBase + queryIndex * unit.depth() + depthIndex);
                        }
                        setF32(output, outputBase + depthIndex, sum);
                    }
                }
                row = nextBatchRow;
            }
        });
    }

    private static void runF32ValueSegment(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            for (int row = start; row < end; row++) {
                int batch = row / unit.keyLen();
                int keyIndex = row - batch * unit.keyLen();
                int outputBase = (batch * unit.keyLen() + keyIndex) * unit.valueDim();
                int weightsBase = batch * unit.queryLen() * unit.keyLen();
                for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                    float sum = 0.0f;
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        sum += f32(inputs.weights, weightsBase + queryIndex * unit.keyLen() + keyIndex)
                                * outGradF32(unit, inputs, batch, queryIndex, valueIndex);
                    }
                    setF32(output, outputBase + valueIndex, sum);
                }
            }
        });
    }

    private static void computeF32ScoreRowSegment(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            int batch,
            int queryIndex,
            float[] scratch,
            int dWeightsBase,
            int dScoresBase
    ) {
        int valueBase = batch * unit.keyLen() * unit.valueDim();
        int weightsBase = batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen();
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float sum = 0.0f;
            int valueRowBase = valueBase + keyIndex * unit.valueDim();
            for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                sum += outGradF32(unit, inputs, batch, queryIndex, valueIndex)
                        * f32(inputs.value, valueRowBase + valueIndex);
            }
            scratch[dWeightsBase + keyIndex] = sum;
        }
        float dot = 0.0f;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            dot += scratch[dWeightsBase + keyIndex] * f32(inputs.weights, weightsBase + keyIndex);
        }
        int maskBase = unit.hasMask()
                ? batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen()
                : -1;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float dScore = f32(inputs.weights, weightsBase + keyIndex) * (scratch[dWeightsBase + keyIndex] - dot);
            if (inputs.mask != null && bool(inputs.mask, maskBase + keyIndex) == 0) {
                dScore = 0.0f;
            }
            scratch[dScoresBase + keyIndex] = dScore * unit.scaleF32();
        }
    }

    private static void runF32QuerySegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            float[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            for (int row = start; row < end; row++) {
                int batch = row / unit.queryLen();
                int queryIndex = row - batch * unit.queryLen();
                computeF32ScoreRowSegmentVector(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int outputBase = (batch * unit.queryLen() + queryIndex) * unit.depth();
                int keyBase = batch * unit.keyLen() * unit.depth();
                zeroF32Segment(output, outputBase, unit.depth());
                for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                    float dScore = scratch[dScoresBase + keyIndex];
                    if (dScore != 0.0f) {
                        accumulateF32SegmentVector(
                                output,
                                outputBase,
                                inputs.key,
                                keyBase + keyIndex * unit.depth(),
                                dScore,
                                unit.depth()
                        );
                    }
                }
            }
        });
    }

    private static void runF32KeySegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output,
            float[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
            for (int row = start; row < end; ) {
                int batch = row / unit.keyLen();
                int keyStart = row - batch * unit.keyLen();
                int nextBatchRow = Math.min(end, (batch + 1) * unit.keyLen());
                int keyEnd = nextBatchRow - batch * unit.keyLen();
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    computeF32ScoreRowSegmentVector(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                    int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                    System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
                }

                int outputBatchBase = batch * unit.keyLen() * unit.depth();
                int queryBatchBase = batch * unit.queryLen() * unit.depth();
                for (int keyIndex = keyStart; keyIndex < keyEnd; keyIndex++) {
                    int outputBase = outputBatchBase + keyIndex * unit.depth();
                    zeroF32Segment(output, outputBase, unit.depth());
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        float dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                        if (dScore != 0.0f) {
                            accumulateF32SegmentVector(
                                    output,
                                    outputBase,
                                    inputs.query,
                                    queryBatchBase + queryIndex * unit.depth(),
                                    dScore,
                                    unit.depth()
                            );
                        }
                    }
                }
                row = nextBatchRow;
            }
        });
    }

    private static void runF32ValueSegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            MemorySegment output
    ) {
        if (!inputs.outGradDense) {
            runF32ValueSegment(unit, inputs, output);
            return;
        }
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            for (int row = start; row < end; row++) {
                int batch = row / unit.keyLen();
                int keyIndex = row - batch * unit.keyLen();
                int outputBase = (batch * unit.keyLen() + keyIndex) * unit.valueDim();
                int weightsBase = batch * unit.queryLen() * unit.keyLen();
                int outGradBatchBase = batch * unit.queryLen() * unit.valueDim();
                zeroF32Segment(output, outputBase, unit.valueDim());
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    float weight = f32(inputs.weights, weightsBase + queryIndex * unit.keyLen() + keyIndex);
                    if (weight != 0.0f) {
                        accumulateF32SegmentVector(
                                output,
                                outputBase,
                                inputs.outGrad,
                                outGradBatchBase + queryIndex * unit.valueDim(),
                                weight,
                                unit.valueDim()
                        );
                    }
                }
            }
        });
    }

    private static void computeF32ScoreRowSegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            int batch,
            int queryIndex,
            float[] scratch,
            int dWeightsBase,
            int dScoresBase
    ) {
        int valueBase = batch * unit.keyLen() * unit.valueDim();
        int weightsBase = batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen();
        if (inputs.outGradDense) {
            int outGradBase = batch * unit.queryLen() * unit.valueDim() + queryIndex * unit.valueDim();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                int valueRowBase = valueBase + keyIndex * unit.valueDim();
                scratch[dWeightsBase + keyIndex] = dotF32SegmentVector(
                        inputs.outGrad,
                        outGradBase,
                        inputs.value,
                        valueRowBase,
                        unit.valueDim()
                );
            }
        } else {
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                float sum = 0.0f;
                int valueRowBase = valueBase + keyIndex * unit.valueDim();
                for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                    sum += outGradF32(unit, inputs, batch, queryIndex, valueIndex)
                            * f32(inputs.value, valueRowBase + valueIndex);
                }
                scratch[dWeightsBase + keyIndex] = sum;
            }
        }
        float dot = dotF32ArraySegmentVector(scratch, dWeightsBase, inputs.weights, weightsBase, unit.keyLen());
        int maskBase = unit.hasMask()
                ? batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen()
                : -1;
        if (inputs.mask == null) {
            computeF32DScoresNoMaskSegmentVector(unit, inputs, scratch, dWeightsBase, dScoresBase, weightsBase, dot);
            return;
        }
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            float dScore = f32(inputs.weights, weightsBase + keyIndex) * (scratch[dWeightsBase + keyIndex] - dot);
            if (bool(inputs.mask, maskBase + keyIndex) == 0) {
                dScore = 0.0f;
            }
            scratch[dScoresBase + keyIndex] = dScore * unit.scaleF32();
        }
    }

    private static void computeF32DScoresNoMaskSegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            float[] scratch,
            int dWeightsBase,
            int dScoresBase,
            int weightsBase,
            float dot
    ) {
        FloatVector dotVector = FloatVector.broadcast(F32, dot);
        FloatVector scaleVector = FloatVector.broadcast(F32, unit.scaleF32());
        int i = 0;
        int upper = F32.loopBound(unit.keyLen());
        for (; i < upper; i += F32.length()) {
            FloatVector weights = FloatVector.fromMemorySegment(
                    F32,
                    inputs.weights,
                    (long) (weightsBase + i) * Float.BYTES,
                    ORDER
            );
            FloatVector dWeights = FloatVector.fromArray(F32, scratch, dWeightsBase + i);
            weights.mul(dWeights.sub(dotVector)).mul(scaleVector).intoArray(scratch, dScoresBase + i);
        }
        for (; i < unit.keyLen(); i++) {
            scratch[dScoresBase + i] = f32(inputs.weights, weightsBase + i)
                    * (scratch[dWeightsBase + i] - dot)
                    * unit.scaleF32();
        }
    }

    private static void runF64Query(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            double[] output,
            double[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            for (int row = start; row < end; row++) {
                int batch = row / unit.queryLen();
                int queryIndex = row - batch * unit.queryLen();
                computeF64ScoreRow(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int outputBase = (batch * unit.queryLen() + queryIndex) * unit.depth();
                int keyBase = batch * unit.keyLen() * unit.depth();
                for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                    double sum = 0.0d;
                    for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                        sum += scratch[dScoresBase + keyIndex]
                                * inputs.key[keyBase + keyIndex * unit.depth() + depthIndex];
                    }
                    output[outputBase + depthIndex] = sum;
                }
            }
        });
    }

    private static void runF64Key(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            double[] output,
            double[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
            for (int row = start; row < end; ) {
                int batch = row / unit.keyLen();
                int keyStart = row - batch * unit.keyLen();
                int nextBatchRow = Math.min(end, (batch + 1) * unit.keyLen());
                int keyEnd = nextBatchRow - batch * unit.keyLen();
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    computeF64ScoreRow(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                    int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                    System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
                }

                int outputBatchBase = batch * unit.keyLen() * unit.depth();
                int queryBatchBase = batch * unit.queryLen() * unit.depth();
                for (int keyIndex = keyStart; keyIndex < keyEnd; keyIndex++) {
                    int outputBase = outputBatchBase + keyIndex * unit.depth();
                    for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                        double sum = 0.0d;
                        for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                            double dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                            sum += dScore * inputs.query[queryBatchBase + queryIndex * unit.depth() + depthIndex];
                        }
                        output[outputBase + depthIndex] = sum;
                    }
                }
                row = nextBatchRow;
            }
        });
    }

    private static void runF64Value(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            double[] output
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            for (int row = start; row < end; row++) {
                int batch = row / unit.keyLen();
                int keyIndex = row - batch * unit.keyLen();
                int outputBase = (batch * unit.keyLen() + keyIndex) * unit.valueDim();
                int weightsBase = batch * unit.queryLen() * unit.keyLen();
                for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                    double sum = 0.0d;
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        sum += inputs.weights[weightsBase + queryIndex * unit.keyLen() + keyIndex]
                                * outGradF64(unit, inputs, batch, queryIndex, valueIndex);
                    }
                    output[outputBase + valueIndex] = sum;
                }
            }
        });
    }

    private static void computeF64ScoreRow(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            int batch,
            int queryIndex,
            double[] scratch,
            int dWeightsBase,
            int dScoresBase
    ) {
        int valueBase = batch * unit.keyLen() * unit.valueDim();
        int weightsBase = batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen();
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double sum = 0.0d;
            int valueRowBase = valueBase + keyIndex * unit.valueDim();
            for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                sum += outGradF64(unit, inputs, batch, queryIndex, valueIndex) * inputs.value[valueRowBase + valueIndex];
            }
            scratch[dWeightsBase + keyIndex] = sum;
        }
        double dot = 0.0d;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            dot += scratch[dWeightsBase + keyIndex] * inputs.weights[weightsBase + keyIndex];
        }
        int maskBase = unit.hasMask()
                ? batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen()
                : -1;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double dScore = inputs.weights[weightsBase + keyIndex] * (scratch[dWeightsBase + keyIndex] - dot);
            if (inputs.mask != null && inputs.mask[maskBase + keyIndex] == 0) {
                dScore = 0.0d;
            }
            scratch[dScoresBase + keyIndex] = dScore * unit.scale();
        }
    }

    private static void runF64QueryVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            double[] output,
            double[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            for (int row = start; row < end; row++) {
                int batch = row / unit.queryLen();
                int queryIndex = row - batch * unit.queryLen();
                computeF64ScoreRowVector(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int outputBase = (batch * unit.queryLen() + queryIndex) * unit.depth();
                int keyBase = batch * unit.keyLen() * unit.depth();
                Arrays.fill(output, outputBase, outputBase + unit.depth(), 0.0d);
                for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                    double dScore = scratch[dScoresBase + keyIndex];
                    if (dScore != 0.0d) {
                        accumulateF64Vector(
                                output,
                                outputBase,
                                inputs.key,
                                keyBase + keyIndex * unit.depth(),
                                dScore,
                                unit.depth()
                        );
                    }
                }
            }
        });
    }

    private static void runF64KeyVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            double[] output,
            double[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
            for (int row = start; row < end; ) {
                int batch = row / unit.keyLen();
                int keyStart = row - batch * unit.keyLen();
                int nextBatchRow = Math.min(end, (batch + 1) * unit.keyLen());
                int keyEnd = nextBatchRow - batch * unit.keyLen();
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    computeF64ScoreRowVector(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                    int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                    System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
                }

                int outputBatchBase = batch * unit.keyLen() * unit.depth();
                int queryBatchBase = batch * unit.queryLen() * unit.depth();
                for (int keyIndex = keyStart; keyIndex < keyEnd; keyIndex++) {
                    int outputBase = outputBatchBase + keyIndex * unit.depth();
                    Arrays.fill(output, outputBase, outputBase + unit.depth(), 0.0d);
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        double dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                        if (dScore != 0.0d) {
                            accumulateF64Vector(
                                    output,
                                    outputBase,
                                    inputs.query,
                                    queryBatchBase + queryIndex * unit.depth(),
                                    dScore,
                                    unit.depth()
                            );
                        }
                    }
                }
                row = nextBatchRow;
            }
        });
    }

    private static void runF64ValueVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            double[] output
    ) {
        if (!inputs.outGradDense) {
            runF64Value(unit, inputs, output);
            return;
        }
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            for (int row = start; row < end; row++) {
                int batch = row / unit.keyLen();
                int keyIndex = row - batch * unit.keyLen();
                int outputBase = (batch * unit.keyLen() + keyIndex) * unit.valueDim();
                int weightsBase = batch * unit.queryLen() * unit.keyLen();
                int outGradBatchBase = batch * unit.queryLen() * unit.valueDim();
                Arrays.fill(output, outputBase, outputBase + unit.valueDim(), 0.0d);
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    double weight = inputs.weights[weightsBase + queryIndex * unit.keyLen() + keyIndex];
                    if (weight != 0.0d) {
                        accumulateF64Vector(
                                output,
                                outputBase,
                                inputs.outGrad,
                                outGradBatchBase + queryIndex * unit.valueDim(),
                                weight,
                                unit.valueDim()
                        );
                    }
                }
            }
        });
    }

    private static void computeF64ScoreRowVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            int batch,
            int queryIndex,
            double[] scratch,
            int dWeightsBase,
            int dScoresBase
    ) {
        int valueBase = batch * unit.keyLen() * unit.valueDim();
        int weightsBase = batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen();
        if (inputs.outGradDense) {
            int outGradBase = batch * unit.queryLen() * unit.valueDim() + queryIndex * unit.valueDim();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                int valueRowBase = valueBase + keyIndex * unit.valueDim();
                scratch[dWeightsBase + keyIndex] = dotF64Vector(
                        inputs.outGrad,
                        outGradBase,
                        inputs.value,
                        valueRowBase,
                        unit.valueDim()
                );
            }
        } else {
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                double sum = 0.0d;
                int valueRowBase = valueBase + keyIndex * unit.valueDim();
                for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                    sum += outGradF64(unit, inputs, batch, queryIndex, valueIndex)
                            * inputs.value[valueRowBase + valueIndex];
                }
                scratch[dWeightsBase + keyIndex] = sum;
            }
        }
        double dot = dotF64Vector(scratch, dWeightsBase, inputs.weights, weightsBase, unit.keyLen());
        int maskBase = unit.hasMask()
                ? batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen()
                : -1;
        if (inputs.mask == null) {
            computeF64DScoresNoMaskVector(unit, inputs, scratch, dWeightsBase, dScoresBase, weightsBase, dot);
            return;
        }
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double dScore = inputs.weights[weightsBase + keyIndex] * (scratch[dWeightsBase + keyIndex] - dot);
            if (inputs.mask[maskBase + keyIndex] == 0) {
                dScore = 0.0d;
            }
            scratch[dScoresBase + keyIndex] = dScore * unit.scale();
        }
    }

    private static void computeF64DScoresNoMaskVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            double[] scratch,
            int dWeightsBase,
            int dScoresBase,
            int weightsBase,
            double dot
    ) {
        DoubleVector dotVector = DoubleVector.broadcast(F64, dot);
        DoubleVector scaleVector = DoubleVector.broadcast(F64, unit.scale());
        int i = 0;
        int upper = F64.loopBound(unit.keyLen());
        for (; i < upper; i += F64.length()) {
            DoubleVector weights = DoubleVector.fromArray(F64, inputs.weights, weightsBase + i);
            DoubleVector dWeights = DoubleVector.fromArray(F64, scratch, dWeightsBase + i);
            weights.mul(dWeights.sub(dotVector)).mul(scaleVector).intoArray(scratch, dScoresBase + i);
        }
        for (; i < unit.keyLen(); i++) {
            scratch[dScoresBase + i] = inputs.weights[weightsBase + i]
                    * (scratch[dWeightsBase + i] - dot)
                    * unit.scale();
        }
    }

    private static void runF64QuerySegment(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output,
            double[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            for (int row = start; row < end; row++) {
                int batch = row / unit.queryLen();
                int queryIndex = row - batch * unit.queryLen();
                computeF64ScoreRowSegment(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int outputBase = (batch * unit.queryLen() + queryIndex) * unit.depth();
                int keyBase = batch * unit.keyLen() * unit.depth();
                for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                    double sum = 0.0d;
                    for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                        sum += scratch[dScoresBase + keyIndex]
                                * f64(inputs.key, keyBase + keyIndex * unit.depth() + depthIndex);
                    }
                    setF64(output, outputBase + depthIndex, sum);
                }
            }
        });
    }

    private static void runF64KeySegment(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output,
            double[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
            for (int row = start; row < end; ) {
                int batch = row / unit.keyLen();
                int keyStart = row - batch * unit.keyLen();
                int nextBatchRow = Math.min(end, (batch + 1) * unit.keyLen());
                int keyEnd = nextBatchRow - batch * unit.keyLen();
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    computeF64ScoreRowSegment(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                    int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                    System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
                }

                int outputBatchBase = batch * unit.keyLen() * unit.depth();
                int queryBatchBase = batch * unit.queryLen() * unit.depth();
                for (int keyIndex = keyStart; keyIndex < keyEnd; keyIndex++) {
                    int outputBase = outputBatchBase + keyIndex * unit.depth();
                    for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                        double sum = 0.0d;
                        for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                            double dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                            sum += dScore * f64(inputs.query, queryBatchBase + queryIndex * unit.depth() + depthIndex);
                        }
                        setF64(output, outputBase + depthIndex, sum);
                    }
                }
                row = nextBatchRow;
            }
        });
    }

    private static void runF64ValueSegment(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            for (int row = start; row < end; row++) {
                int batch = row / unit.keyLen();
                int keyIndex = row - batch * unit.keyLen();
                int outputBase = (batch * unit.keyLen() + keyIndex) * unit.valueDim();
                int weightsBase = batch * unit.queryLen() * unit.keyLen();
                for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                    double sum = 0.0d;
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        sum += f64(inputs.weights, weightsBase + queryIndex * unit.keyLen() + keyIndex)
                                * outGradF64(unit, inputs, batch, queryIndex, valueIndex);
                    }
                    setF64(output, outputBase + valueIndex, sum);
                }
            }
        });
    }

    private static void computeF64ScoreRowSegment(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            int batch,
            int queryIndex,
            double[] scratch,
            int dWeightsBase,
            int dScoresBase
    ) {
        int valueBase = batch * unit.keyLen() * unit.valueDim();
        int weightsBase = batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen();
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double sum = 0.0d;
            int valueRowBase = valueBase + keyIndex * unit.valueDim();
            for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                sum += outGradF64(unit, inputs, batch, queryIndex, valueIndex)
                        * f64(inputs.value, valueRowBase + valueIndex);
            }
            scratch[dWeightsBase + keyIndex] = sum;
        }
        double dot = 0.0d;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            dot += scratch[dWeightsBase + keyIndex] * f64(inputs.weights, weightsBase + keyIndex);
        }
        int maskBase = unit.hasMask()
                ? batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen()
                : -1;
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double dScore = f64(inputs.weights, weightsBase + keyIndex) * (scratch[dWeightsBase + keyIndex] - dot);
            if (inputs.mask != null && bool(inputs.mask, maskBase + keyIndex) == 0) {
                dScore = 0.0d;
            }
            scratch[dScoresBase + keyIndex] = dScore * unit.scale();
        }
    }

    private static void runF64QuerySegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output,
            double[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            for (int row = start; row < end; row++) {
                int batch = row / unit.queryLen();
                int queryIndex = row - batch * unit.queryLen();
                computeF64ScoreRowSegmentVector(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int outputBase = (batch * unit.queryLen() + queryIndex) * unit.depth();
                int keyBase = batch * unit.keyLen() * unit.depth();
                zeroF64Segment(output, outputBase, unit.depth());
                for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                    double dScore = scratch[dScoresBase + keyIndex];
                    if (dScore != 0.0d) {
                        accumulateF64SegmentVector(
                                output,
                                outputBase,
                                inputs.key,
                                keyBase + keyIndex * unit.depth(),
                                dScore,
                                unit.depth()
                        );
                    }
                }
            }
        });
    }

    private static void runF64KeySegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output,
            double[] scratch
    ) {
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            int scratchBase = slot * unit.scratchElementsPerSlot();
            int dWeightsBase = scratchBase;
            int dScoresBase = scratchBase + unit.keyLen();
            int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
            for (int row = start; row < end; ) {
                int batch = row / unit.keyLen();
                int keyStart = row - batch * unit.keyLen();
                int nextBatchRow = Math.min(end, (batch + 1) * unit.keyLen());
                int keyEnd = nextBatchRow - batch * unit.keyLen();
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    computeF64ScoreRowSegmentVector(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                    int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                    System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
                }

                int outputBatchBase = batch * unit.keyLen() * unit.depth();
                int queryBatchBase = batch * unit.queryLen() * unit.depth();
                for (int keyIndex = keyStart; keyIndex < keyEnd; keyIndex++) {
                    int outputBase = outputBatchBase + keyIndex * unit.depth();
                    zeroF64Segment(output, outputBase, unit.depth());
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        double dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                        if (dScore != 0.0d) {
                            accumulateF64SegmentVector(
                                    output,
                                    outputBase,
                                    inputs.query,
                                    queryBatchBase + queryIndex * unit.depth(),
                                    dScore,
                                    unit.depth()
                            );
                        }
                    }
                }
                row = nextBatchRow;
            }
        });
    }

    private static void runF64ValueSegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            MemorySegment output
    ) {
        if (!inputs.outGradDense) {
            runF64ValueSegment(unit, inputs, output);
            return;
        }
        Cpu1RangeLauncher.launchIndexed(unit.rowCount(), unit.launchConfig(), (slot, start, end) -> {
            for (int row = start; row < end; row++) {
                int batch = row / unit.keyLen();
                int keyIndex = row - batch * unit.keyLen();
                int outputBase = (batch * unit.keyLen() + keyIndex) * unit.valueDim();
                int weightsBase = batch * unit.queryLen() * unit.keyLen();
                int outGradBatchBase = batch * unit.queryLen() * unit.valueDim();
                zeroF64Segment(output, outputBase, unit.valueDim());
                for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                    double weight = f64(inputs.weights, weightsBase + queryIndex * unit.keyLen() + keyIndex);
                    if (weight != 0.0d) {
                        accumulateF64SegmentVector(
                                output,
                                outputBase,
                                inputs.outGrad,
                                outGradBatchBase + queryIndex * unit.valueDim(),
                                weight,
                                unit.valueDim()
                        );
                    }
                }
            }
        });
    }

    private static void computeF64ScoreRowSegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            int batch,
            int queryIndex,
            double[] scratch,
            int dWeightsBase,
            int dScoresBase
    ) {
        int valueBase = batch * unit.keyLen() * unit.valueDim();
        int weightsBase = batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen();
        if (inputs.outGradDense) {
            int outGradBase = batch * unit.queryLen() * unit.valueDim() + queryIndex * unit.valueDim();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                int valueRowBase = valueBase + keyIndex * unit.valueDim();
                scratch[dWeightsBase + keyIndex] = dotF64SegmentVector(
                        inputs.outGrad,
                        outGradBase,
                        inputs.value,
                        valueRowBase,
                        unit.valueDim()
                );
            }
        } else {
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                double sum = 0.0d;
                int valueRowBase = valueBase + keyIndex * unit.valueDim();
                for (int valueIndex = 0; valueIndex < unit.valueDim(); valueIndex++) {
                    sum += outGradF64(unit, inputs, batch, queryIndex, valueIndex)
                            * f64(inputs.value, valueRowBase + valueIndex);
                }
                scratch[dWeightsBase + keyIndex] = sum;
            }
        }
        double dot = dotF64ArraySegmentVector(scratch, dWeightsBase, inputs.weights, weightsBase, unit.keyLen());
        int maskBase = unit.hasMask()
                ? batch * unit.queryLen() * unit.keyLen() + queryIndex * unit.keyLen()
                : -1;
        if (inputs.mask == null) {
            computeF64DScoresNoMaskSegmentVector(unit, inputs, scratch, dWeightsBase, dScoresBase, weightsBase, dot);
            return;
        }
        for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
            double dScore = f64(inputs.weights, weightsBase + keyIndex) * (scratch[dWeightsBase + keyIndex] - dot);
            if (bool(inputs.mask, maskBase + keyIndex) == 0) {
                dScore = 0.0d;
            }
            scratch[dScoresBase + keyIndex] = dScore * unit.scale();
        }
    }

    private static void computeF64DScoresNoMaskSegmentVector(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            double[] scratch,
            int dWeightsBase,
            int dScoresBase,
            int weightsBase,
            double dot
    ) {
        DoubleVector dotVector = DoubleVector.broadcast(F64, dot);
        DoubleVector scaleVector = DoubleVector.broadcast(F64, unit.scale());
        int i = 0;
        int upper = F64.loopBound(unit.keyLen());
        for (; i < upper; i += F64.length()) {
            DoubleVector weights = DoubleVector.fromMemorySegment(
                    F64,
                    inputs.weights,
                    (long) (weightsBase + i) * Double.BYTES,
                    ORDER
            );
            DoubleVector dWeights = DoubleVector.fromArray(F64, scratch, dWeightsBase + i);
            weights.mul(dWeights.sub(dotVector)).mul(scaleVector).intoArray(scratch, dScoresBase + i);
        }
        for (; i < unit.keyLen(); i++) {
            scratch[dScoresBase + i] = f64(inputs.weights, weightsBase + i)
                    * (scratch[dWeightsBase + i] - dot)
                    * unit.scale();
        }
    }

    private static float outGradF32(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32Inputs inputs,
            int batch,
            int queryIndex,
            int valueIndex
    ) {
        int index = inputs.outGradDense
                ? batch * unit.queryLen() * unit.valueDim() + queryIndex * unit.valueDim() + valueIndex
                : stridedOutGradOffset(inputs.outGradShape, inputs.outGradStrides, inputs.outGradStorageOffset,
                batch, queryIndex, valueIndex);
        return inputs.outGrad[index];
    }

    private static float outGradF32(
            Cpu1PreparedAttentionBackwardUnit unit,
            F32SegmentInputs inputs,
            int batch,
            int queryIndex,
            int valueIndex
    ) {
        int index = inputs.outGradDense
                ? batch * unit.queryLen() * unit.valueDim() + queryIndex * unit.valueDim() + valueIndex
                : stridedOutGradOffset(inputs.outGradShape, inputs.outGradStrides, inputs.outGradStorageOffset,
                batch, queryIndex, valueIndex);
        return f32(inputs.outGrad, index);
    }

    private static double outGradF64(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64Inputs inputs,
            int batch,
            int queryIndex,
            int valueIndex
    ) {
        int index = inputs.outGradDense
                ? batch * unit.queryLen() * unit.valueDim() + queryIndex * unit.valueDim() + valueIndex
                : stridedOutGradOffset(inputs.outGradShape, inputs.outGradStrides, inputs.outGradStorageOffset,
                batch, queryIndex, valueIndex);
        return inputs.outGrad[index];
    }

    private static double outGradF64(
            Cpu1PreparedAttentionBackwardUnit unit,
            F64SegmentInputs inputs,
            int batch,
            int queryIndex,
            int valueIndex
    ) {
        int index = inputs.outGradDense
                ? batch * unit.queryLen() * unit.valueDim() + queryIndex * unit.valueDim() + valueIndex
                : stridedOutGradOffset(inputs.outGradShape, inputs.outGradStrides, inputs.outGradStorageOffset,
                batch, queryIndex, valueIndex);
        return f64(inputs.outGrad, index);
    }

    private static int stridedOutGradOffset(
            int[] shape,
            int[] strides,
            int storageOffset,
            int batch,
            int queryIndex,
            int valueIndex
    ) {
        int rank = shape.length;
        int offset = storageOffset
                + queryIndex * strides[rank - 2]
                + valueIndex * strides[rank - 1];
        int remaining = batch;
        for (int dim = rank - 3; dim >= 0; dim--) {
            int dimIndex = remaining % shape[dim];
            remaining /= shape[dim];
            offset += dimIndex * strides[dim];
        }
        return offset;
    }

    private static F32Inputs f32Inputs(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        Cpu1TensorView weights = inputArrayView("weights", unit.weightsNodeId(), unit.weightsShape(), unit.dataType(), context);
        Cpu1TensorView outGrad = outGradArrayView(unit, context);
        Cpu1TensorView query = unit.queryNodeId() >= 0
                ? inputArrayView("query", unit.queryNodeId(), unit.queryShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView key = unit.keyNodeId() >= 0
                ? inputArrayView("key", unit.keyNodeId(), unit.keyShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView value = unit.valueNodeId() >= 0
                ? inputArrayView("value", unit.valueNodeId(), unit.valueShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView mask = unit.hasMask()
                ? inputArrayView("mask", unit.maskNodeId(), unit.maskShape(), DataType.BOOL, context)
                : null;
        return new F32Inputs(
                weights.float32Array(),
                outGrad.float32Array(),
                unit.outGradShape(),
                unit.outGradAccessPlan().strides(),
                unit.outGradAccessPlan().storageOffset(),
                unit.outGradDenseContiguousNoOffset(),
                query == null ? null : query.float32Array(),
                key == null ? null : key.float32Array(),
                value == null ? null : value.float32Array(),
                mask == null ? null : mask.boolArray()
        );
    }

    private static F64Inputs f64Inputs(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        Cpu1TensorView weights = inputArrayView("weights", unit.weightsNodeId(), unit.weightsShape(), unit.dataType(), context);
        Cpu1TensorView outGrad = outGradArrayView(unit, context);
        Cpu1TensorView query = unit.queryNodeId() >= 0
                ? inputArrayView("query", unit.queryNodeId(), unit.queryShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView key = unit.keyNodeId() >= 0
                ? inputArrayView("key", unit.keyNodeId(), unit.keyShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView value = unit.valueNodeId() >= 0
                ? inputArrayView("value", unit.valueNodeId(), unit.valueShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView mask = unit.hasMask()
                ? inputArrayView("mask", unit.maskNodeId(), unit.maskShape(), DataType.BOOL, context)
                : null;
        return new F64Inputs(
                weights.float64Array(),
                outGrad.float64Array(),
                unit.outGradShape(),
                unit.outGradAccessPlan().strides(),
                unit.outGradAccessPlan().storageOffset(),
                unit.outGradDenseContiguousNoOffset(),
                query == null ? null : query.float64Array(),
                key == null ? null : key.float64Array(),
                value == null ? null : value.float64Array(),
                mask == null ? null : mask.boolArray()
        );
    }

    private static F32SegmentInputs f32SegmentInputs(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        Cpu1TensorView weights = inputSegmentView("weights", unit.weightsNodeId(), unit.weightsShape(), unit.dataType(), context);
        Cpu1TensorView outGrad = outGradSegmentView(unit, context);
        Cpu1TensorView query = unit.queryNodeId() >= 0
                ? inputSegmentView("query", unit.queryNodeId(), unit.queryShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView key = unit.keyNodeId() >= 0
                ? inputSegmentView("key", unit.keyNodeId(), unit.keyShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView value = unit.valueNodeId() >= 0
                ? inputSegmentView("value", unit.valueNodeId(), unit.valueShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView mask = unit.hasMask()
                ? inputSegmentView("mask", unit.maskNodeId(), unit.maskShape(), DataType.BOOL, context)
                : null;
        return new F32SegmentInputs(
                weights.segment(),
                outGrad.segment(),
                unit.outGradShape(),
                unit.outGradAccessPlan().strides(),
                unit.outGradAccessPlan().storageOffset(),
                unit.outGradDenseContiguousNoOffset(),
                query == null ? null : query.segment(),
                key == null ? null : key.segment(),
                value == null ? null : value.segment(),
                mask == null ? null : mask.segment()
        );
    }

    private static F64SegmentInputs f64SegmentInputs(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        Cpu1TensorView weights = inputSegmentView("weights", unit.weightsNodeId(), unit.weightsShape(), unit.dataType(), context);
        Cpu1TensorView outGrad = outGradSegmentView(unit, context);
        Cpu1TensorView query = unit.queryNodeId() >= 0
                ? inputSegmentView("query", unit.queryNodeId(), unit.queryShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView key = unit.keyNodeId() >= 0
                ? inputSegmentView("key", unit.keyNodeId(), unit.keyShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView value = unit.valueNodeId() >= 0
                ? inputSegmentView("value", unit.valueNodeId(), unit.valueShape(), unit.dataType(), context)
                : null;
        Cpu1TensorView mask = unit.hasMask()
                ? inputSegmentView("mask", unit.maskNodeId(), unit.maskShape(), DataType.BOOL, context)
                : null;
        return new F64SegmentInputs(
                weights.segment(),
                outGrad.segment(),
                unit.outGradShape(),
                unit.outGradAccessPlan().strides(),
                unit.outGradAccessPlan().storageOffset(),
                unit.outGradDenseContiguousNoOffset(),
                query == null ? null : query.segment(),
                key == null ? null : key.segment(),
                value == null ? null : value.segment(),
                mask == null ? null : mask.segment()
        );
    }

    private static Cpu1TensorView outGradArrayView(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        context.requireCpuReadable(unit.outGradNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Tensor tensor = context.runtimeTensorForNodeId(unit.outGradNodeId());
        if (tensor.getDataType() != unit.dataType()) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD outGrad dtype mismatch. expected="
                    + unit.dataType() + ", actual=" + tensor.getDataType());
        }
        Cpu1TensorView view = Cpu1TensorView.fromTensor(tensor);
        requireShape("outGrad", view.shape(), unit.outGradShape());
        requireOutGradRuntimeAccess(unit, view, Cpu1StorageKind.JAVA_ARRAY);
        return view;
    }

    private static Cpu1TensorView outGradSegmentView(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        NativeTensorStorage nativeStorage = context.requireNativeReadable(
                unit.outGradNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        Tensor tensor = context.runtimeTensorForNodeId(unit.outGradNodeId());
        if (tensor.getDataType() != unit.dataType()) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD outGrad dtype mismatch. expected="
                    + unit.dataType() + ", actual=" + tensor.getDataType());
        }
        Cpu1TensorView view = Cpu1TensorView.fromNativeStorage(tensor, nativeStorage);
        requireShape("outGrad", view.shape(), unit.outGradShape());
        requireOutGradRuntimeAccess(unit, view, Cpu1StorageKind.MEMORY_SEGMENT);
        return view;
    }

    private static Cpu1TensorView inputArrayView(
            String role,
            int nodeId,
            int[] expectedShape,
            DataType expectedDataType,
            ExecutionContext context
    ) {
        context.requireCpuReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != expectedDataType) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD " + role + " dtype mismatch. expected="
                    + expectedDataType + ", actual=" + tensor.getDataType());
        }
        Cpu1TensorView view = Cpu1TensorView.fromTensor(tensor);
        requireDenseNoOffset(role, view, Cpu1StorageKind.JAVA_ARRAY);
        requireShape(role, view.shape(), expectedShape);
        return view;
    }

    private static Cpu1TensorView inputSegmentView(
            String role,
            int nodeId,
            int[] expectedShape,
            DataType expectedDataType,
            ExecutionContext context
    ) {
        NativeTensorStorage nativeStorage = context.requireNativeReadable(
                nodeId,
                CpuMaterializationReason.CPU_CONSUMER
        );
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        if (tensor.getDataType() != expectedDataType) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD " + role + " dtype mismatch. expected="
                    + expectedDataType + ", actual=" + tensor.getDataType());
        }
        Cpu1TensorView view = Cpu1TensorView.fromNativeStorage(tensor, nativeStorage);
        requireDenseNoOffset(role, view, Cpu1StorageKind.MEMORY_SEGMENT);
        requireShape(role, view.shape(), expectedShape);
        return view;
    }

    private static Cpu1ScratchBuffer requireScratch(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context) {
        Cpu1ScratchBuffer scratchBuffer = context.cpu1ScratchBufferForNodeId(unit.nodeId());
        if (scratchBuffer == null) {
            throw new IllegalStateException("cpu1 SDPA_BACKWARD nodeId=" + unit.nodeId()
                    + " requires prepared score scratch buffer.");
        }
        return scratchBuffer;
    }

    private static void requireDenseNoOffset(String role, Cpu1TensorView view, Cpu1StorageKind expectedStorageKind) {
        if (view.storageKind() != expectedStorageKind) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD dense route supports only "
                    + expectedStorageKind + " " + role + " runtime storage, got " + view.storageKind());
        }
        if (!view.contiguous() || view.storageOffset() != 0) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD dense route supports only dense contiguous "
                    + "no-offset " + role + " runtime view; contiguous=" + view.contiguous()
                    + ", storageOffset=" + view.storageOffset());
        }
    }

    private static void requireOutGradRuntimeAccess(
            Cpu1PreparedAttentionBackwardUnit unit,
            Cpu1TensorView view,
            Cpu1StorageKind expectedStorageKind
    ) {
        Cpu1StorageAccessPlan plan = unit.outGradAccessPlan();
        if (view.storageKind() != expectedStorageKind) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD dense route supports only "
                    + expectedStorageKind + " outGrad runtime storage, got " + view.storageKind());
        }
        if (view.storageOffset() != plan.storageOffset()) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD outGrad runtime storage offset mismatch. "
                    + "expected=" + plan.storageOffset() + ", actual=" + view.storageOffset());
        }
        if (!Arrays.equals(view.strides(), plan.strides())) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD outGrad runtime strides mismatch. expected="
                    + Arrays.toString(plan.strides()) + ", actual=" + Arrays.toString(view.strides()));
        }
        if (unit.outGradDenseContiguousNoOffset()) {
            if (view.contiguous()) {
                return;
            }
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD prepared dense outGrad requires dense "
                    + "contiguous runtime view.");
        }
        if (unit.outGradBroadcastNoOffset()) {
            if (hasZeroStride(view.strides())) {
                return;
            }
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD prepared broadcast outGrad requires "
                    + "a zero-stride runtime view.");
        }
        throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD first route supports only dense contiguous "
                + "or explicit broadcast outGrad runtime view; preparedKind=" + plan.kind());
    }

    private static boolean hasZeroStride(int[] strides) {
        for (int stride : strides) {
            if (stride == 0) {
                return true;
            }
        }
        return false;
    }

    private static void requireShape(String role, int[] actual, int[] expected) {
        if (Arrays.equals(actual, expected)) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD " + role + " shape mismatch. expected="
                + Arrays.toString(expected) + ", actual=" + Arrays.toString(actual));
    }

    private static float dotF32Vector(float[] left, int leftBase, float[] right, int rightBase, int length) {
        FloatVector sum = FloatVector.zero(F32);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            FloatVector leftVector = FloatVector.fromArray(F32, left, leftBase + i);
            FloatVector rightVector = FloatVector.fromArray(F32, right, rightBase + i);
            sum = leftVector.fma(rightVector, sum);
        }
        float scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += left[leftBase + i] * right[rightBase + i];
        }
        return scalarSum;
    }

    private static double dotF64Vector(double[] left, int leftBase, double[] right, int rightBase, int length) {
        DoubleVector sum = DoubleVector.zero(F64);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            DoubleVector leftVector = DoubleVector.fromArray(F64, left, leftBase + i);
            DoubleVector rightVector = DoubleVector.fromArray(F64, right, rightBase + i);
            sum = leftVector.fma(rightVector, sum);
        }
        double scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += left[leftBase + i] * right[rightBase + i];
        }
        return scalarSum;
    }

    private static void accumulateF32Vector(
            float[] output,
            int outputBase,
            float[] input,
            int inputBase,
            float weight,
            int length
    ) {
        FloatVector weightVector = FloatVector.broadcast(F32, weight);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            FloatVector outputVector = FloatVector.fromArray(F32, output, outputBase + i);
            FloatVector inputVector = FloatVector.fromArray(F32, input, inputBase + i);
            inputVector.fma(weightVector, outputVector).intoArray(output, outputBase + i);
        }
        for (; i < length; i++) {
            output[outputBase + i] += input[inputBase + i] * weight;
        }
    }

    private static void accumulateF64Vector(
            double[] output,
            int outputBase,
            double[] input,
            int inputBase,
            double weight,
            int length
    ) {
        DoubleVector weightVector = DoubleVector.broadcast(F64, weight);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            DoubleVector outputVector = DoubleVector.fromArray(F64, output, outputBase + i);
            DoubleVector inputVector = DoubleVector.fromArray(F64, input, inputBase + i);
            inputVector.fma(weightVector, outputVector).intoArray(output, outputBase + i);
        }
        for (; i < length; i++) {
            output[outputBase + i] += input[inputBase + i] * weight;
        }
    }

    private static float dotF32SegmentVector(
            MemorySegment left,
            int leftBase,
            MemorySegment right,
            int rightBase,
            int length
    ) {
        FloatVector sum = FloatVector.zero(F32);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            FloatVector leftVector = FloatVector.fromMemorySegment(
                    F32,
                    left,
                    (long) (leftBase + i) * Float.BYTES,
                    ORDER
            );
            FloatVector rightVector = FloatVector.fromMemorySegment(
                    F32,
                    right,
                    (long) (rightBase + i) * Float.BYTES,
                    ORDER
            );
            sum = leftVector.fma(rightVector, sum);
        }
        float scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += f32(left, leftBase + i) * f32(right, rightBase + i);
        }
        return scalarSum;
    }

    private static double dotF64SegmentVector(
            MemorySegment left,
            int leftBase,
            MemorySegment right,
            int rightBase,
            int length
    ) {
        DoubleVector sum = DoubleVector.zero(F64);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            DoubleVector leftVector = DoubleVector.fromMemorySegment(
                    F64,
                    left,
                    (long) (leftBase + i) * Double.BYTES,
                    ORDER
            );
            DoubleVector rightVector = DoubleVector.fromMemorySegment(
                    F64,
                    right,
                    (long) (rightBase + i) * Double.BYTES,
                    ORDER
            );
            sum = leftVector.fma(rightVector, sum);
        }
        double scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += f64(left, leftBase + i) * f64(right, rightBase + i);
        }
        return scalarSum;
    }

    private static float dotF32ArraySegmentVector(
            float[] left,
            int leftBase,
            MemorySegment right,
            int rightBase,
            int length
    ) {
        FloatVector sum = FloatVector.zero(F32);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            FloatVector leftVector = FloatVector.fromArray(F32, left, leftBase + i);
            FloatVector rightVector = FloatVector.fromMemorySegment(
                    F32,
                    right,
                    (long) (rightBase + i) * Float.BYTES,
                    ORDER
            );
            sum = leftVector.fma(rightVector, sum);
        }
        float scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += left[leftBase + i] * f32(right, rightBase + i);
        }
        return scalarSum;
    }

    private static double dotF64ArraySegmentVector(
            double[] left,
            int leftBase,
            MemorySegment right,
            int rightBase,
            int length
    ) {
        DoubleVector sum = DoubleVector.zero(F64);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            DoubleVector leftVector = DoubleVector.fromArray(F64, left, leftBase + i);
            DoubleVector rightVector = DoubleVector.fromMemorySegment(
                    F64,
                    right,
                    (long) (rightBase + i) * Double.BYTES,
                    ORDER
            );
            sum = leftVector.fma(rightVector, sum);
        }
        double scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            scalarSum += left[leftBase + i] * f64(right, rightBase + i);
        }
        return scalarSum;
    }

    private static void accumulateF32SegmentVector(
            MemorySegment output,
            int outputBase,
            MemorySegment input,
            int inputBase,
            float weight,
            int length
    ) {
        FloatVector weightVector = FloatVector.broadcast(F32, weight);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            long outputOffset = (long) (outputBase + i) * Float.BYTES;
            long inputOffset = (long) (inputBase + i) * Float.BYTES;
            FloatVector outputVector = FloatVector.fromMemorySegment(F32, output, outputOffset, ORDER);
            FloatVector inputVector = FloatVector.fromMemorySegment(F32, input, inputOffset, ORDER);
            inputVector.fma(weightVector, outputVector).intoMemorySegment(output, outputOffset, ORDER);
        }
        for (; i < length; i++) {
            long outputOffset = (long) (outputBase + i) * Float.BYTES;
            float next = output.get(JAVA_FLOAT, outputOffset) + f32(input, inputBase + i) * weight;
            output.set(JAVA_FLOAT, outputOffset, next);
        }
    }

    private static void accumulateF64SegmentVector(
            MemorySegment output,
            int outputBase,
            MemorySegment input,
            int inputBase,
            double weight,
            int length
    ) {
        DoubleVector weightVector = DoubleVector.broadcast(F64, weight);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            long outputOffset = (long) (outputBase + i) * Double.BYTES;
            long inputOffset = (long) (inputBase + i) * Double.BYTES;
            DoubleVector outputVector = DoubleVector.fromMemorySegment(F64, output, outputOffset, ORDER);
            DoubleVector inputVector = DoubleVector.fromMemorySegment(F64, input, inputOffset, ORDER);
            inputVector.fma(weightVector, outputVector).intoMemorySegment(output, outputOffset, ORDER);
        }
        for (; i < length; i++) {
            long outputOffset = (long) (outputBase + i) * Double.BYTES;
            double next = output.get(JAVA_DOUBLE, outputOffset) + f64(input, inputBase + i) * weight;
            output.set(JAVA_DOUBLE, outputOffset, next);
        }
    }

    private static void zeroF32Segment(MemorySegment output, int outputBase, int length) {
        FloatVector zero = FloatVector.zero(F32);
        int i = 0;
        int upper = F32.loopBound(length);
        for (; i < upper; i += F32.length()) {
            zero.intoMemorySegment(output, (long) (outputBase + i) * Float.BYTES, ORDER);
        }
        for (; i < length; i++) {
            setF32(output, outputBase + i, 0.0f);
        }
    }

    private static void zeroF64Segment(MemorySegment output, int outputBase, int length) {
        DoubleVector zero = DoubleVector.zero(F64);
        int i = 0;
        int upper = F64.loopBound(length);
        for (; i < upper; i += F64.length()) {
            zero.intoMemorySegment(output, (long) (outputBase + i) * Double.BYTES, ORDER);
        }
        for (; i < length; i++) {
            setF64(output, outputBase + i, 0.0d);
        }
    }

    private static float f32(MemorySegment segment, int index) {
        return segment.get(JAVA_FLOAT, (long) index * Float.BYTES);
    }

    private static void setF32(MemorySegment segment, int index, float value) {
        segment.set(JAVA_FLOAT, (long) index * Float.BYTES, value);
    }

    private static double f64(MemorySegment segment, int index) {
        return segment.get(JAVA_DOUBLE, (long) index * Double.BYTES);
    }

    private static void setF64(MemorySegment segment, int index, double value) {
        segment.set(JAVA_DOUBLE, (long) index * Double.BYTES, value);
    }

    private static byte bool(MemorySegment segment, int index) {
        return segment.get(JAVA_BYTE, index);
    }

    private record F32Inputs(
            float[] weights,
            float[] outGrad,
            int[] outGradShape,
            int[] outGradStrides,
            int outGradStorageOffset,
            boolean outGradDense,
            float[] query,
            float[] key,
            float[] value,
            byte[] mask
    ) {
    }

    private record F64Inputs(
            double[] weights,
            double[] outGrad,
            int[] outGradShape,
            int[] outGradStrides,
            int outGradStorageOffset,
            boolean outGradDense,
            double[] query,
            double[] key,
            double[] value,
            byte[] mask
    ) {
    }

    private record F32SegmentInputs(
            MemorySegment weights,
            MemorySegment outGrad,
            int[] outGradShape,
            int[] outGradStrides,
            int outGradStorageOffset,
            boolean outGradDense,
            MemorySegment query,
            MemorySegment key,
            MemorySegment value,
            MemorySegment mask
    ) {
    }

    private record F64SegmentInputs(
            MemorySegment weights,
            MemorySegment outGrad,
            int[] outGradShape,
            int[] outGradStrides,
            int outGradStorageOffset,
            boolean outGradDense,
            MemorySegment query,
            MemorySegment key,
            MemorySegment value,
            MemorySegment mask
    ) {
    }
}
