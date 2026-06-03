package backend.cpu1.kernels.matmul;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.exec.Cpu1Workspace;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1MatmulPostOp;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tensor.Tensor;

/**
 * Java Vector API matmul loops for dense cpu1 array storage.
 */
public final class Cpu1JavaVectorMatmulLoops {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;

    private Cpu1JavaVectorMatmulLoops() {
    }

    public static void matmulF32DensePackedBVector(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Cpu1TensorView left = inputView(unit.leftNodeId(), context);
        Cpu1TensorView right = inputView(unit.rightNodeId(), context);
        Cpu1TensorView bias = unit.hasBias() ? inputView(unit.biasNodeId(), context) : null;
        Cpu1TensorView output = outputView(unit, context);
        float[] packedB = packedBWorkspace(unit, context);

        packBColumns(right.float32Array(), packedB, right.storageOffset(), unit);
        runPackedB(
                left.float32Array(),
                packedB,
                bias == null ? null : bias.float32Array(),
                output.float32Array(),
                left.storageOffset(),
                bias == null ? 0 : bias.storageOffset(),
                output.storageOffset(),
                unit
        );
        markOutputWritten(unit, output, context);
    }

    public static void matmulF64DensePackedBVector(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Cpu1TensorView left = inputView(unit.leftNodeId(), context);
        Cpu1TensorView right = inputView(unit.rightNodeId(), context);
        Cpu1TensorView bias = unit.hasBias() ? inputView(unit.biasNodeId(), context) : null;
        Cpu1TensorView output = outputView(unit, context);
        double[] packedB = packedBF64Workspace(unit, context);

        packBF64Columns(right.float64Array(), packedB, right.storageOffset(), unit);
        runPackedBF64(
                left.float64Array(),
                packedB,
                bias == null ? null : bias.float64Array(),
                output.float64Array(),
                left.storageOffset(),
                bias == null ? 0 : bias.storageOffset(),
                output.storageOffset(),
                unit
        );
        markOutputWritten(unit, output, context);
    }

    private static void packBColumns(
            float[] right,
            float[] packedB,
            int rightStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        int n = unit.n();
        int k = unit.k();
        int rightRowStride = unit.rightRowStride();
        int rightColStride = unit.rightColStride();
        int batchPackedSize = Math.multiplyExact(n, k);
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            int rightBatchBase = rightStorageOffset + unit.rightBatchOffset(batch);
            int packedBatchBase = batch * batchPackedSize;
            for (int col = 0; col < n; col++) {
                int rightColBase = rightBatchBase + col * rightColStride;
                int packedColBase = packedBatchBase + col * k;
                for (int index = 0; index < k; index++) {
                    packedB[packedColBase + index] = right[rightColBase + index * rightRowStride];
                }
            }
        }
    }

    private static void packBF64Columns(
            double[] right,
            double[] packedB,
            int rightStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        int n = unit.n();
        int k = unit.k();
        int rightRowStride = unit.rightRowStride();
        int rightColStride = unit.rightColStride();
        int batchPackedSize = Math.multiplyExact(n, k);
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            int rightBatchBase = rightStorageOffset + unit.rightBatchOffset(batch);
            int packedBatchBase = batch * batchPackedSize;
            for (int col = 0; col < n; col++) {
                int rightColBase = rightBatchBase + col * rightColStride;
                int packedColBase = packedBatchBase + col * k;
                for (int index = 0; index < k; index++) {
                    packedB[packedColBase + index] = right[rightColBase + index * rightRowStride];
                }
            }
        }
    }

    private static void runPackedB(
            float[] left,
            float[] packedB,
            float[] bias,
            float[] output,
            int leftStorageOffset,
            int biasStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        int m = unit.m();
        int n = unit.n();
        int k = unit.k();
        int leftRowStride = unit.leftRowStride();
        int outputRowStride = unit.outputRowStride();
        int outputColStride = unit.outputColStride();
        int batchPackedSize = Math.multiplyExact(n, k);
        int outputRows = Math.multiplyExact(unit.batchCount(), m);
        int vectorUpper = F32.loopBound(k);
        Cpu1MatmulPostOp postOp = unit.postOp();
        boolean hasBias = unit.hasBias();

        Cpu1RangeLauncher.launch(outputRows, unit.launchConfig(), (startRow, endRow) -> {
            for (int rowIndex = startRow; rowIndex < endRow; rowIndex++) {
                int batch = rowIndex / m;
                int row = rowIndex % m;
                int leftBatchBase = leftStorageOffset + unit.leftBatchOffset(batch);
                int outputBatchBase = outputStorageOffset + unit.outputBatchOffset(batch);
                int packedBatchBase = batch * batchPackedSize;
                int leftRowBase = leftBatchBase + row * leftRowStride;
                int outputRowBase = outputBatchBase + row * outputRowStride;
                for (int col = 0; col < n; col++) {
                    int packedColBase = packedBatchBase + col * k;
                    float sum = dotContiguousF32(left, leftRowBase, packedB, packedColBase, k, vectorUpper);
                    if (hasBias) {
                        int biasIndex = biasStorageOffset + unit.biasBatchOffset(batch)
                                + row * unit.biasRowStride()
                                + col * unit.biasColStride();
                        output[outputRowBase + col * outputColStride] = postOp.apply(sum, bias[biasIndex]);
                    } else {
                        output[outputRowBase + col * outputColStride] = postOp.apply(sum);
                    }
                }
            }
        });
    }

    private static void runPackedBF64(
            double[] left,
            double[] packedB,
            double[] bias,
            double[] output,
            int leftStorageOffset,
            int biasStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        int m = unit.m();
        int n = unit.n();
        int k = unit.k();
        int leftRowStride = unit.leftRowStride();
        int outputRowStride = unit.outputRowStride();
        int outputColStride = unit.outputColStride();
        int batchPackedSize = Math.multiplyExact(n, k);
        int outputRows = Math.multiplyExact(unit.batchCount(), m);
        int vectorUpper = F64.loopBound(k);
        Cpu1MatmulPostOp postOp = unit.postOp();
        boolean hasBias = unit.hasBias();

        Cpu1RangeLauncher.launch(outputRows, unit.launchConfig(), (startRow, endRow) -> {
            for (int rowIndex = startRow; rowIndex < endRow; rowIndex++) {
                int batch = rowIndex / m;
                int row = rowIndex % m;
                int leftBatchBase = leftStorageOffset + unit.leftBatchOffset(batch);
                int outputBatchBase = outputStorageOffset + unit.outputBatchOffset(batch);
                int packedBatchBase = batch * batchPackedSize;
                int leftRowBase = leftBatchBase + row * leftRowStride;
                int outputRowBase = outputBatchBase + row * outputRowStride;
                for (int col = 0; col < n; col++) {
                    int packedColBase = packedBatchBase + col * k;
                    double sum = dotContiguousF64(left, leftRowBase, packedB, packedColBase, k, vectorUpper);
                    if (hasBias) {
                        int biasIndex = biasStorageOffset + unit.biasBatchOffset(batch)
                                + row * unit.biasRowStride()
                                + col * unit.biasColStride();
                        output[outputRowBase + col * outputColStride] = postOp.apply(sum, bias[biasIndex]);
                    } else {
                        output[outputRowBase + col * outputColStride] = postOp.apply(sum);
                    }
                }
            }
        });
    }

    private static float dotContiguousF32(
            float[] left,
            int leftBase,
            float[] packedB,
            int packedBase,
            int k,
            int vectorUpper
    ) {
        FloatVector sum = FloatVector.zero(F32);
        int index = 0;
        for (; index < vectorUpper; index += F32.length()) {
            FloatVector leftVector = FloatVector.fromArray(F32, left, leftBase + index);
            FloatVector rightVector = FloatVector.fromArray(F32, packedB, packedBase + index);
            sum = leftVector.fma(rightVector, sum);
        }
        float scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; index < k; index++) {
            scalarSum += left[leftBase + index] * packedB[packedBase + index];
        }
        return scalarSum;
    }

    private static double dotContiguousF64(
            double[] left,
            int leftBase,
            double[] packedB,
            int packedBase,
            int k,
            int vectorUpper
    ) {
        DoubleVector sum = DoubleVector.zero(F64);
        int index = 0;
        for (; index < vectorUpper; index += F64.length()) {
            DoubleVector leftVector = DoubleVector.fromArray(F64, left, leftBase + index);
            DoubleVector rightVector = DoubleVector.fromArray(F64, packedB, packedBase + index);
            sum = leftVector.fma(rightVector, sum);
        }
        double scalarSum = sum.reduceLanes(VectorOperators.ADD);
        for (; index < k; index++) {
            scalarSum += left[leftBase + index] * packedB[packedBase + index];
        }
        return scalarSum;
    }

    private static float[] packedBWorkspace(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Cpu1Workspace workspace = context.cpu1WorkspaceForNodeId(unit.nodeId());
        if (workspace == null) {
            throw new IllegalStateException("cpu1 packed-B vector MATMUL requires prepared F32 workspace for nodeId="
                    + unit.nodeId());
        }
        return workspace.requireF32Array(Math.toIntExact(Math.multiplyExact(
                Math.multiplyExact((long) unit.batchCount(), unit.n()),
                unit.k()
        )));
    }

    private static double[] packedBF64Workspace(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Cpu1Workspace workspace = context.cpu1WorkspaceForNodeId(unit.nodeId());
        if (workspace == null) {
            throw new IllegalStateException("cpu1 packed-B vector MATMUL requires prepared F64 workspace for nodeId="
                    + unit.nodeId());
        }
        return workspace.requireF64Array(Math.toIntExact(Math.multiplyExact(
                Math.multiplyExact((long) unit.batchCount(), unit.n()),
                unit.k()
        )));
    }

    private static Cpu1TensorView inputView(int nodeId, ExecutionContext context) {
        context.requireCpuReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(nodeId);
        return Cpu1TensorView.fromTensor(input);
    }

    private static Cpu1TensorView outputView(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static void markOutputWritten(
            Cpu1PreparedMatmulUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.route() + " packed-B vector matmul wrote CPU array");
    }
}
