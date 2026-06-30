package backend.cpu1.kernels.matmul;

import backend.blas.OpenBlasArrayGemm;
import backend.blas.OpenBlasRuntime;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import runtime.contract.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;

/**
 * OpenBLAS array-copy matmul route for dense cpu1 Java-array storage.
 */
public final class Cpu1OpenBlasArrayMatmulLoops {
    private Cpu1OpenBlasArrayMatmulLoops() {
    }

    public static void matmulF32ArrayCopy(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        if (!OpenBlasRuntime.isFloat32GemmAvailable()) {
            throw new IllegalStateException("cpu1 OPENBLAS_ARRAY_COPYING F32 MATMUL requires OpenBLAS sgemm: "
                    + OpenBlasRuntime.unavailableReason());
        }
        Cpu1TensorView left = inputView(unit.leftNodeId(), context);
        Cpu1TensorView right = inputView(unit.rightNodeId(), context);
        Cpu1TensorView output = outputView(unit, context);
        runF32(
                left.float32Array(),
                right.float32Array(),
                output.float32Array(),
                left.storageOffset(),
                right.storageOffset(),
                output.storageOffset(),
                unit
        );
        markOutputWritten(unit, output, context);
    }

    public static void matmulF64ArrayCopy(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        if (!OpenBlasRuntime.isFloat64GemmAvailable()) {
            throw new IllegalStateException("cpu1 OPENBLAS_ARRAY_COPYING F64 MATMUL requires OpenBLAS dgemm: "
                    + OpenBlasRuntime.unavailableReason());
        }
        Cpu1TensorView left = inputView(unit.leftNodeId(), context);
        Cpu1TensorView right = inputView(unit.rightNodeId(), context);
        Cpu1TensorView output = outputView(unit, context);
        runF64(
                left.float64Array(),
                right.float64Array(),
                output.float64Array(),
                left.storageOffset(),
                right.storageOffset(),
                output.storageOffset(),
                unit
        );
        markOutputWritten(unit, output, context);
    }

    private static void runF32(
            float[] left,
            float[] right,
            float[] output,
            int leftStorageOffset,
            int rightStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        int m = unit.m();
        int n = unit.n();
        int k = unit.k();
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            OpenBlasArrayGemm.sgemmRowMajorNoTransOffsets(
                    m,
                    n,
                    k,
                    1.0f,
                    left,
                    leftStorageOffset + unit.leftBatchOffset(batch),
                    k,
                    right,
                    rightStorageOffset + unit.rightBatchOffset(batch),
                    n,
                    0.0f,
                    output,
                    outputStorageOffset + unit.outputBatchOffset(batch),
                    n
            );
        }
    }

    private static void runF64(
            double[] left,
            double[] right,
            double[] output,
            int leftStorageOffset,
            int rightStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        int m = unit.m();
        int n = unit.n();
        int k = unit.k();
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            OpenBlasArrayGemm.dgemmRowMajorNoTransOffsets(
                    m,
                    n,
                    k,
                    1.0d,
                    left,
                    leftStorageOffset + unit.leftBatchOffset(batch),
                    k,
                    right,
                    rightStorageOffset + unit.rightBatchOffset(batch),
                    n,
                    0.0d,
                    output,
                    outputStorageOffset + unit.outputBatchOffset(batch),
                    n
            );
        }
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
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.route() + " matmul wrote CPU array");
    }
}
