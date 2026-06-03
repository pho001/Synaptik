package backend.cpu1.kernels.matmul;

import backend.blas.OpenBlasRuntime;
import backend.blas.OpenBlasSegmentGemm;
import backend.cpu1.exec.Cpu1Workspace;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.DataType;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

/**
 * OpenBLAS native MemorySegment matmul route for dense cpu1 storage.
 */
public final class Cpu1OpenBlasNativeSegmentMatmulLoops {
    private Cpu1OpenBlasNativeSegmentMatmulLoops() {
    }

    public static void matmulF32NativeSegment(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        if (!OpenBlasRuntime.isFloat32GemmAvailable()) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT F32 MATMUL requires OpenBLAS sgemm: "
                    + OpenBlasRuntime.unavailableReason());
        }
        NativeTensorStorage left = inputStorage(unit.leftNodeId(), DataType.FLOAT32, context);
        NativeTensorStorage right = inputStorage(unit.rightNodeId(), DataType.FLOAT32, context);
        NativeTensorStorage outputStorage = outputStorage(unit, context);
        runF32(left, right, outputStorage, unit);
        markOutputWritten(unit, outputStorage, context);
    }

    public static void matmulF64NativeSegment(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        if (!OpenBlasRuntime.isFloat64GemmAvailable()) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT F64 MATMUL requires OpenBLAS dgemm: "
                    + OpenBlasRuntime.unavailableReason());
        }
        NativeTensorStorage left = inputStorage(unit.leftNodeId(), DataType.FLOAT64, context);
        NativeTensorStorage right = inputStorage(unit.rightNodeId(), DataType.FLOAT64, context);
        NativeTensorStorage outputStorage = outputStorage(unit, context);
        runF64(left, right, outputStorage, unit);
        markOutputWritten(unit, outputStorage, context);
    }

    private static void runF32(
            NativeTensorStorage left,
            NativeTensorStorage right,
            NativeTensorStorage output,
            Cpu1PreparedMatmulUnit unit
    ) {
        int m = unit.m();
        int n = unit.n();
        int k = unit.k();
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outputSegment = output.segment();
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            OpenBlasSegmentGemm.sgemmRowMajorNoTransSegment(
                    m,
                    n,
                    k,
                    1.0f,
                    leftSegment,
                    byteOffset(unit.leftBatchOffset(batch), Float.BYTES),
                    k,
                    rightSegment,
                    byteOffset(unit.rightBatchOffset(batch), Float.BYTES),
                    n,
                    0.0f,
                    outputSegment,
                    byteOffset(unit.outputBatchOffset(batch), Float.BYTES),
                    n
            );
        }
    }

    private static void runF64(
            NativeTensorStorage left,
            NativeTensorStorage right,
            NativeTensorStorage output,
            Cpu1PreparedMatmulUnit unit
    ) {
        int m = unit.m();
        int n = unit.n();
        int k = unit.k();
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outputSegment = output.segment();
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            OpenBlasSegmentGemm.dgemmRowMajorNoTransSegment(
                    m,
                    n,
                    k,
                    1.0d,
                    leftSegment,
                    byteOffset(unit.leftBatchOffset(batch), Double.BYTES),
                    k,
                    rightSegment,
                    byteOffset(unit.rightBatchOffset(batch), Double.BYTES),
                    n,
                    0.0d,
                    outputSegment,
                    byteOffset(unit.outputBatchOffset(batch), Double.BYTES),
                    n
            );
        }
    }

    private static NativeTensorStorage inputStorage(int nodeId, DataType dataType, ExecutionContext context) {
        NativeTensorStorage nativeInput = context.requireNativeReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (nativeInput.getType() != dataType) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT MATMUL requires " + dataType
                    + " native input storage, got " + nativeInput.getType());
        }
        return nativeInput;
    }

    private static NativeTensorStorage outputStorage(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Cpu1Workspace workspace = context.cpu1WorkspaceForNodeId(unit.nodeId());
        if (workspace == null) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT MATMUL nodeId=" + unit.nodeId()
                    + " requires prepared native output workspace.");
        }
        NativeTensorStorage output = workspace.requireNativeOutputStorage(
                unit.dataType(),
                Math.multiplyExact(Math.multiplyExact(unit.batchCount(), unit.m()), unit.n()),
                unit.nodeId(),
                context,
                "cpu1-node-" + unit.nodeId() + ":openblas-native-segment"
        );
        if (output.getType() != unit.dataType()) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT MATMUL allocated wrong output dtype. expected="
                    + unit.dataType() + ", actual=" + output.getType());
        }
        return output;
    }

    private static void markOutputWritten(
            Cpu1PreparedMatmulUnit unit,
            NativeTensorStorage outputStorage,
            ExecutionContext context
    ) {
        outputStorage.markModified();
        context.attachNativeStorage(unit.nodeId(), outputStorage, "cpu1 OPENBLAS_NATIVE_SEGMENT matmul wrote native CPU segment");
    }

    private static long byteOffset(int batchOffset, int elementBytes) {
        return Math.multiplyExact((long) batchOffset, elementBytes);
    }
}
