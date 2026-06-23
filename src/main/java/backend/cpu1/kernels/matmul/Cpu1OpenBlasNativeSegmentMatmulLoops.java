package backend.cpu1.kernels.matmul;

import backend.blas.OpenBlasRuntime;
import backend.blas.OpenBlasSegmentGemm;
import backend.cpu1.prepare.Cpu1MatmulPostOp;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.DataType;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

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
        NativeTensorStorage left = inputStorage("left", unit.leftNodeId(), DataType.FLOAT32, context);
        NativeTensorStorage right = inputStorage("right", unit.rightNodeId(), DataType.FLOAT32, context);
        NativeTensorStorage bias = unit.hasBias()
                ? inputStorage("bias", unit.biasNodeId(), DataType.FLOAT32, context)
                : null;
        NativeTensorStorage outputStorage = outputStorage(unit, context);
        runF32(left, right, bias, outputStorage, unit);
        markOutputWritten(unit, outputStorage, context);
    }

    public static void matmulF64NativeSegment(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        if (!OpenBlasRuntime.isFloat64GemmAvailable()) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT F64 MATMUL requires OpenBLAS dgemm: "
                    + OpenBlasRuntime.unavailableReason());
        }
        NativeTensorStorage left = inputStorage("left", unit.leftNodeId(), DataType.FLOAT64, context);
        NativeTensorStorage right = inputStorage("right", unit.rightNodeId(), DataType.FLOAT64, context);
        NativeTensorStorage bias = unit.hasBias()
                ? inputStorage("bias", unit.biasNodeId(), DataType.FLOAT64, context)
                : null;
        NativeTensorStorage outputStorage = outputStorage(unit, context);
        runF64(left, right, bias, outputStorage, unit);
        markOutputWritten(unit, outputStorage, context);
    }

    private static void runF32(
            NativeTensorStorage left,
            NativeTensorStorage right,
            NativeTensorStorage bias,
            NativeTensorStorage output,
            Cpu1PreparedMatmulUnit unit
    ) {
        int m = unit.m();
        int n = unit.n();
        int k = unit.k();
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment biasSegment = bias == null ? null : bias.segment();
        MemorySegment outputSegment = output.segment();
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            int outputBatchOffset = unit.outputBatchOffset(batch);
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
                    byteOffset(outputBatchOffset, Float.BYTES),
                    n
            );
            applyF32Epilogue(outputSegment, biasSegment, outputBatchOffset, batch, unit);
        }
    }

    private static void runF64(
            NativeTensorStorage left,
            NativeTensorStorage right,
            NativeTensorStorage bias,
            NativeTensorStorage output,
            Cpu1PreparedMatmulUnit unit
    ) {
        int m = unit.m();
        int n = unit.n();
        int k = unit.k();
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment biasSegment = bias == null ? null : bias.segment();
        MemorySegment outputSegment = output.segment();
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            int outputBatchOffset = unit.outputBatchOffset(batch);
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
                    byteOffset(outputBatchOffset, Double.BYTES),
                    n
            );
            applyF64Epilogue(outputSegment, biasSegment, outputBatchOffset, batch, unit);
        }
    }

    private static void applyF32Epilogue(
            MemorySegment outputSegment,
            MemorySegment biasSegment,
            int outputBatchOffset,
            int batch,
            Cpu1PreparedMatmulUnit unit
    ) {
        Cpu1MatmulPostOp postOp = unit.postOp();
        if (postOp == Cpu1MatmulPostOp.NONE) {
            return;
        }
        if (!unit.hasBias() || biasSegment == null) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT " + postOp + " requires native bias storage.");
        }
        int biasBatchOffset = unit.biasBatchOffset(batch);
        for (int row = 0; row < unit.m(); row++) {
            int outputRowBase = outputBatchOffset + row * unit.outputRowStride();
            int biasRowBase = biasBatchOffset + row * unit.biasRowStride();
            for (int col = 0; col < unit.n(); col++) {
                int outputIndex = outputRowBase + col * unit.outputColStride();
                int biasIndex = biasRowBase + col * unit.biasColStride();
                float value = outputSegment.get(JAVA_FLOAT, byteOffset(outputIndex, Float.BYTES));
                float bias = biasSegment.get(JAVA_FLOAT, byteOffset(biasIndex, Float.BYTES));
                outputSegment.set(JAVA_FLOAT, byteOffset(outputIndex, Float.BYTES), postOp.apply(value, bias));
            }
        }
    }

    private static void applyF64Epilogue(
            MemorySegment outputSegment,
            MemorySegment biasSegment,
            int outputBatchOffset,
            int batch,
            Cpu1PreparedMatmulUnit unit
    ) {
        Cpu1MatmulPostOp postOp = unit.postOp();
        if (postOp == Cpu1MatmulPostOp.NONE) {
            return;
        }
        if (!unit.hasBias() || biasSegment == null) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT " + postOp + " requires native bias storage.");
        }
        int biasBatchOffset = unit.biasBatchOffset(batch);
        for (int row = 0; row < unit.m(); row++) {
            int outputRowBase = outputBatchOffset + row * unit.outputRowStride();
            int biasRowBase = biasBatchOffset + row * unit.biasRowStride();
            for (int col = 0; col < unit.n(); col++) {
                int outputIndex = outputRowBase + col * unit.outputColStride();
                int biasIndex = biasRowBase + col * unit.biasColStride();
                double value = outputSegment.get(JAVA_DOUBLE, byteOffset(outputIndex, Double.BYTES));
                double bias = biasSegment.get(JAVA_DOUBLE, byteOffset(biasIndex, Double.BYTES));
                outputSegment.set(JAVA_DOUBLE, byteOffset(outputIndex, Double.BYTES), postOp.apply(value, bias));
            }
        }
    }

    private static NativeTensorStorage inputStorage(
            String role,
            int nodeId,
            DataType dataType,
            ExecutionContext context
    ) {
        NativeTensorStorage nativeInput = context.requireNativeReadable(
                nodeId,
                CpuMaterializationReason.CPU_CONSUMER
        );
        if (nativeInput.getType() != dataType) {
            throw new IllegalStateException("cpu1 OPENBLAS_NATIVE_SEGMENT MATMUL requires " + dataType
                    + " native " + role + " storage, got " + nativeInput.getType());
        }
        return nativeInput;
    }

    private static NativeTensorStorage outputStorage(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        NativeTensorStorage output = context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                Math.multiplyExact(Math.multiplyExact(unit.batchCount(), unit.m()), unit.n()),
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
