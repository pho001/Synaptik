package backend.cpu.nativecpu.layout;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.execution.CpuThreadPool;
import backend.cpu.kernels.elementwise.where.CpuWhereKernel;
import backend.cpu.kernels.elementwise.unary.support.CpuPowSupport;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import tensor.DataType;
import utils.FastTranscendentals;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.DoubleAdder;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Correctness-first MemorySegment strided kernels.
 *
 * <p>These loops are dtype-specialized and intentionally separate from planner selection. They provide the
 * physical segment access needed by native CPU partition planning, but scalar strided variants remain guarded by
 * performance policy before they can become an AUTO hot path.</p>
 */
public final class NativeSegmentStridedKernels {
    private NativeSegmentStridedKernels() {
    }

    public static boolean supportsUnary(Operation op, DataType dataType) {
        if (op == null) {
            return false;
        }
        return switch (Objects.requireNonNull(dataType, "dataType cannot be null")) {
            case FLOAT32 -> isF32Unary(op.opType());
            case FLOAT64 -> isF64Unary(op.opType());
            case BFLOAT16 -> isBF16Unary(op.opType());
            case BOOL -> op.opType() == Operation.OpType.LOGICAL_NOT;
            case INT32, INT64 -> false;
        };
    }

    public static boolean supportsBinary(Operation op, DataType dataType) {
        if (op == null) {
            return false;
        }
        if (dataType == DataType.BFLOAT16 && op.opType() == Operation.OpType.POW_TENSOR) {
            return false;
        }
        if (dataType == DataType.BOOL) {
            return op.opType() == Operation.OpType.LOGICAL_AND
                    || op.opType() == Operation.OpType.LOGICAL_OR;
        }
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16)
                && isBinary(op.opType());
    }

    public static boolean supportsCompare(Operation op, DataType dataType) {
        if (op == null) {
            return false;
        }
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16)
                && isCompare(op.opType());
    }

    public static void runUnary(
            Operation op,
            NativeSegmentView input,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        Objects.requireNonNull(op, "op cannot be null");
        validateSameShapeDenseWrite(input, output, "unary");
        if (!supportsUnary(op, input.physicalView().dataType())) {
            throw new UnsupportedOperationException("Unsupported native segment strided unary opType="
                    + op.opType() + ", dtype=" + input.physicalView().dataType());
        }
        switch (input.physicalView().dataType()) {
            case FLOAT32 -> runUnaryF32(op, input, output, useFastExpApprox, useFastTanhApprox);
            case FLOAT64 -> runUnaryF64(op, input, output, useFastExpApprox, useFastTanhApprox);
            case BFLOAT16 -> runUnaryBF16(op, input, output);
            case BOOL -> runUnaryBool(op, input, output);
            case INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment unary dtype="
                    + input.physicalView().dataType());
        }
        output.storage().markModified();
    }

    public static NativeSegmentDispatchResult runUnaryWithDispatch(
            Operation op,
            NativeSegmentView input,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            NativeSegmentDispatchConfig dispatchConfig
    ) {
        NativeSegmentDispatchConfig config = dispatchConfig == null ? NativeSegmentDispatchConfig.scalar() : dispatchConfig;
        Objects.requireNonNull(op, "op cannot be null");
        validateSameShapeDenseWrite(input, output, "unary");
        int size = logicalSize(input);
        if (!supportsUnary(op, input.physicalView().dataType())) {
            throw new UnsupportedOperationException("Unsupported native segment strided unary opType="
                    + op.opType() + ", dtype=" + input.physicalView().dataType());
        }
        if (!supportsParallelUnary(input.physicalView().dataType()) || !config.parallelEligible(size)) {
            runUnary(op, input, output, useFastExpApprox, useFastTanhApprox);
            return NativeSegmentDispatchResult.scalar(scalarFamily(input), size, scalarReason(config, size));
        }
        int chunks = config.chunks(size);
        switch (input.physicalView().dataType()) {
            case FLOAT32 -> parallelUnaryF32(op, input, output, useFastExpApprox, useFastTanhApprox, config, chunks);
            case FLOAT64 -> parallelUnaryF64(op, input, output, useFastExpApprox, useFastTanhApprox, config, chunks);
            case BFLOAT16, BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment parallel unary dtype="
                    + input.physicalView().dataType());
        }
        output.storage().markModified();
        return NativeSegmentDispatchResult.parallel(NativeSegmentKernelFamily.SEGMENT_PARALLEL, size, chunks, "segment-parallel-unary");
    }

    public static void copyRaw(NativeSegmentView input, NativeSegmentView output) {
        validateSameShapeDenseWrite(input, output, "copy");
        switch (input.physicalView().dataType()) {
            case FLOAT32 -> copyF32(input, output);
            case FLOAT64 -> copyF64(input, output);
            case BFLOAT16 -> copyBF16(input, output);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment raw copy dtype="
                    + input.physicalView().dataType());
        }
        output.storage().markModified();
    }

    public static void runBinary(
            Operation op,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output
    ) {
        Objects.requireNonNull(op, "op cannot be null");
        validateBinaryDenseWrite(left, right, output, "binary");
        if (!supportsBinary(op, output.physicalView().dataType())) {
            throw new UnsupportedOperationException("Unsupported native segment strided binary opType="
                    + op.opType() + ", dtype=" + output.physicalView().dataType());
        }
        switch (output.physicalView().dataType()) {
            case FLOAT32 -> runBinaryF32(op, left, right, output);
            case FLOAT64 -> runBinaryF64(op, left, right, output);
            case BFLOAT16 -> runBinaryBF16(op, left, right, output);
            case BOOL -> runBinaryBool(op, left, right, output);
            case INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment binary dtype="
                    + output.physicalView().dataType());
        }
        output.storage().markModified();
    }

    public static NativeSegmentDispatchResult runBinaryWithDispatch(
            Operation op,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output,
            NativeSegmentDispatchConfig dispatchConfig
    ) {
        NativeSegmentDispatchConfig config = dispatchConfig == null ? NativeSegmentDispatchConfig.scalar() : dispatchConfig;
        Objects.requireNonNull(op, "op cannot be null");
        validateBinaryDenseWrite(left, right, output, "binary");
        int size = logicalSize(output);
        if (!supportsBinary(op, output.physicalView().dataType())) {
            throw new UnsupportedOperationException("Unsupported native segment strided binary opType="
                    + op.opType() + ", dtype=" + output.physicalView().dataType());
        }
        if (!supportsParallelBinary(output.physicalView().dataType()) || !config.parallelEligible(size)) {
            runBinary(op, left, right, output);
            return NativeSegmentDispatchResult.scalar(scalarFamily(left, right), size, scalarReason(config, size));
        }
        int chunks = config.chunks(size);
        switch (output.physicalView().dataType()) {
            case FLOAT32 -> parallelBinaryF32(op, left, right, output, config, chunks);
            case FLOAT64 -> parallelBinaryF64(op, left, right, output, config, chunks);
            case BFLOAT16, BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment parallel binary dtype="
                    + output.physicalView().dataType());
        }
        output.storage().markModified();
        return NativeSegmentDispatchResult.parallel(NativeSegmentKernelFamily.SEGMENT_PARALLEL, size, chunks, "segment-parallel-binary");
    }

    public static NativeSegmentDispatchResult runFusedBinaryUnary(
            Operation binaryOp,
            Operation unaryOp,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        Objects.requireNonNull(binaryOp, "binaryOp cannot be null");
        Objects.requireNonNull(unaryOp, "unaryOp cannot be null");
        validateBinaryDenseWrite(left, right, output, "fused-binary-unary");
        DataType dataType = output.physicalView().dataType();
        if (!supportsBinary(binaryOp, dataType) || !supportsUnary(unaryOp, dataType)) {
            throw new UnsupportedOperationException("Unsupported native segment fused binary-unary. binary="
                    + binaryOp.opType() + ", unary=" + unaryOp.opType() + ", dtype=" + dataType);
        }
        switch (dataType) {
            case FLOAT32 -> runFusedBinaryUnaryF32(binaryOp, unaryOp, left, right, output, useFastExpApprox, useFastTanhApprox);
            case FLOAT64 -> runFusedBinaryUnaryF64(binaryOp, unaryOp, left, right, output, useFastExpApprox, useFastTanhApprox);
            case BFLOAT16, BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment fused binary-unary dtype="
                    + dataType);
        }
        output.storage().markModified();
        return NativeSegmentDispatchResult.scalar(NativeSegmentKernelFamily.SEGMENT_FUSED, logicalSize(output), "segment-fused-binary-unary");
    }

    public static void runWhere(
            CpuWhereKernel kernel,
            byte[] condition,
            TensorPhysicalView conditionView,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        Objects.requireNonNull(kernel, "kernel cannot be null");
        validateWhereDenseWrite(condition, conditionView, ifTrue, ifFalse, output);
        switch (output.physicalView().dataType()) {
            case FLOAT32 -> runWhereF32(kernel, condition, conditionView, ifTrue, ifFalse, output);
            case FLOAT64 -> runWhereF64(kernel, condition, conditionView, ifTrue, ifFalse, output);
            case BFLOAT16 -> runWhereBF16(condition, conditionView, ifTrue, ifFalse, output);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment WHERE dtype="
                    + output.physicalView().dataType());
        }
        output.storage().markModified();
    }

    public static void runWhere(
            CpuWhereKernel kernel,
            NativeSegmentView condition,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        Objects.requireNonNull(kernel, "kernel cannot be null");
        validateWhereDenseWrite(condition, ifTrue, ifFalse, output);
        switch (output.physicalView().dataType()) {
            case FLOAT32 -> runWhereF32(kernel, condition, ifTrue, ifFalse, output);
            case FLOAT64 -> runWhereF64(kernel, condition, ifTrue, ifFalse, output);
            case BFLOAT16 -> runWhereBF16(condition, ifTrue, ifFalse, output);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment WHERE dtype="
                    + output.physicalView().dataType());
        }
        output.storage().markModified();
    }

    public static void runCompare(
            Operation op,
            NativeSegmentView left,
            NativeSegmentView right,
            byte[] output,
            TensorPhysicalView outputView
    ) {
        Objects.requireNonNull(op, "op cannot be null");
        validateCompareDenseWrite(op, left, right, output, outputView);
        switch (left.physicalView().dataType()) {
            case FLOAT32 -> runCompareF32(op.opType(), left, right, output, outputView);
            case FLOAT64 -> runCompareF64(op.opType(), left, right, output, outputView);
            case BFLOAT16 -> runCompareBF16(op.opType(), left, right, output, outputView);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment compare dtype="
                    + left.physicalView().dataType());
        }
    }

    public static void runCompare(
            Operation op,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output
    ) {
        Objects.requireNonNull(op, "op cannot be null");
        validateCompareDenseWrite(op, left, right, output);
        switch (left.physicalView().dataType()) {
            case FLOAT32 -> runCompareF32(op.opType(), left, right, output);
            case FLOAT64 -> runCompareF64(op.opType(), left, right, output);
            case BFLOAT16 -> runCompareBF16(op.opType(), left, right, output);
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment compare dtype="
                    + left.physicalView().dataType());
        }
        output.storage().markModified();
    }

    public static boolean supportsReduction(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.BOOL) {
            return opType == Operation.OpType.REDUCE_ALL || opType == Operation.OpType.REDUCE_ANY;
        }
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN;
        }
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64)
                && (opType == Operation.OpType.SUM
                || opType == Operation.OpType.MEAN
                || opType == Operation.OpType.REDUCE_MIN
                || opType == Operation.OpType.REDUCE_MAX);
    }

    public static void runReduction(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension
    ) {
        validateReductionDenseWrite(opType, input, output, dimension);
        switch (input.physicalView().dataType()) {
            case FLOAT32 -> runReductionF32(opType, input, output, dimension);
            case FLOAT64 -> runReductionF64(opType, input, output, dimension);
            case BFLOAT16 -> runReductionBF16(opType, input, output, dimension);
            case BOOL -> runReductionBool(opType, input, output, dimension);
            case INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment reduction dtype="
                    + input.physicalView().dataType());
        }
        output.storage().markModified();
    }

    public static NativeSegmentDispatchResult runReductionWithDispatch(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension,
            NativeSegmentDispatchConfig dispatchConfig
    ) {
        NativeSegmentDispatchConfig config = dispatchConfig == null ? NativeSegmentDispatchConfig.scalar() : dispatchConfig;
        validateReductionDenseWrite(opType, input, output, dimension);
        int size = dimension == -1 ? logicalSize(input) : logicalSize(output);
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            runReduction(opType, input, output, dimension);
            return NativeSegmentDispatchResult.scalar(
                    scalarFamily(input),
                    size,
                    "segment-scalar-dispatch:minmax-reduction-correctness-only"
            );
        }
        if (input.physicalView().dataType() == DataType.BFLOAT16) {
            runReduction(opType, input, output, dimension);
            return NativeSegmentDispatchResult.scalar(
                    scalarFamily(input),
                    size,
                    "segment-scalar-dispatch:bf16-promoted-reduction-correctness-only"
            );
        }
        if (!config.parallelEligible(size)) {
            runReduction(opType, input, output, dimension);
            return NativeSegmentDispatchResult.scalar(scalarFamily(input), size, scalarReason(config, size));
        }
        int chunks = config.chunks(size);
        switch (input.physicalView().dataType()) {
            case FLOAT32 -> parallelReductionF32(opType, input, output, dimension, config, chunks);
            case FLOAT64 -> parallelReductionF64(opType, input, output, dimension, config, chunks);
            case BFLOAT16, BOOL, INT32, INT64 -> throw new UnsupportedOperationException("Unsupported native segment parallel reduction dtype="
                    + input.physicalView().dataType());
        }
        output.storage().markModified();
        return NativeSegmentDispatchResult.parallel(NativeSegmentKernelFamily.SEGMENT_PARALLEL, size, chunks, "segment-parallel-reduction");
    }

    private static void runUnaryF32(
            Operation op,
            NativeSegmentView input,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        float scalar = scalarParameter(op);
        if (input.physicalView().denseContiguous()) {
            long inOffset = input.baseByteOffset();
            long outOffset = output.baseByteOffset();
            for (int i = 0; i < size; i++) {
                out.set(JAVA_FLOAT, outOffset, applyF32(op.opType(), in.get(JAVA_FLOAT, inOffset), scalar, useFastExpApprox, useFastTanhApprox));
                inOffset += Float.BYTES;
                outOffset += Float.BYTES;
            }
            return;
        }
        NativeSegmentOffsetCursor cursor = new NativeSegmentOffsetCursor(
                input.physicalView().shape(),
                new long[][]{input.byteStrides()},
                new long[]{input.baseByteOffset()}
        );
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            out.set(JAVA_FLOAT, outOffset, applyF32(op.opType(), in.get(JAVA_FLOAT, cursor.offset(0)), scalar, useFastExpApprox, useFastTanhApprox));
            outOffset += Float.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runUnaryF64(
            Operation op,
            NativeSegmentView input,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        double scalar = scalarParameterF64(op);
        if (input.physicalView().denseContiguous()) {
            long inOffset = input.baseByteOffset();
            long outOffset = output.baseByteOffset();
            for (int i = 0; i < size; i++) {
                out.set(JAVA_DOUBLE, outOffset, applyF64(op.opType(), in.get(JAVA_DOUBLE, inOffset), scalar, useFastExpApprox, useFastTanhApprox));
                inOffset += Double.BYTES;
                outOffset += Double.BYTES;
            }
            return;
        }
        NativeSegmentOffsetCursor cursor = new NativeSegmentOffsetCursor(
                input.physicalView().shape(),
                new long[][]{input.byteStrides()},
                new long[]{input.baseByteOffset()}
        );
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            out.set(JAVA_DOUBLE, outOffset, applyF64(op.opType(), in.get(JAVA_DOUBLE, cursor.offset(0)), scalar, useFastExpApprox, useFastTanhApprox));
            outOffset += Double.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runUnaryBF16(Operation op, NativeSegmentView input, NativeSegmentView output) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        float scalar = scalarParameter(op);
        NativeSegmentOffsetCursor cursor = new NativeSegmentOffsetCursor(
                input.physicalView().shape(),
                new long[][]{input.byteStrides()},
                new long[]{input.baseByteOffset()}
        );
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            short bits = in.get(JAVA_SHORT, cursor.offset(0));
            float value = TensorDTypeOps.fromBFloat16Bits(bits);
            out.set(JAVA_SHORT, outOffset, TensorDTypeOps.toBFloat16Bits(applyBF16(op.opType(), value, scalar)));
            outOffset += Short.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runUnaryBool(Operation op, NativeSegmentView input, NativeSegmentView output) {
        if (op.opType() != Operation.OpType.LOGICAL_NOT) {
            throw new UnsupportedOperationException("Unsupported native segment BOOL unary opType=" + op.opType());
        }
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        NativeSegmentOffsetCursor cursor = cursor(input);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            out.set(JAVA_BYTE, outOffset, in.get(JAVA_BYTE, cursor.offset(0)) == 0 ? (byte) 1 : (byte) 0);
            outOffset += Byte.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void parallelUnaryF32(
            Operation op,
            NativeSegmentView input,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            NativeSegmentDispatchConfig config,
            int chunks
    ) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        float scalar = scalarParameter(op);
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunkStart(chunk, config.chunkSize(), size);
            int end = chunkEnd(start, config.chunkSize(), size);
            long outOffset = output.baseByteOffset() + (long) start * Float.BYTES;
            for (int i = start; i < end; i++) {
                float value = in.get(JAVA_FLOAT, input.byteOffsetForLogicalIndex(i));
                out.set(JAVA_FLOAT, outOffset, applyF32(op.opType(), value, scalar, useFastExpApprox, useFastTanhApprox));
                outOffset += Float.BYTES;
            }
        });
    }

    private static void parallelUnaryF64(
            Operation op,
            NativeSegmentView input,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox,
            NativeSegmentDispatchConfig config,
            int chunks
    ) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        double scalar = scalarParameterF64(op);
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunkStart(chunk, config.chunkSize(), size);
            int end = chunkEnd(start, config.chunkSize(), size);
            long outOffset = output.baseByteOffset() + (long) start * Double.BYTES;
            for (int i = start; i < end; i++) {
                double value = in.get(JAVA_DOUBLE, input.byteOffsetForLogicalIndex(i));
                out.set(JAVA_DOUBLE, outOffset, applyF64(op.opType(), value, scalar, useFastExpApprox, useFastTanhApprox));
                outOffset += Double.BYTES;
            }
        });
    }

    private static void copyF32(NativeSegmentView input, NativeSegmentView output) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        if (input.physicalView().denseContiguous()) {
            MemorySegment.copy(in, JAVA_FLOAT, input.baseByteOffset(), out, JAVA_FLOAT, output.baseByteOffset(), size);
            return;
        }
        NativeSegmentOffsetCursor cursor = cursor(input);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            out.set(JAVA_FLOAT, outOffset, in.get(JAVA_FLOAT, cursor.offset(0)));
            outOffset += Float.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void copyF64(NativeSegmentView input, NativeSegmentView output) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        if (input.physicalView().denseContiguous()) {
            MemorySegment.copy(in, JAVA_DOUBLE, input.baseByteOffset(), out, JAVA_DOUBLE, output.baseByteOffset(), size);
            return;
        }
        NativeSegmentOffsetCursor cursor = cursor(input);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            out.set(JAVA_DOUBLE, outOffset, in.get(JAVA_DOUBLE, cursor.offset(0)));
            outOffset += Double.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void copyBF16(NativeSegmentView input, NativeSegmentView output) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int size = logicalSize(input);
        if (input.physicalView().denseContiguous()) {
            MemorySegment.copy(in, JAVA_SHORT, input.baseByteOffset(), out, JAVA_SHORT, output.baseByteOffset(), size);
            return;
        }
        NativeSegmentOffsetCursor cursor = cursor(input);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            out.set(JAVA_SHORT, outOffset, in.get(JAVA_SHORT, cursor.offset(0)));
            outOffset += Short.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runBinaryF32(Operation op, NativeSegmentView left, NativeSegmentView right, NativeSegmentView output) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = binaryCursor(left, right);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            float leftValue = leftSegment.get(JAVA_FLOAT, cursor.offset(0));
            float rightValue = rightSegment.get(JAVA_FLOAT, cursor.offset(1));
            outSegment.set(JAVA_FLOAT, outOffset, applyBinaryF32(op.opType(), leftValue, rightValue));
            outOffset += Float.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runBinaryF64(Operation op, NativeSegmentView left, NativeSegmentView right, NativeSegmentView output) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = binaryCursor(left, right);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            double leftValue = leftSegment.get(JAVA_DOUBLE, cursor.offset(0));
            double rightValue = rightSegment.get(JAVA_DOUBLE, cursor.offset(1));
            outSegment.set(JAVA_DOUBLE, outOffset, applyBinaryF64(op.opType(), leftValue, rightValue));
            outOffset += Double.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runBinaryBF16(Operation op, NativeSegmentView left, NativeSegmentView right, NativeSegmentView output) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = binaryCursor(left, right);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            float leftValue = TensorDTypeOps.fromBFloat16Bits(leftSegment.get(JAVA_SHORT, cursor.offset(0)));
            float rightValue = TensorDTypeOps.fromBFloat16Bits(rightSegment.get(JAVA_SHORT, cursor.offset(1)));
            outSegment.set(JAVA_SHORT, outOffset, TensorDTypeOps.toBFloat16Bits(applyBinaryF32(op.opType(), leftValue, rightValue)));
            outOffset += Short.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runBinaryBool(Operation op, NativeSegmentView left, NativeSegmentView right, NativeSegmentView output) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = binaryCursor(left, right);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            byte leftValue = leftSegment.get(JAVA_BYTE, cursor.offset(0));
            byte rightValue = rightSegment.get(JAVA_BYTE, cursor.offset(1));
            outSegment.set(JAVA_BYTE, outOffset, applyBinaryBool(op.opType(), leftValue, rightValue));
            outOffset += Byte.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void parallelBinaryF32(
            Operation op,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output,
            NativeSegmentDispatchConfig config,
            int chunks
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunkStart(chunk, config.chunkSize(), size);
            int end = chunkEnd(start, config.chunkSize(), size);
            long outOffset = output.baseByteOffset() + (long) start * Float.BYTES;
            for (int i = start; i < end; i++) {
                float leftValue = leftSegment.get(JAVA_FLOAT, left.byteOffsetForLogicalIndex(i));
                float rightValue = rightSegment.get(JAVA_FLOAT, right.byteOffsetForLogicalIndex(i));
                outSegment.set(JAVA_FLOAT, outOffset, applyBinaryF32(op.opType(), leftValue, rightValue));
                outOffset += Float.BYTES;
            }
        });
    }

    private static void parallelBinaryF64(
            Operation op,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output,
            NativeSegmentDispatchConfig config,
            int chunks
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunkStart(chunk, config.chunkSize(), size);
            int end = chunkEnd(start, config.chunkSize(), size);
            long outOffset = output.baseByteOffset() + (long) start * Double.BYTES;
            for (int i = start; i < end; i++) {
                double leftValue = leftSegment.get(JAVA_DOUBLE, left.byteOffsetForLogicalIndex(i));
                double rightValue = rightSegment.get(JAVA_DOUBLE, right.byteOffsetForLogicalIndex(i));
                outSegment.set(JAVA_DOUBLE, outOffset, applyBinaryF64(op.opType(), leftValue, rightValue));
                outOffset += Double.BYTES;
            }
        });
    }

    private static void runFusedBinaryUnaryF32(
            Operation binaryOp,
            Operation unaryOp,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        float scalar = scalarParameter(unaryOp);
        NativeSegmentOffsetCursor cursor = binaryCursor(left, right);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            float leftValue = leftSegment.get(JAVA_FLOAT, cursor.offset(0));
            float rightValue = rightSegment.get(JAVA_FLOAT, cursor.offset(1));
            float intermediate = applyBinaryF32(binaryOp.opType(), leftValue, rightValue);
            outSegment.set(JAVA_FLOAT, outOffset, applyF32(unaryOp.opType(), intermediate, scalar, useFastExpApprox, useFastTanhApprox));
            outOffset += Float.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runFusedBinaryUnaryF64(
            Operation binaryOp,
            Operation unaryOp,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        double scalar = scalarParameterF64(unaryOp);
        NativeSegmentOffsetCursor cursor = binaryCursor(left, right);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            double leftValue = leftSegment.get(JAVA_DOUBLE, cursor.offset(0));
            double rightValue = rightSegment.get(JAVA_DOUBLE, cursor.offset(1));
            double intermediate = applyBinaryF64(binaryOp.opType(), leftValue, rightValue);
            outSegment.set(JAVA_DOUBLE, outOffset, applyF64(unaryOp.opType(), intermediate, scalar, useFastExpApprox, useFastTanhApprox));
            outOffset += Double.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runWhereF32(
            CpuWhereKernel kernel,
            byte[] condition,
            TensorPhysicalView conditionView,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        MemorySegment trueSegment = ifTrue.segment();
        MemorySegment falseSegment = ifFalse.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = whereCursor(conditionView, ifTrue, ifFalse);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            byte conditionValue = condition[Math.toIntExact(cursor.offset(0))];
            float trueValue = trueSegment.get(JAVA_FLOAT, cursor.offset(1));
            float falseValue = falseSegment.get(JAVA_FLOAT, cursor.offset(2));
            outSegment.set(JAVA_FLOAT, outOffset, kernel.applyF32(conditionValue, trueValue, falseValue));
            outOffset += Float.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runWhereF64(
            CpuWhereKernel kernel,
            byte[] condition,
            TensorPhysicalView conditionView,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        MemorySegment trueSegment = ifTrue.segment();
        MemorySegment falseSegment = ifFalse.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = whereCursor(conditionView, ifTrue, ifFalse);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            byte conditionValue = condition[Math.toIntExact(cursor.offset(0))];
            double trueValue = trueSegment.get(JAVA_DOUBLE, cursor.offset(1));
            double falseValue = falseSegment.get(JAVA_DOUBLE, cursor.offset(2));
            outSegment.set(JAVA_DOUBLE, outOffset, kernel.applyF64(conditionValue, trueValue, falseValue));
            outOffset += Double.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runWhereBF16(
            byte[] condition,
            TensorPhysicalView conditionView,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        MemorySegment trueSegment = ifTrue.segment();
        MemorySegment falseSegment = ifFalse.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = whereCursor(conditionView, ifTrue, ifFalse);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            short value = condition[Math.toIntExact(cursor.offset(0))] != 0
                    ? trueSegment.get(JAVA_SHORT, cursor.offset(1))
                    : falseSegment.get(JAVA_SHORT, cursor.offset(2));
            outSegment.set(JAVA_SHORT, outOffset, value);
            outOffset += Short.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runWhereF32(
            CpuWhereKernel kernel,
            NativeSegmentView condition,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        MemorySegment conditionSegment = condition.segment();
        MemorySegment trueSegment = ifTrue.segment();
        MemorySegment falseSegment = ifFalse.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = whereCursor(condition, ifTrue, ifFalse);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            byte conditionValue = conditionSegment.get(JAVA_BYTE, cursor.offset(0));
            float trueValue = trueSegment.get(JAVA_FLOAT, cursor.offset(1));
            float falseValue = falseSegment.get(JAVA_FLOAT, cursor.offset(2));
            outSegment.set(JAVA_FLOAT, outOffset, kernel.applyF32(conditionValue, trueValue, falseValue));
            outOffset += Float.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runWhereF64(
            CpuWhereKernel kernel,
            NativeSegmentView condition,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        MemorySegment conditionSegment = condition.segment();
        MemorySegment trueSegment = ifTrue.segment();
        MemorySegment falseSegment = ifFalse.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = whereCursor(condition, ifTrue, ifFalse);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            byte conditionValue = conditionSegment.get(JAVA_BYTE, cursor.offset(0));
            double trueValue = trueSegment.get(JAVA_DOUBLE, cursor.offset(1));
            double falseValue = falseSegment.get(JAVA_DOUBLE, cursor.offset(2));
            outSegment.set(JAVA_DOUBLE, outOffset, kernel.applyF64(conditionValue, trueValue, falseValue));
            outOffset += Double.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runWhereBF16(
            NativeSegmentView condition,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        MemorySegment conditionSegment = condition.segment();
        MemorySegment trueSegment = ifTrue.segment();
        MemorySegment falseSegment = ifFalse.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(output);
        NativeSegmentOffsetCursor cursor = whereCursor(condition, ifTrue, ifFalse);
        long outOffset = output.baseByteOffset();
        for (int i = 0; i < size; i++) {
            short value = conditionSegment.get(JAVA_BYTE, cursor.offset(0)) != 0
                    ? trueSegment.get(JAVA_SHORT, cursor.offset(1))
                    : falseSegment.get(JAVA_SHORT, cursor.offset(2));
            outSegment.set(JAVA_SHORT, outOffset, value);
            outOffset += Short.BYTES;
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runCompareF32(
            Operation.OpType opType,
            NativeSegmentView left,
            NativeSegmentView right,
            byte[] output,
            TensorPhysicalView outputView
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        int size = logicalSize(left);
        NativeSegmentOffsetCursor cursor = compareCursor(left, right, outputView);
        for (int i = 0; i < size; i++) {
            float leftValue = leftSegment.get(JAVA_FLOAT, cursor.offset(0));
            float rightValue = rightSegment.get(JAVA_FLOAT, cursor.offset(1));
            output[Math.toIntExact(cursor.offset(2))] = compare(opType, leftValue, rightValue);
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runCompareF64(
            Operation.OpType opType,
            NativeSegmentView left,
            NativeSegmentView right,
            byte[] output,
            TensorPhysicalView outputView
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        int size = logicalSize(left);
        NativeSegmentOffsetCursor cursor = compareCursor(left, right, outputView);
        for (int i = 0; i < size; i++) {
            double leftValue = leftSegment.get(JAVA_DOUBLE, cursor.offset(0));
            double rightValue = rightSegment.get(JAVA_DOUBLE, cursor.offset(1));
            output[Math.toIntExact(cursor.offset(2))] = compare(opType, leftValue, rightValue);
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runCompareBF16(
            Operation.OpType opType,
            NativeSegmentView left,
            NativeSegmentView right,
            byte[] output,
            TensorPhysicalView outputView
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        int size = logicalSize(left);
        NativeSegmentOffsetCursor cursor = compareCursor(left, right, outputView);
        for (int i = 0; i < size; i++) {
            float leftValue = TensorDTypeOps.fromBFloat16Bits(leftSegment.get(JAVA_SHORT, cursor.offset(0)));
            float rightValue = TensorDTypeOps.fromBFloat16Bits(rightSegment.get(JAVA_SHORT, cursor.offset(1)));
            output[Math.toIntExact(cursor.offset(2))] = compare(opType, leftValue, rightValue);
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runCompareF32(
            Operation.OpType opType,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(left);
        NativeSegmentOffsetCursor cursor = compareCursor(left, right, output);
        for (int i = 0; i < size; i++) {
            float leftValue = leftSegment.get(JAVA_FLOAT, cursor.offset(0));
            float rightValue = rightSegment.get(JAVA_FLOAT, cursor.offset(1));
            outSegment.set(JAVA_BYTE, cursor.offset(2), compare(opType, leftValue, rightValue));
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runCompareF64(
            Operation.OpType opType,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(left);
        NativeSegmentOffsetCursor cursor = compareCursor(left, right, output);
        for (int i = 0; i < size; i++) {
            double leftValue = leftSegment.get(JAVA_DOUBLE, cursor.offset(0));
            double rightValue = rightSegment.get(JAVA_DOUBLE, cursor.offset(1));
            outSegment.set(JAVA_BYTE, cursor.offset(2), compare(opType, leftValue, rightValue));
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runCompareBF16(
            Operation.OpType opType,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output
    ) {
        MemorySegment leftSegment = left.segment();
        MemorySegment rightSegment = right.segment();
        MemorySegment outSegment = output.segment();
        int size = logicalSize(left);
        NativeSegmentOffsetCursor cursor = compareCursor(left, right, output);
        for (int i = 0; i < size; i++) {
            float leftValue = TensorDTypeOps.fromBFloat16Bits(leftSegment.get(JAVA_SHORT, cursor.offset(0)));
            float rightValue = TensorDTypeOps.fromBFloat16Bits(rightSegment.get(JAVA_SHORT, cursor.offset(1)));
            outSegment.set(JAVA_BYTE, cursor.offset(2), compare(opType, leftValue, rightValue));
            if (i + 1 < size) {
                cursor.step();
            }
        }
    }

    private static void runReductionF32(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension
    ) {
        if (dimension == -1) {
            output.segment().set(JAVA_FLOAT, output.baseByteOffset(), reduceAllF32(opType, input));
            return;
        }
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int[] shape = input.physicalView().shape();
        long[] inputStrides = input.byteStrides();
        int outSize = logicalSize(output);
        int reducedSize = shape[dimension];
        long axisStride = inputStrides[dimension];
        int[] reducedStrides = denseStridesExcludingDim(shape, dimension);
        long outOffset = output.baseByteOffset();
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            long inputBase = inputBaseByteOffset(outIndex, shape, inputStrides, input.baseByteOffset(), reducedStrides, dimension);
            out.set(JAVA_FLOAT, outOffset, reduceAxisF32Value(opType, in, inputBase, axisStride, reducedSize));
            outOffset += Float.BYTES;
        }
    }

    private static void runReductionF64(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension
    ) {
        if (dimension == -1) {
            output.segment().set(JAVA_DOUBLE, output.baseByteOffset(), reduceAllF64(opType, input));
            return;
        }
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int[] shape = input.physicalView().shape();
        long[] inputStrides = input.byteStrides();
        int outSize = logicalSize(output);
        int reducedSize = shape[dimension];
        long axisStride = inputStrides[dimension];
        int[] reducedStrides = denseStridesExcludingDim(shape, dimension);
        long outOffset = output.baseByteOffset();
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            long inputBase = inputBaseByteOffset(outIndex, shape, inputStrides, input.baseByteOffset(), reducedStrides, dimension);
            out.set(JAVA_DOUBLE, outOffset, reduceAxisF64Value(opType, in, inputBase, axisStride, reducedSize));
            outOffset += Double.BYTES;
        }
    }

    private static void runReductionBool(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension
    ) {
        if (dimension == -1) {
            output.segment().set(JAVA_BYTE, output.baseByteOffset(), reduceAllBool(opType, input));
            return;
        }
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int[] shape = input.physicalView().shape();
        long[] inputStrides = input.byteStrides();
        int outSize = logicalSize(output);
        int reducedSize = shape[dimension];
        long axisStride = inputStrides[dimension];
        int[] reducedStrides = denseStridesExcludingDim(shape, dimension);
        long outOffset = output.baseByteOffset();
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            long inputBase = inputBaseByteOffset(outIndex, shape, inputStrides, input.baseByteOffset(), reducedStrides, dimension);
            boolean result = opType == Operation.OpType.REDUCE_ALL;
            for (int k = 0; k < reducedSize; k++) {
                boolean value = in.get(JAVA_BYTE, inputBase + (long) k * axisStride) != 0;
                if (opType == Operation.OpType.REDUCE_ALL) {
                    result &= value;
                    if (!result) {
                        break;
                    }
                } else {
                    result |= value;
                    if (result) {
                        break;
                    }
                }
            }
            out.set(JAVA_BYTE, outOffset, result ? (byte) 1 : (byte) 0);
            outOffset += Byte.BYTES;
        }
    }

    private static void runReductionBF16(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension
    ) {
        if (dimension == -1) {
            output.segment().set(JAVA_SHORT, output.baseByteOffset(), TensorDTypeOps.toBFloat16Bits(reduceAllBF16(opType, input)));
            return;
        }
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int[] shape = input.physicalView().shape();
        long[] inputStrides = input.byteStrides();
        int outSize = logicalSize(output);
        int reducedSize = shape[dimension];
        long axisStride = inputStrides[dimension];
        int[] reducedStrides = denseStridesExcludingDim(shape, dimension);
        long outOffset = output.baseByteOffset();
        for (int outIndex = 0; outIndex < outSize; outIndex++) {
            long inputBase = inputBaseByteOffset(outIndex, shape, inputStrides, input.baseByteOffset(), reducedStrides, dimension);
            float value = reduceAxisBF16Value(opType, in, inputBase, axisStride, reducedSize);
            out.set(JAVA_SHORT, outOffset, TensorDTypeOps.toBFloat16Bits(value));
            outOffset += Short.BYTES;
        }
    }

    private static void parallelReductionF32(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension,
            NativeSegmentDispatchConfig config,
            int chunks
    ) {
        if (dimension == -1) {
            DoubleAdder sum = new DoubleAdder();
            MemorySegment in = input.segment();
            int size = logicalSize(input);
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunkStart(chunk, config.chunkSize(), size);
                int end = chunkEnd(start, config.chunkSize(), size);
                double local = 0.0d;
                for (int i = start; i < end; i++) {
                    local += in.get(JAVA_FLOAT, input.byteOffsetForLogicalIndex(i));
                }
                sum.add(local);
            });
            double value = sum.sum();
            if (opType == Operation.OpType.MEAN) {
                value /= size;
            }
            output.segment().set(JAVA_FLOAT, output.baseByteOffset(), (float) value);
            return;
        }
        parallelReductionAxisF32(opType, input, output, dimension, config, chunks);
    }

    private static void parallelReductionF64(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension,
            NativeSegmentDispatchConfig config,
            int chunks
    ) {
        if (dimension == -1) {
            DoubleAdder sum = new DoubleAdder();
            MemorySegment in = input.segment();
            int size = logicalSize(input);
            CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
                int start = chunkStart(chunk, config.chunkSize(), size);
                int end = chunkEnd(start, config.chunkSize(), size);
                double local = 0.0d;
                for (int i = start; i < end; i++) {
                    local += in.get(JAVA_DOUBLE, input.byteOffsetForLogicalIndex(i));
                }
                sum.add(local);
            });
            double value = sum.sum();
            if (opType == Operation.OpType.MEAN) {
                value /= size;
            }
            output.segment().set(JAVA_DOUBLE, output.baseByteOffset(), value);
            return;
        }
        parallelReductionAxisF64(opType, input, output, dimension, config, chunks);
    }

    private static void parallelReductionAxisF32(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension,
            NativeSegmentDispatchConfig config,
            int chunks
    ) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int[] shape = input.physicalView().shape();
        long[] inputStrides = input.byteStrides();
        int outSize = logicalSize(output);
        int reducedSize = shape[dimension];
        long axisStride = inputStrides[dimension];
        int[] reducedStrides = denseStridesExcludingDim(shape, dimension);
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunkStart(chunk, config.chunkSize(), outSize);
            int end = chunkEnd(start, config.chunkSize(), outSize);
            long outOffset = output.baseByteOffset() + (long) start * Float.BYTES;
            for (int outIndex = start; outIndex < end; outIndex++) {
                long inputBase = inputBaseByteOffset(outIndex, shape, inputStrides, input.baseByteOffset(), reducedStrides, dimension);
                double sum = 0.0d;
                for (int k = 0; k < reducedSize; k++) {
                    sum += in.get(JAVA_FLOAT, inputBase + (long) k * axisStride);
                }
                if (opType == Operation.OpType.MEAN) {
                    sum /= reducedSize;
                }
                out.set(JAVA_FLOAT, outOffset, (float) sum);
                outOffset += Float.BYTES;
            }
        });
    }

    private static void parallelReductionAxisF64(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension,
            NativeSegmentDispatchConfig config,
            int chunks
    ) {
        MemorySegment in = input.segment();
        MemorySegment out = output.segment();
        int[] shape = input.physicalView().shape();
        long[] inputStrides = input.byteStrides();
        int outSize = logicalSize(output);
        int reducedSize = shape[dimension];
        long axisStride = inputStrides[dimension];
        int[] reducedStrides = denseStridesExcludingDim(shape, dimension);
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunkStart(chunk, config.chunkSize(), outSize);
            int end = chunkEnd(start, config.chunkSize(), outSize);
            long outOffset = output.baseByteOffset() + (long) start * Double.BYTES;
            for (int outIndex = start; outIndex < end; outIndex++) {
                long inputBase = inputBaseByteOffset(outIndex, shape, inputStrides, input.baseByteOffset(), reducedStrides, dimension);
                double sum = 0.0d;
                for (int k = 0; k < reducedSize; k++) {
                    sum += in.get(JAVA_DOUBLE, inputBase + (long) k * axisStride);
                }
                if (opType == Operation.OpType.MEAN) {
                    sum /= reducedSize;
                }
                out.set(JAVA_DOUBLE, outOffset, sum);
                outOffset += Double.BYTES;
            }
        });
    }

    private static float reduceAllF32(Operation.OpType opType, NativeSegmentView input) {
        MemorySegment in = input.segment();
        int size = logicalSize(input);
        NativeSegmentOffsetCursor cursor = cursor(input);
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            float best = in.get(JAVA_FLOAT, cursor.offset(0));
            for (int i = 1; i < size; i++) {
                cursor.step();
                float value = in.get(JAVA_FLOAT, cursor.offset(0));
                best = opType == Operation.OpType.REDUCE_MAX ? Math.max(best, value) : Math.min(best, value);
            }
            return best;
        }
        double sum = 0.0d;
        for (int i = 0; i < size; i++) {
            sum += in.get(JAVA_FLOAT, cursor.offset(0));
            if (i + 1 < size) {
                cursor.step();
            }
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= size;
        }
        return (float) sum;
    }

    private static double reduceAllF64(Operation.OpType opType, NativeSegmentView input) {
        MemorySegment in = input.segment();
        int size = logicalSize(input);
        NativeSegmentOffsetCursor cursor = cursor(input);
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            double best = in.get(JAVA_DOUBLE, cursor.offset(0));
            for (int i = 1; i < size; i++) {
                cursor.step();
                double value = in.get(JAVA_DOUBLE, cursor.offset(0));
                best = opType == Operation.OpType.REDUCE_MAX ? Math.max(best, value) : Math.min(best, value);
            }
            return best;
        }
        double sum = 0.0d;
        for (int i = 0; i < size; i++) {
            sum += in.get(JAVA_DOUBLE, cursor.offset(0));
            if (i + 1 < size) {
                cursor.step();
            }
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= size;
        }
        return sum;
    }

    private static byte reduceAllBool(Operation.OpType opType, NativeSegmentView input) {
        MemorySegment in = input.segment();
        int size = logicalSize(input);
        NativeSegmentOffsetCursor cursor = cursor(input);
        boolean result = opType == Operation.OpType.REDUCE_ALL;
        for (int i = 0; i < size; i++) {
            boolean value = in.get(JAVA_BYTE, cursor.offset(0)) != 0;
            if (opType == Operation.OpType.REDUCE_ALL) {
                result &= value;
                if (!result) {
                    return 0;
                }
            } else {
                result |= value;
                if (result) {
                    return 1;
                }
            }
            if (i + 1 < size) {
                cursor.step();
            }
        }
        return result ? (byte) 1 : (byte) 0;
    }

    private static float reduceAllBF16(Operation.OpType opType, NativeSegmentView input) {
        MemorySegment in = input.segment();
        int size = logicalSize(input);
        NativeSegmentOffsetCursor cursor = cursor(input);
        double sum = 0.0d;
        for (int i = 0; i < size; i++) {
            sum += TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, cursor.offset(0)));
            if (i + 1 < size) {
                cursor.step();
            }
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= size;
        }
        return (float) sum;
    }

    private static float reduceAxisF32Value(
            Operation.OpType opType,
            MemorySegment in,
            long inputBase,
            long axisStride,
            int reducedSize
    ) {
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            float best = in.get(JAVA_FLOAT, inputBase);
            for (int k = 1; k < reducedSize; k++) {
                float value = in.get(JAVA_FLOAT, inputBase + (long) k * axisStride);
                best = opType == Operation.OpType.REDUCE_MAX ? Math.max(best, value) : Math.min(best, value);
            }
            return best;
        }
        double sum = 0.0d;
        for (int k = 0; k < reducedSize; k++) {
            sum += in.get(JAVA_FLOAT, inputBase + (long) k * axisStride);
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= reducedSize;
        }
        return (float) sum;
    }

    private static double reduceAxisF64Value(
            Operation.OpType opType,
            MemorySegment in,
            long inputBase,
            long axisStride,
            int reducedSize
    ) {
        if (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX) {
            double best = in.get(JAVA_DOUBLE, inputBase);
            for (int k = 1; k < reducedSize; k++) {
                double value = in.get(JAVA_DOUBLE, inputBase + (long) k * axisStride);
                best = opType == Operation.OpType.REDUCE_MAX ? Math.max(best, value) : Math.min(best, value);
            }
            return best;
        }
        double sum = 0.0d;
        for (int k = 0; k < reducedSize; k++) {
            sum += in.get(JAVA_DOUBLE, inputBase + (long) k * axisStride);
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= reducedSize;
        }
        return sum;
    }

    private static float reduceAxisBF16Value(
            Operation.OpType opType,
            MemorySegment in,
            long inputBase,
            long axisStride,
            int reducedSize
    ) {
        double sum = 0.0d;
        for (int k = 0; k < reducedSize; k++) {
            sum += TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, inputBase + (long) k * axisStride));
        }
        if (opType == Operation.OpType.MEAN) {
            sum /= reducedSize;
        }
        return (float) sum;
    }

    private static NativeSegmentOffsetCursor cursor(NativeSegmentView input) {
        return new NativeSegmentOffsetCursor(
                input.physicalView().shape(),
                new long[][]{input.byteStrides()},
                new long[]{input.baseByteOffset()}
        );
    }

    private static NativeSegmentOffsetCursor binaryCursor(NativeSegmentView left, NativeSegmentView right) {
        return new NativeSegmentOffsetCursor(
                left.physicalView().shape(),
                new long[][]{left.byteStrides(), right.byteStrides()},
                new long[]{left.baseByteOffset(), right.baseByteOffset()}
        );
    }

    private static NativeSegmentOffsetCursor whereCursor(
            TensorPhysicalView conditionView,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse
    ) {
        return new NativeSegmentOffsetCursor(
                conditionView.shape(),
                new long[][]{elementStridesAsLong(conditionView), ifTrue.byteStrides(), ifFalse.byteStrides()},
                new long[]{conditionView.storageOffsetElements(), ifTrue.baseByteOffset(), ifFalse.baseByteOffset()}
        );
    }

    private static NativeSegmentOffsetCursor whereCursor(
            NativeSegmentView condition,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse
    ) {
        return new NativeSegmentOffsetCursor(
                condition.physicalView().shape(),
                new long[][]{condition.byteStrides(), ifTrue.byteStrides(), ifFalse.byteStrides()},
                new long[]{condition.baseByteOffset(), ifTrue.baseByteOffset(), ifFalse.baseByteOffset()}
        );
    }

    private static NativeSegmentOffsetCursor compareCursor(
            NativeSegmentView left,
            NativeSegmentView right,
            TensorPhysicalView outputView
    ) {
        return new NativeSegmentOffsetCursor(
                outputView.shape(),
                new long[][]{left.byteStrides(), right.byteStrides(), elementStridesAsLong(outputView)},
                new long[]{left.baseByteOffset(), right.baseByteOffset(), outputView.storageOffsetElements()}
        );
    }

    private static NativeSegmentOffsetCursor compareCursor(
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output
    ) {
        return new NativeSegmentOffsetCursor(
                output.physicalView().shape(),
                new long[][]{left.byteStrides(), right.byteStrides(), output.byteStrides()},
                new long[]{left.baseByteOffset(), right.baseByteOffset(), output.baseByteOffset()}
        );
    }

    private static long[] elementStridesAsLong(TensorPhysicalView view) {
        int[] elementStrides = view.elementStrides();
        long[] out = new long[elementStrides.length];
        for (int i = 0; i < elementStrides.length; i++) {
            out[i] = elementStrides[i];
        }
        return out;
    }

    private static void validateSameShapeDenseWrite(NativeSegmentView input, NativeSegmentView output, String family) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        TensorPhysicalView inputView = input.physicalView();
        TensorPhysicalView outputView = output.physicalView();
        if (inputView.dataType() != outputView.dataType()) {
            throw new IllegalArgumentException("native segment " + family + " dtype mismatch. input="
                    + inputView.dataType() + ", output=" + outputView.dataType());
        }
        if (!Arrays.equals(inputView.shape(), outputView.shape())) {
            throw new IllegalArgumentException("native segment " + family + " shape mismatch. input="
                    + Arrays.toString(inputView.shape()) + ", output=" + Arrays.toString(outputView.shape()));
        }
        if (!outputView.denseContiguous()) {
            throw new IllegalArgumentException("native segment " + family + " requires dense contiguous output. output="
                    + outputView.describe());
        }
        if (inputView.logicalElementCount() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("native segment " + family + " logical size exceeds int loop bound: "
                    + inputView.logicalElementCount());
        }
    }

    private static void validateBinaryDenseWrite(
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output,
            String family
    ) {
        validateSameShapeDenseWrite(left, output, family);
        Objects.requireNonNull(right, "right cannot be null");
        TensorPhysicalView rightView = right.physicalView();
        TensorPhysicalView outputView = output.physicalView();
        if (rightView.dataType() != outputView.dataType()) {
            throw new IllegalArgumentException("native segment " + family + " dtype mismatch. right="
                    + rightView.dataType() + ", output=" + outputView.dataType());
        }
        if (!Arrays.equals(rightView.shape(), outputView.shape())) {
            throw new IllegalArgumentException("native segment " + family + " shape mismatch. right="
                    + Arrays.toString(rightView.shape()) + ", output=" + Arrays.toString(outputView.shape()));
        }
    }

    private static void validateWhereDenseWrite(
            byte[] condition,
            TensorPhysicalView conditionView,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        Objects.requireNonNull(condition, "condition cannot be null");
        Objects.requireNonNull(conditionView, "conditionView cannot be null");
        validateBinaryDenseWrite(ifTrue, ifFalse, output, "where");
        TensorPhysicalView outputView = output.physicalView();
        if (conditionView.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("native segment where condition must be BOOL. condition="
                    + conditionView.dataType());
        }
        if (conditionView.storageFamily() != NativeCpuStorageFamily.CPU_ARRAY) {
            throw new IllegalArgumentException("native segment where currently requires CPU_ARRAY condition view. condition="
                    + conditionView.storageFamily());
        }
        if (!Arrays.equals(conditionView.shape(), outputView.shape())) {
            throw new IllegalArgumentException("native segment where condition shape mismatch. condition="
                    + Arrays.toString(conditionView.shape()) + ", output=" + Arrays.toString(outputView.shape()));
        }
        if (conditionView.logicalElementCount() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("native segment where condition logical size exceeds int loop bound: "
                    + conditionView.logicalElementCount());
        }
        if (conditionView.physicalElementSpan() > condition.length) {
            throw new IllegalArgumentException("native segment where condition view exceeds condition array length. required="
                    + conditionView.physicalElementSpan() + ", actual=" + condition.length);
        }
    }

    private static void validateWhereDenseWrite(
            NativeSegmentView condition,
            NativeSegmentView ifTrue,
            NativeSegmentView ifFalse,
            NativeSegmentView output
    ) {
        Objects.requireNonNull(condition, "condition cannot be null");
        validateBinaryDenseWrite(ifTrue, ifFalse, output, "where");
        TensorPhysicalView conditionView = condition.physicalView();
        TensorPhysicalView outputView = output.physicalView();
        if (conditionView.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("native segment where condition must be BOOL. condition="
                    + conditionView.dataType());
        }
        if (conditionView.storageFamily() != NativeCpuStorageFamily.CPU_NATIVE) {
            throw new IllegalArgumentException("native segment where native-mask overload requires CPU_NATIVE condition view. condition="
                    + conditionView.storageFamily());
        }
        if (!Arrays.equals(conditionView.shape(), outputView.shape())) {
            throw new IllegalArgumentException("native segment where condition shape mismatch. condition="
                    + Arrays.toString(conditionView.shape()) + ", output=" + Arrays.toString(outputView.shape()));
        }
        if (conditionView.logicalElementCount() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("native segment where condition logical size exceeds int loop bound: "
                    + conditionView.logicalElementCount());
        }
    }

    private static void validateCompareDenseWrite(
            Operation op,
            NativeSegmentView left,
            NativeSegmentView right,
            byte[] output,
            TensorPhysicalView outputView
    ) {
        Objects.requireNonNull(left, "left cannot be null");
        Objects.requireNonNull(right, "right cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        Objects.requireNonNull(outputView, "outputView cannot be null");
        TensorPhysicalView leftView = left.physicalView();
        TensorPhysicalView rightView = right.physicalView();
        if (!supportsCompare(op, leftView.dataType())) {
            throw new UnsupportedOperationException("Unsupported native segment compare opType="
                    + op.opType() + ", dtype=" + leftView.dataType());
        }
        if (leftView.dataType() != rightView.dataType()) {
            throw new IllegalArgumentException("native segment compare dtype mismatch. left="
                    + leftView.dataType() + ", right=" + rightView.dataType());
        }
        if (outputView.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("native segment compare output must be BOOL. output="
                    + outputView.dataType());
        }
        if (outputView.storageFamily() != NativeCpuStorageFamily.CPU_ARRAY) {
            throw new IllegalArgumentException("native segment compare currently writes CPU_ARRAY bool output. output="
                    + outputView.storageFamily());
        }
        if (!Arrays.equals(leftView.shape(), outputView.shape())
                || !Arrays.equals(rightView.shape(), outputView.shape())) {
            throw new IllegalArgumentException("native segment compare shape mismatch. left="
                    + Arrays.toString(leftView.shape()) + ", right=" + Arrays.toString(rightView.shape())
                    + ", output=" + Arrays.toString(outputView.shape()));
        }
        if (!outputView.denseContiguous()) {
            throw new IllegalArgumentException("native segment compare requires dense contiguous bool output. output="
                    + outputView.describe());
        }
        if (outputView.logicalElementCount() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("native segment compare logical size exceeds int loop bound: "
                    + outputView.logicalElementCount());
        }
        if (outputView.physicalElementSpan() > output.length) {
            throw new IllegalArgumentException("native segment compare output view exceeds bool array length. required="
                    + outputView.physicalElementSpan() + ", actual=" + output.length);
        }
    }

    private static void validateCompareDenseWrite(
            Operation op,
            NativeSegmentView left,
            NativeSegmentView right,
            NativeSegmentView output
    ) {
        Objects.requireNonNull(left, "left cannot be null");
        Objects.requireNonNull(right, "right cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        TensorPhysicalView leftView = left.physicalView();
        TensorPhysicalView rightView = right.physicalView();
        TensorPhysicalView outputView = output.physicalView();
        if (!supportsCompare(op, leftView.dataType())) {
            throw new UnsupportedOperationException("Unsupported native segment compare opType="
                    + op.opType() + ", dtype=" + leftView.dataType());
        }
        if (leftView.dataType() != rightView.dataType()) {
            throw new IllegalArgumentException("native segment compare dtype mismatch. left="
                    + leftView.dataType() + ", right=" + rightView.dataType());
        }
        if (outputView.dataType() != DataType.BOOL) {
            throw new IllegalArgumentException("native segment compare output must be BOOL. output="
                    + outputView.dataType());
        }
        if (outputView.storageFamily() != NativeCpuStorageFamily.CPU_NATIVE) {
            throw new IllegalArgumentException("native segment compare native-mask overload requires CPU_NATIVE bool output. output="
                    + outputView.storageFamily());
        }
        if (!Arrays.equals(leftView.shape(), outputView.shape())
                || !Arrays.equals(rightView.shape(), outputView.shape())) {
            throw new IllegalArgumentException("native segment compare shape mismatch. left="
                    + Arrays.toString(leftView.shape()) + ", right=" + Arrays.toString(rightView.shape())
                    + ", output=" + Arrays.toString(outputView.shape()));
        }
        if (!outputView.denseContiguous()) {
            throw new IllegalArgumentException("native segment compare requires dense contiguous bool output. output="
                    + outputView.describe());
        }
        if (outputView.logicalElementCount() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("native segment compare logical size exceeds int loop bound: "
                    + outputView.logicalElementCount());
        }
    }

    private static void validateReductionDenseWrite(
            Operation.OpType opType,
            NativeSegmentView input,
            NativeSegmentView output,
            int dimension
    ) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        TensorPhysicalView inputView = input.physicalView();
        TensorPhysicalView outputView = output.physicalView();
        if (!supportsReduction(opType, inputView.dataType())) {
            throw new UnsupportedOperationException("Unsupported native segment reduction opType="
                    + opType + ", dtype=" + inputView.dataType());
        }
        if (inputView.dataType() != outputView.dataType()) {
            throw new IllegalArgumentException("native segment reduction dtype mismatch. input="
                    + inputView.dataType() + ", output=" + outputView.dataType());
        }
        if (!outputView.denseContiguous()) {
            throw new IllegalArgumentException("native segment reduction requires dense contiguous output. output="
                    + outputView.describe());
        }
        int[] shape = inputView.shape();
        if (shape.length == 0) {
            throw new IllegalArgumentException("native segment reduction requires rank >= 1 input");
        }
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("native segment reduction dimension out of range. dimension="
                    + dimension + ", rank=" + shape.length);
        }
        if (inputView.logicalElementCount() <= 0L || inputView.logicalElementCount() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("native segment reduction logical size out of range: "
                    + inputView.logicalElementCount());
        }
        long expectedOutputElements = expectedReductionOutputElements(shape, dimension);
        if (outputView.logicalElementCount() != expectedOutputElements) {
            throw new IllegalArgumentException("native segment reduction output element count mismatch. expected="
                    + expectedOutputElements + ", actual=" + outputView.logicalElementCount());
        }
        if (!validReductionOutputShape(shape, outputView.shape(), dimension)) {
            throw new IllegalArgumentException("native segment reduction output shape mismatch. input="
                    + Arrays.toString(shape) + ", dimension=" + dimension
                    + ", output=" + Arrays.toString(outputView.shape()));
        }
        if (dimension >= 0 && shape[dimension] <= 0) {
            throw new IllegalArgumentException("native segment reduction reduced dimension must be positive. dimension="
                    + dimension + ", size=" + shape[dimension]);
        }
    }

    private static int logicalSize(NativeSegmentView input) {
        return Math.toIntExact(input.physicalView().logicalElementCount());
    }

    private static boolean supportsParallelUnary(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static boolean supportsParallelBinary(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static NativeSegmentKernelFamily scalarFamily(NativeSegmentView input) {
        return input.physicalView().denseContiguous()
                ? NativeSegmentKernelFamily.SEGMENT_DENSE_SCALAR
                : NativeSegmentKernelFamily.SEGMENT_STRIDED_SCALAR;
    }

    private static NativeSegmentKernelFamily scalarFamily(NativeSegmentView left, NativeSegmentView right) {
        return left.physicalView().denseContiguous() && right.physicalView().denseContiguous()
                ? NativeSegmentKernelFamily.SEGMENT_DENSE_SCALAR
                : NativeSegmentKernelFamily.SEGMENT_STRIDED_SCALAR;
    }

    private static String scalarReason(NativeSegmentDispatchConfig config, int logicalSize) {
        if (config.plannedWorkers() <= 1) {
            return "segment-scalar-dispatch:single-worker";
        }
        if (logicalSize < config.parallelMinElements()) {
            return "segment-scalar-dispatch:below-parallel-threshold";
        }
        return "segment-scalar-dispatch:unsupported-parallel-family";
    }

    private static int chunkStart(int chunk, int chunkSize, int totalSize) {
        return Math.min(totalSize, Math.multiplyExact(chunk, chunkSize));
    }

    private static int chunkEnd(int start, int chunkSize, int totalSize) {
        return Math.min(totalSize, start + chunkSize);
    }

    private static long expectedReductionOutputElements(int[] shape, int dimension) {
        if (dimension == -1) {
            return 1L;
        }
        long size = 1L;
        for (int dim = 0; dim < shape.length; dim++) {
            if (dim != dimension) {
                size = Math.multiplyExact(size, shape[dim]);
            }
        }
        return size;
    }

    private static boolean validReductionOutputShape(int[] inputShape, int[] outputShape, int dimension) {
        if (dimension == -1) {
            return outputShape.length == 0
                    || outputShape.length == 1 && outputShape[0] == 1
                    || allOnes(outputShape);
        }
        if (outputShape.length == inputShape.length - 1) {
            int outDim = 0;
            for (int dim = 0; dim < inputShape.length; dim++) {
                if (dim == dimension) {
                    continue;
                }
                if (outputShape[outDim++] != inputShape[dim]) {
                    return false;
                }
            }
            return true;
        }
        if (outputShape.length == inputShape.length && outputShape[dimension] == 1) {
            for (int dim = 0; dim < inputShape.length; dim++) {
                if (dim != dimension && outputShape[dim] != inputShape[dim]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    private static boolean allOnes(int[] shape) {
        for (int dimension : shape) {
            if (dimension != 1) {
                return false;
            }
        }
        return true;
    }

    private static long inputBaseByteOffset(
            int outIndex,
            int[] inputShape,
            long[] inputStrides,
            long inputBaseOffset,
            int[] reducedDenseStrides,
            int dimension
    ) {
        int rem = outIndex;
        long offset = inputBaseOffset;
        int outAxis = 0;
        for (int dim = 0; dim < inputShape.length; dim++) {
            if (dim == dimension) {
                continue;
            }
            int coord = rem / reducedDenseStrides[outAxis];
            rem %= reducedDenseStrides[outAxis];
            offset += (long) coord * inputStrides[dim];
            outAxis++;
        }
        return offset;
    }

    private static int[] denseStridesExcludingDim(int[] shape, int dimension) {
        int[] strides = new int[Math.max(0, shape.length - 1)];
        int stride = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            if (dim == dimension) {
                continue;
            }
            strides[dim < dimension ? dim : dim - 1] = stride;
            stride = Math.multiplyExact(stride, shape[dim]);
        }
        return strides;
    }

    private static boolean isBinary(Operation.OpType opType) {
        return opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.MIN
                || opType == Operation.OpType.MAX
                || opType == Operation.OpType.POW_TENSOR;
    }

    private static boolean isCompare(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    private static byte compare(Operation.OpType opType, double left, double right) {
        boolean result = switch (opType) {
            case GT -> left > right;
            case GE -> left >= right;
            case LT -> left < right;
            case LE -> left <= right;
            case EQ -> left == right;
            case NE -> left != right;
            default -> throw new UnsupportedOperationException("Unsupported native segment compare opType=" + opType);
        };
        return result ? (byte) 1 : (byte) 0;
    }

    private static boolean isF32Unary(Operation.OpType opType) {
        return opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.NEG
                || opType == Operation.OpType.RELU
                || opType == Operation.OpType.CLAMP_MIN
                || opType == Operation.OpType.CLAMP_MAX
                || opType == Operation.OpType.LOG
                || opType == Operation.OpType.EXP
                || opType == Operation.OpType.FAST_EXP
                || opType == Operation.OpType.SQRT
                || opType == Operation.OpType.ABS
                || opType == Operation.OpType.FLOOR
                || opType == Operation.OpType.CEIL
                || opType == Operation.OpType.SIGN
                || opType == Operation.OpType.POW
                || opType == Operation.OpType.TANH
                || opType == Operation.OpType.FAST_TANH
                || opType == Operation.OpType.SIGMOID
                || opType == Operation.OpType.INV;
    }

    private static boolean isF64Unary(Operation.OpType opType) {
        return isF32Unary(opType);
    }

    private static boolean isBF16Unary(Operation.OpType opType) {
        return opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.NEG
                || opType == Operation.OpType.RELU
                || opType == Operation.OpType.CLAMP_MIN
                || opType == Operation.OpType.CLAMP_MAX
                || opType == Operation.OpType.ABS;
    }

    private static float applyBinaryF32(Operation.OpType opType, float left, float right) {
        return switch (opType) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            case POW_TENSOR -> CpuPowSupport.applyF32(left, right);
            default -> throw new IllegalArgumentException("Unsupported native segment F32 binary op: " + opType);
        };
    }

    private static double applyBinaryF64(Operation.OpType opType, double left, double right) {
        return switch (opType) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            case POW_TENSOR -> CpuPowSupport.applyF64(left, right);
            default -> throw new IllegalArgumentException("Unsupported native segment F64 binary op: " + opType);
        };
    }

    private static byte applyBinaryBool(Operation.OpType opType, byte left, byte right) {
        return switch (opType) {
            case LOGICAL_AND -> left != 0 && right != 0 ? (byte) 1 : (byte) 0;
            case LOGICAL_OR -> left != 0 || right != 0 ? (byte) 1 : (byte) 0;
            default -> throw new UnsupportedOperationException("Unsupported native segment BOOL binary opType="
                    + opType.name().toLowerCase());
        };
    }

    private static float applyF32(
            Operation.OpType opType,
            float value,
            float scalar,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        return switch (opType) {
            case MUL_SCALAR -> value * scalar;
            case NEG -> -value;
            case RELU -> Math.max(0.0f, value);
            case CLAMP_MIN -> Math.max(value, scalar);
            case CLAMP_MAX -> Math.min(value, scalar);
            case LOG -> (float) Math.log(value);
            case EXP -> useFastExpApprox ? FastTranscendentals.fastExpF32(value) : (float) Math.exp(value);
            case FAST_EXP -> FastTranscendentals.fastExpF32(value);
            case SQRT -> (float) Math.sqrt(value);
            case ABS -> Math.abs(value);
            case FLOOR -> (float) Math.floor(value);
            case CEIL -> (float) Math.ceil(value);
            case SIGN -> Math.signum(value);
            case POW -> CpuPowSupport.applyF32(value, scalar);
            case TANH -> useFastTanhApprox ? FastTranscendentals.fastTanhF32(value) : (float) Math.tanh(value);
            case FAST_TANH -> FastTranscendentals.fastTanhF32(value);
            case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-value));
            case INV -> 1.0f / value;
            default -> throw new IllegalArgumentException("Unsupported native segment F32 unary op: " + opType);
        };
    }

    private static double applyF64(
            Operation.OpType opType,
            double value,
            double scalar,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        return switch (opType) {
            case MUL_SCALAR -> value * scalar;
            case NEG -> -value;
            case RELU -> Math.max(0.0d, value);
            case CLAMP_MIN -> Math.max(value, scalar);
            case CLAMP_MAX -> Math.min(value, scalar);
            case LOG -> Math.log(value);
            case EXP -> useFastExpApprox ? FastTranscendentals.fastExpF64(value) : Math.exp(value);
            case FAST_EXP -> FastTranscendentals.fastExpF64(value);
            case SQRT -> Math.sqrt(value);
            case ABS -> Math.abs(value);
            case FLOOR -> Math.floor(value);
            case CEIL -> Math.ceil(value);
            case SIGN -> Math.signum(value);
            case POW -> CpuPowSupport.applyF64(value, scalar);
            case TANH -> useFastTanhApprox ? FastTranscendentals.fastTanhF64(value) : Math.tanh(value);
            case FAST_TANH -> FastTranscendentals.fastTanhF64(value);
            case SIGMOID -> 1.0d / (1.0d + Math.exp(-value));
            case INV -> 1.0d / value;
            default -> throw new IllegalArgumentException("Unsupported native segment F64 unary op: " + opType);
        };
    }

    private static float applyBF16(Operation.OpType opType, float value, float scalar) {
        return switch (opType) {
            case MUL_SCALAR -> value * scalar;
            case NEG -> -value;
            case RELU -> Math.max(0.0f, value);
            case CLAMP_MIN -> Math.max(value, scalar);
            case CLAMP_MAX -> Math.min(value, scalar);
            case ABS -> Math.abs(value);
            default -> throw new IllegalArgumentException("Unsupported native segment BF16 unary op: " + opType);
        };
    }

    private static float scalarParameter(Operation op) {
        if (op instanceof mulScalar mul) {
            return mul.getScalarF32();
        }
        if (op instanceof clampMin clamp) {
            return clamp.getMinValueF32();
        }
        if (op instanceof clampMax clamp) {
            return clamp.getMaxValueF32();
        }
        if (op instanceof pow power) {
            return power.getExponentF32();
        }
        return 0.0f;
    }

    private static double scalarParameterF64(Operation op) {
        if (op instanceof mulScalar mul) {
            return mul.getScalar();
        }
        if (op instanceof clampMin clamp) {
            return clamp.getMinValue();
        }
        if (op instanceof clampMax clamp) {
            return clamp.getMaxValue();
        }
        if (op instanceof pow power) {
            return power.getExponent();
        }
        return 0.0d;
    }
}
