package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLowering;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Objects;

/** Optimal clean-Java full-K scalar oracle for portable MATMUL generated forms. */
public final class CpuMatmulReferenceKernel {
    /** Prevents instantiation of this stateless oracle. */
    private CpuMatmulReferenceKernel() { }

    /**
     * Evaluates one normalized MATMUL into the caller-owned result carrier.
     *
     * <p>Each output is initialized once, traverses increasing logical K exactly once, and is
     * stored once. The method performs no K blocking, splitting, partial-result allocation, or
     * worker-dependent reduction. Input and output layouts may be arbitrary non-negative strides.
     * Integral operations use Java wrapping at the promoted result width.</p>
     *
     * @param geometry checked non-null normalized geometry
     * @param numericalForm exact floating multiply-add decision shared with generated code
     * @param left exact primitive carrier matching the left type
     * @param right exact primitive carrier matching the right type
     * @param result exact mutable primitive carrier matching the result type
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if a carrier does not match its represented type
     * @throws ArithmeticException if exact address arithmetic overflows
     */
    public static void evaluate(CpuMatmulLowering.Geometry geometry,
            CpuMatmulIr.NumericalForm numericalForm, Object left, Object right, Object result) {
        Objects.requireNonNull(geometry, "geometry"); Objects.requireNonNull(numericalForm, "numericalForm");
        requireCarrier(geometry.leftType(), left); requireCarrier(geometry.rightType(), right);
        requireCarrier(geometry.resultType(), result);
        long[] batch = geometry.batchExtents(), aBatch = geometry.leftBatchStrides();
        long[] bBatch = geometry.rightBatchStrides(), cBatch = geometry.resultBatchStrides();
        long[] coordinates = new long[batch.length];
        for (long batchOrdinal = 0; batchOrdinal < geometry.batchCount(); batchOrdinal++) {
            long aBase = geometry.leftOffset(), bBase = geometry.rightOffset();
            long cBase = geometry.resultOffset();
            for (int axis = 0; axis < batch.length; axis++) {
                aBase = addCoordinate(aBase, coordinates[axis], aBatch[axis]);
                bBase = addCoordinate(bBase, coordinates[axis], bBatch[axis]);
                cBase = addCoordinate(cBase, coordinates[axis], cBatch[axis]);
            }
            for (long m = 0; m < geometry.m(); m++) for (long n = 0; n < geometry.n(); n++) {
                long a = addCoordinate(aBase, m, geometry.leftMStride());
                long b = addCoordinate(bBase, n, geometry.rightNStride());
                long output = addCoordinate(addCoordinate(cBase, m, geometry.resultMStride()),
                        n, geometry.resultNStride());
                switch (geometry.resultType()) {
                    case FLOAT64 -> {
                        double sum = 0.0;
                        for (long k = 0; k < geometry.k(); k++) {
                            double x = floating(geometry.leftType(), left, a);
                            double y = floating(geometry.rightType(), right, b);
                            sum = numericalForm == CpuMatmulIr.NumericalForm.FUSED_MULTIPLY_ADD
                                    ? Math.fma(x, y, sum) : sum + x * y;
                            a = Math.addExact(a, geometry.leftKStride());
                            b = Math.addExact(b, geometry.rightKStride());
                        }
                        ((double[]) result)[Math.toIntExact(output)] = sum;
                    }
                    case BFLOAT16, FLOAT32 -> {
                        float sum = 0.0f;
                        for (long k = 0; k < geometry.k(); k++) {
                            float x = (float) floating(geometry.leftType(), left, a);
                            float y = (float) floating(geometry.rightType(), right, b);
                            sum = numericalForm == CpuMatmulIr.NumericalForm.FUSED_MULTIPLY_ADD
                                    ? Math.fma(x, y, sum) : sum + x * y;
                            a = Math.addExact(a, geometry.leftKStride());
                            b = Math.addExact(b, geometry.rightKStride());
                        }
                        int address = Math.toIntExact(output);
                        if (geometry.resultType() == DataType.FLOAT32) ((float[]) result)[address] = sum;
                        else ((short[]) result)[address] = BFloat16Bits.fromFloat(sum);
                    }
                    case INT32 -> {
                        int sum = 0;
                        for (long k = 0; k < geometry.k(); k++) {
                            sum += integral32(geometry.leftType(), left, a)
                                    * integral32(geometry.rightType(), right, b);
                            a = Math.addExact(a, geometry.leftKStride());
                            b = Math.addExact(b, geometry.rightKStride());
                        }
                        ((int[]) result)[Math.toIntExact(output)] = sum;
                    }
                    case INT64 -> {
                        long sum = 0;
                        for (long k = 0; k < geometry.k(); k++) {
                            sum += integral64(geometry.leftType(), left, a)
                                    * integral64(geometry.rightType(), right, b);
                            a = Math.addExact(a, geometry.leftKStride());
                            b = Math.addExact(b, geometry.rightKStride());
                        }
                        ((long[]) result)[Math.toIntExact(output)] = sum;
                    }
                    case BOOL -> throw new AssertionError("BOOL MATMUL is not admitted");
                }
            }
            increment(coordinates, batch);
        }
    }

    private static long addCoordinate(long base, long coordinate, long stride) {
        return Math.addExact(base, Math.multiplyExact(coordinate, stride));
    }
    private static void increment(long[] coordinates, long[] extents) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            if (++coordinates[axis] < extents[axis]) return;
            coordinates[axis] = 0;
        }
    }
    private static double floating(DataType type, Object carrier, long address) {
        int index = Math.toIntExact(address);
        return switch (type) {
            case FLOAT64 -> ((double[]) carrier)[index];
            case FLOAT32 -> ((float[]) carrier)[index];
            case BFLOAT16 -> BFloat16Bits.toFloat(((short[]) carrier)[index]);
            default -> throw new IllegalArgumentException("floating MATMUL carrier type disagrees");
        };
    }
    private static int integral32(DataType type, Object carrier, long address) {
        if (type != DataType.INT32) throw new IllegalArgumentException("INT32 accumulator type disagrees");
        return ((int[]) carrier)[Math.toIntExact(address)];
    }
    private static long integral64(DataType type, Object carrier, long address) {
        int index = Math.toIntExact(address);
        return type == DataType.INT64 ? ((long[]) carrier)[index]
                : type == DataType.INT32 ? ((int[]) carrier)[index]
                : throwType();
    }
    private static long throwType() {
        throw new IllegalArgumentException("INT64 accumulator type disagrees");
    }
    private static void requireCarrier(DataType type, Object carrier) {
        Objects.requireNonNull(carrier, "carrier");
        boolean valid = switch (type) {
            case FLOAT64 -> carrier instanceof double[];
            case FLOAT32 -> carrier instanceof float[];
            case BFLOAT16 -> carrier instanceof short[];
            case INT32 -> carrier instanceof int[];
            case INT64 -> carrier instanceof long[];
            case BOOL -> false;
        };
        if (!valid) throw new IllegalArgumentException("primitive carrier does not match " + type);
    }
}
