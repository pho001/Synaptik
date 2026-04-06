package graph.codegen;

import backend.kernels.cpu.CpuDTypeOps;
import tensor.Tensor;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

public final class FusedStorageOps {
    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private FusedStorageOps() {}

    public static double loadScalar(Tensor tensor, int index, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> tensor.getFloat64Data()[index];
            case FusedDTypeOps.MODE_F32 -> tensor.getFloat32Data()[index];
            case FusedDTypeOps.MODE_BF16 -> CpuDTypeOps.fromBFloat16Bits(tensor.getBFloat16Data()[index]);
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static void storeScalar(Tensor tensor, int index, double value, int mode) {
        switch (mode) {
            case FusedDTypeOps.MODE_F64 -> tensor.getFloat64Data()[index] = value;
            case FusedDTypeOps.MODE_F32 -> tensor.getFloat32Data()[index] = (float) value;
            case FusedDTypeOps.MODE_BF16 -> tensor.getBFloat16Data()[index] = CpuDTypeOps.toBFloat16Bits((float) value);
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        }
    }

    public static Object loadVector(Tensor tensor, int index, int mode) {
        switch (mode) {
            case FusedDTypeOps.MODE_F64 -> {
                return loadVectorF64(tensor, index);
            }
            case FusedDTypeOps.MODE_F32 -> {
                return loadVectorF32(tensor, index);
            }
            case FusedDTypeOps.MODE_BF16 -> {
                double[] lanes = new double[DOUBLE_SPECIES.length()];
                short[] src = tensor.getBFloat16Data();
                for (int i = 0; i < lanes.length; i++) lanes[i] = CpuDTypeOps.fromBFloat16Bits(src[index + i]);
                return DoubleVector.fromArray(DOUBLE_SPECIES, lanes, 0);
            }
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        }
    }

    public static void storeVector(Tensor tensor, int index, Object vector, int mode) {
        switch (mode) {
            case FusedDTypeOps.MODE_F64 -> storeVectorF64(tensor, index, (DoubleVector) vector);
            case FusedDTypeOps.MODE_F32 -> {
                storeVectorF32(tensor, index, (FloatVector) vector);
            }
            case FusedDTypeOps.MODE_BF16 -> {
                double[] lanes = new double[DOUBLE_SPECIES.length()];
                ((DoubleVector) vector).intoArray(lanes, 0);
                short[] dst = tensor.getBFloat16Data();
                for (int i = 0; i < lanes.length; i++) dst[index + i] = CpuDTypeOps.toBFloat16Bits((float) lanes[i]);
            }
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        }
    }

    public static double loadScalarF16Array(short[] src, int index) {
        return CpuDTypeOps.fromBFloat16Bits(src[index]);
    }

    public static void storeScalarF16Array(short[] dst, int index, double value) {
        dst[index] = CpuDTypeOps.toBFloat16Bits((float) value);
    }

    public static DoubleVector loadVectorF64(Tensor tensor, int index) {
        return DoubleVector.fromArray(DOUBLE_SPECIES, tensor.getFloat64Data(), index);
    }

    public static FloatVector loadVectorF32(Tensor tensor, int index) {
        return FloatVector.fromArray(FLOAT_SPECIES, tensor.getFloat32Data(), index);
    }

    public static void storeVectorF64(Tensor tensor, int index, DoubleVector vector) {
        vector.intoArray(tensor.getFloat64Data(), index);
    }

    public static void storeVectorF32(Tensor tensor, int index, FloatVector vector) {
        vector.intoArray(tensor.getFloat32Data(), index);
    }

    public static Object loadVectorF16Array(short[] src, int index) {
        double[] lanes = new double[DOUBLE_SPECIES.length()];
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = CpuDTypeOps.fromBFloat16Bits(src[index + i]);
        }
        return DoubleVector.fromArray(DOUBLE_SPECIES, lanes, 0);
    }

    public static Object loadMaskF32Array(byte[] src, int index) {
        boolean[] lanes = new boolean[FLOAT_SPECIES.length()];
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = src[index + i] != 0;
        }
        return VectorMask.fromArray(FLOAT_SPECIES, lanes, 0);
    }

    public static Object loadMaskF64Array(byte[] src, int index) {
        boolean[] lanes = new boolean[DOUBLE_SPECIES.length()];
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = src[index + i] != 0;
        }
        return VectorMask.fromArray(DOUBLE_SPECIES, lanes, 0);
    }

    public static void storeMaskF32Array(byte[] dst, int index, Object maskObject) {
        VectorMask<Float> mask = (VectorMask<Float>) maskObject;
        for (int i = 0; i < FLOAT_SPECIES.length(); i++) {
            dst[index + i] = mask.laneIsSet(i) ? (byte) 1 : (byte) 0;
        }
    }

    public static void storeMaskF64Array(byte[] dst, int index, Object maskObject) {
        VectorMask<Double> mask = (VectorMask<Double>) maskObject;
        for (int i = 0; i < DOUBLE_SPECIES.length(); i++) {
            dst[index + i] = mask.laneIsSet(i) ? (byte) 1 : (byte) 0;
        }
    }

    public static void storeVectorF16Array(short[] dst, int index, Object vector) {
        double[] lanes = new double[DOUBLE_SPECIES.length()];
        ((DoubleVector) vector).intoArray(lanes, 0);
        for (int i = 0; i < lanes.length; i++) {
            dst[index + i] = CpuDTypeOps.toBFloat16Bits((float) lanes[i]);
        }
    }
}
