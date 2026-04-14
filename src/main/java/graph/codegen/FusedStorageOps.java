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

    public static double loadScalarBF16Array(short[] src, int index) {
        return CpuDTypeOps.fromBFloat16Bits(src[index]);
    }

    public static void storeScalarBF16Array(short[] dst, int index, double value) {
        dst[index] = CpuDTypeOps.toBFloat16Bits((float) value);
    }

    public static DoubleVector loadVectorF64(Tensor tensor, int index) {
        return DoubleVector.fromArray(DOUBLE_SPECIES, tensor.getFloat64Data(), index);
    }

    public static FloatVector loadVectorF32(Tensor tensor, int index) {
        return FloatVector.fromArray(FLOAT_SPECIES, tensor.getFloat32Data(), index);
    }

    public static DoubleVector loadVectorF64Array(double[] src, int index, int width) {
        return DoubleVector.fromArray(speciesF64(width), src, index);
    }

    public static FloatVector loadVectorF32Array(float[] src, int index, int width) {
        return FloatVector.fromArray(speciesF32(width), src, index);
    }

    public static void storeVectorF64(Tensor tensor, int index, DoubleVector vector) {
        vector.intoArray(tensor.getFloat64Data(), index);
    }

    public static void storeVectorF32(Tensor tensor, int index, FloatVector vector) {
        vector.intoArray(tensor.getFloat32Data(), index);
    }

    public static Object loadVectorBF16Array(short[] src, int index) {
        double[] lanes = new double[DOUBLE_SPECIES.length()];
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = CpuDTypeOps.fromBFloat16Bits(src[index + i]);
        }
        return DoubleVector.fromArray(DOUBLE_SPECIES, lanes, 0);
    }

    public static Object loadMaskF32Array(byte[] src, int index, int width) {
        VectorSpecies<Float> species = speciesF32(width);
        long bits = 0L;
        for (int i = 0; i < species.length(); i++) {
            if (src[index + i] != 0) {
                bits |= (1L << i);
            }
        }
        return VectorMask.fromLong(species, bits);
    }

    public static Object loadMaskF64Array(byte[] src, int index, int width) {
        VectorSpecies<Double> species = speciesF64(width);
        long bits = 0L;
        for (int i = 0; i < species.length(); i++) {
            if (src[index + i] != 0) {
                bits |= (1L << i);
            }
        }
        return VectorMask.fromLong(species, bits);
    }

    public static void storeMaskF32Array(byte[] dst, int index, Object maskObject, int width) {
        VectorMask<Float> mask = (VectorMask<Float>) maskObject;
        for (int i = 0; i < speciesF32(width).length(); i++) {
            dst[index + i] = mask.laneIsSet(i) ? (byte) 1 : (byte) 0;
        }
    }

    public static void storeMaskF64Array(byte[] dst, int index, Object maskObject, int width) {
        VectorMask<Double> mask = (VectorMask<Double>) maskObject;
        for (int i = 0; i < speciesF64(width).length(); i++) {
            dst[index + i] = mask.laneIsSet(i) ? (byte) 1 : (byte) 0;
        }
    }

    public static void storeVectorBF16Array(short[] dst, int index, Object vector) {
        double[] lanes = new double[DOUBLE_SPECIES.length()];
        ((DoubleVector) vector).intoArray(lanes, 0);
        for (int i = 0; i < lanes.length; i++) {
            dst[index + i] = CpuDTypeOps.toBFloat16Bits((float) lanes[i]);
        }
    }

    private static VectorSpecies<Float> speciesF32(int width) {
        return switch (normalizeWidth(width)) {
            case 2 -> FloatVector.SPECIES_64;
            case 4 -> FloatVector.SPECIES_128;
            case 8 -> FloatVector.SPECIES_256;
            default -> FLOAT_SPECIES;
        };
    }

    private static VectorSpecies<Double> speciesF64(int width) {
        return switch (normalizeWidth(width)) {
            case 1 -> DoubleVector.SPECIES_64;
            case 2 -> DoubleVector.SPECIES_128;
            case 4 -> DoubleVector.SPECIES_256;
            case 8 -> DoubleVector.SPECIES_512;
            default -> DOUBLE_SPECIES;
        };
    }

    private static int normalizeWidth(int width) {
        if (width <= 1) {
            return 1;
        }
        if (width <= 2) {
            return 2;
        }
        if (width <= 4) {
            return 4;
        }
        return 8;
    }
}
