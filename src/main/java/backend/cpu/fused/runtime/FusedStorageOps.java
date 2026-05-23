package backend.cpu.fused.runtime;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuDTypeOps;
import tensor.Tensor;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Internal runtime helper methods invoked by generated fused kernels for tensor storage access.
 */
public final class FusedStorageOps {
    private FusedStorageOps() {}

    public static double loadScalar(Tensor tensor, int index, int mode) {
        return switch (mode) {
            case FusedDTypeOps.MODE_F64 -> TensorInternalAccess.float64Data(tensor)[index];
            case FusedDTypeOps.MODE_F32 -> TensorInternalAccess.float32Data(tensor)[index];
            case FusedDTypeOps.MODE_BF16 -> CpuDTypeOps.fromBFloat16Bits(TensorInternalAccess.bfloat16Data(tensor)[index]);
            default -> throw new IllegalArgumentException("Unsupported mode: " + mode);
        };
    }

    public static void storeScalar(Tensor tensor, int index, double value, int mode) {
        switch (mode) {
            case FusedDTypeOps.MODE_F64 -> TensorInternalAccess.float64Data(tensor)[index] = value;
            case FusedDTypeOps.MODE_F32 -> TensorInternalAccess.float32Data(tensor)[index] = (float) value;
            case FusedDTypeOps.MODE_BF16 -> TensorInternalAccess.bfloat16Data(tensor)[index] = CpuDTypeOps.toBFloat16Bits((float) value);
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
                float[] lanes = new float[FloatVector.SPECIES_PREFERRED.length()];
                short[] src = TensorInternalAccess.bfloat16Data(tensor);
                for (int i = 0; i < lanes.length; i++) lanes[i] = CpuDTypeOps.fromBFloat16Bits(src[index + i]);
                return FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, lanes, 0);
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
                float[] lanes = new float[FloatVector.SPECIES_PREFERRED.length()];
                ((FloatVector) vector).intoArray(lanes, 0);
                short[] dst = TensorInternalAccess.bfloat16Data(tensor);
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

    public static float loadScalarF32Segment(MemorySegment segment, int index) {
        return segment.get(JAVA_FLOAT, (long) index * Float.BYTES);
    }

    public static double loadScalarF64Segment(MemorySegment segment, int index) {
        return segment.get(JAVA_DOUBLE, (long) index * Double.BYTES);
    }

    public static double loadScalarBF16Segment(MemorySegment segment, int index) {
        return CpuDTypeOps.fromBFloat16Bits(segment.get(JAVA_SHORT, (long) index * Short.BYTES));
    }

    public static int loadScalarBoolSegment(MemorySegment segment, int index) {
        return segment.get(JAVA_BYTE, index) == 0 ? 0 : 1;
    }

    public static void storeScalarF32Segment(MemorySegment segment, int index, float value) {
        segment.set(JAVA_FLOAT, (long) index * Float.BYTES, value);
    }

    public static void storeScalarF64Segment(MemorySegment segment, int index, double value) {
        segment.set(JAVA_DOUBLE, (long) index * Double.BYTES, value);
    }

    public static void storeScalarBF16Segment(MemorySegment segment, int index, double value) {
        segment.set(JAVA_SHORT, (long) index * Short.BYTES, CpuDTypeOps.toBFloat16Bits((float) value));
    }

    public static void storeScalarBoolSegment(MemorySegment segment, int index, int value) {
        segment.set(JAVA_BYTE, index, value == 0 ? (byte) 0 : (byte) 1);
    }

    public static DoubleVector loadVectorF64(Tensor tensor, int index) {
        return DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, TensorInternalAccess.float64Data(tensor), index);
    }

    public static FloatVector loadVectorF32(Tensor tensor, int index) {
        return FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, TensorInternalAccess.float32Data(tensor), index);
    }

    public static DoubleVector loadVectorF64Array(double[] src, int index, int width) {
        return DoubleVector.fromArray(FusedVectorSpecies.f64(width), src, index);
    }

    public static FloatVector loadVectorF32Array(float[] src, int index, int width) {
        return FloatVector.fromArray(FusedVectorSpecies.f32(width), src, index);
    }

    public static void storeVectorF64(Tensor tensor, int index, DoubleVector vector) {
        vector.intoArray(TensorInternalAccess.float64Data(tensor), index);
    }

    public static void storeVectorF32(Tensor tensor, int index, FloatVector vector) {
        vector.intoArray(TensorInternalAccess.float32Data(tensor), index);
    }

    public static Object loadVectorBF16Array(short[] src, int index, int width) {
        var species = FusedVectorSpecies.f32(width);
        float[] lanes = new float[species.length()];
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = CpuDTypeOps.fromBFloat16Bits(src[index + i]);
        }
        return FloatVector.fromArray(species, lanes, 0);
    }

    public static Object loadMaskF32Array(byte[] src, int index, int width) {
        var species = FusedVectorSpecies.f32(width);
        long bits = 0L;
        for (int i = 0; i < species.length(); i++) {
            if (src[index + i] != 0) {
                bits |= (1L << i);
            }
        }
        return VectorMask.fromLong(species, bits);
    }

    public static Object loadMaskF64Array(byte[] src, int index, int width) {
        var species = FusedVectorSpecies.f64(width);
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
        for (int i = 0; i < FusedVectorSpecies.f32(width).length(); i++) {
            dst[index + i] = mask.laneIsSet(i) ? (byte) 1 : (byte) 0;
        }
    }

    public static void storeMaskF64Array(byte[] dst, int index, Object maskObject, int width) {
        VectorMask<Double> mask = (VectorMask<Double>) maskObject;
        for (int i = 0; i < FusedVectorSpecies.f64(width).length(); i++) {
            dst[index + i] = mask.laneIsSet(i) ? (byte) 1 : (byte) 0;
        }
    }

    public static void storeVectorBF16Array(short[] dst, int index, Object vector) {
        FloatVector typed = (FloatVector) vector;
        float[] lanes = new float[typed.species().length()];
        typed.intoArray(lanes, 0);
        for (int i = 0; i < lanes.length; i++) {
            dst[index + i] = CpuDTypeOps.toBFloat16Bits((float) lanes[i]);
        }
    }

}
