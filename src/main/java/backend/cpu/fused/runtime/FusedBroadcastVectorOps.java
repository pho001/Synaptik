package backend.cpu.fused.runtime;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * Internal vector load helpers for broadcasted fused inputs.
 */
public final class FusedBroadcastVectorOps {
    private FusedBroadcastVectorOps() {}

    public static FloatVector loadVectorF32(FusedBroadcastCursor cursor, float[] input, int width) {
        var species = FusedVectorSpecies.f32(width);
        if (cursor.isScalarBroadcast()) {
            return FloatVector.broadcast(species, input[cursor.idx()]);
        }
        if (cursor.staysWithinInnermostDimension(species.length())) {
            int base = cursor.idx();
            int laneStride = cursor.innermostLaneStride();
            cursor.advance(species.length());
            if (laneStride == 0) {
                return FloatVector.broadcast(species, input[base]);
            }
            if (laneStride == 1) {
                return FloatVector.fromArray(species, input, base);
            }
        }
        int[] idx = cursor.nextIndices(species.length());
        return FloatVector.fromArray(species, input, 0, idx, 0);
    }

    public static DoubleVector loadVectorF64(FusedBroadcastCursor cursor, double[] input, int width) {
        var species = FusedVectorSpecies.f64(width);
        if (cursor.isScalarBroadcast()) {
            return DoubleVector.broadcast(species, input[cursor.idx()]);
        }
        if (cursor.staysWithinInnermostDimension(species.length())) {
            int base = cursor.idx();
            int laneStride = cursor.innermostLaneStride();
            cursor.advance(species.length());
            if (laneStride == 0) {
                return DoubleVector.broadcast(species, input[base]);
            }
            if (laneStride == 1) {
                return DoubleVector.fromArray(species, input, base);
            }
        }
        int[] idx = cursor.nextIndices(species.length());
        return DoubleVector.fromArray(species, input, 0, idx, 0);
    }

    public static Object loadVectorBF16(FusedBroadcastCursor cursor, short[] input, int width) {
        var species = FusedVectorSpecies.f32(width);
        if (cursor.isScalarBroadcast()) {
            return FloatVector.broadcast(species, backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(input[cursor.idx()]));
        }
        float[] scratch = new float[species.length()];
        if (cursor.staysWithinInnermostDimension(species.length())) {
            int base = cursor.idx();
            int laneStride = cursor.innermostLaneStride();
            cursor.advance(species.length());
            if (laneStride == 0) {
                return FloatVector.broadcast(species, backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(input[base]));
            }
            if (laneStride == 1) {
                for (int lane = 0; lane < species.length(); lane++) {
                    scratch[lane] = backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(input[base + lane]);
                }
                return FloatVector.fromArray(species, scratch, 0);
            }
        }
        int[] idx = cursor.nextIndices(species.length());
        for (int lane = 0; lane < species.length(); lane++) {
            scratch[lane] = backend.cpu.kernels.CpuDTypeOps.fromBFloat16Bits(input[idx[lane]]);
        }
        return FloatVector.fromArray(species, scratch, 0);
    }

    public static Object loadMaskF32(FusedBroadcastCursor cursor, byte[] input, int width) {
        var species = FusedVectorSpecies.f32(width);
        if (cursor.isScalarBroadcast()) {
            return maskFromBroadcastF32(species, input[cursor.idx()] != 0);
        }
        if (cursor.staysWithinInnermostDimension(species.length()) && cursor.innermostLaneStride() == 0) {
            int base = cursor.idx();
            cursor.advance(species.length());
            return maskFromBroadcastF32(species, input[base] != 0);
        }
        int[] idx = cursor.nextIndices(species.length());
        return maskFromIndicesF32(species, input, idx);
    }

    public static Object loadMaskF64(FusedBroadcastCursor cursor, byte[] input, int width) {
        var species = FusedVectorSpecies.f64(width);
        if (cursor.isScalarBroadcast()) {
            return maskFromBroadcastF64(species, input[cursor.idx()] != 0);
        }
        if (cursor.staysWithinInnermostDimension(species.length()) && cursor.innermostLaneStride() == 0) {
            int base = cursor.idx();
            cursor.advance(species.length());
            return maskFromBroadcastF64(species, input[base] != 0);
        }
        int[] idx = cursor.nextIndices(species.length());
        return maskFromIndicesF64(species, input, idx);
    }

    private static VectorMask<Float> maskFromBroadcastF32(VectorSpecies<Float> species, boolean set) {
        return VectorMask.fromLong(species, set ? allLanesMask(species.length()) : 0L);
    }

    private static VectorMask<Double> maskFromBroadcastF64(VectorSpecies<Double> species, boolean set) {
        return VectorMask.fromLong(species, set ? allLanesMask(species.length()) : 0L);
    }

    private static VectorMask<Float> maskFromIndicesF32(VectorSpecies<Float> species, byte[] input, int[] idx) {
        long bits = 0L;
        for (int i = 0; i < species.length(); i++) {
            if (input[idx[i]] != 0) {
                bits |= (1L << i);
            }
        }
        return VectorMask.fromLong(species, bits);
    }

    private static VectorMask<Double> maskFromIndicesF64(VectorSpecies<Double> species, byte[] input, int[] idx) {
        long bits = 0L;
        for (int i = 0; i < species.length(); i++) {
            if (input[idx[i]] != 0) {
                bits |= (1L << i);
            }
        }
        return VectorMask.fromLong(species, bits);
    }

    private static long allLanesMask(int laneCount) {
        if (laneCount >= Long.SIZE) {
            return -1L;
        }
        return (1L << laneCount) - 1L;
    }
}
