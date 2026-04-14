package graph.codegen;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

public final class FusedBroadcastVectorOps {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;

    private FusedBroadcastVectorOps() {}

    public static FloatVector loadVectorF32(FusedBroadcastCursor cursor, float[] input, int width) {
        VectorSpecies<Float> species = speciesF32(width);
        if (cursor.isScalarBroadcast()) {
            return FloatVector.broadcast(species, input[cursor.idx()]);
        }
        int[] idx = cursor.nextIndices(species.length());
        return FloatVector.fromArray(species, input, 0, idx, 0);
    }

    public static DoubleVector loadVectorF64(FusedBroadcastCursor cursor, double[] input, int width) {
        VectorSpecies<Double> species = speciesF64(width);
        if (cursor.isScalarBroadcast()) {
            return DoubleVector.broadcast(species, input[cursor.idx()]);
        }
        int[] idx = cursor.nextIndices(species.length());
        return DoubleVector.fromArray(species, input, 0, idx, 0);
    }

    public static Object loadMaskF32(FusedBroadcastCursor cursor, byte[] input, int width) {
        VectorSpecies<Float> species = speciesF32(width);
        if (cursor.isScalarBroadcast()) {
            return maskFromBroadcastF32(species, input[cursor.idx()] != 0);
        }
        int[] idx = cursor.nextIndices(species.length());
        return maskFromIndicesF32(species, input, idx);
    }

    public static Object loadMaskF64(FusedBroadcastCursor cursor, byte[] input, int width) {
        VectorSpecies<Double> species = speciesF64(width);
        if (cursor.isScalarBroadcast()) {
            return maskFromBroadcastF64(species, input[cursor.idx()] != 0);
        }
        int[] idx = cursor.nextIndices(species.length());
        return maskFromIndicesF64(species, input, idx);
    }

    private static VectorSpecies<Float> speciesF32(int width) {
        return switch (normalizeWidth(width)) {
            case 2 -> FloatVector.SPECIES_64;
            case 4 -> FloatVector.SPECIES_128;
            case 8 -> FloatVector.SPECIES_256;
            default -> F32;
        };
    }

    private static VectorSpecies<Double> speciesF64(int width) {
        return switch (normalizeWidth(width)) {
            case 1 -> DoubleVector.SPECIES_64;
            case 2 -> DoubleVector.SPECIES_128;
            case 4 -> DoubleVector.SPECIES_256;
            case 8 -> DoubleVector.SPECIES_512;
            default -> F64;
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
