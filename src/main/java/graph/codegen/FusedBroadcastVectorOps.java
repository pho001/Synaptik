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
        int[] idx = cursor.nextIndices(species.length());
        return FloatVector.fromArray(species, input, 0, idx, 0);
    }

    public static DoubleVector loadVectorF64(FusedBroadcastCursor cursor, double[] input, int width) {
        VectorSpecies<Double> species = speciesF64(width);
        int[] idx = cursor.nextIndices(species.length());
        return DoubleVector.fromArray(species, input, 0, idx, 0);
    }

    public static Object loadMaskF32(FusedBroadcastCursor cursor, byte[] input, int width) {
        VectorSpecies<Float> species = speciesF32(width);
        int[] idx = cursor.nextIndices(species.length());
        boolean[] lanes = new boolean[species.length()];
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = input[idx[i]] != 0;
        }
        return VectorMask.fromArray(species, lanes, 0);
    }

    public static Object loadMaskF64(FusedBroadcastCursor cursor, byte[] input, int width) {
        VectorSpecies<Double> species = speciesF64(width);
        int[] idx = cursor.nextIndices(species.length());
        boolean[] lanes = new boolean[species.length()];
        for (int i = 0; i < lanes.length; i++) {
            lanes[i] = input[idx[i]] != 0;
        }
        return VectorMask.fromArray(species, lanes, 0);
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
}
