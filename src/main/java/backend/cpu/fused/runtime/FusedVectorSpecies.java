package backend.cpu.fused.runtime;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class FusedVectorSpecies {
    public static final int PREFERRED_WIDTH = 0;

    private FusedVectorSpecies() {
    }

    public static VectorSpecies<Float> f32(int width) {
        return switch (normalizeF32LaneCount(width)) {
            case 2 -> FloatVector.SPECIES_64;
            case 4 -> FloatVector.SPECIES_128;
            case 8 -> FloatVector.SPECIES_256;
            case 16 -> FloatVector.SPECIES_512;
            default -> FloatVector.SPECIES_PREFERRED;
        };
    }

    public static VectorSpecies<Double> f64(int width) {
        return switch (normalizeF64LaneCount(width)) {
            case 1 -> DoubleVector.SPECIES_64;
            case 2 -> DoubleVector.SPECIES_128;
            case 4 -> DoubleVector.SPECIES_256;
            case 8 -> DoubleVector.SPECIES_512;
            default -> DoubleVector.SPECIES_PREFERRED;
        };
    }

    public static String f32FieldName(int width) {
        return switch (normalizeF32LaneCount(width)) {
            case 2 -> "SPECIES_64";
            case 4 -> "SPECIES_128";
            case 8 -> "SPECIES_256";
            case 16 -> "SPECIES_512";
            default -> "SPECIES_PREFERRED";
        };
    }

    public static String f64FieldName(int width) {
        return switch (normalizeF64LaneCount(width)) {
            case 1 -> "SPECIES_64";
            case 2 -> "SPECIES_128";
            case 4 -> "SPECIES_256";
            case 8 -> "SPECIES_512";
            default -> "SPECIES_PREFERRED";
        };
    }

    public static int normalizeLaneCount(int width) {
        if (width <= 0) {
            return PREFERRED_WIDTH;
        }
        if (width <= 1) {
            return 1;
        }
        if (width <= 2) {
            return 2;
        }
        if (width <= 4) {
            return 4;
        }
        if (width <= 8) {
            return 8;
        }
        return 16;
    }

    public static int normalizeF32LaneCount(int width) {
        return normalizeLaneCount(width);
    }

    public static int normalizeF64LaneCount(int width) {
        int normalized = normalizeLaneCount(width);
        return normalized > 8 ? 8 : normalized;
    }
}
