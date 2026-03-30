package graph.codegen;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class FusedBroadcastVectorOps {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;

    private FusedBroadcastVectorOps() {}

    public static FloatVector loadVectorF32(FusedBroadcastCursor cursor, float[] input) {
        int[] idx = cursor.nextIndices(F32.length());
        return FloatVector.fromArray(F32, input, 0, idx, 0);
    }

    public static DoubleVector loadVectorF64(FusedBroadcastCursor cursor, double[] input) {
        int[] idx = cursor.nextIndices(F64.length());
        return DoubleVector.fromArray(F64, input, 0, idx, 0);
    }
}
