package backend.kernels.cpu.elementwise.strided;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

final class StridedVectorSupport {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private StridedVectorSupport() {
    }

    static void binaryBroadcastRightF64(
            StridedElementWiseSemantics.BinaryKind kind,
            double[] left,
            int leftBase,
            double right,
            double[] out,
            int outBase,
            int cols
    ) {
        int width = F64.length();
        int upper = cols - (cols % width);
        int col = 0;
        DoubleVector rightVector = DoubleVector.broadcast(F64, right);
        switch (kind) {
            case ADD -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).add(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] + right;
            }
            case SUB -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).sub(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] - right;
            }
            case MUL -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).mul(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] * right;
            }
            case DIV -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).div(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] / right;
            }
            case MIN -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).min(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.min(left[leftBase + col], right);
            }
            case MAX -> {
                for (; col < upper; col += width) DoubleVector.fromArray(F64, left, leftBase + col).max(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.max(left[leftBase + col], right);
            }
        }
    }

    static void binaryBroadcastRightF32(
            StridedElementWiseSemantics.BinaryKind kind,
            float[] left,
            int leftBase,
            float right,
            float[] out,
            int outBase,
            int cols
    ) {
        int width = F32.length();
        int upper = cols - (cols % width);
        int col = 0;
        FloatVector rightVector = FloatVector.broadcast(F32, right);
        switch (kind) {
            case ADD -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).add(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] + right;
            }
            case SUB -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).sub(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] - right;
            }
            case MUL -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).mul(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] * right;
            }
            case DIV -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).div(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left[leftBase + col] / right;
            }
            case MIN -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).min(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.min(left[leftBase + col], right);
            }
            case MAX -> {
                for (; col < upper; col += width) FloatVector.fromArray(F32, left, leftBase + col).max(rightVector).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.max(left[leftBase + col], right);
            }
        }
    }

    static void binaryBroadcastLeftF64(
            StridedElementWiseSemantics.BinaryKind kind,
            double left,
            double[] right,
            int rightBase,
            double[] out,
            int outBase,
            int cols
    ) {
        int width = F64.length();
        int upper = cols - (cols % width);
        int col = 0;
        DoubleVector leftVector = DoubleVector.broadcast(F64, left);
        switch (kind) {
            case ADD -> {
                for (; col < upper; col += width) leftVector.add(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left + right[rightBase + col];
            }
            case SUB -> {
                for (; col < upper; col += width) leftVector.sub(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left - right[rightBase + col];
            }
            case MUL -> {
                for (; col < upper; col += width) leftVector.mul(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left * right[rightBase + col];
            }
            case DIV -> {
                for (; col < upper; col += width) leftVector.div(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left / right[rightBase + col];
            }
            case MIN -> {
                for (; col < upper; col += width) leftVector.min(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.min(left, right[rightBase + col]);
            }
            case MAX -> {
                for (; col < upper; col += width) leftVector.max(DoubleVector.fromArray(F64, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.max(left, right[rightBase + col]);
            }
        }
    }

    static void binaryBroadcastLeftF32(
            StridedElementWiseSemantics.BinaryKind kind,
            float left,
            float[] right,
            int rightBase,
            float[] out,
            int outBase,
            int cols
    ) {
        int width = F32.length();
        int upper = cols - (cols % width);
        int col = 0;
        FloatVector leftVector = FloatVector.broadcast(F32, left);
        switch (kind) {
            case ADD -> {
                for (; col < upper; col += width) leftVector.add(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left + right[rightBase + col];
            }
            case SUB -> {
                for (; col < upper; col += width) leftVector.sub(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left - right[rightBase + col];
            }
            case MUL -> {
                for (; col < upper; col += width) leftVector.mul(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left * right[rightBase + col];
            }
            case DIV -> {
                for (; col < upper; col += width) leftVector.div(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = left / right[rightBase + col];
            }
            case MIN -> {
                for (; col < upper; col += width) leftVector.min(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.min(left, right[rightBase + col]);
            }
            case MAX -> {
                for (; col < upper; col += width) leftVector.max(FloatVector.fromArray(F32, right, rightBase + col)).intoArray(out, outBase + col);
                for (; col < cols; col++) out[outBase + col] = Math.max(left, right[rightBase + col]);
            }
        }
    }

    static void fillRowF64(double[] out, int outBase, int cols, double value) {
        int width = F64.length();
        int upper = cols - (cols % width);
        int col = 0;
        DoubleVector vector = DoubleVector.broadcast(F64, value);
        for (; col < upper; col += width) {
            vector.intoArray(out, outBase + col);
        }
        for (; col < cols; col++) {
            out[outBase + col] = value;
        }
    }

    static void fillRowF32(float[] out, int outBase, int cols, float value) {
        int width = F32.length();
        int upper = cols - (cols % width);
        int col = 0;
        FloatVector vector = FloatVector.broadcast(F32, value);
        for (; col < upper; col += width) {
            vector.intoArray(out, outBase + col);
        }
        for (; col < cols; col++) {
            out[outBase + col] = value;
        }
    }
}
