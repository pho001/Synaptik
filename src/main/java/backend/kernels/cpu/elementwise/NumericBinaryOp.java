package backend.kernels.cpu.elementwise;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

public enum NumericBinaryOp {
    ADD {
        @Override
        public double applyF64(double left, double right) { return left + right; }

        @Override
        public float applyF32(float left, float right) { return left + right; }

        @Override
        public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) { return left.add(right); }

        @Override
        public FloatVector applyVectorF32(FloatVector left, FloatVector right) { return left.add(right); }
    },
    SUB {
        @Override
        public double applyF64(double left, double right) { return left - right; }

        @Override
        public float applyF32(float left, float right) { return left - right; }

        @Override
        public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) { return left.sub(right); }

        @Override
        public FloatVector applyVectorF32(FloatVector left, FloatVector right) { return left.sub(right); }
    },
    MUL {
        @Override
        public double applyF64(double left, double right) { return left * right; }

        @Override
        public float applyF32(float left, float right) { return left * right; }

        @Override
        public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) { return left.mul(right); }

        @Override
        public FloatVector applyVectorF32(FloatVector left, FloatVector right) { return left.mul(right); }
    },
    DIV {
        @Override
        public double applyF64(double left, double right) { return left / right; }

        @Override
        public float applyF32(float left, float right) { return left / right; }

        @Override
        public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) { return left.div(right); }

        @Override
        public FloatVector applyVectorF32(FloatVector left, FloatVector right) { return left.div(right); }
    },
    MIN {
        @Override
        public double applyF64(double left, double right) { return Math.min(left, right); }

        @Override
        public float applyF32(float left, float right) { return Math.min(left, right); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) { return left.lanewise(VectorOperators.MIN, right); }

        @Override
        public FloatVector applyVectorF32(FloatVector left, FloatVector right) { return left.lanewise(VectorOperators.MIN, right); }
    },
    MAX {
        @Override
        public double applyF64(double left, double right) { return Math.max(left, right); }

        @Override
        public float applyF32(float left, float right) { return Math.max(left, right); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) { return left.lanewise(VectorOperators.MAX, right); }

        @Override
        public FloatVector applyVectorF32(FloatVector left, FloatVector right) { return left.lanewise(VectorOperators.MAX, right); }
    };

    public abstract double applyF64(double left, double right);

    public abstract float applyF32(float left, float right);

    public float applyBF16(float left, float right) {
        return applyF32(left, right);
    }

    public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) {
        throw new UnsupportedOperationException(name() + " does not support F64 vector execution");
    }

    public FloatVector applyVectorF32(FloatVector left, FloatVector right) {
        throw new UnsupportedOperationException(name() + " does not support F32 vector execution");
    }

    public boolean supportsVectorF64() {
        return true;
    }

    public boolean supportsVectorF32() {
        return true;
    }
}
