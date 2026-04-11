package backend.kernels.cpu.elementwise;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import utils.FastExp;

public enum UnaryOp {
    NEG {
        @Override
        public double applyF64(double value) { return -value; }

        @Override
        public float applyF32(float value) { return -value; }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) { return value.neg(); }

        @Override
        public FloatVector applyVectorF32(FloatVector value) { return value.neg(); }
    },
    INV {
        @Override
        public double applyF64(double value) { return 1.0d / value; }

        @Override
        public float applyF32(float value) { return 1.0f / value; }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) { return DoubleVector.broadcast(value.species(), 1.0d).div(value); }

        @Override
        public FloatVector applyVectorF32(FloatVector value) { return FloatVector.broadcast(value.species(), 1.0f).div(value); }
    },
    RELU {
        @Override
        public double applyF64(double value) { return Math.max(0.0d, value); }

        @Override
        public float applyF32(float value) { return Math.max(0.0f, value); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) { return value.max(DoubleVector.zero(value.species())); }

        @Override
        public FloatVector applyVectorF32(FloatVector value) { return value.max(FloatVector.zero(value.species())); }
    },
    ABS {
        @Override
        public double applyF64(double value) { return Math.abs(value); }

        @Override
        public float applyF32(float value) { return Math.abs(value); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) { return value.lanewise(VectorOperators.ABS); }

        @Override
        public FloatVector applyVectorF32(FloatVector value) { return value.lanewise(VectorOperators.ABS); }
    },
    EXP {
        @Override
        public double applyF64(double value) { return Math.exp(value); }

        @Override
        public float applyF32(float value) { return (float) Math.exp(value); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) { return value.lanewise(VectorOperators.EXP); }

        @Override
        public FloatVector applyVectorF32(FloatVector value) { return value.lanewise(VectorOperators.EXP); }
    },
    FAST_EXP {
        @Override
        public double applyF64(double value) { return FastExp.fastExpF64(value); }

        @Override
        public float applyF32(float value) { return FastExp.fastExpF32(value); }

        @Override
        public boolean supportsVectorF64() { return false; }

        @Override
        public boolean supportsVectorF32() { return false; }
    },
    LOG {
        @Override
        public double applyF64(double value) { return Math.log(value); }

        @Override
        public float applyF32(float value) { return (float) Math.log(value); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) { return value.lanewise(VectorOperators.LOG); }

        @Override
        public FloatVector applyVectorF32(FloatVector value) { return value.lanewise(VectorOperators.LOG); }
    },
    TANH {
        @Override
        public double applyF64(double value) { return Math.tanh(value); }

        @Override
        public float applyF32(float value) { return (float) Math.tanh(value); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) { return value.lanewise(VectorOperators.TANH); }

        @Override
        public FloatVector applyVectorF32(FloatVector value) { return value.lanewise(VectorOperators.TANH); }
    },
    FAST_TANH {
        @Override
        public double applyF64(double value) { return FastExp.fastTanhF64(value); }

        @Override
        public float applyF32(float value) { return FastExp.fastTanhF32(value); }

        @Override
        public boolean supportsVectorF64() { return false; }

        @Override
        public boolean supportsVectorF32() { return false; }
    },
    SQRT {
        @Override
        public double applyF64(double value) { return Math.sqrt(value); }

        @Override
        public float applyF32(float value) { return (float) Math.sqrt(value); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) { return value.lanewise(VectorOperators.SQRT); }

        @Override
        public FloatVector applyVectorF32(FloatVector value) { return value.lanewise(VectorOperators.SQRT); }
    },
    SIGMOID {
        @Override
        public double applyF64(double value) { return 1.0d / (1.0d + Math.exp(-value)); }

        @Override
        public float applyF32(float value) { return 1.0f / (1.0f + (float) Math.exp(-value)); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value) {
            DoubleVector half = DoubleVector.broadcast(value.species(), 0.5d);
            DoubleVector one = DoubleVector.broadcast(value.species(), 1.0d);
            return value.mul(half).lanewise(VectorOperators.TANH).add(one).mul(half);
        }

        @Override
        public FloatVector applyVectorF32(FloatVector value) {
            FloatVector half = FloatVector.broadcast(value.species(), 0.5f);
            FloatVector one = FloatVector.broadcast(value.species(), 1.0f);
            return value.mul(half).lanewise(VectorOperators.TANH).add(one).mul(half);
        }
    };

    public abstract double applyF64(double value);

    public abstract float applyF32(float value);

    public float applyBF16(float value) {
        return applyF32(value);
    }

    public DoubleVector applyVectorF64(DoubleVector value) {
        throw new UnsupportedOperationException(name() + " does not support F64 vector execution");
    }

    public FloatVector applyVectorF32(FloatVector value) {
        throw new UnsupportedOperationException(name() + " does not support F32 vector execution");
    }

    public boolean supportsVectorF64() {
        return true;
    }

    public boolean supportsVectorF32() {
        return true;
    }
}
