package backend.kernels.cpu.elementwise;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

public enum ScalarUnaryOp {
    CLAMP_MIN {
        @Override
        public double applyF64(double value, double parameter) { return Math.max(parameter, value); }

        @Override
        public float applyF32(float value, float parameter) { return Math.max(parameter, value); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value, double parameter) {
            return value.max(DoubleVector.broadcast(value.species(), parameter));
        }

        @Override
        public FloatVector applyVectorF32(FloatVector value, float parameter) {
            return value.max(FloatVector.broadcast(value.species(), parameter));
        }
    },
    CLAMP_MAX {
        @Override
        public double applyF64(double value, double parameter) { return Math.min(parameter, value); }

        @Override
        public float applyF32(float value, float parameter) { return Math.min(parameter, value); }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value, double parameter) {
            return value.min(DoubleVector.broadcast(value.species(), parameter));
        }

        @Override
        public FloatVector applyVectorF32(FloatVector value, float parameter) {
            return value.min(FloatVector.broadcast(value.species(), parameter));
        }
    },
    MUL_SCALAR {
        @Override
        public double applyF64(double value, double parameter) { return value * parameter; }

        @Override
        public float applyF32(float value, float parameter) { return value * parameter; }

        @Override
        public DoubleVector applyVectorF64(DoubleVector value, double parameter) {
            return value.mul(DoubleVector.broadcast(value.species(), parameter));
        }

        @Override
        public FloatVector applyVectorF32(FloatVector value, float parameter) {
            return value.mul(FloatVector.broadcast(value.species(), parameter));
        }
    },
    POW {
        @Override
        public double applyF64(double value, double parameter) { return Math.pow(value, parameter); }

        @Override
        public float applyF32(float value, float parameter) { return (float) Math.pow(value, parameter); }

        @Override
        public boolean supportsVectorF64() { return false; }

        @Override
        public boolean supportsVectorF32() { return false; }
    };

    public abstract double applyF64(double value, double parameter);

    public abstract float applyF32(float value, float parameter);

    public float applyBF16(float value, float parameter) {
        return applyF32(value, parameter);
    }

    public DoubleVector applyVectorF64(DoubleVector value, double parameter) {
        throw new UnsupportedOperationException(name() + " does not support F64 vector execution");
    }

    public FloatVector applyVectorF32(FloatVector value, float parameter) {
        throw new UnsupportedOperationException(name() + " does not support F32 vector execution");
    }

    public boolean supportsVectorF64() {
        return true;
    }

    public boolean supportsVectorF32() {
        return true;
    }
}
