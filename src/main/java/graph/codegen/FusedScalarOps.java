package graph.codegen;

import utils.FastTranscendentals;

public final class FusedScalarOps {
    private FusedScalarOps() {}

    public static float logF32(float x) {
        return (float) Math.log(x);
    }

    public static float expF32(float x, boolean useFastExpApprox) {
        if (useFastExpApprox) {
            return FastTranscendentals.fastExpF32(x);
        }
        return (float) Math.exp(x);
    }

    public static double expF64(double x, boolean useFastExpApprox) {
        if (useFastExpApprox) {
            return FastTranscendentals.fastExpF64(x);
        }
        return Math.exp(x);
    }

    public static float fastExpF32(float x) {
        return FastTranscendentals.fastExpF32(x);
    }

    public static double fastExpF64(double x) {
        return FastTranscendentals.fastExpF64(x);
    }

    public static float tanhF32(float x, boolean useFastTanhApprox) {
        if (useFastTanhApprox) {
            return FastTranscendentals.fastTanhF32(x);
        }
        return (float) Math.tanh(x);
    }

    public static double tanhF64(double x, boolean useFastTanhApprox) {
        if (useFastTanhApprox) {
            return FastTranscendentals.fastTanhF64(x);
        }
        return Math.tanh(x);
    }

    public static float fastTanhF32(float x) {
        return FastTranscendentals.fastTanhF32(x);
    }

    public static double fastTanhF64(double x) {
        return FastTranscendentals.fastTanhF64(x);
    }

    public static float powF32(float x, float exponent) {
        if (exponent == 0.0f) {
            return 1.0f;
        }
        if (exponent == 1.0f) {
            return x;
        }
        if (exponent == 2.0f) {
            return x * x;
        }
        if (exponent == 0.5f) {
            return (float) Math.sqrt(x);
        }
        if (exponent == -1.0f) {
            return 1.0f / x;
        }
        return (float) Math.pow(x, exponent);
    }

    public static double powF64(double x, double exponent) {
        if (exponent == 0.0d) {
            return 1.0d;
        }
        if (exponent == 1.0d) {
            return x;
        }
        if (exponent == 2.0d) {
            return x * x;
        }
        if (exponent == 0.5d) {
            return Math.sqrt(x);
        }
        if (exponent == -1.0d) {
            return 1.0d / x;
        }
        return Math.pow(x, exponent);
    }

}
