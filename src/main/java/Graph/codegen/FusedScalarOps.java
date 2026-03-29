package Graph.codegen;

import Backend.ComputeEngine;
import Utils.FastExp;

public final class FusedScalarOps {
    private FusedScalarOps() {}

    public static float logF32(float x) {
        return (float) Math.log(x);
    }

    public static float expF32(float x) {
        if (ComputeEngine.useFastExpApprox()) {
            return FastExp.fastExpF32(x);
        }
        return (float) Math.exp(x);
    }

    public static double expF64(double x) {
        if (ComputeEngine.useFastExpApprox()) {
            return FastExp.fastExpF64(x);
        }
        return Math.exp(x);
    }

    public static float fastExpF32(float x) {
        return FastExp.fastExpF32(x);
    }

    public static double fastExpF64(double x) {
        return FastExp.fastExpF64(x);
    }

    public static float tanhF32(float x) {
        if (ComputeEngine.useFastTanhApprox()) {
            return FastExp.fastTanhF32(x);
        }
        return (float) Math.tanh(x);
    }

    public static double tanhF64(double x) {
        if (ComputeEngine.useFastTanhApprox()) {
            return FastExp.fastTanhF64(x);
        }
        return Math.tanh(x);
    }

    public static float fastTanhF32(float x) {
        return FastExp.fastTanhF32(x);
    }

    public static double fastTanhF64(double x) {
        return FastExp.fastTanhF64(x);
    }

    public static float powF32(float x, float exponent) {
        return (float) Math.pow(x, exponent);
    }

    public static double powF64(double x, double exponent) {
        return Math.pow(x, exponent);
    }

}
