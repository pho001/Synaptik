package Graph.codegen;

public final class FusedScalarOps {
    private FusedScalarOps() {}

    public static float logF32(float x) {
        return (float) Math.log(x);
    }

    public static float expF32(float x) {
        return (float) Math.exp(x);
    }

    public static float tanhF32(float x) {
        return (float) Math.tanh(x);
    }

}
