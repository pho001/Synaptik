package backend.cpu1.fused.ir;

public record Cpu1FusedScalarParameter(boolean present, float f32, double f64) {
    public static final Cpu1FusedScalarParameter NONE =
            new Cpu1FusedScalarParameter(false, 0.0f, 0.0d);

    public static Cpu1FusedScalarParameter of(float f32, double f64) {
        return new Cpu1FusedScalarParameter(true, f32, f64);
    }
}
