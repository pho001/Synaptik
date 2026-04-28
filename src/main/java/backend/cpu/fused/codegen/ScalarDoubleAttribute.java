package backend.cpu.fused.codegen;

/**
 * Scalar double payload attached to fused scalar operations.
 */
public record ScalarDoubleAttribute(double value) implements FusedNodeAttributes {
}
