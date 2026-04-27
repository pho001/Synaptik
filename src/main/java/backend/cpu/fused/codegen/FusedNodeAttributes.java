package backend.cpu.fused.codegen;

public sealed interface FusedNodeAttributes
        permits NoAttributes, ScalarDoubleAttribute, WhereAttributes {
}
