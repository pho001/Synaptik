package backend.cpu.fused.ir;

public sealed interface FusedNodeAttributes
        permits NoAttributes, ScalarDoubleAttribute, WhereAttributes {
}
