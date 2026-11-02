package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalAttentionSupportTest {
    @Test
    void emptyPrimaryDoesNotRequireRepresentableUnusedScoreWorkspace() {
        long rows = 10_000_000_000L;
        var inputs = List.of(descriptor(DataType.FLOAT32, false, rows, 1),
                descriptor(DataType.FLOAT32, false, rows, 1),
                descriptor(DataType.FLOAT32, false, rows, 0));
        var outputs = List.of(descriptor(DataType.FLOAT32, false, rows, 0));
        assertTrue(MetalAttentionSupport.supports(operation(Optional.empty()), inputs, outputs));
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        inputs.forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        outputs.forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        assertDoesNotThrow(() -> MetalAttentionSupport.validate(node(false), values));
    }

    @Test
    void positiveOverlappingInputCannotPassSourceAdmissionOrIndependentRawValidation() {
        Shape shape = Shape.of(2, 2);
        var overlap = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 1}, 0, true)), false);
        var inputs = List.of(overlap, descriptor(DataType.FLOAT32, false, 3, 2),
                descriptor(DataType.FLOAT32, false, 3, 4));
        var outputs = List.of(descriptor(DataType.FLOAT32, false, 2, 4));
        assertFalse(MetalAttentionSupport.supports(operation(Optional.empty()), inputs, outputs));
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        inputs.forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        outputs.forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        assertThrows(IllegalArgumentException.class, () -> MetalAttentionSupport.validate(node(false), values));
    }

    @Test
    void outputAndCanonicalSiblingKeepDifferentOriginalRoleGradientMetadata() {
        Operation operation = operation(Optional.empty());
        var inputs = List.of(descriptor(DataType.FLOAT32, false, 1, 2),
                descriptor(DataType.FLOAT32, false, 3, 2), descriptor(DataType.FLOAT32, true, 3, 4));
        var outputs = List.of(descriptor(DataType.FLOAT32, true, 1, 4),
                descriptor(DataType.FLOAT32, false, 1, 3));
        assertTrue(MetalAttentionSupport.supports(operation, inputs, outputs));
        var node = MetalAttentionSupport.lower(operation, new int[] {0, 1, 2}, new int[] {3, 4}, inputs, outputs);
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        inputs.forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        outputs.forEach(d -> values.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        assertDoesNotThrow(() -> MetalAttentionSupport.validate(node, values));
        values.set(4, new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {1, 3}, true));
        assertThrows(IllegalArgumentException.class, () -> MetalAttentionSupport.validate(node, values));
    }

    @Test
    void originalLowCarrierAndNonF32ScalarAreNotAdmittedByProjectedWorkingType() {
        var inputs = List.of(descriptor(DataType.BFLOAT16, false, 1, 2),
                descriptor(DataType.FLOAT32, false, 3, 2), descriptor(DataType.FLOAT32, false, 3, 4));
        var outputs = List.of(descriptor(DataType.FLOAT32, false, 1, 4));
        assertFalse(MetalAttentionSupport.supports(operation(Optional.empty()), inputs, outputs));
        var f32 = List.of(descriptor(DataType.FLOAT32, false, 1, 2), inputs.get(1), inputs.get(2));
        assertFalse(MetalAttentionSupport.supports(operation(Optional.of(ScalarValue.float64(0.5))), f32, outputs));
        var values = List.of(new MetalMpsGraphProgram.ValueDescriptor(DataType.BFLOAT16, new long[] {1, 2}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {3, 2}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {3, 4}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {1, 4}, false));
        assertThrows(IllegalArgumentException.class, () -> MetalAttentionSupport.validate(node(false), values));
    }

    @Test
    void independentRawValidatorRejectsMaskBroadcastAndNoncanonicalSiblingClaims() {
        var values = new ArrayList<>(List.of(
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 2}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {3, 2}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {3, 4}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.BOOL, new long[] {2, 2}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 4}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 3}, false)));
        assertThrows(IllegalArgumentException.class, () -> MetalAttentionSupport.validate(node(true), values));
        values.set(3, new MetalMpsGraphProgram.ValueDescriptor(DataType.BOOL, new long[] {3}, false));
        assertDoesNotThrow(() -> MetalAttentionSupport.validate(node(true), values));
        values.set(5, new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {2, 3},
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {1, 2}, 0, true)), false, false));
        assertThrows(IllegalArgumentException.class, () -> MetalAttentionSupport.validate(node(true), values));
    }

    @Test
    void emptyKeyAndEmptyBatchRemainValidWithoutWeakeningEmbeddingPositivity() {
        var emptyKeys = List.of(descriptor(DataType.FLOAT32, false, 2, 3),
                descriptor(DataType.FLOAT32, false, 0, 3), descriptor(DataType.FLOAT32, false, 0, 4));
        assertTrue(MetalAttentionSupport.supports(operation(Optional.empty()), emptyKeys,
                List.of(descriptor(DataType.FLOAT32, false, 2, 4), descriptor(DataType.FLOAT32, false, 2, 0))));
        var emptyBatch = List.of(descriptor(DataType.FLOAT32, false, 0, 2, 3),
                descriptor(DataType.FLOAT32, false, 1, 3, 3), descriptor(DataType.FLOAT32, false, 1, 3, 4));
        assertTrue(MetalAttentionSupport.supports(operation(Optional.empty()), emptyBatch,
                List.of(descriptor(DataType.FLOAT32, false, 0, 2, 4))));
        var noEmbedding = List.of(descriptor(DataType.FLOAT32, false, 2, 0),
                descriptor(DataType.FLOAT32, false, 3, 0), descriptor(DataType.FLOAT32, false, 3, 4));
        assertFalse(MetalAttentionSupport.supports(operation(Optional.empty()), noEmbedding,
                List.of(descriptor(DataType.FLOAT32, false, 2, 4))));
    }

    private static Operation operation(Optional<ScalarValue> scale) {
        return new Operation(ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                new ScaledDotProductAttentionAttrs(scale, false));
    }

    private static TensorDescriptor descriptor(DataType type, boolean grad, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    private static MetalMpsGraphProgram.Node node(boolean masked) {
        return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SCALED_DOT_PRODUCT_ATTENTION,
                masked ? new int[] {0, 1, 2, 3} : new int[] {0, 1, 2},
                masked ? new int[] {4, 5} : new int[] {3},
                MetalMpsGraphProgram.AttributeKind.ATTENTION, new long[] {0, 0});
    }
}
