package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

public final class ScaledDotProductAttentionResultTest {
    @Test
    void exposesExactlyTwoTensorComponentsAndOrdinaryRecordSemantics() {
        var components = ScaledDotProductAttentionResult.class.getRecordComponents();
        Tensor output = tensor(1);
        Tensor weights = tensor(2);
        var result = new ScaledDotProductAttentionResult(output, weights);
        var equal = new ScaledDotProductAttentionResult(output, weights);
        var reversed = new ScaledDotProductAttentionResult(weights, output);

        assertAll(
                () -> assertTrue(Modifier.isPublic(
                        ScaledDotProductAttentionResult.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(
                        ScaledDotProductAttentionResult.class.getModifiers())),
                () -> assertTrue(ScaledDotProductAttentionResult.class.isRecord()),
                () -> assertEquals(List.of("output", "weights"), Arrays.stream(components)
                        .map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(Tensor.class, Tensor.class), Arrays.stream(components)
                        .map(component -> component.getType()).toList()),
                () -> assertEquals(0,
                        ScaledDotProductAttentionResult.class.getDeclaredClasses().length),
                () -> assertSame(output, result.output()),
                () -> assertSame(weights, result.weights()),
                () -> assertEquals(result, equal),
                () -> assertEquals(result.hashCode(), equal.hashCode()),
                () -> assertNotEquals(result, reversed));
    }

    @Test
    void rejectsNullComponentsInDeclarationOrderWithExactMessages() {
        Tensor tensor = tensor(1);

        assertAll(
                () -> assertEquals("output", assertThrows(
                        NullPointerException.class,
                        () -> new ScaledDotProductAttentionResult(null, null)).getMessage()),
                () -> assertEquals("weights", assertThrows(
                        NullPointerException.class,
                        () -> new ScaledDotProductAttentionResult(tensor, null)).getMessage()));
    }

    private static Tensor tensor(long id) {
        return new Tensor(
                new TensorId(id),
                new TensorDescriptor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }
}
