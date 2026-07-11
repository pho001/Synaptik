package io.github.pho001.synaptik.model.operation.random;

import static io.github.pho001.synaptik.model.operation.OperationSignatureTest.assertSignatureEnumShape;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

public final class DropoutSemanticsTest {
    @Test
    void acceptsTheCompleteProbabilityDomainAndRetainsSignedZeroBits() {
        for (double probability : new double[] {
                0.0d, -0.0d, Double.MIN_VALUE, 0.1d, Math.nextDown(1.0d)
        }) {
            DropoutAttrs attrs = new DropoutAttrs(probability);
            assertEquals(
                    Double.doubleToRawLongBits(probability),
                    Double.doubleToRawLongBits(attrs.probability()));
            assertEquals(attrs, new DropoutAttrs(probability));
        }

        assertNotEquals(new DropoutAttrs(0.0d), new DropoutAttrs(-0.0d));
    }

    @Test
    void rejectsEveryProbabilityOutsideTheDomainWithTheExactMessage() {
        for (double probability : new double[] {
                Double.NaN,
                Double.NEGATIVE_INFINITY,
                Double.POSITIVE_INFINITY,
                -Double.MIN_VALUE,
                -1.0d,
                1.0d,
                Math.nextUp(1.0d),
                Double.MAX_VALUE
        }) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> new DropoutAttrs(probability));
            assertEquals(
                    "probability must be finite and in [0.0, 1.0): " + probability,
                    failure.getMessage());
        }
    }

    @Test
    void exposesTheExactImmutableAttributeAndKindSurfaces() {
        var components = DropoutAttrs.class.getRecordComponents();
        OperationSignature expected =
                OperationSignature.fixed(DropoutAttrs.class, 2, 3);
        Operation operation = new Operation(DropoutKind.DROPOUT, new DropoutAttrs(0.25d));

        assertAll(
                () -> assertTrue(DropoutAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(DropoutAttrs.class.getInterfaces())),
                () -> assertEquals(List.of("probability"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(double.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(0, DropoutAttrs.class.getDeclaredClasses().length),
                () -> assertEquals(List.of(expected), DropoutKind.DROPOUT.signatures()),
                () -> assertEquals(expected, operation.signature()),
                () -> assertSame(DropoutKind.DROPOUT, operation.kind()),
                () -> assertTrue(operation.attrs() instanceof DropoutAttrs),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> DropoutKind.DROPOUT.signatures().clear()),
                () -> assertEquals(List.of(OperationKind.class),
                        List.of(DropoutKind.class.getInterfaces())));
        assertSignatureEnumShape(DropoutKind.class);
    }
}
