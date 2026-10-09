package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalRandomDropoutCapabilityTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();

    @Test
    void initialStateIsExactAndRequiresCanonicalNoGradInt64Pair() {
        Operation initial = new Operation(
                GraphRngKind.INITIAL_STATE,
                new GraphRngStateAttrs(0xfedc_ba98_7654_3210L, 0x0123_4567_89ab_cdefL));
        TensorDescriptor state = descriptor(DataType.INT64, Shape.of(2), false);
        assertTrue(supports(initial, List.of(), List.of(state)), "selected occurrence");
        assertFalse(supports(initial, List.of(),
                List.of(descriptor(DataType.INT64, Shape.of(1), false))));
        assertFalse(supports(initial, List.of(),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), false))));
        assertFalse(supports(initial, List.of(),
                List.of(new TensorDescriptor(
                        DataType.INT64, Shape.of(2), Optional.empty(), false))));

    }

    @Test
    void acceleratorDropoutAcceptsOnlyFloat32CanonicalStaticGeometryAndExactRoles() {
        Shape shape = Shape.of(2, 3);
        Operation dropout = new Operation(DropoutKind.DROPOUT, new DropoutAttrs(0.25d));
        TensorDescriptor input = descriptor(DataType.FLOAT32, shape, true);
        TensorDescriptor state = descriptor(DataType.INT64, Shape.of(2), false);
        List<TensorDescriptor> outputs = List.of(
                descriptor(DataType.FLOAT32, shape, true),
                descriptor(DataType.BOOL, shape, false),
                descriptor(DataType.INT64, Shape.of(2), false));
        assertTrue(supports(
                dropout, List.of(input, state), outputs));
        assertFalse(supports(dropout,
                List.of(descriptor(DataType.FLOAT64, shape, true), state),
                List.of(
                        descriptor(DataType.FLOAT64, shape, true),
                        outputs.get(1), outputs.get(2))));
        assertFalse(supports(dropout,
                List.of(input, state),
                List.of(
                        descriptor(DataType.FLOAT32, shape, false),
                        outputs.get(1), outputs.get(2))));
        assertFalse(supports(dropout,
                List.of(input, state),
                List.of(outputs.get(0), descriptor(DataType.INT32, shape, false), outputs.get(2))));
        assertFalse(supports(dropout,
                List.of(new TensorDescriptor(
                        DataType.FLOAT32, shape, Optional.empty(), true), state), outputs));
    }

    @Test
    void scalarRankSixteenAndUnsignedMaximumBoundariesAreLiteral() {
        Operation dropout = new Operation(DropoutKind.DROPOUT, new DropoutAttrs(-0.0d));
        TensorDescriptor state = descriptor(DataType.INT64, Shape.of(2), false);
        for (Shape shape : List.of(
                Shape.scalar(),
                Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7),
                Shape.of(0xffff_ffffL))) {
            TensorDescriptor value = descriptor(DataType.FLOAT32, shape, false);
            assertTrue(supports(dropout,
                    List.of(value, state),
                    List.of(value, descriptor(DataType.BOOL, shape, false), state)),
                    shape.toString());
        }
        Shape extentOver = Shape.of(0x1_0000_0000L);
        TensorDescriptor over = descriptor(DataType.FLOAT32, extentOver, false);
        assertFalse(supports(dropout,
                List.of(over, state),
                List.of(over, descriptor(DataType.BOOL, extentOver, false), state)));
        Shape productOver = Shape.of(65_536, 65_536);
        TensorDescriptor product = descriptor(DataType.FLOAT32, productOver, false);
        assertFalse(supports(dropout,
                List.of(product, state),
                List.of(product, descriptor(DataType.BOOL, productOver, false), state)));
    }

    @Test
    void recurrentBoundaryRemainsFailClosedForEveryVariantDirectionAndBiasArity() {
        TensorDescriptor input =
                descriptor(DataType.FLOAT32, Shape.of(2, 2, 3), true);
        TensorDescriptor lengths = descriptor(DataType.INT64, Shape.of(2), false);
        TensorDescriptor hidden = descriptor(DataType.FLOAT32, Shape.of(2, 4), true);
        TensorDescriptor output = descriptor(DataType.FLOAT32, Shape.of(2, 2, 4), true);
        for (RecurrentScanKind kind : RecurrentScanKind.values()) {
            int gates = switch (kind) {
                case RNN_TANH -> 1;
                case GRU_RESET_AFTER -> 3;
                case LSTM -> 4;
            };
            for (RecurrentDirection direction : RecurrentDirection.values()) {
                for (boolean biased : List.of(false, true)) {
                    var inputs = new java.util.ArrayList<TensorDescriptor>();
                    inputs.add(input);
                    inputs.add(lengths);
                    inputs.add(hidden);
                    if (kind == RecurrentScanKind.LSTM) inputs.add(hidden);
                    inputs.add(descriptor(
                            DataType.FLOAT32, Shape.of((long) gates * 4, 3), true));
                    inputs.add(descriptor(
                            DataType.FLOAT32, Shape.of((long) gates * 4, 4), true));
                    if (biased) {
                        inputs.add(descriptor(
                                DataType.FLOAT32, Shape.of((long) gates * 4), true));
                    }
                    var outputs = new java.util.ArrayList<>(
                            List.of(output, hidden));
                    if (kind == RecurrentScanKind.LSTM) outputs.add(hidden);
                    Operation operation = new Operation(kind, direction);
                    assertFalse(supports(operation, inputs, outputs),
                            kind + " " + direction + " bias=" + biased);

                }
            }
        }
    }

    private boolean supports(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        return provider.supports(new OperationCapabilityQuery(operation, inputs, outputs));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(
                type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }
}
