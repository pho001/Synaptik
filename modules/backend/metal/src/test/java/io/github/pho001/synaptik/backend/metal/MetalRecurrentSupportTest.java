package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exercises source and independent encoded admission without loading a native library. */
class MetalRecurrentSupportTest {
    @Test
    void everyFixedSignatureDirectionAndFloatingGradientRoleLowersWithCanonicalSiblings() {
        for (var kind : RecurrentScanKind.values()) {
            for (var direction : RecurrentDirection.values()) {
                for (boolean biased : List.of(false, true))
                        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                    Case base = occurrence(kind, direction, biased, 3, 2, type);
                    assertQualified(base);
                    for (int role = 0; role < base.inputs.size(); role++) {
                        if (role == 1) continue;
                        var reads = new ArrayList<>(base.inputs);
                        reads.set(role, gradient(reads.get(role), true));
                        var outputs = base.outputs.stream().map(value -> gradient(value, true)).toList();
                        assertQualified(new Case(base.operation, reads, outputs));
                    }
                }
            }
        }
    }

    @Test
    void zeroTimeAndBatchAreAllowedOnlyWithTheExactRecurrentRoleShapes() {
        for (var kind : RecurrentScanKind.values()) {
            for (var direction : RecurrentDirection.values()) {
                for (int[] shape : List.of(new int[] {0, 2}, new int[] {3, 0}, new int[] {0, 0}))
                        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                    assertQualified(occurrence(kind, direction, false, shape[0], shape[1], type));
                }
            }
        }
        Case base = occurrence(RecurrentScanKind.RNN_TANH, RecurrentDirection.FORWARD,
                false, 3, 2, DataType.FLOAT32);
        var reads = new ArrayList<>(base.inputs);
        reads.set(0, descriptor(DataType.FLOAT32, false, 3, 2, 0));
        assertFalse(MetalRecurrentSupport.supports(base.operation, reads, base.outputs));
        reads = new ArrayList<>(base.inputs);
        reads.set(2, descriptor(DataType.FLOAT32, false, 2, 0));
        assertFalse(MetalRecurrentSupport.supports(base.operation, reads, base.outputs));
    }

    @Test
    void float64AndNonfloatingCarriersCannotInheritTheNumericPredicate() {
        for (var kind : RecurrentScanKind.values()) {
            for (var type : List.of(DataType.FLOAT64, DataType.INT32, DataType.INT64, DataType.BOOL)) {
                Case invalid = occurrence(kind, RecurrentDirection.REVERSE, true, 3, 2, type);
                assertFalse(MetalRecurrentSupport.supports(invalid.operation, invalid.inputs, invalid.outputs));
                assertThrows(IllegalArgumentException.class,
                        () -> MetalRecurrentSupport.validate(node(invalid), values(invalid)));
            }
        }
    }

    @Test
    void mixedOriginalFloatingRolesCannotInheritTheCommonCarrier() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            Case base = occurrence(RecurrentScanKind.LSTM, RecurrentDirection.REVERSE, true, 3, 2, type);
            var valid = values(base);
            for (int role = 0; role < valid.size(); role++) {
                if (role == 1) continue;
                for (var other : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                    if (other == type) continue;
                    var mixed = new ArrayList<>(valid);
                    var original = valid.get(role);
                    mixed.set(role, new MetalMpsGraphProgram.ValueDescriptor(
                            other, original.dimensions(), original.requiresGrad()));
                    assertThrows(IllegalArgumentException.class,
                            () -> MetalRecurrentSupport.validate(node(base), mixed));
                }
            }
        }
    }

    @Test
    void independentEncodedChecksRejectChangedShapesTypesPhysicalLayoutsAndGradients() {
        Case base = occurrence(RecurrentScanKind.LSTM, RecurrentDirection.FORWARD,
                true, 3, 2, DataType.FLOAT32);
        var node = node(base);
        List<MetalMpsGraphProgram.ValueDescriptor> valid = values(base);
        assertDoesNotThrow(() -> MetalRecurrentSupport.validate(node, valid));
        int output = base.inputs.size();
        for (int role = 0; role < valid.size(); role++) {
            var wrongCarrier = new ArrayList<>(valid);
            var original = valid.get(role);
            wrongCarrier.set(role, new MetalMpsGraphProgram.ValueDescriptor(
                    role == 1 ? DataType.INT32 : DataType.FLOAT64,
                    original.dimensions(), original.requiresGrad()));
            assertThrows(IllegalArgumentException.class,
                    () -> MetalRecurrentSupport.validate(node, wrongCarrier), "carrier role " + role);
            var wrongGradient = new ArrayList<>(valid);
            wrongGradient.set(role, new MetalMpsGraphProgram.ValueDescriptor(
                    original.dataType(), original.dimensions(), true));
            assertThrows(IllegalArgumentException.class,
                    () -> MetalRecurrentSupport.validate(node, wrongGradient), "gradient role " + role);
            var view = new ArrayList<>(valid);
            Shape shape = Shape.of(original.dimensions());
            LayoutDescriptor canonical = LayoutDescriptor.contiguous(shape);
            view.set(role, new MetalMpsGraphProgram.ValueDescriptor(original.dataType(),
                    original.dimensions(), Optional.of(LayoutDescriptor.of(shape, canonical.strides(), 0, true)),
                    false, false));
            assertThrows(IllegalArgumentException.class,
                    () -> MetalRecurrentSupport.validate(node, view), "view role " + role);
        }
        var wrongOutputShape = new ArrayList<>(valid);
        wrongOutputShape.set(output, new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32, new long[] {3, 2, 4}, false));
        assertThrows(IllegalArgumentException.class,
                () -> MetalRecurrentSupport.validate(node, wrongOutputShape));
        var wrongPackedWeight = new ArrayList<>(valid);
        wrongPackedWeight.set(4, new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32, new long[] {9, 2}, false));
        assertThrows(IllegalArgumentException.class,
                () -> MetalRecurrentSupport.validate(node, wrongPackedWeight));
    }

    @Test
    void loweringPreservesGraphIndicesAndRejectsMissingAliasedOrOutOfRangeSiblings() {
        Case base = occurrence(RecurrentScanKind.GRU_RESET_AFTER, RecurrentDirection.REVERSE,
                true, 3, 2, DataType.FLOAT32);
        int[] reads = {5, 8, 13, 21, 34, 55};
        int[] writes = {89, 144};
        var lowered = MetalRecurrentSupport.lower(base.operation, reads, writes, base.inputs, base.outputs);
        assertArrayEquals(reads, lowered.inputs());
        assertArrayEquals(writes, lowered.outputs());
        assertArrayEquals(new long[] {2}, lowered.attributeWords());
        assertThrows(IllegalArgumentException.class, () -> MetalRecurrentSupport.lower(
                base.operation, reads, new int[] {89, 89}, base.inputs, base.outputs));
        assertThrows(IllegalArgumentException.class, () -> MetalRecurrentSupport.lower(
                base.operation, reads, new int[] {5, 89}, base.inputs, base.outputs));
        assertThrows(IllegalArgumentException.class, () -> MetalRecurrentSupport.lower(
                base.operation, reads, new int[] {89}, base.inputs, base.outputs));
        var valid = values(base);
        var duplicateOutput = MetalMpsGraphProgram.Node.generic(lowered.kind(),
                sequence(0, base.inputs.size()), new int[] {6, 6},
                MetalMpsGraphProgram.AttributeKind.RECURRENT_DIRECTION, new long[] {2});
        assertThrows(IllegalArgumentException.class,
                () -> MetalRecurrentSupport.validate(duplicateOutput, valid));
        var outOfRange = MetalMpsGraphProgram.Node.generic(lowered.kind(),
                sequence(0, base.inputs.size()), new int[] {6, 99},
                MetalMpsGraphProgram.AttributeKind.RECURRENT_DIRECTION, new long[] {2});
        assertThrows(IllegalArgumentException.class,
                () -> MetalRecurrentSupport.validate(outOfRange, valid));
        assertThrows(IllegalArgumentException.class, () -> MetalRecurrentSupport.validate(null, valid));
    }

    private static void assertQualified(Case occurrence) {
        assertTrue(MetalRecurrentSupport.handles(occurrence.operation));
        assertTrue(MetalRecurrentSupport.supports(occurrence.operation, occurrence.inputs, occurrence.outputs));
        var lowered = MetalRecurrentSupport.lower(occurrence.operation,
                sequence(0, occurrence.inputs.size()),
                sequence(occurrence.inputs.size(), occurrence.outputs.size()),
                occurrence.inputs, occurrence.outputs);
        assertTrue(MetalRecurrentSupport.handles(lowered));
        assertDoesNotThrow(() -> MetalRecurrentSupport.validate(lowered, values(occurrence)));
    }

    private static Case occurrence(RecurrentScanKind kind, RecurrentDirection direction,
            boolean biased, int time, int batch, DataType type) {
        int gates = kind == RecurrentScanKind.RNN_TANH ? 1
                : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 4;
        var inputs = new ArrayList<TensorDescriptor>();
        inputs.add(descriptor(type, false, time, batch, 2));
        inputs.add(descriptor(DataType.INT64, false, batch));
        inputs.add(descriptor(type, false, batch, 3));
        if (gates == 4) inputs.add(descriptor(type, false, batch, 3));
        inputs.add(descriptor(type, false, gates * 3, 2));
        inputs.add(descriptor(type, false, gates * 3, 3));
        if (biased) inputs.add(descriptor(type, false, gates * 3));
        var outputs = new ArrayList<TensorDescriptor>();
        outputs.add(descriptor(type, false, time, batch, 3));
        outputs.add(descriptor(type, false, batch, 3));
        if (gates == 4) outputs.add(descriptor(type, false, batch, 3));
        return new Case(new Operation(kind, direction), inputs, outputs);
    }

    private static MetalMpsGraphProgram.Node node(Case occurrence) {
        var kind = switch ((RecurrentScanKind) occurrence.operation.kind()) {
            case RNN_TANH -> MetalMpsGraphProgram.NodeKind.RNN_TANH;
            case GRU_RESET_AFTER -> MetalMpsGraphProgram.NodeKind.GRU_RESET_AFTER;
            case LSTM -> MetalMpsGraphProgram.NodeKind.LSTM;
        };
        return MetalMpsGraphProgram.Node.generic(kind, sequence(0, occurrence.inputs.size()),
                sequence(occurrence.inputs.size(), occurrence.outputs.size()),
                MetalMpsGraphProgram.AttributeKind.RECURRENT_DIRECTION,
                new long[] {occurrence.operation.attrs() == RecurrentDirection.FORWARD ? 1 : 2});
    }

    private static List<MetalMpsGraphProgram.ValueDescriptor> values(Case occurrence) {
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        for (var descriptor : occurrence.inputs) values.add(MetalMpsGraphProgram.ValueDescriptor.from(descriptor));
        for (var descriptor : occurrence.outputs) values.add(MetalMpsGraphProgram.ValueDescriptor.from(descriptor));
        return values;
    }

    private static TensorDescriptor descriptor(DataType type, boolean grad, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    private static TensorDescriptor gradient(TensorDescriptor descriptor, boolean grad) {
        return new TensorDescriptor(descriptor.dataType(), descriptor.shape(), descriptor.layout(), grad);
    }

    private static int[] sequence(int first, int count) {
        int[] indices = new int[count];
        for (int i = 0; i < count; i++) indices[i] = first + i;
        return indices;
    }

    private record Case(Operation operation, List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {}
}
