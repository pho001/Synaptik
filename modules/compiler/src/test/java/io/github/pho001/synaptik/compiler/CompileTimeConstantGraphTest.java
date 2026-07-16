package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class CompileTimeConstantGraphTest {
    @Test
    void exposesOnlyThePlannedPackagePrivateRecordShape() {
        assertAll(
                () -> assertTrue(CompileTimeConstantGraph.class.isRecord()),
                () -> assertFalse(Modifier.isPublic(CompileTimeConstantGraph.class.getModifiers())),
                () -> assertEquals(List.of("graph", "constants"), Arrays.stream(
                                CompileTimeConstantGraph.class.getRecordComponents())
                        .map(component -> component.getName()).toList()),
                () -> assertTrue(CompileTimeConstantGraph.Splat.class.isRecord()),
                () -> assertTrue(CompileTimeConstantGraph.Binding.class.isRecord()),
                () -> assertTrue(CompileTimeConstantGraph.Ingress.class.isRecord()),
                () -> assertFalse(Modifier.isPublic(
                        CompileTimeConstantGraph.Splat.class.getModifiers())),
                () -> assertEquals(List.of("value"), Arrays.stream(
                                CompileTimeConstantGraph.Splat.class.getRecordComponents())
                        .map(component -> component.getName()).toList()),
                () -> assertEquals(List.of("tensor", "splat"), Arrays.stream(
                                CompileTimeConstantGraph.Binding.class.getRecordComponents())
                        .map(component -> component.getName()).toList()),
                () -> assertEquals(List.of("bindings"), Arrays.stream(
                                CompileTimeConstantGraph.Ingress.class.getRecordComponents())
                        .map(component -> component.getName()).toList()));
    }

    @Test
    void splatPreservesAllExactScalarDomainsAndBitEquality() {
        List<ScalarValue> values = List.of(
                ScalarValue.float64(Double.longBitsToDouble(0x7ff8_0000_0000_0042L)),
                ScalarValue.float32(Float.intBitsToFloat(0x7fc0_0042)),
                ScalarValue.bfloat16Bits((short) 0xFFC1),
                ScalarValue.int32(Integer.MIN_VALUE),
                ScalarValue.int64(Long.MAX_VALUE),
                ScalarValue.bool(true));

        for (ScalarValue value : values) {
            var splat = new CompileTimeConstantGraph.Splat(value);
            assertAll(
                    () -> assertSame(value, splat.value()),
                    () -> assertEquals(splat,
                            new CompileTimeConstantGraph.Splat(value)),
                    () -> assertEquals(splat.hashCode(),
                            new CompileTimeConstantGraph.Splat(value).hashCode()));
        }
        assertAll(
                () -> assertNotEquals(
                        new CompileTimeConstantGraph.Splat(ScalarValue.float32(+0.0f)),
                        new CompileTimeConstantGraph.Splat(ScalarValue.float32(-0.0f))),
                () -> assertNotEquals(
                        new CompileTimeConstantGraph.Splat(ScalarValue.float64(
                                Double.longBitsToDouble(0x7ff8_0000_0000_0001L))),
                        new CompileTimeConstantGraph.Splat(ScalarValue.float64(
                                Double.longBitsToDouble(0x7ff8_0000_0000_0002L)))),
                () -> assertEquals("value", assertThrows(NullPointerException.class,
                        () -> new CompileTimeConstantGraph.Splat(null)).getMessage()));
    }

    @Test
    void validatesFactsSnapshotsContainersAndDerivesBindableOrder() {
        TensorDescriptor int32 = descriptor(DataType.INT32, Shape.of(0, 3), Optional.empty(), false);
        TensorDescriptor int64 = descriptor(
                DataType.INT64,
                Shape.ofDimensions(new DynamicDimension("N")),
                Optional.empty(),
                false);
        TensorDescriptor bool = descriptor(
                DataType.BOOL,
                Shape.scalar(),
                Optional.of(LayoutDescriptor.contiguous(Shape.scalar())),
                false);
        CompiledGraphModel graph = inputsGraph(List.of(int32, int64, bool), 1);
        var int64Splat = new CompileTimeConstantGraph.Splat(ScalarValue.int64(-7));
        var sourceFacts = new HashMap<ValueId, CompileTimeConstantGraph.Splat>();
        sourceFacts.put(new ValueId(1), int64Splat);

        CompileTimeConstantGraph constantGraph =
                new CompileTimeConstantGraph(graph, sourceFacts);
        sourceFacts.clear();

        assertAll(
                () -> assertSame(graph, constantGraph.graph()),
                () -> assertEquals(Map.of(new ValueId(1), int64Splat), constantGraph.constants()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(2)),
                        constantGraph.bindableInputs()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> constantGraph.constants().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> constantGraph.bindableInputs().clear()));
    }

    @Test
    void rejectsNullUnknownMismatchedAndGradientFactsDeterministically() {
        TensorDescriptor int32 = descriptor(DataType.INT32, Shape.of(1), Optional.empty(), false);
        TensorDescriptor gradient = descriptor(DataType.FLOAT32, Shape.of(1), Optional.empty(), true);
        CompiledGraphModel graph = inputsGraph(List.of(int32, gradient), 0);
        var int32Splat = new CompileTimeConstantGraph.Splat(ScalarValue.int32(1));
        var floatSplat = new CompileTimeConstantGraph.Splat(ScalarValue.float32(1));

        Map<ValueId, CompileTimeConstantGraph.Splat> nullKey = new HashMap<>();
        nullKey.put(null, int32Splat);
        Map<ValueId, CompileTimeConstantGraph.Splat> nullValue = new HashMap<>();
        nullValue.put(new ValueId(0), null);

        assertAll(
                () -> assertEquals("graph", assertThrows(NullPointerException.class,
                        () -> new CompileTimeConstantGraph(null, Map.of())).getMessage()),
                () -> assertEquals("constants", assertThrows(NullPointerException.class,
                        () -> new CompileTimeConstantGraph(graph, null)).getMessage()),
                () -> assertEquals("constants contains null key",
                        assertThrows(NullPointerException.class,
                                () -> new CompileTimeConstantGraph(graph, nullKey)).getMessage()),
                () -> assertEquals("constants[ValueId[value=0]]",
                        assertThrows(NullPointerException.class,
                                () -> new CompileTimeConstantGraph(graph, nullValue)).getMessage()),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new CompileTimeConstantGraph(
                                graph, Map.of(new ValueId(9), int32Splat)))
                        .getMessage().contains("not a graph input")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new CompileTimeConstantGraph(
                                graph, Map.of(new ValueId(0), floatSplat)))
                        .getMessage().contains("does not match")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new CompileTimeConstantGraph(
                                graph, Map.of(new ValueId(1), floatSplat)))
                        .getMessage().contains("requires gradients")));
    }

    @Test
    void ingressSnapshotsUsesTensorIdentityAndRejectsProducedOrGradientLeaves() {
        Tensor first = tensor(DataType.INT32, false);
        Tensor equalDescriptorButDistinct = tensor(DataType.INT32, false);
        var firstBinding = new CompileTimeConstantGraph.Binding(
                first, new CompileTimeConstantGraph.Splat(ScalarValue.int32(4)));
        var secondBinding = new CompileTimeConstantGraph.Binding(
                equalDescriptorButDistinct,
                new CompileTimeConstantGraph.Splat(ScalarValue.int32(4)));
        var mutable = new ArrayList<>(List.of(firstBinding, secondBinding));
        var ingress = new CompileTimeConstantGraph.Ingress(mutable);
        mutable.clear();

        assertAll(
                () -> assertEquals(List.of(firstBinding, secondBinding), ingress.bindings()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> ingress.bindings().clear()),
                () -> assertTrue(CompileTimeConstantGraph.Ingress.empty().bindings().isEmpty()),
                () -> assertEquals("bindings[1] duplicates bindings[0] tensor",
                        assertThrows(IllegalArgumentException.class,
                                () -> new CompileTimeConstantGraph.Ingress(
                                        List.of(firstBinding, firstBinding))).getMessage()),
                () -> assertEquals("tensor", assertThrows(NullPointerException.class,
                        () -> new CompileTimeConstantGraph.Binding(null,
                                firstBinding.splat())).getMessage()),
                () -> assertEquals("splat", assertThrows(NullPointerException.class,
                        () -> new CompileTimeConstantGraph.Binding(first, null)).getMessage()),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new CompileTimeConstantGraph.Binding(
                                first.add(equalDescriptorButDistinct), firstBinding.splat()))
                        .getMessage().contains("provenance-free")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new CompileTimeConstantGraph.Binding(
                                tensor(DataType.FLOAT32, true),
                                new CompileTimeConstantGraph.Splat(ScalarValue.float32(1))))
                        .getMessage().contains("must not require gradients")));
    }

    @Test
    void replacesGraphByInputPositionAndRejectsBoundaryContradictions() {
        TensorDescriptor first = descriptor(DataType.INT32, Shape.of(2), Optional.empty(), false);
        TensorDescriptor second = descriptor(DataType.INT64, Shape.of(2), Optional.empty(), false);
        CompiledGraphModel original = inputsGraph(List.of(first, second), 0);
        CompileTimeConstantGraph sidecar = new CompileTimeConstantGraph(
                original,
                Map.of(new ValueId(1),
                        new CompileTimeConstantGraph.Splat(ScalarValue.int64(8))));
        CompiledGraphModel replacement = inputsGraphWithIds(
                List.of(first, second), List.of(20L, 10L), 0);

        CompileTimeConstantGraph remapped =
                sidecar.replaceGraphPreservingInputRoles(replacement);

        assertAll(
                () -> assertSame(sidecar, sidecar.replaceGraphPreservingInputRoles(original)),
                () -> assertSame(replacement, remapped.graph()),
                () -> assertEquals(List.of(new ValueId(20)), remapped.bindableInputs()),
                () -> assertEquals(ScalarValue.int64(8),
                        remapped.constants().get(new ValueId(10)).value()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sidecar.replaceGraphPreservingInputRoles(
                                inputsGraph(List.of(first), 0))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> sidecar.replaceGraphPreservingInputRoles(
                                inputsGraph(List.of(second, first), 0))));
    }

    private static Tensor tensor(DataType type, boolean requiresGrad) {
        return TensorFactory.create(descriptor(
                type, Shape.of(2), Optional.empty(), requiresGrad));
    }

    private static TensorDescriptor descriptor(
            DataType type,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad) {
        return new TensorDescriptor(type, shape, layout, requiresGrad);
    }

    private static CompiledGraphModel inputsGraph(
            List<TensorDescriptor> descriptors, int outputIndex) {
        List<Long> ids = new ArrayList<>();
        for (int index = 0; index < descriptors.size(); index++) {
            ids.add((long) index);
        }
        return inputsGraphWithIds(descriptors, ids, outputIndex);
    }

    private static CompiledGraphModel inputsGraphWithIds(
            List<TensorDescriptor> descriptors, List<Long> ids, int outputIndex) {
        List<GraphValue> values = new ArrayList<>();
        List<ValueId> inputs = new ArrayList<>();
        for (int index = 0; index < descriptors.size(); index++) {
            ValueId id = new ValueId(ids.get(index));
            values.add(new GraphValue(id, descriptors.get(index)));
            inputs.add(id);
        }
        return new CompiledGraphModel(
                values, List.of(), inputs, List.of(inputs.get(outputIndex)), Map.of());
    }
}
