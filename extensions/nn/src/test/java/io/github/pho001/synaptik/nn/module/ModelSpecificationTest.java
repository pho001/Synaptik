package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.*;
import io.github.pho001.synaptik.nn.module.ModelSpecification.*;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class ModelSpecificationTest {
    private static final List<DataType> FLOATS = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16, DataType.FLOAT16);

    @Test
    void exactClosedPublicSurfaceAndTypedLayerSnapshots() throws Exception {
        assertTrue(Modifier.isFinal(ModelSpecification.class.getModifiers()));
        assertTrue(Step.class.isSealed());
        assertEquals(Set.of(Sequence.class, LinearStep.class, LayerNormStep.class, EmbeddingStep.class,
                Conv1dStep.class, Conv2dStep.class, Conv3dStep.class, ActivationStep.class),
                Set.of(Step.class.getPermittedSubclasses()));
        assertEquals(Set.of("describe", "root", "aliases", "stateDescriptors", "outputDescriptor"),
                publicMethods(ModelSpecification.class));
        assertEquals(Set.of("reconstruct", "describe", "forward"), publicMethods(DeclarativeModel.class));
        assertEquals(0, Arrays.stream(DeclarativeModel.class.getDeclaredConstructors())
                .filter(c -> Modifier.isPublic(c.getModifiers())).count());
        assertEquals(Model.class, DeclarativeModel.class.getSuperclass());
        assertEquals(Set.of("specification", "forward"), publicMethods(Activation.class));
        assertEquals(List.of(ActivationKind.RELU, ActivationKind.TANH, ActivationKind.SIGMOID),
                List.of(ActivationKind.values()));
        Map<Class<?>, Class<?>> layers = Map.of(Linear.class, LinearStep.class,
                LayerNorm.class, LayerNormStep.class, Embedding.class, EmbeddingStep.class,
                Conv1d.class, Conv1dStep.class, Conv2d.class, Conv2dStep.class, Conv3d.class, Conv3dStep.class,
                Activation.class, ActivationStep.class);
        for (var entry : layers.entrySet())
            assertEquals(entry.getValue(), entry.getKey().getMethod("specification").getReturnType());
        for (Class<?> nested : ModelSpecification.class.getDeclaredClasses()) {
            if (!Modifier.isPublic(nested.getModifiers())) continue;
            for (var field : nested.getDeclaredFields())
                assertFalse(Tensor.class.isAssignableFrom(field.getType()));
        }
        assertEquals(Linear.class, Linear.class.getMethod("fromState", Tensor.class,
                Optional.class, boolean.class).getReturnType());
    }

    @Test
    void listsAreDefensiveAndNumericStateOrderIsNotLexical() {
        List<Step> mutable = new ArrayList<>();
        for (int i = 0; i < 12; i++) mutable.add(new LinearStep(2, 2, true, DataType.FLOAT32, false));
        Sequence sequence = new Sequence(mutable);
        mutable.clear();
        List<AliasGroup> groups = new ArrayList<>();
        List<String> paths = new ArrayList<>(List.of("0.weight", "10.weight"));
        groups.add(new AliasGroup(paths));
        ModelSpecification spec = new ModelSpecification(sequence, groups);
        paths.clear();
        groups.clear();
        assertEquals(12, spec.root().steps().size());
        assertEquals(List.of("0.weight", "10.weight"), spec.aliases().getFirst().paths());
        assertEquals("2.weight", spec.stateDescriptors().get(4).path());
        assertEquals("10.weight", spec.stateDescriptors().get(20).path());
        assertThrows(UnsupportedOperationException.class, () -> spec.root().steps().clear());
        assertThrows(UnsupportedOperationException.class, () -> spec.aliases().clear());
        assertThrows(UnsupportedOperationException.class, () -> spec.stateDescriptors().clear());
        assertThrows(UnsupportedOperationException.class, () -> spec.aliases().getFirst().paths().clear());
    }

    @Test
    void nestedNormalizationPreservesTypesEpsilonBitsPathsAndVariableLeadingAxes() {
        for (DataType type : FLOATS) {
            Shape normShape = Shape.of(3);
            ScalarValue epsilon = epsilon(type);
            Sequential source = new Sequential(List.of(new Linear(leaf(type, Shape.of(3, 2))),
                    new Sequential(List.of(new LayerNorm(leaf(type, normShape), leaf(type, normShape), epsilon)))));
            source.train();
            ModelSpecification spec = ModelSpecification.describe(source);
            assertEquals(ForwardMode.TRAINING, source.mode());
            assertEquals(List.of("0.weight", "1.0.scale", "1.0.bias"),
                    spec.stateDescriptors().stream().map(StateDescriptor::path).toList());
            LayerNormStep step = (LayerNormStep) ((Sequence) spec.root().steps().get(1)).steps().getFirst();
            assertSame(epsilon, step.epsilon());
            assertEquals(epsilon, step.epsilon());
            Shape inputShape = Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(5),
                    new StaticDimension(2));
            var input = descriptor(type, inputShape, false);
            var ordinary = source.forward(TensorFactory.create(input));
            assertEquals(ordinary.descriptor(), spec.outputDescriptor(input));
            assertEquals(Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(5),
                    new StaticDimension(3)), ordinary.descriptor().shape());
            assertEquals(spec.root(), DeclarativeModel.reconstruct(spec).describe().root());
        }
    }

    @Test
    void embeddingThenProjectionSupportsEveryFloatingAndBothIndexTypesAtAnyRank() {
        for (DataType type : FLOATS) {
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                var source = new Sequential(List.of(new Embedding(leaf(type, Shape.of(7, 3))),
                        new Linear(leaf(type, Shape.of(2, 3)), leaf(type, Shape.of(2)))));
                var spec = ModelSpecification.describe(source);
                for (Shape indices : List.of(Shape.scalar(), Shape.of(2), Shape.of(2, 4, 1))) {
                    var input = descriptor(indexType, indices, false);
                    assertEquals(source.forward(TensorFactory.create(input)).descriptor(),
                            spec.outputDescriptor(input));
                    assertEquals(type, spec.outputDescriptor(input).dataType());
                }
                assertThrows(IllegalArgumentException.class,
                        () -> spec.outputDescriptor(descriptor(DataType.FLOAT32, Shape.of(2), false)));
            }
        }
    }

    @Test
    void suppliedLinearPromotesAutomaticLinearAndConvolutionsRequireExactType() {
        var supplied = specification(new LinearStep(3, 2, false, DataType.FLOAT32, false));
        assertEquals(DataType.FLOAT64,
                supplied.outputDescriptor(descriptor(DataType.FLOAT64, Shape.of(4, 3), false)).dataType());
        var automatic = specification(new LinearStep(3, 2, false, DataType.FLOAT32, true));
        assertThrows(IllegalArgumentException.class,
                () -> automatic.outputDescriptor(descriptor(DataType.FLOAT64, Shape.of(4, 3), false)));
        assertThrows(IllegalArgumentException.class,
                () -> automatic.outputDescriptor(descriptor(DataType.FLOAT32,
                        Shape.ofDimensions(new DynamicDimension("F")), false)));
        assertThrows(IllegalArgumentException.class,
                () -> automatic.outputDescriptor(descriptor(DataType.FLOAT32, Shape.of(4, 4), false)));
        assertThrows(IllegalArgumentException.class,
                () -> automatic.outputDescriptor(descriptor(DataType.FLOAT32, Shape.scalar(), false)));
        assertThrows(IllegalArgumentException.class,
                () -> supplied.outputDescriptor(descriptor(DataType.FLOAT32, Shape.of(4, 4), false)));
        var conv = specification(new Conv1dStep(2, 2, 1, false, DataType.FLOAT32, Conv1dAttrs.defaults()));
        assertThrows(IllegalArgumentException.class,
                () -> conv.outputDescriptor(descriptor(DataType.FLOAT64, Shape.of(1, 2, 5), false)));
    }

    @Test
    void layerNormPromotionRetainsEpsilonCompatibilityWithoutConversion() {
        var single = specification(new LayerNormStep(Shape.of(3), DataType.FLOAT32, epsilon(DataType.FLOAT32)));
        assertEquals(DataType.FLOAT32,
                single.outputDescriptor(descriptor(DataType.FLOAT16, Shape.of(2, 3), false)).dataType());
        assertThrows(IllegalArgumentException.class,
                () -> single.outputDescriptor(descriptor(DataType.FLOAT64, Shape.of(2, 3), false)));
        assertThrows(IllegalArgumentException.class,
                () -> single.outputDescriptor(descriptor(DataType.FLOAT32, Shape.of(2, 4), false)));
        var half = specification(new LayerNormStep(Shape.of(3), DataType.FLOAT16, epsilon(DataType.FLOAT16)));
        assertEquals(DataType.FLOAT32,
                half.outputDescriptor(descriptor(DataType.FLOAT32, Shape.of(2, 3), false)).dataType());
    }

    @Test
    void activationAndEmptyTreesPreserveOrdinaryMetadataAndIdentity() {
        for (DataType type : FLOATS) {
            for (ActivationKind kind : ActivationKind.values()) {
                var spec = specification(new ActivationStep(kind));
                var input = descriptor(type, Shape.of(2, 3), true);
                assertEquals(new Activation(kind).forward(TensorFactory.create(input)).descriptor(),
                        spec.outputDescriptor(input));
                assertEquals(kind, ((ActivationStep) spec.root().steps().getFirst()).kind());
            }
        }
        var input = descriptor(DataType.INT32, Shape.of(2), false);
        var empty = new ModelSpecification(new Sequence(List.of()), List.of());
        assertSame(input, empty.outputDescriptor(input));
        Tensor leaf = TensorFactory.create(input);
        assertSame(leaf, DeclarativeModel.reconstruct(empty).forward(leaf));
        assertThrows(IllegalArgumentException.class,
                () -> specification(new ActivationStep(ActivationKind.RELU)).outputDescriptor(input));
    }

    @Test
    void rejectsMalformedSchemasAndNullsBeforeReturningSpecifications() {
        assertThrows(NullPointerException.class, () -> new ModelSpecification(null, List.of()));
        assertThrows(NullPointerException.class, () -> new Sequence(null));
        assertThrows(NullPointerException.class, () -> new Sequence(Arrays.asList((Step) null)));
        assertThrows(NullPointerException.class, () -> new ActivationStep(null));
        assertThrows(NullPointerException.class, () -> new Activation(null));
        assertThrows(NullPointerException.class, () -> new LinearStep(1, 1, false, null, false));
        assertThrows(NullPointerException.class, () -> new AliasGroup(Arrays.asList("0.weight", null)));
        assertThrows(IllegalArgumentException.class, () -> new LinearStep(0, 1, false, DataType.FLOAT32, false));
        assertThrows(IllegalArgumentException.class, () -> new LinearStep(1, 1, false, DataType.INT32, false));
        assertThrows(IllegalArgumentException.class, () -> new EmbeddingStep(-1, 2, DataType.FLOAT32));
        assertThrows(IllegalArgumentException.class, () -> new LayerNormStep(Shape.scalar(), DataType.FLOAT32,
                epsilon(DataType.FLOAT32)));
        assertThrows(IllegalArgumentException.class, () -> new LayerNormStep(
                Shape.ofDimensions(new DynamicDimension("N")), DataType.FLOAT32, epsilon(DataType.FLOAT32)));
        assertThrows(IllegalArgumentException.class, () -> new LayerNormStep(Shape.of(3), DataType.FLOAT32,
                ScalarValue.float32(-0.0f)));
        assertThrows(IllegalArgumentException.class, () -> new LayerNormStep(Shape.of(3), DataType.FLOAT32,
                ScalarValue.float64(0.01)));
        assertThrows(IllegalArgumentException.class, () -> new Conv1dStep(3, 4, 1, false, DataType.FLOAT32,
                new Conv1dAttrs(1, 0, 1, 2)));
        assertThrows(IllegalArgumentException.class, () -> new Conv2dStep(2, 2, 0, 1, false,
                DataType.FLOAT32, Conv2dAttrs.defaults()));
        assertThrows(ArithmeticException.class, () -> new Conv1dStep(1, 1, 2, false, DataType.FLOAT32,
                new Conv1dAttrs(1, Long.MAX_VALUE, 1, 1)));
        assertThrows(ArithmeticException.class,
                () -> new LinearStep(Long.MAX_VALUE, 2, false, DataType.FLOAT64, false));
        assertThrows(IllegalArgumentException.class, () -> new StateDescriptor("0.weight", StateKind.BUFFER,
                DataType.FLOAT32, Shape.of(1), true));
        assertThrows(IllegalArgumentException.class, () -> new StateDescriptor("0..weight", StateKind.PARAMETER,
                DataType.FLOAT32, Shape.of(1), true));
        long oversized = (long) Integer.MAX_VALUE + 1;
        assertThrows(IllegalArgumentException.class,
                () -> new LinearStep(oversized, 1, false, DataType.FLOAT32, true));
        assertThrows(IllegalArgumentException.class,
                () -> new Conv1dStep(oversized, 1, 1, false, DataType.FLOAT32, Conv1dAttrs.defaults()));
        assertThrows(IllegalArgumentException.class,
                () -> new Conv2dStep(oversized, 1, 1, 1, false, DataType.FLOAT32, Conv2dAttrs.defaults()));
        assertThrows(IllegalArgumentException.class,
                () -> new Conv3dStep(oversized, 1, 1, 1, 1, false, DataType.FLOAT32, Conv3dAttrs.defaults()));
        assertDoesNotThrow(() -> new LinearStep(oversized, 1, false, DataType.FLOAT32, false));
        assertDoesNotThrow(() -> new EmbeddingStep(1, oversized, DataType.FLOAT32));
        assertDoesNotThrow(() -> new LayerNormStep(Shape.of(oversized), DataType.FLOAT32, epsilon(DataType.FLOAT32)));
    }

    @Test
    void boundsCountOccurrencesIncludingReusedRecordsAndRootAndRejectDeepSourcesStackSafely() {
        Step activation = new ActivationStep(ActivationKind.RELU);
        assertEquals(4095, new Sequence(Collections.nCopies(4095, activation)).steps().size());
        assertThrows(IllegalArgumentException.class,
                () -> new Sequence(Collections.nCopies(4096, activation)));
        Sequence repeated = new Sequence(Collections.nCopies(63, activation));
        assertDoesNotThrow(() -> new Sequence(Collections.nCopies(63, repeated)));
        assertThrows(IllegalArgumentException.class, () -> new Sequence(Collections.nCopies(64, repeated)));
        Sequence deep = new Sequence(List.of());
        for (int i = 1; i < 64; i++) deep = new Sequence(List.of(deep));
        Sequence boundary = deep;
        assertDoesNotThrow(() -> new ModelSpecification(boundary, List.of()));
        assertThrows(IllegalArgumentException.class, () -> new Sequence(List.of(boundary)));
        Sequential source = new Sequential(List.of());
        for (int i = 0; i < 1000; i++) source = new Sequential(List.of(source));
        Sequential tooDeep = source;
        assertThrows(IllegalArgumentException.class, () -> ModelSpecification.describe(tooDeep));
        List<UnaryTensorModule> manyChildren = new ArrayList<>();
        for (int i = 0; i < 4096; i++) manyChildren.add(new Activation(ActivationKind.RELU));
        Sequential tooWide = new Sequential(manyChildren);
        assertThrows(IllegalArgumentException.class, () -> ModelSpecification.describe(tooWide));
    }

    @Test
    void rejectsUnsupportedAndUnboundSourcesWithoutForwardOrOwnershipChange() {
        UnaryTensorModule custom = new UnaryTensorModule() {
            @Override public Tensor forward(Tensor input) { fail("must not capture forward"); return input; }
        };
        Sequential unknown = new Sequential(List.of(new Sequential(List.of(custom))));
        var failure = assertThrows(IllegalArgumentException.class, () -> ModelSpecification.describe(unknown));
        assertTrue(failure.getMessage().contains("0.0"));
        Linear lazy = new Linear(3, false, DataType.FLOAT32, ParameterInitialization.zeros(),
                java.util.random.RandomGeneratorFactory.of("L64X128MixRandom"), 1);
        Sequential unbound = new Sequential(List.of(lazy));
        assertThrows(IllegalStateException.class, () -> ModelSpecification.describe(unbound));
        assertThrows(IllegalStateException.class, lazy::weight);
        assertSame(lazy, unbound.children().get("0"));
    }

    @Test
    void rejectsUnknownRepeatedOverlappingMismatchedAndNoncanonicalAliasGroups() {
        Sequence root = new Sequence(List.of(new LinearStep(2, 2, true, DataType.FLOAT32, false),
                new LinearStep(2, 2, true, DataType.FLOAT32, false),
                new LinearStep(2, 2, true, DataType.FLOAT64, false)));
        assertThrows(IllegalArgumentException.class, () -> new AliasGroup(List.of("0.weight")));
        assertThrows(IllegalArgumentException.class, () -> new AliasGroup(List.of("0.weight", "0.weight")));
        for (List<String> bad : List.of(List.of("0.weight", "missing"),
                List.of("1.weight", "0.weight"), List.of("0.weight", "0.bias"),
                List.of("0.weight", "2.weight")))
            assertThrows(IllegalArgumentException.class, () -> new ModelSpecification(root, List.of(new AliasGroup(bad))));
        var weight = new AliasGroup(List.of("0.weight", "1.weight"));
        var bias = new AliasGroup(List.of("0.bias", "1.bias"));
        assertThrows(IllegalArgumentException.class, () -> new ModelSpecification(root, List.of(bias, weight)));
        assertThrows(IllegalArgumentException.class, () -> new ModelSpecification(root, List.of(weight, weight)));
        assertDoesNotThrow(() -> new ModelSpecification(root, List.of(weight, bias)));
    }

    static ModelSpecification specification(Step step) {
        return new ModelSpecification(new Sequence(List.of(step)), List.of());
    }

    static Tensor leaf(DataType type, Shape shape) {
        return TensorFactory.create(descriptor(type, shape, true));
    }

    static TensorDescriptor descriptor(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape, Optional.empty(), grad);
    }

    static ScalarValue epsilon(DataType type) {
        return switch (type) {
            case FLOAT64 -> ScalarValue.float64(0.00001000000000001);
            case BFLOAT16 -> ScalarValue.bfloat16(0.00001f);
            default -> ScalarValue.float32(0.000010000001f);
        };
    }

    private static Set<String> publicMethods(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods()).filter(m -> Modifier.isPublic(m.getModifiers()))
                .map(java.lang.reflect.Method::getName).collect(Collectors.toSet());
    }
}
