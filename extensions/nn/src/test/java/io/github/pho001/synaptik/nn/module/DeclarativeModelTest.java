package io.github.pho001.synaptik.nn.module;

import static io.github.pho001.synaptik.nn.module.ModelSpecificationTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.*;
import io.github.pho001.synaptik.nn.module.ModelSpecification.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class DeclarativeModelTest {
    private static final List<DataType> FLOATS = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16, DataType.FLOAT16);

    @Test
    void rootOwnsNumericChildrenDirectlyAndEveryOccurrenceIsIndependentAndEval() {
        Step linear = new LinearStep(2, 2, true, DataType.FLOAT32, false);
        var spec = new ModelSpecification(new Sequence(List.of(linear, new Sequence(List.of(linear)))), List.of());
        var model = DeclarativeModel.reconstruct(spec);
        assertEquals(List.of("0", "1"), List.copyOf(model.children().keySet()));
        assertEquals(List.of("0.weight", "0.bias", "1.0.weight", "1.0.bias"),
                model.stateDictionary().entries().stream().map(StateEntry::path).toList());
        assertNotSame(model.children().get("0"), model.children().get("1").children().get("0"));
        assertNotSame(model.parametersRecursively().get("0.weight").value(),
                model.parametersRecursively().get("1.0.weight").value());
        assertEquals(spec.root(), model.describe().root());
        assertEquals(ForwardMode.EVALUATION, model.mode());
        for (Module module : model.children().values()) assertEquals(ForwardMode.EVALUATION, module.mode());
        assertEquals(ForwardMode.EVALUATION, model.children().get("1").children().get("0").mode());
    }

    @Test
    void userGuideReconstructionExampleProducesDocumentedIntermediateAndFinalState() {
        Linear first = new Linear(2, 2, false, DataType.FLOAT32, new Random(7));
        Linear second = new Linear(first.weight().value());
        Sequential source = new Sequential(List.of(first, new Activation(ActivationKind.RELU), second));
        var specification = ModelSpecification.describe(source);
        var candidate = DeclarativeModel.reconstruct(specification);
        assertEquals(List.of("0.weight", "2.weight"), specification.aliases().getFirst().paths());
        assertTrue(candidate.parametersRecursively().get("0.weight").value().hostStorage().isEmpty());
        candidate.loadStateDictionary(source.stateDictionary());
        assertSame(candidate.parametersRecursively().get("0.weight").value(),
                candidate.parametersRecursively().get("2.weight").value());
        assertEquals(10, specification.outputDescriptor(descriptor(DataType.FLOAT32, Shape.of(5, 2), false))
                .shape().knownElementCount().orElseThrow());
    }

    @Test
    void exactAliasesFollowParameterReplacementAndStrictLoadJoinsAndSplitsInBothDescriptions() {
        Tensor shared = leaf(DataType.FLOAT32, Shape.of(2, 2));
        Linear first = new Linear(shared);
        Linear second = new Linear(shared);
        Sequential source = new Sequential(List.of(first, new Sequential(List.of(second))));
        var expected = List.of(new AliasGroup(List.of("0.weight", "1.0.weight")));
        assertEquals(expected, ModelSpecification.describe(source).aliases());
        second.weight().replace(leaf(DataType.FLOAT32, Shape.of(2, 2)));
        assertTrue(ModelSpecification.describe(source).aliases().isEmpty());
        second.weight().replace(shared);
        assertEquals(expected, ModelSpecification.describe(source).aliases());
        var model = DeclarativeModel.reconstruct(ModelSpecification.describe(source));
        assertSame(model.parametersRecursively().get("0.weight").value(),
                model.parametersRecursively().get("1.0.weight").value());
        Tensor a = leaf(DataType.FLOAT32, Shape.of(2, 2));
        Tensor b = leaf(DataType.FLOAT32, Shape.of(2, 2));
        StateDictionary split = dictionary("0.weight", a, "1.0.weight", b);
        source.loadStateDictionary(split);
        model.loadStateDictionary(split);
        assertTrue(ModelSpecification.describe(source).aliases().isEmpty());
        assertTrue(model.describe().aliases().isEmpty());
        StateDictionary joined = dictionary("0.weight", b, "1.0.weight", b);
        source.loadStateDictionary(joined);
        model.loadStateDictionary(joined);
        assertEquals(expected, ModelSpecification.describe(source).aliases());
        assertEquals(expected, model.describe().aliases());
        model.parametersRecursively().get("1.0.weight").replace(a);
        assertTrue(model.describe().aliases().isEmpty());
        model.parametersRecursively().get("0.weight").replace(a);
        assertEquals(expected, model.describe().aliases());
        assertSame(a, model.stateDictionary().entries().getFirst().value());
    }

    @Test
    void independentEqualValuesNeverBecomeAliasesAndAliasPreservingReloadRetainsExactReferences() {
        Linear a = new Linear(2, 2, false, DataType.FLOAT32, new Random(2));
        Linear b = new Linear(2, 2, false, DataType.FLOAT32, new Random(2));
        var source = new Sequential(List.of(a, b));
        assertNotSame(a.weight().value(), b.weight().value());
        assertTrue(ModelSpecification.describe(source).aliases().isEmpty());
        var spec = new ModelSpecification(ModelSpecification.describe(source).root(),
                List.of(new AliasGroup(List.of("0.weight", "1.weight"))));
        var model = DeclarativeModel.reconstruct(spec);
        Tensor fresh = leaf(DataType.FLOAT32, Shape.of(2, 2));
        model.loadStateDictionary(dictionary("0.weight", fresh, "1.weight", fresh));
        assertSame(fresh, model.parametersRecursively().get("0.weight").value());
        assertSame(fresh, model.parametersRecursively().get("1.weight").value());
        assertEquals(spec.aliases(), model.describe().aliases());
    }

    @Test
    void everyConvolutionRetainsCompleteAttrsAndOrdinaryProvenanceAcrossFloatingMatrix() {
        for (DataType type : FLOATS) {
            Conv1dAttrs a1 = new Conv1dAttrs(2, 3, 2, 2);
            Conv2dAttrs a2 = new Conv2dAttrs(2, 3, 1, 2, 2, 1, 2);
            Conv3dAttrs a3 = new Conv3dAttrs(2, 1, 3, 1, 2, 3, 2, 1, 2, 2);
            List<Step> steps = List.of(new Conv1dStep(4, 6, 3, true, type, a1),
                    new Conv2dStep(4, 6, 2, 3, true, type, a2),
                    new Conv3dStep(4, 6, 2, 3, 2, true, type, a3));
            List<Shape> shapes = List.of(Shape.of(2, 4, 9), Shape.of(2, 4, 9, 11),
                    Shape.of(2, 4, 9, 11, 13));
            List<UnaryTensorModule> direct = List.of(
                    new Conv1d(6, 3, 2, 3, 2, 2, true, type, ParameterInitialization.normal(3, 2), 123),
                    new Conv2d(6, 2, 3, 2, 3, 1, 2, 2, 1, 2, true, type,
                            ParameterInitialization.ones(), 99),
                    new Conv3d(6, 2, 3, 2, 2, 1, 3, 1, 2, 3, 2, 1, 2, 2, true, type,
                            ParameterInitialization.glorotUniform(), -1));
            for (int i = 0; i < steps.size(); i++) {
                var spec = specification(steps.get(i));
                var skeleton = DeclarativeModel.reconstruct(spec);
                var source = new Sequential(List.of(direct.get(i)));
                source.loadStateDictionary(skeleton.stateDictionary());
                assertEquals(spec.root(), ModelSpecification.describe(source).root());
                assertEquals(spec.root(), skeleton.describe().root());
                Tensor input = TensorFactory.create(descriptor(type, shapes.get(i), false));
                Tensor expected = source.forward(input);
                Tensor actual = skeleton.forward(input);
                assertEquals(expected.descriptor(), actual.descriptor());
                assertProvenance(expected, actual);
                assertEquals(expected.descriptor(), spec.outputDescriptor(input.descriptor()));
                List<Dimension> dimensions = new ArrayList<>(shapes.get(i).dimensions());
                dimensions.set(0, new DynamicDimension("B"));
                dimensions.set(dimensions.size() - 1, new DynamicDimension("W"));
                var variable = descriptor(type, Shape.ofDimensions(dimensions.toArray(Dimension[]::new)), false);
                assertEquals(source.forward(TensorFactory.create(variable)).descriptor(), spec.outputDescriptor(variable));
                int rank = shapes.get(i).rank();
                assertThrows(IllegalArgumentException.class,
                        () -> skeleton.forward(TensorFactory.create(descriptor(type, Shape.of(2, 4), false))));
                dimensions.set(1, new DynamicDimension("C"));
                assertThrows(IllegalArgumentException.class, () -> skeleton.forward(
                        TensorFactory.create(descriptor(type, Shape.ofDimensions(dimensions.toArray(Dimension[]::new)), false))));
                long[] badChannels = new long[rank];
                Arrays.fill(badChannels, 10);
                badChannels[1] = 2;
                assertThrows(IllegalArgumentException.class, () -> skeleton.forward(
                        TensorFactory.create(descriptor(type, Shape.of(badChannels), false))));
            }
        }
    }

    @Test
    void hugeSkeletonsAllocateOnlyOneLeafPerIdentityGroupAndNoThrowawayTensorOrStorage() throws Exception {
        for (DataType type : FLOATS) {
            long huge = Integer.MAX_VALUE;
            List<Step> steps = List.of(new LinearStep(huge, 1, false, type, true),
                    new LinearStep(huge, 1, false, type, true),
                    new EmbeddingStep(1, huge, type),
                    new LayerNormStep(Shape.of(huge), type, epsilon(type)),
                    new Conv1dStep(huge, 1, 1, false, type, Conv1dAttrs.defaults()),
                    new Conv2dStep(huge, 1, 1, 1, false, type, Conv2dAttrs.defaults()),
                    new Conv3dStep(huge, 1, 1, 1, 1, false, type, Conv3dAttrs.defaults()));
            var spec = new ModelSpecification(new Sequence(steps),
                    List.of(new AliasGroup(List.of("0.weight", "1.weight", "2.weight"))));
            AtomicLong ids = ids();
            long before = ids.get();
            var model = DeclarativeModel.reconstruct(spec);
            assertEquals(before + spec.stateDescriptors().size() - 2, ids.get());
            for (StateEntry state : model.stateDictionary().entries()) {
                assertTrue(state.value().hostStorage().isEmpty());
                assertTrue(state.value().provenance().isEmpty());
                assertTrue(state.value().descriptor().layout().isEmpty());
                assertTrue(state.value().descriptor().requiresGrad());
                assertTrue(state.value().label().isEmpty());
            }
            assertEquals(spec.root(), model.describe().root());
            assertEquals(spec.aliases(), model.describe().aliases());
            assertEquals(before + spec.stateDescriptors().size() - 2, ids.get());
        }
    }

    @Test
    void linearFromStateCreatesNoTensorOrRandomSourceAndRejectsInvalidCompleteGroups() throws Exception {
        for (DataType type : FLOATS) {
            Tensor weight = leaf(type, Shape.of(2, 3));
            Tensor bias = leaf(type, Shape.of(2));
            long before = ids().get();
            Linear bound = Linear.fromState(weight, Optional.of(bias), true);
            assertEquals(before, ids().get());
            assertSame(weight, bound.weight().value());
            assertSame(bias, bound.bias().orElseThrow().value());
            assertTrue(bound.specification().automaticInputGuards());
            assertTrue(Linear.fromState(weight, Optional.empty(), true).bias().isEmpty());
            assertFalse(new Linear(weight).specification().automaticInputGuards());
            assertFalse(Linear.fromState(weight, Optional.of(bias), false).specification().automaticInputGuards());
            Tensor input = TensorFactory.create(descriptor(type, Shape.of(5, 3), false));
            assertProvenance(new Linear(weight, bias).forward(input), bound.forward(input));
            assertThrows(IllegalArgumentException.class, () -> Linear.fromState(weight,
                    Optional.of(leaf(type, Shape.of(3))), true));
            assertThrows(IllegalArgumentException.class, () -> Linear.fromState(
                    TensorFactory.create(descriptor(type, Shape.of(2, 3), false)), Optional.empty(), false));
        }
        assertThrows(NullPointerException.class, () -> Linear.fromState(null, Optional.empty(), true));
        assertThrows(NullPointerException.class, () -> Linear.fromState(leaf(DataType.FLOAT32, Shape.of(1, 1)),
                null, true));
        Tensor oversized = leaf(DataType.FLOAT32, Shape.of(1, (long) Integer.MAX_VALUE + 1));
        assertThrows(IllegalArgumentException.class, () -> Linear.fromState(oversized, Optional.empty(), true));
        assertDoesNotThrow(() -> Linear.fromState(oversized, Optional.empty(), false));
    }

    @Test
    void descriptionRetainsAutomaticVersusEagerAndStrictLoadedGuardPolicyWithoutInitialization() throws Exception {
        Linear automatic = new Linear(2, true, DataType.FLOAT32, ParameterInitialization.glorotUniform(),
                java.util.random.RandomGeneratorFactory.of("L64X128MixRandom"), 789);
        Tensor weight = leaf(DataType.FLOAT32, Shape.of(2, 3));
        Tensor bias = leaf(DataType.FLOAT32, Shape.of(2));
        automatic.loadStateDictionary(dictionary("weight", weight, "bias", bias));
        long before = ids().get();
        var source = new Sequential(List.of(automatic));
        var spec = ModelSpecification.describe(source);
        assertEquals(before, ids().get());
        assertTrue(((LinearStep) spec.root().steps().getFirst()).automaticInputGuards());
        var rebuilt = DeclarativeModel.reconstruct(spec);
        rebuilt.loadStateDictionary(source.stateDictionary());
        assertSame(weight, rebuilt.parametersRecursively().get("0.weight").value());
        var promoted = TensorFactory.create(descriptor(DataType.FLOAT64, Shape.of(1, 3), false));
        assertThrows(IllegalArgumentException.class, () -> source.forward(promoted));
        assertThrows(IllegalArgumentException.class, () -> rebuilt.forward(promoted));
        var eager = new Linear(3, 2, false, DataType.FLOAT32, new Random(1));
        assertFalse(eager.specification().automaticInputGuards());
    }

    @Test
    void invalidReloadLeavesSkeletonBindingsIntactAndNullOperationsFail() {
        var model = DeclarativeModel.reconstruct(specification(new LinearStep(3, 2, true, DataType.FLOAT32, true)));
        var prior = model.stateDictionary();
        assertThrows(IllegalArgumentException.class,
                () -> model.loadStateDictionary(dictionary("0.weight", leaf(DataType.FLOAT32, Shape.of(2, 3)),
                        "0.bias", leaf(DataType.FLOAT32, Shape.of(3)))));
        assertEquals(prior, model.stateDictionary());
        assertThrows(NullPointerException.class, () -> model.forward(null));
        assertThrows(NullPointerException.class, () -> DeclarativeModel.reconstruct(null));
        assertThrows(NullPointerException.class, () -> model.describe().outputDescriptor(null));
        assertThrows(NullPointerException.class, () -> ModelSpecification.describe(null));
    }

    private static StateDictionary dictionary(String a, Tensor av, String b, Tensor bv) {
        return new StateDictionary(List.of(new StateEntry(a, StateKind.PARAMETER, av),
                new StateEntry(b, StateKind.PARAMETER, bv)));
    }

    private static AtomicLong ids() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static void assertProvenance(Tensor expected, Tensor actual) {
        assertEquals(expected.descriptor(), actual.descriptor());
        assertEquals(expected.provenance().isPresent(), actual.provenance().isPresent());
        if (expected.provenance().isEmpty()) {
            assertSame(expected, actual);
            return;
        }
        var ep = expected.provenance().orElseThrow();
        var ap = actual.provenance().orElseThrow();
        assertEquals(ep.operation(), ap.operation());
        assertEquals(ep.inputs().size(), ap.inputs().size());
        for (int i = 0; i < ep.inputs().size(); i++) assertProvenance(ep.inputs().get(i), ap.inputs().get(i));
    }
}
