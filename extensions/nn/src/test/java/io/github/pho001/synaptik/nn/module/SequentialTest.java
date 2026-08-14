package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.layers.BatchNorm;
import io.github.pho001.synaptik.nn.layers.Dropout;
import io.github.pho001.synaptik.nn.layers.Embedding;
import io.github.pho001.synaptik.nn.layers.LayerNorm;
import io.github.pho001.synaptik.nn.layers.Linear;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class SequentialTest {
    @Test
    void exposesExactlyThePlannedTypesAndMembers() throws ReflectiveOperationException {
        Constructor<UnaryTensorModule> baseConstructor =
                UnaryTensorModule.class.getDeclaredConstructor();
        Method baseForward = UnaryTensorModule.class.getDeclaredMethod("forward", Tensor.class);
        Set<List<Class<?>>> sequentialConstructors =
                Arrays.stream(Sequential.class.getDeclaredConstructors())
                        .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                        .map(Constructor::getParameterTypes)
                        .map(List::of)
                        .collect(Collectors.toSet());
        Set<String> sequentialMethods = Arrays.stream(Sequential.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());
        Method sequentialForward = Sequential.class.getDeclaredMethod("forward", Tensor.class);
        Field modules = Sequential.class.getDeclaredField("modules");
        Method indexedRegistration = Module.class.getDeclaredMethod(
                "registerIndexedChildren", List.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(UnaryTensorModule.class.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(UnaryTensorModule.class.getModifiers())),
                () -> assertSame(Module.class, UnaryTensorModule.class.getSuperclass()),
                () -> assertTrue(Modifier.isProtected(baseConstructor.getModifiers())),
                () -> assertTrue(Modifier.isPublic(baseForward.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(baseForward.getModifiers())),
                () -> assertSame(Tensor.class, baseForward.getReturnType()),
                () -> assertEquals(0, UnaryTensorModule.class.getDeclaredFields().length),
                () -> assertEquals(0, UnaryTensorModule.class.getDeclaredClasses().length),
                () -> assertEquals(0, UnaryTensorModule.class.getInterfaces().length),
                () -> assertTrue(Modifier.isPublic(Sequential.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(Sequential.class.getModifiers())),
                () -> assertSame(UnaryTensorModule.class, Sequential.class.getSuperclass()),
                () -> assertEquals(Set.of(List.of(List.class)), sequentialConstructors),
                () -> assertEquals(Set.of("forward"), sequentialMethods),
                () -> assertTrue(Modifier.isFinal(sequentialForward.getModifiers())),
                () -> assertSame(Tensor.class, sequentialForward.getReturnType()),
                () -> assertEquals(1, Sequential.class.getDeclaredFields().length),
                () -> assertSame(List.class, modules.getType()),
                () -> assertTrue(Modifier.isPrivate(modules.getModifiers())),
                () -> assertTrue(Modifier.isFinal(modules.getModifiers())),
                () -> assertEquals(0, Sequential.class.getDeclaredClasses().length),
                () -> assertEquals(0, Sequential.class.getInterfaces().length),
                () -> assertTrue(Modifier.isFinal(indexedRegistration.getModifiers())),
                () -> assertFalse(Modifier.isPublic(indexedRegistration.getModifiers())),
                () -> assertFalse(Modifier.isProtected(indexedRegistration.getModifiers())),
                () -> assertSame(UnaryTensorModule.class, Linear.class.getSuperclass()),
                () -> assertSame(UnaryTensorModule.class, LayerNorm.class.getSuperclass()),
                () -> assertSame(UnaryTensorModule.class, Embedding.class.getSuperclass()),
                () -> assertSame(Module.class, BatchNorm.class.getSuperclass()),
                () -> assertSame(Module.class, Dropout.class.getSuperclass()),
                () -> assertFalse(UnaryTensorModule.class.isAssignableFrom(BatchNorm.class)),
                () -> assertFalse(UnaryTensorModule.class.isAssignableFrom(Dropout.class)),
                () -> assertFalse(Arrays.stream(Module.class.getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("forward"))));
    }

    @Test
    void rejectsNullsAndDuplicateIdentityBeforeInstallingAnyOwnership() {
        RecordingUnary first = identityUnary();
        RecordingUnary second = identityUnary();
        ArrayList<UnaryTensorModule> withNull = new ArrayList<>();
        withNull.add(first);
        withNull.add(null);
        withNull.add(second);

        NullPointerException nullList = assertThrows(
                NullPointerException.class, () -> new Sequential(null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class, () -> new Sequential(withNull));

        assertAll(
                () -> assertEquals("modules", nullList.getMessage()),
                () -> assertEquals("modules[1]", nullElement.getMessage()),
                () -> assertTrue(first.children().isEmpty()),
                () -> assertTrue(second.children().isEmpty()));

        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new Sequential(List.of(first, second, first)));
        assertTrue(duplicate.getMessage().contains("repeated"));

        Sequential reusable = new Sequential(List.of(first, second));
        assertEquals(List.of("0", "1"), List.copyOf(reusable.children().keySet()));
        assertSame(first, reusable.children().get("0"));
        assertSame(second, reusable.children().get("1"));
    }

    @Test
    void duplicateDetectionUsesIdentityRatherThanEquality() {
        EqualUnary first = new EqualUnary();
        EqualUnary second = new EqualUnary();

        Sequential sequential = new Sequential(List.of(first, second));

        assertEquals(List.of("0", "1"), List.copyOf(sequential.children().keySet()));
        assertSame(first, sequential.children().get("0"));
        assertSame(second, sequential.children().get("1"));
    }

    @Test
    void lateOwnershipAndCycleFailuresDoNotStrandAValidPrefix() {
        RecordingUnary owned = identityUnary();
        Sequential existingOwner = new Sequential(List.of(owned));
        RecordingUnary prefix = identityUnary();

        assertThrows(
                IllegalStateException.class,
                () -> new Sequential(List.of(prefix, owned)));

        Sequential nextOwner = new Sequential(List.of(prefix));
        assertAll(
                () -> assertSame(owned, existingOwner.children().get("0")),
                () -> assertSame(prefix, nextOwner.children().get("0")));

        IndexedModule root = new IndexedModule(false);
        IndexedModule descendant = new IndexedModule(false);
        IndexedModule validPrefix = new IndexedModule(false);
        root.attach("descendant", descendant);

        assertThrows(
                IllegalArgumentException.class,
                () -> descendant.attachIndexed(List.of(validPrefix, root)));
        assertTrue(descendant.children().isEmpty());

        IndexedModule laterOwner = new IndexedModule(false);
        laterOwner.attachIndexed(List.of(validPrefix));
        assertSame(validPrefix, laterOwner.children().get("0"));
    }

    @Test
    void indexedRegistrationPreflightsNumericNamesBeforeOwnershipInstallation() {
        IndexedModule receiver = new IndexedModule(true);
        IndexedModule first = new IndexedModule(false);
        IndexedModule second = new IndexedModule(false);

        assertThrows(
                IllegalArgumentException.class,
                () -> receiver.attachIndexed(List.of(first, second)));

        assertTrue(receiver.children().isEmpty());
        IndexedModule nextOwner = new IndexedModule(false);
        nextOwner.attachIndexed(List.of(first, second));
        assertEquals(List.of("0", "1"), List.copyOf(nextOwner.children().keySet()));
    }

    @Test
    void snapshotsTheCallerListAndExposesOnlyImmutableNumericChildren() {
        RecordingUnary first = identityUnary();
        RecordingUnary second = identityUnary();
        RecordingUnary later = identityUnary();
        ArrayList<UnaryTensorModule> supplied = new ArrayList<>(List.of(first, second));
        Sequential sequential = new Sequential(supplied);

        supplied.clear();
        supplied.add(later);
        Map<String, Module> children = sequential.children();

        assertAll(
                () -> assertEquals(List.of("0", "1"), List.copyOf(children.keySet())),
                () -> assertSame(first, children.get("0")),
                () -> assertSame(second, children.get("1")),
                () -> assertFalse(children.containsValue(later)),
                () -> assertThrows(UnsupportedOperationException.class, children::clear));

        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        sequential.forward(input);
        assertEquals(1, first.calls());
        assertEquals(1, second.calls());
        assertEquals(0, later.calls());
    }

    @Test
    void emptyForwardIsExactIdentityAndConsumesNoTensorIdentifier()
            throws ReflectiveOperationException {
        Sequential sequential = new Sequential(List.of());
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        AtomicLong nextTensorId = nextTensorIdState();
        long before = nextTensorId.get();

        Tensor result = sequential.forward(input);
        NullPointerException nullInput = assertThrows(
                NullPointerException.class, () -> sequential.forward(null));

        assertAll(
                () -> assertSame(input, result),
                () -> assertEquals(before, nextTensorId.get()),
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertTrue(sequential.children().isEmpty()),
                () -> assertTrue(sequential.parametersRecursively().isEmpty()),
                () -> assertTrue(sequential.buffersRecursively().isEmpty()));
    }

    @Test
    void forwardsExactReferencesLeftToRightOnceAndReturnsTheFinalIdentity() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor firstOutput = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor secondOutput = tensor(DataType.FLOAT64, Shape.of(4), true);
        Tensor finalOutput = tensor(DataType.BFLOAT16, Shape.scalar(), false);
        RecordingUnary first = new RecordingUnary(ignored -> firstOutput);
        RecordingUnary second = new RecordingUnary(ignored -> secondOutput);
        RecordingUnary third = new RecordingUnary(ignored -> finalOutput);
        Sequential sequential = new Sequential(List.of(first, second, third));

        Tensor result = sequential.forward(input);

        assertAll(
                () -> assertSame(finalOutput, result),
                () -> assertEquals(List.of(input), first.inputs()),
                () -> assertEquals(List.of(firstOutput), second.inputs()),
                () -> assertEquals(List.of(secondOutput), third.inputs()),
                () -> assertEquals(1, first.calls()),
                () -> assertEquals(1, second.calls()),
                () -> assertEquals(1, third.calls()));
    }

    @Test
    void nullInputNullResultAndThrownFailureSuppressTheSuffixAndKeepPrefixEffects() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Linear prefix = new Linear(weight);
        RecordingUnary nullResult = new RecordingUnary(ignored -> null);
        RecordingUnary suffix = identityUnary();
        Sequential nullSequence = new Sequential(List.of(prefix, nullResult, suffix));

        NullPointerException nullInput = assertThrows(
                NullPointerException.class, () -> nullSequence.forward(null));
        NullPointerException nullOutput = assertThrows(
                NullPointerException.class, () -> nullSequence.forward(input));
        Tensor retainedPrefix = nullResult.inputs().getFirst();

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("modules[1] output", nullOutput.getMessage()),
                () -> assertSame(MatmulKind.MATMUL,
                        retainedPrefix.provenance().orElseThrow().operation().kind()),
                () -> assertEquals(1, nullResult.calls()),
                () -> assertEquals(0, suffix.calls()));

        DeliberateFailure failure = new DeliberateFailure();
        RecordingUnary throwingPrefix = new RecordingUnary(ignored -> retainedPrefix);
        RecordingUnary throwingSuffix = identityUnary();
        Sequential throwingSequence =
                new Sequential(List.of(throwingPrefix, new ThrowingUnary(failure), throwingSuffix));

        DeliberateFailure thrown = assertThrows(
                DeliberateFailure.class, () -> throwingSequence.forward(input));
        assertAll(
                () -> assertSame(failure, thrown),
                () -> assertEquals(1, throwingPrefix.calls()),
                () -> assertEquals(1, failure.calls()),
                () -> assertSame(retainedPrefix, failure.input()),
                () -> assertEquals(0, throwingSuffix.calls()));
    }

    @Test
    void composesCurrentLinearLayerNormAndEmbeddingExpressionsWithoutAWrapper() {
        Linear linear = new Linear(tensor(DataType.FLOAT32, Shape.of(4, 3), true));
        LayerNorm layerNorm = new LayerNorm(
                tensor(DataType.FLOAT32, Shape.of(4), true),
                tensor(DataType.FLOAT32, Shape.of(4), true),
                ScalarValue.float32(1.0e-5f));
        Sequential dense = new Sequential(List.of(linear, layerNorm));
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);

        Tensor first = dense.forward(input);
        Tensor second = dense.forward(input);
        Tensor linearResult = first.provenance().orElseThrow().inputs().getFirst();

        assertAll(
                () -> assertSame(LayerNormKind.LAYER_NORM,
                        first.provenance().orElseThrow().operation().kind()),
                () -> assertSame(MatmulKind.MATMUL,
                        linearResult.provenance().orElseThrow().operation().kind()),
                () -> assertNotSame(first, second),
                () -> assertNotSame(
                        first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertEquals(Shape.of(2, 4), first.descriptor().shape()));

        Embedding embedding = new Embedding(
                tensor(DataType.FLOAT32, Shape.of(10, 3), true));
        Linear projection = new Linear(
                tensor(DataType.FLOAT32, Shape.of(4, 3), true));
        Sequential embedded = new Sequential(List.of(embedding, projection));
        Tensor indices = tensor(DataType.INT64, Shape.of(2), false);

        Tensor embeddedResult = embedded.forward(indices);
        Tensor gather = embeddedResult.provenance().orElseThrow().inputs().getFirst();
        assertAll(
                () -> assertSame(MatmulKind.MATMUL,
                        embeddedResult.provenance().orElseThrow().operation().kind()),
                () -> assertSame(AxisGatherKind.GATHER,
                        gather.provenance().orElseThrow().operation().kind()),
                () -> assertEquals(Shape.of(2, 4), embeddedResult.descriptor().shape()));
    }

    @Test
    void preservesNestedNumericPathsStateLoadAndReplacementSnapshots() {
        Tensor firstWeight = tensor(DataType.FLOAT32, Shape.of(2), true);
        Tensor nestedWeight = tensor(DataType.FLOAT32, Shape.of(3), true);
        StateUnary first = new StateUnary(firstWeight);
        StateUnary nested = new StateUnary(nestedWeight);
        Sequential inner = new Sequential(List.of(nested));
        Sequential outer = new Sequential(List.of(first, inner));

        assertAll(
                () -> assertEquals(List.of("0", "1"), List.copyOf(outer.children().keySet())),
                () -> assertSame(inner, outer.children().get("1")),
                () -> assertEquals(List.of("0"), List.copyOf(inner.children().keySet())),
                () -> assertEquals(
                        List.of("0.weight", "1.0.weight"),
                        List.copyOf(outer.parametersRecursively().keySet())),
                () -> assertEquals(
                        List.of("0.weight", "1.0.weight"),
                        outer.stateDictionary().entries().stream().map(StateEntry::path).toList()));

        Tensor nextFirst = tensor(DataType.FLOAT32, Shape.of(2), true);
        Tensor nextNested = tensor(DataType.FLOAT32, Shape.of(3), true);
        outer.loadStateDictionary(new StateDictionary(List.of(
                new StateEntry("1.0.weight", StateKind.PARAMETER, nextNested),
                new StateEntry("0.weight", StateKind.PARAMETER, nextFirst))));

        assertAll(
                () -> assertSame(nextFirst, first.weight().value()),
                () -> assertSame(nextNested, nested.weight().value()),
                () -> assertEquals(
                        List.of("0.weight", "1.0.weight"),
                        outer.stateDictionary().entries().stream().map(StateEntry::path).toList()));

        Tensor oldLinearWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Linear linear = new Linear(oldLinearWeight);
        Sequential sequence = new Sequential(List.of(linear));
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor before = sequence.forward(input);
        Tensor nextLinearWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        linear.weight().replace(nextLinearWeight);
        Tensor after = sequence.forward(input);

        assertAll(
                () -> assertSame(oldLinearWeight, linearWeightInput(before)),
                () -> assertSame(nextLinearWeight, linearWeightInput(after)),
                () -> assertSame(nextLinearWeight,
                        sequence.stateDictionary().entries().getFirst().value()));
    }

    @Test
    void nestedModePropagationUsesInheritedTreeBehaviorWithoutNormalizingConstruction() {
        RecordingUnary first = identityUnary();
        RecordingUnary nestedChild = identityUnary();
        nestedChild.eval();
        Sequential inner = new Sequential(List.of(nestedChild));
        Sequential outer = new Sequential(List.of(first, inner));

        assertAll(
                () -> assertSame(ForwardMode.TRAINING, outer.mode()),
                () -> assertSame(ForwardMode.TRAINING, inner.mode()),
                () -> assertSame(ForwardMode.TRAINING, first.mode()),
                () -> assertSame(ForwardMode.EVALUATION, nestedChild.mode()));

        outer.eval();
        assertTrue(List.of(outer, first, inner, nestedChild).stream()
                .allMatch(module -> module.mode() == ForwardMode.EVALUATION));
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        assertSame(input, outer.forward(input));

        outer.train();
        assertTrue(List.of(outer, first, inner, nestedChild).stream()
                .allMatch(module -> module.mode() == ForwardMode.TRAINING));
        assertSame(input, outer.forward(input));
    }

    private static RecordingUnary identityUnary() {
        return new RecordingUnary(Function.identity());
    }

    private static Tensor linearWeightInput(Tensor result) {
        Tensor transposedWeight = result.provenance().orElseThrow().inputs().get(1);
        return transposedWeight.provenance().orElseThrow().inputs().getFirst();
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static class RecordingUnary extends UnaryTensorModule {
        private final Function<Tensor, Tensor> function;
        private final List<Tensor> inputs = new ArrayList<>();

        private RecordingUnary(Function<Tensor, Tensor> function) {
            this.function = function;
        }

        @Override
        public Tensor forward(Tensor input) {
            inputs.add(input);
            return function.apply(input);
        }

        private int calls() {
            return inputs.size();
        }

        private List<Tensor> inputs() {
            return List.copyOf(inputs);
        }
    }

    private static final class EqualUnary extends RecordingUnary {
        private EqualUnary() {
            super(Function.identity());
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof EqualUnary;
        }

        @Override
        public int hashCode() {
            return 1;
        }
    }

    private static final class DeliberateFailure extends RuntimeException {
        private int calls;
        private Tensor input;

        private DeliberateFailure() {
            super("deliberate");
        }

        private Tensor forward(Tensor input) {
            this.calls++;
            this.input = input;
            throw this;
        }

        private int calls() {
            return calls;
        }

        private Tensor input() {
            return input;
        }
    }

    private static final class ThrowingUnary extends UnaryTensorModule {
        private final DeliberateFailure failure;

        private ThrowingUnary(DeliberateFailure failure) {
            this.failure = failure;
        }

        @Override
        public Tensor forward(Tensor input) {
            return failure.forward(input);
        }
    }

    private static final class StateUnary extends UnaryTensorModule {
        private final Parameter weight;

        private StateUnary(Tensor weight) {
            this.weight = parameter("weight", weight);
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }

        private Parameter weight() {
            return weight;
        }
    }

    private static final class IndexedModule extends Module {
        private IndexedModule(boolean reserveZero) {
            if (reserveZero) {
                parameter("0", tensor(DataType.FLOAT32, Shape.scalar(), true));
            }
        }

        private <T extends Module> T attach(String name, T module) {
            return child(name, module);
        }

        private <T extends Module> List<T> attachIndexed(List<? extends T> modules) {
            return registerIndexedChildren(modules);
        }
    }
}
